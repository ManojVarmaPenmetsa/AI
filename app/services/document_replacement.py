"""Orchestrates staged, verified, generation-based document replacement."""

from pathlib import Path

from app.rag import vectorstore
from app.services.document_registry import DocumentRegistry
from app.services.ingestion import chunk_documents, content_version, ingest_file


class ReplacementError(RuntimeError):
    def __init__(self, message: str, *, stage: str, document_id: str, generation_id: str):
        self.stage = stage
        self.document_id = document_id
        self.generation_id = generation_id
        super().__init__(message)


def replace_document(
    registry: DocumentRegistry,
    path: Path,
    source_filename: str,
    content: bytes,
    document_id: str,
) -> dict:
    version = content_version(content)
    with registry.ingestion_lock(document_id):
        generation_id = registry.begin_generation(document_id, version)
        stage = "parse"
        try:
            def parse_generation(current_id: str):
                parsed = ingest_file(
                    path,
                    source_filename=source_filename,
                    document_id=document_id,
                    generation_id=current_id,
                )
                return parsed, chunk_documents(parsed.documents)

            result, chunks = parse_generation(generation_id)
            if not chunks:
                raise ValueError("No searchable content was produced")

            previous_ids = registry.generation_chunk_ids(generation_id)
            expected = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            if previous_ids and set(previous_ids) != set(expected):
                # Vision/OCR output can vary across retry attempts. Keep those
                # chunks in a separate inactive generation so a partial prior
                # write can never leak into the newly active generation.
                generation_id = registry.rotate_generation(
                    document_id, generation_id, version
                )
                result, chunks = parse_generation(generation_id)
                if not chunks:
                    raise ValueError("No searchable content was produced")

            stage = "upsert"
            expected = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
            registry.set_generation_status(
                generation_id, "pending", chunk_ids=expected, warnings=result.warnings
            )
            ids = vectorstore.upsert_generation_documents(chunks)
            registry.set_generation_status(generation_id, "upserted", chunk_ids=expected)

            stage = "activation"
            registry.activate_generation(document_id, generation_id, ids)

            warnings = list(result.warnings)
            cleanup_pending = False
            stage = "cleanup"
            try:
                stale_ids = registry.stale_chunk_ids(document_id)
                if stale_ids:
                    vectorstore.delete_vector_ids(stale_ids)
                vectorstore.delete_inactive_document_generations(document_id, generation_id)
                registry.clear_stale_chunk_ids(document_id)
            except Exception as exc:
                cleanup_pending = True
                warnings.append(
                    "The new generation is active, but old vectors remain pending cleanup "
                    f"({type(exc).__name__}). Retry ingestion or clean up this document later."
                )
            registry.set_generation_status(
                generation_id, "active", chunk_ids=expected, warnings=warnings
            )
            return {
                "result": result,
                "chunks": chunks,
                "ids": ids,
                "document_id": document_id,
                "document_version": version,
                "generation_id": generation_id,
                "cleanup_pending": cleanup_pending,
                "warnings": warnings,
            }
        except Exception as exc:
            failure_status = {
                "parse": "parse_failed",
                "upsert": "upsert_failed",
                "activation": "activation_failed",
                "cleanup": "active",
            }.get(stage, "failed")
            try:
                registry.set_generation_status(generation_id, failure_status)
            except Exception:
                pass
            if isinstance(exc, ReplacementError):
                raise
            raise ReplacementError(
                f"Document generation failed during {stage} ({type(exc).__name__})",
                stage=stage,
                document_id=document_id,
                generation_id=generation_id,
            ) from exc

