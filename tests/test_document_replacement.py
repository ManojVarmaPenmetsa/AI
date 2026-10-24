from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

from app.rag import vectorstore
from app.services import document_replacement
from app.services.document_registry import DocumentRegistry
from app.services.document_replacement import ReplacementError, replace_document


class FakeVectorIndex:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.deleted_filters: list[dict] = []
        self.deleted_ids: list[str] = []
        self.guard = threading.Lock()

    def upsert(self, documents):
        with self.guard:
            for document in documents:
                self.records[document.metadata["chunk_id"]] = dict(document.metadata)

    def delete_inactive(self, document_id: str, active_generation_id: str):
        with self.guard:
            self.deleted_filters.append({
                "document_id": document_id,
                "generation_id": {"$ne": active_generation_id},
            })
            doomed = [
                chunk_id for chunk_id, metadata in self.records.items()
                if metadata.get("document_id") == document_id
                and metadata.get("generation_id") != active_generation_id
            ]
            for chunk_id in doomed:
                del self.records[chunk_id]

    def delete_ids(self, ids):
        with self.guard:
            self.deleted_ids.extend(ids)
            for chunk_id in ids:
                self.records.pop(chunk_id, None)


class DocumentReplacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = DocumentRegistry(str(Path(self.temp.name) / "registry.db"))
        self.path = Path(self.temp.name) / "manual.pdf"
        self.pdf_bytes = b"synthetic-pdf-payload"
        self.path.write_bytes(self.pdf_bytes)
        self.store = FakeVectorIndex()
        self.original_upsert = vectorstore.upsert_generation_documents
        self.original_cleanup = vectorstore.delete_inactive_document_generations
        self.original_delete_ids = vectorstore.delete_vector_ids
        self.mock_upsert = patch.object(vectorstore, "upsert_generation_documents", self._upsert)
        self.mock_cleanup = patch.object(
            vectorstore, "delete_inactive_document_generations", self.store.delete_inactive
        )
        self.mock_delete_ids = patch.object(vectorstore, "delete_vector_ids", self.store.delete_ids)
        self.mock_upsert.start()
        self.mock_cleanup.start()
        self.mock_delete_ids.start()
        self.addCleanup(self.mock_upsert.stop)
        self.addCleanup(self.mock_cleanup.stop)
        self.addCleanup(self.mock_delete_ids.stop)
        self.output_factory = lambda _call: "Troubleshooting detail " * 15
        self.parser_calls = 0
        self.parser = patch.object(document_replacement, "ingest_file", self._parse)
        self.parser.start()
        self.addCleanup(self.parser.stop)

    def _parse(self, path, *, source_filename, document_id, generation_id):
        self.parser_calls += 1
        text = self.output_factory(self.parser_calls)
        doc = Document(
            page_content=text,
            metadata={
                "source": source_filename,
                "source_filename": source_filename,
                "document_id": document_id,
                "document_version": hashlib.sha256(self.pdf_bytes).hexdigest(),
                "generation_id": generation_id,
                "ingestion_timestamp": "2026-10-24T00:00:00+00:00",
                "page": 1,
                "content_type": "ocr",
                "provenance": "ocr_page",
            },
        )
        return SimpleNamespace(
            documents=[doc], warnings=[], vision_status="completed", page_count=1
        )

    def _upsert(self, chunks):
        self.store.upsert(chunks)
        return [chunk.metadata["chunk_id"] for chunk in chunks]

    def _document_id(self, filename="manual.pdf", **kwargs):
        return self.registry.resolve_document(
            filename, legacy_id=hashlib.sha256(filename.casefold().encode()).hexdigest(), **kwargs
        )

    def _replace(self, document_id):
        return replace_document(
            self.registry, self.path, "manual.pdf", self.pdf_bytes, document_id
        )

    def _active(self, document_id):
        with self.registry._connection() as db:
            row = db.execute(
                "SELECT active_generation_id FROM documents WHERE document_id=?",
                (document_id,),
            ).fetchone()
        return row[0]

    def test_same_pdf_twice_activates_one_replacement_generation(self):
        document_id = self._document_id()
        first = self._replace(document_id)
        first_ids = set(first["ids"])
        second = self._replace(document_id)
        self.assertNotEqual(first["generation_id"], second["generation_id"])
        self.assertEqual(self._active(document_id), second["generation_id"])
        self.assertTrue(first_ids.isdisjoint(set(second["ids"])))
        self.assertEqual(
            {meta["generation_id"] for meta in self.store.records.values()},
            {second["generation_id"]},
        )

    def test_changed_ocr_or_vision_output_for_same_bytes_replaces_prior_chunks(self):
        document_id = self._document_id()
        self.output_factory = lambda call: f"OCR description variant {call}. " * 20
        first = self._replace(document_id)
        self.output_factory = lambda call: f"Vision description variant {call}. " * 20
        second = self._replace(document_id)
        self.assertEqual(first["document_version"], second["document_version"])
        self.assertNotEqual(first["ids"], second["ids"])
        self.assertEqual({m["generation_id"] for m in self.store.records.values()}, {second["generation_id"]})

    def test_shorter_new_generation_removes_old_extra_chunks(self):
        document_id = self._document_id()
        self.output_factory = lambda _call: "A" * 2500
        first = self._replace(document_id)
        self.assertGreater(len(first["ids"]), 1)
        self.output_factory = lambda _call: "Short replacement text."
        second = self._replace(document_id)
        self.assertLess(len(second["ids"]), len(first["ids"]))
        self.assertEqual({m["generation_id"] for m in self.store.records.values()}, {second["generation_id"]})

    def test_same_filename_can_resolve_to_two_distinct_documents(self):
        first = self._document_id()
        second = self._document_id(new_document=True)
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "More than one document"):
            self._document_id()
        self.assertEqual(self._document_id(document_id=second), second)

    def test_partial_upsert_keeps_old_active_and_retry_reuses_generation(self):
        document_id = self._document_id()
        first = self._replace(document_id)
        calls = {"count": 0}

        def fail_half(chunks):
            calls["count"] += 1
            self.store.upsert(chunks[:1])
            if calls["count"] == 1:
                raise RuntimeError("injected partial write")
            self.store.upsert(chunks[1:])
            return [chunk.metadata["chunk_id"] for chunk in chunks]

        self.output_factory = lambda _call: "replacement " * 100
        with patch.object(vectorstore, "upsert_generation_documents", fail_half):
            with self.assertRaises(ReplacementError) as caught:
                self._replace(document_id)
            self.assertEqual(caught.exception.stage, "upsert")
            self.assertEqual(self._active(document_id), first["generation_id"])
            retried = self._replace(document_id)
        self.assertEqual(retried["generation_id"], caught.exception.generation_id)
        self.assertEqual(self._active(document_id), retried["generation_id"])
        self.assertEqual({m["generation_id"] for m in self.store.records.values()}, {retried["generation_id"]})

    def test_activation_failure_preserves_previous_generation(self):
        document_id = self._document_id()
        first = self._replace(document_id)
        with patch.object(self.registry, "activate_generation", side_effect=RuntimeError("db down")):
            with self.assertRaises(ReplacementError) as caught:
                self._replace(document_id)
        self.assertEqual(caught.exception.stage, "activation")
        self.assertEqual(self._active(document_id), first["generation_id"])
        # New vectors exist but remain excluded by the persisted active pointer.
        self.assertEqual(
            self.registry.active_generation_filter()["$or"][0]["$and"][1]["generation_id"],
            first["generation_id"],
        )

    def test_cleanup_failure_keeps_new_generation_active_and_reports_pending(self):
        document_id = self._document_id()
        first = self._replace(document_id)
        with patch.object(vectorstore, "delete_inactive_document_generations", side_effect=RuntimeError("offline")):
            outcome = self._replace(document_id)
        self.assertTrue(outcome["cleanup_pending"])
        self.assertEqual(self._active(document_id), outcome["generation_id"])
        self.assertNotEqual(self._active(document_id), first["generation_id"])
        self.assertTrue(any("pending cleanup" in warning for warning in outcome["warnings"]))

    def test_same_document_uploads_are_serialized(self):
        document_id = self._document_id()
        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        def slow_parse(*args, **kwargs):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.12)
            try:
                return self._parse(*args, **kwargs)
            finally:
                with counter_lock:
                    active -= 1

        errors = []

        def run_one():
            try:
                self._replace(document_id)
            except Exception as exc:
                errors.append(exc)

        with patch.object(document_replacement, "ingest_file", slow_parse):
            workers = [threading.Thread(target=run_one) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10)
        self.assertFalse(errors)
        self.assertEqual(max_active, 1)

    def test_retrieval_filter_excludes_inactive_and_incomplete_generations(self):
        document_id = self._document_id()
        first = self._replace(document_id)
        incomplete = self.registry.begin_generation(document_id, "different-version")
        self.registry.set_generation_status(incomplete, "upsert_failed")
        second = self._replace(document_id)
        filter_value = self.registry.active_generation_filter()
        active_clause = next(
            clause for clause in filter_value["$or"]
            if clause.get("$and", [{}])[0].get("document_id") == document_id
        )
        self.assertEqual(active_clause["$and"][1]["generation_id"], second["generation_id"])
        self.assertNotIn(first["generation_id"], str(filter_value))
        self.assertNotIn(incomplete, str(filter_value))

    def test_legacy_vectors_without_generation_metadata_remain_searchable(self):
        legacy_id = self._document_id()
        filter_before = self.registry.active_generation_filter()
        self.assertEqual(filter_before, {"$or": [{"generation_id": {"$exists": False}}]})
        self._replace(legacy_id)
        filter_after = self.registry.active_generation_filter()
        self.assertIn("'$exists': False", str(filter_after))
        self.assertIn(legacy_id, str(filter_after))
        self.assertIn("'$nin'", str(filter_after))

    def test_deleted_document_tombstone_excludes_late_legacy_vectors(self):
        document_id = self._document_id()
        self._replace(document_id)
        self.registry.mark_deleted(document_id)
        filter_value = self.registry.active_generation_filter()
        self.assertEqual(
            filter_value["$or"][0]["$and"],
            [
                {"generation_id": {"$exists": False}},
                {"document_id": {"$nin": [document_id]}},
            ],
        )


if __name__ == "__main__":
    unittest.main()
