import secrets
import re
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import (
    delete_document_vectors,
)
from app.services.ingestion import (
    content_version,
    stable_document_id,
    SUPPORTED,
)
from app.services.document_registry import AmbiguousDocumentError
from app.services.document_replacement import ReplacementError, replace_document
from app.services.audit import write_audit, write_ingestion_audit
from app.services.conversations import (
    create_conversation,
    new_owner_token,
    owns_conversation,
    read_messages,
)

router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    conversation_id: str | None = None


OWNER_COOKIE = "conversation_owner"


async def _audit_ingestion_safely(
    document_id: str,
    source_filename: str,
    document_version: str,
    status: str,
    vision_status: str,
    diagnostics: list[str],
) -> None:
    try:
        await run_in_threadpool(
            write_ingestion_audit,
            document_id,
            source_filename,
            document_version,
            status,
            vision_status,
            diagnostics,
        )
    except Exception as exc:
        logger.warning("Could not write ingestion audit record (%s)", type(exc).__name__)


def get_owner_token(request: Request, response: Response, *, create: bool) -> str | None:
    token = request.cookies.get(OWNER_COOKIE)
    if token:
        return token
    if not create:
        return None
    token = new_owner_token()
    response.set_cookie(
        OWNER_COOKIE,
        token,
        max_age=60 * 60 * 24 * 365,
        httponly=True,
        secure=settings.app_env.lower() not in {"development", "local"},
        samesite="lax",
        path="/",
    )
    return token


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/conversations", status_code=201)
def new_conversation(request: Request, response: Response):
    owner = get_owner_token(request, response, create=True)
    with request.app.state.conversation_lock:
        thread_id = create_conversation(request.app.state.conversation_store, owner)
    return {"conversation_id": thread_id}


@router.get("/conversations/{conversation_id}")
def resume_conversation(conversation_id: str, request: Request, response: Response):
    owner = get_owner_token(request, response, create=False)
    with request.app.state.conversation_lock:
        if not owner or not owns_conversation(
            request.app.state.conversation_store, conversation_id, owner
        ):
            raise HTTPException(status_code=404, detail="Conversation not found")
        try:
            messages = read_messages(request.app.state.agent_graph, conversation_id)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
    return {"conversation_id": conversation_id, "messages": messages}


@router.post("/chat")
def chat(payload: ChatRequest, request: Request, response: Response):
    try:
        owner = get_owner_token(
            request, response, create=payload.conversation_id is None
        )
        conversation_id = payload.conversation_id
        with request.app.state.conversation_lock:
            if conversation_id is None:
                conversation_id = create_conversation(
                    request.app.state.conversation_store, owner
                )
            elif not owner or not owns_conversation(
                request.app.state.conversation_store, conversation_id, owner
            ):
                raise HTTPException(status_code=404, detail="Conversation not found")

            result = ask(
                payload.question,
                thread_id=conversation_id,
                graph=request.app.state.agent_graph,
            )
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
            "conversation_id": conversation_id,
        }

    except HTTPException:
        raise
    except Exception as exc:
        status_code = 400 if isinstance(exc, ValueError) else 500
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc



@router.post("/ingest")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    new_document: bool = Form(default=False),
    x_admin_key: str = Header(default=""),
):
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Document upload is not configured")
    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    original_filename = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    if not original_filename or original_filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded document is empty")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    version = content_version(content)
    try:
        document_id = request.app.state.document_registry.resolve_document(
            original_filename,
            document_id=document_id,
            new_document=new_document,
            legacy_id=stable_document_id(original_filename),
        )
    except AmbiguousDocumentError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "document_ids": exc.document_ids},
        ) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    document_dir = upload_dir / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    dest = document_dir / f"{version}{suffix}"
    dest.write_bytes(content)

    try:
        replacement = await run_in_threadpool(
            replace_document,
            request.app.state.document_registry,
            dest,
            original_filename,
            content,
            document_id,
        )
        result = replacement["result"]
        chunks = replacement["chunks"]
        ids = replacement["ids"]
    except ReplacementError as exc:
        await _audit_ingestion_safely(
            document_id, original_filename, version, "failed", "not_processed", [exc.stage],
        )
        code = 502 if exc.stage in {"upsert", "activation", "cleanup"} else 422
        raise HTTPException(
            status_code=code,
            detail={
                "message": str(exc),
                "document_id": document_id,
                "generation_id": exc.generation_id,
                "stage": exc.stage,
                "previous_generation_remains_active": exc.stage != "cleanup",
            },
        ) from exc
    warnings = replacement["warnings"]
    await _audit_ingestion_safely(
        document_id,
        original_filename,
        version,
        "partial" if warnings or replacement["cleanup_pending"] else "completed",
        result.vision_status,
        warnings,
    )
    return {
        "message": "Document indexed",
        "file": original_filename,
        "document_id": document_id,
        "document_version": version,
        "generation_id": replacement["generation_id"],
        "pages": result.page_count,
        "ocr_pages": result.ocr_pages,
        "vision_pages": result.vision_pages,
        "visual_pages_detected": result.visual_pages_detected,
        "vision_status": result.vision_status,
        "chunks": len(chunks),
        "ids_created": len(ids or []),
        "cleanup_pending": replacement["cleanup_pending"],
        "partial": bool(warnings),
        "warnings": warnings,
    }


@router.delete("/ingest/{document_id}")
def delete_ingested_document(
    document_id: str, request: Request, x_admin_key: str = Header(default="")
):
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Document management is not configured")
    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    if not re.fullmatch(r"(?:[a-f0-9]{64}|[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})", document_id):
        raise HTTPException(status_code=400, detail="Invalid document ID")
    try:
        delete_document_vectors(document_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not delete the document's Pinecone vectors ({type(exc).__name__}).",
        ) from exc
    request.app.state.document_registry.mark_deleted(document_id)

    upload_dir = Path(settings.upload_dir).resolve()
    document_dir = (upload_dir / document_id).resolve()
    warnings: list[str] = []
    if document_dir.parent == upload_dir and document_dir.is_dir():
        try:
            for version_file in document_dir.iterdir():
                if version_file.is_file() or version_file.is_symlink():
                    version_file.unlink()
            document_dir.rmdir()
        except OSError as exc:
            warnings.append(
                "Pinecone vectors were deleted, but some local upload files could not "
                f"be removed ({type(exc).__name__})."
            )
    return {
        "message": "Document and all indexed versions deleted",
        "document_id": document_id,
        "warnings": warnings,
    }
