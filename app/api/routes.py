import secrets
import re
import logging
from time import perf_counter
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header, Request, Response, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.openrouter_errors import classify_openrouter_failure, http_retry_after
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
from app.services import audit as audit_service
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
    *, request_id: str | None = None, latency_ms: float | None = None,
    error_category: str | None = None,
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
            request_id=request_id,
            latency_ms=latency_ms,
            error_category=error_category,
        )
    except Exception as exc:
        logger.warning("Could not write ingestion audit record (%s)", type(exc).__name__)


def _write_query_audit_safely(*args, **kwargs) -> None:
    try:
        write_audit(*args, **kwargs)
    except Exception as exc:
        logger.warning("Could not write query audit record (%s)", type(exc).__name__)


def _safe_trace(trace: list[str]) -> list[str]:
    """Keep workflow stage labels but exclude rewritten or user-derived query text."""
    return ["Query rewrite" if item.startswith("Query rewrite") else item for item in trace]


def _safe_web_references(citations: list[dict]) -> list[dict[str, str]]:
    references = []
    for citation in citations:
        if not isinstance(citation, dict) or citation.get("type") != "web" or not citation.get("url"):
            continue
        try:
            parts = urlsplit(str(citation["url"]))
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                continue
            host = parts.hostname
            if parts.port:
                host += f":{parts.port}"
            references.append({
                "title": str(citation.get("title") or host)[:300],
                "url": urlunsplit((parts.scheme, host, parts.path, "", "")),
                "type": "web",
            })
        except (TypeError, ValueError):
            continue
    return references


def _require_admin(x_admin_key: str) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Audit inspection is not configured")
    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")


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


@router.get("/admin/audit")
async def inspect_audit(
    request: Request,
    x_admin_key: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    conversation_id: str | None = None,
    request_id: str | None = None,
    outcome: str | None = None,
    selected_route: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
):
    _require_admin(x_admin_key)
    try:
        return await run_in_threadpool(
            audit_service.list_query_audit, limit=limit, offset=offset,
            conversation_id=conversation_id, request_id=request_id, outcome=outcome,
            selected_route=selected_route, from_ts=from_ts, to_ts=to_ts,
        )
    except Exception as exc:
        logger.warning("Audit history query failed request_id=%s (%s)", request.state.request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail=f"Audit history is temporarily unavailable. Request ID: {request.state.request_id}") from exc


@router.get("/admin/audit/metrics")
async def inspect_audit_metrics(
    request: Request,
    x_admin_key: str = Header(default=""),
    from_ts: str | None = None,
    to_ts: str | None = None,
    outcome: str | None = None,
    selected_route: str | None = None,
):
    _require_admin(x_admin_key)
    try:
        return await run_in_threadpool(
            audit_service.audit_metrics, from_ts=from_ts, to_ts=to_ts,
            outcome=outcome, selected_route=selected_route,
        )
    except Exception as exc:
        logger.warning("Audit metrics query failed request_id=%s (%s)", request.state.request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail=f"Audit metrics are temporarily unavailable. Request ID: {request.state.request_id}") from exc


@router.get("/admin/audit/ingestion")
async def inspect_ingestion_audit(
    request: Request,
    x_admin_key: str = Header(default=""),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    document_id: str | None = None,
    request_id: str | None = None,
    status: str | None = None,
    from_ts: str | None = None,
    to_ts: str | None = None,
):
    _require_admin(x_admin_key)
    try:
        return await run_in_threadpool(
            audit_service.list_ingestion_audit, limit=limit, offset=offset,
            document_id=document_id, request_id=request_id, status=status,
            from_ts=from_ts, to_ts=to_ts,
        )
    except Exception as exc:
        logger.warning("Ingestion audit query failed request_id=%s (%s)", request.state.request_id, type(exc).__name__)
        raise HTTPException(status_code=503, detail=f"Ingestion audit is temporarily unavailable. Request ID: {request.state.request_id}") from exc


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
    started = perf_counter()
    conversation_id: str | None = payload.conversation_id
    request_id = request.state.request_id
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
        source_used = result["source_used"]
        references = result.get("retrieval_results", [])
        supporting_ids = set(result.get("kb_supporting_chunk_ids", []))
        evidence = {
            "kb_grade": result.get("kb_grade"),
            "web_grade": result.get("web_grade"),
            "supporting_chunk_ids": sorted(supporting_ids),
            "web_references": _safe_web_references(result.get("citations", [])),
        }
        _write_query_audit_safely(
            payload.question, source_used, _safe_trace(result.get("trace", [])),
            request_id=request_id, conversation_id=conversation_id,
            outcome="insufficient_evidence" if source_used == "insufficient_evidence" else "answered",
            selected_route=result.get("selected_route"),
            retrieval_results=references,
            evidence=evidence,
            web_fallback=source_used == "web_search" or any("Web fallback" in item for item in result.get("trace", [])),
            query_rewrite_count=result.get("query_rewrite_count", 0),
            retry_count=result.get("retry_count", 0),
            latency_ms=(perf_counter() - started) * 1000,
            stage_timings_ms=result.get("stage_timings_ms", {}),
        )

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
        failure = classify_openrouter_failure(exc)
        if failure.status_code == 429:
            status_code = 429
            if failure.category == "openrouter_account_or_model_limit":
                detail = "The configured OpenRouter account or model is rate-limited or out of quota. Please try again later."
            elif failure.category == "upstream_provider_rate_limit":
                detail = "The selected AI provider is temporarily rate-limited. Please wait and try again."
            else:
                detail = "The AI service is rate-limited. Please wait and try again."
        elif failure.category == "transient_upstream_error":
            status_code = 503
            detail = "The AI service is temporarily unavailable. Please wait and try again."
        else:
            status_code = 400 if isinstance(exc, ValueError) else 500
            detail = "The request could not be completed."

        diagnostics = []
        headers = {}
        if status_code in {429, 503}:
            diagnostics.append(f"openrouter:{failure.category}")
            retry_after = http_retry_after(exc)
            if retry_after:
                headers["Retry-After"] = retry_after
            logger.warning(
                "Chat AI service unavailable request_id=%s category=%s provider=%s",
                request_id,
                failure.category,
                failure.provider or "unknown",
            )
        _write_query_audit_safely(
            payload.question, "failed", [], request_id=request_id,
            conversation_id=conversation_id, outcome="failed",
            latency_ms=(perf_counter() - started) * 1000,
            error_category=type(exc).__name__,
            diagnostics=diagnostics or ["workflow_execution_failed"],
        )
        if status_code == 400 or status_code == 500:
            logger.warning("Chat request failed request_id=%s category=%s", request_id, type(exc).__name__)
        if status_code == 500:
            detail += f" Reference request ID: {request_id}"
        raise HTTPException(status_code=status_code, detail=detail, headers=headers) from exc



@router.post("/ingest")
async def ingest(
    request: Request,
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    new_document: bool = Form(default=False),
    x_admin_key: str = Header(default=""),
):
    started = perf_counter()
    request_id = request.state.request_id
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
            document_id, original_filename, version, "failed", "not_processed", [f"stage:{exc.stage}"],
            request_id=request_id,
            latency_ms=(perf_counter() - started) * 1000,
            error_category=exc.stage,
        )
        code = 502 if exc.stage in {"upsert", "activation", "cleanup"} else 422
        raise HTTPException(
            status_code=code,
            detail={
                "message": f"Document ingestion failed during {exc.stage}.",
                "document_id": document_id,
                "generation_id": exc.generation_id,
                "stage": exc.stage,
                "previous_generation_remains_active": exc.stage != "cleanup",
            },
        ) from exc
    except Exception as exc:
        await _audit_ingestion_safely(
            document_id, original_filename, version, "failed", "not_processed",
            ["stage:unexpected"], request_id=request_id,
            latency_ms=(perf_counter() - started) * 1000,
            error_category=type(exc).__name__,
        )
        logger.warning("Document ingestion failed request_id=%s category=%s", request_id, type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed. Reference request ID: {request_id}",
        ) from exc
    warnings = replacement["warnings"]
    text_ocr_types = {"text", "table", "ocr", "screenshot_ocr"}
    text_ocr_chunk_count = sum(
        1 for chunk in chunks if chunk.metadata.get("content_type") in text_ocr_types
    )
    await _audit_ingestion_safely(
        document_id,
        original_filename,
        version,
        "partial" if warnings or replacement["cleanup_pending"] else "completed",
        result.vision_status,
        warnings,
        request_id=request_id,
        latency_ms=(perf_counter() - started) * 1000,
        error_category=None,
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
        "vision_analysis_status": result.vision_status,
        "vision_failed_pages": result.vision_failed_pages,
        "document_index_status": "completed",
        "text_ocr_index_status": "completed" if text_ocr_chunk_count else "no_content",
        "text_ocr_chunks": text_ocr_chunk_count,
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
