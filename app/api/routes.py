import secrets
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Request, Response
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit
from app.services.conversations import (
    create_conversation,
    new_owner_token,
    owns_conversation,
    read_messages,
)

router = APIRouter(prefix="/api")
settings = get_settings()

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    conversation_id: str | None = None


OWNER_COOKIE = "conversation_owner"


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
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Document upload is not configured")
    if not secrets.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}
