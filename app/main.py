from contextlib import asynccontextmanager
import logging
from threading import RLock
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import BASE_DIR, get_settings
from app.core.logging import configure_logging
from app.services.audit import init_db
from app.rag.workflow import build_graph
from app.services.conversations import open_conversation_store
from app.services.document_registry import DocumentRegistry
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            init_db()
        except Exception as exc:
            logger.error("Audit database initialization failed (%s)", type(exc).__name__)
        connection = open_conversation_store(settings.conversation_db_path)
        try:
            checkpointer = SqliteSaver(
                connection,
                serde=JsonPlusSerializer(allowed_msgpack_modules=None),
            )
            checkpointer.setup()
            app.state.conversation_store = connection
            app.state.conversation_lock = RLock()
            app.state.document_registry = DocumentRegistry(settings.document_registry_db_path)
            app.state.agent_graph = build_graph(checkpointer=checkpointer)
            yield
        finally:
            connection.close()

    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def assign_request_id(request: Request, call_next):
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.include_router(router)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "app_name": settings.app_name},
        )

    return app


app = create_app()
