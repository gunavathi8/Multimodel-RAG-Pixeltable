from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.chat.qa_service import ask_question, get_chat_history, list_chat_sessions
from src.config.settings import get_settings
from src.core.pixeltable_env import bootstrap_pixeltable
from src.ingestion.service import ingest_document, list_documents, save_uploaded_file
from src.utils.exceptions import AppError
from src.utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


class ChatAskRequest(BaseModel):
    document_name: str
    question: str
    session_id: str | None = None
    session_title: str | None = None


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Creating FastAPI application")

    app = FastAPI(title="Multimodal RAG API", version="1.0.0")

    @app.exception_handler(AppError)
    async def app_error_handler(_, exc: AppError):
        logger.exception("Application error: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "extra": exc.details},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(_, exc: Exception):
        logger.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info("Bootstrapping application state")
        bootstrap_pixeltable(settings)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/documents")
    async def documents() -> list[dict[str, object]]:
        logger.info("GET /documents")
        return jsonable_encoder(list_documents(settings).to_dict(orient="records"))

    @app.post("/documents/ingest")
    async def ingest(file: UploadFile = File(...)) -> dict[str, object]:
        logger.info("POST /documents/ingest for %s", file.filename)
        payload = await file.read()
        destination = save_uploaded_file(Path(file.filename).name, payload, settings)
        return jsonable_encoder(ingest_document(destination, settings))

    @app.get("/chat/history")
    async def chat_history(
        document_name: str | None = Query(default=None),
        session_id: str | None = Query(default=None),
    ) -> list[dict[str, object]]:
        logger.info("GET /chat/history for document=%s session=%s", document_name, session_id)
        return get_chat_history(document_name, settings, session_id=session_id, limit=50)

    @app.get("/chat/sessions")
    async def chat_sessions() -> list[dict[str, object]]:
        logger.info("GET /chat/sessions")
        return list_chat_sessions(settings, limit=100)

    @app.post("/chat/ask")
    async def chat_ask(payload: ChatAskRequest) -> dict[str, object]:
        document_name = payload.document_name
        question = payload.question
        session_id = payload.session_id
        session_title = payload.session_title
        logger.info("POST /chat/ask for %s", document_name)
        result = ask_question(
            question,
            document_name,
            settings,
            session_id=session_id,
            session_title=session_title,
        )
        return jsonable_encoder(result.iloc[0].to_dict())

    return app
