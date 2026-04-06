from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.config.settings import AppSettings, get_settings
from src.core.pixeltable_env import ensure_documents_table
from src.processing.page_descriptions import ensure_page_descriptions, is_openrouter_configured
from src.processing.page_images import count_page_images, ensure_page_images_view
from src.processing.text_chunks import count_text_chunks, ensure_text_chunks_view
from src.retrieval.hybrid_retrieval import prepare_retrieval_assets
from src.utils.exceptions import AppError
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    document_name: str
    local_path: str
    status: str


def _compute_document_id(file_path: Path) -> str:
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return digest[:16]


def _normalize_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise AppError(f"Document not found: {path}", status_code=404)
    if not path.is_file():
        raise AppError(f"Expected a file path, got: {path}")
    return path


def _record_exists(documents, document_id: str) -> bool:
    return documents.where(documents.document_id == document_id).count() > 0


def register_document(
    file_path: str | Path,
    settings: AppSettings | None = None,
) -> tuple[str, DocumentRecord]:
    settings = settings or get_settings()
    logger.info("Registering document from path %s", file_path)
    documents = ensure_documents_table(settings)
    normalized_path = _normalize_path(file_path)
    document_id = _compute_document_id(normalized_path)

    record = DocumentRecord(
        document_id=document_id,
        document_name=normalized_path.name,
        local_path=str(normalized_path),
        status="registered",
    )

    if _record_exists(documents, document_id):
        logger.info("Document %s already registered; skipping insert", record.document_name)
        return "skipped", record

    documents.insert(
        [
            {
                "document_id": record.document_id,
                "document_name": record.document_name,
                "local_path": record.local_path,
                "document": record.local_path,
                "status": record.status,
            }
        ]
    )
    logger.info("Document %s inserted into documents table", record.document_name)
    return "inserted", record


def list_documents(settings: AppSettings | None = None):
    settings = settings or get_settings()
    logger.info("Listing registered documents")
    documents = ensure_documents_table(settings)
    result = (
        documents.select(
            documents.document_id,
            documents.document_name,
            documents.local_path,
            documents.status,
        )
        .order_by(documents.document_name)
        .collect()
    )
    return result.to_pandas()


def save_uploaded_file(
    file_name: str,
    content: bytes,
    settings: AppSettings | None = None,
) -> Path:
    settings = settings or get_settings()
    destination = settings.uploads_dir / file_name
    counter = 1
    while destination.exists():
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix
        destination = settings.uploads_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    logger.info("Writing uploaded document to %s", destination)
    destination.write_bytes(content)
    return destination


def ingest_document(file_path: str | Path, settings: AppSettings | None = None) -> dict[str, object]:
    settings = settings or get_settings()
    logger.info("Starting full ingestion pipeline for %s", file_path)
    action, record = register_document(file_path, settings)
    ensure_text_chunks_view(settings)
    ensure_page_images_view(settings)
    vision_status = "skipped"
    if is_openrouter_configured(settings):
        ensure_page_descriptions(settings)
        vision_status = "ready"
    retrieval_status = prepare_retrieval_assets(settings)
    result = {
        "document_name": record.document_name,
        "registration_action": action,
        "text_chunks": count_text_chunks(settings),
        "page_images": count_page_images(settings),
        "vision_status": vision_status,
        "retrieval_status": retrieval_status,
    }
    logger.info("Completed ingestion pipeline for %s", record.document_name)
    return result
