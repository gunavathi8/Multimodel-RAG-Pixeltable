from __future__ import annotations

import pixeltable as pxt
from pixeltable.functions.document import document_splitter

from src.config.settings import AppSettings, get_settings
from src.core.pixeltable_env import ensure_documents_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_text_chunks_view(settings: AppSettings | None = None):
    settings = settings or get_settings()
    documents = ensure_documents_table(settings)
    view_name = f"{settings.pixeltable_namespace}.text_chunks"

    try:
        logger.info("Using text chunks view %s", view_name)
        return pxt.get_table(view_name)
    except Exception:
        logger.info("Creating text chunks view %s", view_name)
        return pxt.create_view(
            view_name,
            documents,
            iterator=document_splitter(
                documents.document,
                separators="sentence,token_limit",
                limit=1000,
                overlap=100,
                metadata="page,bounding_box",
            ),
        )


def rebuild_text_chunks_view(settings: AppSettings | None = None):
    settings = settings or get_settings()
    documents = ensure_documents_table(settings)
    view_name = f"{settings.pixeltable_namespace}.text_chunks"
    logger.info("Rebuilding text chunks view %s", view_name)
    return pxt.create_view(
        view_name,
        documents,
        iterator=document_splitter(
            documents.document,
            separators="sentence,token_limit",
            limit=1000,
            overlap=100,
            metadata="page,bounding_box",
        ),
        if_exists="replace_force",
    )


def drop_text_chunks_view(settings: AppSettings | None = None) -> str:
    settings = settings or get_settings()
    view_name = f"{settings.pixeltable_namespace}.text_chunks"
    logger.info("Dropping text chunks view %s", view_name)
    pxt.drop_table(view_name, if_not_exists="ignore")
    return view_name


def count_text_chunks(settings: AppSettings | None = None) -> int:
    chunks = ensure_text_chunks_view(settings)
    return chunks.count()


def preview_text_chunks(settings: AppSettings | None = None, limit: int = 10):
    chunks = ensure_text_chunks_view(settings)
    result = (
        chunks.select(
            chunks.document_name,
            chunks.page,
            chunks.pos,
            chunks.text,
        )
        .order_by(chunks.document_name, chunks.pos)
        .limit(limit)
        .collect()
    )
    return result.to_pandas()
