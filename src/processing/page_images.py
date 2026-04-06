from __future__ import annotations

import pixeltable as pxt
from pixeltable.functions.document import document_splitter

from src.config.settings import AppSettings, get_settings
from src.core.pixeltable_env import ensure_documents_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_page_images_view(settings: AppSettings | None = None):
    settings = settings or get_settings()
    documents = ensure_documents_table(settings)
    view_name = f"{settings.pixeltable_namespace}.page_images"

    try:
        logger.info("Using page images view %s", view_name)
        return pxt.get_table(view_name)
    except Exception:
        logger.info("Creating page images view %s", view_name)
        return pxt.create_view(
            view_name,
            documents,
            iterator=document_splitter(
                documents.document,
                separators="page",
                elements=["image"],
                metadata="page",
            ),
        )


def count_page_images(settings: AppSettings | None = None) -> int:
    page_images = ensure_page_images_view(settings)
    return page_images.count()


def preview_page_images(settings: AppSettings | None = None, limit: int = 5):
    page_images = ensure_page_images_view(settings)
    result = (
        page_images.select(
            page_images.document_name,
            page_images.page,
            page_images.pos,
            page_images.image,
        )
        .order_by(page_images.document_name, page_images.pos)
        .limit(limit)
        .collect()
    )
    return result.to_pandas()
