from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf

from src.config.settings import AppSettings, get_settings
from src.processing.page_images import ensure_page_images_view
from src.utils.logging import get_logger

logger = get_logger(__name__)


VISION_PROMPT = (
    "Analyze this PDF page carefully. Describe all meaningful information visible on the page. "
    "Include headings, body text that is clearly readable, tables, charts, graphs, legends, axes, "
    "callouts, figures, and image content. If the page contains a table or chart, summarize its "
    "structure and key takeaways. Be factual and concise."
)


def is_openrouter_configured(settings: AppSettings | None = None) -> bool:
    settings = settings or get_settings()
    return bool(settings.openrouter_api_key)


def _has_column(table, column_name: str) -> bool:
    columns = table.columns() if callable(table.columns) else table.columns
    return column_name in columns


def ensure_page_descriptions(settings: AppSettings | None = None):
    from pixeltable.functions import openrouter

    settings = settings or get_settings()
    if not is_openrouter_configured(settings):
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    page_images = ensure_page_images_view(settings)
    if not _has_column(page_images, "image_data_url"):
        logger.info("Creating image_data_url computed column")
        page_images.add_computed_column(
            image_data_url=pxtf.string.format(
                "data:image/jpeg;base64,{0}",
                page_images.image.b64_encode("jpeg"),
            )
        )

    if not _has_column(page_images, "vision_description"):
        logger.info("Creating vision_description computed column with OpenRouter")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": page_images.image_data_url}},
                ],
            }
        ]
        page_images.add_computed_column(
            vision_description=openrouter.chat_completions(
                messages=messages,
                model=settings.models.vision_model,
                model_kwargs={"max_tokens": 500, "temperature": 0.1},
            )
            .choices[0]
            .message.content
        )

    if not _has_column(page_images, "vision_text"):
        logger.info("Creating vision_text computed column")
        page_images.add_computed_column(
            vision_text=page_images.vision_description.astype(pxt.String)
        )
    return page_images


def preview_page_descriptions(settings: AppSettings | None = None, limit: int = 5):
    page_images = ensure_page_descriptions(settings)
    result = (
        page_images.select(
            page_images.document_name,
            page_images.page,
            page_images.pos,
            page_images.vision_text,
        )
        .order_by(page_images.document_name, page_images.pos)
        .limit(limit)
        .collect()
    )
    return result.to_pandas()
