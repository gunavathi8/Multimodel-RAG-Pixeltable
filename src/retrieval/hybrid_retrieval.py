from __future__ import annotations

from pixeltable.functions.huggingface import sentence_transformer

from src.config.settings import AppSettings, get_settings
from src.processing.page_descriptions import is_openrouter_configured
from src.processing.text_chunks import ensure_text_chunks_view
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_text_chunk_index(settings: AppSettings | None = None):
    settings = settings or get_settings()
    chunks = ensure_text_chunks_view(settings)
    logger.info("Ensuring text embedding index")
    chunks.add_embedding_index(
        "text",
        idx_name="text_idx",
        string_embed=sentence_transformer.using(
            model_id=settings.models.embedding_model
        ),
        if_exists="ignore",
    )
    return chunks


def ensure_page_description_index(settings: AppSettings | None = None):
    settings = settings or get_settings()
    if not is_openrouter_configured(settings):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured, so page descriptions are unavailable."
        )

    from src.processing.page_descriptions import ensure_page_descriptions

    pages = ensure_page_descriptions(settings)
    logger.info("Ensuring page description embedding index")
    pages.add_embedding_index(
        "vision_text",
        idx_name="vision_text_idx",
        string_embed=sentence_transformer.using(
            model_id=settings.models.embedding_model
        ),
        if_exists="ignore",
    )
    return pages


def prepare_retrieval_assets(settings: AppSettings | None = None) -> dict[str, str]:
    settings = settings or get_settings()
    logger.info("Preparing retrieval assets")
    ensure_text_chunk_index(settings)
    status = {"text_index": "ready"}
    if is_openrouter_configured(settings):
        ensure_page_description_index(settings)
        status["vision_index"] = "ready"
    else:
        status["vision_index"] = "skipped_no_openrouter_key"
    return status


def retrieve_text_evidence(
    query_text: str,
    settings: AppSettings | None = None,
    *,
    document_name: str | None = None,
    limit: int = 5,
):
    settings = settings or get_settings()
    logger.info("Retrieving text evidence for query: %s", query_text)
    chunks = ensure_text_chunk_index(settings)
    sim = chunks.text.similarity(query_text, idx="text_idx")
    query = chunks.order_by(sim, asc=False)
    if document_name:
        query = query.where(chunks.document_name == document_name)
    result = (
        query.limit(limit)
        .select(
            chunks.document_name,
            chunks.page,
            chunks.pos,
            chunks.text,
            similarity=sim,
        )
        .collect()
    )
    return result.to_pandas()


def retrieve_visual_evidence(
    query_text: str,
    settings: AppSettings | None = None,
    *,
    document_name: str | None = None,
    limit: int = 5,
):
    settings = settings or get_settings()
    logger.info("Retrieving visual evidence for query: %s", query_text)
    pages = ensure_page_description_index(settings)
    sim = pages.vision_text.similarity(query_text, idx="vision_text_idx")
    query = pages.order_by(sim, asc=False)
    if document_name:
        query = query.where(pages.document_name == document_name)
    result = (
        query.limit(limit)
        .select(
            pages.document_name,
            pages.page,
            pages.pos,
            pages.vision_text,
            similarity=sim,
        )
        .collect()
    )
    return result.to_pandas()
