import pixeltable as pxt
import pixeltable.functions as pxtf
from fastapi.encoders import jsonable_encoder
from uuid import uuid4

from src.config.settings import AppSettings, get_settings
from src.processing.page_descriptions import is_openrouter_configured
from src.retrieval.hybrid_retrieval import (
    ensure_page_description_index,
    ensure_text_chunk_index,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_qa_table(settings: AppSettings | None = None):
    settings = settings or get_settings()
    table_name = f"{settings.pixeltable_namespace}.qa_sessions"
    try:
        logger.info("Using QA table %s", table_name)
        qa = pxt.get_table(table_name)
    except Exception:
        logger.info("Creating QA table %s", table_name)
        qa = pxt.create_table(
            table_name,
            {
                "session_id": pxt.String,
                "session_title": pxt.String,
                "document_name": pxt.String,
                "question": pxt.String,
                "created_at": pxt.Timestamp,
            },
        )
    qa_columns = qa.columns() if callable(qa.columns) else qa.columns
    if "session_id" not in qa_columns:
        logger.info("Migrating QA table to add session_id column")
        qa.add_column(session_id=pxt.String, if_exists="ignore")
    if "session_title" not in qa_columns:
        logger.info("Migrating QA table to add session_title column")
        qa.add_column(session_title=pxt.String, if_exists="ignore")
    if "created_at" not in qa_columns:
        logger.info("Migrating QA table to add created_at column")
        qa.add_column(created_at=pxt.Timestamp, if_exists="ignore")
    return qa


def ensure_qa_pipeline(settings: AppSettings | None = None):
    from pixeltable.functions import openrouter

    settings = settings or get_settings()
    if not is_openrouter_configured(settings):
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")

    logger.info("Ensuring QA pipeline")
    chunks = ensure_text_chunk_index(settings)
    pages = ensure_page_description_index(settings)
    qa = ensure_qa_table(settings)

    qa_columns = qa.columns() if callable(qa.columns) else qa.columns

    if "text_context" not in qa_columns:
        logger.info("Creating text_context computed column")
        @pxt.query
        def text_context_for_question(
            document_name: pxt.String,
            question: pxt.String,
            limit: pxt.Int = 5,
        ):
            sim = chunks.text.similarity(question, idx="text_idx")
            return (
                chunks.where(chunks.document_name == document_name)
                .order_by(sim, asc=False)
                .limit(limit)
                .select(
                    chunks.pos,
                    chunks.text,
                    similarity=sim,
                )
            )

        qa.add_computed_column(
            text_context=text_context_for_question(qa.document_name, qa.question, 5)
        )

    if "visual_context" not in qa_columns:
        logger.info("Creating visual_context computed column")
        @pxt.query
        def visual_context_for_question(
            document_name: pxt.String,
            question: pxt.String,
            limit: pxt.Int = 3,
        ):
            sim = pages.vision_text.similarity(question, idx="vision_text_idx")
            return (
                pages.where(pages.document_name == document_name)
                .order_by(sim, asc=False)
                .limit(limit)
                .select(
                    pages.page,
                    pages.vision_text,
                    similarity=sim,
                )
            )

        qa.add_computed_column(
            visual_context=visual_context_for_question(qa.document_name, qa.question, 3)
        )

    qa_columns = qa.columns() if callable(qa.columns) else qa.columns

    if "final_prompt_v2" not in qa_columns:
        logger.info("Creating final_prompt_v2 computed column")
        qa.add_computed_column(
            final_prompt_v2=pxtf.string.format(
                (
                    "You are answering questions about a selected PDF document.\n"
                    "Use only the retrieved evidence below. If the evidence is insufficient, say so clearly.\n\n"
                    "DOCUMENT:\n{0}\n\n"
                    "QUESTION:\n{1}\n\n"
                    "TEXT EVIDENCE:\n{2}\n\n"
                    "VISUAL EVIDENCE:\n{3}\n\n"
                    "Provide a clear, complete answer grounded in the evidence.\n"
                    "When the question asks for an explanation of a diagram, architecture, or workflow, cover all major components "
                    "that appear in the evidence instead of stopping after the first section."
                ),
                qa.document_name,
                qa.question,
                qa.text_context,
                qa.visual_context,
            )
        )

    qa_columns = qa.columns() if callable(qa.columns) else qa.columns

    if "answer_json_v2" not in qa_columns:
        logger.info("Creating answer_json_v2 computed column")
        messages = [{"role": "user", "content": qa.final_prompt_v2}]
        qa.add_computed_column(
            answer_json_v2=openrouter.chat_completions(
                messages=messages,
                model=settings.models.chat_model,
                model_kwargs={"max_tokens": 1800, "temperature": 0.1},
            )
        )

    qa_columns = qa.columns() if callable(qa.columns) else qa.columns

    if "answer_v2" not in qa_columns:
        logger.info("Creating answer_v2 computed column")
        qa.add_computed_column(answer_v2=qa.answer_json_v2.choices[0].message.content.astype(pxt.String))

    return qa


def ask_question(
    question: str,
    document_name: str,
    settings: AppSettings | None = None,
    *,
    session_id: str | None = None,
    session_title: str | None = None,
):
    from datetime import datetime, timezone

    settings = settings or get_settings()
    logger.info("Asking question against document %s", document_name)
    qa = ensure_qa_pipeline(settings)
    session_id = session_id or str(uuid4())
    session_title = session_title or question.strip()[:80]
    qa.insert(
        [
            {
                "session_id": session_id,
                "session_title": session_title,
                "document_name": document_name,
                "question": question,
                "created_at": datetime.now(timezone.utc),
            }
        ]
    )
    logger.info("Question inserted into QA table")
    result = (
        qa.where((qa.document_name == document_name) & (qa.session_id == session_id))
        .order_by(qa.created_at, asc=False)
        .limit(1)
        .select(
            qa.session_id,
            qa.session_title,
            qa.document_name,
            qa.question,
            qa.created_at,
            qa.text_context,
            qa.visual_context,
            answer=qa.answer_v2,
        )
        .collect()
    )
    return result.to_pandas()


def get_chat_history(
    document_name: str | None,
    settings: AppSettings | None = None,
    *,
    session_id: str | None = None,
    limit: int = 10,
):
    settings = settings or get_settings()
    logger.info("Fetching chat history for document=%s session=%s", document_name, session_id)
    qa = ensure_qa_table(settings)
    qa_columns = qa.columns() if callable(qa.columns) else qa.columns
    if "answer" not in qa_columns:
        if "answer_v2" in qa_columns:
            answer_col = qa.answer_v2
        else:
            return []
    else:
        answer_col = qa.answer_v2 if "answer_v2" in qa_columns else qa.answer

    query = qa.order_by(qa.created_at, asc=False)
    if session_id:
        query = query.where(qa.session_id == session_id)
    elif document_name:
        query = query.where(qa.document_name == document_name)

    result = query.limit(limit).select(
        qa.session_id,
        qa.session_title,
        qa.document_name,
        qa.question,
        created_at=qa.created_at,
        answer=answer_col,
    ).collect()
    history_df = result.to_pandas()
    records = history_df.iloc[::-1].to_dict(orient="records")
    return jsonable_encoder(records)


def list_chat_sessions(settings: AppSettings | None = None, *, limit: int = 100):
    settings = settings or get_settings()
    logger.info("Listing chat sessions")
    qa = ensure_qa_table(settings)
    qa_columns = qa.columns() if callable(qa.columns) else qa.columns
    required = {"session_id", "session_title"}
    if not required.issubset(set(qa_columns)):
        return []

    result = qa.order_by(qa.created_at, asc=False).limit(limit).select(
        qa.session_id,
        qa.session_title,
        qa.document_name,
        created_at=qa.created_at,
    ).collect()
    sessions_df = result.to_pandas()
    if sessions_df.empty:
        return []
    sessions_df = sessions_df.drop_duplicates(subset=["session_id"], keep="first")
    return jsonable_encoder(sessions_df.to_dict(orient="records"))
