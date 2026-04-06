from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ModelSettings:
    chat_model: str = os.getenv("MMRAG_CHAT_MODEL", "qwen/qwen3-14b")
    vision_model: str = os.getenv("MMRAG_VISION_MODEL", "qwen/qwen3-vl-8b-instruct")
    embedding_model: str = os.getenv(
        "MMRAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    pixeltable_home: Path
    data_dir: Path
    uploads_dir: Path
    pixeltable_namespace: str
    openrouter_api_key: str | None
    api_host: str
    api_port: int
    streamlit_port: int
    log_level: str
    models: ModelSettings


def get_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parents[2]
    pixeltable_home = Path(
        os.getenv("PIXELTABLE_HOME", project_root / ".pixeltable")
    ).expanduser()
    data_dir = project_root / "data"
    uploads_dir = data_dir / "uploads"
    namespace = os.getenv("MMRAG_PIXELTABLE_NAMESPACE", "multimodal_rag")

    return AppSettings(
        project_root=project_root,
        pixeltable_home=pixeltable_home,
        data_dir=data_dir,
        uploads_dir=uploads_dir,
        pixeltable_namespace=namespace,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        api_host=os.getenv("MMRAG_API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("MMRAG_API_PORT", "8000")),
        streamlit_port=int(os.getenv("MMRAG_STREAMLIT_PORT", "8501")),
        log_level=os.getenv("MMRAG_LOG_LEVEL", "INFO"),
        models=ModelSettings(),
    )
