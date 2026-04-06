from __future__ import annotations

import os

import pixeltable as pxt

from src.config.settings import AppSettings, get_settings
from src.core.schema import documents_schema
from src.utils.logging import get_logger

logger = get_logger(__name__)


def configure_environment(settings: AppSettings | None = None) -> AppSettings:
    settings = settings or get_settings()
    logger.info("Configuring Pixeltable environment at %s", settings.pixeltable_home)
    os.environ["PIXELTABLE_HOME"] = str(settings.pixeltable_home)
    settings.pixeltable_home.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings


def ensure_namespace(settings: AppSettings | None = None) -> str:
    settings = configure_environment(settings)
    existing_dirs = set(pxt.list_dirs())
    if settings.pixeltable_namespace not in existing_dirs:
        logger.info("Creating Pixeltable namespace %s", settings.pixeltable_namespace)
        pxt.create_dir(settings.pixeltable_namespace)
    return settings.pixeltable_namespace


def ensure_documents_table(settings: AppSettings | None = None):
    settings = configure_environment(settings)
    namespace = ensure_namespace(settings)
    table_name = f"{namespace}.documents"
    try:
        logger.info("Using documents table %s", table_name)
        return pxt.get_table(table_name)
    except Exception:
        logger.info("Creating documents table %s", table_name)
        return pxt.create_table(table_name, documents_schema())


def bootstrap_pixeltable(settings: AppSettings | None = None) -> dict[str, str]:
    settings = configure_environment(settings)
    namespace = ensure_namespace(settings)
    ensure_documents_table(settings)
    table_name = f"{namespace}.documents"
    logger.info("Pixeltable bootstrap complete for namespace %s", namespace)
    return {
        "pixeltable_home": str(settings.pixeltable_home),
        "uploads_dir": str(settings.uploads_dir),
        "namespace": namespace,
        "documents_table": table_name,
    }


def format_bootstrap_report(report: dict[str, str]) -> str:
    lines = [
        "Pixeltable bootstrap complete.",
        f"PIXELTABLE_HOME: {report['pixeltable_home']}",
        f"Uploads directory: {report['uploads_dir']}",
        f"Namespace: {report['namespace']}",
        f"Documents table: {report['documents_table']}",
    ]
    return "\n".join(lines)


def main() -> None:
    report = bootstrap_pixeltable()
    print(format_bootstrap_report(report))


if __name__ == "__main__":
    main()
