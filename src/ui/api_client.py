from __future__ import annotations

import httpx

from src.config.settings import get_settings
from src.utils.exceptions import AppError
from src.utils.logging import configure_logging, get_logger


configure_logging(get_settings().log_level)
logger = get_logger(__name__)


def _base_url() -> str:
    settings = get_settings()
    return f"http://{settings.api_host}:{settings.api_port}"


def api_get(path: str, *, params: dict[str, object] | None = None) -> object:
    url = f"{_base_url()}{path}"
    logger.info("GET %s", url)
    with httpx.Client(timeout=120.0) as client:
        response = client.get(url, params=params)
    if response.is_error:
        raise AppError(f"API GET failed for {path}", status_code=response.status_code, details=response.text)
    return response.json()


def api_post(path: str, *, json: dict[str, object] | None = None, files: dict[str, tuple[str, bytes, str]] | None = None) -> object:
    url = f"{_base_url()}{path}"
    logger.info("POST %s", url)
    with httpx.Client(timeout=600.0) as client:
        response = client.post(url, json=json, files=files)
    if response.is_error:
        raise AppError(f"API POST failed for {path}", status_code=response.status_code, details=response.text)
    return response.json()
