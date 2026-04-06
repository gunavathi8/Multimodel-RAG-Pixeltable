from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx

from src.config.settings import get_settings
from src.utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


def _wait_for_backend(base_url: str, timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=5.0)
            if response.is_success:
                logger.info("Backend is healthy at %s", base_url)
                return
        except Exception:
            time.sleep(1)
            continue
        time.sleep(1)
    raise RuntimeError(f"Backend did not become healthy within {timeout_seconds} seconds")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting Multimodal RAG application")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(settings.project_root)

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:create_app",
        "--factory",
        "--host",
        settings.api_host,
        "--port",
        str(settings.api_port),
    ]
    logger.info("Launching FastAPI backend: %s", " ".join(backend_cmd))
    backend = subprocess.Popen(backend_cmd, cwd=settings.project_root, env=env)

    try:
        _wait_for_backend(f"http://{settings.api_host}:{settings.api_port}")
        streamlit_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/ui/Home.py",
            "--server.port",
            str(settings.streamlit_port),
        ]
        logger.info("Launching Streamlit frontend: %s", " ".join(streamlit_cmd))
        frontend = subprocess.Popen(streamlit_cmd, cwd=settings.project_root, env=env)
        frontend.wait()
    finally:
        logger.info("Stopping backend process")
        backend.terminate()
        backend.wait(timeout=10)


if __name__ == "__main__":
    main()
