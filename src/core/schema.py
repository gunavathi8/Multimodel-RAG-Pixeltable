from __future__ import annotations

from typing import Any

import pixeltable as pxt


def documents_schema() -> dict[str, Any]:
    return {
        "document_id": pxt.String,
        "document_name": pxt.String,
        "local_path": pxt.String,
        "document": pxt.Document,
        "status": pxt.String,
    }
