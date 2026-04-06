from __future__ import annotations

from src.processing.text_chunks import count_text_chunks, ensure_text_chunks_view


def main() -> None:
    chunks = ensure_text_chunks_view()
    print(f"Text chunks view ready: {chunks}")
    print(f"Total chunk rows: {count_text_chunks()}")


if __name__ == "__main__":
    main()
