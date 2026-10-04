"""Session 2 — split documents into overlapping chunks before embedding."""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter


def default_chunk_settings() -> tuple[int, int]:
    """Env-configurable defaults (CHUNK_SIZE / CHUNK_OVERLAP); read at call time so .env applies."""

    return int(os.getenv("CHUNK_SIZE", "800")), int(os.getenv("CHUNK_OVERLAP", "100"))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split on paragraphs, then lines, then words, so chunks break at natural boundaries."""

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
