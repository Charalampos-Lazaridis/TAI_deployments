"""
Session 2 — send a text/Markdown file to a running /ingest, so you don't paste JSON by hand.

Usage (server must be running):
    python ingest_file.py my_doc.txt
    python ingest_file.py my_doc.txt --document-id my-doc --chunk-size 400
    python ingest_file.py my_doc.txt --url https://tai-deployments.onrender.com
"""

import argparse
import sys
from pathlib import Path

import httpx


def post_file(
    url: str,
    file: Path,
    document_id: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> httpx.Response:
    """POST one file to /ingest; shared with ingest_folder.py."""

    payload = {
        "text": file.read_text(encoding="utf-8"),
        "document_id": document_id,
        "source": file.name,
    }
    # Only send chunk settings when given, so the server's defaults apply otherwise.
    if chunk_size is not None:
        payload["chunk_size"] = chunk_size
    if overlap is not None:
        payload["chunk_overlap"] = overlap
    return httpx.post(f"{url}/ingest", json=payload, timeout=120.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a file to POST /ingest.")
    parser.add_argument("file", type=Path, help="Text or Markdown file to ingest")
    parser.add_argument("--document-id", help="Defaults to the file name without extension")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--overlap", type=int)
    args = parser.parse_args()

    try:
        response = post_file(
            args.url, args.file, args.document_id or args.file.stem, args.chunk_size, args.overlap
        )
    except httpx.ConnectError:
        sys.exit(f"Could not reach {args.url} - is uvicorn running?")

    print(f"HTTP {response.status_code}")
    print(response.text)


if __name__ == "__main__":
    main()
