"""
Session 2 — ingest every .txt file in a folder via POST /ingest, then report the vector store total.

document_id comes from the file's "Document ID: ..." header line (e.g. POL-101), falling back to
the file name, so re-running the script replaces each document instead of duplicating it.

Usage (server must be running):
    python ingest_folder.py northwind-sample-docs
    python ingest_folder.py northwind-sample-docs --url https://tai-deployments.onrender.com
"""

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

from ingest_file import post_file

DOCUMENT_ID_LINE = re.compile(r"^Document ID:\s*(\S+)", re.MULTILINE)


def stable_document_id(file: Path) -> str:
    match = DOCUMENT_ID_LINE.search(file.read_text(encoding="utf-8"))
    return match.group(1) if match else file.stem


def vector_counts(url: str) -> tuple[int, int]:
    """(total in index, in the namespace /ask searches), from /health/pinecone."""

    report = httpx.get(f"{url}/health/pinecone", timeout=30.0).json()
    return report.get("total_vector_count", 0), report.get("namespace_vector_count", 0)


def settled_vector_counts(url: str, attempts: int = 10) -> tuple[int, int]:
    """Pinecone's stats lag writes by a few seconds; wait until two readings agree."""

    previous = vector_counts(url)
    for _ in range(attempts):
        time.sleep(3)
        current = vector_counts(url)
        if current == previous:
            return current
        previous = current
    return previous


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all .txt files in a folder.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to a legacy code page.

    files = sorted(args.folder.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files in {args.folder}")

    failures = 0
    total_chunks = 0
    print(f"{'file':32} {'document_id':12} chunks")
    for file in files:
        document_id = stable_document_id(file)
        try:
            response = post_file(args.url, file, document_id)
        except httpx.ConnectError:
            sys.exit(f"Could not reach {args.url} - is uvicorn running?")
        if response.status_code == 200:
            chunks = response.json()["chunks_indexed"]
            total_chunks += chunks
            print(f"{file.name:32} {document_id:12} {chunks}")
        else:
            failures += 1
            print(f"{file.name:32} {document_id:12} FAILED HTTP {response.status_code}: {response.text}")

    print(f"\nChunks ingested this run: {total_chunks} from {len(files) - failures} file(s)")
    total, in_namespace = settled_vector_counts(args.url)
    print(f"Total chunks in the vector store: {total} ({in_namespace} in the namespace /ask searches)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
