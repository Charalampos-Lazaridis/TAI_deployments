"""
Session 2 — preview how /ingest will chunk a document. Free: no OpenAI or Pinecone calls.

Usage:
    python inspect_chunks.py my_doc.txt
    python inspect_chunks.py my_doc.txt --chunk-size 400 --overlap 50
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from chunking import chunk_text, default_chunk_settings

load_dotenv(Path(__file__).resolve().parent / ".env")  # So CHUNK_SIZE/CHUNK_OVERLAP match the server.

PREVIEW = 60  # Characters shown at each chunk boundary.


def shared_overlap(previous: str, current: str) -> int:
    """Length of the longest end of `previous` that `current` starts with."""

    for size in range(min(len(previous), len(current)), 0, -1):
        if previous.endswith(current[:size]):
            return size
    return 0


def main() -> None:
    default_size, default_overlap = default_chunk_settings()
    parser = argparse.ArgumentParser(description="Preview how /ingest will chunk a document.")
    parser.add_argument("file", type=Path, help="Text or Markdown file to chunk")
    parser.add_argument("--chunk-size", type=int, default=default_size)
    parser.add_argument("--overlap", type=int, default=default_overlap)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to a legacy code page.

    text = args.file.read_text(encoding="utf-8")
    chunks = chunk_text(text, args.chunk_size, args.overlap)  # Same function /ingest uses.
    if not chunks:
        print("No chunks — the file is empty or whitespace only.")
        return

    lengths = [len(c) for c in chunks]
    overlaps = [shared_overlap(a, b) for a, b in zip(chunks, chunks[1:])]

    print(f"File: {args.file}  ({len(text)} characters)")
    print(f"Settings: chunk_size={args.chunk_size}, overlap={args.overlap}\n")
    print(f"Chunks: {len(chunks)}")
    print(f"Length: min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)}")
    if overlaps:
        with_overlap = sum(1 for o in overlaps if o > 0)
        print(f"Neighbours sharing text: {with_overlap} of {len(overlaps)} "
              f"(avg {sum(overlaps) // len(overlaps)} chars shared)")

    for i, chunk in enumerate(chunks):
        print(f"\n--- chunk {i}  ({len(chunk)} chars"
              + (f", shares {overlaps[i - 1]} chars with chunk {i - 1}" if i else "") + ") ---")
        print(f"  starts: {chunk[:PREVIEW]!r}")
        print(f"  ends:   {chunk[-PREVIEW:]!r}")


if __name__ == "__main__":
    main()
