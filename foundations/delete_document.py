"""
Session 2 — remove every chunk of one document from Pinecone. Runs locally with your .env keys;
deliberately not an HTTP endpoint, because the deployed service has no authentication.

Usage:
    python delete_document.py digifab
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from vector_store import delete_stale_chunks, get_namespace  # noqa: E402  (needs .env loaded first)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all chunks of a document from Pinecone.")
    parser.add_argument("document_id")
    args = parser.parse_args()

    # Keeping no IDs means every chunk of this document counts as stale.
    deleted = delete_stale_chunks(args.document_id, keep_ids=set())
    print(f"Deleted {deleted} chunk(s) of '{args.document_id}' from namespace '{get_namespace()}'.")


if __name__ == "__main__":
    main()
