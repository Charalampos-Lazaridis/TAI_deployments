"""Session 2 — Pinecone vector store + OpenAI embeddings, configured only via env vars."""

import os
from functools import lru_cache
from typing import Any

from openai import OpenAI
from pinecone import Pinecone, PineconeError

# Fixed in code, not env, so ingest and query can never drift onto different embedding spaces.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536  # Your Pinecone index must be created with exactly this dimension.
EMBEDDING_PRICE_PER_1K = 0.00002  # USD per 1K input tokens (OpenAI list price).
BATCH_SIZE = 100  # Chunks per embeddings call / upsert request — well under both APIs' limits.


class VectorStoreConfigError(RuntimeError):
    """Raised when a required Pinecone env var is missing, so the cause is obvious."""


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise VectorStoreConfigError(f"Missing required environment variable: {name}")
    return value


def get_namespace() -> str:
    """Optional namespace so several experiments can share one index without mixing."""

    return os.getenv("PINECONE_NAMESPACE", "").strip()


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    """One Pinecone client per process; created lazily so /ask still boots without Pinecone."""

    return Pinecone(api_key=_require_env("PINECONE_API_KEY"))


@lru_cache(maxsize=1)
def get_index():
    """Resolve the index host once, then reuse the data-plane connection."""

    pc = get_pinecone()
    description = pc.describe_index(_require_env("PINECONE_INDEX_NAME"))
    return pc.Index(host=description.host)


def embed_texts(client: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    """Embed texts in batches; returns vectors (same order as texts) plus tokens billed."""

    vectors: list[list[float]] = []
    tokens = 0
    for start in range(0, len(texts), BATCH_SIZE):
        response = client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts[start : start + BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in response.data)
        tokens += response.usage.total_tokens if response.usage else 0
    return vectors, tokens


def chunk_vector_id(document_id: str, chunk_index: int) -> str:
    """Deterministic IDs, so re-ingesting a document overwrites its chunks instead of duplicating."""

    return f"{document_id}#chunk-{chunk_index}"


def upsert_vectors(
    ids: list[str], vectors: list[list[float]], metadatas: list[dict[str, Any]]
) -> int:
    """Write vectors with their metadata (store the chunk text here so queries can return it)."""

    records = [
        {"id": vid, "values": vec, "metadata": meta}
        for vid, vec, meta in zip(ids, vectors, metadatas, strict=True)
    ]
    get_index().upsert(
        vectors=records, namespace=get_namespace(), batch_size=BATCH_SIZE, show_progress=False
    )
    return len(records)


def delete_stale_chunks(document_id: str, keep_ids: set[str]) -> int:
    """
    After re-ingesting a shorter version of a document, remove its leftover higher-index chunks.
    Runs after upsert, so a failure here never leaves the document missing.
    """

    index = get_index()
    namespace = get_namespace()
    existing = [
        item.id
        for page in index.list(prefix=f"{document_id}#chunk-", namespace=namespace)
        for item in page.vectors
        if item.id
    ]
    stale = [vid for vid in existing if vid not in keep_ids]
    for start in range(0, len(stale), 1000):  # Pinecone deletes at most 1000 IDs per call.
        index.delete(ids=stale[start : start + 1000], namespace=namespace)
    return len(stale)


def query_vectors(vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Nearest-neighbour search; returns plain dicts so callers don't depend on SDK types."""

    result = get_index().query(
        vector=vector, top_k=top_k, namespace=get_namespace(), include_metadata=True
    )
    return [
        {"id": match.id, "score": match.score, "metadata": match.metadata or {}}
        for match in result.matches
    ]


def pinecone_health() -> dict[str, Any]:
    """
    Debug check: is Pinecone configured, reachable, ready, and sized for our embeddings?
    Never includes the API key in its output.
    """

    report: dict[str, Any] = {
        "ok": False,
        "index_name": os.getenv("PINECONE_INDEX_NAME", "").strip() or None,
        "namespace": get_namespace(),
        "embedding_model": EMBEDDING_MODEL,
        "expected_dimension": EMBEDDING_DIMENSION,
    }

    try:
        pc = get_pinecone()
        description = pc.describe_index(_require_env("PINECONE_INDEX_NAME"))
        report.update(
            host=description.host,
            ready=description.status.ready,
            state=description.status.state,
            metric=description.metric,
            index_dimension=description.dimension,
            dimension_matches=description.dimension == EMBEDDING_DIMENSION,
        )

        stats = get_index().describe_index_stats()
        # Stats report the blank namespace under the key "__default__".
        namespace_summary = stats.namespaces.get(get_namespace() or "__default__")
        report.update(
            total_vector_count=stats.total_vector_count,
            namespace_vector_count=namespace_summary.vector_count if namespace_summary else 0,
        )

        report["ok"] = bool(report["ready"] and report["dimension_matches"])
        if not report["dimension_matches"]:
            report["error"] = (
                f"Index dimension {description.dimension} != {EMBEDDING_DIMENSION} "
                f"required by {EMBEDDING_MODEL}; recreate the index."
            )
    except (VectorStoreConfigError, PineconeError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    return report
