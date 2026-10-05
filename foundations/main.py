"""Week 1 live demo — five stages in one file, built up live in class."""

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI, OpenAIError
from pinecone import PineconeError
from pydantic import BaseModel, Field, ValidationError

from chunking import chunk_text, default_chunk_settings
from rag import GROUNDING_SYSTEM_PROMPT, REFUSAL_ANSWER, build_user_prompt, min_score
from vector_store import (
    EMBEDDING_PRICE_PER_1K,
    VectorStoreConfigError,
    chunk_vector_id,
    delete_stale_chunks,
    embed_texts,
    get_index,
    get_namespace,
    pinecone_health,
    query_vectors,
    upsert_vectors,
)

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool
    citations: list[str]  # Session 2 — chunk IDs ("document_id#chunk-N") the answer is based on.


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.
    top_k: int = Field(default=5, ge=1, le=20)  # Session 2 — chunks to retrieve as context.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int  # Question embedding + generation.
    model: str
    latency_ms: int  # Retrieval + generation.
    cost_usd: float  # Question embedding + generation.
    retrieved_chunk_ids: list[str]  # Chunks given to the model as context (above RAG_MIN_SCORE).


# Pinecone metadata only accepts strings, numbers, booleans, or lists of strings.
MetadataValue = str | int | float | bool | list[str]


class IngestRequest(BaseModel):
    """One document to chunk, embed, and store; chunk settings fall back to env defaults."""

    text: str
    document_id: str
    source: str | None = None  # e.g. the original filename; stored on every chunk.
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)


class IngestResponse(BaseModel):
    """Same cost visibility as /ask, so ingestion spend is never invisible."""

    document_id: str
    chunks_indexed: int
    status: str
    tokens_used: int
    cost_usd: float


class RetrievedChunk(BaseModel):
    """One search hit, flattened so it's easy to eyeball in /docs."""

    rank: int
    score: float  # Cosine similarity: higher = closer; compare scores, not absolute values.
    id: str
    document_id: str | None
    chunk_index: int | None
    source: str | None
    text: str | None


class RetrieveResponse(BaseModel):
    """Retrieval only — no LLM call, so this costs just the question's embedding."""

    question: str
    top_k: int
    namespace: str
    matches: list[RetrievedChunk]
    tokens_used: int
    cost_usd: float


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def build_messages(question: str, system: str | None) -> list[dict[str, str]]:
    """Session 2 — optional system message carries the grounding rules."""

    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": "user", "content": question}]


def call_model_structured(
    question: str, model: str, system: str | None = None
) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=build_messages(question, system),
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(
    question: str, model: str, system: str | None = None
) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=build_messages(
            f"{question}\n\n"
            "Reply with ONLY a JSON object using keys answer, confidence, sources_needed, citations. "
            "Set confidence to the string 'very high' (not a number).",
            system,
        ),
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


HOME_PAGE_TEXT = """All good! The service is online.

HOW TO USE THE /ask ENDPOINT
============================

1. Send a POST request to /ask with a JSON body:

     curl -X POST https://tai-deployments.onrender.com/ask \\
       -H "Content-Type: application/json" \\
       -d '{"question": "What are the objectives of the project?", "model": "gpt-4o-mini"}'

   - "question" (required): what you want to ask. Answers come ONLY from ingested documents.
   - "model" (optional): gpt-4o (default), gpt-4o-mini, or o3-mini.
   - "top_k" (optional): how many document chunks to retrieve as context (default 5).

2. You get back JSON like this:

     {
       "answer": {"answer": "... [doc#chunk-3]", "confidence": 0.9,
                  "sources_needed": false, "citations": ["doc#chunk-3"]},
       "tokens_used": 1650,
       "model": "gpt-4o-mini",
       "latency_ms": 2100,
       "cost_usd": 0.000310,
       "retrieved_chunk_ids": ["doc#chunk-3", "doc#chunk-7", "..."]
     }

   If the documents don't contain the answer, "answer" says so and "citations" is empty.

3. Prefer clicking to typing? Open /docs in your browser,
   pick POST /ask, click "Try it out", fill in the JSON, and hit "Execute".

Note: opening /ask directly in the browser will not work,
because browsers send GET requests and /ask only accepts POST.

4. Open /health/pinecone in your browser to check the vector store connection.

5. Add documents to the knowledge base with POST /ingest:

     curl -X POST https://tai-deployments.onrender.com/ingest \\
       -H "Content-Type: application/json" \\
       -d '{"document_id": "rag-notes", "source": "rag_notes.md", "text": "..."}'

   Returns {"document_id": "...", "chunks_indexed": N, "status": "indexed", ...}

6. Check what retrieval finds (no LLM call) by opening in your browser:

     /debug/retrieve?q=What is RAG?&top_k=5
"""


@app.get("/", response_class=PlainTextResponse)
def home() -> str:
    """Health check plus a short usage guide, so visiting the root URL isn't a 404."""

    return HOME_PAGE_TEXT


@app.get("/health/pinecone")
def health_pinecone() -> JSONResponse:
    """Session 2 debug check — 200 if Pinecone is reachable and sized right, else 503 with why."""

    report = pinecone_health()
    return JSONResponse(content=report, status_code=200 if report["ok"] else 503)


# Example (works in a browser too):
#   curl "http://127.0.0.1:8000/debug/retrieve?q=What%20is%20RAG%3F&top_k=5"
@app.get("/debug/retrieve")
def debug_retrieve(
    q: str = Query(..., description="Question to search for"),
    top_k: int = Query(5, ge=1, le=20),
) -> RetrieveResponse:
    """Embed the question and return the nearest chunks — verifies retrieval without the LLM."""

    question = q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'q' must not be empty.")

    try:
        get_index()  # Fail on missing/bad Pinecone config before paying for the embedding.
        vectors, tokens_used = embed_texts(client, [question])
        hits = query_vectors(vectors[0], top_k=top_k)
    except VectorStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OpenAIError, PineconeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed: {type(exc).__name__}: {exc}"
        ) from exc

    matches = [
        RetrievedChunk(
            rank=rank,
            score=round(hit["score"], 4),
            id=hit["id"],
            document_id=hit["metadata"].get("document_id"),
            # Pinecone returns stored numbers as floats; cast back for readability.
            chunk_index=int(hit["metadata"]["chunk_index"]) if "chunk_index" in hit["metadata"] else None,
            source=hit["metadata"].get("source"),
            text=hit["metadata"].get("text"),
        )
        for rank, hit in enumerate(hits, start=1)
    ]
    return RetrieveResponse(
        question=question,
        top_k=top_k,
        namespace=get_namespace(),
        matches=matches,
        tokens_used=tokens_used,
        cost_usd=round(tokens_used / 1000 * EMBEDDING_PRICE_PER_1K, 6),
    )


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer one question from retrieved document chunks (RAG), with guardrails and cost visibility."""

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="'question' must not be empty.")

    model = body.model or DEFAULT_MODEL
    start = time.perf_counter()

    # Session 2: retrieve once, outside the retry loop — retries only re-run generation.
    try:
        get_index()  # Fail on missing/bad Pinecone config before paying for the embedding.
        vectors, embedding_tokens = embed_texts(client, [question])
        hits = query_vectors(vectors[0], top_k=body.top_k)
    except VectorStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OpenAIError, PineconeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed: {type(exc).__name__}: {exc}"
        ) from exc

    hits = [hit for hit in hits if hit["score"] >= min_score()]
    retrieved_ids = [hit["id"] for hit in hits]
    embedding_cost = embedding_tokens / 1000 * EMBEDDING_PRICE_PER_1K

    # Nothing relevant retrieved: refuse without calling the LLM, so it can't answer from memory.
    if not hits:
        return AskResponse(
            answer=Answer(answer=REFUSAL_ANSWER, confidence=0.0, sources_needed=True, citations=[]),
            tokens_used=embedding_tokens,
            model=model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            cost_usd=round(embedding_cost, 6),
            retrieved_chunk_ids=[],
        )

    grounded_question = build_user_prompt(question, hits)
    last_error: str | None = None

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    grounded_question, model, system=GROUNDING_SYSTEM_PROMPT
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    grounded_question, model, system=GROUNDING_SYSTEM_PROMPT
                )

            # Guardrail: keep only citations that point at chunks we actually supplied.
            answer.citations = [cid for cid in answer.citations if cid in retrieved_ids]

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens) + embedding_cost

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used + embedding_tokens,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


# Example:
#   curl -X POST http://127.0.0.1:8000/ingest \
#     -H "Content-Type: application/json" \
#     -d '{"document_id": "rag-notes", "source": "rag_notes.md",
#          "text": "Retrieval-Augmented Generation (RAG) grounds answers in your own documents..."}'
#
# Optional fields: "metadata": {"author": "..."}, "chunk_size": 800, "chunk_overlap": 100.
# Re-sending the same document_id replaces that document's chunks.
@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk -> embed (text-embedding-3-small) -> upsert into Pinecone."""

    document_id = body.document_id.strip()
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="'text' must not be empty.")
    if not document_id:
        raise HTTPException(status_code=400, detail="'document_id' must not be empty.")
    if "#" in document_id:
        # '#' separates document_id from the chunk number in vector IDs.
        raise HTTPException(status_code=400, detail="'document_id' must not contain '#'.")

    default_size, default_overlap = default_chunk_settings()
    chunk_size = body.chunk_size or default_size
    chunk_overlap = body.chunk_overlap if body.chunk_overlap is not None else default_overlap
    try:
        chunks = chunk_text(body.text, chunk_size, chunk_overlap)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    source = body.source or "unknown"
    ids = [chunk_vector_id(document_id, i) for i in range(len(chunks))]
    # Reserved keys go last so caller metadata can't overwrite them; chunk text is stored for retrieval.
    metadatas = [
        {**body.metadata, "document_id": document_id, "chunk_index": i, "source": source, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]

    try:
        get_index()  # Fail on missing/bad Pinecone config before paying for embeddings.
        vectors, tokens_used = embed_texts(client, chunks)
        chunks_indexed = upsert_vectors(ids, vectors, metadatas)
        delete_stale_chunks(document_id, keep_ids=set(ids))
    except VectorStoreConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OpenAIError, PineconeError) as exc:
        raise HTTPException(
            status_code=502, detail=f"Ingestion failed: {type(exc).__name__}: {exc}"
        ) from exc

    return IngestResponse(
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        status="indexed",
        tokens_used=tokens_used,
        cost_usd=round(tokens_used / 1000 * EMBEDDING_PRICE_PER_1K, 6),
    )
