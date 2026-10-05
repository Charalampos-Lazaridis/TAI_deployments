"""Session 2 — minimal Streamlit UI for the RAG API. All RAG logic stays in the FastAPI service.

Run this page:
  streamlit run rag_ui.py

API URL comes from RAG_API_URL (env or .env) and can be changed in the sidebar.
"""

import json
import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_API_URL = os.getenv("RAG_API_URL", "https://tai-deployments.onrender.com")
TIMEOUT = 120.0  # Render's free tier can take ~30-60 s to wake from sleep.


def call_api(method: str, base_url: str, path: str, **kwargs) -> tuple[int, dict | str]:
    """One place for HTTP + error handling; status 0 means the request never got a response."""

    try:
        response = httpx.request(method, f"{base_url.rstrip('/')}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.ConnectError:
        return 0, f"Cannot reach {base_url}. Is the server running and the URL correct?"
    except httpx.HTTPError as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    try:
        return response.status_code, response.json()
    except json.JSONDecodeError:
        return response.status_code, response.text


def show_error(status: int, data: dict | str) -> None:
    detail = data.get("detail", data) if isinstance(data, dict) else data
    st.error(f"HTTP {status}: {detail}" if status else str(detail))


def show_ask_result(data: dict) -> None:
    answer = data["answer"]
    citations = answer["citations"]

    # The API refuses by returning no citations; make that impossible to miss.
    if citations:
        st.success(answer["answer"])
        st.markdown("**Citations:** " + "  ".join(f"`{cid}`" for cid in citations))
    else:
        st.warning(f"**Refused, not answerable from the documents.**\n\n{answer['answer']}")

    cols = st.columns(5)
    cols[0].metric("Confidence", f"{answer['confidence']:.2f}")
    cols[1].metric("Tokens", data["tokens_used"])
    cols[2].metric("Cost (USD)", f"${data['cost_usd']:.6f}")
    cols[3].metric("Latency", f"{data['latency_ms']} ms")
    cols[4].metric("Model", data["model"])

    retrieved = data["retrieved_chunk_ids"]
    st.markdown(f"**Retrieved chunks ({len(retrieved)})** — ✅ = cited in the answer")
    if retrieved:
        for cid in retrieved:
            st.markdown(f"- {'✅' if cid in citations else '▫️'} `{cid}`")
    else:
        st.caption("No chunk scored above the relevance threshold, so the LLM was not called.")

    with st.expander("Raw JSON response"):
        st.json(data)


st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("Session 2 — RAG Demo")

base_url = st.sidebar.text_input("API base URL", DEFAULT_API_URL)
st.sidebar.caption("Default comes from the RAG_API_URL env var.")
if st.sidebar.button("Check connection"):
    with st.sidebar:
        with st.spinner("Calling /health/pinecone..."):
            status, data = call_api("GET", base_url, "/health/pinecone")
        if status == 200 and isinstance(data, dict):
            st.success(f"Connected. {data['namespace_vector_count']} chunks in the vector store.")
        else:
            show_error(status, data)

ask_tab, ingest_tab = st.tabs(["Ask", "Ingest"])

with ask_tab:
    with st.form("ask"):
        question = st.text_input("Question", placeholder="e.g. How many remote days are allowed?")
        left, right = st.columns(2)
        model = left.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "o3-mini"])
        top_k = right.slider("Chunks to retrieve (top_k)", 1, 20, 5)
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        if not question.strip():
            st.error("Type a question first.")
        else:
            with st.spinner("Calling /ask..."):
                status, data = call_api(
                    "POST", base_url, "/ask", json={"question": question, "model": model, "top_k": top_k}
                )
            if status == 200 and isinstance(data, dict):
                show_ask_result(data)
            else:
                show_error(status, data)

with ingest_tab:
    with st.form("ingest"):
        left, right = st.columns(2)
        document_id = left.text_input("document_id", placeholder="e.g. POL-999")
        source = right.text_input("source (optional)", placeholder="e.g. my_policy.txt")
        text = st.text_area("Document text", height=250)
        submitted = st.form_submit_button("Ingest", type="primary")
    if submitted:
        payload = {"document_id": document_id, "text": text}
        if source.strip():
            payload["source"] = source
        with st.spinner("Calling /ingest..."):
            status, data = call_api("POST", base_url, "/ingest", json=payload)
        if status == 200 and isinstance(data, dict):
            st.success(f"Indexed **{data['chunks_indexed']}** chunk(s) for `{data['document_id']}`.")
            cols = st.columns(3)
            cols[0].metric("Status", data["status"])
            cols[1].metric("Tokens", data["tokens_used"])
            cols[2].metric("Cost (USD)", f"${data['cost_usd']:.6f}")
            st.caption("Re-ingesting the same document_id replaces its previous chunks.")
        else:
            show_error(status, data)  # e.g. the API's 400 for empty text / document_id.
