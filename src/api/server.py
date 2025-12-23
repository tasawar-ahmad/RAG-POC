from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, Form, HTTPException, BackgroundTasks
import requests
from pydantic import BaseModel

from src.core.config import get_config
from src.core.documents import load_documents
from src.core.embeddings import load_vectorstore, create_vectorstore
from src.core.question_answering import answer_all_strategies
from src.core.splitters import split_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Slack Bridge", version="0.1.0")


class QueryRequest(BaseModel):
    query: str
    recreate_store: Optional[bool] = False  # force rebuild vectorstore if true


class QueryResponse(BaseModel):
    strategy: str
    answer: str
    used_query: Optional[str] = None
    scores: Optional[List[float]] = None


class MultiStrategyResponse(BaseModel):
    results: List[QueryResponse]


_CACHED_VECTORSTORE = None  # simple in-process cache


def ensure_vectorstore(recreate: bool = False):
    """
    Ensure vectorstore exists; create from docs if missing or recreate=True.
    Uses an in-process cache to avoid rebuilding on every request.
    """
    global _CACHED_VECTORSTORE
    if _CACHED_VECTORSTORE is not None and not recreate:
        return _CACHED_VECTORSTORE

    cfg = get_config()
    vs = None
    if not recreate:
        vs = load_vectorstore(cfg)
    if vs is None or recreate:
        logger.info("Building vectorstore (recreate=%s)...", recreate)
        docs = load_documents(cfg)
        chunks = split_documents(docs, cfg)
        vs = create_vectorstore(chunks, cfg)

    _CACHED_VECTORSTORE = vs
    return vs


def _format_slack_results(multi):
    lines = []
    for item in multi.results:
        header = f"*{item.strategy}*"
        if item.used_query:
            header += f" (query: {item.used_query})"
        body = item.answer
        lines.append(f"{header}\n{body}")
    return "\n\n".join(lines)


def _post_to_response_url(response_url: str, text: str):
    try:
        requests.post(
            response_url,
            json={"response_type": "ephemeral", "text": text},
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to post to Slack response_url: %s", exc)


@app.on_event("startup")
def warm_start_vectorstore():
    """
    Pre-build or load the vectorstore at startup to avoid per-request cold starts.
    Also triggers a tiny retrieval to warm up Ollama/Chroma connections.
    """
    try:
        vs = ensure_vectorstore(recreate=False)
        # Light warm-up query to initialize embedding/LLM plumbing; ignored if empty.
        try:
            _ = vs.similarity_search("warmup", k=1)
        except Exception:
            # Best-effort warmup; continue even if it fails.
            logger.info("Warm-up retrieval skipped or failed; continuing.")
        logger.info("Vectorstore ready at startup.")
    except Exception as exc:  # pragma: no cover - startup only
        logger.exception("Startup vectorstore preparation failed: %s", exc)


@app.post("/query")
def query_endpoint(
    background_tasks: BackgroundTasks,
    payload: Optional[QueryRequest] = Body(None),
    query: Optional[str] = Form(
        None,
        description="Query text (for form submissions such as Slack slash commands)",
    ),
    text: Optional[str] = Form(
        None,
        description="Slack sends the user message in the 'text' form field",
    ),
    response_url: Optional[str] = Form(
        None, description="Slack response_url for delayed reply"
    ),
    recreate_store: Optional[bool] = Form(
        False, description="Force rebuild of vectorstore", alias="recreate_store"
    ),
):
    """
    Single endpoint for both direct clients and Slack:
      - If response_url is provided, respond immediately and post results later.
      - Otherwise, return answers synchronously.
    Accepts both JSON (QueryRequest) and form-encoded (Slack) inputs.
    """
    print ('response_url', response_url)
    # Normalize inputs (priority: payload.query > text > query)
    effective_query = None
    effective_recreate = False

    if payload and payload.query:
        effective_query = payload.query
        effective_recreate = bool(payload.recreate_store)
    if text:
        effective_query = text
        effective_recreate = bool(recreate_store)
    if query:
        effective_query = query
        effective_recreate = bool(recreate_store)

    if not effective_query or not effective_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    def _compute_and_respond(sync_mode: bool = False):
        # Ensure vectorstore exists
        vectorstore = ensure_vectorstore(recreate=effective_recreate)
        results = answer_all_strategies(vectorstore, effective_query)

        response_items = []
        for name, data in results.items():
            response_items.append(
                QueryResponse(
                    strategy=name,
                    answer=data.get("answer", ""),
                    used_query=data.get("used_query"),
                    scores=data.get("scores"),
                )
            )

        if sync_mode:
            return MultiStrategyResponse(results=response_items)

        # async post to Slack
        formatted = _format_slack_results(
            MultiStrategyResponse(results=response_items)
        )
        if response_url:
            _post_to_response_url(response_url, formatted)
        else:
            logger.info("Slack response (no response_url):\n%s", formatted)
        return None

    # If response_url provided, do async work and ack immediately
    if response_url:
        background_tasks.add_task(_compute_and_respond, False)
        return {"response_type": "ephemeral", "text": "Working on it…"}

    # Otherwise synchronous response
    result = _compute_and_respond(sync_mode=True)
    return result


@app.get("/health")
def health():
    return {"status": "ok"}

