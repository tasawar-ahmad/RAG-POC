from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document  # type: ignore[import-not-found]
from langchain_ollama import ChatOllama  # type: ignore[import-not-found]

from .config import AppConfig, get_config

logger = logging.getLogger(__name__)


def build_llm(config: Optional[AppConfig] = None) -> ChatOllama:
    """Return a local Ollama chat model for light-weight query rewrites."""
    cfg = config or get_config()
    return ChatOllama(model=cfg.llm_model_name, temperature=0.0)


def rewrite_query(
    query: str, *, config: Optional[AppConfig] = None, llm: Optional[ChatOllama] = None
) -> str:
    """Rewrite a user query to be more explicit for retrieval; fall back to original on error."""
    llm_instance = llm or build_llm(config)
    prompt = (
        "You are optimizing a search query for retrieving the most relevant chunks "
        "from HR and company policy documents (leave policy, OPD policy, etc.). "
        "Rewrite the following query to be more explicit and keyword-rich, "
        "while keeping the same intent. Respond with ONLY the rewritten query.\n\n"
        f"Query: {query!r}"
    )
    try:
        response = llm_instance.invoke(prompt)
        rewritten = response.content.strip()
        return rewritten or query
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Query rewrite failed, using original. Error: %s", exc)
        return query


def similarity_with_scores(
    vectorstore,
    query: str,
    k: int = 4,
    rewrite: bool = False,
    config: Optional[AppConfig] = None,
    llm: Optional[ChatOllama] = None,

) -> List[Tuple[Document, str]]:
    """Baseline similarity search with scores."""
    effective_query = rewrite_query(query, config=config, llm=llm) if rewrite else query
    docs = vectorstore.similarity_search(effective_query, k=k)
    return docs, effective_query

def mmr_retrieve(
    vectorstore,
    query: str,
    *,
    k: int = 4,
    fetch_k: int = 15,
    lambda_mult: float = 0.5,
    rewrite: bool = True,
    config: Optional[AppConfig] = None,
    llm: Optional[ChatOllama] = None,
) -> Tuple[List[Document], str]:
    """
    Run Max Marginal Relevance retrieval, optionally with an LLM query rewrite.

    Returns the retrieved documents and the (possibly rewritten) query used.
    """
    effective_query = rewrite_query(query, config=config, llm=llm) if rewrite else query
    docs = vectorstore.max_marginal_relevance_search(
        effective_query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )
    return docs, effective_query


def recommended_retrieve(
    vectorstore,
    query: str,
    *,
    k: int = 4,
    config: Optional[AppConfig] = None,
) -> Tuple[List[Document], str]:
    """
    Recommended retrieval strategy for this project:
    1) Rewrite the query with the local LLM (llama3.2).
    2) Use similarity_with_scores to get top-k results.
    """
    llm = build_llm(config)
    return similarity_with_scores(
        vectorstore,
        query,
        k=k,
        rewrite=False,
        config=config,
        llm=llm,
    )
