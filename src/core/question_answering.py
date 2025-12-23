from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document  # type: ignore[import-not-found]
from langchain_ollama import ChatOllama  # type: ignore[import-not-found]

from .config import AppConfig, get_config
from .retrieval import (
    recommended_retrieve,
    mmr_retrieve,
    similarity_with_scores,
    rewrite_query,
)

logger = logging.getLogger(__name__)


def build_llm(config: Optional[AppConfig] = None) -> ChatOllama:
    """Return the chat LLM to craft answers."""
    cfg = config or get_config()
    return ChatOllama(model=cfg.llm_model_name, temperature=0)


def _format_context(docs: List[Document], max_chars: int = 5000) -> str:
    """Concatenate top documents into a single context block."""
    parts: List[str] = []
    total = 0
    for doc in docs:
        snippet = doc.page_content
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = snippet[:remaining]
        parts.append(f"Source: {doc.metadata.get('file_name', 'unknown')}\n{snippet}")
        total += len(snippet)
    return "\n\n".join(parts)


def _answer_with_context(
    llm: ChatOllama,
    question: str,
    docs: List[Document],
) -> str:
    """Generate an answer using provided docs as context."""
    if not docs:
        return "No relevant context found."
    context = _format_context(docs)
    prompt = (
        "You are an assistant answering questions about HR and company policies. "
        "Use the provided context to answer concisely. If unsure, say you don't know.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    try:
        return llm.invoke(prompt).content.strip()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Answer generation failed: %s", exc)
        return "Answer generation failed."


def answer_all_strategies(
    vectorstore,
    question: str,
    *,
    k: int = 4,
    fetch_k: int = 15,
    lambda_mult: float = 0.5,
    config: Optional[AppConfig] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Run multiple retrieval strategies and produce answers for each.

    Strategies:
      - similarity: vanilla similarity_search_with_score
      - mmr: max marginal relevance (no rewrite)
      - recommended: rewrite + MMR (default choice)
    """
    cfg = config or get_config()
    llm = build_llm(cfg)

    # Strategy 1: similarity search with scores
    sim_results, sim_query = similarity_with_scores(vectorstore, question, k=k, rewrite=False)
    sim_docs = [doc for doc in sim_results]
    sim_answer = _answer_with_context(llm, question, sim_docs)

    # # Strategy 2: MMR without rewrite
    # mmr_docs, mmr_query = mmr_retrieve(
    #     vectorstore,
    #     question,
    #     k=k,
    #     fetch_k=fetch_k,
    #     lambda_mult=lambda_mult,
    #     rewrite=False,
    #     config=cfg,
    # )
    # mmr_answer = _answer_with_context(llm, question, mmr_docs)

    # # # Strategy 3: recommended (rewrite + Similarity)
    # rec_docs, rec_query = recommended_retrieve(
    #     vectorstore,
    #     question,
    #     k=k,
    #     # fetch_k=fetch_k,
    #     # lambda_mult=lambda_mult,
    #     config=cfg,
    # )
    # rec_answer = _answer_with_context(llm, question, rec_docs)

    return {
        "similarity": {
            "answer": sim_answer,
            "docs": sim_docs,
            # "scores": [score for _, score in sim_results],
            "used_query": sim_query,
        },
        # "mmr": {
        #     "answer": mmr_answer,
        #     "docs": mmr_docs,
        #     "used_query": mmr_query,
        # },
        # "recommended": {
        #     "answer": rec_answer,
        #     "docs": rec_docs,
        #     "used_query": rec_query,
        # },
    }


__all__ = [
    "answer_all_strategies",
    "rewrite_query",
]
