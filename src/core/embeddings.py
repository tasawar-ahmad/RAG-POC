from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document  # type: ignore[import-not-found]
from langchain_chroma import Chroma  # type: ignore[import-not-found]
from langchain_ollama import OllamaEmbeddings

from .config import AppConfig, get_config

logger = logging.getLogger(__name__)


def build_embeddings(config: Optional[AppConfig] = None) -> OllamaEmbeddings:
    """
    Create an Ollama embeddings model instance.

    Uses a local Ollama model (e.g. \"nomic-embed-text\") so no external API key
    or network calls are required.
    """
    cfg = config or get_config()
    return OllamaEmbeddings(
        model=cfg.embedding_model_name,
    )


def get_vectorstore_path(config: Optional[AppConfig] = None) -> Path:
    """Return the path where the Chroma vectorstore should be persisted."""
    cfg = config or get_config()
    cfg.ensure_runtime_dirs()
    # Use a subdirectory for the Chroma database
    return cfg.vectorstore_dir / "chroma_db"


def create_vectorstore(
    documents: List[Document],
    config: Optional[AppConfig] = None,
    collection_name: str = "rag_documents",
) -> Chroma:
    """
    Create a new Chroma vectorstore from documents and persist it.

    If a vectorstore already exists at the configured path, it will be overwritten.
    """
    cfg = config or get_config()
    cfg.ensure_runtime_dirs()

    embeddings = build_embeddings(cfg)
    persist_directory = str(get_vectorstore_path(cfg))

    logger.info(
        "Creating Chroma vectorstore with %s documents at %s",
        len(documents),
        persist_directory,
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    logger.info("Vectorstore created and persisted successfully")
    return vectorstore

def load_vectorstore(
    config: Optional[AppConfig] = None,
    collection_name: str = "rag_documents",
) -> Optional[Chroma]:
    """
    Load an existing Chroma vectorstore from disk.

    Returns None if the vectorstore doesn't exist yet.
    """
    cfg = config or get_config()
    persist_directory = get_vectorstore_path(cfg)
    
    if not persist_directory.exists():
        logger.info("No existing vectorstore found at %s", persist_directory)
        return None

    logger.info("Loading existing vectorstore from %s", persist_directory)

    embeddings = build_embeddings(cfg)
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    logger.info("Vectorstore loaded successfully")
    return vectorstore


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    config: Optional[AppConfig] = None,
    collection_name: str = "rag_documents",
    force_recreate: bool = False,
) -> Chroma:
    """
    Get an existing vectorstore or create a new one.

    If documents are provided and no vectorstore exists (or force_recreate=True),
    a new vectorstore will be created from the documents.

    Args:
        documents: Documents to use if creating a new vectorstore
        config: Optional config override
        collection_name: Chroma collection name
        force_recreate: If True, recreate the vectorstore even if it exists

    Returns:
        A Chroma vectorstore instance

    Raises:
        RuntimeError: If no vectorstore exists and no documents provided
    """
    cfg = config or get_config()

    if not force_recreate:
        existing = load_vectorstore(cfg, collection_name)
        if existing is not None:
            return existing

    if documents is None:
        raise RuntimeError(
            "No existing vectorstore found and no documents provided. "
            "Either create a vectorstore first or provide documents."
        )

    return create_vectorstore(documents, cfg, collection_name)


