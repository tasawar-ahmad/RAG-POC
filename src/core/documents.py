from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document  # type: ignore[import-not-found]
from langchain_community.document_loaders import PyPDFLoader  # type: ignore[import-not-found]
from langchain_community.document_loaders.merge import MergedDataLoader  # type: ignore[import-not-found]

from .config import AppConfig, get_config

logger = logging.getLogger(__name__)

def _iter_document_paths(root: Path) -> Iterable[Path]:
    """Yield PDF files within the document root."""
    if not root.exists():
        logger.warning("Documents directory %s does not exist; creating it.", root)
        root.mkdir(parents=True, exist_ok=True)
        return []

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".pdf":
            yield path


WHITESPACE_PATTERN = re.compile(r"\s+")


def _clean_page_content(content: str) -> str:
    """Normalize whitespace to avoid awkward newlines and double spaces."""
    normalized = WHITESPACE_PATTERN.sub(" ", content).strip()
    return normalized


def _load_with_loader(path: Path) -> List[Document]:
    """Load a PDF using LangChain's PyPDFLoader."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.page_content = _clean_page_content(doc.page_content)
    return docs


def _augment_metadata(doc: Document, path: Path, root: Path) -> Document:
    """Ensure every document carries consistent metadata fields."""
    metadata = {**doc.metadata}
    metadata.setdefault("source", str(path))
    metadata.setdefault("relative_source", str(path.relative_to(root)))
    metadata.setdefault("file_name", path.name)
    doc.metadata = metadata
    return doc


def load_documents(config: Optional[AppConfig] = None) -> List[Document]:
    """
    Load documents via LangChain loaders.

    PDFs go through PyPDFLoader (page-aware) and text files through TextLoader.
    """

    cfg = config or get_config()
    cfg.ensure_runtime_dirs()

    documents: List[Document] = []
    for path in _iter_document_paths(cfg.documents_dir):
        raw_docs = _load_with_loader(path)
        if not raw_docs:
            logger.warning("Loader returned no content for %s; skipping.", path)
            continue

        documents.extend(_augment_metadata(doc, path, cfg.documents_dir) for doc in raw_docs)

    if not documents:
        raise RuntimeError(
            f"No supported documents found in {cfg.documents_dir}. "
            "Add PDF or text files before running ingestion."
        )

    logger.info("Loaded %s documents from %s", len(documents), cfg.documents_dir)
    return documents
