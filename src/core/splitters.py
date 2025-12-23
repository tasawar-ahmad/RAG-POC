from __future__ import annotations

from typing import Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore[import-not-found]
from langchain_core.documents import Document  # type: ignore[import-not-found]

from .config import AppConfig, get_config

DEFAULT_SEPARATORS = [
    ". ",
    "? ",
    "! ",
    " ",
    "",
]


def build_text_splitter(config: Optional[AppConfig] = None) -> RecursiveCharacterTextSplitter:
    """
    Create a LangChain text splitter based on application configuration.

    Uses RecursiveCharacterTextSplitter so we get sensible splits for PDFs that
    may contain headings, paragraphs, and tables.
    """

    cfg = config or get_config()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
    )


def split_documents(
    documents: Iterable[Document], config: Optional[AppConfig] = None
) -> List[Document]:
    """
    Split documents into overlapping chunks using the configured splitter.

    Returns a flat list of Documents with preserved metadata, ready for embeddings.
    """

    splitter = build_text_splitter(config)
    docs = splitter.split_documents(list(documents))
    return docs