from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _require_env(name: str) -> str:
    """Return required environment variable or raise."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Environment variable '{name}' is missing. "
            "Configure it in your shell or .env file."
        )
    return value


def _path_from_env(name: str, default: str) -> Path:
    """Coerce env var into an absolute Path, falling back to default."""
    raw_value = os.getenv(name, default)
    path = Path(raw_value)
    return path if path.is_absolute() else (Path.cwd() / path)


@dataclass(frozen=True)
class AppConfig:
    """Central configuration state for the RAG CLI application."""

    llm_model_name: str
    # Embedding model name; with Ollama we use "nomic-embed-text" by default
    embedding_model_name: str
    chunk_size: int
    chunk_overlap: int
    documents_dir: Path
    vectorstore_dir: Path

    def ensure_runtime_dirs(self) -> None:
        """Create directories that must exist before ingestion or chat."""
        for directory in (self.documents_dir, self.vectorstore_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """
    Load configuration from environment once and cache it.

    Exposed as a function so entrypoints can simply call:

        settings = get_config()
        settings.ensure_runtime_dirs()
    """

    return AppConfig(
        llm_model_name=os.getenv("LLM_MODEL_NAME", "llama3.2"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        documents_dir=_path_from_env("DOCUMENTS_DIR", "data/raw"),
        vectorstore_dir=_path_from_env("VECTORSTORE_DIR", "data/vectorstores"),
    )
