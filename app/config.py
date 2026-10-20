"""Environment configuration for the RAG service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _clean(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    base_url: str | None
    chat_model: str
    embed_model: str
    data_dir: Path
    storage_dir: Path
    top_k: int = 3


def load_config() -> AppConfig:
    base_url = _clean("OPENAI_BASE_URL") or None
    if base_url:
        base_url = base_url.rstrip("/")
    return AppConfig(
        api_key=_clean("OPENAI_API_KEY"),
        base_url=base_url,
        chat_model=_clean("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        embed_model=_clean("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        or "text-embedding-3-small",
        data_dir=Path(_clean("DATA_DIR", "data") or "data"),
        storage_dir=Path(_clean("STORAGE_DIR", "storage") or "storage"),
    )
