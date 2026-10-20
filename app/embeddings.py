"""Local ONNX embeddings for hosts that have chat but no /v1/embeddings."""

from __future__ import annotations

from app.config import AppConfig


def _load_fastembed():
    """Import FastEmbed only for EMBEDDING_PROVIDER=local. The OpenAI path never loads ONNX."""
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    return FastEmbedEmbedding


def build_local_embed_model(config: AppConfig):
    """Build a fastembed ONNX embedder. The model file downloads on first use."""
    cache_dir = config.embed_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastembed_cls = _load_fastembed()
    return fastembed_cls(
        model_name=config.local_embed_model,
        cache_dir=str(cache_dir),
    )
