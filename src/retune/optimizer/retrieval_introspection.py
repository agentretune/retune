"""Introspect the adapter for retrieval config at optimize-time."""
from __future__ import annotations

import logging
from typing import Any

from retune.optimizer.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)


def introspect_retrieval_config(adapter: Any) -> RetrievalConfig | None:
    """Return the adapter's current retrieval config, or None if not a RAG adapter."""
    if adapter is None:
        return None

    # Common attribute names the SDK adapters may expose
    rc_dict = getattr(adapter, "retrieval_config", None)
    if isinstance(rc_dict, RetrievalConfig):
        return rc_dict
    if isinstance(rc_dict, dict):
        try:
            return RetrievalConfig(**rc_dict)
        except Exception as e:
            logger.debug("Retrieval config parse failed: %s", e)
            return None

    # Fallback: pull individual attrs
    retriever = getattr(adapter, "retriever", None)
    if retriever is None:
        return None
    try:
        _reranker = getattr(adapter, "reranker", None)
        _reranker_model = getattr(_reranker, "model", None) if _reranker is not None else None
        _embeddings = getattr(retriever, "embeddings", None)
        _embedding_model_raw = getattr(_embeddings, "model", None) if _embeddings is not None else None
        _embedding_model = _embedding_model_raw if isinstance(_embedding_model_raw, str) else None
        _chunk_overlap_raw = getattr(retriever, "chunk_overlap", 200)
        _chunk_overlap = int(_chunk_overlap_raw) if isinstance(_chunk_overlap_raw, (int, float)) else 200
        return RetrievalConfig(
            retrieval_k=int(getattr(retriever, "search_kwargs", {}).get("k", 5)),
            chunk_size=int(getattr(retriever, "chunk_size", 1000) or 1000),
            chunk_overlap=_chunk_overlap,
            reranker_enabled=bool(_reranker is not None),
            reranker_model=_reranker_model if isinstance(_reranker_model, str) else None,
            retrieval_strategy=getattr(retriever, "search_type", "dense") or "dense",
            embedding_model=_embedding_model,
        )
    except Exception as e:
        logger.debug("Retrieval introspection failed: %s", e)
        return None
