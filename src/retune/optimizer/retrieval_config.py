"""Serializable retrieval-config envelope for SDK→cloud upload."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Current retrieval configuration as introspected from the adapter."""
    retrieval_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    reranker_enabled: bool = False
    reranker_model: str | None = None
    query_rewriting_enabled: bool = False
    retrieval_strategy: str = "dense"  # "dense" | "sparse" | "hybrid"
    embedding_model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
