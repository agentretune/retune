"""RetrievalConfig introspection from adapter."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.retrieval_introspection import introspect_retrieval_config
from retune.optimizer.retrieval_config import RetrievalConfig


def test_adapter_with_retrieval_config_dict():
    adapter = MagicMock()
    adapter.retrieval_config = {"retrieval_k": 8, "chunk_size": 500}
    config = introspect_retrieval_config(adapter)
    assert config is not None
    assert config.retrieval_k == 8
    assert config.chunk_size == 500


def test_adapter_with_retriever_fallback():
    adapter = MagicMock()
    adapter.retrieval_config = None
    retriever = MagicMock()
    retriever.search_kwargs = {"k": 10}
    retriever.chunk_size = 800
    retriever.search_type = "hybrid"
    adapter.retriever = retriever
    adapter.reranker = None
    config = introspect_retrieval_config(adapter)
    assert config is not None
    assert config.retrieval_k == 10
    assert config.chunk_size == 800
    assert config.retrieval_strategy == "hybrid"


def test_adapter_none_returns_none():
    assert introspect_retrieval_config(None) is None


def test_non_rag_adapter_returns_none():
    adapter = MagicMock()
    adapter.retrieval_config = None
    adapter.retriever = None
    assert introspect_retrieval_config(adapter) is None
