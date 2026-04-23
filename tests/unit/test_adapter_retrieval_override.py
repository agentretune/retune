"""Adapter.apply_retrieval_override — base no-op + LangChain mutates search_kwargs."""
from __future__ import annotations


def test_base_adapter_no_op():
    from retune.adapters.base import BaseAdapter

    class DummyAdapter(BaseAdapter):
        def run(self, query: str) -> str:  # type: ignore[override]
            return "ok"

        def get_config(self):  # type: ignore[override]
            from retune.core.models import OptimizationConfig
            return OptimizationConfig()

        def apply_config(self, config) -> None:  # type: ignore[override]
            pass

    d = DummyAdapter(agent=lambda q: "ok")
    # Should not raise, should not error
    d.apply_retrieval_override(retrieval_k=8, chunk_size=500)


def test_custom_adapter_inherits_no_op():
    from retune.adapters.custom_adapter import CustomAdapter
    def fn(q: str) -> str:
        return "ok"
    a = CustomAdapter(agent=fn)
    a.apply_retrieval_override(retrieval_k=10)   # no-op, no error
