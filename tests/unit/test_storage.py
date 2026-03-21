"""Tests for SQLite storage."""

import pytest

from retune.core.models import ExecutionTrace, OptimizationConfig
from retune.storage.sqlite_storage import SQLiteStorage


class TestSQLiteStorage:
    def test_save_and_get_trace(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        trace = ExecutionTrace(query="test query", response="test response")

        storage.save_trace(trace)
        retrieved = storage.get_trace(trace.trace_id)

        assert retrieved is not None
        assert retrieved.query == "test query"
        assert retrieved.response == "test response"

    def test_get_nonexistent_trace(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        assert storage.get_trace("nonexistent") is None

    def test_get_traces_with_limit(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "test.db"))

        for i in range(5):
            storage.save_trace(ExecutionTrace(query=f"query {i}"))

        traces = storage.get_traces(limit=3)
        assert len(traces) == 3

    def test_get_traces_by_session(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "test.db"))

        storage.save_trace(ExecutionTrace(query="q1", session_id="s1"))
        storage.save_trace(ExecutionTrace(query="q2", session_id="s1"))
        storage.save_trace(ExecutionTrace(query="q3", session_id="s2"))

        traces = storage.get_traces(session_id="s1")
        assert len(traces) == 2

    def test_save_and_get_config(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        config = OptimizationConfig(top_k=5, temperature=0.3)

        storage.save_config("v1", config)
        retrieved = storage.get_config("v1")

        assert retrieved is not None
        assert retrieved.top_k == 5
        assert retrieved.temperature == 0.3

    def test_list_configs(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "test.db"))
        storage.save_config("v1", OptimizationConfig(top_k=3))
        storage.save_config("v2", OptimizationConfig(top_k=5))

        configs = storage.list_configs()
        assert "v1" in configs
        assert "v2" in configs
