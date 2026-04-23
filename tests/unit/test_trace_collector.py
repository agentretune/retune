"""SDK trace collector for last_n_traces upload."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.trace_collector import collect_last_n_local_traces


def test_collect_last_n_from_storage():
    storage = MagicMock()
    storage.get_traces.return_value = [
        {"id": "t1", "query": "q1", "response": "r1", "duration_ms": 100},
        {"id": "t2", "query": "q2", "response": "r2", "duration_ms": 120},
    ]
    traces = collect_last_n_local_traces(storage, n=2)
    assert len(traces) == 2
    assert traces[0]["query"] == "q1"
    storage.get_traces.assert_called_once_with(limit=2)


def test_collect_last_n_empty_storage():
    storage = MagicMock()
    storage.get_traces.return_value = []
    assert collect_last_n_local_traces(storage, n=50) == []


def test_collect_last_n_fewer_than_requested():
    storage = MagicMock()
    storage.get_traces.return_value = [{"id": "t1", "query": "q"}]
    traces = collect_last_n_local_traces(storage, n=50)
    assert len(traces) == 1
