"""SDK local dashboard serves HTML from SQLite traces."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402


def test_dashboard_home_with_traces():
    mock_storage = MagicMock()
    mock_storage.get_traces.return_value = [
        {"query": "hello world", "response": "hi there", "duration_ms": 123.0},
        {"query": "another", "response": "another resp", "duration_ms": 200.0},
    ]
    with patch("retune.dashboard.app._get_storage", return_value=mock_storage):
        from retune.dashboard.app import app
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "hello world" in r.text
        assert "hi there" in r.text
        assert "2 recent traces" in r.text


def test_dashboard_home_empty_storage():
    mock_storage = MagicMock()
    mock_storage.get_traces.return_value = []
    with patch("retune.dashboard.app._get_storage", return_value=mock_storage):
        from retune.dashboard.app import app
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "No traces yet" in r.text


def test_dashboard_health():
    from retune.dashboard.app import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
