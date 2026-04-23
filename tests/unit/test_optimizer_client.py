"""OptimizerClient — HTTP wrappers for cloud optimize endpoints."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from retune.optimizer.client import OptimizerClient


@patch("retune.optimizer.client.urlopen")
def test_preauthorize_returns_run_id(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = b'{"run_id": "run_abc", "runs_remaining": 14}'
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    out = client.preauthorize(source="last_n_traces", n_traces=50, axes=["prompt"])
    assert out["run_id"] == "run_abc"
    assert out["runs_remaining"] == 14


@patch("retune.optimizer.client.urlopen")
def test_preauthorize_raises_on_402(mock_urlopen):
    from urllib.error import HTTPError
    mock_urlopen.side_effect = HTTPError(
        "url", 402, "Payment Required", hdrs=None, fp=None  # type: ignore
    )
    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    import pytest
    with pytest.raises(RuntimeError, match="limit reached|402"):
        client.preauthorize(source="last_n_traces", n_traces=50, axes=["prompt"])


@patch("retune.optimizer.client.urlopen")
def test_poll_pending_returns_message(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = b'{"type": "run_candidate", "candidate_id": "c1"}'
    resp.status = 200
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    msg = client.poll_pending("run_abc", timeout=1)
    assert msg is not None
    assert msg["type"] == "run_candidate"


@patch("retune.optimizer.client.urlopen")
def test_poll_pending_returns_none_on_204(mock_urlopen):
    resp = MagicMock()
    resp.status = 204
    resp.read.return_value = b""
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    assert client.poll_pending("run_abc", timeout=1) is None


@patch("retune.optimizer.client.urlopen")
def test_fetch_report(mock_urlopen):
    resp = MagicMock()
    resp.read.return_value = (
        b'{"run_id": "r", "tier1": [], "tier2": [], "tier3": [],'
        b' "markdown": "# empty", "understanding": "",'
        b' "summary": {}, "pareto_data": []}'
    )
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    rep = client.fetch_report("r")
    assert rep["run_id"] == "r"


@patch("retune.optimizer.client.urlopen")
def test_preauthorize_includes_traces_in_body(mock_urlopen):
    import json as _json
    resp = MagicMock()
    resp.read.return_value = b'{"run_id": "r", "runs_remaining": 14}'
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    traces = [{"query": "q1"}, {"query": "q2"}]
    client.preauthorize(
        source="last_n_traces", n_traces=2, axes=["prompt"],
        traces=traces,
    )
    req = mock_urlopen.call_args[0][0]
    body = _json.loads(req.data)
    assert body["traces"] == traces
