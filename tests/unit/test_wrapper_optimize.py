"""Retuner.optimize() — triggers cloud run, runs worker loop, returns report."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from retune import Retuner, Mode


def _mk_agent():
    def _agent(q: str) -> str:
        return f"echo: {q}"
    return _agent


@patch("retune.wrapper.OptimizerClient")
@patch("retune.wrapper.SDKWorker")
def test_optimize_historical_source(mock_worker_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client.preauthorize.return_value = {"run_id": "r1", "runs_remaining": 14}
    mock_client.fetch_report.return_value = {
        "run_id": "r1",
        "understanding": "",
        "summary": {"baseline_score": 0.0, "best_score": 0.0, "improvement_pct": 0.0},
        "tier1": [], "tier2": [], "tier3": [],
        "pareto_data": [],
        "markdown": "# empty",
    }
    mock_client_cls.return_value = mock_client

    mock_worker = MagicMock()
    mock_worker.run.return_value = "/api/v1/optimize/r1/report"
    mock_worker_cls.return_value = mock_worker

    retuner = Retuner(
        agent=_mk_agent(), adapter="custom",
        mode=Mode.IMPROVE, api_key="rt-test",
        agent_purpose="echo bot",
    )
    report = retuner.optimize(source="last_n_traces", n=50, axes=["prompt"])

    mock_client.preauthorize.assert_called_once()
    mock_worker.run.assert_called_once()
    mock_client.commit.assert_called_once_with("r1")
    assert report.run_id == "r1"


def test_optimize_requires_api_key():
    retuner = Retuner(
        agent=_mk_agent(), adapter="custom",
        mode=Mode.IMPROVE, api_key=None,
        agent_purpose="bot",
    )
    import pytest
    with pytest.raises(RuntimeError, match="api_key"):
        retuner.optimize(source="last_n_traces", n=10)


def test_optimize_requires_agent_purpose():
    import pytest
    with pytest.raises(ValueError, match="agent_purpose"):
        Retuner(
            agent=_mk_agent(), adapter="custom",
            mode=Mode.IMPROVE, api_key="rt-test",
            # agent_purpose missing
        )
