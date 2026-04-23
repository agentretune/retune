"""Retuner.optimize uploads retrieval_config when rag axis requested."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from retune import Mode, Retuner
from retune.optimizer.retrieval_config import RetrievalConfig


@patch("retune.wrapper.OptimizerClient")
@patch("retune.wrapper.SDKWorker")
@patch("retune.wrapper.introspect_retrieval_config")
def test_optimize_uploads_retrieval_config_when_rag_in_axes(
    mock_introspect, mock_worker_cls, mock_client_cls,
):
    mock_introspect.return_value = RetrievalConfig(
        retrieval_k=8, chunk_size=500,
    )
    mock_client = MagicMock()
    mock_client.preauthorize.return_value = {"run_id": "r", "runs_remaining": 14}
    mock_client.fetch_report.return_value = {
        "run_id": "r", "understanding": "", "summary": {},
        "tier1": [], "tier2": [], "tier3": [], "pareto_data": [], "markdown": "",
    }
    mock_client_cls.return_value = mock_client
    mock_worker_cls.return_value.run.return_value = "/x"

    def agent(q: str) -> str: return "r"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner.optimize(source="last_n_traces", n=5, axes=["rag"])

    kwargs = mock_client.preauthorize.call_args.kwargs
    assert kwargs["retrieval_config"]["retrieval_k"] == 8


@patch("retune.wrapper.OptimizerClient")
@patch("retune.wrapper.SDKWorker")
def test_optimize_skips_retrieval_when_axis_absent(mock_worker_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client.preauthorize.return_value = {"run_id": "r", "runs_remaining": 14}
    mock_client.fetch_report.return_value = {
        "run_id": "r", "understanding": "", "summary": {},
        "tier1": [], "tier2": [], "tier3": [], "pareto_data": [], "markdown": "",
    }
    mock_client_cls.return_value = mock_client
    mock_worker_cls.return_value.run.return_value = "/x"

    def agent(q: str) -> str: return "r"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner.optimize(source="last_n_traces", n=5, axes=["prompt"])
    kwargs = mock_client.preauthorize.call_args.kwargs
    assert kwargs.get("retrieval_config") is None
