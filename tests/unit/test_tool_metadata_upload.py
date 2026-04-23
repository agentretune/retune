"""Retuner.optimize introspects tools and sends them to cloud."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from retune import Mode, Retuner


@patch("retune.wrapper.OptimizerClient")
@patch("retune.wrapper.SDKWorker")
@patch("retune.wrapper.introspect_tools")
def test_optimize_passes_tool_metadata_when_tools_in_axes(
    mock_introspect, mock_worker_cls, mock_client_cls
):
    from retune.optimizer.tool_metadata import ToolMetadata
    mock_introspect.return_value = [
        ToolMetadata(name="search", description="d", args_schema={}),
    ]
    mock_client = MagicMock()
    mock_client.preauthorize.return_value = {"run_id": "r", "runs_remaining": 14}
    mock_client.fetch_report.return_value = {
        "run_id": "r", "understanding": "", "summary": {},
        "tier1": [], "tier2": [], "tier3": [],
        "pareto_data": [], "markdown": "# empty",
    }
    mock_client_cls.return_value = mock_client
    mock_worker = MagicMock()
    mock_worker.run.return_value = "/x"
    mock_worker_cls.return_value = mock_worker

    def agent(q: str) -> str: return "resp"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner.optimize(source="last_n_traces", n=10, axes=["prompt", "tools"])

    # tool_metadata passed
    kwargs = mock_client.preauthorize.call_args.kwargs
    assert kwargs["tool_metadata"] == [{"name": "search", "description": "d",
                                         "args_schema": {}, "is_async": False}]


@patch("retune.wrapper.OptimizerClient")
@patch("retune.wrapper.SDKWorker")
def test_optimize_skips_tool_metadata_when_axis_absent(mock_worker_cls, mock_client_cls):
    mock_client = MagicMock()
    mock_client.preauthorize.return_value = {"run_id": "r", "runs_remaining": 14}
    mock_client.fetch_report.return_value = {
        "run_id": "r", "understanding": "", "summary": {},
        "tier1": [], "tier2": [], "tier3": [],
        "pareto_data": [], "markdown": "# empty",
    }
    mock_client_cls.return_value = mock_client
    mock_worker_cls.return_value.run.return_value = "/x"

    def agent(q: str) -> str: return "resp"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner.optimize(source="last_n_traces", n=10, axes=["prompt"])
    kwargs = mock_client.preauthorize.call_args.kwargs
    # Not set (or None) since tools axis not requested
    assert kwargs.get("tool_metadata") is None
