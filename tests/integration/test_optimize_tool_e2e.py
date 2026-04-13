"""Phase 3 E2E: SDK uploads traces + tool_metadata → cloud routes to both
Prompt and Tool subagents → report has merged Tier 1 (prompt rewrite + tool drop)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from server.app import app
from server.optimizer.models import PromptCandidate


@pytest.fixture
def fake_db_phase3():
    """Extends the Phase 2 fake DB with tool-metadata storage."""
    state = {
        "runs": {}, "reports": {}, "traces_bundles": {}, "tools_bundles": {},
        "org_used": {"org_1": 0}, "org_limit": {"org_1": 15},
    }

    def create_opt_run(run_id, org_id, source, n_traces, axes, reward_spec, rewriter_llm):
        state["runs"][run_id] = {
            "id": run_id, "org_id": org_id, "status": "pending",
            "source": source, "n_traces": n_traces, "axes": axes,
            "reward_spec": reward_spec, "rewriter_llm": rewriter_llm,
            "created_at": None, "started_at": None, "completed_at": None,
            "failure_reason": None,
        }

    def get_opt_run(run_id): return state["runs"].get(run_id)

    def update_opt_run_status(run_id, status, failure_reason=None):
        if run_id in state["runs"]:
            state["runs"][run_id]["status"] = status
            if failure_reason:
                state["runs"][run_id]["failure_reason"] = failure_reason

    def save_opt_run_traces(run_id, traces):
        state["traces_bundles"][run_id] = (list(traces), len(traces))

    def get_opt_run_traces(run_id):
        return state["traces_bundles"].get(run_id, ([], 0))

    def save_opt_run_tools(run_id, tools):
        state["tools_bundles"][run_id] = list(tools)

    def get_opt_run_tools(run_id):
        return state["tools_bundles"].get(run_id, [])

    def save_opt_report(**kwargs):
        state["reports"][kwargs["run_id"]] = kwargs

    def get_opt_report(run_id): return state["reports"].get(run_id)

    def count_opt_runs_used(org_id):
        return state["org_used"].get(org_id, 0), state["org_limit"].get(org_id, 15)

    def increment_opt_runs_used(org_id):
        state["org_used"][org_id] = state["org_used"].get(org_id, 0) + 1

    def decrement_opt_runs_used(org_id):
        state["org_used"][org_id] = max(0, state["org_used"].get(org_id, 0) - 1)

    ns = type("FakeDB", (), {})()
    ns.runs = state["runs"]
    ns.reports = state["reports"]
    ns.traces_bundles = state["traces_bundles"]
    ns.tools_bundles = state["tools_bundles"]
    ns.org_used = state["org_used"]
    ns.org_limit = state["org_limit"]
    ns.create_opt_run = create_opt_run
    ns.get_opt_run = get_opt_run
    ns.update_opt_run_status = update_opt_run_status
    ns.save_opt_run_traces = save_opt_run_traces
    ns.get_opt_run_traces = get_opt_run_traces
    ns.save_opt_run_tools = save_opt_run_tools
    ns.get_opt_run_tools = get_opt_run_tools
    ns.save_opt_report = save_opt_report
    ns.get_opt_report = get_opt_report
    ns.count_opt_runs_used = count_opt_runs_used
    ns.increment_opt_runs_used = increment_opt_runs_used
    ns.decrement_opt_runs_used = decrement_opt_runs_used
    return ns


def test_phase3_combined_routing_e2e(fake_db_phase3):
    """Upload traces+tools → Orchestrator routes both → merged Tier 1."""
    from server.optimizer.job_queue import get_queue, get_results
    get_queue().reset()
    get_results().reset()

    # Traces that call "used_tool" 3 times but never "unused_tool"
    traces = [
        {"id": f"t{i}", "query": f"q{i}", "response": f"r{i}",
         "config_snapshot": {"system_prompt": "BASE"},
         "steps": [{"type": "tool_call", "name": "used_tool", "args": {"x": i}}]}
        for i in range(3)
    ]
    tool_metadata = [
        {"name": "used_tool", "description": "Active tool", "args_schema": {}, "is_async": False},
        {"name": "unused_tool", "description": "Never called",
         "args_schema": {}, "is_async": False},
    ]

    baseline_cand = PromptCandidate(
        candidate_id="cand_base", system_prompt="BASE", generation_round=0,
    )
    rewrite_cand = PromptCandidate(
        candidate_id="cand_rewrite", system_prompt="PRECISE BASE", generation_round=1,
    )

    with patch("server.routes.optimize.db", fake_db_phase3), \
         patch("server.routes.jobs.db", fake_db_phase3), \
         patch("server.optimizer.orchestrator.db", fake_db_phase3), \
         patch("server.routes.optimize.require_auth", return_value={"org": "org_1"}), \
         patch("server.routes.jobs.require_auth", return_value={"org": "org_1"}), \
         patch("server.optimizer.orchestrator.PromptOptimizerAgent") as mock_prompt_cls, \
         patch("server.optimizer.orchestrator.ToolOptimizerAgent") as mock_tool_cls, \
         patch("server.routes.optimize.BackgroundTasks.add_task"):

        mock_prompt = MagicMock()
        mock_prompt.generate_candidates.return_value = [baseline_cand, rewrite_cand]
        mock_prompt_cls.return_value = mock_prompt

        mock_tool = MagicMock()
        mock_tool.optimize.return_value = (
            [{
                "tier": 1, "axis": "tools",
                "title": "Drop unused tool: unused_tool",
                "description": "`unused_tool` was never invoked.",
                "confidence": "H",
                "estimated_impact": {"tool_count_reduction": 1.0},
                "apply_payload": {"action": "drop_tool", "tool_name": "unused_tool"},
                "evidence_trace_ids": [],
            }],
            [],
            [],
        )
        mock_tool_cls.return_value = mock_tool

        client = TestClient(app)
        auth = {"Authorization": "Bearer test"}

        r = client.post(
            "/api/v1/optimize/preauthorize",
            json={
                "source": "last_n_traces", "n_traces": 3,
                "axes": ["prompt", "tools"],
                "traces": traces, "tool_metadata": tool_metadata,
            },
            headers=auth,
        )
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        # Seed candidate results for the prompt subagent
        results = get_results()
        for cid, score in [("cand_base", 5.0), ("cand_rewrite", 8.0)]:
            results.put(run_id, cid, {
                "run_id": run_id, "candidate_id": cid,
                "trace": {}, "eval_scores": {"llm_judge": score},
            })

        from server.optimizer.orchestrator import OptimizerOrchestrator
        OptimizerOrchestrator().run(run_id, candidate_result_timeout=0.3)

        # Verify both subagents dispatched
        mock_prompt_cls.assert_called_once()
        mock_tool_cls.assert_called_once()
        mock_tool.optimize.assert_called_once()

        # Fetch report
        r = client.get(f"/api/v1/optimize/{run_id}/report", headers=auth)
        assert r.status_code == 200
        body = r.json()
        tier1 = body["tier1"]
        axes_seen = {s.get("axis") for s in tier1}
        assert "prompt" in axes_seen, f"tier1 missing prompt suggestion: {tier1}"
        assert "tools" in axes_seen, f"tier1 missing tools suggestion: {tier1}"
