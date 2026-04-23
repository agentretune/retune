"""Phase 2 E2E: SDK uploads traces → real Orchestrator dispatches prompt
candidates → SDK posts mock results → Report has a Tier 1 suggestion."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402
from server.app import app  # noqa: E402
from server.optimizer.models import PromptCandidate, ScoredCandidate  # noqa: E402


@pytest.fixture
def fake_db_phase2():
    """In-memory fake DB for Phase 2 tables."""
    state = {
        "runs": {}, "reports": {}, "traces_bundles": {},
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
    ns.org_used = state["org_used"]
    ns.org_limit = state["org_limit"]
    ns.create_opt_run = create_opt_run
    ns.get_opt_run = get_opt_run
    ns.update_opt_run_status = update_opt_run_status
    ns.save_opt_run_traces = save_opt_run_traces
    ns.get_opt_run_traces = get_opt_run_traces
    ns.save_opt_report = save_opt_report
    ns.get_opt_report = get_opt_report
    ns.count_opt_runs_used = count_opt_runs_used
    ns.increment_opt_runs_used = increment_opt_runs_used
    ns.decrement_opt_runs_used = decrement_opt_runs_used
    return ns


def test_prompt_optimization_e2e(fake_db_phase2):
    """Full Phase 2 flow: upload traces → dispatch candidates → mock results → Tier 1."""
    from server.optimizer.job_queue import get_queue, get_results
    get_queue().reset()
    get_results().reset()

    traces = [
        {"id": f"t{i}", "query": f"q{i}", "response": f"r{i}",
         "config_snapshot": {"system_prompt": "You are helpful."}}
        for i in range(5)
    ]

    baseline_cand = PromptCandidate(
        candidate_id="cand_base", system_prompt="You are helpful.",
        generation_round=0,
    )
    rewrite_cand = PromptCandidate(
        candidate_id="cand_rewrite", system_prompt="You are a precise helper.",
        generation_round=1,
    )

    with patch("server.routes.optimize.db", fake_db_phase2), \
         patch("server.routes.jobs.db", fake_db_phase2), \
         patch("server.optimizer.orchestrator.db", fake_db_phase2), \
         patch("server.routes.optimize.require_auth", return_value={"org": "org_1"}), \
         patch("server.routes.jobs.require_auth", return_value={"org": "org_1"}), \
         patch("server.optimizer.orchestrator.PromptOptimizerAgent") as mock_prompt_cls, \
         patch("server.optimizer.orchestrator.get_queue"), \
         patch("server.optimizer.orchestrator.get_results"):

        mock_prompt = MagicMock()
        mock_prompt.run_iterative.return_value = [
            ScoredCandidate(
                candidate=rewrite_cand, scalar_score=8.0,
                dimensions={"llm_judge": 8.0}, guardrails_held=True,
            ),
            ScoredCandidate(
                candidate=baseline_cand, scalar_score=5.0,
                dimensions={"llm_judge": 5.0}, guardrails_held=True,
            ),
        ]
        mock_prompt_cls.return_value = mock_prompt

        client = TestClient(app)
        auth = {"Authorization": "Bearer test"}

        # Preauthorize — but patch BackgroundTasks to prevent the auto-orchestrator
        # from running before we drive it synchronously below.
        with patch("server.routes.optimize.BackgroundTasks.add_task"):
            r = client.post(
                "/api/v1/optimize/preauthorize",
                json={
                    "source": "last_n_traces", "n_traces": 5,
                    "axes": ["prompt"], "traces": traces,
                    "rewriter_llm": "gpt-4o-mini",
                },
                headers=auth,
            )
            assert r.status_code == 200
            run_id = r.json()["run_id"]

        # Drive the orchestrator synchronously
        from server.optimizer.orchestrator import OptimizerOrchestrator
        OptimizerOrchestrator().run(run_id, candidate_result_timeout=0.5)

        # Fetch report
        r = client.get(f"/api/v1/optimize/{run_id}/report", headers=auth)
        assert r.status_code == 200
        body = r.json()
        assert len(body["tier1"]) >= 1
        assert any(
            "precise helper" in (s.get("description") or "")
            for s in body["tier1"]
        )
