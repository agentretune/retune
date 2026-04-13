# tests/integration/test_optimize_e2e.py
"""End-to-end: SDK triggers optimize → cloud creates run → orchestrator
completes → SDK receives report → slot committed.

Uses FastAPI TestClient as transport; fakes the DB layer in-process.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from server.app import app
from server.optimizer.models import PromptCandidate, ScoredCandidate


@pytest.fixture
def fake_db():
    """In-memory fake DB for optimizer tables."""
    state = {
        "runs": {},
        "reports": {},
        "org_used": {"org_1": 0},
        "org_limit": {"org_1": 15},
    }

    def create_opt_run(run_id, org_id, source, n_traces, axes, reward_spec, rewriter_llm):
        state["runs"][run_id] = {
            "id": run_id, "org_id": org_id, "status": "pending",
            "source": source, "n_traces": n_traces, "axes": axes,
            "reward_spec": reward_spec, "rewriter_llm": rewriter_llm,
            "created_at": None, "started_at": None, "completed_at": None,
            "failure_reason": None,
        }

    def get_opt_run(run_id):
        return state["runs"].get(run_id)

    def update_opt_run_status(run_id, status, failure_reason=None):
        if run_id in state["runs"]:
            state["runs"][run_id]["status"] = status
            if failure_reason:
                state["runs"][run_id]["failure_reason"] = failure_reason

    def save_opt_report(run_id, understanding, summary, tier1, tier2, tier3, pareto_data, markdown):
        state["reports"][run_id] = {
            "run_id": run_id,
            "understanding": understanding, "summary": summary,
            "tier1": tier1, "tier2": tier2, "tier3": tier3,
            "pareto_data": pareto_data, "markdown": markdown,
        }

    def get_opt_report(run_id):
        return state["reports"].get(run_id)

    def count_opt_runs_used(org_id):
        return state["org_used"].get(org_id, 0), state["org_limit"].get(org_id, 15)

    def increment_opt_runs_used(org_id):
        state["org_used"][org_id] = state["org_used"].get(org_id, 0) + 1

    def decrement_opt_runs_used(org_id):
        state["org_used"][org_id] = max(0, state["org_used"].get(org_id, 0) - 1)

    def get_opt_run_traces(run_id):
        return (
            [{"query": "q1", "response": "r1", "id": "t1",
              "config_snapshot": {"system_prompt": "You are helpful."}}],
            1,
        )

    ns = type("FakeDB", (), {})()
    ns.runs = state["runs"]
    ns.reports = state["reports"]
    ns.org_used = state["org_used"]
    ns.org_limit = state["org_limit"]
    ns.create_opt_run = create_opt_run
    ns.get_opt_run = get_opt_run
    ns.update_opt_run_status = update_opt_run_status
    ns.save_opt_report = save_opt_report
    ns.get_opt_report = get_opt_report
    ns.count_opt_runs_used = count_opt_runs_used
    ns.increment_opt_runs_used = increment_opt_runs_used
    ns.decrement_opt_runs_used = decrement_opt_runs_used
    ns.get_opt_run_traces = get_opt_run_traces
    return ns


def test_noop_optimize_e2e(fake_db):
    """Drive the full loop: preauthorize → background orchestrator completes
    → long-poll delivers JobComplete → report fetch → commit."""
    # Reset the job queue between tests
    from server.optimizer.job_queue import get_queue, get_results
    get_queue().reset()
    get_results().reset()

    # Prompt agent returns one baseline scored candidate (only round=0, so no tier1 entry).
    _baseline_cand = PromptCandidate(
        candidate_id="cand_base", system_prompt="You are helpful.", generation_round=0
    )

    mock_prompt_agent = MagicMock()
    mock_prompt_agent.run_iterative.return_value = [
        ScoredCandidate(
            candidate=_baseline_cand, scalar_score=5.0,
            dimensions={"llm_judge": 5.0}, guardrails_held=True,
        ),
    ]

    with patch("server.routes.optimize.db", fake_db), \
         patch("server.routes.jobs.db", fake_db), \
         patch("server.optimizer.orchestrator.db", fake_db), \
         patch("server.optimizer.orchestrator.PromptOptimizerAgent",
               return_value=mock_prompt_agent), \
         patch("server.optimizer.orchestrator.get_queue"), \
         patch("server.optimizer.orchestrator.get_results"), \
         patch("server.routes.optimize.require_auth", return_value={"org": "org_1"}), \
         patch("server.routes.jobs.require_auth", return_value={"org": "org_1"}):

        client = TestClient(app)
        auth = {"Authorization": "Bearer test-key"}

        # Step 1: preauthorize
        r = client.post(
            "/api/v1/optimize/preauthorize",
            json={"source": "last_n_traces", "n_traces": 10, "axes": ["prompt"]},
            headers=auth,
        )
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        # By here BackgroundTasks ran synchronously (TestClient waits for them).
        # The noop orchestrator should have completed and pushed JobComplete
        # via _notify_complete_when_ready.
        # Status check:
        assert fake_db.runs[run_id]["status"] == "completed"

        # Step 2: long-poll — should see job_complete immediately
        r = client.get(f"/api/v1/jobs/pending?run_id={run_id}&timeout=5", headers=auth)
        assert r.status_code == 200
        assert r.json()["type"] == "job_complete"

        # Step 3: fetch report
        r = client.get(f"/api/v1/optimize/{run_id}/report", headers=auth)
        assert r.status_code == 200
        body = r.json()
        assert body["tier1"] == []
        assert body["tier2"] == []
        assert body["tier3"] == []
        assert "markdown" in body
        assert "Optimization Report" in body["markdown"]

        # Step 4: commit (no-op in Phase 1)
        r = client.post(f"/api/v1/optimize/{run_id}/commit", headers=auth)
        assert r.status_code == 200


def test_quota_exhausted_returns_402(fake_db):
    # Exhaust quota
    fake_db.org_used["org_1"] = 15

    with patch("server.routes.optimize.db", fake_db), \
         patch("server.routes.optimize.require_auth", return_value={"org": "org_1"}):

        client = TestClient(app)
        r = client.post(
            "/api/v1/optimize/preauthorize",
            json={"source": "last_n_traces", "n_traces": 10, "axes": ["prompt"]},
            headers={"Authorization": "Bearer test-key"},
        )
        assert r.status_code == 402
