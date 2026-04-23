"""Phase 4 E2E: SDK uploads traces + retrieval_config → Orchestrator dispatches
RAG subagent → report has Tier 2 retrieval suggestion with apply_retrieval_override
action."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from server.app import app
from server.optimizer.models import PromptCandidate, RAGCandidate, ScoredCandidate


@pytest.fixture
def fake_db_phase4():
    """Fake DB with traces + tools + retrieval_config storage."""
    state = {
        "runs": {}, "reports": {}, "traces_bundles": {},
        "tools_bundles": {}, "retrieval_configs": {},
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

    def save_opt_run_retrieval_config(run_id, config):
        state["retrieval_configs"][run_id] = dict(config)

    def get_opt_run_retrieval_config(run_id):
        return state["retrieval_configs"].get(run_id)

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
    for k, v in state.items():
        setattr(ns, k, v)
    ns.create_opt_run = create_opt_run
    ns.get_opt_run = get_opt_run
    ns.update_opt_run_status = update_opt_run_status
    ns.save_opt_run_traces = save_opt_run_traces
    ns.get_opt_run_traces = get_opt_run_traces
    ns.save_opt_run_tools = save_opt_run_tools
    ns.get_opt_run_tools = get_opt_run_tools
    ns.save_opt_run_retrieval_config = save_opt_run_retrieval_config
    ns.get_opt_run_retrieval_config = get_opt_run_retrieval_config
    ns.save_opt_report = save_opt_report
    ns.get_opt_report = get_opt_report
    ns.count_opt_runs_used = count_opt_runs_used
    ns.increment_opt_runs_used = increment_opt_runs_used
    ns.decrement_opt_runs_used = decrement_opt_runs_used
    return ns


def test_phase4_rag_routing_e2e(fake_db_phase4):
    """Full flow: upload traces + retrieval_config → Orchestrator → Tier 2 RAG suggestion."""
    from server.optimizer.job_queue import get_queue, get_results
    get_queue().reset()
    get_results().reset()

    traces = [
        {"id": f"t{i}", "query": f"q{i}", "response": f"r{i}",
         "config_snapshot": {"system_prompt": "BASE"},
         "steps": [{"type": "retrieval", "output": [{"page_content": f"doc{i}"}],
                    "duration_ms": 20}]}
        for i in range(3)
    ]
    retrieval_config = {
        "retrieval_k": 3, "chunk_size": 1500,   # large chunk → triggers chunk_sweep
        "retrieval_strategy": "dense", "reranker_enabled": False,
    }

    baseline_prompt_cand = PromptCandidate(
        candidate_id="cp_b", system_prompt="BASE", generation_round=0,
    )
    rag_baseline = RAGCandidate(
        candidate_id="rag_b", config_overrides={}, kind="baseline",
        rationale="Current retrieval config.",
    )
    rag_variant = RAGCandidate(
        candidate_id="rag_v1", config_overrides={"chunk_size": 800, "chunk_overlap": 150},
        kind="chunk_sweep",
        rationale="Chunk size of 1500 is large; 800 often gives better precision.",
    )

    with patch("server.routes.optimize.db", fake_db_phase4), \
         patch("server.routes.jobs.db", fake_db_phase4), \
         patch("server.optimizer.orchestrator.db", fake_db_phase4), \
         patch("server.routes.optimize.require_auth", return_value={"org": "org_1"}), \
         patch("server.routes.jobs.require_auth", return_value={"org": "org_1"}), \
         patch("server.optimizer.orchestrator.PromptOptimizerAgent") as mock_prompt_cls, \
         patch("server.optimizer.orchestrator.RAGOptimizerAgent") as mock_rag_cls, \
         patch("server.routes.optimize.BackgroundTasks.add_task"):

        mock_prompt = MagicMock()
        mock_prompt.run_iterative.return_value = [
            ScoredCandidate(
                candidate=baseline_prompt_cand, scalar_score=5.0,
                dimensions={"llm_judge": 5.0}, guardrails_held=True,
            ),
        ]
        mock_prompt_cls.return_value = mock_prompt

        mock_rag = MagicMock()
        mock_rag.propose_candidates.return_value = [rag_baseline, rag_variant]
        mock_rag.conceptual_suggestions.return_value = []
        mock_rag_cls.return_value = mock_rag

        client = TestClient(app)
        auth = {"Authorization": "Bearer test"}

        # Preauthorize with all three: traces, (no tool metadata), retrieval_config
        r = client.post(
            "/api/v1/optimize/preauthorize",
            json={
                "source": "last_n_traces", "n_traces": 3,
                "axes": ["prompt", "rag"],
                "traces": traces,
                "retrieval_config": retrieval_config,
            },
            headers=auth,
        )
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        from server.optimizer.orchestrator import OptimizerOrchestrator
        OptimizerOrchestrator().run(run_id, candidate_result_timeout=0.3)

        # Verify RAG dispatched
        mock_rag_cls.assert_called_once()
        mock_rag.propose_candidates.assert_called_once()

        # Fetch report — tier2 should have the chunk_sweep suggestion
        r = client.get(f"/api/v1/optimize/{run_id}/report", headers=auth)
        assert r.status_code == 200
        body = r.json()
        rag_tier2 = [s for s in body["tier2"] if s.get("axis") == "rag"]
        assert len(rag_tier2) >= 1, f"Missing rag tier2: {body['tier2']}"
        # apply_payload.action should be apply_retrieval_override
        first_rag = rag_tier2[0]
        assert first_rag["apply_payload"]["action"] == "apply_retrieval_override"
        assert first_rag["apply_payload"]["chunk_size"] == 800
