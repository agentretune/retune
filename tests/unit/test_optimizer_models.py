"""Tests for shared optimizer envelope models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from retune.optimizer.models import (
    RunCandidateMsg,
    CandidateResultMsg,
    JobCompleteMsg,
    Suggestion,
    OptimizationReport,
)


def test_run_candidate_msg_roundtrip():
    msg = RunCandidateMsg(
        run_id="run_1",
        candidate_id="cand_1",
        config_overrides={"retrieval_k": 8},
        query_set=[{"query": "hello", "trace_id": "t1"}],
    )
    parsed = RunCandidateMsg.model_validate(msg.model_dump())
    assert parsed.run_id == "run_1"
    assert parsed.config_overrides["retrieval_k"] == 8
    assert len(parsed.query_set) == 1


def test_candidate_result_msg_requires_eval_scores():
    with pytest.raises(ValidationError):
        CandidateResultMsg(run_id="r", candidate_id="c", trace={})


def test_suggestion_tier_must_be_1_2_or_3():
    with pytest.raises(ValidationError):
        Suggestion(
            tier=4, axis="prompt", title="x", description="y",
            confidence="H", estimated_impact={},
        )


def test_optimization_report_empty_renders():
    report = OptimizationReport(
        run_id="r",
        understanding="",
        summary={"baseline_score": 0.0, "best_score": 0.0, "improvement_pct": 0.0},
        tier1=[], tier2=[], tier3=[],
        pareto_data=[],
    )
    assert report.run_id == "r"
    assert report.tier1 == []


def test_job_complete_msg_carries_report_url():
    msg = JobCompleteMsg(run_id="r", report_url="/v1/optimize/r/report")
    assert msg.run_id == "r"
