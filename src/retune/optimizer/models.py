"""Shared Pydantic envelope models for SDK↔cloud optimizer protocol."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunCandidateMsg(BaseModel):
    """Cloud → SDK: execute this candidate config over this query set."""
    run_id: str
    candidate_id: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    query_set: list[dict[str, Any]] = Field(default_factory=list)


class CandidateResultMsg(BaseModel):
    """SDK → Cloud: here is the trace + eval scores for candidate X."""
    run_id: str
    candidate_id: str
    trace: dict[str, Any]
    eval_scores: dict[str, float]


class JobCompleteMsg(BaseModel):
    """Cloud → SDK: run is done, fetch the report at this URL."""
    run_id: str
    report_url: str


class JobFailedMsg(BaseModel):
    """Cloud → SDK: run failed. Slot will be refunded."""
    run_id: str
    reason: str


class Suggestion(BaseModel):
    """One entry in the tiered apply-manifest."""
    tier: Literal[1, 2, 3]
    axis: Literal["prompt", "tools", "rag"]
    title: str
    description: str
    confidence: Literal["H", "M", "L"]
    estimated_impact: dict[str, float] = Field(default_factory=dict)
    evidence_trace_ids: list[str] = Field(default_factory=list)
    apply_payload: dict[str, Any] | None = None
    code_snippet: str | None = None


class OptimizationReport(BaseModel):
    """Final report rendered by ReportWriterAgent."""
    run_id: str
    understanding: str
    summary: dict[str, float]
    tier1: list[Suggestion]
    tier2: list[Suggestion]
    tier3: list[Suggestion]
    pareto_data: list[dict[str, Any]]
    feedback: str | None = None
