# Optimizer Phase 1 (Infra) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the optimizer infrastructure end-to-end with a **noop Orchestrator** so every other piece (SDK↔Cloud protocol, billing preauth/commit, JudgeAgent, ReportWriter shell, DB schema) is proven before Phase 2 wires in real subagents.

**Architecture:** Cloud FastAPI service exposes `/v1/optimize/*` (preauthorize, commit, cancel, get_run, get_report) and `/v1/jobs/*` (long-poll pending, submit result). The SDK's new `retuner.optimize()` method calls preauthorize, spawns a long-poll worker that listens for `RunCandidate` commands, then waits for `JobComplete`. In Phase 1 the Orchestrator dispatches nothing, so the cloud run transitions `pending → running → completed` immediately and the SDK receives an empty but well-formed report. The JudgeAgent, reward parser, and ReportWriter shell are fully built and unit-tested — they're just not yet invoked in the noop flow (Phase 2 wires them).

**Tech Stack:** Python 3.10+, FastAPI (existing), PostgreSQL (existing), Pydantic v2 (existing), pytest + pytest-asyncio (existing), urllib (existing SDK HTTP pattern). No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-04-12-optimizer-design.md`

---

## File Structure

### New files — SDK side (`src/retune/optimizer/`)

| File | Responsibility |
|---|---|
| `src/retune/optimizer/__init__.py` | Public exports: `OptimizationReport`, `OptimizeJob` |
| `src/retune/optimizer/models.py` | Pydantic models shared with cloud: `RunCandidateMsg`, `CandidateResultMsg`, `JobCompleteMsg`, `OptimizationReport`, `Suggestion` (tiered) |
| `src/retune/optimizer/client.py` | `OptimizerClient` — HTTP wrappers for `/v1/optimize/*` (preauthorize, commit, cancel, get_report) |
| `src/retune/optimizer/worker.py` | `SDKWorker` — long-poll loop; consumes `RunCandidate`, executes candidate against the wrapped agent, posts `CandidateResult` back |
| `src/retune/optimizer/report.py` | `OptimizationReport.show()` / `apply(tier=1)` / `copy_snippets()` |

### Modified files — SDK side

| File | Change |
|---|---|
| `src/retune/wrapper.py` | Add `Retuner.optimize(source, n, axes, reward, rewriter_llm, guardrails)` method; require `agent_purpose` in `__init__` when `Mode.IMPROVE` |
| `src/retune/usage_gate.py` | Extend gate: new `preauthorize_run()` / `commit_run()` / `refund_run()` methods that hit `/v1/optimize/*` |
| `src/retune/__init__.py` | Re-export `OptimizationReport`, `OptimizeJob` |

### New files — cloud side (`retune-cloud/server/`)

| File | Responsibility |
|---|---|
| `retune-cloud/server/optimizer/__init__.py` | Package init |
| `retune-cloud/server/optimizer/models.py` | Pydantic: `OptimizationRun` (DB row), `Candidate`, `Report`, `RewardSpec`, message envelopes |
| `retune-cloud/server/optimizer/reward_parser.py` | `parse_reward_spec(dict) -> RewardSpec` with validation |
| `retune-cloud/server/optimizer/judge.py` | `JudgeAgent.score(candidate, reward_spec, baseline) -> float + dimensions` |
| `retune-cloud/server/optimizer/report_writer.py` | `ReportWriterAgent.render(run) -> Report` (markdown + JSON) |
| `retune-cloud/server/optimizer/orchestrator.py` | `OptimizerOrchestrator.run(run_id)` — Phase 1 noop (sets status=completed immediately) |
| `retune-cloud/server/optimizer/job_queue.py` | In-memory per-run queue of pending `RunCandidate` messages (Phase 1; Redis in later phases) |
| `retune-cloud/server/routes/optimize.py` | Routes: `POST /preauthorize`, `POST /{run_id}/commit`, `POST /{run_id}/cancel`, `GET /{run_id}`, `GET /{run_id}/report` |
| `retune-cloud/server/routes/jobs.py` | Routes: `GET /pending?run_id=...` (long-poll), `POST /result` |

### Modified files — cloud side

| File | Change |
|---|---|
| `retune-cloud/server/db/schema.sql` | Append `optimization_runs` + `optimization_candidates` + `optimization_reports` tables |
| `retune-cloud/server/db/postgres.py` | Add helpers: `create_opt_run`, `get_opt_run`, `update_opt_run_status`, `save_opt_report`, `count_opt_runs_used` |
| `retune-cloud/server/routes/billing.py` | `get_usage` returns new `{runs_used, runs_limit}` fields |
| `retune-cloud/server/app.py` | Include new `optimize` + `jobs` routers |

### New tests

| File | Covers |
|---|---|
| `tests/unit/test_optimizer_models.py` | Pydantic model validation |
| `tests/unit/test_reward_parser.py` | Declarative reward JSON parsing + validation errors |
| `tests/unit/test_judge_agent.py` | Default judge+guardrails scoring; guardrail violation → 0 |
| `tests/unit/test_report_writer.py` | Markdown + JSON rendering with tiered suggestions (incl. empty) |
| `tests/unit/test_orchestrator_noop.py` | Noop orchestrator transitions run pending→completed with empty report |
| `tests/unit/test_optimizer_client.py` | HTTP client mocks (preauthorize, commit) |
| `tests/unit/test_optimize_routes.py` | Cloud route unit tests (mocked DB) |
| `tests/unit/test_jobs_routes.py` | Long-poll endpoint with mocked queue |
| `tests/unit/test_usage_gate_per_run.py` | `preauthorize_run` / `commit_run` / `refund_run` |
| `tests/integration/test_optimize_e2e.py` | Full E2E: SDK triggers optimize → cloud creates run → orchestrator noop completes → SDK receives empty report → slot committed |

---

## Task Summary

Fifteen tasks. Build bottom-up so every later task can call working code from earlier tasks.

1. Shared Pydantic models (both sides)
2. Cloud DB schema + migration
3. Cloud DB helpers
4. Reward parser
5. JudgeAgent (default + declarative)
6. ReportWriterAgent shell
7. OptimizerOrchestrator noop
8. Job queue + jobs routes (long-poll)
9. Optimize routes (preauth/commit/cancel/get)
10. Wire routers into `app.py`; extend `billing.get_usage`
11. SDK `OptimizerClient`
12. SDK `SDKWorker` (long-poll consumer)
13. SDK `OptimizationReport` + `Retuner.optimize()` method
14. SDK `UsageGate.preauthorize_run` / `commit_run` / `refund_run`
15. End-to-end integration test

---

## Task 1: Shared Pydantic models

**Files:**
- Create: `src/retune/optimizer/__init__.py`
- Create: `src/retune/optimizer/models.py`
- Create: `retune-cloud/server/optimizer/__init__.py`
- Create: `retune-cloud/server/optimizer/models.py`
- Test: `tests/unit/test_optimizer_models.py`

The SDK and cloud hold structurally identical envelope models. We duplicate rather than sharing a package to avoid coupling — the cloud Pydantic models may later grow DB-specific fields that don't belong in the SDK.

- [ ] **Step 1.1: Create empty package init files**

```python
# src/retune/optimizer/__init__.py
"""Retune cloud optimizer client-side."""
```

```python
# retune-cloud/server/optimizer/__init__.py
"""Retune cloud-side optimizer (orchestrator, judge, report writer)."""
```

- [ ] **Step 1.2: Write the failing model tests**

```python
# tests/unit/test_optimizer_models.py
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
```

Run: `pytest tests/unit/test_optimizer_models.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 1.3: Implement SDK-side models**

```python
# src/retune/optimizer/models.py
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
```

- [ ] **Step 1.4: Implement cloud-side models**

```python
# retune-cloud/server/optimizer/models.py
"""Cloud-side optimizer models. Shape-compatible with src/retune/optimizer/models.py."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class OptimizationRun(BaseModel):
    """Maps to optimization_runs DB row."""
    id: str
    org_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    source: Literal["last_n_traces", "collect_next"]
    n_traces: int
    axes: list[str]
    reward_spec: dict[str, Any]
    rewriter_llm: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    slots_consumed: int = 1


class Candidate(BaseModel):
    id: str
    run_id: str
    config_overrides: dict[str, Any]
    scalar_score: float | None = None
    dimensions: dict[str, float] = Field(default_factory=dict)


class RewardSpec(BaseModel):
    """Parsed declarative reward, also used for the default case."""
    objective: Literal["maximize", "minimize"] = "maximize"
    primary_evaluator: str = "llm_judge"
    primary_weight: float = 1.0
    guardrails: list[dict[str, Any]] = Field(default_factory=list)
    soft_penalties: list[dict[str, Any]] = Field(default_factory=list)
    extra_metrics: list[str] = Field(default_factory=list)


class Report(BaseModel):
    run_id: str
    understanding: str
    summary: dict[str, float]
    tier1: list[dict[str, Any]]
    tier2: list[dict[str, Any]]
    tier3: list[dict[str, Any]]
    pareto_data: list[dict[str, Any]]
    markdown: str
```

- [ ] **Step 1.5: Run tests, verify pass**

Run: `pytest tests/unit/test_optimizer_models.py -v`
Expected: PASS (5 tests).

- [ ] **Step 1.6: Commit**

```bash
git add src/retune/optimizer/__init__.py src/retune/optimizer/models.py \
        retune-cloud/server/optimizer/__init__.py retune-cloud/server/optimizer/models.py \
        tests/unit/test_optimizer_models.py
git commit -m "optimizer: add shared SDK↔cloud envelope models"
```

---

## Task 2: Cloud DB schema + migration

**Files:**
- Modify: `retune-cloud/server/db/schema.sql` (append at EOF)

No test at the SQL level — validated indirectly when DB helpers land in Task 3.

- [ ] **Step 2.1: Append schema to `schema.sql`**

```sql
-- ============ Optimizer v0.3 (Phase 1) ============

-- Optimization runs
CREATE TABLE IF NOT EXISTS optimization_runs (
    id VARCHAR(64) PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    source VARCHAR(20) NOT NULL,
    n_traces INTEGER NOT NULL,
    axes JSONB NOT NULL DEFAULT '[]',
    reward_spec JSONB NOT NULL DEFAULT '{}',
    rewriter_llm VARCHAR(100),
    slots_consumed INTEGER NOT NULL DEFAULT 1,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    failure_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_opt_runs_org
    ON optimization_runs(org_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_opt_runs_status
    ON optimization_runs(status) WHERE status IN ('pending','running');

-- Candidate executions within a run
CREATE TABLE IF NOT EXISTS optimization_candidates (
    id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES optimization_runs(id) ON DELETE CASCADE,
    config_overrides JSONB NOT NULL DEFAULT '{}',
    scalar_score FLOAT,
    dimensions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_opt_candidates_run
    ON optimization_candidates(run_id);

-- Persistent reports (survive raw-trace purge)
CREATE TABLE IF NOT EXISTS optimization_reports (
    run_id VARCHAR(64) PRIMARY KEY REFERENCES optimization_runs(id) ON DELETE CASCADE,
    understanding TEXT NOT NULL DEFAULT '',
    summary JSONB NOT NULL DEFAULT '{}',
    tier1 JSONB NOT NULL DEFAULT '[]',
    tier2 JSONB NOT NULL DEFAULT '[]',
    tier3 JSONB NOT NULL DEFAULT '[]',
    pareto_data JSONB NOT NULL DEFAULT '[]',
    markdown TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Usage counter addition: per-org lifetime optimize-run count (free-trial gate)
ALTER TABLE organizations
    ADD COLUMN IF NOT EXISTS optimize_runs_used INTEGER NOT NULL DEFAULT 0;
ALTER TABLE organizations
    ADD COLUMN IF NOT EXISTS optimize_runs_limit INTEGER NOT NULL DEFAULT 15;
```

- [ ] **Step 2.2: Commit**

```bash
git add retune-cloud/server/db/schema.sql
git commit -m "optimizer: add DB schema for runs, candidates, reports"
```

---

## Task 3: Cloud DB helpers

**Files:**
- Modify: `retune-cloud/server/db/postgres.py`
- Test: `retune-cloud/tests/test_opt_db.py` (create)

We test with a real Postgres only if available; fall back to mocked helpers if not. Existing `postgres.py` follows the pattern of raising on no connection — we match.

- [ ] **Step 3.1: Write failing tests (mock connection)**

```python
# retune-cloud/tests/test_opt_db.py
"""Unit tests for optimizer DB helpers (mocked connection)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from server.db import postgres as db


@patch.object(db, "_conn", None)
@patch.object(db, "get_connection")
def test_create_opt_run_inserts_row(mock_get_conn):
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    mock_get_conn.return_value = conn

    db.create_opt_run(
        run_id="run_1", org_id="org_1",
        source="last_n_traces", n_traces=50, axes=["prompt"],
        reward_spec={"objective": "maximize"},
        rewriter_llm="claude-3-7-sonnet",
    )
    assert cur.execute.called
    sql = cur.execute.call_args[0][0]
    assert "INSERT INTO optimization_runs" in sql


@patch.object(db, "get_connection")
def test_count_opt_runs_used(mock_get_conn):
    cur = MagicMock()
    cur.fetchone.return_value = {"optimize_runs_used": 5, "optimize_runs_limit": 15}
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    mock_get_conn.return_value = conn

    used, limit = db.count_opt_runs_used("org_1")
    assert used == 5
    assert limit == 15


@patch.object(db, "get_connection")
def test_update_opt_run_status(mock_get_conn):
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    mock_get_conn.return_value = conn

    db.update_opt_run_status("run_1", "completed")
    assert "UPDATE optimization_runs" in cur.execute.call_args[0][0]
```

Run: `pytest retune-cloud/tests/test_opt_db.py -v`
Expected: FAIL — functions do not exist.

- [ ] **Step 3.2: Append DB helpers to `postgres.py`**

Add at the end of the existing `retune-cloud/server/db/postgres.py`:

```python
# ============ Optimizer v0.3 helpers ============

import json as _json


def create_opt_run(
    run_id: str,
    org_id: str,
    source: str,
    n_traces: int,
    axes: list[str],
    reward_spec: dict,
    rewriter_llm: str | None,
) -> None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO optimization_runs
                (id, org_id, status, source, n_traces, axes, reward_spec, rewriter_llm)
            VALUES (%s, %s, 'pending', %s, %s, %s::jsonb, %s::jsonb, %s)
            """,
            (
                run_id, org_id, source, n_traces,
                _json.dumps(axes), _json.dumps(reward_spec), rewriter_llm,
            ),
        )
    conn.commit()


def get_opt_run(run_id: str) -> dict | None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM optimization_runs WHERE id = %s",
            (run_id,),
        )
        return cur.fetchone()


def update_opt_run_status(
    run_id: str,
    status: str,
    failure_reason: str | None = None,
) -> None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE optimization_runs
               SET status = %s,
                   failure_reason = %s,
                   started_at = COALESCE(started_at,
                                   CASE WHEN %s = 'running' THEN NOW() ELSE NULL END),
                   completed_at = CASE WHEN %s IN ('completed','failed','cancelled')
                                       THEN NOW() ELSE completed_at END
             WHERE id = %s
            """,
            (status, failure_reason, status, status, run_id),
        )
    conn.commit()


def save_opt_report(
    run_id: str,
    understanding: str,
    summary: dict,
    tier1: list,
    tier2: list,
    tier3: list,
    pareto_data: list,
    markdown: str,
) -> None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO optimization_reports
                (run_id, understanding, summary, tier1, tier2, tier3, pareto_data, markdown)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (run_id) DO UPDATE
              SET understanding = EXCLUDED.understanding,
                  summary = EXCLUDED.summary,
                  tier1 = EXCLUDED.tier1,
                  tier2 = EXCLUDED.tier2,
                  tier3 = EXCLUDED.tier3,
                  pareto_data = EXCLUDED.pareto_data,
                  markdown = EXCLUDED.markdown
            """,
            (
                run_id, understanding,
                _json.dumps(summary),
                _json.dumps(tier1),
                _json.dumps(tier2),
                _json.dumps(tier3),
                _json.dumps(pareto_data),
                markdown,
            ),
        )
    conn.commit()


def get_opt_report(run_id: str) -> dict | None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM optimization_reports WHERE run_id = %s",
            (run_id,),
        )
        return cur.fetchone()


def count_opt_runs_used(org_id: str) -> tuple[int, int]:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT optimize_runs_used, optimize_runs_limit FROM organizations WHERE id = %s",
            (org_id,),
        )
        row = cur.fetchone()
        if not row:
            return 0, 15
        return int(row["optimize_runs_used"]), int(row["optimize_runs_limit"])


def increment_opt_runs_used(org_id: str) -> None:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE organizations SET optimize_runs_used = optimize_runs_used + 1 WHERE id = %s",
            (org_id,),
        )
    conn.commit()


def decrement_opt_runs_used(org_id: str) -> None:
    """Refund on run failure."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE organizations SET optimize_runs_used = GREATEST(0, optimize_runs_used - 1) WHERE id = %s",
            (org_id,),
        )
    conn.commit()
```

- [ ] **Step 3.3: Run tests, verify pass**

Run: `pytest retune-cloud/tests/test_opt_db.py -v`
Expected: PASS (3 tests).

- [ ] **Step 3.4: Commit**

```bash
git add retune-cloud/server/db/postgres.py retune-cloud/tests/test_opt_db.py
git commit -m "optimizer: add DB helpers for runs, reports, usage counter"
```

---

## Task 4: Reward parser

**Files:**
- Create: `retune-cloud/server/optimizer/reward_parser.py`
- Test: `tests/unit/test_reward_parser.py`

- [ ] **Step 4.1: Write failing tests**

```python
# tests/unit/test_reward_parser.py
"""Reward spec declarative JSON parser."""
from __future__ import annotations

import pytest

from server.optimizer.reward_parser import parse_reward_spec, default_reward_spec


def test_default_reward_spec():
    spec = default_reward_spec()
    assert spec.primary_evaluator == "llm_judge"
    assert spec.objective == "maximize"
    # Default guardrails: cost <= 1.5x baseline, p95 latency <= 1.2x baseline
    assert len(spec.guardrails) == 2
    names = [g["evaluator"] for g in spec.guardrails]
    assert "cost" in names
    assert "latency" in names


def test_parse_declarative_reward():
    raw = {
        "objective": "maximize",
        "primary": {"evaluator": "llm_judge", "weight": 1.0},
        "penalties": [
            {"evaluator": "cost",    "threshold": "<= 0.002", "hard": True},
            {"evaluator": "latency", "threshold": "<= 3.0",   "hard": False, "weight": 0.2},
        ],
        "extra_metrics": [{"evaluator": "retrieval_precision"}],
    }
    spec = parse_reward_spec(raw)
    assert spec.primary_evaluator == "llm_judge"
    assert len(spec.guardrails) == 1   # hard penalty
    assert len(spec.soft_penalties) == 1
    assert spec.extra_metrics == ["retrieval_precision"]


def test_parse_rejects_invalid_objective():
    with pytest.raises(ValueError, match="objective"):
        parse_reward_spec({"objective": "sideways", "primary": {"evaluator": "x"}})


def test_parse_rejects_missing_primary():
    with pytest.raises(ValueError, match="primary"):
        parse_reward_spec({"objective": "maximize"})


def test_parse_rejects_malformed_threshold():
    with pytest.raises(ValueError, match="threshold"):
        parse_reward_spec({
            "primary": {"evaluator": "x"},
            "penalties": [{"evaluator": "cost", "threshold": "cheap"}],
        })
```

Run: `pytest tests/unit/test_reward_parser.py -v`
Expected: FAIL.

- [ ] **Step 4.2: Implement parser**

```python
# retune-cloud/server/optimizer/reward_parser.py
"""Parse declarative reward specs into internal RewardSpec objects."""
from __future__ import annotations

import re
from typing import Any

from server.optimizer.models import RewardSpec

_THRESHOLD_RE = re.compile(r"^\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)\s*s?\s*$")


def _validate_threshold(raw: str) -> None:
    if not isinstance(raw, str) or not _THRESHOLD_RE.match(raw):
        raise ValueError(
            f"Invalid threshold {raw!r}: expected '<= 0.002', '>= 3.0', etc."
        )


def default_reward_spec() -> RewardSpec:
    return RewardSpec(
        objective="maximize",
        primary_evaluator="llm_judge",
        primary_weight=1.0,
        guardrails=[
            {"evaluator": "cost",    "threshold": "<= 1.5",  "relative_to_baseline": True},
            {"evaluator": "latency", "threshold": "<= 1.2",  "relative_to_baseline": True},
        ],
        soft_penalties=[],
        extra_metrics=[],
    )


def parse_reward_spec(raw: dict[str, Any]) -> RewardSpec:
    """Parse a user-supplied declarative reward JSON into RewardSpec."""
    objective = raw.get("objective", "maximize")
    if objective not in ("maximize", "minimize"):
        raise ValueError(f"Invalid objective {objective!r}; must be 'maximize' or 'minimize'")

    primary = raw.get("primary")
    if not isinstance(primary, dict) or "evaluator" not in primary:
        raise ValueError("Reward spec missing required 'primary' with 'evaluator'")

    guardrails: list[dict[str, Any]] = []
    soft_penalties: list[dict[str, Any]] = []
    for pen in raw.get("penalties", []):
        if "evaluator" not in pen or "threshold" not in pen:
            raise ValueError("Each penalty requires 'evaluator' and 'threshold'")
        _validate_threshold(pen["threshold"])
        (guardrails if pen.get("hard") else soft_penalties).append(pen)

    extra = []
    for m in raw.get("extra_metrics", []):
        if isinstance(m, dict) and "evaluator" in m:
            extra.append(m["evaluator"])

    return RewardSpec(
        objective=objective,
        primary_evaluator=primary["evaluator"],
        primary_weight=float(primary.get("weight", 1.0)),
        guardrails=guardrails,
        soft_penalties=soft_penalties,
        extra_metrics=extra,
    )
```

- [ ] **Step 4.3: Configure pytest to find cloud server package**

The tests import `server.optimizer.reward_parser`. Cloud tests need `retune-cloud/` on `sys.path`. Append to the existing `pyproject.toml`'s pytest config:

```toml
# pyproject.toml  — inside [tool.pytest.ini_options]
pythonpath = ["src", "retune-cloud"]
```

Edit the existing block in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "retune-cloud/tests"]
asyncio_mode = "auto"
pythonpath = ["src", "retune-cloud"]
```

- [ ] **Step 4.4: Run tests, verify pass**

Run: `pytest tests/unit/test_reward_parser.py -v`
Expected: PASS (5 tests).

- [ ] **Step 4.5: Commit**

```bash
git add retune-cloud/server/optimizer/reward_parser.py \
        tests/unit/test_reward_parser.py pyproject.toml
git commit -m "optimizer: declarative reward JSON parser"
```

---

## Task 5: JudgeAgent

**Files:**
- Create: `retune-cloud/server/optimizer/judge.py`
- Test: `tests/unit/test_judge_agent.py`

Phase 1 scope: Judge computes `scalar_score = primary_rating × guardrails_held ? 1 : 0` from a candidate's eval_scores dict. No LLM call yet — the Judge receives pre-computed scores (the SDK's evaluators produce them in the CandidateResultMsg). Phase 2+ wires the judge LLM for rubric-based scoring of arbitrary outputs.

- [ ] **Step 5.1: Write failing tests**

```python
# tests/unit/test_judge_agent.py
"""JudgeAgent scoring logic (default + declarative)."""
from __future__ import annotations

from server.optimizer.judge import JudgeAgent
from server.optimizer.reward_parser import default_reward_spec


def test_scalar_score_default_passes_guardrails():
    judge = JudgeAgent()
    spec = default_reward_spec()
    baseline = {"cost": 0.001, "latency": 1.0}
    scores = {"llm_judge": 8.0, "cost": 0.0011, "latency": 1.1}  # within 1.5×/1.2×
    result = judge.score(scores, spec, baseline)
    assert result.scalar_score == 8.0
    assert result.guardrails_held is True


def test_scalar_score_default_guardrail_violation_zeroes_out():
    judge = JudgeAgent()
    spec = default_reward_spec()
    baseline = {"cost": 0.001, "latency": 1.0}
    scores = {"llm_judge": 8.0, "cost": 0.01, "latency": 1.0}  # 10× cost
    result = judge.score(scores, spec, baseline)
    assert result.scalar_score == 0.0
    assert result.guardrails_held is False
    assert "cost" in result.guardrail_violations


def test_scalar_score_declarative_soft_penalty():
    """Soft penalty reduces score but doesn't zero it."""
    judge = JudgeAgent()
    from server.optimizer.reward_parser import parse_reward_spec
    spec = parse_reward_spec({
        "primary": {"evaluator": "llm_judge", "weight": 1.0},
        "penalties": [
            {"evaluator": "latency", "threshold": "<= 1.0", "hard": False, "weight": 0.5},
        ],
    })
    scores = {"llm_judge": 8.0, "latency": 2.0}  # over threshold
    result = judge.score(scores, spec, baseline={})
    # Soft penalty: score = primary - weight × (over_amount / threshold_value)
    # penalty = 0.5 × (2.0 - 1.0)/1.0 = 0.5
    assert result.scalar_score == 7.5
    assert result.guardrails_held is True


def test_dimensions_always_logged():
    judge = JudgeAgent()
    spec = default_reward_spec()
    scores = {"llm_judge": 7.0, "cost": 0.001, "latency": 1.0, "retrieval_precision": 0.9}
    result = judge.score(scores, spec, baseline={"cost": 0.001, "latency": 1.0})
    # All input scores preserved as dimensions for Pareto viz
    assert result.dimensions == scores
```

Run: `pytest tests/unit/test_judge_agent.py -v`
Expected: FAIL.

- [ ] **Step 5.2: Implement JudgeAgent**

```python
# retune-cloud/server/optimizer/judge.py
"""JudgeAgent — scores a candidate against a RewardSpec.

Phase 1: deterministic scoring from pre-computed eval_scores.
Phase 2+: adds LLM-based rubric judging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from server.optimizer.models import RewardSpec


_THRESHOLD_RE = re.compile(r"^\s*(<=|>=|<|>|==)\s*([0-9]*\.?[0-9]+)\s*s?\s*$")


@dataclass
class JudgeResult:
    scalar_score: float
    dimensions: dict[str, float]
    guardrails_held: bool
    guardrail_violations: list[str] = field(default_factory=list)
    soft_penalty_total: float = 0.0


def _parse_threshold(raw: str) -> tuple[str, float]:
    m = _THRESHOLD_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid threshold {raw!r}")
    return m.group(1), float(m.group(2))


def _check(op: str, value: float, threshold: float) -> bool:
    if op == "<=": return value <= threshold
    if op == ">=": return value >= threshold
    if op == "<":  return value < threshold
    if op == ">":  return value > threshold
    if op == "==": return value == threshold
    raise ValueError(f"Unknown op {op}")


def _resolve_target(
    score: float,
    threshold_val: float,
    rule: dict,
    baseline: dict,
    evaluator: str,
) -> float:
    """If rule says relative_to_baseline, multiply threshold by baseline value."""
    if rule.get("relative_to_baseline") and evaluator in baseline:
        return threshold_val * baseline[evaluator]
    return threshold_val


class JudgeAgent:
    def score(
        self,
        eval_scores: dict[str, float],
        spec: RewardSpec,
        baseline: dict[str, float],
    ) -> JudgeResult:
        dimensions = dict(eval_scores)  # preserve for Pareto
        primary = eval_scores.get(spec.primary_evaluator, 0.0)
        scalar = primary * spec.primary_weight

        # Hard guardrails — any violation zeroes the score
        violations: list[str] = []
        for rule in spec.guardrails:
            ev = rule["evaluator"]
            if ev not in eval_scores:
                continue
            op, thr_val = _parse_threshold(rule["threshold"])
            target = _resolve_target(eval_scores[ev], thr_val, rule, baseline, ev)
            if not _check(op, eval_scores[ev], target):
                violations.append(ev)

        if violations:
            return JudgeResult(
                scalar_score=0.0,
                dimensions=dimensions,
                guardrails_held=False,
                guardrail_violations=violations,
            )

        # Soft penalties — reduce (never below zero)
        soft_total = 0.0
        for rule in spec.soft_penalties:
            ev = rule["evaluator"]
            if ev not in eval_scores:
                continue
            op, thr_val = _parse_threshold(rule["threshold"])
            target = _resolve_target(eval_scores[ev], thr_val, rule, baseline, ev)
            weight = float(rule.get("weight", 1.0))
            if not _check(op, eval_scores[ev], target):
                over = abs(eval_scores[ev] - target) / max(abs(target), 1e-9)
                soft_total += weight * over

        scalar = max(0.0, scalar - soft_total)

        return JudgeResult(
            scalar_score=scalar,
            dimensions=dimensions,
            guardrails_held=True,
            soft_penalty_total=soft_total,
        )
```

- [ ] **Step 5.3: Run tests, verify pass**

Run: `pytest tests/unit/test_judge_agent.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5.4: Commit**

```bash
git add retune-cloud/server/optimizer/judge.py tests/unit/test_judge_agent.py
git commit -m "optimizer: JudgeAgent with guardrails and soft penalties"
```

---

## Task 6: ReportWriterAgent shell

**Files:**
- Create: `retune-cloud/server/optimizer/report_writer.py`
- Test: `tests/unit/test_report_writer.py`

Shell because Phase 1 noop never produces real suggestions — but the rendering for empty + tiered cases is fully implemented and tested so Phase 2+ just plugs data in.

- [ ] **Step 6.1: Write failing tests**

```python
# tests/unit/test_report_writer.py
"""ReportWriterAgent markdown + JSON rendering."""
from __future__ import annotations

from server.optimizer.report_writer import ReportWriterAgent


def test_render_empty_report():
    writer = ReportWriterAgent()
    report = writer.render(
        run_id="r",
        understanding="",
        summary={"baseline_score": 0.0, "best_score": 0.0, "improvement_pct": 0.0},
        tier1=[], tier2=[], tier3=[],
        pareto_data=[],
    )
    assert report.run_id == "r"
    assert "No suggestions generated" in report.markdown
    assert report.tier1 == []


def test_render_tiered_suggestions():
    writer = ReportWriterAgent()
    tier1 = [{"axis": "prompt", "title": "Rewrite system prompt",
              "description": "Clearer role framing",
              "confidence": "H",
              "estimated_impact": {"judge": 0.8}}]
    report = writer.render(
        run_id="r2",
        understanding="Customer support bot.",
        summary={"baseline_score": 6.8, "best_score": 7.6, "improvement_pct": 11.8},
        tier1=tier1, tier2=[], tier3=[],
        pareto_data=[{"candidate_id": "c1", "judge": 7.6, "cost": 0.0011, "latency": 0.9}],
    )
    assert "Tier 1" in report.markdown
    assert "Rewrite system prompt" in report.markdown
    assert "+11.8%" in report.markdown
    assert len(report.tier1) == 1


def test_render_includes_understanding_first():
    writer = ReportWriterAgent()
    report = writer.render(
        run_id="r3",
        understanding="This is a RAG bot for billing queries.",
        summary={"baseline_score": 5.0, "best_score": 6.0, "improvement_pct": 20.0},
        tier1=[], tier2=[], tier3=[],
        pareto_data=[],
    )
    md = report.markdown
    # Understanding section must appear before summary
    assert md.index("Understanding") < md.index("Summary")
    assert "This is a RAG bot" in md
```

Run: `pytest tests/unit/test_report_writer.py -v`
Expected: FAIL.

- [ ] **Step 6.2: Implement ReportWriterAgent**

```python
# retune-cloud/server/optimizer/report_writer.py
"""ReportWriterAgent — assembles the tiered apply-manifest as markdown + JSON."""
from __future__ import annotations

from server.optimizer.models import Report


def _format_impact(impact: dict) -> str:
    if not impact:
        return ""
    parts = []
    for k, v in impact.items():
        sign = "+" if v >= 0 else ""
        parts.append(f"{sign}{v:.2f} {k}")
    return " · ".join(parts)


def _render_suggestion_line(s: dict) -> str:
    impact = _format_impact(s.get("estimated_impact", {}))
    confidence = s.get("confidence", "M")
    axis = s.get("axis", "?")
    title = s.get("title", "")
    line = f"- **[{axis}]** {title}  _(confidence: {confidence}_"
    if impact:
        line += f"_, impact: {impact}_"
    line += ")"
    if s.get("description"):
        line += f"\n  - {s['description']}"
    return line


class ReportWriterAgent:
    def render(
        self,
        run_id: str,
        understanding: str,
        summary: dict,
        tier1: list,
        tier2: list,
        tier3: list,
        pareto_data: list,
    ) -> Report:
        md_parts: list[str] = [f"# Optimization Report — `{run_id}`\n"]

        md_parts.append("## Understanding of Your Agent\n")
        if understanding:
            md_parts.append(f"{understanding}\n")
            md_parts.append("_If this is wrong, correct it in the feedback box — it will guide the next run._\n")
        else:
            md_parts.append("_No agent purpose was supplied. Provide `agent_purpose=` at wrapper init for better results._\n")

        md_parts.append("## Summary\n")
        base = summary.get("baseline_score", 0.0)
        best = summary.get("best_score", 0.0)
        pct = summary.get("improvement_pct", 0.0)
        sign = "+" if pct >= 0 else ""
        md_parts.append(f"- Baseline judge score: **{base:.2f}** → Best candidate: **{best:.2f}** ({sign}{pct:.1f}%)\n")

        md_parts.append("## Pareto Frontier\n")
        if pareto_data:
            md_parts.append(f"_{len(pareto_data)} candidates explored — see dashboard for interactive plot._\n")
        else:
            md_parts.append("_No candidates were evaluated in this run._\n")

        if not (tier1 or tier2 or tier3):
            md_parts.append("\n## Suggestions\n\nNo suggestions generated.\n")
        else:
            if tier1:
                md_parts.append("## Tier 1 — One-Click Apply\n")
                md_parts.extend(_render_suggestion_line(s) + "\n" for s in tier1)
            if tier2:
                md_parts.append("\n## Tier 2 — Copy-Paste Snippets\n")
                md_parts.extend(_render_suggestion_line(s) + "\n" for s in tier2)
            if tier3:
                md_parts.append("\n## Tier 3 — Conceptual Suggestions\n")
                md_parts.extend(_render_suggestion_line(s) + "\n" for s in tier3)

        md_parts.append("\n## Your Feedback (feeds next optimization run)\n\n_(submit in dashboard or via SDK)_\n")

        return Report(
            run_id=run_id,
            understanding=understanding,
            summary=summary,
            tier1=tier1,
            tier2=tier2,
            tier3=tier3,
            pareto_data=pareto_data,
            markdown="".join(md_parts),
        )
```

- [ ] **Step 6.3: Run tests, verify pass**

Run: `pytest tests/unit/test_report_writer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6.4: Commit**

```bash
git add retune-cloud/server/optimizer/report_writer.py tests/unit/test_report_writer.py
git commit -m "optimizer: ReportWriterAgent renders markdown + JSON for all tiers"
```

---

## Task 7: OptimizerOrchestrator (noop)

**Files:**
- Create: `retune-cloud/server/optimizer/orchestrator.py`
- Test: `tests/unit/test_orchestrator_noop.py`

The noop orchestrator immediately transitions pending→running→completed and saves an empty report. It's the driver the cloud's background worker calls. Phase 2 replaces its body with real subagent dispatch.

- [ ] **Step 7.1: Write failing tests**

```python
# tests/unit/test_orchestrator_noop.py
"""Phase 1 noop Orchestrator — transitions run to completed with empty report."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from server.optimizer.orchestrator import OptimizerOrchestrator


@patch("server.optimizer.orchestrator.db")
def test_noop_orchestrator_marks_running_then_completed(mock_db):
    mock_db.get_opt_run.return_value = {
        "id": "run_1", "org_id": "org_1", "status": "pending",
        "source": "last_n_traces", "n_traces": 50,
        "axes": ["prompt"], "reward_spec": {}, "rewriter_llm": None,
    }
    orch = OptimizerOrchestrator()
    orch.run("run_1")

    calls = [c.args for c in mock_db.update_opt_run_status.call_args_list]
    # Was set to 'running' then 'completed'
    statuses = [c[1] for c in calls]
    assert "running" in statuses
    assert "completed" in statuses


@patch("server.optimizer.orchestrator.db")
def test_noop_orchestrator_saves_empty_report(mock_db):
    mock_db.get_opt_run.return_value = {
        "id": "run_2", "org_id": "org_1", "status": "pending",
        "source": "last_n_traces", "n_traces": 50,
        "axes": [], "reward_spec": {}, "rewriter_llm": None,
    }
    orch = OptimizerOrchestrator()
    orch.run("run_2")

    assert mock_db.save_opt_report.called
    kwargs = mock_db.save_opt_report.call_args.kwargs
    assert kwargs["run_id"] == "run_2"
    assert kwargs["tier1"] == []
    assert kwargs["tier2"] == []
    assert kwargs["tier3"] == []


@patch("server.optimizer.orchestrator.db")
def test_noop_orchestrator_marks_failed_on_exception(mock_db):
    mock_db.get_opt_run.return_value = None  # simulate missing run
    orch = OptimizerOrchestrator()
    orch.run("run_missing")

    # 'failed' status recorded
    calls = [c.args for c in mock_db.update_opt_run_status.call_args_list]
    statuses = [c[1] for c in calls]
    assert "failed" in statuses
```

Run: `pytest tests/unit/test_orchestrator_noop.py -v`
Expected: FAIL.

- [ ] **Step 7.2: Implement the noop Orchestrator**

```python
# retune-cloud/server/optimizer/orchestrator.py
"""OptimizerOrchestrator — Phase 1 noop.

Phase 2+ fills in subagent dispatch (PromptOptimizer, ToolOptimizer, RAGOptimizer)
based on trace analysis.
"""
from __future__ import annotations

import logging

from server.db import postgres as db
from server.optimizer.report_writer import ReportWriterAgent

logger = logging.getLogger(__name__)


class OptimizerOrchestrator:
    def __init__(self) -> None:
        self._writer = ReportWriterAgent()

    def run(self, run_id: str) -> None:
        """Drive a single optimization run to completion.

        Phase 1: no candidates are generated or evaluated; an empty report
        is written immediately and the run is marked completed.
        """
        try:
            row = db.get_opt_run(run_id)
            if row is None:
                raise ValueError(f"Run {run_id} not found")

            db.update_opt_run_status(run_id, "running")

            # Phase 1: no subagents dispatched. Phase 2 fills in here.
            report = self._writer.render(
                run_id=run_id,
                understanding="",
                summary={"baseline_score": 0.0, "best_score": 0.0, "improvement_pct": 0.0},
                tier1=[], tier2=[], tier3=[],
                pareto_data=[],
            )
            db.save_opt_report(
                run_id=run_id,
                understanding=report.understanding,
                summary=report.summary,
                tier1=report.tier1,
                tier2=report.tier2,
                tier3=report.tier3,
                pareto_data=report.pareto_data,
                markdown=report.markdown,
            )
            db.update_opt_run_status(run_id, "completed")

        except Exception as e:
            logger.exception("Orchestrator run failed: %s", e)
            db.update_opt_run_status(run_id, "failed", failure_reason=str(e))
            # Refund the slot — best effort
            try:
                if row is not None:
                    db.decrement_opt_runs_used(row["org_id"])
            except Exception:
                pass
```

- [ ] **Step 7.3: Run tests, verify pass**

Run: `pytest tests/unit/test_orchestrator_noop.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7.4: Commit**

```bash
git add retune-cloud/server/optimizer/orchestrator.py tests/unit/test_orchestrator_noop.py
git commit -m "optimizer: noop Orchestrator (Phase 1) — writes empty report"
```

---

## Task 8: Job queue + jobs routes (long-poll)

**Files:**
- Create: `retune-cloud/server/optimizer/job_queue.py`
- Create: `retune-cloud/server/routes/jobs.py`
- Test: `tests/unit/test_jobs_routes.py`

Phase 1 in-memory queue; later phases migrate to Redis. The queue is per-run: `RunCandidate` messages are pushed by the Orchestrator (not yet, in noop) and pulled by the SDK via long-poll. `JobCompleteMsg` is also pushed through the same queue so the SDK's single poll loop can receive it.

- [ ] **Step 8.1: Write failing tests**

```python
# tests/unit/test_jobs_routes.py
"""Long-poll jobs endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient
from unittest.mock import patch

from server.app import app
from server.optimizer.job_queue import get_queue


client = TestClient(app)


def _auth():
    # Minimal auth bypass for tests — real API keys in integration tests
    return {"Authorization": "Bearer test-key"}


@patch("server.routes.jobs._verify_run_auth", return_value={"org": "org_1"})
def test_pending_returns_immediately_when_message_present(_):
    q = get_queue()
    q.reset()
    q.push("run_x", {"type": "run_candidate", "candidate_id": "c1", "config_overrides": {}})
    r = client.get("/api/v1/jobs/pending?run_id=run_x&timeout=1", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "run_candidate"
    assert body["candidate_id"] == "c1"


@patch("server.routes.jobs._verify_run_auth", return_value={"org": "org_1"})
def test_pending_times_out_cleanly(_):
    q = get_queue()
    q.reset()
    r = client.get("/api/v1/jobs/pending?run_id=empty_run&timeout=1", headers=_auth())
    # Server returns 204 No Content on timeout — SDK just polls again
    assert r.status_code == 204


@patch("server.routes.jobs._verify_run_auth", return_value={"org": "org_1"})
def test_submit_result_stores_candidate(_):
    from server.optimizer.job_queue import get_results
    results = get_results()
    results.reset()

    r = client.post(
        "/api/v1/jobs/result",
        json={
            "run_id": "run_y",
            "candidate_id": "c2",
            "trace": {"query": "hello", "response": "hi"},
            "eval_scores": {"llm_judge": 7.0, "cost": 0.001, "latency": 1.0},
        },
        headers=_auth(),
    )
    assert r.status_code == 200
    assert results.get("run_y", "c2") is not None
```

Run: `pytest tests/unit/test_jobs_routes.py -v`
Expected: FAIL — module(s) do not exist.

- [ ] **Step 8.2: Implement the in-memory queue**

```python
# retune-cloud/server/optimizer/job_queue.py
"""In-memory per-run job queue + results store.

Phase 1 uses stdlib queue + threading.Event. Later phases swap in Redis.
"""
from __future__ import annotations

import queue
import threading
from typing import Any


class _JobQueue:
    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _q(self, run_id: str) -> queue.Queue:
        with self._lock:
            if run_id not in self._queues:
                self._queues[run_id] = queue.Queue()
            return self._queues[run_id]

    def push(self, run_id: str, message: dict[str, Any]) -> None:
        self._q(run_id).put(message)

    def pop(self, run_id: str, timeout: float) -> dict[str, Any] | None:
        try:
            return self._q(run_id).get(timeout=timeout)
        except queue.Empty:
            return None

    def reset(self) -> None:
        with self._lock:
            self._queues.clear()


class _ResultsStore:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def put(self, run_id: str, candidate_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._store[(run_id, candidate_id)] = payload

    def get(self, run_id: str, candidate_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._store.get((run_id, candidate_id))

    def reset(self) -> None:
        with self._lock:
            self._store.clear()


_queue_singleton = _JobQueue()
_results_singleton = _ResultsStore()


def get_queue() -> _JobQueue:
    return _queue_singleton


def get_results() -> _ResultsStore:
    return _results_singleton
```

- [ ] **Step 8.3: Implement jobs routes**

```python
# retune-cloud/server/routes/jobs.py
"""Long-poll jobs endpoints for SDK worker."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Response

from server.db import postgres as db
from server.middleware import require_auth
from server.optimizer.job_queue import get_queue, get_results


router = APIRouter(redirect_slashes=False)


def _verify_run_auth(authorization: str | None, run_id: str) -> dict[str, Any]:
    payload = require_auth(authorization)
    row = db.get_opt_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if str(row["org_id"]) != payload["org"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return payload


@router.get("/pending")
def pending(
    run_id: str,
    timeout: float = 15.0,
    authorization: str | None = Header(None),
) -> Response:
    """Long-poll for the next job on this run.

    Returns 200 with JSON if a message arrives within timeout seconds,
    204 No Content if the timeout expires so the SDK can re-poll.
    """
    _verify_run_auth(authorization, run_id)
    timeout = min(max(timeout, 1.0), 30.0)
    msg = get_queue().pop(run_id, timeout=timeout)
    if msg is None:
        return Response(status_code=204)
    # fastapi will serialize dict → JSON
    from fastapi.responses import JSONResponse
    return JSONResponse(msg)


@router.post("/result")
def submit_result(
    payload: dict,
    authorization: str | None = Header(None),
) -> dict:
    run_id = payload.get("run_id")
    candidate_id = payload.get("candidate_id")
    if not run_id or not candidate_id:
        raise HTTPException(status_code=400, detail="run_id and candidate_id required")
    _verify_run_auth(authorization, run_id)
    get_results().put(run_id, candidate_id, payload)
    return {"accepted": True}
```

- [ ] **Step 8.4: Pre-register router in `app.py` (just enough to run tests)**

Edit `retune-cloud/server/app.py`, add to the imports block:

```python
from server.routes import jobs
```

Add after the other `app.include_router` calls:

```python
app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
```

- [ ] **Step 8.5: Run tests, verify pass**

Run: `pytest tests/unit/test_jobs_routes.py -v`
Expected: PASS (3 tests).

- [ ] **Step 8.6: Commit**

```bash
git add retune-cloud/server/optimizer/job_queue.py \
        retune-cloud/server/routes/jobs.py \
        retune-cloud/server/app.py \
        tests/unit/test_jobs_routes.py
git commit -m "optimizer: in-memory job queue + long-poll jobs routes"
```

---

## Task 9: Optimize routes (preauth/commit/cancel/get)

**Files:**
- Create: `retune-cloud/server/routes/optimize.py`
- Test: `tests/unit/test_optimize_routes.py`

- [ ] **Step 9.1: Write failing tests**

```python
# tests/unit/test_optimize_routes.py
"""Cloud /v1/optimize/* endpoints (mocked DB + orchestrator)."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def _auth():
    return {"Authorization": "Bearer test-key"}


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
@patch("server.routes.optimize.BackgroundTasks.add_task")
def test_preauthorize_checks_quota_and_creates_run(mock_add_task, mock_db, _auth_fn):
    mock_db.count_opt_runs_used.return_value = (3, 15)

    r = client.post(
        "/api/v1/optimize/preauthorize",
        json={
            "source": "last_n_traces",
            "n_traces": 50,
            "axes": ["prompt"],
            "reward_spec": None,
        },
        headers=_auth(),
    )
    assert r.status_code == 200
    body = r.json()
    assert "run_id" in body
    assert body["runs_remaining"] == 12
    assert mock_db.create_opt_run.called
    assert mock_db.increment_opt_runs_used.called
    assert mock_add_task.called  # orchestrator scheduled


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
def test_preauthorize_rejects_when_over_limit(mock_db, _auth_fn):
    mock_db.count_opt_runs_used.return_value = (15, 15)
    r = client.post(
        "/api/v1/optimize/preauthorize",
        json={"source": "last_n_traces", "n_traces": 50, "axes": ["prompt"]},
        headers=_auth(),
    )
    assert r.status_code == 402  # Payment Required
    assert "limit" in r.json()["detail"].lower()
    assert not mock_db.create_opt_run.called


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
def test_get_run_returns_status(mock_db, _auth_fn):
    mock_db.get_opt_run.return_value = {
        "id": "run_1", "org_id": "org_1", "status": "completed",
        "source": "last_n_traces", "n_traces": 50,
        "axes": [], "reward_spec": {}, "rewriter_llm": None,
        "created_at": None, "started_at": None, "completed_at": None,
    }
    r = client.get("/api/v1/optimize/run_1", headers=_auth())
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
def test_cancel_refunds_slot(mock_db, _auth_fn):
    mock_db.get_opt_run.return_value = {
        "id": "run_1", "org_id": "org_1", "status": "pending",
    }
    r = client.post("/api/v1/optimize/run_1/cancel", headers=_auth())
    assert r.status_code == 200
    assert mock_db.update_opt_run_status.called
    assert mock_db.decrement_opt_runs_used.called


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
def test_get_report_fetches_saved(mock_db, _auth_fn):
    mock_db.get_opt_run.return_value = {"id": "r", "org_id": "org_1"}
    mock_db.get_opt_report.return_value = {
        "run_id": "r",
        "understanding": "",
        "summary": {},
        "tier1": [], "tier2": [], "tier3": [],
        "pareto_data": [],
        "markdown": "# Empty",
    }
    r = client.get("/api/v1/optimize/r/report", headers=_auth())
    assert r.status_code == 200
    assert r.json()["markdown"] == "# Empty"
```

Run: `pytest tests/unit/test_optimize_routes.py -v`
Expected: FAIL.

- [ ] **Step 9.2: Implement optimize routes**

```python
# retune-cloud/server/routes/optimize.py
"""POST /v1/optimize/* endpoints — preauth, commit, cancel, get run, get report."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel, Field

from server.db import postgres as db
from server.middleware import require_auth
from server.optimizer.job_queue import get_queue
from server.optimizer.orchestrator import OptimizerOrchestrator
from server.optimizer.reward_parser import default_reward_spec, parse_reward_spec

router = APIRouter(redirect_slashes=False)


class PreauthorizeRequest(BaseModel):
    source: str  # "last_n_traces" | "collect_next"
    n_traces: int = Field(ge=1, le=10_000)
    axes: list[str] = Field(default_factory=lambda: ["prompt", "tools", "rag"])
    reward_spec: dict | None = None
    rewriter_llm: str | None = None


def _verify_run_belongs_to_org(run_id: str, org_id: str) -> dict[str, Any]:
    row = db.get_opt_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if str(row["org_id"]) != org_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return row


@router.post("/preauthorize")
def preauthorize(
    req: PreauthorizeRequest,
    background: BackgroundTasks,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Reserve a run slot, create the run row, schedule the orchestrator."""
    payload = require_auth(authorization)
    org_id = payload["org"]

    used, limit = db.count_opt_runs_used(org_id)
    if used >= limit:
        raise HTTPException(
            status_code=402,
            detail=f"Optimization run limit reached ({used}/{limit}). Upgrade at https://agentretune.com/pricing",
        )

    # Validate reward spec up front so the orchestrator doesn't fail later
    try:
        spec = (
            parse_reward_spec(req.reward_spec) if req.reward_spec is not None
            else default_reward_spec()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid reward_spec: {e}")

    run_id = f"run_{uuid.uuid4().hex[:16]}"
    db.create_opt_run(
        run_id=run_id,
        org_id=org_id,
        source=req.source,
        n_traces=req.n_traces,
        axes=req.axes,
        reward_spec=spec.model_dump(),
        rewriter_llm=req.rewriter_llm,
    )
    db.increment_opt_runs_used(org_id)

    # Schedule the orchestrator in the background. Phase 1 noop completes in <1s.
    background.add_task(OptimizerOrchestrator().run, run_id)

    # Push a JobComplete message so the SDK worker's long-poll resolves
    # once the orchestrator finishes writing the report.
    # The orchestrator itself doesn't push JobComplete — we do it here
    # after scheduling, via a tiny poller helper task.
    background.add_task(_notify_complete_when_ready, run_id)

    return {"run_id": run_id, "runs_remaining": limit - used - 1}


def _notify_complete_when_ready(run_id: str) -> None:
    """Poll DB until run is completed/failed, then enqueue JobComplete."""
    import time

    for _ in range(120):  # up to ~2 minutes
        row = db.get_opt_run(run_id)
        if row and row["status"] in ("completed", "failed", "cancelled"):
            q = get_queue()
            if row["status"] == "completed":
                q.push(run_id, {
                    "type": "job_complete",
                    "run_id": run_id,
                    "report_url": f"/api/v1/optimize/{run_id}/report",
                })
            else:
                q.push(run_id, {
                    "type": "job_failed",
                    "run_id": run_id,
                    "reason": row.get("failure_reason", "unknown"),
                })
            return
        time.sleep(1.0)


@router.post("/{run_id}/commit")
def commit(
    run_id: str,
    authorization: str | None = Header(None),
) -> dict:
    """SDK acknowledges it consumed the report. No-op in Phase 1 (slot was charged at preauth)."""
    payload = require_auth(authorization)
    _verify_run_belongs_to_org(run_id, payload["org"])
    return {"committed": True}


@router.post("/{run_id}/cancel")
def cancel(
    run_id: str,
    authorization: str | None = Header(None),
) -> dict:
    """Cancel a pending/running run; refund the slot."""
    payload = require_auth(authorization)
    row = _verify_run_belongs_to_org(run_id, payload["org"])
    if row["status"] in ("completed", "failed", "cancelled"):
        return {"cancelled": False, "reason": f"already {row['status']}"}
    db.update_opt_run_status(run_id, "cancelled")
    db.decrement_opt_runs_used(payload["org"])
    get_queue().push(run_id, {"type": "job_failed", "run_id": run_id, "reason": "cancelled"})
    return {"cancelled": True}


@router.get("/{run_id}")
def get_run(
    run_id: str,
    authorization: str | None = Header(None),
) -> dict:
    payload = require_auth(authorization)
    row = _verify_run_belongs_to_org(run_id, payload["org"])
    return {
        "run_id": run_id,
        "status": row["status"],
        "source": row["source"],
        "n_traces": row["n_traces"],
        "axes": row["axes"],
        "created_at": str(row["created_at"]) if row.get("created_at") else None,
        "started_at": str(row["started_at"]) if row.get("started_at") else None,
        "completed_at": str(row["completed_at"]) if row.get("completed_at") else None,
    }


@router.get("/{run_id}/report")
def get_report(
    run_id: str,
    authorization: str | None = Header(None),
) -> dict:
    payload = require_auth(authorization)
    _verify_run_belongs_to_org(run_id, payload["org"])
    report = db.get_opt_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not ready yet")
    return {
        "run_id": run_id,
        "understanding": report["understanding"],
        "summary": report["summary"],
        "tier1": report["tier1"],
        "tier2": report["tier2"],
        "tier3": report["tier3"],
        "pareto_data": report["pareto_data"],
        "markdown": report["markdown"],
    }
```

- [ ] **Step 9.3: Run tests, verify pass**

Run: `pytest tests/unit/test_optimize_routes.py -v`
Expected: PASS (5 tests).

- [ ] **Step 9.4: Commit**

```bash
git add retune-cloud/server/routes/optimize.py tests/unit/test_optimize_routes.py
git commit -m "optimizer: /v1/optimize routes (preauth, commit, cancel, get, report)"
```

---

## Task 10: Wire optimize router into `app.py`; extend `billing.get_usage`

**Files:**
- Modify: `retune-cloud/server/app.py`
- Modify: `retune-cloud/server/routes/billing.py`

- [ ] **Step 10.1: Include optimize router in `app.py`**

Add to the imports block in `retune-cloud/server/app.py`:

```python
from server.routes import jobs, optimize
```

Add after the existing `include_router` calls:

```python
app.include_router(optimize.router, prefix="/api/v1/optimize", tags=["optimize"])
```

(jobs was added in Task 8.)

- [ ] **Step 10.2: Extend `get_usage` in `billing.py` to include optimize counters**

In `retune-cloud/server/routes/billing.py`, locate the `get_usage` function (around line 95–114) and replace its body:

```python
@router.get("/usage")
def get_usage(authorization: str | None = Header(None)):
    """Get current month's usage for the org."""
    from server.middleware import require_permission
    payload = require_permission(authorization, "billing:read")

    try:
        from server.db import postgres as db
        org = db.get_org(payload["org"])
        usage = db.get_usage(payload["org"])
        opt_used, opt_limit = db.count_opt_runs_used(payload["org"])
        return {
            "plan": org["plan"] if org else "free",
            "trace_limit": org["trace_limit"] if org else 1000,
            "runs_used": opt_used,
            "runs_limit": opt_limit,
            **usage,
        }
    except ImportError:
        return {
            "plan": "free",
            "trace_limit": 1000,
            "runs_used": 0,
            "runs_limit": 15,
        }
```

- [ ] **Step 10.3: Verify cloud app still imports cleanly**

Run: `python -c "from server.app import app; print(len(app.routes))"` (from `retune-cloud/` dir)
Expected: prints a positive number, no ImportError.

- [ ] **Step 10.4: Commit**

```bash
git add retune-cloud/server/app.py retune-cloud/server/routes/billing.py
git commit -m "optimizer: wire optimize router; expose runs_used/runs_limit in billing"
```

---

## Task 11: SDK `OptimizerClient`

**Files:**
- Create: `src/retune/optimizer/client.py`
- Test: `tests/unit/test_optimizer_client.py`

Uses existing `urllib` pattern (same as `cloud/client.py`) — no new deps.

- [ ] **Step 11.1: Write failing tests**

```python
# tests/unit/test_optimizer_client.py
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
    resp.read.return_value = b'{"run_id": "r", "tier1": [], "tier2": [], "tier3": [], "markdown": "# empty", "understanding": "", "summary": {}, "pareto_data": []}'
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    rep = client.fetch_report("r")
    assert rep["run_id"] == "r"
```

Run: `pytest tests/unit/test_optimizer_client.py -v`
Expected: FAIL.

- [ ] **Step 11.2: Implement OptimizerClient**

```python
# src/retune/optimizer/client.py
"""OptimizerClient — HTTP wrappers for the cloud /v1/optimize/* and /v1/jobs/* APIs."""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OptimizerClient:
    def __init__(self, api_key: str, base_url: str = "https://api.agentretune.com") -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _headers(self, content: bool = False) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "retune-sdk/0.3.0",
        }
        if content:
            h["Content-Type"] = "application/json"
        return h

    def _post(self, path: str, body: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        req = Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(content=True),
            method="POST",
        )
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except HTTPError as e:
            if e.code == 402:
                raise RuntimeError(
                    f"Optimization run limit reached (402). Upgrade at https://agentretune.com/pricing"
                ) from e
            raise RuntimeError(f"Cloud {path} failed: {e.code} {e.reason}") from e
        except URLError as e:
            raise RuntimeError(f"Cloud {path} unreachable: {e.reason}") from e

    def _get(self, path: str, timeout: float = 10.0) -> dict[str, Any] | None:
        url = f"{self._base_url}{path}"
        req = Request(url, headers=self._headers(), method="GET")
        try:
            with urlopen(req, timeout=timeout) as resp:
                if resp.status == 204:
                    return None
                return json.loads(resp.read() or b"{}")
        except HTTPError as e:
            raise RuntimeError(f"Cloud {path} failed: {e.code} {e.reason}") from e

    def preauthorize(
        self,
        source: str,
        n_traces: int,
        axes: list[str],
        reward_spec: dict[str, Any] | None = None,
        rewriter_llm: str | None = None,
    ) -> dict[str, Any]:
        return self._post("/api/v1/optimize/preauthorize", {
            "source": source,
            "n_traces": n_traces,
            "axes": axes,
            "reward_spec": reward_spec,
            "rewriter_llm": rewriter_llm,
        })

    def commit(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/optimize/{run_id}/commit", {})

    def cancel(self, run_id: str) -> dict[str, Any]:
        return self._post(f"/api/v1/optimize/{run_id}/cancel", {})

    def get_run(self, run_id: str) -> dict[str, Any]:
        out = self._get(f"/api/v1/optimize/{run_id}")
        if out is None:
            raise RuntimeError(f"Run {run_id} not found")
        return out

    def fetch_report(self, run_id: str) -> dict[str, Any]:
        out = self._get(f"/api/v1/optimize/{run_id}/report")
        if out is None:
            raise RuntimeError(f"Report for {run_id} not ready")
        return out

    def poll_pending(self, run_id: str, timeout: float = 15.0) -> dict[str, Any] | None:
        """Long-poll. Returns message dict, or None on timeout."""
        # Server-side timeout + a small client buffer
        client_timeout = timeout + 5.0
        return self._get(
            f"/api/v1/jobs/pending?run_id={run_id}&timeout={timeout}",
            timeout=client_timeout,
        )

    def submit_result(
        self,
        run_id: str,
        candidate_id: str,
        trace: dict[str, Any],
        eval_scores: dict[str, float],
    ) -> dict[str, Any]:
        return self._post("/api/v1/jobs/result", {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "trace": trace,
            "eval_scores": eval_scores,
        })
```

- [ ] **Step 11.3: Run tests, verify pass**

Run: `pytest tests/unit/test_optimizer_client.py -v`
Expected: PASS (5 tests).

- [ ] **Step 11.4: Commit**

```bash
git add src/retune/optimizer/client.py tests/unit/test_optimizer_client.py
git commit -m "optimizer: SDK OptimizerClient (HTTP wrappers)"
```

---

## Task 12: SDK `SDKWorker` (long-poll consumer)

**Files:**
- Create: `src/retune/optimizer/worker.py`
- Test: `tests/unit/test_sdk_worker.py`

Consumes `RunCandidate` messages, calls a `candidate_runner` callback (which in Phase 1 is a trivial stub — the `Retuner.optimize()` method wires the real one in Task 13), submits the `CandidateResult`, and exits when `JobComplete` or `JobFailed` arrives.

- [ ] **Step 12.1: Write failing tests**

```python
# tests/unit/test_sdk_worker.py
"""SDKWorker — long-poll consumer."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.worker import SDKWorker


def test_worker_exits_on_job_complete():
    client = MagicMock()
    # First poll returns a run_candidate, second returns job_complete.
    client.poll_pending.side_effect = [
        {"type": "run_candidate", "candidate_id": "c1",
         "config_overrides": {}, "query_set": []},
        {"type": "job_complete", "run_id": "r1",
         "report_url": "/x"},
    ]
    runner_calls = []
    def runner(overrides, queries):
        runner_calls.append(overrides)
        return ({"trace": "fake"}, {"llm_judge": 7.0})

    worker = SDKWorker(client=client, run_id="r1", candidate_runner=runner)
    report_url = worker.run()

    assert report_url == "/x"
    assert len(runner_calls) == 1
    client.submit_result.assert_called_once()


def test_worker_raises_on_job_failed():
    client = MagicMock()
    client.poll_pending.side_effect = [
        {"type": "job_failed", "run_id": "r1", "reason": "boom"},
    ]
    worker = SDKWorker(client=client, run_id="r1",
                       candidate_runner=lambda o, q: ({}, {}))
    import pytest
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()


def test_worker_retries_on_timeout():
    client = MagicMock()
    # 2 timeouts (None) then completion
    client.poll_pending.side_effect = [
        None, None, {"type": "job_complete", "run_id": "r1", "report_url": "/y"},
    ]
    worker = SDKWorker(client=client, run_id="r1",
                       candidate_runner=lambda o, q: ({}, {}))
    assert worker.run() == "/y"
    assert client.poll_pending.call_count == 3
```

Run: `pytest tests/unit/test_sdk_worker.py -v`
Expected: FAIL.

- [ ] **Step 12.2: Implement SDKWorker**

```python
# src/retune/optimizer/worker.py
"""SDKWorker — long-poll consumer that executes candidates on the SDK side."""
from __future__ import annotations

import logging
from typing import Any, Callable

from retune.optimizer.client import OptimizerClient

logger = logging.getLogger(__name__)

CandidateRunner = Callable[[dict[str, Any], list[dict[str, Any]]], tuple[dict[str, Any], dict[str, float]]]


class SDKWorker:
    """Long-polls the cloud for RunCandidate commands; runs them locally.

    The `candidate_runner` callback is what actually executes the user's
    agent with overridden config against a query set — it's supplied by
    Retuner.optimize() and encapsulates all adapter-specific logic.
    """

    def __init__(
        self,
        client: OptimizerClient,
        run_id: str,
        candidate_runner: CandidateRunner,
        poll_timeout: float = 15.0,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._runner = candidate_runner
        self._poll_timeout = poll_timeout

    def run(self) -> str:
        """Drive the worker loop to completion. Returns report_url."""
        while True:
            msg = self._client.poll_pending(self._run_id, timeout=self._poll_timeout)
            if msg is None:
                continue  # timeout — re-poll

            mtype = msg.get("type")
            if mtype == "run_candidate":
                self._handle_candidate(msg)
            elif mtype == "job_complete":
                return msg["report_url"]
            elif mtype == "job_failed":
                raise RuntimeError(f"Optimization run failed: {msg.get('reason', 'unknown')}")
            else:
                logger.warning("SDKWorker: unknown message type %r", mtype)

    def _handle_candidate(self, msg: dict[str, Any]) -> None:
        cid = msg["candidate_id"]
        overrides = msg.get("config_overrides", {})
        queries = msg.get("query_set", [])
        try:
            trace, eval_scores = self._runner(overrides, queries)
        except Exception as e:
            logger.exception("Candidate %s failed: %s", cid, e)
            # Submit a failed result — cloud treats as 0-score candidate
            trace, eval_scores = ({"error": str(e)}, {})
        self._client.submit_result(
            run_id=self._run_id,
            candidate_id=cid,
            trace=trace,
            eval_scores=eval_scores,
        )
```

- [ ] **Step 12.3: Run tests, verify pass**

Run: `pytest tests/unit/test_sdk_worker.py -v`
Expected: PASS (3 tests).

- [ ] **Step 12.4: Commit**

```bash
git add src/retune/optimizer/worker.py tests/unit/test_sdk_worker.py
git commit -m "optimizer: SDK long-poll worker (candidate runner + job completion)"
```

---

## Task 13: SDK `OptimizationReport` + `Retuner.optimize()` method

**Files:**
- Create: `src/retune/optimizer/report.py`
- Modify: `src/retune/wrapper.py`
- Modify: `src/retune/__init__.py`
- Test: `tests/unit/test_optimization_report.py`
- Test: `tests/unit/test_wrapper_optimize.py`

- [ ] **Step 13.1: Write failing tests for OptimizationReport**

```python
# tests/unit/test_optimization_report.py
"""OptimizationReport — model + apply/show/copy_snippets."""
from __future__ import annotations

from retune.optimizer.report import OptimizationReport


def test_from_cloud_dict():
    raw = {
        "run_id": "r",
        "understanding": "hello",
        "summary": {"baseline_score": 1.0, "best_score": 2.0, "improvement_pct": 100.0},
        "tier1": [{"axis": "prompt", "title": "rewrite", "description": "x",
                    "tier": 1, "confidence": "H", "estimated_impact": {}}],
        "tier2": [], "tier3": [],
        "pareto_data": [],
        "markdown": "# r",
    }
    rep = OptimizationReport.from_cloud_dict(raw)
    assert rep.run_id == "r"
    assert len(rep.tier1) == 1
    assert rep.tier1[0].axis == "prompt"


def test_apply_tier1_invokes_callback(tmp_path):
    rep = OptimizationReport(
        run_id="r", understanding="", summary={},
        tier1=[], tier2=[], tier3=[], pareto_data=[],
    )
    applied = []
    rep.apply(tier=1, apply_fn=lambda s: applied.append(s))
    assert applied == []  # empty tier1, no-op


def test_show_returns_markdown(capsys):
    rep = OptimizationReport(
        run_id="r", understanding="", summary={},
        tier1=[], tier2=[], tier3=[], pareto_data=[],
        markdown="# Report\nHello"
    )
    rep.show()
    captured = capsys.readouterr()
    assert "# Report" in captured.out
```

Run: `pytest tests/unit/test_optimization_report.py -v`
Expected: FAIL.

- [ ] **Step 13.2: Implement OptimizationReport**

```python
# src/retune/optimizer/report.py
"""Client-side OptimizationReport with apply/show/copy_snippets."""
from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel, Field

from retune.optimizer.models import OptimizationReport as _BaseReport, Suggestion


class OptimizationReport(_BaseReport):
    """Extends the shared model with client-side convenience methods."""
    markdown: str = ""

    @classmethod
    def from_cloud_dict(cls, raw: dict[str, Any]) -> "OptimizationReport":
        def _hydrate(items: list) -> list[Suggestion]:
            out = []
            for item in items:
                if "tier" not in item:
                    # The cloud Report model doesn't carry tier on each item —
                    # we assign tier based on which list it came from.
                    continue
                out.append(Suggestion.model_validate(item))
            return out

        # The cloud response nests items per tier — tag each with its tier.
        def _tag(items: list, tier: int) -> list[Suggestion]:
            tagged = []
            for it in items:
                if "tier" not in it:
                    it = {**it, "tier": tier}
                tagged.append(Suggestion.model_validate(it))
            return tagged

        return cls(
            run_id=raw["run_id"],
            understanding=raw.get("understanding", ""),
            summary=raw.get("summary", {}),
            tier1=_tag(raw.get("tier1", []), 1),
            tier2=_tag(raw.get("tier2", []), 2),
            tier3=_tag(raw.get("tier3", []), 3),
            pareto_data=raw.get("pareto_data", []),
            markdown=raw.get("markdown", ""),
        )

    def show(self) -> None:
        """Print the markdown report to stdout."""
        print(self.markdown)

    def apply(
        self,
        tier: int = 1,
        apply_fn: Callable[[Suggestion], None] | None = None,
    ) -> list[Suggestion]:
        """Apply tier-N suggestions.

        Phase 1 contract: `apply_fn` (supplied by Retuner) mutates wrapper
        config for tier-1 suggestions. Phase 2+ wires real payloads.
        """
        if tier == 1:
            items = self.tier1
        elif tier == 2:
            items = self.tier2
        elif tier == 3:
            items = self.tier3
        else:
            raise ValueError("tier must be 1, 2, or 3")

        if apply_fn is None:
            return items
        for s in items:
            apply_fn(s)
        return items

    def copy_snippets(self, to: str = "stdout") -> str:
        """Return concatenated Tier-2 code snippets (or print them)."""
        blocks = []
        for s in self.tier2:
            if s.code_snippet:
                blocks.append(f"# {s.title}\n{s.code_snippet}")
        text = "\n\n".join(blocks)
        if to == "stdout":
            print(text)
        return text
```

- [ ] **Step 13.3: Write failing tests for `Retuner.optimize()`**

```python
# tests/unit/test_wrapper_optimize.py
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
```

Run: `pytest tests/unit/test_wrapper_optimize.py -v`
Expected: FAIL.

- [ ] **Step 13.4: Modify `wrapper.py` — require agent_purpose + add optimize()**

Open `src/retune/wrapper.py`. In the `Retuner.__init__` signature, add `agent_purpose` and `success_criteria` parameters. Near line 215 (where `UsageGate` is constructed), add:

```python
self._agent_purpose = kwargs.get("agent_purpose") or agent_purpose
self._success_criteria = kwargs.get("success_criteria") or success_criteria
if mode == Mode.IMPROVE and not self._agent_purpose:
    raise ValueError(
        "agent_purpose='...' is required when mode=Mode.IMPROVE. "
        "Supply a one-line description of what your agent does."
    )
```

(Concrete edit: find the existing `__init__` parameter list and append `agent_purpose: str | None = None, success_criteria: str | None = None` before `**kwargs`. Then the body snippet above stores them.)

At the end of the `Retuner` class, add the new `optimize()` method:

```python
def optimize(
    self,
    source: str = "last_n_traces",
    n: int = 50,
    axes: list[str] | str = "auto",
    reward: str | dict = "judge_with_guardrails",
    rewriter_llm: str | None = None,
    guardrails: dict | None = None,
) -> "OptimizationReport":
    """Trigger a cloud optimization run.

    Args:
        source: "last_n_traces" (historical replay) or "collect_next".
        n: number of traces to optimize over.
        axes: list of axes to optimize, or "auto" for orchestrator-selected.
        reward: "judge_with_guardrails" (default) or a dict with the
            declarative reward spec (see docs).
        rewriter_llm: model used by PromptOptimizerAgent for rewrites.
        guardrails: overrides for the default cost/latency guardrails.
    """
    if not self._api_key:
        raise RuntimeError("api_key is required to call optimize()")

    from retune.optimizer.client import OptimizerClient
    from retune.optimizer.worker import SDKWorker
    from retune.optimizer.report import OptimizationReport

    axes_list = ["prompt", "tools", "rag"] if axes == "auto" else list(axes)
    reward_spec = reward if isinstance(reward, dict) else None
    if guardrails and reward_spec is None:
        # Layer user guardrails on top of the default judge_with_guardrails
        reward_spec = {
            "primary": {"evaluator": "llm_judge", "weight": 1.0},
            "penalties": [
                {"evaluator": k, "threshold": v, "hard": True}
                for k, v in guardrails.items()
            ],
        }

    from retune.config import settings
    client = OptimizerClient(api_key=self._api_key, base_url=settings.cloud_base_url)
    resp = client.preauthorize(
        source=source, n_traces=n, axes=axes_list,
        reward_spec=reward_spec, rewriter_llm=rewriter_llm,
    )
    run_id = resp["run_id"]

    worker = SDKWorker(
        client=client,
        run_id=run_id,
        candidate_runner=self._make_candidate_runner(),
    )
    try:
        worker.run()
    except Exception:
        client.cancel(run_id)
        raise

    raw = client.fetch_report(run_id)
    client.commit(run_id)
    return OptimizationReport.from_cloud_dict(raw)


def _make_candidate_runner(self):
    """Return a callable that runs the wrapped agent with config overrides.

    Phase 1: since the noop orchestrator sends no RunCandidate messages,
    this function is effectively unreachable in the happy path. We keep
    a minimal implementation so unit tests can exercise the SDKWorker
    loop end-to-end. Phase 2 wires real config override injection.
    """
    def _runner(overrides: dict, queries: list) -> tuple[dict, dict]:
        # Phase 1: ignore overrides, run agent on each query, return first trace.
        if not queries:
            return ({"query": "", "response": ""}, {"llm_judge": 0.0})
        q = queries[0].get("query", "")
        resp = self._adapter.run(q) if self._adapter else ""
        return (
            {"query": q, "response": str(resp)},
            {"llm_judge": 0.0, "cost": 0.0, "latency": 0.0},
        )
    return _runner
```

Also at the top of `wrapper.py`, add an `OptimizerClient` + `SDKWorker` import guarded by `TYPE_CHECKING` to keep the import path resolvable for the tests that patch `retune.wrapper.OptimizerClient`:

```python
# Near other imports at the top of wrapper.py:
from retune.optimizer.client import OptimizerClient  # noqa: F401 (test patch target)
from retune.optimizer.worker import SDKWorker        # noqa: F401 (test patch target)
```

- [ ] **Step 13.5: Re-export from `src/retune/__init__.py`**

Append to `src/retune/__init__.py`:

```python
from retune.optimizer.report import OptimizationReport  # noqa: E402

__all__ = [*__all__, "OptimizationReport"] if "__all__" in dir() else ["OptimizationReport"]
```

- [ ] **Step 13.6: Run tests, verify pass**

Run: `pytest tests/unit/test_optimization_report.py tests/unit/test_wrapper_optimize.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 13.7: Commit**

```bash
git add src/retune/optimizer/report.py src/retune/wrapper.py src/retune/__init__.py \
        tests/unit/test_optimization_report.py tests/unit/test_wrapper_optimize.py
git commit -m "optimizer: Retuner.optimize() method + client-side OptimizationReport"
```

---

## Task 14: Extend `UsageGate` for per-run gating

**Files:**
- Modify: `src/retune/usage_gate.py`
- Test: `tests/unit/test_usage_gate_per_run.py`

The current `UsageGate` tracks "deep operations" in-memory. Phase 1 migrates the gate boundary: per-run checks happen via cloud (`preauthorize_run` returns 402 on exhaustion). The local gate becomes a cached *reflection* of cloud state (`runs_remaining` after preauth) so `Retuner.status()` can report without an extra HTTP call.

- [ ] **Step 14.1: Write failing tests**

```python
# tests/unit/test_usage_gate_per_run.py
"""UsageGate — per-run gating via cloud."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from retune.usage_gate import UsageGate


def test_reflect_cloud_run_count():
    gate = UsageGate(api_key="rt-x")
    gate.note_preauthorize_response({"runs_remaining": 12})
    status = gate.get_status()
    assert status["runs_remaining"] == 12


def test_local_count_noop_when_cloud_authoritative():
    """Without calling note_preauthorize_response, local-only behavior still works."""
    gate = UsageGate(api_key=None)
    # Legacy per-call behavior still intact for OBSERVE/EVALUATE
    assert gate.check("optimize") is True
```

Run: `pytest tests/unit/test_usage_gate_per_run.py -v`
Expected: FAIL — method doesn't exist.

- [ ] **Step 14.2: Add `note_preauthorize_response` to `UsageGate`**

Edit `src/retune/usage_gate.py`. Add these methods to the `UsageGate` class:

```python
def note_preauthorize_response(self, response: dict) -> None:
    """Record runs_remaining returned by /v1/optimize/preauthorize.

    Cloud is authoritative; this is a local cache for status display.
    """
    if "runs_remaining" in response:
        remaining = int(response["runs_remaining"])
        self._cloud_count = max(0, self._limit - remaining)
        self._last_check = __import__("time").time()
```

- [ ] **Step 14.3: Run tests, verify pass**

Run: `pytest tests/unit/test_usage_gate_per_run.py -v`
Expected: PASS (2 tests).

- [ ] **Step 14.4: Commit**

```bash
git add src/retune/usage_gate.py tests/unit/test_usage_gate_per_run.py
git commit -m "optimizer: UsageGate reflects cloud runs_remaining after preauth"
```

---

## Task 15: End-to-end integration test

**Files:**
- Create: `tests/integration/__init__.py` (empty if missing)
- Create: `tests/integration/test_optimize_e2e.py`

Full stack test with a real FastAPI TestClient + mocked DB. The SDK client is rewired to use the TestClient transport so no real HTTP/ports are involved but the actual route handlers, orchestrator, job queue, and SDK worker loop all run.

- [ ] **Step 15.1: Write the integration test**

```python
# tests/integration/test_optimize_e2e.py
"""End-to-end: SDK triggers optimize → cloud creates run → noop orchestrator
completes → SDK receives empty report → slot committed.

Uses FastAPI TestClient as transport; mocks the DB layer in-process.
"""
from __future__ import annotations

import json
import threading
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def fake_db():
    """In-memory fake DB for optimizer tables."""
    state = {
        "runs": {},      # run_id -> dict
        "reports": {},   # run_id -> dict
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

    ns = MagicMock()
    ns.create_opt_run = create_opt_run
    ns.get_opt_run = get_opt_run
    ns.update_opt_run_status = update_opt_run_status
    ns.save_opt_report = save_opt_report
    ns.get_opt_report = get_opt_report
    ns.count_opt_runs_used = count_opt_runs_used
    ns.increment_opt_runs_used = increment_opt_runs_used
    ns.decrement_opt_runs_used = decrement_opt_runs_used
    return ns


def test_noop_optimize_e2e(fake_db):
    """Drive the full loop: preauthorize → background orchestrator completes
    → long-poll delivers JobComplete → report fetch → commit."""
    with patch("server.routes.optimize.db", fake_db), \
         patch("server.routes.jobs.db", fake_db), \
         patch("server.optimizer.orchestrator.db", fake_db), \
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

        # BackgroundTasks runs synchronously in TestClient — so by here the
        # noop orchestrator has already completed and pushed JobComplete.
        assert fake_db.runs[run_id]["status"] == "completed" if hasattr(fake_db, "runs") else True

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
```

Note: the `state` dict is what actually backs the mock; attach it as `runs`/`reports`/`org_used`/`org_limit` attributes so tests can assert on DB state. Update the fixture:

Replace the `ns = MagicMock()` block with:

```python
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
return ns
```

- [ ] **Step 15.2: Run the integration test**

Run: `pytest tests/integration/test_optimize_e2e.py -v`
Expected: PASS (2 tests).

- [ ] **Step 15.3: Run the full test suite to check nothing regressed**

Run: `pytest tests/ -v --tb=short`
Expected: all tests pass (pre-existing + new Phase 1 tests).

Run: `ruff check src/ tests/ retune-cloud/server/`
Expected: no errors.

Run: `mypy src/retune/ --ignore-missing-imports`
Expected: no new errors introduced.

- [ ] **Step 15.4: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_optimize_e2e.py
git commit -m "optimizer: Phase 1 end-to-end integration test (noop flow)"
```

---

## Phase 1 Exit Gate

All of the following must be green before starting Phase 2:

- [ ] `pytest tests/ -v --tb=short` — all tests pass (pre-existing + Phase 1)
- [ ] `ruff check src/ tests/ retune-cloud/server/` — no lint errors
- [ ] `mypy src/retune/ --ignore-missing-imports` — no new type errors
- [ ] **Manual smoke test** — spin up `uvicorn server.app:app --port 8000` against a local Postgres with the schema applied, then in a separate shell:

```python
from retune import Retuner, Mode
def my_agent(q): return f"echo: {q}"
r = Retuner(
    agent=my_agent, adapter="custom", mode=Mode.IMPROVE,
    api_key="<test-org-key>",
    agent_purpose="echo bot",
)
report = r.optimize(source="last_n_traces", n=5, axes=["prompt"])
report.show()   # prints empty-but-well-formed markdown
```

- [ ] Verify in Postgres: `SELECT status FROM optimization_runs ORDER BY created_at DESC LIMIT 1;` → `completed`
- [ ] Verify: `SELECT optimize_runs_used FROM organizations WHERE id = '<test-org-id>';` → incremented by 1
- [ ] Run preauthorize 16 times → 16th returns 402

Only after all of the above is green, start **Phase 2 (PromptOptimizer)**.
