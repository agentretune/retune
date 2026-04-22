# Optimizer Phase 2 (PromptOptimizer + Real Orchestrator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 1's noop Orchestrator with a real implementation that dispatches a deepagents-based `PromptOptimizerAgent` (Beam Search APO + LLM-proposed rewrites) end-to-end: trace upload → orchestrator plans → prompt subagent generates candidates → job queue dispatches to SDK → SDK executes with config overrides → JudgeAgent scores → ReportWriterAgent emits real Tier 1 suggestions.

**Architecture:** Cloud-side `PromptOptimizerAgent` is refactored from the existing `retune-cloud/premium/agents/optimizer/` package into `retune-cloud/server/optimizer/prompt_optimizer/`, preserving the deepagents framework + BeamSearchAPO core but adapting the I/O to the new Phase 1 contracts (Orchestrator, JobQueue, JudgeAgent, ReportWriter). SDK's Retuner uploads the last-N local traces at preauthorize time; SDK worker applies candidate config_overrides by temporarily mutating `_config.system_prompt` / `_config.few_shot_examples` during each candidate's execution.

**Tech Stack:** Python 3.10+, `deepagents` (becomes a required dep in `retune-cloud/pyproject.toml`), `langgraph` fallback path (already present), Pydantic v2, pytest. No new deps on the SDK side.

**Reference spec:** `docs/superpowers/specs/2026-04-12-optimizer-design.md` §5 (PromptOptimizerAgent), §11 Phase 2 row. Reference plan: `docs/superpowers/plans/2026-04-12-optimizer-phase1-infra.md` (Phase 1 that this builds on).

**Phase 1 baseline:** branch `phase1/optimizer-infra` on both repos, 117 tests passing. Phase 2 branches from Phase 1.

---

## Scope

### In

- Trace upload pipeline (SDK → cloud at preauthorize time for `source="last_n_traces"`)
- `retune-cloud/server/optimizer/prompt_optimizer/` package: refactored from `retune-cloud/premium/agents/optimizer/*`, keeps deepagents + BeamSearchAPO, adapted to new I/O contracts
- Real `OptimizerOrchestrator.run()` implementation (replaces Phase 1 noop): reads traces, invokes PromptOptimizerAgent, dispatches candidates via JobQueue, waits for results, scores via JudgeAgent, writes real report
- SDK-side `Retuner._make_candidate_runner()` — real config-override application (system prompt + few-shot examples)
- `rewriter_llm` parameter wired end-to-end (user picks the LLM for prompt rewrites)

### Out (deferred to later phases)

- ToolOptimizerAgent (Phase 3)
- RAGOptimizerAgent (Phase 4)
- Feedback-loop memory (Phase 5)
- Dashboard UI (Phase 5)
- Multi-axis routing logic in Orchestrator (Phase 3+ — for Phase 2, the Orchestrator only ever dispatches PromptOptimizer)

---

## File Structure

### New files — cloud side (`retune-cloud/server/optimizer/prompt_optimizer/`)

| File | Responsibility |
|---|---|
| `server/optimizer/prompt_optimizer/__init__.py` | Public export: `PromptOptimizerAgent` |
| `server/optimizer/prompt_optimizer/agent.py` | `PromptOptimizerAgent` class — deepagents-based entry point, new I/O contract |
| `server/optimizer/prompt_optimizer/beam_search.py` | `BeamSearchAPO` — copied + trimmed from `premium/agents/optimizer/beam_search.py` |
| `server/optimizer/prompt_optimizer/prompts.py` | LLM prompts — copied from `premium/agents/optimizer/prompts.py` |
| `server/optimizer/prompt_optimizer/state.py` | State dataclasses — trimmed from `premium/agents/optimizer/state.py` |
| `server/optimizer/prompt_optimizer/llm.py` | `create_rewriter_llm(model_name)` — accepts user's choice, returns LangChain LLM |

### Modified files — cloud side

| File | Change |
|---|---|
| `server/optimizer/orchestrator.py` | Real `run()` implementation: replaces noop body with real dispatch loop |
| `server/optimizer/models.py` | Add `CandidateExecutionRequest`/`CandidateExecutionResult` envelopes + `PromptCandidate` data class |
| `server/routes/optimize.py` | `PreauthorizeRequest` accepts optional `traces: list[dict]` payload; stores them for the run |
| `server/db/postgres.py` | Add `save_opt_run_traces(run_id, traces)` + `get_opt_run_traces(run_id)` — JSONB-stored per-run trace bundle |
| `server/db/schema.sql` | Add `optimization_run_traces` table (JSONB blob, 1:1 with run, auto-purged on run completion) |
| `retune-cloud/pyproject.toml` | Promote `deepagents` from optional to required dependency |

### New files — SDK side

| File | Responsibility |
|---|---|
| `src/retune/optimizer/trace_collector.py` | `collect_last_n_local_traces(storage, n)` → `list[dict]` for upload |

### Modified files — SDK side

| File | Change |
|---|---|
| `src/retune/wrapper.py` | `Retuner.optimize()`: when `source="last_n_traces"`, collect traces from `self._storage` and pass in preauthorize payload. `_make_candidate_runner` applies `config_overrides` via temp mutation of `_config` (restores after). |
| `src/retune/optimizer/client.py` | `OptimizerClient.preauthorize(..., traces=...)` accepts optional traces list |

### Tests

| File | Covers |
|---|---|
| `retune-cloud/tests/test_prompt_optimizer_beam.py` | BeamSearchAPO unit: single round of critique→rewrite→score, mocked LLM |
| `retune-cloud/tests/test_prompt_optimizer_agent.py` | PromptOptimizerAgent: accepts traces + reward, emits `list[PromptCandidate]` with mocked beam search |
| `retune-cloud/tests/test_orchestrator_real.py` | Real Orchestrator dispatches PromptOptim, waits for candidates via mocked queue, scores, writes report with Tier 1 suggestion |
| `retune-cloud/tests/test_optimize_routes_traces.py` | Preauthorize accepts + stores traces payload |
| `retune-cloud/tests/test_opt_run_traces_db.py` | New DB helpers for per-run trace storage |
| `tests/unit/test_trace_collector.py` | `collect_last_n_local_traces` from SQLiteStorage |
| `tests/unit/test_wrapper_optimize_overrides.py` | `_make_candidate_runner` applies system_prompt override during execution and restores after |
| `tests/integration/test_optimize_prompt_e2e.py` | Full E2E: SDK uploads traces → Orchestrator dispatches prompt candidates → SDK executes overrides → Judge scores → Report has real Tier 1 suggestion |

---

## Task Summary

Thirteen tasks. Bottom-up: new models + DB first, then the subagent, then Orchestrator glue, then SDK trace upload + override, then E2E.

1. DB: `optimization_run_traces` table + helpers
2. Models: new envelopes for prompt candidates + execution
3. Route: preauthorize accepts traces payload
4. SDK: `trace_collector` helper
5. SDK: `OptimizerClient.preauthorize` carries traces
6. SDK: `_make_candidate_runner` applies config overrides
7. Cloud: `prompt_optimizer/llm.py` — rewriter LLM factory
8. Cloud: `prompt_optimizer/prompts.py` + `state.py` — copy+trim from `premium/`
9. Cloud: `prompt_optimizer/beam_search.py` — copy+adapt, new tests
10. Cloud: `prompt_optimizer/agent.py` — deepagents-based PromptOptimizerAgent
11. Cloud: Real `OptimizerOrchestrator.run()` (replaces noop)
12. Cloud: Promote deepagents to required dep
13. E2E integration test + phase exit verification

---

## Task 1: DB — optimization_run_traces table + helpers

**Files:**
- Modify: `retune-cloud/server/db/schema.sql` (append)
- Modify: `retune-cloud/server/db/postgres.py` (append 2 helpers)
- Test: `retune-cloud/tests/test_opt_run_traces_db.py`

- [ ] **Step 1.1: Append schema**

Append to the end of `retune-cloud/server/db/schema.sql`:

```sql
-- ============ Phase 2: Per-run trace bundle (auto-purged) ============

CREATE TABLE IF NOT EXISTS optimization_run_traces (
    run_id VARCHAR(64) PRIMARY KEY REFERENCES optimization_runs(id) ON DELETE CASCADE,
    traces JSONB NOT NULL DEFAULT '[]',
    trace_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 1.2: Write failing tests**

```python
# retune-cloud/tests/test_opt_run_traces_db.py
"""DB helpers for per-run trace storage."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from server.db import postgres as db


@patch.object(db, "put_conn")
@patch.object(db, "get_conn")
def test_save_opt_run_traces(mock_get_conn, _put):
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    mock_get_conn.return_value = conn

    traces = [{"query": "q1", "response": "r1"}, {"query": "q2", "response": "r2"}]
    db.save_opt_run_traces("run_1", traces)
    assert cur.execute.called
    sql = cur.execute.call_args[0][0]
    assert "INSERT INTO optimization_run_traces" in sql


@patch.object(db, "put_conn")
@patch.object(db, "get_conn")
def test_get_opt_run_traces(mock_get_conn, _put):
    cur = MagicMock()
    cur.fetchone.return_value = ([{"query": "q1"}], 1)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    mock_get_conn.return_value = conn

    traces, count = db.get_opt_run_traces("run_1")
    assert traces == [{"query": "q1"}]
    assert count == 1


@patch.object(db, "put_conn")
@patch.object(db, "get_conn")
def test_get_opt_run_traces_missing(mock_get_conn, _put):
    cur = MagicMock()
    cur.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    mock_get_conn.return_value = conn

    traces, count = db.get_opt_run_traces("run_missing")
    assert traces == []
    assert count == 0
```

Run: `pytest retune-cloud/tests/test_opt_run_traces_db.py -v`
Expected: FAIL.

- [ ] **Step 1.3: Implement helpers**

Append to `retune-cloud/server/db/postgres.py`:

```python
def save_opt_run_traces(run_id: str, traces: list[dict]) -> None:
    """Store the trace bundle for an optimization run. Auto-purged with the run."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO optimization_run_traces (run_id, traces, trace_count)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (run_id) DO UPDATE
                  SET traces = EXCLUDED.traces,
                      trace_count = EXCLUDED.trace_count,
                      uploaded_at = NOW()
                """,
                (run_id, _json.dumps(traces), len(traces)),
            )
        conn.commit()
    finally:
        put_conn(conn)


def get_opt_run_traces(run_id: str) -> tuple[list[dict], int]:
    """Return (traces, count) for a run; (empty, 0) if none stored."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT traces, trace_count FROM optimization_run_traces WHERE run_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return [], 0
            return list(row[0] or []), int(row[1] or 0)
    finally:
        put_conn(conn)
```

- [ ] **Step 1.4: Run + commit**

Run: `pytest retune-cloud/tests/test_opt_run_traces_db.py -v` — expect 3 pass.
Run: `pytest tests/ retune-cloud/tests/ -q` — 120 passed (117 prior + 3 new).

```bash
cd retune-cloud
git add server/db/schema.sql server/db/postgres.py tests/test_opt_run_traces_db.py
git commit -m "optimizer: schema + helpers for per-run trace bundle"
```

---

## Task 2: Models — PromptCandidate + execution envelopes

**Files:**
- Modify: `retune-cloud/server/optimizer/models.py` (append classes)
- Test: `retune-cloud/tests/test_prompt_optimizer_models.py`

- [ ] **Step 2.1: Write failing tests**

```python
# retune-cloud/tests/test_prompt_optimizer_models.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.optimizer.models import (
    PromptCandidate,
    CandidateExecutionRequest,
    CandidateExecutionResult,
)


def test_prompt_candidate_fields():
    c = PromptCandidate(
        candidate_id="cand_1",
        system_prompt="You are a helpful assistant.",
        few_shot_examples=[],
        generation_round=1,
        parent_id=None,
    )
    assert c.candidate_id == "cand_1"
    assert c.generation_round == 1


def test_candidate_execution_request():
    req = CandidateExecutionRequest(
        run_id="run_1",
        candidate_id="cand_1",
        config_overrides={"system_prompt": "New prompt"},
        query_set=[{"query": "hello", "trace_id": "t1"}],
    )
    assert req.run_id == "run_1"


def test_candidate_execution_result_requires_scores():
    with pytest.raises(ValidationError):
        CandidateExecutionResult(run_id="r", candidate_id="c", trace={})
```

Run — expect FAIL.

- [ ] **Step 2.2: Add models**

Append to `retune-cloud/server/optimizer/models.py` (before the envelope classes RunCandidateMsg etc.):

```python
class PromptCandidate(BaseModel):
    """One prompt variant explored by the beam search."""
    candidate_id: str
    system_prompt: str
    few_shot_examples: list[dict[str, Any]] = Field(default_factory=list)
    generation_round: int = 1
    parent_id: str | None = None


class CandidateExecutionRequest(BaseModel):
    """Cloud → SDK: execute this candidate's config against these queries.
    Same shape as RunCandidateMsg (Phase 1) but with explicit naming for
    the new Phase 2 dispatch flow."""
    run_id: str
    candidate_id: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    query_set: list[dict[str, Any]] = Field(default_factory=list)


class CandidateExecutionResult(BaseModel):
    """SDK → Cloud: execution trace + eval scores for a candidate."""
    run_id: str
    candidate_id: str
    trace: dict[str, Any]
    eval_scores: dict[str, float]
```

- [ ] **Step 2.3: Verify + commit**

Run: `pytest retune-cloud/tests/test_prompt_optimizer_models.py -v` — 3 pass.
Run full suite — 123 pass.

```bash
cd retune-cloud
git add server/optimizer/models.py tests/test_prompt_optimizer_models.py
git commit -m "optimizer: add PromptCandidate + execution envelopes"
```

---

## Task 3: Route — preauthorize accepts traces payload

**Files:**
- Modify: `retune-cloud/server/routes/optimize.py`
- Test: `retune-cloud/tests/test_optimize_routes_traces.py`

- [ ] **Step 3.1: Write failing test**

```python
# retune-cloud/tests/test_optimize_routes_traces.py
from __future__ import annotations

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
@patch("server.routes.optimize.BackgroundTasks.add_task")
def test_preauthorize_saves_traces(mock_add_task, mock_db, _auth):
    mock_db.count_opt_runs_used.return_value = (0, 15)

    r = client.post(
        "/api/v1/optimize/preauthorize",
        json={
            "source": "last_n_traces",
            "n_traces": 2,
            "axes": ["prompt"],
            "traces": [
                {"query": "q1", "response": "r1"},
                {"query": "q2", "response": "r2"},
            ],
        },
        headers={"Authorization": "Bearer test"},
    )
    assert r.status_code == 200
    # save_opt_run_traces called with the 2-trace bundle
    mock_db.save_opt_run_traces.assert_called_once()
    call_args = mock_db.save_opt_run_traces.call_args
    assert len(call_args.args[1]) == 2  # 2 traces


@patch("server.routes.optimize.require_auth", return_value={"org": "org_1"})
@patch("server.routes.optimize.db")
@patch("server.routes.optimize.BackgroundTasks.add_task")
def test_preauthorize_no_traces_still_works(mock_add_task, mock_db, _auth):
    """source='collect_next' has no traces at preauth time."""
    mock_db.count_opt_runs_used.return_value = (0, 15)

    r = client.post(
        "/api/v1/optimize/preauthorize",
        json={"source": "collect_next", "n_traces": 10, "axes": ["prompt"]},
        headers={"Authorization": "Bearer test"},
    )
    assert r.status_code == 200
    # save_opt_run_traces NOT called when no traces provided
    assert not mock_db.save_opt_run_traces.called
```

Run — expect FAIL.

- [ ] **Step 3.2: Modify PreauthorizeRequest + route**

In `retune-cloud/server/routes/optimize.py`, change `PreauthorizeRequest`:

```python
class PreauthorizeRequest(BaseModel):
    source: str
    n_traces: int = Field(ge=1, le=10_000)
    axes: list[str] = Field(default_factory=lambda: ["prompt", "tools", "rag"])
    reward_spec: dict | None = None
    rewriter_llm: str | None = None
    traces: list[dict[str, Any]] | None = None  # NEW — only for source="last_n_traces"
```

Inside the `preauthorize` function, after `db.create_opt_run(...)` and before `db.increment_opt_runs_used(...)`:

```python
    # Store uploaded traces (for source="last_n_traces" only)
    if req.traces:
        db.save_opt_run_traces(run_id, req.traces)
```

- [ ] **Step 3.3: Run + commit**

Run new tests — 2 pass. Run full suite — 125 pass (123 + 2).

```bash
cd retune-cloud
git add server/routes/optimize.py tests/test_optimize_routes_traces.py
git commit -m "optimizer: preauthorize accepts + stores traces payload"
```

---

## Task 4: SDK — trace_collector helper

**Files:**
- Create: `src/retune/optimizer/trace_collector.py`
- Test: `tests/unit/test_trace_collector.py`

- [ ] **Step 4.1: Write failing test**

```python
# tests/unit/test_trace_collector.py
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.trace_collector import collect_last_n_local_traces


def test_collect_last_n_from_storage():
    storage = MagicMock()
    storage.get_traces.return_value = [
        {"id": "t1", "query": "q1", "response": "r1", "duration_ms": 100},
        {"id": "t2", "query": "q2", "response": "r2", "duration_ms": 120},
    ]

    traces = collect_last_n_local_traces(storage, n=2)
    assert len(traces) == 2
    assert traces[0]["query"] == "q1"
    storage.get_traces.assert_called_once_with(limit=2)


def test_collect_last_n_empty_storage():
    storage = MagicMock()
    storage.get_traces.return_value = []
    assert collect_last_n_local_traces(storage, n=50) == []


def test_collect_last_n_fewer_than_requested():
    storage = MagicMock()
    storage.get_traces.return_value = [{"id": "t1", "query": "q"}]
    traces = collect_last_n_local_traces(storage, n=50)
    assert len(traces) == 1  # only what was available
```

Run — expect FAIL.

- [ ] **Step 4.2: Implement**

```python
# src/retune/optimizer/trace_collector.py
"""Collect recent local traces for upload to the cloud optimizer."""
from __future__ import annotations

from typing import Any, Protocol


class _StorageLike(Protocol):
    def get_traces(self, limit: int) -> list[dict[str, Any]]: ...


def collect_last_n_local_traces(storage: _StorageLike, n: int) -> list[dict[str, Any]]:
    """Return up to `n` most recent traces from local storage, newest first.

    Used when `Retuner.optimize(source="last_n_traces")` — the SDK uploads
    these as the payload in /v1/optimize/preauthorize.
    """
    return storage.get_traces(limit=n)
```

- [ ] **Step 4.3: Run + commit (outer repo)**

Run: `pytest tests/unit/test_trace_collector.py -v` — 3 pass.
Run full suite — 128 pass.

```bash
git add src/retune/optimizer/trace_collector.py tests/unit/test_trace_collector.py
git commit -m "optimizer: SDK trace_collector for last_n_traces upload"
```

---

## Task 5: SDK — OptimizerClient.preauthorize carries traces

**Files:**
- Modify: `src/retune/optimizer/client.py`
- Test: append to `tests/unit/test_optimizer_client.py`

- [ ] **Step 5.1: Write failing test**

Append to existing `tests/unit/test_optimizer_client.py`:

```python
@patch("retune.optimizer.client.urlopen")
def test_preauthorize_includes_traces_in_body(mock_urlopen):
    import json as _json
    resp = MagicMock()
    resp.read.return_value = b'{"run_id": "r", "runs_remaining": 14}'
    resp.__enter__.return_value = resp
    mock_urlopen.return_value = resp

    client = OptimizerClient(api_key="rt-test", base_url="https://api.example.com")
    traces = [{"query": "q1"}, {"query": "q2"}]
    client.preauthorize(
        source="last_n_traces", n_traces=2, axes=["prompt"],
        traces=traces,
    )
    # Inspect the POSTed body
    req = mock_urlopen.call_args[0][0]
    body = _json.loads(req.data)
    assert body["traces"] == traces
```

Run — expect FAIL (parameter not accepted).

- [ ] **Step 5.2: Modify client**

In `src/retune/optimizer/client.py`, change the `preauthorize` method signature + body:

```python
    def preauthorize(
        self,
        source: str,
        n_traces: int,
        axes: list[str],
        reward_spec: dict[str, Any] | None = None,
        rewriter_llm: str | None = None,
        traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body = {
            "source": source,
            "n_traces": n_traces,
            "axes": axes,
            "reward_spec": reward_spec,
            "rewriter_llm": rewriter_llm,
        }
        if traces is not None:
            body["traces"] = traces
        return self._post("/api/v1/optimize/preauthorize", body)
```

- [ ] **Step 5.3: Also update `Retuner.optimize` in `src/retune/wrapper.py`**

Find `Retuner.optimize`. Near the `client.preauthorize(...)` call, add trace collection for `source="last_n_traces"`:

```python
    traces_payload = None
    if source == "last_n_traces":
        from retune.optimizer.trace_collector import collect_last_n_local_traces
        traces_payload = collect_last_n_local_traces(self._storage, n=n)

    resp = client.preauthorize(
        source=source, n_traces=n, axes=axes_list,
        reward_spec=reward_spec, rewriter_llm=rewriter_llm,
        traces=traces_payload,
    )
```

(`self._storage` already exists on Retuner from Phase 1. If its name differs, use the actual attribute — check wrapper.py.)

- [ ] **Step 5.4: Run + commit**

Run outer tests — should pass. Full suite — 129 pass.

```bash
git add src/retune/optimizer/client.py src/retune/wrapper.py tests/unit/test_optimizer_client.py
git commit -m "optimizer: SDK uploads last_n_traces at preauth time"
```

---

## Task 6: SDK — _make_candidate_runner applies config overrides

**Files:**
- Modify: `src/retune/wrapper.py`
- Test: `tests/unit/test_wrapper_optimize_overrides.py`

- [ ] **Step 6.1: Write failing test**

```python
# tests/unit/test_wrapper_optimize_overrides.py
from __future__ import annotations

from unittest.mock import MagicMock

from retune import Retuner, Mode


def test_runner_applies_system_prompt_override_and_restores():
    captured_prompts = []

    def agent(q: str) -> str:
        # Stub captures the _config.system_prompt at call time
        return f"resp"

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    # Set an initial system prompt via config
    retuner._config.system_prompt = "ORIGINAL"

    # Patch the adapter.run to capture what prompt was active during the call
    orig_run = retuner._adapter.run
    def capturing_run(q: str):
        captured_prompts.append(retuner._config.system_prompt)
        return orig_run(q)
    retuner._adapter.run = capturing_run

    runner = retuner._make_candidate_runner()
    # Execute with an override
    trace, scores = runner(
        {"system_prompt": "OVERRIDDEN"},
        [{"query": "hello", "trace_id": "t1"}],
    )

    assert captured_prompts == ["OVERRIDDEN"]  # override was applied during the call
    # After the runner returns, original prompt is restored
    assert retuner._config.system_prompt == "ORIGINAL"


def test_runner_no_overrides_leaves_config_unchanged():
    def agent(q: str) -> str:
        return "resp"

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner._config.system_prompt = "ORIGINAL"
    runner = retuner._make_candidate_runner()

    trace, scores = runner({}, [{"query": "hello"}])
    assert retuner._config.system_prompt == "ORIGINAL"
```

Run — expect FAIL.

- [ ] **Step 6.2: Replace `_make_candidate_runner`**

In `src/retune/wrapper.py`, replace the Phase 1 stub body of `_make_candidate_runner` with:

```python
    def _make_candidate_runner(self):
        """Return a callable that runs the wrapped agent with config overrides.

        Phase 2: applies `config_overrides` by temporarily mutating
        self._config (system_prompt, few_shot_examples), running the candidate,
        then restoring the original config. Adapter implementations read from
        self._config at call time, so this override propagates correctly.
        """
        def _runner(overrides: dict, queries: list):
            # Snapshot the fields we're about to override
            snapshot = {}
            for key in ("system_prompt", "few_shot_examples"):
                if key in overrides:
                    snapshot[key] = getattr(self._config, key, None)
                    setattr(self._config, key, overrides[key])

            try:
                if not queries:
                    return ({"query": "", "response": ""}, {"llm_judge": 0.0})
                q = queries[0].get("query", "")
                try:
                    resp = self._adapter.run(q) if self._adapter else ""
                except Exception:
                    resp = ""
                return (
                    {"query": q, "response": str(resp)},
                    {"llm_judge": 0.0, "cost": 0.0, "latency": 0.0},
                )
            finally:
                # Restore snapshot
                for key, old_val in snapshot.items():
                    setattr(self._config, key, old_val)
        return _runner
```

**Note:** scores are still stubs — real eval scoring is wired by JudgeAgent on the cloud side, not by the SDK runner. The SDK's job is just to produce a trace with the overridden config; evaluation happens later in the Orchestrator.

- [ ] **Step 6.3: Run + commit**

Run: `pytest tests/unit/test_wrapper_optimize_overrides.py -v` — 2 pass.
Run full suite — 131 pass.

```bash
git add src/retune/wrapper.py tests/unit/test_wrapper_optimize_overrides.py
git commit -m "optimizer: SDK candidate runner applies + restores config overrides"
```

---

## Task 7: Cloud — rewriter LLM factory

**Files:**
- Create: `retune-cloud/server/optimizer/prompt_optimizer/__init__.py`
- Create: `retune-cloud/server/optimizer/prompt_optimizer/llm.py`
- Test: `retune-cloud/tests/test_prompt_optimizer_llm.py`

- [ ] **Step 7.1: Write failing test**

```python
# retune-cloud/tests/test_prompt_optimizer_llm.py
from __future__ import annotations

from unittest.mock import patch

import pytest

from server.optimizer.prompt_optimizer.llm import create_rewriter_llm


@patch("server.optimizer.prompt_optimizer.llm._create_openai")
def test_create_rewriter_llm_openai(mock_openai):
    mock_openai.return_value = "openai-instance"
    llm = create_rewriter_llm("gpt-4o-mini")
    assert llm == "openai-instance"


@patch("server.optimizer.prompt_optimizer.llm._create_anthropic")
def test_create_rewriter_llm_anthropic(mock_anth):
    mock_anth.return_value = "anthropic-instance"
    llm = create_rewriter_llm("claude-3-7-sonnet")
    assert llm == "anthropic-instance"


def test_create_rewriter_llm_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        create_rewriter_llm("not-a-real-model")
```

Run — expect FAIL.

- [ ] **Step 7.2: Create package + implement**

```python
# retune-cloud/server/optimizer/prompt_optimizer/__init__.py
"""PromptOptimizerAgent — Beam Search APO for the prompt axis (Phase 2)."""
```

```python
# retune-cloud/server/optimizer/prompt_optimizer/llm.py
"""LLM factory for the prompt rewriter — user's choice of model."""
from __future__ import annotations

from typing import Any


def _create_openai(model: str) -> Any:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=0.7)


def _create_anthropic(model: str) -> Any:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, temperature=0.7)


def _create_google(model: str) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=0.7)


_OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"}
_ANTHROPIC_MODELS = {"claude-3-7-sonnet", "claude-3-5-sonnet-20241022", "claude-3-opus"}
_GOOGLE_MODELS = {"gemini-1.5-pro", "gemini-1.5-flash"}


def create_rewriter_llm(model: str) -> Any:
    """Return a LangChain LLM for the given model name."""
    if model in _OPENAI_MODELS or model.startswith("gpt-"):
        return _create_openai(model)
    if model in _ANTHROPIC_MODELS or model.startswith("claude-"):
        return _create_anthropic(model)
    if model in _GOOGLE_MODELS or model.startswith("gemini-"):
        return _create_google(model)
    raise ValueError(
        f"Unknown model {model!r}. Supported: gpt-*, claude-*, gemini-*."
    )
```

- [ ] **Step 7.3: Run + commit**

Run new tests — 3 pass. Full — 134 pass.

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/__init__.py server/optimizer/prompt_optimizer/llm.py tests/test_prompt_optimizer_llm.py
git commit -m "optimizer: rewriter LLM factory (user's choice of model)"
```

---

## Task 8: Cloud — prompt_optimizer/prompts.py + state.py (copy from premium)

**Files:**
- Create: `retune-cloud/server/optimizer/prompt_optimizer/prompts.py`
- Create: `retune-cloud/server/optimizer/prompt_optimizer/state.py`

- [ ] **Step 8.1: Read the premium originals**

Read `retune-cloud/premium/agents/optimizer/prompts.py` (139 lines) and `retune-cloud/premium/agents/optimizer/state.py` (44 lines) in full. They contain the APO critique/rewrite prompts + BeamSearchState dataclasses.

- [ ] **Step 8.2: Copy the prompts verbatim into the new location**

Copy the **entire contents** of `premium/agents/optimizer/prompts.py` to `server/optimizer/prompt_optimizer/prompts.py` unchanged. These are stable prompt strings — no adaptation needed.

- [ ] **Step 8.3: Copy state.py, trim multi-strategy bits**

Copy `premium/agents/optimizer/state.py` to `server/optimizer/prompt_optimizer/state.py`, but DELETE any `config_tuner_state` / `tool_curator_state` fields — PromptOptimizerAgent only uses `beam_search_state`. If the only state classes are already prompt-specific, copy as-is.

- [ ] **Step 8.4: Run + commit**

Run full suite — still 134 pass (no new tests, no imports breaking).

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/prompts.py server/optimizer/prompt_optimizer/state.py
git commit -m "optimizer: copy APO prompts + state from premium/ (Phase 2 refactor)"
```

---

## Task 9: Cloud — prompt_optimizer/beam_search.py (copy + adapt)

**Files:**
- Create: `retune-cloud/server/optimizer/prompt_optimizer/beam_search.py`
- Test: `retune-cloud/tests/test_prompt_optimizer_beam.py`

The existing `premium/agents/optimizer/beam_search.py` (359 lines) implements `BeamSearchAPO` with methods like `critique`, `rewrite`, `score`, `run_beam_round`. Adapt to the new interfaces:

- Input: list of traces (dicts) + baseline system prompt + reward spec
- Output: list of `PromptCandidate` objects (the new model)
- LLM calls go through `create_rewriter_llm(...)`
- Scoring happens OUTSIDE this class — `BeamSearchAPO` only generates candidates; scoring is done by the Orchestrator via the JobQueue + JudgeAgent round-trip.

- [ ] **Step 9.1: Write failing test**

```python
# retune-cloud/tests/test_prompt_optimizer_beam.py
from __future__ import annotations

from unittest.mock import MagicMock

from server.optimizer.prompt_optimizer.beam_search import BeamSearchAPO


def test_generate_candidates_returns_prompt_candidates():
    # Mock LLM that returns fixed critiques and rewrites
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content="Critique: too vague"),
        MagicMock(content="Rewrite: Be more specific."),
        MagicMock(content="Rewrite: Use examples."),
    ]

    apo = BeamSearchAPO(
        llm=mock_llm,
        beam_width=2,
        branch_factor=2,
    )
    traces = [{"query": "q1", "response": "bad response"}]
    baseline_prompt = "You are a helpful assistant."

    candidates = apo.generate_candidates(
        traces=traces,
        baseline_prompt=baseline_prompt,
        max_candidates=4,
    )
    # Expect at least the baseline + some rewrites
    assert len(candidates) >= 1
    assert all(c.system_prompt for c in candidates)
    assert candidates[0].generation_round == 0  # baseline is round 0


def test_generate_candidates_respects_max():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Rewrite: x")
    apo = BeamSearchAPO(llm=mock_llm, beam_width=2, branch_factor=3)
    traces = [{"query": "q"}]

    candidates = apo.generate_candidates(
        traces=traces, baseline_prompt="base", max_candidates=3,
    )
    assert len(candidates) <= 3
```

Run — expect FAIL.

- [ ] **Step 9.2: Implement BeamSearchAPO (adapt from premium)**

Read the existing `retune-cloud/premium/agents/optimizer/beam_search.py` for reference. Create a simplified version in `retune-cloud/server/optimizer/prompt_optimizer/beam_search.py`:

```python
# retune-cloud/server/optimizer/prompt_optimizer/beam_search.py
"""Beam Search APO — generates candidate prompts by critique+rewrite cycles.

Adapted from premium/agents/optimizer/beam_search.py with these changes:
- No scoring inside this class (scoring happens via JobQueue→SDK→Judge round-trip)
- Output is list[PromptCandidate] instead of scored rows
- LLM is injected (not constructed from a model string)
"""
from __future__ import annotations

import uuid
from typing import Any

from server.optimizer.models import PromptCandidate
from server.optimizer.prompt_optimizer.prompts import (
    CRITIQUE_PROMPT,
    REWRITE_PROMPT,
)


class BeamSearchAPO:
    def __init__(
        self,
        llm: Any,
        beam_width: int = 2,
        branch_factor: int = 2,
    ) -> None:
        self._llm = llm
        self._beam_width = beam_width
        self._branch_factor = branch_factor

    def _mk_id(self) -> str:
        return f"cand_{uuid.uuid4().hex[:12]}"

    def _critique(self, prompt: str, traces: list[dict]) -> str:
        trace_excerpt = "\n".join(
            f"Q: {t.get('query', '')}\nA: {t.get('response', '')}"
            for t in traces[:5]
        )
        msg = CRITIQUE_PROMPT.format(prompt=prompt, traces=trace_excerpt)
        return str(self._llm.invoke(msg).content)

    def _rewrite(self, prompt: str, critique: str) -> str:
        msg = REWRITE_PROMPT.format(prompt=prompt, critique=critique)
        return str(self._llm.invoke(msg).content)

    def generate_candidates(
        self,
        traces: list[dict],
        baseline_prompt: str,
        max_candidates: int = 6,
    ) -> list[PromptCandidate]:
        """Return the baseline + up to `max_candidates-1` rewrites.

        One round of critique → k rewrites per beam candidate.
        """
        baseline = PromptCandidate(
            candidate_id=self._mk_id(),
            system_prompt=baseline_prompt,
            generation_round=0,
            parent_id=None,
        )
        candidates: list[PromptCandidate] = [baseline]

        # Round 1: critique the baseline, generate branch_factor rewrites
        critique = self._critique(baseline_prompt, traces)
        rewrites_remaining = max_candidates - 1
        for _ in range(min(self._branch_factor, rewrites_remaining)):
            new_prompt = self._rewrite(baseline_prompt, critique)
            candidates.append(PromptCandidate(
                candidate_id=self._mk_id(),
                system_prompt=new_prompt,
                generation_round=1,
                parent_id=baseline.candidate_id,
            ))

        return candidates[:max_candidates]
```

**Note:** The existing `premium/` version does MUCH more (multi-round, scoring, verification, budget tracking). Phase 2 keeps it minimal — one round of critique+rewrite. Later phases can extend to multi-round with scored feedback.

- [ ] **Step 9.3: Check prompts.py has the expected constants**

Verify `CRITIQUE_PROMPT` and `REWRITE_PROMPT` exist in `retune-cloud/server/optimizer/prompt_optimizer/prompts.py` (copied in Task 8). If they use different names, either:
- Rename them in prompts.py to match
- Or update the imports in beam_search.py to match the actual names

- [ ] **Step 9.4: Run + commit**

Run: `pytest retune-cloud/tests/test_prompt_optimizer_beam.py -v` — 2 pass.
Full suite — 136 pass.

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/beam_search.py tests/test_prompt_optimizer_beam.py
git commit -m "optimizer: beam search APO — single-round candidate generation"
```

---

## Task 10: Cloud — PromptOptimizerAgent (deepagents-based)

**Files:**
- Create: `retune-cloud/server/optimizer/prompt_optimizer/agent.py`
- Test: `retune-cloud/tests/test_prompt_optimizer_agent.py`

- [ ] **Step 10.1: Write failing test**

```python
# retune-cloud/tests/test_prompt_optimizer_agent.py
from __future__ import annotations

from unittest.mock import patch, MagicMock

from server.optimizer.prompt_optimizer.agent import PromptOptimizerAgent
from server.optimizer.models import PromptCandidate


@patch("server.optimizer.prompt_optimizer.agent.BeamSearchAPO")
@patch("server.optimizer.prompt_optimizer.agent.create_rewriter_llm")
def test_generate_candidates_delegates_to_beam(mock_llm_factory, mock_beam_cls):
    mock_llm_factory.return_value = "llm-inst"
    mock_beam = MagicMock()
    mock_beam.generate_candidates.return_value = [
        PromptCandidate(
            candidate_id="c1", system_prompt="p1", generation_round=0,
        ),
    ]
    mock_beam_cls.return_value = mock_beam

    agent = PromptOptimizerAgent(rewriter_llm="gpt-4o-mini")
    candidates = agent.generate_candidates(
        traces=[{"query": "q"}],
        baseline_prompt="base",
        max_candidates=4,
    )
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "c1"
    mock_llm_factory.assert_called_once_with("gpt-4o-mini")
    mock_beam.generate_candidates.assert_called_once()


@patch("server.optimizer.prompt_optimizer.agent.create_rewriter_llm")
def test_default_rewriter_llm(mock_llm_factory):
    mock_llm_factory.return_value = MagicMock()
    agent = PromptOptimizerAgent()  # no rewriter_llm specified
    # Default should be claude-3-7-sonnet per spec
    mock_llm_factory.assert_called_once_with("claude-3-7-sonnet")
```

Run — expect FAIL.

- [ ] **Step 10.2: Implement PromptOptimizerAgent**

```python
# retune-cloud/server/optimizer/prompt_optimizer/agent.py
"""PromptOptimizerAgent — Phase 2 prompt-axis subagent.

Wraps BeamSearchAPO with a deepagents-style interface. In Phase 2 we use
deepagents directly for the outer loop; BeamSearchAPO is the inner algorithm
that produces candidate prompts. The Orchestrator calls `generate_candidates`
and then dispatches each candidate through the JobQueue for SDK-side
execution + cloud-side scoring.
"""
from __future__ import annotations

import logging
from typing import Any

from server.optimizer.models import PromptCandidate
from server.optimizer.prompt_optimizer.beam_search import BeamSearchAPO
from server.optimizer.prompt_optimizer.llm import create_rewriter_llm

logger = logging.getLogger(__name__)


class PromptOptimizerAgent:
    """Prompt-axis subagent. Generates candidate prompts via beam search APO."""

    def __init__(
        self,
        rewriter_llm: str = "claude-3-7-sonnet",
        beam_width: int = 2,
        branch_factor: int = 2,
    ) -> None:
        self._llm = create_rewriter_llm(rewriter_llm)
        self._beam = BeamSearchAPO(
            llm=self._llm,
            beam_width=beam_width,
            branch_factor=branch_factor,
        )

    def generate_candidates(
        self,
        traces: list[dict[str, Any]],
        baseline_prompt: str,
        max_candidates: int = 6,
    ) -> list[PromptCandidate]:
        """Run beam search APO and return candidate prompts.

        Phase 2: one critique+rewrite round. The Orchestrator is responsible
        for dispatching each candidate through the JobQueue → SDK → Judge
        pipeline and selecting the winner based on scored results.
        """
        if not baseline_prompt:
            logger.warning("No baseline prompt provided; using empty string")
            baseline_prompt = ""

        try:
            return self._beam.generate_candidates(
                traces=traces,
                baseline_prompt=baseline_prompt,
                max_candidates=max_candidates,
            )
        except Exception as e:
            logger.exception("Beam search failed: %s", e)
            # Fall back to baseline-only — at least don't break the run
            return [
                PromptCandidate(
                    candidate_id="cand_baseline_fallback",
                    system_prompt=baseline_prompt,
                    generation_round=0,
                ),
            ]
```

**Note on deepagents:** the spec's design rationale for "keep deepagents" is that the subagent framework brings TodoListMiddleware / SubAgentMiddleware / FilesystemMiddleware. For Phase 2 minimum, we don't yet need those — the beam search loop is simple enough that direct LLM calls suffice. The Phase 3+ tasks (ToolOptimizer, RAGOptimizer) will layer in full deepagents when the subagent delegation becomes non-trivial. If you want the deepagents scaffolding here for forward compatibility, add a `_build_deep_agent()` method that wraps BeamSearchAPO as a deepagents sub-agent — but don't call it in generate_candidates yet.

- [ ] **Step 10.3: Run + commit**

Run new tests — 2 pass. Full — 138 pass.

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/agent.py tests/test_prompt_optimizer_agent.py
git commit -m "optimizer: PromptOptimizerAgent — generates candidates via beam search"
```

---

## Task 11: Cloud — Real OptimizerOrchestrator.run()

**Files:**
- Modify: `retune-cloud/server/optimizer/orchestrator.py` (major rewrite of `run()`)
- Test: `retune-cloud/tests/test_orchestrator_real.py`

This is the Phase 2 centerpiece. Orchestrator becomes:

1. Load the run row + uploaded traces from DB
2. Instantiate PromptOptimizerAgent with the run's `rewriter_llm`
3. Extract baseline prompt from first trace's `config_snapshot.system_prompt` (or `""` if missing)
4. Call `PromptOptimizerAgent.generate_candidates(traces, baseline, max=6)`
5. For each candidate: push `RunCandidateMsg` with `config_overrides={"system_prompt": candidate.system_prompt}` to the JobQueue, using a subset of trace queries as the query_set
6. Wait for all candidate results via `get_results()` polling (with timeout)
7. For each result: call `JudgeAgent.score(eval_scores, reward_spec, baseline)` → JudgeResult
8. Pick winner (max scalar_score)
9. Build Tier 1 suggestion from winner (if score > baseline)
10. Build pareto_data from all JudgeResults
11. Render Report via ReportWriterAgent with real data, save it, mark run completed

- [ ] **Step 11.1: Write failing test**

```python
# retune-cloud/tests/test_orchestrator_real.py
from __future__ import annotations

from unittest.mock import patch, MagicMock

from server.optimizer.orchestrator import OptimizerOrchestrator
from server.optimizer.models import PromptCandidate


@patch("server.optimizer.orchestrator.PromptOptimizerAgent")
@patch("server.optimizer.orchestrator.JudgeAgent")
@patch("server.optimizer.orchestrator.get_queue")
@patch("server.optimizer.orchestrator.get_results")
@patch("server.optimizer.orchestrator.db")
def test_real_orchestrator_dispatches_candidates_and_writes_tier1(
    mock_db, mock_get_results, mock_get_queue, mock_judge_cls, mock_prompt_cls
):
    # Setup: run exists, 3 traces stored, baseline prompt exists
    mock_db.get_opt_run.return_value = {
        "id": "run_1", "org_id": "org_1", "status": "pending",
        "source": "last_n_traces", "n_traces": 3, "axes": ["prompt"],
        "reward_spec": {}, "rewriter_llm": "gpt-4o-mini",
    }
    traces = [
        {"query": "q1", "response": "r1",
         "config_snapshot": {"system_prompt": "You are helpful."}},
        {"query": "q2", "response": "r2",
         "config_snapshot": {"system_prompt": "You are helpful."}},
        {"query": "q3", "response": "r3",
         "config_snapshot": {"system_prompt": "You are helpful."}},
    ]
    mock_db.get_opt_run_traces.return_value = (traces, 3)

    # PromptOptimizer returns baseline + 1 rewrite
    mock_prompt = MagicMock()
    mock_prompt.generate_candidates.return_value = [
        PromptCandidate(
            candidate_id="cand_base", system_prompt="You are helpful.",
            generation_round=0,
        ),
        PromptCandidate(
            candidate_id="cand_rewrite",
            system_prompt="You are a precise helper.",
            generation_round=1,
        ),
    ]
    mock_prompt_cls.return_value = mock_prompt

    # Judge gives the rewrite a higher score
    mock_judge = MagicMock()
    def score_side_effect(eval_scores, spec, baseline):
        r = MagicMock()
        r.scalar_score = 8.0 if eval_scores.get("_cand") == "cand_rewrite" else 5.0
        r.guardrails_held = True
        r.dimensions = eval_scores
        return r
    mock_judge.score.side_effect = score_side_effect
    mock_judge_cls.return_value = mock_judge

    # Job queue + results
    mock_queue = MagicMock()
    mock_get_queue.return_value = mock_queue
    mock_results = MagicMock()
    def get_side_effect(run_id, candidate_id):
        return {
            "run_id": run_id, "candidate_id": candidate_id,
            "trace": {"query": "q", "response": "r"},
            "eval_scores": {"llm_judge": 7.0, "_cand": candidate_id},
        }
    mock_results.get.side_effect = get_side_effect
    mock_get_results.return_value = mock_results

    orch = OptimizerOrchestrator()
    # Short timeout since everything is mocked
    orch.run("run_1", candidate_result_timeout=0.5)

    # Verify: both candidates were pushed to the queue
    assert mock_queue.push.call_count == 2
    # Verify: a Tier 1 suggestion was written for the winning rewrite
    save_kwargs = mock_db.save_opt_report.call_args.kwargs
    tier1 = save_kwargs["tier1"]
    assert len(tier1) >= 1
    assert any("precise helper" in str(s) for s in tier1)
    # Run marked completed
    calls = [c.args for c in mock_db.update_opt_run_status.call_args_list]
    statuses = [c[1] for c in calls]
    assert "completed" in statuses


@patch("server.optimizer.orchestrator.db")
def test_real_orchestrator_no_traces_marks_failed(mock_db):
    mock_db.get_opt_run.return_value = {
        "id": "run_2", "org_id": "org_1", "status": "pending",
        "source": "last_n_traces", "n_traces": 0, "axes": ["prompt"],
        "reward_spec": {}, "rewriter_llm": None,
    }
    mock_db.get_opt_run_traces.return_value = ([], 0)

    orch = OptimizerOrchestrator()
    orch.run("run_2")

    calls = [c.args for c in mock_db.update_opt_run_status.call_args_list]
    assert any(c[1] == "failed" for c in calls)
```

Run — expect FAIL (orchestrator is still the Phase 1 noop).

- [ ] **Step 11.2: Replace Orchestrator.run() body**

Rewrite `retune-cloud/server/optimizer/orchestrator.py`:

```python
"""OptimizerOrchestrator — Phase 2 real implementation.

Reads the run's uploaded traces, invokes PromptOptimizerAgent to generate
candidate prompts, dispatches each candidate through the JobQueue for
SDK-side execution, collects results, scores them via JudgeAgent, and
writes a real Tier 1 suggestion for the winning candidate.

Phase 3+ extends this to also dispatch ToolOptimizer and RAGOptimizer
based on trace analysis of bottlenecks.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from server.db import postgres as db
from server.optimizer.judge import JudgeAgent
from server.optimizer.job_queue import get_queue, get_results
from server.optimizer.prompt_optimizer.agent import PromptOptimizerAgent
from server.optimizer.report_writer import ReportWriterAgent
from server.optimizer.reward_parser import default_reward_spec, parse_reward_spec

logger = logging.getLogger(__name__)

_DEFAULT_CANDIDATE_TIMEOUT = 60.0  # seconds per candidate result
_DEFAULT_MAX_CANDIDATES = 6
_DEFAULT_QUERIES_PER_CANDIDATE = 3  # subsample of the uploaded traces


class OptimizerOrchestrator:
    def __init__(self) -> None:
        self._writer = ReportWriterAgent()
        self._judge = JudgeAgent()

    def run(
        self,
        run_id: str,
        candidate_result_timeout: float = _DEFAULT_CANDIDATE_TIMEOUT,
    ) -> None:
        row = None
        try:
            row = db.get_opt_run(run_id)
            if row is None:
                raise ValueError(f"Run {run_id} not found")

            db.update_opt_run_status(run_id, "running")

            traces, trace_count = db.get_opt_run_traces(run_id)
            if trace_count == 0:
                raise ValueError(
                    "No traces available for optimization. "
                    "For source='last_n_traces', the SDK must upload traces at preauth time."
                )

            # Reward spec
            reward_raw = row.get("reward_spec") or {}
            spec = parse_reward_spec(reward_raw) if reward_raw else default_reward_spec()

            # Baseline prompt from the first trace's config snapshot
            baseline_prompt = ""
            for t in traces:
                cs = t.get("config_snapshot", {})
                if cs.get("system_prompt"):
                    baseline_prompt = cs["system_prompt"]
                    break

            # Generate candidates
            prompt_agent = PromptOptimizerAgent(
                rewriter_llm=row.get("rewriter_llm") or "claude-3-7-sonnet",
            )
            candidates = prompt_agent.generate_candidates(
                traces=traces,
                baseline_prompt=baseline_prompt,
                max_candidates=_DEFAULT_MAX_CANDIDATES,
            )

            # Subsample queries for each candidate
            query_set = [
                {"query": t.get("query", ""), "trace_id": t.get("id", "")}
                for t in traces[:_DEFAULT_QUERIES_PER_CANDIDATE]
            ]

            # Dispatch each candidate via JobQueue
            q = get_queue()
            for cand in candidates:
                q.push(run_id, {
                    "type": "run_candidate",
                    "candidate_id": cand.candidate_id,
                    "config_overrides": {"system_prompt": cand.system_prompt},
                    "query_set": query_set,
                })

            # Collect results with per-candidate timeout
            results_store = get_results()
            scored: list[dict[str, Any]] = []
            pareto_data: list[dict[str, Any]] = []
            baseline_scores: dict[str, float] = {}

            deadline = time.time() + candidate_result_timeout * len(candidates)
            for cand in candidates:
                while time.time() < deadline:
                    result = results_store.get(run_id, cand.candidate_id)
                    if result is not None:
                        break
                    time.sleep(0.25)
                else:
                    logger.warning("Timeout waiting for candidate %s", cand.candidate_id)
                    continue

                eval_scores = result.get("eval_scores", {})
                if cand.generation_round == 0:
                    baseline_scores = eval_scores

                judge_result = self._judge.score(eval_scores, spec, baseline_scores)
                scored.append({
                    "candidate": cand,
                    "judge_result": judge_result,
                    "eval_scores": eval_scores,
                })
                pareto_data.append({
                    "candidate_id": cand.candidate_id,
                    **judge_result.dimensions,
                    "scalar_score": judge_result.scalar_score,
                })

            # Pick winner
            tier1: list[dict[str, Any]] = []
            summary = {
                "baseline_score": 0.0, "best_score": 0.0, "improvement_pct": 0.0,
            }
            if scored:
                scored.sort(key=lambda s: s["judge_result"].scalar_score, reverse=True)
                winner = scored[0]
                baseline = next(
                    (s for s in scored if s["candidate"].generation_round == 0),
                    winner,
                )
                best = winner["judge_result"].scalar_score
                base = baseline["judge_result"].scalar_score
                improvement = ((best - base) / max(base, 1e-9)) * 100 if base > 0 else 0.0
                summary = {
                    "baseline_score": base,
                    "best_score": best,
                    "improvement_pct": improvement,
                }
                if winner["candidate"].candidate_id != baseline["candidate"].candidate_id:
                    tier1.append({
                        "tier": 1,
                        "axis": "prompt",
                        "title": "Rewrite system prompt",
                        "description": winner["candidate"].system_prompt,
                        "confidence": "H" if improvement > 10 else "M",
                        "estimated_impact": {"judge": best - base},
                        "evidence_trace_ids": [t.get("id", "") for t in traces[:3]],
                        "apply_payload": {"system_prompt": winner["candidate"].system_prompt},
                    })

            # Render + save
            report = self._writer.render(
                run_id=run_id,
                understanding="",
                summary=summary,
                tier1=tier1, tier2=[], tier3=[],
                pareto_data=pareto_data,
            )
            db.save_opt_report(
                run_id=run_id,
                understanding=report.understanding,
                summary=report.summary,
                tier1=[s.model_dump() for s in report.tier1],
                tier2=[s.model_dump() for s in report.tier2],
                tier3=[s.model_dump() for s in report.tier3],
                pareto_data=report.pareto_data,
                markdown=report.markdown,
            )
            db.update_opt_run_status(run_id, "completed")

        except Exception as e:
            logger.exception("Orchestrator run failed: %s", e)
            db.update_opt_run_status(run_id, "failed", failure_reason=str(e))
            try:
                if row is not None:
                    db.decrement_opt_runs_used(row["org_id"])
            except Exception:
                pass
```

- [ ] **Step 11.3: Run + commit**

Run new tests — 2 pass. Phase 1's `test_orchestrator_noop.py` will likely FAIL because the orchestrator no longer uses the noop path — update the Phase 1 noop tests to match the new behavior, OR delete them (they're superseded). Recommendation: **delete `retune-cloud/tests/test_orchestrator_noop.py`** — Phase 2 tests replace it.

Run full suite — 140 pass (138 prior + 2 new − 3 removed noop tests = 140).

```bash
cd retune-cloud
git add server/optimizer/orchestrator.py tests/test_orchestrator_real.py
git rm tests/test_orchestrator_noop.py
git commit -m "optimizer: real Orchestrator — dispatches PromptOptimizer, writes Tier 1"
```

---

## Task 12: Cloud — promote deepagents to required dep

**Files:**
- Modify: `retune-cloud/pyproject.toml`

- [ ] **Step 12.1: Check current dep status**

Open `retune-cloud/pyproject.toml`. If `deepagents` is listed as optional-deps or not at all, move it into `dependencies`.

- [ ] **Step 12.2: Modify dependencies**

Example (actual edit depends on current file contents — adjust accordingly):

Before:
```toml
[project.optional-dependencies]
deep = ["deepagents>=0.2"]
```

After:
```toml
[project]
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    ...
    "deepagents>=0.2",          # required for PromptOptimizerAgent (Phase 2)
    "langchain-core>=0.2",
    "langchain-openai>=0.1",    # rewriter LLM factory
    "langchain-anthropic>=0.1",
    "langchain-google-genai>=1.0",
]
```

**Note:** only `retune-cloud/pyproject.toml` — do NOT promote these in the outer public SDK's pyproject.toml. Cloud needs them; the SDK does not.

- [ ] **Step 12.3: Install locally + verify**

```bash
cd retune-cloud
pip install -e .
python -c "import deepagents; from langchain_openai import ChatOpenAI; print('ok')"
```

Expected: `ok`.

- [ ] **Step 12.4: Commit**

```bash
cd retune-cloud
git add pyproject.toml
git commit -m "deploy: promote deepagents + langchain LLM clients to required deps"
```

---

## Task 13: End-to-end integration test + Phase 2 exit verification

**Files:**
- Create: `tests/integration/test_optimize_prompt_e2e.py`

Full-stack test: SDK calls `optimize(source="last_n_traces", n=5, axes=["prompt"])` with a mocked local storage that returns 5 traces. TestClient receives the preauth, stores the traces via fake_db, real Orchestrator runs, dispatches candidates, SDK's mocked candidate runner executes each (producing consistent eval scores), Orchestrator scores via real JudgeAgent, writes Tier 1, SDK fetches report and verifies it has a Tier 1 suggestion.

- [ ] **Step 13.1: Write the integration test**

```python
# tests/integration/test_optimize_prompt_e2e.py
"""Phase 2 E2E: SDK uploads traces → real Orchestrator dispatches prompt
candidates → SDK executes with config_overrides → Judge scores →
Report has a Tier 1 suggestion."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.optimizer.models import PromptCandidate


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
    """Full flow: upload traces → dispatch candidates → score → Tier 1."""
    from server.optimizer.job_queue import get_queue, get_results
    get_queue().reset()
    get_results().reset()

    # Seed traces
    traces = [
        {"id": f"t{i}", "query": f"q{i}", "response": f"r{i}",
         "config_snapshot": {"system_prompt": "You are helpful."}}
        for i in range(5)
    ]
    run_id_ref = {"id": None}

    # Mock the PromptOptimizerAgent to return a deterministic candidate set
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
         patch("server.optimizer.orchestrator.PromptOptimizerAgent") as mock_prompt_cls:

        mock_prompt = MagicMock()
        mock_prompt.generate_candidates.return_value = [baseline_cand, rewrite_cand]
        mock_prompt_cls.return_value = mock_prompt

        client = TestClient(app)
        auth = {"Authorization": "Bearer test"}

        # Step 1: preauthorize with traces
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
        run_id_ref["id"] = run_id

        # TestClient runs background tasks, so the orchestrator has been
        # invoked. It pushes 2 candidates to the queue.
        # Simulate the SDK worker posting results for both candidates.
        for cand_id, judge_score in [("cand_base", 5.0), ("cand_rewrite", 8.0)]:
            r = client.post(
                "/api/v1/jobs/result",
                json={
                    "run_id": run_id, "candidate_id": cand_id,
                    "trace": {"query": "q", "response": "r"},
                    "eval_scores": {"llm_judge": judge_score},
                },
                headers=auth,
            )
            assert r.status_code == 200

        # The orchestrator was dispatched in a background task BEFORE we
        # posted results. It's blocking waiting for results.get(...) in a
        # loop. The results are now there, so the next iteration of its
        # wait loop will find them and proceed. But TestClient's background
        # tasks run after the response completes and are synchronous to the
        # route — so the orchestrator is still blocked when preauthorize's
        # background tasks started. This is the timing challenge for E2E.

        # For the E2E in Phase 2, the orchestrator's blocking wait should
        # be short enough that by the time we post results (immediately
        # after preauth), they're present. Orchestrator's candidate timeout
        # per-round of 0.25s should pick them up quickly.

        # Alternatively: manually invoke the orchestrator.run() after results
        # are posted, instead of relying on background-task timing.

        # --- Wait for orchestrator to complete (poll DB) ---
        import time
        for _ in range(30):
            if run_id in fake_db_phase2.reports:
                break
            time.sleep(0.5)

        # Step 3: fetch report
        r = client.get(f"/api/v1/optimize/{run_id}/report", headers=auth)
        assert r.status_code == 200
        body = r.json()
        # Tier 1 should have a suggestion for the winning rewrite
        assert len(body["tier1"]) >= 1
        assert any("precise helper" in (s.get("description") or "") for s in body["tier1"])
```

**Honest note:** this E2E test has a timing coupling — the orchestrator's background task runs in parallel with our result-posting. The simplest fix is to **call the orchestrator explicitly after posting results** rather than relying on `BackgroundTasks` execution order. If the test is flaky, refactor the orchestrator invocation in the preauth route to optionally be scheduled-for-later (a future `POST /start` endpoint) so the test can drive it.

**Simpler E2E alternative:** mock the orchestrator's background task scheduling, drive everything sequentially (preauth → results posted → orchestrator run synchronously → report fetched). This is what the spec reviewer will likely ask for anyway.

- [ ] **Step 13.2: Run full suite + lint**

```bash
pytest tests/ retune-cloud/tests/ -q
```
Expected: all tests pass including the new E2E.

```bash
ruff check src/ tests/ retune-cloud/server/
```
Expected: no new errors.

```bash
mypy src/retune/ --ignore-missing-imports
```
Expected: no new errors.

- [ ] **Step 13.3: Commit**

```bash
git add tests/integration/test_optimize_prompt_e2e.py
git commit -m "optimizer: Phase 2 end-to-end prompt optimization test"
```

---

## Phase 2 Exit Gate

All of the following must be green before moving to Phase 3:

- [ ] `pytest tests/ retune-cloud/tests/ -q` — all tests pass (Phase 1 + Phase 2 additions − removed noop orchestrator tests)
- [ ] `ruff check src/ tests/ retune-cloud/server/` — no new errors
- [ ] `mypy src/retune/ --ignore-missing-imports` — no new errors introduced
- [ ] **Manual smoke test** — against a local Postgres with the schema applied and a real LLM API key:

```python
from retune import Retuner, Mode
retuner = Retuner(
    agent=my_real_agent, adapter="custom", mode=Mode.OBSERVE,
    api_key="<your-key>",
    agent_purpose="test bot",
)
# Run the agent 10 times to seed traces
for q in sample_queries:
    retuner.run(q)

retuner.set_mode(Mode.IMPROVE)
report = retuner.optimize(
    source="last_n_traces", n=10, axes=["prompt"],
    rewriter_llm="gpt-4o-mini",
)
report.show()
```

- [ ] The report's Tier 1 has at least one suggestion that's a genuinely rewritten prompt (not a copy of the baseline)
- [ ] Running the agent with `report.apply(tier=1)` applied shows the new prompt in effect

Once green, start **Phase 3 (ToolOptimizer)**.

---

## Known limitations for Phase 2

These are intentional scope-cuts that become Phase 3+ work:

1. **Single-round beam search only.** Phase 2's `BeamSearchAPO.generate_candidates` runs one critique→rewrite cycle. The original `premium/` version does multi-round with scored feedback; that becomes Phase 2.1 if needed before Phase 3.
2. **Orchestrator only dispatches PromptOptimizer.** The multi-axis routing logic (Phase 3+) is not written yet — Orchestrator hard-codes PromptOptimizerAgent.
3. **SDK candidate runner produces stub eval scores.** `_make_candidate_runner` returns `{"llm_judge": 0.0, "cost": 0.0, "latency": 0.0}` — real evaluation needs the SDK-side evaluators (already open-source in `src/retune/evaluators/`) to actually run against the candidate's response. That wiring is Phase 2.1.
4. **Few-shot curation deferred.** Spec §5 says PromptOptimizerAgent "curates few-shot examples from high-scoring traces." Phase 2 only does system-prompt rewrites; few-shot curation is a Phase 2.1 extension.
5. **deepagents framework barely used.** Phase 2 uses direct LLM calls inside `BeamSearchAPO`. The TodoListMiddleware / SubAgentMiddleware scaffolding comes in Phase 3 when ToolOptimizer's multi-step delegation actually benefits from it.
