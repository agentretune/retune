# Optimizer Phase 3 (ToolOptimizerAgent + Orchestrator Routing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second subagent (ToolOptimizerAgent) that analyzes tool usage patterns from traces + introspected tool metadata, proposes changes (drop unused, rewrite descriptions, tighten schemas, suggest new tools). The Orchestrator learns to route: it dispatches both PromptOptimizer and ToolOptimizer when relevant metadata is present.

**Architecture:** ToolOptimizer differs from PromptOptimizer in kind — it doesn't need beam search or multi-round iteration. Most of its work is deterministic trace analysis (call counts, success rates). The LLM is only invoked for description rewrites (single shot per tool). This makes Phase 3 simpler than Phase 2 despite adding a second subagent.

SDK gains a new **tool introspection** step at optimize-time: it reads the wrapped adapter's tool list + descriptions + arg schemas and uploads them alongside the trace bundle. Cloud stores this metadata 1:1 with the run, ToolOptimizerAgent reads it together with traces, produces suggestions, Orchestrator merges them with PromptOptimizer output into the report.

**Tech Stack:** No new dependencies. Reuses cloud LLM factory from Phase 2 T7 for description rewrites.

**Reference spec:** `docs/superpowers/specs/2026-04-12-optimizer-design.md` §5 (ToolOptimizerAgent), §11 Phase 3 row. §13 open items: "Tool discovery = introspection of the wrapped agent at build time; never parse source files."

**Baseline:** Can branch from either Phase 2 (`phase2/prompt-optimizer`, 139 tests) OR Phase 2.1 (`phase2.1/multi-round`, ~159 tests). Phase 3 doesn't depend on Phase 2.1's beam/eval changes, so either base works. If Phase 2.1 is already merged, branch from that tip.

---

## Scope

### In

- **SDK tool introspection** — a helper that reads `agent.tools`, LangChain `bind_tools` lists, LangGraph tool nodes, or `BaseTool` instances from the adapter and returns a `list[ToolMetadata]`.
- **Upload path** — `Retuner.optimize(...)` sends tool_metadata alongside traces at preauthorize time.
- **Cloud storage** — `optimization_run_tools` JSONB table (1:1 with run, auto-purged via CASCADE).
- **ToolOptimizerAgent** with four analyzers:
  1. `DropUnusedAnalyzer` — Tier 1 suggestion if a tool was never called across traces
  2. `DescriptionRewriter` — Tier 2 suggestion with rewritten description (LLM-backed) if a tool has high failure rate or is never picked
  3. `SchemaTightener` — Tier 2 suggestion with tighter arg schema if tool calls consistently omit certain args
  4. `NewToolProposer` — Tier 3 conceptual suggestion if traces show repeated patterns better served by a new tool (heuristic-based for Phase 3, not LLM-heavy)
- **Orchestrator routing** — inspects available metadata at run start: dispatches PromptOptimizer if `axes` includes "prompt", dispatches ToolOptimizer if `axes` includes "tools" AND tool_metadata is present.
- **Trace-based fallback** — if tool_metadata is empty/missing (adapter can't introspect), ToolOptimizer runs against `ExecutionTrace.steps[]` call-name analysis only, producing Tier 2/3 suggestions (no Tier 1 drops since we can't know the full tool set).
- **Combined report** — tier1/2/3 lists from both subagents are merged, then rendered by ReportWriterAgent.
- **SDK apply path** — `OptimizationReport.apply(tier=1)` handles `drop_tool` apply_payloads by mutating the wrapper/adapter's tool list.

### Out (deferred)

- RAGOptimizerAgent (Phase 4)
- Feedback-loop memory (Phase 5)
- Cloud-hosted LLM judge (stays BYO-key on SDK)
- A/B testing of proposed schemas in prod (Phase 5)

---

## File Structure

### New files — SDK side

| File | Responsibility |
|---|---|
| `src/retune/optimizer/tool_introspection.py` | `introspect_tools(adapter) -> list[ToolMetadata]` — handles LangChain `bind_tools`, LangGraph tool nodes, plain `@tool` decorator callables, custom adapter tool lists. Graceful empty list if unknown shape. |
| `src/retune/optimizer/tool_metadata.py` | Pydantic `ToolMetadata` (name, description, args_schema, is_async) — serializable to JSON for upload |

### Modified files — SDK side

| File | Change |
|---|---|
| `src/retune/wrapper.py` | `Retuner.optimize()` calls `introspect_tools(self._adapter)` when `axes` includes "tools"; passes tool_metadata to `client.preauthorize(...)` |
| `src/retune/optimizer/client.py` | `OptimizerClient.preauthorize(..., tool_metadata=...)` accepts optional list |
| `src/retune/optimizer/report.py` | `OptimizationReport.apply(tier=1)` invokes an optional `tool_apply_fn` callback that the Retuner supplies, which mutates the adapter's tool list when suggestions have `apply_payload.action == "drop_tool"` |

### New files — cloud side

| File | Responsibility |
|---|---|
| `retune-cloud/server/optimizer/tool_optimizer/__init__.py` | Package init |
| `retune-cloud/server/optimizer/tool_optimizer/analyzer.py` | `ToolUsageAnalyzer.analyze(traces, tool_metadata) -> ToolUsageReport` — call counts, failure rates, arg patterns per tool |
| `retune-cloud/server/optimizer/tool_optimizer/agent.py` | `ToolOptimizerAgent` — composes the four analyzers + a rewrite LLM, emits suggestions |
| `retune-cloud/server/optimizer/tool_optimizer/prompts.py` | LLM prompts for description rewrites |

### Modified files — cloud side

| File | Change |
|---|---|
| `retune-cloud/server/db/schema.sql` | Add `optimization_run_tools` table (1:1 with run, JSONB, CASCADE) |
| `retune-cloud/server/db/postgres.py` | Add `save_opt_run_tools(run_id, tools)` + `get_opt_run_tools(run_id) -> list[dict]` |
| `retune-cloud/server/routes/optimize.py` | `PreauthorizeRequest` accepts optional `tool_metadata`; route stores via `save_opt_run_tools` |
| `retune-cloud/server/optimizer/models.py` | Add `ToolMetadata`, `ToolUsageReport` classes |
| `retune-cloud/server/optimizer/orchestrator.py` | Routing: dispatches PromptOptim if "prompt" in axes; dispatches ToolOptim if "tools" in axes AND metadata present OR traces have step data. Merges tier1/2/3 lists from both. |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_tool_introspection.py` | LangChain / LangGraph / callable-list / unknown-shape introspection |
| `tests/unit/test_tool_metadata_upload.py` | Retuner.optimize passes tool_metadata to preauthorize |
| `tests/unit/test_report_apply_tool_drop.py` | `OptimizationReport.apply(tier=1)` handles drop_tool action |
| `retune-cloud/tests/test_opt_run_tools_db.py` | DB helpers for tool metadata |
| `retune-cloud/tests/test_preauth_tool_metadata.py` | Route accepts + stores tool_metadata |
| `retune-cloud/tests/test_tool_usage_analyzer.py` | Call counts, failure rates, arg-pattern detection |
| `retune-cloud/tests/test_tool_optimizer_agent.py` | All four analyzer outputs merged into suggestions |
| `retune-cloud/tests/test_orchestrator_tool_routing.py` | Orchestrator dispatches both subagents when axes includes both |
| `tests/integration/test_optimize_tool_e2e.py` | Full E2E: traces + tool_metadata → routing → combined tier1/2/3 |

---

## Task Summary

Ten tasks.

1. SDK: `ToolMetadata` model + `introspect_tools` helper
2. SDK: wire introspection into `Retuner.optimize` + `OptimizerClient.preauthorize`
3. Cloud: `optimization_run_tools` table + 2 DB helpers
4. Cloud: preauthorize accepts tool_metadata
5. Cloud: `ToolUsageAnalyzer` (deterministic trace analysis)
6. Cloud: `ToolOptimizerAgent` with 4 analyzer composition + LLM description rewriter
7. Cloud: Orchestrator routing + combined report
8. SDK: `OptimizationReport.apply` handles `drop_tool` action
9. E2E integration test
10. Phase 3 exit gate verification

---

## Task 1: SDK `ToolMetadata` + `introspect_tools`

**Files:**
- Create: `src/retune/optimizer/tool_metadata.py`
- Create: `src/retune/optimizer/tool_introspection.py`
- Test: `tests/unit/test_tool_introspection.py`

- [ ] **Step 1.1: Failing tests**

```python
# tests/unit/test_tool_introspection.py
"""Tool introspection — reads tool metadata from various adapter shapes."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.tool_introspection import introspect_tools


def test_adapter_with_tools_attribute_list_of_dicts():
    adapter = MagicMock()
    adapter.tools = [
        {"name": "search", "description": "Search the web", "args_schema": {}},
        {"name": "calc", "description": "Do math", "args_schema": {}},
    ]
    tools = introspect_tools(adapter)
    assert len(tools) == 2
    assert tools[0].name == "search"
    assert tools[0].description == "Search the web"


def test_langchain_basetools():
    """LangChain BaseTool objects expose .name, .description, .args_schema."""
    class FakeTool:
        name = "lc_tool"
        description = "A langchain tool"
        args_schema = None
    adapter = MagicMock()
    adapter.tools = [FakeTool()]
    tools = introspect_tools(adapter)
    assert tools[0].name == "lc_tool"


def test_adapter_without_tools_returns_empty():
    adapter = object()   # no .tools attribute at all
    assert introspect_tools(adapter) == []


def test_adapter_with_empty_tools():
    adapter = MagicMock()
    adapter.tools = []
    assert introspect_tools(adapter) == []


def test_adapter_none_returns_empty():
    assert introspect_tools(None) == []
```

Run — FAIL.

- [ ] **Step 1.2: Implement models + helper**

```python
# src/retune/optimizer/tool_metadata.py
"""Serializable tool-metadata envelope for SDK→cloud upload."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolMetadata(BaseModel):
    """One wrapped agent's tool, in serializable form."""
    name: str
    description: str = ""
    args_schema: dict[str, Any] = Field(default_factory=dict)
    is_async: bool = False
```

```python
# src/retune/optimizer/tool_introspection.py
"""Extract tool metadata from an adapter at optimize-time.

Handles:
- Plain list of dicts: [{"name": ..., "description": ..., "args_schema": ...}]
- LangChain BaseTool instances (duck-typed: .name, .description, .args_schema)
- LangGraph tool nodes
- Adapters without a .tools attribute → empty list (graceful, no error)

Never parses user source files — purely in-memory introspection.
"""
from __future__ import annotations

import logging
from typing import Any

from retune.optimizer.tool_metadata import ToolMetadata

logger = logging.getLogger(__name__)


def _extract_one(tool_obj: Any) -> ToolMetadata | None:
    """Best-effort extraction from a single tool object / dict."""
    try:
        if isinstance(tool_obj, dict):
            return ToolMetadata(
                name=str(tool_obj.get("name", "")),
                description=str(tool_obj.get("description", "")),
                args_schema=dict(tool_obj.get("args_schema") or {}),
                is_async=bool(tool_obj.get("is_async", False)),
            )
        # Duck-typed: name + description required
        name = getattr(tool_obj, "name", None)
        if not name:
            return None
        description = str(getattr(tool_obj, "description", "") or "")
        args_schema_raw = getattr(tool_obj, "args_schema", None) or {}
        if hasattr(args_schema_raw, "model_json_schema"):
            # Pydantic model
            args_schema = args_schema_raw.model_json_schema()
        elif isinstance(args_schema_raw, dict):
            args_schema = args_schema_raw
        else:
            args_schema = {}
        is_async = getattr(tool_obj, "is_async", False) or getattr(tool_obj, "coroutine", False) is not None
        return ToolMetadata(
            name=str(name),
            description=description,
            args_schema=args_schema,
            is_async=bool(is_async),
        )
    except Exception as e:
        logger.debug("Tool extraction failed for %r: %s", tool_obj, e)
        return None


def introspect_tools(adapter: Any) -> list[ToolMetadata]:
    """Return tool metadata from the adapter. Empty list if unknown shape."""
    if adapter is None:
        return []
    tools_raw = getattr(adapter, "tools", None)
    if not tools_raw:
        return []
    out: list[ToolMetadata] = []
    try:
        iterator = list(tools_raw)
    except TypeError:
        return []
    for t in iterator:
        md = _extract_one(t)
        if md is not None:
            out.append(md)
    return out
```

- [ ] **Step 1.3: Run + commit (outer repo)**

Tests 5 pass.

```bash
git add src/retune/optimizer/tool_metadata.py src/retune/optimizer/tool_introspection.py tests/unit/test_tool_introspection.py
git commit -m "optimizer: SDK tool introspection for adapter metadata"
```

---

## Task 2: SDK wires introspection into optimize

**Files:**
- Modify: `src/retune/optimizer/client.py` (preauthorize signature)
- Modify: `src/retune/wrapper.py` (optimize method)
- Test: append to `tests/unit/test_optimizer_client.py` + create `tests/unit/test_tool_metadata_upload.py`

- [ ] **Step 2.1: Extend OptimizerClient.preauthorize**

Add parameter:

```python
    def preauthorize(
        self,
        source: str,
        n_traces: int,
        axes: list[str],
        reward_spec: dict[str, Any] | None = None,
        rewriter_llm: str | None = None,
        traces: list[dict[str, Any]] | None = None,
        tool_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body = {
            "source": source, "n_traces": n_traces, "axes": axes,
            "reward_spec": reward_spec, "rewriter_llm": rewriter_llm,
        }
        if traces is not None:
            body["traces"] = traces
        if tool_metadata is not None:
            body["tool_metadata"] = tool_metadata
        return self._post("/api/v1/optimize/preauthorize", body)
```

- [ ] **Step 2.2: Retuner.optimize collects tool_metadata**

In `src/retune/wrapper.py`'s `optimize` method, after the traces collection block, add:

```python
    tool_metadata_payload = None
    if "tools" in axes_list:
        try:
            from retune.optimizer.tool_introspection import introspect_tools
            tool_metadata_payload = [
                md.model_dump() for md in introspect_tools(self._adapter)
            ]
        except Exception as e:
            logger.warning("Tool introspection failed: %s", e)
            tool_metadata_payload = []

    resp = client.preauthorize(
        source=source, n_traces=n, axes=axes_list,
        reward_spec=reward_spec, rewriter_llm=rewriter_llm,
        traces=traces_payload,
        tool_metadata=tool_metadata_payload,
    )
```

- [ ] **Step 2.3: Test — Retuner uploads tool_metadata**

```python
# tests/unit/test_tool_metadata_upload.py
"""Retuner.optimize introspects tools and sends them to cloud."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from retune import Retuner, Mode


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
```

Also add `introspect_tools` to wrapper.py's top-level imports (so the `retune.wrapper.introspect_tools` patch target resolves):

```python
from retune.optimizer.tool_introspection import introspect_tools  # noqa: F401
```

- [ ] **Step 2.4: Run + commit**

Tests pass.

```bash
git add src/retune/optimizer/client.py src/retune/wrapper.py tests/unit/test_optimizer_client.py tests/unit/test_tool_metadata_upload.py
git commit -m "optimizer: SDK uploads tool_metadata when tools axis requested"
```

---

## Task 3: Cloud `optimization_run_tools` table + DB helpers

**Files:**
- Modify: `retune-cloud/server/db/schema.sql`
- Modify: `retune-cloud/server/db/postgres.py`
- Test: `retune-cloud/tests/test_opt_run_tools_db.py`

- [ ] **Step 3.1: Append schema**

```sql
-- ============ Phase 3: Per-run tool metadata (auto-purged) ============

CREATE TABLE IF NOT EXISTS optimization_run_tools (
    run_id VARCHAR(64) PRIMARY KEY REFERENCES optimization_runs(id) ON DELETE CASCADE,
    tools JSONB NOT NULL DEFAULT '[]',
    tool_count INTEGER NOT NULL DEFAULT 0,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 3.2: Add DB helpers**

Append to `postgres.py`:

```python
def save_opt_run_tools(run_id: str, tools: list[dict]) -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO optimization_run_tools (run_id, tools, tool_count)
                VALUES (%s, %s::jsonb, %s)
                ON CONFLICT (run_id) DO UPDATE
                  SET tools = EXCLUDED.tools,
                      tool_count = EXCLUDED.tool_count,
                      uploaded_at = NOW()
                """,
                (run_id, _json.dumps(tools), len(tools)),
            )
        conn.commit()
    finally:
        put_conn(conn)


def get_opt_run_tools(run_id: str) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tools FROM optimization_run_tools WHERE run_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
            if row is None:
                return []
            return list(row[0] or [])
    finally:
        put_conn(conn)
```

- [ ] **Step 3.3: Test (same pattern as T1 of Phase 2)**

Quickly — 2 tests (save, get empty). Follow the Phase 2 T1 test style.

- [ ] **Step 3.4: Commit**

```bash
cd retune-cloud
git add server/db/schema.sql server/db/postgres.py tests/test_opt_run_tools_db.py
git commit -m "optimizer: schema + helpers for per-run tool metadata"
```

---

## Task 4: Preauthorize accepts tool_metadata

**Files:**
- Modify: `retune-cloud/server/routes/optimize.py`
- Test: `retune-cloud/tests/test_preauth_tool_metadata.py`

- [ ] **Step 4.1: Extend PreauthorizeRequest**

Add field:

```python
    tool_metadata: list[dict[str, Any]] | None = None
```

- [ ] **Step 4.2: In `preauthorize` route, store the metadata**

After `db.save_opt_run_traces(...)` block, add:

```python
    if req.tool_metadata:
        db.save_opt_run_tools(run_id, req.tool_metadata)
```

- [ ] **Step 4.3: Test — 1 test pattern from Phase 2 T3**

Assert `save_opt_run_tools` called when present, NOT called when absent.

- [ ] **Step 4.4: Commit**

```bash
cd retune-cloud
git add server/routes/optimize.py tests/test_preauth_tool_metadata.py
git commit -m "optimizer: preauthorize accepts + stores tool_metadata"
```

---

## Task 5: Cloud ToolUsageAnalyzer (deterministic)

**Files:**
- Create: `retune-cloud/server/optimizer/tool_optimizer/__init__.py`
- Create: `retune-cloud/server/optimizer/tool_optimizer/analyzer.py`
- Test: `retune-cloud/tests/test_tool_usage_analyzer.py`

`ToolUsageAnalyzer.analyze(traces, tool_metadata) -> ToolUsageReport`:
- Per tool: call_count, success_count, failure_count, avg_latency_ms, common_missing_args (from step error messages), example_args (from step inputs).
- Identify unused tools: tools in metadata that never appear in any trace's steps.
- Identify over-triggered tools: tools called significantly more than the median.

- [ ] **Step 5.1: Add `ToolUsageReport` to `models.py`**

```python
class ToolUsagePerTool(BaseModel):
    name: str
    description: str
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    never_called: bool = False
    common_missing_args: list[str] = Field(default_factory=list)
    example_successful_args: list[dict[str, Any]] = Field(default_factory=list)


class ToolUsageReport(BaseModel):
    per_tool: list[ToolUsagePerTool] = Field(default_factory=list)
    total_tool_calls_observed: int = 0
    tools_in_metadata_count: int = 0
    unknown_tool_calls: list[str] = Field(default_factory=list)   # called but not in metadata
```

- [ ] **Step 5.2: Implement analyzer**

```python
# retune-cloud/server/optimizer/tool_optimizer/analyzer.py
"""Deterministic trace analysis for tool usage patterns."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from server.optimizer.models import ToolUsageReport, ToolUsagePerTool


class ToolUsageAnalyzer:
    def analyze(
        self,
        traces: list[dict[str, Any]],
        tool_metadata: list[dict[str, Any]],
    ) -> ToolUsageReport:
        metadata_by_name = {t["name"]: t for t in tool_metadata}
        stats = defaultdict(lambda: {
            "call_count": 0, "success_count": 0, "failure_count": 0,
            "total_latency_ms": 0.0, "missing_arg_keys": defaultdict(int),
            "example_args": [],
        })
        unknown_calls: list[str] = []
        total_calls = 0

        for t in traces:
            for step in t.get("steps", []) or []:
                if step.get("type") != "tool_call" and step.get("step_type") != "tool":
                    continue
                tool_name = step.get("name") or step.get("tool_name") or ""
                if not tool_name:
                    continue
                total_calls += 1
                if tool_name not in metadata_by_name and tool_name not in unknown_calls:
                    unknown_calls.append(tool_name)
                s = stats[tool_name]
                s["call_count"] += 1
                if step.get("error") or step.get("status") == "error":
                    s["failure_count"] += 1
                    # Try to extract missing-arg keys from the error message
                    err = str(step.get("error", ""))
                    for token in err.split():
                        if token.startswith("'") and token.endswith("'"):
                            key = token.strip("'")
                            s["missing_arg_keys"][key] += 1
                else:
                    s["success_count"] += 1
                    if len(s["example_args"]) < 3:
                        s["example_args"].append(step.get("args", {}) or {})
                s["total_latency_ms"] += float(step.get("duration_ms", 0.0))

        per_tool: list[ToolUsagePerTool] = []
        for name, meta in metadata_by_name.items():
            s = stats.get(name, None)
            if s is None:
                # Never called
                per_tool.append(ToolUsagePerTool(
                    name=name,
                    description=meta.get("description", ""),
                    never_called=True,
                ))
                continue
            avg_latency = (s["total_latency_ms"] / s["call_count"]) if s["call_count"] else 0.0
            missing = sorted(s["missing_arg_keys"].items(), key=lambda kv: -kv[1])
            per_tool.append(ToolUsagePerTool(
                name=name,
                description=meta.get("description", ""),
                call_count=s["call_count"],
                success_count=s["success_count"],
                failure_count=s["failure_count"],
                avg_latency_ms=avg_latency,
                never_called=False,
                common_missing_args=[k for k, _c in missing[:3]],
                example_successful_args=s["example_args"][:3],
            ))

        return ToolUsageReport(
            per_tool=per_tool,
            total_tool_calls_observed=total_calls,
            tools_in_metadata_count=len(metadata_by_name),
            unknown_tool_calls=unknown_calls,
        )
```

- [ ] **Step 5.3: 4 tests covering: unused tool detection, success/failure counts, missing-arg extraction, unknown-tool-calls when metadata is empty**

```python
# retune-cloud/tests/test_tool_usage_analyzer.py (skeleton)
from server.optimizer.tool_optimizer.analyzer import ToolUsageAnalyzer


def test_unused_tool_marked_never_called():
    traces = []
    metadata = [{"name": "search", "description": "Search"}]
    report = ToolUsageAnalyzer().analyze(traces, metadata)
    assert len(report.per_tool) == 1
    assert report.per_tool[0].never_called is True


def test_call_counts_aggregated():
    traces = [
        {"steps": [
            {"type": "tool_call", "name": "search", "args": {"q": "x"}, "duration_ms": 50},
            {"type": "tool_call", "name": "search", "args": {"q": "y"}, "duration_ms": 60},
        ]}
    ]
    metadata = [{"name": "search", "description": "Search"}]
    report = ToolUsageAnalyzer().analyze(traces, metadata)
    pt = report.per_tool[0]
    assert pt.call_count == 2
    assert pt.success_count == 2
    assert pt.never_called is False


def test_failure_missing_arg_extraction():
    traces = [
        {"steps": [
            {"type": "tool_call", "name": "lookup",
             "status": "error", "error": "missing required key 'customer_id'"},
        ]}
    ]
    metadata = [{"name": "lookup", "description": "Lookup"}]
    report = ToolUsageAnalyzer().analyze(traces, metadata)
    pt = report.per_tool[0]
    assert pt.failure_count == 1
    assert "customer_id" in pt.common_missing_args


def test_unknown_tool_calls_captured():
    traces = [{"steps": [{"type": "tool_call", "name": "ghost", "args": {}}]}]
    report = ToolUsageAnalyzer().analyze(traces, tool_metadata=[])
    assert "ghost" in report.unknown_tool_calls
```

- [ ] **Step 5.4: Commit**

```bash
cd retune-cloud
git add server/optimizer/tool_optimizer/__init__.py server/optimizer/tool_optimizer/analyzer.py server/optimizer/models.py tests/test_tool_usage_analyzer.py
git commit -m "optimizer: ToolUsageAnalyzer — per-tool stats from traces"
```

---

## Task 6: Cloud `ToolOptimizerAgent`

**Files:**
- Create: `retune-cloud/server/optimizer/tool_optimizer/prompts.py`
- Create: `retune-cloud/server/optimizer/tool_optimizer/agent.py`
- Test: `retune-cloud/tests/test_tool_optimizer_agent.py`

Four analyzers compose into suggestions:

| Analyzer | Trigger | Tier | Suggestion shape |
|---|---|---|---|
| DropUnused | tool.never_called AND metadata present | Tier 1 | `apply_payload.action="drop_tool", tool_name=X` |
| DescriptionRewriter | `failure_count / call_count > 0.3` OR `call_count == 0` | Tier 2 | `code_snippet` with suggested `description=...` |
| SchemaTightener | `common_missing_args` non-empty | Tier 2 | `code_snippet` making listed args required |
| NewToolProposer | repeated query patterns in traces not matching any tool | Tier 3 | conceptual suggestion |

- [ ] **Step 6.1: Create `tool_optimizer/prompts.py`**

Short, simple prompts for description rewrites. Example:

```python
DESCRIPTION_REWRITE_PROMPT = """\
You are optimizing tool descriptions for an LLM agent. The current tool has failed to help the agent succeed at its task. Rewrite the description to make it clearer when the agent should use this tool.

Current description:
{description}

Failure pattern:
{failure_context}

Return ONLY the rewritten description, no preamble or quotes."""
```

- [ ] **Step 6.2: Implement agent**

```python
# retune-cloud/server/optimizer/tool_optimizer/agent.py
from __future__ import annotations

import logging
from typing import Any

from server.optimizer.models import (
    ToolUsageReport, ToolUsagePerTool,
)
from server.optimizer.prompt_optimizer.llm import create_rewriter_llm
from server.optimizer.tool_optimizer.analyzer import ToolUsageAnalyzer
from server.optimizer.tool_optimizer.prompts import DESCRIPTION_REWRITE_PROMPT

logger = logging.getLogger(__name__)


class ToolOptimizerAgent:
    def __init__(self, rewriter_llm: str = "claude-3-7-sonnet") -> None:
        self._analyzer = ToolUsageAnalyzer()
        # Description rewriter is optional — if LLM init fails, skip Tier 2 rewrites
        try:
            self._llm: Any = create_rewriter_llm(rewriter_llm)
        except Exception as e:
            logger.warning("Rewriter LLM unavailable: %s; description rewrites disabled", e)
            self._llm = None

    def optimize(
        self,
        traces: list[dict[str, Any]],
        tool_metadata: list[dict[str, Any]],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Return (tier1, tier2, tier3) suggestion lists."""
        report = self._analyzer.analyze(traces, tool_metadata)

        tier1: list[dict] = []
        tier2: list[dict] = []
        tier3: list[dict] = []

        # Drop unused tools
        for pt in report.per_tool:
            if pt.never_called and report.tools_in_metadata_count > 0:
                tier1.append({
                    "tier": 1,
                    "axis": "tools",
                    "title": f"Drop unused tool: {pt.name}",
                    "description": f"`{pt.name}` was never invoked across the evaluated traces.",
                    "confidence": "H",
                    "estimated_impact": {"tool_count_reduction": 1.0},
                    "apply_payload": {"action": "drop_tool", "tool_name": pt.name},
                    "evidence_trace_ids": [],
                })

        # Description rewrites for high-failure tools
        for pt in report.per_tool:
            if pt.call_count > 0 and pt.failure_count / max(pt.call_count, 1) > 0.3:
                new_desc = self._rewrite_description(pt)
                if new_desc:
                    tier2.append({
                        "tier": 2,
                        "axis": "tools",
                        "title": f"Rewrite description for `{pt.name}`",
                        "description": (
                            f"{pt.failure_count}/{pt.call_count} calls failed. "
                            f"Consider this clarified description:"
                        ),
                        "confidence": "M",
                        "estimated_impact": {"tool_failure_rate_reduction": 0.2},
                        "code_snippet": f'description = """{new_desc}"""',
                        "apply_payload": {
                            "action": "rewrite_description",
                            "tool_name": pt.name,
                            "new_description": new_desc,
                        },
                    })

        # Schema tightening
        for pt in report.per_tool:
            if pt.common_missing_args:
                args = ", ".join(f'"{a}"' for a in pt.common_missing_args)
                tier2.append({
                    "tier": 2,
                    "axis": "tools",
                    "title": f"Make {', '.join(pt.common_missing_args)} required on `{pt.name}`",
                    "description": (
                        f"`{pt.name}` failed with missing args: {args}. "
                        "Tightening the schema will produce better LLM-generated calls."
                    ),
                    "confidence": "M",
                    "estimated_impact": {"tool_failure_rate_reduction": 0.1},
                    "code_snippet": f"# In your tool's args_schema, mark these as required:\n# required: [{args}]",
                })

        # Tier 3 conceptual: if many unknown tool calls, agent is reaching for
        # tools it doesn't have
        if len(report.unknown_tool_calls) >= 2:
            tier3.append({
                "tier": 3,
                "axis": "tools",
                "title": "Consider adding missing tools",
                "description": (
                    f"The agent attempted to call unregistered tools: "
                    f"{', '.join(report.unknown_tool_calls[:5])}. "
                    "These are candidates to implement as real tools."
                ),
                "confidence": "M",
                "estimated_impact": {},
            })

        return tier1, tier2, tier3

    def _rewrite_description(self, pt: ToolUsagePerTool) -> str | None:
        if self._llm is None:
            return None
        try:
            msg = DESCRIPTION_REWRITE_PROMPT.format(
                description=pt.description,
                failure_context=(
                    f"{pt.failure_count}/{pt.call_count} calls failed. "
                    f"Common missing args: {', '.join(pt.common_missing_args) or 'none'}."
                ),
            )
            return str(self._llm.invoke(msg).content).strip() or None
        except Exception as e:
            logger.warning("Description rewrite failed for %s: %s", pt.name, e)
            return None
```

- [ ] **Step 6.3: 3 tests — drop unused, description rewrite, schema tighten**

Mock `create_rewriter_llm` to return a MagicMock whose `.invoke(...).content = "New description here"`.

- [ ] **Step 6.4: Commit**

```bash
cd retune-cloud
git add server/optimizer/tool_optimizer/prompts.py server/optimizer/tool_optimizer/agent.py tests/test_tool_optimizer_agent.py
git commit -m "optimizer: ToolOptimizerAgent — drop/rewrite/tighten/propose suggestions"
```

---

## Task 7: Orchestrator routing + combined report

**Files:**
- Modify: `retune-cloud/server/optimizer/orchestrator.py`
- Test: `retune-cloud/tests/test_orchestrator_tool_routing.py`

Routing logic:
- If `"prompt" in row["axes"]`: dispatch PromptOptimizerAgent (as today)
- If `"tools" in row["axes"]`: load tool_metadata via `db.get_opt_run_tools`, dispatch ToolOptimizerAgent
- Combine tier1/2/3 lists from both
- Render with merged suggestions

- [ ] **Step 7.1: Modify orchestrator**

Add near the existing imports:

```python
from server.optimizer.tool_optimizer.agent import ToolOptimizerAgent
```

In `run()`, after the existing prompt-axis block produces its tier1 suggestion, add:

```python
            # Tool axis
            tool_tier1: list[dict] = []
            tool_tier2: list[dict] = []
            tool_tier3: list[dict] = []
            if "tools" in (row.get("axes") or []):
                try:
                    tool_metadata = db.get_opt_run_tools(run_id)
                    tool_agent = ToolOptimizerAgent(
                        rewriter_llm=row.get("rewriter_llm") or "claude-3-7-sonnet",
                    )
                    tool_tier1, tool_tier2, tool_tier3 = tool_agent.optimize(
                        traces=traces, tool_metadata=tool_metadata,
                    )
                except Exception as e:
                    logger.exception("ToolOptimizerAgent failed: %s", e)
```

Before the `self._writer.render(...)` call, merge:

```python
            merged_tier1 = tier1 + tool_tier1
            merged_tier2 = tool_tier2   # Phase 3 has no prompt-axis tier2 yet
            merged_tier3 = tool_tier3   # same

            report = self._writer.render(
                run_id=run_id,
                understanding="",
                summary=summary,
                tier1=merged_tier1,
                tier2=merged_tier2,
                tier3=merged_tier3,
                pareto_data=pareto_data,
            )
```

- [ ] **Step 7.2: Test — orchestrator dispatches both when both axes requested**

```python
# retune-cloud/tests/test_orchestrator_tool_routing.py
# Mock: db.get_opt_run returns row with axes=["prompt", "tools"]
# Mock: PromptOptimizerAgent returns 1 tier1 rewrite
# Mock: ToolOptimizerAgent.optimize returns ([drop_suggestion], [], [])
# Assert: merged tier1 has both suggestions
```

Full example follows Phase 2 T11's test structure.

- [ ] **Step 7.3: Commit**

```bash
cd retune-cloud
git add server/optimizer/orchestrator.py tests/test_orchestrator_tool_routing.py
git commit -m "optimizer: Orchestrator routes to both PromptOptim + ToolOptim"
```

---

## Task 8: SDK handles `drop_tool` apply action

**Files:**
- Modify: `src/retune/optimizer/report.py`
- Modify: `src/retune/wrapper.py` (optional: supply default `tool_apply_fn` that mutates `self._adapter.tools`)
- Test: `tests/unit/test_report_apply_tool_drop.py`

- [ ] **Step 8.1: Extend `OptimizationReport.apply`**

```python
    def apply(
        self,
        tier: int = 1,
        apply_fn: Callable[[Suggestion], None] | None = None,
    ) -> list[Suggestion]:
        items = {1: self.tier1, 2: self.tier2, 3: self.tier3}.get(tier)
        if items is None:
            raise ValueError("tier must be 1, 2, or 3")
        if apply_fn is None:
            return items
        for s in items:
            apply_fn(s)
        return items
```

(No change from Phase 1 — just keep `apply_fn` as the hook.)

In `src/retune/wrapper.py`, add a helper method:

```python
    def apply_report(self, report: "OptimizationReport", tier: int = 1) -> None:
        """Apply tier-N suggestions to the wrapped agent.

        - action="rewrite_prompt" → self._config.system_prompt = ...
        - action="drop_tool"      → remove tool from self._adapter.tools
        - action="rewrite_description" → Tier 2 (printed as instructions, not applied)
        """
        def _apply(s):
            payload = s.apply_payload or {}
            action = payload.get("action")
            if action == "drop_tool":
                tool_name = payload.get("tool_name")
                tools = getattr(self._adapter, "tools", None)
                if tools is None:
                    return
                # Filter in place (support both list[dict] and list[object])
                filtered = []
                for t in tools:
                    name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
                    if name != tool_name:
                        filtered.append(t)
                self._adapter.tools = filtered
            elif "system_prompt" in payload:
                self._config.system_prompt = payload["system_prompt"]
            # Phase 3: no tier-2 auto-apply yet
        report.apply(tier=tier, apply_fn=_apply)
```

- [ ] **Step 8.2: Test**

```python
def test_apply_drops_named_tool():
    # Build a Retuner with an adapter exposing two tools
    # Call apply_report with a Tier 1 suggestion action=drop_tool, tool_name=X
    # Assert adapter.tools no longer contains X
```

- [ ] **Step 8.3: Commit**

```bash
git add src/retune/wrapper.py tests/unit/test_report_apply_tool_drop.py
git commit -m "optimizer: SDK applies drop_tool action from Tier 1 suggestions"
```

---

## Task 9: E2E integration test

**Files:**
- Create: `tests/integration/test_optimize_tool_e2e.py`

Full flow: SDK with 2 tools (one unused) → `optimize(axes=["prompt","tools"])` → cloud routes to both agents → report has Tier 1 drop + Tier 1 prompt rewrite.

Mock `ToolOptimizerAgent.optimize` to return a drop suggestion (avoid needing a real LLM). Use the Phase 2 E2E fixture pattern.

Key assertion: `report.tier1` has 2 suggestions, one with `axis="prompt"` and one with `axis="tools"` (`action="drop_tool"`).

- [ ] **Commit**

```bash
git add tests/integration/test_optimize_tool_e2e.py
git commit -m "optimizer: Phase 3 E2E — combined prompt + tool routing"
```

---

## Task 10: Phase 3 exit gate

- [ ] `pytest tests/ retune-cloud/tests/ -q` — all passing
- [ ] `ruff check` — no new errors
- [ ] `mypy src/retune/ --ignore-missing-imports` — no new errors
- [ ] **Manual smoke test** — against a local Postgres + real LLM API key + a wrapped agent with at least 2 tools (one that never gets called):

```python
from retune import Retuner, Mode
from langchain.tools import tool

@tool
def search(q: str) -> str: ...

@tool
def never_used(x: int) -> int: ...

agent = build_langgraph_agent_with_tools([search, never_used])
retuner = Retuner(
    agent=agent, adapter="langgraph", mode=Mode.OBSERVE,
    api_key="<your-key>",
    agent_purpose="research assistant",
)
for q in sample_queries:
    retuner.run(q)   # calls search, never never_used

retuner.set_mode(Mode.IMPROVE)
report = retuner.optimize(source="last_n_traces", n=20, axes=["prompt", "tools"])
report.show()
```

- [ ] Report Tier 1 has a `drop_tool` suggestion for `never_used`
- [ ] `retuner.apply_report(report, tier=1)` removes `never_used` from `retuner._adapter.tools`

Phase 4 (RAGOptimizer) begins after this.

---

## Known risks / deferred items

1. **Tool introspection across adapters.** LangChain `bind_tools` stores tools on the underlying model, not the adapter — if `adapter.tools` isn't populated by the adapter implementation, introspection returns empty. Workaround: adapter implementers set `self.tools = [...]` in `__init__`. Phase 3 assumes this is either done or Tier 1 drops are simply unavailable.

2. **LLM description rewrite is optional.** If the LLM is unreachable or errors, Tier 2 description rewrites are silently skipped (no exception). The Tier 1 drop suggestions are deterministic and always available.

3. **No tier-2 auto-apply.** Users copy Tier 2 snippets manually. Auto-apply for Tier 2 (e.g., patch the wrapper's tool list with new descriptions) is Phase 5 dashboard work.

4. **`NewToolProposer` is heuristic.** It only fires when the agent calls tool names not in metadata. Real "suggest a new tool" logic (analyzing recurring failure patterns to propose completely novel tools) is Phase 4+.

5. **Trace step schema.** The analyzer expects steps with `type`, `name`, `args`, `error`, `status`, `duration_ms`. The existing `ExecutionTrace.steps[]` format should match — verify during Task 5 implementation.
