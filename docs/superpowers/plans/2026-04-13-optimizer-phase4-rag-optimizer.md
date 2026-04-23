# Optimizer Phase 4 (RAGOptimizerAgent) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third subagent (RAGOptimizerAgent) that proposes retrieval-configuration changes based on retrieval trace analysis. Orchestrator routes to it when `"rag"` is in `axes` AND retrieval-like steps appear in traces. Per the spec, RAG optimization **never modifies user code** — suggestions are Tier 2 (code snippets) or Tier 3 (conceptual), with Tier 1 only available when the adapter exposes runtime override hooks (e.g. LangChain `RunnableConfig`).

**Architecture:** RAGOptimizer differs from both PromptOptim (which uses beam search + LLM rewrites) and ToolOptim (deterministic analysis + one-shot LLM rewrites). RAG optimization is **parameter sweeping** — the axes are numerical/categorical: `retrieval_k`, `chunk_size`, `chunk_overlap`, `reranker_enabled`, `reranker_model`, `query_rewriting_enabled`. The subagent:
1. Analyzes traces for retrieval patterns (how many chunks retrieved, how often retrieval_precision is low, query vs. retrieved-doc similarity)
2. Proposes parameter variants (e.g., "increase k from 5 to 8", "enable hybrid retrieval", "add reranker")
3. The Orchestrator dispatches these as candidate configs via JobQueue — the SDK worker applies the override at candidate execution time (if the adapter exposes override hooks) or produces a baseline-only trace otherwise
4. JudgeAgent scores based on `retrieval_precision`, `groundedness`, `llm_judge` dimensions
5. Winning config → Tier 1 or Tier 2 per the spec's override-availability rule

**Tech Stack:** Reuses PromptOptim's LLM factory for query-rewriter suggestions. No new deps.

**Reference spec:** §5 RAGOptimizerAgent, §11 Phase 4 row, §13 item 3 (never modify user code).

**Baseline:** `phase3/tool-optimizer` (162 tests). Phase 4 branches from here.

---

## Scope

### In

- **SDK-side retrieval-config introspection** — extract `retrieval_k`, `chunk_size`, etc. from the adapter when available; upload as metadata
- **SDK-side adapter override hook** — a new `_apply_retrieval_override(overrides)` helper that sets runtime config on supported adapters (LangChain `RunnableConfig`, retriever kwargs). Tier 1 applicability depends on adapter support.
- **Retrieval trace analysis** (cloud) — `RAGUsageAnalyzer` extracts from `ExecutionTrace.steps[]`:
  - Average # documents retrieved per query
  - Retrieval latency
  - Retrieval-result → final-answer groundedness (heuristic: keyword overlap)
  - Query-to-retrieved-doc similarity (when embeddings available)
- **Candidate proposer** — `RAGCandidateProposer` generates 4-6 parameter-sweep candidates
- **RAGOptimizerAgent** — composes analyzer + proposer, integrates with Orchestrator via the same JobQueue dispatch pattern Phase 2's PromptOptim uses
- **Orchestrator routing** — dispatches RAGOptim when `"rag" in axes` AND (retrieval-config metadata present OR traces show retrieval steps)
- **Tier classification based on adapter capability** — candidates land in Tier 1 if adapter supports runtime override, else Tier 2 (copy-paste snippet) or Tier 3 (conceptual)

### Out (deferred)

- Reindexing (genuinely modifies corpus — Tier 3 only, no auto-apply)
- Hybrid retrieval bootstrap (adds a new retriever — Tier 3 conceptual)
- Reranker model swap for models the user doesn't already have installed (Tier 3)
- Phase 5: dashboard UI, feedback memory, public GA

---

## File Structure

### New files — SDK side

| File | Responsibility |
|---|---|
| `src/retune/optimizer/retrieval_introspection.py` | `introspect_retrieval_config(adapter) -> RetrievalConfig` — reads `retrieval_k`, `chunk_size`, `reranker`, etc. from adapter |
| `src/retune/optimizer/retrieval_config.py` | `RetrievalConfig` Pydantic model |

### Modified — SDK side

| File | Change |
|---|---|
| `src/retune/wrapper.py` | `Retuner.optimize()` includes retrieval_config in preauth payload when `"rag"` in axes. `_make_candidate_runner` applies retrieval-config overrides via adapter hooks. |
| `src/retune/optimizer/client.py` | `preauthorize(..., retrieval_config=...)` param |
| `src/retune/adapters/base.py` | Add optional `apply_retrieval_override(k=..., chunk_size=..., reranker=...)` method with default no-op |
| `src/retune/adapters/langchain_adapter.py` | Override `apply_retrieval_override` to mutate `runnable.with_config({"configurable": {...}})` |
| `src/retune/adapters/langgraph_adapter.py` | Override `apply_retrieval_override` to set runtime state on retriever nodes |

### New files — cloud side

| File | Responsibility |
|---|---|
| `retune-cloud/server/optimizer/rag_optimizer/__init__.py` | Package init |
| `retune-cloud/server/optimizer/rag_optimizer/analyzer.py` | `RAGUsageAnalyzer.analyze(traces) -> RAGUsageReport` — retrieval patterns |
| `retune-cloud/server/optimizer/rag_optimizer/proposer.py` | `RAGCandidateProposer.propose(usage_report, baseline_config) -> list[RAGCandidate]` — parameter sweep |
| `retune-cloud/server/optimizer/rag_optimizer/agent.py` | `RAGOptimizerAgent` — composes analyzer + proposer + scoring via orchestrator callback |

### Modified — cloud side

| File | Change |
|---|---|
| `retune-cloud/server/db/schema.sql` | Add `optimization_run_retrieval_config` table |
| `retune-cloud/server/db/postgres.py` | `save_opt_run_retrieval_config` + `get_opt_run_retrieval_config` helpers |
| `retune-cloud/server/routes/optimize.py` | `PreauthorizeRequest.retrieval_config` field |
| `retune-cloud/server/optimizer/models.py` | `RAGCandidate`, `RAGUsageReport`, `RetrievalStepStats`, `RAGCandidateKind` enum |
| `retune-cloud/server/optimizer/orchestrator.py` | Routing: dispatches RAGOptim when `"rag" in axes` AND (retrieval_config OR retrieval steps) |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_retrieval_introspection.py` | Extract retrieval config from LangChain/LangGraph/custom adapters |
| `tests/unit/test_wrapper_retrieval_upload.py` | Retuner.optimize sends retrieval_config when "rag" in axes |
| `tests/unit/test_adapter_retrieval_override.py` | `apply_retrieval_override` default no-op + LangChain/LangGraph overrides |
| `retune-cloud/tests/test_opt_run_retrieval_config_db.py` | DB helpers |
| `retune-cloud/tests/test_preauth_retrieval_config.py` | Route accepts + stores retrieval_config |
| `retune-cloud/tests/test_rag_usage_analyzer.py` | Trace analysis: retrieval count, latency, groundedness heuristic |
| `retune-cloud/tests/test_rag_candidate_proposer.py` | Parameter sweep: k variants, chunk variants, reranker toggle |
| `retune-cloud/tests/test_rag_optimizer_agent.py` | Full subagent: analyzer + proposer → suggestions |
| `retune-cloud/tests/test_orchestrator_rag_routing.py` | Orchestrator dispatches RAGOptim when criteria met |
| `tests/integration/test_optimize_rag_e2e.py` | E2E: retrieval config upload → analyzer → proposer → Tier 2 suggestions in report |

---

## Task Summary

Eleven tasks.

1. SDK: `RetrievalConfig` model + `introspect_retrieval_config` helper
2. SDK: wire retrieval_config into `Retuner.optimize` + `OptimizerClient.preauthorize`
3. SDK: `Adapter.apply_retrieval_override` default + LangChain/LangGraph overrides
4. Cloud: `optimization_run_retrieval_config` DB
5. Cloud: preauthorize stores retrieval_config
6. Cloud: `RAGUsageAnalyzer` (deterministic)
7. Cloud: `RAGCandidateProposer` (parameter sweep)
8. Cloud: `RAGOptimizerAgent` (composes + integrates with orchestrator score_fn pattern if available, else one-shot score loop)
9. Cloud: Orchestrator routing to RAGOptim
10. E2E integration test
11. Phase 4 exit gate

---

## Task 1: SDK `RetrievalConfig` + introspection

**Files:**
- Create: `src/retune/optimizer/retrieval_config.py`
- Create: `src/retune/optimizer/retrieval_introspection.py`
- Test: `tests/unit/test_retrieval_introspection.py`

- [ ] **Step 1.1: Create model**

```python
# src/retune/optimizer/retrieval_config.py
"""Serializable retrieval-config envelope for SDK→cloud upload."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Current retrieval configuration as introspected from the adapter."""
    retrieval_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    reranker_enabled: bool = False
    reranker_model: str | None = None
    query_rewriting_enabled: bool = False
    retrieval_strategy: str = "dense"  # "dense" | "sparse" | "hybrid"
    embedding_model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 1.2: Implement introspection helper**

```python
# src/retune/optimizer/retrieval_introspection.py
"""Introspect the adapter for retrieval config at optimize-time."""
from __future__ import annotations

import logging
from typing import Any

from retune.optimizer.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)


def introspect_retrieval_config(adapter: Any) -> RetrievalConfig | None:
    """Return the adapter's current retrieval config, or None if not a RAG adapter."""
    if adapter is None:
        return None

    # Common attribute names the SDK adapters may expose
    rc_dict = getattr(adapter, "retrieval_config", None)
    if isinstance(rc_dict, RetrievalConfig):
        return rc_dict
    if isinstance(rc_dict, dict):
        try:
            return RetrievalConfig(**rc_dict)
        except Exception as e:
            logger.debug("Retrieval config parse failed: %s", e)
            return None

    # Fallback: pull individual attrs
    retriever = getattr(adapter, "retriever", None)
    if retriever is None:
        return None
    try:
        return RetrievalConfig(
            retrieval_k=int(getattr(retriever, "search_kwargs", {}).get("k", 5)),
            chunk_size=int(getattr(retriever, "chunk_size", 1000) or 1000),
            chunk_overlap=int(getattr(retriever, "chunk_overlap", 200) or 200),
            reranker_enabled=bool(getattr(adapter, "reranker", None) is not None),
            reranker_model=getattr(getattr(adapter, "reranker", None), "model", None),
            retrieval_strategy=getattr(retriever, "search_type", "dense") or "dense",
            embedding_model=getattr(getattr(retriever, "embeddings", None), "model", None),
        )
    except Exception as e:
        logger.debug("Retrieval introspection failed: %s", e)
        return None
```

- [ ] **Step 1.3: Tests**

```python
# tests/unit/test_retrieval_introspection.py
"""RetrievalConfig introspection from adapter."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.retrieval_introspection import introspect_retrieval_config
from retune.optimizer.retrieval_config import RetrievalConfig


def test_adapter_with_retrieval_config_dict():
    adapter = MagicMock()
    adapter.retrieval_config = {"retrieval_k": 8, "chunk_size": 500}
    config = introspect_retrieval_config(adapter)
    assert config is not None
    assert config.retrieval_k == 8
    assert config.chunk_size == 500


def test_adapter_with_retriever_fallback():
    adapter = MagicMock()
    adapter.retrieval_config = None
    retriever = MagicMock()
    retriever.search_kwargs = {"k": 10}
    retriever.chunk_size = 800
    retriever.search_type = "hybrid"
    adapter.retriever = retriever
    adapter.reranker = None
    config = introspect_retrieval_config(adapter)
    assert config is not None
    assert config.retrieval_k == 10
    assert config.chunk_size == 800
    assert config.retrieval_strategy == "hybrid"


def test_adapter_none_returns_none():
    assert introspect_retrieval_config(None) is None


def test_non_rag_adapter_returns_none():
    adapter = MagicMock()
    adapter.retrieval_config = None
    adapter.retriever = None
    assert introspect_retrieval_config(adapter) is None
```

- [ ] **Step 1.4: Run + commit**

```bash
pytest tests/unit/test_retrieval_introspection.py -v
pytest tests/ retune-cloud/tests/ -q   # 166 passed (162 + 4)

git add src/retune/optimizer/retrieval_config.py src/retune/optimizer/retrieval_introspection.py tests/unit/test_retrieval_introspection.py
git commit -m "optimizer: SDK retrieval config introspection"
```

---

## Task 2: SDK wires retrieval_config into optimize

**Files:**
- Modify: `src/retune/optimizer/client.py` (preauthorize signature + body)
- Modify: `src/retune/wrapper.py` (optimize method)
- Test: `tests/unit/test_wrapper_retrieval_upload.py`

- [ ] **Step 2.1: Extend OptimizerClient.preauthorize**

Add `retrieval_config: dict[str, Any] | None = None` parameter; include in body dict when provided. Follow the same pattern Phase 3 T2 used for `tool_metadata`.

- [ ] **Step 2.2: Retuner.optimize collects retrieval_config**

In `src/retune/wrapper.py`'s `optimize` method, after the tool-metadata block added in Phase 3 T2:

```python
    retrieval_config_payload = None
    if "rag" in axes_list:
        try:
            from retune.optimizer.retrieval_introspection import introspect_retrieval_config
            rc = introspect_retrieval_config(self._adapter)
            retrieval_config_payload = rc.model_dump() if rc else None
        except Exception as e:
            logger.warning("Retrieval introspection failed: %s", e)
            retrieval_config_payload = None

    resp = client.preauthorize(
        ...,
        retrieval_config=retrieval_config_payload,
    )
```

Also add module-level import in `wrapper.py` for patchability:

```python
from retune.optimizer.retrieval_introspection import introspect_retrieval_config  # noqa: F401
```

- [ ] **Step 2.3: Test** (2 tests — patches `introspect_retrieval_config`, asserts payload forwarded)

- [ ] **Step 2.4: Run + commit**

```bash
git add src/retune/optimizer/client.py src/retune/wrapper.py tests/unit/test_wrapper_retrieval_upload.py
git commit -m "optimizer: SDK uploads retrieval_config when rag axis requested"
```

---

## Task 3: Adapter `apply_retrieval_override`

**Files:**
- Modify: `src/retune/adapters/base.py` (add default method)
- Modify: `src/retune/adapters/custom.py` (no-op inherited, confirm)
- Modify: `src/retune/adapters/langchain_adapter.py` (override)
- Modify: `src/retune/adapters/langgraph_adapter.py` (override)
- Test: `tests/unit/test_adapter_retrieval_override.py`

- [ ] **Step 3.1: Base method**

In `src/retune/adapters/base.py`'s `BaseAdapter` class, add:

```python
    def apply_retrieval_override(self, **kwargs: Any) -> None:
        """Apply runtime retrieval config overrides (k, chunk_size, etc.).

        Default no-op. Adapters that support runtime override (LangChain, LangGraph)
        override this. Adapters that don't support overrides inherit the no-op —
        optimizer suggestions for them land in Tier 2 (copy-paste) only.
        """
        pass   # no-op default
```

- [ ] **Step 3.2: LangChain override**

In `src/retune/adapters/langchain_adapter.py`'s `LangChainAdapter`:

```python
    def apply_retrieval_override(self, **kwargs: Any) -> None:
        # Many LangChain retrievers support .search_kwargs mutation
        retriever = getattr(self, "_retriever", None) or getattr(self._chain, "retriever", None)
        if retriever is None:
            return
        if "retrieval_k" in kwargs:
            retriever.search_kwargs = {**getattr(retriever, "search_kwargs", {}), "k": int(kwargs["retrieval_k"])}
        # Other kwargs: degrade to no-op for Phase 4
```

- [ ] **Step 3.3: LangGraph override** — similar, depending on the specific graph retriever node shape; if unknown, inherit the no-op

- [ ] **Step 3.4: Test** (3 tests — base no-op, LangChain k-override, no-op when retriever missing)

- [ ] **Step 3.5: Commit**

```bash
git add src/retune/adapters/base.py src/retune/adapters/langchain_adapter.py src/retune/adapters/langgraph_adapter.py tests/unit/test_adapter_retrieval_override.py
git commit -m "optimizer: adapter apply_retrieval_override hook (base no-op + LC/LG overrides)"
```

---

## Task 4: Cloud `optimization_run_retrieval_config` DB

**Files:**
- Modify: `retune-cloud/server/db/schema.sql` (append)
- Modify: `retune-cloud/server/db/postgres.py` (append 2 helpers)
- Test: `retune-cloud/tests/test_opt_run_retrieval_config_db.py`

- [ ] **Step 4.1: Append schema**

```sql
-- ============ Phase 4: Per-run retrieval config (auto-purged) ============

CREATE TABLE IF NOT EXISTS optimization_run_retrieval_config (
    run_id VARCHAR(64) PRIMARY KEY REFERENCES optimization_runs(id) ON DELETE CASCADE,
    config JSONB NOT NULL DEFAULT '{}',
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 4.2: Helpers** — follow Phase 3 T3 pattern for `save_opt_run_retrieval_config(run_id, config)` and `get_opt_run_retrieval_config(run_id) -> dict | None`.

- [ ] **Step 4.3: Tests** (3 tests)

- [ ] **Step 4.4: Commit**

```bash
cd retune-cloud
git add server/db/schema.sql server/db/postgres.py tests/test_opt_run_retrieval_config_db.py
git commit -m "optimizer: schema + helpers for per-run retrieval config"
```

---

## Task 5: Preauthorize accepts retrieval_config

**Files:**
- Modify: `retune-cloud/server/routes/optimize.py`
- Test: `retune-cloud/tests/test_preauth_retrieval_config.py`

Same pattern as Phase 3 T4.

Add `retrieval_config: dict[str, Any] | None = None` field to `PreauthorizeRequest`; in the route handler, call `db.save_opt_run_retrieval_config(run_id, req.retrieval_config)` when provided.

---

## Task 6: Cloud `RAGUsageAnalyzer`

**Files:**
- Create: `retune-cloud/server/optimizer/rag_optimizer/__init__.py`
- Create: `retune-cloud/server/optimizer/rag_optimizer/analyzer.py`
- Modify: `retune-cloud/server/optimizer/models.py` (add `RAGUsageReport`, `RetrievalStepStats`)
- Test: `retune-cloud/tests/test_rag_usage_analyzer.py`

The analyzer examines traces' steps for retrieval-like calls (step.type == "retrieval" or step.step_type == "retrieve"). Produces `RAGUsageReport` with:
- `avg_docs_retrieved: float`
- `avg_retrieval_latency_ms: float`
- `groundedness_score: float` (keyword overlap between retrieved doc content and final response — heuristic)
- `retrieval_calls: int`
- `traces_with_retrieval: int`
- `queries_with_empty_results: int`
- `overall_quality_signal: Literal["poor", "mediocre", "ok"]`

The quality signal is a simple heuristic: `poor` if groundedness < 0.3 or >20% queries have empty results; `mediocre` if groundedness 0.3-0.6; `ok` otherwise. This drives what suggestions the proposer generates.

- [ ] **4 tests**: step detection, groundedness heuristic, empty-results counter, quality-signal thresholds.

```bash
cd retune-cloud
git add server/optimizer/rag_optimizer/__init__.py server/optimizer/rag_optimizer/analyzer.py server/optimizer/models.py tests/test_rag_usage_analyzer.py
git commit -m "optimizer: RAGUsageAnalyzer — retrieval-step trace analysis"
```

---

## Task 7: Cloud `RAGCandidateProposer`

**Files:**
- Create: `retune-cloud/server/optimizer/rag_optimizer/proposer.py`
- Modify: `retune-cloud/server/optimizer/models.py` (add `RAGCandidate`)
- Test: `retune-cloud/tests/test_rag_candidate_proposer.py`

`RAGCandidate` fields: `candidate_id`, `config_overrides: dict[str, Any]` (the parameter delta), `kind: Literal["k_sweep", "chunk_sweep", "reranker_toggle", "strategy_switch"]`, `rationale: str`.

`RAGCandidateProposer.propose(usage_report, baseline_config) -> list[RAGCandidate]`:

Rules (deterministic, parameter sweeps — no LLM):
- If `usage_report.overall_quality_signal == "poor"` AND `baseline.retrieval_strategy == "dense"`:
  - Candidate A: `retrieval_strategy="hybrid"`
  - Candidate B: `reranker_enabled=True, reranker_model="bge-reranker-base"`
- If `usage_report.avg_docs_retrieved >= baseline.retrieval_k * 0.9` (always maxing out retrieval):
  - Candidate C: `retrieval_k = baseline.retrieval_k + 3`
- If `usage_report.avg_docs_retrieved <= baseline.retrieval_k * 0.3` (way over-fetching):
  - Candidate D: `retrieval_k = max(2, baseline.retrieval_k - 2)`
- If `baseline.chunk_size > 1200`:
  - Candidate E: `chunk_size = 800, chunk_overlap = 150`
- If `baseline.chunk_size < 500`:
  - Candidate F: `chunk_size = 800`

Always include the baseline (config_overrides = {}, kind = "baseline") as Candidate 0.

Max 6 candidates total (truncate if more rules fire).

- [ ] **4 tests**: baseline always included, poor-quality triggers strategy/reranker, maxed-retrieval triggers k-up, over-fetching triggers k-down.

```bash
cd retune-cloud
git add server/optimizer/rag_optimizer/proposer.py server/optimizer/models.py tests/test_rag_candidate_proposer.py
git commit -m "optimizer: RAGCandidateProposer — deterministic parameter sweep"
```

---

## Task 8: Cloud `RAGOptimizerAgent`

**Files:**
- Create: `retune-cloud/server/optimizer/rag_optimizer/agent.py`
- Test: `retune-cloud/tests/test_rag_optimizer_agent.py`

`RAGOptimizerAgent` composes analyzer + proposer. API shape matches PromptOptim's pattern:

```python
class RAGOptimizerAgent:
    def __init__(self) -> None:
        self._analyzer = RAGUsageAnalyzer()
        self._proposer = RAGCandidateProposer()

    def propose_candidates(
        self,
        traces: list[dict[str, Any]],
        retrieval_config: dict[str, Any] | None,
    ) -> list[RAGCandidate]:
        """Return candidate configs to dispatch. If no retrieval_config provided,
        returns only conceptual suggestions (no dispatch)."""
        usage = self._analyzer.analyze(traces)
        if retrieval_config is None:
            return []  # Can't propose parameter sweeps without a baseline
        return self._proposer.propose(usage, retrieval_config)

    def conceptual_suggestions(
        self,
        traces: list[dict[str, Any]],
    ) -> list[dict]:
        """Return Tier 3 conceptual suggestions based on trace analysis alone,
        regardless of whether we have retrieval_config to sweep over."""
        usage = self._analyzer.analyze(traces)
        out: list[dict] = []
        if usage.queries_with_empty_results > 0:
            out.append({
                "tier": 3, "axis": "rag",
                "title": f"{usage.queries_with_empty_results} queries returned no retrieval results",
                "description": "Consider expanding the corpus, relaxing similarity thresholds, "
                               "or adding a fallback to keyword search.",
                "confidence": "M",
            })
        if usage.overall_quality_signal == "poor":
            out.append({
                "tier": 3, "axis": "rag",
                "title": "Consider hybrid BM25 + dense retrieval",
                "description": "Groundedness is below acceptable threshold; dense retrieval "
                               "alone may be missing lexical matches. Hybrid retrieval often "
                               "significantly improves on this.",
                "confidence": "M",
            })
        return out
```

- [ ] **3 tests**: propose_candidates with config, propose_candidates without config returns empty, conceptual_suggestions fires on poor quality.

```bash
cd retune-cloud
git add server/optimizer/rag_optimizer/agent.py tests/test_rag_optimizer_agent.py
git commit -m "optimizer: RAGOptimizerAgent composes analyzer + proposer"
```

---

## Task 9: Orchestrator routes to RAGOptim + dispatches via JobQueue

**Files:**
- Modify: `retune-cloud/server/optimizer/orchestrator.py`
- Test: `retune-cloud/tests/test_orchestrator_rag_routing.py`

In `OptimizerOrchestrator.run()`, after the tools-axis block, add the rag-axis block:

```python
            # RAG axis
            rag_tier1: list[dict] = []
            rag_tier2: list[dict] = []
            rag_tier3: list[dict] = []
            if "rag" in (row.get("axes") or []):
                try:
                    retrieval_config = db.get_opt_run_retrieval_config(run_id)
                    rag_agent = RAGOptimizerAgent()
                    rag_candidates = rag_agent.propose_candidates(traces, retrieval_config)
                    rag_tier3 = rag_agent.conceptual_suggestions(traces)

                    # Dispatch rag candidates through JobQueue (same as Phase 2 does for prompt)
                    for cand in rag_candidates[1:]:  # skip baseline (cand[0])
                        q.push(run_id, {
                            "type": "run_candidate",
                            "candidate_id": cand.candidate_id,
                            "config_overrides": cand.config_overrides,
                            "query_set": query_set,
                        })

                    # Score each RAG candidate via same pattern as prompt candidates
                    # [scoring loop — abbreviated; follows Phase 2 T11 pattern]

                    # Classify winning candidate into tier1 or tier2 based on whether
                    # adapter supports runtime override (unknown at cloud side — default Tier 2)
                    if rag_winner:
                        rag_tier2.append({
                            "tier": 2, "axis": "rag",
                            "title": f"Tune retrieval ({rag_winner.kind})",
                            "description": rag_winner.rationale,
                            "confidence": "M",
                            "code_snippet": f"# In your adapter init:\n# {rag_winner.config_overrides}",
                            "apply_payload": {
                                "action": "apply_retrieval_override",
                                **rag_winner.config_overrides,
                            },
                        })
                except Exception as e:
                    logger.exception("RAGOptimizerAgent failed: %s", e)
```

Merge `rag_tier1 + rag_tier2 + rag_tier3` into the render call.

**Implementation note:** whether rag suggestions become Tier 1 vs. Tier 2 depends on SDK-side adapter capability — cloud has no way to know reliably, so Phase 4 defaults them all to Tier 2 (copy-paste). A future enhancement: the SDK reports adapter capabilities in the preauth payload and cloud decides tier accordingly.

- [ ] **1 test**: mocks all three subagents, asserts RAGOptim dispatched when `"rag" in axes`.

```bash
cd retune-cloud
git add server/optimizer/orchestrator.py tests/test_orchestrator_rag_routing.py
git commit -m "optimizer: Orchestrator routes to RAGOptim + JobQueue dispatch"
```

---

## Task 10: E2E integration test

**Files:**
- Create: `tests/integration/test_optimize_rag_e2e.py`

Extend Phase 3's `fake_db_phase3` fixture with retrieval-config helpers. Test:
1. SDK uploads traces + retrieval_config
2. Orchestrator dispatches RAGOptim (mocked) which returns 3 candidates
3. SDK worker posts mock results
4. Report has Tier 2 retrieval suggestion with `apply_retrieval_override` action

```bash
git add tests/integration/test_optimize_rag_e2e.py
git commit -m "optimizer: Phase 4 E2E — RAG parameter sweep end-to-end"
```

---

## Task 11: Phase 4 exit gate

- [ ] `pytest tests/ retune-cloud/tests/ -q` — all pass
- [ ] `ruff check` — clean
- [ ] `mypy src/retune/ --ignore-missing-imports` — no new errors
- [ ] **Manual smoke test** — wrap a real LangChain RAG chain:

```python
from retune import Retuner, Mode
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
# ... build a LangChain RAG chain with retriever.search_kwargs = {"k": 5} ...

retuner = Retuner(
    agent=my_rag_chain, adapter="langchain",
    mode=Mode.OBSERVE, api_key="<key>",
    agent_purpose="customer support RAG bot",
)
for q in sample_queries:
    retuner.run(q)

retuner.set_mode(Mode.IMPROVE)
report = retuner.optimize(
    source="last_n_traces", n=20,
    axes=["prompt", "tools", "rag"],
)
report.show()
```

- [ ] Report has at least one Tier 2 suggestion with `apply_retrieval_override` action
- [ ] Tier 3 has a conceptual suggestion like "consider hybrid retrieval" or "expand corpus"

---

## Known limitations for Phase 4

1. **SDK adapter capability is not communicated to cloud.** All RAG suggestions default to Tier 2. A future enhancement sends adapter capability flags in preauth so the cloud can promote certain suggestions to Tier 1.
2. **Groundedness heuristic is keyword-overlap.** Real groundedness requires entailment scoring (LLM or NLI model) — Phase 5 can upgrade.
3. **No reindexing suggestions.** Reindexing modifies the corpus, which is always Tier 3 conceptual.
4. **RAGCandidateProposer is purely deterministic.** A future phase could use an LLM to propose more nuanced variants (e.g., "try a domain-specific embedding model"). Deterministic is fine for Phase 4.
5. **No A/B testing.** Each candidate gets a single execution per query — no statistical significance testing. Phase 5 dashboard adds multi-run A/B if needed.
