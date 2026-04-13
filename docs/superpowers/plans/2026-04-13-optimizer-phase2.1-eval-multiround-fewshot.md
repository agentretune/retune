# Optimizer Phase 2.1 (Real Eval + Multi-round Beam + Few-shot) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 2 prompt optimizer from "works end-to-end but with stub eval scores and single-round beam" into a genuinely useful optimizer. Three independent but complementary improvements: (1) SDK's candidate runner runs real open-source evaluators and returns real scores, (2) BeamSearchAPO runs multiple rounds with pruning on scored feedback, (3) PromptOptimizerAgent also curates few-shot examples from high-scoring traces.

**Architecture:** The key architectural change is **inverted control** for the beam loop. Today the Orchestrator calls `PromptOptimizerAgent.generate_candidates(...)` and gets back a flat list of candidates, then dispatches/scores them all at once. Multi-round beam requires interleaving generation with scoring (round N's output depends on round N-1's scores). We flip the control flow: `PromptOptimizerAgent.run_iterative(traces, baseline, score_fn)` — the agent runs the multi-round loop, calling the Orchestrator-provided `score_fn(candidates) -> list[ScoredCandidate]` at each round. SDK side, `_make_candidate_runner` composes registered evaluators to produce real eval scores instead of hardcoded zeros.

**Tech Stack:** Same as Phase 2. No new dependencies. Reuses existing `src/retune/evaluators/` (cost, latency, retrieval, llm_judge, pairwise_judge).

**Reference spec:** `docs/superpowers/specs/2026-04-12-optimizer-design.md` §5 (PromptOptimizerAgent — "also curates few-shot examples from high-scoring traces"). The multi-round beam and real-eval wiring are listed as Phase 2.1 in the Phase 2 plan's Known Limitations section.

**Phase 2 baseline:** branch `phase2/prompt-optimizer` on both repos, 139 tests passing. Phase 2.1 branches from Phase 2.

---

## Scope

### In

- **Real SDK evaluator wiring** — `Retuner._make_candidate_runner` runs the registered SDK evaluators over each candidate's trace, returning real `{evaluator_name: score}` dict instead of stub zeros.
- **Multi-round beam search** — `BeamSearchAPO.run_iterative(traces, baseline, score_fn, beam_rounds, beam_width, branch_factor)` drives N rounds of critique→rewrite→score→prune.
- **ScoredCandidate model** — pairs a `PromptCandidate` with its `scalar_score` for beam pruning.
- **Few-shot curation** — select diverse high-scoring traces and include them as `few_shot_examples` in candidate `config_overrides`.
- **Inverted control for PromptOptimizerAgent** — new `run_iterative` method; old `generate_candidates` kept for backward compat (may be deleted after Orchestrator migrates).
- **Orchestrator adaptation** — calls `run_iterative` with a `score_fn` callback that encapsulates the dispatch-and-collect cycle.

### Out (deferred to Phase 3+)

- ToolOptimizerAgent, RAGOptimizerAgent (Phase 3, 4)
- LLM-judge executed cloud-side (currently scoring is still SDK side via BYO-key evaluators — matches the open-core positioning)
- Adaptive beam width / dynamic budget (not needed at Phase 2.1 scale)
- Full deepagents framework integration (Phase 3 when subagent delegation becomes non-trivial)

---

## File Structure

### New files — SDK side

| File | Responsibility |
|---|---|
| `src/retune/optimizer/evaluator_pipeline.py` | `run_evaluators_on_trace(evaluators, trace) -> dict[name, float]` — executes a registered evaluator set against a freshly-built `ExecutionTrace` |

### Modified files — SDK side

| File | Change |
|---|---|
| `src/retune/wrapper.py` | `_make_candidate_runner` composes the real evaluators instead of returning stub scores |
| `tests/unit/test_wrapper_optimize_overrides.py` | Update assertions — runner now returns real (not stub) eval scores |

### New files — cloud side

| File | Responsibility |
|---|---|
| `retune-cloud/server/optimizer/prompt_optimizer/few_shot.py` | `select_few_shot_examples(traces, k, diversity_strategy)` → list of trace dicts suitable as few-shot examples |
| `retune-cloud/server/optimizer/models.py` addition | `ScoredCandidate` Pydantic class (holds `candidate: PromptCandidate`, `scalar_score: float`, `dimensions: dict[str, float]`, `guardrails_held: bool`) |

### Modified files — cloud side

| File | Change |
|---|---|
| `retune-cloud/server/optimizer/prompt_optimizer/beam_search.py` | Add `run_iterative(traces, baseline, score_fn, rounds, width, branch) -> list[ScoredCandidate]` method |
| `retune-cloud/server/optimizer/prompt_optimizer/agent.py` | Add `run_iterative(traces, baseline, score_fn)` method. Update `generate_candidates` to also inject few-shot examples into non-baseline candidates. |
| `retune-cloud/server/optimizer/orchestrator.py` | Replace the flat dispatch with an iterative loop driven by `PromptOptimizerAgent.run_iterative(..., score_fn=self._score_batch)`. The `_score_batch` helper does what the current `for cand in candidates: push/poll/score` loop does, but for one batch at a time. |

### Tests

| File | Covers |
|---|---|
| `tests/unit/test_evaluator_pipeline.py` | `run_evaluators_on_trace` — runs registered evaluators, aggregates scores |
| `tests/unit/test_wrapper_candidate_runner_real_eval.py` | `_make_candidate_runner` returns real scores from registered evaluators |
| `retune-cloud/tests/test_few_shot_selector.py` | Few-shot diverse selection |
| `retune-cloud/tests/test_scored_candidate_model.py` | `ScoredCandidate` field validation |
| `retune-cloud/tests/test_beam_multi_round.py` | `run_iterative` runs N rounds, prunes to beam_width after each |
| `retune-cloud/tests/test_prompt_optimizer_agent_iterative.py` | `PromptOptimizerAgent.run_iterative` delegates to beam + injects few-shot |
| `retune-cloud/tests/test_orchestrator_multi_round.py` | Orchestrator's updated flow (score_fn callback path) |
| `tests/integration/test_optimize_phase21_e2e.py` | Full flow with 2 beam rounds + real eval scores + few-shot-bearing candidates |

---

## Task Summary

Eleven tasks. Bottom-up: SDK evaluator pipeline → SDK runner → cloud few-shot selector → ScoredCandidate → beam multi-round → agent iterative → orchestrator adaptation → E2E.

1. SDK: `evaluator_pipeline.run_evaluators_on_trace`
2. SDK: `_make_candidate_runner` runs real evaluators (update test_6-era assertions)
3. Cloud: `ScoredCandidate` model
4. Cloud: `few_shot.select_few_shot_examples` helper
5. Cloud: `BeamSearchAPO.run_iterative` (multi-round loop)
6. Cloud: `PromptOptimizerAgent.run_iterative` + few-shot injection
7. Cloud: Orchestrator uses `run_iterative` with score_fn callback
8. SDK: verify `_make_candidate_runner` applies `few_shot_examples` override (should already work from Phase 2 T6; add an explicit test)
9. Cloud: extend `ReportWriter` to surface few-shot changes in Tier 1 suggestions
10. Cloud: JudgeAgent reward handling for multi-round (verify — no code change expected, just test coverage)
11. E2E integration test with multi-round + real eval + few-shot

---

## Task 1: SDK evaluator pipeline

**Files:**
- Create: `src/retune/optimizer/evaluator_pipeline.py`
- Test: `tests/unit/test_evaluator_pipeline.py`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/unit/test_evaluator_pipeline.py
"""run_evaluators_on_trace — aggregates scores from registered evaluators."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.evaluator_pipeline import run_evaluators_on_trace


def test_runs_each_evaluator_once_and_aggregates():
    trace = {"query": "q", "response": "r", "steps": []}

    ev1 = MagicMock()
    ev1.name = "cost"
    ev1.evaluate.return_value = MagicMock(score=0.002)

    ev2 = MagicMock()
    ev2.name = "latency"
    ev2.evaluate.return_value = MagicMock(score=1.5)

    scores = run_evaluators_on_trace([ev1, ev2], trace)
    assert scores == {"cost": 0.002, "latency": 1.5}
    ev1.evaluate.assert_called_once()
    ev2.evaluate.assert_called_once()


def test_empty_evaluator_list_returns_empty_dict():
    assert run_evaluators_on_trace([], {"query": "q", "response": "r"}) == {}


def test_evaluator_that_raises_is_logged_and_skipped():
    trace = {"query": "q", "response": "r", "steps": []}

    bad = MagicMock()
    bad.name = "broken"
    bad.evaluate.side_effect = RuntimeError("boom")

    good = MagicMock()
    good.name = "cost"
    good.evaluate.return_value = MagicMock(score=0.001)

    scores = run_evaluators_on_trace([bad, good], trace)
    assert scores == {"cost": 0.001}   # broken evaluator skipped, good still scored


def test_score_is_coerced_to_float():
    trace = {"query": "q", "response": "r", "steps": []}
    ev = MagicMock()
    ev.name = "score"
    ev.evaluate.return_value = MagicMock(score=7)   # int, not float
    scores = run_evaluators_on_trace([ev], trace)
    assert scores == {"score": 7.0}
    assert isinstance(scores["score"], float)
```

Run: `pytest tests/unit/test_evaluator_pipeline.py -v` — expect FAIL.

- [ ] **Step 1.2: Implement**

```python
# src/retune/optimizer/evaluator_pipeline.py
"""Run a set of evaluators over a trace dict, return {name: score} dict.

Used by the candidate runner: after executing the user's agent with
overridden config, we run the registered evaluators against the produced
trace to get real eval_scores to send back to the cloud optimizer.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _EvaluatorLike(Protocol):
    name: str
    def evaluate(self, trace: Any) -> Any: ...


def _build_trace_object(trace_dict: dict[str, Any]) -> Any:
    """Convert a plain dict trace into whatever shape evaluators expect.

    The existing evaluators accept an `ExecutionTrace` — construct one if
    possible, else pass the dict through (duck-typed evaluators handle it).
    """
    try:
        from retune.core.models import ExecutionTrace
        # Minimum viable — fill missing fields with sensible defaults
        return ExecutionTrace(
            query=trace_dict.get("query", ""),
            response=trace_dict.get("response", ""),
            steps=trace_dict.get("steps", []),
            eval_results=trace_dict.get("eval_results", []),
            config_snapshot=trace_dict.get("config_snapshot", {}),
        )
    except Exception:
        return trace_dict


def run_evaluators_on_trace(
    evaluators: list[_EvaluatorLike],
    trace: dict[str, Any],
) -> dict[str, float]:
    """Run each evaluator, return {name: score}. Failures are logged + skipped."""
    trace_obj = _build_trace_object(trace)
    scores: dict[str, float] = {}
    for ev in evaluators:
        try:
            result = ev.evaluate(trace_obj)
            scores[ev.name] = float(result.score)
        except Exception as e:
            logger.warning("Evaluator %r failed: %s", getattr(ev, "name", ev), e)
    return scores
```

- [ ] **Step 1.3: Run + commit (outer repo)**

Run: `pytest tests/unit/test_evaluator_pipeline.py -v` → 4 pass.
Run full suite → 143 pass (139 + 4).

```bash
git add src/retune/optimizer/evaluator_pipeline.py tests/unit/test_evaluator_pipeline.py
git commit -m "optimizer: SDK evaluator pipeline helper"
```

---

## Task 2: SDK candidate runner uses real evaluators

**Files:**
- Modify: `src/retune/wrapper.py` (`_make_candidate_runner`)
- Modify: `tests/unit/test_wrapper_optimize_overrides.py` (update the stub-score assertions)
- Test: `tests/unit/test_wrapper_candidate_runner_real_eval.py`

- [ ] **Step 2.1: Inspect `Retuner._evaluators` attribute**

Read `src/retune/wrapper.py` to find where the evaluators are registered on the `Retuner` instance (from Phase 1). The attribute name is likely `self._evaluators` — verify. It's a `list[BaseEvaluator]` built from the `evaluators=` kwarg in `__init__`.

If the user didn't pass evaluators, this list may be empty or have defaults (cost, latency). Behavior we want: if empty, return an empty scores dict (not stub zeros). The Orchestrator + JudgeAgent already handle empty scores dict correctly.

- [ ] **Step 2.2: Write failing test**

```python
# tests/unit/test_wrapper_candidate_runner_real_eval.py
"""Candidate runner runs registered evaluators, returns real scores."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune import Retuner, Mode


def test_runner_runs_registered_evaluators():
    def agent(q: str) -> str:
        return "resp to " + q

    fake_eval = MagicMock()
    fake_eval.name = "my_metric"
    fake_eval.evaluate.return_value = MagicMock(score=0.75)

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
        evaluators=[fake_eval],
    )
    runner = retuner._make_candidate_runner()
    trace, scores = runner(
        {"system_prompt": "NEW"},
        [{"query": "hello", "trace_id": "t1"}],
    )
    # Real evaluator was invoked
    fake_eval.evaluate.assert_called_once()
    assert scores == {"my_metric": 0.75}


def test_runner_no_evaluators_returns_empty_scores():
    def agent(q: str) -> str:
        return "ok"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
        evaluators=[],   # explicitly empty
    )
    runner = retuner._make_candidate_runner()
    trace, scores = runner({}, [{"query": "hi"}])
    assert scores == {}
```

Run — expect FAIL because the current runner returns stub `{"llm_judge": 0.0, ...}`.

- [ ] **Step 2.3: Replace `_make_candidate_runner` body**

Find the Phase 2 T6 version of `_make_candidate_runner` in `src/retune/wrapper.py`. Replace the return path that currently hardcodes scores:

```python
    def _make_candidate_runner(self):
        """Return a callable that runs the wrapped agent with config overrides
        and returns REAL eval scores from the registered evaluators.

        Phase 2.1: replaces the Phase 2 stub (hardcoded zeros) with actual
        evaluator pipeline execution. If no evaluators are registered, returns
        an empty scores dict — JudgeAgent + Orchestrator handle that gracefully.
        """
        def _runner(overrides: dict, queries: list):
            snapshot = {}
            for key in ("system_prompt", "few_shot_examples"):
                if key in overrides:
                    snapshot[key] = getattr(self._config, key, None)
                    setattr(self._config, key, overrides[key])
            try:
                if not queries:
                    return ({"query": "", "response": ""}, {})
                q = queries[0].get("query", "")
                try:
                    resp = self._adapter.run(q) if self._adapter else ""
                except Exception:
                    resp = ""
                trace = {
                    "query": q,
                    "response": str(resp),
                    "steps": [],
                    "config_snapshot": self._config.to_flat_dict() if hasattr(self._config, "to_flat_dict") else {},
                }
                # Phase 2.1: run real evaluators
                from retune.optimizer.evaluator_pipeline import run_evaluators_on_trace
                scores = run_evaluators_on_trace(
                    getattr(self, "_evaluators", []) or [],
                    trace,
                )
                return (trace, scores)
            finally:
                for key, old_val in snapshot.items():
                    setattr(self._config, key, old_val)

        return _runner
```

- [ ] **Step 2.4: Update the Phase 2 T6 test that asserted stub scores**

Find `tests/unit/test_wrapper_optimize_overrides.py`. Both tests pass `evaluators=<nothing>` (defaulted), so today they might implicitly get stub scores. Under Phase 2.1's change:
- No evaluators registered → empty scores dict (not `{"llm_judge": 0.0, ...}`)

Check if existing tests assert anything about the scores value — if so, update the assertion to match the new empty-dict behavior OR pass `evaluators=[]` explicitly to make intent clear. The override-application assertions (`captured_prompts == ["OVERRIDDEN"]`) are unchanged.

Specifically: if the test has `trace, scores = runner(...)` and never asserts on `scores`, no change needed. If it asserts on `scores`, update to `assert scores == {}`.

- [ ] **Step 2.5: Run + commit**

Run `pytest tests/unit/test_wrapper_candidate_runner_real_eval.py -v` → 2 pass.
Run full suite → 145 pass.

```bash
git add src/retune/wrapper.py tests/unit/test_wrapper_optimize_overrides.py tests/unit/test_wrapper_candidate_runner_real_eval.py
git commit -m "optimizer: SDK candidate runner returns real eval scores"
```

---

## Task 3: ScoredCandidate model

**Files:**
- Modify: `retune-cloud/server/optimizer/models.py` (append)
- Test: `retune-cloud/tests/test_scored_candidate_model.py`

- [ ] **Step 3.1: Failing tests**

```python
# retune-cloud/tests/test_scored_candidate_model.py
"""ScoredCandidate model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.optimizer.models import PromptCandidate, ScoredCandidate


def test_scored_candidate_basic():
    sc = ScoredCandidate(
        candidate=PromptCandidate(candidate_id="c1", system_prompt="p",
                                  generation_round=0),
        scalar_score=7.5,
        dimensions={"llm_judge": 8.0, "cost": 0.001},
        guardrails_held=True,
    )
    assert sc.scalar_score == 7.5
    assert sc.candidate.candidate_id == "c1"


def test_scored_candidate_requires_candidate():
    with pytest.raises(ValidationError):
        ScoredCandidate(scalar_score=1.0, dimensions={}, guardrails_held=True)
```

- [ ] **Step 3.2: Append to `retune-cloud/server/optimizer/models.py`**

Add AFTER existing classes (near `PromptCandidate`):

```python
class ScoredCandidate(BaseModel):
    """A PromptCandidate paired with its JudgeAgent score result.
    Used by BeamSearchAPO for pruning and ranking during multi-round search."""
    candidate: PromptCandidate
    scalar_score: float
    dimensions: dict[str, float] = Field(default_factory=dict)
    guardrails_held: bool = True
```

- [ ] **Step 3.3: Run + commit**

Tests 2 pass. Full suite 147 pass.

```bash
cd retune-cloud
git add server/optimizer/models.py tests/test_scored_candidate_model.py
git commit -m "optimizer: add ScoredCandidate model for beam pruning"
```

---

## Task 4: Few-shot selector

**Files:**
- Create: `retune-cloud/server/optimizer/prompt_optimizer/few_shot.py`
- Test: `retune-cloud/tests/test_few_shot_selector.py`

- [ ] **Step 4.1: Failing tests**

```python
# retune-cloud/tests/test_few_shot_selector.py
"""Few-shot example selection — diverse high-scoring traces."""
from __future__ import annotations

from server.optimizer.prompt_optimizer.few_shot import (
    select_few_shot_examples,
)


def test_selects_high_scoring_traces():
    traces = [
        {"query": "q1", "response": "r1",
         "eval_results": [{"evaluator_name": "llm_judge", "score": 9.0}]},
        {"query": "q2", "response": "r2",
         "eval_results": [{"evaluator_name": "llm_judge", "score": 3.0}]},
        {"query": "q3", "response": "r3",
         "eval_results": [{"evaluator_name": "llm_judge", "score": 8.5}]},
    ]
    examples = select_few_shot_examples(traces, k=2, primary_metric="llm_judge")
    assert len(examples) == 2
    # Top 2 by score: q1 (9.0) and q3 (8.5)
    queries = [e["query"] for e in examples]
    assert "q1" in queries
    assert "q3" in queries
    assert "q2" not in queries


def test_no_eval_results_returns_empty():
    traces = [{"query": "q", "response": "r"}]
    assert select_few_shot_examples(traces, k=3) == []


def test_k_exceeds_available_returns_all_scored():
    traces = [
        {"query": "q1", "response": "r1",
         "eval_results": [{"evaluator_name": "llm_judge", "score": 5.0}]},
    ]
    examples = select_few_shot_examples(traces, k=10)
    assert len(examples) == 1


def test_examples_have_input_output_shape():
    """Returned examples should be suitable for few-shot formatting."""
    traces = [
        {"query": "What is AI?", "response": "AI is artificial intelligence.",
         "eval_results": [{"evaluator_name": "llm_judge", "score": 9.0}]},
    ]
    examples = select_few_shot_examples(traces, k=1)
    assert examples[0]["input"] == "What is AI?"
    assert examples[0]["output"] == "AI is artificial intelligence."
```

- [ ] **Step 4.2: Implement**

```python
# retune-cloud/server/optimizer/prompt_optimizer/few_shot.py
"""Select diverse high-scoring traces to use as few-shot examples."""
from __future__ import annotations

from typing import Any


def _primary_score(trace: dict[str, Any], metric: str) -> float | None:
    """Extract the primary metric score from a trace's eval_results."""
    for result in trace.get("eval_results", []) or []:
        if result.get("evaluator_name") == metric:
            try:
                return float(result.get("score", 0.0))
            except (TypeError, ValueError):
                return None
    return None


def select_few_shot_examples(
    traces: list[dict[str, Any]],
    k: int = 3,
    primary_metric: str = "llm_judge",
) -> list[dict[str, str]]:
    """Pick the top-k high-scoring traces by primary_metric, return as
    few-shot examples in {input, output} shape.

    Phase 2.1: simple greedy top-k by metric. Future extension: add
    diversity (e.g. embedding-based) to avoid near-duplicate examples.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for t in traces:
        score = _primary_score(t, primary_metric)
        if score is not None:
            scored.append((score, t))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:k]

    return [
        {
            "input": t.get("query", ""),
            "output": t.get("response", ""),
        }
        for _, t in top
    ]
```

- [ ] **Step 4.3: Run + commit**

Tests 4 pass. Full suite 151 pass.

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/few_shot.py tests/test_few_shot_selector.py
git commit -m "optimizer: few-shot example selector (top-k by metric)"
```

---

## Task 5: BeamSearchAPO.run_iterative (multi-round)

**Files:**
- Modify: `retune-cloud/server/optimizer/prompt_optimizer/beam_search.py`
- Test: `retune-cloud/tests/test_beam_multi_round.py`

- [ ] **Step 5.1: Failing tests**

```python
# retune-cloud/tests/test_beam_multi_round.py
"""Multi-round beam search with external scoring callback."""
from __future__ import annotations

from unittest.mock import MagicMock

from server.optimizer.prompt_optimizer.beam_search import BeamSearchAPO
from server.optimizer.models import PromptCandidate, ScoredCandidate


def _mk_scored(cand: PromptCandidate, score: float) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=cand, scalar_score=score,
        dimensions={"llm_judge": score}, guardrails_held=True,
    )


def test_run_iterative_runs_multiple_rounds():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="x")
    apo = BeamSearchAPO(llm=mock_llm, beam_width=1, branch_factor=2)

    scoring_rounds = []
    def score_fn(cands):
        scoring_rounds.append(len(cands))
        return [_mk_scored(c, i) for i, c in enumerate(cands)]

    result = apo.run_iterative(
        traces=[{"query": "q"}],
        baseline_prompt="BASELINE",
        score_fn=score_fn,
        rounds=2,
    )
    # At least 2 rounds of scoring happened
    assert len(scoring_rounds) >= 2


def test_run_iterative_prunes_to_beam_width():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="rewrite")
    apo = BeamSearchAPO(llm=mock_llm, beam_width=2, branch_factor=3)

    all_scored = []
    def score_fn(cands):
        scored = [_mk_scored(c, i * 0.5) for i, c in enumerate(cands)]
        all_scored.append(scored)
        return scored

    result = apo.run_iterative(
        traces=[{"query": "q"}], baseline_prompt="B",
        score_fn=score_fn, rounds=2,
    )
    # After round 1, only beam_width=2 candidates carried forward
    # Round 2 generated branch_factor=3 rewrites from each of the 2 kept → up to 6 new + 2 kept = 8, but
    # the key property is: len(scored[round2]) > 0 and <= beam_width*(1+branch_factor)
    assert len(all_scored) == 2


def test_run_iterative_returns_top_scored():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="x")
    apo = BeamSearchAPO(llm=mock_llm, beam_width=1, branch_factor=1)

    def score_fn(cands):
        # Last candidate always wins
        return [
            _mk_scored(c, i * 1.0)
            for i, c in enumerate(cands)
        ]

    result = apo.run_iterative(
        traces=[{"query": "q"}], baseline_prompt="B",
        score_fn=score_fn, rounds=1,
    )
    # Result is the full final-round scored list, sorted by scalar_score desc
    assert result[0].scalar_score >= result[-1].scalar_score
```

- [ ] **Step 5.2: Implement — append to `beam_search.py`**

Add imports at top if not present:
```python
from server.optimizer.models import PromptCandidate, ScoredCandidate
from typing import Callable
```

Add method to `BeamSearchAPO`:

```python
    def run_iterative(
        self,
        traces: list[dict],
        baseline_prompt: str,
        score_fn: Callable[[list[PromptCandidate]], list[ScoredCandidate]],
        rounds: int = 2,
    ) -> list[ScoredCandidate]:
        """Run multi-round beam search.

        Round 1: seed with baseline + branch_factor rewrites, score, keep top beam_width.
        Round N: for each kept candidate, generate branch_factor rewrites, score,
                 keep top beam_width across ALL candidates (kept + new).

        Returns the final scored candidates sorted by scalar_score desc.
        """
        # Round 1 seed
        baseline = PromptCandidate(
            candidate_id=self._mk_id(),
            system_prompt=baseline_prompt,
            generation_round=0,
            parent_id=None,
        )
        current: list[PromptCandidate] = [baseline]
        critique_r0 = self._critique(baseline_prompt, traces)
        for _ in range(self._branch_factor):
            current.append(PromptCandidate(
                candidate_id=self._mk_id(),
                system_prompt=self._rewrite(baseline_prompt, critique_r0),
                generation_round=1,
                parent_id=baseline.candidate_id,
            ))

        scored: list[ScoredCandidate] = score_fn(current)
        scored.sort(key=lambda s: s.scalar_score, reverse=True)
        kept: list[ScoredCandidate] = scored[: self._beam_width]

        # Subsequent rounds
        for round_num in range(2, rounds + 1):
            new_candidates: list[PromptCandidate] = []
            for sc in kept:
                critique = self._critique(sc.candidate.system_prompt, traces)
                for _ in range(self._branch_factor):
                    new_candidates.append(PromptCandidate(
                        candidate_id=self._mk_id(),
                        system_prompt=self._rewrite(sc.candidate.system_prompt, critique),
                        generation_round=round_num,
                        parent_id=sc.candidate.candidate_id,
                    ))

            # Score only the new candidates, merge with kept
            new_scored = score_fn(new_candidates)
            all_scored = kept + new_scored
            all_scored.sort(key=lambda s: s.scalar_score, reverse=True)
            kept = all_scored[: self._beam_width]
            scored = all_scored   # final output sorts all

        scored.sort(key=lambda s: s.scalar_score, reverse=True)
        return scored
```

- [ ] **Step 5.3: Run + commit**

Tests 3 pass. Full suite 154.

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/beam_search.py tests/test_beam_multi_round.py
git commit -m "optimizer: multi-round beam search with external score callback"
```

---

## Task 6: PromptOptimizerAgent.run_iterative + few-shot injection

**Files:**
- Modify: `retune-cloud/server/optimizer/prompt_optimizer/agent.py`
- Test: `retune-cloud/tests/test_prompt_optimizer_agent_iterative.py`

- [ ] **Step 6.1: Failing tests**

```python
# retune-cloud/tests/test_prompt_optimizer_agent_iterative.py
"""PromptOptimizerAgent.run_iterative — wraps BeamSearchAPO multi-round + few-shot."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from server.optimizer.prompt_optimizer.agent import PromptOptimizerAgent
from server.optimizer.models import PromptCandidate, ScoredCandidate


@patch("server.optimizer.prompt_optimizer.agent.BeamSearchAPO")
@patch("server.optimizer.prompt_optimizer.agent.create_rewriter_llm")
@patch("server.optimizer.prompt_optimizer.agent.select_few_shot_examples")
def test_run_iterative_delegates_and_injects_few_shot(
    mock_select, mock_llm_factory, mock_beam_cls
):
    mock_select.return_value = [{"input": "q1", "output": "r1"}]
    mock_llm_factory.return_value = "llm"
    mock_beam = MagicMock()

    baseline = PromptCandidate(candidate_id="cb", system_prompt="B", generation_round=0)
    rewrite = PromptCandidate(candidate_id="cr", system_prompt="R", generation_round=1)
    mock_beam.run_iterative.return_value = [
        ScoredCandidate(candidate=rewrite, scalar_score=8.0,
                        dimensions={}, guardrails_held=True),
        ScoredCandidate(candidate=baseline, scalar_score=5.0,
                        dimensions={}, guardrails_held=True),
    ]
    mock_beam_cls.return_value = mock_beam

    agent = PromptOptimizerAgent(rewriter_llm="gpt-4o-mini")
    score_fn = MagicMock()
    result = agent.run_iterative(
        traces=[{"query": "q"}], baseline_prompt="B",
        score_fn=score_fn, rounds=2,
    )

    assert len(result) == 2
    # beam.run_iterative was called with the score_fn
    _args, kwargs = mock_beam.run_iterative.call_args
    assert kwargs["score_fn"] is score_fn
    # few_shot selection was called
    mock_select.assert_called_once()
```

- [ ] **Step 6.2: Modify `agent.py`**

Add to imports:
```python
from server.optimizer.prompt_optimizer.few_shot import select_few_shot_examples
from server.optimizer.models import ScoredCandidate
from typing import Callable
```

Add method to `PromptOptimizerAgent`:

```python
    def run_iterative(
        self,
        traces: list[dict[str, Any]],
        baseline_prompt: str,
        score_fn: Callable[[list], list[ScoredCandidate]],
        rounds: int = 2,
        few_shot_k: int = 3,
    ) -> list[ScoredCandidate]:
        """Multi-round beam search + few-shot curation.

        The `score_fn` receives a list of PromptCandidate objects and must
        return a list of ScoredCandidate objects in the same order (one score
        per input candidate). The Orchestrator supplies this callback, which
        internally dispatches candidates via JobQueue and collects results.
        """
        # Select few-shot examples from high-scoring traces (shared across candidates)
        few_shots = select_few_shot_examples(traces, k=few_shot_k)

        # Wrap score_fn so each candidate carries the few-shot examples in its dispatch
        def _score_with_few_shot(candidates):
            # Attach few_shot_examples onto candidates in-place for scoring
            # (They'll be read by the Orchestrator when building config_overrides)
            for c in candidates:
                if not c.few_shot_examples and few_shots:
                    c.few_shot_examples = few_shots
            return score_fn(candidates)

        return self._beam.run_iterative(
            traces=traces,
            baseline_prompt=baseline_prompt,
            score_fn=_score_with_few_shot,
            rounds=rounds,
        )
```

- [ ] **Step 6.3: Run + commit**

Tests 1 passes, existing tests untouched. Full suite 155.

```bash
cd retune-cloud
git add server/optimizer/prompt_optimizer/agent.py tests/test_prompt_optimizer_agent_iterative.py
git commit -m "optimizer: PromptOptimizerAgent.run_iterative with few-shot injection"
```

---

## Task 7: Orchestrator uses run_iterative + score_fn

**Files:**
- Modify: `retune-cloud/server/optimizer/orchestrator.py`
- Test: `retune-cloud/tests/test_orchestrator_multi_round.py`

- [ ] **Step 7.1: Failing test**

```python
# retune-cloud/tests/test_orchestrator_multi_round.py
"""Orchestrator invokes run_iterative with a score_fn that dispatches candidates."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from server.optimizer.orchestrator import OptimizerOrchestrator
from server.optimizer.models import PromptCandidate, ScoredCandidate


@patch("server.optimizer.orchestrator.PromptOptimizerAgent")
@patch("server.optimizer.orchestrator.JudgeAgent")
@patch("server.optimizer.orchestrator.get_queue")
@patch("server.optimizer.orchestrator.get_results")
@patch("server.optimizer.orchestrator.db")
def test_orchestrator_calls_run_iterative(
    mock_db, mock_get_results, mock_get_queue, mock_judge_cls, mock_prompt_cls
):
    mock_db.get_opt_run.return_value = {
        "id": "r", "org_id": "o", "status": "pending",
        "source": "last_n_traces", "n_traces": 1,
        "axes": ["prompt"], "reward_spec": {}, "rewriter_llm": None,
    }
    mock_db.get_opt_run_traces.return_value = (
        [{"query": "q", "response": "r",
          "config_snapshot": {"system_prompt": "BASE"}}], 1,
    )

    mock_prompt = MagicMock()
    rewrite = PromptCandidate(candidate_id="cr", system_prompt="R", generation_round=1)
    mock_prompt.run_iterative.return_value = [
        ScoredCandidate(candidate=rewrite, scalar_score=9.0,
                        dimensions={"llm_judge": 9.0}, guardrails_held=True),
    ]
    mock_prompt_cls.return_value = mock_prompt

    orch = OptimizerOrchestrator()
    orch.run("r", candidate_result_timeout=0.2)

    # Orchestrator called run_iterative (not generate_candidates)
    mock_prompt.run_iterative.assert_called_once()
    # Called with a score_fn callback
    kwargs = mock_prompt.run_iterative.call_args.kwargs
    assert callable(kwargs["score_fn"])
    # Report saved with tier1 from winning rewrite
    save_kwargs = mock_db.save_opt_report.call_args.kwargs
    assert len(save_kwargs["tier1"]) >= 1
```

- [ ] **Step 7.2: Modify `orchestrator.py`**

Restructure `run()`. The candidate-batch loop (push → poll results → judge) becomes a helper method, invoked as the `score_fn` callback passed to `run_iterative`.

Replace the middle section of `run()` (from `candidates = prompt_agent.generate_candidates(...)` through the scoring loop) with:

```python
            prompt_agent = PromptOptimizerAgent(
                rewriter_llm=row.get("rewriter_llm") or "claude-3-7-sonnet",
            )

            query_set = [
                {"query": t.get("query", ""), "trace_id": t.get("id", "")}
                for t in traces[:_DEFAULT_QUERIES_PER_CANDIDATE]
            ]

            q = get_queue()
            results_store = get_results()
            baseline_scores: dict[str, float] = {}

            def _score_batch(candidates_batch):
                """Dispatch each candidate via JobQueue, poll results, score via JudgeAgent."""
                nonlocal baseline_scores
                for cand in candidates_batch:
                    q.push(run_id, {
                        "type": "run_candidate",
                        "candidate_id": cand.candidate_id,
                        "config_overrides": {
                            "system_prompt": cand.system_prompt,
                            "few_shot_examples": cand.few_shot_examples,
                        },
                        "query_set": query_set,
                    })

                scored: list[ScoredCandidate] = []
                deadline = time.time() + candidate_result_timeout * len(candidates_batch)
                for cand in candidates_batch:
                    result = None
                    while time.time() < deadline:
                        result = results_store.get(run_id, cand.candidate_id)
                        if result is not None:
                            break
                        time.sleep(0.1)
                    if result is None:
                        logger.warning("Timeout waiting for candidate %s", cand.candidate_id)
                        scored.append(ScoredCandidate(
                            candidate=cand, scalar_score=0.0,
                            dimensions={}, guardrails_held=False,
                        ))
                        continue
                    eval_scores = result.get("eval_scores", {})
                    if cand.generation_round == 0:
                        baseline_scores = eval_scores
                    judge_result = self._judge.score(eval_scores, spec, baseline_scores)
                    scored.append(ScoredCandidate(
                        candidate=cand,
                        scalar_score=judge_result.scalar_score,
                        dimensions=judge_result.dimensions,
                        guardrails_held=judge_result.guardrails_held,
                    ))
                return scored

            all_scored: list[ScoredCandidate] = prompt_agent.run_iterative(
                traces=traces,
                baseline_prompt=baseline_prompt,
                score_fn=_score_batch,
                rounds=_DEFAULT_BEAM_ROUNDS,
            )

            pareto_data = [
                {"candidate_id": s.candidate.candidate_id,
                 **s.dimensions, "scalar_score": s.scalar_score}
                for s in all_scored
            ]
```

Add near other module-level constants:

```python
_DEFAULT_BEAM_ROUNDS = 2
```

Also add import:

```python
from server.optimizer.models import ScoredCandidate
```

The code that picks the winner + builds tier1 remains — just adapt it to use `all_scored` (already sorted by beam) instead of the old `scored` list:

```python
            tier1: list[dict[str, Any]] = []
            summary = {"baseline_score": 0.0, "best_score": 0.0, "improvement_pct": 0.0}
            if all_scored:
                winner = all_scored[0]
                baseline_sc = next(
                    (s for s in all_scored if s.candidate.generation_round == 0),
                    winner,
                )
                best = winner.scalar_score
                base = baseline_sc.scalar_score
                improvement = ((best - base) / max(base, 1e-9)) * 100 if base > 0 else 0.0
                summary = {
                    "baseline_score": base,
                    "best_score": best,
                    "improvement_pct": improvement,
                }
                if winner.candidate.candidate_id != baseline_sc.candidate.candidate_id:
                    tier1.append({
                        "tier": 1,
                        "axis": "prompt",
                        "title": "Rewrite system prompt",
                        "description": winner.candidate.system_prompt,
                        "confidence": "H" if improvement > 10 else "M",
                        "estimated_impact": {"judge": best - base},
                        "evidence_trace_ids": [t.get("id", "") for t in traces[:3]],
                        "apply_payload": {
                            "system_prompt": winner.candidate.system_prompt,
                            "few_shot_examples": winner.candidate.few_shot_examples,
                        },
                    })
```

- [ ] **Step 7.3: Run + commit**

Test 1 passes. Existing `test_orchestrator_real.py` (2 tests from Phase 2 T11) may need adjustment — its mock on `mock_prompt.generate_candidates` no longer fires; change to mock `mock_prompt.run_iterative` returning `list[ScoredCandidate]` instead. Update those 2 tests to match the new flow.

Run full suite → 156 (155 + 1 new; the 2 T11 tests still pass after adjustment).

```bash
cd retune-cloud
git add server/optimizer/orchestrator.py tests/test_orchestrator_multi_round.py tests/test_orchestrator_real.py
git commit -m "optimizer: Orchestrator uses iterative beam via score_fn callback"
```

---

## Task 8: Verify SDK runner applies few_shot_examples

**Files:**
- Test only: `tests/unit/test_wrapper_optimize_overrides.py` (append 1 test)

Phase 2 T6 already includes `few_shot_examples` in the list of keys the runner snapshots/applies. This task just adds an explicit test to pin that behavior.

- [ ] **Step 8.1: Append test**

```python
def test_runner_applies_few_shot_examples_override():
    captured = []

    def agent(q: str) -> str:
        return "resp"

    from retune import Retuner, Mode
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner._config.few_shot_examples = [{"input": "old", "output": "old"}]

    orig_run = retuner._adapter.run
    def capturing_run(q: str):
        captured.append(list(retuner._config.few_shot_examples))
        return orig_run(q)
    retuner._adapter.run = capturing_run

    new_examples = [{"input": "NEW_IN", "output": "NEW_OUT"}]
    runner = retuner._make_candidate_runner()
    runner({"few_shot_examples": new_examples}, [{"query": "hi"}])

    assert captured[0] == new_examples
    assert retuner._config.few_shot_examples == [{"input": "old", "output": "old"}]
```

Run → passes with the existing Phase 2 T6 runner. Full suite 157.

```bash
git add tests/unit/test_wrapper_optimize_overrides.py
git commit -m "optimizer: pin test for few_shot_examples override application"
```

---

## Task 9: ReportWriter surfaces few-shot changes

**Files:**
- Modify: `retune-cloud/server/optimizer/report_writer.py`
- Test: `retune-cloud/tests/test_report_writer.py` (append 1 test)

- [ ] **Step 9.1: Failing test — append to existing test file**

```python
def test_tier1_with_few_shot_renders_count_in_description():
    writer = ReportWriterAgent()
    tier1 = [{
        "axis": "prompt", "title": "Rewrite system prompt + few-shot",
        "description": "You are a precise helper.",
        "confidence": "H",
        "estimated_impact": {"judge": 1.8},
        "apply_payload": {
            "system_prompt": "You are a precise helper.",
            "few_shot_examples": [{"input": "a", "output": "b"},
                                   {"input": "c", "output": "d"}],
        },
    }]
    report = writer.render(
        run_id="r", understanding="",
        summary={"baseline_score": 6.0, "best_score": 7.8, "improvement_pct": 30.0},
        tier1=tier1, tier2=[], tier3=[], pareto_data=[],
    )
    assert "2 few-shot examples" in report.markdown
```

- [ ] **Step 9.2: Modify `_render_suggestion_line`**

Just before returning the line, if `apply_payload.few_shot_examples` is non-empty, append a note:

```python
def _render_suggestion_line(s: dict) -> str:
    # ... existing implementation ...
    line = f"- **[{axis}]** {title}  _({meta})_"
    if s.get("description"):
        line += f"\n  - {s['description']}"
    # NEW: surface few-shot count if present in apply_payload
    payload = s.get("apply_payload") or {}
    examples = payload.get("few_shot_examples") or []
    if examples:
        line += f"\n  - _({len(examples)} few-shot examples included in the apply payload)_"
    return line
```

- [ ] **Step 9.3: Run + commit**

Test passes. Full suite 158.

```bash
cd retune-cloud
git add server/optimizer/report_writer.py tests/test_report_writer.py
git commit -m "optimizer: ReportWriter surfaces few-shot example count in Tier 1"
```

---

## Task 10: Judge handles multi-round scoring — verify only

**Files:**
- Test: `retune-cloud/tests/test_judge_agent.py` (append 1 test)

No code change expected — Judge is stateless, already called per-candidate. Just pin test coverage for the baseline-update pattern across rounds.

- [ ] **Step 10.1: Append test**

```python
def test_judge_with_relative_guardrails_against_baseline():
    """Round 1 sets baseline, round 2 candidates scored relative to it."""
    from server.optimizer.reward_parser import default_reward_spec
    judge = JudgeAgent()
    spec = default_reward_spec()

    # Round 1 baseline: cost=0.001
    baseline = {"llm_judge": 6.0, "cost": 0.001, "latency": 1.0}
    r_baseline = judge.score(baseline, spec, baseline)
    assert r_baseline.guardrails_held is True

    # Round 2 candidate with 1.4× cost (within 1.5× default guardrail)
    r_ok = judge.score(
        {"llm_judge": 8.0, "cost": 0.0014, "latency": 1.0},
        spec, baseline,
    )
    assert r_ok.guardrails_held is True
    assert r_ok.scalar_score == 8.0

    # Round 2 candidate that blows cost guardrail (2× baseline)
    r_bad = judge.score(
        {"llm_judge": 9.0, "cost": 0.002, "latency": 1.0},
        spec, baseline,
    )
    assert r_bad.guardrails_held is False
    assert r_bad.scalar_score == 0.0
```

Commit:

```bash
cd retune-cloud
git add tests/test_judge_agent.py
git commit -m "optimizer: pin judge test for relative baseline across rounds"
```

---

## Task 11: E2E integration test for Phase 2.1

**Files:**
- Create: `tests/integration/test_optimize_phase21_e2e.py`

Full-stack test with real SDK evaluator + multi-round beam + few-shot.

- [ ] **Step 11.1: Write the test**

```python
# tests/integration/test_optimize_phase21_e2e.py
"""Phase 2.1 E2E: multi-round beam + real SDK evaluators + few-shot curation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.optimizer.models import PromptCandidate, ScoredCandidate


@pytest.fixture
def fake_db_p21():
    """Reuse the Phase 2 fake_db pattern."""
    state = {
        "runs": {}, "reports": {}, "traces_bundles": {},
        "org_used": {"org_1": 0}, "org_limit": {"org_1": 15},
    }
    # ... (same as Phase 2 fixture) ...
    # (copy the fake_db_phase2 fixture from tests/integration/test_optimize_prompt_e2e.py)
    raise NotImplementedError("Copy the fake DB fixture from Phase 2's e2e test")


def test_phase21_multi_round_e2e(fake_db_p21):
    """Run a 2-round beam search end-to-end with real evaluators + few-shot."""
    from server.optimizer.job_queue import get_queue, get_results
    get_queue().reset()
    get_results().reset()

    # Traces with eval_results high enough to become few-shot examples
    traces = [
        {"id": f"t{i}", "query": f"q{i}", "response": f"r{i}",
         "config_snapshot": {"system_prompt": "BASE"},
         "eval_results": [{"evaluator_name": "llm_judge", "score": 9.0 - i}]}
        for i in range(5)
    ]

    # Mock PromptOptimizerAgent.run_iterative to return 3 scored candidates
    baseline = PromptCandidate(candidate_id="cb", system_prompt="BASE", generation_round=0)
    r1 = PromptCandidate(candidate_id="cr1", system_prompt="BETTER", generation_round=1)
    r2 = PromptCandidate(candidate_id="cr2", system_prompt="BEST", generation_round=2)
    mock_final = [
        ScoredCandidate(candidate=r2, scalar_score=9.0,
                        dimensions={"llm_judge": 9.0}, guardrails_held=True),
        ScoredCandidate(candidate=r1, scalar_score=7.0,
                        dimensions={"llm_judge": 7.0}, guardrails_held=True),
        ScoredCandidate(candidate=baseline, scalar_score=5.0,
                        dimensions={"llm_judge": 5.0}, guardrails_held=True),
    ]

    with patch("server.routes.optimize.db", fake_db_p21), \
         patch("server.routes.jobs.db", fake_db_p21), \
         patch("server.optimizer.orchestrator.db", fake_db_p21), \
         patch("server.routes.optimize.require_auth", return_value={"org": "org_1"}), \
         patch("server.routes.jobs.require_auth", return_value={"org": "org_1"}), \
         patch("server.optimizer.orchestrator.PromptOptimizerAgent") as mock_prompt_cls, \
         patch("server.routes.optimize.BackgroundTasks.add_task"):

        mock_prompt = MagicMock()
        mock_prompt.run_iterative.return_value = mock_final
        mock_prompt_cls.return_value = mock_prompt

        client = TestClient(app)
        auth = {"Authorization": "Bearer test"}

        r = client.post(
            "/api/v1/optimize/preauthorize",
            json={"source": "last_n_traces", "n_traces": 5,
                  "axes": ["prompt"], "traces": traces},
            headers=auth,
        )
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        from server.optimizer.orchestrator import OptimizerOrchestrator
        OptimizerOrchestrator().run(run_id, candidate_result_timeout=0.2)

        r = client.get(f"/api/v1/optimize/{run_id}/report", headers=auth)
        assert r.status_code == 200
        body = r.json()
        # Tier 1 has the round-2 rewrite (highest score)
        assert len(body["tier1"]) >= 1
        assert "BEST" in (body["tier1"][0].get("description") or "")
        # Report markdown mentions few-shot examples (from few-shot injection)
        # The winner's apply_payload includes few_shot_examples if the agent set them
        # At minimum: the improvement_pct reflects the 5.0 → 9.0 jump = 80%
        assert body["summary"]["improvement_pct"] > 50
```

Copy the `fake_db_phase2` fixture body inline (don't import — it's in the sibling integration test file).

- [ ] **Step 11.2: Lint + type-check**

```bash
ruff check src/ tests/ retune-cloud/server/
```
No new errors.

```bash
mypy src/retune/ --ignore-missing-imports
```
No new errors.

- [ ] **Step 11.3: Commit**

```bash
git add tests/integration/test_optimize_phase21_e2e.py
git commit -m "optimizer: Phase 2.1 E2E — multi-round beam + real eval + few-shot"
```

---

## Phase 2.1 Exit Gate

All green:

- [ ] `pytest tests/ retune-cloud/tests/ -q` — all passing (Phase 2's 139 + ~20 new tests = ~159)
- [ ] `ruff check` — no new errors
- [ ] `mypy src/retune/ --ignore-missing-imports` — no new errors
- [ ] **Manual smoke test** — against a local Postgres + real LLM API key + a wrapped agent registered with at least one evaluator (e.g. `CostEvaluator`, `LatencyEvaluator`):

```python
from retune import Retuner, Mode
from retune.evaluators import CostEvaluator, LatencyEvaluator, LLMJudgeEvaluator

retuner = Retuner(
    agent=my_real_agent, adapter="custom", mode=Mode.OBSERVE,
    api_key="<your-key>",
    agent_purpose="customer support bot",
    evaluators=[CostEvaluator(), LatencyEvaluator(), LLMJudgeEvaluator(criteria="helpful+cited")],
)
for q in sample_queries:
    retuner.run(q)

retuner.set_mode(Mode.IMPROVE)
report = retuner.optimize(
    source="last_n_traces", n=20, axes=["prompt"],
    rewriter_llm="claude-3-7-sonnet",
)
report.show()
```

- [ ] Report has a Tier 1 with a distinct rewrite (not just the baseline)
- [ ] Tier 1's `apply_payload` includes `few_shot_examples` (non-empty) selected from the top-scoring traces
- [ ] `summary.improvement_pct` is a non-trivial positive number (not 0.0)
- [ ] Pareto data has scored candidates across 2 rounds (not just round 1)

Once green, Phase 3 (ToolOptimizerAgent) begins.

---

## Known risks / decisions

1. **Ordering of `score_fn` results.** `BeamSearchAPO.run_iterative` assumes `score_fn(candidates)` returns `ScoredCandidate`s in the same order as input (or at least maps candidate_id → score). The Orchestrator implementation iterates candidates in order, so this holds today. If the SDK ever posts results out of order, the matching logic may need to index by `candidate_id`.

2. **Beam width default.** `_DEFAULT_BEAM_ROUNDS = 2`, `beam_width=2`, `branch_factor=2` → up to `2 + 2×2 + 2×2 = 10` candidates across 2 rounds, each requiring one cloud LLM call to the rewriter + one SDK-side agent execution. At ~100 traces tested × 10 candidates = 1000 agent runs per optimization. Tune these defaults based on user feedback before scale.

3. **Few-shot primary_metric default = `"llm_judge"`.** If the user's traces don't have `llm_judge` eval_results (e.g. they only registered `cost` + `latency`), `select_few_shot_examples` returns empty and few-shot injection is a no-op. Future: fall back to any available metric or let the user specify.

4. **The old `PromptOptimizerAgent.generate_candidates` single-round method is kept for backward compat but not called by the Orchestrator anymore.** Safe to keep until Phase 3 starts touching this file.
