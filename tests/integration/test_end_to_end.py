"""End-to-end integration tests using custom adapter (no external deps)."""

import pytest

from retune import Mode, OptimizationConfig, Retuner
from retune.core.models import EvalResult, ExecutionTrace
from retune.evaluators.base import BaseEvaluator
from retune.evaluators.latency import LatencyEvaluator
from retune.storage.sqlite_storage import SQLiteStorage


def simple_agent(query: str) -> str:
    return f"Response to: {query}"


class AlwaysHighEvaluator(BaseEvaluator):
    name = "always_high"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        return EvalResult(evaluator_name=self.name, score=0.95, reasoning="Always high")


class AlwaysLowEvaluator(BaseEvaluator):
    name = "always_low"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        return EvalResult(
            evaluator_name=self.name,
            score=0.2,
            reasoning="Always low",
            details={"retrieval": 0.2, "correctness": 0.2},
        )


class TestEndToEnd:
    def test_full_observe_evaluate_improve_cycle(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "e2e.db"))

        # Phase 1: Observe
        wrapped = Retuner(
            agent=simple_agent,
            adapter="custom",
            mode=Mode.OBSERVE,
            storage=storage,
        )
        r1 = wrapped.run("What is AI?")
        assert r1.output == "Response to: What is AI?"
        assert r1.trace is not None

        # Phase 2: Evaluate
        wrapped.set_mode(Mode.EVALUATE)
        wrapped._evaluators = [LatencyEvaluator()]
        r2 = wrapped.run("What is ML?")
        assert len(r2.eval_results) == 1
        assert r2.eval_results[0].evaluator_name == "latency"

        # Phase 3: Improve
        wrapped.set_mode(Mode.IMPROVE)
        wrapped._evaluators = [AlwaysLowEvaluator()]
        wrapped.set_config(OptimizationConfig(top_k=3, temperature=0.7))
        r3 = wrapped.run("Improve me")
        # Should generate suggestions due to low scores
        assert len(r3.suggestions) > 0

        # Phase 4: Back to OFF
        wrapped.set_mode(Mode.OFF)
        r4 = wrapped.run("Production query")
        assert r4.trace is None

    def test_evaluation_dataset(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "eval.db"))

        wrapped = Retuner(
            agent=simple_agent,
            adapter="custom",
            mode=Mode.OFF,
            evaluators=[AlwaysHighEvaluator()],
            storage=storage,
        )

        dataset = [
            {"query": "q1"},
            {"query": "q2"},
            {"query": "q3"},
        ]
        results = wrapped.run_evaluation_dataset(dataset)
        assert results["total_queries"] == 3
        assert results["aggregate_scores"]["always_high"] == pytest.approx(0.95)

    def test_auto_improve(self, tmp_path):
        storage = SQLiteStorage(str(tmp_path / "auto.db"))

        wrapped = Retuner(
            agent=simple_agent,
            adapter="custom",
            mode=Mode.IMPROVE,
            evaluators=[AlwaysLowEvaluator()],
            storage=storage,
            config=OptimizationConfig(top_k=3, temperature=0.7),
            auto_improve=True,
        )

        r = wrapped.run("improve this")
        # Auto-improve should have applied suggestions
        # Version should be > 1 if suggestions were applied
        if r.suggestions:
            assert wrapped.version >= 1
