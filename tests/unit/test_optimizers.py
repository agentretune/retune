"""Tests for optimizers."""

from retune.core.models import EvalResult, ExecutionTrace, OptimizationConfig
from retune.optimizers.basic import BasicOptimizer


class TestBasicOptimizer:
    def test_no_traces(self):
        opt = BasicOptimizer()
        suggestions = opt.suggest([], OptimizationConfig())
        assert suggestions == []

    def test_low_retrieval_suggests_top_k(self):
        opt = BasicOptimizer(score_threshold=0.7)
        traces = [
            ExecutionTrace(
                query="test",
                eval_results=[
                    EvalResult(
                        evaluator_name="retrieval",
                        score=0.3,
                        details={"retrieval": 0.3},
                    )
                ],
            )
        ]
        config = OptimizationConfig(top_k=3)
        suggestions = opt.suggest(traces, config)

        param_names = [s.param_name for s in suggestions]
        assert "top_k" in param_names
        assert "use_reranker" in param_names

    def test_high_scores_no_suggestions(self):
        opt = BasicOptimizer(score_threshold=0.7)
        traces = [
            ExecutionTrace(
                query="test",
                eval_results=[
                    EvalResult(evaluator_name="llm_judge", score=0.95),
                    EvalResult(evaluator_name="retrieval", score=0.9),
                    EvalResult(evaluator_name="latency", score=0.85),
                    EvalResult(evaluator_name="cost", score=0.9),
                ],
            )
        ]
        suggestions = opt.suggest(traces, OptimizationConfig())
        assert len(suggestions) == 0
