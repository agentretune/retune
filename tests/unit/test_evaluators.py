"""Tests for evaluators."""

from datetime import datetime, timedelta, timezone

from retune.core.enums import StepType
from retune.core.models import ExecutionTrace, Step, TokenUsage
from retune.evaluators.cost import CostEvaluator
from retune.evaluators.latency import LatencyEvaluator
from retune.evaluators.retrieval import RetrievalEvaluator


class TestLatencyEvaluator:
    def test_fast_execution(self):
        now = datetime.now(timezone.utc)
        trace = ExecutionTrace(
            query="test",
            started_at=now,
            ended_at=now + timedelta(milliseconds=500),
        )
        result = LatencyEvaluator().evaluate(trace)
        assert result.score == 1.0

    def test_slow_execution(self):
        now = datetime.now(timezone.utc)
        trace = ExecutionTrace(
            query="test",
            started_at=now,
            ended_at=now + timedelta(seconds=15),
        )
        result = LatencyEvaluator().evaluate(trace)
        assert result.score == 0.0

    def test_medium_execution(self):
        now = datetime.now(timezone.utc)
        trace = ExecutionTrace(
            query="test",
            started_at=now,
            ended_at=now + timedelta(seconds=5),
        )
        result = LatencyEvaluator().evaluate(trace)
        assert 0.0 < result.score < 1.0


class TestRetrievalEvaluator:
    def test_no_retrieval_steps(self):
        trace = ExecutionTrace(query="test")
        result = RetrievalEvaluator().evaluate(trace)
        assert result.score == 1.0  # No retrieval = not penalized

    def test_successful_retrieval(self):
        trace = ExecutionTrace(
            query="test",
            response="Machine learning is great",
            steps=[
                Step(
                    step_type=StepType.RETRIEVAL,
                    name="retriever",
                    output_data={
                        "num_docs": 3,
                        "documents": [
                            {"content": "Machine learning is a field of AI", "metadata": {}},
                        ],
                    },
                )
            ],
        )
        result = RetrievalEvaluator().evaluate(trace)
        assert result.score > 0.5


class TestCostEvaluator:
    def test_low_cost(self):
        trace = ExecutionTrace(
            query="test",
            steps=[
                Step(
                    step_type=StepType.LLM_CALL,
                    name="llm",
                    token_usage=TokenUsage(total_tokens=200),
                )
            ],
        )
        result = CostEvaluator().evaluate(trace)
        assert result.score == 1.0

    def test_high_cost(self):
        trace = ExecutionTrace(
            query="test",
            steps=[
                Step(
                    step_type=StepType.LLM_CALL,
                    name="llm",
                    token_usage=TokenUsage(total_tokens=15000),
                )
            ],
        )
        result = CostEvaluator().evaluate(trace)
        assert result.score == 0.0
