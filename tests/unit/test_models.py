"""Tests for core data models."""

from retune.core.enums import Mode, StepType
from retune.core.models import (
    EvalResult,
    ExecutionTrace,
    OptimizationConfig,
    Step,
    Suggestion,
    TokenUsage,
    WrapperResponse,
)


class TestStep:
    def test_create_step(self):
        step = Step(
            step_type=StepType.LLM_CALL,
            name="test_llm",
            input_data={"prompt": "hello"},
            output_data={"response": "world"},
        )
        assert step.step_type == StepType.LLM_CALL
        assert step.name == "test_llm"
        assert step.step_id  # auto-generated

    def test_step_with_token_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        step = Step(
            step_type=StepType.LLM_CALL,
            name="test",
            token_usage=usage,
        )
        assert step.token_usage.total_tokens == 30


class TestExecutionTrace:
    def test_create_trace(self):
        trace = ExecutionTrace(query="test query", response="test response")
        assert trace.query == "test query"
        assert trace.trace_id
        assert trace.mode == Mode.OBSERVE

    def test_weighted_score_empty(self):
        trace = ExecutionTrace(query="test")
        assert trace.weighted_score is None

    def test_weighted_score(self):
        trace = ExecutionTrace(
            query="test",
            eval_results=[
                EvalResult(evaluator_name="a", score=0.8),
                EvalResult(evaluator_name="b", score=0.6),
            ],
        )
        assert trace.weighted_score == pytest.approx(0.7)

    def test_total_tokens(self):
        trace = ExecutionTrace(
            query="test",
            steps=[
                Step(
                    step_type=StepType.LLM_CALL,
                    name="s1",
                    token_usage=TokenUsage(total_tokens=100),
                ),
                Step(
                    step_type=StepType.LLM_CALL,
                    name="s2",
                    token_usage=TokenUsage(total_tokens=200),
                ),
            ],
        )
        assert trace.total_tokens == 300

    def test_get_score(self):
        trace = ExecutionTrace(
            query="test",
            eval_results=[
                EvalResult(evaluator_name="llm_judge", score=0.9),
            ],
        )
        assert trace.get_score("llm_judge") == 0.9
        assert trace.get_score("nonexistent") is None


class TestOptimizationConfig:
    def test_to_flat_dict_excludes_none(self):
        config = OptimizationConfig(top_k=5, temperature=0.3)
        flat = config.to_flat_dict()
        assert flat == {"top_k": 5, "temperature": 0.3, "custom_params": {}}

    def test_default_config_empty(self):
        config = OptimizationConfig()
        flat = config.to_flat_dict()
        assert flat == {"custom_params": {}}


class TestSuggestion:
    def test_create_suggestion(self):
        s = Suggestion(
            param_name="top_k",
            old_value=3,
            new_value=5,
            reasoning="Improve retrieval",
            confidence=0.8,
            category="rag",
        )
        assert s.param_name == "top_k"
        assert s.confidence == 0.8


import pytest
