"""Tests for Deep Agent v2 evaluator and optimizer with mocked deepagents."""

from unittest.mock import patch

from retune.core.enums import StepType
from retune.core.models import (
    EvalResult,
    ExecutionTrace,
    Span,
    Step,
)


class TestSpanModel:
    """Tests for the new Span model."""

    def test_span_creation(self):
        span = Span(
            step_id="step-1",
            step_type=StepType.LLM_CALL,
            name="generate_answer",
            contribution_score=0.8,
            is_bottleneck=False,
            reasoning="Key generation step",
        )
        assert span.step_id == "step-1"
        assert span.contribution_score == 0.8
        assert not span.is_bottleneck
        assert span.span_id  # auto-generated

    def test_span_bottleneck(self):
        span = Span(
            step_id="step-2",
            step_type=StepType.RETRIEVAL,
            name="retrieve_docs",
            contribution_score=-0.5,
            is_bottleneck=True,
            reasoning="Poor retrieval quality caused failure",
        )
        assert span.is_bottleneck
        assert span.contribution_score == -0.5


class TestEvaluatorDeepAgent:
    """Tests for the EvaluatorDeepAgent with fallback mode."""

    def _make_trace(self, score: float = 0.7) -> ExecutionTrace:
        return ExecutionTrace(
            query="What is Python?",
            response="Python is a programming language.",
            steps=[
                Step(
                    step_type=StepType.RETRIEVAL,
                    name="retrieve_docs",
                    input_data={"query": "What is Python?"},
                    output_data={"documents": [{"content": "Python is a language"}]},
                ),
                Step(
                    step_type=StepType.LLM_CALL,
                    name="generate_answer",
                    input_data={"prompt": "Answer the question"},
                    output_data={"response": "Python is a programming language."},
                ),
            ],
            eval_results=[
                EvalResult(evaluator_name="test", score=score),
            ],
        )

    def test_heuristic_fallback(self):
        """Test that evaluator works with pure heuristic when no LLM available."""
        from retune.agents.evaluator.agent import EvaluatorDeepAgent

        evaluator = EvaluatorDeepAgent(model="gpt-4o-mini")
        # Force heuristic mode
        evaluator._use_deep_agents = False

        trace = self._make_trace()

        # Mock _build_graph to simulate LangGraph not available
        with patch.object(evaluator, "_build_graph", side_effect=ImportError("no langgraph")):
            result = evaluator.evaluate(trace)

        assert isinstance(result, EvalResult)
        assert 0.0 <= result.score <= 1.0
        assert result.evaluator_name == "deep_evaluator"
        assert "heuristic" in result.details.get("mode", "").lower() or result.reasoning

    def test_supervisor_routing(self):
        """Test supervisor correctly routes based on step types."""
        from retune.agents.evaluator.agent import supervisor_node

        # Trace with tool calls and LLM calls
        state = {
            "trace": {
                "steps": [
                    {"step_type": "tool_call", "name": "search"},
                    {"step_type": "llm_call", "name": "generate"},
                    {"step_type": "retrieval", "name": "retrieve"},
                ],
            }
        }
        result = supervisor_node(state)
        assert "trace_analyzer" in result["subagents_to_run"]
        assert "credit_assigner" in result["subagents_to_run"]
        assert "tool_auditor" in result["subagents_to_run"]
        assert "hallucination_detector" in result["subagents_to_run"]

    def test_supervisor_no_tools(self):
        """Test supervisor skips tool auditor when no tool calls."""
        from retune.agents.evaluator.agent import supervisor_node

        state = {
            "trace": {
                "steps": [
                    {"step_type": "llm_call", "name": "generate"},
                ],
            }
        }
        result = supervisor_node(state)
        assert "tool_auditor" not in result["subagents_to_run"]
        assert "hallucination_detector" in result["subagents_to_run"]

    def test_trace_analyzer_node(self):
        """Test trace analyzer node produces analysis."""
        from retune.agents.evaluator.agent import trace_analyzer_node

        state = {
            "trace": {
                "steps": [
                    {
                        "step_type": "retrieval",
                        "name": "retrieve",
                        "started_at": "2024-01-01T00:00:00Z",
                        "ended_at": "2024-01-01T00:00:01Z",
                        "token_usage": {"total_tokens": 100},
                    },
                ],
                "query": "test",
                "response": "test response",
            },
            "subagents_completed": [],
        }
        result = trace_analyzer_node(state)
        assert "trace_analysis" in result
        assert "trace_analyzer" in result["subagents_completed"]

    def test_tool_auditor_no_tools(self):
        """Test tool auditor with no tool calls gives score 1.0."""
        from retune.agents.evaluator.agent import tool_auditor_node

        state = {
            "trace": {
                "steps": [],
                "query": "test",
                "response": "test",
            },
            "subagents_completed": [],
        }
        result = tool_auditor_node(state)
        assert result["tool_audit"]["score"] == 1.0

    def test_route_next(self):
        """Test routing logic."""
        from retune.agents.evaluator.agent import _route_next

        state = {
            "subagents_to_run": ["trace_analyzer", "credit_assigner"],
            "subagents_completed": ["trace_analyzer"],
        }
        assert _route_next(state) == "credit_assigner"

        state["subagents_completed"] = ["trace_analyzer", "credit_assigner"]
        assert _route_next(state) == "synthesizer"


class TestOptimizerDeepAgent:
    """Tests for the OptimizerDeepAgent with fallback mode."""

    def _make_traces(self, avg_score: float = 0.5) -> list[ExecutionTrace]:
        return [
            ExecutionTrace(
                query="What is ML?",
                response="ML is machine learning",
                steps=[
                    Step(
                        step_type=StepType.LLM_CALL,
                        name="generate",
                        output_data={"response": "ML is machine learning"},
                    ),
                ],
                eval_results=[
                    EvalResult(evaluator_name="deep_evaluator", score=avg_score),
                    EvalResult(
                        evaluator_name="correctness",
                        score=avg_score,
                        details={"correctness": avg_score},
                    ),
                ],
            )
        ]

    def test_planner_selects_strategies(self):
        """Test planner selects APO when quality is low."""
        from retune.agents.optimizer.agent import planner_node

        state = {
            "traces": [
                {
                    "eval_results": [
                        {"evaluator_name": "deep_evaluator", "score": 0.4, "details": {}},
                        {
                            "evaluator_name": "correctness",
                            "score": 0.3,
                            "details": {"correctness": 0.3},
                        },
                    ],
                    "steps": [],
                }
            ],
            "current_config": {},
        }
        result = planner_node(state)
        assert "apo" in result["strategies_to_run"]

    def test_planner_no_strategies_high_scores(self):
        """Test planner selects nothing when scores are high."""
        from retune.agents.optimizer.agent import planner_node

        state = {
            "traces": [
                {
                    "eval_results": [
                        {"evaluator_name": "deep_evaluator", "score": 0.95, "details": {}},
                    ],
                    "steps": [],
                }
            ],
            "current_config": {},
        }
        result = planner_node(state)
        # Either no strategies or just config_tuner as a safety net
        assert "apo" not in result["strategies_to_run"]

    def test_planner_empty_traces(self):
        """Test planner handles empty traces."""
        from retune.agents.optimizer.agent import planner_node

        state = {"traces": [], "current_config": {}}
        result = planner_node(state)
        assert result["strategies_to_run"] == []

    def test_config_tuner_low_retrieval(self):
        """Test config tuner suggests top_k increase for low retrieval."""
        from retune.agents.optimizer.agent import config_tuner_node

        state = {
            "traces": [],
            "current_config": {"top_k": 3},
            "analysis_summary": {
                "avg_scores": {"retrieval": 0.4, "correctness": 0.8},
            },
            "model": "gpt-4o-mini",
            "strategies_completed": [],
        }
        result = config_tuner_node(state)
        param_names = [s["param_name"] for s in result["config_suggestions"]]
        assert "top_k" in param_names

    def test_tool_curator_wasteful_tools(self):
        """Test tool curator identifies wasteful tools."""
        from retune.agents.optimizer.agent import tool_curator_node

        state = {
            "traces": [
                {
                    "steps": [
                        {
                            "step_type": "tool_call",
                            "name": "bad_tool",
                            "output_data": {"output": "irrelevant data xyz123"},
                        },
                        {
                            "step_type": "tool_call",
                            "name": "bad_tool",
                            "output_data": {"output": "more irrelevant data abc456"},
                        },
                    ],
                    "response": "The answer is 42",
                }
            ],
            "current_config": {},
            "strategies_completed": [],
        }
        result = tool_curator_node(state)
        assert len(result["tool_suggestions"]) > 0

    def test_aggregator_deduplicates(self):
        """Test aggregator keeps highest confidence per param."""
        from retune.agents.optimizer.agent import aggregator_node

        state = {
            "apo_rewritten_prompt": "new prompt",
            "apo_confidence": 0.8,
            "apo_critique": "needs improvement",
            "current_config": {"system_prompt": "old prompt"},
            "config_suggestions": [
                {"param_name": "top_k", "old_value": 3, "new_value": 5,
                 "reasoning": "test", "confidence": 0.9, "category": "rag"},
                {"param_name": "top_k", "old_value": 3, "new_value": 7,
                 "reasoning": "test2", "confidence": 0.6, "category": "rag"},
            ],
            "tool_suggestions": [],
            "beam_search_result": None,
        }
        result = aggregator_node(state)
        # Should have system_prompt + one top_k (deduplicated)
        param_names = [s["param_name"] for s in result["final_suggestions"]]
        assert param_names.count("top_k") == 1

    def test_aggregator_includes_beam_result(self):
        """Test aggregator includes beam search result when present."""
        from retune.agents.optimizer.agent import aggregator_node

        state = {
            "apo_rewritten_prompt": "",
            "apo_confidence": 0.0,
            "apo_critique": "",
            "current_config": {"system_prompt": "old"},
            "config_suggestions": [],
            "tool_suggestions": [],
            "beam_search_result": {
                "best_prompt": "beam optimized prompt",
                "best_score": 0.85,
                "improvement": 0.2,
                "candidates_explored": 8,
                "rounds_completed": 2,
                "verified": True,
            },
        }
        result = aggregator_node(state)
        prompts = [s for s in result["final_suggestions"] if s["param_name"] == "system_prompt"]
        assert len(prompts) == 1
        assert "Beam Search APO" in prompts[0]["reasoning"]

    def test_route_next_strategy(self):
        """Test strategy routing."""
        from retune.agents.optimizer.agent import _route_next_strategy

        state = {
            "strategies_to_run": ["apo", "config_tuner"],
            "strategies_completed": [],
        }
        assert _route_next_strategy(state) == "apo_evaluate"

        state["strategies_completed"] = ["apo"]
        assert _route_next_strategy(state) == "config_tuner"

        state["strategies_completed"] = ["apo", "config_tuner"]
        assert _route_next_strategy(state) == "aggregator"


class TestSubagentDefinitions:
    """Test subagent definitions are well-formed."""

    def test_evaluator_subagent_definitions(self):
        from retune.agents.evaluator.subagents.definitions import (
            get_all_evaluator_subagent_definitions,
        )

        defs = get_all_evaluator_subagent_definitions()
        assert len(defs) == 4

        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "system_prompt" in d
            assert "tools" in d
            assert len(d["tools"]) > 0
            assert len(d["system_prompt"]) > 50

    def test_optimizer_subagent_definitions(self):
        from retune.agents.optimizer.subagents.definitions import (
            get_all_optimizer_subagent_definitions,
        )

        defs = get_all_optimizer_subagent_definitions()
        assert len(defs) == 4

        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "system_prompt" in d
            assert "tools" in d
            assert len(d["tools"]) > 0

    def test_evaluator_subagent_names(self):
        from retune.agents.evaluator.subagents.definitions import (
            get_all_evaluator_subagent_definitions,
        )

        names = [d["name"] for d in get_all_evaluator_subagent_definitions()]
        assert "trace-analyzer" in names
        assert "credit-assigner" in names
        assert "tool-auditor" in names
        assert "hallucination-detector" in names

    def test_optimizer_subagent_names(self):
        from retune.agents.optimizer.subagents.definitions import (
            get_all_optimizer_subagent_definitions,
        )

        names = [d["name"] for d in get_all_optimizer_subagent_definitions()]
        assert "prompt-critic" in names
        assert "prompt-rewriter" in names
        assert "config-tuner" in names
        assert "tool-curator" in names
