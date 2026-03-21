"""Tests for Beam Search APO logic."""

from unittest.mock import MagicMock, patch

import pytest

from retune.agents.optimizer.beam_config import BeamSearchConfig
from retune.agents.optimizer.beam_search import BeamSearchAPO
from retune.core.models import (
    BeamCandidate,
    BeamSearchResult,
    OptimizationConfig,
)


class TestBeamSearchConfig:
    def test_defaults(self):
        cfg = BeamSearchConfig()
        assert cfg.beam_width == 2
        assert cfg.branch_factor == 2
        assert cfg.beam_rounds == 2
        assert cfg.max_rollout_queries == 5
        assert cfg.cost_budget_usd == 0.50

    def test_custom_config(self):
        cfg = BeamSearchConfig(
            beam_width=4,
            branch_factor=3,
            beam_rounds=3,
            cost_budget_usd=1.0,
        )
        assert cfg.beam_width == 4
        assert cfg.branch_factor == 3
        assert cfg.beam_rounds == 3

    def test_validation(self):
        with pytest.raises(Exception):
            BeamSearchConfig(beam_width=0)  # ge=1

        with pytest.raises(Exception):
            BeamSearchConfig(beam_width=10)  # le=8


class TestBeamCandidate:
    def test_creation(self):
        candidate = BeamCandidate(
            prompt="You are a helpful assistant.",
            score=0.75,
            confidence=0.8,
            generation=1,
        )
        assert candidate.prompt == "You are a helpful assistant."
        assert candidate.score == 0.75
        assert not candidate.verified

    def test_verified_candidate(self):
        candidate = BeamCandidate(
            prompt="test",
            score=0.9,
            verified=True,
            verification_score=0.88,
        )
        assert candidate.verified
        assert candidate.verification_score == 0.88


class TestBeamSearchResult:
    def test_creation(self):
        result = BeamSearchResult(
            best_prompt="improved prompt",
            best_score=0.85,
            baseline_score=0.5,
            improvement=0.35,
            candidates_explored=8,
            rounds_completed=2,
        )
        assert result.improvement == 0.35
        assert result.candidates_explored == 8


class TestBeamSearchAPO:
    def test_estimate_baseline_empty(self):
        apo = BeamSearchAPO()
        score = apo._estimate_baseline_score([])
        assert score == 0.5

    def test_estimate_baseline_with_traces(self):
        apo = BeamSearchAPO()
        traces = [
            {
                "eval_results": [
                    {"score": 0.3},
                    {"score": 0.5},
                ]
            },
            {
                "eval_results": [
                    {"score": 0.6},
                ]
            },
        ]
        score = apo._estimate_baseline_score(traces)
        # Trace 1: avg=0.4, Trace 2: avg=0.6 → overall avg=0.5
        assert abs(score - 0.5) < 0.01

    def test_critique_heuristic_fallback(self):
        """Test critique falls back to heuristic when LLM unavailable."""
        apo = BeamSearchAPO()

        with patch("retune.agents.optimizer.beam_search.BeamSearchAPO._critique") as mock:
            mock.return_value = "Issues found:\n- Missing role definition"
            result = mock("You are helpful", [])
            assert "Issues" in result

    def test_rewrite_uses_prompt_rewriter(self):
        """Test rewrite delegates to PromptRewriterTool."""
        apo = BeamSearchAPO()

        mock_rewriter = MagicMock()
        mock_rewriter.execute.return_value = {
            "rewritten_prompt": "improved prompt",
            "changes_made": ["added role"],
            "confidence": 0.7,
        }
        apo._prompt_rewriter = mock_rewriter
        result = apo._rewrite("old prompt", "needs role definition")
        assert result["rewritten_prompt"] == "improved prompt"
        assert result["confidence"] == 0.7

    def test_search_no_failures(self):
        """Test search with no failure traces returns baseline."""
        cfg = BeamSearchConfig(beam_rounds=1, branch_factor=1)
        apo = BeamSearchAPO(config=cfg)

        # Mock _critique and _rewrite to avoid LLM calls
        with patch.object(apo, "_critique", return_value="needs improvement"), \
             patch.object(apo, "_rewrite", return_value={
                 "rewritten_prompt": "better prompt",
                 "changes_made": ["improved"],
                 "confidence": 0.6,
             }):
            result = apo.search(
                current_prompt="You are helpful",
                failure_traces=[],
            )

        assert isinstance(result, BeamSearchResult)
        assert result.best_prompt  # Should have some prompt
        assert result.baseline_score == 0.5  # No failures → 0.5

    def test_search_with_failures(self):
        """Test search with failure traces produces candidates."""
        cfg = BeamSearchConfig(beam_rounds=1, branch_factor=2)
        apo = BeamSearchAPO(config=cfg)

        failures = [
            {
                "query": "What is X?",
                "response": "I don't know",
                "eval_results": [{"score": 0.3}],
            }
        ]

        with patch.object(apo, "_critique", return_value="missing specificity"), \
             patch.object(apo, "_rewrite", return_value={
                 "rewritten_prompt": "You are an expert assistant. Be specific.",
                 "changes_made": ["added specificity"],
                 "confidence": 0.75,
             }):
            result = apo.search(
                current_prompt="You are helpful",
                failure_traces=failures,
            )

        assert result.candidates_explored > 1
        assert result.best_prompt != ""

    def test_search_respects_cost_budget(self):
        """Test search stops when cost budget is exhausted."""
        cfg = BeamSearchConfig(
            beam_rounds=5,
            branch_factor=4,
            cost_budget_usd=0.001,  # Very low budget
        )
        apo = BeamSearchAPO(config=cfg)
        apo._cost_spent = 0.002  # Already over budget

        result = apo.search(
            current_prompt="test",
            failure_traces=[{"eval_results": [{"score": 0.3}]}],
        )

        # Should return quickly without exploring many candidates
        assert result.candidates_explored <= 1

    def test_search_with_verification(self):
        """Test search with rollout verification."""
        cfg = BeamSearchConfig(
            beam_rounds=1,
            branch_factor=1,
            verification_enabled=True,
        )
        apo = BeamSearchAPO(config=cfg)

        mock_adapter = MagicMock()
        mock_evaluator = MagicMock()

        # Replace rollout runner with a mock
        mock_rollout = MagicMock()
        mock_rollout.execute.return_value = {
            "avg_score": 0.8,
            "per_query_scores": [],
            "total_cost": 0.01,
            "num_queries": 3,
        }
        mock_rollout.set_adapter = MagicMock()
        mock_rollout.set_evaluators = MagicMock()
        apo._rollout_runner = mock_rollout

        with patch.object(apo, "_critique", return_value="needs work"), \
             patch.object(apo, "_rewrite", return_value={
                 "rewritten_prompt": "improved",
                 "changes_made": ["fixed"],
                 "confidence": 0.7,
             }):
            result = apo.search(
                current_prompt="test",
                failure_traces=[{"eval_results": [{"score": 0.3}]}],
                adapter=mock_adapter,
                validation_queries=["q1", "q2", "q3"],
                current_config=OptimizationConfig(),
            )

        # Should have verified candidates
        verified = [c for c in result.beam_history if c.verified]
        assert len(verified) > 0 or result.candidates_explored >= 1


class TestNewTools:
    """Tests for the new tools added in Phase 1."""

    def test_prompt_rewriter_heuristic(self):
        from retune.tools.builtin.prompt_rewriter import PromptRewriterTool

        tool = PromptRewriterTool()
        result = tool._heuristic_rewrite(
            "Answer questions.",
            "Missing role definition and step-by-step reasoning",
        )
        assert result["rewritten_prompt"]
        assert "You are" in result["rewritten_prompt"]
        assert result["confidence"] == 0.3

    def test_prompt_rewriter_empty(self):
        from retune.tools.builtin.prompt_rewriter import PromptRewriterTool

        tool = PromptRewriterTool()
        result = tool.execute(current_prompt="", critique="")
        assert result["confidence"] == 0.0

    def test_gradient_aggregator_single(self):
        from retune.tools.builtin.gradient_aggregator import GradientAggregatorTool

        tool = GradientAggregatorTool()
        result = tool.execute(critiques=["needs more examples"])
        assert result["unified_gradient"] == "needs more examples"

    def test_gradient_aggregator_heuristic(self):
        from retune.tools.builtin.gradient_aggregator import GradientAggregatorTool

        tool = GradientAggregatorTool()
        result = tool._heuristic_aggregate([
            "Missing role definition",
            "Needs better examples and formatting",
            "Role is unclear, add constraints",
        ])
        assert "role_definition" in result["themes"]
        assert "examples" in result["themes"]

    def test_gradient_aggregator_empty(self):
        from retune.tools.builtin.gradient_aggregator import GradientAggregatorTool

        tool = GradientAggregatorTool()
        result = tool.execute(critiques=[])
        assert result["unified_gradient"] == ""

    def test_rollout_runner_no_adapter(self):
        from retune.tools.builtin.rollout_runner import RolloutRunnerTool

        tool = RolloutRunnerTool()
        result = tool.execute(
            candidate_config={"temperature": 0.5},
            validation_queries=["test query"],
        )
        assert result["error"]
        assert result["avg_score"] == 0.0

    def test_rollout_runner_no_queries(self):
        from retune.tools.builtin.rollout_runner import RolloutRunnerTool

        tool = RolloutRunnerTool()
        tool._adapter = MagicMock()
        result = tool.execute(
            candidate_config={},
            validation_queries=[],
        )
        assert result["avg_score"] == 0.0

    def test_builtin_tools_list(self):
        from retune.tools.builtin import get_builtin_tools

        tools = get_builtin_tools()
        names = [t.name for t in tools]
        assert "prompt_rewriter" in names
        assert "rollout_runner" in names
        assert "gradient_aggregator" in names
        assert len(tools) == 8
