"""Tests for Retuner -- the core of the system."""


from retune import Mode, Retuner
from retune.core.enums import SuggestionStatus
from retune.core.models import (
    EvalResult,
    ExecutionTrace,
    OptimizationConfig,
    Suggestion,
)
from retune.evaluators.base import BaseEvaluator
from retune.evaluators.latency import LatencyEvaluator


def mock_agent(query: str) -> str:
    return f"Answer: {query}"


class MockEvaluator(BaseEvaluator):
    name = "mock"

    def evaluate(self, trace: ExecutionTrace) -> EvalResult:
        return EvalResult(
            evaluator_name=self.name,
            score=0.85,
            reasoning="Mock evaluation",
        )


class TestRetuner:
    def test_off_mode(self, tmp_path):
        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OFF,
            storage=__import__(
            "retune.storage.sqlite_storage", fromlist=["SQLiteStorage"]
        ).SQLiteStorage(str(tmp_path / "test.db")),
        )
        response = wrapped.run("hello")
        assert response.output == "Answer: hello"
        assert response.trace is None
        assert response.mode == Mode.OFF

    def test_observe_mode(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OBSERVE,
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )
        response = wrapped.run("test query")
        assert response.output == "Answer: test query"
        assert response.trace is not None
        assert response.trace.query == "test query"
        assert len(response.trace.steps) > 0

    def test_evaluate_mode(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.EVALUATE,
            evaluators=[MockEvaluator()],
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )
        response = wrapped.run("test")
        assert len(response.eval_results) == 1
        assert response.eval_results[0].score == 0.85

    def test_improve_mode(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.IMPROVE,
            evaluators=[MockEvaluator()],
            storage=SQLiteStorage(str(tmp_path / "test.db")),
            config=OptimizationConfig(top_k=3),
        )
        response = wrapped.run("test")
        assert response.trace is not None
        # Suggestions may or may not be generated depending on scores

    def test_mode_switching(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OFF,
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )
        assert wrapped.get_mode() == Mode.OFF

        wrapped.set_mode(Mode.OBSERVE)
        assert wrapped.get_mode() == Mode.OBSERVE

        wrapped.set_mode("evaluate")
        assert wrapped.get_mode() == Mode.EVALUATE

    def test_get_traces(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OBSERVE,
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )
        wrapped.run("query 1")
        wrapped.run("query 2")

        traces = wrapped.get_traces()
        assert len(traces) == 2

    def test_eval_summary(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.EVALUATE,
            evaluators=[MockEvaluator(), LatencyEvaluator()],
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )
        wrapped.run("q1")
        wrapped.run("q2")

        summary = wrapped.get_eval_summary()
        assert summary["total_traces"] == 2
        assert "mock" in summary["scores"]
        assert "latency" in summary["scores"]

    def test_run_evaluation_dataset(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OFF,  # Will be switched internally
            evaluators=[MockEvaluator()],
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )

        dataset = [
            {"query": "What is AI?"},
            {"query": "What is ML?"},
        ]
        results = wrapped.run_evaluation_dataset(dataset)
        assert results["total_queries"] == 2
        assert "mock" in results["aggregate_scores"]

    def test_config_management(self, tmp_path):
        from retune.storage.sqlite_storage import SQLiteStorage

        config = OptimizationConfig(top_k=5, temperature=0.3)
        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OFF,
            config=config,
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )

        assert wrapped.get_config().top_k == 5
        wrapped.set_config(OptimizationConfig(top_k=10))
        assert wrapped.get_config().top_k == 10


class TestSuggestionManagement:
    """Tests for the accept/reject/revert suggestion flow."""

    def _make_wrapper(self, tmp_path, config=None):
        from retune.storage.sqlite_storage import SQLiteStorage

        return Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.IMPROVE,
            evaluators=[MockEvaluator()],
            config=config or OptimizationConfig(top_k=3, temperature=0.7),
            storage=SQLiteStorage(str(tmp_path / "test.db")),
        )

    def test_suggestions_start_pending(self, tmp_path):
        """Suggestions from IMPROVE mode should be PENDING."""
        wrapped = self._make_wrapper(tmp_path)
        wrapped.run("test")

        # All suggestions (if any) should be PENDING
        for s in wrapped.get_all_suggestions():
            assert s.status == SuggestionStatus.PENDING

    def test_accept_suggestion(self, tmp_path):
        """Test accepting a suggestion applies it."""
        wrapped = self._make_wrapper(tmp_path)

        # Manually inject a suggestion
        s = Suggestion(
            param_name="top_k",
            old_value=3,
            new_value=7,
            reasoning="Increase retrieval",
            confidence=0.8,
            category="rag",
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        assert wrapped.get_config().top_k == 3
        result = wrapped.accept_suggestion(s.suggestion_id)
        assert result is True
        assert wrapped.get_config().top_k == 7
        assert s.status == SuggestionStatus.ACCEPTED
        assert len(wrapped.get_pending_suggestions()) == 0

    def test_reject_suggestion(self, tmp_path):
        """Test rejecting a suggestion does NOT apply it."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="top_k",
            old_value=3,
            new_value=7,
            reasoning="Increase retrieval",
            confidence=0.8,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        result = wrapped.reject_suggestion(s.suggestion_id)
        assert result is True
        assert wrapped.get_config().top_k == 3  # NOT changed
        assert s.status == SuggestionStatus.REJECTED
        assert len(wrapped.get_pending_suggestions()) == 0

    def test_revert_suggestion(self, tmp_path):
        """Test reverting an accepted suggestion restores old value."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="temperature",
            old_value=0.7,
            new_value=0.2,
            reasoning="Lower temperature",
            confidence=0.9,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        # Accept first
        wrapped.accept_suggestion(s.suggestion_id)
        assert wrapped.get_config().temperature == 0.2

        # Then revert
        result = wrapped.revert_suggestion(s.suggestion_id)
        assert result is True
        assert wrapped.get_config().temperature == 0.7
        assert s.status == SuggestionStatus.REVERTED

    def test_accept_all(self, tmp_path):
        """Test accepting all pending suggestions."""
        wrapped = self._make_wrapper(tmp_path)

        s1 = Suggestion(
            param_name="top_k", old_value=3, new_value=5,
            reasoning="test", confidence=0.8,
        )
        s2 = Suggestion(
            param_name="temperature", old_value=0.7, new_value=0.3,
            reasoning="test", confidence=0.7,
        )
        wrapped._pending_suggestions.extend([s1, s2])
        wrapped._all_suggestions.extend([s1, s2])

        count = wrapped.accept_all()
        assert count == 2
        assert wrapped.get_config().top_k == 5
        assert wrapped.get_config().temperature == 0.3
        assert len(wrapped.get_pending_suggestions()) == 0

    def test_reject_all(self, tmp_path):
        """Test rejecting all pending suggestions."""
        wrapped = self._make_wrapper(tmp_path)

        s1 = Suggestion(
            param_name="top_k", old_value=3, new_value=5,
            reasoning="test", confidence=0.8,
        )
        s2 = Suggestion(
            param_name="temperature", old_value=0.7, new_value=0.3,
            reasoning="test", confidence=0.7,
        )
        wrapped._pending_suggestions.extend([s1, s2])
        wrapped._all_suggestions.extend([s1, s2])

        count = wrapped.reject_all()
        assert count == 2
        assert wrapped.get_config().top_k == 3  # Unchanged
        assert wrapped.get_config().temperature == 0.7  # Unchanged

    def test_revert_all(self, tmp_path):
        """Test reverting all changes restores original config."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="top_k", old_value=3, new_value=10,
            reasoning="test", confidence=0.9,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        wrapped.accept_suggestion(s.suggestion_id)
        assert wrapped.get_config().top_k == 10

        wrapped.revert_all()
        assert wrapped.get_config().top_k == 3
        assert s.status == SuggestionStatus.REVERTED

    def test_cannot_accept_rejected(self, tmp_path):
        """Test cannot accept an already rejected suggestion."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="top_k", old_value=3, new_value=5,
            reasoning="test", confidence=0.8,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        wrapped.reject_suggestion(s.suggestion_id)
        result = wrapped.accept_suggestion(s.suggestion_id)
        assert result is False

    def test_cannot_revert_pending(self, tmp_path):
        """Test cannot revert a suggestion that hasn't been accepted."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="top_k", old_value=3, new_value=5,
            reasoning="test", confidence=0.8,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        result = wrapped.revert_suggestion(s.suggestion_id)
        assert result is False

    def test_suggestion_not_found(self, tmp_path):
        """Test handling of non-existent suggestion ID."""
        wrapped = self._make_wrapper(tmp_path)
        assert wrapped.accept_suggestion("nonexistent") is False
        assert wrapped.reject_suggestion("nonexistent") is False
        assert wrapped.revert_suggestion("nonexistent") is False

    def test_get_suggestion_by_id(self, tmp_path):
        """Test retrieving a specific suggestion by ID."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="top_k", old_value=3, new_value=5,
            reasoning="test", confidence=0.8,
        )
        wrapped._all_suggestions.append(s)

        found = wrapped.get_suggestion(s.suggestion_id)
        assert found is not None
        assert found.param_name == "top_k"

        not_found = wrapped.get_suggestion("bad-id")
        assert not_found is None

    def test_improvement_history_tracks_actions(self, tmp_path):
        """Test that improvement history records accept and revert actions."""
        wrapped = self._make_wrapper(tmp_path)

        s = Suggestion(
            param_name="top_k", old_value=3, new_value=8,
            reasoning="test", confidence=0.9,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        wrapped.accept_suggestion(s.suggestion_id)
        history = wrapped.get_improvement_history()
        assert len(history) >= 1
        assert history[-1]["action"] == "accept"
        assert history[-1]["param_name"] == "top_k"

        wrapped.revert_suggestion(s.suggestion_id)
        history = wrapped.get_improvement_history()
        assert history[-1]["action"] == "revert"

    def test_version_increments(self, tmp_path):
        """Test that version increments on accept and revert."""
        wrapped = self._make_wrapper(tmp_path)
        initial_version = wrapped.version

        s = Suggestion(
            param_name="top_k", old_value=3, new_value=5,
            reasoning="test", confidence=0.8,
        )
        wrapped._pending_suggestions.append(s)
        wrapped._all_suggestions.append(s)

        wrapped.accept_suggestion(s.suggestion_id)
        assert wrapped.version == initial_version + 1

        wrapped.revert_suggestion(s.suggestion_id)
        assert wrapped.version == initial_version + 2

    def test_auto_improve_uses_confidence_threshold(self, tmp_path):
        """Test auto_improve only applies high-confidence suggestions."""
        from retune.storage.sqlite_storage import SQLiteStorage

        wrapped = Retuner(
            agent=mock_agent,
            adapter="custom",
            mode=Mode.OFF,
            config=OptimizationConfig(top_k=3),
            storage=SQLiteStorage(str(tmp_path / "test.db")),
            auto_improve=True,
        )

        low_conf = Suggestion(
            param_name="top_k", old_value=3, new_value=10,
            reasoning="test", confidence=0.3,
        )
        high_conf = Suggestion(
            param_name="temperature", old_value=None, new_value=0.2,
            reasoning="test", confidence=0.8,
        )

        wrapped._auto_apply_suggestions([low_conf, high_conf])

        # Low confidence should NOT be applied
        assert wrapped.get_config().top_k == 3
        # High confidence should be applied
        assert wrapped.get_config().temperature == 0.2
