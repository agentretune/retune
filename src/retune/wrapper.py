"""Retuner -- the main user-facing class.

This is the control plane that wraps any agent/RAG system and orchestrates
observation, evaluation, and optimization based on the selected mode.

User flow:
  1. Wrap agent -> run in OBSERVE mode (captures traces)
  2. Switch to EVALUATE mode -> get eval scores
  3. Switch to IMPROVE mode -> get suggestions (PENDING)
  4. Review suggestions: accept_suggestion() / reject_suggestion()
  5. Accepted suggestions are applied to config
  6. Verify improvement by running more queries in EVALUATE mode
  7. Satisfied? Keep changes. Not satisfied? revert_suggestion()
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from retune.adapters import get_adapter
from retune.adapters.base import BaseAdapter
from retune.agents.optimizer.beam_config import BeamSearchConfig
from retune.config import settings
from retune.core.enums import Mode, SuggestionStatus
from retune.core.models import (
    EvalResult,
    ExecutionTrace,
    OptimizationConfig,
    Suggestion,
    WrapperResponse,
)
from retune.evaluators import get_evaluator
from retune.evaluators.base import BaseEvaluator
from retune.memory.store import MemoryStore
from retune.optimizers.base import BaseOptimizer
from retune.optimizers.basic import BasicOptimizer
from retune.storage.base import BaseStorage
from retune.storage.sqlite_storage import SQLiteStorage

logger = logging.getLogger(__name__)


def _create_optimizer(
    use_deep: bool,
    model: str,
    beam_config: BeamSearchConfig | None,
) -> BaseOptimizer:
    """Create the appropriate optimizer based on available dependencies.

    Priority:
    1. OptimizerDeepAgent with beam_config (if deepagents or langgraph available)
    2. BasicOptimizer (always works, no deps)
    """
    if use_deep:
        try:
            from retune.agents.optimizer.agent import OptimizerDeepAgent
            return OptimizerDeepAgent(model=model, beam_config=beam_config)
        except ImportError:
            logger.debug("Deep optimizer deps not available, using BasicOptimizer")
    return BasicOptimizer()


class Retuner:
    """Universal wrapper that makes any agent/RAG system self-improving.

    Think of the mode as a fan regulator:
    - OFF: pure passthrough, zero overhead
    - OBSERVE: capture traces (cheap)
    - EVALUATE: capture + score (medium)
    - IMPROVE: capture + score + optimize (heavy, user-triggered)

    Suggestions are generated in IMPROVE mode but NOT auto-applied.
    The user reviews them and explicitly calls accept_suggestion() or
    reject_suggestion(). This gives the user full control.

    Usage:
        from retune import Retuner, Mode

        # Step 1: Wrap your agent
        wrapped = Retuner(
            agent=my_agent,
            adapter="custom",
            mode=Mode.OBSERVE,
            evaluators=["latency", "cost"],
        )

        # Step 2: Run some queries (observe)
        for q in queries:
            wrapped.run(q)

        # Step 3: Switch to evaluate
        wrapped.set_mode(Mode.EVALUATE)
        for q in queries:
            resp = wrapped.run(q)
            print(resp.eval_results)  # See scores

        # Step 4: Switch to improve (generates suggestions)
        wrapped.set_mode(Mode.IMPROVE)
        resp = wrapped.run("test query")
        print(resp.suggestions)  # See pending suggestions

        # Step 5: Review and accept/reject
        for s in wrapped.get_pending_suggestions():
            print(f"{s.param_name}: {s.old_value} -> {s.new_value}")
            print(f"  Reason: {s.reasoning}")
            print(f"  Confidence: {s.confidence:.0%}")

        wrapped.accept_suggestion(suggestion_id)   # Apply one
        wrapped.reject_suggestion(other_id)         # Skip one
        wrapped.accept_all()                        # Accept all pending

        # Step 6: Verify improvement
        wrapped.set_mode(Mode.EVALUATE)
        resp = wrapped.run("test query")
        print(resp.eval_results)  # Better scores?

        # Step 7: Not happy? Revert
        wrapped.revert_suggestion(suggestion_id)

        # With Beam Search APO (multi-round prompt optimization):
        wrapped = Retuner(
            agent=my_agent,
            adapter="custom",
            mode=Mode.IMPROVE,
            beam_config=BeamSearchConfig(beam_width=2, beam_rounds=2),
            validation_queries=["q1", "q2", "q3"],
            use_deep_optimizer=True,
        )
    """

    def __init__(
        self,
        agent: Any,
        adapter: str | BaseAdapter = "langchain",
        mode: str | Mode = Mode.OBSERVE,
        config: OptimizationConfig | None = None,
        evaluators: list[str | BaseEvaluator] | None = None,
        optimizer: BaseOptimizer | None = None,
        storage: BaseStorage | None = None,
        session_id: str | None = None,
        auto_improve: bool = False,
        beam_config: BeamSearchConfig | None = None,
        validation_queries: list[str] | None = None,
        use_deep_optimizer: bool = False,
        eval_llm_model: str = "gpt-4o-mini",
        llm: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        # Resolve adapter
        if isinstance(adapter, str):
            self._adapter = get_adapter(adapter, agent)
        else:
            self._adapter = adapter

        # Mode
        self._mode = Mode(mode) if isinstance(mode, str) else mode

        # Config
        self._config = config or OptimizationConfig()
        self._original_config = self._config.model_copy()  # For revert

        # Evaluators
        self._evaluators: list[BaseEvaluator] = []
        if evaluators:
            for ev in evaluators:
                if isinstance(ev, str):
                    self._evaluators.append(get_evaluator(ev))
                else:
                    self._evaluators.append(ev)

        # Set global LLM if user provided one (makes all components use it)
        if llm is not None:
            from retune.core.llm import set_default_llm
            set_default_llm(llm)

        # Optimizer: smart selection
        # If user passes explicit optimizer, use it.
        # Otherwise, if use_deep_optimizer or beam_config is set, try deep optimizer.
        # Otherwise, use BasicOptimizer.
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            should_use_deep = use_deep_optimizer or beam_config is not None
            self._optimizer = _create_optimizer(
                use_deep=should_use_deep,
                model=eval_llm_model,
                beam_config=beam_config,
            )

        # Storage: cloud-synced if API key provided, local-only otherwise
        if storage is not None:
            self._storage = storage
        elif api_key or settings.api_key:
            key = api_key or settings.api_key
            from retune.cloud.storage import CloudStorage
            self._storage = CloudStorage(
                api_key=key,  # type: ignore[arg-type]
                base_url=settings.cloud_base_url,
                db_path=settings.storage_path,
            )
        else:
            self._storage = SQLiteStorage(settings.storage_path)

        # Memory
        self._memory = MemoryStore()

        # Session
        self._session_id = session_id or str(uuid4())

        # Auto-improve: automatically apply HIGH confidence suggestions
        self._auto_improve = auto_improve

        # Beam Search APO config
        self._beam_config = beam_config

        # Validation queries for rollout verification
        self._validation_queries = validation_queries or []

        # Version tracking
        self._version = 1
        self._improvement_history: list[dict[str, Any]] = []

        # Suggestion management
        self._pending_suggestions: list[Suggestion] = []
        self._all_suggestions: list[Suggestion] = []  # Full history

    def run(self, query: str, **kwargs: Any) -> WrapperResponse:
        """Execute the wrapped agent and optionally evaluate/optimize.

        Args:
            query: Input query
            **kwargs: Additional arguments passed to the adapter

        Returns:
            WrapperResponse containing output, trace, eval results, and suggestions
        """
        expected_answer = kwargs.pop("_expected_answer", None)

        # --- Mode: OFF ---
        if self._mode == Mode.OFF:
            trace = self._adapter.run(query, config=self._config, **kwargs)
            return WrapperResponse(output=trace.response, mode=self._mode)

        # --- Mode: OBSERVE / EVALUATE / IMPROVE ---
        trace = self._adapter.run(query, config=self._config, **kwargs)

        if expected_answer is not None:
            trace.metadata["expected_answer"] = expected_answer
        trace.session_id = self._session_id
        trace.mode = self._mode
        trace.config_snapshot = self._config.to_flat_dict()

        eval_results: list[EvalResult] = []
        suggestions: list[Suggestion] = []

        # --- Mode: EVALUATE or IMPROVE ---
        if self._mode in (Mode.EVALUATE, Mode.IMPROVE):
            eval_results = self._evaluate(trace)
            trace.eval_results = eval_results

        # --- Mode: IMPROVE ---
        if self._mode == Mode.IMPROVE:
            suggestions = self._optimize(trace)

            # All suggestions start as PENDING
            for s in suggestions:
                s.status = SuggestionStatus.PENDING

            self._pending_suggestions.extend(suggestions)
            self._all_suggestions.extend(suggestions)

            # Auto-improve: only if user explicitly opted in
            if self._auto_improve and suggestions:
                self._auto_apply_suggestions(suggestions)

        # Store trace
        self._storage.save_trace(trace)

        # Update memory
        self._memory.add_from_trace(trace)

        return WrapperResponse(
            output=trace.response,
            trace=trace,
            eval_results=eval_results,
            suggestions=suggestions,
            mode=self._mode,
        )

    def _evaluate(self, trace: ExecutionTrace) -> list[EvalResult]:
        """Run all configured evaluators on the trace."""
        results = []
        for evaluator in self._evaluators:
            try:
                result = evaluator.evaluate(trace)
                results.append(result)
            except Exception as e:
                logger.warning(f"Evaluator '{evaluator.name}' failed: {e}")
                results.append(
                    EvalResult(
                        evaluator_name=evaluator.name,
                        score=0.0,
                        reasoning=f"Evaluator error: {e}",
                    )
                )
        return results

    def _optimize(self, trace: ExecutionTrace) -> list[Suggestion]:
        """Run the optimizer to get improvement suggestions.

        Passes the adapter and validation queries for beam search rollouts.
        """
        try:
            recent_traces = self._storage.get_traces(
                limit=20, session_id=self._session_id
            )
            all_traces = [trace] + recent_traces
            return self._optimizer.suggest(
                all_traces,
                self._config,
                adapter=self._adapter,
                validation_queries=self._validation_queries or None,
            )
        except Exception as e:
            logger.warning(f"Optimizer failed: {e}")
            return []

    def _auto_apply_suggestions(self, suggestions: list[Suggestion]) -> None:
        """Auto-apply high-confidence suggestions (when auto_improve=True)."""
        for s in suggestions:
            if s.confidence >= 0.6:
                self._apply_single_suggestion(s)

    def _apply_single_suggestion(self, suggestion: Suggestion) -> bool:
        """Apply a single suggestion to the config. Returns True if applied."""
        if hasattr(self._config, suggestion.param_name):
            old = getattr(self._config, suggestion.param_name)
            setattr(self._config, suggestion.param_name, suggestion.new_value)
            if suggestion.param_name == "system_prompt":
                self._adapter.set_system_prompt(suggestion.new_value)
            suggestion.status = SuggestionStatus.ACCEPTED
            logger.info(
                f"Applied suggestion: {suggestion.param_name} "
                f"{old} -> {suggestion.new_value}"
            )
            # Remove from pending
            self._pending_suggestions = [
                s for s in self._pending_suggestions
                if s.suggestion_id != suggestion.suggestion_id
            ]
            # Track version
            self._version += 1
            self._improvement_history.append({
                "version": self._version,
                "action": "accept",
                "suggestion_id": suggestion.suggestion_id,
                "param_name": suggestion.param_name,
                "old_value": old,
                "new_value": suggestion.new_value,
                "confidence": suggestion.confidence,
                "config": self._config.to_flat_dict(),
            })
            self._storage.save_config(
                f"v{self._version}_{self._session_id[:8]}", self._config
            )
            if hasattr(self._storage, 'send_suggestion_event'):
                self._storage.send_suggestion_event({
                    "action": "accept",
                    "session_id": self._session_id,
                    "suggestion_id": suggestion.suggestion_id,
                    "param_name": suggestion.param_name,
                    "old_value": old,
                    "new_value": suggestion.new_value,
                    "confidence": suggestion.confidence,
                })
            return True
        else:
            logger.warning(
                f"Cannot apply suggestion: config has no field '{suggestion.param_name}'"
            )
            return False

    # =====================================================================
    # Suggestion Management API -- the user's control panel
    # =====================================================================

    def get_pending_suggestions(self) -> list[Suggestion]:
        """Get all suggestions awaiting user review."""
        return [s for s in self._pending_suggestions if s.status == SuggestionStatus.PENDING]

    def get_all_suggestions(self) -> list[Suggestion]:
        """Get full suggestion history (pending + accepted + rejected + reverted)."""
        return list(self._all_suggestions)

    def get_suggestion(self, suggestion_id: str) -> Suggestion | None:
        """Get a specific suggestion by ID."""
        for s in self._all_suggestions:
            if s.suggestion_id == suggestion_id:
                return s
        return None

    def accept_suggestion(self, suggestion_id: str) -> bool:
        """Accept and apply a pending suggestion.

        Args:
            suggestion_id: The ID of the suggestion to accept

        Returns:
            True if the suggestion was found and applied
        """
        suggestion = self.get_suggestion(suggestion_id)
        if suggestion is None:
            logger.warning(f"Suggestion {suggestion_id} not found")
            return False

        if suggestion.status != SuggestionStatus.PENDING:
            logger.warning(
                f"Suggestion {suggestion_id} is not pending "
                f"(status={suggestion.status.value})"
            )
            return False

        return self._apply_single_suggestion(suggestion)

    def reject_suggestion(self, suggestion_id: str) -> bool:
        """Reject a pending suggestion (don't apply it).

        Args:
            suggestion_id: The ID of the suggestion to reject

        Returns:
            True if the suggestion was found and rejected
        """
        suggestion = self.get_suggestion(suggestion_id)
        if suggestion is None:
            logger.warning(f"Suggestion {suggestion_id} not found")
            return False

        if suggestion.status != SuggestionStatus.PENDING:
            logger.warning(
                f"Suggestion {suggestion_id} is not pending "
                f"(status={suggestion.status.value})"
            )
            return False

        suggestion.status = SuggestionStatus.REJECTED
        self._pending_suggestions = [
            s for s in self._pending_suggestions
            if s.suggestion_id != suggestion_id
        ]
        logger.info(
            f"Rejected suggestion: {suggestion.param_name} "
            f"({suggestion.old_value} -> {suggestion.new_value})"
        )
        if hasattr(self._storage, 'send_suggestion_event'):
            self._storage.send_suggestion_event({
                "action": "reject",
                "session_id": self._session_id,
                "suggestion_id": suggestion.suggestion_id,
                "param_name": suggestion.param_name,
            })
        return True

    def accept_all(self) -> int:
        """Accept all pending suggestions.

        Returns:
            Number of suggestions accepted
        """
        pending = self.get_pending_suggestions()
        count = 0
        for s in pending:
            if self._apply_single_suggestion(s):
                count += 1
        return count

    def reject_all(self) -> int:
        """Reject all pending suggestions.

        Returns:
            Number of suggestions rejected
        """
        pending = self.get_pending_suggestions()
        count = 0
        for s in pending:
            s.status = SuggestionStatus.REJECTED
            count += 1
        self._pending_suggestions = []
        return count

    def revert_suggestion(self, suggestion_id: str) -> bool:
        """Revert a previously accepted suggestion back to the old value.

        Args:
            suggestion_id: The ID of the suggestion to revert

        Returns:
            True if the suggestion was found and reverted
        """
        suggestion = self.get_suggestion(suggestion_id)
        if suggestion is None:
            logger.warning(f"Suggestion {suggestion_id} not found")
            return False

        if suggestion.status != SuggestionStatus.ACCEPTED:
            logger.warning(
                f"Suggestion {suggestion_id} is not accepted "
                f"(status={suggestion.status.value})"
            )
            return False

        if hasattr(self._config, suggestion.param_name):
            current = getattr(self._config, suggestion.param_name)
            setattr(self._config, suggestion.param_name, suggestion.old_value)
            suggestion.status = SuggestionStatus.REVERTED
            logger.info(
                f"Reverted suggestion: {suggestion.param_name} "
                f"{current} -> {suggestion.old_value}"
            )
            self._version += 1
            self._improvement_history.append({
                "version": self._version,
                "action": "revert",
                "suggestion_id": suggestion.suggestion_id,
                "param_name": suggestion.param_name,
                "reverted_from": current,
                "reverted_to": suggestion.old_value,
                "config": self._config.to_flat_dict(),
            })
            return True

        return False

    def revert_all(self) -> None:
        """Revert ALL changes back to the original config."""
        self._config = self._original_config.model_copy()
        for s in self._all_suggestions:
            if s.status == SuggestionStatus.ACCEPTED:
                s.status = SuggestionStatus.REVERTED
        self._pending_suggestions = []
        self._version += 1
        self._improvement_history.append({
            "version": self._version,
            "action": "revert_all",
            "config": self._config.to_flat_dict(),
        })
        logger.info("Reverted all changes to original config")

    # =====================================================================
    # Public API -- mode, config, traces, summaries
    # =====================================================================

    def set_mode(self, mode: str | Mode) -> None:
        """Change the execution mode (fan regulator)."""
        self._mode = Mode(mode) if isinstance(mode, str) else mode
        logger.info(f"Mode changed to: {self._mode.value}")

    def get_mode(self) -> Mode:
        return self._mode

    def set_config(self, config: OptimizationConfig) -> None:
        """Manually set the optimization config."""
        self._config = config

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def set_beam_config(self, beam_config: BeamSearchConfig | None) -> None:
        """Set or clear the beam search APO configuration."""
        self._beam_config = beam_config

    def set_validation_queries(self, queries: list[str]) -> None:
        """Set validation queries for beam search rollout verification."""
        self._validation_queries = queries

    def get_traces(self, limit: int = 10) -> list[ExecutionTrace]:
        """Retrieve recent traces for this session."""
        return self._storage.get_traces(limit=limit, session_id=self._session_id)

    def get_all_traces(self, limit: int = 50) -> list[ExecutionTrace]:
        """Retrieve all recent traces across sessions."""
        return self._storage.get_traces(limit=limit)

    def get_eval_summary(self) -> dict[str, Any]:
        """Get aggregated evaluation scores across recent traces."""
        traces = self.get_traces(limit=50)
        if not traces:
            return {"total_traces": 0, "scores": {}}

        scores: dict[str, list[float]] = {}
        for trace in traces:
            for result in trace.eval_results:
                scores.setdefault(result.evaluator_name, []).append(result.score)

        summary = {
            "total_traces": len(traces),
            "scores": {
                name: {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "count": len(vals),
                }
                for name, vals in scores.items()
            },
        }
        return summary

    def get_improvement_history(self) -> list[dict[str, Any]]:
        """Get the history of applied improvements."""
        return self._improvement_history

    def get_best_config(self) -> dict[str, Any] | None:
        """Get the config that produced the best results from memory."""
        return self._memory.get_best_config()

    @property
    def version(self) -> int:
        return self._version

    @property
    def session_id(self) -> str:
        return self._session_id

    def compare_configs(
        self,
        query: str,
        config_a: OptimizationConfig | None = None,
        config_b: OptimizationConfig | None = None,
    ) -> dict[str, Any]:
        """A/B test: run query with two configs and compare via pairwise judge.

        Args:
            query: The test query
            config_a: First config (default: current config)
            config_b: Second config to compare against
        """
        config_a = config_a or self._config
        if config_b is None:
            config_b = self._original_config

        trace_a = self._adapter.run(query, config=config_a)
        trace_b = self._adapter.run(query, config=config_b)

        from retune.evaluators.pairwise_judge import PairwiseJudgeEvaluator
        judge = PairwiseJudgeEvaluator()
        result = judge.compare(trace_a, trace_b)

        return {
            "query": query,
            "response_a": str(trace_a.response)[:500],
            "response_b": str(trace_b.response)[:500],
            "winner": result.details.get("winner", "tie"),
            "score": result.score,
            "reasoning": result.reasoning,
            "details": result.details,
        }

    def run_evaluation_dataset(
        self,
        dataset: list[dict[str, str]],
        query_key: str = "query",
    ) -> dict[str, Any]:
        """Run evaluation on a dataset and return aggregate scores.

        Args:
            dataset: List of dicts with at least a query key
            query_key: Key name for the query field

        Returns:
            Aggregate evaluation results
        """
        original_mode = self._mode
        self._mode = Mode.EVALUATE

        results = []
        for item in dataset:
            query = item[query_key]
            expected = item.get("expected")
            response = self.run(query, _expected_answer=expected)
            results.append(
                {
                    "query": query,
                    "output": response.output,
                    "scores": {r.evaluator_name: r.score for r in response.eval_results},
                    "expected": item.get("expected", None),
                }
            )

        self._mode = original_mode

        # Aggregate
        all_scores: dict[str, list[float]] = {}
        for r in results:
            for name, score in r["scores"].items():
                all_scores.setdefault(name, []).append(score)

        return {
            "total_queries": len(results),
            "results": results,
            "aggregate_scores": {
                name: sum(vals) / len(vals) for name, vals in all_scores.items()
            },
        }
