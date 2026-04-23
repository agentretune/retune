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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from retune.optimizer.report import OptimizationReport

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
from retune.optimizer.client import OptimizerClient  # noqa: F401
from retune.optimizer.retrieval_introspection import introspect_retrieval_config  # noqa: F401
from retune.optimizer.tool_introspection import introspect_tools  # noqa: F401
from retune.optimizer.worker import SDKWorker  # noqa: F401
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
        # Auto-evaluation and RL loop
        auto_eval_every: int = 0,
        auto_optimize: bool = False,
        drift_threshold: float = 0.1,
        max_free_optimizations: int = 15,
        enable_few_shot: bool = False,
        enable_routing: bool = False,
        agent_purpose: str | None = None,
        success_criteria: str | None = None,
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

        # Cloud optimize parameters
        self._api_key = api_key or settings.api_key
        self._agent_purpose = agent_purpose
        self._success_criteria = success_criteria
        if self._mode == Mode.IMPROVE and not self._agent_purpose:
            raise ValueError(
                "agent_purpose='...' is required when mode=Mode.IMPROVE. "
                "Supply a one-line description of what your agent does."
            )

        # Usage gate (15 free deep operations, unlimited for premium)
        from retune.usage_gate import UsageGate
        self._usage_gate = UsageGate(api_key=api_key or settings.api_key)

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

        # Auto-eval controller (the "RL loop")
        self._auto_eval_controller = None
        if auto_eval_every > 0 or auto_optimize:
            from retune.auto_eval import AutoEvalController
            self._auto_eval_controller = AutoEvalController(
                eval_every_n_calls=auto_eval_every or 50,
                optimize_on_drift=auto_optimize,
                drift_threshold=drift_threshold,
                max_free_optimizations=max_free_optimizations,
            )
            # Premium if API key is set
            if api_key or settings.api_key:
                self._auto_eval_controller.set_premium(True)

        # Few-shot optimizer
        self._few_shot = None
        if enable_few_shot:
            from retune.few_shot import FewShotOptimizer
            self._few_shot = FewShotOptimizer()

        # Strategy router
        self._router = None
        if enable_routing:
            from retune.strategy_router import StrategyRouter
            self._router = StrategyRouter()

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

        # Few-shot: inject relevant examples into system prompt
        if self._few_shot and self._few_shot.example_count > 0:
            examples_text = self._few_shot.build_examples_prompt(query)
            if examples_text and self._config.system_prompt:
                # Temporarily augment the system prompt with examples
                augmented = self._config.system_prompt + examples_text
                self._adapter.set_system_prompt(augmented)

        eval_results: list[EvalResult] = []
        suggestions: list[Suggestion] = []

        # --- Mode: EVALUATE or IMPROVE ---
        if self._mode in (Mode.EVALUATE, Mode.IMPROVE):
            eval_results = self._evaluate(trace)
            trace.eval_results = eval_results

        # --- Mode: IMPROVE ---
        if self._mode == Mode.IMPROVE:
            if not self._usage_gate.check("optimize"):
                logger.warning(
                    "Deep optimization limit reached. "
                    "Upgrade at https://agentretune.com/pricing"
                )
            else:
                suggestions = self._optimize(trace)
                if suggestions:
                    self._usage_gate.record_usage("optimize")

            # All suggestions start as PENDING
            for s in suggestions:
                s.status = SuggestionStatus.PENDING

            self._pending_suggestions.extend(suggestions)
            self._all_suggestions.extend(suggestions)

            # Auto-improve: only if user explicitly opted in
            if self._auto_improve and suggestions:
                self._auto_apply_suggestions(suggestions)

        # Auto-eval: track this call and check triggers
        auto_status = None
        if self._auto_eval_controller:
            auto_status = self._auto_eval_controller.on_trace(trace, eval_results)

            # Few-shot: store good traces as examples
            if self._few_shot and eval_results:
                self._few_shot.add_from_trace(trace)

            # Strategy router: record result
            if self._router and eval_results:
                avg = sum(r.score for r in eval_results) / len(eval_results)
                self._router.record_result(avg)

            # Auto-optimize if triggered
            if auto_status and auto_status.get("should_optimize"):
                if self._auto_eval_controller.can_optimize:
                    self._auto_optimize(trace)

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

    def _auto_optimize(self, latest_trace: ExecutionTrace) -> None:
        """Run the automatic optimization cycle."""
        if not self._auto_eval_controller:
            return
        if not self._auto_eval_controller.can_optimize:
            logger.warning(
                "Free optimization limit reached. "
                "Set api_key for unlimited optimizations."
            )
            return

        logger.info("Auto-optimization triggered")

        # Get suggestions
        suggestions = self._optimize(latest_trace)
        if not suggestions:
            logger.info("No optimization suggestions generated")
            return

        # Auto-apply high-confidence suggestions
        applied = 0
        for s in suggestions:
            if s.confidence >= 0.7:
                if self._apply_single_suggestion(s):
                    applied += 1

        if applied > 0:
            self._auto_eval_controller.record_optimization()
            self._auto_eval_controller.update_baseline()
            logger.info(
                f"Auto-optimization applied {applied} suggestions "
                f"(remaining: {self._auto_eval_controller.optimizations_remaining})"
            )

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

    def get_usage_status(self) -> dict[str, Any]:
        """Get deep optimization usage status (free tier limit)."""
        return self._usage_gate.get_status()

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

    def get_auto_eval_status(self) -> dict[str, Any]:
        """Get the auto-evaluation and RL loop status."""
        if not self._auto_eval_controller:
            return {"enabled": False}
        return {"enabled": True, **self._auto_eval_controller.get_summary()}

    def get_few_shot_examples(self) -> list[dict[str, Any]]:
        """Get stored few-shot examples."""
        if not self._few_shot:
            return []
        return self._few_shot.get_all_examples()

    def get_routing_status(self) -> dict[str, Any]:
        """Get strategy routing status."""
        if not self._router:
            return {"enabled": False}
        return {"enabled": True, **self._router.get_summary()}

    def add_strategy_variant(
        self,
        name: str,
        config: OptimizationConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Add a strategy variant for routing."""
        if not self._router:
            from retune.strategy_router import StrategyRouter
            self._router = StrategyRouter()
        config = config or OptimizationConfig()
        self._router.add_variant(name, config, system_prompt)

    # =====================================================================
    # Cloud Optimization API
    # =====================================================================

    def apply_report(self, report: "OptimizationReport", tier: int = 1) -> None:
        """Apply tier-N suggestions from an OptimizationReport to the wrapped agent.

        Phase 3 handles these apply_payload.action values:
        - "drop_tool"     → remove tool from self._adapter.tools
        - "system_prompt" (implicit, via payload key) → self._config.system_prompt = ...

        Other actions are currently no-op (future phases will extend).
        """
        from retune.optimizer.report import OptimizationReport  # noqa: F401

        def _apply(s):
            payload = s.apply_payload or {}
            action = payload.get("action")
            if action == "drop_tool":
                tool_name = payload.get("tool_name")
                tools = getattr(self._adapter, "tools", None)
                if tools is None:
                    return
                filtered = []
                for t in tools:
                    name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
                    if name != tool_name:
                        filtered.append(t)
                self._adapter.tools = filtered
            elif "system_prompt" in payload:
                self._config.system_prompt = payload["system_prompt"]
            # Other actions: no-op for Phase 3

        report.apply(tier=tier, apply_fn=_apply)

    def optimize(
        self,
        source: str = "last_n_traces",
        n: int = 50,
        axes: "list[str] | str" = "auto",
        reward: "str | dict" = "judge_with_guardrails",
        rewriter_llm: str | None = None,
        guardrails: dict | None = None,
    ):
        """Trigger a cloud optimization run.

        Args:
            source: "last_n_traces" (historical replay) or "collect_next".
            n: number of traces to optimize over.
            axes: list of axes to optimize, or "auto" for orchestrator-selected.
            reward: "judge_with_guardrails" (default) or a dict with the
                declarative reward spec.
            rewriter_llm: model used by PromptOptimizerAgent for rewrites.
            guardrails: overrides for the default cost/latency guardrails.
        """
        from retune.optimizer.report import OptimizationReport

        if not self._api_key:
            raise RuntimeError("api_key is required to call optimize()")

        axes_list = ["prompt", "tools", "rag"] if axes == "auto" else list(axes)
        reward_spec = reward if isinstance(reward, dict) else None
        if guardrails and reward_spec is None:
            reward_spec = {
                "primary": {"evaluator": "llm_judge", "weight": 1.0},
                "penalties": [
                    {"evaluator": k, "threshold": v, "hard": True}
                    for k, v in guardrails.items()
                ],
            }

        traces_payload = None
        if source == "last_n_traces":
            try:
                from retune.optimizer.trace_collector import collect_last_n_local_traces
                traces_payload = collect_last_n_local_traces(self._storage, n=n)
            except Exception as e:
                logger.warning("Failed to collect local traces: %s", e)
                traces_payload = []

        tool_metadata_payload = None
        if "tools" in axes_list:
            try:
                tool_metadata_payload = [
                    md.model_dump() for md in introspect_tools(self._adapter)
                ]
            except Exception as e:
                logger.warning("Tool introspection failed: %s", e)
                tool_metadata_payload = []

        retrieval_config_payload = None
        if "rag" in axes_list:
            try:
                rc = introspect_retrieval_config(self._adapter)
                retrieval_config_payload = rc.model_dump() if rc else None
            except Exception as e:
                logger.warning("Retrieval introspection failed: %s", e)
                retrieval_config_payload = None

        client = OptimizerClient(api_key=self._api_key, base_url=settings.cloud_base_url)
        resp = client.preauthorize(
            source=source, n_traces=n, axes=axes_list,
            reward_spec=reward_spec, rewriter_llm=rewriter_llm,
            traces=traces_payload,
            tool_metadata=tool_metadata_payload,
            retrieval_config=retrieval_config_payload,
        )
        run_id = resp["run_id"]

        worker = SDKWorker(
            client=client,
            run_id=run_id,
            candidate_runner=self._make_candidate_runner(),
        )
        try:
            worker.run()
        except Exception:
            client.cancel(run_id)
            raise

        raw = client.fetch_report(run_id)
        client.commit(run_id)
        return OptimizationReport.from_cloud_dict(raw)

    def _make_candidate_runner(self):
        """Return a callable that runs the wrapped agent with config overrides
        and returns REAL eval scores from the registered evaluators."""
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
                    "config_snapshot": (
                        self._config.to_flat_dict()
                        if hasattr(self._config, "to_flat_dict") else {}
                    ),
                }
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
