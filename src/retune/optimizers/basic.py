"""Basic rule-based optimizer — MVP optimization engine."""

from __future__ import annotations

from retune.core.models import ExecutionTrace, OptimizationConfig, Suggestion
from retune.optimizers.base import BaseOptimizer


class BasicOptimizer(BaseOptimizer):
    """Rule-based optimizer that suggests improvements from common failure patterns.

    Analyzes traces and applies heuristic rules:
    - Low retrieval scores → increase top_k, suggest reranker
    - High latency → reduce top_k, lower temperature
    - Low correctness → suggest prompt improvements
    - High token usage → reduce max_tokens
    """

    name = "basic"

    def __init__(self, score_threshold: float = 0.7) -> None:
        self._threshold = score_threshold

    def suggest(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        adapter: object | None = None,
        validation_queries: list[str] | None = None,
    ) -> list[Suggestion]:
        if not traces:
            return []

        suggestions: list[Suggestion] = []

        # Aggregate scores across traces
        scores = self._aggregate_scores(traces)

        # Rule 1: Low retrieval quality
        if scores.get("retrieval", 1.0) < self._threshold:
            suggestions.extend(self._suggest_retrieval_improvements(traces, current_config))

        # Rule 2: Low correctness
        if scores.get("correctness", 1.0) < self._threshold:
            suggestions.extend(self._suggest_correctness_improvements(traces, current_config))

        # Rule 3: High latency
        if scores.get("latency", 1.0) < self._threshold:
            suggestions.extend(self._suggest_latency_improvements(traces, current_config))

        # Rule 4: High cost
        if scores.get("cost", 1.0) < self._threshold:
            suggestions.extend(self._suggest_cost_improvements(traces, current_config))

        return suggestions

    def _aggregate_scores(self, traces: list[ExecutionTrace]) -> dict[str, float]:
        """Average eval scores across traces by evaluator name."""
        scores: dict[str, list[float]] = {}
        for trace in traces:
            for result in trace.eval_results:
                scores.setdefault(result.evaluator_name, []).append(result.score)
                # Also extract sub-scores from details
                for key in ("correctness", "retrieval", "latency", "cost"):
                    if key in result.details:
                        scores.setdefault(key, []).append(result.details[key])

        return {k: sum(v) / len(v) for k, v in scores.items() if v}

    def _suggest_retrieval_improvements(
        self, traces: list[ExecutionTrace], config: OptimizationConfig
    ) -> list[Suggestion]:
        suggestions = []
        current_top_k = config.top_k or 4

        # Increase top_k
        if current_top_k < 10:
            suggestions.append(
                Suggestion(
                    param_name="top_k",
                    old_value=current_top_k,
                    new_value=min(current_top_k + 3, 15),
                    reasoning=(
                        "Retrieval quality is below threshold. "
                        "Increasing top_k to retrieve more potentially relevant documents."
                    ),
                    confidence=0.7,
                    category="rag",
                )
            )

        # Suggest reranker
        if not config.use_reranker:
            suggestions.append(
                Suggestion(
                    param_name="use_reranker",
                    old_value=False,
                    new_value=True,
                    reasoning=(
                        "Adding a cross-encoder reranker can significantly improve "
                        "retrieval precision by re-scoring documents after initial retrieval."
                    ),
                    confidence=0.8,
                    category="rag",
                )
            )

        # Suggest hybrid search
        if config.search_type != "hybrid":
            suggestions.append(
                Suggestion(
                    param_name="search_type",
                    old_value=config.search_type or "similarity",
                    new_value="hybrid",
                    reasoning=(
                        "Hybrid search (combining dense + sparse retrieval) "
                        "often improves recall for diverse query types."
                    ),
                    confidence=0.6,
                    category="rag",
                )
            )

        return suggestions

    def _suggest_correctness_improvements(
        self, traces: list[ExecutionTrace], config: OptimizationConfig
    ) -> list[Suggestion]:
        suggestions = []

        # Lower temperature for more deterministic outputs
        current_temp = config.temperature or 0.7
        if current_temp > 0.3:
            suggestions.append(
                Suggestion(
                    param_name="temperature",
                    old_value=current_temp,
                    new_value=max(current_temp - 0.3, 0.0),
                    reasoning=(
                        "Lowering temperature reduces randomness and can improve "
                        "factual accuracy for Q&A tasks."
                    ),
                    confidence=0.6,
                    category="agent",
                )
            )

        # Suggest structured reasoning
        if config.reasoning_strategy != "cot":
            suggestions.append(
                Suggestion(
                    param_name="reasoning_strategy",
                    old_value=config.reasoning_strategy or "default",
                    new_value="cot",
                    reasoning=(
                        "Chain-of-thought reasoning can improve correctness "
                        "by forcing step-by-step thinking before answering."
                    ),
                    confidence=0.5,
                    category="agent",
                )
            )

        return suggestions

    def _suggest_latency_improvements(
        self, traces: list[ExecutionTrace], config: OptimizationConfig
    ) -> list[Suggestion]:
        suggestions = []

        # Reduce top_k if high
        current_top_k = config.top_k or 4
        if current_top_k > 5:
            suggestions.append(
                Suggestion(
                    param_name="top_k",
                    old_value=current_top_k,
                    new_value=max(current_top_k - 2, 3),
                    reasoning="Reducing top_k decreases retrieval and context processing time.",
                    confidence=0.6,
                    category="rag",
                )
            )

        # Reduce max_tokens
        current_max = config.max_tokens or 2048
        if current_max > 1024:
            suggestions.append(
                Suggestion(
                    param_name="max_tokens",
                    old_value=current_max,
                    new_value=1024,
                    reasoning="Limiting max output tokens reduces generation time.",
                    confidence=0.5,
                    category="agent",
                )
            )

        return suggestions

    def _suggest_cost_improvements(
        self, traces: list[ExecutionTrace], config: OptimizationConfig
    ) -> list[Suggestion]:
        suggestions = []

        avg_tokens = sum(t.total_tokens for t in traces) / max(len(traces), 1)

        if avg_tokens > 5000:
            current_max = config.max_tokens or 2048
            suggestions.append(
                Suggestion(
                    param_name="max_tokens",
                    old_value=current_max,
                    new_value=max(current_max // 2, 512),
                    reasoning=(
                        f"Average token usage is {avg_tokens:.0f}. "
                        "Reducing max_tokens to control costs."
                    ),
                    confidence=0.5,
                    category="agent",
                )
            )

        return suggestions
