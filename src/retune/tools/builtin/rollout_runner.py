"""Rollout runner tool — runs adapter with candidate configs for verification."""

from __future__ import annotations

import logging
from typing import Any

from retune.tools.base import RetuneTool

logger = logging.getLogger(__name__)


class RolloutRunnerTool(RetuneTool):
    """Runs an adapter with a candidate config against validation queries.

    Used in Beam Search APO for verification rollouts — tests whether a candidate
    prompt/config actually improves performance before accepting it.
    """

    name: str = "rollout_runner"
    description: str = (
        "Run the wrapped agent with a candidate configuration against validation queries. "
        "Input: adapter, candidate_config, validation_queries, evaluators. "
        "Output: avg_score, per_query_scores, total_cost."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "candidate_config": {
                "type": "object",
                "description": "The candidate OptimizationConfig to test",
            },
            "validation_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Queries to run the agent against",
            },
            "max_queries": {
                "type": "integer",
                "description": "Maximum number of queries to run",
                "default": 5,
            },
        },
        "required": ["candidate_config", "validation_queries"],
    }

    # These are set externally before use (not serialized)
    _adapter: Any = None
    _evaluators: list[Any] = []

    model_config = {"arbitrary_types_allowed": True}

    def set_adapter(self, adapter: Any) -> None:
        """Set the adapter to use for rollouts."""
        self._adapter = adapter

    def set_evaluators(self, evaluators: list[Any]) -> None:
        """Set evaluators for scoring rollout results."""
        self._evaluators = evaluators

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        candidate_config_dict = kwargs.get("candidate_config", {})
        validation_queries = kwargs.get("validation_queries", [])
        max_queries = kwargs.get("max_queries", 5)

        if self._adapter is None:
            return {
                "error": "No adapter configured for rollout runner",
                "avg_score": 0.0,
                "per_query_scores": [],
                "total_cost": 0.0,
            }

        if not validation_queries:
            return {
                "error": "No validation queries provided",
                "avg_score": 0.0,
                "per_query_scores": [],
                "total_cost": 0.0,
            }

        from retune.core.models import OptimizationConfig

        # Build config from dict
        candidate_config = OptimizationConfig(**candidate_config_dict)

        # Limit queries
        queries = validation_queries[:max_queries]

        per_query_scores = []
        total_cost = 0.0

        for query in queries:
            try:
                trace = self._adapter.run(query, config=candidate_config)
                total_cost += trace.total_cost

                # Evaluate
                query_scores = []
                for evaluator in self._evaluators:
                    try:
                        result = evaluator.evaluate(trace)
                        query_scores.append(result.score)
                    except Exception as e:
                        logger.debug(f"Evaluator failed during rollout: {e}")
                        query_scores.append(0.5)

                avg_query_score = (
                    sum(query_scores) / len(query_scores) if query_scores else 0.5
                )
                per_query_scores.append({
                    "query": query,
                    "score": round(avg_query_score, 3),
                    "cost": trace.total_cost,
                })

            except Exception as e:
                logger.warning(f"Rollout query failed: {e}")
                per_query_scores.append({
                    "query": query,
                    "score": 0.0,
                    "cost": 0.0,
                    "error": str(e),
                })

        all_scores = [r["score"] for r in per_query_scores]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "avg_score": round(avg_score, 3),
            "per_query_scores": per_query_scores,
            "total_cost": round(total_cost, 4),
            "num_queries": len(queries),
        }
