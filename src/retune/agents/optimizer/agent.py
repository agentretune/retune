"""Deep Optimizer — calls cloud API for premium optimization.

With API key: sends traces to agentretune.com, gets optimization suggestions.
Without API key: falls back to BasicOptimizer (rule-based heuristics).
The actual deep optimization (APO, beam search, config tuning, tool
curation) runs on the server.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from retune.agents.optimizer.beam_config import BeamSearchConfig
from retune.core.models import ExecutionTrace, OptimizationConfig, Suggestion
from retune.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


class OptimizerDeepAgent(BaseOptimizer):
    """Deep optimizer — premium cloud optimization with local fallback.

    With API key (free or paid):
    - Sends traces + config to cloud API
    - Server runs APO, beam search, config tuning, tool curation
    - Returns optimization suggestions
    - Counts against 15 free deep operations

    Without API key:
    - BasicOptimizer with rule-based heuristics
    - No LLM calls, no usage counted
    """

    name = "deep_optimizer"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        beam_config: BeamSearchConfig | None = None,
    ) -> None:
        self._model = model
        self._beam_config = beam_config

    def suggest(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        adapter: Any | None = None,
        validation_queries: list[str] | None = None,
    ) -> list[Suggestion]:
        # Try cloud optimization first
        try:
            from retune.config import settings
            if settings.api_key:
                result = self._cloud_optimize(
                    traces, current_config, settings.api_key,
                    settings.cloud_base_url, validation_queries,
                )
                if result:
                    return result
        except Exception as e:
            logger.debug(f"Cloud optimization failed: {e}")

        # Fallback to BasicOptimizer
        from retune.optimizers.basic import BasicOptimizer
        return BasicOptimizer().suggest(traces, current_config)

    def _cloud_optimize(
        self,
        traces: list[ExecutionTrace],
        current_config: OptimizationConfig,
        api_key: str,
        base_url: str,
        validation_queries: list[str] | None = None,
    ) -> list[Suggestion] | None:
        """Send traces to cloud API for deep optimization."""
        url = f"{base_url}/api/v1/ingest/optimize"

        # Summarize traces for the API
        trace_summaries = []
        for t in traces[:10]:
            trace_summaries.append({
                "trace_id": t.trace_id,
                "query": t.query[:200],
                "response": str(t.response)[:500],
                "eval_results": [
                    {"evaluator_name": r.evaluator_name, "score": r.score}
                    for r in t.eval_results
                ],
                "duration_ms": t.duration_ms,
                "total_tokens": t.total_tokens,
            })

        payload = json.dumps({
            "traces": trace_summaries,
            "current_config": current_config.model_dump(mode="json"),
            "model": self._model,
            "beam_config": (
                self._beam_config.model_dump() if self._beam_config else None
            ),
            "validation_queries": validation_queries,
        }, default=str).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "retune-sdk/0.1.0",
        }

        try:
            req = Request(url, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=60.0) as resp:
                data = json.loads(resp.read())
                suggestions = []
                for s in data.get("suggestions", []):
                    suggestions.append(Suggestion(
                        param_name=s.get("param_name", "unknown"),
                        old_value=s.get("old_value"),
                        new_value=s.get("new_value"),
                        reasoning=s.get("reasoning", ""),
                        confidence=float(s.get("confidence", 0.5)),
                        category=s.get("category", "general"),
                    ))
                return suggestions if suggestions else None
        except URLError as e:
            logger.debug(f"Cloud optimize request failed: {e}")
        except Exception as e:
            logger.debug(f"Cloud optimize parse failed: {e}")

        return None
