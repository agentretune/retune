"""LangChain callback handler for token/cost tracking."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from retune.core.models import TokenUsage

logger = logging.getLogger(__name__)


class TokenTrackingHandler(BaseCallbackHandler):
    """Tracks token usage across LLM calls.

    Subclasses BaseCallbackHandler so LangChain/LangGraph callback
    managers recognize it and route events correctly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.llm_calls: list[dict[str, Any]] = []
        self._pending: dict[str, dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._pending[str(run_id)] = {
            "started_at": datetime.now(timezone.utc),
            "prompt_text": prompts[0] if prompts else "",
        }

    def on_llm_end(
        self, response: Any, *, run_id: Any, **kwargs: Any
    ) -> None:
        pending = self._pending.pop(str(run_id), {})
        token_usage = None
        model_name = ""
        cost_usd = 0.0

        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or {}
        model_name = llm_output.get("model_name", "")

        # Try response metadata (newer LangChain versions)
        if not usage and hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    meta = getattr(gen, "generation_info", {}) or {}
                    if "usage" in meta:
                        usage = meta["usage"]
                    rm = (
                        getattr(gen.message, "response_metadata", {})
                        if hasattr(gen, "message")
                        else {}
                    )
                    if "usage" in rm:
                        usage = rm["usage"]
                    if not model_name:
                        model_name = rm.get("model", "")

        if usage:
            prompt_tokens = usage.get(
                "prompt_tokens", usage.get("input_tokens", 0)
            )
            completion_tokens = usage.get(
                "completion_tokens", usage.get("output_tokens", 0)
            )
            total_tokens = usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            )
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            from retune.utils.cost_tracker import estimate_cost

            cost_usd = estimate_cost(
                model_name, prompt_tokens, completion_tokens
            )

        self.llm_calls.append(
            {
                "token_usage": token_usage,
                "model_name": model_name,
                "cost_usd": cost_usd,
                "started_at": pending.get("started_at"),
                "ended_at": datetime.now(timezone.utc),
            }
        )

    @property
    def total_tokens(self) -> int:
        return sum(
            c["token_usage"].total_tokens
            for c in self.llm_calls
            if c.get("token_usage")
        )

    @property
    def total_cost(self) -> float:
        return float(sum(c.get("cost_usd", 0) for c in self.llm_calls))
