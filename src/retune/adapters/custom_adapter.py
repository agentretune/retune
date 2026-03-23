"""Custom adapter — wraps any callable (function, class) as an agent."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from retune.adapters.base import BaseAdapter
from retune.core.enums import StepType
from retune.core.exceptions import AdapterError
from retune.core.models import ExecutionTrace, OptimizationConfig, Step


class CustomAdapter(BaseAdapter):
    """Adapter for custom functions or callable objects.

    Wraps any callable that takes a string query and returns a response.
    The user can optionally provide a trace_fn to capture internal steps.

    Usage:
        def my_rag_pipeline(query: str) -> str:
            docs = retriever.search(query)
            return llm.generate(query, context=docs)

        adapter = CustomAdapter(agent=my_rag_pipeline)
        trace = adapter.run("What is AI?")
    """

    def __init__(
        self,
        agent: Callable[..., Any],
        trace_fn: Callable[..., list[Step]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent=agent, **kwargs)
        self._trace_fn = trace_fn
        self._config = OptimizationConfig()
        self._system_prompt: str | None = None
        self._agent_accepts_system_prompt: bool | None = None

    def run(
        self,
        query: str,
        config: OptimizationConfig | None = None,
        **kwargs: Any,
    ) -> ExecutionTrace:
        if config:
            self.apply_config(config)

        started_at = datetime.now(timezone.utc)
        steps: list[Step] = []

        if self._system_prompt and self._system_prompt not in kwargs.values():
            if self._agent_accepts_system_prompt is None:
                try:
                    sig = inspect.signature(self.agent)
                    self._agent_accepts_system_prompt = "system_prompt" in sig.parameters
                except (ValueError, TypeError):
                    self._agent_accepts_system_prompt = False
            if self._agent_accepts_system_prompt:
                kwargs["system_prompt"] = self._system_prompt

        try:
            result = self.agent(query, **kwargs)
        except Exception as e:
            raise AdapterError(f"Custom agent execution failed: {e}") from e

        ended_at = datetime.now(timezone.utc)

        # If user provided a trace function, call it
        if self._trace_fn:
            try:
                steps = self._trace_fn(query, result)
            except Exception:
                pass  # Don't fail on tracing errors

        # If no trace function, create a single step
        if not steps:
            steps = [
                Step(
                    step_type=StepType.CUSTOM,
                    name="custom_execution",
                    input_data={"query": query},
                    output_data={"response": str(result)[:2000]},
                    started_at=started_at,
                    ended_at=ended_at,
                )
            ]

        response_text = str(result) if not isinstance(result, str) else result

        return ExecutionTrace(
            trace_id=str(uuid4()),
            query=query,
            response=response_text,
            steps=steps,
            config_snapshot=self._config.to_flat_dict(),
            started_at=started_at,
            ended_at=ended_at,
        )

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def apply_config(self, config: OptimizationConfig) -> None:
        flat = config.to_flat_dict()
        if "system_prompt" in flat:
            self.set_system_prompt(flat["system_prompt"])
        for key, value in flat.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
