"""LangChain adapter — wraps LangChain Runnables with tracing callbacks."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from retune.adapters.base import BaseAdapter
from retune.core.enums import StepType
from retune.core.exceptions import AdapterError
from retune.core.models import ExecutionTrace, OptimizationConfig, Step, TokenUsage

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class _TracingCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that captures execution steps."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[Step] = []
        self._pending: dict[str, dict[str, Any]] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        self._pending[str(run_id)] = {
            "type": StepType.LLM_CALL,
            "name": serialized.get("id", ["unknown"])[-1],
            "input": prompts[0] if prompts else "",
            "started_at": datetime.now(timezone.utc),
            "metadata": {
                "serialized_id": serialized.get("id", []),
                "invocation_params": kwargs.get("invocation_params", {}),
            },
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        pending = self._pending.pop(key, None)
        if pending is None:
            return

        output_text = ""
        token_usage = None

        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text

        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            token_usage = TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )

        self.steps.append(
            Step(
                step_type=pending["type"],
                name=pending["name"],
                input_data={"prompt": pending["input"]},
                output_data={"response": output_text},
                metadata=pending.get("metadata", {}),
                started_at=pending["started_at"],
                ended_at=datetime.now(timezone.utc),
                token_usage=token_usage,
            )
        )

    def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        name = "retriever"
        if serialized:
            name = serialized.get("id", ["retriever"])[-1]
        self._pending[str(run_id)] = {
            "type": StepType.RETRIEVAL,
            "name": name,
            "input": query,
            "started_at": datetime.now(timezone.utc),
        }

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        pending = self._pending.pop(key, None)
        if pending is None:
            return

        doc_data = []
        if documents:
            for doc in documents:
                doc_data.append(
                    {
                        "content": getattr(doc, "page_content", str(doc))[:500],
                        "metadata": getattr(doc, "metadata", {}),
                    }
                )

        self.steps.append(
            Step(
                step_type=pending["type"],
                name=pending["name"],
                input_data={"query": pending["input"]},
                output_data={"documents": doc_data, "num_docs": len(doc_data)},
                started_at=pending["started_at"],
                ended_at=datetime.now(timezone.utc),
            )
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        serialized = serialized or {}
        self._pending[str(run_id)] = {
            "type": StepType.TOOL_CALL,
            "name": serialized.get("name", serialized.get("id", ["tool"])[-1]),
            "input": input_str,
            "started_at": datetime.now(timezone.utc),
        }

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        pending = self._pending.pop(key, None)
        if pending is None:
            return

        self.steps.append(
            Step(
                step_type=pending["type"],
                name=pending["name"],
                input_data={"input": pending["input"]},
                output_data={"output": str(output)[:2000]},
                started_at=pending["started_at"],
                ended_at=datetime.now(timezone.utc),
            )
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._pending.pop(str(run_id), None)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._pending.pop(str(run_id), None)

    def on_retriever_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._pending.pop(str(run_id), None)


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain Runnables (chains, agents, RAG pipelines).

    Wraps any LangChain Runnable and captures execution traces via callbacks.

    Usage:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        chain = prompt | llm
        adapter = LangChainAdapter(agent=chain)
        trace = adapter.run("What is AI?")
    """

    def __init__(self, agent: Any, **kwargs: Any) -> None:
        if not HAS_LANGCHAIN:
            raise AdapterError(
                "langchain-core is required. Install with: pip install retune[langchain]"
            )
        super().__init__(agent=agent, **kwargs)
        self._config = OptimizationConfig()

    def run(
        self,
        query: str,
        config: OptimizationConfig | None = None,
        **kwargs: Any,
    ) -> ExecutionTrace:
        if config:
            self.apply_config(config)

        handler = _TracingCallbackHandler()
        started_at = datetime.now(timezone.utc)

        try:
            # LangChain Runnables accept callbacks in the config
            invoke_kwargs = {**kwargs}
            invoke_config = invoke_kwargs.pop("config", {})
            if isinstance(invoke_config, dict):
                existing_callbacks = invoke_config.get("callbacks", [])
                invoke_config["callbacks"] = existing_callbacks + [handler]
            else:
                invoke_config = {"callbacks": [handler]}

            # Try invoke with string input first, fall back to dict
            try:
                result = self.agent.invoke(query, config=invoke_config, **invoke_kwargs)
            except (TypeError, ValueError):
                result = self.agent.invoke(
                    {"input": query}, config=invoke_config, **invoke_kwargs
                )

        except Exception as e:
            raise AdapterError(f"LangChain execution failed: {e}") from e

        ended_at = datetime.now(timezone.utc)

        # Extract the response text
        if isinstance(result, str):
            response_text = result
        elif hasattr(result, "content"):
            response_text = result.content
        elif isinstance(result, dict):
            response_text = result.get("output", result.get("answer", str(result)))
        else:
            response_text = str(result)

        return ExecutionTrace(
            trace_id=str(uuid4()),
            query=query,
            response=response_text,
            steps=handler.steps,
            config_snapshot=self._config.to_flat_dict(),
            started_at=started_at,
            ended_at=ended_at,
        )

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def apply_config(self, config: OptimizationConfig) -> None:
        """Apply config to the wrapped chain.

        For LangChain, this primarily affects prompt templates and LLM parameters.
        """
        flat = config.to_flat_dict()

        if "temperature" in flat and hasattr(self.agent, "temperature"):
            self.agent.temperature = flat["temperature"]

        if "system_prompt" in flat:
            # Store for potential prompt modification
            self._config.system_prompt = flat["system_prompt"]

        # Update internal config state
        for key, value in flat.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
