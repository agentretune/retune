"""LangChain adapter — wraps LangChain Runnables with tracing callbacks."""

from __future__ import annotations

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

        cost_usd = None
        if token_usage:
            model_name = (response.llm_output or {}).get("model_name", "")
            from retune.utils.cost_tracker import estimate_cost
            cost_usd = estimate_cost(
                model_name,
                token_usage.prompt_tokens,
                token_usage.completion_tokens,
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
                cost_usd=cost_usd,
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
        self._system_prompt: str | None = None

    def _inject_system_prompt(self, input_data: Any) -> Any:
        """Try to inject system prompt into the chain input."""
        if not self._system_prompt:
            return input_data

        # If input is a dict, add system_prompt key
        if isinstance(input_data, dict):
            input_data["system_prompt"] = self._system_prompt

        # Also try to modify the chain's prompt template directly
        try:
            if hasattr(self.agent, 'first'):
                prompt = self.agent.first
                if hasattr(prompt, 'messages') and prompt.messages:
                    from langchain_core.prompts import SystemMessagePromptTemplate
                    # Check if first message is a system template
                    if isinstance(prompt.messages[0], SystemMessagePromptTemplate):
                        prompt.messages[0] = SystemMessagePromptTemplate.from_template(
                            self._system_prompt
                        )
        except Exception:
            pass  # Best effort -- don't break the chain

        return input_data

    def _find_llm(self) -> Any:
        """Try to find the LLM component in the chain."""
        # Direct LLM
        if hasattr(self.agent, 'temperature'):
            return self.agent
        # RunnableSequence -- look for LLM in the chain
        if hasattr(self.agent, 'middle'):
            for step in self.agent.middle:
                if hasattr(step, 'temperature'):
                    return step
        if hasattr(self.agent, 'last') and hasattr(self.agent.last, 'temperature'):
            return self.agent.last
        return None

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
                input_data = self._inject_system_prompt(query)
                result = self.agent.invoke(input_data, config=invoke_config, **invoke_kwargs)
            except (TypeError, ValueError):
                input_data = self._inject_system_prompt({"input": query})
                result = self.agent.invoke(
                    input_data, config=invoke_config, **invoke_kwargs
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
            response_text = result.get("output") or result.get("answer") or str(result)
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

    def set_system_prompt(self, prompt: str) -> None:
        self._system_prompt = prompt

    def apply_config(self, config: OptimizationConfig) -> None:
        """Apply config to the wrapped chain.

        For LangChain, this primarily affects prompt templates and LLM parameters.
        """
        flat = config.to_flat_dict()

        if "temperature" in flat and hasattr(self.agent, "temperature"):
            self.agent.temperature = flat["temperature"]

        if "system_prompt" in flat:
            self.set_system_prompt(flat["system_prompt"])
            # Store for potential prompt modification
            self._config.system_prompt = flat["system_prompt"]

        # Try to modify the LLM's parameters directly
        llm = self._find_llm()
        if llm:
            if "temperature" in flat and flat["temperature"] is not None:
                try:
                    llm.temperature = flat["temperature"]
                except Exception:
                    pass
            if "max_tokens" in flat and flat["max_tokens"] is not None:
                try:
                    llm.max_tokens = flat["max_tokens"]
                except Exception:
                    pass

        # Update internal config state
        for key, value in flat.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def apply_retrieval_override(self, **kwargs: Any) -> None:
        # Find the retriever on the adapter or its underlying chain
        retriever = getattr(self, "_retriever", None)
        if retriever is None:
            chain = getattr(self, "_chain", None) or getattr(self, "_agent", None)
            retriever = getattr(chain, "retriever", None) if chain else None
        if retriever is None:
            return
        if "retrieval_k" in kwargs:
            try:
                retriever.search_kwargs = {
                    **getattr(retriever, "search_kwargs", {}),
                    "k": int(kwargs["retrieval_k"]),
                }
            except Exception:
                pass   # Adapter doesn't support search_kwargs — no-op
        # Other overrides (chunk_size, reranker) are no-op in Phase 4
