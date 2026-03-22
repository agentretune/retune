"""LangGraph adapter — wraps compiled StateGraphs with stream-based step capture."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from retune.adapters.base import BaseAdapter
from retune.core.enums import StepType
from retune.core.exceptions import AdapterError
from retune.core.models import ExecutionTrace, OptimizationConfig, Step

try:
    from langgraph.graph.state import CompiledStateGraph  # noqa: F401

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


def _infer_step_type(node_name: str, output: Any) -> StepType:
    """Infer the step type from the node name and output."""
    name_lower = node_name.lower()
    if any(kw in name_lower for kw in ("retriev", "search", "fetch", "rag")):
        return StepType.RETRIEVAL
    if any(kw in name_lower for kw in ("tool", "action", "execute")):
        return StepType.TOOL_CALL
    if any(kw in name_lower for kw in ("reason", "think", "plan")):
        return StepType.REASONING
    return StepType.LLM_CALL


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph compiled state graphs.

    Uses LangGraph's stream() to capture per-node execution steps,
    giving natural granularity without needing custom callbacks.

    Usage:
        from langgraph.graph import StateGraph

        graph = StateGraph(MyState)
        # ... build graph ...
        compiled = graph.compile()

        adapter = LangGraphAdapter(agent=compiled)
        trace = adapter.run("What is AI?")
    """

    def __init__(self, agent: Any, input_key: str = "messages", **kwargs: Any) -> None:
        if not HAS_LANGGRAPH:
            raise AdapterError(
                "langgraph is required. Install with: pip install retune[langgraph]"
            )
        super().__init__(agent=agent, **kwargs)
        self._input_key = input_key
        self._config = OptimizationConfig()

    def run(
        self,
        query: str,
        config: OptimizationConfig | None = None,
        **kwargs: Any,
    ) -> ExecutionTrace:
        if config:
            self.apply_config(config)

        steps: list[Step] = []
        started_at = datetime.now(timezone.utc)
        final_output: Any = None

        try:
            # Prepare input — LangGraph expects a dict matching the state schema
            graph_input = kwargs.pop("graph_input", None)
            if graph_input is None:
                # Default: try common patterns
                try:
                    from langchain_core.messages import HumanMessage

                    graph_input = {self._input_key: [HumanMessage(content=query)]}
                except ImportError:
                    graph_input = {self._input_key: query}

            # Stream to capture per-node outputs
            for node_output in self.agent.stream(graph_input, **kwargs):
                step_started = datetime.now(timezone.utc)

                for node_name, output in node_output.items():
                    step_type = _infer_step_type(node_name, output)

                    # Extract meaningful data from the output
                    if isinstance(output, dict):
                        output_data = {
                            k: str(v)[:2000] if not isinstance(v, (int, float, bool)) else v
                            for k, v in output.items()
                        }
                    else:
                        output_data = {"output": str(output)[:2000]}

                    steps.append(
                        Step(
                            step_type=step_type,
                            name=node_name,
                            input_data={"query": query},
                            output_data=output_data,
                            started_at=step_started,
                            ended_at=datetime.now(timezone.utc),
                        )
                    )

                    final_output = output

        except Exception as e:
            raise AdapterError(f"LangGraph execution failed: {e}") from e

        ended_at = datetime.now(timezone.utc)

        # Extract response text from final output
        response_text = self._extract_response(final_output)

        return ExecutionTrace(
            trace_id=str(uuid4()),
            query=query,
            response=response_text,
            steps=steps,
            config_snapshot=self._config.to_flat_dict(),
            started_at=started_at,
            ended_at=ended_at,
        )

    def _extract_response(self, output: Any) -> str:
        """Extract human-readable response from LangGraph output."""
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            # Try common state keys
            for key in ("messages", "output", "answer", "response"):
                if key in output:
                    val = output[key]
                    if isinstance(val, list) and val:
                        last = val[-1]
                        return getattr(last, "content", str(last))
                    return str(val)
            return str(output)
        return str(output)

    def get_config(self) -> OptimizationConfig:
        return self._config.model_copy()

    def apply_config(self, config: OptimizationConfig) -> None:
        flat = config.to_flat_dict()
        for key, value in flat.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
