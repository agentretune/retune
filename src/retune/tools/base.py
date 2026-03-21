"""Universal tool base class — MCP-compatible interface for all Retune tools.

Every tool in the system (built-in, community, user-defined) follows this format.
Tools can be converted to LangChain tools for use in LangGraph agents.

Design inspired by:
- MCP (Model Context Protocol) tool format
- Microsoft Agent Lightning span tools
- LangChain StructuredTool interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class RetuneTool(ABC, BaseModel):
    """Base class for all Retune tools.

    Implements an MCP-compatible interface:
    - name: unique identifier
    - description: what the tool does (used by LLM for tool selection)
    - input_schema: JSON Schema for the input
    - execute(**kwargs) -> dict: run the tool

    Usage:
        class MyTool(RetuneTool):
            name = "my_tool"
            description = "Does something useful"
            input_schema = {"type": "object", "properties": {"x": {"type": "string"}}}

            def execute(self, **kwargs) -> dict:
                return {"result": kwargs["x"].upper()}

        # Register globally
        registry.register(MyTool())

        # Use in LangGraph
        langchain_tool = my_tool.to_langchain_tool()
    """

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the given inputs.

        Args:
            **kwargs: Tool inputs matching input_schema

        Returns:
            Dict with tool outputs
        """
        ...

    def to_langchain_tool(self) -> Any:
        """Convert this tool to a LangChain StructuredTool for use in LangGraph.

        Returns a langchain_core.tools.StructuredTool that wraps this tool's execute method.
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            raise ImportError(
                "langchain-core is required to convert tools. "
                "Install with: pip install retune[agents]"
            )

        tool_ref = self

        def _run(**kwargs: Any) -> str:
            import json
            result = tool_ref.execute(**kwargs)
            return json.dumps(result, default=str)

        return StructuredTool.from_function(
            func=_run,
            name=self.name,
            description=self.description,
        )

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Allow direct calling: tool(x="hello")."""
        return self.execute(**kwargs)
