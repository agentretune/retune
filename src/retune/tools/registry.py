"""Tool registry — discover, register, and manage Retune tools.

Provides a central registry for all tools (built-in + user-defined + community).
Tools are lazily loaded and can be retrieved by name for use in LangGraph agents.
"""

from __future__ import annotations

import logging
from typing import Any

from retune.tools.base import RetuneTool

logger = logging.getLogger(__name__)

_global_registry: ToolRegistry | None = None


class ToolRegistry:
    """Central registry for Retune tools.

    Usage:
        registry = ToolRegistry()
        registry.register(MyCustomTool())

        tool = registry.get("my_custom_tool")
        result = tool.execute(x="hello")

        # Get all tools as LangChain tools for LangGraph
        lc_tools = registry.get_langchain_tools()
    """

    def __init__(self, load_builtins: bool = True) -> None:
        self._tools: dict[str, RetuneTool] = {}
        if load_builtins:
            self._load_builtins()

    def register(self, tool: RetuneTool) -> None:
        """Register a tool. Overwrites if name already exists."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> RetuneTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all(self) -> list[RetuneTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_langchain_tools(self, names: list[str] | None = None) -> list[Any]:
        """Convert tools to LangChain StructuredTools for LangGraph.

        Args:
            names: Specific tool names to convert. None = all tools.
        """
        tools = []
        target = names or self.list_tools()
        for name in target:
            tool = self._tools.get(name)
            if tool:
                tools.append(tool.to_langchain_tool())
        return tools

    def _load_builtins(self) -> None:
        """Load all built-in tools."""
        try:
            from retune.tools.builtin import get_builtin_tools

            for tool in get_builtin_tools():
                self.register(tool)
        except Exception as e:
            logger.debug(f"Could not load built-in tools: {e}")


def get_registry() -> ToolRegistry:
    """Get the global tool registry (creates one if needed)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry
