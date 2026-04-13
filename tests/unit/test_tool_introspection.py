"""Tool introspection — reads tool metadata from various adapter shapes."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.tool_introspection import introspect_tools


def test_adapter_with_tools_attribute_list_of_dicts():
    adapter = MagicMock()
    adapter.tools = [
        {"name": "search", "description": "Search the web", "args_schema": {}},
        {"name": "calc", "description": "Do math", "args_schema": {}},
    ]
    tools = introspect_tools(adapter)
    assert len(tools) == 2
    assert tools[0].name == "search"
    assert tools[0].description == "Search the web"


def test_langchain_basetools():
    """LangChain BaseTool objects expose .name, .description, .args_schema."""
    class FakeTool:
        name = "lc_tool"
        description = "A langchain tool"
        args_schema = None
    adapter = MagicMock()
    adapter.tools = [FakeTool()]
    tools = introspect_tools(adapter)
    assert tools[0].name == "lc_tool"


def test_adapter_without_tools_returns_empty():
    adapter = object()   # no .tools attribute at all
    assert introspect_tools(adapter) == []


def test_adapter_with_empty_tools():
    adapter = MagicMock()
    adapter.tools = []
    assert introspect_tools(adapter) == []


def test_adapter_none_returns_empty():
    assert introspect_tools(None) == []
