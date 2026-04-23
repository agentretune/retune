"""Extract tool metadata from an adapter at optimize-time.

Handles:
- Plain list of dicts: [{"name": ..., "description": ..., "args_schema": ...}]
- LangChain BaseTool instances (duck-typed: .name, .description, .args_schema)
- LangGraph tool nodes
- Adapters without a .tools attribute → empty list (graceful, no error)

Never parses user source files — purely in-memory introspection.
"""
from __future__ import annotations

import logging
from typing import Any

from retune.optimizer.tool_metadata import ToolMetadata

logger = logging.getLogger(__name__)


def _extract_one(tool_obj: Any) -> ToolMetadata | None:
    """Best-effort extraction from a single tool object / dict."""
    try:
        if isinstance(tool_obj, dict):
            return ToolMetadata(
                name=str(tool_obj.get("name", "")),
                description=str(tool_obj.get("description", "")),
                args_schema=dict(tool_obj.get("args_schema") or {}),
                is_async=bool(tool_obj.get("is_async", False)),
            )
        # Duck-typed: name + description required
        name = getattr(tool_obj, "name", None)
        if not name:
            return None
        description = str(getattr(tool_obj, "description", "") or "")
        args_schema_raw = getattr(tool_obj, "args_schema", None) or {}
        if hasattr(args_schema_raw, "model_json_schema"):
            # Pydantic model
            args_schema = args_schema_raw.model_json_schema()
        elif isinstance(args_schema_raw, dict):
            args_schema = args_schema_raw
        else:
            args_schema = {}
        is_async = (
            getattr(tool_obj, "is_async", False)
            or getattr(tool_obj, "coroutine", False) is not None
        )
        return ToolMetadata(
            name=str(name),
            description=description,
            args_schema=args_schema,
            is_async=bool(is_async),
        )
    except Exception as e:
        logger.debug("Tool extraction failed for %r: %s", tool_obj, e)
        return None


def introspect_tools(adapter: Any) -> list[ToolMetadata]:
    """Return tool metadata from the adapter. Empty list if unknown shape."""
    if adapter is None:
        return []
    tools_raw = getattr(adapter, "tools", None)
    if not tools_raw:
        return []
    out: list[ToolMetadata] = []
    try:
        iterator = list(tools_raw)
    except TypeError:
        return []
    for t in iterator:
        md = _extract_one(t)
        if md is not None:
            out.append(md)
    return out
