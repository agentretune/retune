"""Framework adapters -- convert any execution into universal ExecutionTrace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retune.core.exceptions import AdapterNotFoundError

if TYPE_CHECKING:
    from retune.adapters.base import BaseAdapter

# Registry of available adapters
_ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[BaseAdapter]) -> None:
    """Register an adapter class by name."""
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str, agent: object, **kwargs) -> BaseAdapter:
    """Get an adapter instance by name.

    Args:
        name: Adapter name (e.g., "langchain", "langgraph", "custom")
        agent: The agent/chain/pipeline to wrap
        **kwargs: Additional adapter-specific arguments

    Returns:
        Configured adapter instance
    """
    if name not in _ADAPTER_REGISTRY:
        # Try lazy loading
        _lazy_load(name)

    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(_ADAPTER_REGISTRY.keys()) or "none"
        raise AdapterNotFoundError(
            f"Adapter '{name}' not found. Available: {available}. "
            f"Install the required extras: pip install retune[{name}]"
        )

    adapter_cls = _ADAPTER_REGISTRY[name]
    adapter = adapter_cls(agent=agent, **kwargs)
    return adapter


def _lazy_load(name: str) -> None:
    """Attempt to import and register an adapter by name."""
    try:
        if name == "langchain":
            from retune.adapters.langchain_adapter import LangChainAdapter

            register_adapter("langchain", LangChainAdapter)
        elif name == "langgraph":
            from retune.adapters.langgraph_adapter import LangGraphAdapter

            register_adapter("langgraph", LangGraphAdapter)
        elif name == "custom":
            from retune.adapters.custom_adapter import CustomAdapter

            register_adapter("custom", CustomAdapter)
    except ImportError:
        pass  # Adapter's dependencies not installed


def list_adapters() -> list[str]:
    """List all registered adapter names."""
    return list(_ADAPTER_REGISTRY.keys())


__all__ = [
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
