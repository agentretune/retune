"""Collect recent local traces for upload to the cloud optimizer."""
from __future__ import annotations

from typing import Any, Protocol


class _StorageLike(Protocol):
    def get_traces(self, limit: int) -> list[Any]: ...


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Normalize ExecutionTrace / Pydantic model / dict into a plain JSON-safe dict."""
    # Pydantic v2 model
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass
    # Dataclass or Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj
    # Last-resort: attribute introspection
    if hasattr(obj, "__dict__"):
        return {
            k: v for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    return {"value": str(obj)}


def collect_last_n_local_traces(storage: _StorageLike, n: int) -> list[dict[str, Any]]:
    """Return up to `n` most recent traces from local storage, newest first.

    Normalizes ExecutionTrace instances (or any Pydantic model) into plain
    JSON-serializable dicts so the result can be sent to the cloud as-is.
    """
    raw = storage.get_traces(limit=n)
    return [_to_plain_dict(t) for t in raw]
