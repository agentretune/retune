"""Collect recent local traces for upload to the cloud optimizer."""
from __future__ import annotations

from typing import Any, Protocol


class _StorageLike(Protocol):
    def get_traces(self, limit: int) -> list[dict[str, Any]]: ...


def collect_last_n_local_traces(storage: _StorageLike, n: int) -> list[dict[str, Any]]:
    """Return up to `n` most recent traces from local storage, newest first.

    Used when Retuner.optimize(source="last_n_traces") — the SDK uploads
    these as the payload in /v1/optimize/preauthorize.
    """
    return storage.get_traces(limit=n)
