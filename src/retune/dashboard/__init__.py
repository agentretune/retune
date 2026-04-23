"""Retune local dashboard — pure-SDK FastAPI app serving SQLite traces."""

from retune.dashboard.app import app  # noqa: F401

__all__ = ["app"]
