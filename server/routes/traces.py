"""Trace API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from automlagent.storage.sqlite_storage import SQLiteStorage
from server.config import server_settings

router = APIRouter()

_storage = SQLiteStorage(server_settings.storage_path)


@router.get("/")
def list_traces(limit: int = 50, session_id: str | None = None):
    """List recent execution traces."""
    traces = _storage.get_traces(limit=limit, session_id=session_id)
    return {
        "total": len(traces),
        "traces": [t.model_dump() for t in traces],
    }


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    """Get a specific trace by ID."""
    trace = _storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump()


@router.get("/{trace_id}/eval")
def get_trace_evaluations(trace_id: str):
    """Get evaluation results for a specific trace."""
    trace = _storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "trace_id": trace_id,
        "eval_results": [r.model_dump() for r in trace.eval_results],
        "weighted_score": trace.weighted_score,
    }
