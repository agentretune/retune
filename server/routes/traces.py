"""Trace API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from retune.storage.sqlite_storage import SQLiteStorage
from server.config import server_settings

router = APIRouter(redirect_slashes=False)

_storage = SQLiteStorage(server_settings.storage_path)


@router.get("")
def list_traces(
    limit: int = Query(50, ge=1, le=500),
    session_id: str | None = None,
):
    """List recent execution traces."""
    traces = _storage.get_traces(limit=limit, session_id=session_id)
    return {
        "total": len(traces),
        "traces": [
            {
                "trace_id": t.trace_id,
                "session_id": t.session_id,
                "query": t.query,
                "response": str(t.response)[:300],
                "mode": t.mode.value,
                "duration_ms": round(t.duration_ms, 1),
                "total_tokens": t.total_tokens,
                "total_cost": round(t.total_cost, 6),
                "num_steps": len(t.steps),
                "weighted_score": t.weighted_score,
                "started_at": t.started_at.isoformat(),
            }
            for t in traces
        ],
    }


@router.get("/{trace_id}")
def get_trace(trace_id: str):
    """Get full trace detail including steps and evals."""
    trace = _storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.model_dump(mode="json")


@router.get("/{trace_id}/steps")
def get_trace_steps(trace_id: str):
    """Get steps for a specific trace."""
    trace = _storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "trace_id": trace_id,
        "steps": [
            {
                "step_id": s.step_id,
                "step_type": s.step_type.value,
                "name": s.name,
                "duration_ms": round(s.duration_ms, 1),
                "token_usage": s.token_usage.model_dump() if s.token_usage else None,
                "cost_usd": s.cost_usd,
                "input_preview": str(s.input_data)[:200],
                "output_preview": str(s.output_data)[:200],
            }
            for s in trace.steps
        ],
    }


@router.get("/{trace_id}/evals")
def get_trace_evals(trace_id: str):
    """Get evaluation results for a trace."""
    trace = _storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "trace_id": trace_id,
        "weighted_score": trace.weighted_score,
        "eval_results": [r.model_dump(mode="json") for r in trace.eval_results],
    }


@router.delete("/{trace_id}")
def delete_trace(trace_id: str):
    """Delete a specific trace."""
    trace = _storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    # Note: SQLiteStorage doesn't have single-delete yet, return 501
    raise HTTPException(status_code=501, detail="Single trace deletion not yet implemented")
