"""Minimal local FastAPI dashboard for Retune SDK traces.

Serves a single HTML page listing recent traces from the local SQLiteStorage.
Pure-SDK — does not depend on any cloud/server code.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


app = FastAPI(title="Retune Local Dashboard")


def _get_storage():
    """Build a SQLiteStorage against the configured path."""
    from retune.storage.sqlite_storage import SQLiteStorage
    path = os.environ.get("RETUNE_STORAGE_PATH", "./retune.db")
    return SQLiteStorage(path)


def _render_trace_row(trace) -> str:
    if hasattr(trace, "query"):
        query = (str(trace.query) or "")[:80].replace("<", "&lt;").replace(">", "&gt;")
        response = (str(trace.response) or "")[:80].replace("<", "&lt;").replace(">", "&gt;")
        dur = trace.duration_ms if hasattr(trace, "duration_ms") else 0
        mode = trace.mode.value if hasattr(trace.mode, "value") else str(trace.mode)
        steps = len(trace.steps) if hasattr(trace, "steps") else 0
        scores = ", ".join(
            f"{r.evaluator_name}={r.score:.2f}"
            for r in (trace.eval_results or [])
        ) or "N/A"
        return (
            f"<tr><td>{query}</td><td>{response}</td>"
            f"<td>{mode}</td><td>{steps}</td>"
            f"<td>{dur:.0f} ms</td><td>{scores}</td></tr>"
        )
    # Fallback for dict
    query = (trace.get("query") or "")[:80].replace("<", "&lt;").replace(">", "&gt;")
    response = (trace.get("response") or "")[:80].replace("<", "&lt;").replace(">", "&gt;")
    dur = trace.get("duration_ms") or 0
    return f"<tr><td>{query}</td><td>{response}</td><td colspan='4'>{dur:.0f} ms</td></tr>"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    try:
        storage = _get_storage()
        traces = storage.get_traces(limit=100)
    except Exception as e:
        return f"""
        <html><body>
        <h1>Retune Local Dashboard</h1>
        <p style="color:red">Could not load traces: {e}</p>
        <p>Set RETUNE_STORAGE_PATH to point at your retune.db file.</p>
        </body></html>
        """
    if not traces:
        return """
        <html><body>
        <h1>Retune Local Dashboard</h1>
        <p>No traces yet. Wrap your agent with Retuner(mode=Mode.OBSERVE, ...) and run some queries.</p>
        </body></html>
        """
    rows = "\n".join(_render_trace_row(t) for t in traces)
    return f"""
    <html>
    <head>
      <title>Retune Local Dashboard</title>
      <style>
        body {{ font-family: system-ui, sans-serif; margin: 2em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
        th {{ background: #f5f5f5; }}
      </style>
    </head>
    <body>
      <h1>Retune Local Dashboard</h1>
      <p>{len(traces)} recent traces (local SQLite).</p>
      <table>
        <tr><th>Query</th><th>Response</th><th>Mode</th><th>Steps</th><th>Duration</th><th>Eval Scores</th></tr>
        {rows}
      </table>
    </body>
    </html>
    """


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
