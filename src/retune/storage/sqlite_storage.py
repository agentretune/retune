"""SQLite storage backend — zero-config persistence for the SDK."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from retune.core.models import ExecutionTrace, OptimizationConfig
from retune.storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """SQLite-backed storage for traces and configs.

    Uses WAL mode for concurrent read support. Stores traces as JSON blobs.
    """

    def __init__(self, db_path: str = "retune.db") -> None:
        self._db_path = str(Path(db_path).resolve())
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_traces_session
                ON traces(session_id);

                CREATE INDEX IF NOT EXISTS idx_traces_created
                ON traces(created_at DESC);

                CREATE TABLE IF NOT EXISTS configs (
                    name TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_trace(self, trace: ExecutionTrace) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO traces "
                "(trace_id, session_id, query, data) VALUES (?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.session_id,
                    trace.query,
                    trace.model_dump_json(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trace(self, trace_id: str) -> ExecutionTrace | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT data FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return None
            return ExecutionTrace.model_validate_json(row["data"])
        finally:
            conn.close()

    def get_traces(
        self,
        limit: int = 50,
        session_id: str | None = None,
    ) -> list[ExecutionTrace]:
        conn = self._get_conn()
        try:
            if session_id:
                rows = conn.execute(
                    "SELECT data FROM traces WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT data FROM traces ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [ExecutionTrace.model_validate_json(row["data"]) for row in rows]
        finally:
            conn.close()

    def save_config(self, name: str, config: OptimizationConfig) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO configs (name, data) VALUES (?, ?)",
                (name, config.model_dump_json()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_config(self, name: str) -> OptimizationConfig | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT data FROM configs WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                return None
            return OptimizationConfig.model_validate_json(row["data"])
        finally:
            conn.close()

    def list_configs(self) -> list[str]:
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT name FROM configs ORDER BY name").fetchall()
            return [row["name"] for row in rows]
        finally:
            conn.close()

    def delete_traces(self, older_than_days: int = 30) -> int:
        """Delete traces older than N days. Returns count of deleted rows."""
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM traces WHERE created_at < datetime('now', ?)",
                (f"-{older_than_days} days",),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
