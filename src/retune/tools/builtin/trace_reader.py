"""Trace reader tool — structured analysis of execution traces."""

from __future__ import annotations

from typing import Any

from retune.tools.base import RetuneTool


class TraceReaderTool(RetuneTool):
    """Reads and summarizes an execution trace into a structured analysis.

    Extracts: step breakdown, durations, token usage, retrieval details,
    tool calls, and identifies bottlenecks.
    """

    name: str = "trace_reader"
    description: str = (
        "Analyze an execution trace. Input: a serialized ExecutionTrace dict. "
        "Output: structured summary of steps, durations, token usage, "
        "retrieval details, tool call patterns, and bottleneck identification."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "trace": {"type": "object", "description": "Serialized ExecutionTrace"},
        },
        "required": ["trace"],
    }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        trace = kwargs.get("trace", {})
        steps = trace.get("steps", [])

        # Step breakdown
        step_summaries = []
        total_tokens = 0
        total_duration_ms: float = 0
        slowest_step = None
        slowest_duration: float = 0

        for step in steps:
            step_type = step.get("step_type", "unknown")
            name = step.get("name", "unknown")

            # Calculate duration
            started = step.get("started_at", "")
            ended = step.get("ended_at", "")
            duration_ms = self._calc_duration_ms(started, ended)
            total_duration_ms += duration_ms

            # Token usage
            token_usage = step.get("token_usage")
            step_tokens = 0
            if token_usage:
                step_tokens = token_usage.get("total_tokens", 0)
                total_tokens += step_tokens

            # Track slowest
            if duration_ms > slowest_duration:
                slowest_duration = duration_ms
                slowest_step = name

            summary = {
                "step_type": step_type,
                "name": name,
                "duration_ms": round(duration_ms, 1),
                "tokens": step_tokens,
            }

            # Retrieval-specific
            if step_type == "retrieval":
                output = step.get("output_data", {})
                summary["num_docs"] = output.get("num_docs", 0)
                summary["has_documents"] = bool(output.get("documents"))

            # Tool-specific
            if step_type == "tool_call":
                summary["tool_input"] = str(step.get("input_data", {}))[:200]
                summary["tool_output"] = str(step.get("output_data", {}))[:200]

            step_summaries.append(summary)

        # Step type counts
        type_counts: dict[str, int] = {}
        for s in step_summaries:
            t = s["step_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "query": trace.get("query", ""),
            "response_preview": str(trace.get("response", ""))[:300],
            "total_steps": len(steps),
            "step_type_counts": type_counts,
            "total_duration_ms": round(total_duration_ms, 1),
            "total_tokens": total_tokens,
            "slowest_step": slowest_step,
            "slowest_step_duration_ms": round(slowest_duration, 1),
            "steps": step_summaries,
            "has_reasoning": type_counts.get("reasoning", 0) > 0,
            "has_retrieval": type_counts.get("retrieval", 0) > 0,
            "has_tool_calls": type_counts.get("tool_call", 0) > 0,
        }

    def _calc_duration_ms(self, started: Any, ended: Any) -> float:
        """Calculate duration in ms from datetime strings or objects."""
        from datetime import datetime

        try:
            if isinstance(started, str) and isinstance(ended, str):
                # Handle ISO format with microseconds and timezone
                s = datetime.fromisoformat(started.replace("Z", "+00:00"))
                e = datetime.fromisoformat(ended.replace("Z", "+00:00"))
                return (e - s).total_seconds() * 1000
            elif hasattr(started, "timestamp") and hasattr(ended, "timestamp"):
                return float((ended - started).total_seconds() * 1000)
        except Exception:
            pass
        return 0.0
