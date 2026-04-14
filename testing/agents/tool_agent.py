"""Agent with two tools — one actively used, one intentionally unused.

Use in: 03_trial_tool.py — ToolOptimizerAgent should propose a
Tier 1 `drop_tool` suggestion for the unused tool.

Uses a deterministic dispatch rule (no LLM needed) so the flow test is
fast + cheap. The agent.tools attribute is what introspect_tools reads.
"""
from __future__ import annotations

from typing import Any


def _search_fake(query: str) -> str:
    """Fake search tool — returns hardcoded results by query keyword."""
    if "refund" in query.lower():
        return "Refunds are available within 30 days of purchase."
    if "cancel" in query.lower():
        return "You can cancel anytime from the Billing page."
    if "payment" in query.lower():
        return "Update payment methods from Account → Billing → Payment."
    return f"No result for: {query}"


def _obsolete_metrics_tool(metric_name: str) -> str:
    """Fake metrics tool — the agent never calls this. Should be dropped."""
    return f"Metric {metric_name}: N/A"


class ToolAgent:
    """Deterministic tool-using agent.

    Routes every query to `search_fake` (based on keyword matching). Never
    calls `obsolete_metrics` even though it's registered — so
    ToolOptimizerAgent should suggest dropping it.
    """

    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = [
            {
                "name": "search_fake",
                "description": "Search the knowledge base for billing and account questions.",
                "args_schema": {"type": "object", "properties": {
                    "query": {"type": "string"}
                }, "required": ["query"]},
                "is_async": False,
            },
            {
                "name": "obsolete_metrics",
                "description": "Fetch deprecated internal metrics.",
                "args_schema": {"type": "object", "properties": {
                    "metric_name": {"type": "string"}
                }, "required": ["metric_name"]},
                "is_async": False,
            },
        ]
        self.system_prompt = "You are a help desk agent. Use search_fake to answer billing questions."
        self._call_log: list[dict[str, Any]] = []

    def __call__(self, query: str) -> str:
        # Always dispatch to search_fake
        result = _search_fake(query)
        self._call_log.append({
            "tool": "search_fake",
            "args": {"query": query},
            "result": result[:120],
        })
        return result

    def get_trace_steps(self) -> list[dict[str, Any]]:
        """Return the log in ExecutionTrace.steps[] format so the cloud
        analyzer sees "tool_call" steps."""
        return [
            {
                "type": "tool_call",
                "name": entry["tool"],
                "args": entry["args"],
                "duration_ms": 5.0,
                "status": "success",
            }
            for entry in self._call_log
        ]


def make_tool_agent() -> ToolAgent:
    return ToolAgent()


SAMPLE_QUERIES = [
    "How do I get a refund?",
    "Can I cancel my plan?",
    "How do I update payment info?",
    "Is there a refund window?",
    "Where do I cancel the subscription?",
]
