"""Simplest possible test agent — echoes the query back with a prefix.

Use in: 01_pure_sdk_free.py (observability + evaluation demo).
Not useful for optimization (nothing to tune).
"""
from __future__ import annotations


def make_echo_agent():
    """Return a callable agent: query -> response."""
    def agent(query: str) -> str:
        return f"Echo: {query}"
    return agent
