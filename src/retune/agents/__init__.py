"""Deep agents -- LangGraph-based evaluator and optimizer agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retune.agents.evaluator.agent import EvaluatorDeepAgent
    from retune.agents.optimizer.agent import OptimizerDeepAgent

__all__ = [
    "EvaluatorDeepAgent",
    "OptimizerDeepAgent",
]


def __getattr__(name: str):
    if name == "EvaluatorDeepAgent":
        from retune.agents.evaluator.agent import EvaluatorDeepAgent

        return EvaluatorDeepAgent
    if name == "OptimizerDeepAgent":
        from retune.agents.optimizer.agent import OptimizerDeepAgent

        return OptimizerDeepAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
