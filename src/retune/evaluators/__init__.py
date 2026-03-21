"""Evaluation engine — modular evaluators for scoring execution traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retune.evaluators.base import BaseEvaluator

_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {}


def register_evaluator(name: str, evaluator_cls: type[BaseEvaluator]) -> None:
    _EVALUATOR_REGISTRY[name] = evaluator_cls


def get_evaluator(name: str, **kwargs) -> BaseEvaluator:
    if name not in _EVALUATOR_REGISTRY:
        _lazy_load(name)
    if name not in _EVALUATOR_REGISTRY:
        available = ", ".join(_EVALUATOR_REGISTRY.keys()) or "none"
        raise ValueError(f"Evaluator '{name}' not found. Available: {available}")
    return _EVALUATOR_REGISTRY[name](**kwargs)


def _lazy_load(name: str) -> None:
    try:
        if name == "llm_judge":
            from retune.evaluators.llm_judge import LLMJudgeEvaluator

            register_evaluator("llm_judge", LLMJudgeEvaluator)
        elif name == "retrieval":
            from retune.evaluators.retrieval import RetrievalEvaluator

            register_evaluator("retrieval", RetrievalEvaluator)
        elif name == "latency":
            from retune.evaluators.latency import LatencyEvaluator

            register_evaluator("latency", LatencyEvaluator)
        elif name == "cost":
            from retune.evaluators.cost import CostEvaluator

            register_evaluator("cost", CostEvaluator)
        elif name == "deep":
            from retune.agents.evaluator.agent import EvaluatorDeepAgent

            register_evaluator("deep", EvaluatorDeepAgent)
    except ImportError:
        pass


def list_evaluators() -> list[str]:
    return list(_EVALUATOR_REGISTRY.keys())


__all__ = [
    "get_evaluator",
    "list_evaluators",
    "register_evaluator",
]
