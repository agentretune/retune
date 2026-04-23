"""run_evaluators_on_trace — aggregates scores from registered evaluators."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.evaluator_pipeline import run_evaluators_on_trace


def test_runs_each_evaluator_once_and_aggregates():
    trace = {"query": "q", "response": "r", "steps": []}

    ev1 = MagicMock()
    ev1.name = "cost"
    ev1.evaluate.return_value = MagicMock(score=0.002)

    ev2 = MagicMock()
    ev2.name = "latency"
    ev2.evaluate.return_value = MagicMock(score=1.5)

    scores = run_evaluators_on_trace([ev1, ev2], trace)
    assert scores == {"cost": 0.002, "latency": 1.5}
    ev1.evaluate.assert_called_once()
    ev2.evaluate.assert_called_once()


def test_empty_evaluator_list_returns_empty_dict():
    assert run_evaluators_on_trace([], {"query": "q", "response": "r"}) == {}


def test_evaluator_that_raises_is_logged_and_skipped():
    trace = {"query": "q", "response": "r", "steps": []}

    bad = MagicMock()
    bad.name = "broken"
    bad.evaluate.side_effect = RuntimeError("boom")

    good = MagicMock()
    good.name = "cost"
    good.evaluate.return_value = MagicMock(score=0.001)

    scores = run_evaluators_on_trace([bad, good], trace)
    assert scores == {"cost": 0.001}   # broken evaluator skipped, good still scored


def test_score_is_coerced_to_float():
    trace = {"query": "q", "response": "r", "steps": []}
    ev = MagicMock()
    ev.name = "score"
    ev.evaluate.return_value = MagicMock(score=7)   # int, not float
    scores = run_evaluators_on_trace([ev], trace)
    assert scores == {"score": 7.0}
    assert isinstance(scores["score"], float)
