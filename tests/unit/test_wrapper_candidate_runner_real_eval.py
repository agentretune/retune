"""Candidate runner runs registered evaluators, returns real scores."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune import Mode, Retuner


def test_runner_runs_registered_evaluators():
    def agent(q: str) -> str:
        return "resp to " + q

    fake_eval = MagicMock()
    fake_eval.name = "my_metric"
    fake_eval.evaluate.return_value = MagicMock(score=0.75)

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
        evaluators=[fake_eval],
    )
    runner = retuner._make_candidate_runner()
    trace, scores = runner(
        {"system_prompt": "NEW"},
        [{"query": "hello", "trace_id": "t1"}],
    )
    fake_eval.evaluate.assert_called_once()
    assert scores == {"my_metric": 0.75}


def test_runner_no_evaluators_returns_empty_scores():
    def agent(q: str) -> str: return "ok"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
        evaluators=[],
    )
    runner = retuner._make_candidate_runner()
    trace, scores = runner({}, [{"query": "hi"}])
    assert scores == {}
