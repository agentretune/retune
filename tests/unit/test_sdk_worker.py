"""SDKWorker — long-poll consumer."""
from __future__ import annotations

from unittest.mock import MagicMock

from retune.optimizer.worker import SDKWorker


def test_worker_exits_on_job_complete():
    client = MagicMock()
    # First poll returns a run_candidate, second returns job_complete.
    client.poll_pending.side_effect = [
        {"type": "run_candidate", "candidate_id": "c1",
         "config_overrides": {}, "query_set": []},
        {"type": "job_complete", "run_id": "r1",
         "report_url": "/x"},
    ]
    runner_calls = []
    def runner(overrides, queries):
        runner_calls.append(overrides)
        return ({"trace": "fake"}, {"llm_judge": 7.0})

    worker = SDKWorker(client=client, run_id="r1", candidate_runner=runner)
    report_url = worker.run()

    assert report_url == "/x"
    assert len(runner_calls) == 1
    client.submit_result.assert_called_once()


def test_worker_raises_on_job_failed():
    client = MagicMock()
    client.poll_pending.side_effect = [
        {"type": "job_failed", "run_id": "r1", "reason": "boom"},
    ]
    worker = SDKWorker(client=client, run_id="r1",
                       candidate_runner=lambda o, q: ({}, {}))
    import pytest
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()


def test_worker_retries_on_timeout():
    client = MagicMock()
    # 2 timeouts (None) then completion
    client.poll_pending.side_effect = [
        None, None, {"type": "job_complete", "run_id": "r1", "report_url": "/y"},
    ]
    worker = SDKWorker(client=client, run_id="r1",
                       candidate_runner=lambda o, q: ({}, {}))
    assert worker.run() == "/y"
    assert client.poll_pending.call_count == 3
