"""SDK candidate runner applies + restores config overrides."""
from __future__ import annotations

from retune import Mode, Retuner


def test_runner_applies_system_prompt_override_and_restores():
    captured_prompts = []

    def agent(q: str) -> str:
        return "resp"

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner._config.system_prompt = "ORIGINAL"

    # Patch adapter.run to capture the active prompt at call time
    orig_run = retuner._adapter.run
    def capturing_run(q: str):
        captured_prompts.append(retuner._config.system_prompt)
        return orig_run(q)
    retuner._adapter.run = capturing_run

    runner = retuner._make_candidate_runner()
    trace, scores = runner(
        {"system_prompt": "OVERRIDDEN"},
        [{"query": "hello", "trace_id": "t1"}],
    )

    assert captured_prompts == ["OVERRIDDEN"]
    assert retuner._config.system_prompt == "ORIGINAL"


def test_runner_no_overrides_leaves_config_unchanged():
    def agent(q: str) -> str:
        return "resp"

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner._config.system_prompt = "ORIGINAL"
    runner = retuner._make_candidate_runner()

    trace, scores = runner({}, [{"query": "hello"}])
    assert retuner._config.system_prompt == "ORIGINAL"


def test_runner_applies_few_shot_examples_override():
    captured = []

    def agent(q: str) -> str:
        return "resp"

    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner._config.few_shot_examples = [{"input": "old", "output": "old"}]

    # Patch adapter.run to capture the active examples at call time
    orig_run = retuner._adapter.run
    def capturing_run(q: str):
        captured.append(list(retuner._config.few_shot_examples))
        return orig_run(q)
    retuner._adapter.run = capturing_run

    new_examples = [{"input": "NEW_IN", "output": "NEW_OUT"}]
    runner = retuner._make_candidate_runner()
    trace, scores = runner(
        {"few_shot_examples": new_examples},
        [{"query": "hi"}],
    )

    assert captured[0] == new_examples
    assert retuner._config.few_shot_examples == [{"input": "old", "output": "old"}]
