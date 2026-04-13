"""Retuner.apply_report handles drop_tool action by mutating adapter.tools."""
from __future__ import annotations

from retune import Retuner, Mode
from retune.optimizer.models import Suggestion
from retune.optimizer.report import OptimizationReport


def test_apply_drops_named_tool():
    def agent(q: str) -> str:
        return "r"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    # Seed adapter.tools
    retuner._adapter.tools = [
        {"name": "keep_me", "description": "useful"},
        {"name": "drop_me", "description": "unused"},
    ]

    report = OptimizationReport(
        run_id="r", understanding="", summary={},
        tier1=[
            Suggestion(
                tier=1, axis="tools",
                title="Drop unused tool: drop_me",
                description="never called",
                confidence="H",
                apply_payload={"action": "drop_tool", "tool_name": "drop_me"},
            ),
        ],
        tier2=[], tier3=[], pareto_data=[],
    )

    retuner.apply_report(report, tier=1)
    names = [t["name"] if isinstance(t, dict) else t.name for t in retuner._adapter.tools]
    assert "keep_me" in names
    assert "drop_me" not in names


def test_apply_system_prompt_from_payload():
    def agent(q: str) -> str: return "r"
    retuner = Retuner(
        agent=agent, adapter="custom", mode=Mode.IMPROVE,
        api_key="rt-test", agent_purpose="test",
    )
    retuner._config.system_prompt = "OLD"

    report = OptimizationReport(
        run_id="r", understanding="", summary={},
        tier1=[
            Suggestion(
                tier=1, axis="prompt",
                title="Rewrite prompt",
                description="new prompt",
                confidence="H",
                apply_payload={"system_prompt": "NEW"},
            ),
        ],
        tier2=[], tier3=[], pareto_data=[],
    )
    retuner.apply_report(report, tier=1)
    assert retuner._config.system_prompt == "NEW"
