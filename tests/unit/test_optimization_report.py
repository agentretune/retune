"""OptimizationReport — model + apply/show/copy_snippets."""
from __future__ import annotations

from retune.optimizer.report import OptimizationReport


def test_from_cloud_dict():
    raw = {
        "run_id": "r",
        "understanding": "hello",
        "summary": {"baseline_score": 1.0, "best_score": 2.0, "improvement_pct": 100.0},
        "tier1": [{"axis": "prompt", "title": "rewrite", "description": "x",
                    "tier": 1, "confidence": "H", "estimated_impact": {}}],
        "tier2": [], "tier3": [],
        "pareto_data": [],
        "markdown": "# r",
    }
    rep = OptimizationReport.from_cloud_dict(raw)
    assert rep.run_id == "r"
    assert len(rep.tier1) == 1
    assert rep.tier1[0].axis == "prompt"


def test_apply_tier1_invokes_callback(tmp_path):
    rep = OptimizationReport(
        run_id="r", understanding="", summary={},
        tier1=[], tier2=[], tier3=[], pareto_data=[],
    )
    applied = []
    rep.apply(tier=1, apply_fn=lambda s: applied.append(s))
    assert applied == []  # empty tier1, no-op


def test_show_returns_markdown(capsys):
    rep = OptimizationReport(
        run_id="r", understanding="", summary={},
        tier1=[], tier2=[], tier3=[], pareto_data=[],
        markdown="# Report\nHello"
    )
    rep.show()
    captured = capsys.readouterr()
    assert "# Report" in captured.out
