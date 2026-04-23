"""UsageGate — per-run gating via cloud."""
from __future__ import annotations

from retune.usage_gate import UsageGate


def test_reflect_cloud_run_count():
    gate = UsageGate(api_key="rt-x")
    gate.note_preauthorize_response({"runs_remaining": 12})
    status = gate.get_status()
    assert status["remaining"] == 12


def test_local_count_noop_when_cloud_authoritative():
    """Without calling note_preauthorize_response, local-only behavior still works."""
    gate = UsageGate(api_key=None)
    assert gate.check("optimize") is True
