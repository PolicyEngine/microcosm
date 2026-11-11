"""Replayed numerical gate outcomes retain the same enforcement policy."""

from dataclasses import replace

import pytest

from microcosm.build.country_spec import GateSelectionSpec, GatesManifest
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
    evaluate_phase,
    gate_phase_report_from_payload,
    gate_phase_report_payload,
)


def _manifest():
    return GatesManifest(
        country="xx",
        version=1,
        policy="test",
        phases=("terminal",),
        gates=(
            GateSelectionSpec(
                id="mass",
                gate="input_mass_parity",
                phase="terminal",
                criticality="release_blocking",
                parameters={"relative_tolerance": 0.01},
            ),
        ),
    )


def test_replayed_gate_failure_is_persisted_before_it_blocks(tmp_path):
    gates = _manifest()
    report = evaluate_phase(
        gates,
        "terminal",
        EvidenceContext(
            artifacts={
                "candidate_input_mass_totals": {"income": 80.0},
                "reference_input_mass_totals": {"income": 100.0},
            }
        ),
    )
    payload = gate_phase_report_payload(report, gates=gates)
    restored = gate_phase_report_from_payload(payload, gates=gates)
    run = GateBatteryRun(
        gates,
        release_id="replay",
        report_path=tmp_path / "gates.json",
        release_candidate=False,
    )
    run.record_phase(restored)
    with pytest.raises(GateBatteryBlockedError):
        run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
    assert run.report_path.is_file()
    assert run.report_payload()["blocked_at_phase"] == "terminal"
    assert run.phase_report("terminal").failures == report.failures


def test_replay_rejects_changed_policy_and_missing_outcomes():
    gates = _manifest()
    report = evaluate_phase(gates, "terminal", EvidenceContext())
    payload = gate_phase_report_payload(report, gates=gates)
    changed = replace(
        gates, gates=(replace(gates.gates[0], parameters={"relative_tolerance": 1.0}),)
    )
    with pytest.raises(ValueError, match="manifest"):
        gate_phase_report_from_payload(payload, gates=changed)
    payload["outcomes"] = []
    with pytest.raises(ValueError, match="outcomes"):
        gate_phase_report_from_payload(payload, gates=gates)
