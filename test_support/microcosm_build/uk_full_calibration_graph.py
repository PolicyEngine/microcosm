"""A synthetic preflight gate report on the full-gate manifest, shared
by the calibration-graph and target-graph tests."""

from __future__ import annotations

from microcosm.graph.canonical import canonical_json


def preflight_payload(passed=True, selection=None):
    from microcosm.build.gate_battery import (
        GateOutcome,
        GatePhaseReport,
        GateStatus,
        gate_phase_report_payload,
    )
    from microcosm.build.gates import GateResult
    from microcosm.build.uk_runtime.full_gates import (
        classify_full_gate_outcomes,
        uk_full_gate_manifest,
    )

    selection = (
        selection
        if selection is not None
        else {
            "schema": "microcosm.calibrate.target-selection.v1",
            "selector": {"geography_levels": ["country"], "explicit": True},
            "included": [{"name": "count", "period": 0, "geography_level": "country"}],
            "excluded": [],
        }
    )
    gates = uk_full_gate_manifest(selection)
    report = GatePhaseReport(
        "preflight",
        tuple(
            GateOutcome(
                entry=entry,
                status=GateStatus.PASSED if passed else GateStatus.FAILED,
                result=GateResult(
                    name=entry.gate,
                    passed=passed,
                    failures=() if passed else ("fixture blocked",),
                ),
            )
            for entry in gates.gates
            if entry.phase == "preflight"
        ),
    )
    return canonical_json(
        {
            "schema_version": 1,
            "kind": "uk_full_gate_report",
            "selection_receipt": selection,
            "sample_fraction": 1.0,
            "release_candidate": True,
            "report": gate_phase_report_payload(report, gates=gates),
            "enforcement": classify_full_gate_outcomes(
                report, sample_fraction=1.0, release_candidate=True
            ),
        }
    )


__all__ = [name for name in globals() if not name.startswith("__")]
