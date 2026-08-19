"""The runnable worked example behind ``docs/gate-battery-contract.md``.

A minimal greenfield country ("xx") declares three gates and runs them on
the shared registry alone — no custom bindings, no country runtime module.
The narrated walk in the contract doc quotes this file; CI keeps the quote
honest. If this test changes shape, update the doc's worked-example
section in the same commit.
"""

from __future__ import annotations

import base64
import json

import pytest

from microcosm.build.country_spec import GatesManifest
from microcosm.build.gate_battery import (
    GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
    GATE_BATTERY_SCHEMA_VERSION,
    BlockingMode,
    EvidenceContext,
    GateBatteryBlockedError,
    GateBatteryRun,
    GateStatus,
    gate_signing_key_env,
)

KEY = base64.b64encode(b"\x2a" * 32).decode("ascii")

#: The country's whole gate policy, as pure data. In a real country this is
#: ``<cc>/gates.json`` declared in ``<cc>/country_package.json`` and loaded
#: via ``load_country_spec``; the example inlines it so the shape is visible
#: in one place.
EXAMPLE_GATES = {
    "version": 1,
    "country": "xx",
    "policy": (
        "Worked example for the gate-battery contract doc: one passing "
        "gate, one failing gate, one named evidence gap."
    ),
    "phases": ["terminal"],
    "gates": [
        {
            "id": "xx_input_mass_parity",
            "gate": "input_mass_parity",
            "phase": "terminal",
            "criticality": "release_blocking",
            "parameters": {"relative_tolerance": 0.1},
            "notes": "Total input mass within ten percent of the reference.",
        },
        {
            "id": "xx_tail_concentration",
            "gate": "tail_concentration",
            "phase": "terminal",
            "criticality": "release_blocking",
            "parameters": {
                "top_k": 1,
                "max_top_share": 0.5,
                "min_nonzero_records": 2,
            },
            "notes": (
                "No imputed column may carry half its weighted mass in a "
                "single record."
            ),
        },
        {
            "id": "xx_weights_audit",
            "gate": "weights_audit",
            "phase": "terminal",
            "criticality": "release_blocking",
            "parameters": {},
            "notes": "Every production fit resolves weighted.",
        },
    ],
}


class TestContractWorkedExample:
    def test_the_narrated_run(self, tmp_path, monkeypatch) -> None:
        # Input (f) of the onboarding checklist: the per-country signing key.
        monkeypatch.setenv(gate_signing_key_env("xx"), KEY)

        manifest = GatesManifest.from_mapping(EXAMPLE_GATES, country="xx")
        report_path = tmp_path / "terminal_gates.json"
        run = GateBatteryRun(
            manifest,
            release_id="xx-2026-example",
            report_path=report_path,
            release_candidate=False,  # a dev build: absent evidence records
            # a named gap instead of blocking
        )

        # The build tool supplies runtime evidence under the canonical
        # artifact keys the bindings declare. weights_audit's key
        # (fit_weight_records) is deliberately missing.
        context = EvidenceContext(
            artifacts={
                "candidate_input_mass_totals": {"employment_income": 95.0},
                "reference_input_mass_totals": {"employment_income": 100.0},
                "tail_concentration_values": {"imputed_gains": [100.0, 1.0, 1.0]},
                "tail_concentration_weights": {"imputed_gains": [1.0, 1.0, 1.0]},
            }
        )

        phase = run.run_phase("terminal", context)
        statuses = {o.entry.id: o.status for o in phase.outcomes}
        assert statuses == {
            # 5% drift within the declared 10% tolerance.
            "xx_input_mass_parity": GateStatus.PASSED,
            # Top-1 record carries ~98% of weighted mass > declared 50%.
            "xx_tail_concentration": GateStatus.FAILED,
            # Declared but unevidenced: a named gap, never a silent pass.
            "xx_weights_audit": GateStatus.EVIDENCE_ABSENT,
        }

        # Blocking is a two-axis decision: FAILED always blocks a
        # release-blocking entry; EVIDENCE_ABSENT blocks release candidates
        # only. This dev build blocks on the failure alone.
        assert {
            o.entry.id for o in phase.blocking_outcomes(release_candidate=False)
        } == {"xx_tail_concentration"}
        assert {
            o.entry.id for o in phase.blocking_outcomes(release_candidate=True)
        } == {"xx_tail_concentration", "xx_weights_audit"}

        # Write-then-block: enforce raises only after the full report —
        # including the block itself — is on disk.
        with pytest.raises(GateBatteryBlockedError) as blocked:
            run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
        assert "xx_tail_concentration" in str(blocked.value)

        written = json.loads(report_path.read_text())
        assert written["schema_version"] == GATE_BATTERY_SCHEMA_VERSION
        assert written["blocked_at_phase"] == "terminal"
        assert written["shippable"] is False
        assert written["gates"]["xx_weights_audit"]["status"] == "evidence_absent"
        # The pins that move with the spec: the policy hash covers every
        # declared entry (minus notes); the manifest hash covers notes too.
        assert len(written["policy_sha256"]) == 64
        assert len(written["gates_manifest_sha256"]) == 64
        assert (
            written["attestation"]["schema_version"]
            == GATE_BATTERY_ATTESTATION_SCHEMA_VERSION
        )
