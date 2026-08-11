"""The schema-4 verifier's mirrors, held to the producer they mirror.

microcosm-data deliberately does not import microcosm-build, so every
expectation its schema-4 checker enforces is a hand-mirrored constant.
These tests are the lockstep: they import both shards (tests may) and hold
each mirror equal to the live producer — the executor's constants, the
committed UK spec's digests, and the canonical-JSON signature scheme.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import inspect
import json

import pytest

from microcosm.build import load_country_spec
from microcosm.build.gate_battery import (
    GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
    GATE_BATTERY_PRODUCER,
    GATE_BATTERY_SCHEMA_VERSION,
    EvidenceContext,
    GateBatteryRun,
    gate_signing_key_env,
)
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
)
from microcosm.data import contract as data_contract

KEY = base64.b64encode(b"\x07" * 32).decode("ascii")


def _uk_run(tmp_path, **overrides) -> GateBatteryRun:
    arguments = {
        "release_id": "populace-uk-2023-frs-k535080",
        "report_path": tmp_path / "terminal_gates.json",
        "release_candidate": False,
        "registry": UK_GATE_REGISTRY,
        **overrides,
    }
    return GateBatteryRun(load_country_spec("uk").gates, **arguments)


class TestMirrorConstants:
    def test_executor_identity_mirrors(self) -> None:
        assert (
            data_contract._UK_GATE_BATTERY_SCHEMA_VERSION == GATE_BATTERY_SCHEMA_VERSION
        )
        assert (
            data_contract._UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION
            == GATE_BATTERY_ATTESTATION_SCHEMA_VERSION
        )
        assert data_contract._UK_GATE_BATTERY_PRODUCER == GATE_BATTERY_PRODUCER
        assert data_contract._UK_GATE_BATTERY_SIGNING_KEY_ENV == (
            gate_signing_key_env("uk")
        )

    def test_vintage_pins_mirror_the_committed_spec(self, tmp_path) -> None:
        run = _uk_run(tmp_path)
        payload = run.report_payload()
        assert data_contract._UK_GATE_BATTERY_POLICY_SHA256 == payload["policy_sha256"]
        assert (
            data_contract._UK_GATE_BATTERY_GATES_MANIFEST_SHA256
            == payload["gates_manifest_sha256"]
        )
        assert (
            data_contract._UK_GATE_BATTERY_SPEC_FINGERPRINT
            == payload["spec_fingerprint"]
        )

    def test_entry_membership_mirrors_the_committed_spec(self) -> None:
        spec = load_country_spec("uk")
        assert data_contract._UK_GATE_BATTERY_ENTRY_IDS == {
            entry.id for entry in spec.gates.gates
        }
        assert set(data_contract._UK_GATE_BATTERY_ENTRY_LEGACY_NAMES) <= {
            entry.id for entry in spec.gates.gates
        }

    def test_input_mass_evidence_pin_mirrors_the_wrapped_reference(self) -> None:
        from microcosm.build.gate_battery import _canonical_sha256

        assert data_contract._UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256 == (
            _canonical_sha256(
                {"reference_evidence_sha256": (UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256)}
            )
        )


class TestProducerRoundTrip:
    """A real executor report against the real verifier.

    The report is produced unarmed (no build evidence), so verdict-driven
    refusals are expected; what must NOT appear is any identity, pin, or
    signature failure — those would mean a mirror drifted from the producer.
    """

    MIRROR_DRIFT_NEEDLES = (
        "policy_sha256 does not match",
        "gates_manifest_sha256 does not match",
        "spec_fingerprint does not match",
        "producer must name",
        "signature does not authenticate",
        "signing_key_sha256 does not identify",
        "gate_outcomes_sha256 does not match",
        "exactly the declared UK entry ids",
    )

    def test_signature_scheme_is_verifiable_across_the_shards(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(gate_signing_key_env("uk"), KEY)
        run = _uk_run(tmp_path)
        run.run_phase("preflight", EvidenceContext())
        run.run_phase("terminal", EvidenceContext())
        report = json.loads((tmp_path / "terminal_gates.json").read_text())

        signature = report["attestation"]["signature"]
        assert signature is not None
        report["attestation"]["signature"] = None
        recomputed = hmac.new(
            base64.b64decode(KEY),
            data_contract._canonical_json_bytes(report),
            hashlib.sha256,
        ).hexdigest()
        assert recomputed == signature, (
            "the data shard's canonical JSON must reproduce the producer's signed bytes"
        )

    def test_a_real_report_survives_every_mirror_check(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(gate_signing_key_env("uk"), KEY)
        arguments: dict = {}
        if "release_evidence" in inspect.signature(GateBatteryRun.__init__).parameters:
            arguments["release_evidence"] = {"calibration_diagnostics_sha256": "c" * 64}
        else:  # pragma: no cover - pre-consumer-flip executors only
            pytest.skip(
                "GateBatteryRun has no release_evidence slot yet; flip when "
                "the consumer PR (uk-battery-consumer-a2) merges."
            )
        run = _uk_run(tmp_path, release_candidate=True, **arguments)
        run.run_phase("preflight", EvidenceContext())
        run.run_phase("terminal", EvidenceContext())
        report = json.loads((tmp_path / "terminal_gates.json").read_text())

        failures: list[str] = []
        data_contract._check_uk_gate_battery_report(
            report,
            release_id="populace-uk-2023-frs-k535080",
            calibration_diagnostics_sha256="c" * 64,
            build_manifest=None,
            calibration_diagnostics=None,
            failures=failures,
        )
        # Unarmed evidence: every blocking entry is a named gap, so verdict
        # refusals are the honest outcome. Mirror drift is not.
        assert failures, "an unarmed candidate report cannot verify clean"
        drifted = [
            line
            for line in failures
            if any(needle in line for needle in self.MIRROR_DRIFT_NEEDLES)
        ]
        assert drifted == [], drifted
