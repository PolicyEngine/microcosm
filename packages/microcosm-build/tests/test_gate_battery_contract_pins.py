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
import json

import pytest

from microcosm.build import load_country_spec
from microcosm.build.gate_battery import (
    GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
    GATE_BATTERY_PRODUCER,
    GATE_BATTERY_SCHEMA_VERSION,
    EvidenceContext,
    GateBatteryRun,
    GateOutcome,
    GateStatus,
    gate_signing_key_env,
)
from microcosm.build.gates import GateResult
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE,
    UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
    UK_INPUT_MASS_REFERENCE_REGISTRY,
    uk_default_input_mass_reviewed_exclusions,
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

    def test_entry_metadata_mirrors_the_committed_spec(self) -> None:
        # The verifier pins each entry's gate, phase, and criticality, and
        # refuses not_applicable outright; all three restate the spec exactly.
        spec = load_country_spec("uk")
        assert data_contract._UK_GATE_BATTERY_ENTRY_GATES == {
            entry.id: (entry.gate, entry.phase) for entry in spec.gates.gates
        }
        assert data_contract._UK_GATE_BATTERY_DIAGNOSTIC_IDS == {
            entry.id for entry in spec.gates.gates if entry.criticality == "diagnostic"
        }
        assert all(entry.not_applicable is None for entry in spec.gates.gates)

    def test_outcome_envelope_mirrors_the_producer_construction_invariants(
        self,
    ) -> None:
        # The schema-4 checker's per-status envelope (passed -> no failures,
        # no reason; failed -> named failures, no reason; excused statuses ->
        # no failures, empty details, a non-empty reason) is a hand-mirror of
        # these construction invariants and of to_payload's emission shape.
        # Hold it to actually constructed instances so a producer change
        # breaks this test, not a release assembly.
        entry = load_country_spec("uk").gates.gates[0]

        with pytest.raises(ValueError, match="cannot pass with failures"):
            GateResult(name=entry.gate, passed=True, failures=("caveat",))
        with pytest.raises(ValueError, match="cannot fail without naming"):
            GateResult(name=entry.gate, passed=False)
        with pytest.raises(ValueError, match="requires a reason"):
            GateOutcome(entry=entry, status=GateStatus.EVIDENCE_ABSENT)
        with pytest.raises(ValueError, match="requires a reason"):
            GateOutcome(entry=entry, status=GateStatus.NOT_APPLICABLE)

        passed = GateOutcome(
            entry=entry,
            status=GateStatus.PASSED,
            result=GateResult(name=entry.gate, passed=True),
        ).to_payload()
        assert passed["failures"] == []
        assert passed["reason"] is None
        failed = GateOutcome(
            entry=entry,
            status=GateStatus.FAILED,
            result=GateResult(
                name=entry.gate, passed=False, failures=("named failure",)
            ),
        ).to_payload()
        assert failed["failures"] == ["named failure"]
        assert failed["reason"] is None
        absent = GateOutcome(
            entry=entry,
            status=GateStatus.EVIDENCE_ABSENT,
            reason="missing evidence: fixture",
        ).to_payload()
        assert absent["failures"] == []
        assert absent["details"] == {}
        assert absent["reason"] == "missing evidence: fixture"

    def test_degenerate_evidence_pin_mirrors_the_committed_register(self) -> None:
        from microcosm.build.gate_battery import _canonical_sha256
        from microcosm.build.uk_runtime.terminal_gates import (
            uk_default_degenerate_reviewed_exclusions,
        )

        records = uk_default_degenerate_reviewed_exclusions()
        payload = {
            "exclusions_policy": "committed",
            "reviewed_exclusions": {
                name: record.policy_payload()
                for name, record in sorted(records.items())
            },
        }
        assert data_contract._UK_GATE_BATTERY_DEGENERATE_EVIDENCE_SHA256 == (
            _canonical_sha256(payload)
        )

    def test_input_mass_evidence_pin_mirrors_the_wrapped_reference(self) -> None:
        from microcosm.build.gate_battery import _canonical_sha256

        reference = "efrs-post-calibration"
        records = uk_default_input_mass_reviewed_exclusions()[reference]
        assert data_contract._UK_GATE_BATTERY_INPUT_MASS_EVIDENCE_SHA256 == (
            _canonical_sha256(
                {
                    "reference": reference,
                    "reference_evidence_sha256": (
                        UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256
                    ),
                    "exclusions_policy": "committed",
                    "reviewed_exclusions": {
                        name: record.policy_payload()
                        for name, record in sorted(records.items())
                    },
                }
            )
        )

    def test_numeric_threshold_mirrors_match_the_committed_spec(self) -> None:
        spec = load_country_spec("uk")
        params = {entry.id: entry.parameters for entry in spec.gates.gates}
        input_mass = params["uk_input_mass_parity"]
        reference = input_mass["reference"]

        assert (
            data_contract._UK_MAX_TO_MEDIAN_WEIGHT_RATIO
            == (params["uk_weight_ratio"]["maximum_max_to_median_ratio"])
        )
        assert (
            data_contract._UK_MIN_ESS_FRACTION
            == (params["uk_weight_ess"]["minimum_ess_fraction"])
        )
        assert (
            data_contract._UK_MAX_TARGET_ABS_RELATIVE_ERROR
            == (params["uk_target_fit"]["max_abs_relative_error"])
        )
        assert data_contract._UK_INPUT_MASS_ACTIVE_REFERENCE == reference
        assert (
            data_contract._UK_INPUT_MASS_REFERENCE_IDENTITY
            == input_mass["reference_registry"][reference]["identity"]
        )
        assert (
            data_contract._UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256
            == (input_mass["reference_registry"][reference]["totals_sha256"])
        )
        assert input_mass["reviewed_exclusions_resource"] == (
            UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE
        )
        qrf = params["uk_qrf_tail_concentration"]
        assert (
            data_contract._UK_INPUT_MASS_RELATIVE_TOLERANCE
            == (input_mass["relative_tolerance"])
        )
        assert (
            data_contract._UK_INPUT_MASS_MINIMUM_REFERENCE_TOTAL
            == (input_mass["minimum_reference_total"])
        )
        assert data_contract._UK_QRF_TAIL_TOP_K == qrf["top_k"]
        assert data_contract._UK_QRF_TAIL_MAX_TOP_SHARE == qrf["max_top_share"]
        assert (
            data_contract._UK_QRF_TAIL_MIN_NONZERO_RECORDS
            == (qrf["min_nonzero_records"])
        )
        expected_registry = {
            name: descriptor.spec_payload()
            for name, descriptor in UK_INPUT_MASS_REFERENCE_REGISTRY.items()
        }
        assert input_mass["reference_registry"] == expected_registry


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
        run.run_phase("assembled", EvidenceContext())
        run.run_phase("transferred", EvidenceContext())
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
        run = _uk_run(
            tmp_path,
            release_candidate=True,
            release_evidence={"calibration_diagnostics_sha256": "c" * 64},
        )
        run.run_phase("preflight", EvidenceContext())
        run.run_phase("assembled", EvidenceContext())
        run.run_phase("transferred", EvidenceContext())
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


class TestCertificationMirrors:
    """The data shard's certification mirrors track the build shard."""

    def test_part_scopes_mirror_the_ownership_partition(self) -> None:
        from microcosm.build.uk_runtime.calibration_run import (
            UK_CALIBRATION_GATE_SCOPE,
            UK_LOCAL_GATE_SCOPE,
            UK_NATIONAL_GATE_SCOPE,
            UK_SHARED_GATE_IDS,
            UK_SPINE_GATE_SCOPE,
        )

        assert data_contract._UK_CERTIFICATION_PART_SCOPES["spine"] == frozenset(
            UK_SPINE_GATE_SCOPE
        )
        assert data_contract._UK_CERTIFICATION_PART_SCOPES[
            "calibration_seam"
        ] == frozenset(UK_CALIBRATION_GATE_SCOPE)
        assert data_contract._UK_CERTIFICATION_PART_SCOPES["release_cut"] == frozenset(
            UK_NATIONAL_GATE_SCOPE
        )
        assert data_contract._UK_CERTIFICATION_SHARED_GATE_IDS == frozenset(
            UK_SHARED_GATE_IDS
        )
        assert data_contract._UK_CERTIFICATION_EXCLUDED_GATE_IDS == frozenset(
            UK_LOCAL_GATE_SCOPE
        )

    def test_part_digests_mirror_the_live_scoped_manifests(self) -> None:
        from microcosm.build.uk_runtime.release_certification import (
            _PART_SCOPES,
            _scoped_digests,
        )

        for part_name, spec in _PART_SCOPES.items():
            live = _scoped_digests(
                frozenset(spec["scope"]),
                phases=tuple(spec["phases"]),
                policy_suffix=str(spec["policy_suffix"]),
            )
            mirrored = data_contract._UK_CERTIFICATION_PART_DIGESTS[part_name]
            assert (
                mirrored["gates_manifest_sha256"] == (live["gates_manifest_sha256"])
            ), part_name
            assert mirrored["policy_sha256"] == live["policy_sha256"], part_name
            assert list(data_contract._UK_CERTIFICATION_PART_PHASES[part_name]) == list(
                spec["phases"]
            )

    def test_certification_identity_mirrors(self) -> None:
        from microcosm.build.uk_runtime import release_certification

        assert data_contract._UK_RELEASE_CERTIFICATION_SCHEMA_VERSION == (
            release_certification.UK_RELEASE_CERTIFICATION_SCHEMA_VERSION
        )
        assert data_contract._UK_RELEASE_CERTIFICATION_KIND == (
            release_certification.UK_RELEASE_CERTIFICATION_KIND
        )

    def test_national_release_id_mirrors_the_build_shard(self) -> None:
        from microcosm.build.uk_runtime.release_identity import (
            UK_NATIONAL_RELEASE_ID,
        )

        assert data_contract._UK_NATIONAL_RELEASE_ID == UK_NATIONAL_RELEASE_ID


class TestDenseLineMirrors:
    """The dense joint line's contract mirrors (microcosm#762 A18)."""

    def test_release_id_mirrors_the_build_shard(self) -> None:
        from microcosm.build.uk_runtime.release_identity import UK_DENSE_RELEASE_ID

        assert data_contract._UK_DENSE_RELEASE_ID == UK_DENSE_RELEASE_ID
        assert data_contract._UK_DENSE_CUT_TAG_RE.fullmatch(
            f"{UK_DENSE_RELEASE_ID}-20260903T105422Z-ca611e43"
        )
        assert data_contract._UK_DENSE_CUT_TAG_RE.fullmatch(UK_DENSE_RELEASE_ID) is None

    def test_entry_ids_mirror_the_local_battery_scope(self) -> None:
        from microcosm.build.uk_runtime.calibration_run import UK_LOCAL_GATE_SCOPE

        assert data_contract._UK_DENSE_GATE_ENTRY_IDS == frozenset(UK_LOCAL_GATE_SCOPE)
        assert data_contract._UK_DENSE_RELEASE_BLOCKING_IDS < (
            data_contract._UK_DENSE_GATE_ENTRY_IDS
        )

    def test_scoped_digests_mirror_the_live_local_manifest(self) -> None:
        import importlib.util
        from pathlib import Path

        from microcosm.build.uk_runtime.calibration_run import UK_LOCAL_GATE_SCOPE
        from microcosm.build.uk_runtime.release_certification import _scoped_digests

        spec = importlib.util.spec_from_file_location(
            "build_uk_rowwise_candidate",
            Path(__file__).resolve().parents[3]
            / "tools"
            / "build_uk_rowwise_candidate.py",
        )
        builder = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(builder)
        live = _scoped_digests(
            frozenset(UK_LOCAL_GATE_SCOPE),
            phases=tuple(data_contract._UK_DENSE_GATE_PHASES),
            policy_suffix=str(builder._LOCAL_GATE_POLICY_SUFFIX),
        )
        for field, mirrored in data_contract._UK_DENSE_GATE_DIGESTS.items():
            assert mirrored == live[field], field
