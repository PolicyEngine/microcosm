"""The country-agnostic gate battery: taxonomy, write-then-block, attestation.

The battery's contract (populace#611): every declared gate resolves to
exactly one of five statuses, the full report is on disk before any blocking
decision, an upstream block leaves downstream entries ``unreached`` (never
``not_applicable``), and the attestation is valid over failed reports too.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest

from populace.build import FitWeightRecord, GateResult, load_country_spec
from populace.build.country_spec import GatesManifest
from populace.build.gate_battery import (
    DEFAULT_REGISTRY,
    BlockingMode,
    EvidenceContext,
    FunctionBinding,
    GateBatteryBlockedError,
    GateBatteryRun,
    GateStatus,
    evaluate_phase,
    gate_signing_key_env,
)

KEY = base64.b64encode(b"\x07" * 32).decode("ascii")


def _manifest(gates: list[dict], phases: list[str] | None = None) -> GatesManifest:
    return GatesManifest.from_mapping(
        {
            "version": 1,
            "country": "xx",
            "policy": "test battery",
            "phases": phases or ["preflight", "terminal"],
            "gates": gates,
        }
    )


def _entry(id_: str, **overrides) -> dict:
    entry = {
        "id": id_,
        "gate": "exported_nonzero",
        "phase": "terminal",
        "criticality": "release_blocking",
    }
    entry.update(overrides)
    return entry


def _binding(gate_name: str, *, passes: bool = True, raises: Exception | None = None):
    def gate(**kwargs) -> GateResult:
        if raises is not None:
            raise raises
        if passes:
            return GateResult(name=gate_name, passed=True)
        return GateResult(
            name=gate_name, passed=False, failures=(f"{gate_name} failed",)
        )

    return FunctionBinding(name=gate_name, gate=gate)


@pytest.fixture
def signing_env(monkeypatch):
    monkeypatch.setenv(gate_signing_key_env("xx"), KEY)


class TestOutcomeTaxonomy:
    def test_every_status_is_reachable_in_one_report(self, tmp_path, signing_env):
        manifest = _manifest(
            [
                _entry("pf_pass", gate="support", phase="preflight"),
                _entry("t_pass", gate="exported_nonzero"),
                _entry("t_fail", gate="nonconstant_columns"),
                _entry(
                    "t_na",
                    gate="parity",
                    not_applicable="no incumbent dataset yet",
                ),
                _entry("t_no_impl", gate="macro_realism"),
                _entry("t_no_evidence", gate="weights_audit"),
            ]
        )
        registry = {
            "support": _binding("support"),
            "exported_nonzero": _binding("exported_nonzero"),
            "nonconstant_columns": _binding("nonconstant_columns", passes=False),
            **{
                name: DEFAULT_REGISTRY[name]
                for name in ("weights_audit",)
            },
        }
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=False,
            registry=registry,
        )
        run.run_phase("preflight", EvidenceContext())
        assert run.enforce("preflight", mode=BlockingMode.BLOCKS_ARTIFACT) is False
        run.run_phase("terminal", EvidenceContext())
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        statuses = {name: gate["status"] for name, gate in report["gates"].items()}
        assert statuses == {
            "pf_pass": "passed",
            "t_pass": "passed",
            "t_fail": "failed",
            "t_na": "not_applicable",
            "t_no_impl": "evidence_absent",
            "t_no_evidence": "evidence_absent",
        }
        assert report["gates"]["t_na"]["reason"] == "no incumbent dataset yet"
        assert "macro_realism" in report["gates"]["t_no_impl"]["reason"]
        assert "fit_weight_records" in report["gates"]["t_no_evidence"]["reason"]
        assert report["gates"]["t_fail"]["failures"] == [
            "nonconstant_columns failed"
        ]

    def test_unreached_is_distinct_from_not_applicable(self, tmp_path, signing_env):
        manifest = _manifest(
            [
                _entry("pf_fail", gate="support", phase="preflight"),
                _entry("t_gate", gate="exported_nonzero"),
                _entry(
                    "t_na", gate="parity", not_applicable="no incumbent dataset"
                ),
            ]
        )
        registry = {"support": _binding("support", passes=False)}
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry=registry,
        )
        run.run_phase("preflight", EvidenceContext())
        with pytest.raises(GateBatteryBlockedError):
            run.enforce("preflight", mode=BlockingMode.BLOCKS_ARTIFACT)
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        assert report["blocked_at_phase"] == "preflight"
        assert report["phases_evaluated"] == ["preflight"]
        assert report["gates"]["t_gate"]["status"] == "unreached"
        assert report["gates"]["t_na"]["status"] == "not_applicable"
        assert report["shippable"] is False
        attestation = report["attestation"]
        assert attestation["blocked_at_phase"] == "preflight"
        assert attestation["phases_evaluated"] == ["preflight"]

    def test_blocked_battery_refuses_later_phases(self, tmp_path, signing_env):
        manifest = _manifest(
            [
                _entry("pf_fail", gate="support", phase="preflight"),
                _entry("t_gate", gate="exported_nonzero"),
            ]
        )
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry={"support": _binding("support", passes=False)},
        )
        run.run_phase("preflight", EvidenceContext())
        assert run.enforce("preflight", mode=BlockingMode.MARKS_ARTIFACT) is True
        with pytest.raises(ValueError, match="blocked at phase 'preflight'"):
            run.run_phase("terminal", EvidenceContext())


class TestWriteThenBlock:
    def test_report_is_on_disk_before_the_raise(self, tmp_path, signing_env):
        report_path = tmp_path / "terminal_gates.json"
        manifest = _manifest([_entry("t_fail", gate="exported_nonzero")], ["terminal"])
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=report_path,
            release_candidate=True,
            registry={"exported_nonzero": _binding("exported_nonzero", passes=False)},
        )
        run.run_phase("terminal", EvidenceContext())
        assert report_path.exists(), "run_phase must persist before enforce"
        with pytest.raises(GateBatteryBlockedError) as excinfo:
            run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
        assert excinfo.value.report_path == report_path
        report = json.loads(report_path.read_text())
        assert report["blocked_at_phase"] == "terminal"
        assert report["gates"]["t_fail"]["status"] == "failed"

    def test_marks_artifact_records_without_raising(self, tmp_path, signing_env):
        manifest = _manifest([_entry("t_fail", gate="exported_nonzero")], ["terminal"])
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry={"exported_nonzero": _binding("exported_nonzero", passes=False)},
        )
        run.run_phase("terminal", EvidenceContext())
        assert run.enforce("terminal", mode=BlockingMode.MARKS_ARTIFACT) is True
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        assert report["blocked_at_phase"] == "terminal"
        assert report["shippable"] is False

    def test_phases_run_in_declared_order_exactly_once(self, tmp_path, signing_env):
        manifest = _manifest(
            [
                _entry("pf", gate="support", phase="preflight"),
                _entry("t", gate="exported_nonzero"),
            ]
        )
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry={
                "support": _binding("support"),
                "exported_nonzero": _binding("exported_nonzero"),
            },
        )
        with pytest.raises(ValueError, match="out of order"):
            run.run_phase("terminal", EvidenceContext())
        run.run_phase("preflight", EvidenceContext())
        with pytest.raises(ValueError, match="out of order"):
            run.run_phase("preflight", EvidenceContext())
        run.run_phase("terminal", EvidenceContext())
        with pytest.raises(ValueError, match="already run"):
            run.run_phase("terminal", EvidenceContext())


class TestFailClosed:
    def test_raising_evaluator_fails_by_name_and_the_batch_continues(
        self, tmp_path, signing_env
    ):
        manifest = _manifest(
            [
                _entry("t_crash", gate="support"),
                _entry("t_pass", gate="exported_nonzero"),
            ],
            ["terminal"],
        )
        registry = {
            "support": _binding("support", raises=RuntimeError("evidence exploded")),
            "exported_nonzero": _binding("exported_nonzero"),
        }
        report = evaluate_phase(
            manifest, "terminal", EvidenceContext(), registry=registry
        )
        by_id = {outcome.entry.id: outcome for outcome in report.outcomes}
        assert by_id["t_crash"].status is GateStatus.FAILED
        assert "evidence exploded" in by_id["t_crash"].result.failures[0]
        assert by_id["t_pass"].status is GateStatus.PASSED

    def test_wrong_gate_name_fails_closed(self, tmp_path):
        manifest = _manifest([_entry("t", gate="support")], ["terminal"])
        registry = {"support": _binding("parity")}  # returns the wrong name
        report = evaluate_phase(
            manifest, "terminal", EvidenceContext(), registry=registry
        )
        (outcome,) = report.outcomes
        assert outcome.status is GateStatus.FAILED
        assert "expected 'support'" in outcome.result.failures[0]

    def test_unattestable_evidence_turns_a_pass_into_a_failure(self, tmp_path):
        def bad_evidence(context, parameters):
            raise ValueError("not canonicalizable")

        manifest = _manifest([_entry("t", gate="support")], ["terminal"])
        binding = FunctionBinding(
            name="support",
            gate=lambda **kwargs: GateResult(name="support", passed=True),
            evidence=bad_evidence,
        )
        report = evaluate_phase(
            manifest, "terminal", EvidenceContext(), registry={"support": binding}
        )
        (outcome,) = report.outcomes
        assert outcome.status is GateStatus.FAILED
        assert "could not be attested" in outcome.result.failures[0]


class TestEvidenceAbsence:
    def test_blocks_release_candidates_and_marks_dev_builds(
        self, tmp_path, signing_env
    ):
        manifest = _manifest([_entry("t", gate="weights_audit")], ["terminal"])
        for release_candidate, blocked in ((True, True), (False, False)):
            run = GateBatteryRun(
                manifest,
                release_id="xx-test-build",
                report_path=tmp_path / f"report_{release_candidate}.json",
                release_candidate=release_candidate,
            )
            run.run_phase("terminal", EvidenceContext())
            assert (
                run.enforce("terminal", mode=BlockingMode.MARKS_ARTIFACT) is blocked
            )
            report = json.loads(run.report_path.read_text())
            assert report["gates"]["t"]["status"] == "evidence_absent"
            assert report["shippable"] is False

    def test_diagnostic_entries_never_block(self, tmp_path, signing_env):
        manifest = _manifest(
            [
                _entry("t_diag", gate="support", criticality="diagnostic"),
                _entry("t_anchor", gate="exported_nonzero"),
            ],
            ["terminal"],
        )
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry={
                "support": _binding("support", passes=False),
                "exported_nonzero": _binding("exported_nonzero"),
            },
        )
        run.run_phase("terminal", EvidenceContext())
        assert run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT) is False
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        assert report["gates"]["t_diag"]["status"] == "failed"
        assert report["shippable"] is True


class TestDefaultRegistry:
    def test_weights_audit_end_to_end(self, tmp_path, signing_env):
        manifest = _manifest([_entry("t", gate="weights_audit")], ["terminal"])
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
        )
        context = EvidenceContext(
            artifacts={
                "fit_weight_records": (
                    FitWeightRecord("donor_fit", "design"),
                    FitWeightRecord("sneaky_fit", "none"),
                )
            }
        )
        report = run.run_phase("terminal", context)
        (outcome,) = report.outcomes
        assert outcome.status is GateStatus.FAILED
        assert "sneaky_fit" in outcome.result.failures[0]

    def test_input_mass_parity_end_to_end_with_evidence_hash(
        self, tmp_path, signing_env
    ):
        manifest = _manifest(
            [
                _entry(
                    "t",
                    gate="input_mass_parity",
                    parameters={"relative_tolerance": 0.5},
                )
            ],
            ["terminal"],
        )
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
        )
        context = EvidenceContext(
            artifacts={
                "candidate_input_mass_totals": {"employment_income": 95.0},
                "reference_input_mass_totals": {"employment_income": 100.0},
            }
        )
        report = run.run_phase("terminal", context)
        (outcome,) = report.outcomes
        assert outcome.status is GateStatus.PASSED
        assert outcome.evidence_sha256 is not None
        written = json.loads((tmp_path / "terminal_gates.json").read_text())
        assert written["evidence_sha256"]["t"] == outcome.evidence_sha256

    def test_declared_thresholds_reach_the_gate(self, tmp_path):
        manifest = _manifest(
            [
                _entry(
                    "t",
                    gate="input_mass_parity",
                    parameters={"relative_tolerance": 0.01},
                )
            ],
            ["terminal"],
        )
        context = EvidenceContext(
            artifacts={
                "candidate_input_mass_totals": {"employment_income": 95.0},
                "reference_input_mass_totals": {"employment_income": 100.0},
            }
        )
        report = evaluate_phase(manifest, "terminal", context)
        (outcome,) = report.outcomes
        assert outcome.status is GateStatus.FAILED  # 5% drift > 1% tolerance


class TestAttestation:
    def test_signature_is_valid_over_a_failed_report(self, tmp_path, signing_env):
        manifest = _manifest([_entry("t", gate="support")], ["terminal"])
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry={"support": _binding("support", passes=False)},
        )
        run.run_phase("terminal", EvidenceContext())
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        signature = report["attestation"]["signature"]
        assert signature is not None
        report["attestation"]["signature"] = None
        recomputed = hmac.new(
            base64.b64decode(KEY),
            json.dumps(
                report, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert recomputed == signature

    def test_missing_key_records_the_hole_and_is_never_shippable(self, tmp_path):
        manifest = _manifest([_entry("t", gate="support")], ["terminal"])
        run = GateBatteryRun(
            manifest,
            release_id="xx-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            registry={"support": _binding("support")},
        )
        run.run_phase("terminal", EvidenceContext())
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        assert report["gates"]["t"]["status"] == "passed"
        assert report["attestation"]["signature"] is None
        assert "signing_error" in report["attestation"]
        assert report["shippable"] is False

    def test_policy_hash_moves_with_a_threshold(self, tmp_path, signing_env):
        def payload_for(tolerance: float) -> dict:
            manifest = _manifest(
                [
                    _entry(
                        "t",
                        gate="input_mass_parity",
                        parameters={"relative_tolerance": tolerance},
                    )
                ],
                ["terminal"],
            )
            return GateBatteryRun(
                manifest,
                release_id="xx-test-build",
                report_path=tmp_path / "terminal_gates.json",
                release_candidate=True,
            ).report_payload()

        assert (
            payload_for(0.5)["policy_sha256"] != payload_for(0.25)["policy_sha256"]
        ), "a threshold outside the policy hash is not attested"


class TestBelgianCompatibility:
    def test_the_be_spec_runs_as_declared_with_named_gaps(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(gate_signing_key_env("be"), KEY)
        spec = load_country_spec("be")
        run = GateBatteryRun(
            spec.gates,
            release_id="be-test-build",
            report_path=tmp_path / "terminal_gates.json",
            release_candidate=True,
            spec_fingerprint=spec.fingerprint,
            gates_manifest_sha256=spec.resource_hashes["gates.json"],
        )
        run.run_phase("terminal", EvidenceContext())
        report = json.loads((tmp_path / "terminal_gates.json").read_text())
        assert set(report["gates"]) == {gate.id for gate in spec.gates.gates}
        assert all(
            gate["status"] == "evidence_absent" for gate in report["gates"].values()
        ), "unimplemented BE gates are named gaps, never crashes or passes"
        assert report["shippable"] is False
        assert report["gates_manifest_sha256"] == spec.resource_hashes["gates.json"]
        with pytest.raises(GateBatteryBlockedError):
            run.enforce("terminal", mode=BlockingMode.BLOCKS_ARTIFACT)
