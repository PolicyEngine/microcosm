"""Shared signed green-certification fixtures for UK release tests."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import (
    BlockingMode,
    EvidenceContext,
    FunctionBinding,
    GateBatteryRun,
    gate_signing_key_env,
)
from microcosm.build.gates import GateResult
from microcosm.build.uk_runtime.calibration_run import (
    UK_CALIBRATION_GATE_SCOPE,
    UK_NATIONAL_GATE_SCOPE,
    UK_SPINE_GATE_SCOPE,
    resign_uk_gate_report,
    uk_scoped_gate_manifest,
)
from microcosm.build.uk_runtime.release_certification import (
    uk_release_cut_scope_exclusions,
)

TEST_KEY = base64.b64encode(bytes(range(32))).decode("ascii")


@pytest.fixture(name="uk_certification_signing_key", autouse=True)
def signing_key(monkeypatch):
    monkeypatch.setenv(gate_signing_key_env("uk"), TEST_KEY)


def stub_registry():
    """Return a registry that passes every declared gate over the real spec."""

    spec = load_country_spec("uk").gates
    parameter_keys: dict[str, set[str]] = {}
    for entry in spec.gates:
        parameter_keys.setdefault(entry.gate, set()).update(entry.parameters)

    def passing(name):
        def gate(**_kwargs):
            return GateResult(name=name, passed=True)

        return gate

    return {
        gate: FunctionBinding(
            name=gate,
            gate=passing(gate),
            parameter_keys=frozenset(keys),
        )
        for gate, keys in parameter_keys.items()
    }


def write_part(
    path: Path,
    scope,
    phases,
    *,
    release_id,
    release_candidate,
    release_evidence=None,
    augment=None,
    block_phase=None,
):
    registry = stub_registry()
    manifest = uk_scoped_gate_manifest(
        frozenset(scope),
        phases=tuple(phases),
        policy_suffix={
            frozenset(UK_SPINE_GATE_SCOPE): "spine_build_scope",
            frozenset(UK_CALIBRATION_GATE_SCOPE): "calibration_seam_scope",
            frozenset(UK_NATIONAL_GATE_SCOPE): "release_cut_scope",
        }[frozenset(scope)],
    )
    battery = GateBatteryRun(
        manifest,
        release_id=release_id,
        report_path=path,
        release_candidate=release_candidate,
        registry=registry,
        release_evidence=release_evidence or {},
    )
    for phase in phases:
        battery.run_phase(phase, EvidenceContext(artifacts={}))
        battery.enforce(phase, mode=BlockingMode.MARKS_ARTIFACT)
    payload = battery.report_payload()
    if augment:
        payload.update(augment)
        resign_uk_gate_report(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(name="green_certification_inputs")
def green_certification_inputs(tmp_path: Path):
    """Three green signed parts plus a closed identity join."""

    candidate = tmp_path / "microcosm_uk_2024.h5"
    candidate.write_bytes(b"candidate-bytes")
    candidate_sha = sha256(candidate)
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text('{"targets": []}', encoding="utf-8")
    diagnostics_sha = sha256(diagnostics)

    spine_report = tmp_path / "spine.spine_gates.json"
    write_part(
        spine_report,
        UK_SPINE_GATE_SCOPE,
        ("assembled", "transferred"),
        release_id="uk-frs-spine-test",
        release_candidate=True,
    )
    seam_report = tmp_path / "terminal_gates.json"
    write_part(
        seam_report,
        UK_CALIBRATION_GATE_SCOPE,
        ("terminal",),
        release_id="dev-seam-test",
        release_candidate=False,
        release_evidence={"calibration_diagnostics_sha256": diagnostics_sha},
        augment={
            "posture": "calibration_seam",
            "scope_exclusions": {},
            "aggregate_admin_measurement": {},
        },
    )
    release_cut_report = tmp_path / "release_cut_gates.json"
    write_part(
        release_cut_report,
        UK_NATIONAL_GATE_SCOPE,
        ("preflight", "terminal"),
        release_id="uk-757-first-certified-cut",
        release_candidate=True,
        release_evidence={"calibration_diagnostics_sha256": diagnostics_sha},
        augment={
            "posture": "release_cut",
            "scope_exclusions": uk_release_cut_scope_exclusions(),
            "aggregate_admin_measurement": {},
        },
    )
    seam_report_sha = sha256(seam_report)
    sidecar = {
        "stages": ["frs_spine"],
        "spine_gate_report": {
            "path": str(spine_report),
            "sha256": sha256(spine_report),
        },
    }
    build_record = {
        "run_config": {
            "doctrine": {"epochs": 1500},
            "doctrine_overrides": {"epochs": {"default": 256, "effective": 1500}},
        },
        "spine_provenance": {
            "spine_gate_report": {"sha256": sha256(spine_report)},
        },
        "artifacts": {
            "staging_h5": {"sha256": candidate_sha},
            "diagnostics_json": {"sha256": diagnostics_sha},
            "terminal_gate_json": {"sha256": seam_report_sha},
        },
    }
    score_receipt = tmp_path / "score_vs_enhanced_frs.json"
    score_receipt.write_text(
        json.dumps(
            {
                "artifacts": {
                    "candidate": {"sha256": candidate_sha, "size_bytes": 15},
                    "incumbent": {"sha256": "9" * 64, "size_bytes": 1},
                },
                "candidate_target_wins": 293,
            }
        ),
        encoding="utf-8",
    )
    return {
        "release_id": "uk-757-first-certified-cut",
        "candidate_name": "microcosm_uk_2024",
        "candidate_path": candidate,
        "candidate_sha256": candidate_sha,
        "spine_report_path": spine_report,
        "seam_report_path": seam_report,
        "release_cut_report_path": release_cut_report,
        "spine_sidecar": sidecar,
        "build_record": build_record,
        "score_receipt_path": score_receipt,
        "exclusions_evaluated_on": date(2026, 8, 27),
        "certification_path": tmp_path / "release_certification.json",
    }
