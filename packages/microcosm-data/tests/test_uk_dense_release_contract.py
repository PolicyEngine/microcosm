"""The dense joint UK line's release contract (microcosm#762 A18)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from microcosm.data import contract as dc
from microcosm.data.contract import (
    ReleaseContractError,
    _canonical_json_bytes,
    _canonical_sha256,
    validate_release_dir,
)

DENSE_ID = "microcosm-uk-2024-25-dense"
CUT_TAG = f"{DENSE_ID}-20260903T105422Z-ca611e43"
KEY = base64.b64encode(b"\x07" * 32).decode()
KEY_BYTES = base64.b64decode(KEY)


@pytest.fixture(autouse=True)
def _trusted_key(monkeypatch) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _signed_report(*, blocking_status: str = "passed", release_candidate: bool = True):
    gates = {}
    for entry_id in sorted(dc._UK_DENSE_GATE_ENTRY_IDS):
        blocking = entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        gates[entry_id] = {
            "gate": entry_id.removeprefix("uk_local_"),
            "phase": "terminal",
            "criticality": "release_blocking" if blocking else "diagnostic",
            "status": blocking_status if blocking else "failed",
            "failures": [] if blocking else ["diagnostic miss"],
            "details": {},
            "reason": None,
        }
    attestation = {
        "schema_version": dc._UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
        "producer": dc._UK_GATE_BATTERY_PRODUCER,
        "country": "uk",
        "release_id": "uk-local-candidate-f100-s42-20260903T105422Z-ca611e43",
        "release_candidate": release_candidate,
        "spec_fingerprint": dc._UK_DENSE_GATE_DIGESTS["spec_fingerprint"],
        "gates_manifest_sha256": dc._UK_DENSE_GATE_DIGESTS["gates_manifest_sha256"],
        "policy_sha256": dc._UK_DENSE_GATE_DIGESTS["policy_sha256"],
        "phases": ["terminal"],
        "phases_evaluated": ["terminal"],
        "blocked_at_phase": None,
        "release_evidence": {},
        "evidence_sha256": {},
        "gate_outcomes_sha256": _canonical_sha256(gates),
        "signature_algorithm": "hmac-sha256",
        "signing_key_sha256": hashlib.sha256(KEY_BYTES).hexdigest(),
        "signature": None,
    }
    report = {
        "schema_version": dc._UK_GATE_BATTERY_SCHEMA_VERSION,
        "country": "uk",
        "release_id": attestation["release_id"],
        "release_candidate": release_candidate,
        "spec_fingerprint": attestation["spec_fingerprint"],
        "gates_manifest_sha256": attestation["gates_manifest_sha256"],
        "phases": ["terminal"],
        "phases_evaluated": ["terminal"],
        "blocked_at_phase": None,
        "shippable": blocking_status == "passed" and release_candidate,
        "gates": gates,
        "policy_sha256": attestation["policy_sha256"],
        "release_evidence": {},
        "evidence_sha256": {},
        "attestation": attestation,
        "posture": "local_candidate",
        "scope_exclusions": {},
        "aggregate_admin_measurement": None,
    }
    attestation["signature"] = hmac.new(
        KEY_BYTES, _canonical_json_bytes(report), hashlib.sha256
    ).hexdigest()
    return report


def write_dense_bundle(
    root: Path,
    *,
    release_id: str = DENSE_ID,
    revision: str = CUT_TAG,
    namespace: str = "uk_dense",
    default_datasets: dict | None = None,
    report: dict | None = None,
    drop_coverage_key: str | None = None,
    drop_households: bool = False,
) -> Path:
    release_dir = root / "releases" / release_id
    release_dir.mkdir(parents=True)
    report = report or _signed_report()
    diagnostics = {
        "schema_version": 6,
        "households": 4,
        "n_targets": 2,
        "n_records": 4,
        "final_loss": 0.01,
        "fraction_within_10pct": 0.98,
        "targets": [
            {
                "name": "ons.age.20_30@E14000001@2025",
                "target": 10.0,
                "compiled_target": 10.0,
                "initial_estimate": 9.0,
                "final_estimate": 10.0,
            },
            {
                "name": "dwp.uc.households@2025",
                "target": 5.0,
                "compiled_target": 5.0,
                "initial_estimate": 4.0,
                "final_estimate": 5.0,
            },
        ],
    }
    if drop_households:
        del diagnostics["households"]
    gate_summary = {
        "gates": {
            entry_id: {"passed": report["gates"][entry_id]["status"] == "passed"}
            for entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        },
        "diagnostic_gates": {},
        "reviewed_limitations": [],
    }
    coverage = {
        "schema_version": 1,
        "spine": {"path_name": "spine-m.h5", "sha256": "a" * 64},
        "ledger_artifact": {
            "path_name": "chronicle-uk-artifact-1cab809",
            "facts_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        "geography_ladder": {"sha256": "d" * 64},
        "incumbent": {"snapshot": {"incumbent_h5": {"sha256": "e" * 64}}},
        "doctrine": {"max_weight_ratio": 10.0},
        "measure_exclusions": {"obr.housing_benefit": {"expires_on": "2026-10-03"}},
        "signed_deferrals": {"binding_adjudications": {"x": {}}},
        "holdout": {"mean_holdout_loss": 0.2, "n_folds": 5},
        "uprating": {"applied": True, "factor": 1.03},
    }
    if drop_coverage_key:
        del coverage[drop_coverage_key]
    build_manifest = {"build_id": release_id, "build_sha": "abc1234"}
    score = {
        "candidate_fitted_surface_loss": 0.01,
        "incumbent_fitted_surface_loss": 0.2,
    }
    payloads = {
        "calibration_diagnostics.json": json.dumps(diagnostics, indent=1).encode(),
        "gate_summary.json": json.dumps(gate_summary, indent=1).encode(),
        "uk_source_coverage.json": json.dumps(coverage, indent=1).encode(),
        "uk_local_gates.json": json.dumps(report, indent=1).encode(),
        "score_vs_incumbent.json": json.dumps(score, indent=1).encode(),
        "build_manifest.json": json.dumps(build_manifest, indent=1).encode(),
    }
    for name, payload in payloads.items():
        (release_dir / name).write_bytes(payload)
    h5_bytes = b"dense-h5-stand-in"
    (root / "microcosm_uk_2025_dense.h5").write_bytes(h5_bytes)

    def artifact(kind, path, sha):
        return {
            "kind": kind,
            "path": path,
            "repo_id": "policyengine/populace-uk-private",
            "revision": revision,
            "sha256": sha,
        }

    manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "dataset_role": "non_default_local_area",
        "is_default": False,
        "default_datasets": default_datasets if default_datasets is not None else {},
        "namespace": namespace,
        "build": {"build_id": release_id},
        "artifacts": {
            "microcosm_uk_2025_dense": artifact(
                "microdata", "microcosm_uk_2025_dense.h5", _sha(h5_bytes)
            ),
            **{
                Path(name).stem: artifact("diagnostics", name, _sha(payload))
                for name, payload in payloads.items()
                if name != "build_manifest.json"
            },
        },
        "reviewed_limitations": [{"id": "rotated_holdout"}],
    }
    manifest_text = json.dumps(manifest, indent=1)
    (release_dir / "release_manifest.json").write_text(manifest_text)
    ledger = {name: _sha(payload) for name, payload in payloads.items()}
    ledger["release_manifest.json"] = _sha(manifest_text.encode())
    ledger["microcosm_uk_2025_dense.h5"] = _sha(h5_bytes)
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items()))
    )
    return release_dir


def test_valid_dense_bundle_passes(tmp_path: Path) -> None:
    validate_release_dir(write_dense_bundle(tmp_path))


def test_dense_bundle_accepts_the_release_id_as_revision(tmp_path: Path) -> None:
    validate_release_dir(write_dense_bundle(tmp_path, revision=DENSE_ID))


def _failures(release_dir: Path) -> str:
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    return str(excinfo.value)


def test_unpinned_revision_is_rejected(tmp_path: Path) -> None:
    assert "not pinned" in _failures(write_dense_bundle(tmp_path, revision="main"))


def test_wrong_namespace_is_rejected(tmp_path: Path) -> None:
    assert "namespace" in _failures(write_dense_bundle(tmp_path, namespace="uk_local"))


def test_default_slot_claim_is_rejected(tmp_path: Path) -> None:
    text = _failures(write_dense_bundle(tmp_path, default_datasets={"national": "x"}))
    assert "default_datasets" in text


def test_failing_release_blocking_gate_is_rejected(tmp_path: Path) -> None:
    text = _failures(
        write_dense_bundle(tmp_path, report=_signed_report(blocking_status="failed"))
    )
    assert "did not pass" in text and "shippable" in text


def test_dev_posture_report_is_rejected(tmp_path: Path) -> None:
    text = _failures(
        write_dense_bundle(tmp_path, report=_signed_report(release_candidate=False))
    )
    assert "release_candidate must be true" in text


def test_tampered_signature_is_rejected(tmp_path: Path) -> None:
    report = _signed_report()
    report["attestation"]["signature"] = "0" * 64
    assert "signature does not authenticate" in _failures(
        write_dense_bundle(tmp_path, report=report)
    )


def test_report_edited_after_signing_is_rejected(tmp_path: Path) -> None:
    report = _signed_report()
    report["gates"]["uk_local_target_fit"]["status"] = "passed"
    text = _failures(write_dense_bundle(tmp_path, report=report))
    assert "gate_outcomes_sha256" in text


def test_foreign_entry_ids_are_rejected(tmp_path: Path) -> None:
    report = _signed_report()
    report["gates"]["uk_terminal_something"] = report["gates"].pop(
        "uk_local_weight_ess"
    )
    assert "exactly the local battery entries" in _failures(
        write_dense_bundle(tmp_path, report=report)
    )


def test_missing_coverage_object_is_rejected(tmp_path: Path) -> None:
    assert "coverage object 'incumbent'" in _failures(
        write_dense_bundle(tmp_path, drop_coverage_key="incumbent")
    )


def test_diagnostics_without_households_are_rejected(tmp_path: Path) -> None:
    assert "households" in _failures(write_dense_bundle(tmp_path, drop_households=True))


def test_missing_key_env_names_the_hole(tmp_path: Path, monkeypatch) -> None:
    release_dir = write_dense_bundle(tmp_path)
    monkeypatch.delenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY")
    assert "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY" in _failures(release_dir)


def test_other_local_area_ids_keep_the_generic_contract(tmp_path: Path) -> None:
    # A non-dense local-area id never reaches the dense validator.
    release_dir = write_dense_bundle(
        tmp_path,
        release_id="populace-uk-2025-frs-k9",
        revision="populace-uk-2025-frs-k9",
    )
    text = _failures(release_dir)
    assert "us_source_coverage.json" in text
