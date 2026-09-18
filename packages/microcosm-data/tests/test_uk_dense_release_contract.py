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
    attempt_id: str | None = None,
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
            entry_id: {
                "passed": report["gates"].get(entry_id, {}).get("status") == "passed"
            }
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
        "measure_exclusions": {
            "obr.housing_benefit": {
                "reason": "synthetic gap",
                "tracking": "microcosm#869",
                "approved_by": "synthetic_reviewer",
                "adjudication": "synthetic decision",
                "approved_on": "2026-09-03",
                "expires_on": "2026-10-03",
            }
        },
        "signed_deferrals": {"binding_adjudications": {"x": {}}},
        "holdout": {"mean_holdout_loss": 0.2, "n_folds": 5},
        "uprating": {"applied": True, "factor": 1.03},
    }
    if drop_coverage_key:
        del coverage[drop_coverage_key]
    build_manifest = {
        "build_id": release_id,
        "build_sha": "abc1234",
        "code": {"git_commit": "a" * 40, "git_dirty": False},
        "attempt_id": attempt_id if attempt_id is not None else report["release_id"],
    }
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
    source_diagnostics = payloads["calibration_diagnostics.json"]
    candidate = {
        "identity": {
            "code": {"git_commit": "a" * 40, "git_dirty": False},
            "targets": {
                "chronicle": {
                    "facts_sha256": "b" * 64,
                    "manifest_sha256": "c" * 64,
                }
            },
        },
        "outputs": {
            "dataset": {"sha256": _sha(b"dense-h5-stand-in")},
            "calibration_diagnostics": {"sha256": _sha(source_diagnostics)},
            "local_gate_report": {"sha256": _sha(payloads["uk_local_gates.json"])},
        },
    }
    candidate["measure_exclusions"] = coverage.get("measure_exclusions", {})
    incumbent = {
        "inputs": {"incumbent_h5": {"sha256": "e" * 64}},
        "outputs": {"metrics": {"sha256": "6" * 64}, "weights": {"sha256": "7" * 64}},
    }
    payloads["rowwise_candidate_manifest.json"] = json.dumps(candidate).encode()
    payloads["incumbent_manifest.json"] = json.dumps(incumbent).encode()
    payloads["source_calibration_diagnostics.json"] = source_diagnostics
    evaluation = {
        "schema_version": 2,
        "kind": "uk_incumbent_surface_evaluation",
        "measure_resolution": {"blocks": 1},
        **_surface_rows(),
        "identity": {
            "candidate_dataset_sha256": _sha(b"dense-h5-stand-in"),
            "candidate_manifest_sha256": _sha(
                payloads["rowwise_candidate_manifest.json"]
            ),
            "candidate_diagnostics_sha256": _sha(source_diagnostics),
            "ledger_facts_sha256": "b" * 64,
            "ledger_manifest_sha256": "c" * 64,
            "incumbent_manifest_sha256": _sha(payloads["incumbent_manifest.json"]),
            "incumbent_metrics_sha256": "6" * 64,
            "incumbent_weights_sha256": "7" * 64,
        },
    }
    diagnostics["source_diagnostics_sha256"] = _sha(source_diagnostics)
    payloads["calibration_diagnostics.json"] = json.dumps(diagnostics).encode()
    score["artifacts"] = {
        "candidate_diagnostics": {"sha256": _sha(source_diagnostics)},
        "incumbent_household_metrics": {"sha256": "6" * 64},
        "incumbent_wide_weights": {"sha256": "7" * 64},
    }
    payloads["score_vs_incumbent.json"] = json.dumps(score).encode()
    evaluation["summary"] = dc.uk_incumbent_surface_assessment(evaluation)
    payloads["incumbent_surface_evaluation.json"] = json.dumps(evaluation).encode()
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


def test_dense_bundle_rejects_stale_ledger_identity_key(tmp_path: Path) -> None:
    release_dir = write_dense_bundle(tmp_path)
    candidate_path = release_dir / "rowwise_candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["identity"]["ledger"] = candidate["identity"].pop("targets")["chronicle"]
    candidate_path.write_text(json.dumps(candidate))
    assert "identity.targets.chronicle" in _failures(release_dir)


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
    report["gates"]["uk_local_target_fit"]["status"] = "failed"
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


def test_report_from_another_attempt_is_rejected(tmp_path: Path) -> None:
    release_dir = write_dense_bundle(
        tmp_path, attempt_id="uk-local-candidate-f100-s42-20260101T000000Z-00000000"
    )
    assert "does not belong to this run" in _failures(release_dir)


def test_build_manifest_without_attempt_id_is_rejected(tmp_path: Path) -> None:
    release_dir = write_dense_bundle(tmp_path)
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    del build_manifest["attempt_id"]
    (release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=1)
    )
    assert "must carry the calibration 'attempt_id'" in _failures(release_dir)


@pytest.mark.parametrize("dirty", [True, None, "false", 0, "missing"])
def test_dense_contract_requires_measured_clean_code(tmp_path, dirty):
    release_dir = write_dense_bundle(tmp_path)
    path = release_dir / "build_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["code"] = {} if dirty == "missing" else {"git_dirty": dirty}
    path.write_text(json.dumps(manifest))
    assert "code.git_dirty" in _failures(release_dir)


@pytest.mark.parametrize(
    "when,passes",
    [
        ("2026-09-02", False),
        ("2026-09-03", True),
        ("2026-10-03", True),
        ("2026-10-04", False),
    ],
)
def test_measure_exclusion_time_window_is_checked_at_validation(when, passes):
    from datetime import date

    record = {
        "name": "example",
        "reason": "measured gap",
        "tracking": "microcosm#869",
        "approved_by": "reviewer",
        "adjudication": "review decision",
        "approved_on": "2026-09-03",
        "expires_on": "2026-10-03",
    }
    failures = []
    dc._check_uk_measure_exclusions(
        {"example": record}, failures, today=date.fromisoformat(when)
    )
    assert (not failures) is passes


@pytest.mark.parametrize(
    "field,value",
    [
        ("expires_on", None),
        ("expires_on", ""),
        ("expires_on", "2026-99-03"),
        ("approved_on", None),
        ("approved_on", "20260903"),
        ("approved_by", ""),
    ],
)
def test_measure_exclusion_requires_full_approval_provenance(field, value):
    from datetime import date

    record = {
        "reason": "gap",
        "tracking": "microcosm#869",
        "approved_by": "reviewer",
        "adjudication": "decision",
        "approved_on": "2026-09-03",
        "expires_on": "2026-10-03",
    }
    record[field] = value
    failures = []
    dc._check_uk_measure_exclusions(
        {"example": record}, failures, today=date(2026, 9, 7)
    )
    assert any(field in failure for failure in failures)


def _surface_rows():
    return {
        "national_rows": [
            {
                "incumbent_name": "synthetic_national",
                "incumbent_target": 100.0,
                "family": "synthetic_program",
                "candidate_estimate": 100.0,
                "status": "measure_excluded",
            }
        ],
        "local_rows": [
            {
                "incumbent_name": "synthetic_local",
                "incumbent_target": 100.0,
                "area_type": "local_authority",
                "geography_id": "SYNTHETIC",
                "incumbent_metric": "synthetic_metric",
                "candidate_estimate": 100.0,
                "incumbent_estimate": 99.0,
                "status": "signed_deferred",
            }
        ],
    }


@pytest.fixture(autouse=True)
def _synthetic_surface_inventory(monkeypatch):
    from datetime import date

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 7)

    monkeypatch.setattr(dc, "date", FixedDate)
    monkeypatch.setattr(
        dc,
        "_UK_DENSE_SURFACE_INVENTORIES",
        {
            grain: {
                "rows": len(rows),
                "sha256": dc._canonical_sha256(
                    [dc._uk_incumbent_row_identity(r, grain) for r in rows]
                ),
            }
            for grain, rows in (
                ("national", _surface_rows()["national_rows"]),
                ("local", _surface_rows()["local_rows"]),
            )
        },
    )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing_file", "required file 'incumbent_surface_evaluation.json'"),
        ("missing_row", "complete nonempty row list"),
        ("nonfinite_candidate", "not valid JSON"),
        ("missing_comparator", "incumbent_estimate"),
        ("changed_identity", "identity.candidate_dataset_sha256"),
        ("changed_summary", "summary does not match"),
        ("deferred_miss", "absolute relative error"),
    ],
)
def test_dense_release_refuses_bad_incumbent_surface_evidence(
    tmp_path, mutation, expected
):
    release_dir = write_dense_bundle(tmp_path)
    path = release_dir / "incumbent_surface_evaluation.json"
    if mutation == "missing_file":
        path.unlink()
    else:
        payload = json.loads(path.read_text())
        if mutation == "missing_row":
            payload["national_rows"] = []
        elif mutation == "nonfinite_candidate":
            payload["national_rows"][0]["candidate_estimate"] = float("nan")
        elif mutation == "missing_comparator":
            payload["local_rows"][0]["incumbent_estimate"] = None
        elif mutation == "changed_identity":
            payload["identity"]["candidate_dataset_sha256"] = "9" * 64
        elif mutation == "changed_summary":
            payload["summary"]["grains"]["local"]["measured"] = 100
        elif mutation == "deferred_miss":
            payload["local_rows"][0]["candidate_estimate"] = 1.8
            payload["summary"] = dc.uk_incumbent_surface_assessment(payload)
        path.write_text(json.dumps(payload))
    assert expected in _failures(release_dir)


def test_incumbent_surface_limits_keep_exact_existing_boundaries():
    payload = _surface_rows()
    payload["national_rows"][0]["candidate_estimate"] = 125.0
    result = dc.uk_incumbent_surface_assessment(payload)
    # At 25% the row limit passes. The within-10 share remains diagnostic.
    assert not any(
        "absolute relative error" in failure for failure in result["failures"]
    )
    assert result["passed"] is True
    payload["national_rows"][0]["candidate_estimate"] = 125.0001
    assert any(
        "absolute relative error" in failure
        for failure in dc.uk_incumbent_surface_assessment(payload)["failures"]
    )
    payload["national_rows"][0]["candidate_estimate"] = 110.0
    assert dc.uk_incumbent_surface_assessment(payload)["passed"] is True


@pytest.mark.parametrize(
    "when,passes",
    [
        ("2026-09-02", False),
        ("2026-09-03", True),
        ("2026-10-03", True),
        ("2026-10-04", False),
    ],
)
def test_dense_directory_rechecks_exclusion_dates_on_each_validation(
    tmp_path, monkeypatch, when, passes
):
    from datetime import date

    release_dir = write_dense_bundle(tmp_path)

    class ValidationDate(date):
        @classmethod
        def today(cls):
            return date.fromisoformat(when)

    monkeypatch.setattr(dc, "date", ValidationDate)
    if passes:
        validate_release_dir(release_dir)
    else:
        assert "measure exclusion" in _failures(release_dir)


@pytest.mark.parametrize(
    "name,mutate,expected",
    [
        ("uk_source_coverage.json", "ledger", "Ledger facts_sha256"),
        ("uk_source_coverage.json", "incumbent", "incumbent snapshot"),
        ("calibration_diagnostics.json", "diagnostics", "changed original evaluated"),
        ("score_vs_incumbent.json", "score", "candidate_diagnostics does not match"),
    ],
)
def test_dense_release_joins_every_receipt_to_the_evaluated_sources(
    tmp_path, name, mutate, expected
):
    release_dir = write_dense_bundle(tmp_path)
    path = release_dir / name
    data = json.loads(path.read_text())
    if mutate == "ledger":
        data["ledger_artifact"]["facts_sha256"] = "9" * 64
    elif mutate == "incumbent":
        data["incumbent"]["snapshot"] = {"other": "snapshot"}
    elif mutate == "diagnostics":
        data["targets"][0]["final_estimate"] = 11.0
    else:
        data["artifacts"]["candidate_diagnostics"]["sha256"] = "9" * 64
    path.write_text(json.dumps(data))
    assert expected in _failures(release_dir)


def _refresh_packaging_checksums(release_dir):
    """Unsigned packaging edits cannot replace original approval provenance."""
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"].values():
        local = release_dir / artifact["path"]
        if local.is_file():
            artifact["sha256"] = _sha(local.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    sums_path = release_dir / "sha256sums.txt"
    sums = {}
    for line in sums_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        local = release_dir / name
        sums[name] = _sha(local.read_bytes()) if local.is_file() else digest
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )


@pytest.mark.parametrize("mutation", ["extend", "delete"])
def test_refreshing_unsigned_coverage_cannot_renew_original_approvals(
    tmp_path, monkeypatch, mutation
):
    from datetime import date

    release_dir = write_dense_bundle(tmp_path)

    class AfterExpiry(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 4)

    monkeypatch.setattr(dc, "date", AfterExpiry)
    path = release_dir / "uk_source_coverage.json"
    coverage = json.loads(path.read_text())
    if mutation == "extend":
        coverage["measure_exclusions"]["obr.housing_benefit"]["expires_on"] = (
            "2099-01-01"
        )
    else:
        coverage["measure_exclusions"] = {}
    path.write_text(json.dumps(coverage))
    _refresh_packaging_checksums(release_dir)
    failures = _failures(release_dir)
    assert "original candidate approvals" in failures
    assert "expired 2026-10-03" in failures
