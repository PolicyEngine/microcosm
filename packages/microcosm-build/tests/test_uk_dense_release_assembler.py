"""The dense release assembler stages a contract-valid bundle from a run."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path

import pytest

from microcosm.build.gate_battery import _canonical_json_bytes, _canonical_sha256
from microcosm.build.uk_runtime.calibration_run import UK_LOCAL_GATE_SCOPE
from microcosm.build.uk_runtime.release_identity import UK_DENSE_RELEASE_ID
from microcosm.data import contract as dc

REPO_ROOT = Path(__file__).resolve().parents[3]
KEY = base64.b64encode(b"\x09" * 32).decode()
KEY_BYTES = base64.b64decode(KEY)
ATTEMPT = "uk-local-candidate-f100-s42-20260903T105422Z-ca611e43"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed_report():
    gates = {}
    for entry_id in sorted(UK_LOCAL_GATE_SCOPE):
        blocking = entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        gates[entry_id] = {
            "gate": entry_id.removeprefix("uk_local_"),
            "phase": "terminal",
            "criticality": "release_blocking" if blocking else "diagnostic",
            "status": "passed" if blocking else "failed",
            "failures": [] if blocking else ["a diagnostic miss"],
            "details": {},
            "reason": None,
        }
    attestation = {
        "schema_version": dc._UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
        "producer": dc._UK_GATE_BATTERY_PRODUCER,
        "country": "uk",
        "release_id": ATTEMPT,
        "release_candidate": True,
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
        "release_id": ATTEMPT,
        "release_candidate": True,
        "spec_fingerprint": attestation["spec_fingerprint"],
        "gates_manifest_sha256": attestation["gates_manifest_sha256"],
        "phases": ["terminal"],
        "phases_evaluated": ["terminal"],
        "blocked_at_phase": None,
        "shippable": True,
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


def _candidate_dir(root: Path) -> tuple[Path, Path, Path]:
    candidate = root / "f100-k15-RC"
    (candidate / "logbook-spool").mkdir(parents=True)
    spine = root / "spine.h5"
    spine.write_bytes(b"spine-stand-in")
    h5 = candidate / "microcosm_uk_2025_local.h5"
    h5.write_bytes(b"dense-candidate-stand-in")
    diagnostics = {
        "schema_version": 6,
        "n_records": 30,
        "final_loss": 0.0156,
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
    (candidate / "calibration_diagnostics.json").write_text(json.dumps(diagnostics))
    report = _signed_report()
    (candidate / "microcosm_uk_2025_local.local_gates.json").write_text(
        json.dumps(report)
    )
    (candidate / "score_vs_incumbent.json").write_text(
        json.dumps(
            {
                "candidate_fitted_surface_loss": 0.0146,
                "incumbent_fitted_surface_loss": 0.181,
                "rows_compared": 2,
                "incumbent_missing_areas": {},
            }
        )
    )
    (candidate / "logbook-spool" / "row.json").write_text(
        json.dumps(
            {"build_id": ATTEMPT, "row_digest": "f" * 64, "prev_row_digest": "e" * 64}
        )
    )
    manifest = {
        "schema_version": 2,
        "created_at": "2026-09-03T10:54:22+00:00",
        "git_commit": "b" * 40,
        "parameters": {
            "release_candidate": True,
            "epochs": 1500,
            "n_clones": 15,
            "skip_holdout": False,
            "doctrine": {
                "max_weight_ratio": 10.0,
                "target_weight_rule": "grain_equal",
                "solve_epochs": 1500,
                "clone_count": 15,
                "target_loss_cap": 10.0,
                "scale_rule": "default_target_loss_scales",
            },
        },
        "releasable": True,
        "blocked_at_f100": False,
        "blocking_failures": [],
        "solve": {
            "measure_resolution": {"blocks": 1},
            "target_weight_rule_override": {},
            "n_households": 30,
            "final_loss": 0.0156,
            "n_targets_by_kind": {"local": 1, "ladder": 0, "national": 1},
            "binding_adjudications": {"stood_on": {"age_structure/constituency": {}}},
            "area_support_exclusions": {
                "entries_stood_on": ["local_authority/E06000053"]
            },
        },
        "fit": {
            "rotated_holdout": {
                "n_folds": 5,
                "mean_holdout_loss": 0.204,
                "worst_holdout_loss": 0.213,
                "folds": [{"fold": 0}],
            }
        },
        "weights": {
            "calibration_mass_change": {"declared_factor": 0.989},
            "realized_max_weight_ratio_vs_design": 10.0,
        },
        "cross_grain": {
            "unbound_bridges": [
                {
                    "bridge_id": "national_household_composition_partition_vs_census_households"
                }
            ],
            "empty_legs_licensed": [1, 2],
        },
        "ladder_household_uprating": {
            "applied": True,
            "factor": 1.0336,
            "reason": "x",
            "tenure_cells": {"applied": True, "cells": 3},
        },
        "measure_exclusions": {
            "obr.housing_benefit": {
                "reason": "unreachable",
                "tracking": "microcosm#736",
                "expires_on": "2099-01-01",
            }
        },
        "identity": {
            "spine": {
                "pin_verified": True,
                "sha256": _sha(spine),
                "path": str(spine),
                "bytes": spine.stat().st_size,
                "spine_provenance": {
                    "rules_engine": {"package": "policyengine-uk", "version": "2.92.1"},
                    "source_vintages": {"frs": "2024_25"},
                    "stages": ["frs_spine"],
                    "stochastic_contract_sha256": "9" * 64,
                },
            },
            "ladder": {
                "pin_verified": True,
                "sha256": "d" * 64,
                "bytes": 1,
                "layer_vintages": {"constituency": "2024_pcon"},
            },
            "ledger": {
                "path_name": "chronicle-uk-artifact-1cab809",
                "facts_sha256": "1" * 64,
                "manifest_sha256": "2" * 64,
                "fact_row_count": 3,
                "schema_version": "v1",
            },
            "code": {"git_commit": "b" * 40},
            "runtime": {
                "policyengine-core": "3.31.0",
                "policyengine-uk": "2.92.1",
                "microcosm-data": "0.1.0",
                "python": "3.13.14",
            },
        },
        "outputs": {
            "dataset": {"path": str(h5), "sha256": _sha(h5)},
            "calibration_diagnostics": {
                "path": str(candidate / "calibration_diagnostics.json"),
                "sha256": _sha(candidate / "calibration_diagnostics.json"),
            },
            "local_gate_report": {
                "path": str(candidate / "microcosm_uk_2025_local.local_gates.json"),
                "sha256": _sha(candidate / "microcosm_uk_2025_local.local_gates.json"),
            },
        },
    }
    (candidate / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    incumbent_manifest = root / "incumbent_local_surface_manifest.json"
    incumbent_manifest.write_text(
        json.dumps(
            {
                "period": 2025,
                "households": 52846,
                "inputs": {"incumbent_h5": {"sha256": "5" * 64}},
            }
        )
    )
    return candidate, spine, incumbent_manifest


def test_assembler_stages_a_contract_valid_dense_release(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    out = tmp_path / "releases"
    assert (
        assembler.main(
            [
                "--candidate-dir",
                str(candidate),
                "--spine-h5",
                str(spine),
                "--incumbent-manifest",
                str(incumbent),
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    release_dir = out / UK_DENSE_RELEASE_ID
    assert summary["release_id"] == UK_DENSE_RELEASE_ID
    assert summary["cut_tag"] == f"{UK_DENSE_RELEASE_ID}-20260903T105422Z-ca611e43"
    assert release_dir.is_dir()
    for name in dc._UK_DENSE_REQUIRED_RELEASE_FILES:
        assert (release_dir / name).is_file(), name
    published = candidate / "microcosm_uk_2025_dense.h5"
    assert (
        published.read_bytes()
        == (candidate / "microcosm_uk_2025_local.h5").read_bytes()
    )
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert manifest["dataset_role"] == "non_default_local_area"
    assert manifest["default_datasets"] == {} and manifest["is_default"] is False
    assert (
        manifest["artifacts"]["microcosm_uk_2025_dense"]["revision"]
        == summary["cut_tag"]
    )
    assert (
        manifest["artifacts"]["microcosm_uk_2025_dense"]["repo_id"]
        == "policyengine/populace-uk-private"
    )
    assert any(
        item["id"].startswith("measure_exclusion:obr.housing_benefit")
        for item in manifest["reviewed_limitations"]
    )
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    assert diagnostics["households"] == 30 and diagnostics["n_targets"] == 2
    gate_summary = json.loads((release_dir / "gate_summary.json").read_text())
    assert set(gate_summary["gates"]) == set(dc._UK_DENSE_RELEASE_BLOCKING_IDS)
    assert set(gate_summary["diagnostic_gates"]) == set(UK_LOCAL_GATE_SCOPE) - set(
        dc._UK_DENSE_RELEASE_BLOCKING_IDS
    )
    assert (
        "--no-latest" in summary["publish_command"]
        and "populace-uk-private" in summary["publish_command"]
    )
    # The contract re-validates the finished directory, and so does the pre-flight.
    dc.validate_release_dir(release_dir)
    preflight = _load("preflight_uk_local_release_candidate")
    assert preflight.main(["--release-dir", str(release_dir)]) == 0


def test_assembler_refuses_a_dev_posture_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["release_candidate"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="pre-flight failed"):
        assembler.main(
            [
                "--candidate-dir",
                str(candidate),
                "--spine-h5",
                str(spine),
                "--incumbent-manifest",
                str(incumbent),
                "--out-dir",
                str(tmp_path / "releases"),
            ]
        )


def test_assembler_refuses_tampered_candidate_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    (candidate / "microcosm_uk_2025_local.h5").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="candidate bytes"):
        assembler.main(
            [
                "--candidate-dir",
                str(candidate),
                "--spine-h5",
                str(spine),
                "--incumbent-manifest",
                str(incumbent),
                "--out-dir",
                str(tmp_path / "releases"),
            ]
        )
