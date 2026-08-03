from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.exact_k_ladder import calibrate_exact_k_ladder
from populace.build.us_runtime.h5_io import (
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    load_simulation_ready_us_multispine_pool,
    write_nullable_us_h5,
)
from populace.calibrate import TargetRegistry, TargetSpec
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_exact_k_e2e", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_frame() -> Frame:
    ids = np.arange(1, 9, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                US_SCHEMA.membership_column(entity): ids
                for entity in US_SCHEMA.group_entities
            },
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    tables["household"]["fixture_measure"] = np.asarray(
        [0.0, 1.0, 2.0, 3.0, 6.0, 9.0, 12.0, 20.0]
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.arange(1.0, 9.0),
                WeightKind.IMPORTANCE,
            )
        },
    )


def _write_ready_pool(tmp_path: Path) -> Path:
    run_id = "fixture-publication"
    pool_path = tmp_path / "pool.h5"
    diagnostics_path = tmp_path / "pool.agreement.json"
    manifest_path = tmp_path / "pool.manifest.json"
    agreement_gate = {
        "passed": True,
        "gates": {
            "us_spine_agreement": {
                "passed": True,
                "failures": [],
                "details": {"fixture": True},
            }
        },
    }
    write_nullable_us_h5(
        _pool_frame(),
        pool_path,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        publication_run_id=run_id,
    )
    diagnostics_path.write_text(
        json.dumps(
            {
                "artifact_kind": (US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND),
                "schema_version": US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
                "simulation_ready": True,
                "publication_run_id": run_id,
                "agreement_gate": agreement_gate,
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
                "schema_version": US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
                "status": "simulation_ready",
                "simulation_ready": True,
                "publication_run_id": run_id,
                "period": 2024,
                "stage_checkpoints": {
                    "agreement": {
                        "source": "always_fresh",
                        "cached": False,
                        "terminal_verdict_persisted": False,
                    }
                },
                "agreement_gate": agreement_gate,
                "provenance_counts": {"household": {"rows": 8}},
                "pool_h5": {
                    "path": str(pool_path.resolve()),
                    "sha256": _sha256(pool_path),
                    "artifact_kind": US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
                    "publication_run_id": run_id,
                },
                "agreement_diagnostics": {
                    "path": str(diagnostics_path.resolve()),
                    "sha256": _sha256(diagnostics_path),
                    "publication_run_id": run_id,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.parametrize(
    ("ladder_point", "k", "expected_design"),
    (
        ("N", 8, "full-pool"),
        ("57,240 fixture analogue", 6, "sampford"),
        ("20,000 fixture analogue", 4, "sampford"),
    ),
)
def test_ready_pool_to_refit_and_release_manifests_for_each_ladder_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ladder_point: str,
    k: int,
    expected_design: str,
) -> None:
    pytest.importorskip("tables")
    builder = _builder_module()
    pool_manifest_path = _write_ready_pool(tmp_path)
    pool, pool_manifest = load_simulation_ready_us_multispine_pool(pool_manifest_path)
    target = TargetSpec(
        name="fixture_measure",
        entity="household",
        value=float(
            pool.weights_for("household").values
            @ pool.table("household")["fixture_measure"].to_numpy()
        )
        * 0.9,
        measure="fixture_measure",
        period=2024,
        source="fixture frozen register",
        family="fixture",
    )
    registry = TargetRegistry((target,), country="us")
    outcome = calibrate_exact_k_ladder(
        pool,
        registry.to_target_set(),
        k=k,
        pi_hi=1.0,
        seed=17,
        epochs=3,
        refit_epochs=3,
        learning_rate=0.02,
        max_weight_ratio=20.0,
        l0_lambda=1e-8,
        target_loss_weights=np.ones(1),
    )

    diagnostic = outcome.result.diagnostics[0]
    incumbent_rows = {
        diagnostic.name: {
            "target": diagnostic.target,
            "final_estimate": diagnostic.target * 11.0,
        }
    }
    loss_basis = builder._fiscal_target_loss_basis(registry, np.ones(1))
    incumbent_gate = builder._exact_k_frozen_register_fit_gate(
        outcome.result,
        incumbent_rows,
        target_registry=registry,
        target_loss_weights=np.ones(1),
        configured_loss_basis=loss_basis,
        incumbent_loss_basis=loss_basis,
    )
    assert incumbent_gate.passed
    target_surface = builder.diagnostics_payload(
        outcome.result,
        target_registry=registry,
    )["target_surface"]
    args = argparse.Namespace(
        exact_k=k,
        seed=17,
        pool_release_id="fixture-pool-release",
        pool_manifest_sha256=_sha256(pool_manifest_path),
        incumbent_diagnostics_sha256="f" * 64,
    )
    ladder_receipt = builder._exact_k_ladder_manifest_payload(
        args=args,
        outcome=outcome,
        pool_manifest=pool_manifest,
        ledger_artifact={
            "facts_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
        target_surface=target_surface,
        target_loss_basis=loss_basis,
        incumbent_diagnostics_sha256="f" * 64,
        incumbent_fit_gate=incumbent_gate,
        puf_tail_gate=builder.GateResult(
            name="exact_k_puf_capital_gains_tail",
            passed=True,
            details={"status": "fixture_support_retained"},
        ),
    )

    release_id = f"populace-us-2024-k{k}-fixture"
    release_dir = tmp_path / "release" / release_id
    artifact_root = tmp_path / "artifacts"
    release_dir.mkdir(parents=True)
    artifact_root.mkdir()
    (artifact_root / builder.DATASET_FILENAME).write_bytes(b"fixture h5")
    (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(b"fixture npz")
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    (release_dir / "us_source_coverage.json").write_text("{}")
    (release_dir / "us_ssi_take_up.json").write_text("{}")
    monkeypatch.setattr(
        builder,
        "_runtime_versions",
        lambda: {
            "python": "3.14.0",
            "populace-data": "0.1.0",
            "policyengine-core": "3.26.11",
            "policyengine-us": "1.752.2",
        },
    )
    monkeypatch.setattr(builder, "_git_output", lambda *args: "a" * 40)
    monkeypatch.setattr(builder, "_release_gate_failures", lambda *args, **kwargs: [])
    builder._build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=outcome.result,
        registry=registry,
        dropped={"dropped_target_names": []},
        target_profile_gate=builder.GateResult(
            name="target_profile_coverage",
            passed=True,
            details={"requirements_checked": 1},
        ),
        ledger_artifact={
            "facts_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
        },
        default_dataset={
            "method": "full_pool_refit" if k == 8 else "exact_k_sampford_refit",
            "n_candidate_households": 8,
            "n_selected_households": k,
        },
        exact_k_ladder=ladder_receipt,
    )

    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert ladder_point
    assert outcome.result.frame.n("household") == k
    assert outcome.result.frame.weights_for("household").kind is WeightKind.CALIBRATED
    assert ladder_receipt["k"] == k
    assert ladder_receipt["seed"] == 17
    assert ladder_receipt["selection_receipt"] == outcome.selection_receipt
    assert ladder_receipt["selection_receipt"]["design"] == expected_design
    assert ladder_receipt["pool"] == {
        "release_id": "fixture-pool-release",
        "release_id_source": "release_config",
        "manifest_sha256": _sha256(pool_manifest_path),
        "publication_run_id": "fixture-publication",
        "pool_h5_sha256": pool_manifest["pool_h5"]["sha256"],
        "agreement_diagnostics_sha256": pool_manifest["agreement_diagnostics"][
            "sha256"
        ],
    }
    assert ladder_receipt["agreement_gate_reference"] == {
        "passed": True,
        "publication_run_id": "fixture-publication",
        "diagnostics_sha256": pool_manifest["agreement_diagnostics"]["sha256"],
        "verdict": pool_manifest["agreement_gate"],
    }
    assert ladder_receipt["frozen_target_register"]["ledger_artifact"] == {
        "facts_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    assert (
        ladder_receipt["frozen_target_register"]["target_surface_sha256"]
        == target_surface["sha256"]
    )
    assert ladder_receipt["frozen_target_register"]["target_loss_basis"] == loss_basis
    assert (
        ladder_receipt["frozen_target_register"]["incumbent_diagnostics_sha256"]
        == "f" * 64
    )
    assert ladder_receipt["frozen_target_register"]["incumbent_fit"]["passed"]
    assert ladder_receipt["invariant_battery"] == {
        "puf_capital_gains_tail": {
            "passed": True,
            "failures": [],
            "details": {"status": "fixture_support_retained"},
        }
    }
    assert ladder_receipt["refit_baseline_diagnostics"]["method"] == (
        "full_pool_original_frame_weights"
        if k == 8
        else "normalized_horvitz_thompson_w_over_q"
    )
    assert (
        ladder_receipt["refit_baseline_diagnostics"]["source_weight_kind"]
        == "importance"
    )
    assert build_manifest["exact_k_ladder"] == ladder_receipt
    assert release_manifest["build"]["exact_k_ladder"] == ladder_receipt
    assert release_manifest["default_datasets"] == {"national": "populace_us_2024"}
    assert (
        release_manifest["artifacts"]["populace_us_2024"]["path"]
        == builder.DATASET_FILENAME
    )
    if k == 8:
        np.testing.assert_array_equal(outcome.support, np.arange(8))
        assert not np.array_equal(
            outcome.result.weights,
            pool.weights_for("household").values,
        )
