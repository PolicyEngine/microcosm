from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.h5_io import (
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    load_simulation_ready_us_multispine_pool,
    write_nullable_us_h5,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool_frame() -> Frame:
    ids = np.asarray([10, 20, 30], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                US_SCHEMA.membership_column(entity): ids
                for entity in US_SCHEMA.group_entities
            },
            "nullable_input": np.asarray([True, None, False], dtype=object),
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 3.0, 5.0]),
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
                "provenance_counts": {"household": {"rows": 3}},
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


def test_ready_pool_loader_preserves_importance_weights_and_nullable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    expected_manifest_sha256 = _sha256(manifest_path)
    original_read_bytes = Path.read_bytes
    manifest_reads = 0

    def replace_after_pinned_read(path: Path) -> bytes:
        nonlocal manifest_reads
        raw = original_read_bytes(path)
        if path == manifest_path:
            manifest_reads += 1
            replacement = json.loads(raw)
            replacement["publication_run_id"] = "replacement-publication"
            path.write_text(json.dumps(replacement), encoding="utf-8")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_pinned_read)

    frame, manifest = load_simulation_ready_us_multispine_pool(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    weights = frame.weights_for("household")
    assert weights.kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(weights.values, [2.0, 3.0, 5.0])
    assert frame.table("person")["nullable_input"].tolist() == [True, None, False]
    assert frame.n("household") == 3
    assert manifest["publication_run_id"] == "fixture-publication"
    assert manifest_reads == 1
    assert json.loads(manifest_path.read_text())["publication_run_id"] == (
        "replacement-publication"
    )


def test_ready_pool_loader_reconciles_manifest_and_h5_household_counts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance_counts"]["household"]["rows"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="household row count 4.*H5 count 3"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_requires_explicitly_green_agreement_receipt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agreement_gate"]["passed"] = False
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["agreement_gate"]["passed"] = False
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no passing agreement-gate verdict"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_binds_diagnostics_agreement_verdict(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["agreement_gate"]["gates"]["us_spine_agreement"]["details"] = {
        "fixture": False
    }
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="verdict does not match"):
        load_simulation_ready_us_multispine_pool(manifest_path)
