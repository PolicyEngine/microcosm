import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime import l0_refit_export
from populace.build.us_runtime.l0_refit_export import (
    assert_required_us_release_source_columns,
    attach_l0_refit_entity_weights,
    attach_l0_refit_weights,
    export_us_l0_refit_h5,
    load_l0_refit_npz,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _us_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 20], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 201], dtype="int64"),
            "person_spm_unit_id": np.asarray([1000, 2000, 2000], dtype="int64"),
            "person_family_id": np.asarray([10000, 20000, 20000], dtype="int64"),
            "person_marital_unit_id": np.asarray(
                [100000, 200000, 200001], dtype="int64"
            ),
            "ssn_card_type": ["CITIZEN", "CITIZEN", "NONE"],
            "immigration_status_str": ["CITIZEN", "CITIZEN", "UNDOCUMENTED"],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": np.asarray([10, 20], dtype="int64")}
            ),
            "tax_unit": pd.DataFrame(
                {
                    "tax_unit_id": np.asarray([100, 200, 201], dtype="int64"),
                    "takes_up_aca_if_eligible": np.asarray(
                        [False, True, False], dtype=bool
                    ),
                    "selected_marketplace_plan_benchmark_ratio": np.asarray(
                        [1.0, 0.8, 1.2], dtype="float64"
                    ),
                }
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": np.asarray([1000, 2000], dtype="int64")}
            ),
            "family": pd.DataFrame(
                {"family_id": np.asarray([10000, 20000], dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": np.asarray([100000, 200000, 200001], dtype="int64")}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 200.0]), WeightKind.CALIBRATED)},
    )


def test_load_l0_refit_npz_validates_full_candidate_vector(tmp_path: Path) -> None:
    path = tmp_path / "weights.npz"
    metadata = {"candidate_records": 3, "n_selected": 2, "weight_entity": "household"}
    np.savez_compressed(
        path,
        weights=np.asarray([0.0, 4.0, 5.0]),
        metadata_json=json.dumps(metadata),
    )

    solution = load_l0_refit_npz(path, expected_candidate_records=3)

    assert solution.selected_mask.tolist() == [False, True, True]
    np.testing.assert_allclose(solution.selected_weights, np.asarray([4.0, 5.0]))
    assert solution.metadata == metadata


def test_load_l0_refit_npz_rejects_metadata_candidate_count_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weights.npz"
    np.savez_compressed(
        path,
        weights=np.asarray([0.0, 4.0, 5.0]),
        metadata_json=json.dumps({"candidate_records": 4, "n_selected": 2}),
    )

    with pytest.raises(ValueError, match="candidate_records"):
        load_l0_refit_npz(path, expected_candidate_records=3)


def test_attach_l0_refit_weights_subsets_clean_base_support() -> None:
    frame = _us_frame()
    solution = l0_refit_export.L0RefitWeights(
        weights=np.asarray([0.0, 333.0]),
        selected_mask=np.asarray([False, True]),
        metadata={},
    )

    exported = attach_l0_refit_weights(frame, solution)

    assert exported.table("household")["household_id"].to_list() == [20]
    assert exported.table("person")["person_id"].to_list() == [2, 3]
    assert exported.table("tax_unit")["tax_unit_id"].to_list() == [200, 201]
    assert exported.table("spm_unit")["spm_unit_id"].to_list() == [2000]
    np.testing.assert_allclose(
        exported.weights_for("household").values,
        np.asarray([333.0]),
    )
    assert exported.weights_for("household").kind == WeightKind.CALIBRATED


def test_attach_l0_refit_entity_weights_rejects_misaligned_weights() -> None:
    frame = _us_frame()

    with pytest.raises(ValueError, match="same shape"):
        attach_l0_refit_entity_weights(
            frame,
            weight_entity="household",
            selected_entity_ids=np.asarray([10, 20]),
            selected_weights=np.asarray([333.0]),
            reason="test",
        )


def test_required_us_release_source_columns_rejects_missing_source_stage() -> None:
    frame = _us_frame()
    raw_tax_units = frame.table("tax_unit").drop(
        columns=[
            "takes_up_aca_if_eligible",
            "selected_marketplace_plan_benchmark_ratio",
        ]
    )
    raw_frame = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.schema.entities},
            "tax_unit": raw_tax_units,
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(ValueError, match="takes_up_aca_if_eligible: missing"):
        assert_required_us_release_source_columns(raw_frame)


def test_export_us_l0_refit_h5_uses_existing_policyengine_writer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _us_frame()
    npz = tmp_path / "weights.npz"
    np.savez_compressed(
        npz,
        weights=np.asarray([0.0, 333.0]),
        metadata_json=json.dumps({"candidate_records": 2, "n_selected": 1}),
    )
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    attrs_h5 = tmp_path / "attrs.h5"
    attrs_h5.write_text("attrs")
    output = tmp_path / "populace_us_2024.h5"
    copied_from: list[Path] = []

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: frame)
    monkeypatch.setattr(
        l0_refit_export,
        "copy_populace_root_attrs",
        lambda source, destination: (
            copied_from.append(Path(source)) or ("populace_test_attr",)
        ),
    )

    class FakeEngine:
        def write_dataset(self, bundle, path, period):
            assert Path(path) == output
            assert period == 2024
            assert bundle.table("household")["household_id"].to_list() == [20]
            np.testing.assert_allclose(
                bundle.weights_for("household").values,
                np.asarray([333.0]),
            )
            Path(path).write_text("sentinel")

    monkeypatch.setattr(l0_refit_export, "PolicyEngineUSEngine", FakeEngine)

    summary = export_us_l0_refit_h5(
        base_h5=base_h5,
        weights_npz=npz,
        output_h5=output,
        root_attrs_h5=attrs_h5,
    )

    assert output.read_text() == "sentinel"
    assert copied_from == [attrs_h5]
    manifest_path = output.with_suffix(".l0_refit_export_summary.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "us_l0_refit_h5_export"
    assert manifest["summary_json_path"] == str(manifest_path)
    assert manifest["required_source_columns_checked"] is True
    assert manifest["base_h5"]["sha256"] == sha256(b"base").hexdigest()
    assert manifest["root_attrs_h5"]["sha256"] == sha256(b"attrs").hexdigest()
    assert manifest["weights_npz"]["sha256"] == sha256(npz.read_bytes()).hexdigest()
    assert manifest["output_h5"]["sha256"] == sha256(b"sentinel").hexdigest()
    assert summary["candidate_households"] == 2
    assert summary["selected_households"] == 1
    assert summary["selected_weight_sum"] == pytest.approx(333.0)
    assert summary["copied_root_attrs"] == ["populace_test_attr"]
    assert (
        summary["summary_json"]["sha256"]
        == sha256(manifest_path.read_bytes()).hexdigest()
    )
