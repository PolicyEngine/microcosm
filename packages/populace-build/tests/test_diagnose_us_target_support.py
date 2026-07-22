import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


def _load_tool_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "diagnose_us_target_support.py"
    spec = importlib.util.spec_from_file_location("diagnose_us_target_support", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TARGET_NAME = "irs_soi.synthetic.state.xx.amount"
HOUSEHOLD_IDS = np.array([11, 12, 13, 14, 15], dtype=np.int64)
COMPILED = np.array([100.0, 0.0, 0.0, 0.0, 1000.0])
DESIGN_WEIGHTS = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
FINAL_WEIGHTS = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
FINAL_ESTIMATE = float((FINAL_WEIGHTS * COMPILED).sum())  # 51,000
TARGET_VALUE = 20_000.0


def _write_fixture(
    tmp_path: Path,
    *,
    checkpoint_ids: np.ndarray = HOUSEHOLD_IDS,
    recorded_final: float = FINAL_ESTIMATE,
) -> dict[str, Path]:
    pytest.importorskip("tables")  # pandas HDF backend
    diagnostics = {
        "realized_max_weight_ratio": 5.0,
        "targets": [
            {
                "name": f"{TARGET_NAME}@2024",
                "entity": "household",
                "target": TARGET_VALUE,
                "final_estimate": recorded_final,
            },
            {
                "name": "person_entity_row@2024",
                "entity": "person",
                "target": 1.0,
                "final_estimate": 1.0,
            },
        ],
    }
    diagnostics_path = tmp_path / "calibration_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics))

    checkpoint_path = tmp_path / "target_frame_checkpoint.h5"
    with h5py.File(checkpoint_path, "w") as checkpoint:
        for key, name, values in (
            ("00000", "household_id", checkpoint_ids.astype(np.float64)),
            ("00001", TARGET_NAME, COMPILED),
        ):
            group = checkpoint.create_group(f"tables/household/columns/{key}")
            group.attrs["name"] = name
            group.create_dataset("values", data=values)
        checkpoint.create_dataset("weights/household/values", data=DESIGN_WEIGHTS)

    dataset_path = tmp_path / "populace_us_2024.h5"
    household = pd.DataFrame(
        {
            "household_id": HOUSEHOLD_IDS,
            "household_weight": FINAL_WEIGHTS,
            "household_support_channel": [
                "asec",
                "asec",
                "puf_tax_detail",
                "asec",
                "puf_tax_detail",
            ],
            "household_support_clone_index": [0, 0, 1, 0, 1],
            "household_source_id": [11, 12, 3, 14, 5],
            "state_fips": [5, 5, 5, 47, 47],
        }
    )
    household.to_hdf(dataset_path, key="household", format="table")

    return {
        "diagnostics": diagnostics_path,
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
    }


def _diagnose(module, paths: dict[str, Path], patterns: list[str]) -> dict:
    return module.diagnose_target_support(
        diagnostics_path=paths["diagnostics"],
        checkpoint_path=paths["checkpoint"],
        dataset_path=paths["dataset"],
        target_patterns=patterns,
        top=3,
    )


def test__given_aligned_release_triple__then_concentration_and_provenance_reported(
    tmp_path,
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path)

    payload = _diagnose(module, paths, [TARGET_NAME])
    (report,) = payload["targets"]

    assert report["name"] == f"{TARGET_NAME}@2024"
    assert report["carrier_count"] == 2
    assert report["final_estimate"] == pytest.approx(FINAL_ESTIMATE)
    assert report["design_estimate"] == pytest.approx(11_000.0)
    assert report["final_relative_error"] == pytest.approx(
        FINAL_ESTIMATE / TARGET_VALUE - 1.0
    )
    assert report["design_relative_error"] == pytest.approx(-0.45)
    assert report["top_1_share"] == pytest.approx(50_000.0 / 51_000.0)

    top = report["top_carriers"][0]
    assert top["household_id"] == 15
    assert top["weight_ratio"] == pytest.approx(5.0)
    assert top["household_support_channel"] == "puf_tax_detail"
    assert top["household_support_clone_index"] == 1
    assert top["state_fips"] == 47
    # Ratios among the two carriers are 1.0 and 5.0; the near-cap threshold
    # is 0.9 x realized_max_weight_ratio = 4.5, so half the carriers sit
    # near the cap.
    assert report["carrier_share_near_cap"] == pytest.approx(0.5)

    formatted = module._format_report(payload)
    assert "hh 15" in formatted
    assert "puf_tax_detail" in formatted


def test__given_period_suffixed_or_substring_pattern__then_same_target_matches(
    tmp_path,
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path)

    by_suffix = _diagnose(module, paths, [f"{TARGET_NAME}@2024"])
    by_substring = _diagnose(module, paths, ["state.xx"])

    assert [r["name"] for r in by_suffix["targets"]] == [f"{TARGET_NAME}@2024"]
    assert [r["name"] for r in by_substring["targets"]] == [f"{TARGET_NAME}@2024"]


def test__given_unknown_pattern__then_error_names_it(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path)

    with pytest.raises(module.TargetSupportError, match="no_such_target"):
        _diagnose(module, paths, ["no_such_target"])


def test__given_misaligned_household_ids__then_decomposition_refuses(
    tmp_path,
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, checkpoint_ids=HOUSEHOLD_IDS[::-1].copy())

    with pytest.raises(module.TargetSupportError, match="household_id order"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_foreign_checkpoint_final_mismatch__then_decomposition_refuses(
    tmp_path,
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, recorded_final=FINAL_ESTIMATE * 2.0)

    with pytest.raises(module.TargetSupportError, match="does not reproduce"):
        _diagnose(module, paths, [TARGET_NAME])
