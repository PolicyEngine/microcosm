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
    dataset_ids: np.ndarray = HOUSEHOLD_IDS,
    compiled: np.ndarray = COMPILED,
    design_weights: np.ndarray = DESIGN_WEIGHTS,
    recorded_final: float | None = None,
    target_value: float = TARGET_VALUE,
    drop_provenance: tuple[str, ...] = (),
) -> dict[str, Path]:
    pytest.importorskip("tables")  # pandas HDF backend
    tmp_path.mkdir(parents=True, exist_ok=True)
    if recorded_final is None:
        recorded_final = float((FINAL_WEIGHTS * compiled).sum())
    diagnostics = {
        "realized_max_weight_ratio": 5.0,
        "targets": [
            {
                "name": f"{TARGET_NAME}@2024",
                "entity": "household",
                "target": target_value,
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
            ("00000", "household_id", checkpoint_ids),
            ("00001", TARGET_NAME, compiled),
        ):
            group = checkpoint.create_group(f"tables/household/columns/{key}")
            group.attrs["name"] = name
            group.create_dataset("values", data=values)
        checkpoint.create_dataset("weights/household/values", data=design_weights)

    dataset_path = tmp_path / "populace_us_2024.h5"
    household = pd.DataFrame(
        {
            "household_id": dataset_ids,
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
    household = household.drop(columns=list(drop_provenance))
    household.to_hdf(dataset_path, key="household", format="table")

    return {
        "diagnostics": diagnostics_path,
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
    }


def _diagnose(
    module, paths: dict[str, Path], patterns: list[str], *, top: int = 3
) -> dict:
    return module.diagnose_target_support(
        diagnostics_path=paths["diagnostics"],
        checkpoint_path=paths["checkpoint"],
        dataset_path=paths["dataset"],
        target_patterns=patterns,
        top=top,
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
    assert report["carriers_with_nonpositive_design_weight"] == 0
    assert report["final_estimate"] == pytest.approx(FINAL_ESTIMATE)
    assert report["design_estimate"] == pytest.approx(11_000.0)
    assert report["final_relative_error"] == pytest.approx(
        FINAL_ESTIMATE / TARGET_VALUE - 1.0
    )
    assert report["design_relative_error"] == pytest.approx(-0.45)
    assert report["top_1_share"] == pytest.approx(50_000.0 / 51_000.0)
    assert "missing_provenance_columns" not in payload

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
    assert "top-3" in formatted
    assert "near-cap 50.0%" in formatted


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


def test__given_reversed_ids_beyond_float_precision__then_still_refuses(
    tmp_path,
) -> None:
    # 2**53 and its neighbors collapse to the same float64; the id comparison
    # must stay lossless (int64) so the order refusal still fires.
    module = _load_tool_module()
    big = np.array([2**53 + i for i in range(5)], dtype=np.int64)
    paths = _write_fixture(tmp_path, checkpoint_ids=big[::-1].copy(), dataset_ids=big)

    with pytest.raises(module.TargetSupportError, match="household_id order"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_lossy_float_ids__then_refuses_rather_than_compares(
    tmp_path,
) -> None:
    module = _load_tool_module()
    big = np.array([2**53 + i for i in range(5)], dtype=np.int64)
    paths = _write_fixture(
        tmp_path,
        checkpoint_ids=big.astype(np.float64),
        dataset_ids=big,
    )

    with pytest.raises(module.TargetSupportError, match="lossy"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_small_integral_float_ids__then_accepted(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, checkpoint_ids=HOUSEHOLD_IDS.astype(np.float64))

    payload = _diagnose(module, paths, [TARGET_NAME])
    assert payload["targets"][0]["carrier_count"] == 2


def test__given_foreign_checkpoint_final_mismatch__then_decomposition_refuses(
    tmp_path,
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, recorded_final=FINAL_ESTIMATE * 2.0)

    with pytest.raises(module.TargetSupportError, match="does not reproduce"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_dollar_scale_final_drift__then_decomposition_refuses(
    tmp_path,
) -> None:
    # A $10 drift on a $51k aggregate is far beyond float re-summation noise
    # (measured <= $0.41 absolute on the Build N release) and must refuse.
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, recorded_final=FINAL_ESTIMATE + 10.0)

    with pytest.raises(module.TargetSupportError, match="does not reproduce"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_wrong_length_design_weights__then_refuses(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, design_weights=np.array([10.0]))

    with pytest.raises(module.TargetSupportError, match="design weights"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_negative_dominant_carrier__then_ranked_by_magnitude(
    tmp_path,
) -> None:
    compiled = np.array([-1000.0, 0.0, 0.0, 0.0, 10.0])
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, compiled=compiled)

    payload = _diagnose(module, paths, [TARGET_NAME])
    (report,) = payload["targets"]

    # total = -10,000 + 500 = -9,500; the negative household dominates and
    # must lead the carrier list rather than the small positive row.
    assert report["carrier_count"] == 2
    assert report["top_carriers"][0]["household_id"] == 11
    assert report["top_carriers"][0]["weighted_contribution"] == pytest.approx(
        -10_000.0
    )
    assert report["top_1_share"] == pytest.approx((-10_000.0) / (-9_500.0))


def test__given_zero_target__then_relative_errors_report_na(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, target_value=0.0)

    payload = _diagnose(module, paths, [TARGET_NAME])
    (report,) = payload["targets"]
    assert report["final_relative_error"] is None

    formatted = module._format_report(payload)
    assert "(n/a)" in formatted
    assert "+0.0%" not in formatted


def test__given_nonpositive_top__then_refuses(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path)

    with pytest.raises(module.TargetSupportError, match="--top"):
        _diagnose(module, paths, [TARGET_NAME], top=0)


def test__given_json_output_aliasing_an_input__then_main_refuses(
    tmp_path, monkeypatch, capsys
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path)
    argv = [
        "diagnose_us_target_support.py",
        "--diagnostics",
        str(paths["diagnostics"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--dataset",
        str(paths["dataset"]),
        "--target",
        TARGET_NAME,
        "--json-output",
        str(paths["dataset"]),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(module.TargetSupportError, match="refusing to overwrite"):
        module.main()

    dataset_bytes = paths["dataset"].read_bytes()
    assert len(dataset_bytes) > 0  # input untouched


def test__given_hard_linked_json_output__then_main_refuses(
    tmp_path, monkeypatch
) -> None:
    import os

    module = _load_tool_module()
    paths = _write_fixture(tmp_path)
    alias = tmp_path / "alias.json"
    os.link(paths["dataset"], alias)
    original = paths["dataset"].read_bytes()
    argv = [
        "diagnose_us_target_support.py",
        "--diagnostics",
        str(paths["diagnostics"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--dataset",
        str(paths["dataset"]),
        "--target",
        TARGET_NAME,
        "--json-output",
        str(alias),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(module.TargetSupportError, match="refusing to overwrite"):
        module.main()

    assert paths["dataset"].read_bytes() == original


def test__given_uint64_ids_beyond_int64__then_refuses(tmp_path) -> None:
    module = _load_tool_module()
    big = np.array([2**63 + i for i in range(5)], dtype=np.uint64)
    paths = _write_fixture(tmp_path, checkpoint_ids=big)

    with pytest.raises(module.TargetSupportError, match="int64 range"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_float32_ids__then_refuses_unsupported_dtype(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, checkpoint_ids=HOUSEHOLD_IDS.astype(np.float32))

    with pytest.raises(module.TargetSupportError, match="unsupported dtype"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_nonfinite_target_value__then_refuses(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, target_value=float("inf"))

    with pytest.raises(module.TargetSupportError, match="not finite"):
        _diagnose(module, paths, [TARGET_NAME])


def test__given_design_aggregate_overflow__then_refuses(tmp_path) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, design_weights=np.full(5, 1e308))

    with pytest.raises(module.TargetSupportError, match="not finite"):
        _diagnose(module, paths, [TARGET_NAME])


def test__final_estimate_tolerance_bounds_are_the_documented_ones(
    tmp_path,
) -> None:
    module = _load_tool_module()
    # $0.75 absolute drift: inside the old abs_tol=1.0, outside the new $0.50.
    refused = _write_fixture(
        tmp_path / "abs_refuse", recorded_final=FINAL_ESTIMATE + 0.75
    )
    with pytest.raises(module.TargetSupportError, match="does not reproduce"):
        _diagnose(module, refused, [TARGET_NAME])

    # $0.30 absolute drift: inside the new bound (measured release noise is
    # <= $0.41), so it must be accepted.
    accepted = _write_fixture(
        tmp_path / "abs_accept", recorded_final=FINAL_ESTIMATE + 0.30
    )
    report = _diagnose(module, accepted, [TARGET_NAME])["targets"][0]
    assert report["carrier_count"] == 2

    # At large scale the relative bound governs: 5e-10 passes, 5e-9 refuses.
    scaled = COMPILED * 1e9
    scaled_final = float((FINAL_WEIGHTS * scaled).sum())
    rel_ok = _write_fixture(
        tmp_path / "rel_accept",
        compiled=scaled,
        recorded_final=scaled_final * (1 + 5e-10),
    )
    assert _diagnose(module, rel_ok, [TARGET_NAME])["targets"][0]["carrier_count"] == 2
    rel_bad = _write_fixture(
        tmp_path / "rel_refuse",
        compiled=scaled,
        recorded_final=scaled_final * (1 + 5e-9),
    )
    with pytest.raises(module.TargetSupportError, match="does not reproduce"):
        _diagnose(module, rel_bad, [TARGET_NAME])


def test__given_nonpositive_design_weight_carrier__then_counted_and_noted(
    tmp_path,
) -> None:
    module = _load_tool_module()
    weights = DESIGN_WEIGHTS.copy()
    weights[0] = 0.0  # the 100-valued carrier household
    paths = _write_fixture(tmp_path, design_weights=weights)

    payload = _diagnose(module, paths, [TARGET_NAME])
    (report,) = payload["targets"]
    assert report["carriers_with_nonpositive_design_weight"] == 1
    # Only the weight-5.0 carrier remains in the ratio statistics.
    assert report["carrier_weight_ratio"]["max"] == pytest.approx(5.0)
    assert "nonpositive design" in module._format_report(payload)


def test__given_missing_provenance_column__then_listed_and_noted(
    tmp_path,
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path, drop_provenance=("household_source_id",))

    payload = _diagnose(module, paths, [TARGET_NAME])
    assert payload["missing_provenance_columns"] == ["household_source_id"]
    formatted = module._format_report(payload)
    assert "lacks provenance column(s): household_source_id" in formatted
    assert "household_support_channel=puf_tax_detail" in formatted


def test__given_fresh_json_output__then_main_writes_payload(
    tmp_path, monkeypatch, capsys
) -> None:
    module = _load_tool_module()
    paths = _write_fixture(tmp_path)
    out = tmp_path / "payload.json"
    argv = [
        "diagnose_us_target_support.py",
        "--diagnostics",
        str(paths["diagnostics"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--dataset",
        str(paths["dataset"]),
        "--target",
        TARGET_NAME,
        "--json-output",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", argv)

    module.main()

    payload = json.loads(out.read_text())
    assert payload["targets"][0]["carrier_count"] == 2
    assert "hh 15" in capsys.readouterr().out
