"""Engine-free SSI asset-gradient evidence-estimator behavior checks."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from test_support.paths import paths_for

_TOOL_PATH = (
    paths_for("microcosm-build").repository / "tools/estimate_ssi_asset_gradient.py"
)
_SPEC = importlib.util.spec_from_file_location("ssi_asset_estimator", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
estimator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(estimator)


def test_source_mappings_and_pin_match_manifest():
    path = (
        paths_for("microcosm-build").package
        / "src/microcosm/build/us/source_stages.json"
    )
    stage = next(
        row
        for row in json.loads(path.read_text())["stages"]
        if row["stage"] == "scf_wealth"
    )
    artifact = next(row for row in stage["artifacts"] if row["member"] == "pu2023.csv")
    read = next(
        row for row in stage["operations"] if row.get("table") == "sipp_2023_person"
    )
    assert estimator.DONOR_ARTIFACT == artifact
    assert estimator.ASSET_COLUMNS == tuple(read["targets"].values())
    assert set(estimator.ASSET_ALLOCATION_COLUMNS) == {
        column
        for columns in read["target_allocation_status_columns"].values()
        for column in columns
    }


def _raw() -> pd.DataFrame:
    rows = []
    for i in range(8):
        row = {column: 0.0 for column in estimator.SOURCE_COLUMNS}
        row.update(
            {
                "SSUID": f"household-{i}",
                "PNUM": 1,
                "MONTHCODE": 12,
                "WPFINWGT": 100.0,
                "TAGE": 70,
                "EMS": 6,
                "TPTOTINC": 100.0,
                "RSSI_MNYN": 2,
                "RSSI_YRYN": 2,
                "TVAL_BANK": 100.0,
                "TVAL_STMF": 20.0,
                "TVAL_BOND": 30.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_sample_uses_all_three_own_assets():
    sample, _ = estimator.prepare_sample(_raw())
    np.testing.assert_array_equal(sample["liquid_assets"], 150.0)


def test_receipt_does_not_create_disability_and_ssi_does_not_disqualify_income():
    raw = _raw()
    raw.loc[0, ["TPTOTINC", "TSSI_AMT", "RSSI_MNYN", "RSSI_YRYN"]] = [1000, 900, 1, 1]
    raw.loc[1, ["TAGE", "RSSI_MNYN", "RSSI_YRYN"]] = [40, 1, 1]
    raw.loc[2, ["TAGE", "EDISABL"]] = [40, 1]
    sample, _ = estimator.prepare_sample(raw)
    assert 0 in sample.index
    assert sample.loc[0, "countable_income"] == 80.0
    assert 1 not in sample.index
    assert 2 in sample.index


@pytest.mark.parametrize(
    "column,value",
    [
        ("TVAL_BANK", np.nan),
        ("AJSSAVVAL", 2),
        ("ASSI_MNYN", 2),
        ("EMS", 1),
        ("WPFINWGT", 0),
        ("TAGE", 14),
        ("TPTOTINC", 2000),
    ],
)
def test_sample_excludes_unmeasured_imputed_incomplete_and_ineligible_records(
    column, value
):
    raw = _raw()
    raw.loc[0, column] = value
    if column == "TAGE":
        raw.loc[0, "EDISABL"] = 1
    sample, _ = estimator.prepare_sample(raw)
    assert 0 not in sample.index
    assert len(sample) == 7


def test_duplicate_source_people_refuse():
    raw = _raw()
    raw.loc[1, ["SSUID", "PNUM"]] = raw.loc[0, ["SSUID", "PNUM"]].values
    with pytest.raises(ValueError, match="duplicate source-person"):
        estimator.prepare_sample(raw)


def _two_point_sample() -> pd.DataFrame:
    rows = []
    for assets, positives in ((0, 80), (1000, 40)):
        for i in range(100):
            receipt = int(i < positives)
            # Positive observations get twice the design weight.  Analytic
            # weighted odds are 8 at zero and 4/3 at $1,000.
            rows.append(
                {
                    "liquid_assets": assets,
                    "receipt": receipt,
                    "weight": 2.0 if receipt else 1.0,
                    "household": f"{assets}-{i}",
                }
            )
    return pd.DataFrame(rows)


def test_weighted_logit_matches_analytic_odds_and_is_order_and_weight_scale_stable():
    frame = _two_point_sample()
    result = estimator.fit_weighted_logit(frame)
    assert result["intercept"] == pytest.approx(np.log(8), abs=1e-7)
    assert result["slope"] == pytest.approx(
        np.log((4 / 3) / 8) / np.log1p(1000), abs=1e-7
    )
    shuffled = frame.sample(frac=1, random_state=42)
    shuffled["weight"] *= 100
    other = estimator.fit_weighted_logit(shuffled)
    assert other["slope"] == pytest.approx(result["slope"], abs=1e-7)
    assert other["slope_standard_error"] == pytest.approx(
        result["slope_standard_error"], abs=1e-9
    )


def test_standard_errors_cluster_households():
    frame = _two_point_sample()
    first = estimator.fit_weighted_logit(frame)
    # Cloning observations does not create independent sampled households.
    cloned = pd.concat((frame, frame), ignore_index=True)
    second = estimator.fit_weighted_logit(cloned)
    assert second["households"] == first["households"]
    assert second["slope_standard_error"] == pytest.approx(
        first["slope_standard_error"], rel=0.01
    )


def test_single_class_or_no_asset_variation_is_not_estimable():
    frame = _two_point_sample()
    assert (
        estimator.fit_weighted_logit(frame.loc[frame["receipt"].eq(1)])["status"]
        == "not_estimable"
    )
    frame["liquid_assets"] = 0
    assert estimator.fit_weighted_logit(frame)["status"] == "not_estimable"


@pytest.mark.parametrize("quasi", [False, True])
def test_complete_and_quasi_separation_do_not_report_finite_estimates(quasi):
    frame = pd.DataFrame(
        {
            "liquid_assets": [0, 1, 1 if quasi else 2, 3],
            "receipt": [0, 0, 1, 1],
            "weight": [1, 1, 1, 1],
            "household": ["a", "b", "c", "d"],
        }
    )
    result = estimator.fit_weighted_logit(frame)
    assert result["status"] == "not_estimable"
    assert "separation" in result["reason"]
    assert "slope" not in result


def test_primary_fit_excludes_resource_ineligible_range_and_reports_it_separately():
    two_point = _two_point_sample()
    raw = pd.concat([_raw().iloc[[0]]] * (len(two_point) + 1), ignore_index=True)
    raw["SSUID"] = [f"person-{i}" for i in range(len(raw))]
    raw["TVAL_STMF"] = 0
    raw["TVAL_BOND"] = 0
    raw.loc[: len(two_point) - 1, "TVAL_BANK"] = two_point["liquid_assets"].values
    raw.loc[: len(two_point) - 1, "RSSI_MNYN"] = np.where(two_point["receipt"], 1, 2)
    raw["RSSI_YRYN"] = raw["RSSI_MNYN"]
    raw.loc[len(two_point), "TVAL_BANK"] = 100000
    evidence = estimator.estimate(raw)
    primary = evidence["fits"]["65_plus"]["within_resource_limit"]
    full = evidence["fits"]["65_plus"]["full_observed_range_confounded_by_eligibility"]
    assert primary["rows"] == 200
    assert primary["maximum_assets"] == 1000
    assert full["rows"] == 201
    assert full["maximum_assets"] == 100000


def test_packaged_summary_transfers_adult_child_slope_without_reform_tuning():
    evidence = {
        "transform": "log1p(own assets)",
        "identification_limits": ["association"],
        "fits": {
            band: {
                "within_resource_limit": {
                    "status": "estimated",
                    "slope": slope,
                    "slope_standard_error": 0.02,
                    "rows": 100,
                }
            }
            for band, slope in (("under_18", -0.9), ("18_64", -0.1), ("65_plus", -0.2))
        },
    }
    packaged = estimator.packaged_summary(evidence)
    assert packaged["bands"]["under_18"]["slope"] == -0.1
    assert packaged["bands"]["under_18"]["source_band"] == "18_64"
    assert packaged["bands"]["65_plus"]["slope"] == -0.2
