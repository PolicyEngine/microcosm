import json

import pandas as pd
import pytest

from populace.build.us_runtime.snap_local_proxy import (
    snap_local_proxy_diagnostics,
    write_snap_local_proxy_diagnostics,
)


def test_snap_local_proxy_diagnostics_reports_validation_only_payload() -> None:
    household_frame = pd.DataFrame(
        {
            "district": ["CA-01", "CA-01", "CA-01", "NY-01"],
            "state": ["CA", "CA", "CA", "NY"],
            "weight": [10.0, 20.0, 30.0, 5.0],
            "snap_receipt": [1, 0, 1, 1],
            "snap_amount": [100.0, 0.0, 200.0, 50.0],
            "state_snap_relative_error": [0.12, 0.12, 0.12, 0.01],
        }
    )
    acs_reference = pd.DataFrame(
        {
            "district": ["CA-01", "NY-01"],
            "acs_snap_households": [30.0, 5.0],
            "acs_snap_households_moe": [5.0, 10.0],
        }
    )

    diagnostics = snap_local_proxy_diagnostics(
        household_frame,
        district_column="district",
        weight_column="weight",
        snap_receipt_column="snap_receipt",
        snap_amount_column="snap_amount",
        state_column="state",
        state_snap_relative_error_column="state_snap_relative_error",
        acs_reference=acs_reference,
        acs_snap_households_moe_column="acs_snap_households_moe",
        low_positive_sample_threshold=3,
        low_positive_ess_threshold=2.0,
        state_outlier_abs_relative_error=0.10,
        validation_source={
            "package_alias": "census-acs-s2201-congressional-district-snap-2024"
        },
    )

    assert diagnostics["classification"] == "validation_only"
    assert diagnostics["source_family"] == "snap_local_proxy"
    assert diagnostics["summary"]["districts"] == 2
    assert diagnostics["summary"]["districts_with_acs_reference"] == 2
    assert diagnostics["summary"]["outside_acs_moe_districts"] == 1
    assert diagnostics["summary"]["weighted_snap_households"] == 45.0
    assert diagnostics["summary"]["snap_dollars"] == 7_250.0
    assert diagnostics["validation_source"] == {
        "package_alias": "census-acs-s2201-congressional-district-snap-2024"
    }

    ca = diagnostics["districts"][0]
    assert ca["congressional_district"] == "CA-01"
    assert ca["weighted_households"] == 60.0
    assert ca["weighted_snap_households"] == 40.0
    assert ca["raw_positive_snap_households"] == 2
    assert ca["positive_snap_ess"] == pytest.approx(1.6)
    assert ca["positive_snap_max_to_mean_weight_ratio"] == pytest.approx(1.5)
    assert ca["snap_household_difference"] == 10.0
    assert ca["snap_household_relative_error"] == pytest.approx(1 / 3)
    assert ca["outside_acs_moe"] is True
    assert ca["flags"] == [
        "low_positive_ess",
        "low_positive_sample",
        "outside_acs_moe",
        "state_snap_outlier",
    ]


def test_snap_local_proxy_diagnostics_requires_positive_weight_district_rows() -> None:
    household_frame = pd.DataFrame(
        {"district": ["CA-01"], "weight": [0.0], "snap_receipt": [1]}
    )

    with pytest.raises(ValueError, match="positive-weight district rows"):
        snap_local_proxy_diagnostics(
            household_frame,
            district_column="district",
            weight_column="weight",
            snap_receipt_column="snap_receipt",
        )


def test_snap_local_proxy_diagnostics_rejects_duplicate_acs_rows() -> None:
    household_frame = pd.DataFrame(
        {"district": ["CA-01"], "weight": [1.0], "snap_receipt": [1]}
    )
    acs_reference = pd.DataFrame(
        {
            "district": ["CA-01", "CA-01"],
            "acs_snap_households": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate district rows"):
        snap_local_proxy_diagnostics(
            household_frame,
            district_column="district",
            weight_column="weight",
            snap_receipt_column="snap_receipt",
            acs_reference=acs_reference,
        )


def test_write_snap_local_proxy_diagnostics_writes_strict_json(tmp_path) -> None:
    payload = {"schema_version": 1, "classification": "validation_only"}
    path = write_snap_local_proxy_diagnostics(
        payload, tmp_path / "snap_local_proxy.json"
    )
    assert json.loads(path.read_text()) == payload
