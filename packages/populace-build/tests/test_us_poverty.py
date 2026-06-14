import json

import pandas as pd
import pytest

from populace.build.us.poverty import (
    spm_resource_diagnostics,
    write_spm_resource_diagnostics,
)


def test_spm_resource_diagnostics_reports_validation_only_payload() -> None:
    frame = pd.DataFrame(
        {
            "resources": [9_000.0, 40_000.0, -500.0],
            "threshold": [15_000.0, 30_000.0, 10_000.0],
            "weight": [2.0, 3.0, 1.0],
            "state": ["CA", "CA", "NY"],
        }
    )

    diagnostics = spm_resource_diagnostics(
        frame,
        resource_column="resources",
        threshold_column="threshold",
        weight_column="weight",
        group_columns=("state",),
        validation_references={"census_cps_spm": {"classification": "validation_only"}},
    )

    assert diagnostics["classification"] == "validation_only"
    assert diagnostics["population"] == 6.0
    assert diagnostics["poverty_count"] == 3.0
    assert diagnostics["poverty_rate"] == pytest.approx(0.5)
    assert diagnostics["negative_resources"]["count"] == 1.0
    assert diagnostics["negative_resources"]["minimum"] == -500.0
    assert diagnostics["groups"]["state"]["CA"]["poverty_count"] == 2.0
    assert diagnostics["validation_references"]["census_cps_spm"] == {
        "classification": "validation_only"
    }


def test_spm_resource_diagnostics_requires_positive_finite_rows() -> None:
    frame = pd.DataFrame({"resources": [1.0], "threshold": [2.0], "weight": [0.0]})
    with pytest.raises(ValueError, match="positive-weight finite rows"):
        spm_resource_diagnostics(
            frame,
            resource_column="resources",
            threshold_column="threshold",
            weight_column="weight",
        )


def test_spm_resource_diagnostics_ignores_invalid_rows_in_counts() -> None:
    frame = pd.DataFrame(
        {
            "resources": [9_000.0, 40_000.0, 1_000.0, 2_000.0],
            "threshold": [15_000.0, 30_000.0, 10_000.0, float("nan")],
            "weight": [2.0, 3.0, float("nan"), 5.0],
        }
    )

    diagnostics = spm_resource_diagnostics(
        frame,
        resource_column="resources",
        threshold_column="threshold",
        weight_column="weight",
    )

    assert diagnostics["population"] == 5.0
    assert diagnostics["poverty_count"] == 2.0
    assert diagnostics["poverty_rate"] == pytest.approx(0.4)
    assert diagnostics["negative_resources"]["count"] == 0.0


def test_write_spm_resource_diagnostics_writes_strict_json(tmp_path) -> None:
    payload = {"schema_version": 1, "classification": "validation_only"}
    path = write_spm_resource_diagnostics(payload, tmp_path / "spm.json")
    assert json.loads(path.read_text()) == payload
