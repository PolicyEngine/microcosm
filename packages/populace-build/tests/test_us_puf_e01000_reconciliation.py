from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from populace.build.us_runtime.puf_e01000_reconciliation import (
    PUF_E01000_RECONCILIATION_SCHEMA_VERSION,
    build_puf_e01000_reconciliation_basis,
    finalize_puf_e01000_reconciliation,
    puf_capital_gains_joint_metrics,
    puf_processed_capital_gains_stage,
    puf_raw_e01000_stage,
)

_METRIC_KEYS = {
    "record_count",
    "total_weight",
    "positive_carrier_count",
    "positive_carrier_weight",
    "weighted_signed_mass",
    "weighted_positive_mass",
}
_REAL_SOURCE_PUF = (
    Path.home()
    / "PolicyEngine"
    / ("policyengine" + "-us-data")
    / ("policyengine" + "_us_data")
    / "storage"
    / "puf_2015.csv"
)
_REAL_SOURCE_PUF_SHA256 = (
    "0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df"
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_raw_source(path: Path) -> None:
    pd.DataFrame(
        {
            "RECID": [1, 2, 999996, 999997, 999998, 999999],
            "S006": [100, 200, 300, 400, 500, 600],
            "E01000": [100.0, -10.0, 200.0, -100.0, 50.0, 0.0],
            # RECID 1 proves positivity is applied only after summing the
            # signed legs: +100 plus -150 is not a positive joint carrier.
            "P22250": [100.0, -20.0, 10.0, -20.0, 60.0, 0.0],
            "P23250": [-150.0, 50.0, 20.0, 10.0, -10.0, 0.0],
        }
    ).to_csv(path, index=False)


def _donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 1_000_000, 1_000_001],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "short_term_capital_gains": [100.0, -20.0, 40.0, 0.0],
            "long_term_capital_gains_before_response": [
                -150.0,
                50.0,
                -10.0,
                0.0,
            ],
        }
    )


def _screen_receipt(donor: pd.DataFrame) -> dict[str, object]:
    metric = puf_capital_gains_joint_metrics(
        donor,
        mask=[True, False, True, False],
    )
    return {
        "method": "field_local_zero",
        "screened_record_count": 2,
        "capital_gains_preserved": {
            "columns": [
                "short_term_capital_gains",
                "long_term_capital_gains_before_response",
            ],
            "before": metric,
            "after": dict(metric),
            "difference": {key: 0 for key in metric},
        },
    }


def _frame_metric(*, positive_mass: float) -> dict[str, float | int]:
    return {
        "record_count": 10,
        "total_weight": 100.0,
        "positive_carrier_count": 4,
        "positive_carrier_weight": 30.0,
        "weighted_signed_mass": positive_mass - 10.0,
        "weighted_positive_mass": positive_mass,
    }


def test_raw_receipt_uses_s006_over_100_and_row_joint_clip(tmp_path: Path) -> None:
    source = tmp_path / "puf_2015.csv"
    _write_raw_source(source)

    receipt = puf_raw_e01000_stage(source)

    assert set(receipt) == {"period", "regular", "aggregate", "all"}
    assert receipt["period"] == 2015
    for cohort in ("regular", "aggregate", "all"):
        assert set(receipt[cohort]) == {
            "e01000",
            "p22250_plus_p23250",
        }
        assert set(receipt[cohort]["e01000"]) == _METRIC_KEYS
        assert set(receipt[cohort]["p22250_plus_p23250"]) == _METRIC_KEYS
    assert receipt["regular"]["e01000"] == {
        "record_count": 2,
        "total_weight": 3.0,
        "positive_carrier_count": 1,
        "positive_carrier_weight": 1.0,
        "weighted_signed_mass": 80.0,
        "weighted_positive_mass": 100.0,
    }
    assert receipt["regular"]["p22250_plus_p23250"] == {
        "record_count": 2,
        "total_weight": 3.0,
        "positive_carrier_count": 1,
        "positive_carrier_weight": 2.0,
        "weighted_signed_mass": 10.0,
        "weighted_positive_mass": 60.0,
    }
    assert receipt["aggregate"]["e01000"]["weighted_positive_mass"] == 850.0
    assert receipt["aggregate"]["p22250_plus_p23250"]["weighted_positive_mass"] == 340.0
    assert receipt["all"]["e01000"]["weighted_positive_mass"] == 950.0
    assert receipt["all"]["p22250_plus_p23250"]["weighted_positive_mass"] == 400.0


def test_processed_receipt_observes_synthetic_replacement_without_mutation() -> None:
    donor = _donor()
    snapshot = donor.copy(deep=True)

    receipt = puf_processed_capital_gains_stage(donor)

    pd.testing.assert_frame_equal(donor, snapshot)
    assert receipt["source_aggregate_record_ids_absent"]
    assert receipt["synthetic_recid_start"] == 1_000_000
    assert receipt["synthetic_tail_support_eligible"]
    assert receipt["regular"]["weighted_signed_mass"] == 10.0
    assert receipt["regular"]["weighted_positive_mass"] == 60.0
    assert receipt["synthetic"]["weighted_signed_mass"] == 90.0
    assert receipt["synthetic"]["weighted_positive_mass"] == 90.0
    assert receipt["all"]["weighted_signed_mass"] == 100.0
    assert receipt["all"]["weighted_positive_mass"] == 150.0


def test_processed_receipt_rejects_an_e01000_carrier() -> None:
    donor = _donor()
    donor["E01000"] = 0.0

    with pytest.raises(ValueError, match="unexpectedly carries E01000"):
        puf_processed_capital_gains_stage(donor)


def test_reconciliation_schema_is_pinned_and_e01000_stays_audit_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "puf_2015.csv"
    _write_raw_source(source)
    donor = _donor()
    donor_snapshot = donor.copy(deep=True)
    replacement = puf_processed_capital_gains_stage(donor)

    basis = build_puf_e01000_reconciliation_basis(
        source,
        donor,
        processed_before_screen=replacement,
        mortgage_screen=_screen_receipt(donor),
        target_year=2024,
        source_sha256="source-sha",
    )
    receipt = finalize_puf_e01000_reconciliation(
        basis,
        {
            "tail_distribution_receipts": {
                "frame_before_stage": _frame_metric(positive_mass=200.0),
                "frame_after_stage": _frame_metric(positive_mass=700.0),
            }
        },
        frame_columns={
            "person": [
                "short_term_capital_gains",
                "long_term_capital_gains_before_response",
            ],
            "tax_unit": ["tax_unit_id"],
        },
    )

    pd.testing.assert_frame_equal(donor, donor_snapshot)
    assert set(receipt) == {
        "artifact_kind",
        "schema_version",
        "source",
        "target",
        "carrier",
        "concepts",
        "concept_divergence",
        "reconciliation_policy",
        "stages",
    }
    assert receipt["schema_version"] == PUF_E01000_RECONCILIATION_SCHEMA_VERSION
    assert set(receipt["stages"]) == {
        "raw_source",
        "synthetic_replacement",
        "mortgage_screen",
        "donor",
        "frame",
    }
    assert receipt["source"] == {
        "tax_year": 2015,
        "path": str(source.resolve()),
        "sha256": "source-sha",
        "weight_column": "S006",
        "weight_scale": 0.01,
        "aggregate_record_ids": [999996, 999997, 999998, 999999],
    }
    assert receipt["carrier"] == {
        "source_column": "E01000",
        "source_status_after_raw": "audit_only_not_carried",
        "donor_column": None,
        "frame_column": None,
        "materialized": False,
        "carrier_changed": False,
        "donor_column_absence_verified": True,
        "frame_column_absence_verified": True,
        "frame_entities_checked": ["person", "tax_unit"],
    }
    assert receipt["concept_divergence"] == {
        "basis": "weighted_positive_mass",
        "denominator": "e01000_weighted_positive_mass",
        "e01000_weighted_positive_mass": 950.0,
        "p22250_plus_p23250_weighted_positive_mass": 400.0,
        "difference": -550.0,
        "ratio": -550.0 / 950.0,
        "percent": 100.0 * -550.0 / 950.0,
        "interpretation": "expected_source_concept_divergence_not_stage_loss",
    }
    for stage in ("synthetic_replacement", "mortgage_screen", "donor", "frame"):
        assert receipt["stages"][stage]["e01000_status"] == "not_carried"
    screen = receipt["stages"]["mortgage_screen"]
    assert (
        screen["all_records"]["before"]
        == receipt["stages"]["synthetic_replacement"]["all"]
    )
    assert screen["all_records"]["after"] == receipt["stages"]["donor"]["all"]
    assert set(screen["all_records"]["difference"].values()) == {0}
    assert set(receipt["stages"]["frame"]["before_tail_transfer"]) == _METRIC_KEYS
    assert set(receipt["stages"]["frame"]["after_tail_transfer"]) == _METRIC_KEYS
    assert (
        receipt["stages"]["frame"]["after_tail_transfer"]["weighted_positive_mass"]
        == 700.0
    )
    with pytest.raises(ValueError, match="Final frame unexpectedly carries E01000"):
        finalize_puf_e01000_reconciliation(
            basis,
            {
                "tail_distribution_receipts": {
                    "frame_before_stage": _frame_metric(positive_mass=200.0),
                    "frame_after_stage": _frame_metric(positive_mass=700.0),
                }
            },
            frame_columns={"person": ["E01000"]},
        )


@pytest.mark.skipif(
    not _REAL_SOURCE_PUF.is_file(),
    reason=f"Pinned restricted PUF not present at {_REAL_SOURCE_PUF}",
)
def test_ty2015_raw_e01000_receipt_matches_pinned_artifact() -> None:
    assert _sha256(_REAL_SOURCE_PUF) == _REAL_SOURCE_PUF_SHA256

    receipt = puf_raw_e01000_stage(_REAL_SOURCE_PUF)

    regular = receipt["regular"]
    aggregate = receipt["aggregate"]
    all_records = receipt["all"]
    assert regular["e01000"]["weighted_positive_mass"] == pytest.approx(
        633_565_325_418.06,
        abs=0.01,
        rel=0,
    )
    assert aggregate["e01000"]["weighted_positive_mass"] == pytest.approx(
        86_694_319_431.0729,
        abs=0.01,
        rel=0,
    )
    assert all_records["e01000"]["weighted_positive_mass"] == pytest.approx(
        720_259_644_849.1329,
        abs=0.01,
        rel=0,
    )
    assert regular["p22250_plus_p23250"]["weighted_positive_mass"] == pytest.approx(
        674_979_536_370.26, abs=0.01, rel=0
    )
    assert aggregate["p22250_plus_p23250"]["weighted_positive_mass"] == pytest.approx(
        82_117_882_783.3678, abs=0.01, rel=0
    )
    joint_positive = all_records["p22250_plus_p23250"]["weighted_positive_mass"]
    e01000_positive = all_records["e01000"]["weighted_positive_mass"]
    assert joint_positive == pytest.approx(
        757_097_419_153.6278,
        abs=0.01,
        rel=0,
    )
    assert joint_positive - e01000_positive == pytest.approx(
        36_837_774_304.4949,
        abs=0.01,
        rel=0,
    )
    assert (joint_positive - e01000_positive) / e01000_positive == pytest.approx(
        0.0511451315757264,
        abs=1e-15,
        rel=0,
    )
    assert all_records["e01000"]["weighted_signed_mass"] == pytest.approx(
        701_378_975_535.2129,
        abs=0.01,
        rel=0,
    )
    assert all_records["p22250_plus_p23250"]["weighted_signed_mass"] == pytest.approx(
        676_142_092_687.9446, abs=0.01, rel=0
    )
