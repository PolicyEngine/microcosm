"""Contracts for IRS PUF aggregate-record disaggregation."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.puf_aggregate_records import (
    AGGREGATE_RECIDS,
    SYNTHETIC_RECID_START,
    PufAggregateDisaggregationSpec,
    _assign_s006_values,
    _choose_n_synthetic,
    _get_amount_columns,
    audit_puf_aggregate_disaggregation,
    derive_puf_policyengine_variables,
    disaggregate_puf_aggregate_records,
    load_default_puf_aggregate_disaggregation_spec,
)
from microcosm.calibrate import relative_error_loss

_DERIVED_POLICYENGINE_COLUMNS = (
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)


def _make_regular_rows() -> list[dict]:
    rng = np.random.default_rng(123)
    rows: list[dict] = []
    next_recid = 1
    bucket_specs = [
        (999996, -18_000_000, -400_000, 14),
        (999997, 150_000, 9_500_000, 18),
        (999998, 12_000_000, 85_000_000, 16),
        (999999, 120_000_000, 480_000_000, 14),
    ]

    for bucket_recid, low, high, count in bucket_specs:
        for i in range(count):
            agi = rng.uniform(low, high)
            mars = rng.choice([1, 2, 3, 4], p=[0.18, 0.72, 0.05, 0.05])
            xtot_lower = 2 if mars == 2 else 0
            xtot = int(rng.integers(xtot_lower, 6))
            tail_boost = rng.uniform(2.0, 5.0) if i % 4 == 0 else 1.0
            abs_agi = abs(agi)
            p23250 = abs_agi * rng.uniform(0.05, 0.55) * tail_boost
            e26270 = abs_agi * rng.uniform(0.02, 0.18) * tail_boost
            e00900 = abs_agi * rng.uniform(0.0, 0.10) * tail_boost
            e02100 = abs_agi * rng.uniform(0.0, 0.03) * tail_boost
            p22250 = abs_agi * rng.uniform(0.0, 0.12) * tail_boost
            qualified_dividends = abs_agi * rng.uniform(0.01, 0.10)
            ordinary_dividends = qualified_dividends + abs_agi * rng.uniform(
                0.0,
                0.04,
            )
            if bucket_recid == 999996:
                p23250 *= -1 if i % 2 == 0 else 1
                e26270 *= -1 if i % 3 == 0 else 1
                e00900 *= -1 if i % 2 == 0 else 1
                e02100 *= -1 if i % 2 == 0 else 1
                p22250 *= -1
            rows.append(
                {
                    "RECID": next_recid,
                    "S006": int(rng.integers(300, 1800)),
                    "AGIR1": 19 if bucket_recid in (999998, 999999) else 11,
                    "F6251": int(rng.binomial(1, 0.2)),
                    "MARS": int(mars),
                    "XTOT": int(xtot),
                    "DSI": int(rng.binomial(1, 0.03)),
                    "EIC": int(rng.integers(0, 4)),
                    "E00100": float(agi),
                    "E00200": float(max(0.0, abs_agi * rng.uniform(0.02, 0.18))),
                    "P23250": float(p23250),
                    "P22250": float(p22250),
                    "E00650": float(qualified_dividends),
                    "E00300": float(abs_agi * rng.uniform(0.01, 0.08)),
                    "E26270": float(e26270),
                    "E00900": float(e00900),
                    "E02100": float(e02100),
                    "E00400": float(abs_agi * rng.uniform(0.0, 0.03)),
                    "E00600": float(ordinary_dividends),
                    "E18400": float(abs_agi * rng.uniform(0.0, 0.02)),
                    "E19800": float(abs_agi * rng.uniform(0.0, 0.04)),
                    "T27800": float(e02100),
                    "age": int(rng.integers(35, 85)),
                }
            )
            next_recid += 1
    return rows


def _make_aggregate_rows() -> list[dict]:
    rows = [
        (999996, 179, -5_000_000, 100_000, -3_000_000, -1_000_000, -1_500_000),
        (999997, 324, 5_000_000, 1_000_000, 2_000_000, 500_000, 800_000),
        (999998, 448, 30_000_000, 3_000_000, 15_000_000, 2_000_000, 5_000_000),
        (999999, 349, 300_000_000, 10_000_000, 200_000_000, 20_000_000, 40_000_000),
    ]
    return [
        {
            "RECID": recid,
            "S006": pop * 100,
            "AGIR1": 0,
            "F6251": 0,
            "MARS": 0,
            "XTOT": 1,
            "DSI": 0,
            "EIC": 0,
            "E00100": agi,
            "E00200": wages,
            "P23250": ltcg,
            "P22250": stcg,
            "E00650": abs(agi) * 0.10,
            "E00300": abs(agi) * 0.02,
            "E26270": passthrough,
            "E00900": abs(agi) * 0.02 * (-1 if agi < 0 else 1),
            "E02100": abs(agi) * 0.001 * (-1 if agi < 0 else 1),
            "E00400": abs(agi) * 0.01,
            "E00600": abs(agi) * 0.12,
            "E18400": abs(agi) * 0.01,
            "E19800": abs(agi) * 0.03,
            "T27800": abs(agi) * 0.001 * (-1 if agi < 0 else 1),
            "age": 0,
        }
        for recid, pop, agi, wages, ltcg, stcg, passthrough in rows
    ]


def _make_mini_puf() -> pd.DataFrame:
    return pd.DataFrame(_make_regular_rows() + _make_aggregate_rows())


def _synthetic_bucket(
    result: pd.DataFrame,
    original: pd.DataFrame,
    recid: int,
) -> pd.DataFrame:
    offset = 0
    for bucket_recid in AGGREGATE_RECIDS:
        pop_weight = (
            original.loc[original["RECID"] == bucket_recid, "S006"].iloc[0] / 100
        )
        n_synthetic = _choose_n_synthetic(pop_weight)
        start = SYNTHETIC_RECID_START + offset
        end = start + n_synthetic
        if bucket_recid == recid:
            return result[(result["RECID"] >= start) & (result["RECID"] < end)]
        offset += n_synthetic
    raise AssertionError(f"unknown aggregate recid {recid}")


def _weighted_total(df: pd.DataFrame, column: str) -> float:
    return float((df[column] * df["S006"] / 100).sum())


def _expected_columns_after_puf_derivation(columns: pd.Index) -> list[str]:
    expected = list(columns)
    for column in _DERIVED_POLICYENGINE_COLUMNS:
        if column not in expected:
            expected.append(column)
    return expected


@pytest.fixture
def mini_puf() -> pd.DataFrame:
    return _make_mini_puf()


@pytest.fixture
def result(mini_puf: pd.DataFrame) -> pd.DataFrame:
    return disaggregate_puf_aggregate_records(mini_puf, seed=42)


def test_default_spec_is_non_forbes_and_source_only() -> None:
    spec = load_default_puf_aggregate_disaggregation_spec()
    assert spec.enabled
    assert not spec.forbes_top_tail
    assert spec.synthetic_tail_support_eligible
    assert spec.aggregate_recids == AGGREGATE_RECIDS
    assert spec.buckets[999999].synthetic_agi_upper == 1_250_000_000
    assert "target" not in str(spec).lower()


def test_spec_rejects_unknown_keys() -> None:
    raw = {
        "enabled": True,
        "forbes_top_tail": False,
        "source": "test source",
        "aggregate_recids": [999999],
        "synthetic_recid_start": SYNTHETIC_RECID_START,
        "synthetic_tail_support_eligible": True,
        "screened_fields": ["E00100"],
        "buckets": {
            "999999": {
                "description": "Positive AGI $100M+",
                "agi_lower": 100_000_000,
                "agi_upper": None,
                "synthetic_agi_upper": 1_250_000_000,
            }
        },
        "hardcoded_targets": {"E00100": 1.0},
    }
    with pytest.raises(ValueError, match="unsupported keys"):
        PufAggregateDisaggregationSpec.from_dict(raw)

    del raw["hardcoded_targets"]
    raw["buckets"]["999999"]["forbes_template"] = "not allowed"
    with pytest.raises(ValueError, match="unsupported keys"):
        PufAggregateDisaggregationSpec.from_dict(raw)


def test_forbes_path_is_explicitly_refused() -> None:
    raw = load_default_puf_aggregate_disaggregation_spec()
    with pytest.raises(ValueError, match="Forbes top-tail"):
        PufAggregateDisaggregationSpec(
            enabled=raw.enabled,
            forbes_top_tail=True,
            source=raw.source,
            aggregate_recids=raw.aggregate_recids,
            synthetic_recid_start=raw.synthetic_recid_start,
            screened_fields=raw.screened_fields,
            synthetic_tail_support_eligible=raw.synthetic_tail_support_eligible,
            buckets=raw.buckets,
        ).validate()


def test_aggregate_rows_removed_and_regular_rows_preserved(
    mini_puf: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    assert not result["RECID"].isin(AGGREGATE_RECIDS).any()
    expected_regular = set(mini_puf[~mini_puf["RECID"].isin(AGGREGATE_RECIDS)]["RECID"])
    actual_regular = set(result[result["RECID"] < min(AGGREGATE_RECIDS)]["RECID"])
    assert actual_regular == expected_regular
    assert list(result.columns) == _expected_columns_after_puf_derivation(
        mini_puf.columns,
    )


def test_policyengine_dividend_variables_are_derived_from_raw_puf(
    mini_puf: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    derived = derive_puf_policyengine_variables(mini_puf)
    for column in _DERIVED_POLICYENGINE_COLUMNS:
        assert column in derived.columns
        assert column in result.columns

    assert np.allclose(result["qualified_dividend_income"], result["E00650"])
    assert np.allclose(
        result["non_qualified_dividend_income"],
        result["E00600"] - result["E00650"],
    )
    assert "ordinary_dividend_income" not in result
    assert "dividend_income" not in result
    assert (result["non_qualified_dividend_income"] >= -1e-9).all()


def test_invalid_puf_dividend_sources_raise_before_disaggregation(
    mini_puf: pd.DataFrame,
) -> None:
    puf = mini_puf.copy()
    puf.loc[puf.index[0], "E00650"] = puf.loc[puf.index[0], "E00600"] + 1.0

    with pytest.raises(ValueError, match="qualified dividends above ordinary"):
        disaggregate_puf_aggregate_records(puf, seed=42)


def test_synthetic_structural_fields_are_valid(result: pd.DataFrame) -> None:
    synthetic = result[result["RECID"] >= SYNTHETIC_RECID_START]
    assert 80 <= len(synthetic) <= 160
    assert synthetic["MARS"].isin([1, 2, 3, 4]).all()
    assert synthetic["XTOT"].isin([0, 1, 2, 3, 4, 5]).all()
    assert synthetic["DSI"].isin([0, 1]).all()
    assert synthetic["EIC"].isin([0, 1, 2, 3]).all()
    assert (synthetic.loc[synthetic["MARS"] == 2, "XTOT"] >= 2).all()


def test_weights_sum_by_aggregate_bucket(
    mini_puf: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    for recid in AGGREGATE_RECIDS:
        expected = mini_puf.loc[mini_puf["RECID"] == recid, "S006"].iloc[0] / 100
        bucket = _synthetic_bucket(result, mini_puf, recid)
        assert (bucket["S006"] / 100).sum() == pytest.approx(expected)


def test_fractional_hundredth_weights_are_preserved() -> None:
    weights = _assign_s006_values(total_s006=14_031, n=20, rng=np.random.default_rng(0))
    assert weights.sum() == 14_031
    assert (weights > 0).all()


def test_tiny_weight_assignment_never_creates_negative_weights() -> None:
    weights = _assign_s006_values(total_s006=10, n=10, rng=np.random.default_rng(0))
    assert weights.sum() == 10
    assert (weights > 0).all()
    with pytest.raises(ValueError, match="total_s006 must be at least n"):
        _assign_s006_values(total_s006=10, n=20, rng=np.random.default_rng(0))


def test_exact_weighted_totals_for_amount_columns(
    mini_puf: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    for recid in AGGREGATE_RECIDS:
        bucket = _synthetic_bucket(result, mini_puf, recid)
        target_row = mini_puf.loc[mini_puf["RECID"] == recid].iloc[0]
        pop_weight = target_row["S006"] / 100
        for column in _get_amount_columns(mini_puf.columns):
            assert _weighted_total(bucket, column) == pytest.approx(
                pop_weight * target_row[column],
                rel=1e-9,
                abs=1e-6,
            )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_aggregate_nonfinite_amount_targets_are_zeroed(
    mini_puf: pd.DataFrame,
    bad_value: float,
) -> None:
    puf = mini_puf.copy()
    puf["E00800"] = 123.0
    puf.loc[puf["RECID"].isin(AGGREGATE_RECIDS), "E00800"] = bad_value

    result = disaggregate_puf_aggregate_records(puf, seed=42)
    synthetic = result[result["RECID"] >= SYNTHETIC_RECID_START]

    assert np.isfinite(synthetic["E00800"]).all()
    assert (synthetic["E00800"] == 0).all()
    assert _weighted_total(synthetic, "E00800") == pytest.approx(0.0)


def test_bucket_agi_bounds(mini_puf: pd.DataFrame, result: pd.DataFrame) -> None:
    bounds = {
        999996: (None, 0),
        999997: (0, 10_000_000),
        999998: (10_000_000, 100_000_000),
        999999: (100_000_000, 1_250_000_000),
    }
    for recid, (lower, upper) in bounds.items():
        bucket = _synthetic_bucket(result, mini_puf, recid)
        if lower is not None:
            assert (bucket["E00100"] >= lower).all()
        if upper is not None:
            assert (bucket["E00100"] <= upper).all()


def test_infeasible_agi_bounds_raise_instead_of_silently_clipping(
    mini_puf: pd.DataFrame,
) -> None:
    raw = load_default_puf_aggregate_disaggregation_spec()
    constrained_top_bucket = replace(
        raw.buckets[999999],
        synthetic_agi_upper=101_000_000,
    )
    spec = replace(
        raw,
        buckets={**raw.buckets, 999999: constrained_top_bucket},
    )
    with pytest.raises(ValueError, match="could not preserve E00100"):
        disaggregate_puf_aggregate_records(mini_puf, seed=42, spec=spec)


def test_reproducible_by_seed(mini_puf: pd.DataFrame) -> None:
    first = disaggregate_puf_aggregate_records(mini_puf, seed=42)
    second = disaggregate_puf_aggregate_records(mini_puf, seed=42)
    pd.testing.assert_frame_equal(first, second)
    different = disaggregate_puf_aggregate_records(mini_puf, seed=99)
    assert not np.allclose(
        first[first["RECID"] >= SYNTHETIC_RECID_START]["E00100"],
        different[different["RECID"] >= SYNTHETIC_RECID_START]["E00100"],
    )


def test_audit_reports_source_reconstruction_recovery(
    mini_puf: pd.DataFrame,
) -> None:
    audit = audit_puf_aggregate_disaggregation(mini_puf, seed=42)

    assert audit["forbes_top_tail"] is False
    assert audit["raw_rows"] == len(mini_puf)
    assert audit["raw_aggregate_rows"] == len(AGGREGATE_RECIDS)
    assert audit["aggregate_rows_after"] == 0
    assert audit["synthetic_rows"] > len(AGGREGATE_RECIDS)
    assert audit["raw_aggregate_s006"] == pytest.approx(audit["synthetic_s006"])

    old_loss = audit["source_reconstruction_loss"]["old_drop_aggregate"]
    disaggregated_loss = audit["source_reconstruction_loss"]["disaggregated"]
    assert old_loss["loss"] > 0
    assert old_loss["loss_formula"] == (
        "mean(min(abs((estimate - target) / max(abs(target), 1)), 10))"
    )
    subset_audit = audit_puf_aggregate_disaggregation(
        mini_puf,
        seed=42,
        columns=["E00100", "P23250"],
    )
    subset_rows = subset_audit["source_reconstruction_loss"]["old_drop_aggregate"][
        "worst_columns"
    ]
    assert len(subset_rows) == 2
    assert subset_audit["source_reconstruction_loss"]["old_drop_aggregate"][
        "loss"
    ] == pytest.approx(
        relative_error_loss(
            np.asarray([row["estimate_total"] for row in subset_rows]),
            np.asarray([row["target_total"] for row in subset_rows]),
        )
    )
    assert old_loss["within_10pct"] < 1.0
    assert disaggregated_loss["loss"] < 1e-12
    assert disaggregated_loss["within_10pct"] == pytest.approx(1.0)

    agi = next(
        row
        for row in audit["field_totals"]
        if row["attribute"] == "adjusted_gross_income"
    )
    assert agi["source_column"] == "E00100"
    assert agi["aggregate_row_total"] != 0
    assert abs(agi["old_relative_error"]) > 0
    assert abs(agi["disaggregated_relative_error"]) < 1e-12

    capital_gains = next(
        row
        for row in audit["field_totals"]
        if row["attribute"] == "capital_gains_proxy"
    )
    assert set(capital_gains["source_columns"]) == {"P22250", "P23250"}

    assert len(audit["bucket_summaries"]) == len(AGGREGATE_RECIDS)
    assert (
        max(
            summary["max_abs_weighted_total_delta"]
            for summary in audit["bucket_summaries"]
        )
        < 2e-5
    )
    assert "forbes" not in json.dumps(audit["field_totals"]).lower()
    json.dumps(audit, allow_nan=False)


def test_audit_requires_raw_aggregate_rows(mini_puf: pd.DataFrame) -> None:
    transformed = disaggregate_puf_aggregate_records(mini_puf, seed=42)

    with pytest.raises(ValueError, match="requires the raw IRS aggregate RECIDs"):
        audit_puf_aggregate_disaggregation(transformed, seed=42)


def test_audit_rejects_missing_or_non_amount_columns(
    mini_puf: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="missing"):
        audit_puf_aggregate_disaggregation(
            mini_puf,
            seed=42,
            columns=["E00100", "NOT_A_COLUMN"],
        )

    with pytest.raises(ValueError, match="amount columns"):
        audit_puf_aggregate_disaggregation(
            mini_puf,
            seed=42,
            columns=["E00100", "age"],
        )
