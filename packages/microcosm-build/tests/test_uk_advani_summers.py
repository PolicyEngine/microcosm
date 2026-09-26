"""The shared Advani-Summers surface: loader, inverse and stratified mapping."""

from __future__ import annotations

import hashlib
from importlib.resources import files

import numpy as np
import pytest

from microcosm.build.uk_runtime.advani_summers import (
    ADVANI_SUMMERS_RESOURCE,
    CGT_PRIOR_PERCENTILE_COLUMNS,
    CGT_QUANTILE_POINTS,
    advani_summers_band_index,
    advani_summers_knots,
    advani_summers_quantile_function,
    advani_summers_resource_sha256,
    advani_summers_rows,
    draw_banded_priors,
    exempt_range_quantiles,
    load_advani_summers_distribution,
    quantile_for_amount,
    stratified_amounts_within_range,
)

# One synthetic band whose spline crosses zero at q = 0.1 and 50 at q = 0.5.
SYNTHETIC_KNOTS = np.asarray([-10.0, 0.0, 25.0, 50.0, 75.0, 100.0, 125.0])


def _synthetic_distribution() -> dict:
    return {
        "rows": [
            {
                "percentile": "<40",
                "minimum_total_income": 0,
                "percent_with_gains": 0.01,
                **dict(zip(CGT_PRIOR_PERCENTILE_COLUMNS, SYNTHETIC_KNOTS, strict=True)),
            },
            {
                "percentile": 40,
                "minimum_total_income": 10_000,
                "percent_with_gains": 0.02,
                **dict(
                    zip(CGT_PRIOR_PERCENTILE_COLUMNS, SYNTHETIC_KNOTS * 2, strict=True)
                ),
            },
        ]
    }


def test_committed_rows_validate_and_place_zero_on_the_p10_p25_segment() -> None:
    rows = advani_summers_rows(load_advani_summers_distribution())

    assert len(rows) == 61
    for row in rows:
        assert row["p10"] < 0 < row["p25"], row["minimum_total_income"]


def test_quantile_for_amount_inverts_the_prior_spline_on_every_row() -> None:
    rows = advani_summers_rows(load_advani_summers_distribution())
    for row in rows:
        knots = advani_summers_knots(row)
        spline = advani_summers_quantile_function(knots)
        for amount in (
            knots[0] - 5_000.0,
            0.0,
            3_000.0,
            6_000.0,
            12_300.0,
            knots[-1] + 50_000.0,
        ):
            quantile = quantile_for_amount(knots, amount)
            assert float(spline(quantile)) == pytest.approx(amount, abs=1e-6)


def test_quantile_for_amount_refuses_a_flat_segment_and_misaligned_knots() -> None:
    with pytest.raises(ValueError, match="flat"):
        quantile_for_amount(np.asarray([0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 40.0]), 0.0)
    with pytest.raises(ValueError, match="align"):
        quantile_for_amount(np.asarray([0.0, 1.0, 2.0]), 1.0)


def test_exempt_range_quantiles_are_ordered_and_inside_the_body() -> None:
    rows = advani_summers_rows(load_advani_summers_distribution())
    for exempt_amount, upper_bounds in (
        (3_000.0, (0.22, 0.31)),
        (6_000.0, (0.30, 0.36)),
    ):
        for row in rows:
            lower, upper = exempt_range_quantiles(
                advani_summers_knots(row), exempt_amount
            )
            assert 0.05 < lower < upper < 0.95
            assert 0.16 <= lower <= 0.25
            assert upper_bounds[0] <= upper <= upper_bounds[1]
    with pytest.raises(ValueError, match="positive"):
        exempt_range_quantiles(SYNTHETIC_KNOTS, 0.0)


def test_synthetic_crossings_are_exact() -> None:
    assert exempt_range_quantiles(SYNTHETIC_KNOTS, 50.0) == pytest.approx((0.1, 0.5))


def test_stratified_amounts_take_stratum_midpoints_in_order() -> None:
    equal = stratified_amounts_within_range(
        SYNTHETIC_KNOTS, weights=np.ones(4), lower_quantile=0.1, upper_quantile=0.5
    )
    # Strata midpoints at q = 0.15, 0.25, 0.35, 0.45 on the two segments.
    assert equal == pytest.approx([25.0 / 3.0, 25.0, 35.0, 45.0])
    unequal = stratified_amounts_within_range(
        SYNTHETIC_KNOTS,
        weights=np.asarray([1.0, 3.0]),
        lower_quantile=0.1,
        upper_quantile=0.5,
    )
    # Strata (0.1, 0.2) and (0.2, 0.5): midpoints q = 0.15 and 0.35.
    assert unequal == pytest.approx([25.0 / 3.0, 35.0])
    assert (
        stratified_amounts_within_range(
            SYNTHETIC_KNOTS, weights=np.zeros(0), lower_quantile=0.1, upper_quantile=0.5
        ).size
        == 0
    )


def test_stratified_amounts_keep_zero_weight_persons_strictly_inside() -> None:
    amounts = stratified_amounts_within_range(
        SYNTHETIC_KNOTS,
        weights=np.asarray([0.0, 0.0, 4.0]),
        lower_quantile=0.1,
        upper_quantile=0.5,
    )
    assert (amounts > 0.0).all() and (amounts < 50.0).all()
    assert (np.diff(amounts) >= 0.0).all()
    with pytest.raises(ValueError, match="finite"):
        stratified_amounts_within_range(
            SYNTHETIC_KNOTS,
            weights=np.asarray([1.0, -1.0]),
            lower_quantile=0.1,
            upper_quantile=0.5,
        )
    with pytest.raises(ValueError, match="width"):
        stratified_amounts_within_range(
            SYNTHETIC_KNOTS, weights=np.ones(2), lower_quantile=0.5, upper_quantile=0.5
        )


def test_band_index_matches_the_prior_draw_banding() -> None:
    rows = advani_summers_rows(_synthetic_distribution())
    index = advani_summers_band_index(
        rows, np.asarray([-1.0, 0.0, 9_999.0, 10_000.0, 1e9])
    )
    assert index.tolist() == [0, 0, 0, 1, 1]


def test_draw_banded_priors_evaluates_each_band_spline() -> None:
    distribution = _synthetic_distribution()
    draws = draw_banded_priors(
        np.asarray([5_000.0, 20_000.0]),
        np.asarray([0.05, 0.05]),
        distribution=distribution,
    )
    assert draws == pytest.approx([-10.0, -20.0])


def test_rows_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        advani_summers_rows({"rows": []})
    bad = _synthetic_distribution()
    bad["rows"][1]["p50"] = -1.0
    with pytest.raises(ValueError, match="non-decreasing"):
        advani_summers_rows(bad)


def test_resource_sha256_is_the_committed_bytes() -> None:
    digest = advani_summers_resource_sha256()
    expected = hashlib.sha256(
        files("microcosm.build.uk").joinpath(ADVANI_SUMMERS_RESOURCE).read_bytes()
    ).hexdigest()
    assert digest == expected
    assert len(digest) == 64 and digest == digest.lower()
    assert CGT_QUANTILE_POINTS == (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
