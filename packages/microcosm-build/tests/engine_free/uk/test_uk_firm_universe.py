from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.firm_universe import (
    N_HMRC_BANDS,
    assign_vat_scope_and_registration,
    registration_propensity,
    stratified_thin,
)

THIN = {
    "window_lo_k": 15.0,
    "window_hi_k": 150.0,
    "window_fraction": 0.3,
    "tail_fraction": 0.05,
    "cell_floor": 5,
    "seed": 42,
}


def _draw(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    turnover = rng.lognormal(mean=4.0, sigma=1.5, size=n)
    sic = rng.choice(
        np.array([1, 41, 47, 62, 86]), size=n, p=[0.05, 0.2, 0.4, 0.3, 0.05]
    )
    bands = np.digitize(
        turnover, [0.0, 90.0, 150.0, 300.0, 500.0, 1000.0, 10000.0], right=True
    )
    return turnover, sic, bands.astype(np.int64)


def _strata(turnover, sic, bands) -> np.ndarray:
    window = (turnover >= THIN["window_lo_k"]) & (turnover <= THIN["window_hi_k"])
    keys = np.stack([sic, bands, window.astype(np.int64)], axis=1)
    return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)


def test_full_fractions_return_every_row_at_unit_weight() -> None:
    turnover, sic, bands = _draw()
    params = {**THIN, "window_fraction": 1.0, "tail_fraction": 1.0}
    keep, base = stratified_thin(turnover, sic, bands, **params)
    np.testing.assert_array_equal(keep, np.arange(len(turnover)))
    np.testing.assert_array_equal(base, np.ones(len(turnover)))


def test_base_weighted_stratum_totals_reproduce_the_draw() -> None:
    turnover, sic, bands = _draw()
    keep, base = stratified_thin(turnover, sic, bands, **THIN)
    assert len(keep) < len(turnover) / 2
    strata = _strata(turnover, sic, bands)
    original = np.bincount(strata)
    weighted = np.bincount(strata[keep], weights=base, minlength=len(original))
    np.testing.assert_allclose(weighted, original, rtol=1e-12, atol=0.0)
    assert np.round(weighted).astype(np.int64).tolist() == original.tolist()


def test_cell_floor_keeps_small_strata() -> None:
    turnover, sic, bands = _draw()
    keep, _ = stratified_thin(turnover, sic, bands, **THIN)
    strata = _strata(turnover, sic, bands)
    sizes = np.bincount(strata)
    kept = np.bincount(strata[keep], minlength=len(sizes))
    assert np.all(kept >= np.minimum(THIN["cell_floor"], sizes))
    # Three-row stratum, zero tail fraction: the floor keeps all three.
    params = {**THIN, "window_fraction": 0.0, "tail_fraction": 0.0}
    keep, base = stratified_thin(
        np.array([500.0, 600.0, 700.0]),
        np.array([1, 1, 1]),
        np.array([5, 5, 5]),
        **params,
    )
    np.testing.assert_array_equal(keep, [0, 1, 2])
    np.testing.assert_array_equal(base, [1.0, 1.0, 1.0])


def test_floor_top_up_takes_lowest_draws_and_weights_by_ratio() -> None:
    n = 12
    turnover = np.full(n, 500.0)
    params = {**THIN, "tail_fraction": 0.0, "cell_floor": 4, "seed": 7}
    keep, base = stratified_thin(
        turnover, np.ones(n, dtype=np.int64), np.full(n, 4), **params
    )
    u = np.random.default_rng(8).random(n)
    np.testing.assert_array_equal(keep, np.sort(np.argsort(u)[:4]))
    np.testing.assert_array_equal(base, np.full(4, 3.0))


def test_thinning_is_deterministic_by_seed() -> None:
    turnover, sic, bands = _draw()
    first = stratified_thin(turnover, sic, bands, **THIN)
    second = stratified_thin(turnover, sic, bands, **THIN)
    other = stratified_thin(turnover, sic, bands, **{**THIN, "seed": 43})
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])


def test_propensity_is_hmrc_count_over_frame_mass() -> None:
    bands = np.array([0, 1, 1, 1, 2, 2, 3, 0, 7])
    frame = np.array([True, True, True, True, True, True, True, False, True])
    base = np.array([1.0, 10.0, 20.0, 30.0, 4.0, 6.0, 2.0, 1.0, 5.0])
    hmrc = np.array([99.0, 15.0, 5.0, 8.0, 0.0, 0.0, 0.0, 7.0])
    propensity = registration_propensity(bands, frame, base, hmrc)
    expected = [0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(propensity, expected, rtol=0, atol=1e-12)


def test_propensity_leaves_empty_bands_at_zero() -> None:
    bands = np.array([4, 4])
    propensity = registration_propensity(
        bands, np.array([True, True]), np.zeros(2), np.full(N_HMRC_BANDS, 5.0)
    )
    np.testing.assert_array_equal(propensity, [0.0, 0.0])


def test_registration_totals_match_hmrc_within_one_weight() -> None:
    rng = np.random.default_rng(3)
    n = 5000
    bands = rng.integers(0, N_HMRC_BANDS, size=n)
    frame = rng.random(n) > 0.05
    weights = rng.uniform(1.0, 20.0, size=n)
    mass = np.bincount(bands[frame], weights=weights[frame], minlength=N_HMRC_BANDS)
    hmrc = np.round(mass * np.array([0.0, 0.4, 0.9, 0.8, 0.7, 0.6, 0.5, 0.95]))
    scope, registered = assign_vat_scope_and_registration(
        bands, frame, weights, hmrc, seed=42
    )
    assert scope[~frame].all() and registered[~frame].all()
    assert not (scope | registered)[frame & (bands == 0)].any()
    assert scope[frame & (bands == 1)].all()
    assert np.array_equal(scope[frame & (bands >= 2)], registered[frame & (bands >= 2)])
    assert not (registered & ~scope).any()
    for band in range(1, N_HMRC_BANDS):
        rows = frame & (bands == band)
        total = weights[rows & registered].sum()
        assert hmrc[band] <= total < hmrc[band] + weights[rows].max()
    again = assign_vat_scope_and_registration(bands, frame, weights, hmrc, seed=42)
    np.testing.assert_array_equal(again[1], registered)


def test_registration_selects_whole_band_when_target_exceeds_mass() -> None:
    bands = np.array([2, 2, 2])
    hmrc = np.zeros(N_HMRC_BANDS)
    hmrc[2] = 100.0
    scope, registered = assign_vat_scope_and_registration(
        bands, np.ones(3, dtype=bool), np.ones(3), hmrc, seed=1
    )
    assert scope.all() and registered.all()


def test_hmrc_bands_must_align_with_band_index() -> None:
    with pytest.raises(ValueError, match="hmrc_bands"):
        registration_propensity(np.array([1]), np.array([True]), np.ones(1), np.ones(7))
