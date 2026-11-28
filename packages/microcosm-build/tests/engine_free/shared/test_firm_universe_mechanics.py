"""The country-neutral firm mechanics: band draws, thinning, registration.

Two registration specs are used throughout. ``EIGHT_BANDS`` has the shape of
a VAT register with a threshold: an excluded band for traders outside the
frame, one registrable band below the threshold and six selected bands above
it. ``FOUR_BANDS`` has no registrable band and no excluded band, so the
functions are tested on a shape no current country uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.firms import (
    BAND_ROLES,
    RegistrationSpec,
    assign_scope_and_registration,
    draw_in_band,
    registration_propensity,
    select_to_weighted_target,
    stratified_thin,
)

EIGHT_BANDS = RegistrationSpec(
    tuple(f"band_{b}" for b in range(8)),
    ("excluded", "registrable", *("selected",) * 6),
)
FOUR_BANDS = RegistrationSpec(("a", "b", "c", "d"), ("selected",) * 4)

THIN = {
    "window_lo": 15.0,
    "window_hi": 150.0,
    "window_fraction": 0.3,
    "tail_fraction": 0.05,
    "cell_floor": 5,
    "rng_seed": 43,
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
    window = (turnover >= THIN["window_lo"]) & (turnover <= THIN["window_hi"])
    keys = np.stack([sic, bands, window.astype(np.int64)], axis=1)
    return np.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)


# Band draws.


def test_closed_band_draw_is_uniform_inside_the_band() -> None:
    unit = np.array([0.0, 0.25, 0.5, 0.999])
    values = draw_in_band(unit, 10.0, 50.0)
    np.testing.assert_allclose(values, [10.0, 20.0, 30.0, 49.96])


def test_floor_lifts_a_zero_lower_edge() -> None:
    values = draw_in_band(np.array([0.0, 0.5]), 0.0, 10.0, floor=2.0)
    np.testing.assert_allclose(values, [2.0, 6.0])


def test_open_band_draw_is_log_uniform_and_stays_below_the_cap() -> None:
    unit = np.array([0.0, 0.5, 0.999999])
    values = draw_in_band(unit, 100.0, 10000.0, log_uniform=True)
    np.testing.assert_allclose(values[:2], [100.0, 1000.0])
    assert values[2] < 10000.0


def test_band_draw_keeps_torch_type_and_dtype() -> None:
    torch = pytest.importorskip("torch")
    unit = torch.tensor([0.0, 0.5], dtype=torch.float32)
    for log_uniform in (False, True):
        values = draw_in_band(unit, 1.0, 100.0, log_uniform=log_uniform)
        assert isinstance(values, torch.Tensor)
        assert values.dtype == torch.float32


def test_band_draw_refuses_inverted_and_non_positive_log_bands() -> None:
    with pytest.raises(ValueError, match="below its low edge"):
        draw_in_band(np.array([0.5]), 10.0, 5.0)
    with pytest.raises(ValueError, match="positive low edge"):
        draw_in_band(np.array([0.5]), 0.0, 5.0, log_uniform=True)


# Thinning.


def test_full_fractions_return_every_row_at_unit_weight() -> None:
    turnover, sic, bands = _draw()
    params = {**THIN, "window_fraction": 1.0, "tail_fraction": 1.0}
    keep, base = stratified_thin(turnover, [sic, bands], **params)
    np.testing.assert_array_equal(keep, np.arange(len(turnover)))
    np.testing.assert_array_equal(base, np.ones(len(turnover)))


def test_base_weighted_stratum_totals_reproduce_the_draw() -> None:
    turnover, sic, bands = _draw()
    keep, base = stratified_thin(turnover, [sic, bands], **THIN)
    assert len(keep) < len(turnover) / 2
    strata = _strata(turnover, sic, bands)
    original = np.bincount(strata)
    weighted = np.bincount(strata[keep], weights=base, minlength=len(original))
    np.testing.assert_allclose(weighted, original, rtol=1e-12, atol=0.0)


def test_cell_floor_keeps_small_strata() -> None:
    turnover, sic, bands = _draw()
    keep, _ = stratified_thin(turnover, [sic, bands], **THIN)
    strata = _strata(turnover, sic, bands)
    sizes = np.bincount(strata)
    kept = np.bincount(strata[keep], minlength=len(sizes))
    assert np.all(kept >= np.minimum(THIN["cell_floor"], sizes))
    params = {**THIN, "window_fraction": 0.0, "tail_fraction": 0.0}
    keep, base = stratified_thin(
        np.array([500.0, 600.0, 700.0]), [np.array([1, 1, 1])], **params
    )
    np.testing.assert_array_equal(keep, [0, 1, 2])
    np.testing.assert_array_equal(base, [1.0, 1.0, 1.0])


def test_floor_top_up_takes_lowest_draws_and_weights_by_ratio() -> None:
    n = 12
    params = {**THIN, "tail_fraction": 0.0, "cell_floor": 4, "rng_seed": 8}
    keep, base = stratified_thin(
        np.full(n, 500.0), [np.ones(n, dtype=np.int64)], **params
    )
    u = np.random.default_rng(8).random(n)
    np.testing.assert_array_equal(keep, np.sort(np.argsort(u)[:4]))
    np.testing.assert_array_equal(base, np.full(4, 3.0))


def test_thinning_without_keys_stratifies_by_window_side() -> None:
    values = np.array([10.0] * 6 + [100.0] * 4)
    params = {**THIN, "window_fraction": 0.0, "tail_fraction": 0.0, "cell_floor": 2}
    keep, base = stratified_thin(values, [], **params)
    assert len(keep) == 4
    assert sorted(base.tolist()) == [2.0, 2.0, 3.0, 3.0]


def test_thinning_is_deterministic_by_seed() -> None:
    turnover, sic, bands = _draw()
    first = stratified_thin(turnover, [sic, bands], **THIN)
    second = stratified_thin(turnover, [sic, bands], **THIN)
    other = stratified_thin(turnover, [sic, bands], **{**THIN, "rng_seed": 44})
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], other[0])


def test_thinning_refuses_misaligned_or_non_integer_keys() -> None:
    with pytest.raises(ValueError, match=r"strata\[0\]"):
        stratified_thin(np.ones(3), [np.ones(2, dtype=np.int64)], **THIN)
    with pytest.raises(TypeError, match="integer"):
        stratified_thin(np.ones(3), [np.ones(3)], **THIN)


# Registration spec.


def test_spec_records_bands_by_role() -> None:
    assert EIGHT_BANDS.n_bands == 8
    assert EIGHT_BANDS.bands_with_role("excluded") == (0,)
    assert EIGHT_BANDS.bands_with_role("registrable") == (1,)
    assert EIGHT_BANDS.bands_with_role("selected") == tuple(range(2, 8))
    assert set(BAND_ROLES) == {"excluded", "registrable", "selected"}


@pytest.mark.parametrize(
    ("labels", "roles", "message"),
    [
        ((), (), "at least one band"),
        (("a", "b"), ("selected",), "2 band labels and 1 band roles"),
        (("a", "a"), ("selected", "selected"), "unique"),
        (("a",), ("threshold",), "unknown registration band roles"),
    ],
)
def test_spec_refuses_malformed_declarations(labels, roles, message) -> None:
    with pytest.raises(ValueError, match=message):
        RegistrationSpec(labels, roles)


# Propensity.


def test_propensity_is_registered_count_over_frame_mass() -> None:
    bands = np.array([0, 1, 1, 1, 2, 2, 3, 0, 7])
    frame = np.array([True, True, True, True, True, True, True, False, True])
    base = np.array([1.0, 10.0, 20.0, 30.0, 4.0, 6.0, 2.0, 1.0, 5.0])
    counts = np.array([99.0, 15.0, 5.0, 8.0, 0.0, 0.0, 0.0, 7.0])
    propensity = registration_propensity(EIGHT_BANDS, bands, frame, base, counts)
    expected = [0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1.0]
    np.testing.assert_allclose(propensity, expected, rtol=0, atol=1e-12)


def test_propensity_without_an_excluded_band_covers_every_band() -> None:
    bands = np.array([0, 0, 1, 3])
    frame = np.array([True, True, True, False])
    base = np.array([2.0, 2.0, 5.0, 1.0])
    counts = np.array([1.0, 10.0, 0.0, 0.0])
    propensity = registration_propensity(FOUR_BANDS, bands, frame, base, counts)
    np.testing.assert_allclose(propensity, [0.25, 0.25, 1.0, 1.0])


def test_propensity_leaves_empty_bands_at_zero() -> None:
    propensity = registration_propensity(
        EIGHT_BANDS,
        np.array([4, 4]),
        np.array([True, True]),
        np.zeros(2),
        np.full(8, 5.0),
    )
    np.testing.assert_array_equal(propensity, [0.0, 0.0])


def test_registered_counts_must_align_with_the_spec() -> None:
    with pytest.raises(ValueError, match="registered_counts"):
        registration_propensity(
            EIGHT_BANDS, np.array([1]), np.array([True]), np.ones(1), np.ones(7)
        )
    with pytest.raises(ValueError, match="band_idx"):
        registration_propensity(
            FOUR_BANDS, np.array([4]), np.array([True]), np.ones(1), np.ones(4)
        )


# Scope and registration.


@pytest.mark.parametrize("spec", [EIGHT_BANDS, FOUR_BANDS])
def test_registration_totals_match_counts_within_one_weight(spec) -> None:
    rng = np.random.default_rng(3)
    n = 5000
    bands = rng.integers(0, spec.n_bands, size=n)
    frame = rng.random(n) > 0.05
    weights = rng.uniform(1.0, 20.0, size=n)
    mass = np.bincount(bands[frame], weights=weights[frame], minlength=spec.n_bands)
    counts = np.round(mass * rng.uniform(0.3, 0.95, size=spec.n_bands))
    scope, registered = assign_scope_and_registration(
        spec, bands, frame, weights, counts, seed=42
    )
    assert scope[~frame].all() and registered[~frame].all()
    assert not (registered & ~scope).any()
    for band, role in enumerate(spec.band_roles):
        rows = frame & (bands == band)
        if role == "excluded":
            assert not (scope | registered)[rows].any()
            continue
        if role == "registrable":
            assert scope[rows].all()
        else:
            assert np.array_equal(scope[rows], registered[rows])
        total = weights[rows & registered].sum()
        assert counts[band] <= total < counts[band] + weights[rows].max()
    again = assign_scope_and_registration(spec, bands, frame, weights, counts, seed=42)
    np.testing.assert_array_equal(again[1], registered)


def test_registration_selects_whole_band_when_count_exceeds_mass() -> None:
    counts = np.zeros(8)
    counts[2] = 100.0
    scope, registered = assign_scope_and_registration(
        EIGHT_BANDS,
        np.array([2, 2, 2]),
        np.ones(3, dtype=bool),
        np.ones(3),
        counts,
        seed=1,
    )
    assert scope.all() and registered.all()


def test_weighted_selection_draws_nothing_for_an_empty_or_zero_target() -> None:
    rng = np.random.default_rng(5)
    state = rng.bit_generator.state
    empty = select_to_weighted_target(np.zeros(0, dtype=np.int64), np.ones(3), 2.0, rng)
    zero = select_to_weighted_target(np.arange(3), np.ones(3), 0.0, rng)
    assert len(empty) == len(zero) == 0
    assert rng.bit_generator.state == state
