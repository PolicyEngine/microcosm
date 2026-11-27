from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    assign_binary_with_anchored_residual,
    clipped_normal_from_uniforms,
    integer_record_keys,
    keyed_tie_order,
    keyed_tie_uniforms,
    sample_categorical_from_counts,
    stable_identity_uniform,
    stable_identity_uniforms,
    weight_balanced_uniforms,
)


def test_no_anchor_falls_back_to_draws_less_than_rate() -> None:
    draws = np.random.default_rng(0).random(1000)

    result = assign_binary_with_anchored_residual(draws, 0.3)

    assert abs(result.mean() - 0.3) < 0.05
    np.testing.assert_array_equal(result, draws < 0.3)


def test_reported_anchor_forces_true_for_reporters() -> None:
    draws = np.random.default_rng(1).random(1000)
    anchor = np.zeros(1000, dtype=bool)
    anchor[:100] = True

    result = assign_binary_with_anchored_residual(draws, 0.3, anchor)

    assert result[:100].all()


def test_reported_anchor_hits_target_rate() -> None:
    draws = np.random.default_rng(2).random(10000)
    anchor = np.zeros(10000, dtype=bool)
    anchor[:1000] = True

    result = assign_binary_with_anchored_residual(draws, 0.3, anchor)

    assert abs(result.mean() - 0.3) < 0.02


def test_reported_anchor_when_reporters_exceed_target() -> None:
    draws = np.random.default_rng(3).random(1000)
    anchor = np.zeros(1000, dtype=bool)
    anchor[:500] = True

    result = assign_binary_with_anchored_residual(draws, 0.3, anchor)

    assert result[:500].all()
    assert not result[500:].any()


def test_reported_anchor_length_validation() -> None:
    draws = np.random.default_rng(4).random(100)

    with pytest.raises(ValueError, match="must align"):
        assign_binary_with_anchored_residual(draws, 0.3, np.zeros(50, dtype=bool))


def test_anchor_uses_unweighted_int_floor_target() -> None:
    draws = np.array([0.0, 0.2, 0.39, 0.4, 0.99])
    anchor = np.array([False, True, False, False, False])

    result = assign_binary_with_anchored_residual(draws, 0.59, anchor)

    np.testing.assert_array_equal(result, np.array([True, True, False, False, False]))


def test_assign_binary_from_rate_validates_rate() -> None:
    np.testing.assert_array_equal(
        assign_binary_from_rate(np.array([0.1, 0.4, 0.9]), 0.5),
        np.array([True, True, False]),
    )
    with pytest.raises(ValueError, match="rate must be"):
        assign_binary_from_rate(np.array([0.1]), 1.2)


def test_clipped_normal_from_uniforms_uses_inverse_cdf_and_bounds() -> None:
    result = clipped_normal_from_uniforms(
        np.array([0.5, 1e-12, 1 - 1e-12]),
        mean=15.0,
        sd=5.0,
        lower=0.0,
        upper=30.0,
    )

    assert result[0] == pytest.approx(15.0)
    assert result[1] == 0.0
    assert result[2] == 30.0


def test_count_table_sampling_uses_hand_computed_cdf() -> None:
    draws = np.array([0.0, 0.249, 0.25, 0.749, 0.75, 0.999])

    result = sample_categorical_from_counts(
        draws,
        counts={"A": 1, "B": 2, "C": 1},
    )

    np.testing.assert_array_equal(
        result, np.array(["A", "A", "B", "B", "C", "C"], dtype=object)
    )


def test_count_table_sampling_fails_closed_for_missing_cell() -> None:
    with pytest.raises(ValueError, match="empty"):
        sample_categorical_from_counts(np.array([0.1]), counts={})


def test_stable_identity_uniforms_separates_seed_salt_and_id() -> None:
    ids = np.array([3, 1, 2])
    draws = stable_identity_uniforms(ids, seed=0, salt="x")

    assert ((0 <= draws) & (draws < 1)).all()
    assert not np.array_equal(draws, stable_identity_uniforms(ids, seed=1, salt="x"))
    assert not np.array_equal(draws, stable_identity_uniforms(ids, seed=0, salt="y"))
    assert draws[0] != stable_identity_uniforms(np.array([4]), seed=0, salt="x")[0]


def test_stable_identity_uniforms_is_permutation_equivariant() -> None:
    ids = np.array(["a", "b", "c"])
    draws = stable_identity_uniforms(ids, seed=7, salt="flag")
    permuted_ids = ids[[2, 0, 1]]
    permuted_draws = stable_identity_uniforms(permuted_ids, seed=7, salt="flag")

    np.testing.assert_array_equal(permuted_draws[[1, 2, 0]], draws)


def test_stable_identity_uniform_is_the_array_helpers_draw_for_one_label() -> None:
    # Labels of the kind the UK stages key a walk's offset on (microcosm#931).
    for key, seed, salt in (
        ("cell:0:45:london", 552, "hmrc_cgt_gains_spine:offset:2024"),
        ("0|10000", 555, "assign_badr_qualifying_gains:offset:2024"),
        (1001, 7, "flag"),
    ):
        value = stable_identity_uniform(key, seed=seed, salt=salt)
        assert type(value) is float
        assert 0.0 <= value < 1.0
        assert value == stable_identity_uniforms([key], seed=seed, salt=salt)[0]
    base = stable_identity_uniform("a", seed=0, salt="x")
    assert base != stable_identity_uniform("a", seed=1, salt="x")
    assert base != stable_identity_uniform("a", seed=0, salt="y")
    assert base != stable_identity_uniform("b", seed=0, salt="x")


def test_stable_identity_uniform_refuses_a_sequence_for_its_label() -> None:
    # A list would be formatted whole into the key and draw something else.
    for key in (["a"], ("a", "b"), np.array([1, 2])):
        with pytest.raises(TypeError, match="single value"):
            stable_identity_uniform(key, seed=0, salt="x")


def test_anchored_residual_population_restricts_the_draw_and_the_target() -> None:
    """The rate is a share of the population; rows outside it are never drawn."""

    # 60 population rows with evenly spread draws, then 40 rows outside it.
    draws = np.concatenate([np.linspace(0.0, 0.99, 60), np.linspace(0.0, 0.99, 40)])
    population = np.arange(100) < 60
    anchor = np.zeros(100, dtype=bool)
    anchor[5] = True  # inside the population
    anchor[95] = True  # outside it: reported receipt stays true

    result = assign_binary_with_anchored_residual(
        draws, 0.5, anchor, population=population
    )

    # int(0.5 * 60) = 30 inside the population: the anchor plus 29 residual
    # rows at the adjusted threshold 29 / 59 over the evenly spread draws.
    assert int(result[population].sum()) == 30
    assert result[95] and int(result[~population].sum()) == 1
    assert not (result & ~population & ~anchor).any()


def test_anchored_residual_population_without_anchor_and_shape_refusal() -> None:
    draws = np.linspace(0.0, 0.99, 10)
    population = np.array([True] * 5 + [False] * 5)

    result = assign_binary_with_anchored_residual(draws, 0.4, population=population)

    assert result.tolist() == [True, True, True, True, False] + [False] * 5
    with pytest.raises(ValueError, match="population and draws must align"):
        assign_binary_with_anchored_residual(
            draws, 0.4, np.zeros(10, dtype=bool), population=np.ones(3, dtype=bool)
        )


# --- the anchored residual within strata (microcosm#931) ---------------------


def _anchored_residual_without_strata(draws, rate, anchor=None, population=None):
    """The anchored residual's arithmetic as it stood before ``strata`` was
    added, copied unchanged (its refusals aside), as the oracle that the
    calls without strata, and with every row in one stratum, are held to
    byte for byte."""

    draws = np.asarray(draws, dtype=np.float64)
    if population is None:
        population = np.ones(draws.shape, dtype=bool)
    else:
        population = np.asarray(population, dtype=bool)
    if anchor is None:
        return population & (draws < rate)
    anchor = np.asarray(anchor, dtype=bool)
    result = anchor.copy()
    target = int(rate * int(population.sum()))
    remaining_needed = max(0, target - int((anchor & population).sum()))
    non_anchored = ~anchor & population
    if remaining_needed == 0 or not non_anchored.any():
        return result
    adjusted = remaining_needed / int(non_anchored.sum())
    result |= non_anchored & (draws < adjusted)
    return result


def _random_anchored_cases(seed: int, trials: int):
    """Invented draws, rates, anchors and populations, with or without each.

    Draws sometimes repeat or sit on the rate; anchors and populations run
    from none to every row, so the residual fills, overshoots, has no row left
    to fill and meets no anchor."""

    rng = np.random.default_rng(seed)
    densities = (0.0, 0.05, 0.3, 0.9, 1.0)
    for trial in range(trials):
        n = int(rng.integers(0, 120))
        draws = rng.random(n)
        if trial % 4 == 0 and n:
            draws = np.round(draws, 1)
        rate = float(rng.choice([0.0, 1.0, 0.5, *rng.random(3)]))
        if trial % 5 == 0 and n:
            draws[int(rng.integers(0, n))] = rate
        anchor = (
            None
            if trial % 6 == 0
            else rng.random(n) < densities[trial % len(densities)]
        )
        population = (
            None if trial % 3 == 0 else rng.random(n) < float(rng.choice(densities))
        )
        yield draws, rate, anchor, population


def test_without_strata_or_in_one_stratum_the_flags_are_unchanged() -> None:
    """``strata=None``, the default, and every row in one stratum, under three
    labels and integer dtypes, give the flags the arithmetic gave before
    strata were added, byte for byte."""

    branches = {"no anchor": 0, "filled": 0, "nothing to fill": 0}
    for draws, rate, anchor, population in _random_anchored_cases(11, 600):
        expected = _anchored_residual_without_strata(draws, rate, anchor, population)
        assert expected.dtype == np.bool_
        n = draws.size
        results = [
            assign_binary_with_anchored_residual(draws, rate, anchor, population),
            assign_binary_with_anchored_residual(
                draws, rate, anchor, population, strata=None
            ),
            assign_binary_with_anchored_residual(
                draws, rate, anchor, population, strata=np.zeros(n, dtype=np.int64)
            ),
            assign_binary_with_anchored_residual(
                draws, rate, anchor, population, strata=np.full(n, -7, dtype=np.int8)
            ),
            assign_binary_with_anchored_residual(
                draws, rate, anchor, population, strata=np.full(n, 3, dtype=np.uint16)
            ),
        ]
        for result in results:
            assert result.dtype == expected.dtype
            assert result.shape == expected.shape
            assert result.tobytes() == expected.tobytes()
        if anchor is None:
            branches["no anchor"] += 1
        elif (expected & ~anchor).any():
            branches["filled"] += 1
        else:
            branches["nothing to fill"] += 1
    assert min(branches.values()) > 50, branches


def _two_strata() -> pd.DataFrame:
    """Two invented strata, interleaved, at a rate of one half.

    Stratum -2: eight reachable rows, one anchored, so its target is
    int(0.5 x 8) = 4 and its residual fills 3 of its 7 other rows, at the
    threshold 3 / 7, which the draws (2i + 1) / 14 cross after their third.
    Stratum 5: six rows, four reachable, three of those anchored, and an
    anchor outside the reach: its target int(0.5 x 4) = 2 is already
    exceeded, so it fills nothing. One count over both strata would aim at
    int(0.5 x 12) = 6, need 2 of its 8 unanchored reachable rows, at the
    threshold 2 / 8, and so take three: stratum 5's one (drawn at 0.1) and
    two of stratum -2's."""

    rows = [
        # (stratum, draw, anchor, reachable, with strata, without strata)
        (5, 0.10, False, True, False, True),
        (-2, 9 / 14, False, True, False, False),
        (-2, 0.90, True, True, True, True),
        (5, 0.95, True, True, True, True),
        (-2, 1 / 14, False, True, True, True),
        (5, 0.40, True, True, True, True),
        (-2, 13 / 14, False, True, False, False),
        (-2, 5 / 14, False, True, True, False),
        (5, 0.01, False, False, False, False),
        (-2, 3 / 14, False, True, True, True),
        (5, 0.70, True, False, True, True),
        (-2, 11 / 14, False, True, False, False),
        (5, 0.20, True, True, True, True),
        (-2, 7 / 14, False, True, False, False),
    ]
    return pd.DataFrame(
        rows,
        columns=["stratum", "draw", "anchor", "reachable", "strata", "pooled"],
    )


def test_strata_count_each_stratums_target_on_its_own_rows() -> None:
    """Each stratum meets its own target on its own rows (:func:`_two_strata`),
    where one count over both meets one target over every row."""

    table = _two_strata()
    draws = table["draw"].to_numpy()
    anchor = table["anchor"].to_numpy()
    population = table["reachable"].to_numpy()
    strata = table["stratum"].to_numpy(dtype=np.int64)

    result = assign_binary_with_anchored_residual(
        draws, 0.5, anchor, population, strata=strata
    )

    assert result.tolist() == table["strata"].tolist()
    pooled = assign_binary_with_anchored_residual(draws, 0.5, anchor, population)
    assert pooled.tolist() == table["pooled"].tolist()
    # Each stratum's flags are those its rows take alone.
    for label in (-2, 5):
        rows = strata == label
        np.testing.assert_array_equal(
            result[rows],
            assign_binary_with_anchored_residual(
                draws[rows], 0.5, anchor[rows], population[rows]
            ),
        )
    # The labels only name the strata, and the rows' order does not matter.
    renamed = np.where(strata == 5, 0, 1).astype(np.int32)
    np.testing.assert_array_equal(
        assign_binary_with_anchored_residual(
            draws, 0.5, anchor, population, strata=renamed
        ),
        result,
    )
    order = np.random.default_rng(2).permutation(len(table))
    np.testing.assert_array_equal(
        assign_binary_with_anchored_residual(
            draws[order],
            0.5,
            anchor[order],
            population[order],
            strata=strata[order],
        ),
        result[order],
    )


def test_a_stratum_takes_its_own_flags_whatever_the_others_hold() -> None:
    """Stratum 0 takes the flags its rows take alone at every rate, beside
    strata of other draws over the same anchors and reach, as geographic
    clone 0 does beside the other clones (microcosm#931). One count over every
    stratum moves them at some rates, so the strata are not idle."""

    rng = np.random.default_rng(17)
    n, copies = 40, 3
    draws = rng.random(n)
    anchor = rng.random(n) < 0.15
    population = rng.random(n) < 0.8
    stacked_draws = np.concatenate([draws, rng.random(n * (copies - 1))])
    stacked_anchor = np.tile(anchor, copies)
    stacked_population = np.tile(population, copies)
    strata = np.repeat(np.arange(copies, dtype=np.int64), n)
    pooled_moves = 0
    for rate in np.linspace(0.0, 1.0, 81):
        alone = assign_binary_with_anchored_residual(draws, rate, anchor, population)
        stratified = assign_binary_with_anchored_residual(
            stacked_draws, rate, stacked_anchor, stacked_population, strata=strata
        )
        np.testing.assert_array_equal(stratified[:n], alone)
        pooled = assign_binary_with_anchored_residual(
            stacked_draws, rate, stacked_anchor, stacked_population
        )
        pooled_moves += not np.array_equal(pooled[:n], alone)
    assert pooled_moves > 0


def test_without_an_anchor_strata_change_nothing() -> None:
    rng = np.random.default_rng(8)
    draws = rng.random(50)
    population = rng.random(50) < 0.6
    strata = rng.integers(0, 4, 50)

    for rate in (0.0, 0.3, 1.0):
        expected = population & (draws < rate)
        np.testing.assert_array_equal(
            assign_binary_with_anchored_residual(
                draws, rate, population=population, strata=strata
            ),
            expected,
        )
        np.testing.assert_array_equal(
            assign_binary_with_anchored_residual(draws, rate, strata=strata),
            draws < rate,
        )


@pytest.mark.parametrize("with_anchor", [True, False])
def test_strata_are_refused_misaligned_non_integer_or_missing(with_anchor) -> None:
    """Strata are checked whether or not an anchor is given, and every refusal
    names no count, label or row. An integer dtype holds no missing value, so
    a missing label is refused with the other dtypes that are not integers."""

    draws = np.linspace(0.0, 0.9, 4)
    anchor = np.array([True, False, False, False]) if with_anchor else None
    refused = (
        (np.zeros(3, dtype=np.int64), ValueError, "strata and draws must align"),
        (np.zeros((2, 2), dtype=np.int64), ValueError, "strata and draws must align"),
        (np.zeros(4), TypeError, "strata must be integers"),
        (np.zeros(4, dtype=bool), TypeError, "strata must be integers"),
        (np.array([0, 1, 0, 1], dtype=object), TypeError, "strata must be integers"),
        (np.array(["a", "b", "a", "b"]), TypeError, "strata must be integers"),
        (np.array([0.0, np.nan, 1.0, 1.0]), TypeError, "none missing"),
        (pd.array([0, None, 1, 1], dtype="Int64"), TypeError, "none missing"),
        (np.array([0, None, 1, 1], dtype=object), TypeError, "none missing"),
    )
    for strata, error, message in refused:
        with pytest.raises(error, match=message) as caught:
            assign_binary_with_anchored_residual(draws, 0.5, anchor, strata=strata)
        assert not any(character.isdigit() for character in str(caught.value))


def test_integer_record_keys_pass_unique_integer_ids_through() -> None:
    ids = np.array([3, 1, 2], dtype="int64")

    np.testing.assert_array_equal(integer_record_keys(ids, label="ids"), ids)


def test_a_float_id_keys_a_different_draw_so_it_is_refused() -> None:
    # The key string formats each id, so 1001 and 1001.0 draw differently:
    # a merge that turned an id column into floats would redraw every record.
    assert not np.array_equal(
        stable_identity_uniforms([1001], seed=0, salt="s"),
        stable_identity_uniforms([1001.0], seed=0, salt="s"),
    )
    with pytest.raises(TypeError, match="integer ids"):
        integer_record_keys(np.array([1001.0, 1002.0]), label="ids")
    with pytest.raises(TypeError, match="integer ids"):
        integer_record_keys(np.array(["1001"], dtype=object), label="ids")


def test_a_repeated_id_is_refused() -> None:
    with pytest.raises(ValueError, match="each record once"):
        integer_record_keys(np.array([1, 2, 1]), label="ids")


def test_integer_record_keys_find_repeats_as_np_unique_does() -> None:
    # The sort-based check gives np.unique's verdict on every array: random
    # integer ids of several dtypes, narrow ranges making repeats likely,
    # empty and single-id arrays, and the extremes of int64 (microcosm#931).
    # Ids that are not one-dimensional are still refused by type first, and
    # so are timedeltas, numpy integers by subtype whose NaT never repeat.
    for refused in (
        np.array([[1, 2], [1, 2]]),
        np.array([1, 2], dtype="m8[ns]"),
        np.array(["NaT", "NaT"], dtype="m8[ns]"),
    ):
        with pytest.raises(TypeError, match="one-dimensional"):
            integer_record_keys(refused, label="ids")
    rng = np.random.default_rng(23)
    dtypes = (np.int64, np.int32, np.int8, np.uint64, np.uint16)
    verdicts = {True: 0, False: 0}
    extremes = np.array([np.iinfo(np.int64).min, -1, 0, np.iinfo(np.int64).max])
    arrays = [extremes, np.append(extremes, extremes[-1]), np.array([], dtype="int64")]
    for trial in range(400):
        n = int(rng.integers(0, 50))
        span = int(rng.integers(1, 120))
        arrays.append(rng.integers(0, span, n).astype(dtypes[trial % len(dtypes)]))
    for ids in arrays:
        repeated = np.unique(ids).size != ids.size
        verdicts[repeated] += 1
        if repeated:
            with pytest.raises(ValueError, match="each record once"):
                integer_record_keys(ids, label="ids")
        else:
            keys = integer_record_keys(ids, label="ids")
            assert keys.dtype == ids.dtype
            np.testing.assert_array_equal(keys, ids)
    assert min(verdicts.values()) > 50


def test_balanced_uniforms_give_each_stratum_its_rate_to_within_half_a_record() -> None:
    # At midpoints the last record taken sits below the rate and the next one
    # at or above it, so a stratum misses by at most half its heaviest record
    # (the independent review of microcosm#1420, finding 13).
    rng = np.random.default_rng(3)
    n = 500
    strata = rng.integers(0, 6, n)
    draws = rng.random(n)
    weights = rng.uniform(0.5, 5.0, n)
    positions = weight_balanced_uniforms(strata, draws, weights)
    for rate in (0.1, 0.6, 0.9):
        for stratum in np.unique(strata):
            members = strata == stratum
            claimed = float(weights[members][positions[members] < rate].sum())
            assert abs(claimed - rate * weights[members].sum()) <= (
                0.5 * weights[members].max() + 1e-9
            )


def test_a_start_per_stratum_keeps_one_record_and_restores_the_rate() -> None:
    # Rotated by a start per stratum (systematic sampling), a stratum misses by
    # at most half a record at each end of its arc, and every record is taken
    # with probability equal to the rate, which midpoints alone miss in a
    # stratum of one record.
    rng = np.random.default_rng(5)
    n = 600
    strata = rng.integers(0, 8, n)
    draws = rng.random(n)
    weights = rng.uniform(0.5, 5.0, n)
    per_stratum = rng.random(8)
    positions = weight_balanced_uniforms(
        strata, draws, weights, starts=per_stratum[strata]
    )
    for rate in (0.1, 0.6, 0.9):
        for stratum in np.unique(strata):
            members = strata == stratum
            claimed = float(weights[members][positions[members] < rate].sum())
            assert abs(claimed - rate * weights[members].sum()) <= (
                weights[members].max() + 1e-9
            )
    singles = np.arange(1_000)
    rotated = weight_balanced_uniforms(
        singles,
        np.zeros(singles.size),
        np.ones(singles.size),
        starts=(np.arange(1_000) + 0.5) / 1_000,
    )
    assert (rotated < 0.6).mean() == pytest.approx(0.6, abs=0.002)
    with pytest.raises(ValueError, match="aligned"):
        weight_balanced_uniforms(strata, draws, weights, starts=per_stratum)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        weight_balanced_uniforms(strata, draws, weights, starts=np.ones(n))


def test_balanced_uniforms_follow_the_draws_and_not_the_row_order() -> None:
    rng = np.random.default_rng(4)
    n = 200
    strata, draws = rng.integers(0, 3, n), rng.random(n)
    weights = rng.uniform(1.0, 3.0, n)
    positions = weight_balanced_uniforms(strata, draws, weights)
    order = rng.permutation(n)
    np.testing.assert_array_equal(
        weight_balanced_uniforms(strata[order], draws[order], weights[order]),
        positions[order],
    )
    for stratum in np.unique(strata):
        members = strata == stratum
        np.testing.assert_array_equal(
            np.argsort(positions[members]), np.argsort(draws[members])
        )


def test_a_weightless_stratum_keeps_its_draws_and_misalignment_is_refused() -> None:
    positions = weight_balanced_uniforms(
        np.array([0, 0, 1]), np.array([0.3, 0.7, 0.25]), np.array([0.0, 0.0, 2.0])
    )
    np.testing.assert_allclose(positions, [0.3, 0.7, 0.5])
    with pytest.raises(ValueError, match="aligned"):
        weight_balanced_uniforms(np.zeros(2), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        weight_balanced_uniforms(np.zeros(2), np.zeros(2), np.array([1.0, np.nan]))


def test_binomial_share_tolerance_is_sigma_standard_errors_floored_and_capped() -> None:
    from microcosm.build.stochastic_assignment import binomial_share_tolerance

    assert binomial_share_tolerance(target=0.3, rows=1_000, sigma=4.0) == (
        4.0 * (0.3 * 0.7 / 1_000) ** 0.5
    )
    assert binomial_share_tolerance(target=0.3, rows=1_000, sigma=2.0) == (
        2.0 * (0.3 * 0.7 / 1_000) ** 0.5
    )
    # Floored at one row's worth, capped at one, and one without rows.
    assert binomial_share_tolerance(target=1.0, rows=50, sigma=4.0) == 1.0 / 50
    assert binomial_share_tolerance(target=0.5, rows=1, sigma=4.0) == 1.0
    assert binomial_share_tolerance(target=0.5, rows=0, sigma=4.0) == 1.0


def test_the_latent_attribute_band_is_the_shared_band_at_four_sigma() -> None:
    # The stage-health name stays an alias of the moved band (microcosm#931).
    from microcosm.build.stochastic_assignment import binomial_share_tolerance
    from microcosm.build.uk_runtime.stage_health import (
        LATENT_ATTRIBUTE_SIGMA,
        latent_attribute_tolerance,
    )

    assert LATENT_ATTRIBUTE_SIGMA == 4.0
    for target in (0.0, 0.01, 0.3, 0.5, 0.97, 1.0):
        for rows in (0, 1, 7, 250, 40_000):
            # The band as stage_health computed it before the move.
            before = (
                1.0
                if rows <= 0
                else min(
                    1.0,
                    max(1.0 / rows, 4.0 * math.sqrt(target * (1.0 - target) / rows)),
                )
            )
            assert latent_attribute_tolerance(target=target, rows=rows) == before
            assert binomial_share_tolerance(target=target, rows=rows, sigma=4.0) == (
                before
            )


# --- keyed tie-breaks (microcosm#931) ----------------------------------------

_TIE = {"seed": 7, "salt": "test:tie"}


def _ties(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic distinct integer ids and values drawn from a few atoms."""

    ids = rng.choice(10**12, n, replace=False).astype(np.int64)
    values = rng.integers(0, 6, n).astype(float)
    return ids, values


def test_keyed_tie_order_is_the_id_order_where_nothing_ties() -> None:
    rng = np.random.default_rng(3)
    ids = rng.choice(10**12, 500, replace=False).astype(np.int64)
    values = rng.permutation(500).astype(float) * 1.5

    order = keyed_tie_order(values, ids, **_TIE)

    np.testing.assert_array_equal(order, np.lexsort((ids, values)))
    np.testing.assert_array_equal(
        keyed_tie_order(-values, ids, **_TIE), np.lexsort((ids, -values))
    )


def test_keyed_tie_order_sorts_ties_on_their_uniforms_then_ids() -> None:
    # The full form the helper's contract states: np.lexsort((ids, u,
    # primary)) with u every record's uniform, though only tied records are
    # drawn. NaNs tie with each other and -0.0 with 0.0.
    rng = np.random.default_rng(4)
    for trial in range(60):
        ids, values = _ties(int(rng.integers(1, 80)), rng)
        if trial % 3 == 0:
            values[rng.random(values.size) < 0.2] = np.nan
        if trial % 4 == 0:
            values[values == 0.0] = -0.0
        uniforms = stable_identity_uniforms(ids, **_TIE)

        order = keyed_tie_order(values, ids, **_TIE)

        np.testing.assert_array_equal(order, np.lexsort((ids, uniforms, values)))
        np.testing.assert_array_equal(np.sort(order), np.arange(values.size))
    np.testing.assert_array_equal(
        keyed_tie_uniforms(ids, **_TIE), stable_identity_uniforms(ids, **_TIE)
    )


def test_keyed_tie_order_breaks_a_tie_otherwise_than_the_ids_would() -> None:
    # Three records tie; their uniforms, not their ids, set their order.
    ids = np.asarray([11, 12, 13, 14, 15], dtype=np.int64)
    values = np.asarray([2.0, 1.0, 1.0, 1.0, 0.0])
    tied = ids[1:4]
    keyed = tied[np.argsort(stable_identity_uniforms(tied, **_TIE))]
    assert keyed.tolist() != sorted(keyed.tolist())

    order = keyed_tie_order(values, ids, **_TIE)

    assert ids[order].tolist() == [15, *keyed.tolist(), 11]


def test_keyed_tie_order_does_not_depend_on_the_row_order() -> None:
    rng = np.random.default_rng(5)
    ids, values = _ties(300, rng)
    expected = ids[keyed_tie_order(values, ids, **_TIE)]
    for _ in range(5):
        rows = rng.permutation(ids.size)
        order = keyed_tie_order(values[rows], ids[rows], **_TIE)
        np.testing.assert_array_equal(ids[rows][order], expected)


def test_keyed_tie_order_draws_on_its_salt_and_refuses_unusable_ids() -> None:
    rng = np.random.default_rng(6)
    ids, values = _ties(200, rng)
    salted = keyed_tie_order(values, ids, seed=7, salt="test:other")
    assert not np.array_equal(keyed_tie_order(values, ids, **_TIE), salted)
    with pytest.raises(TypeError, match="integer ids"):
        keyed_tie_order([1.0, 1.0], [1.0, 2.0], **_TIE)
    with pytest.raises(ValueError, match="each record once"):
        keyed_tie_order([1.0, 1.0], np.asarray([3, 3]), **_TIE)
    with pytest.raises(ValueError, match="aligned"):
        keyed_tie_order([1.0, 1.0, 2.0], np.asarray([3, 4]), **_TIE)
    assert keyed_tie_order([], np.asarray([], dtype=np.int64), **_TIE).size == 0
