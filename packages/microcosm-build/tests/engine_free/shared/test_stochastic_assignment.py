from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    assign_binary_with_anchored_residual,
    clipped_normal_from_uniforms,
    integer_record_keys,
    sample_categorical_from_counts,
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
