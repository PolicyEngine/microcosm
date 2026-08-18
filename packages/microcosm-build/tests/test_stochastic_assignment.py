from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    assign_binary_with_anchored_residual,
    clipped_normal_from_uniforms,
    sample_categorical_from_counts,
    stable_identity_uniforms,
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
