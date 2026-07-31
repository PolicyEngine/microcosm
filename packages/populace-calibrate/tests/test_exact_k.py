"""Exact-cardinality Sampford selection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from populace.calibrate import assert_exact_k_support, select_exact_k


def test_certainty_units_are_force_included_before_boundary_draw() -> None:
    pi = np.asarray([0.99, 0.98, 0.60, 0.30, 0.10])

    support, receipt = select_exact_k(pi, k=3, pi_hi=0.95, seed=17)

    assert len(support) == 3
    assert {0, 1}.issubset(support.tolist())
    assert receipt["certainty_count"] == 2
    assert receipt["boundary_pool_size"] == 3


@pytest.mark.parametrize(
    ("k", "pi_hi"),
    [
        (0, 1.0),
        (1, 0.90),
        (3, 0.90),
        (5, 0.80),
        (7, 0.60),
        (10, 1.0),
    ],
)
def test_exact_cardinality_for_threshold_and_budget_combinations(
    k: int,
    pi_hi: float,
) -> None:
    pi = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    support, receipt = select_exact_k(pi, k=k, pi_hi=pi_hi, seed=42)

    assert len(support) == k
    assert np.unique(support).size == k
    assert ((0 <= support) & (support < len(pi))).all()
    assert receipt["k"] == k


def test_census_and_proportional_take_alls_are_deterministic() -> None:
    census, _ = select_exact_k([0.0, 0.2, 0.8], k=3, pi_hi=1.0, seed=0)
    take_all, _ = select_exact_k([0.0, 0.5, 0.5], k=2, pi_hi=1.0, seed=0)

    np.testing.assert_array_equal(census, [0, 1, 2])
    np.testing.assert_array_equal(take_all, [1, 2])


def test_mixed_boundary_take_all_and_fractional_draw() -> None:
    # Scaling by 2 / 1.6 gives target probabilities [1, .5, .25, .25].
    pi = np.asarray([0.8, 0.4, 0.2, 0.2])
    draws = 2_000
    counts = np.zeros(len(pi), dtype=np.int64)

    for seed in range(draws):
        support, receipt = select_exact_k(pi, k=2, pi_hi=0.95, seed=seed)
        counts[support] += 1

    frequencies = counts / draws
    assert receipt["certainty_count"] == 0
    np.testing.assert_allclose(frequencies, [1.0, 0.5, 0.25, 0.25], atol=0.04)


def test_majority_draw_uses_complementary_sampford_design() -> None:
    pi = np.full(10, 0.7)
    draws = 2_000
    counts = np.zeros(len(pi), dtype=np.int64)

    for seed in range(draws):
        support, _ = select_exact_k(pi, k=7, pi_hi=1.0, seed=seed)
        counts[support] += 1

    assert counts.sum() == draws * 7
    np.testing.assert_allclose(counts / draws, pi, atol=0.04)


def test_near_deterministic_feasible_design_does_not_stall() -> None:
    pi = np.asarray([0.4999999999995, 0.4999999999995, 5e-13, 5e-13])

    support, _ = select_exact_k(pi, k=2, pi_hi=0.95, seed=0)

    assert len(support) == 2
    assert np.unique(support).size == 2


@pytest.mark.parametrize(
    "pi",
    [
        [0.1, np.nan],
        [0.1, np.inf],
        [-0.1, 0.5],
        [0.1, 1.1],
    ],
)
def test_invalid_probability_values_fail(pi: list[float]) -> None:
    with pytest.raises(ValueError, match=r"finite|\[0, 1\]"):
        select_exact_k(pi, k=1, pi_hi=0.95, seed=0)


@pytest.mark.parametrize("pi_hi", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_certainty_threshold_fails(pi_hi: float) -> None:
    with pytest.raises(ValueError, match="pi_hi must be a finite value"):
        select_exact_k([0.2, 0.8], k=1, pi_hi=pi_hi, seed=0)


def test_pi_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="pi must be one-dimensional"):
        select_exact_k([[0.2, 0.8]], k=1, pi_hi=0.95, seed=0)


def test_invalid_cardinality_and_certainty_conflicts_fail() -> None:
    with pytest.raises(ValueError, match="exceeds the pool size"):
        select_exact_k([0.2, 0.8], k=3, pi_hi=0.95, seed=0)
    with pytest.raises(ValueError, match="smaller than.*certainty"):
        select_exact_k([0.99, 0.98, 0.1], k=1, pi_hi=0.95, seed=0)
    with pytest.raises(ValueError, match="k must be at least 0"):
        select_exact_k([0.2, 0.8], k=-1, pi_hi=0.95, seed=0)


def test_degenerate_boundary_mass_fails() -> None:
    with pytest.raises(ValueError, match="fewer positive pi values"):
        select_exact_k([0.0, 0.0, 0.8], k=2, pi_hi=0.95, seed=0)
    with pytest.raises(ValueError, match="greater than one"):
        select_exact_k([0.90, 0.01, 0.01], k=2, pi_hi=0.95, seed=0)
    with pytest.raises(ValueError, match="greater than one"):
        select_exact_k([0.0, 0.20, 0.80], k=2, pi_hi=1.0, seed=0)


def test_same_seed_is_identical_and_different_seeds_change_valid_support() -> None:
    pi = np.full(20, 0.35)

    first, first_receipt = select_exact_k(pi, k=7, pi_hi=0.95, seed=123)
    replay, replay_receipt = select_exact_k(pi, k=7, pi_hi=0.95, seed=123)
    different, _ = select_exact_k(pi, k=7, pi_hi=0.95, seed=124)

    np.testing.assert_array_equal(first, replay)
    assert first.tobytes() == replay.tobytes()
    assert first_receipt == replay_receipt
    assert not np.array_equal(first, different)
    assert len(different) == 7


def test_receipt_has_manifest_ready_shape() -> None:
    _, receipt = select_exact_k([0.2, 0.4, 0.6, 0.8], k=2, pi_hi=0.95, seed=11)

    assert receipt == {
        "k": 2,
        "pi_hi": 0.95,
        "seed": 11,
        "certainty_count": 0,
        "boundary_pool_size": 4,
        "design": "sampford",
    }
    assert all(isinstance(value, (int, float, str)) for value in receipt.values())


def test_empirical_inclusion_frequencies_match_pi() -> None:
    pi = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80])
    assert float(pi.sum()) == 4.0
    draws = 10_000
    counts = np.zeros(len(pi), dtype=np.int64)

    for seed in range(draws):
        support, _ = select_exact_k(pi, k=4, pi_hi=0.95, seed=seed)
        counts[support] += 1

    frequencies = counts / draws
    np.testing.assert_allclose(frequencies, pi, rtol=0.0, atol=0.02)


def test_group_ids_unique_only_seam_is_explicit() -> None:
    pi = np.asarray([0.2, 0.4, 0.6, 0.8])
    ungrouped, _ = select_exact_k(pi, k=2, pi_hi=0.95, seed=9)
    grouped, _ = select_exact_k(
        pi,
        k=2,
        pi_hi=0.95,
        seed=9,
        group_ids=np.asarray(["a", "b", "c", "d"]),
    )

    np.testing.assert_array_equal(grouped, ungrouped)
    with pytest.raises(ValueError, match="deferred to the increment-2 pool"):
        select_exact_k(
            pi,
            k=2,
            pi_hi=0.95,
            seed=9,
            group_ids=np.asarray(["a", "a", "c", "d"]),
        )
    with pytest.raises(ValueError, match="aligned with pi"):
        select_exact_k(pi, k=2, pi_hi=0.95, seed=9, group_ids=np.asarray(["a"]))
    with pytest.raises(ValueError, match="non-reflexive"):
        select_exact_k(
            pi,
            k=2,
            pi_hi=0.95,
            seed=9,
            group_ids=np.asarray([np.nan, np.nan, "c", "d"], dtype=object),
        )


def test_named_cardinality_gate_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"exact-k cardinality gate failed.*2 != k=3"):
        assert_exact_k_support([1, 2], 3, pool_size=4)
    with pytest.raises(ValueError, match="must be unique"):
        assert_exact_k_support([1, 1], 2, pool_size=4)
    with pytest.raises(ValueError, match="valid range"):
        assert_exact_k_support([1, 4], 2, pool_size=4)

    normalized = assert_exact_k_support([3, 1], 2, pool_size=4)
    np.testing.assert_array_equal(normalized, [1, 3])
