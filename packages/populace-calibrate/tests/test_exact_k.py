"""Exact-cardinality Sampford selection contracts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import populace.calibrate.exact_k as exact_k_module
from populace.calibrate import assert_exact_k_support, select_exact_k


def _set_dp_cell_limits(monkeypatch: pytest.MonkeyPatch, cells: int) -> None:
    monkeypatch.setattr(exact_k_module, "_SAMPFORD_DP_ALWAYS_MAX_CELLS", cells)
    monkeypatch.setattr(exact_k_module, "_SAMPFORD_DP_MAX_CELLS", cells)


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


def test_ulp_scale_take_alls_are_extracted_before_residual_correction() -> None:
    support, receipt = select_exact_k(
        [0.2, 0.2, 0.4, 0.4],
        k=3,
        pi_hi=1.0,
        seed=0,
    )

    assert {2, 3}.issubset(support.tolist())
    assert len(set(support).intersection({0, 1})) == 1
    assert receipt["certainty_count"] == 0
    assert receipt["boundary_pool_size"] == 4


def test_majority_draw_uses_nonuniform_nontrivial_complement_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pi = np.linspace(0.42, 0.98, 40)
    assert float(pi.sum()) == 28.0
    draws = 4_000
    counts = np.zeros(len(pi), dtype=np.int64)
    core_calls: list[tuple[int, int]] = []
    original_core = exact_k_module._sampford_core

    def track_core(
        probabilities: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        core_calls.append((len(probabilities), sample_size))
        return original_core(probabilities, sample_size, rng)

    _set_dp_cell_limits(monkeypatch, 0)
    monkeypatch.setattr(exact_k_module, "_sampford_core", track_core)

    for seed in range(draws):
        support, _ = select_exact_k(pi, k=28, pi_hi=1.0, seed=seed)
        counts[support] += 1

    frequencies = counts / draws
    assert core_calls[:2] == [(40, 28), (40, 12)]
    assert counts.sum() == draws * 28
    np.testing.assert_allclose(frequencies, pi, rtol=0.0, atol=0.03)
    # An SRSWOR implementation would give every record inclusion probability .7.
    assert float(np.max(np.abs(frequencies - 0.7))) > 0.20


def test_near_deterministic_feasible_design_does_not_stall() -> None:
    pi = np.asarray([0.4999999999995, 0.4999999999995, 5e-13, 5e-13])

    support, _ = select_exact_k(pi, k=2, pi_hi=0.95, seed=0)

    assert len(support) == 2
    assert np.unique(support).size == 2


def test_concentrated_design_just_above_old_dp_cutoff_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_size = 2_001
    sample_size = 999
    target_probabilities = np.concatenate(
        (
            np.full(sample_size, 0.999999),
            np.full(
                pool_size - sample_size,
                (sample_size - sample_size * 0.999999) / (pool_size - sample_size),
            ),
        )
    )
    pi = 0.9 * target_probabilities
    dp_calls: list[tuple[int, int]] = []
    original_dp = exact_k_module._sampford_dynamic_programming

    def recording_dp(
        probabilities: np.ndarray,
        draw_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dp_calls.append((len(probabilities), draw_size))
        return original_dp(probabilities, draw_size, rng)

    monkeypatch.setattr(
        exact_k_module,
        "_sampford_dynamic_programming",
        recording_dp,
    )

    support, receipt = select_exact_k(
        pi,
        k=sample_size,
        pi_hi=0.95,
        seed=0,
    )
    replay, _ = select_exact_k(
        pi,
        k=sample_size,
        pi_hi=0.95,
        seed=0,
    )

    np.testing.assert_array_equal(
        support,
        assert_exact_k_support(support, sample_size, pool_size=pool_size),
    )
    np.testing.assert_array_equal(replay, support)
    assert receipt["boundary_pool_size"] == pool_size
    assert dp_calls == [(pool_size, sample_size), (pool_size, sample_size)]


def test_realistic_ladder_shaped_large_design_completes() -> None:
    near_one = np.full(5_000, 1.0 - 1e-6)
    near_zero = np.full(5_000, 1e-6)
    ladder = np.linspace(0.001, 0.149, 40_000)
    pi = np.concatenate((near_one, ladder, near_zero))
    assert len(pi) == 50_000
    np.testing.assert_allclose(
        np.sum(pi, dtype=np.float64),
        8_000.0,
        rtol=0.0,
        atol=1e-9,
    )

    support, receipt = select_exact_k(pi, k=8_000, pi_hi=1.0, seed=0)

    np.testing.assert_array_equal(
        support,
        assert_exact_k_support(support, 8_000, pool_size=len(pi)),
    )
    assert receipt["boundary_pool_size"] == len(pi)


def test_forced_rejection_path_returns_exact_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pi = np.asarray(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.70]
    )
    assert float(pi.sum()) == 4.0
    _set_dp_cell_limits(monkeypatch, 0)

    support, _ = select_exact_k(pi, k=4, pi_hi=1.0, seed=73)

    np.testing.assert_array_equal(
        support,
        assert_exact_k_support(support, 4, pool_size=len(pi)),
    )


def test_no_viable_path_error_names_rejection_and_dp_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pi = np.asarray(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.70]
    )
    _set_dp_cell_limits(monkeypatch, 0)
    monkeypatch.setattr(exact_k_module, "_SAMPFORD_MAX_REJECTION_ATTEMPTS", 0)

    with pytest.raises(
        RuntimeError,
        match=r"rejection.*budget.*dynamic programming.*memory bound",
    ):
        select_exact_k(pi, k=4, pi_hi=1.0, seed=73)


def test_dp_rejection_and_complement_paths_agree_on_inclusion_frequencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probabilities = np.asarray(
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.70]
    )
    sample_size = 4
    draws = 3_000

    def inclusion_frequencies(
        *,
        seed_offset: int,
        complement: bool = False,
    ) -> np.ndarray:
        counts = np.zeros(len(probabilities), dtype=np.int64)
        for draw in range(draws):
            seed = seed_offset + draw
            if complement:
                selected, _ = select_exact_k(
                    1.0 - probabilities,
                    k=len(probabilities) - sample_size,
                    pi_hi=1.0,
                    seed=seed,
                )
                selected_mask = np.ones(len(probabilities), dtype=bool)
                selected_mask[selected] = False
                selected = np.flatnonzero(selected_mask)
            else:
                selected, _ = select_exact_k(
                    probabilities,
                    k=sample_size,
                    pi_hi=1.0,
                    seed=seed,
                )
            counts[selected] += 1
        return counts / draws

    _set_dp_cell_limits(monkeypatch, 10_000)
    dp_frequencies = inclusion_frequencies(seed_offset=0)
    _set_dp_cell_limits(monkeypatch, 0)
    rejection_frequencies = inclusion_frequencies(seed_offset=10_000)
    complement_frequencies = inclusion_frequencies(
        seed_offset=20_000,
        complement=True,
    )

    for frequencies in (
        dp_frequencies,
        rejection_frequencies,
        complement_frequencies,
    ):
        np.testing.assert_allclose(
            frequencies,
            probabilities,
            rtol=0.0,
            atol=0.025,
        )
    np.testing.assert_allclose(
        dp_frequencies,
        rejection_frequencies,
        rtol=0.0,
        atol=0.04,
    )
    np.testing.assert_allclose(
        dp_frequencies,
        complement_frequencies,
        rtol=0.0,
        atol=0.04,
    )


def test_enumerated_subsets_match_sampford_joint_law() -> None:
    probabilities = np.asarray([0.15, 0.35, 0.55, 0.65, 0.60, 0.70])
    sample_size = 3
    subsets = list(itertools.combinations(range(len(probabilities)), sample_size))
    subset_positions = {subset: position for position, subset in enumerate(subsets)}
    odds = probabilities / (1.0 - probabilities)
    masses = np.asarray(
        [
            np.sum(1.0 - probabilities[list(subset)], dtype=np.float64)
            * np.prod(odds[list(subset)], dtype=np.float64)
            for subset in subsets
        ]
    )
    expected_probabilities = masses / masses.sum()
    draws = 50_000
    observed = np.zeros(len(subsets), dtype=np.int64)

    for seed in range(draws):
        support, _ = select_exact_k(
            probabilities,
            k=sample_size,
            pi_hi=1.0,
            seed=seed,
        )
        observed[subset_positions[tuple(support.tolist())]] += 1

    expected = draws * expected_probabilities
    chi_squared = float(np.sum((observed - expected) ** 2 / expected))
    assert float(expected.min()) > 100.0
    assert chi_squared == pytest.approx(13.24, abs=0.01)
    assert chi_squared < 43.82  # chi-square 0.999 quantile with 19 df

    # This probability transfer preserves every first-order inclusion
    # probability but changes the joint law. The chi-square gate rejects it,
    # showing why marginal-frequency checks alone are insufficient.
    wrong_probabilities = expected_probabilities.copy()
    subtract = ((0, 1, 3), (2, 4, 5))
    add = ((0, 1, 2), (3, 4, 5))
    delta = 0.25 * min(
        wrong_probabilities[subset_positions[subset]] for subset in subtract
    )
    for subset in subtract:
        wrong_probabilities[subset_positions[subset]] -= delta
    for subset in add:
        wrong_probabilities[subset_positions[subset]] += delta
    incidence = np.asarray(
        [
            [index in subset for index in range(len(probabilities))]
            for subset in subsets
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        wrong_probabilities @ incidence,
        expected_probabilities @ incidence,
        rtol=0.0,
        atol=1e-15,
    )
    wrong_chi_squared = draws * float(
        np.sum(
            (wrong_probabilities - expected_probabilities) ** 2 / expected_probabilities
        )
    )
    assert wrong_chi_squared > 43.82


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
    np.testing.assert_array_equal(first, [1, 2, 3, 4, 11, 13, 17])
    assert first.dtype.str == "<i8"
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


@pytest.mark.parametrize(
    "group_ids",
    [
        pytest.param(
            np.asarray([np.nan, 1.0, 2.0, 3.0]),
            id="float-nan",
        ),
        pytest.param(
            np.asarray([np.nan + 0.0j, 1.0, 2.0, 3.0], dtype=np.complex128),
            id="complex-nan",
        ),
        pytest.param(
            np.asarray(
                ["NaT", "2020-01-01", "2020-01-02", "2020-01-03"], dtype="M8[D]"
            ),
            id="datetime-nat",
        ),
        pytest.param(
            np.asarray(["NaT", 1, 2, 3], dtype="m8[D]"),
            id="timedelta-nat",
        ),
    ],
)
def test_group_ids_reject_missing_values_for_nonobject_dtypes(
    group_ids: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="missing"):
        select_exact_k(
            [0.2, 0.4, 0.6, 0.8],
            k=2,
            pi_hi=0.95,
            seed=9,
            group_ids=group_ids,
        )


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
