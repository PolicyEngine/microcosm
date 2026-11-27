"""Rotated holdout: deterministic, disjoint, covering — and honestly summarized."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build import rotated_folds, summarize_rotations


class TestRotatedFolds:
    def test_folds_partition_the_targets(self) -> None:
        folds = rotated_folds(103, n_folds=5, seed=7)
        union = np.concatenate(folds)
        assert len(union) == 103
        assert len(np.unique(union)) == 103  # disjoint + covering
        sizes = [len(fold) for fold in folds]
        assert max(sizes) - min(sizes) <= 1

    def test_same_seed_same_folds(self) -> None:
        a = rotated_folds(50, n_folds=4, seed=1)
        b = rotated_folds(50, n_folds=4, seed=1)
        for fold_a, fold_b in zip(a, b, strict=True):
            np.testing.assert_array_equal(fold_a, fold_b)

    def test_different_seed_different_folds(self) -> None:
        a = rotated_folds(50, n_folds=4, seed=1)
        b = rotated_folds(50, n_folds=4, seed=2)
        assert any(
            not np.array_equal(fold_a, fold_b)
            for fold_a, fold_b in zip(a, b, strict=True)
        )

    def test_bad_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="n_targets must be positive"):
            rotated_folds(0)
        with pytest.raises(ValueError, match="n_folds must be between"):
            rotated_folds(10, n_folds=1)
        with pytest.raises(ValueError, match="n_folds must be between"):
            rotated_folds(3, n_folds=4)


class TestSummary:
    def test_mean_and_worst(self) -> None:
        summary = summarize_rotations([0.1, 0.3, 0.2])
        assert summary.n_folds == 3
        assert summary.mean_holdout_loss == pytest.approx(0.2)
        assert summary.worst_holdout_loss == pytest.approx(0.3)

    def test_empty_and_nonfinite_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_rotations([])
        with pytest.raises(ValueError, match="non-finite"):
            summarize_rotations([0.1, float("nan")])


# ---------------------------------------------------------------------------
# The shared split and held estimates (microcosm#1759), against the code they
# replaced in uk_runtime.local_rowwise and microcosm.build.target_holdouts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_targets", [2, 5, 17, 103])
@pytest.mark.parametrize("seed", [1, 20260529])
def test_fold_splits_are_the_folds_the_uk_holdout_built(n_targets, seed) -> None:
    from microcosm.build.holdout import fold_splits

    n_folds = min(5, n_targets)
    every_row = np.arange(n_targets, dtype=np.int64)
    old = [
        (np.setdiff1d(every_row, held, assume_unique=True), held)
        for held in rotated_folds(n_targets, n_folds=n_folds, seed=seed)
    ]
    new = fold_splits(n_targets, n_folds=n_folds, seed=seed)
    assert [split.name for split in new] == [f"fold_{i}" for i in range(n_folds)]
    for (old_train, old_held), split in zip(old, new, strict=True):
        assert split.train_rows.dtype == old_train.dtype
        assert split.held_rows.dtype == old_held.dtype
        assert split.train_rows.tobytes() == old_train.tobytes()
        assert split.held_rows.tobytes() == old_held.tobytes()


def _sparse(seed: int, shape, dtype, kind):

    rng = np.random.default_rng(seed)
    dense = rng.normal(0.0, 3.0, size=shape)
    dense[rng.uniform(size=shape) < 0.7] = 0.0
    return kind(dense.astype(dtype))


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("weight_dtype", [np.float64, np.float32])
def test_held_estimates_match_the_uk_expression(seed, weight_dtype) -> None:
    from scipy import sparse

    from microcosm.build.holdout import held_estimates

    rng = np.random.default_rng(seed + 100)
    matrix = _sparse(seed, (40, 60), np.float64, sparse.csr_matrix)
    rows = np.sort(rng.choice(40, size=9, replace=False)).astype(np.int64)
    weights = rng.uniform(0.0, 5.0, size=60).astype(weight_dtype)
    for columns in (np.arange(60), np.sort(rng.choice(60, size=25, replace=False))):
        used = weights[: len(columns)]
        old = np.asarray(matrix[rows][:, columns] @ used, dtype=np.float64).reshape(-1)
        new = held_estimates(matrix, rows, used, columns=columns)
        assert new.dtype == old.dtype
        assert new.tobytes() == old.tobytes()


def test_held_estimates_widen_a_float32_matrix_before_the_product() -> None:
    # The premise of the UK equality above: the UK problem matrix is float64.
    # For a float32 matrix with float32 weights the replaced UK expression
    # multiplied in float32 and the shared function multiplies in float64, so
    # the bytes differ (the review of microcosm#1755). No build passes that.
    from scipy import sparse

    from microcosm.build.holdout import held_estimates

    matrix = _sparse(3, (40, 60), np.float32, sparse.csr_matrix)
    rows = np.arange(0, 40, 4, dtype=np.int64)
    weights = np.random.default_rng(7).uniform(0.0, 5.0, size=60).astype(np.float32)
    in_float32 = np.asarray(matrix[rows] @ weights, dtype=np.float64).reshape(-1)
    in_float64 = np.asarray(
        matrix[rows].astype(np.float64) @ weights.astype(np.float64)
    ).reshape(-1)
    new = held_estimates(matrix, rows, weights)
    assert new.tobytes() == in_float64.tobytes()
    assert new.tobytes() != in_float32.tobytes()
    np.testing.assert_allclose(new, in_float32, rtol=1e-5)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("weight_dtype", [np.float64, np.float32])
def test_held_estimates_match_the_us_expression(seed, weight_dtype) -> None:
    from scipy import sparse

    from microcosm.build.holdout import held_estimates

    rng = np.random.default_rng(seed + 200)
    matrix = _sparse(seed, (30, 45), np.float32, sparse.csr_array)
    rows = sorted(int(row) for row in rng.choice(30, size=7, replace=False))
    weights = rng.uniform(0.1, 4.0, size=45).astype(weight_dtype)
    held = sparse.csr_array(matrix[rows]).astype(np.float64)
    old = held @ np.asarray(weights, dtype=np.float64)
    new = held_estimates(matrix, rows, weights)
    assert new.dtype == old.dtype
    assert new.tobytes() == old.tobytes()


def test_role_split_holds_the_roles() -> None:
    from microcosm.build.target_holdouts import (
        holdout_rows,
        role_split,
        train_rows,
    )

    records = [{"role": role} for role in ("train", "holdout", "train", "holdout")]
    split = role_split(records)
    assert split.name == "roles"
    assert split.train_rows.tolist() == train_rows(records) == [0, 2]
    assert split.held_rows.tolist() == holdout_rows(records) == [1, 3]
    assert split.train_rows.dtype == split.held_rows.dtype == np.int64
