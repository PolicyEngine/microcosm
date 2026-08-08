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
