"""Contribution-informed L0 initialization (microcosm#346, #355)."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GateInitialization:
    """Aligned open probabilities and records excluded from gate pruning."""

    probabilities: np.ndarray
    protected: np.ndarray


def contribution_initialization(
    matrix: sparse.sparray,
    weights: np.ndarray,
    targets: np.ndarray,
) -> GateInitialization:
    """Initialize search from maximum absolute target-contribution share.

    Protect the largest absolute weighted carrier of each nonzero target
    (first column breaks ties deterministically). The smooth bounded prior
    uses the median positive share as its scale. It only initializes L0;
    neither these scores nor their ranks select the final support. Zero
    targets use a unit denominator, matching the default calibration scale.
    Work and storage are sparse in the target matrix, never targets x pool.
    """
    matrix = sparse.csr_array(matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    weights = np.asarray(weights, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if weights.shape != (matrix.shape[1],) or targets.shape != (matrix.shape[0],):
        raise ValueError("matrix, weights and targets must align.")
    if (
        not np.isfinite(matrix.data).all()
        or not np.isfinite(weights).all()
        or not np.isfinite(targets).all()
        or (weights <= 0).any()
    ):
        raise ValueError(
            "contribution initialization requires finite inputs and positive weights."
        )
    scores = np.zeros(len(weights))
    protected = np.zeros(len(weights), dtype=bool)
    for row, target in enumerate(targets):
        start, stop = matrix.indptr[row : row + 2]
        indices = matrix.indices[start:stop]
        contributions = np.abs(matrix.data[start:stop]) * weights[indices]
        if not len(indices):
            if target != 0:
                raise ValueError(f"nonzero target at row {row} has no support.")
            continue
        shares = contributions / (abs(target) if target != 0 else 1.0)
        np.maximum.at(scores, indices, shares)
        if target != 0:
            protected[indices[np.argmax(contributions)]] = True
    positive = scores[scores > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    probabilities = 0.1 + 0.8 * scores / (scores + scale)
    return GateInitialization(probabilities, protected)
