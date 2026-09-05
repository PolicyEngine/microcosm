"""Exact household-count UK candidates on a fixed, materialized target surface."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from microcosm.calibrate import (
    CalibrationResult,
    TargetSet,
    assert_exact_k_support,
    calibrate,
    refit_l0_selection,
    select_exact_k,
)
from microcosm.calibrate.initialization import contribution_initialization
from microcosm.frame import Frame


@dataclass(frozen=True)
class UKDatasetSize:
    """A compact refit with its full-pool row identities and selection evidence."""

    result: CalibrationResult
    support: np.ndarray
    receipt: dict[str, object]


def refit_uk_dataset_size(
    frame: Frame,
    dense: CalibrationResult,
    *,
    households: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> UKDatasetSize:
    """Run informed L0, a fixed-size draw, and refit under the dense doctrine.

    Freeze the already compiled household contributions, including engine
    measures that depend on the full population. Re-evaluating those formulas
    on a subset would change the target system. Target values, order, loss
    weights, cap and stretch bound remain those of the dense solve.
    """
    n = frame.n("household")
    if (
        isinstance(households, bool)
        or not isinstance(households, int)
        or not 0 < households <= n
    ):
        raise ValueError(f"households must be an integer in [1, {n}]; never clamped.")
    if (
        dense.skipped
        or len(dense.weights) != n
        or dense.l0_lambda != 0
        or dense.weight_entity != "household"
        or not np.array_equal(
            frame.table("household")["household_id"].to_numpy(),
            dense.frame.table("household")["household_id"].to_numpy(),
        )
        or not np.array_equal(
            frame.weights_for("household").values, dense.initial_weights
        )
    ):
        raise ValueError(
            "size selection requires an aligned, fully compiled dense solve."
        )
    if households == n:
        return UKDatasetSize(
            dense,
            np.arange(n),
            {
                "method": "full_pool",
                "requested_households": n,
                "realized_households": n,
                "pool_households": n,
                "seed": seed,
            },
        )
    problem = dense.problem
    # Input weights are the pool design, not the concentrated dense weights.
    init = contribution_initialization(
        problem.matrix, dense.initial_weights, problem.target_vector
    )
    if int(init.protected.sum()) > households:
        raise ValueError(
            f"requested {households} households cannot retain {int(init.protected.sum())} protected target carriers."
        )
    common = dict(
        weight_entity="household",
        epochs=epochs,
        learning_rate=learning_rate,
        mass=dense.options["mass"],
        max_weight_ratio=dense.options["max_weight_ratio"],
        seed=seed,
        target_loss_weights=dense.target_loss_weights,
        target_loss_scales=dense.target_loss_scales,
        target_loss_cap=dense.target_loss_cap,
    )
    selection = calibrate(
        frame,
        TargetSet(problem.targets),
        target_records=households,
        gate_initialization=init,
        mass_reason=dense.options["mass_reason"],
        **common,
    )
    probabilities = selection.gate_open_probabilities
    if probabilities is None:
        raise RuntimeError("informed L0 returned no selection probabilities.")
    # Exact-one certainties are protected by the gates themselves; no top-k
    # ranking or post-hoc promotion of learned boundary scores is performed.
    support, sampling, q = select_exact_k(
        probabilities, households, pi_hi=1.0, seed=seed
    )
    support = assert_exact_k_support(support, households, pool_size=n)
    if not np.isin(np.flatnonzero(init.protected), support).all():
        raise RuntimeError("exact-count selection lost a protected carrier.")
    frozen = _frozen_targets(frame, dense, support)
    refit = refit_l0_selection(
        frame,
        frozen,
        selection,
        support=support,
        k=households,
        support_inclusion_probabilities=q,
        mass_reason=dense.options["mass_reason"],
        **common,
    ).refit
    if (
        refit.skipped
        or refit.frame.n("household") != households
        or (refit.weights <= 0).any()
    ):
        raise RuntimeError("compact refit lost targets or positive household support.")
    if not np.array_equal(refit.problem.target_vector, problem.target_vector):
        raise RuntimeError("compact refit changed target values.")
    errors_dense = np.asarray([d.final_estimate for d in dense.diagnostics])
    errors_small = np.asarray([d.final_estimate for d in refit.diagnostics])
    return UKDatasetSize(
        refit,
        support,
        {
            "method": "contribution_informed_l0_exact_count_refit",
            "requested_households": households,
            "realized_households": households,
            "pool_households": n,
            "seed": seed,
            "protected_carriers": int(init.protected.sum()),
            "selection_receipt": sampling,
            "selection_l0_lambda": selection.l0_lambda,
            "selection_epochs": epochs,
            "refit_epochs": epochs,
            "pool_row_indices": support.tolist(),
            "inclusion_probabilities": q.tolist(),
            "refit_baseline": "normalized_horvitz_thompson_w_over_q",
            "stretch_reference": "normalized_horvitz_thompson_w_over_q",
            "dense_loss": dense.final_loss,
            "compact_loss": refit.final_loss,
            "max_target_scaled_change": float(
                np.max(np.abs(errors_small - errors_dense) / dense.target_loss_scales)
            ),
            "certification": "candidate_only_pending_matched_comparison_and_promotion_scorecard",
        },
    )


def _frozen_targets(
    frame: Frame, dense: CalibrationResult, support: np.ndarray
) -> TargetSet:
    """Bind sparse contribution rows to selected ids, with one shared id join."""
    ids = pd.Index(frame.table("household")["household_id"].iloc[support])
    matrix = dense.problem.matrix[:, support].tocsr()
    cached_ids = None
    cached_positions = None

    def positions(subset):
        nonlocal cached_ids, cached_positions
        current = subset.table("household")["household_id"]
        if cached_ids is None or not current.equals(cached_ids):
            cached_positions = ids.get_indexer(current)
            if (cached_positions < 0).any():
                raise ValueError("frozen target requested unknown household ids.")
            cached_ids = current.copy()
        return cached_positions

    def measure(row):
        def values(subset):
            return np.asarray(matrix[[row], :].toarray()).reshape(-1)[positions(subset)]

        return values

    return TargetSet(
        [
            replace(target, entity="household", measure=measure(row), filter=None)
            for row, target in enumerate(dense.problem.targets)
        ]
    )
