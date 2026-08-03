"""Exact-cardinality calibration seam for the US release ladder.

The multispine pool supplies one original importance-weighted frame.  A
non-census ladder point first learns hard-concrete open probabilities on that
full frame, draws a seeded fixed-size Sampford support, and refits ordinary
calibration from the normalized Horvitz--Thompson ``w / q`` baseline.  The
full-pool point skips the stochastic draw but still runs ordinary calibration
and emits the same six-scalar receipt shape with ``design="full-pool"``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from populace.calibrate import (
    CONSERVE_MASS,
    CalibrationResult,
    L0RefitResult,
    TargetSet,
    assert_exact_k_support,
    calibrate,
    effective_sample_size,
    refit_l0_selection,
    select_exact_k,
)
from populace.frame import Frame

__all__ = ["ExactKLadderCalibration", "calibrate_exact_k_ladder"]


@dataclass(frozen=True)
class ExactKLadderCalibration:
    """One exact-k selection/refit result and its release-sized receipts."""

    result: CalibrationResult | L0RefitResult
    support: np.ndarray
    selected_inclusion_probabilities: np.ndarray
    selection_receipt: dict[str, int | float | str]
    refit_baseline_diagnostics: dict[str, int | float | str]


def calibrate_exact_k_ladder(
    frame: Frame,
    targets: TargetSet,
    *,
    k: int,
    pi_hi: float,
    seed: int,
    weight_entity: str = "household",
    epochs: int = 256,
    refit_epochs: int | None = None,
    learning_rate: float = 0.02,
    refit_learning_rate: float | None = None,
    mass: str = CONSERVE_MASS,
    max_weight_ratio: float | None = None,
    l0_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    refit_l2_lambda: float | None = None,
    l2_anchor: str = "initial",
    refit_l2_anchor: str | None = None,
    init_mean: float = 0.999,
    temperature: float = 0.25,
    budget_iters: int = 10,
    target_loss_weights: np.ndarray | None = None,
    target_loss_scales: np.ndarray | None = None,
    target_loss_cap: float = 10.0,
    warm_start_weights: np.ndarray | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> ExactKLadderCalibration:
    """Calibrate one exact-count point from an original weighted pool.

    ``k == N`` is an identity support selection followed by ordinary
    calibration.  ``k < N`` uses the public exact-k API and passes its aligned
    marginal inclusion probabilities into the public explicit-support refit.
    That refit is the #585 authority for subsetting the *original* frame and
    constructing the normalized ``w / q`` baseline.
    """

    pool_size = int(frame.n(weight_entity))
    target = _nonnegative_integer(k, name="k")
    if target == 0:
        raise ValueError("k must be positive for a release dataset.")
    if target > pool_size:
        raise ValueError(
            f"k={target} exceeds the pool size {pool_size}; ladder selection "
            "never clamps the requested cardinality."
        )
    random_seed = _nonnegative_integer(seed, name="seed")
    certainty_threshold = _probability(pi_hi, name="pi_hi")
    refit_steps = epochs if refit_epochs is None else refit_epochs
    refit_rate = learning_rate if refit_learning_rate is None else refit_learning_rate
    refit_penalty = l2_lambda if refit_l2_lambda is None else refit_l2_lambda
    refit_anchor = l2_anchor if refit_l2_anchor is None else refit_l2_anchor

    if target == pool_size:
        support = assert_exact_k_support(
            np.arange(pool_size, dtype=np.int64),
            target,
            pool_size=pool_size,
        )
        selected_q = np.ones(pool_size, dtype=np.float64)
        receipt: dict[str, int | float | str] = {
            "k": target,
            "pi_hi": certainty_threshold,
            "seed": random_seed,
            "certainty_count": pool_size,
            "boundary_pool_size": 0,
            "design": "full-pool",
        }
        result: CalibrationResult | L0RefitResult = calibrate(
            frame,
            targets,
            weight_entity=weight_entity,
            epochs=refit_steps,
            learning_rate=refit_rate,
            mass=mass,
            max_weight_ratio=max_weight_ratio,
            l0_lambda=0.0,
            l2_lambda=refit_penalty,
            l2_anchor=refit_anchor,
            init_mean=init_mean,
            temperature=temperature,
            budget_iters=budget_iters,
            seed=random_seed,
            target_loss_weights=target_loss_weights,
            target_loss_scales=target_loss_scales,
            target_loss_cap=target_loss_cap,
            warm_start_weights=warm_start_weights,
            progress_callback=_phase_callback(progress_callback, "full_pool_refit"),
        )
    else:
        if not math.isfinite(l0_lambda) or l0_lambda <= 0.0:
            raise ValueError(
                "k<N ladder calibration requires a positive finite l0_lambda "
                "to learn selection probabilities."
            )
        selection = calibrate(
            frame,
            targets,
            weight_entity=weight_entity,
            epochs=epochs,
            learning_rate=learning_rate,
            mass=mass,
            max_weight_ratio=max_weight_ratio,
            l0_lambda=l0_lambda,
            l2_lambda=l2_lambda,
            l2_anchor=l2_anchor,
            init_mean=init_mean,
            temperature=temperature,
            budget_iters=budget_iters,
            seed=random_seed,
            target_loss_weights=target_loss_weights,
            target_loss_scales=target_loss_scales,
            target_loss_cap=target_loss_cap,
            warm_start_weights=warm_start_weights,
            progress_callback=_phase_callback(progress_callback, "l0_selection"),
        )
        pi = selection.gate_open_probabilities
        if pi is None:  # pragma: no cover - positive-L0 solver invariant
            raise RuntimeError("L0 selection returned no gate open probabilities.")
        support, receipt, selected_q = select_exact_k(
            pi,
            k=target,
            pi_hi=certainty_threshold,
            seed=random_seed,
        )
        # Keep the public named gate immediately adjacent to the release refit,
        # in addition to the two gates enforced inside refit_l0_selection.
        support = assert_exact_k_support(support, target, pool_size=pool_size)
        result = refit_l0_selection(
            frame,
            targets,
            selection,
            weight_entity=weight_entity,
            support=support,
            k=target,
            support_inclusion_probabilities=selected_q,
            epochs=refit_steps,
            learning_rate=refit_rate,
            mass=mass,
            max_weight_ratio=max_weight_ratio,
            l2_lambda=refit_penalty,
            l2_anchor=refit_anchor,
            init_mean=init_mean,
            temperature=temperature,
            budget_iters=budget_iters,
            seed=random_seed,
            target_loss_weights=target_loss_weights,
            target_loss_scales=target_loss_scales,
            target_loss_cap=target_loss_cap,
            progress_callback=_phase_callback(progress_callback, "exact_k_refit"),
        )

    diagnostics = _refit_baseline_diagnostics(
        frame,
        result,
        weight_entity=weight_entity,
        support=support,
        selected_inclusion_probabilities=selected_q,
        full_pool=target == pool_size,
    )
    return ExactKLadderCalibration(
        result=result,
        support=support,
        selected_inclusion_probabilities=selected_q,
        selection_receipt=receipt,
        refit_baseline_diagnostics=diagnostics,
    )


def _refit_baseline_diagnostics(
    frame: Frame,
    result: CalibrationResult | L0RefitResult,
    *,
    weight_entity: str,
    support: np.ndarray,
    selected_inclusion_probabilities: np.ndarray,
    full_pool: bool,
) -> dict[str, int | float | str]:
    original_weights = frame.resolve_weights(weight_entity)
    source = np.asarray(original_weights.values, dtype=np.float64)
    selected_source = source[support]
    q = np.asarray(selected_inclusion_probabilities, dtype=np.float64)
    unnormalized = selected_source / q
    normalization_factor = float(source.sum() / unnormalized.sum())
    expected_baseline = unnormalized * normalization_factor
    observed_baseline = np.asarray(result.initial_weights, dtype=np.float64)
    if not np.allclose(
        observed_baseline,
        expected_baseline,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Exact-k refit baseline no longer equals the normalized original-frame "
            "w/q projection."
        )
    return {
        "method": (
            "full_pool_original_frame_weights"
            if full_pool
            else "normalized_horvitz_thompson_w_over_q"
        ),
        "source_weight_kind": original_weights.kind.value,
        "pool_size": int(source.size),
        "selected_size": int(support.size),
        "pool_weight_total": float(source.sum()),
        "selected_original_weight_total": float(selected_source.sum()),
        "unnormalized_ht_weight_total": float(unnormalized.sum()),
        "normalization_factor": normalization_factor,
        "refit_baseline_weight_total": float(observed_baseline.sum()),
        "refit_baseline_minimum": float(observed_baseline.min()),
        "refit_baseline_maximum": float(observed_baseline.max()),
        "refit_baseline_effective_sample_size": float(
            effective_sample_size(observed_baseline)
        ),
        "inclusion_probability_minimum": float(q.min()),
        "inclusion_probability_maximum": float(q.max()),
    }


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return parsed


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite value in [0, 1], got {value!r}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a finite value in [0, 1], got {value!r}."
        ) from None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1], got {value!r}.")
    return parsed


def _phase_callback(
    callback: Callable[[dict[str, object]], None] | None,
    phase: str,
) -> Callable[[dict[str, object]], None] | None:
    if callback is None:
        return None

    def with_phase(event: dict[str, object]) -> None:
        callback({"phase": phase, **event})

    return with_phase
