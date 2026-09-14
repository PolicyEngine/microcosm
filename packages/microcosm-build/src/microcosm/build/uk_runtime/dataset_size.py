"""Exact household-count UK candidates on a fixed, materialized target surface."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from microcosm.calibrate import (
    CalibrationResult,
    TargetSet,
    assert_exact_k_support,
    calibrate,
    exact_k_design_feasibility,
    refit_l0_selection,
    select_exact_k,
)
from microcosm.calibrate.initialization import contribution_initialization
from microcosm.calibrate.solve import BUDGET_BASIS_OPEN_PROBABILITY_MASS
from microcosm.frame import Frame


@dataclass(frozen=True)
class UKDatasetSize:
    """A compact refit with its full-pool row identities and selection evidence."""

    result: CalibrationResult
    support: np.ndarray
    receipt: dict[str, object]


@dataclass(frozen=True)
class UKSizeSelection:
    """The informed L0 search a size draw starts from.

    Kept apart from the draw and the refit so a run can checkpoint the dense
    solve and this search before the exact-count draw: a draw refusal then
    costs a re-draw, not the hours the pool solve took (microcosm#355, S2
    2026-09-08). ``search_pi_hi`` is the certainty threshold the budget
    search stopped on; a later draw may use another threshold, recorded
    beside it in the size receipt.
    """

    selection: CalibrationResult
    protected: np.ndarray
    households: int
    epochs: int
    learning_rate: float
    seed: int
    search_pi_hi: float


@dataclass(frozen=True)
class UKSizeDraw:
    """An executed exact-count draw, independently reusable by the refit."""

    support: np.ndarray
    sampling: dict[str, object]
    inclusion_probabilities: np.ndarray
    feasibility: dict[str, object]
    seed: int
    pi_hi: float
    probabilities_sha256: str


def _probability_digest(probabilities: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(probabilities, dtype="<f8").tobytes()).hexdigest()


def draw_uk_dataset_size(
    frame: Frame,
    dense: CalibrationResult,
    *,
    selection: UKSizeSelection,
    households: int,
    seed: int,
    pi_hi: float = 1.0,
) -> UKSizeDraw:
    """Execute only the existing exact-count draw, without search or refit."""
    n = _check_size_inputs(frame, dense, households)
    pi_hi = _check_pi_hi(pi_hi)
    probabilities = selection.selection.gate_open_probabilities
    if (
        selection.households != households
        or selection.seed != seed
        or probabilities is None
        or len(probabilities) != n
        or len(selection.protected) != n
    ):
        raise ValueError("Exact-count draw requires its aligned selection and seed.")
    search_result = selection.selection
    feasibility = selection_feasibility(
        probabilities,
        households,
        protected=selection.protected,
        n_nonzero=int(search_result.n_nonzero),
        l0_lambda=float(search_result.l0_lambda),
        requested_pi_hi=pi_hi,
        budget_search=search_result.options.get("budget_search"),
        search_pi_hi=selection.search_pi_hi,
    )
    try:
        support, sampling, q = select_exact_k(
            probabilities, households, pi_hi=pi_hi, seed=seed
        )
    except ValueError as error:
        raise ValueError(
            f"{error} Selection feasibility (requested pi_hi={pi_hi:g}): "
            f"{json.dumps(feasibility, sort_keys=True)}"
        ) from error
    support = assert_exact_k_support(support, households, pool_size=n)
    if not np.isin(np.flatnonzero(selection.protected), support).all():
        raise RuntimeError("exact-count selection lost a protected carrier.")
    return UKSizeDraw(
        support,
        sampling,
        q,
        feasibility,
        seed,
        pi_hi,
        _probability_digest(probabilities),
    )


ProgressCallback = Callable[[dict[str, object]], None]


def _phased(
    progress_callback: ProgressCallback | None, phase: str
) -> ProgressCallback | None:
    """Tag every progress event with the size stage it belongs to."""
    if progress_callback is None:
        return None

    def callback(event: dict[str, object]) -> None:
        progress_callback({"phase": phase, **event})

    return callback


def _check_size_inputs(frame: Frame, dense: CalibrationResult, households: int) -> int:
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
    return n


def _check_pi_hi(pi_hi: object) -> float:
    if not isinstance(pi_hi, float | int) or isinstance(pi_hi, bool):
        raise ValueError("pi_hi must be a number in (0, 1].")
    value = float(pi_hi)
    if not (0.0 < value <= 1.0):
        raise ValueError("pi_hi must be a number in (0, 1].")
    return value


def _solver_common(
    dense: CalibrationResult, *, epochs: int, learning_rate: float, seed: int
) -> dict[str, Any]:
    return dict(
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


def select_uk_dataset_size(
    frame: Frame,
    dense: CalibrationResult,
    *,
    households: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    pi_hi: float = 1.0,
    progress_callback: ProgressCallback | None = None,
) -> UKSizeSelection:
    """Run the informed L0 budget search for an exact-count draw of ``households``.

    Refuses by name any nonzero target no pool household supports, protects
    each remaining target's largest carrier, and searches the L0 penalty on
    the gates' open-probability mass until a draw of ``households`` at
    ``pi_hi`` is feasible on the learned probabilities (or the probe budget
    is spent; then the draw refuses with the measurement).
    """
    n = _check_size_inputs(frame, dense, households)
    pi_hi = _check_pi_hi(pi_hi)
    if households == n:
        raise ValueError("a full-pool size needs no selection.")
    problem = dense.problem
    unsupported = unsupported_nonzero_targets(problem)
    if unsupported:
        shown = ", ".join(unsupported[:12])
        more = "" if len(unsupported) <= 12 else f" (+{len(unsupported) - 12} more)"
        raise ValueError(
            f"{len(unsupported)} nonzero target(s) have no supporting household in "
            f"the pool, so no selection can carry them: {shown}{more}. The dense "
            "solve tolerates such rows as misses; a size selection refuses them "
            "by name so the binding defect is fixed or the row is excluded with "
            "a signed reason, never selected around."
        )
    # Input weights are the pool design, not the concentrated dense weights.
    init = contribution_initialization(
        problem.matrix, dense.initial_weights, problem.target_vector
    )
    if int(init.protected.sum()) > households:
        raise ValueError(
            f"requested {households} households cannot retain {int(init.protected.sum())} protected target carriers."
        )
    # The exact-count draw can only draw from the gates' open-probability
    # mass, so the budget search targets that mass, not the count of
    # not-fully-closed gates (microcosm#355 ruling 2026-09-08), and it stops
    # only on a probe whose gates make the draw at ``pi_hi`` feasible: the
    # draw's inequality is one-sided, the budget band is not, and S2 (2026-09-08)
    # landed 166 rows under the request inside the band and was refused.
    selection = calibrate(
        frame,
        TargetSet(problem.targets),
        target_records=households,
        gate_initialization=init,
        mass_reason=dense.options["mass_reason"],
        budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS,
        feasible_draw_pi_hi=pi_hi,
        progress_callback=_phased(progress_callback, "size_search"),
        **_solver_common(dense, epochs=epochs, learning_rate=learning_rate, seed=seed),
    )
    if selection.gate_open_probabilities is None:
        raise RuntimeError("informed L0 returned no selection probabilities.")
    return UKSizeSelection(
        selection=selection,
        protected=np.asarray(init.protected, dtype=bool),
        households=households,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        search_pi_hi=pi_hi,
    )


def refit_uk_dataset_size(
    frame: Frame,
    dense: CalibrationResult,
    *,
    households: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    pi_hi: float = 1.0,
    selection: UKSizeSelection | None = None,
    draw: UKSizeDraw | None = None,
    progress_callback: ProgressCallback | None = None,
) -> UKDatasetSize:
    """Run informed L0, a fixed-size draw, and refit under the dense doctrine.

    Freeze the already compiled household contributions, including engine
    measures that depend on the full population. Re-evaluating those formulas
    on a subset would change the target system. Target values, order, loss
    weights, cap and stretch bound remain those of the dense solve.

    ``pi_hi`` is the exact-count draw's certainty threshold: gates whose
    learned open probability reaches it are taken with certainty and only the
    rest are drawn. The default ``1.0`` keeps exactly the protected carriers
    as certainties; a lower threshold (the US exact-k ladder runs 0.95)
    promotes learned near-certain gates and is a reviewed candidate-run
    setting recorded in the size receipt, never a release default.

    ``selection`` skips the informed L0 search and draws from an existing
    :class:`UKSizeSelection` (a checkpoint restored by
    :mod:`microcosm.build.uk_runtime.size_checkpoint`); it must have been
    searched for the same size, epochs, learning rate and seed on this pool.
    ``draw`` additionally reuses an authenticated completed draw without
    consuming its random stream again; the probability binding must match.
    """
    n = _check_size_inputs(frame, dense, households)
    pi_hi = _check_pi_hi(pi_hi)
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
    if selection is None:
        selection = select_uk_dataset_size(
            frame,
            dense,
            households=households,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            pi_hi=pi_hi,
            progress_callback=progress_callback,
        )
        reused = False
    else:
        reused = True
        expected = {
            "households": households,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        }
        mismatched = sorted(
            f"{key}: selection {getattr(selection, key)!r} != {value!r}"
            for key, value in expected.items()
            if getattr(selection, key) != value
        )
        probabilities_given = selection.selection.gate_open_probabilities
        if (
            probabilities_given is None
            or len(probabilities_given) != n
            or len(selection.protected) != n
            or len(selection.selection.weights) != n
            or selection.selection.l0_lambda <= 0.0
        ):
            mismatched.append("pool: the selection was not searched on this pool")
        if mismatched:
            raise ValueError(
                "cannot draw from a selection searched for different inputs: "
                + "; ".join(mismatched)
                + "."
            )
    init_protected = selection.protected
    common = _solver_common(
        dense, epochs=epochs, learning_rate=learning_rate, seed=seed
    )
    probabilities = selection.selection.gate_open_probabilities
    assert probabilities is not None
    search_result = selection.selection
    if draw is None:
        draw = draw_uk_dataset_size(
            frame,
            dense,
            selection=selection,
            households=households,
            seed=seed,
            pi_hi=pi_hi,
        )
    if (
        draw.seed != seed
        or draw.pi_hi != pi_hi
        or draw.probabilities_sha256 != _probability_digest(probabilities)
    ):
        raise ValueError("Reused exact-count draw differs from its selection or seed.")
    support = assert_exact_k_support(draw.support, households, pool_size=n)
    sampling, q, feasibility = (
        draw.sampling,
        draw.inclusion_probabilities,
        draw.feasibility,
    )
    if (
        np.asarray(q).shape != (households,)
        or not np.isfinite(q).all()
        or (q <= 0).any()
        or (q > 1).any()
    ):
        raise ValueError("Reused exact-count draw has invalid inclusion probabilities.")
    if not np.isin(np.flatnonzero(init_protected), support).all():
        raise RuntimeError("exact-count selection lost a protected carrier.")
    frozen = _frozen_targets(frame, dense, support)
    refit = refit_l0_selection(
        frame,
        frozen,
        search_result,
        support=support,
        k=households,
        support_inclusion_probabilities=q,
        mass_reason=dense.options["mass_reason"],
        progress_callback=_phased(progress_callback, "size_refit"),
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
            "protected_carriers": int(init_protected.sum()),
            # How much of the count the threshold decided versus the draw: with
            # polarised gates the "draw" is mostly a threshold on learned pi.
            "certainty_share": float(sampling["certainty_count"]) / float(households),
            "boundary_draws": int(households - sampling["certainty_count"]),
            # Rows whose target is exactly zero: the contribution prior uses a
            # unit denominator there, so a survivor would saturate the prior.
            "zero_target_rows": int(np.count_nonzero(problem.target_vector == 0.0)),
            "selection_receipt": sampling,
            "selection_pi_hi": pi_hi,
            "selection_search_pi_hi": selection.search_pi_hi,
            "selection_reused": reused,
            "selection_budget_basis": BUDGET_BASIS_OPEN_PROBABILITY_MASS,
            "selection_budget_search": search_result.options.get("budget_search"),
            "selection_feasibility": feasibility,
            "selection_l0_lambda": search_result.l0_lambda,
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


def unsupported_nonzero_targets(problem: Any) -> list[str]:
    """Names of nonzero targets whose constraint row has no nonzero entry.

    ``contribution_initialization`` refuses such a row by index; a size run
    should refuse it by name, because the cure is upstream (a measure that
    resolves to zero on the cloned pool, or a filter no pool household
    satisfies), not in the selection.
    """

    matrix = sparse.csr_array(problem.matrix)
    targets = np.asarray(problem.target_vector, dtype=np.float64)
    support = np.diff(matrix.indptr)
    nonzero_entries = np.asarray(
        [
            int(np.count_nonzero(matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]]))
            for i in range(matrix.shape[0])
        ]
    )
    names = list(getattr(problem, "names", [str(i) for i in range(matrix.shape[0])]))
    return [
        str(names[i])
        for i in range(matrix.shape[0])
        if targets[i] != 0.0 and (support[i] == 0 or nonzero_entries[i] == 0)
    ]


_FEASIBILITY_PI_HI_GRID = (0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5)


def selection_feasibility(
    probabilities: np.ndarray,
    households: int,
    *,
    protected: np.ndarray,
    n_nonzero: int,
    l0_lambda: float,
    requested_pi_hi: float = 1.0,
    budget_search: dict[str, object] | None = None,
    search_pi_hi: float | None = None,
) -> dict[str, object]:
    """Measure whether an exact-count draw is feasible on these gate probabilities.

    ``select_exact_k`` with ``pi_hi=1.0`` keeps every gate below one in the
    boundary and scales its open probabilities to the remaining draw size
    ``m``; the scaled value of the largest boundary gate must not exceed one,
    i.e. ``m * max(pi_boundary) <= sum(pi_boundary)``. The L0 budget search
    stops on the count of not-fully-closed gates, which can sit well above the
    open-probability mass when gates are only partly polarised, so the draw
    can refuse. This records the measured mass and the two ways out — the
    smallest certainty threshold on a fixed grid that makes the design
    feasible, and the largest household count feasible at ``pi_hi=1.0`` — so
    the refusal is a ruling with numbers, never a silent clamp.
    Every verdict comes from :func:`exact_k_design_feasibility`, the draw's own
    inequality, so the scan and the draw can never disagree. ``budget_search``
    is the search receipt (probes, the verdict each stopped on) when the
    selection was searched with a feasibility-aware stop.
    """

    pi = np.asarray(probabilities, dtype=np.float64)
    protected_mask = np.asarray(protected, dtype=bool)
    if pi.shape != protected_mask.shape:
        raise ValueError("selection feasibility needs aligned probabilities and mask.")
    certainty = pi >= 1.0
    boundary = pi[~certainty]
    positive = boundary[boundary > 0.0]
    m = int(households) - int(certainty.sum())
    boundary_mass = float(positive.sum()) if positive.size else 0.0
    boundary_max = float(positive.max()) if positive.size else 0.0
    at_one = exact_k_design_feasibility(pi, int(households), 1.0)
    feasible = bool(at_one["feasible"])
    max_feasible_k = (
        int(certainty.sum()) + int(np.floor(boundary_mass / boundary_max))
        if boundary_max > 0.0
        else int(certainty.sum())
    )
    scan: dict[str, object] = {}
    smallest_feasible_pi_hi: float | None = 1.0 if feasible else None
    for threshold in _FEASIBILITY_PI_HI_GRID:
        design = exact_k_design_feasibility(pi, int(households), threshold)
        ok = bool(design["feasible"])
        scan[f"{threshold:g}"] = {
            "certainties": int(design["certainty_count"]),
            "boundary_draw": int(design["boundary_draw"]),
            "boundary_mass": float(design["boundary_mass"]),
            "boundary_max": float(design["boundary_max"]),
            "feasible": ok,
            "reason": str(design["reason"]),
        }
        if ok and smallest_feasible_pi_hi is None:
            smallest_feasible_pi_hi = threshold
    quantiles = (
        {f"p{q * 100:g}": float(np.quantile(pi, q)) for q in (0.5, 0.9, 0.99, 0.999)}
        if pi.size
        else {}
    )
    requested = exact_k_design_feasibility(pi, int(households), float(requested_pi_hi))
    return {
        "requested_households": int(households),
        "requested_pi_hi": float(requested_pi_hi),
        "feasible_at_requested_pi_hi": bool(requested["feasible"]),
        "requested_pi_hi_verdict": str(requested["reason"]),
        "search_pi_hi": search_pi_hi,
        "budget_search": budget_search,
        "pool_households": int(pi.size),
        "protected_carriers": int(protected_mask.sum()),
        "certainties_at_pi_hi_1": int(certainty.sum()),
        "boundary_draw": m,
        "boundary_positive_gates": int(positive.size),
        "boundary_mass": boundary_mass,
        "boundary_max": boundary_max,
        "feasible_at_pi_hi_1": bool(feasible),
        "max_feasible_households_at_pi_hi_1": max_feasible_k,
        "smallest_feasible_pi_hi_on_grid": smallest_feasible_pi_hi,
        "pi_hi_scan": scan,
        "pi_sum": float(pi.sum()),
        "pi_quantiles": quantiles,
        "gates_above": {
            f"{t:g}": int((pi >= t).sum()) for t in (0.5, 0.9, 0.99, 0.999)
        },
        "budget_search_n_nonzero": int(n_nonzero),
        "budget_search_basis": BUDGET_BASIS_OPEN_PROBABILITY_MASS,
        "selection_l0_lambda": float(l0_lambda),
    }


def _feasible_at(pi: np.ndarray, households: int, threshold: float) -> bool:
    return bool(exact_k_design_feasibility(pi, households, threshold)["feasible"])


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
