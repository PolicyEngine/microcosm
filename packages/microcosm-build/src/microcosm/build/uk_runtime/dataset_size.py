"""Exact household-count UK candidates on a fixed, materialized target surface."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from microcosm.build.uk_runtime.target_weights import UKStageTargetWeighting
from microcosm.calibrate import (
    CalibrationResult,
    TargetSet,
    assert_exact_k_support,
    calibrate,
    chi_square_distance,
    exact_k_design_feasibility,
    refit_l0_selection,
    relative_error_loss,
    select_exact_k,
)
from microcosm.calibrate.initialization import contribution_initialization
from microcosm.calibrate.solve import (
    BUDGET_BASIS_OPEN_PROBABILITY_MASS,
    L2_BASIS_CHI_SQUARE,
)
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
    initial_lambda: float | None = None


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

    @classmethod
    def from_payload(
        cls, payload: bytes | str | Mapping[str, object], *, problem_sha256: str
    ) -> "UKSizeDraw | None":
        """Rebuild a stored draw artifact; ``None`` for a full-pool draw.

        The graph's size-draw node writes this JSON; the refit node and the
        size-experiment harness read it back through this one parser.
        """

        draw = dict(payload if isinstance(payload, Mapping) else json.loads(payload))
        if draw.pop("problem_sha256", None) != problem_sha256:
            raise ValueError("Exact-count draw belongs to another ordered problem.")
        method = draw.pop("method", None)
        if method == "full_pool":
            return None
        if method != "exact_count":
            raise ValueError("Compact refit requires a completed exact-count draw.")
        return cls(
            **{
                **draw,
                "support": np.asarray(draw["support"]),
                "inclusion_probabilities": np.asarray(
                    draw["inclusion_probabilities"], dtype=np.float64
                ),
            }
        )


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
        # The draw refuses rather than clamps; carry the measured gate mass
        # with the refusal so the ruling it needs can be made from the receipt.
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


def _check_baseline_pi_floor(value: object) -> float:
    if not isinstance(value, float | int) or isinstance(value, bool):
        raise ValueError("baseline_pi_floor must be a number in [0, 1].")
    floor = float(value)
    if not (0.0 <= floor <= 1.0):
        raise ValueError("baseline_pi_floor must be a number in [0, 1].")
    return floor


def _check_initial_lambda(value: object) -> float | None:
    """``None`` (cold search) or a positive finite warm-start penalty."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or float(value) <= 0.0
    ):
        raise ValueError("initial_lambda must be None or a positive finite number.")
    return float(value)


def _check_pi_hi(pi_hi: object) -> float:
    if not isinstance(pi_hi, float | int) or isinstance(pi_hi, bool):
        raise ValueError("pi_hi must be a number in (0, 1].")
    value = float(pi_hi)
    if not (0.0 < value <= 1.0):
        raise ValueError("pi_hi must be a number in (0, 1].")
    return value


#: The L2 anchors a UK size stage admits (microcosm#1124). ``initial`` is the
#: stage's own starting weights: the pool design weights in the search, the
#: normalized (optionally floored) Horvitz–Thompson baseline in the refit.
#: ``uniform`` is their mean, a direct Kish-ESS control.
UK_SIZE_L2_ANCHORS = ("initial", "uniform")
#: The L2 basis a UK size stage admits. Under the UK's free mass the record
#: basis has its target-free optimum at every weight zero, so it only shrinks
#: mass; the chi-square basis is zero at the anchor (penalized GREG).
UK_SIZE_L2_BASES = (L2_BASIS_CHI_SQUARE,)
UK_SIZE_L2_STAGES = ("selection", "refit")
#: The flat keyword names a size stage's L2 travels under (graph node
#: parameters and the size functions' keywords).
UK_SIZE_L2_PARAM_KEYS = tuple(
    f"{stage}_l2_{field}"
    for stage in UK_SIZE_L2_STAGES
    for field in ("lambda", "anchor", "basis")
)


@dataclass(frozen=True)
class UKSizeL2:
    """One size stage's L2 concentration penalty; an instance means it is on."""

    stage: str
    l2_lambda: float
    anchor: str = "initial"
    basis: str = L2_BASIS_CHI_SQUARE

    def __post_init__(self) -> None:
        if self.stage not in UK_SIZE_L2_STAGES:
            raise ValueError(f"L2 stage must be one of {UK_SIZE_L2_STAGES}.")
        value = self.l2_lambda
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{self.stage}_l2_lambda of an L2 penalty must be > 0.")
        if self.anchor not in UK_SIZE_L2_ANCHORS:
            raise ValueError(
                f"{self.stage}_l2_anchor must be one of {UK_SIZE_L2_ANCHORS}, "
                f"got {self.anchor!r}."
            )
        if self.basis not in UK_SIZE_L2_BASES:
            raise ValueError(
                f"{self.stage}_l2_basis must be one of {UK_SIZE_L2_BASES}, "
                f"got {self.basis!r}."
            )
        object.__setattr__(self, "l2_lambda", float(value))

    def solver_kwargs(self) -> dict[str, object]:
        return {
            "l2_lambda": self.l2_lambda,
            "l2_anchor": self.anchor,
            "l2_basis": self.basis,
        }

    def params(self) -> dict[str, object]:
        """The flat keywords (node parameters) this penalty travels under."""

        return {
            f"{self.stage}_l2_lambda": self.l2_lambda,
            f"{self.stage}_l2_anchor": self.anchor,
            f"{self.stage}_l2_basis": self.basis,
        }

    def as_dict(self) -> dict[str, object]:
        return {"lambda": self.l2_lambda, "anchor": self.anchor, "basis": self.basis}


def uk_size_l2(
    stage: str,
    *,
    l2_lambda: object = 0.0,
    anchor: object = None,
    basis: object = None,
) -> UKSizeL2 | None:
    """Validate one stage's flat L2 settings; ``None`` when the penalty is off.

    ``l2_lambda`` 0 (the default) is off, and then no anchor or basis may be
    given. The record basis and a ``"design"`` anchor are refused with their
    reasons: the first only shrinks mass under free mass; the second is, on
    the exact-count refit, the same vector as ``"initial"`` (the normalized
    Horvitz–Thompson baseline), and in the search ``"initial"`` already names
    the pool design weights.
    """

    if stage not in UK_SIZE_L2_STAGES:
        raise ValueError(f"L2 stage must be one of {UK_SIZE_L2_STAGES}.")
    if (
        isinstance(l2_lambda, bool)
        or not isinstance(l2_lambda, int | float)
        or not np.isfinite(l2_lambda)
        or l2_lambda < 0.0
    ):
        raise ValueError(f"{stage}_l2_lambda must be a finite number >= 0.")
    if float(l2_lambda) == 0.0:
        if anchor is not None or basis is not None:
            raise ValueError(
                f"{stage} L2 anchor or basis given without a positive "
                f"{stage}_l2_lambda."
            )
        return None
    anchor = "initial" if anchor is None else str(anchor)
    basis = L2_BASIS_CHI_SQUARE if basis is None else str(basis)
    if anchor == "design":
        raise ValueError(
            f"{stage}_l2_anchor 'design' is not a size-stage anchor: on the "
            "exact-count refit it is the same vector as 'initial' (the "
            "normalized Horvitz–Thompson baseline), and in the search 'initial' "
            "already names the pool design weights."
        )
    if basis == "record":
        raise ValueError(
            f"{stage}_l2_basis 'record' is refused under the UK's free mass: its "
            "target-free optimum is every weight at zero, so it only shrinks "
            "mass. Use 'chi_square'."
        )
    return UKSizeL2(stage, float(l2_lambda), anchor, basis)


def uk_size_l2_from_options(options: Mapping[str, object]) -> dict[str, object] | None:
    """The L2 penalty a solved stage recorded in its options (``None`` when off)."""

    value = float(options.get("l2_lambda", 0.0) or 0.0)
    if value == 0.0:
        return None
    return {
        "lambda": value,
        "anchor": str(options.get("l2_anchor", "initial")),
        "basis": str(options.get("l2_basis", "record")),
    }


def _l2_solver_kwargs(l2: UKSizeL2 | None) -> dict[str, object]:
    # Off means no L2 keyword reaches the solver: its recorded defaults (and
    # every byte of a default run) stay what they were before #1124.
    return {} if l2 is None else l2.solver_kwargs()


def _stage_loss_weights(
    dense: CalibrationResult, weighting: object
) -> np.ndarray | None:
    """A stage's loss weights from a named rule, or ``None`` (the dense's)."""

    if weighting is None:
        return None
    if not isinstance(weighting, UKStageTargetWeighting):
        raise TypeError(
            "a size stage's target weighting must be a UKStageTargetWeighting "
            "(a named rule on the problem's rows), never a raw vector."
        )
    return weighting.loss_weights(dense.problem.names)


def _solver_common(
    dense: CalibrationResult,
    *,
    epochs: int,
    learning_rate: float,
    seed: int,
    target_loss_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    return dict(
        weight_entity="household",
        epochs=epochs,
        learning_rate=learning_rate,
        mass=dense.options["mass"],
        max_weight_ratio=dense.options["max_weight_ratio"],
        seed=seed,
        target_loss_weights=(
            dense.target_loss_weights
            if target_loss_weights is None
            else target_loss_weights
        ),
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
    initial_lambda: float | None = None,
    selection_l2_lambda: float = 0.0,
    selection_l2_anchor: str | None = None,
    selection_l2_basis: str | None = None,
    target_weighting: UKStageTargetWeighting | None = None,
    progress_callback: ProgressCallback | None = None,
) -> UKSizeSelection:
    """Run the informed L0 budget search for an exact-count draw of ``households``.

    Refuses by name any nonzero target no pool household supports, protects
    each remaining target's largest carrier, and searches the L0 penalty on
    the gates' open-probability mass until a draw of ``households`` at
    ``pi_hi`` is feasible on the learned probabilities (or the probe budget
    is spent; then the draw refuses with the measurement).

    ``initial_lambda`` warm-starts the search at a known penalty (the one a
    previous search on the same pool and targets selected, microcosm#1115):
    the search probes it first and stops there when the draw is feasible and
    within tolerance, saving the full bisection. The search still verifies
    every probe, so a stale hint costs probes, never feasibility; any penalty
    whose draw lands inside the budget window is a valid stop, so a warm and a
    cold search can settle on different penalties and select different rows.

    ``selection_l2_*`` add a chi-square L2 penalty on the search's pre-gate
    weights (microcosm#1124; see :func:`uk_size_l2`); off by default, and an
    off penalty passes no L2 keyword to the solver. ``target_weighting``
    searches under a named rule other than the dense solve's weights (the
    size-experiment seam; graph builds never pass it).
    """
    n = _check_size_inputs(frame, dense, households)
    pi_hi = _check_pi_hi(pi_hi)
    initial_lambda = _check_initial_lambda(initial_lambda)
    selection_l2 = uk_size_l2(
        "selection",
        l2_lambda=selection_l2_lambda,
        anchor=selection_l2_anchor,
        basis=selection_l2_basis,
    )
    search_weights = _stage_loss_weights(dense, target_weighting)
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
        l0_lambda=0.0 if initial_lambda is None else initial_lambda,
        progress_callback=_phased(progress_callback, "size_search"),
        **_l2_solver_kwargs(selection_l2),
        **_solver_common(
            dense,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            target_loss_weights=search_weights,
        ),
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
        initial_lambda=initial_lambda,
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
    initial_lambda: float | None = None,
    baseline_pi_floor: float = 0.0,
    selection_l2_lambda: float = 0.0,
    selection_l2_anchor: str | None = None,
    selection_l2_basis: str | None = None,
    refit_l2_lambda: float = 0.0,
    refit_l2_anchor: str | None = None,
    refit_l2_basis: str | None = None,
    selection_target_weighting: UKStageTargetWeighting | None = None,
    refit_target_weighting: UKStageTargetWeighting | None = None,
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

    ``baseline_pi_floor`` trims the Horvitz–Thompson baseline the refit starts
    from and stretches against: each selected row's dense weight is divided by
    ``max(q, baseline_pi_floor)`` instead of its inclusion probability ``q``
    before the baseline is normalised to the pool mass. ``0.0`` (default) is
    the untrimmed baseline. A boundary row drawn at ``q`` of a few in a
    million otherwise starts at millions of households and, after the
    normalisation, leaves the certainties too little mass to reach the
    targets under the stretch bound (microcosm#355, Q50 2026-09-10: 24 such
    rows held 92 % of the baseline; the refit ended 24 % short of the pool
    mass at 10× the dense loss). Candidate-only, recorded in the size receipt
    with the number of rows it trimmed and the certainties' baseline share.

    ``selection`` skips the informed L0 search and draws from an existing
    :class:`UKSizeSelection` (a checkpoint restored by
    :mod:`microcosm.build.uk_runtime.size_checkpoint`); it must have been
    searched for the same size, epochs, learning rate and seed on this pool.
    ``draw`` additionally reuses an authenticated completed draw without
    consuming its random stream again; the probability binding must match.

    ``selection_l2_*`` are the search's L2 settings (a reused selection must
    have been searched under the same ones) and ``refit_l2_*`` the refit's
    own, independent of the search's (microcosm#1124; see :func:`uk_size_l2`).
    On the exact-count path the refit's ``"initial"`` anchor is the
    normalized Horvitz–Thompson baseline, floored by ``baseline_pi_floor``.
    ``selection_target_weighting`` and ``refit_target_weighting`` run a stage
    under a named rule other than the dense solve's weights (the
    size-experiment seam; graph builds never pass them). Every new setting is
    off by default and then leaves the call, its solver options and its
    receipt exactly as before.
    """
    n = _check_size_inputs(frame, dense, households)
    pi_hi = _check_pi_hi(pi_hi)
    baseline_pi_floor = _check_baseline_pi_floor(baseline_pi_floor)
    selection_l2 = uk_size_l2(
        "selection",
        l2_lambda=selection_l2_lambda,
        anchor=selection_l2_anchor,
        basis=selection_l2_basis,
    )
    refit_l2 = uk_size_l2(
        "refit", l2_lambda=refit_l2_lambda, anchor=refit_l2_anchor, basis=refit_l2_basis
    )
    search_weights = _stage_loss_weights(dense, selection_target_weighting)
    refit_weights = _stage_loss_weights(dense, refit_target_weighting)
    if households == n:
        if (
            selection_l2 is not None
            or refit_l2 is not None
            or search_weights is not None
            or refit_weights is not None
        ):
            raise ValueError(
                "a full-pool size returns the dense solve unchanged, so it cannot "
                "honour a size-stage L2 penalty or target weighting."
            )
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
            initial_lambda=initial_lambda,
            selection_l2_lambda=selection_l2_lambda,
            selection_l2_anchor=selection_l2_anchor,
            selection_l2_basis=selection_l2_basis,
            target_weighting=selection_target_weighting,
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
        stored_l2 = uk_size_l2_from_options(selection.selection.options)
        requested_l2 = None if selection_l2 is None else selection_l2.as_dict()
        if stored_l2 != requested_l2 or selection.selection.options.get(
            "l2_anchor_weights_supplied", False
        ):
            mismatched.append(
                f"selection L2: searched under {stored_l2!r}, requested "
                f"{requested_l2!r}"
            )
        expected_search_weights = (
            dense.target_loss_weights if search_weights is None else search_weights
        )
        if not np.array_equal(
            np.asarray(selection.selection.target_loss_weights, dtype=np.float64),
            np.asarray(expected_search_weights, dtype=np.float64),
        ):
            mismatched.append(
                "target loss weights: the selection was searched under other weights"
            )
        if mismatched:
            raise ValueError(
                "cannot draw from a selection searched for different inputs: "
                + "; ".join(mismatched)
                + "."
            )
    init_protected = selection.protected
    common = _solver_common(
        dense,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        target_loss_weights=refit_weights,
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
    q = np.asarray(q, dtype=np.float64)
    baseline_q = q if baseline_pi_floor == 0.0 else np.maximum(q, baseline_pi_floor)
    baseline_floored_rows = int(np.count_nonzero(q < baseline_pi_floor))
    refit = refit_l0_selection(
        frame,
        frozen,
        search_result,
        support=support,
        k=households,
        support_inclusion_probabilities=baseline_q,
        mass_reason=dense.options["mass_reason"],
        progress_callback=_phased(progress_callback, "size_refit"),
        **_l2_solver_kwargs(refit_l2),
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
    baseline = np.asarray(refit.initial_weights, dtype=np.float64)
    baseline_total = float(baseline.sum())
    certainty_rows = q >= 1.0
    baseline_name = "normalized_horvitz_thompson_w_over_q" + (
        "_floored" if baseline_pi_floor > 0.0 else ""
    )
    receipt_extras = _size_stage_receipt(
        dense=dense,
        search_result=search_result,
        refit=refit,
        refit_l2=refit_l2,
        baseline_name=baseline_name,
        selection_weighting=selection_target_weighting,
        refit_weighting=refit_target_weighting,
    )
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
            "refit_baseline": baseline_name,
            "stretch_reference": baseline_name,
            "baseline_pi_floor": baseline_pi_floor,
            "baseline_floored_rows": baseline_floored_rows,
            # How the normalised baseline splits between the certainties and
            # the boundary draws: the certainties' 10x ceiling has to cover
            # the pool mass for the refit to reach the targets at all.
            "baseline_mass_share_certainties": (
                float(baseline[certainty_rows].sum() / baseline_total)
                if baseline_total > 0.0
                else None
            ),
            "dense_loss": dense.final_loss,
            "compact_loss": refit.final_loss,
            "max_target_scaled_change": float(
                np.max(np.abs(errors_small - errors_dense) / dense.target_loss_scales)
            ),
            "certification": "candidate_only_pending_matched_comparison_and_promotion_scorecard",
            **receipt_extras,
        },
    )


def _size_stage_receipt(
    *,
    dense: CalibrationResult,
    search_result: CalibrationResult,
    refit: CalibrationResult,
    refit_l2: UKSizeL2 | None,
    baseline_name: str,
    selection_weighting: UKStageTargetWeighting | None,
    refit_weighting: UKStageTargetWeighting | None,
) -> dict[str, object]:
    """The microcosm#1124 receipt fields, present only when a setting is on.

    A default run (no L2, the dense's weights in both stages) adds nothing,
    so its size receipt is byte-identical to the one before these settings
    existed.
    """

    extras: dict[str, object] = {}
    searched_l2 = uk_size_l2_from_options(search_result.options)
    if searched_l2 is not None:
        extras["selection_l2"] = {
            **searched_l2,
            "penalty": search_result.options.get("l2_penalty"),
            "anchor_reference": (
                "pool_design"
                if searched_l2["anchor"] == "initial"
                else "pool_design_mean"
            ),
        }
    if refit_l2 is not None:
        start = np.asarray(refit.initial_weights, dtype=np.float64)
        anchor = (
            start if refit_l2.anchor == "initial" else np.full_like(start, start.mean())
        )
        extras["refit_l2"] = {
            **refit_l2.as_dict(),
            "penalty": refit.options.get("l2_penalty"),
            "anchor_reference": (
                baseline_name
                if refit_l2.anchor == "initial"
                else f"{baseline_name}_mean"
            ),
            "chi_square_distance_to_anchor": float(
                chi_square_distance(np.asarray(refit.weights, dtype=np.float64), anchor)
            ),
        }
        extras["refit_iterate_selection"] = refit.options.get("iterate_selection")
    names = dense.problem.names
    if selection_weighting is not None:
        extras["selection_target_weighting"] = selection_weighting.receipt(names)
    if refit_weighting is not None:
        extras["refit_target_weighting"] = refit_weighting.receipt(names)
        # The compact loss under the dense solve's own weights keeps the
        # dense-versus-compact comparison on one yardstick.
        estimates = np.asarray(
            [diagnostic.final_estimate for diagnostic in refit.diagnostics],
            dtype=np.float64,
        )
        extras["compact_loss_under_dense_weights"] = float(
            relative_error_loss(
                estimates,
                np.asarray(refit.problem.target_vector, dtype=np.float64),
                target_loss_weights=dense.target_loss_weights,
                target_loss_scales=dense.target_loss_scales,
                target_loss_cap=dense.target_loss_cap,
            )
        )
    return extras


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
