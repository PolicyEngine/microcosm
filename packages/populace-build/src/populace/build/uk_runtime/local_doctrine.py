"""UK local solve doctrine: one uniform operator, declared bounds (#495).

The doctrine adjudicated on the populace#492/#493 lane: **no per-target
calibration knobs**. The UK local solve is one operator over the whole
area x metric surface — the canonical capped relative-error loss on the
default target-defined scales, uniform target weights, one declared loss
cap, one declared weight-ratio stretch bound. A miss is local-support work
(rows, clones, ladder coverage) or target work (fix or fence the target),
never a knob.

This module is the release path's solve surface. The low-level
:func:`populace.build.uk_runtime.local_solver.solve_stacked_local_weights`
remains the research harness (it accepts explicit per-target vectors for
experiments); :func:`solve_uk_local_weights_under_doctrine` exposes none of
them — the refusal is structural, not a runtime flag — and every solve
carries the populace#492 past-cap census so written-off rows are first-class
diagnostics.

The declared values below are the current reviewed contract. Revising either
constant is a doctrine change: it must edit this module (and the pinned
test), which forces review. The populace#493 one-stretch-contract
adjudication may revise ``UK_LOCAL_MAX_WEIGHT_RATIO``; the first calibrated
rowwise candidate review (#495 increment 6) adjudicates the cap against
measured fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from populace.build.uk_runtime.local_geography import StackedLocalMatrix
from populace.build.uk_runtime.local_solver import (
    StackedLocalSolveResult,
    solve_stacked_local_weights,
)

__all__ = [
    "UK_LOCAL_MAX_WEIGHT_RATIO",
    "UK_LOCAL_SOLVE_DOCTRINE",
    "UK_LOCAL_TARGET_LOSS_CAP",
    "UKLocalSolveDoctrine",
    "solve_uk_local_weights_under_doctrine",
]

#: Declared uniform loss cap for the UK local solve (scaled absolute
#: relative-error units on the default target-defined scales).
UK_LOCAL_TARGET_LOSS_CAP = 10.0

#: Declared weight-ratio stretch bound for the UK local solve.
UK_LOCAL_MAX_WEIGHT_RATIO = 100.0

_ALLOWED_SCALE_RULES = ("default_target_loss_scales",)
_ALLOWED_TARGET_WEIGHT_RULES = ("uniform",)


@dataclass(frozen=True)
class UKLocalSolveDoctrine:
    """The declared, reviewed bounds of the uniform UK local solve operator.

    ``scale_rule`` and ``target_weight_rule`` are closed vocabularies: the
    only admissible scale rule is the canonical target-defined default, and
    the only admissible target weighting is uniform. A future family-level
    weighting would be a new reviewed rule name here — never a per-target
    vector.
    """

    target_loss_cap: float = UK_LOCAL_TARGET_LOSS_CAP
    max_weight_ratio: float | None = UK_LOCAL_MAX_WEIGHT_RATIO
    scale_rule: str = "default_target_loss_scales"
    target_weight_rule: str = "uniform"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_loss_cap, int | float)
            or isinstance(self.target_loss_cap, bool)
            or not np.isfinite(self.target_loss_cap)
            or self.target_loss_cap <= 0
        ):
            raise ValueError(
                "doctrine target_loss_cap must be a positive finite number, "
                f"got {self.target_loss_cap!r}."
            )
        if self.max_weight_ratio is not None and (
            not isinstance(self.max_weight_ratio, int | float)
            or isinstance(self.max_weight_ratio, bool)
            or not np.isfinite(self.max_weight_ratio)
            or self.max_weight_ratio <= 1
        ):
            raise ValueError(
                "doctrine max_weight_ratio must be None or a finite number "
                f"greater than 1, got {self.max_weight_ratio!r}."
            )
        if self.scale_rule not in _ALLOWED_SCALE_RULES:
            raise ValueError(
                f"doctrine scale_rule must be one of {_ALLOWED_SCALE_RULES}, "
                f"got {self.scale_rule!r}."
            )
        if self.target_weight_rule not in _ALLOWED_TARGET_WEIGHT_RULES:
            raise ValueError(
                "doctrine target_weight_rule must be one of "
                f"{_ALLOWED_TARGET_WEIGHT_RULES}, got "
                f"{self.target_weight_rule!r}."
            )


#: The reviewed doctrine instance every release-path solve uses.
UK_LOCAL_SOLVE_DOCTRINE = UKLocalSolveDoctrine()


def solve_uk_local_weights_under_doctrine(
    problem: StackedLocalMatrix,
    base_weights: Sequence[float],
    *,
    doctrine: UKLocalSolveDoctrine = UK_LOCAL_SOLVE_DOCTRINE,
    epochs: int = 512,
    learning_rate: float = 0.15,
    conserve_mass: bool = False,
    target_records: int | None = None,
    l0_lambda: float = 0.0,
    min_initial_weight: float = 1e-4,
    budget_iters: int = 10,
    seed: int = 0,
) -> StackedLocalSolveResult:
    """Solve UK local weights as one uniform operator with declared bounds.

    Structurally knob-free: there are no per-target weight or scale
    parameters on this signature. Scales follow the doctrine's canonical
    rule (applied by the underlying solver's default), target weights are
    uniform, and the loss cap and stretch bound come from the doctrine
    instance — whose fields validate themselves, so a tampered contract
    fails before any solve runs.
    """

    if not isinstance(doctrine, UKLocalSolveDoctrine):
        raise TypeError(
            f"doctrine must be a UKLocalSolveDoctrine instance, got {doctrine!r}."
        )
    return solve_stacked_local_weights(
        problem,
        base_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=doctrine.max_weight_ratio,
        conserve_mass=conserve_mass,
        target_records=target_records,
        l0_lambda=l0_lambda,
        min_initial_weight=min_initial_weight,
        target_loss_cap=doctrine.target_loss_cap,
        budget_iters=budget_iters,
        seed=seed,
    )
