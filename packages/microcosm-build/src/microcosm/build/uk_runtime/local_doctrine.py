"""UK local solve doctrine: one uniform operator, declared bounds (#495).

The doctrine adjudicated on the microcosm#492/#493 lane: **no per-target
calibration knobs**. The UK local solve is one operator over the whole
area x metric surface — the canonical capped relative-error loss on the
default target-defined scales, uniform target weights, one declared loss
cap, one declared weight-ratio stretch bound. A miss is local-support work
(rows, clones, ladder coverage) or target work (fix or fence the target),
never a knob.

This module declares the release path's reviewed bounds. The solve surface
that consumes them is the rowwise doctrine solve
(:func:`microcosm.build.uk_runtime.local_rowwise.solve_uk_rowwise_weights_under_doctrine`);
it exposes no per-target parameters — the refusal is structural, not a
runtime flag — and every solve carries the microcosm#492 past-cap census so
written-off rows are first-class diagnostics. The pre-ladder stacked
research harness was removed with microcosm#612 increment 2.

The declared values below are the current reviewed contract. Revising either
constant is a doctrine change: it must edit this module (and the pinned
test), which forces review. The microcosm#493 one-stretch-contract
adjudication may revise ``UK_LOCAL_MAX_WEIGHT_RATIO``; the first calibrated
rowwise candidate review (#495 increment 6) adjudicates the cap against
measured fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "UK_LOCAL_MAX_WEIGHT_RATIO",
    "UK_LOCAL_SOLVE_DOCTRINE",
    "UK_LOCAL_TARGET_LOSS_CAP",
    "UKLocalSolveDoctrine",
]

#: Declared uniform loss cap for the UK local solve (scaled absolute
#: relative-error units on the default target-defined scales).
UK_LOCAL_TARGET_LOSS_CAP = 10.0

#: Declared weight-ratio stretch bound for the UK local solve.
#:
#: microcosm#493 adjudication record (2026-08-07): the bound stays 100.0,
#: recorded against the US design bound of 5.0 (the ACS local and fiscal
#: refresh defaults; realized max ratio 4.994 in production) and the US
#: exact-k ladder's 20.0. The UK value is not yet a measured-fit choice —
#: it is the bound the local solve inherited when it was a second entry
#: point owning its own defaults — and tightening it without measured fit
#: would be an arbitrary cutoff. The revision path remains the #495
#: increment-6 calibrated-candidate review, which adjudicates against
#: measured fit; any revision must edit this constant and its pinned test,
#: which forces review.
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
