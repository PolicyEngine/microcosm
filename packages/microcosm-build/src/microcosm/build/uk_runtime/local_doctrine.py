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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace

import numpy as np

__all__ = [
    "UK_LOCAL_CLONE_COUNT",
    "UK_LOCAL_MAX_WEIGHT_RATIO",
    "UK_LOCAL_SOLVE_DOCTRINE",
    "UK_LOCAL_SOLVE_EPOCHS",
    "UK_LOCAL_TARGET_LOSS_CAP",
    "UK_LOCAL_TARGET_WEIGHT_RULE",
    "UKLocalSolveDoctrine",
    "uk_local_doctrine_with_overrides",
    "uk_local_target_loss_weights",
]

#: Declared uniform loss cap for the UK local solve (scaled absolute
#: relative-error units on the default target-defined scales).
UK_LOCAL_TARGET_LOSS_CAP = 10.0

#: Declared weight-ratio stretch bound for the UK local solve.
#:
#: microcosm#762 A3 adjudication (2026-09-03, receipts R8–R10 in
#: ``experiments/762-uk-rowwise-candidate-receipts.md``): 10.0, measured
#: against 100.0 and 20.0 on spine-m at K=10 under both weighting rules.
#: Tightening 100 → 10 left the final loss and every family's within-10 %
#: share unchanged to a tenth of a point; the extra stretch bought weight
#: concentration, not fit (max/median 312 → 105 under ``uniform``, 1,014 →
#: 578 under ``grain_equal``), and 10 was the only bound at which any run
#: cleared the ESS ≥ 50 support floor. It is also the national doctrine's
#: bound, so the two UK solves now agree. The #493 record (2026-08-07) had
#: held 100.0 explicitly as the inherited, not-yet-measured value pending
#: this review. Any revision must edit this constant and its pinned test.
UK_LOCAL_MAX_WEIGHT_RATIO = 10.0

#: Declared target-weighting rule for the UK local solve.
#:
#: microcosm#762 A2 adjudication (2026-09-03, receipts R6–R10):
#: ``grain_equal`` — the national rows, the constituency rows and the
#: local-authority rows take one equal share of the loss each, uniform
#: within. Under ``uniform`` the 364 national rows are 1.8 % of the 20,480-
#: row matrix and are effectively ignored (78 % within 10 %; OBR aggregates
#: 43 %); ``grain_equal`` holds them at 92–93 % for about two points of
#: census-household and tenure fit. ``uniform`` stays a receipted override.
UK_LOCAL_TARGET_WEIGHT_RULE = "grain_equal"

#: Declared solve length for the UK local solve (Adam epochs).
#:
#: microcosm#762 adjudication (2026-09-03, receipt R10): 1500, María's
#: ruling, matching the national certified-cut posture. The solver has no
#: stopping criterion; at 512 the loss was still falling 2–6 % per 128
#: epochs, at 1500 it is 0.4 % (``uniform``) / 2.8 % (``grain_equal``).
UK_LOCAL_SOLVE_EPOCHS = 1500

#: Declared clone count K for the UK local candidate (rows per spine
#: household through the OA ladder).
#:
#: microcosm#762 adjudication (2026-09-03, receipt R9): 15. The plan's
#: K=4 default refused 172 UC child-band cells with zero support; K=10
#: refused 86 band-H cells (A14) and, under ``grain_equal`` at bound 10,
#: left five constituencies 0.3–7.7 ESS under the floor; K=15 clears the
#: floor (min constituency ESS 54.1) at unchanged fit and flat memory.
UK_LOCAL_CLONE_COUNT = 15

_ALLOWED_SCALE_RULES = ("default_target_loss_scales",)
_ALLOWED_TARGET_WEIGHT_RULES = ("uniform", "grain_equal")
_OVERRIDABLE_FIELDS = frozenset({"target_weight_rule"})


@dataclass(frozen=True)
class UKLocalSolveDoctrine:
    """The declared, reviewed bounds of the uniform UK local solve operator.

    ``scale_rule`` and ``target_weight_rule`` are closed vocabularies: the
    only admissible scale rule is the canonical target-defined default, and
    the admissible target weightings are ``grain_equal`` (the reviewed
    default) and ``uniform`` (a receipted override). Any other weighting
    would be a new reviewed rule name here — never a per-target vector.
    """

    target_loss_cap: float = UK_LOCAL_TARGET_LOSS_CAP
    max_weight_ratio: float | None = UK_LOCAL_MAX_WEIGHT_RATIO
    scale_rule: str = "default_target_loss_scales"
    target_weight_rule: str = UK_LOCAL_TARGET_WEIGHT_RULE

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


def uk_local_target_loss_weights(
    grain_labels: Sequence[str],
    *,
    rule: str,
) -> np.ndarray | None:
    """Derive doctrine weights from row-grain labels, never target knobs."""

    if rule not in _ALLOWED_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {_ALLOWED_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    if rule == "uniform":
        return None
    from microcosm.build.uk_runtime.national_doctrine import (
        uk_national_target_loss_weights,
    )

    return uk_national_target_loss_weights(grain_labels, rule="family_equal")


def uk_local_doctrine_with_overrides(
    doctrine: UKLocalSolveDoctrine,
    overrides: Mapping[str, object],
) -> tuple[UKLocalSolveDoctrine, dict[str, dict[str, object]]]:
    """Apply the sole reviewed local override and return its receipt."""

    if not isinstance(doctrine, UKLocalSolveDoctrine):
        raise TypeError("doctrine must be a UKLocalSolveDoctrine instance.")
    fields_by_name = {field.name for field in fields(UKLocalSolveDoctrine)}
    unknown = sorted(set(overrides) - fields_by_name)
    if unknown:
        raise ValueError(
            f"unknown UK local doctrine field(s) {unknown}; frozen fields are "
            f"{sorted(fields_by_name)}."
        )
    frozen = sorted(set(overrides) - _OVERRIDABLE_FIELDS)
    if frozen:
        raise ValueError(
            "UK local doctrine field(s) are reviewed constants, not knobs: "
            f"{frozen}; overridable fields are {sorted(_OVERRIDABLE_FIELDS)}."
        )
    effective_doctrine = replace(doctrine, **overrides)
    receipt: dict[str, dict[str, object]] = {}
    for name in sorted(_OVERRIDABLE_FIELDS):
        default = getattr(doctrine, name)
        effective = getattr(effective_doctrine, name)
        if effective != default:
            receipt[name] = {"default": default, "effective": effective}
    return effective_doctrine, receipt
