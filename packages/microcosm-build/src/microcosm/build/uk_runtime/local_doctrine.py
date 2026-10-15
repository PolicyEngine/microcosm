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
    "UK_LOCAL_DATASET_HOUSEHOLDS",
    "UK_LOCAL_MAX_WEIGHT_RATIO",
    "UK_LOCAL_ROW_METADATA_RULES",
    "UK_LOCAL_SIZE_BASELINE_PI_FLOOR",
    "UK_LOCAL_SIZE_REFIT_L2_ANCHOR",
    "UK_LOCAL_SIZE_REFIT_L2_BASIS",
    "UK_LOCAL_SIZE_REFIT_L2_LAMBDA",
    "UK_LOCAL_SIZE_SELECTION_PI_HI",
    "UK_LOCAL_SOLVE_DOCTRINE",
    "UK_LOCAL_SOLVE_EPOCHS",
    "UK_LOCAL_TARGET_LOSS_CAP",
    "UK_LOCAL_TARGET_WEIGHT_RULE",
    "UK_LOCAL_TARGET_WEIGHT_RULES",
    "UKLocalSolveDoctrine",
    "uk_local_doctrine_with_overrides",
    "uk_local_size_doctrine",
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
#:
#: microcosm#1124 (PR #1163 experiments on the K=25 pool, and the K=25 /
#: 100,000-household build of 2026-10-11): ``nation_grain_family_equal_sqrt_count``
#: — each grain's loss share is split equally across its target families
#: (national rows by nation as a fourth grain), and count rows within a
#: family by the square root of their count. Under ``grain_equal`` the 662
#: HMRC rows took 18.9 % of the loss and the size selection kept a lone-person
#: share of 23 % against the dense solve's 28.6 %; under this rule the selection
#: keeps the composition (28.8 %) and the Northern Ireland share (2.73 %). It
#: dilutes the HMRC top-income bands, which is where the national misses
#: remain (microcosm#1123). ``grain_equal`` and ``uniform`` stay receipted
#: overrides; the other row-metadata rules too.
UK_LOCAL_TARGET_WEIGHT_RULE = "nation_grain_family_equal_sqrt_count"

#: Declared solve length for the UK local solve (Adam epochs).
#:
#: microcosm#762 adjudication (2026-09-03, receipt R10): 1500, María's
#: ruling, matching the national certified-cut posture. The solver has no
#: stopping criterion; at 512 the loss was still falling 2–6 % per 128
#: epochs, at 1500 it is 0.4 % (``uniform``) / 2.8 % (``grain_equal``).
#:
#: microcosm#1124: 2000, the length every #1124 measurement ran at (the
#: 2026-10-06 and 2026-10-11 builds and the #1163 experiments). On the
#: 2026-10-11 build's stored trajectories the dense solve's best loss by
#: epoch 1500 is 3.5 % above its best by 2000, and the size search is still
#: falling at 2000 (7.5 % above at 1500); the refit is within 0.2 % of its
#: best by epoch 500. The K=15 rebuild re-measures it per stage.
UK_LOCAL_SOLVE_EPOCHS = 2000

#: Declared clone count K for the UK local candidate (rows per spine
#: household through the OA ladder).
#:
#: microcosm#762 adjudication (2026-09-03, receipt R9): 15. Smaller K left
#: target cells without supporting households and constituencies under the
#: ESS floor; K=15 cleared the floor at unchanged fit and flat memory
#: (receipts R8-R10 in experiments/762-uk-rowwise-candidate-receipts.md).
#:
#: microcosm#1124 (ruling of 2026-10-13): stays 15. Across the dense pools of
#: the uk-dataset-evaluation reports (``55k-candidates`` and ``55k-rebuild`` at
#: K=15, ``55k-k20`` at K=20, ``k25-60k`` and ``k25-100k`` at K=25), more clones
#: did not lift the weakest seat's support, and the 100,000-household draw's
#: support gain on the K=25 pool came from the size settings below, not from
#: K. A K=15 / 100,000-household build re-measures it.
UK_LOCAL_CLONE_COUNT = 15

#: Target-weighting rules that read more row metadata than the grain
#: (microcosm#1124): the row's target family, its geography and its value
#: basis. They are computed in :mod:`microcosm.build.uk_runtime.target_weights`
#: from one row-aligned carrier, and each is a declared function of that
#: metadata, never a per-target vector. ``nation_grain_family_equal_sqrt_count``
#: is the doctrine's rule; the others are receipted overrides, which a release
#: candidate refuses like any rule but the default.
UK_LOCAL_ROW_METADATA_RULES = (
    "grain_family_equal",
    "grain_family_equal_sqrt_count",
    "nation_grain_family_equal",
    "nation_grain_family_equal_sqrt_count",
)
#: The closed target-weighting vocabulary of the UK local solve.
UK_LOCAL_TARGET_WEIGHT_RULES = ("uniform", "grain_equal", *UK_LOCAL_ROW_METADATA_RULES)

#: Declared defaults of a sized (``--dataset-households``) local build, applied
#: when the request leaves them unset (microcosm#1124; PR #1163's
#: ``SAN_f0.5_u0.022`` and the 2026-10-11 K=25 / 100,000-household build):
#: the exact-k draw's inclusion-probability ceiling, the refit's
#: Horvitz–Thompson baseline floor, and a chi-square L2 penalty on the refit's
#: weights toward their uniform mean. The penalty is a Kish-ESS control (with
#: a uniform anchor it equals n / ESS − 1, so its λ means the same at any
#: size); λ is 10⁻² times the doctrine rule's loss ratio to ``grain_equal``
#: (2.1575). A sized release candidate must use exactly these settings at
#: :data:`UK_LOCAL_DATASET_HOUSEHOLDS`.
UK_LOCAL_SIZE_SELECTION_PI_HI = 0.5
UK_LOCAL_SIZE_BASELINE_PI_FLOOR = 0.5
UK_LOCAL_SIZE_REFIT_L2_LAMBDA = 0.0215748772773509
UK_LOCAL_SIZE_REFIT_L2_ANCHOR = "uniform"
UK_LOCAL_SIZE_REFIT_L2_BASIS = "chi_square"

#: The household count a sized local release ships (microcosm#1124, ruling
#: of 2026-10-13): the local release is the exact-k selection from the
#: K-clone pool at this size, under the dense line's publication
#: requirements (rotated holdout, signed gates, incumbent-surface evaluation
#: on one engine). The pool itself is too large for one engine and for the
#: consumers who load the file. A release candidate may also be unsized
#: (the whole pool); any other size stays candidate-only.
UK_LOCAL_DATASET_HOUSEHOLDS = 100_000


def uk_local_size_doctrine() -> dict[str, object]:
    """The settings a sized local release candidate must carry, by name."""

    return {
        "dataset_households": UK_LOCAL_DATASET_HOUSEHOLDS,
        "selection_pi_hi": UK_LOCAL_SIZE_SELECTION_PI_HI,
        "baseline_pi_floor": UK_LOCAL_SIZE_BASELINE_PI_FLOOR,
        "refit_l2_lambda": UK_LOCAL_SIZE_REFIT_L2_LAMBDA,
        "refit_l2_anchor": UK_LOCAL_SIZE_REFIT_L2_ANCHOR,
        "refit_l2_basis": UK_LOCAL_SIZE_REFIT_L2_BASIS,
        # The selection (search) stage carries no L2 penalty.
        "selection_l2": None,
    }


_ALLOWED_SCALE_RULES = (
    "default_target_loss_scales",
    "default_target_loss_scales_with_family_floors",
)
_ALLOWED_TARGET_WEIGHT_RULES = UK_LOCAL_TARGET_WEIGHT_RULES
_OVERRIDABLE_FIELDS = frozenset({"target_weight_rule"})


@dataclass(frozen=True)
class UKLocalSolveDoctrine:
    """The declared, reviewed bounds of the uniform UK local solve operator.

    ``scale_rule`` and ``target_weight_rule`` are closed vocabularies: the
    admissible scale rules are the canonical target-defined default and that
    default with the reviewed per-family floors (microcosm#1240), and
    the admissible target weightings are ``uniform``, ``grain_equal`` and the
    row-metadata rules of :data:`UK_LOCAL_ROW_METADATA_RULES`, of which
    ``nation_grain_family_equal_sqrt_count`` is the reviewed default and the
    rest are receipted overrides. Any other
    weighting would be a new reviewed rule name here — never a per-target
    vector.
    """

    target_loss_cap: float = UK_LOCAL_TARGET_LOSS_CAP
    max_weight_ratio: float | None = UK_LOCAL_MAX_WEIGHT_RATIO
    #: microcosm#1240 review: the default rule with the family floors of
    #: ``target_weights.UK_LOCAL_LOSS_SCALE_FLOORS``.
    scale_rule: str = "default_target_loss_scales_with_family_floors"
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
    """Derive doctrine weights from row-grain labels, never target knobs.

    Only the grain rules (``uniform``, ``grain_equal``) are computable from
    grain labels alone; a row-metadata rule refuses here and is computed by
    :func:`microcosm.build.uk_runtime.target_weights.uk_target_loss_weights_for_rows`.
    """

    if rule not in _ALLOWED_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {_ALLOWED_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    if rule in UK_LOCAL_ROW_METADATA_RULES:
        raise ValueError(
            f"target_weight_rule {rule!r} derives from row metadata (grain, "
            "family, geography and value basis), not grain labels alone; compute "
            "it with microcosm.build.uk_runtime.target_weights."
            "uk_target_loss_weights_for_rows."
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
