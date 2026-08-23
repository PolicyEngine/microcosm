"""UK national calibration doctrine: reviewed solve constants for #623.

The national release path exposes no calibration knobs. Every solver option
that the first Ledger-backed national calibration currently inherits is
declared here so a future change must edit this module and its pinned tests,
forcing review before any armed run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "UK_NATIONAL_LEARNING_RATE",
    "UK_NATIONAL_L0_LAMBDA",
    "UK_NATIONAL_MASS_RULE",
    "UK_NATIONAL_MAX_WEIGHT_RATIO",
    "UK_NATIONAL_SEED",
    "UK_NATIONAL_SOLVE_DOCTRINE",
    "UK_NATIONAL_SOLVE_EPOCHS",
    "UK_NATIONAL_TARGET_LOSS_CAP",
    "UK_NATIONAL_TARGET_WEIGHT_RULE",
    "uk_national_target_loss_weights",
    "UKNationalSolveDoctrine",
]

# 256 was the promoted kernel default the doctrine was first declared with
# (verdict ABSENT — never measured). Raised to 1500 by Maria's adjudication
# on 2026-08-23, informed by the first armed-run receipts: at 256 the solve
# left recoverable targets unconverged (UC caseload reached 5.10m of 6.76m
# with headroom inside the weight-ratio bound). The incumbent's own solve
# runs 512 epochs on a smaller national surface.
UK_NATIONAL_SOLVE_EPOCHS = 1500
UK_NATIONAL_LEARNING_RATE = 0.02
UK_NATIONAL_MAX_WEIGHT_RATIO = 10.0
UK_NATIONAL_SEED = 0
UK_NATIONAL_TARGET_LOSS_CAP = 10.0
UK_NATIONAL_L0_LAMBDA = 0.0
UK_NATIONAL_MASS_RULE = "free"
UK_NATIONAL_TARGET_WEIGHT_RULE = "family_equal"

_ALLOWED_SCALE_RULES = ("default_target_loss_scales",)

# "family_equal" gives every declared target family one equal share of the
# objective, split evenly within the family — the reviewed rule name the
# local doctrine reserved for exactly this ("a future family-level weighting
# would be a new reviewed rule name here — never a per-target vector").
#
# Adjudicated on 2026-08-23 from the first armed-run receipts. Under
# "uniform" the hmrc SPI income-band family is 57% of the target surface and
# supplied 101 of the 102 references that start past the loss cap; pursuing
# them annihilated 24 population and OBR targets that were never past cap
# themselves. Measured on the run-3 diagnostics, the objective's gain from
# annihilating a segment is +4.48 under uniform weights and +0.68 under
# family_equal at cap 10, turning negative (-0.09) at cap 2.
_ALLOWED_TARGET_WEIGHT_RULES = ("uniform", "family_equal")
_ALLOWED_MASS_RULES = ("free",)


@dataclass(frozen=True)
class UKNationalSolveDoctrine:
    """The declared constants of the UK national calibration solve."""

    epochs: int = UK_NATIONAL_SOLVE_EPOCHS
    learning_rate: float = UK_NATIONAL_LEARNING_RATE
    max_weight_ratio: float | None = UK_NATIONAL_MAX_WEIGHT_RATIO
    seed: int = UK_NATIONAL_SEED
    target_loss_cap: float = UK_NATIONAL_TARGET_LOSS_CAP
    scale_rule: str = "default_target_loss_scales"
    target_weight_rule: str = UK_NATIONAL_TARGET_WEIGHT_RULE
    mass_rule: str = UK_NATIONAL_MASS_RULE
    l0_lambda: float = UK_NATIONAL_L0_LAMBDA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.epochs, int)
            or isinstance(self.epochs, bool)
            or self.epochs <= 0
        ):
            raise ValueError(
                f"doctrine epochs must be a positive integer, got {self.epochs!r}."
            )
        if (
            not isinstance(self.learning_rate, int | float)
            or isinstance(self.learning_rate, bool)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "doctrine learning_rate must be a positive finite number, "
                f"got {self.learning_rate!r}."
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
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError(
                f"doctrine seed must be a non-negative integer, got {self.seed!r}."
            )
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
        if self.mass_rule not in _ALLOWED_MASS_RULES:
            raise ValueError(
                f"doctrine mass_rule must be one of {_ALLOWED_MASS_RULES}, "
                f"got {self.mass_rule!r}."
            )
        if (
            not isinstance(self.l0_lambda, int | float)
            or isinstance(self.l0_lambda, bool)
            or not np.isfinite(self.l0_lambda)
            or self.l0_lambda < 0
        ):
            raise ValueError(
                "doctrine l0_lambda must be a non-negative finite number, "
                f"got {self.l0_lambda!r}."
            )


UK_NATIONAL_SOLVE_DOCTRINE = UKNationalSolveDoctrine()


def uk_national_target_loss_weights(
    families: Sequence[str],
    *,
    rule: str = UK_NATIONAL_TARGET_WEIGHT_RULE,
) -> np.ndarray | None:
    """The declared target-weight vector for one compiled register.

    ``uniform`` returns ``None`` — the kernel's own equal weighting, and the
    identity this doctrine shipped with. ``family_equal`` gives each declared
    family one equal share of the objective, divided evenly among its
    members, so an over-supplied family cannot outvote the rest of the
    surface by sheer count.

    The vector is derived from the compiled register's declared families and
    nothing else: it is a rule, not a per-target vector, which is the
    property the local doctrine requires of any weighting.
    """

    if rule not in _ALLOWED_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {_ALLOWED_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    if rule == "uniform":
        return None
    labels = [str(family) for family in families]
    if not labels:
        raise ValueError("family_equal weighting requires a non-empty register.")
    if any(not label for label in labels):
        raise ValueError("every target must declare a family for family_equal.")
    counts = Counter(labels)
    family_count = len(counts)
    return np.asarray(
        [1.0 / (family_count * counts[label]) for label in labels],
        dtype=np.float64,
    )
