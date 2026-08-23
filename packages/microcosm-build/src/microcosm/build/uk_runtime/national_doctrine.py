"""UK national calibration doctrine: reviewed solve constants for #623.

The national release path exposes no calibration knobs. Every solver option
that the first Ledger-backed national calibration currently inherits is
declared here so a future change must edit this module and its pinned tests,
forcing review before any armed run.
"""

from __future__ import annotations

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

_ALLOWED_SCALE_RULES = ("default_target_loss_scales",)
_ALLOWED_TARGET_WEIGHT_RULES = ("uniform",)
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
    target_weight_rule: str = "uniform"
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
