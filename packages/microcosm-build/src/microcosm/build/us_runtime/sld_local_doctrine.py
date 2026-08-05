"""US SLD local solve doctrine: one uniform operator, declared bounds.

The doctrine adjudicated on the populace#492/#493 lane and ported from the
UK local solve (populace#503): **no per-target and no per-district
calibration knobs**. The SLD layer solves every district with one operator —
the canonical capped relative-error loss on the default target-defined
scales, uniform target weights, one declared loss cap, one declared
weight-ratio stretch bound anchored at the artifact's calibrated weights.
A miss is support work (membership coverage, thin districts) or target work
(fix or fence the target), never a knob.

This module is the release path's solve surface. The low-level
:func:`populace.build.us_runtime.sld_local_solver.solve_sld_district_weights`
remains the research harness; the doctrine wrappers expose no bound, scale,
or weight parameters — the refusal is structural, not a runtime flag — and
every solve carries the populace#492 past-cap census per district.

The declared values below are the initial reviewed contract, mirroring the
UK local doctrine's declared defaults. The populace#625 pilot adjudicates
them against measured Utah fit; revising either constant is a doctrine
change that must edit this module and its pinned test, which forces review.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from populace.build.us_runtime.sld_local_solver import (
    SldChamberSolveResult,
    SldDistrictProblem,
    SldDistrictSolveResult,
    solve_sld_chamber,
    solve_sld_district_weights,
)

__all__ = [
    "US_SLD_LOCAL_MAX_WEIGHT_RATIO",
    "US_SLD_LOCAL_SOLVE_DOCTRINE",
    "US_SLD_LOCAL_TARGET_LOSS_CAP",
    "UsSldLocalSolveDoctrine",
    "solve_us_sld_chamber_under_doctrine",
    "solve_us_sld_district_weights_under_doctrine",
]

#: Declared uniform loss cap for the SLD local solve (scaled absolute
#: relative-error units on the default target-defined scales).
US_SLD_LOCAL_TARGET_LOSS_CAP = 10.0

#: Declared weight-ratio stretch bound vs the artifact-weight anchor.
US_SLD_LOCAL_MAX_WEIGHT_RATIO = 100.0

_ALLOWED_SCALE_RULES = ("default_target_loss_scales",)
_ALLOWED_TARGET_WEIGHT_RULES = ("uniform",)
_ALLOWED_ANCHOR_RULES = ("artifact_calibrated_weights",)


@dataclass(frozen=True)
class UsSldLocalSolveDoctrine:
    """The declared, reviewed bounds of the uniform SLD solve operator.

    ``scale_rule``, ``target_weight_rule``, and ``anchor_rule`` are closed
    vocabularies: the only admissible scale rule is the canonical
    target-defined default, the only admissible target weighting is uniform,
    and the only admissible stretch anchor is the artifact's calibrated
    weights (single-stage, per populace#493). A future family-level
    weighting would be a new reviewed rule name here — never a per-target
    vector.
    """

    target_loss_cap: float = US_SLD_LOCAL_TARGET_LOSS_CAP
    max_weight_ratio: float | None = US_SLD_LOCAL_MAX_WEIGHT_RATIO
    scale_rule: str = "default_target_loss_scales"
    target_weight_rule: str = "uniform"
    anchor_rule: str = "artifact_calibrated_weights"

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
        if self.anchor_rule not in _ALLOWED_ANCHOR_RULES:
            raise ValueError(
                f"doctrine anchor_rule must be one of {_ALLOWED_ANCHOR_RULES},"
                f" got {self.anchor_rule!r}."
            )

    def as_record(self) -> dict:
        """The doctrine as a JSON-ready record for sidecar provenance."""
        return {
            "target_loss_cap": float(self.target_loss_cap),
            "max_weight_ratio": (
                None if self.max_weight_ratio is None else float(self.max_weight_ratio)
            ),
            "scale_rule": self.scale_rule,
            "target_weight_rule": self.target_weight_rule,
            "anchor_rule": self.anchor_rule,
        }


#: The reviewed doctrine instance every release-path solve uses.
US_SLD_LOCAL_SOLVE_DOCTRINE = UsSldLocalSolveDoctrine()


def _require_uniform_target_surface(problem: SldDistrictProblem) -> None:
    """Refuse a district surface whose rows repeat a metric.

    A duplicated metric row would double that cell's weight in the uniform
    loss — a per-target knob smuggled through the surface.
    """
    duplicated = problem.target_frame.duplicated(["metric"])
    if duplicated.any():
        metrics = sorted(
            problem.target_frame.loc[duplicated, "metric"].astype(str).unique()
        )
        raise ValueError(
            "doctrine solve refuses a non-uniform target surface: duplicate "
            f"metric row(s) {metrics[:5]} in district {problem.area_code} "
            "would act as implicit per-target weights."
        )


def solve_us_sld_district_weights_under_doctrine(
    problem: SldDistrictProblem,
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    min_initial_weight: float = 1e-4,
    seed: int = 0,
) -> SldDistrictSolveResult:
    """Solve one district as the uniform operator with declared bounds.

    Structurally knob-free: no per-target weight, scale, cap, or ratio
    parameters exist on this signature, and no doctrine parameter either —
    the bounds always come from the reviewed module constant
    ``US_SLD_LOCAL_SOLVE_DOCTRINE``, so a caller cannot mint a locally
    revised contract and route it through the release path.
    """
    doctrine = US_SLD_LOCAL_SOLVE_DOCTRINE
    _require_uniform_target_surface(problem)
    return solve_sld_district_weights(
        problem,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=doctrine.max_weight_ratio,
        target_loss_cap=doctrine.target_loss_cap,
        min_initial_weight=min_initial_weight,
        seed=seed,
    )


def solve_us_sld_chamber_under_doctrine(
    problems: list[SldDistrictProblem],
    *,
    epochs: int = 512,
    learning_rate: float = 0.15,
    min_initial_weight: float = 1e-4,
    seed: int = 0,
) -> SldChamberSolveResult:
    """Solve a whole chamber under the reviewed doctrine constants."""
    doctrine = US_SLD_LOCAL_SOLVE_DOCTRINE
    for problem in problems:
        _require_uniform_target_surface(problem)
    return solve_sld_chamber(
        problems,
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=doctrine.max_weight_ratio,
        target_loss_cap=doctrine.target_loss_cap,
        min_initial_weight=min_initial_weight,
        seed=seed,
    )
