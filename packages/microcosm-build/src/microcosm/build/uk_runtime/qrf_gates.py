"""The QRF sign-gate calibration each UK stage declares, and what each gate got.

A stage operation that fits a :class:`~microcosm.fit.RegimeGatedQRF` names its
``gate_calibration`` in ``uk/spec/sources.yaml`` (microcosm#1359). An
undeclared operation keeps the in-sample gate. The fitted model says which
calibration each target's gate actually has
(:meth:`~microcosm.fit.FittedRegimeGatedQRF.gate_outcomes`), and the stages
record that in their evidence under :data:`UK_GATE_OUTCOMES_EVIDENCE`, so a
target that kept the in-sample gate or fell back to the class rates is seen.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from microcosm.build.source_manifest import SourceStageSpec

#: The calibration an operation that declares none keeps.
UK_DEFAULT_GATE_CALIBRATION = "in_sample"
#: The stage-evidence key holding each fit's gate outcomes.
UK_GATE_OUTCOMES_EVIDENCE = "gate_outcomes"


def declared_gate_calibration(
    stage: SourceStageSpec, kind: str, **match: object
) -> str:
    """The ``gate_calibration`` the stage's ``kind`` operation declares.

    ``match`` narrows to the operation whose parameters carry those values,
    for a stage that declares the kind more than once.
    """

    for operation in stage.operations:
        if operation.kind == kind and all(
            operation.parameters.get(key) == value for key, value in match.items()
        ):
            return str(
                operation.parameters.get(
                    "gate_calibration", UK_DEFAULT_GATE_CALIBRATION
                )
            )
    return UK_DEFAULT_GATE_CALIBRATION


def gate_outcomes(fit_name: str, fitted: Any) -> dict[str, str]:
    """``{"<fit_name>:<target>": outcome}`` for each gated target of a fit."""

    outcomes: Mapping[str, str | None] = fitted.gate_outcomes()
    return {
        f"{fit_name}:{target}": outcome
        for target, outcome in outcomes.items()
        if outcome is not None
    }


def pooled_gate_outcomes(fit_name: str, fits: Sequence[Any]) -> dict[str, str]:
    """One :func:`gate_outcomes` entry per target over fits of the same targets.

    A target whose fits (the folds of a cross-fitted reference, say) all got
    one outcome names it; otherwise the entry counts the fits behind each
    outcome, as ``"cross_fitted 4, in_sample_thin_class 1"``.
    """

    counts: dict[str, Counter[str]] = {}
    for fitted in fits:
        for key, outcome in gate_outcomes(fit_name, fitted).items():
            counts.setdefault(key, Counter())[outcome] += 1
    return {
        key: (
            next(iter(count))
            if len(count) == 1
            else ", ".join(f"{outcome} {n}" for outcome, n in sorted(count.items()))
        )
        for key, count in counts.items()
    }
