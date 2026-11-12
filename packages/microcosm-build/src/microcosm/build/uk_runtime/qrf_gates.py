"""The QRF sign-gate calibration each UK stage declares, and what each gate got.

A stage operation that fits a :class:`~microcosm.fit.RegimeGatedQRF` names its
``gate_calibration`` in ``uk/spec/sources.yaml`` (microcosm#1359). An
undeclared operation keeps the in-sample gate. The fitted model says which
calibration each target's gate actually has
(:meth:`~microcosm.fit.FittedRegimeGatedQRF.gate_outcomes`), and the stages
record that in their evidence under :data:`UK_GATE_OUTCOMES_EVIDENCE`, so a
target that kept the in-sample gate or fell back to the class rates is seen.
An operation may also name ``gate_level_columns``, predictors whose levels a
cross-fitted calibration offsets on their own (microcosm#1276); the levels each
gate got are recorded under :data:`UK_GATE_LEVELS_EVIDENCE`.
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
#: The stage-evidence key holding the predictor levels each gate offsets.
UK_GATE_LEVELS_EVIDENCE = "gate_levels"


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


def declared_gate_level_columns(
    stage: SourceStageSpec, kind: str, **match: object
) -> tuple[str, ...]:
    """The ``gate_level_columns`` the stage's ``kind`` operation declares.

    The operation's own predictor names, in declared order; an operation that
    declares none has none. ``match`` narrows as in
    :func:`declared_gate_calibration`.
    """

    for operation in stage.operations:
        if operation.kind == kind and all(
            operation.parameters.get(key) == value for key, value in match.items()
        ):
            columns = tuple(operation.parameters.get("gate_level_columns", ()))
            if not all(isinstance(column, str) for column in columns):
                raise ValueError(
                    f"{stage.stage} {kind} gate_level_columns must name columns as "
                    f"strings, got {list(columns)!r}."
                )
            if len(set(columns)) != len(columns):
                raise ValueError(
                    f"{stage.stage} {kind} gate_level_columns repeats a column: "
                    f"{list(columns)!r}."
                )
            return columns
    return ()


def gate_levels(fit_name: str, fitted: Any) -> dict[str, list[str]]:
    """``{"<fit_name>:<target>": [level, ...]}`` for each target with level offsets.

    Each level is ``"<encoded column>=<value>"``
    (:meth:`~microcosm.fit.FittedRegimeGatedQRF.gate_levels`); a target whose
    gate offsets no level of its own is left out.
    """

    return {
        f"{fit_name}:{target}": list(levels)
        for target, levels in fitted.gate_levels().items()
        if levels
    }


def pooled_gate_levels(fit_name: str, fits: Sequence[Any]) -> dict[str, dict[str, int]]:
    """Per target, how many of ``fits`` gave each level offsets of its own.

    For fits of the same targets, the folds of a cross-fitted reference say,
    where a level can be thin in some folds and not in others.
    """

    counts: dict[str, Counter[str]] = {}
    for fitted in fits:
        for key, levels in gate_levels(fit_name, fitted).items():
            counts.setdefault(key, Counter()).update(levels)
    return {key: dict(sorted(count.items())) for key, count in counts.items()}


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
