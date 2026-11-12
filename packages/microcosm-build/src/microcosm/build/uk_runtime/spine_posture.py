"""The spine a UK dense build stands on must be geography-first (microcosm#931).

The dense line builds only on a spine that assigns each geographic clone its
area: one that runs every stage
:data:`~microcosm.build.uk_runtime.geography_first_roster.GEOGRAPHY_FIRST_STAGES`
names and every operation ``GEOGRAPHY_FIRST_OPERATIONS`` names, and, where a
build sidecar describes it, records the ``geography`` block under the keys
``SIDECAR_GEOGRAPHY_KEYS`` names. :func:`roster_failures` names what a roster
lacks; :func:`require_geography_first_spine` refuses a spine that lacks
anything. The dense line's preparation calls it right after the engine freeze,
before it pins an input or composes a graph, and again on the roster the build
then binds.

The check reads names only: the stages a spine ran or will run, each stage's
operation kinds, and the keys of the sidecar's ``geography`` block, never their
values. Its texts give names only, never a count, an id or a value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from microcosm.build.uk_runtime.geography_first_roster import (
    GEOGRAPHY_FIRST_OPERATIONS,
    GEOGRAPHY_FIRST_STAGES,
    SIDECAR_GEOGRAPHY_KEYS,
)

__all__ = [
    "require_geography_first_spine",
    "roster_failures",
    "spec_roster",
    "stage_roster",
]


def stage_roster(
    stages: Iterable[object],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """The names of declared ``stages``, in their order, and each one's
    operation kinds: the names a build sidecar records under ``stages`` and
    ``operations``."""

    declared = tuple(stages)
    names = tuple(stage.stage for stage in declared)
    operations = {
        stage.stage: tuple(operation.kind for operation in stage.operations)
        for stage in declared
    }
    return names, operations


def spec_roster(spec) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    """:func:`stage_roster` of the stages a country spec declares, in its
    order."""

    if spec.sources is None:
        raise ValueError("The country spec declares no source stages.")
    return stage_roster(spec.sources.stages)


def _names(value: object) -> tuple[str, ...] | None:
    """``value`` as a sequence of names, or ``None`` if it is not one."""

    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    if not all(isinstance(item, str) for item in value):
        return None
    return tuple(value)


def roster_failures(
    stages: object,
    operations: object,
    geography: object = None,
    *,
    from_sidecar: bool = True,
) -> tuple[str, ...]:
    """The geography-first names this roster lacks, in the leaf's order.

    ``stages`` and ``operations`` are a sidecar's fields of those names, or a
    spine request's roster; of ``geography``, the sidecar's block, only the
    keys are read. Failures read ``stage <name>``, ``operation <kind> in
    <stage>``, ``geography block`` and ``geography key <key>``; a roster that
    is not names fails as ``stage roster`` or ``operation roster``, and a
    stage whose kinds are not names holds none. An operation counts only in a
    stage the roster ran; extra names never fail. With ``from_sidecar=False``
    no block is read, and passing one raises.
    """

    if not from_sidecar and geography is not None:
        raise ValueError(
            "A roster read without a sidecar has no geography block to check."
        )
    failures: list[str] = []
    names = _names(stages)
    if names is None:
        failures.append("stage roster")
    else:
        ran = set(names)
        failures.extend(
            f"stage {stage}" for stage in GEOGRAPHY_FIRST_STAGES if stage not in ran
        )
        if not isinstance(operations, Mapping):
            failures.append("operation roster")
        else:
            for stage, required in GEOGRAPHY_FIRST_OPERATIONS.items():
                kinds = _names(operations.get(stage)) if stage in ran else None
                failures.extend(
                    f"operation {kind} in {stage}"
                    for kind in required
                    if kind not in (kinds or ())
                )
    if from_sidecar:
        if not isinstance(geography, Mapping):
            failures.append("geography block")
        else:
            failures.extend(
                f"geography key {key}"
                for key in SIDECAR_GEOGRAPHY_KEYS
                if key not in geography
            )
    return tuple(failures)


def require_geography_first_spine(
    stages: object,
    operations: object,
    geography: object = None,
    *,
    from_sidecar: bool = True,
) -> None:
    """Refuse a spine whose roster lacks any geography-first name.

    The arguments are :func:`roster_failures`'. The refusal is one sentence
    that names what the roster lacks and nothing else.
    """

    failures = roster_failures(stages, operations, geography, from_sidecar=from_sidecar)
    if failures:
        where = (
            "the bound spine's build sidecar"
            if from_sidecar
            else "the spine request's stage roster"
        )
        raise ValueError(
            "The dense line builds only on a geography-first spine, one that "
            "assigns every geographic clone its area (microcosm#931); "
            f"{where} lacks: {'; '.join(failures)}."
        )
