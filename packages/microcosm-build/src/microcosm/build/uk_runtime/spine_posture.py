"""The spine a UK build stands on must be geography-first (microcosm#931).

The dense and national lines build only on a spine that assigns each
geographic clone its area: one that runs every stage
:data:`~microcosm.build.uk_runtime.geography_first_roster.GEOGRAPHY_FIRST_STAGES`
names and every operation ``GEOGRAPHY_FIRST_OPERATIONS`` names, and, where a
build sidecar describes it, records the ``geography`` block under the keys
``SIDECAR_GEOGRAPHY_KEYS`` names. :func:`roster_failures` names what a roster
lacks; :func:`require_geography_first_spine` refuses a spine that lacks
anything. The dense line's preparation calls it right after the engine freeze,
before it pins an input or composes a graph, and again on the roster the build
then binds.

The national line builds on such a spine of one geographic clone:
:func:`require_national_spine` refuses any other bound spine, by its build
sidecar. The national preparation calls it before it pins the input and again
on the authenticated sidecar, and the national problem, the first national
node that reads the bound frame, before it resolves anything. A national build
also needs the degenerate-surface register's ``household.atomic_area_basis``
exclusion signed for the national line and in force on its review date
(:func:`require_national_exclusion_signed`), which the national preparation
checks before it reads any input. The register decides: an entry that lists
the national line, in force on the review date, lifts the refusal with no code
change.

The roster checks read names only: the stages a spine ran or will run, each
stage's operation kinds, and the keys of the sidecar's ``geography`` block,
never their values; the national check also reads the sidecar's geographic
clone count K. Their texts give names, and K, never a row count, an id or a
value. The exclusion check reads one register entry's release lines and
window, and its texts name the exclusion, the line and the window's dates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from microcosm.build.uk_runtime.geography_first_roster import (
    GEOGRAPHY_FIRST_OPERATIONS,
    GEOGRAPHY_FIRST_STAGES,
    SIDECAR_GEOGRAPHY_KEYS,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    load_uk_reviewed_exclusion_register,
)

__all__ = [
    "NATIONAL_AREA_EXCLUSION",
    "NATIONAL_CLONE_COUNT",
    "NATIONAL_LINE",
    "require_geography_first_spine",
    "require_national_exclusion_signed",
    "require_national_spine",
    "roster_failures",
    "spec_roster",
    "stage_roster",
]

#: The release line a national build runs on, as a gate context and a
#: reviewed exclusion's ``release_lines`` name it.
NATIONAL_LINE = "national"
#: The geographic clone count the national line builds at (microcosm#931).
NATIONAL_CLONE_COUNT = 1
#: The degenerate-surface exclusion that must be signed for the national line
#: before a national build runs (microcosm#931), by its register entry's name.
NATIONAL_AREA_EXCLUSION = "household.atomic_area_basis"


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


def require_national_spine(sidecar: object) -> None:
    """Refuse a bound spine the national line cannot build on (microcosm#931).

    ``sidecar`` is the bound spine's build sidecar, as JSON reads it. The
    national line builds on a geography-first spine of one geographic clone:
    the sidecar's ``geographic_clone_count`` must be the int one (a sidecar
    that records none describes a spine of one clone, as the checkpoint
    authentication reads it), and its ``stages``, ``operations`` and
    ``geography`` block must pass :func:`roster_failures`. The refusal is one
    sentence that names K and what the roster lacks, and nothing else.
    """

    if not isinstance(sidecar, Mapping):
        raise ValueError(
            "The national line builds only on a bound spine whose build "
            "sidecar is a JSON object (microcosm#931)."
        )
    problems: list[str] = []
    clones = sidecar.get("geographic_clone_count", NATIONAL_CLONE_COUNT)
    if type(clones) is not int or clones != NATIONAL_CLONE_COUNT:
        problems.append(f"the bound spine's geographic clone count is K = {clones!r}")
    failures = roster_failures(
        sidecar.get("stages"), sidecar.get("operations"), sidecar.get("geography")
    )
    if failures:
        problems.append(f"the bound spine's build sidecar lacks: {'; '.join(failures)}")
    if problems:
        raise ValueError(
            "The national line builds only on a geography-first spine of one "
            "geographic clone, one that assigns every household its area "
            f"(microcosm#931); {'; '.join(problems)}."
        )


def require_national_exclusion_signed(
    review_date: date, *, source: str | Path | None = None
) -> None:
    """Refuse a national build until its area-basis exclusion is signed for it.

    Loads the degenerate-surface register (the committed one, or ``source``,
    a register file of the same schema) through
    :func:`~microcosm.build.uk_runtime.weighted_integrity.load_uk_reviewed_exclusion_register`
    and requires its ``household.atomic_area_basis`` entry to apply on the
    national line (``UKReviewedExclusion.applies_on``; an entry that names no
    line applies on every line) and to be in force on ``review_date``, neither
    expired nor premature: the build's own review date (``--review-date``),
    the date the national graph's target and gate nodes evaluate their
    registers on (microcosm#931). No gate of the national build reads this
    register; the release cut reads it again, on the day it certifies, and
    fails there on an entry that has lapsed since. The register decides: once
    the entry lists the national line and is in force, the check passes with
    no code change.

    ``review_date`` must be a ``datetime.date`` and never a ``datetime``; the
    check reads no clock of its own. The refusals name the exclusion, the
    national line and the entry's window, and nothing else.
    """

    if isinstance(review_date, datetime) or not isinstance(review_date, date):
        raise TypeError(
            "The national exclusion check needs the build's review date as a "
            f"datetime.date, got {type(review_date).__name__}."
        )
    records = load_uk_reviewed_exclusion_register(
        source, resource=UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
    )
    lead = (
        "The national line builds only once the degenerate-surface exclusion "
        f"{NATIONAL_AREA_EXCLUSION} is signed for the {NATIONAL_LINE} line and "
        "in force on the build's review date (microcosm#931); "
    )
    record = records.get(NATIONAL_AREA_EXCLUSION)
    if record is None:
        raise ValueError(lead + "the register has no such entry.")
    window = f"from {record.approved_on} through {record.expires_on}"
    if not record.applies_on(NATIONAL_LINE):
        raise ValueError(
            lead + f"its entry, in force {window}, does not list the "
            f"{NATIONAL_LINE} line."
        )
    if record.expired(review_date):
        raise ValueError(lead + f"its entry has expired: it was in force {window}.")
    if record.premature(review_date):
        raise ValueError(
            lead + f"its entry is not yet in force: it is in force {window}."
        )
