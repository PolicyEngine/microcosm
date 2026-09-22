"""Declare the US SPM measurement universe from each source's record type.

PolicyEngine/policyengine-us#9462 makes the SPM measurement universe a
**source declaration**: a dataset supplies ``spm_unit_spm_universe_status`` in
``{INCLUDED, OUTSIDE, UNRESOLVED}`` for every SPM unit and year, and the model
never infers scope from composition. This module is Microcosm's producer for
that input.

It is a source declaration, not a model. Nothing here fits, draws, ages,
weights, or imputes: the status is read off ``TYPEHUGQ`` — the ACS PUMS
household record type the source already puts on the frame's household table
(``acs_pums.py`` ``_HOUSEHOLD_FRAME_COLUMNS``) — and off the record's
provenance channel. ``OUTSIDE`` changes no weight, no membership, and no row's
presence in the file; it declares only that the source does not measure that
unit's SPM poverty.

The rule
--------

===============================================  ==========
ACS housing unit (``TYPEHUGQ == 1``)             INCLUDED
ACS group quarters (``TYPEHUGQ`` in ``{2, 3}``)  OUTSIDE
ASEC record                                      INCLUDED
===============================================  ==========

Both arms come from Census SEHSD-WP2020-09 pages 6-7 (see
:data:`SPM_UNIVERSE_METHODOLOGY`), and they are not symmetric. The ACS
exclusion is a *data limitation* Census states explicitly — the public PUMS
cannot separate institutional from noninstitutional group quarters, so the ACS
SPM sample is limited to people living in households. The ASEC ruling is the
opposite shape: the CPS ASEC *sample frame* already equals the CPS SPM
universe, so there is no ASEC record outside it to declare. That ruling is
therefore made at the spine, not per record, and carries its own refusal
(:data:`ASEC_RECORD_TYPE_FIELDS`) so that a future ASEC vintage which admits a
record-type field cannot be silently declared in-universe.

``UNRESOLVED`` is the engine's *absence* sentinel, not a value a producer
emits. A producer that cannot decide refuses; it never writes ``UNRESOLVED``.

Never an inference
------------------

The ACS problem was discovered as SPM units with no classified adult, and it
would be easy — and wrong — to mark zero-adult units ``OUTSIDE``. That would
brand a genuine household-unit data defect as out-of-universe and hide it, and
it would make the universe move whenever the role column moved. The input is
``TYPEHUGQ``, which the source fixes before any composition is computed. The
anti-inference property is pinned by test: a zero-adult *housing* unit is
``INCLUDED``.

Never a spine-agreement surface
-------------------------------

A new categorical ``spm_unit`` column looks like something that belongs in
``spine_agreement``. It does not. The universe status is *intended* to differ
by spine — that is its entire content — and registering it there would assert
that the ACS and ASEC arms should agree about a fact they are defined to
disagree about. The status must never become a declared transfer, a derived
transfer, a take-up contract entry or a simulated output, which is what would
be required to admit it to that gate.

Never a target
--------------

The status is not a calibration target, not a selection criterion, and not a
gate. Poverty in this repository is a comparison only.

Import weight
-------------

This producer is a leaf: a preflight or a release tool must be able to call it
without dragging the ACS loader into the process. Both
``microcosm.build.us_runtime.acs_pums`` and ``...stacked_spine`` pull the
country engine: re-measured in separate processes on 2026-09-22 after merging
main, each import took ~36-38 s on a warm cache (the authoring lane measured
~90 s for ``stacked_spine`` cold). So the four channel
tag values below are named here rather than imported from those modules. They
are not a second source of truth: :mod:`test_us_spm_universe_source` parses
each canonical definition out of its defining module with :mod:`ast` — no
import — and fails if this table drifts from it. The one canonical constant
that is free to import (:data:`BASE_ASEC_SUPPORT_CHANNEL`) is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    support_channel_column,
    support_clone_index_column,
)

__all__ = [
    "ACS_ARM",
    "ACS_GROUP_QUARTERS_KINDS",
    "ACS_HOUSEHOLD_KIND_COLUMN",
    "ACS_HOUSEHOLD_KINDS",
    "ACS_HOUSING_UNIT_KIND",
    "ACS_SPM_OUTSIDE_AUTHORITY",
    "ASEC_ARM",
    "ASEC_NATIVE_UNIT_COLUMN",
    "ASEC_RECORD_TYPE_FIELDS",
    "ASEC_SPM_INCLUDED_AUTHORITY",
    "INCLUDED",
    "OUTSIDE",
    "SPM_UNIVERSE_CHANNEL_ARMS",
    "SPM_UNIVERSE_METHODOLOGY",
    "SPM_UNIVERSE_PROFILE_NAME",
    "SPM_UNIVERSE_REFUSALS",
    "SPM_UNIVERSE_RULE",
    "SPM_UNIVERSE_STATUSES",
    "SpmUniverseClassification",
    "SpmUniverseSourceResult",
    "UNIVERSE_INPUT",
    "UNRESOLVED",
    "attach_spm_universe_status",
    "classify_spm_universe",
]

#: The engine input this producer supplies (policyengine-us#9462). The name
#: and the three-state vocabulary below are deliberately identical to the
#: graph-native line's ``spm_input_contract`` (unmerged native-SPM branches,
#: first at commit 774a90091; not on main); there is no reason for two
#: spellings of the same declaration.
UNIVERSE_INPUT = "spm_unit_spm_universe_status"

INCLUDED = "INCLUDED"
OUTSIDE = "OUTSIDE"
#: The engine's absence sentinel. Read back from a defaulted dataset; never
#: written by this producer.
UNRESOLVED = "UNRESOLVED"
SPM_UNIVERSE_STATUSES = (INCLUDED, OUTSIDE, UNRESOLVED)

#: The two source universes this producer rules on.
ACS_ARM = "acs"
ASEC_ARM = "asec"

#: The ACS PUMS household record type, as the source stages it.
ACS_HOUSEHOLD_KIND_COLUMN = "TYPEHUGQ"
ACS_HOUSING_UNIT_KIND = 1
ACS_GROUP_QUARTERS_KINDS = (2, 3)
ACS_HOUSEHOLD_KINDS = (ACS_HOUSING_UNIT_KIND, *ACS_GROUP_QUARTERS_KINDS)

#: The ASEC native SPM unit id. Its presence and completeness is what proves
#: the frame's SPM partition is the native one rather than the documented
#: household fallback (``microcosm.frame.units.assign_us_unit_structure``).
ASEC_NATIVE_UNIT_COLUMN = "SPM_ID"

#: ASEC household record-type fields. Microcosm reads none of them at this
#: head, which is why the ASEC ruling is made at the spine. If one ever
#: arrives on a frame, the spine-level ruling has to be re-derived against it
#: rather than silently applied, so its presence refuses.
ASEC_RECORD_TYPE_FIELDS = ("H_TYPE", "HRHTYPE", "H_HHTYPE", "HUNITS", "H_LIVQRT")

SPM_UNIVERSE_METHODOLOGY = (
    "Census SEHSD-WP2020-09, pages 6-7: all people living in group quarters "
    "are excluded from the ACS SPM because the public ACS PUMS cannot "
    "distinguish noninstitutional group quarters, while the CPS ASEC sample "
    "frame already equals the CPS SPM universe."
)
#: Named authority for the ACS exclusion; identical to the graph-native line's
#: ``acs_spm_scope.ACS_SPM_OUTSIDE_AUTHORITY`` (same unmerged branches).
ACS_SPM_OUTSIDE_AUTHORITY = "outside_acs_household_universe"
#: Named authority for the ASEC ruling. This one has no upstream: it is this
#: producer's own source-universe judgement, made on the Census quote above
#: plus the zero-adult reconciliation ``spm_role_source`` already enforces
#: against the pinned complete Census ASEC files. Signed off as this named,
#: cited constant by Max Ghenis on 2026-09-22 (microcosm#977).
ASEC_SPM_INCLUDED_AUTHORITY = "asec_sample_frame_equals_cps_spm_universe"
#: The named analysis scope a release declares when it carries this status.
SPM_UNIVERSE_PROFILE_NAME = "us_acs_household_and_asec_frame_universe_v1"

SPM_UNIVERSE_RULE = (
    "TYPEHUGQ == 1 -> INCLUDED; TYPEHUGQ in {2, 3} -> OUTSIDE; ASEC record -> INCLUDED"
)

#: Provenance tag values this producer has a universe ruling for. Anything
#: else refuses; see the module docstring on why these are named rather than
#: imported.
SPM_UNIVERSE_CHANNEL_ARMS: Mapping[str, str] = MappingProxyType(
    {
        # microcosm.build.us_runtime.acs_pums.ACS_2024_1YR_SPINE
        "acs_2024_1yr": ACS_ARM,
        # microcosm.build.us_runtime.stacked_spine.ACS_STACKED_SUPPORT_CHANNEL
        "acs": ACS_ARM,
        # microcosm.build.us_runtime.base_pool.ASEC_PUF_SPINE
        "asec_puf": ASEC_ARM,
        BASE_ASEC_SUPPORT_CHANNEL: ASEC_ARM,
    }
)

#: The household column naming each row's base spine
#: (``base_pool.spine_column("household")``), then the assembled support
#: channel, in the order the producer looks for them.
HOUSEHOLD_SPINE_COLUMN = "household_spine"
HOUSEHOLD_SUPPORT_CHANNEL_COLUMN = support_channel_column("household")
_CHANNEL_COLUMN_ORDER = (HOUSEHOLD_SPINE_COLUMN, HOUSEHOLD_SUPPORT_CHANNEL_COLUMN)
_PERSON_CLONE_INDEX_COLUMN = support_clone_index_column("person")
_NATIVE_CLONE_INDEX = 0

SPM_UNIVERSE_REFUSALS = (
    "SPM_UNIVERSE_COLUMN_EXISTS",
    "SPM_UNIVERSE_UNDECLARED_SPINE",
    "SPM_UNIVERSE_UNKNOWN_HOUSEHOLD_KIND",
    "SPM_UNIVERSE_OFF_ARM_HOUSEHOLD_KIND",
    "SPM_UNIVERSE_MIXED_HOUSEHOLD",
    "SPM_UNIVERSE_GQ_MULTI_PERSON",
    "SPM_UNIVERSE_ORPHAN_MEMBER",
    "SPM_UNIVERSE_EMPTY_UNIT",
    "SPM_UNIVERSE_UNIT_SPANS_KINDS",
    "SPM_UNIVERSE_DEGRADED_PARTITION",
    "SPM_UNIVERSE_ASEC_RECORD_TYPE_UNREVIEWED",
)

_MAX_REPORTED_EXAMPLES = 5


@dataclass(frozen=True)
class SpmUniverseClassification:
    """One frame's universe declaration, plus the evidence it was made from."""

    #: Member-name strings in ``unit_ids`` order.
    status: np.ndarray
    #: ``unit_ids`` as classified, in the order given.
    unit_ids: np.ndarray
    #: ``ACS_ARM``/``ASEC_ARM`` per household row.
    household_arm: pd.Series
    #: ACS record type per household row; missing on the ASEC arm.
    household_kind: pd.Series
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SpmUniverseSourceResult:
    """A frame carrying the declaration, with the classification behind it."""

    frame: Any
    classification: SpmUniverseClassification

    @property
    def status(self) -> np.ndarray:
        return self.classification.status

    @property
    def provenance(self) -> dict[str, Any]:
        return self.classification.provenance


def _require(condition: bool, code: str, detail: str) -> None:
    """Refuse, named and fail-closed, in ``spm_role_source``'s idiom."""
    if not condition:
        raise ValueError(f"{code}: {detail}")


def _examples(values: Any) -> str:
    unique = pd.unique(np.asarray(values, dtype=object))
    shown = ", ".join(repr(value) for value in unique[:_MAX_REPORTED_EXAMPLES])
    if len(unique) > _MAX_REPORTED_EXAMPLES:
        shown = f"{shown}, ... ({len(unique)} distinct)"
    return shown


def _resolve_channel(
    household: pd.DataFrame,
    *,
    declared_channel: str | None,
    channel_column: str | None,
) -> tuple[pd.Series, str]:
    """Return each household's provenance tag and where it came from.

    Resolution order is explicit and fail-closed: an operator-named column, the
    base-spine tag, the assembled support channel, then a caller declaration
    for a frame that carries no tag at all. A caller may not declare over a
    frame that already declares for itself.
    """
    present = [name for name in _CHANNEL_COLUMN_ORDER if name in household.columns]
    if channel_column is not None:
        _require(
            channel_column in household.columns,
            "SPM_UNIVERSE_UNDECLARED_SPINE",
            f"the household table carries no requested channel column "
            f"{channel_column!r}; present: {sorted(household.columns)[:20]}.",
        )
        _require(
            declared_channel is None,
            "SPM_UNIVERSE_UNDECLARED_SPINE",
            "declared_channel and channel_column are mutually exclusive.",
        )
        return household[channel_column].astype(object), channel_column
    if present:
        _require(
            declared_channel is None,
            "SPM_UNIVERSE_UNDECLARED_SPINE",
            f"the household table already declares its own provenance in "
            f"{present[0]!r}; a caller declaration would override it.",
        )
        return household[present[0]].astype(object), present[0]
    _require(
        declared_channel is not None,
        "SPM_UNIVERSE_UNDECLARED_SPINE",
        "the household table carries no provenance tag "
        f"({' or '.join(_CHANNEL_COLUMN_ORDER)}) and the caller declared no "
        "channel, so no universe ruling applies to these rows.",
    )
    return (
        pd.Series(declared_channel, index=household.index, dtype=object),
        f"declared:{declared_channel}",
    )


def _household_arms(channel: pd.Series) -> pd.Series:
    arm = channel.map(SPM_UNIVERSE_CHANNEL_ARMS)
    unruled = channel[arm.isna()]
    _require(
        unruled.empty,
        "SPM_UNIVERSE_UNDECLARED_SPINE",
        f"{len(unruled)} household row(s) carry a provenance channel this "
        f"producer has no universe ruling for: {_examples(unruled)}; ruled "
        f"channels: {sorted(SPM_UNIVERSE_CHANNEL_ARMS)}.",
    )
    return arm.astype(object)


def _household_kinds(household: pd.DataFrame, arm: pd.Series) -> pd.Series:
    """Validate the ACS record type and its confinement to the ACS arm."""
    if ACS_HOUSEHOLD_KIND_COLUMN in household.columns:
        kind = pd.to_numeric(
            household[ACS_HOUSEHOLD_KIND_COLUMN], errors="coerce"
        ).astype(float)
    else:
        kind = pd.Series(np.nan, index=household.index, dtype=float)
    acs = arm.eq(ACS_ARM).to_numpy()
    unknown = acs & ~kind.isin(ACS_HOUSEHOLD_KINDS).to_numpy()
    if ACS_HOUSEHOLD_KIND_COLUMN in household.columns:
        offending = _examples(household.loc[unknown, ACS_HOUSEHOLD_KIND_COLUMN])
    else:
        offending = "<column absent>"
    _require(
        not unknown.any(),
        "SPM_UNIVERSE_UNKNOWN_HOUSEHOLD_KIND",
        f"{int(unknown.sum())} ACS household row(s) carry a "
        f"{ACS_HOUSEHOLD_KIND_COLUMN} outside {list(ACS_HOUSEHOLD_KINDS)} "
        f"(missing counts as outside): {offending}.",
    )
    # TYPEHUGQ == 1 means "ACS housing unit", not "not group quarters": ASEC
    # rows carry it as missing by design, so a non-missing value on the ASEC
    # arm is a bad merge, not a housing unit.
    off_arm = (~acs) & kind.notna().to_numpy()
    _require(
        not off_arm.any(),
        "SPM_UNIVERSE_OFF_ARM_HOUSEHOLD_KIND",
        f"{int(off_arm.sum())} non-ACS household row(s) carry an ACS "
        f"{ACS_HOUSEHOLD_KIND_COLUMN} value "
        f"({_examples(kind[off_arm])}); the ACS record type does not "
        "describe them and cannot be read as one.",
    )
    return kind


def _kind_keys(arm: pd.Series, kind: pd.Series) -> pd.Series:
    """One hashable ``(arm, record type)`` per household row."""
    return pd.Series(
        [
            (str(a), None if pd.isna(k) else int(k))
            for a, k in zip(arm.to_numpy(), kind.to_numpy(), strict=True)
        ],
        index=arm.index,
        dtype=object,
    )


def _assert_one_kind_per_household(
    household_ids: pd.Series, kind_keys: pd.Series
) -> None:
    """A household id resolving to more than one kind is a bad merge.

    Unreachable through a validated :class:`~microcosm.frame.Frame`, whose
    constructor already refuses a duplicated group id. It is reachable when the
    producer is handed tables directly, and picking a majority kind there would
    be a fabricated universe decision.
    """
    distinct = kind_keys.groupby(household_ids.to_numpy(), sort=False).nunique()
    mixed = distinct[distinct > 1]
    _require(
        mixed.empty,
        "SPM_UNIVERSE_MIXED_HOUSEHOLD",
        f"{len(mixed)} household id(s) resolve to more than one "
        f"(arm, record type): {_examples(mixed.index)}.",
    )


def _person_clone_copies(person: pd.DataFrame) -> np.ndarray:
    """Each person's support-clone copy, with a missing index read as native.

    A support clone deep-copies every column of a table and remaps only the
    structural id and membership columns
    (``puf_support._clone_entity_table``), so clone copy ``k`` carries the
    native row's source fields under new ids. Reading an absent or missing
    clone index as the native copy can only make the checks that use it
    stricter: it merges rows *into* the native copy, where they can collide
    with native rows, and never separates rows the truth would merge.
    """
    if _PERSON_CLONE_INDEX_COLUMN not in person.columns:
        return np.full(len(person), _NATIVE_CLONE_INDEX, dtype=float)
    return (
        pd.to_numeric(person[_PERSON_CLONE_INDEX_COLUMN], errors="coerce")
        .fillna(_NATIVE_CLONE_INDEX)
        .to_numpy(dtype=float)
    )


def _assert_group_quarters_are_single_native_records(
    *,
    household_ids: pd.Series,
    arm: pd.Series,
    kind: pd.Series,
    person_household_ids: pd.Series,
    clone_copies: np.ndarray,
) -> int:
    """Mirror the stacked spine's one-native-person-per-GQ-placeholder rule.

    ACS group-quarters rows are one-person ``WGTP == 0`` placeholders at the
    source (``acs_pums._occupied_households``), and the stacked assembly
    re-asserts it. Clones of such a placeholder are expected and are not
    counted here; more than one *native* person in one is not. A person
    whose clone index is missing counts as native (see
    :func:`_person_clone_copies`), so a pool that lost the index on one arm
    cannot pass this check vacuously.
    """
    gq = (arm.eq(ACS_ARM) & kind.isin(ACS_GROUP_QUARTERS_KINDS)).to_numpy()
    gq_ids = pd.Index(household_ids.to_numpy()[gq]).unique()
    if len(gq_ids) == 0:
        return 0
    native = clone_copies == _NATIVE_CLONE_INDEX
    counts = (
        pd.Series(1, index=person_household_ids.to_numpy()[native])
        .groupby(level=0)
        .sum()
        .reindex(gq_ids, fill_value=0)
    )
    crowded = counts[counts > 1]
    _require(
        crowded.empty,
        "SPM_UNIVERSE_GQ_MULTI_PERSON",
        f"{len(crowded)} ACS group-quarters household row(s) carry more than "
        f"one native person: {_examples(crowded.index)}.",
    )
    return int(len(gq_ids))


def _assert_membership(
    values: pd.Series, owner_ids: np.ndarray, *, column: str, owner: str
) -> None:
    """Both directions of a membership join, independently of the frame kernel."""
    owner_index = pd.Index(owner_ids)
    orphan = values[~pd.Index(values).isin(owner_index)]
    _require(
        orphan.empty,
        "SPM_UNIVERSE_ORPHAN_MEMBER",
        f"{len(orphan)} person row(s) carry a {column!r} value matching no "
        f"{owner} id: {_examples(orphan)}.",
    )
    if owner == "spm_unit":
        referenced = pd.Index(pd.unique(values.to_numpy()))
        empty = owner_index[~owner_index.isin(referenced)]
        _require(
            empty.empty,
            "SPM_UNIVERSE_EMPTY_UNIT",
            f"{len(empty)} {owner} id(s) have no member in {column!r}: "
            f"{_examples(empty)}; an empty unit has no source record to read "
            "a universe from.",
        )


def _assert_native_asec_partition(
    *,
    person: pd.DataFrame,
    asec_person: np.ndarray,
    unit_membership_column: str,
    clone_copies: np.ndarray,
) -> None:
    """Refuse a silently degraded SPM partition on the ASEC arm.

    ``assign_us_unit_structure`` uses the native ``SPM_ID`` when it is fully
    present and falls back to ``household_id`` otherwise — and *any* missing
    ``SPM_ID`` triggers that fallback for the whole frame, while ``asec_pool``
    deliberately tolerates a missing ``SPM_ID`` when globalizing. Declaring
    ``INCLUDED`` over a degraded partition would attest to a unit structure
    that does not exist, so the ASEC arm must prove its partition is the
    native one.

    A support clone copies ``SPM_ID`` verbatim and gives each copy new unit
    ids (see :func:`_person_clone_copies`), so every clone copy is its own
    instance of the native partition. The native unit key is therefore
    ``(clone copy, SPM_ID)``, and the frame's units must be a one-to-one
    densification of that key. Keying on ``SPM_ID`` alone would read every
    support-cloned ASEC frame as degraded.

    The check is deliberately ASEC-only. ACS PUMS supplies no ``SPM_ID`` at
    all, so the ACS arm reaches ``assign_us_unit_structure`` without one and
    takes the household fallback by construction — there, one SPM unit per
    household *is* the native partition and carries no information.
    """
    if not asec_person.any():
        return
    _require(
        ASEC_NATIVE_UNIT_COLUMN in person.columns,
        "SPM_UNIVERSE_DEGRADED_PARTITION",
        f"the person table carries no {ASEC_NATIVE_UNIT_COLUMN!r}, so the ASEC "
        "arm's SPM partition cannot be shown to be the native one rather than "
        "the documented household fallback.",
    )
    native = person.loc[asec_person, ASEC_NATIVE_UNIT_COLUMN]
    missing = int(native.isna().sum())
    _require(
        missing == 0,
        "SPM_UNIVERSE_DEGRADED_PARTITION",
        f"{missing} ASEC person row(s) carry a missing "
        f"{ASEC_NATIVE_UNIT_COLUMN}; any missing value silently converts every "
        "SPM unit in the frame into a household.",
    )
    units = person.loc[asec_person, unit_membership_column].to_numpy()
    copies = clone_copies[asec_person]
    native_units = len(pd.MultiIndex.from_arrays([copies, native.to_numpy()]).unique())
    frame_units = len(pd.unique(units))
    pairings = len(
        pd.MultiIndex.from_arrays([copies, native.to_numpy(), units]).unique()
    )
    _require(
        pairings == native_units == frame_units,
        "SPM_UNIVERSE_DEGRADED_PARTITION",
        f"the ASEC arm's {unit_membership_column!r} is not a one-to-one "
        f"densification of (clone copy, {ASEC_NATIVE_UNIT_COLUMN}) "
        f"({native_units} native unit(s), {frame_units} frame unit(s), "
        f"{pairings} distinct pairing(s)), so the native partition did not "
        "survive assembly.",
    )


def _assert_asec_record_type_unreviewed(
    *, household: pd.DataFrame, person: pd.DataFrame, asec_any: bool
) -> None:
    """Guard the spine-level ASEC ruling against a vintage that outgrows it."""
    if not asec_any:
        return
    found = sorted(
        field
        for field in ASEC_RECORD_TYPE_FIELDS
        if field in household.columns or field in person.columns
    )
    _require(
        not found,
        "SPM_UNIVERSE_ASEC_RECORD_TYPE_UNREVIEWED",
        f"the frame carries ASEC household record-type field(s) {found}. The "
        f"ASEC ruling ({ASEC_SPM_INCLUDED_AUTHORITY}) was made at the spine "
        "precisely because Microcosm reads none of these; it must be "
        "re-derived against the field rather than applied over it.",
    )


def classify_spm_universe(
    *,
    household: pd.DataFrame,
    person: pd.DataFrame,
    unit_ids: np.ndarray,
    household_id_column: str = "household_id",
    household_membership_column: str = "person_household_id",
    unit_membership_column: str = "person_spm_unit_id",
    declared_channel: str | None = None,
    channel_column: str | None = None,
) -> SpmUniverseClassification:
    """Classify every SPM unit's measurement universe from source record type.

    Args:
        household: The household table, carrying its provenance tag and, on
            the ACS arm, ``TYPEHUGQ``.
        person: The person table, carrying household and SPM-unit membership.
        unit_ids: The SPM unit ids to classify, in the order the caller wants
            the returned status.
        household_id_column: Id column on ``household``.
        household_membership_column: Person column linking to ``household``.
        unit_membership_column: Person column linking to the SPM unit table.
        declared_channel: Universe channel for a frame that carries no
            provenance tag of its own. Refused when the frame does carry one.
        channel_column: Explicit provenance column to read instead of the
            default resolution order.

    Returns:
        The per-unit status as member-name strings, the household-level
        evidence it was derived from, and a provenance dict.

    Raises:
        ValueError: Any of :data:`SPM_UNIVERSE_REFUSALS`. Every refusal is
            fail-closed: this producer never resolves an ambiguity by guessing,
            and never writes ``UNRESOLVED`` to mean "I could not tell".
    """
    for table, columns, label in (
        (household, (household_id_column,), "household"),
        (person, (household_membership_column, unit_membership_column), "person"),
    ):
        missing = sorted(set(columns) - set(table.columns))
        _require(
            not missing,
            "SPM_UNIVERSE_ORPHAN_MEMBER",
            f"the {label} table is missing required column(s) {missing}.",
        )

    unit_ids = np.asarray(unit_ids)
    household_ids = household[household_id_column]
    channel, channel_source = _resolve_channel(
        household, declared_channel=declared_channel, channel_column=channel_column
    )
    arm = _household_arms(channel)
    kind = _household_kinds(household, arm)
    kind_keys = _kind_keys(arm, kind)
    _assert_one_kind_per_household(household_ids, kind_keys)

    person_household_ids = person[household_membership_column]
    person_units = person[unit_membership_column]
    _assert_membership(
        person_household_ids,
        household_ids.to_numpy(),
        column=household_membership_column,
        owner="household",
    )
    _assert_membership(
        person_units, unit_ids, column=unit_membership_column, owner="spm_unit"
    )

    clone_copies = _person_clone_copies(person)
    group_quarters_households = _assert_group_quarters_are_single_native_records(
        household_ids=household_ids,
        arm=arm,
        kind=kind,
        person_household_ids=person_household_ids,
        clone_copies=clone_copies,
    )

    kind_by_household = pd.Series(
        kind_keys.to_numpy(), index=pd.Index(household_ids.to_numpy())
    )
    # Every duplicate id resolves to one kind by now, so dropping repeats
    # cannot change a lookup; it only keeps ``map`` from fanning rows out.
    kind_by_household = kind_by_household[~kind_by_household.index.duplicated()]
    person_kind = person_household_ids.map(kind_by_household)
    asec_person = np.asarray(
        [key[0] == ASEC_ARM for key in person_kind.to_numpy()], dtype=bool
    )
    _assert_asec_record_type_unreviewed(
        household=household, person=person, asec_any=bool(asec_person.any())
    )
    _assert_native_asec_partition(
        person=person,
        asec_person=asec_person,
        unit_membership_column=unit_membership_column,
        clone_copies=clone_copies,
    )

    by_unit = pd.DataFrame(
        {"unit": person_units.to_numpy(), "kind": person_kind.to_numpy()}
    )
    distinct = by_unit.groupby("unit", sort=False)["kind"].nunique()
    spanning = distinct[distinct > 1]
    _require(
        spanning.empty,
        "SPM_UNIVERSE_UNIT_SPANS_KINDS",
        f"{len(spanning)} SPM unit(s) have members in households of different "
        f"(arm, record type): {_examples(spanning.index)}. Picking a majority "
        "kind would be a fabricated universe decision.",
    )

    unit_kind = (
        by_unit.groupby("unit", sort=False)["kind"].first().reindex(pd.Index(unit_ids))
    )
    status = np.asarray(
        [
            OUTSIDE
            if key[0] == ACS_ARM and key[1] in ACS_GROUP_QUARTERS_KINDS
            else INCLUDED
            for key in unit_kind.to_numpy()
        ],
        dtype=object,
    )
    # A producer that cannot decide refuses above; UNRESOLVED is the engine's
    # absence sentinel and must never be emitted.
    _require(
        UNRESOLVED not in set(status),
        "SPM_UNIVERSE_UNDECLARED_SPINE",
        f"the classification emitted {UNRESOLVED}, which is the engine's "
        "absence sentinel rather than a producer value.",
    )

    arm_counts = arm.value_counts()
    kind_counts = {
        int(value): int(count)
        for value, count in kind.value_counts(dropna=True).items()
    }
    provenance = {
        "declared_variable": UNIVERSE_INPUT,
        "declared_entity": "spm_unit",
        "declared_period": "YEAR",
        "stored_encoding": "enum member name string",
        "profile": SPM_UNIVERSE_PROFILE_NAME,
        "rule": SPM_UNIVERSE_RULE,
        "acs_authority": ACS_SPM_OUTSIDE_AUTHORITY,
        "asec_authority": ASEC_SPM_INCLUDED_AUTHORITY,
        "methodology": SPM_UNIVERSE_METHODOLOGY,
        "evidence_status": (
            "Source record type read from the frame's own household table "
            "(ACS TYPEHUGQ), plus a spine-level ruling for the ASEC arm whose "
            "sample frame equals the CPS SPM universe. No composition, age, "
            "tenure, geography or role is read."
        ),
        "channel_source": channel_source,
        "channel_arms": dict(SPM_UNIVERSE_CHANNEL_ARMS),
        "household_rows": int(len(household)),
        "households_by_arm": {str(k): int(v) for k, v in arm_counts.items()},
        "households_by_acs_record_type": kind_counts,
        "acs_group_quarters_households": group_quarters_households,
        "person_rows": int(len(person)),
        "spm_units": int(len(unit_ids)),
        "units_included": int((status == INCLUDED).sum()),
        "units_outside": int((status == OUTSIDE).sum()),
        "units_unresolved": 0,
        "refusals": list(SPM_UNIVERSE_REFUSALS),
        "composition_read": False,
        "weights_used": False,
        "ages_changed": False,
        "is_calibration_target": False,
        "is_spine_agreement_surface": False,
        "period_handling": (
            "The declaration is a construction fact about the source record, "
            "constant across the years a build requests."
        ),
    }
    return SpmUniverseClassification(
        status=status,
        unit_ids=unit_ids,
        household_arm=arm,
        household_kind=kind,
        provenance=provenance,
    )


def attach_spm_universe_status(
    frame: Any,
    *,
    unit_entity: str = "spm_unit",
    declared_channel: str | None = None,
    channel_column: str | None = None,
) -> SpmUniverseSourceResult:
    """Attach :data:`UNIVERSE_INPUT` to ``frame``'s SPM unit table.

    The status is stored as the enum **member name** — ``"OUTSIDE"``, not
    ``2``, not ``"SPMUniverseStatus.OUTSIDE"``, and not the label text —
    because the export adapter normalizes a stored cell to a member name and
    validates it cell-by-cell against the engine's own enum domain.

    Nothing else about the frame changes: no weight, no membership, no row.

    Args:
        frame: A validated :class:`~microcosm.frame.Frame`.
        unit_entity: The SPM unit entity.
        declared_channel: See :func:`classify_spm_universe`.
        channel_column: See :func:`classify_spm_universe`.

    Returns:
        A new frame carrying the column, and the classification behind it.

    Raises:
        ValueError: Any of :data:`SPM_UNIVERSE_REFUSALS`, including
            ``SPM_UNIVERSE_COLUMN_EXISTS`` when any entity table already
            carries the declaration — a producer overwriting an existing
            declaration would silently replace someone else's source ruling,
            and a copy on another entity would collide with it under the
            frame's global column-name rule.
    """
    from microcosm.frame import Frame

    schema = frame.schema
    unit_table = frame.table(unit_entity)
    carriers = [
        entity
        for entity in frame.entities
        if UNIVERSE_INPUT in frame.table(entity).columns
    ]
    _require(
        not carriers,
        "SPM_UNIVERSE_COLUMN_EXISTS",
        f"the {carriers} table(s) already carry {UNIVERSE_INPUT!r}; this "
        "producer never overwrites an existing universe declaration.",
    )
    classification = classify_spm_universe(
        household=frame.table("household"),
        person=frame.table(schema.person_entity),
        unit_ids=unit_table[schema.id_column(unit_entity)].to_numpy(),
        household_id_column=schema.id_column("household"),
        household_membership_column=schema.membership_column("household"),
        unit_membership_column=schema.membership_column(unit_entity),
        declared_channel=declared_channel,
        channel_column=channel_column,
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({name: frame.link(name).copy() for name in frame.links})
    tables[unit_entity][UNIVERSE_INPUT] = classification.status
    declared = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return SpmUniverseSourceResult(frame=declared, classification=classification)
