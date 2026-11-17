"""The d2283 one-population contract: provenance, refusals, role record, absences.

d2283's national child population joins every CPS row of the parent's Route A
export (native ASEC rows at clone index 0 and their PUF tax-detail clones at
clone indices 1 and 2) with the ACS households PR 1's selection
(:mod:`~microcosm.build.us_runtime.one_population_selection`) takes from a
staging run built on that export as donor. The prepare step applies this
library between the selection and the H5 write:

- :func:`to_assembled_provenance` rewrites both origins' support provenance
  into the assembled form (``<entity>_spine_source_id`` on every row) and
  builds the spine assembly record from the clone-0 rows;
- :func:`require_one_population` refuses a joined frame that breaks any
  clause of the contract, with a message that names the clause's rule;
- :func:`combined_spm_role_record` writes the SPM independence role record
  for the joined frame, with per-origin counts;
- :func:`absences` lists the columns one origin carries as a reviewed
  default fill, a zero written in place of a measure, or not at all, each
  with its kind;
- :func:`cps_unit_ids` derives each CPS household's unit key, so the
  selection manifest's digest and the contract's key check share one rule.

The engine's variable classes at 2024 (known, computed, input) come from an
injected :class:`EngineVariableIndex`. The default,
:class:`PolicyEngineUSVariableIndex`, reads the installed policyengine-us
source metadata the first time it is asked, never at import, so importing
this module loads no engine.

The module certifies no numbers: no build calls it yet.
"""

from __future__ import annotations

import numbers
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.one_population_selection import (
    SelectionManifest,
    cps_unit_key,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_CAPITAL_GAINS_TAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_assembly_manifest,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_copy_rank_series,
    support_source_id_column,
    validate_assembly_provenance,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

__all__ = [
    "ABSENCES_FILENAME",
    "ACS_SUPPORT_CHANNEL",
    "ASSEMBLED_CHANNELS",
    "CPS_SPINE_TAG",
    "ENGINE_PERIOD",
    "STRUCTURAL_COLUMNS",
    "EngineVariableIndex",
    "OnePopulationContractError",
    "PolicyEngineUSVariableIndex",
    "absences",
    "combined_spm_role_record",
    "cps_unit_ids",
    "require_one_population",
    "to_assembled_provenance",
]

#: The national period the engine's variable classes are read at: the
#: release tool's ``PERIOD``, which its export guard passes to the same
#: classifier (F-14, F-15).
ENGINE_PERIOD = 2024

#: The support channel (and ``<entity>_spine`` tag) staging gives ACS rows,
#: ``acs_pums.ACS_2024_1YR_SPINE``. A test pins the two together.
ACS_SUPPORT_CHANNEL = "acs_2024_1yr"

#: The ``<entity>_spine`` tag staging gives donor rows,
#: ``base_pool.ASEC_PUF_SPINE``. The contract stamps it on every CPS row
#: (d2283 Q3), so no row of the joined frame has a null tag. A test pins the
#: two together.
CPS_SPINE_TAG = "asec_puf"

#: The channels an assembled one-population frame declares, in record order.
#: ``puf_tax_detail`` is declared because CPS clones carry it, with a clone-0
#: count of 0 on every entity.
ASSEMBLED_CHANNELS = (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ACS_SUPPORT_CHANNEL,
)

#: The file the prepare step writes :func:`absences` to.
ABSENCES_FILENAME = "us_population_absences.json"

#: The person id and each group's id and membership column. None of them is
#: an engine input or an engine-computed column for this contract, as in
#: the release tool's export guard.
STRUCTURAL_COLUMNS = frozenset(
    {
        US_SCHEMA.person_id_column,
        *(US_SCHEMA.id_column(group) for group in US_SCHEMA.group_entities),
        *(US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities),
    }
)

# Raw source columns the clauses read. CPS rows carry the ASEC person id;
# ACS rows carry the PUMS household serial number (household table) and the
# person's line in it (person table), plus the household type code.
_PERIDNUM = "PERIDNUM"
_SERIALNO = "SERIALNO"
_SPORDER = "SPORDER"
_TYPEHUGQ = "TYPEHUGQ"
_SOURCE_YEAR = "source_year"
_SOURCE_HOUSEHOLD_ID = "source_household_id"
_SOURCE_PERSON_ID = "source_person_id"
#: A CPS unit id is the first 20 characters of PERIDNUM (H_IDNUM), as the
#: support-mix bake-off reads it.
_CPS_UNIT_ID_LENGTH = 20
#: The ``source_year`` staging's ACS rows carry: ``acs_pums`` writes the
#: source vintage, 2024 for the 2024 1-year file the channel names.
_ACS_SOURCE_YEAR = 2024

#: Route A's labels on the parent export's rows: the native ASEC row at
#: clone index 0, the PUF tax-detail copy at 1 and the capital-gains tail
#: copy at 2, both copies labelled ``puf_tax_detail``.
_CPS_CHANNELS = (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
_CPS_CLONE_INDICES = (
    0,
    PUF_TAX_DETAIL_CLONE_INDEX,
    PUF_CAPITAL_GAINS_TAIL_CLONE_INDEX,
)
_CPS = "CPS"
_ACS = "ACS"
#: The boundary name the assembly validator reports.
_BOUNDARY = "one-population provenance"


class OnePopulationContractError(ValueError):
    """A one-population frame, or one origin's input, breaks a contract clause.

    Each message names the clause's rule, so a caller (and a test) can tell
    the refusals apart.
    """


# ------------------------------------------------------------ provenance


def _spine_column(entity: str) -> str:
    """The ``<entity>_spine`` tag column staging writes on every pooled row."""

    return f"{entity}_spine"


def _provenance_columns(entity: str) -> tuple[str, str, str, str, str]:
    """``entity``'s source id, channel, clone index, spine source id and spine tag."""

    return (
        support_source_id_column(entity),
        support_channel_column(entity),
        support_clone_index_column(entity),
        spine_source_id_column(entity),
        _spine_column(entity),
    )


def _integral_ids(series: pd.Series, *, refusal: str) -> np.ndarray:
    """``series`` as int64 if every value is a non-null integer, else refuse.

    Integer dtypes pass as they are; float and object values must be finite,
    integral and inside int64 (staging's pooled form turns an origin's
    integer column into float64 when the other origin lacks it).
    """

    if pd.api.types.is_bool_dtype(series.dtype):
        raise OnePopulationContractError(f"{refusal}: it holds booleans, not ids.")
    nulls = int(series.isna().sum())
    if nulls:
        raise OnePopulationContractError(f"{refusal}: {nulls} value(s) are null.")
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.to_numpy(dtype=np.int64)
    numeric = pd.to_numeric(series, errors="coerce")
    if pd.api.types.is_integer_dtype(numeric.dtype):
        # Object-held Python ints: exact, never through float64.
        return numeric.to_numpy(dtype=np.int64)
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    bad = (
        ~np.isfinite(values) | (values != np.floor(values)) | (np.abs(values) >= 2**63)
    )
    if bad.any():
        examples = series[bad].head(5).tolist()
        raise OnePopulationContractError(
            f"{refusal}: {int(bad.sum())} value(s) are not integers, such as "
            f"{examples!r}."
        )
    return values.astype(np.int64)


def _origin_tables(
    tables: Mapping[str, pd.DataFrame], *, origin: str
) -> dict[str, pd.DataFrame]:
    """The six US entity tables of one origin, or a refusal."""

    if not isinstance(tables, Mapping):
        raise TypeError(
            f"{origin} tables must be a mapping of entity to table, got "
            f"{type(tables).__name__}."
        )
    expected = set(US_SCHEMA.entities)
    missing = sorted(expected - set(tables))
    unexpected = sorted(str(name) for name in set(tables) - expected)
    if missing or unexpected:
        raise OnePopulationContractError(
            f"{origin} rows lack complete support metadata: the tables must be "
            f"the six US entities; missing {missing}, unexpected {unexpected}."
        )
    for entity in US_SCHEMA.entities:
        if not isinstance(tables[entity], pd.DataFrame):
            raise TypeError(
                f"{origin} {entity!r} table must be a DataFrame, got "
                f"{type(tables[entity]).__name__}."
            )
    return {entity: tables[entity] for entity in US_SCHEMA.entities}


def _require_support_columns(
    table: pd.DataFrame, entity: str, *, origin: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Presence and completeness of one table's support metadata (P5).

    Returns the integral ids, source ids and clone indices.
    """

    source, channel, clone, spine_source, spine = _provenance_columns(entity)
    id_column = US_SCHEMA.entity_id_column(entity)
    rule = f"{origin} rows lack complete support metadata"
    missing = [column for column in (source, channel, clone) if column not in table]
    if missing:
        raise OnePopulationContractError(
            f"{rule}: {entity!r} is missing {missing}; every entity needs its "
            "source id, support channel and clone index."
        )
    if id_column not in table:
        raise OnePopulationContractError(
            f"{rule}: {entity!r} is missing {id_column!r}."
        )
    if spine_source in table:
        raise OnePopulationContractError(
            f"{origin} rows already carry {spine_source!r}: the contract writes "
            "the spine source id, so its input must not hold one."
        )
    for column in (source, channel, clone):
        nulls = int(table[column].isna().sum())
        if nulls:
            raise OnePopulationContractError(
                f"{rule}: {column!r} has {nulls} null value(s)."
            )
    return (
        _integral_ids(
            table[id_column], refusal=f"{origin} {id_column!r} must hold ids"
        ),
        _integral_ids(
            table[source], refusal=f"{origin} {source!r} must hold source ids"
        ),
        _integral_ids(
            table[clone], refusal=f"{origin} {clone!r} must hold clone indices"
        ),
    )


def _unexpected(values: pd.Series, allowed: Collection[object]) -> list[str]:
    """Sorted reprs of the values outside ``allowed`` (strings only allowed)."""

    good = values.map(lambda value: isinstance(value, str) and value in allowed)
    return sorted({repr(value) for value in values[~good.astype(bool)]})


def _require_cps_support_metadata(tables: Mapping[str, pd.DataFrame]) -> None:
    """P5 on the parent export's rows: Route A's labels, complete."""

    for entity in US_SCHEMA.entities:
        table = tables[entity]
        _, channel, clone, _, spine = _provenance_columns(entity)
        _, _, clones = _require_support_columns(table, entity, origin=_CPS)
        bad_clones = sorted(set(clones.tolist()) - set(_CPS_CLONE_INDICES))
        if bad_clones:
            raise OnePopulationContractError(
                "CPS clone indices must be 0, 1 or 2 (Route A's native rows, PUF "
                f"tax-detail copies and capital-gains tail copies); {clone!r} "
                f"holds {bad_clones[:5]}."
            )
        bad_channels = _unexpected(table[channel], _CPS_CHANNELS)
        if bad_channels:
            raise OnePopulationContractError(
                f"CPS rows must carry channel {BASE_ASEC_SUPPORT_CHANNEL!r} or "
                f"{PUF_TAX_DETAIL_SUPPORT_CHANNEL!r} (Route A's labels); "
                f"{channel!r} holds {bad_channels[:5]}."
            )
        native = table[channel].eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
        disagree = int(np.count_nonzero(native != (clones == 0)))
        if disagree:
            raise OnePopulationContractError(
                f"CPS rows must carry channel {BASE_ASEC_SUPPORT_CHANNEL!r} exactly "
                f"on clone index 0; {disagree} {entity!r} row(s) disagree."
            )
        if spine in table:
            tags = table[spine]
            wrong = tags.notna() & tags.ne(CPS_SPINE_TAG)
            if wrong.any():
                raise OnePopulationContractError(
                    f"CPS rows carry a spine tag other than {CPS_SPINE_TAG!r} in "
                    f"{spine!r}: {sorted({repr(tag) for tag in tags[wrong]})[:5]}; "
                    "staging tags the donor's rows "
                    f"{CPS_SPINE_TAG!r}."
                )


def _require_acs_support_metadata(tables: Mapping[str, pd.DataFrame]) -> None:
    """P5 on staging's ACS rows: staging's labels, complete, one id offset."""

    for entity in US_SCHEMA.entities:
        table = tables[entity]
        _, channel, clone, _, spine = _provenance_columns(entity)
        ids, sources, clones = _require_support_columns(table, entity, origin=_ACS)
        bad_channels = _unexpected(table[channel], (ACS_SUPPORT_CHANNEL,))
        if bad_channels:
            raise OnePopulationContractError(
                f"ACS rows must carry channel {ACS_SUPPORT_CHANNEL!r} (staging's); "
                f"{channel!r} holds {bad_channels[:5]}."
            )
        bad_clones = sorted(set(clones.tolist()) - {0})
        if bad_clones:
            raise OnePopulationContractError(
                f"ACS rows must carry clone index 0; {clone!r} holds {bad_clones[:5]}."
            )
        offsets = sorted(set((ids - sources).tolist()))
        if len(offsets) > 1:
            raise OnePopulationContractError(
                f"ACS {entity!r} ids must sit one constant offset from staging's "
                f"native source ids (staging shifts each entity's ids by one "
                f"amount); found offsets {offsets[:5]}."
            )
        if spine in table:
            tags = table[spine]
            wrong = tags.notna() & tags.ne(ACS_SUPPORT_CHANNEL)
            if wrong.any():
                raise OnePopulationContractError(
                    f"ACS rows carry a spine tag other than {ACS_SUPPORT_CHANNEL!r} "
                    f"in {spine!r}: {sorted({repr(tag) for tag in tags[wrong]})[:5]}."
                )


def _household_weights(values: object, *, origin: str, households: int) -> np.ndarray:
    """One origin's household weights, aligned with its household table."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != households:
        raise OnePopulationContractError(
            f"{origin} household weights must align with the {origin} household "
            f"table: {households} row(s), got shape {array.shape}."
        )
    if not np.isfinite(array).all() or (array < 0).any():
        raise OnePopulationContractError(
            f"{origin} household weights must be finite and non-negative."
        )
    return array


def _object_column(value: object, index: pd.Index) -> pd.Series:
    return pd.Series(value, index=index, dtype=object)


def _assembled_cps(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """P1 and Q3: keep Route A's labels, add the spine source id and tag."""

    result: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        source, _, _, spine_source, spine = _provenance_columns(entity)
        table = tables[entity].copy()
        table[spine_source] = table[source].to_numpy(copy=True)
        table[spine] = _object_column(CPS_SPINE_TAG, table.index)
        result[entity] = table
    return result


def _assembled_acs(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """P2: ``assemble_spines``'s rule on staging's rows, channel unchanged.

    The spine source id takes staging's native source id and the source id
    the pooled ``<entity>_id``; channel ``acs_2024_1yr`` and clone index 0
    stay. Nothing writes the predictor join's ``acs`` channel (P3).
    """

    result: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        source, _, _, spine_source, _ = _provenance_columns(entity)
        table = tables[entity].copy()
        table[spine_source] = table[source].to_numpy(copy=True)
        table[source] = table[US_SCHEMA.entity_id_column(entity)].to_numpy(copy=True)
        result[entity] = table
    return result


def _require_spine_tags(origins: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    """Q3: no row of the joined frame may carry a null ``<entity>_spine``."""

    for origin, tables in origins.items():
        for entity in US_SCHEMA.entities:
            spine = _spine_column(entity)
            table = tables[entity]
            nulls = len(table) if spine not in table else int(table[spine].isna().sum())
            if nulls:
                raise OnePopulationContractError(
                    f"A null spine tag: {nulls} {origin} {entity!r} row(s) carry no "
                    f"{spine!r} after the rewrite; every row needs its origin's tag "
                    f"({CPS_SPINE_TAG!r} on CPS rows, {ACS_SUPPORT_CHANNEL!r} on "
                    "ACS rows)."
                )


def _require_disjoint_ids(origins: Mapping[str, Mapping[str, pd.DataFrame]]) -> None:
    """P6b: each entity's ids unique across origins, memberships resolved.

    Every person membership must resolve to exactly one group row of its own
    origin, and every group row must hold a person of its origin. Staging
    offsets ACS ids clear of its own donor only, so an ACS frame staged on
    another donor than the parent export collides here.
    """

    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        ids = {
            origin: _integral_ids(
                tables[entity][id_column],
                refusal=f"{origin} {id_column!r} must hold ids",
            )
            for origin, tables in origins.items()
        }
        joined = pd.Series(np.concatenate(list(ids.values())))
        repeated = joined[joined.duplicated(keep=False)]
        if len(repeated):
            across = sorted(set(ids[_CPS].tolist()) & set(ids[_ACS].tolist()))
            where = "across origins" if across else "within one origin"
            raise OnePopulationContractError(
                f"Frame ids collide: {id_column!r} repeats {where} "
                f"({sorted(set(repeated.tolist()))[:5]}); each entity's id must be "
                "unique across the joined CPS and ACS rows (an ACS frame staged on "
                "another donor than the parent export does this)."
            )
    for origin, tables in origins.items():
        person = tables[US_SCHEMA.person_entity]
        for group in US_SCHEMA.group_entities:
            membership_column = US_SCHEMA.membership_column(group)
            group_ids = tables[group][US_SCHEMA.id_column(group)]
            if membership_column not in person:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {origin} persons lack "
                    f"{membership_column!r}."
                )
            membership = person[membership_column]
            unresolved = int((~membership.isin(group_ids)).sum())
            if unresolved:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {unresolved} {origin} person(s)' "
                    f"{membership_column!r} match no {origin} {group} row; every "
                    "person membership must resolve to exactly one group row of its "
                    "own origin."
                )
            empty = int((~group_ids.isin(membership)).sum())
            if empty:
                raise OnePopulationContractError(
                    f"Memberships do not resolve: {empty} {origin} {group} row(s) "
                    f"hold no {origin} person."
                )


def _require_unique_source_ranks(
    origins: Mapping[str, Mapping[str, pd.DataFrame]],
) -> None:
    """P6: no (source id, rank) pair repeats anywhere in the joined frame.

    The rank is :func:`support_copy_rank_series`'s, the clone index, read on
    the rewritten (assembled) rows.
    """

    for entity in US_SCHEMA.entities:
        source = support_source_id_column(entity)
        pairs = pd.concat(
            [
                pd.DataFrame(
                    {
                        "source_id": _integral_ids(
                            tables[entity][source],
                            refusal=f"{origin} {source!r} must hold source ids",
                        ),
                        "rank": support_copy_rank_series(
                            tables[entity], entity=entity
                        ).to_numpy(),
                        "origin": origin,
                    }
                )
                for origin, tables in origins.items()
            ],
            ignore_index=True,
        )
        repeated = pairs[pairs.duplicated(["source_id", "rank"], keep=False)]
        if len(repeated):
            examples = sorted(
                {
                    (int(row.source_id), int(row.rank))
                    for row in repeated.itertuples(index=False)
                }
            )
            origins_hit = sorted(set(repeated["origin"]))
            raise OnePopulationContractError(
                f"A (source id, rank) pair repeats in {entity!r}: {examples[:5]} "
                f"({len(examples)} pair(s), on {origins_hit} rows); each support "
                "copy needs its own source id and clone index."
            )


def _lineage_pairs(
    staging_lineage: Collection[tuple[str, int]],
) -> frozenset[tuple[str, int]]:
    """Staging's (SERIALNO, SPORDER) pairs, with SPORDER as an int."""

    rule = "ACS lineage does not survive"
    if isinstance(staging_lineage, str | bytes) or not isinstance(
        staging_lineage, Collection
    ):
        raise TypeError(
            "staging_lineage must be a collection of (SERIALNO, SPORDER) pairs, "
            f"got {type(staging_lineage).__name__}."
        )
    pairs: set[tuple[str, int]] = set()
    for pair in staging_lineage:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise OnePopulationContractError(
                f"{rule}: staging's lineage holds {pair!r}, not a "
                "(SERIALNO, SPORDER) pair."
            )
        serial, sporder = pair
        if (
            not isinstance(serial, str)
            or not serial
            or isinstance(sporder, bool)
            or not isinstance(sporder, numbers.Real)
            or not float(sporder).is_integer()
        ):
            raise OnePopulationContractError(
                f"{rule}: staging's lineage holds {pair!r}; SERIALNO must be a "
                "non-empty string and SPORDER an integer."
            )
        pairs.add((serial, int(sporder)))
    return frozenset(pairs)


def _require_acs_lineage(
    tables: Mapping[str, pd.DataFrame],
    staging: frozenset[tuple[str, int]],
) -> None:
    """P7: every ACS row's ``SERIALNO``/``SPORDER`` lineage survives.

    ``SERIALNO`` is a non-empty string, unique per ACS household; ``SPORDER``
    an integer, with (``SERIALNO``, ``SPORDER``) unique per ACS person through
    the household link; ``source_household_id`` equals the linked
    household's ``household_spine_source_id`` (staging's native id);
    ``source_person_id`` is ``str(SPORDER)``; ``source_year`` is 2024; and
    the persons' (``SERIALNO``, ``SPORDER``) set equals staging's.
    """

    rule = "ACS lineage does not survive"
    household = tables["household"]
    person = tables[US_SCHEMA.person_entity]
    for name, table, columns in (
        ("household", household, (_SERIALNO,)),
        (
            "person",
            person,
            (_SPORDER, _SOURCE_YEAR, _SOURCE_HOUSEHOLD_ID, _SOURCE_PERSON_ID),
        ),
    ):
        missing = [column for column in columns if column not in table]
        if missing:
            raise OnePopulationContractError(
                f"{rule}: the ACS {name} table lacks {missing}."
            )
    serials = household[_SERIALNO]
    bad = ~serials.map(lambda value: isinstance(value, str) and bool(value)).astype(
        bool
    )
    if bad.any():
        raise OnePopulationContractError(
            f"{rule}: every ACS household needs a non-empty string {_SERIALNO}; "
            f"{int(bad.sum())} hold {serials[bad].head(5).tolist()!r}."
        )
    if serials.duplicated().any():
        raise OnePopulationContractError(
            f"{rule}: {_SERIALNO} repeats across ACS households: "
            f"{sorted(set(serials[serials.duplicated()]))[:5]}."
        )
    sporder = _integral_ids(
        person[_SPORDER],
        refusal=f"{rule}: every ACS person needs an integral {_SPORDER}",
    )
    household_id = US_SCHEMA.id_column("household")
    by_household = household.set_index(household_id)
    link = person[US_SCHEMA.membership_column("household")]
    person_serial = link.map(by_household[_SERIALNO]).to_numpy(dtype=object)
    lineage = pd.DataFrame({"serial": person_serial, "sporder": sporder})
    repeated = lineage.duplicated(keep=False)
    if repeated.any():
        raise OnePopulationContractError(
            f"{rule}: ({_SERIALNO}, {_SPORDER}) repeats on "
            f"{int(repeated.sum())} ACS person(s)."
        )
    native = link.map(by_household[spine_source_id_column("household")])
    source_household = pd.to_numeric(person[_SOURCE_HOUSEHOLD_ID], errors="coerce")
    differs = int((source_household.to_numpy() != native.to_numpy()).sum())
    if differs:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_HOUSEHOLD_ID} differs from the linked household's "
            f"native id ({spine_source_id_column('household')}) on {differs} ACS "
            "person(s)."
        )
    expected = [str(value) for value in sporder.tolist()]
    actual = person[_SOURCE_PERSON_ID].tolist()
    differs = sum(1 for want, got in zip(expected, actual, strict=True) if want != got)
    if differs:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_PERSON_ID} differs from str({_SPORDER}) on {differs} "
            "ACS person(s)."
        )
    years = pd.to_numeric(person[_SOURCE_YEAR], errors="coerce")
    wrong_year = int((years != _ACS_SOURCE_YEAR).sum())
    if wrong_year:
        raise OnePopulationContractError(
            f"{rule}: {_SOURCE_YEAR} must be {_ACS_SOURCE_YEAR} on every ACS person; "
            f"{wrong_year} differ."
        )
    selected = frozenset(
        zip(lineage["serial"].tolist(), lineage["sporder"].tolist(), strict=True)
    )
    if selected != staging:
        missing = sorted(staging - selected)
        extra = sorted(selected - staging)
        raise OnePopulationContractError(
            f"{rule}: the selected ACS persons' ({_SERIALNO}, {_SPORDER}) set "
            f"differs from staging's for the selected households; {len(missing)} "
            f"missing (such as {missing[:3]}), {len(extra)} extra (such as "
            f"{extra[:3]})."
        )


def _joined_tables(
    origins: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """CPS rows then ACS rows per entity, group tables sorted by id.

    A zero-row table is left out of the join, so an empty origin changes no
    dtype of the other's columns. Returns the tables and the household
    order applied to the concatenated rows.
    """

    tables: dict[str, pd.DataFrame] = {}
    household_order = np.arange(0)
    for entity in US_SCHEMA.entities:
        parts = [table[entity] for table in origins.values() if len(table[entity])]
        if not parts:
            parts = [origins[_CPS][entity]]
        table = pd.concat(parts, ignore_index=True, sort=False)
        if entity in US_SCHEMA.group_entities:
            order = np.argsort(
                table[US_SCHEMA.id_column(entity)].to_numpy(), kind="stable"
            )
            table = table.iloc[order].reset_index(drop=True)
            if entity == "household":
                household_order = order
        tables[entity] = table
    return tables, household_order


def to_assembled_provenance(
    cps_tables: Mapping[str, pd.DataFrame],
    acs_tables: Mapping[str, pd.DataFrame],
    *,
    cps_household_weights: object,
    acs_household_weights: object,
    staging_lineage: Collection[tuple[str, int]],
) -> Frame:
    """Join CPS and ACS rows in the assembled provenance form (d2283 K-1).

    CPS rows (the parent export's) keep Route A's source id, channel and
    clone index (``asec`` on clone 0, ``puf_tax_detail`` on clones 1 and 2),
    gain ``<entity>_spine_source_id`` equal to the source id, and are tagged
    ``<entity>_spine = asec_puf``, staging's donor tag (d2283 Q3). ACS rows
    (staging's) follow
    :func:`~microcosm.build.us_runtime.spine_assembly.assemble_spines`'s rule,
    re-implemented because that function refuses inputs that already carry
    provenance: the spine source id takes staging's native source id, the
    source id takes the pooled ``<entity>_id`` (staging offset it clear of
    the donor's ids), the channel stays ``acs_2024_1yr`` and the clone index
    0, and staging's ``<entity>_spine`` tag stays. No row is labelled
    ``acs``, so the ACS predictor join passes the frame through. Ids are
    never re-offset (d2283 Q9): an ACS frame staged on another donor than
    the parent export is refused.

    The clauses run in order, each refusing with its own rule's message:

    - complete support metadata on each origin's input (P5). CPS: source id,
      channel and clone index on all six entities, no nulls, channel
      ``asec`` or ``puf_tax_detail``, ``asec`` exactly on clone index 0,
      clone index 0, 1 or 2, no spine source id, and no spine tag other than
      ``asec_puf``. ACS: the same three columns, no nulls, channel
      ``acs_2024_1yr``, clone index 0, ``<entity>_id - <entity>_source_id``
      one constant per entity, no spine source id, and no spine tag other
      than ``acs_2024_1yr``;
    - a non-null ``<entity>_spine`` on every row after the rewrite (Q3);
    - each entity's id unique across the joined rows, every person
      membership resolving to exactly one group row of its own origin, and
      every group row holding a person (P6b);
    - no (source id, rank) pair repeated anywhere, the rank being
      :func:`~microcosm.build.us_runtime.support_provenance.support_copy_rank_series`'s
      (P6);
    - the ACS rows' ``SERIALNO``/``SPORDER`` lineage, against
      ``staging_lineage`` (P7; see :func:`_require_acs_lineage`).

    The spine assembly record (P8) is
    :func:`~microcosm.build.us_runtime.support_provenance.spine_assembly_manifest`
    over every entity's clone-0 rows, declaring :data:`ASSEMBLED_CHANNELS`
    (``puf_tax_detail`` with a clone-0 count of 0), so
    :func:`~microcosm.build.us_runtime.support_provenance.validate_assembly_provenance`
    accepts the clone rows. The frame is checked against it before return.

    Args:
        cps_tables: The parent export's six entity tables (every CPS row:
            native rows and their clones).
        acs_tables: Staging's six entity tables for the selected ACS
            households. A table may have no rows.
        cps_household_weights: One weight per row of
            ``cps_tables["household"]``, in its order (PR 1's starting
            weights).
        acs_household_weights: One weight per row of
            ``acs_tables["household"]``, in its order; empty when no ACS
            household is selected.
        staging_lineage: Staging's (``SERIALNO``, ``SPORDER``) pairs for the
            persons of the selected ACS households, read from staging's
            output, not from ``acs_tables``. ``SPORDER`` may be an integral
            float (staging's pooled form).

    Returns:
        A :class:`~microcosm.frame.Frame` of the CPS rows then the ACS rows,
        each group table sorted by its id. Household weights are the two
        inputs, concatenated and carried through that sort, as
        :attr:`~microcosm.frame.WeightKind.IMPORTANCE` weights (a pool
        assembly's starting weights, the kind staging gives its pooled
        weights). ``metadata`` holds only the spine assembly record, under
        ``SPINE_ASSEMBLY_MANIFEST_KEY``; the mass log is empty.

    Raises:
        OnePopulationContractError: Naming the first clause the inputs break,
            or if the joined tables fail the frame's invariants or the
            assembly validator.
        TypeError: If the tables are not a mapping of DataFrames.
    """

    cps = _origin_tables(cps_tables, origin=_CPS)
    acs = _origin_tables(acs_tables, origin=_ACS)
    _require_cps_support_metadata(cps)
    _require_acs_support_metadata(acs)
    cps_weights = _household_weights(
        cps_household_weights, origin=_CPS, households=len(cps["household"])
    )
    acs_weights = _household_weights(
        acs_household_weights, origin=_ACS, households=len(acs["household"])
    )
    staging = _lineage_pairs(staging_lineage)

    origins = {_CPS: _assembled_cps(cps), _ACS: _assembled_acs(acs)}
    _require_spine_tags(origins)
    _require_disjoint_ids(origins)
    _require_unique_source_ranks(origins)
    _require_acs_lineage(origins[_ACS], staging)

    tables, household_order = _joined_tables(origins)
    record = spine_assembly_manifest(
        {
            entity: table.loc[
                pd.to_numeric(table[support_clone_index_column(entity)]).eq(0)
            ]
            for entity, table in tables.items()
        },
        channels=ASSEMBLED_CHANNELS,
    )
    weights = np.concatenate([cps_weights, acs_weights])[household_order]
    try:
        frame = Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(weights, WeightKind.IMPORTANCE)},
            metadata=record,
        )
        validate_assembly_provenance(frame, boundary=_BOUNDARY)
    except ValueError as exc:
        raise OnePopulationContractError(
            f"The joined one-population frame is not valid: {exc}"
        ) from exc
    return frame


# ------------------------------------------------------- engine variables


@runtime_checkable
class EngineVariableIndex(Protocol):
    """What the contract asks of the engine's variable metadata, at 2024.

    The answers must cohere: every input is known, is not computed at 2024
    and is not a structural column. The contract refuses an index whose
    answers do not.
    """

    def known_variables(self, columns: Iterable[str]) -> frozenset[str]:
        """The names among ``columns`` the engine defines, formulas included."""
        ...

    def computed_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        """The tables' columns the engine computes at 2024."""
        ...

    def inputs_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        """The tables' known, non-structural columns not computed at 2024."""
        ...


def _present_columns(tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
    return frozenset(
        column
        for table in tables.values()
        for column in table.columns
        if isinstance(column, str)
    )


class PolicyEngineUSVariableIndex:
    """:class:`EngineVariableIndex` over the installed policyengine-us sources.

    It wraps
    :class:`~microcosm.frame.adapters.policyengine_us.PolicyEngineUSVariableMetadataIndex`,
    built the first time an answer is needed (it parses the installed
    wheel's variable sources without importing the engine), never at
    construction or import.

    - "Computed at 2024" is that index's
      ``_engine_computed_columns(tables, period=2024)``, the call the release
      tool's export guard makes, so the contract and the release cannot
      disagree. It refuses the four compatibility aggregates at any period
      and never counts a structural id or membership column.
    - "Known" is ``variables()`` plus ``formula_owned_outputs``: every name
      the sources define, inputs and formulas alike.
    - "Input at 2024" is known, not structural and not computed at 2024. A
      variable whose only formula starts after 2024 (``weeks_worked`` at
      policyengine-us 2.2.1) is therefore an input here, although the
      year-blind ``variables()`` leaves it out.

    Args:
        metadata_index: The index to wrap. ``None`` builds the installed one
            on first use.
    """

    def __init__(self, metadata_index: Any | None = None) -> None:
        self._metadata_index = metadata_index

    def _index(self) -> Any:
        if self._metadata_index is None:
            from microcosm.frame.adapters.policyengine_us import (
                PolicyEngineUSVariableMetadataIndex,
            )

            self._metadata_index = PolicyEngineUSVariableMetadataIndex()
        return self._metadata_index

    def known_variables(self, columns: Iterable[str]) -> frozenset[str]:
        names = {column for column in columns if isinstance(column, str)}
        index = self._index()
        inputs = names.intersection(index.variables())
        formulas = names.intersection(index.formula_owned_outputs(names))
        return frozenset(inputs | formulas)

    def computed_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        return frozenset(
            self._index()._engine_computed_columns(tables, period=ENGINE_PERIOD)
        )

    def inputs_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        known = self.known_variables(_present_columns(tables))
        return frozenset(known - STRUCTURAL_COLUMNS - self.computed_at_2024(tables))


@dataclass(frozen=True)
class _EngineColumns:
    """One frame's columns by engine class, from one coherent index answer."""

    known: frozenset[str]
    computed: frozenset[str]
    inputs: frozenset[str]


def _engine_columns(
    tables: Mapping[str, pd.DataFrame],
    variable_index: EngineVariableIndex | None,
) -> _EngineColumns:
    """Ask the index (the installed engine's by default) and check the answers."""

    index = (
        variable_index if variable_index is not None else PolicyEngineUSVariableIndex()
    )
    present = _present_columns(tables)
    known = frozenset(index.known_variables(sorted(present)))
    computed = frozenset(index.computed_at_2024(tables))
    inputs = frozenset(index.inputs_at_2024(tables))
    problems = [
        f"{label} names columns no table holds: {sorted(names - present)[:5]}"
        for label, names in (
            ("known", known),
            ("computed at 2024", computed),
            ("input at 2024", inputs),
        )
        if not names <= present
    ]
    if not inputs <= known:
        problems.append(f"inputs not known: {sorted(inputs - known)[:5]}")
    if inputs & computed:
        problems.append(f"inputs also computed: {sorted(inputs & computed)[:5]}")
    if inputs & STRUCTURAL_COLUMNS:
        problems.append(
            f"inputs include structural columns: {sorted(inputs & STRUCTURAL_COLUMNS)}"
        )
    if problems:
        raise OnePopulationContractError(
            "The engine variable index answers do not cohere (an input must be "
            "known, not computed at 2024 and not structural): "
            + "; ".join(problems)
            + "."
        )
    return _EngineColumns(known=known, computed=computed, inputs=inputs)


# ----------------------------------------------------------- the contract


def require_one_population(
    frame: Frame,
    manifest: SelectionManifest,
    *,
    parent_tail_donor_ids: frozenset[int],
    declared_absences: Mapping[str, str] | None = None,
    variable_index: EngineVariableIndex | None = None,
    register: Mapping[str, str] | None = None,
) -> None:
    """Refuse a joined frame that breaks any clause of the contract.

    The clauses, in the order they run: assembled provenance (R1); one native
    row per CPS key and one row per ACS household, against ``manifest`` (R2);
    no column the engine computes at 2024 (R3); no null ESI premium and the
    premium present (R4); a classified adult in every SPM unit (R5); no null
    in an engine input at 2024 outside the always-on producers' listed
    outputs (R6); no null PUF tail column on a tax unit (R7); the tail donor
    set equal to ``parent_tail_donor_ids`` (R8); every stored lower-case
    column an engine variable or registered (R9); a model-named column filled
    on CPS rows and null on ACS rows only when ``declared_absences`` declares
    it (R10); no location-clone index column (R11); a congressional district
    on every household (R12); ``TYPEHUGQ`` on every ACS household (T1).

    Args:
        frame: The joined frame, built from :func:`to_assembled_provenance`.
        manifest: PR 1's selection manifest for the same selection.
        parent_tail_donor_ids: The parent export's applied capital-gains tail
            donor ids.
        declared_absences: Column to the owner's decision, for each
            model-named column ACS rows lack by decision.
        variable_index: The engine's variable classes at 2024. ``None`` uses
            :class:`PolicyEngineUSVariableIndex`.
        register: The stored non-variable register. ``None`` uses
            ``microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS``.

    Raises:
        OnePopulationContractError: Naming the first clause the frame breaks.
    """

    raise NotImplementedError(
        "require_one_population is built in a later phase of d2283 PR 2."
    )


def combined_spm_role_record(
    cps_record: Mapping[str, Any],
    staging_receipt: Mapping[str, Any],
    frame: Frame,
) -> dict[str, object]:
    """The SPM independence role record for the joined frame (F-17, K-6).

    ``cps_record`` is a gate-keyed record over exactly the frame's CPS rows,
    which the prepare step re-derives from raw ASEC inputs; it is checked
    against a recount of those rows (counts and digest) and must say
    ``role_source == "source_column"``. ``staging_receipt`` is staging's ACS
    role receipt, whose keys are checked. The result recounts the whole
    frame in the keys
    :func:`~microcosm.build.us_runtime.spm_independence_role.us_spm_independence_role_signal_gate`
    reads, keeps per-origin counts, and cites both origins' evidence.

    Raises:
        OnePopulationContractError: If either origin's evidence disagrees
            with the frame.
    """

    raise NotImplementedError(
        "combined_spm_role_record is built in a later phase of d2283 PR 2."
    )


def absences(
    frame: Frame,
    *,
    fill_manifest: Mapping[str, Any],
    zero_written: Mapping[str, str],
    declared: Mapping[str, str],
) -> dict[str, object]:
    """What one origin carries as a default fill, a written zero, or not at all.

    Returns the content of :data:`ABSENCES_FILENAME`: one entry per column
    with its origin, rows and weighted share and a kind. The kinds are
    "owner-declared" (needs the owner's decision, from ``declared``),
    "release-written" (an always-on producer's listed output), "raw source
    column" (upper-case survey columns), "reviewed default fill" (from the
    prepare step's ``fill_manifest``) and "zero written" (``zero_written``
    maps each such column to the receipt that records it). Both directions
    are listed: columns CPS rows fill and ACS rows lack, and the reverse.
    Origins are keyed by support channel.
    """

    raise NotImplementedError("absences is built in a later phase of d2283 PR 2.")


def _is_peridnum(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= _CPS_UNIT_ID_LENGTH
        and value.isascii()
        and value.isdigit()
    )


def cps_unit_ids(person: pd.DataFrame) -> pd.DataFrame:
    """Each CPS household's income year, unit id and selection key.

    The unit id is ``PERIDNUM[:20]`` (H_IDNUM), on which every person of the
    household must agree, as the support-mix bake-off reads it
    (``tools/bakeoff_us_support_mix.py`` ``_row_index_columns``). It comes
    from ``PERIDNUM`` when the column is present, else from
    ``source_person_id`` (the ASEC pool writes ``PERIDNUM`` there). Either
    way every value must be a string of at least 20 ASCII digits: the ASEC
    pool's integer fallback, a number cast to text and a null are refused,
    never sliced (unlike the bake-off's ``astype(str)``). The income year is
    ``source_year``, an integer on which the household's persons agree. The
    key is
    :func:`~microcosm.build.us_runtime.one_population_selection.cps_unit_key`,
    so the prepare step's selection keys (PR 4 builds them with this
    function) and the contract's key check share one rule; digest a key set
    with :func:`~microcosm.build.us_runtime.one_population_selection.keys_sha256`.

    Every physical household gets a row: a source household's PUF clone
    households carry its PERIDNUM, so they share its key. The selection's
    ``cps_keys`` are the clone-0 households' keys.

    Args:
        person: CPS person rows with ``person_household_id``,
            ``source_year`` and ``PERIDNUM`` or ``source_person_id``.

    Returns:
        One row per household id (``person_household_id``), sorted, in an
        index named ``household_id``, with columns ``income_year`` (int64),
        ``unit_id`` and ``key``.

    Raises:
        OnePopulationContractError: If an id or year is missing or malformed,
            or a household's persons disagree.
    """

    if not isinstance(person, pd.DataFrame):
        raise TypeError(f"person must be a DataFrame, got {type(person).__name__}.")
    household_column = US_SCHEMA.membership_column("household")
    if _PERIDNUM in person:
        column = _PERIDNUM
    elif _SOURCE_PERSON_ID in person:
        column = _SOURCE_PERSON_ID
    else:
        raise OnePopulationContractError(
            f"CPS unit ids need {_PERIDNUM} or {_SOURCE_PERSON_ID} on the persons."
        )
    missing = [c for c in (household_column, _SOURCE_YEAR) if c not in person]
    if missing:
        raise OnePopulationContractError(f"CPS unit ids need {missing} on the persons.")
    values = person[column]
    bad = ~values.map(_is_peridnum).astype(bool)
    if bad.any():
        raise OnePopulationContractError(
            f"CPS unit ids need {column} as a string of at least "
            f"{_CPS_UNIT_ID_LENGTH} digits on every person (the ASEC PERIDNUM); "
            f"{int(bad.sum())} person(s) hold values such as "
            f"{values[bad].head(3).tolist()!r}."
        )
    households = _integral_ids(
        person[household_column],
        refusal=f"CPS unit ids need integral {household_column!r}",
    )
    years = _integral_ids(
        person[_SOURCE_YEAR], refusal=f"CPS unit ids need an integral {_SOURCE_YEAR}"
    )
    per_person = pd.DataFrame(
        {
            "household_id": households,
            "income_year": years,
            "unit_id": [value[:_CPS_UNIT_ID_LENGTH] for value in values.tolist()],
        }
    )
    grouped = per_person.groupby("household_id", sort=True)
    for name, label in (
        ("unit_id", f"{column}[:{_CPS_UNIT_ID_LENGTH}] (H_IDNUM)"),
        ("income_year", _SOURCE_YEAR),
    ):
        split = grouped[name].nunique()
        if (split != 1).any():
            raise OnePopulationContractError(
                f"CPS unit ids: persons of one household disagree on {label} in "
                f"household(s) {split.index[split != 1].tolist()[:5]}."
            )
    result = grouped[["income_year", "unit_id"]].first()
    try:
        keys = [
            cps_unit_key(int(year), unit)
            for year, unit in zip(
                result["income_year"].tolist(), result["unit_id"].tolist(), strict=True
            )
        ]
    except ValueError as exc:
        raise OnePopulationContractError(
            f"CPS unit ids: a household's key is malformed: {exc}"
        ) from exc
    return pd.DataFrame(
        {
            "income_year": result["income_year"].to_numpy(dtype=np.int64),
            "unit_id": pd.Series(result["unit_id"].tolist(), dtype=object).to_numpy(),
            "key": pd.Series(keys, dtype=object).to_numpy(),
        },
        index=pd.Index(result.index.to_numpy(dtype=np.int64), name="household_id"),
    )
