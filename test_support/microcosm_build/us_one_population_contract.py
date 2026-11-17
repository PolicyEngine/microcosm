"""A generated d2283 one-population input and a stand-in engine variable index.

d2283's population contract (``us_runtime/one_population_contract.py``) takes
two inputs: the parent Route A export's CPS rows and a staging run's ACS rows.
:func:`generate_population` builds one small, valid pair from a seed:

- CPS tables in the export's form: six entities; native ASEC rows at clone
  index 0 (channel ``asec``), every household's PUF tax-detail copy at clone
  index 1 and, for a few households, the capital-gains tail copy at clone
  index 2 (both ``puf_tax_detail``), with ids shifted by
  ``puf_support._remap_ids``'s multiplier and the tail's extra multiplier;
  ``<entity>_source_id``, ``<entity>_support_channel`` and
  ``<entity>_support_clone_index`` on every entity and no spine source id or
  spine tag; ``PERIDNUM`` (22 digits, the first 20 shared by a household),
  ``source_year`` and the other ASEC lineage columns; the six tail
  provenance columns on every tax unit, with applied donors on the tail
  copies.
- ACS tables in staging's form (``base_pool._prepared_source_view``): channel
  and ``<entity>_spine`` ``acs_2024_1yr``, clone index 0, native ids in
  ``<entity>_source_id`` and pooled ids offset past the CPS maximum by
  ``base_pool._id_remap_offsets``'s rule; household ``SERIALNO`` and
  ``TYPEHUGQ``; person ``SPORDER`` and the PUMS lineage columns. They also
  carry what the prepare step fills before the contract runs: the ESI
  premium (PR 0) and the six tail columns at the transfer's defaults.

The ACS tables :func:`generate_population` returns hold only the ACS rows'
own columns: the SSI take-up column (which the release writes) and the raw
ASEC columns are absent. Staging pools the union of both origins' columns
(``base_pool._aligned_column_orders`` and the concat in
``base_pool._combined_tables``), so on its output those columns are present
and all null on ACS rows, and an ACS-only integer column (``SPORDER``,
``TYPEHUGQ``) comes back as float64. :func:`with_staging_union_columns`
gives that form; a clause that reads "absent" and "all null" differently
should be tested on both.

Both origins come from one column spec, so a shared column keeps one dtype
after the two are joined. Every SPM unit has a classified adult, and the
weighted SPM role shares sit inside the role gate's bands by construction.

:class:`StandInVariableIndex` answers the contract's engine questions without
an engine: its input list is the engine ABI lock's committed projection (the
926 names ``variables()`` returned at policyengine-us 2.2.1), plus explicit
small sets of names computed at 2024 and of names whose only formula starts
after 2024.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cache
from importlib.resources import files
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.one_population_contract import (
    ACS_SUPPORT_CHANNEL,
    ASSEMBLED_CHANNELS,
    CPS_SPINE_TAG,
    STRUCTURAL_COLUMNS,
    to_assembled_provenance,
)
from microcosm.build.us_runtime.one_population_selection import (
    acs_unit_key,
    cps_unit_key,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
)
from microcosm.build.us_runtime.spm_independence_role import (
    US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_assembly_manifest,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

# ------------------------------------------------------ stand-in index

ENGINE_ABI_LOCK = files("microcosm.build.us").joinpath("engine_abi.lock.json")
_PROJECTION_CONSUMER = "_simulation_projection"

#: Formula variables the stand-in computes at 2024 (none is in the lock).
STAND_IN_COMPUTED_AT_2024 = frozenset(
    {"adjusted_gross_income", "employment_income", "income_tax", "snap", "ssi"}
)
#: The aggregates the release's guard refuses at any period
#: (``policyengine_us._FORMULA_OWNED_COMPAT_COLUMNS`` in microcosm-frame).
STAND_IN_COMPAT_AGGREGATES = frozenset(
    {
        "dividend_income",
        "ordinary_dividend_income",
        "partnership_s_corp_income",
        "social_security",
    }
)
#: Known variables whose only formula starts after 2024: policyengine-us
#: 2.2.1 gives ``weeks_worked`` a ``formula_2025`` and no other.
STAND_IN_FORMULA_AFTER_2024 = frozenset({"weeks_worked"})
#: Structural variables with a formula, so absent from the year-blind input
#: list: policyengine-us's ``person_id`` is ``np.arange``.
STAND_IN_STRUCTURAL_FORMULAS = frozenset({"person_id"})

#: The always-on output prepare leaves null on ACS rows and the release
#: writes on both origins (d2283 K-2; on the contract's exempt list).
RELEASE_WRITTEN_ON_ACS = frozenset({"takes_up_ssi_if_eligible"})


@cache
def lock_engine_inputs() -> Mapping[str, str]:
    """The ABI lock's engine-input projection: variable name to entity.

    Read as package data, so no engine is imported. The count and digest
    are checked against the lock's own receipt, computed as
    ``multispine_pool._pool_engine_input_projection`` computes them.
    """

    payload = json.loads(ENGINE_ABI_LOCK.read_text(encoding="utf-8"))
    manifest = payload["remaining_stage_input_manifest"]
    rows = [row for row in manifest["rows"] if row["consumer"] == _PROJECTION_CONSUMER]
    projection = {row["variable"]: row["entity"] for row in rows}
    if len(projection) != len(rows):
        raise AssertionError("The lock's engine-input projection repeats a variable.")
    digest = hashlib.sha256(
        json.dumps(
            [
                {"entity": entity, "variable": variable}
                for variable, entity in sorted(projection.items())
            ],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    contract = manifest["receipt"]["engine_input_projection_contract"]
    if (len(projection), digest) != (contract["input_count"], contract["sha256"]):
        raise AssertionError(
            "The lock's engine-input projection does not match its receipt: "
            f"{len(projection)} names, digest {digest}; receipt {contract}."
        )
    return MappingProxyType(projection)


def _present(tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
    return frozenset(
        column
        for table in tables.values()
        for column in table.columns
        if isinstance(column, str)
    )


@dataclass(frozen=True)
class StandInVariableIndex:
    """An :class:`EngineVariableIndex` with no engine behind it.

    ``inputs`` is the year-blind input list (``variables()``); ``computed``
    holds what the engine computes at 2024, compatibility aggregates
    included; ``formula_after_2024`` holds known names with no formula at
    2024; ``structural_formulas`` holds structural names that only a formula
    defines. Answers follow
    :class:`~microcosm.build.us_runtime.one_population_contract.PolicyEngineUSVariableIndex`'s
    rules: computed never counts a structural column, and an input at 2024 is
    known, not structural and not computed.
    """

    inputs: frozenset[str]
    computed: frozenset[str] = STAND_IN_COMPUTED_AT_2024 | STAND_IN_COMPAT_AGGREGATES
    formula_after_2024: frozenset[str] = STAND_IN_FORMULA_AFTER_2024
    structural_formulas: frozenset[str] = STAND_IN_STRUCTURAL_FORMULAS

    @classmethod
    def from_lock(cls) -> StandInVariableIndex:
        return cls(inputs=frozenset(lock_engine_inputs()))

    @property
    def known(self) -> frozenset[str]:
        return (
            self.inputs
            | self.computed
            | self.formula_after_2024
            | self.structural_formulas
        )

    def known_variables(self, columns: Iterable[str]) -> frozenset[str]:
        return frozenset(set(columns) & self.known)

    def computed_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        return frozenset((_present(tables) & self.computed) - STRUCTURAL_COLUMNS)

    def inputs_at_2024(self, tables: Mapping[str, pd.DataFrame]) -> frozenset[str]:
        known = self.known_variables(_present(tables))
        return frozenset(known - STRUCTURAL_COLUMNS - self.computed_at_2024(tables))


# --------------------------------------------------------- the generator

CPS_INCOME_YEARS = (2023, 2024)
ACS_SOURCE_YEAR = 2024
SPM_ROLE = "is_spm_independent_minor_role"
TAIL_DONOR_ID_BASE = 7_000_000

TAIL_PROVENANCE_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN: False,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN: -1,
        PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN: False,
        PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN: 0,
        PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN: -1,
        PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN: 0.0,
    }
)

#: (state FIPS, districts): states with numbered districts only.
_STATES = ((6, 52), (12, 28), (36, 26), (42, 17), (48, 38))
_ROLE_SHARE_BAND = (0.40, 0.75)
_MINOR_ROLE_SHARE_BAND = (0.003, 0.06)

#: One dtype per column, shared by both origins.
_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "age": "int64",
        "is_female": "bool",
        "is_household_head": "bool",
        SPM_ROLE: "bool",
        "employer_sponsored_insurance_premiums": "float64",
        "tip_income": "float64",
        "takes_up_ssi_if_eligible": "bool",
        "SSI_VAL": "float64",
        "PERIDNUM": object,
        "SERIALNO": object,
        "TYPEHUGQ": "int64",
        "SPORDER": "int64",
        "source_year": "int64",
        "source_household_id": "int64",
        "source_person_id": object,
        "source_row_id": "int64",
        "state_fips": "int64",
        "congressional_district_geoid": "int64",
        PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN: "bool",
        PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN: "int64",
        PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN: "bool",
        PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN: "int64",
        PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN: "int64",
        PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN: "float64",
        **{
            column: "int64"
            for entity in US_SCHEMA.entities
            for column in (
                US_SCHEMA.entity_id_column(entity),
                support_source_id_column(entity),
                support_clone_index_column(entity),
            )
        },
        **{
            US_SCHEMA.membership_column(group): "int64"
            for group in US_SCHEMA.group_entities
        },
        **{
            column: object
            for entity in US_SCHEMA.entities
            for column in (support_channel_column(entity), f"{entity}_spine")
        },
    }
)


def _typed(table: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(table.columns) - set(_DTYPES))
    if unknown:
        raise AssertionError(f"Generator columns without a dtype: {unknown}.")
    return table.astype({column: _DTYPES[column] for column in table.columns})


@dataclass(frozen=True)
class _Member:
    kind: str  # head, spouse, child, adult_child, roommate, minor_head, gq
    age: int
    female: bool


@dataclass(frozen=True)
class _Household:
    members: tuple[_Member, ...]
    state_fips: int
    district: int
    weight: float
    typehugq: int = 1


_ROLE_KINDS = frozenset({"head", "spouse", "roommate", "minor_head", "gq"})
_HEAD_KINDS = frozenset({"head", "minor_head", "gq"})
_PATTERN = ("couple", "single_parent", "single", "couple_adult_child", "roommates")


def _household(
    rng: np.random.Generator, index: int, *, weight: float, kind: str | None = None
) -> _Household:
    kind = kind or _PATTERN[index % len(_PATTERN)]
    head_female = bool(rng.random() < 0.5)
    members = [_Member("head", int(rng.integers(25, 81)), head_female)]
    n_children = 0
    if kind in ("couple", "couple_adult_child"):
        members.append(_Member("spouse", int(rng.integers(25, 81)), not head_female))
    if kind == "couple":
        n_children = int(rng.integers(0, 4))
    elif kind == "single_parent":
        n_children = int(rng.integers(1, 4))
    for child in range(n_children):
        # The first child of every other family is 15 to 17, so the weighted
        # share of role holders among 15-to-17-year-olds stays small.
        teen = child == 0 and index % 2 == 0
        age = int(rng.integers(15, 18)) if teen else int(rng.integers(0, 18))
        members.append(_Member("child", age, bool(rng.random() < 0.5)))
    if kind == "couple_adult_child":
        members.append(
            _Member("adult_child", int(rng.integers(18, 31)), bool(rng.random() < 0.5))
        )
    if kind == "roommates":
        members.append(
            _Member("roommate", int(rng.integers(20, 70)), bool(rng.random() < 0.5))
        )
    state, districts = _STATES[int(rng.integers(0, len(_STATES)))]
    return _Household(
        members=tuple(members),
        state_fips=state,
        district=int(rng.integers(1, districts + 1)),
        weight=weight,
    )


def _unit_groups(members: tuple[_Member, ...]) -> dict[str, list[list[int]]]:
    """Member positions per unit, for each group entity but the household."""

    if members[0].kind in ("minor_head", "gq"):
        everyone = [list(range(len(members)))]
        return {
            "tax_unit": everyone,
            "spm_unit": everyone,
            "family": everyone,
            "marital_unit": [[position] for position in range(len(members))],
        }
    core = [
        position
        for position, member in enumerate(members)
        if member.kind in ("head", "spouse", "child")
    ]
    family = core + [
        position
        for position, member in enumerate(members)
        if member.kind == "adult_child"
    ]
    outsiders = [
        [position]
        for position, member in enumerate(members)
        if member.kind not in ("head", "spouse", "child")
    ]
    roommates = [
        [position]
        for position, member in enumerate(members)
        if member.kind == "roommate"
    ]
    couple = [
        position
        for position, member in enumerate(members)
        if member.kind in ("head", "spouse")
    ]
    singles = [
        [position]
        for position, member in enumerate(members)
        if member.kind not in ("head", "spouse")
    ]
    return {
        "tax_unit": [core, *outsiders],
        "spm_unit": [family, *roommates],
        "family": [family, *roommates],
        "marital_unit": [couple, *singles],
    }


def _native_tables(
    households: list[_Household],
    rng: np.random.Generator,
    *,
    first_person_id: int,
) -> tuple[dict[str, pd.DataFrame], list[list[int]]]:
    """Native tables (ids from 1, persons from ``first_person_id``).

    Returns the tables and, per household, its persons' row positions.
    """

    person_rows: list[dict[str, object]] = []
    household_rows: list[dict[str, object]] = []
    groups: dict[str, list[dict[str, object]]] = {
        group: [] for group in US_SCHEMA.group_entities if group != "household"
    }
    positions: list[list[int]] = []
    for household_index, household in enumerate(households):
        household_id = household_index + 1
        household_rows.append(
            {
                "household_id": household_id,
                "state_fips": household.state_fips,
                "congressional_district_geoid": household.state_fips * 100
                + household.district,
            }
        )
        unit_ids: dict[str, list[int]] = {}
        for group, units in _unit_groups(household.members).items():
            ids = [0] * len(household.members)
            for unit in units:
                unit_id = len(groups[group]) + 1
                groups[group].append({f"{group}_id": unit_id})
                for position in unit:
                    ids[position] = unit_id
            unit_ids[group] = ids
        mine = []
        for position, member in enumerate(household.members):
            adult = member.age >= 22
            mine.append(len(person_rows))
            person_rows.append(
                {
                    "person_id": first_person_id + len(person_rows),
                    "person_household_id": household_id,
                    **{
                        US_SCHEMA.membership_column(group): unit_ids[group][position]
                        for group in unit_ids
                    },
                    "age": member.age,
                    "is_female": member.female,
                    "is_household_head": member.kind in _HEAD_KINDS,
                    SPM_ROLE: member.kind in _ROLE_KINDS,
                    "employer_sponsored_insurance_premiums": (
                        float(rng.integers(10, 80) * 100)
                        if adult and member.age < 65 and rng.random() < 0.5
                        else 0.0
                    ),
                    "tip_income": (
                        float(rng.integers(5, 50) * 100)
                        if adult and rng.random() < 0.1
                        else 0.0
                    ),
                }
            )
        positions.append(mine)
    tables = {
        "person": pd.DataFrame(person_rows),
        "household": pd.DataFrame(household_rows),
        **{group: pd.DataFrame(rows) for group, rows in groups.items()},
    }
    return tables, positions


def _clone_multiplier(tables: Mapping[str, pd.DataFrame]) -> int:
    """``puf_support._id_multiplier_for_values`` over every entity's ids."""

    max_id = max(
        int(tables[entity][US_SCHEMA.entity_id_column(entity)].abs().max())
        for entity in US_SCHEMA.entities
    )
    return 10 ** max(1, len(str(max_id)))


def _shift_ids(table: pd.DataFrame, entity: str, shift: int) -> pd.DataFrame:
    table = table.copy()
    table[US_SCHEMA.entity_id_column(entity)] += shift
    if entity == US_SCHEMA.person_entity:
        for group in US_SCHEMA.group_entities:
            table[US_SCHEMA.membership_column(group)] += shift
    return table


def _cps_tables(
    households: list[_Household],
    rng: np.random.Generator,
    *,
    tail_every: int,
) -> tuple[dict[str, pd.DataFrame], np.ndarray, int, frozenset[int], tuple[str, ...]]:
    native, positions = _native_tables(households, rng, first_person_id=1)
    person = native["person"]

    # ASEC lineage: each household gets an income year, H_IDNUM and PH_SEQ.
    peridnum = [""] * len(person)
    source_year = [0] * len(person)
    source_household_id = [0] * len(person)
    source_row_id = [0] * len(person)
    seen: set[str] = set()
    keys: list[str] = []
    rows_by_year = dict.fromkeys(CPS_INCOME_YEARS, 0)
    households_by_year = dict.fromkeys(CPS_INCOME_YEARS, 0)
    for household_index, rows in enumerate(positions):
        year = CPS_INCOME_YEARS[household_index % len(CPS_INCOME_YEARS)]
        h_idnum = (
            f"{int(rng.integers(1, 10))}{int(rng.integers(0, 10**9)):09d}"
            f"{int(rng.integers(0, 10**10)):010d}"
        )
        if h_idnum in seen:  # pragma: no cover - 1 in 10**19
            raise AssertionError("Generated H_IDNUM repeats.")
        seen.add(h_idnum)
        keys.append(cps_unit_key(year, h_idnum))
        households_by_year[year] += 1
        for line, row in enumerate(rows, start=1):
            peridnum[row] = f"{h_idnum}{line:02d}"
            source_year[row] = year
            source_household_id[row] = households_by_year[year]
            source_row_id[row] = rows_by_year[year]
            rows_by_year[year] += 1
    person["PERIDNUM"] = peridnum
    person["source_year"] = source_year
    person["source_household_id"] = source_household_id
    person["source_person_id"] = peridnum
    person["source_row_id"] = source_row_id
    adult = person["age"].to_numpy() >= 18
    person["SSI_VAL"] = np.where(
        adult & (rng.random(len(person)) < 0.08),
        rng.integers(20, 90, len(person)) * 100.0,
        0.0,
    )
    person["takes_up_ssi_if_eligible"] = rng.random(len(person)) < 0.6
    for column, default in TAIL_PROVENANCE_DEFAULTS.items():
        native["tax_unit"][column] = default

    # puf_support._clone_entity_table: clone 0 keeps its ids, clone 1 moves
    # every id (memberships too) by the multiplier; both keep the native id
    # as the source id.
    multiplier = _clone_multiplier(native)
    cloned: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        copies = []
        for clone_index, channel in enumerate(
            (BASE_ASEC_SUPPORT_CHANNEL, PUF_TAX_DETAIL_SUPPORT_CHANNEL)
        ):
            copy = _shift_ids(native[entity], entity, clone_index * multiplier)
            copy[support_source_id_column(entity)] = native[entity][
                US_SCHEMA.entity_id_column(entity)
            ].to_numpy()
            copy[support_channel_column(entity)] = channel
            copy[support_clone_index_column(entity)] = clone_index
            copies.append(copy)
        cloned[entity] = pd.concat(copies, ignore_index=True)

    # puf_capital_gains_tail._clone_and_transfer: copy the selected
    # households' clone-1 rows on every entity, move their ids by one more
    # multiplier, set clone index 2, keep channel and source id; stamp the
    # recipient tax unit's tail provenance.
    selected = [index + 1 for index in range(0, len(households), tail_every)]
    person = cloned["person"]
    clone_one = person[support_clone_index_column("person")] == 1
    tail_people = person.loc[
        clone_one & person["person_household_id"].isin(np.add(selected, multiplier))
    ]
    tail_ids = {
        "person": set(tail_people["person_id"]),
        **{
            group: set(tail_people[US_SCHEMA.membership_column(group)])
            for group in US_SCHEMA.group_entities
        },
    }
    weights = np.array([household.weight for household in households])
    household_weights = np.concatenate([weights / 2.0, weights / 2.0])
    tail_weights = []
    for entity in US_SCHEMA.entities:
        table = cloned[entity]
        rows = table.loc[
            table[US_SCHEMA.entity_id_column(entity)].isin(tail_ids[entity])
        ]
        tail = _shift_ids(rows, entity, multiplier)
        tail[support_clone_index_column(entity)] = 2
        cloned[entity] = pd.concat([table, tail], ignore_index=True)
    tax_unit = cloned["tax_unit"]
    donors: list[int] = []
    for number, household_id in enumerate(selected):
        weight = households[household_id - 1].weight / 2.0
        household_weights[len(households) + household_id - 1] = weight * 0.75
        tail_weights.append(weight * 0.25)
        head = person.loc[
            clone_one & (person["person_household_id"] == household_id + multiplier)
        ].iloc[0]
        target = tax_unit["tax_unit_id"] == int(head["person_tax_unit_id"]) + multiplier
        donor = TAIL_DONOR_ID_BASE + 17 * number
        donors.append(donor)
        tax_unit.loc[target, PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN] = True
        tax_unit.loc[target, PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN] = donor
        tax_unit.loc[target, PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN] = bool(
            number % 2
        )
        tax_unit.loc[target, PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN] = (
            1 + number % 2
        )
        tax_unit.loc[target, PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN] = number % 3
        tax_unit.loc[target, PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN] = (
            weight * 0.25
        )
    household_weights = np.concatenate([household_weights, tail_weights])
    tables = {entity: _typed(table) for entity, table in cloned.items()}
    return tables, household_weights, multiplier, frozenset(donors), tuple(keys)


def _acs_tables(
    households: list[_Household],
    rng: np.random.Generator,
    cps_tables: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], np.ndarray, dict[str, int], tuple[str, ...]]:
    # acs_pums: households sorted by SERIALNO and numbered 1..N in that
    # order; persons by (SERIALNO, SPORDER), person ids 0..N-1.
    serials = [
        f"{ACS_SOURCE_YEAR}{'GQ' if household.typehugq != 1 else 'HU'}"
        f"{int(rng.integers(0, 10**7)):07d}"
        for household in households
    ]
    if len(set(serials)) != len(serials):  # pragma: no cover - seeded
        raise AssertionError("Generated SERIALNO repeats.")
    order = sorted(range(len(households)), key=serials.__getitem__)
    households = [households[index] for index in order]
    serials = [serials[index] for index in order]
    native, positions = _native_tables(households, rng, first_person_id=0)
    native["household"]["SERIALNO"] = serials
    native["household"]["TYPEHUGQ"] = [household.typehugq for household in households]
    person = native["person"]
    sporder = np.zeros(len(person), dtype=np.int64)
    for rows in positions:
        sporder[rows] = np.arange(1, len(rows) + 1)
    person["SPORDER"] = sporder
    person["source_year"] = ACS_SOURCE_YEAR
    person["source_household_id"] = person["person_household_id"].to_numpy()
    person["source_person_id"] = [str(int(value)) for value in sporder]
    person["source_row_id"] = np.arange(len(person), dtype=np.int64)
    for column, default in TAIL_PROVENANCE_DEFAULTS.items():
        native["tax_unit"][column] = default

    # base_pool._prepared_source_view with the export as donor: tag the
    # spine, copy the native id into the source id, then offset each entity's
    # ids (memberships by their group's offset) past the donor's maximum
    # when the id sets overlap (_id_remap_offsets).
    offsets: dict[str, int] = {}
    for entity in US_SCHEMA.entities:
        id_column = US_SCHEMA.entity_id_column(entity)
        mine = cps_tables[entity][id_column].to_numpy()
        theirs = native[entity][id_column].to_numpy()
        if np.intersect1d(mine, theirs).size:
            offsets[entity] = int(mine.max()) + 1 - int(theirs.min())
    tables: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        source = native[entity]
        table = source.copy()
        table[f"{entity}_spine"] = ACS_SUPPORT_CHANNEL
        table[support_source_id_column(entity)] = source[
            US_SCHEMA.entity_id_column(entity)
        ].to_numpy()
        table[support_channel_column(entity)] = ACS_SUPPORT_CHANNEL
        table[support_clone_index_column(entity)] = 0
        table[US_SCHEMA.entity_id_column(entity)] += offsets.get(entity, 0)
        if entity == US_SCHEMA.person_entity:
            for group in US_SCHEMA.group_entities:
                table[US_SCHEMA.membership_column(group)] += offsets.get(group, 0)
        tables[entity] = _typed(table)
    weights = np.array([household.weight for household in households])
    keys = tuple(acs_unit_key(serial) for serial in serials)
    return tables, weights, offsets, keys


@dataclass(frozen=True, eq=False)
class GeneratedPopulation:
    """One valid input pair for the contract, with what a test needs to know.

    ``cps_household_weights`` and ``acs_household_weights`` align with each
    origin's household table. ``clone_multiplier`` is the CPS clone id shift,
    ``acs_id_offsets`` staging's per-entity ACS offsets, ``tail_donor_ids``
    the applied tail donors, and ``cps_keys``/``acs_keys`` the selection keys
    of the native CPS households and the ACS households.
    """

    seed: int
    cps_tables: Mapping[str, pd.DataFrame]
    acs_tables: Mapping[str, pd.DataFrame]
    cps_household_weights: np.ndarray
    acs_household_weights: np.ndarray
    clone_multiplier: int
    acs_id_offsets: Mapping[str, int]
    tail_donor_ids: frozenset[int]
    cps_keys: tuple[str, ...]
    acs_keys: tuple[str, ...]


def generate_population(
    seed: int = 0,
    *,
    n_cps_households: int = 24,
    n_acs_households: int = 24,
    tail_every: int = 4,
) -> GeneratedPopulation:
    """Generate one valid CPS-and-ACS population input from ``seed``.

    The ACS side always holds one household whose only member at least 15 is
    a 17-year-old head with the SPM role (so the role, not age alone,
    classifies its SPM unit's adult) and two group-quarters households
    (``TYPEHUGQ`` 2 and 3).
    """

    rng = np.random.default_rng(seed)
    cps_households = [
        _household(rng, index, weight=float(rng.integers(800, 1201)))
        for index in range(n_cps_households)
    ]
    acs_households = [
        _household(rng, index, weight=float(rng.integers(80, 121)))
        for index in range(n_acs_households - 3)
    ]
    state, districts = _STATES[0]
    acs_households.append(
        _Household(
            members=(
                _Member("minor_head", 17, bool(rng.random() < 0.5)),
                _Member("child", 12, bool(rng.random() < 0.5)),
            ),
            state_fips=state,
            district=1,
            weight=100.0,
        )
    )
    for typehugq, age in ((2, 82), (3, 19)):
        acs_households.append(
            _Household(
                members=(_Member("gq", age, bool(rng.random() < 0.5)),),
                state_fips=state,
                district=districts,
                weight=100.0,
                typehugq=typehugq,
            )
        )
    cps, cps_weights, multiplier, donors, cps_keys = _cps_tables(
        cps_households, rng, tail_every=tail_every
    )
    acs, acs_weights, offsets, acs_keys = _acs_tables(acs_households, rng, cps)
    population = GeneratedPopulation(
        seed=seed,
        cps_tables=MappingProxyType(cps),
        acs_tables=MappingProxyType(acs),
        cps_household_weights=cps_weights,
        acs_household_weights=acs_weights,
        clone_multiplier=multiplier,
        acs_id_offsets=MappingProxyType(offsets),
        tail_donor_ids=donors,
        cps_keys=cps_keys,
        acs_keys=acs_keys,
    )
    _check_role_bands(joined_frame(population))
    return population


def with_staging_union_columns(population: GeneratedPopulation) -> GeneratedPopulation:
    """``population`` with its ACS tables in staging's pooled column union.

    Staging concatenates the donor's rows (here, the CPS tables, tagged
    ``asec_puf``) and the ACS rows over the union of their columns, donor
    columns first. Its ACS rows therefore carry every CPS-only column all
    null, in the dtype the concat gives it, and their own integer-only
    columns come back as float64. The CPS tables and weights are unchanged.
    """

    acs: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        donor = population.cps_tables[entity].assign(
            **{f"{entity}_spine": CPS_SPINE_TAG}
        )
        pooled = pd.concat(
            [donor, population.acs_tables[entity]], ignore_index=True, sort=False
        )
        acs[entity] = pooled.iloc[len(donor) :].reset_index(drop=True)
    return replace(population, acs_tables=MappingProxyType(acs))


# ------------------------------------------------------ joined frame


def staging_lineage(population: GeneratedPopulation) -> frozenset[tuple[str, int]]:
    """The ACS persons' (``SERIALNO``, ``SPORDER``) pairs, as staging holds them.

    The generated ACS tables are staging's rows for the selected households,
    so their pairs are what the prepare step would read from staging.
    """

    household = population.acs_tables["household"]
    person = population.acs_tables["person"]
    serial = person["person_household_id"].map(
        household.set_index("household_id")["SERIALNO"]
    )
    return frozenset(
        zip(
            serial.tolist(),
            (int(value) for value in person["SPORDER"].tolist()),
            strict=True,
        )
    )


def assembled_frame(population: GeneratedPopulation) -> Frame:
    """The contract's provenance rewrite of ``population``."""

    return to_assembled_provenance(
        population.cps_tables,
        population.acs_tables,
        cps_household_weights=population.cps_household_weights,
        acs_household_weights=population.acs_household_weights,
        staging_lineage=staging_lineage(population),
    )


def joined_frame(
    population: GeneratedPopulation,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Frame:
    """CPS and ACS rows in one frame, as given (no provenance rewrite).

    Group tables are sorted by id and the household weights follow them.
    """

    tables: dict[str, pd.DataFrame] = {}
    weights = np.concatenate(
        [population.cps_household_weights, population.acs_household_weights]
    )
    for entity in US_SCHEMA.entities:
        table = pd.concat(
            [population.cps_tables[entity], population.acs_tables[entity]],
            ignore_index=True,
            sort=False,
        )
        if entity in US_SCHEMA.group_entities:
            order = np.argsort(
                table[US_SCHEMA.id_column(entity)].to_numpy(), kind="stable"
            )
            table = table.iloc[order].reset_index(drop=True)
            if entity == "household":
                weights = weights[order]
        tables[entity] = table
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.IMPORTANCE)},
        metadata=metadata,
    )


def clone_zero_assembly_record(frame: Frame) -> dict[str, object]:
    """The three-channel spine assembly record over the clone-0 rows."""

    return spine_assembly_manifest(
        {
            entity: frame.table(entity).loc[
                frame.table(entity)[support_clone_index_column(entity)] == 0
            ]
            for entity in frame.entities
        },
        channels=ASSEMBLED_CHANNELS,
    )


def spm_role_gate_record(frame: Frame) -> dict[str, object]:
    """A record in the role gate's keys, computed by its rule over ``frame``.

    The digest restates ``spm_independence_role._role_binding_sha256``: the
    sha256 of the uint64 row hashes of the ordered person id, SPM unit
    membership, age and role columns.
    """

    person = frame.table("person")
    columns = ["person_id", "person_spm_unit_id", "age", SPM_ROLE]
    hashes = pd.util.hash_pandas_object(person[columns], index=False)
    return {
        "persons_joined": len(person),
        "table_rows": len(person),
        "native_spm_units": frame.n("spm_unit"),
        "true_role_persons": int(person[SPM_ROLE].sum()),
        "person_role_binding_sha256": hashlib.sha256(
            hashes.to_numpy(dtype="<u8").tobytes()
        ).hexdigest(),
    }


def with_spm_role_record(frame: Frame, record: Mapping[str, object]) -> Frame:
    """``frame`` with ``record`` under the role gate's metadata key."""

    return Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={
            **frame.metadata,
            US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY: dict(record),
        },
    )


def role_shares(frame: Frame) -> tuple[float, float]:
    """Weighted SPM role share overall and among 15-to-17-year-olds."""

    person = frame.table("person")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    role = person[SPM_ROLE].to_numpy(dtype=bool)
    age = person["age"].to_numpy(dtype=np.float64)
    minor = (age >= 15) & (age < 18)
    return (
        float(weights[role].sum() / weights.sum()),
        float(weights[role & minor].sum() / weights[minor].sum()),
    )


def _check_role_bands(frame: Frame) -> None:
    share, minor_share = role_shares(frame)
    for label, value, (lower, upper) in (
        ("role share", share, _ROLE_SHARE_BAND),
        ("minor role share", minor_share, _MINOR_ROLE_SHARE_BAND),
    ):
        if not lower <= value <= upper:
            raise AssertionError(
                f"Generated {label} {value:.4f} is outside [{lower}, {upper}]."
            )
