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

:func:`cps_role_record`, :func:`staging_role_receipt` and
:func:`fill_manifest` give the role record's and the absences file's inputs
in the shapes the prepare step will pass: a CPS-rows record in the role
gate's keys, staging's ACS role receipt, and the reviewed default fill's
manifest.

:class:`StandInVariableIndex` answers the contract's engine questions without
an engine: its input list is the engine ABI lock's committed projection (the
926 names ``variables()`` returned at policyengine-us 2.2.1), plus explicit
small sets of names computed at 2024 and of names whose only formula starts
after 2024.

:func:`reader_population` is the whole-frame differential's (X5) ASEC-only
population: a Route A frame made by the production clone and tail operators,
carrying every :data:`READER_GATES` reader's raw source columns, on which
each of those gates passes, and its provenance rewrite with no ACS row.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import cache
from importlib.resources import files
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.alimony import us_alimony_signal_gate
from microcosm.build.us_runtime.medicare_take_up import (
    US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS,
    us_medicare_take_up_signal_gate,
)
from microcosm.build.us_runtime.one_population_contract import (
    ACS_SUPPORT_CHANNEL,
    ASSEMBLED_CHANNELS,
    CPS_SPINE_TAG,
    STRUCTURAL_COLUMNS,
    to_assembled_provenance,
)
from microcosm.build.us_runtime.one_population_selection import (
    RULE_VERSION,
    SelectionManifest,
    acs_unit_key,
    cps_unit_key,
    keys_sha256,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    us_retirement_contributions_signal_gate,
)
from microcosm.build.us_runtime.retirement_distributions import (
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    us_retirement_distributions_signal_gate,
)
from microcosm.build.us_runtime.spm_independence_role import (
    US_SPM_INDEPENDENCE_ROLE_PROVENANCE_KEY,
)
from microcosm.build.us_runtime.ssi_take_up import (
    US_SSI_TAKE_UP_ANCHOR,
    us_ssi_take_up_reporter_source_ids,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    spine_assembly_manifest,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.weeks_unemployed import us_weeks_unemployed_signal_gate
from microcosm.build.us_runtime.wic_claim import (
    US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS,
    us_wic_claim_signal_gate,
)
from microcosm.build.us_runtime.workers_compensation import (
    US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS,
    us_workers_compensation_signal_gate,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_tail_clone_fixtures import (
    with_capital_gains_tail_copies,
)

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
    origin's household table; :attr:`keyed_cps_household_weights` and
    :attr:`keyed_acs_household_weights` key them as the contract takes them.
    ``clone_multiplier`` is the CPS clone id shift, ``acs_id_offsets``
    staging's per-entity ACS offsets, ``tail_donor_ids`` the applied tail
    donors, and ``cps_keys``/``acs_keys`` the selection keys of the native
    CPS households and the ACS households.
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

    @property
    def keyed_cps_household_weights(self) -> pd.Series:
        """The CPS household weights keyed by ``household_id``."""

        return cps_household_weight_series(
            self.cps_tables["household"], self.cps_household_weights
        )

    @property
    def keyed_acs_household_weights(self) -> pd.Series:
        """The ACS household weights keyed by ``acs_unit_key(SERIALNO)``."""

        return acs_household_weight_series(
            self.acs_tables["household"], self.acs_household_weights
        )


def cps_household_weight_series(household: pd.DataFrame, weights: object) -> pd.Series:
    """CPS household weights keyed as the contract takes them, by household id.

    ``weights`` align with ``household``'s rows.
    """

    return pd.Series(
        np.asarray(weights, dtype=np.float64),
        index=pd.Index(household["household_id"].to_numpy(), name="household_id"),
    )


def acs_household_weight_series(household: pd.DataFrame, weights: object) -> pd.Series:
    """ACS household weights keyed as the contract takes them.

    The key is ``acs_unit_key(SERIALNO)``, the key PR 1's selection keeps its
    ACS starting weights in order of. ``weights`` align with ``household``'s
    rows.
    """

    return pd.Series(
        np.asarray(weights, dtype=np.float64),
        index=pd.Index(
            [acs_unit_key(serial) for serial in household["SERIALNO"]],
            dtype=object,
            name="acs_unit_key",
        ),
    )


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
        cps_household_weights=population.keyed_cps_household_weights,
        acs_household_weights=population.keyed_acs_household_weights,
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


def selection_manifest(population: GeneratedPopulation) -> SelectionManifest:
    """PR 1's selection manifest for selecting every generated household.

    The key digests and counts are the generated keys'; the other fields
    are coherent placeholders (the contract reads only the keys' digests and
    counts).
    """

    by_year: dict[int, int] = {}
    for key in population.cps_keys:
        year = int(key.split("|", 1)[0])
        by_year[year] = by_year.get(year, 0) + 1
    group_quarters = int(
        (population.acs_tables["household"]["TYPEHUGQ"] != 1).sum()
        if len(population.acs_tables["household"])
        else 0
    )
    return SelectionManifest(
        rule_version=RULE_VERSION,
        salt="d2283-generated-population",
        replicate=0,
        budget=len(population.cps_keys) + len(population.acs_keys),
        include_group_quarters=True,
        drop_spm_zero_adult=True,
        acs_households=len(population.acs_keys),
        acs_eligible=len(population.acs_keys),
        acs_spm_zero_adult=0,
        acs_group_quarters=group_quarters,
        cps_income_years=tuple(sorted(by_year)),
        n_cps=len(population.cps_keys),
        n_acs=len(population.acs_keys),
        n_cps_by_income_year=tuple(sorted(by_year.items())),
        threshold_draw=None,
        selected_acs_keys_sha256=keys_sha256(population.acs_keys),
        cps_keys_sha256=keys_sha256(population.cps_keys),
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


def _gate_record(person: pd.DataFrame, spm_units: int) -> dict[str, object]:
    """The role gate's counts and digest over ``person``, by its rule.

    The digest restates ``spm_independence_role._role_binding_sha256``: the
    sha256 of the uint64 row hashes of the ordered person id, SPM unit
    membership, age and role columns.
    """

    columns = ["person_id", "person_spm_unit_id", "age", SPM_ROLE]
    hashes = pd.util.hash_pandas_object(person[columns], index=False)
    return {
        "persons_joined": len(person),
        "table_rows": len(person),
        "native_spm_units": int(spm_units),
        "true_role_persons": int(person[SPM_ROLE].sum()),
        "person_role_binding_sha256": hashlib.sha256(
            hashes.to_numpy(dtype="<u8").tobytes()
        ).hexdigest(),
    }


def spm_role_gate_record(frame: Frame) -> dict[str, object]:
    """A record in the role gate's keys, computed by its rule over ``frame``."""

    return _gate_record(frame.table("person"), frame.n("spm_unit"))


def _cps_rows(table: pd.DataFrame, entity: str) -> np.ndarray:
    return table[support_channel_column(entity)].ne(ACS_SUPPORT_CHANNEL).to_numpy()


def cps_role_record(frame: Frame) -> dict[str, object]:
    """The CPS half's role record PR 4 would derive, over the frame's CPS rows.

    The gate's keys over the CPS person rows in the frame's order and dtypes
    (by the gate's rule, restated), plus the provenance keys
    ``with_us_spm_independence_role`` writes that the contract reads as
    evidence only.
    """

    person = frame.table("person")
    spm_unit = frame.table("spm_unit")
    return {
        **_gate_record(
            person.loc[_cps_rows(person, "person")],
            int(_cps_rows(spm_unit, "spm_unit").sum()),
        ),
        "primitive_column": SPM_ROLE,
        "source_join": [
            "source_year (income year)",
            "PERIDNUM (exact 22-digit string)",
        ],
        "unmatched_persons": 0,
    }


def staging_role_receipt(frame: Frame, *, staged_beyond: int = 5) -> dict[str, object]:
    """Staging's ACS role receipt for a staging run that held ``frame``'s ACS rows.

    The keys and identity are ``acs_inputs.with_acs_spm_independence_role``'s;
    the counts cover ``staged_beyond`` more persons (and one more SPM unit and
    role holder per five) than the selection keeps, as a pre-selection
    receipt does.
    """

    from microcosm.build.us_runtime.acs_inputs import (
        ACS_SPM_ROLE_PARTITION,
        ACS_SPM_ROLE_RULE,
    )
    from microcosm.build.us_runtime.spm_role_source import (
        SPM_ADULT_RULE,
        SPM_ROLE_RULE,
    )

    person = frame.table("person")
    acs = ~_cps_rows(person, "person")
    role = person.loc[acs, SPM_ROLE].to_numpy(dtype=bool)
    age = person.loc[acs, "age"].to_numpy()
    minor = (age >= 15) & (age < 18)
    extra = staged_beyond // 5
    return {
        "column": SPM_ROLE,
        "entity": "person",
        "source_columns": ["RELSHIPP"],
        "rule": ACS_SPM_ROLE_RULE,
        "partition": ACS_SPM_ROLE_PARTITION,
        "asec_rule": SPM_ROLE_RULE,
        "adult_rule": SPM_ADULT_RULE,
        "provenance": "acs_2024_1yr_native",
        "persons": int(acs.sum()) + staged_beyond,
        "spm_units": int((~_cps_rows(frame.table("spm_unit"), "spm_unit")).sum())
        + extra,
        "role_true_persons": int(role.sum()) + extra,
        "role_branches": {
            "reference_person": int(role.sum()),
            "spouse_of_reference_person": 0,
            "group_quarters_sole_person": 2,
        },
        "persons_aged_15_to_17": int(minor.sum()),
        "independent_minor_persons": int((role & minor).sum()),
        "housing_units_classified_only_by_role": 1,
        "group_quarters_units_without_classified_adult": 0,
        "weighted_role_share": 0.6,
        "weighted_minor_role_share": 0.02,
    }


def fill_manifest(fills: Iterable[Mapping[str, object]] = (), *, period: int = 2024):
    """A reviewed default fill's manifest (``fill_reviewed_nulls``'s shape)."""

    fills = [dict(fill) for fill in fills]
    return {
        "period": period,
        "register_entries": len(fills),
        "columns_filled": len(fills),
        "total_values_filled": sum(int(fill["filled_rows"]) for fill in fills),
        "count_mismatch_warnings": [],
        "unregistered_violations": [],
        "fills": fills,
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


# ------------------------------------- X5: a Route A frame for the readers

#: The provenance-reading gates d2283's whole-frame differential (X5) runs:
#: six of the seven assembled-form readers F-19 lists (alimony, Medicare
#: take-up, both retirement families, WIC, workers' compensation; the
#: seventh, ``acs_local_hours``, has no release gate), the weeks-unemployed
#: gate (in F-19's first count of nine; PR 3 owns its UC-rule scope) and
#: the SSI take-up reporter lineage (a reviewed caller of
#: ``support_gate_source_channel_series``; the SSI gate itself grades
#: diagnostics built from an engine-computed ``uncapped_ssi``).
READER_GATES: Mapping[str, Callable[[Frame], object]] = MappingProxyType(
    {
        "alimony": us_alimony_signal_gate,
        "medicare_take_up": us_medicare_take_up_signal_gate,
        "retirement_contributions": us_retirement_contributions_signal_gate,
        "retirement_distributions": us_retirement_distributions_signal_gate,
        "ssi_take_up_reporters": us_ssi_take_up_reporter_source_ids,
        "weeks_unemployed": us_weeks_unemployed_signal_gate,
        "wic_claim": us_wic_claim_signal_gate,
        "workers_compensation": us_workers_compensation_signal_gate,
    }
)

#: Each reader's raw ASEC source columns, all on the reader frame.
READER_SOURCE_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "alimony": ("OI_VAL", "OI_OFF"),
        "medicare_take_up": US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS,
        "retirement_contributions": US_RETIREMENT_CONTRIBUTION_REQUIRED_SOURCE_COLUMNS,
        "retirement_distributions": US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS,
        "ssi_take_up_reporters": (US_SSI_TAKE_UP_ANCHOR,),
        "weeks_unemployed": ("LKWEEKS", "unemployment_compensation"),
        "wic_claim": US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS,
        "workers_compensation": US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS,
    }
)

#: The WIC gate's source-local identity on a frame without spine ids.
WIC_SOURCE_TRIPLE = ("source_year", "source_household_id", "source_person_id")

#: The two clone-only defects d2283 lead call L7 adds to X5. Each changes
#: one cell on one clone row so the row breaks its reader's Route A
#: reconciliation, which the assembled readers leave out: Medicare's
#: assembled source mask keeps stored-``asec`` rows, and the assembled weeks
#: UC rule keys on the stored channel and clone index. Each maps to its
#: reader and the one failure Route A's gate then reports.
READER_CLONE_DEFECTS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "medicare_clone_mcare_mismatch": (
            "medicare_take_up",
            "takes_up_medicare_if_eligible: 1 MCARE reconciliation mismatch(es)",
        ),
        "weeks_clone_uc_mismatch": (
            "weeks_unemployed",
            "weeks_unemployed has 1 row(s) positive outside the "
            "unemployment-compensation constraint.",
        ),
    }
)

#: Each defect's cell: the native person (group, index) whose one clone row
#: changes, the column and the new value. Elder 0 is enrolled (``MCARE``
#: 1, take-up True) and its clone now reports code 2; worker 5 receives
#: unemployment compensation, so its clone carries 10 weeks, and the clone's
#: compensation is now 0. Neither sits in a tail-source household, so each
#: has exactly one clone row.
_CLONE_DEFECT_CELLS: Mapping[str, tuple[tuple[str, int], str, object]] = (
    MappingProxyType(
        {
            "medicare_clone_mcare_mismatch": (("elder", 0), "MCARE", 2),
            "weeks_clone_uc_mismatch": (
                ("worker", 5),
                "unemployment_compensation",
                0.0,
            ),
        }
    )
)

_READER_WEIGHT = 1_000.0
#: One single's household weighs a tenth of the rest, so the Keogh
#: distribution it carries sits inside the gate's 0.5% share bound.
_READER_LIGHT_WEIGHT = 100.0
#: A distribution slot: (code column, amount column).
_DST_SLOTS = tuple(
    (f"DST_SC{suffix}", f"DST_VAL{suffix}") for suffix in ("1", "2", "1_YNG", "2_YNG")
)
_DST_OUTPUT_BY_CODE = dict(enumerate(US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS, 1))


@dataclass(frozen=True)
class ReaderPopulation:
    """The X5 reader population in both provenance forms.

    ``route_a`` is the Route A (unassembled) frame; ``assembled`` is
    :func:`~microcosm.build.us_runtime.one_population_contract.to_assembled_provenance`
    of its tables with no ACS row. ``reporters`` are the native persons with
    a direct ``SSI_VAL`` report (as ``person_source_id`` strings) and
    ``clone_only_ssi`` the person whose clone rows alone carry one.
    """

    route_a: Frame
    assembled: Frame
    reporters: frozenset[str]
    clone_only_ssi: str


def _reader_households() -> list[tuple[float, list[dict[str, object]]]]:
    """200 native persons in 120 households; ``group`` names each role."""

    households: list[tuple[float, list[dict[str, object]]]] = []

    def person(group: str, age: int, female: bool, **extra: object) -> dict:
        return {"group": group, "age": age, "is_female": female, **extra}

    # WIC: postpartum mothers with infants, pregnant women, young children.
    # Claims by category: postpartum 2 of 4, infant 3 of 4, pregnant 1 of 2,
    # child 2 of 4, so every category share sits inside its band.
    for index in range(4):
        households.append(
            (
                _READER_WEIGHT,
                [
                    person("mother", 24 + 2 * index, True, children=1, wic=index < 2),
                    person("infant", 0, index % 2 == 0, wic=index < 3),
                ],
            )
        )
    for index in range(2):
        households.append(
            (
                _READER_WEIGHT,
                [
                    person(
                        "pregnant", 25 + 5 * index, True, pregnant=True, wic=index == 0
                    ),
                    person("partner", 27 + 5 * index, False),
                ],
            )
        )
    for index in range(4):
        households.append(
            (
                _READER_WEIGHT,
                [
                    person("parent", 30 + 2 * index, index % 2 == 0, children=1),
                    person("child", 1 + index, index % 2 == 1, wic=index < 2),
                ],
            )
        )
    # Medicare: 40 enrolled persons, a fifth of the person weight.
    for index in range(20):
        households.append(
            (
                _READER_WEIGHT,
                [
                    person("elder", 66 + index % 14, False),
                    person("elder", 65 + index % 14, True),
                ],
            )
        )
    for index in range(50):
        households.append(
            (
                _READER_WEIGHT,
                [
                    person("worker", 25 + index % 38, False),
                    person("worker", 24 + index % 38, True),
                ],
            )
        )
    for index in range(39):
        households.append(
            (_READER_WEIGHT, [person("single", 22 + index % 40, index % 2 == 0)])
        )
    households.append((_READER_LIGHT_WEIGHT, [person("light", 60, False)]))
    return households


def _reader_native_frame() -> tuple[Frame, dict[str, np.ndarray]]:
    """The native ASEC rows, each producer's direct-source output written.

    Returns the frame and, by group, the person positions in its person
    table (for choosing who carries a rare measure).
    """

    rows: list[dict[str, object]] = []
    household_ids: list[int] = []
    weights: list[float] = []
    marital_unit = 0
    for household_index, (weight, members) in enumerate(_reader_households()):
        household_id = household_index + 1
        household_ids.append(household_id)
        weights.append(weight)
        couple = len(members) == 2 and all(member["age"] >= 18 for member in members)
        for line, member in enumerate(members, 1):
            if line == 1 or not couple:
                marital_unit += 1
            rows.append(
                {
                    **member,
                    "person_household_id": household_id,
                    "person_marital_unit_id": marital_unit,
                    "line": line,
                }
            )
    table = pd.DataFrame(rows)
    count = len(table)
    group = table["group"].to_numpy(dtype=object)
    positions = {
        name: np.flatnonzero(group == name) for name in sorted(set(group.tolist()))
    }
    age = table["age"].to_numpy(dtype=np.int64)
    adult = age >= 15

    def zeros() -> np.ndarray:
        return np.zeros(count, dtype=np.float64)

    workers, elders, singles = (
        positions["worker"],
        positions["elder"],
        positions["single"],
    )
    # Weeks unemployed: five LKWEEKS reports; unemployment compensation on
    # two of them and on one person with none (the clones' UC scope).
    lkweeks = np.where(adult, 0, -1).astype(np.int64)
    lkweeks[workers[:5]] = 20
    unemployment = zeros()
    unemployment[workers[[0, 1, 5]]] = 4_000.0
    # Workers' compensation: one direct report.
    wc = zeros()
    wc[workers[6]] = 12_000.0
    # Other income: alimony (code 20), strike benefits (12), another code.
    oi_off = np.zeros(count, dtype=np.int64)
    oi_val = zeros()
    for position, code, amount in (
        (singles[0], 20, 6_000.0),
        (singles[1], 12, 800.0),
        (singles[2], 7, 400.0),
    ):
        oi_off[position], oi_val[position] = code, amount
    alimony_expense = zeros()
    alimony_expense[singles[3]] = 3_000.0
    # Retirement contributions: one leaf per contributor.
    wages = np.where(np.isin(np.arange(count), workers), 40_000.0, 0.0)
    self_employment = zeros()
    self_employment[workers[12]] = 20_000.0
    retcb = zeros()
    contributions = {
        column: zeros() for column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
    }
    for offset, column in enumerate(US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS):
        amount = 5_000.0 - 1_000.0 * offset
        retcb[workers[8 + offset]] = amount
        contributions[column][workers[8 + offset]] = amount
    # Retirement distributions: every account code, a two-slot sum, an
    # "other" code (7) with no leaf, and Keogh on the light household.
    dst_codes = {code: np.zeros(count, dtype=np.int64) for code, _ in _DST_SLOTS}
    dst_amounts = {amount: zeros() for _, amount in _DST_SLOTS}
    for position, slot, code, amount in (
        (elders[0], 0, 1, 8_000.0),
        (elders[1], 0, 2, 5_000.0),
        (elders[2], 0, 3, 4_000.0),
        (elders[2], 1, 4, 3_000.0),
        (positions["light"][0], 2, 5, 2_000.0),
        (elders[3], 3, 6, 1_500.0),
        (elders[4], 0, 7, 900.0),
        (elders[5], 0, 4, 2_500.0),
        (elders[5], 1, 4, 500.0),
    ):
        code_column, amount_column = _DST_SLOTS[slot]
        dst_codes[code_column][position] = code
        dst_amounts[amount_column][position] = amount
    distributions = {
        column: zeros() for column in US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
    }
    for code_column, amount_column in _DST_SLOTS:
        for code, column in _DST_OUTPUT_BY_CODE.items():
            hit = dst_codes[code_column] == code
            distributions[column][hit] += dst_amounts[amount_column][hit]
    # SSI: two direct reports.
    ssi = zeros()
    ssi[elders[6]] = 9_000.0
    ssi[singles[5]] = 4_000.0
    # Medicare: every elder enrolled; other adults not; children NIU.
    mcare = np.where(np.isin(np.arange(count), elders), 1, np.where(adult, 2, 0))

    household_of = table["person_household_id"].to_numpy(dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype=np.int64),
            "person_household_id": household_of,
            "person_tax_unit_id": household_of,
            "person_spm_unit_id": household_of,
            "person_family_id": household_of,
            "person_marital_unit_id": table["person_marital_unit_id"].astype("int64"),
            "age": age,
            "is_female": table["is_female"].astype(bool),
            "is_pregnant": table.get("pregnant", pd.Series(False, index=table.index))
            .fillna(False)
            .astype(bool),
            "own_children_in_household": table.get(
                "children", pd.Series(0, index=table.index)
            )
            .fillna(0)
            .astype("int64"),
            "source_year": np.full(count, 2024, dtype=np.int64),
            "source_household_id": household_of,
            "source_person_id": table["line"].astype(str).astype(object),
            "LKWEEKS": lkweeks,
            "unemployment_compensation": unemployment,
            "weeks_unemployed": np.where(lkweeks == -1, 0, lkweeks).astype(np.float64),
            "MCARE": mcare.astype(np.int64),
            "takes_up_medicare_if_eligible": mcare == 1,
            "WC_VAL": wc,
            "workers_compensation": wc.copy(),
            "OI_OFF": oi_off,
            "OI_VAL": oi_val,
            "alimony_income": np.where(oi_off == 20, oi_val, 0.0),
            "strike_benefits": np.where(oi_off == 12, oi_val, 0.0),
            "miscellaneous_income": np.where(np.isin(oi_off, (12, 20)), 0.0, oi_val),
            "alimony_expense": alimony_expense,
            "WSAL_VAL": wages,
            "SEMP_VAL": self_employment,
            "RETCB_VAL": retcb,
            **contributions,
            **dst_codes,
            **dst_amounts,
            **distributions,
            US_SSI_TAKE_UP_ANCHOR: ssi,
            "takes_up_wic_if_eligible": table.get(
                "wic", pd.Series(False, index=table.index)
            )
            .fillna(False)
            .astype(bool),
        }
    )
    ids = np.asarray(household_ids, dtype=np.int64)
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(1, marital_unit + 1, dtype=np.int64)}
        ),
    }
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray(weights), WeightKind.DESIGN)},
    )
    return frame, positions


def _on_clones(person: pd.DataFrame, column: str, values: Mapping[int, float]) -> None:
    """Write ``values`` (by native person id) on every clone row, in place."""

    clone = person["person_support_clone_index"].to_numpy() > 0
    source = person["person_source_id"].to_numpy(dtype=np.int64)
    for native_id, value in values.items():
        person.loc[clone & (source == native_id), column] = value


def reader_population(*, clone_defect: str | None = None) -> ReaderPopulation:
    """An ASEC-only population for the whole-frame differential (d2283 X5).

    The native rows (:func:`_reader_native_frame`) carry every reader's raw
    ASEC source columns and each producer's direct-source output. The
    production operators then make Route A's labels, as X4's fixture does:
    ``clone_us_frame_for_puf_support`` adds a PUF tax-detail copy (clone 1,
    ``puf_tax_detail``) of every household on all six entities, writing
    ``<entity>_source_id`` and halving the weights, and
    ``with_capital_gains_tail_copies`` adds tail copies (clone 2) of three
    households: a WIC infant's, a UC recipient's and a contributor's.

    Clone rows then take stand-ins for the PUF imputations where Route A's
    gates do not reconcile clones against the source, so a reader whose
    assembled mask wrongly took clone rows would show it: weeks positive
    only where unemployment compensation is (the clones' UC rule), a
    workers' compensation amount unlike ``WC_VAL`` and one with none, an
    alimony amount unlike the ``OI_VAL`` split and one with none, every
    contribution and distribution leaf scaled, and an ``SSI_VAL`` report on
    a person whose native row has none. Medicare and WIC clones copy their
    natives: Route A's Medicare gate reconciles every row against ``MCARE``,
    and its WIC gate refuses a clone whose claim differs from its source
    person's.

    The WIC gate keys clones by the source triple (``source_year``,
    ``source_household_id``, ``source_person_id``) on Route A and by
    ``person_source_id`` once assembled. The triple is unique per native
    person and both columns are copied to every clone, so the two keys
    partition the rows alike.

    Args:
        clone_defect: A key of :data:`READER_CLONE_DEFECTS`, to change that
            defect's one clone cell (d2283 lead call L7), or ``None``.

    Raises:
        AssertionError: If any reader's gate fails on the Route A frame (or,
            with ``clone_defect``, if the defect's reader does not fail with
            exactly its one failure while every other reader passes), so a
            differential never compares two gates failing for another
            reason.
    """

    native, positions = _reader_native_frame()
    cloned = clone_us_frame_for_puf_support(native)
    household = cloned.table("household")
    clone_one = household["household_support_clone_index"].to_numpy() == 1
    sources = household.loc[clone_one, "household_source_id"].to_numpy(dtype=np.int64)
    native_person = native.table("person")

    def household_of(position: int) -> int:
        return int(native_person["person_household_id"].iloc[position])

    tail_sources = [
        household_of(positions["infant"][0]),
        household_of(positions["worker"][0]),
        household_of(positions["worker"][8]),
    ]
    if not set(tail_sources) <= set(sources.tolist()):
        raise AssertionError("A tail source household has no PUF tax-detail copy.")
    route_a = with_capital_gains_tail_copies(cloned, tail_sources)

    def native_id(group: str, index: int) -> int:
        return int(native_person["person_id"].iloc[positions[group][index]])

    tables = {entity: route_a.table(entity).copy() for entity in route_a.entities}
    person = tables["person"]
    clone = person["person_support_clone_index"].to_numpy() > 0
    uc = person["unemployment_compensation"].to_numpy(dtype=np.float64)
    person.loc[clone, "weeks_unemployed"] = np.where(uc[clone] > 0.0, 10.0, 0.0)
    _on_clones(
        person,
        "workers_compensation",
        {native_id("worker", 6): 15_000.0, native_id("worker", 7): 3_000.0},
    )
    _on_clones(
        person,
        "alimony_income",
        {native_id("single", 0): 7_500.0, native_id("single", 4): 2_000.0},
    )
    for column in (
        *US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS,
        *US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
    ):
        factor = 1.2 if column in US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS else 0.9
        person.loc[clone, column] = person.loc[clone, column] * factor
    clone_only_ssi = native_id("single", 6)
    _on_clones(person, US_SSI_TAKE_UP_ANCHOR, {clone_only_ssi: 2_000.0})
    if clone_defect is not None:
        (group, index), column, value = _CLONE_DEFECT_CELLS[clone_defect]
        source = person["person_source_id"].to_numpy(dtype=np.int64)
        rows = np.flatnonzero(clone & (source == native_id(group, index)))
        if len(rows) != 1:
            raise AssertionError(
                f"The {clone_defect} defect needs one clone row; found {len(rows)}."
            )
        person.loc[person.index[rows[0]], column] = value
    route_a = Frame(
        tables,
        route_a.schema,
        {entity: route_a.weights_for(entity) for entity in route_a.weighted_entities},
        route_a.strata,
        mass_log=route_a.mass_log,
        metadata=route_a.metadata,
    )
    reporters = frozenset(
        str(native_id(group, index)) for group, index in (("elder", 6), ("single", 5))
    )
    _check_reader_gates(route_a, reporters=reporters, clone_defect=clone_defect)

    empty_acs = {
        entity: table.iloc[:0]
        for entity, table in generate_population(0).acs_tables.items()
    }
    assembled = to_assembled_provenance(
        tables,
        empty_acs,
        cps_household_weights=cps_household_weight_series(
            route_a.table("household"), route_a.weights_for("household").values
        ),
        acs_household_weights={},
        staging_lineage=frozenset(),
    )
    return ReaderPopulation(
        route_a=route_a,
        assembled=assembled,
        reporters=reporters,
        clone_only_ssi=str(clone_only_ssi),
    )


def _check_reader_gates(
    frame: Frame, *, reporters: frozenset[str], clone_defect: str | None = None
) -> None:
    expected: dict[str, list[str]] = {}
    if clone_defect is not None:
        defect_reader, failure = READER_CLONE_DEFECTS[clone_defect]
        expected[defect_reader] = [failure]
    failures: dict[str, object] = {}
    for name, reader in READER_GATES.items():
        result = reader(frame)
        if isinstance(result, frozenset):
            if result != reporters:
                failures[name] = sorted(result)
        elif result.passed == (name in expected) or list(result.failures) != (
            expected.get(name, [])
        ):
            failures[name] = list(result.failures)
    if failures:
        raise AssertionError(
            f"Route A reader gates fail on the reader frame: {failures} "
            f"(expected failures: {expected})."
        )
