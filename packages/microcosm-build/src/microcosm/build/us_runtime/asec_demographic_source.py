"""Explicit ASEC demographic source contract: sex, its allocation flag, and the
household reference-person code, restored from the pinned original CSV cohorts.

Scope fence. This module declares and implements the contract and its reader.
It is deliberately **not** wired into
:mod:`.asec_prepared_source`: the root adjudication of 2026-09-06 authorised
preparing this source code and tiny fixtures independently and authorised no
additional full-source read and no new binding. Nothing here is executed
against a genuine Census member by this lane.

What the contract refuses to assume, stated once so a later reader cannot
mistake an omission for an oversight:

* A declared value range is not measured proof that every delivered token is
  inside it. ``A_SEX`` is validated against its own printed codes, and a token
  that is not a printed code leaves that person's sex **unbound**. Nothing here
  maps "not 2" onto male.
* ``AXSEX`` prints two codes inside a wider declared range. A token in the
  range but outside the printed codes is an unresolved allocation provenance,
  so it leaves that person's sex unbound rather than being smoothed into
  "reported".
* ``A_EXPRRP`` codes 1 and 2 are the printed, self-labelled household
  reference person. They are the only admitted head evidence here.
* ``P_SEQ`` labels no code in its own dictionary entry. It is carried and
  diagnosed against the reference code, and it is never a fallback: a
  household whose reference code is absent, duplicated or indeterminate stays
  unbound even when exactly one of its members has ``P_SEQ == 1``.
* ``A_FAMREL`` is a **family** relationship and is not read here. ``A_LINENO``
  is a Basic-CPS roster line number read only as a join crosscheck. Neither
  substitutes for the household reference code.

Every printed fact below was read this session from the three pinned public
dictionaries named in :data:`ASEC_DICTIONARY_AUTHORITY`, at the PDF pages
recorded there, and the four entries are textually identical across the three
years.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.frame import Frame

from . import asec_current_money_source as legacy
from . import asec_person_income_source as restoration
from .asec_student_controls import _snapshot
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

MAGIC = b"MCASDEMO\x01"
FILENAME = "asec_demographic_source.bin"
ARTIFACT_KIND = "microcosm.asec_demographic_source.v1"
_HEADER_MAX = 262_144
_MAX_PERSONS = 600_000
_TOKEN = object()
_COMPOSE_TOKEN = object()


class DemographicSourceRefusalError(ValueError):
    """Value-free refusal; never carries a row, an identifier or an amount."""

    def __init__(self, reason: str, field: str = "contract"):
        self.reason = reason
        self.field = field
        super().__init__(f"{field}: {reason}")


def _require(condition: bool, reason: str, field: str = "contract") -> None:
    if not condition:
        raise DemographicSourceRefusalError(reason, field)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


_ASEC_DICTIONARY_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/cps/datasets/"
    "{survey_year}/march/asec{survey_year}_ddl_pub_full.pdf"
)
#: Survey year -> the pinned public-use dictionary that prints these entries.
#: The three digests are the same documents the reviewed household
#: reported-income contract already pins; no new pin is introduced here.
ASEC_DICTIONARY_AUTHORITY: Mapping[int, Mapping[str, object]] = MappingProxyType(
    {
        2023: MappingProxyType(
            {
                "url": _ASEC_DICTIONARY_URL_TEMPLATE.format(survey_year=2023),
                "sha256": (
                    "66bd6e3fe516233ab63b75c60573451222b3b3d3235d61cfed96f64608de2117"
                ),
            }
        ),
        2024: MappingProxyType(
            {
                "url": _ASEC_DICTIONARY_URL_TEMPLATE.format(survey_year=2024),
                "sha256": (
                    "761c67ea53f5c3264329b3e9ddbdd802826ba4a859122b8a8a2f29ab85b3a840"
                ),
            }
        ),
        2025: MappingProxyType(
            {
                "url": _ASEC_DICTIONARY_URL_TEMPLATE.format(survey_year=2025),
                "sha256": (
                    "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f"
                ),
            }
        ),
    }
)


@dataclass(frozen=True, eq=False)
class AsecDemographicField:
    """One printed dictionary entry, restated exactly, with its own role.

    ``named_codes`` holds only codes the dictionary actually prints. Codes
    inside ``printed_range`` that it does not print are exposed as
    :attr:`unnamed_in_range_codes` and are never treated as a named value.
    """

    name: str
    concept: str
    printed_length: int
    printed_range: tuple[int, int]
    printed_position: int
    named_codes: Mapping[int, str]
    universe_as_printed: str
    role: str
    quoted_locator: str
    pdf_page_1based_by_survey_year: Mapping[int, int]

    @property
    def declared_domain(self) -> tuple[int, ...]:
        low, high = self.printed_range
        return tuple(range(low, high + 1))

    @property
    def unnamed_in_range_codes(self) -> tuple[int, ...]:
        return tuple(
            code for code in self.declared_domain if code not in self.named_codes
        )

    @property
    def token_pattern(self) -> str:
        return rf"[0-9]{{1,{self.printed_length}}}"

    def document(self) -> dict:
        return {
            "name": self.name,
            "concept": self.concept,
            "printed_length": self.printed_length,
            "printed_range": list(self.printed_range),
            "printed_position": self.printed_position,
            "named_codes": {
                str(code): self.named_codes[code] for code in sorted(self.named_codes)
            },
            "unnamed_in_range_codes": list(self.unnamed_in_range_codes),
            "universe_as_printed": self.universe_as_printed,
            "role": self.role,
            "quoted_locator": self.quoted_locator,
            "token_pattern": self.token_pattern,
            "authority": [
                {
                    "survey_year": year,
                    "url": ASEC_DICTIONARY_AUTHORITY[year]["url"],
                    "sha256": ASEC_DICTIONARY_AUTHORITY[year]["sha256"],
                    "pdf_page_1based": page,
                }
                for year, page in sorted(self.pdf_page_1based_by_survey_year.items())
            ],
        }


A_SEX = AsecDemographicField(
    name="A_SEX",
    concept="Sex",
    printed_length=1,
    printed_range=(1, 2),
    printed_position=92,
    named_codes=MappingProxyType({1: "Male", 2: "Female"}),
    universe_as_printed="All Persons",
    role="restored_observation",
    quoted_locator="A_SEX / Sex / 1 (1:2) 92 / Values: 1 = Male 2 = Female / "
    "Universe: All Persons",
    pdf_page_1based_by_survey_year=MappingProxyType({2023: 22, 2024: 23, 2025: 24}),
)
AXSEX = AsecDemographicField(
    name="AXSEX",
    concept="Allocation flag for A_SEX",
    printed_length=1,
    printed_range=(0, 4),
    printed_position=161,
    named_codes=MappingProxyType({0: "No change", 4: "Allocated"}),
    universe_as_printed="All Persons",
    role="restored_observation",
    quoted_locator="AXSEX / Allocation flag for A_SEX / 1 (0:4) 161 / "
    "Values: 0 = No change 4 = Allocated / Universe: All Persons",
    pdf_page_1based_by_survey_year=MappingProxyType({2023: 27, 2024: 28, 2025: 29}),
)
A_EXPRRP = AsecDemographicField(
    name="A_EXPRRP",
    concept="Expanded relationship code",
    printed_length=2,
    printed_range=(1, 14),
    printed_position=82,
    named_codes=MappingProxyType(
        {
            1: "Reference person with relatives",
            2: "Reference person without relatives",
            3: "Husband",
            4: "Wife",
            5: "Own child",
            7: "Grandchild",
            8: "Parent",
            9: "Brother/sister",
            10: "Other relative",
            11: "Foster child",
            12: "Nonrelative with relatives",
            13: "Partner/roommate",
            14: "Nonrelative without relatives",
        }
    ),
    universe_as_printed="All Persons",
    role="restored_observation",
    quoted_locator="A_EXPRRP / Expanded relationship code / 2 (1:14) 82 / "
    "Values: 1 = Reference person with relatives 2 = Reference person "
    "without relatives 3 = Husband 4 = Wife 5 = Own child 7 = Grandchild "
    "8 = Parent 9 = Brother/sister 10 = Other relative 11 = Foster child "
    "12 = Nonrelative with relatives 13 = Partner/roommate 14 = Nonrelative "
    "without relatives / Universe: All Persons",
    pdf_page_1based_by_survey_year=MappingProxyType({2023: 21, 2024: 22, 2025: 23}),
)
P_SEQ = AsecDemographicField(
    name="P_SEQ",
    concept="Sequence number of person in hhld",
    printed_length=2,
    printed_range=(0, 16),
    printed_position=10,
    named_codes=MappingProxyType({}),
    universe_as_printed="All Persons",
    role="diagnostic_crosscheck",
    quoted_locator="P_SEQ / Sequence number of person in hhld / 2 (00:16) 10 / "
    "Values: 0-16 / Universe: All Persons",
    pdf_page_1based_by_survey_year=MappingProxyType({2023: 20, 2024: 21, 2025: 22}),
)

#: The four printed entries this contract covers, in output order.
ASEC_DEMOGRAPHIC_FIELDS: tuple[AsecDemographicField, ...] = (
    A_SEX,
    AXSEX,
    A_EXPRRP,
    P_SEQ,
)
OBSERVATIONS: tuple[str, ...] = tuple(field.name for field in ASEC_DEMOGRAPHIC_FIELDS)
ALIASES: tuple[str, ...] = tuple(f"asec_{name}" for name in OBSERVATIONS)
#: The only codes admitted as household reference-person evidence. Both are
#: printed, self-labelled reference-person codes; the split is with/without
#: relatives and is irrelevant to headship itself.
HOUSEHOLD_REFERENCE_CODES: tuple[int, ...] = (1, 2)

#: Signals deliberately not adopted as semantic proof of headship, each with
#: the reason it is refused. Declared so the refusal survives a later reader.
NON_ADOPTED_HEAD_SIGNALS: tuple[Mapping[str, str], ...] = (
    MappingProxyType(
        {
            "signal": "P_SEQ == 1",
            "kind": "positional_sequence_within_household",
            "refusal": (
                "P_SEQ's own dictionary entry labels no code, including 1, as "
                "reference person or household head; the ordering statement "
                "that would support it is expressly limited to the ASCII file "
                "while this restoration reads the CSV member"
            ),
            "carried_as": "diagnostic_crosscheck",
        }
    ),
    MappingProxyType(
        {
            "signal": "A_FAMREL == 1",
            "kind": "family_relationship",
            "refusal": (
                "A_FAMREL is a family relationship scoped to primary-family "
                "membership; a subfamily reference person is not its code 1, "
                "so it is not a household role and is not read here"
            ),
            "carried_as": "not_read",
        }
    ),
    MappingProxyType(
        {
            "signal": "A_LINENO == 1",
            "kind": "basic_cps_roster_line_number",
            "refusal": (
                "A_LINENO is a separately assigned Basic-CPS roster line "
                "number with no source guarantee of row-for-row agreement "
                "with any ASEC sequence; it is read only as a join crosscheck"
            ),
            "carried_as": "join_crosscheck",
        }
    ),
)

#: Person-level derived production fields, with their declared code meanings.
SEX_BINDING_STATES: Mapping[int, str] = MappingProxyType(
    {0: "unbound", 1: "bound_male", 2: "bound_female"}
)
SEX_UNBOUND_REASONS: Mapping[int, str] = MappingProxyType(
    {
        0: "bound",
        1: "sex_code_outside_printed_codes",
        2: "allocation_flag_outside_printed_codes",
    }
)
SEX_ALLOCATION_STATES: Mapping[int, str] = MappingProxyType(
    {0: "undetermined", 1: "no_change", 2: "census_allocated"}
)
HOUSEHOLD_REFERENCE_STATES: Mapping[int, str] = MappingProxyType(
    {
        0: "unbound",
        1: "bound_household_reference_person",
        2: "bound_not_household_reference_person",
    }
)
HOUSEHOLD_REFERENCE_UNBOUND_REASONS: Mapping[int, str] = MappingProxyType(
    {
        0: "bound",
        1: "relationship_code_outside_printed_codes",
        2: "household_reference_absent",
        3: "household_reference_duplicated",
        4: "household_relationship_code_indeterminate",
    }
)
DERIVED: tuple[str, ...] = (
    "asec_sex_binding_state",
    "asec_sex_unbound_reason",
    "asec_sex_allocation_state",
    "asec_household_reference_state",
    "asec_household_reference_unbound_reason",
)

COORDINATES: tuple[str, ...] = (
    "person_id",
    "income_year",
    "source_household_id",
    "A_LINENO",
    "A_AGE",
)
#: Every column the artifact carries, in body order.
COLUMNS: tuple[str, ...] = COORDINATES + ALIASES + DERIVED
#: The exact readset: the only columns read from any Census member here.
ASEC_DEMOGRAPHIC_SOURCE_COLUMNS: tuple[str, ...] = (
    "PERIDNUM",
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    *OBSERVATIONS,
)
_COORDINATE_COLUMNS = ("PH_SEQ", "A_LINENO", "A_AGE")
_MEMBER_PINS = tuple(
    (year, p.member, p.zip_sha256, p.member_sha256, p.rows, p.member_size_bytes)
    for year, p in sorted(ASEC_EDUCATION_ASSISTANCE_ARCHIVES.items())
)
#: Income year -> survey year. The observation is the interview household one
#: year after the income year; this module harmonizes no period.
COHORTS: Mapping[int, int] = MappingProxyType(
    {year: year + 1 for year, *_rest in _MEMBER_PINS}
)
OBSERVED_PERIOD_KIND = "interview_household_one_year_after_income_year"


def _code_document(codes: Mapping[int, str]) -> dict:
    """Codes are documented with string keys so a JSON replay compares equal."""
    return {str(code): codes[code] for code in sorted(codes)}


def contract_document() -> dict:
    """The full declared contract, as it is recorded in every receipt."""
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "fields": [field.document() for field in ASEC_DEMOGRAPHIC_FIELDS],
        "readset": list(ASEC_DEMOGRAPHIC_SOURCE_COLUMNS),
        "missing_token_policy": "refuse_source_member_without_filling_or_coercion",
        "incumbent_relationship_policy": (
            "A_EXPRRP presence does not attest source observation; compare and "
            "diagnose discrepancies without replacing or trusting the incumbent recode"
        ),
        "production_fields": {
            "restored_observations": [
                {
                    "column": alias,
                    "source_column": name,
                    "role": field.role,
                    "dtype": "int64",
                }
                for alias, name, field in zip(
                    ALIASES, OBSERVATIONS, ASEC_DEMOGRAPHIC_FIELDS, strict=True
                )
            ],
            "derived": [
                {
                    "column": "asec_sex_binding_state",
                    "dtype": "int64",
                    "codes": _code_document(SEX_BINDING_STATES),
                },
                {
                    "column": "asec_sex_unbound_reason",
                    "dtype": "int64",
                    "codes": _code_document(SEX_UNBOUND_REASONS),
                },
                {
                    "column": "asec_sex_allocation_state",
                    "dtype": "int64",
                    "codes": _code_document(SEX_ALLOCATION_STATES),
                },
                {
                    "column": "asec_household_reference_state",
                    "dtype": "int64",
                    "codes": _code_document(HOUSEHOLD_REFERENCE_STATES),
                },
                {
                    "column": "asec_household_reference_unbound_reason",
                    "dtype": "int64",
                    "codes": _code_document(HOUSEHOLD_REFERENCE_UNBOUND_REASONS),
                },
            ],
        },
        "household_reference_codes": list(HOUSEHOLD_REFERENCE_CODES),
        "non_adopted_head_signals": [dict(item) for item in NON_ADOPTED_HEAD_SIGNALS],
        "household_and_family_roles_separate": True,
        "acs_diagnostic_reference_contract": {
            "source_column": "RELSHIPP",
            "declared_range": [20, 38],
            "housing_unit_reference_code": 20,
            "group_quarters_codes": {
                "37": "institutionalized",
                "38": "noninstitutionalized",
            },
            "authority": {
                "url": "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf",
                "sha256": "929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff",
                "pdf_page_1based": 43,
            },
            "production_acs_values_changed": False,
        },
        "family_relationship_field_read": False,
        "line_number_used_for_headship": False,
        "sex_binding_requires": [
            "A_SEX in printed codes",
            "AXSEX in printed codes",
        ],
        "household_reference_binding_requires": [
            "every A_EXPRRP in the household inside printed codes",
            "exactly one household member carrying code 1 or 2",
        ],
        "household_membership_verification": {
            "grouping_column": HOUSEHOLD_GROUPING_COLUMN,
            "published_household_key": list(HOUSEHOLD_MEMBERSHIP_KEY),
            "equivalence": "bidirectional_partition_bijection",
            "verified_before_classification": True,
            "merged_published_households": "refused",
            "split_published_households": "refused",
            "native_household_id_reuse_across_cohorts": "admitted_and_counted",
        },
        "other_than_two_is_male": False,
        "cohorts": {
            str(year): {
                "income_year": year,
                "survey_year": survey,
                "observed_period_kind": OBSERVED_PERIOD_KIND,
                "aged_from_a_date": False,
            }
            for year, survey in sorted(COHORTS.items())
        },
        "release_eligible": False,
        "wired_into_prepared_source": False,
    }


def _implementation() -> dict:
    package = resources.files(__package__)
    return {
        "schema": 1,
        "modules": {
            name: _sha(package.joinpath(name).read_bytes())
            for name in (
                "asec_demographic_source.py",
                "asec_person_income_source.py",
                "asec_student_controls.py",
            )
        },
        "legacy_verification": legacy._verification_identity(),
        "dependencies": {n: metadata.version(n) for n in ("numpy", "pandas")},
        "member_pins": _MEMBER_PINS,
        "contract": contract_document(),
    }


def read_demographic_member(path, *, rows: int, size: int) -> pd.DataFrame:
    """Parse one member's readset as exact tokens; never fill, never coerce.

    Lexical admission comes from the dictionary's own printed field length:
    a token must be digits only and no longer than the printed width, so a
    two-character ``A_SEX`` or a signed token is refused here rather than
    reaching the domain classification as a plausible code.
    """
    path = Path(path)
    _require(path.stat().st_size == size, "MEMBER_SIZE")
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    _require(
        len(header) == len(set(header))
        and set(ASEC_DEMOGRAPHIC_SOURCE_COLUMNS) <= set(header),
        "MEMBER_HEADER",
    )
    table = pd.read_csv(
        path,
        usecols=list(ASEC_DEMOGRAPHIC_SOURCE_COLUMNS),
        dtype="string",
        na_filter=False,
    )
    _require(len(table) == rows, "MEMBER_ROWS")
    _require(
        bool(table.PERIDNUM.str.fullmatch(r"[0-9]{22}").all()), "MEMBER_PERSON_KEY"
    )
    for name in _COORDINATE_COLUMNS:
        _require(
            bool(table[name].str.fullmatch(r"[0-9]+").all()),
            "MEMBER_INTEGER",
            name,
        )
        table[name] = table[name].astype("int64")
    for field in ASEC_DEMOGRAPHIC_FIELDS:
        _require(
            bool(table[field.name].str.fullmatch(field.token_pattern).all()),
            "MEMBER_TOKEN_WIDTH",
            field.name,
        )
        table[field.name] = table[field.name].astype("int64")
    _require(
        not table.PERIDNUM.duplicated().any()
        and not table.duplicated(["PH_SEQ", "A_LINENO"]).any()
        and bool((table.PH_SEQ > 0).all())
        and bool((table.A_LINENO > 0).all()),
        "MEMBER_COORDINATES",
    )
    return table


def _counts(codes: np.ndarray, meanings: Mapping[int, str]) -> dict:
    return {meanings[code]: int((codes == code).sum()) for code in sorted(meanings)}


#: The published household key. ``A_EXPRRP`` names a person's role inside the
#: household the Census published, and that household is identified by its
#: native ``PH_SEQ`` **within one income year**: an unrelated household in
#: another year legitimately carries the same native id.
HOUSEHOLD_MEMBERSHIP_KEY: tuple[str, ...] = ("income_year", "source_household_id")
#: The parent roster column headship is grouped by. It is a remapped id, not
#: the published one, so its partition is proven equal to the key above rather
#: than assumed to be.
HOUSEHOLD_GROUPING_COLUMN = "person_household_id"
#: Counters below that any nonzero value would have refused before an artifact
#: exists. Named so a later reader cannot read their zero as a measurement.
MEMBERSHIP_REFUSAL_COUNTERS: tuple[str, ...] = (
    "grouping_households_spanning_multiple_cohorts",
    "grouping_households_spanning_multiple_published_households",
    "published_households_spanning_multiple_grouping_households",
)
ZERO_COUNTER_SEMANTICS = (
    "derived from the delivered values, but any nonzero value refuses the run "
    "before an artifact exists, so a zero in a successful receipt attests that "
    "refusal and is not an independent measurement of the source"
)


def _dense_codes(*keys: np.ndarray) -> tuple[np.ndarray, int]:
    """Number the distinct key tuples 0..n-1 without combining them numerically.

    Key columns are compared component-wise after a lexicographic sort, so no
    two coordinates are ever multiplied or concatenated into one integer and
    no width assumption can silently collide two distinct households.
    """
    order = np.lexsort(tuple(reversed(keys)))
    changed = np.zeros(len(order), dtype=bool)
    changed[0] = True
    for values in keys:
        sorted_values = values[order]
        changed[1:] |= sorted_values[1:] != sorted_values[:-1]
    codes = np.empty(len(order), dtype=np.int64)
    codes[order] = np.cumsum(changed) - 1
    return codes, int(changed.sum())


def _representatives(codes: np.ndarray, count: int) -> np.ndarray:
    """One arbitrary but deterministic row position per distinct code."""
    positions = np.zeros(count, dtype=np.int64)
    positions[codes] = np.arange(len(codes), dtype=np.int64)
    return positions


def verify_household_membership(
    *,
    income_year: np.ndarray,
    source_household_id: np.ndarray,
    household_id: np.ndarray,
) -> dict:
    """Prove the grouping column partitions persons exactly as the publisher did.

    Household reference status is a statement about the household the Census
    published: the cardinality of ``A_EXPRRP`` 1/2 is evidence only inside
    ``(income_year, source_household_id)``. Classification groups instead by
    the parent roster's remapped :data:`HOUSEHOLD_GROUPING_COLUMN`, and a
    per-row equality between that roster's ``source_household_id`` and the
    member's ``PH_SEQ`` says nothing about how either column *groups*. So the
    two partitions are proven identical in both directions here, before any
    household verdict is formed.

    Refused in both directions: a grouping household carrying persons from
    more than one published household (a merge, whether inside one cohort or
    across cohorts), and a published household split across more than one
    grouping household. Admitted and counted: the same native household id
    reused by unrelated households in different income years, which is
    ordinary and is what makes the income year part of the key.

    Returns the aggregate verification recorded in the artifact receipt. It
    carries counts, column names and cohort years only; no person, household
    or native identifier appears in it or in any refusal raised here.
    """
    rows = len(income_year)
    _require(0 < rows <= _MAX_PERSONS, "MEMBERSHIP_ROWS")
    arrays = {
        "income_year": income_year,
        "source_household_id": source_household_id,
        HOUSEHOLD_GROUPING_COLUMN: household_id,
    }
    for name, values in arrays.items():
        _require(
            isinstance(values, np.ndarray)
            and values.dtype == np.dtype("int64")
            and values.shape == (rows,),
            "MEMBERSHIP_INPUT",
            name,
        )
    _require(
        bool(np.isin(income_year, list(COHORTS)).all()),
        "MEMBERSHIP_COHORT",
        "income_year",
    )
    for name in ("source_household_id", HOUSEHOLD_GROUPING_COLUMN):
        _require(bool((arrays[name] > 0).all()), "MEMBERSHIP_COORDINATE", name)

    published, published_count = _dense_codes(income_year, source_household_id)
    grouping, grouping_count = _dense_codes(household_id)
    edges, edge_count = _dense_codes(grouping, published)
    # The relation between the two partitions is exactly its distinct
    # (grouping household, published household) pairs; take one row per pair.
    edge_rows = _representatives(edges, edge_count)
    edge_grouping = grouping[edge_rows]
    edge_published = published[edge_rows]
    edge_year = income_year[edge_rows]

    _, grouping_degree = np.unique(edge_grouping, return_counts=True)
    _, published_degree = np.unique(edge_published, return_counts=True)
    cohort_edges, cohort_edge_count = _dense_codes(edge_grouping, edge_year)
    _, cohort_degree = np.unique(
        edge_grouping[_representatives(cohort_edges, cohort_edge_count)],
        return_counts=True,
    )
    native_rows = _representatives(published, published_count)
    _, native_degree = np.unique(source_household_id[native_rows], return_counts=True)

    collisions = int((cohort_degree > 1).sum())
    merged = int((grouping_degree > 1).sum())
    split = int((published_degree > 1).sum())
    _require(collisions == 0, "MEMBERSHIP_COHORT_COLLISION", HOUSEHOLD_GROUPING_COLUMN)
    _require(
        merged == 0, "MEMBERSHIP_MERGED_NATIVE_HOUSEHOLDS", HOUSEHOLD_GROUPING_COLUMN
    )
    _require(
        split == 0, "MEMBERSHIP_SPLIT_NATIVE_HOUSEHOLDS", HOUSEHOLD_GROUPING_COLUMN
    )
    # A relation whose edge count equals both vertex counts, with every vertex
    # incident to at least one edge by construction, is a bijection. Asserted
    # independently of the three counters above so a gap in any one of them
    # still cannot let an unequal partition through.
    _require(
        edge_count == grouping_count == published_count,
        "MEMBERSHIP_NOT_BIJECTIVE",
        HOUSEHOLD_GROUPING_COLUMN,
    )

    return {
        "schema_version": 1,
        "grouping_column": HOUSEHOLD_GROUPING_COLUMN,
        "published_household_key": list(HOUSEHOLD_MEMBERSHIP_KEY),
        "equivalence": "bidirectional_partition_bijection",
        "verified_before_classification": True,
        "rows": rows,
        "grouping_households": grouping_count,
        "published_households": published_count,
        "membership_edges": edge_count,
        # Recomputable from the three counts recorded immediately above it.
        "bijective": edge_count == grouping_count == published_count,
        "grouping_households_spanning_multiple_cohorts": collisions,
        "grouping_households_spanning_multiple_published_households": merged,
        "published_households_spanning_multiple_grouping_households": split,
        "native_household_ids_reused_across_cohorts": int((native_degree > 1).sum()),
        "native_household_id_reuse_across_cohorts_is_admitted": True,
        "by_cohort": {
            str(year): {
                "income_year": int(year),
                "survey_year": int(COHORTS[int(year)]),
                "rows": int((income_year == year).sum()),
                "grouping_households": int(
                    np.unique(household_id[income_year == year]).size
                ),
                "published_households": int(
                    np.unique(published[income_year == year]).size
                ),
                "native_household_ids": int(
                    np.unique(source_household_id[income_year == year]).size
                ),
            }
            for year in sorted({int(value) for value in np.unique(income_year)})
        },
        "counters_zero_in_every_successful_receipt": list(MEMBERSHIP_REFUSAL_COUNTERS),
        "zero_counter_semantics": ZERO_COUNTER_SEMANTICS,
    }


@dataclass(frozen=True)
class DemographicClassification:
    """Per-person states plus value-free counts. Holds no source identifier."""

    states: Mapping[str, np.ndarray]
    diagnostics: dict


def classify_asec_demographic_observations(
    *,
    income_year: np.ndarray,
    household_id: np.ndarray,
    a_sex: np.ndarray,
    axsex: np.ndarray,
    a_exprrp: np.ndarray,
    p_seq: np.ndarray,
) -> DemographicClassification:
    """Bind only what the printed codes and the household cardinality support.

    Sex binds when the sex code is printed and its allocation flag is printed;
    the allocation state is carried alongside, never folded into the value.
    Household headship binds when every relationship code in the household is
    printed and exactly one member carries a reference-person code. Every other
    person is reported unbound with the reason, and ``P_SEQ`` is diagnosed
    against the result without ever supplying it.

    ``household_id`` is the grouping column, and the reference-person
    cardinality this reads is only evidence about the household the Census
    published. The caller must therefore have proven that grouping equal to
    ``(income_year, source_household_id)`` with
    :func:`verify_household_membership` first; the cohort-uniformity check
    below is a weaker self-check and is not that proof.
    """
    arrays = {
        "income_year": income_year,
        "household_id": household_id,
        "A_SEX": a_sex,
        "AXSEX": axsex,
        "A_EXPRRP": a_exprrp,
        "P_SEQ": p_seq,
    }
    rows = len(income_year)
    _require(0 < rows <= _MAX_PERSONS, "CLASSIFY_ROWS")
    for name, values in arrays.items():
        _require(
            isinstance(values, np.ndarray)
            and values.dtype == np.dtype("int64")
            and values.shape == (rows,),
            "CLASSIFY_INPUT",
            name,
        )
    _require(
        bool(np.isin(income_year, list(COHORTS)).all())
        and bool((household_id > 0).all()),
        "CLASSIFY_COORDINATES",
    )

    sex_state = np.zeros(rows, dtype=np.int64)
    sex_reason = np.zeros(rows, dtype=np.int64)
    allocation = np.zeros(rows, dtype=np.int64)
    named_sex = np.isin(a_sex, list(A_SEX.named_codes))
    named_flag = np.isin(axsex, list(AXSEX.named_codes))
    allocation[named_flag & (axsex == 0)] = 1
    allocation[named_flag & (axsex == 4)] = 2
    admissible = named_sex & named_flag
    sex_state[admissible & (a_sex == 1)] = 1
    sex_state[admissible & (a_sex == 2)] = 2
    sex_reason[~named_sex] = 1
    sex_reason[named_sex & ~named_flag] = 2

    named_relationship = np.isin(a_exprrp, list(A_EXPRRP.named_codes))
    reference = np.isin(a_exprrp, list(HOUSEHOLD_REFERENCE_CODES))
    head_state = np.zeros(rows, dtype=np.int64)
    head_reason = np.zeros(rows, dtype=np.int64)
    order = np.argsort(household_id, kind="stable")
    grouped = household_id[order]
    starts = np.flatnonzero(np.concatenate(([True], grouped[1:] != grouped[:-1])))
    bounds = np.append(starts, len(grouped))
    households = len(starts)
    household_verdicts = np.zeros(households, dtype=np.int64)
    for index in range(households):
        positions = order[bounds[index] : bounds[index + 1]]
        _require(
            len(np.unique(income_year[positions])) == 1, "CLASSIFY_HOUSEHOLD_COHORT"
        )
        unnamed = ~named_relationship[positions]
        if unnamed.any():
            head_reason[positions] = 4
            head_reason[positions[unnamed]] = 1
            household_verdicts[index] = 4
            continue
        references = positions[reference[positions]]
        if len(references) == 0:
            head_reason[positions] = 2
            household_verdicts[index] = 2
            continue
        if len(references) > 1:
            head_reason[positions] = 3
            household_verdicts[index] = 3
            continue
        head_state[positions] = 2
        head_state[references[0]] = 1
        household_verdicts[index] = 1

    # P_SEQ is diagnosed, never consulted. Nothing above reads it.
    p_seq_one = p_seq == 1
    in_declared_range = np.isin(p_seq, list(P_SEQ.declared_domain))
    household_years = income_year[order[bounds[:-1]]]
    household_p_seq_ones = np.add.reduceat(p_seq_one[order].astype(np.int64), starts)

    diagnostics: dict = {
        "rows": rows,
        "households": households,
        "by_cohort": {},
        "totals": {},
        "p_seq_crosscheck": {},
    }
    cohorts = sorted({int(year) for year in np.unique(income_year)})
    for scope, mask, household_mask in [
        ("totals", np.ones(rows, dtype=bool), np.ones(households, dtype=bool)),
        *[
            (
                str(year),
                income_year == year,
                household_years == year,
            )
            for year in cohorts
        ],
    ]:
        block = {
            "income_year": None if scope == "totals" else int(scope),
            "survey_year": None
            if scope == "totals"
            else int(COHORTS.get(int(scope), int(scope) + 1)),
            "observed_period_kind": OBSERVED_PERIOD_KIND,
            "rows": int(mask.sum()),
            "households": int(household_mask.sum()),
            "field_domains": {
                field.name: {
                    "rows_in_declared_range": int(
                        np.isin(arrays[field.name][mask], field.declared_domain).sum()
                    ),
                    "rows_outside_declared_range": int(
                        (
                            ~np.isin(arrays[field.name][mask], field.declared_domain)
                        ).sum()
                    ),
                    "rows_in_named_codes": int(
                        np.isin(arrays[field.name][mask], list(field.named_codes)).sum()
                    ),
                    "rows_in_unnamed_range_codes": int(
                        np.isin(
                            arrays[field.name][mask], field.unnamed_in_range_codes
                        ).sum()
                    ),
                    "role": field.role,
                }
                for field in ASEC_DEMOGRAPHIC_FIELDS
            },
            "sex_binding_state": _counts(sex_state[mask], SEX_BINDING_STATES),
            "sex_unbound_reason": _counts(sex_reason[mask], SEX_UNBOUND_REASONS),
            "sex_allocation_state": _counts(allocation[mask], SEX_ALLOCATION_STATES),
            "household_reference_state": _counts(
                head_state[mask], HOUSEHOLD_REFERENCE_STATES
            ),
            "household_reference_unbound_reason": _counts(
                head_reason[mask], HOUSEHOLD_REFERENCE_UNBOUND_REASONS
            ),
            "household_reference_verdict": _counts(
                household_verdicts[household_mask],
                MappingProxyType(
                    {
                        1: "exactly_one_reference",
                        2: "no_reference",
                        3: "multiple_references",
                        4: "indeterminate_relationship_code",
                    }
                ),
            ),
            "p_seq_crosscheck": {
                "declared_role": P_SEQ.concept,
                "named_codes": {},
                "is_semantic_proof_of_headship": False,
                "used_as_fallback": False,
                "rows_with_p_seq_one": int(p_seq_one[mask].sum()),
                "rows_outside_declared_range": int((~in_declared_range[mask]).sum()),
                "households_with_one_p_seq_one": int(
                    (household_p_seq_ones[household_mask] == 1).sum()
                ),
                "households_with_no_p_seq_one": int(
                    (household_p_seq_ones[household_mask] == 0).sum()
                ),
                "households_with_multiple_p_seq_one": int(
                    (household_p_seq_ones[household_mask] > 1).sum()
                ),
                "agreement_rows": int((p_seq_one & reference)[mask].sum()),
                "p_seq_one_without_reference_code": int(
                    (p_seq_one & ~reference)[mask].sum()
                ),
                "reference_code_without_p_seq_one": int(
                    (reference & ~p_seq_one)[mask].sum()
                ),
                "unbound_households_with_one_p_seq_one": int(
                    ((household_verdicts != 1) & (household_p_seq_ones == 1))[
                        household_mask
                    ].sum()
                ),
            },
        }
        if scope == "totals":
            diagnostics["totals"] = block
            diagnostics["p_seq_crosscheck"] = block["p_seq_crosscheck"]
        else:
            diagnostics["by_cohort"][scope] = block
    diagnostics["family_relationship_field_read"] = False
    diagnostics["line_number_used_for_headship"] = False
    diagnostics["household_and_family_roles_separate"] = True
    return DemographicClassification(
        MappingProxyType(
            {
                "asec_sex_binding_state": sex_state,
                "asec_sex_unbound_reason": sex_reason,
                "asec_sex_allocation_state": allocation,
                "asec_household_reference_state": head_state,
                "asec_household_reference_unbound_reason": head_reason,
            }
        ),
        diagnostics,
    )


#: The ACS side is *not* rewritten by this contract. These are the pinned 2024
#: 1-year PUMS codes the existing reader already branches on, restated so the
#: shared classifier below can read the ACS arm without inventing a head.
ACS_OBSERVED_REFERENCE_CODE = 20
ACS_GROUP_QUARTERS_CODES: tuple[int, int] = (37, 38)
#: What ``acs_pums`` already writes for a group-quarters-only household. It is
#: a printed A_EXPRRP nonrelative code, deliberately not a reference code, so a
#: synthetic group-quarters representative can never be read as an observed
#: housing-unit reference person.
ACS_SYNTHETIC_GROUP_QUARTERS_RELATIONSHIP_CODE = 14
ACS_REFERENCE_STATES: Mapping[int, str] = MappingProxyType(
    {
        0: "unbound",
        1: "observed_housing_unit_reference_person",
        2: "observed_not_reference_person",
        3: "group_quarters_no_observed_reference_person",
    }
)


def acs_household_reference_states(
    *, household_id: np.ndarray, relshipp: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Read the ACS arm's own observed reference person; never manufacture one.

    ``RELSHIPP == 20`` is the self-labelled reference person of a housing unit.
    Codes 37 and 38 are the institutional and noninstitutional group-quarters
    populations; they are neither a reference person nor a relationship to one,
    so their households report *no* observed reference person and stay
    distinguishable from a housing unit throughout.
    """
    rows = len(household_id)
    _require(0 < rows <= _MAX_PERSONS, "ACS_ROWS")
    for name, values in (("household_id", household_id), ("RELSHIPP", relshipp)):
        _require(
            isinstance(values, np.ndarray)
            and values.dtype == np.dtype("int64")
            and values.shape == (rows,),
            "ACS_INPUT",
            name,
        )
    group_quarters = np.isin(relshipp, list(ACS_GROUP_QUARTERS_CODES))
    valid_relationship = np.isin(relshipp, np.arange(20, 39))
    reference = relshipp == ACS_OBSERVED_REFERENCE_CODE
    states = np.where(reference, 1, 2).astype(np.int64)
    order = np.argsort(household_id, kind="stable")
    grouped = household_id[order]
    starts = np.flatnonzero(np.concatenate(([True], grouped[1:] != grouped[:-1])))
    bounds = np.append(starts, len(grouped))
    housing_units = 0
    gq_households = 0
    unbound_housing_units = 0
    bad_reference_cardinality = 0
    mixed_households = 0
    invalid_households = 0
    for index in range(len(starts)):
        positions = order[bounds[index] : bounds[index + 1]]
        if group_quarters[positions].all():
            states[positions] = 3
            gq_households += 1
            continue
        housing_units += 1
        bad_reference_cardinality += int(int(reference[positions].sum()) != 1)
        mixed_households += int(group_quarters[positions].any())
        invalid_households += int(not valid_relationship[positions].all())
        if (
            group_quarters[positions].any()
            or not valid_relationship[positions].all()
            or int(reference[positions].sum()) != 1
        ):
            states[positions] = 0
            unbound_housing_units += 1
    diagnostics = {
        "rows": rows,
        "households": len(starts),
        "housing_unit_households": housing_units,
        "group_quarters_only_households": gq_households,
        "unbound_housing_unit_households": unbound_housing_units,
        "housing_unit_households_without_exactly_one_reference": bad_reference_cardinality,
        "mixed_housing_unit_group_quarters_households": mixed_households,
        "households_with_invalid_relationship_codes": invalid_households,
        "state_counts": _counts(states, ACS_REFERENCE_STATES),
        "group_quarters_rows": int(group_quarters.sum()),
        "institutional_group_quarters_rows": int((relshipp == 37).sum()),
        "noninstitutional_group_quarters_rows": int((relshipp == 38).sum()),
        "manufactured_group_quarters_reference_person": False,
        "synthetic_group_quarters_relationship_code": (
            ACS_SYNTHETIC_GROUP_QUARTERS_RELATIONSHIP_CODE
        ),
    }
    return states, diagnostics


@dataclass(frozen=True)
class AuthenticatedAsecDemographicSource:
    """Immutable owned numeric buffers issued only by closed reconstruction."""

    _header: bytes
    _body: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _TOKEN, "DEMOGRAPHIC_CONSTRUCTOR_UNAVAILABLE")
        self.validate()

    @property
    def receipt(self) -> dict:
        return json.loads(self._header.decode("ascii"))

    @property
    def content_sha256(self) -> str:
        return _sha(self._header + self._body)

    def validate(self) -> None:
        _require(
            type(self._header) is bytes and 0 < len(self._header) <= _HEADER_MAX,
            "DEMOGRAPHIC_HEADER_SIZE",
        )
        header = self.receipt
        rows = header["rows"]
        _require(type(rows) is int and 0 < rows <= _MAX_PERSONS, "DEMOGRAPHIC_ROWS")
        _require(
            type(self._body) is bytes
            and len(self._body) == rows * len(COLUMNS) * 8
            and _sha(self._body) == header["body_sha256"]
            and header["columns"] == list(COLUMNS)
            and _json(header["contract"]) == _json(contract_document())
            and _json(header["implementation"]) == _json(_implementation()),
            "DEMOGRAPHIC_CONTENT_CHANGED",
        )

    def array(self, name: str) -> np.ndarray:
        self.validate()
        _require(name in COLUMNS, "DEMOGRAPHIC_COLUMN")
        rows = self.receipt["rows"]
        offset = COLUMNS.index(name) * rows * 8
        return np.frombuffer(self._body, dtype="<i8", count=rows, offset=offset)


def _encode(value: AuthenticatedAsecDemographicSource) -> bytes:
    value.validate()
    payload = (
        MAGIC + struct.pack("<I", len(value._header)) + value._header + value._body
    )
    return payload + hashlib.sha256(payload).digest()


def _parent(source) -> None:
    _require(
        type(source) is legacy.AuthenticatedCurrentMoneySource,
        "DEMOGRAPHIC_AUTHENTICATED_PARENT",
    )
    source.validate()


def _reconstruct(source, member_paths: Mapping[int, str | Path]):
    restoration._paths(member_paths)
    _parent(source)
    before = _implementation()
    person = source.frame.person
    rows = len(person)
    _require(0 < rows <= _MAX_PERSONS, "DEMOGRAPHIC_ROWS")
    _require(not set(ALIASES) & set(person), "DEMOGRAPHIC_ALREADY_ATTACHED")
    _require(
        set(COORDINATES[2:]) <= set(person) and "person_household_id" in person,
        "DEMOGRAPHIC_PARENT_ROSTER",
    )
    years = np.asarray(source.scope.person_years, dtype=np.int64)
    keys = np.asarray(source.scope.person_native_keys)
    output: dict[str, np.ndarray] = {
        "person_id": np.asarray(source.scope.person_ids, dtype=np.int64),
        "income_year": years,
    }
    for name in COORDINATES[2:]:
        _require(
            person[name].dtype == np.dtype("int64"),
            "DEMOGRAPHIC_COORDINATE_DTYPE",
            name,
        )
        output[name] = person[name].to_numpy(copy=True)
    # The grouping column is a coordinate of the headship verdict, so it is
    # typed here rather than coerced by the conversion that consumes it.
    _require(
        person[HOUSEHOLD_GROUPING_COLUMN].dtype == np.dtype("int64"),
        "DEMOGRAPHIC_COORDINATE_DTYPE",
        HOUSEHOLD_GROUPING_COLUMN,
    )
    household_id = person[HOUSEHOLD_GROUPING_COLUMN].to_numpy(copy=True)
    for alias in ALIASES:
        output[alias] = np.empty(rows, dtype=np.int64)
    joins = []
    with tempfile.TemporaryDirectory(
        prefix="microcosm-demographic-members-"
    ) as directory:
        for year, member, archive_pin, pin, member_rows, size in _MEMBER_PINS:
            staged = Path(directory) / member
            _require(
                _snapshot(member_paths[year], staged, size=size) == pin,
                "DEMOGRAPHIC_SOURCE_BYTES",
            )
            raw = read_demographic_member(staged, rows=member_rows, size=size)
            staged.unlink()
            positions = np.flatnonzero(years == year)
            _require(len(positions) == member_rows, "DEMOGRAPHIC_COHORT_ROWS")
            recipient = pd.Index(keys[positions])
            _require(not recipient.has_duplicates, "DEMOGRAPHIC_PERSON_KEY")
            indices = pd.Index(raw.PERIDNUM).get_indexer(recipient)
            _require(
                bool((indices >= 0).all()) and len(np.unique(indices)) == member_rows,
                "DEMOGRAPHIC_KEY_COVERAGE",
            )
            joined = raw.iloc[indices]
            coordinate_conflicts = 0
            for name, raw_name in (
                ("source_household_id", "PH_SEQ"),
                ("A_LINENO", "A_LINENO"),
                ("A_AGE", "A_AGE"),
            ):
                conflicts = int(
                    (output[name][positions] != joined[raw_name].to_numpy()).sum()
                )
                coordinate_conflicts += conflicts
                _require(conflicts == 0, "DEMOGRAPHIC_NATIVE_KEY_OR_AGE", name)
            compared = {}
            incumbent_conflicts = 0
            relationship_crosscheck = {
                "incumbent_present": "A_EXPRRP" in person,
                "compared_rows": 0,
                "missing_rows": 0,
                "mismatch_rows": 0,
                "presence_certifies_source_observation": False,
                "used_as_semantic_proof": False,
            }
            for alias, name in zip(ALIASES, OBSERVATIONS, strict=True):
                observed = joined[name].to_numpy(dtype=np.int64)
                output[alias][positions] = observed
                if name not in person:
                    compared[name] = None
                    continue
                incumbent = person[name].iloc[positions]
                known = ~incumbent.isna().to_numpy()
                equal = (
                    incumbent.to_numpy(dtype=np.float64, na_value=np.nan)[known]
                    == observed[known]
                )
                if name == "A_EXPRRP":
                    # Older prepared cohorts carry an A_LINENO-derived recode.
                    # The restored alias owns the CSV observation; an unverified
                    # incumbent may diagnose a discrepancy but cannot veto it.
                    relationship_crosscheck.update(
                        compared_rows=int(known.sum()),
                        missing_rows=int((~known).sum()),
                        mismatch_rows=int((~equal).sum()),
                    )
                    compared[name] = int(known.sum())
                    continue
                conflicts = int((~equal).sum())
                incumbent_conflicts += conflicts
                _require(conflicts == 0, "DEMOGRAPHIC_INCUMBENT_CONFLICT", name)
                compared[name] = int(known.sum())
            joins.append(
                {
                    "income_year": year,
                    "survey_year": COHORTS[year],
                    "member": member,
                    "archive_sha256": archive_pin,
                    "member_sha256": pin,
                    "source_rows": member_rows,
                    "joined_rows": len(positions),
                    "unreferenced_source_rows": int(
                        member_rows - len(np.unique(indices))
                    ),
                    "incumbent_compared_rows": compared,
                    "incumbent_relationship_crosscheck": relationship_crosscheck,
                    "accepted_source_missing_tokens": dict.fromkeys(OBSERVATIONS, 0),
                    "incumbent_absent_columns": sorted(
                        name for name, value in compared.items() if value is None
                    ),
                    "incumbent_conflicts": incumbent_conflicts,
                    "native_key_or_age_conflicts": coordinate_conflicts,
                    "counters_zero_in_every_successful_receipt": [
                        "accepted_source_missing_tokens",
                        "incumbent_conflicts",
                        "native_key_or_age_conflicts",
                        "unreferenced_source_rows",
                    ],
                    "zero_counter_semantics": ZERO_COUNTER_SEMANTICS,
                }
            )
    membership = verify_household_membership(
        income_year=years,
        source_household_id=output["source_household_id"],
        household_id=household_id,
    )
    classification = classify_asec_demographic_observations(
        income_year=years,
        household_id=household_id,
        a_sex=output["asec_A_SEX"],
        axsex=output["asec_AXSEX"],
        a_exprrp=output["asec_A_EXPRRP"],
        p_seq=output["asec_P_SEQ"],
    )
    output.update({name: values for name, values in classification.states.items()})
    _parent(source)
    _require(
        _json(_implementation()) == _json(before),
        "DEMOGRAPHIC_IMPLEMENTATION_CHANGED",
    )
    buffers = [output[name].astype("<i8", copy=False).tobytes() for name in COLUMNS]
    body = b"".join(buffers)
    header = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "rows": rows,
        "columns": list(COLUMNS),
        "dtype": "little_endian_int64",
        "aliases": dict(zip(OBSERVATIONS, ALIASES, strict=True)),
        "source_identity": source.source.identity.decode(),
        "source_frame_sha256": legacy._frame_signature(source.frame),
        "native_person_keys_sha256": _sha(_json(source.scope.person_native_keys)),
        "sources": joins,
        "join_keys": ["source_year", "PERIDNUM"],
        "crosschecks": ["source_household_id", "A_LINENO", "A_AGE"],
        "household_grouping_column": HOUSEHOLD_GROUPING_COLUMN,
        "household_membership": membership,
        "diagnostics": classification.diagnostics,
        "contract": contract_document(),
        "release_eligible": False,
        "implementation": before,
        "body_sha256": _sha(body),
        "column_sha256": dict(zip(COLUMNS, map(_sha, buffers), strict=True)),
        "encoding_contract": ("independently_reconstructed_canonical_numeric_bytes_v1"),
    }
    return AuthenticatedAsecDemographicSource(_json(header), body, _token=_TOKEN)


def load_authenticated_asec_demographic_source(
    source, *, member_paths, candidate_path=None
) -> AuthenticatedAsecDemographicSource:
    """Reconstruct from pinned members; a candidate supplies bytes, not content."""
    try:
        expected = _reconstruct(source, member_paths)
        if candidate_path is not None:
            payload = _encode(expected)
            with tempfile.TemporaryDirectory(
                prefix="microcosm-demographic-candidate-"
            ) as directory:
                digest = _snapshot(
                    candidate_path, Path(directory) / FILENAME, size=len(payload)
                )
            _require(digest == _sha(payload), "DEMOGRAPHIC_CANONICAL_BYTES")
        _parent(source)
        expected.validate()
        return expected
    except DemographicSourceRefusalError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        UnicodeError,
        csv.Error,
        re.error,
    ):
        raise DemographicSourceRefusalError("DEMOGRAPHIC_SOURCE_CONTRACT") from None


def _attachment_receipt(frame, source, observations) -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "microcosm.asec_demographic_source_attachment.v1",
        "source_identity_sha256": _sha(source.source.identity),
        "observations_content_sha256": observations.content_sha256,
        "columns": list(ALIASES + DERIVED),
        "output_frame_sha256": legacy._frame_signature(frame),
        "release_eligible": False,
    }


def _check_attachment_parents(source, observations) -> None:
    _parent(source)
    _require(
        type(observations) is AuthenticatedAsecDemographicSource,
        "DEMOGRAPHIC_AUTHENTICATED_OBSERVATIONS",
    )
    observations.validate()
    _require(
        observations.receipt["source_identity"].encode() == source.source.identity,
        "DEMOGRAPHIC_ATTACHMENT_PARENT",
    )


@dataclass(frozen=True)
class DemographicSourceAttachedAsec:
    """Owned additive Frame beside an unchanged authenticated money parent."""

    frame: Frame
    parent: legacy.AuthenticatedCurrentMoneySource
    observations: AuthenticatedAsecDemographicSource
    _receipt: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(
            _token is _COMPOSE_TOKEN,
            "DEMOGRAPHIC_ATTACHMENT_CONSTRUCTOR_UNAVAILABLE",
        )

    @property
    def receipt(self) -> dict:
        return json.loads(self._receipt.decode("ascii"))

    def validate(self) -> None:
        try:
            _check_attachment_parents(self.parent, self.observations)
            _require(type(self.frame) is Frame, "DEMOGRAPHIC_ATTACHMENT_FRAME")
            for name in ALIASES + DERIVED:
                values = self.frame.person[name]
                _require(
                    values.dtype == np.dtype("int64")
                    and np.array_equal(
                        values.to_numpy(), self.observations.array(name)
                    ),
                    "DEMOGRAPHIC_ATTACHMENT_OUTPUT",
                    name,
                )
            recovered = restoration._owned_frame(self.frame)
            recovered.person.drop(columns=list(ALIASES + DERIVED), inplace=True)
            _require(
                legacy._frame_signature(recovered)
                == self.observations.receipt["source_frame_sha256"]
                and self.receipt
                == _attachment_receipt(self.frame, self.parent, self.observations),
                "DEMOGRAPHIC_ATTACHMENT_CHANGED",
            )
        except DemographicSourceRefusalError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise DemographicSourceRefusalError(
                "DEMOGRAPHIC_ATTACHMENT_CHANGED"
            ) from error


def attach_asec_demographic_source(
    source, observations
) -> DemographicSourceAttachedAsec:
    """Append the restored observations and their states; change nothing else."""
    _check_attachment_parents(source, observations)
    _require(
        not set(ALIASES + DERIVED) & set(source.frame.person),
        "DEMOGRAPHIC_ALREADY_ATTACHED",
    )
    frame = restoration._owned_frame(source.frame)
    for name in ALIASES + DERIVED:
        frame.person[name] = observations.array(name).copy()
    result = DemographicSourceAttachedAsec(
        frame,
        source,
        observations,
        _json(_attachment_receipt(frame, source, observations)),
        _token=_COMPOSE_TOKEN,
    )
    result.validate()
    return result


def write_asec_demographic_source(source, *, member_paths, output_dir) -> dict:
    """Produce a new immutable local bundle; never replace an existing parent."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    value = load_authenticated_asec_demographic_source(
        source, member_paths=member_paths
    )
    payload = _encode(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".asec-demographic-", dir=destination.parent
    ) as directory:
        staging = Path(directory)
        (staging / FILENAME).write_bytes(payload)
        receipt = {
            **value.receipt,
            "content_sha256": value.content_sha256,
            "output_file": FILENAME,
            "output_sha256": _sha(payload),
        }
        (staging / "receipt.json").write_bytes(_json(receipt) + b"\n")
        destination.mkdir()
        for path in staging.iterdir():
            os.link(path, destination / path.name)
    return receipt


__all__ = [
    "ACS_GROUP_QUARTERS_CODES",
    "ACS_OBSERVED_REFERENCE_CODE",
    "ACS_REFERENCE_STATES",
    "ACS_SYNTHETIC_GROUP_QUARTERS_RELATIONSHIP_CODE",
    "ALIASES",
    "ARTIFACT_KIND",
    "ASEC_DEMOGRAPHIC_FIELDS",
    "ASEC_DEMOGRAPHIC_SOURCE_COLUMNS",
    "ASEC_DICTIONARY_AUTHORITY",
    "COHORTS",
    "COLUMNS",
    "DERIVED",
    "HOUSEHOLD_GROUPING_COLUMN",
    "HOUSEHOLD_MEMBERSHIP_KEY",
    "HOUSEHOLD_REFERENCE_CODES",
    "HOUSEHOLD_REFERENCE_STATES",
    "HOUSEHOLD_REFERENCE_UNBOUND_REASONS",
    "MEMBERSHIP_REFUSAL_COUNTERS",
    "NON_ADOPTED_HEAD_SIGNALS",
    "OBSERVATIONS",
    "OBSERVED_PERIOD_KIND",
    "SEX_ALLOCATION_STATES",
    "SEX_BINDING_STATES",
    "SEX_UNBOUND_REASONS",
    "ZERO_COUNTER_SEMANTICS",
    "AsecDemographicField",
    "AuthenticatedAsecDemographicSource",
    "DemographicClassification",
    "DemographicSourceAttachedAsec",
    "DemographicSourceRefusalError",
    "acs_household_reference_states",
    "attach_asec_demographic_source",
    "classify_asec_demographic_observations",
    "contract_document",
    "load_authenticated_asec_demographic_source",
    "read_demographic_member",
    "verify_household_membership",
    "write_asec_demographic_source",
]
