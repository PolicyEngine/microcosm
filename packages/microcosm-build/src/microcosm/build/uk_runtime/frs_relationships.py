"""FRS household-grid relationships and the ONS household-composition split.

The stage (microcosm#791) reads the licensed FRS household grid — ``relhrp``
(each person's relationship to the household reference person) and the
pairwise ``R01``-``R14`` matrix — and derives four frame columns: the
relationship to the head, each person's role inside an ONS family, the family
index within the household, and the household's ONS Families and households
Table 7 type. The ten household-type values are Chronicle's own category
identifiers for the ``ons.household_type`` dimension, so the frame value, the
manifest declaration, the contract row and the published fact share one
string.

Three code lists, three owners, each named for its source:

* ``FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES`` — DWP's FRS relationship codes
  (UK Data Service SN 9563, FRS 2024-25 question instructions, Household
  Grid: Relationship).
* ``ONS_FAMILY_LINK_CODES`` / ``ONS_DEPENDENT_CHILD_RULE`` /
  ``ONS_ONE_PERSON_AGE_SPLIT`` — the ONS household-type definitions
  (Families and households in the UK: 2025, Measuring the data / Glossary)
  expressed over those FRS codes.
* ``CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS`` — the Chronicle
  ``ons-families-households-2025`` package's ``ons.household_type`` value ids.

The manifest declares the same lists as the ``derive_ons_household_composition``
operation's parameters; :func:`assert_frs_relationships_stage_parameters`
refuses a build whose manifest and runtime disagree on any of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import normalize_ids, read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame

__all__ = [
    "CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS",
    "FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES",
    "FRS_RELATIONSHIPS_DERIVE_OPERATION_KIND",
    "FRS_RELATIONSHIPS_DOCUMENTATION",
    "FRS_RELATIONSHIPS_OUTPUT_COLUMNS",
    "FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE",
    "FRS_RELATIONSHIPS_SOURCE_COLUMNS",
    "FRS_RELATIONSHIPS_STAGE_NAME",
    "FRS_RELATIONSHIPS_TABLES",
    "ONS_DEPENDENT_CHILD_RULE",
    "ONS_FAMILY_LINK_CODES",
    "ONS_FAMILY_ROLE_VALUES",
    "ONS_ONE_PERSON_AGE_SPLIT",
    "RELATIONSHIP_TO_HEAD_HEAD_LABEL",
    "RELATIONSHIP_TO_HEAD_VALUES",
    "UKFRSRelationshipsResult",
    "UKFRSRelationshipsStageTransform",
    "add_frs_relationships",
    "assert_frs_relationships_stage_parameters",
    "derive_frs_relationships",
    "frs_relationships_domains",
    "frs_relationships_operation_parameters",
]

FRS_RELATIONSHIPS_STAGE_NAME = "frs_relationships"
FRS_RELATIONSHIPS_DERIVE_OPERATION_KIND = "derive_ons_household_composition"
FRS_RELATIONSHIPS_TABLES = ("adult", "child", "househol")
FRS_RELATIONSHIPS_OUTPUT_COLUMNS = (
    "relationship_to_head",
    "ons_family_role",
    "ons_family_index",
    "ons_household_type",
)
FRS_RELATIONSHIPS_PERSON_OUTPUT_COLUMNS = (
    "relationship_to_head",
    "ons_family_role",
    "ons_family_index",
)
FRS_RELATIONSHIPS_HOUSEHOLD_OUTPUT_COLUMNS = ("ons_household_type",)

#: Raw FRS columns the stage reads. ``grid`` is the pairwise relationship
#: matrix keyed to the within-household person number (``person_id % 1000``);
#: ``full_time_education`` lists the adult-tab flag in fallback order (the
#: 2024-25 tape carries ``educft``; earlier tapes carried ``fted``).
FRS_RELATIONSHIPS_SOURCE_COLUMNS: Mapping[str, Any] = {
    "head": "relhrp",
    "head_flag": "hrpid",
    "head_index": "hrpnum",
    "grid": [f"r{index:02d}" for index in range(1, 15)],
    "full_time_education": ["fted", "educft"],
}
FRS_HOUSEHOLD_GRID_MAX_PERSONS = 14

#: DWP FRS Household Grid relationship codes (UK Data Service SN 9563, FRS
#: 2024-25 question instructions, "Relationship"). Code 19 is not issued and
#: 97 is reserved; blank means the person is the household reference person.
FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES: Mapping[int, str] = {
    1: "SPOUSE",
    2: "COHABITEE",
    3: "CHILD",  # son/daughter, including adopted and legal dependent
    4: "STEP_CHILD",
    5: "FOSTER_CHILD",
    6: "CHILD_IN_LAW",
    7: "PARENT",  # father/mother or guardian
    8: "STEP_PARENT",
    9: "FOSTER_PARENT",
    10: "PARENT_IN_LAW",
    11: "SIBLING",  # brother/sister, including adopted
    12: "STEP_SIBLING",  # half-siblings are coded here per the instructions
    13: "FOSTER_SIBLING",
    14: "SIBLING_IN_LAW",
    15: "GRANDCHILD",
    16: "GRANDPARENT",
    17: "OTHER_RELATIVE",
    18: "OTHER_NON_RELATIVE",
    20: "CIVIL_PARTNER",
}
RELATIONSHIP_TO_HEAD_HEAD_LABEL = "HEAD"
RELATIONSHIP_TO_HEAD_VALUES: tuple[str, ...] = (
    RELATIONSHIP_TO_HEAD_HEAD_LABEL,
    *FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES.values(),
)

#: ONS "family" links over the FRS codes: a couple is any partner pair; a
#: child belongs to a parent's family through a natural/adopted or step link
#: (the parent's record carries the reciprocal parent code). Foster, in-law,
#: grandparent and sibling links never form a family (ONS treats foster
#: children and children living without a parent as one-person family units).
ONS_FAMILY_LINK_CODES: Mapping[str, tuple[int, ...]] = {
    "partner": (1, 2, 20),
    "child_of": (3, 4),
    "parent_of": (7, 8),
}
#: ONS dependent child: under 16, or 16-18 and in full-time education, and
#: (by construction of the family assignment) with no partner or own child in
#: the household. 19+ is non-dependent whatever the education status.
ONS_DEPENDENT_CHILD_RULE: Mapping[str, int] = {
    "unconditional_below_age": 16,
    "full_time_education_through_age": 18,
}
#: ONS one-person households split at 65 ("65 or over").
ONS_ONE_PERSON_AGE_SPLIT = 65
ONS_FAMILY_ROLE_VALUES: tuple[str, ...] = (
    "COUPLE_PARTNER",
    "LONE_PARENT",
    "DEPENDENT_CHILD",
    "NON_DEPENDENT_CHILD",
    "INDIVIDUAL",
)
ONS_INDIVIDUAL_FAMILY_INDEX = 0

#: Chronicle ``ons-families-households-2025`` Table 7 ``ons.household_type``
#: value ids, in the published row order. These are the frame values.
CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS: tuple[str, ...] = (
    "lone_households_under_65",
    "lone_households_over_65",
    "unrelated_adult_households",
    "couple_no_children_households",
    "couple_under_3_children_households",
    "couple_3_plus_children_households",
    "couple_non_dependent_children_only_households",
    "lone_parent_dependent_children_households",
    "lone_parent_non_dependent_children_households",
    "multi_family_households",
)
#: Reviewed count of grid reciprocity mismatches on the pinned 2024-25 tape:
#: three ordered pairs in two households where a child records a parent link
#: the parent records as non-relative or sibling (step-0 probe, microcosm#791).
#: Either side's declaration establishes the link; any other count refuses.
FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE = 3
FRS_RELATIONSHIPS_DOCUMENTATION: Mapping[str, str] = {
    "frs_codes": (
        "UK Data Service SN 9563 (DOI 10.5255/UKDA-SN-9563-1), FRS 2024-25 "
        "question instructions, Household Grid: Relationship; HRP definition "
        "in the FRS background information and methodology"
    ),
    "ons_definitions": (
        "ONS Families and households in the UK: 2025, Measuring the data and "
        "Glossary (family, dependent and non-dependent children, one-family, "
        "two or more unrelated adults, multi-family households)"
    ),
    "category_ids": (
        "chronicle package ons-families-households-2025, Table 7, dimension "
        "ons.household_type value ids"
    ),
}


def frs_relationships_operation_parameters() -> dict[str, Any]:
    """The ``derive_ons_household_composition`` parameters the runtime implements."""

    return {
        "kind": FRS_RELATIONSHIPS_DERIVE_OPERATION_KIND,
        "source_columns": {
            "head": FRS_RELATIONSHIPS_SOURCE_COLUMNS["head"],
            "head_flag": FRS_RELATIONSHIPS_SOURCE_COLUMNS["head_flag"],
            "head_index": FRS_RELATIONSHIPS_SOURCE_COLUMNS["head_index"],
            "grid": list(FRS_RELATIONSHIPS_SOURCE_COLUMNS["grid"]),
            "full_time_education": list(
                FRS_RELATIONSHIPS_SOURCE_COLUMNS["full_time_education"]
            ),
        },
        "frs_household_grid_relationship_codes": [
            {"code": code, "label": label}
            for code, label in FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES.items()
        ],
        "head_label": RELATIONSHIP_TO_HEAD_HEAD_LABEL,
        "family_link_codes": {
            key: list(codes) for key, codes in ONS_FAMILY_LINK_CODES.items()
        },
        "dependent_child": dict(ONS_DEPENDENT_CHILD_RULE),
        "one_person_age_split": ONS_ONE_PERSON_AGE_SPLIT,
        "family_role_values": list(ONS_FAMILY_ROLE_VALUES),
        "household_type_values": list(CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS),
        "reciprocity_mismatch_tolerance": (
            FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE
        ),
        "documentation": dict(FRS_RELATIONSHIPS_DOCUMENTATION),
    }


def assert_frs_relationships_stage_parameters(stage: SourceStageSpec) -> None:
    """Closed-world drift assert: the manifest declares exactly this stage."""

    if stage.stage != FRS_RELATIONSHIPS_STAGE_NAME:
        raise ValueError(f"frs_relationships transform received stage {stage.stage!r}.")
    operations = list(stage.operations)
    kinds = [operation.kind for operation in operations]
    if kinds != ["read_tables", FRS_RELATIONSHIPS_DERIVE_OPERATION_KIND]:
        raise ValueError(
            "frs_relationships stage must declare read_tables followed by "
            f"{FRS_RELATIONSHIPS_DERIVE_OPERATION_KIND}, got {kinds}."
        )
    declared = {"kind": operations[1].kind, **dict(operations[1].parameters)}
    expected = frs_relationships_operation_parameters()
    for key, value in expected.items():
        if _plain(declared.get(key)) != _plain(value):
            raise ValueError(
                f"frs_relationships stage parameter drift: {key!r} declares "
                f"{declared.get(key)!r} but the runtime implements {value!r}."
            )
    extra = set(declared) - set(expected) - {"reason"}
    if extra:
        raise ValueError(
            f"frs_relationships stage declares parameter(s) {sorted(extra)} that "
            "the runtime does not implement."
        )
    if tuple(stage.outputs) != FRS_RELATIONSHIPS_OUTPUT_COLUMNS:
        raise ValueError(
            "frs_relationships stage outputs drift: manifest declares "
            f"{tuple(stage.outputs)!r}, runtime writes "
            f"{FRS_RELATIONSHIPS_OUTPUT_COLUMNS!r}."
        )


def _plain(value: Any) -> Any:
    """Normalise tuples/lists and mapping proxies so JSON and Python compare."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def frs_relationships_domains() -> dict[str, tuple[str, ...]]:
    """Enum domains of the three string outputs, for gates and tests."""

    return {
        "relationship_to_head": RELATIONSHIP_TO_HEAD_VALUES,
        "ons_family_role": ONS_FAMILY_ROLE_VALUES,
        "ons_household_type": CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS,
    }


@dataclass(frozen=True)
class UKFRSRelationshipsResult:
    """Derived person and household columns plus the stage evidence."""

    person_values: pd.DataFrame
    household_values: pd.DataFrame
    evidence: Mapping[str, Any]


class UKFRSRelationshipsStageTransform:
    """Whole-stage callable for the FRS relationship and composition derivation."""

    def __init__(self, raw_dir: str | Path, *, stage: SourceStageSpec) -> None:
        self.raw_dir = Path(raw_dir)
        self.stage = stage
        self.last_result: UKFRSRelationshipsResult | None = None

    def __call__(self, frame: Frame) -> Frame:
        result, new_frame = add_frs_relationships(frame, self.raw_dir, stage=self.stage)
        self.last_result = result
        return new_frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_RELATIONSHIPS_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": dict(self.last_result.evidence)}


def add_frs_relationships(
    frame: Frame, raw_dir: str | Path, *, stage: SourceStageSpec
) -> tuple[UKFRSRelationshipsResult, Frame]:
    """Read the pinned FRS tabs and write the four relationship columns."""

    assert_frs_relationships_stage_parameters(stage)
    artifacts = _artifact_by_table(stage)
    tables = {
        table: normalize_ids(
            read_pinned_tab(
                Path(raw_dir) / str(artifacts[table]["locator"]), artifacts[table]
            )
        )
        for table in FRS_RELATIONSHIPS_TABLES
    }
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    weights = frame.weights_for("household").values
    result = derive_frs_relationships(
        person,
        household,
        raw_adult=tables["adult"],
        raw_child=tables["child"],
        raw_household=tables["househol"],
        household_weights=weights,
    )
    for column in FRS_RELATIONSHIPS_PERSON_OUTPUT_COLUMNS:
        person[column] = result.person_values[column].to_numpy()
    for column in FRS_RELATIONSHIPS_HOUSEHOLD_OUTPUT_COLUMNS:
        household[column] = result.household_values[column].to_numpy()
    new_frame = uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(new_frame)
    return result, new_frame


def _artifact_by_table(stage: SourceStageSpec) -> dict[str, Mapping[str, Any]]:
    by_table = {str(artifact.get("table")): artifact for artifact in stage.artifacts}
    missing = sorted(set(FRS_RELATIONSHIPS_TABLES) - set(by_table))
    if missing:
        raise ValueError(
            f"frs_relationships manifest is missing tab artifact(s): {missing}."
        )
    return by_table


class FRSRelationshipsError(ValueError):
    """A licensed-tape defect the derivation refuses to paper over."""


def derive_frs_relationships(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    raw_adult: pd.DataFrame,
    raw_child: pd.DataFrame,
    raw_household: pd.DataFrame | None = None,
    household_weights: np.ndarray | pd.Series | None = None,
    reciprocity_mismatch_tolerance: int = (
        FRS_RELATIONSHIPS_RECIPROCITY_MISMATCH_TOLERANCE
    ),
) -> UKFRSRelationshipsResult:
    """Derive the relationship, family and household-type columns.

    ``person`` and ``household`` are frame tables (``person_id``,
    ``person_household_id``, ``age``, ``is_household_head``;
    ``household_id``). ``raw_adult`` / ``raw_child`` / ``raw_household`` are
    the id-normalised FRS tabs. Pure pandas/numpy; no engine.
    """

    raw = _aligned_raw_person(person, raw_adult, raw_child)
    person_ids = person["person_id"].to_numpy(dtype="int64")
    household_ids = person["person_household_id"].to_numpy(dtype="int64")
    age_values = pd.to_numeric(person["age"], errors="coerce")
    if age_values.isna().any():
        # A blank age would read as 0 and silently make a dependent child.
        raise FRSRelationshipsError(
            f"{int(age_values.isna().sum())} frame person(s) carry a non-numeric "
            "age; the derivation refuses rather than reading them as 0."
        )
    ages = age_values.to_numpy()
    frame_head = person["is_household_head"].to_numpy(dtype=bool)
    person_index = person_ids % 1000
    grid_columns = list(FRS_RELATIONSHIPS_SOURCE_COLUMNS["grid"])
    grid = (
        raw[grid_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype="int64")
    )
    relhrp = pd.to_numeric(
        raw[FRS_RELATIONSHIPS_SOURCE_COLUMNS["head"]], errors="coerce"
    )
    hrpid = pd.to_numeric(
        raw.get(FRS_RELATIONSHIPS_SOURCE_COLUMNS["head_flag"]), errors="coerce"
    )
    hrpid = (pd.Series(hrpid, index=raw.index).fillna(0) == 1).to_numpy(dtype=bool)
    fte_column = next(
        (
            column
            for column in FRS_RELATIONSHIPS_SOURCE_COLUMNS["full_time_education"]
            if column in raw.columns
        ),
        None,
    )
    fte_flag = (
        (pd.to_numeric(raw[fte_column], errors="coerce").fillna(0) == 1).to_numpy(
            dtype=bool
        )
        if fte_column is not None
        else np.zeros(len(raw), dtype=bool)
    )
    from_child_tab = (raw["_frs_table"] == "child").to_numpy(dtype=bool)
    in_fte = from_child_tab | fte_flag

    relationship_to_head, unmapped = _relationship_to_head(relhrp, hrpid)
    head_index_by_household = _head_index_by_household(raw_household)

    partner_codes = np.asarray(ONS_FAMILY_LINK_CODES["partner"])
    child_of_codes = np.asarray(ONS_FAMILY_LINK_CODES["child_of"])
    parent_of_codes = np.asarray(ONS_FAMILY_LINK_CODES["parent_of"])
    natural_child_code = int(ONS_FAMILY_LINK_CODES["child_of"][0])
    dep_rule = ONS_DEPENDENT_CHILD_RULE

    family_index = np.zeros(len(person), dtype="int64")
    role = np.full(len(person), "INDIVIDUAL", dtype=object)
    household_type: dict[int, str] = {}

    counters = {
        "head_invariant_violations": 0,
        "hrpnum_mismatches": 0,
        "person_index_gaps": 0,
        "grid_reciprocity_mismatches": 0,
        "multi_partner_persons": 0,
        "parent_tie_breaks": 0,
        "one_family_plus_individuals": 0,
    }
    dependency_counts = {
        "under_16": 0,
        "16_18_fte": 0,
        "16_18_not_fte": 0,
        "19_plus": 0,
    }

    order = np.argsort(household_ids, kind="stable")
    boundaries = np.flatnonzero(np.diff(household_ids[order])) + 1
    for rows in np.split(order, boundaries):
        household_id = int(household_ids[rows[0]])
        n = len(rows)
        p = person_index[rows]
        sort = np.argsort(p, kind="stable")
        rows = rows[sort]
        p = p[sort]
        if p.max() > FRS_HOUSEHOLD_GRID_MAX_PERSONS or not np.array_equal(
            p, np.arange(1, n + 1)
        ):
            counters["person_index_gaps"] += 1
        # M[i, j] = person i's relationship code toward person j.
        matrix = grid[rows][:, np.minimum(p, FRS_HOUSEHOLD_GRID_MAX_PERSONS) - 1]
        np.fill_diagonal(matrix, 0)
        partner = np.isin(matrix, partner_codes)
        child_of = np.isin(matrix, child_of_codes)
        parent_of = np.isin(matrix, parent_of_codes)
        counters["grid_reciprocity_mismatches"] += int(
            np.triu(partner != partner.T, 1).sum()
        ) + int((child_of != parent_of.T).sum())

        heads = relationship_to_head[rows] == RELATIONSHIP_TO_HEAD_HEAD_LABEL
        if heads.sum() != 1 or not np.array_equal(heads, frame_head[rows]):
            counters["head_invariant_violations"] += 1
        expected_head_index = head_index_by_household.get(household_id)
        if expected_head_index is not None and (
            heads.sum() != 1 or int(p[heads][0]) != expected_head_index
        ):
            counters["hrpnum_mismatches"] += 1

        partner_any = partner | partner.T
        partner_count = partner_any.sum(axis=1)
        counters["multi_partner_persons"] += int((partner_count > 1).sum())
        has_partner = partner_count > 0
        child_of_any = child_of | parent_of.T
        has_own_child = child_of_any.any(axis=0)
        child_eligible = ~has_partner & ~has_own_child

        local_family = np.zeros(n, dtype="int64")
        local_role = np.full(n, "INDIVIDUAL", dtype=object)
        next_family = 1
        for i in range(n):
            for j in range(i + 1, n):
                if partner_any[i, j] and local_family[i] == 0 and local_family[j] == 0:
                    local_family[i] = local_family[j] = next_family
                    local_role[i] = local_role[j] = "COUPLE_PARTNER"
                    next_family += 1
        for i in range(n):
            if not child_eligible[i]:
                continue
            parents = np.flatnonzero(child_of_any[i])
            if parents.size == 0:
                continue
            couple_families = sorted(
                {
                    int(local_family[q])
                    for q in parents
                    if local_role[q] == "COUPLE_PARTNER"
                }
            )
            if couple_families:
                if len(couple_families) > 1:
                    counters["parent_tie_breaks"] += 1
                local_family[i] = couple_families[0]
                local_role[i] = "CHILD"
                continue
            candidates = [
                q for q in parents if local_role[q] in ("INDIVIDUAL", "LONE_PARENT")
            ]
            if not candidates:
                continue
            ranked = sorted(
                candidates,
                key=lambda q: (0 if matrix[i, q] == natural_child_code else 1, q),
            )
            if len({int(q) for q in ranked}) > 1:
                counters["parent_tie_breaks"] += 1
            q = ranked[0]
            if local_family[q] == 0:
                local_family[q] = next_family
                local_role[q] = "LONE_PARENT"
                next_family += 1
            local_family[i] = local_family[q]
            local_role[i] = "CHILD"

        local_ages = ages[rows]
        local_fte = in_fte[rows]
        for i in np.flatnonzero(local_role == "CHILD"):
            age = int(local_ages[i])
            if age < dep_rule["unconditional_below_age"]:
                dependent = True
                dependency_counts["under_16"] += 1
            elif age <= dep_rule["full_time_education_through_age"]:
                dependent = bool(local_fte[i])
                dependency_counts["16_18_fte" if dependent else "16_18_not_fte"] += 1
            else:
                dependent = False
                dependency_counts["19_plus"] += 1
            local_role[i] = "DEPENDENT_CHILD" if dependent else "NON_DEPENDENT_CHILD"

        families = next_family - 1
        if n == 1:
            value = (
                "lone_households_over_65"
                if int(local_ages[0]) >= ONS_ONE_PERSON_AGE_SPLIT
                else "lone_households_under_65"
            )
        elif families == 0:
            value = "unrelated_adult_households"
        elif families >= 2:
            value = "multi_family_households"
        else:
            members = local_family == 1
            if members.sum() < n:
                counters["one_family_plus_individuals"] += 1
            dependent = int((local_role[members] == "DEPENDENT_CHILD").sum())
            non_dependent = int((local_role[members] == "NON_DEPENDENT_CHILD").sum())
            if (local_role[members] == "COUPLE_PARTNER").any():
                if dependent >= 3:
                    value = "couple_3_plus_children_households"
                elif dependent >= 1:
                    value = "couple_under_3_children_households"
                elif non_dependent >= 1:
                    value = "couple_non_dependent_children_only_households"
                else:
                    value = "couple_no_children_households"
            elif dependent >= 1:
                value = "lone_parent_dependent_children_households"
            else:
                value = "lone_parent_non_dependent_children_households"
        household_type[household_id] = value
        family_index[rows] = local_family
        role[rows] = local_role

    household_table_ids = household["household_id"].to_numpy(dtype="int64")
    missing_households = [
        int(identifier)
        for identifier in household_table_ids
        if int(identifier) not in household_type
    ]
    if missing_households:
        raise FRSRelationshipsError(
            f"{len(missing_households)} household(s) carry no persons on the "
            f"frame and cannot be typed (first: {missing_households[:5]})."
        )
    typed = np.asarray(
        [household_type[int(identifier)] for identifier in household_table_ids],
        dtype=object,
    )
    if household_weights is None:
        weights = np.ones(len(household_table_ids), dtype=float)
    else:
        weights = np.asarray(household_weights, dtype=float)
    type_counts = {
        value: int((typed == value).sum())
        for value in CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS
    }
    type_weighted = {
        value: float(weights[typed == value].sum())
        for value in CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS
    }
    domain_violations = int(
        (~np.isin(relationship_to_head, RELATIONSHIP_TO_HEAD_VALUES)).sum()
        + (~np.isin(role, ONS_FAMILY_ROLE_VALUES)).sum()
        + (~np.isin(typed, CHRONICLE_ONS_HOUSEHOLD_TYPE_VALUE_IDS)).sum()
    )
    family_index_violations = int(((family_index == 0) != (role == "INDIVIDUAL")).sum())
    evidence: dict[str, Any] = {
        "stage": FRS_RELATIONSHIPS_STAGE_NAME,
        "households": int(len(household_table_ids)),
        "persons": int(len(person)),
        **counters,
        "relhrp_unmapped_codes": unmapped,
        "family_index_violations": family_index_violations,
        "domain_violations": domain_violations,
        "dependency_counts": dependency_counts,
        "household_type_counts": type_counts,
        "household_type_weighted": type_weighted,
        "partition_closes": sum(type_counts.values()) == int(len(household_table_ids)),
        "full_time_education_source": fte_column,
        "reciprocity_mismatch_tolerance": int(reciprocity_mismatch_tolerance),
    }
    _refuse_defects(evidence, reciprocity_mismatch_tolerance)
    person_values = pd.DataFrame(
        {
            "relationship_to_head": relationship_to_head,
            "ons_family_role": role,
            "ons_family_index": family_index,
        },
        index=person.index,
    )
    household_values = pd.DataFrame(
        {"ons_household_type": typed}, index=household.index
    )
    return UKFRSRelationshipsResult(
        person_values=person_values,
        household_values=household_values,
        evidence=evidence,
    )


def _refuse_defects(evidence: Mapping[str, Any], tolerance: int) -> None:
    failures: list[str] = []
    for key in (
        "head_invariant_violations",
        "hrpnum_mismatches",
        "person_index_gaps",
        "multi_partner_persons",
        "family_index_violations",
        "domain_violations",
    ):
        if int(evidence[key]) != 0:
            failures.append(f"{key}={evidence[key]}")
    if int(evidence["grid_reciprocity_mismatches"]) > int(tolerance):
        failures.append(
            "grid_reciprocity_mismatches="
            f"{evidence['grid_reciprocity_mismatches']} exceeds tolerance {tolerance}"
        )
    if evidence["relhrp_unmapped_codes"]:
        failures.append(f"relhrp_unmapped_codes={evidence['relhrp_unmapped_codes']}")
    if not evidence["partition_closes"]:
        failures.append("household-type partition does not close")
    if failures:
        raise FRSRelationshipsError(
            "FRS relationship derivation refused: " + "; ".join(failures) + "."
        )


def _aligned_raw_person(
    person: pd.DataFrame, raw_adult: pd.DataFrame, raw_child: pd.DataFrame
) -> pd.DataFrame:
    adult = raw_adult.copy()
    adult["_frs_table"] = "adult"
    child = raw_child.copy()
    child["_frs_table"] = "child"
    raw = pd.concat([adult, child], ignore_index=True, sort=False)
    required = [
        "person_id",
        FRS_RELATIONSHIPS_SOURCE_COLUMNS["head"],
        *FRS_RELATIONSHIPS_SOURCE_COLUMNS["grid"],
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise FRSRelationshipsError(
            f"FRS adult/child tabs are missing relationship column(s): {missing}."
        )
    if FRS_RELATIONSHIPS_SOURCE_COLUMNS["head_flag"] not in adult.columns:
        raise FRSRelationshipsError(
            "FRS adult tab is missing the household reference person flag "
            f"{FRS_RELATIONSHIPS_SOURCE_COLUMNS['head_flag']!r}."
        )
    raw = raw.set_index(raw["person_id"].astype("int64"))
    if raw.index.duplicated().any():
        raise FRSRelationshipsError("FRS adult/child tabs repeat a person_id.")
    frame_ids = person["person_id"].astype("int64")
    absent = frame_ids[~frame_ids.isin(raw.index)]
    if len(absent):
        raise FRSRelationshipsError(
            f"{len(absent)} frame person(s) are absent from the FRS adult/child "
            f"tabs (first: {absent.head().tolist()})."
        )
    return raw.reindex(frame_ids.to_numpy()).reset_index(drop=True)


def _relationship_to_head(
    relhrp: pd.Series, hrpid: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    codes = relhrp.to_numpy(dtype=float)
    blank = np.isnan(codes) | (codes == 0)
    labels = np.full(len(codes), "", dtype=object)
    unmapped: dict[str, int] = {}
    labels[blank & hrpid] = RELATIONSHIP_TO_HEAD_HEAD_LABEL
    blank_non_head = int((blank & ~hrpid).sum())
    if blank_non_head:
        unmapped["blank_non_head"] = blank_non_head
    coded = ~blank
    for code, label in FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES.items():
        labels[coded & (codes == code)] = label
    for code in np.unique(codes[coded]):
        if int(code) not in FRS_HOUSEHOLD_GRID_RELATIONSHIP_CODES:
            unmapped[str(int(code))] = int((codes == code).sum())
    coded_head = int((coded & hrpid).sum())
    if coded_head:
        unmapped["coded_head"] = coded_head
    return labels, unmapped


def _head_index_by_household(raw_household: pd.DataFrame | None) -> dict[int, int]:
    if raw_household is None:
        return {}
    column = FRS_RELATIONSHIPS_SOURCE_COLUMNS["head_index"]
    if (
        column not in raw_household.columns
        or "household_id" not in raw_household.columns
    ):
        raise FRSRelationshipsError(
            f"FRS househol tab is missing {column!r} or household_id."
        )
    values = pd.to_numeric(raw_household[column], errors="coerce")
    ids = raw_household["household_id"].astype("int64")
    return {
        int(identifier): int(value)
        for identifier, value in zip(ids, values, strict=True)
        if pd.notna(value)
    }
