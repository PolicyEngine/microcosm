"""Truth-preserving mappings from native ACS PUMS columns to frame inputs.

Combined ACS amounts stay combined. In particular, ``SSP``, ``RETP``, and
``INTP`` are retained as adjusted ACS predictors rather than being assigned to
one PolicyEngine component or split by invented fractions. The separate fit
transfer stage learns the component leaves from the ASEC x PUF spine. ACS
``RNTP`` and ``GRNTP`` likewise remain native rent predictors: neither is
pre-subsidy rent, so mapping one to that model leaf would double-subtract a
subsequently transferred housing subsidy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.spm_role_source import (
    NATIVE_SPM_ROLE,
    SPM_ADULT_RULE,
    SPM_ROLE_RULE,
)
from microcosm.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_SPM_ROLE_PARTITION",
    "ACS_SPM_ROLE_RULE",
    "ACS_UNRESOLVED_PARENT_ID_MAPPINGS",
    "ACS_GROUP_QUARTERS_CHILD_RULE",
    "AcsGroupQuartersChildrenResult",
    "AcsNativeInputResult",
    "AcsSpmRoleResult",
    "map_acs_native_inputs",
    "with_acs_spm_independence_role",
    "without_acs_group_quarters_children",
]

_INFLATION_FACTOR_DENOMINATOR = 1_000_000.0

_HOUSEHOLD_TENURE = {
    1: "OWNED_WITH_MORTGAGE",
    2: "OWNED_OUTRIGHT",
    3: "RENTED",
    4: "NONE",  # occupied without payment of rent
}
_SPM_TENURE = {
    1: "OWNER_WITH_MORTGAGE",
    2: "OWNER_WITHOUT_MORTGAGE",
    3: "RENTER",
    4: "RENTER",  # occupied without payment: non-owner housing tenure
}

#: The ACS spine's value for the ASEC parent pointers, and why.
#:
#: ``acs_pums._with_structural_columns`` synthesizes PEPAR1/PEPAR2 from
#: RELSHIPP: for a biological/adopted/step child of the reference person
#: (25/26/27) PEPAR1 is always the reference person's line and PEPAR2 always
#: the spouse's, and every other record — grandchild, foster child, child of a
#: non-reference adult — gets nothing. That is a reference-person link, not the
#: ASEC's measured parent pointer, so exporting it under the same name would
#: conflate two constructions; and an ACS-minted ``person_id`` is renumbered by
#: the assembly offset after this mapping runs, so a resolved id would be stale
#: before it reached the pool. The ACS spine therefore reports the pointer as
#: unknown (0), which is exactly the value PolicyEngine-US reads as "fall back
#: to the count-based proxy" (microcosm#884, policyengine-us#9404).
_ACS_UNRESOLVED_PARENT_POINTER_TRANSFORMATION = (
    "0 (unknown): the ACS pointer is a synthesized reference-person link, "
    "not the ASEC measured parent pointer"
)
ACS_UNRESOLVED_PARENT_ID_MAPPINGS: tuple[tuple[str, str], ...] = (
    ("parent_1_id", "PEPAR1"),
    ("parent_2_id", "PEPAR2"),
)

_FORMULA_OWNED_AGGREGATES = frozenset(
    {
        "employment_income",
        "self_employment_income",
        "social_security",
        "taxable_pension_income",
        "interest_income",
        "dividend_income",
        "rent",
    }
)


#: RELSHIPP codes the ACS SPM independence role reads. 20 is the reference
#: person; 21/23 the opposite-/same-sex spouse of the reference person; 37/38
#: the institutional/noninstitutional group-quarters person, whom the PUMS
#: records as the only person of their own household.
_ACS_REFERENCE_PERSON = 20
_ACS_SPOUSE_OF_REFERENCE = (21, 23)
_ACS_GROUP_QUARTERS_PERSON = (37, 38)
_ACS_RELSHIPP_CODES = frozenset(range(20, 39))
_ACS_GROUP_QUARTERS_TYPEHUGQ = (2, 3)

#: The ACS arm of the SPM independence role, the ASEC rule
#: (:data:`~microcosm.build.us_runtime.spm_role_source.SPM_ROLE_RULE`) read on
#: the ACS spine's SPM partition. ``SPM_HEAD`` is each unit's head: the
#: reference person of a housing unit, or a group-quarters person, who is the
#: sole member of their unit. ``A_FAMTYP 1, A_FAMREL 1/2`` is the primary
#: family's reference person and spouse. ACS PUMS identifies no unrelated
#: subfamily, so the rule's ``A_FAMTYP 4`` branch has no ACS counterpart.
ACS_SPM_ROLE_RULE = (
    "RELSHIPP == 20 OR RELSHIPP in {21, 23} OR RELSHIPP in {37, 38}, "
    "on the one-SPM-unit-per-household ACS partition"
)

#: The SPM partition the rule is defined on. ACS PUMS carries no ``SPM_ID``,
#: so ``assign_us_unit_structure`` makes each household one SPM unit.
ACS_SPM_ROLE_PARTITION = "household (ACS PUMS carries no SPM_ID)"


@dataclass(frozen=True)
class AcsNativeInputResult:
    """Mapped ACS frame and JSON-ready native-column provenance."""

    frame: Frame
    native_inputs: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class AcsSpmRoleResult:
    """An ACS frame carrying the SPM independence role, plus its receipt."""

    frame: Frame
    provenance: Mapping[str, Any]


def map_acs_native_inputs(frame: Frame) -> AcsNativeInputResult:
    """Map measured ACS values without filling blanks or splitting totals."""

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS native input mapping requires the US schema.")
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    household = tables["household"]
    spm_unit = tables["spm_unit"]
    native: dict[str, Mapping[str, Any]] = {}

    if "AGEP" in person:
        _add_native(
            person,
            "age",
            pd.to_numeric(person["AGEP"], errors="coerce").to_numpy(dtype=np.float64),
            entity="person",
            source_columns=("AGEP",),
            transformation="identity",
            register=native,
        )
    if "SEX" in person:
        sex = pd.to_numeric(person["SEX"], errors="coerce")
        bad = sex.notna() & ~sex.isin([1, 2])
        if bad.any():
            raise ValueError(
                "ACS SEX contains unsupported code(s): "
                f"{sorted(sex.loc[bad].unique().tolist())}."
            )
        if sex.isna().any():
            raise ValueError("ACS required SEX values must not be blank.")
        _add_native(
            person,
            "is_female",
            (sex == 2).to_numpy(dtype=bool),
            entity="person",
            source_columns=("SEX",),
            transformation="SEX == 2",
            register=native,
        )
    if "RELSHIPP" in person:
        relationship = pd.to_numeric(person["RELSHIPP"], errors="coerce")
        if relationship.isna().any():
            raise ValueError("ACS required RELSHIPP values must not be blank.")
        _add_native(
            person,
            "is_household_head",
            (relationship == 20).to_numpy(dtype=bool),
            entity="person",
            source_columns=("RELSHIPP",),
            transformation="RELSHIPP == 20",
            register=native,
        )

    _map_usual_hours(person, register=native)

    _map_adjusted_person_amount(
        person,
        source="WAGP",
        output="employment_income_before_lsr",
        register=native,
    )
    _map_adjusted_person_amount(
        person,
        source="SEMP",
        output="self_employment_income_before_lsr",
        register=native,
    )
    _map_adjusted_person_amount(
        person,
        source="SSIP",
        output="ssi_reported",
        register=native,
    )
    _map_reported_ssi_receipt(person, register=native)
    _map_adjusted_person_amount(
        person,
        source="SSP",
        output="acs_social_security_income",
        register=native,
    )
    _map_adjusted_person_amount(
        person,
        source="RETP",
        output="acs_retirement_income",
        register=native,
    )
    _map_adjusted_person_amount(
        person,
        source="INTP",
        output="acs_interest_dividend_rental_income",
        register=native,
    )

    for output, pointer_column in ACS_UNRESOLVED_PARENT_ID_MAPPINGS:
        if pointer_column in person:
            _add_native(
                person,
                output,
                np.zeros(len(person), dtype=np.int64),
                entity="person",
                source_columns=(pointer_column,),
                transformation=_ACS_UNRESOLVED_PARENT_POINTER_TRANSFORMATION,
                register=native,
            )

    _map_tenure(person, household, spm_unit, register=native)
    _map_housing_amounts(person, household, register=native)

    forbidden = sorted(
        _FORMULA_OWNED_AGGREGATES.intersection(
            column for table in tables.values() for column in table.columns
        )
    )
    if forbidden:
        raise ValueError(
            "ACS native mappings must not persist formula-owned aggregate "
            f"column(s): {forbidden}."
        )

    mapped = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return AcsNativeInputResult(mapped, native)


def with_acs_spm_independence_role(frame: Frame) -> AcsSpmRoleResult:
    """Derive ``is_spm_independent_minor_role`` for an ACS spine, or refuse.

    The engine declares the role a dataset source input
    (``policyengine_us.spm.DATASET_SOURCE_INPUTS``): a producer must deliver
    it and may not default it. ASEC rows get it from the Census SPM fields
    (:mod:`~microcosm.build.us_runtime.spm_independence_role`). The ACS has no
    SPM fields, but it has no SPM partition either: every ACS household is one
    SPM unit (:data:`ACS_SPM_ROLE_PARTITION`). On that partition the ASEC
    rule's unit head is the household's reference person, or the sole person
    of a group-quarters record, and its family branch is the reference
    person's spouse, so the rule is read off ``RELSHIPP``
    (:data:`ACS_SPM_ROLE_RULE`). On the pinned ASEC files, in households that
    are one SPM unit, this reading reproduces Census's own SPM adult and child
    counts in 215,342 of 215,343 units across the four pinned vintages
    (income years 2022 to 2025;
    ``experiments/acs-spm-role-asec-validation.md``).

    This is deliberately not part of :func:`map_acs_native_inputs`. That
    mapping also feeds the stacked pool, whose ASEC arm carries no role and
    whose ACS-row rule is an open decision (``docs/us-spm-role-stage.md``
    §7 Q1). A caller applies this only where the partner spine delivers the
    role too, so the pooled column is complete on both spines.

    Refuses rather than deriving (a refusal about units names up to five
    example unit ids, the others give counts) when: the frame already
    carries the role; ``RELSHIPP`` is missing, blank or outside the ACS codes
    20 to 38; ``age`` is missing or not finite (no age range is checked);
    the SPM partition is not the household partition; a unit has other than
    exactly one head, more than one spouse, or a group-quarters person beside
    anyone else; ``TYPEHUGQ`` (when present) disagrees with ``RELSHIPP`` about
    group quarters; or a housing-unit SPM unit is left with no classified
    adult. A group-quarters person under 15 is a one-person unit with no
    classified adult under any role; the source places group quarters outside
    the ACS SPM universe (:mod:`~microcosm.build.us_runtime.spm_universe_source`),
    so those units are counted in the receipt, not refused.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS SPM independence role requires the US schema.")
    person = frame.table("person")
    if NATIVE_SPM_ROLE in person:
        raise ValueError(
            f"ACS SPM independence role refuses to overwrite existing column "
            f"{NATIVE_SPM_ROLE!r}."
        )
    missing = [
        column
        for column in ("RELSHIPP", "age", "person_household_id", "person_spm_unit_id")
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"ACS SPM independence role requires person column(s): {missing}."
        )

    relationship = pd.to_numeric(person["RELSHIPP"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    off_domain = np.isnan(relationship) | ~np.isin(
        relationship, sorted(_ACS_RELSHIPP_CODES)
    )
    if off_domain.any():
        raise ValueError(
            "ACS SPM independence role requires every RELSHIPP to be an ACS "
            f"relationship code in [20, 38]; {int(off_domain.sum())} row(s) are "
            "blank or off-domain."
        )
    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(age).all():
        raise ValueError(
            "ACS SPM independence role requires an observed age for every person."
        )

    household_id = person["person_household_id"].to_numpy()
    unit_id = person["person_spm_unit_id"].to_numpy()
    membership = pd.DataFrame({"unit": unit_id, "household": household_id})
    households_per_unit = membership.groupby("unit", sort=False)["household"].nunique()
    units_per_household = membership.groupby("household", sort=False)["unit"].nunique()
    if (households_per_unit != 1).any() or (units_per_household != 1).any():
        raise ValueError(
            "ACS SPM independence role is defined only on the household SPM "
            f"partition ({ACS_SPM_ROLE_PARTITION}); "
            f"{int((households_per_unit != 1).sum())} SPM unit(s) span households "
            f"and {int((units_per_household != 1).sum())} household(s) hold "
            "several SPM units. A reconstructed ACS SPM partition must supply its "
            "own role."
        )

    reference = relationship == _ACS_REFERENCE_PERSON
    spouse = np.isin(relationship, _ACS_SPOUSE_OF_REFERENCE)
    group_quarters = np.isin(relationship, _ACS_GROUP_QUARTERS_PERSON)
    head = reference | group_quarters
    units = pd.DataFrame(
        {
            "unit": unit_id,
            "persons": 1,
            "heads": head.astype(np.int64),
            "spouses": spouse.astype(np.int64),
            "group_quarters": group_quarters.astype(np.int64),
        }
    ).groupby("unit", sort=False)[["persons", "heads", "spouses", "group_quarters"]]
    counts = units.sum()
    _refuse_units(
        counts.index[counts["heads"] != 1],
        "not exactly one head (a RELSHIPP=20 reference person or one RELSHIPP "
        "37/38 group-quarters person)",
    )
    _refuse_units(
        counts.index[counts["spouses"] > 1],
        "more than one RELSHIPP 21/23 spouse of the reference person",
    )
    _refuse_units(
        counts.index[(counts["group_quarters"] > 0) & (counts["persons"] != 1)],
        "a RELSHIPP 37/38 group-quarters person shares the unit",
    )

    household = frame.table("household")
    if "TYPEHUGQ" in household:
        kind = pd.Series(
            pd.to_numeric(household["TYPEHUGQ"], errors="coerce").to_numpy(),
            index=household["household_id"].to_numpy(),
        )
        person_kind = pd.Series(household_id).map(kind).to_numpy(dtype=np.float64)
        if (np.isnan(person_kind) | ~np.isin(person_kind, (1, 2, 3))).any():
            raise ValueError(
                "ACS SPM independence role requires TYPEHUGQ in {1, 2, 3} for "
                "every person's household."
            )
        disagree = np.isin(person_kind, _ACS_GROUP_QUARTERS_TYPEHUGQ) != group_quarters
        if disagree.any():
            raise ValueError(
                "ACS SPM independence role found "
                f"{int(disagree.sum())} person(s) whose RELSHIPP and household "
                "TYPEHUGQ disagree about group quarters."
            )

    role = head | spouse
    adult = (age >= 18) | ((age >= 15) & role)
    unit_frame = pd.DataFrame(
        {
            "unit": unit_id,
            "adult": adult,
            "member_18_plus": age >= 18,
            "group_quarters": group_quarters,
        }
    ).groupby("unit", sort=False)
    unit_flags = unit_frame.agg(
        adult=("adult", "any"),
        member_18_plus=("member_18_plus", "any"),
        group_quarters=("group_quarters", "any"),
    )
    _refuse_units(
        unit_flags.index[~unit_flags["adult"] & ~unit_flags["group_quarters"]],
        "a housing unit left with no classified adult "
        f"({SPM_ADULT_RULE}); a zero-adult housing unit is a data defect, "
        "never an inferred adult",
    )

    minor = (age >= 15) & (age < 18)
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    minor_weight = float(weights[minor].sum())
    provenance = {
        "column": NATIVE_SPM_ROLE,
        "entity": "person",
        "source_columns": ["RELSHIPP"],
        "rule": ACS_SPM_ROLE_RULE,
        "partition": ACS_SPM_ROLE_PARTITION,
        "asec_rule": SPM_ROLE_RULE,
        "adult_rule": SPM_ADULT_RULE,
        "provenance": "acs_2024_1yr_native",
        "persons": int(len(person)),
        "spm_units": int(len(counts)),
        "role_true_persons": int(role.sum()),
        "role_branches": {
            "reference_person": int(reference.sum()),
            "spouse_of_reference_person": int(spouse.sum()),
            "group_quarters_sole_person": int(group_quarters.sum()),
        },
        "persons_aged_15_to_17": int(minor.sum()),
        "independent_minor_persons": int((role & minor).sum()),
        "housing_units_classified_only_by_role": int(
            (
                unit_flags["adult"]
                & ~unit_flags["member_18_plus"]
                & ~unit_flags["group_quarters"]
            ).sum()
        ),
        "group_quarters_units_without_classified_adult": int(
            (~unit_flags["adult"] & unit_flags["group_quarters"]).sum()
        ),
        "weighted_role_share": (
            float(weights[role].sum()) / total_weight if total_weight else 0.0
        ),
        "weighted_minor_role_share": (
            float(weights[role & minor].sum()) / minor_weight if minor_weight else 0.0
        ),
    }

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"][NATIVE_SPM_ROLE] = role
    derived = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return AcsSpmRoleResult(derived, provenance)


#: The ACS records the interim drop removes (microcosm#1263, decision d1484).
ACS_GROUP_QUARTERS_CHILD_RULE = (
    "TYPEHUGQ in {2, 3} (a group-quarters record, the record's only person) "
    "AND age < 15"
)
#: Census SEHSD-WP2020-09 pp. 6-7 limits the ACS SPM sample to housing units;
#: ``spm_universe_source`` declares ACS group quarters OUTSIDE on that basis.
_GROUP_QUARTERS_CHILDREN_AUTHORITY = (
    "Census SEHSD-WP2020-09 pp. 6-7: the ACS SPM universe is people in housing "
    "units; ACS group quarters are outside it "
    "(microcosm.build.us_runtime.spm_universe_source)"
)


@dataclass(frozen=True)
class AcsGroupQuartersChildrenResult:
    """An ACS frame without its group-quarters children, plus the receipt."""

    frame: Frame
    provenance: Mapping[str, Any]


def without_acs_group_quarters_children(
    frame: Frame, *, age_column: str = "age"
) -> AcsGroupQuartersChildrenResult:
    """Remove the ACS group-quarters records whose only person is under 15.

    An interim (microcosm#1263, decision d1484), not a source rule. No role
    can give such a person a classified adult under spm-calculator's rule
    (``age >= 18 OR (age >= 15 AND role)``), and an engine that does not scope
    SPM by measurement universe refuses every SPM measurement on a population
    containing one. The source places ACS group quarters outside the SPM
    universe, so the end state is to keep these records and exclude them from
    SPM measurement alone. Until the engine can, the staging builder removes
    them from the population. That also removes them from every fiscal total,
    which the receipt and the staging's reviewed limitation disclose. The
    builder stops calling this once the certified engine scopes SPM by
    universe.

    Refuses when ``TYPEHUGQ``, the age column or a membership column is
    missing or unreadable, or when a group-quarters record holds other than
    one person. Housing units are never touched, whatever their ages.
    ``age_column`` names the age: ``AGEP`` on the raw PUMS frame, where the
    staging applies this before any receipt is computed, or ``age`` on a
    mapped frame.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS group-quarters children require the US schema.")
    person = frame.table("person")
    household = frame.table("household")
    for table, columns, label in (
        (person, (age_column, "person_household_id"), "person"),
        (household, ("household_id", "TYPEHUGQ"), "household"),
    ):
        missing = [column for column in columns if column not in table]
        if missing:
            raise ValueError(
                f"ACS group-quarters children require {label} column(s): {missing}."
            )
    kind = pd.Series(
        pd.to_numeric(household["TYPEHUGQ"], errors="coerce").to_numpy(),
        index=household["household_id"].to_numpy(),
    )
    if not kind.isin((1, 2, 3)).all():
        raise ValueError(
            "ACS group-quarters children require TYPEHUGQ in {1, 2, 3} for every "
            "household."
        )
    household_id = person["person_household_id"].to_numpy()
    person_kind = pd.Series(household_id).map(kind).to_numpy(dtype=np.float64)
    if np.isnan(person_kind).any():
        raise ValueError(
            "ACS group-quarters children found persons in no declared household."
        )
    age = pd.to_numeric(person[age_column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(age).all():
        raise ValueError(
            "ACS group-quarters children require an observed age for every person."
        )
    group_quarters = np.isin(person_kind, _ACS_GROUP_QUARTERS_TYPEHUGQ)
    persons_per_household = pd.Series(household_id).map(
        pd.Series(household_id).value_counts()
    )
    crowded = group_quarters & (persons_per_household.to_numpy() != 1)
    if crowded.any():
        raise ValueError(
            "ACS group-quarters children found "
            f"{int(crowded.sum())} person(s) in group-quarters records of more "
            "than one person; a group-quarters record is its only person's."
        )
    child = group_quarters & (age < 15)
    weights = frame.weights_for("household")
    household_weight = pd.Series(
        np.asarray(weights.values, dtype=np.float64),
        index=household["household_id"].to_numpy(),
    )
    dropped_weight = float(household_weight.loc[np.unique(household_id[child])].sum())
    total_weight = float(household_weight.sum())
    provenance = {
        "rule": ACS_GROUP_QUARTERS_CHILD_RULE,
        "authority": _GROUP_QUARTERS_CHILDREN_AUTHORITY,
        "decision": "microcosm#1263 (d1484)",
        "status": "interim",
        "flip_condition": (
            "Keep these records, declared OUTSIDE the SPM universe, in the first "
            "release certified against an engine that scopes SPM measurement by "
            "spm_unit_spm_universe_status "
            "(microcosm.build.us_runtime.spm_measurement_gate."
            "engine_scopes_spm_universe)."
        ),
        "households_before": int(len(household)),
        "persons_before": int(len(person)),
        "group_quarters_households": int(group_quarters.sum()),
        "households_dropped": int(child.sum()),
        "persons_dropped": int(child.sum()),
        "household_weight_dropped": dropped_weight,
        "household_weight_share_dropped": (
            dropped_weight / total_weight if total_weight else 0.0
        ),
    }
    if not child.any():
        return AcsGroupQuartersChildrenResult(frame, provenance)
    return AcsGroupQuartersChildrenResult(frame.select(~child), provenance)


def _refuse_units(offending: pd.Index, problem: str) -> None:
    if len(offending):
        examples = [
            value.item() if isinstance(value, np.generic) else value
            for value in offending[:5]
        ]
        raise ValueError(
            f"ACS SPM independence role refuses {len(offending)} SPM unit(s) "
            f"with {problem}. Example unit id(s): {examples}."
        )


def _map_usual_hours(
    person: pd.DataFrame, *, register: dict[str, Mapping[str, Any]]
) -> None:
    """Use annual usual hours, retaining unresolved blanks for transfer.

    The 2024 FTP dictionary (p. 45) uses blank WKHP for NIU, unlike the
    Census API's zero code. WKL can confirm past-year nonwork; age below 16
    only establishes survey-universe absence, not zero work. Current
    employment status cannot zero annual hours.
    """
    if "WKHP" not in person:
        return
    hours = _nullable_source_codes(person["WKHP"], minimum=1, maximum=99)
    worked = (
        _nullable_source_codes(person["WKL"], minimum=1, maximum=3)
        if "WKL" in person
        else pd.Series(np.nan, index=person.index)
    )
    age = (
        pd.to_numeric(person["AGEP"], errors="coerce")
        if "AGEP" in person
        else pd.Series(np.nan, index=person.index)
    )
    under_sixteen = age.ge(0) & age.lt(16)
    not_worked = worked.isin([2, 3])
    if (hours.notna() & (under_sixteen | not_worked)).any() or (
        under_sixteen & worked.notna()
    ).any():
        raise ValueError("ACS WKHP/WKL contradict their age or past-year universe.")
    structural_zero = hours.isna() & not_worked
    values = hours.mask(structural_zero, 0.0).to_numpy(dtype=float, na_value=np.nan)
    allocation = (
        _nullable_source_codes(person["FWKHP"], minimum=0, maximum=1)
        if "FWKHP" in person
        else pd.Series(np.nan, index=person.index)
    )
    output = "weekly_hours_worked_before_lsr"
    _add_native(
        person,
        output,
        values,
        entity="person",
        source_columns=tuple(
            column for column in ("WKHP", "AGEP", "WKL", "FWKHP") if column in person
        ),
        transformation="WKHP; zero only for source-confirmed past-year nonwork",
        register=register,
    )
    register[output] = {
        **register[output],
        "source_value_rows": int(hours.notna().sum()),
        "structural_zero_rows": int(structural_zero.sum()),
        "source_universe_unavailable_rows": int(under_sixteen.sum()),
        "allocated_value_rows": int((hours.notna() & allocation.eq(1)).sum()),
        "allocation_unknown_value_rows": int((hours.notna() & allocation.isna()).sum()),
        "reference": (
            "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
            "data_dict/PUMS_Data_Dictionary_2024.pdf#page=45"
        ),
        "allocation_reference_page": 130,
    }


def _nullable_source_codes(
    source: pd.Series, *, minimum: int, maximum: int
) -> pd.Series:
    blank = source.isna() | source.astype("string").str.strip().eq("").fillna(False)
    values = pd.to_numeric(source.where(~blank), errors="coerce")
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    valid = (
        np.isfinite(numeric)
        & (numeric == np.floor(numeric))
        & (numeric >= minimum)
        & (numeric <= maximum)
    )
    if (~blank.to_numpy(dtype=bool) & ~valid).any():
        raise ValueError(
            f"ACS {source.name} requires blank or integer codes "
            f"within [{minimum}, {maximum}]."
        )
    return values


#: ACS asks income questions, SSI included, of people aged 15 and older.
_ACS_INCOME_UNIVERSE_MIN_AGE = 15


def _map_reported_ssi_receipt(
    person: pd.DataFrame,
    *,
    register: dict[str, Mapping[str, Any]],
) -> None:
    """Map ``receives_ssi`` from the respondent's own past-12-month ACS SSIP.

    A positive ``SSIP`` is reported SSI receipt, the ACS counterpart of the CPS
    rule in ``cps_carried.reported_ssi_receipt``. ``SSIP`` is blank only below
    the ACS income universe (age 15), where it means no reported receipt; the
    register counts those rows as ``source_universe_unavailable_rows``. A
    blank at 15 or older, or a non-numeric value, is refused rather than read
    as no receipt. ACS records no reason for receipt, so a payee receiving SSI
    on behalf of a child cannot be told apart here.
    """

    if "SSIP" not in person:
        return
    raw = person["SSIP"]
    amount = pd.to_numeric(raw, errors="coerce")
    if (raw.notna() & amount.isna()).any():
        raise ValueError("ACS SSIP contains non-numeric values.")
    blank = amount.isna()
    if blank.any():
        if "AGEP" not in person:
            raise ValueError("ACS SSIP blanks require AGEP to check their universe.")
        age = pd.to_numeric(person["AGEP"], errors="coerce")
        if (blank & ~(age < _ACS_INCOME_UNIVERSE_MIN_AGE)).any():
            raise ValueError(
                "ACS SSIP may be blank only below the age-15 income universe."
            )
    _add_native(
        person,
        "receives_ssi",
        (amount.fillna(0.0) > 0.0).to_numpy(dtype=bool),
        entity="person",
        source_columns=("SSIP",),
        transformation="SSIP > 0; blank below age 15 is no reported receipt",
        register=register,
    )
    register["receives_ssi"] = {
        **register["receives_ssi"],
        "source_universe_unavailable_rows": int(blank.sum()),
    }


def _map_adjusted_person_amount(
    person: pd.DataFrame,
    *,
    source: str,
    output: str,
    register: dict[str, Mapping[str, Any]],
) -> None:
    if source not in person:
        return
    values = _adjusted_dollars(person[source], person, factor="ADJINC")
    _add_native(
        person,
        output,
        values,
        entity="person",
        source_columns=(source, "ADJINC"),
        transformation=f"{source} * ADJINC / 1_000_000",
        register=register,
    )


def _map_tenure(
    person: pd.DataFrame,
    household: pd.DataFrame,
    spm_unit: pd.DataFrame,
    *,
    register: dict[str, Mapping[str, Any]],
) -> None:
    if "TEN" not in household:
        return
    codes = pd.to_numeric(household["TEN"], errors="coerce")
    household_values = codes.map(_HOUSEHOLD_TENURE).astype(object)
    spm_values_at_household = codes.map(_SPM_TENURE).astype(object)
    unknown = codes.notna() & ~codes.isin(_HOUSEHOLD_TENURE)
    if unknown.any():
        raise ValueError(
            "ACS TEN contains unsupported code(s): "
            f"{sorted(codes.loc[unknown].unique().tolist())}."
        )
    _add_native(
        household,
        "tenure_type",
        household_values.to_numpy(),
        entity="household",
        source_columns=("TEN",),
        transformation="ACS TEN enum recode",
        register=register,
    )

    household_ids = household["household_id"].to_numpy()
    spm_by_household = pd.Series(
        spm_values_at_household.to_numpy(), index=household_ids
    )
    person_values = person["person_household_id"].map(spm_by_household)
    by_spm = pd.DataFrame(
        {
            "spm_unit_id": person["person_spm_unit_id"].to_numpy(),
            "value": person_values.to_numpy(),
        }
    )
    conflicting = by_spm.dropna().groupby("spm_unit_id")["value"].nunique() > 1
    if conflicting.any():
        raise ValueError("ACS SPM unit spans conflicting household tenure values.")
    spm_lookup = by_spm.drop_duplicates("spm_unit_id").set_index("spm_unit_id")["value"]
    _add_native(
        spm_unit,
        "spm_unit_tenure_type",
        spm_unit["spm_unit_id"].map(spm_lookup).to_numpy(),
        entity="spm_unit",
        source_columns=("TEN",),
        transformation="ACS TEN enum recode through SPM membership",
        register=register,
    )


def _map_housing_amounts(
    person: pd.DataFrame,
    household: pd.DataFrame,
    *,
    register: dict[str, Mapping[str, Any]],
) -> None:
    adjusted: dict[str, np.ndarray] = {}
    for source, output in (
        ("RNTP", "acs_monthly_contract_rent"),
        ("GRNTP", "acs_monthly_gross_rent"),
        ("TAXAMT", "acs_annual_property_tax"),
    ):
        if source not in household:
            continue
        values = _adjusted_dollars(household[source], household, factor="ADJHSG")
        adjusted[source] = values
        _add_native(
            household,
            output,
            values,
            entity="household",
            source_columns=(source, "ADJHSG"),
            transformation=f"{source} * ADJHSG / 1_000_000",
            register=register,
        )

    if "is_household_head" not in person:
        return
    household_id = household["household_id"]
    if "TAXAMT" in adjusted:
        placed = _head_place(person, household_id, adjusted["TAXAMT"])
        _add_native(
            person,
            "real_estate_taxes",
            placed,
            entity="person",
            source_columns=("TAXAMT", "ADJHSG", "RELSHIPP"),
            transformation=("TAXAMT * ADJHSG / 1_000_000; reference-person carry"),
            register=register,
        )


def _head_place(
    person: pd.DataFrame,
    household_ids: pd.Series,
    household_values: np.ndarray,
) -> np.ndarray:
    lookup = pd.Series(household_values, index=household_ids.to_numpy())
    broadcast = person["person_household_id"].map(lookup).to_numpy(dtype=np.float64)
    is_head = person["is_household_head"].to_numpy(dtype=bool)
    # A measured household amount is stored once, on its reference person.
    # A missing source amount stays missing for every member; it is not a zero.
    return np.where(is_head, broadcast, np.where(np.isnan(broadcast), np.nan, 0.0))


def _adjusted_dollars(
    values: pd.Series,
    table: pd.DataFrame,
    *,
    factor: str,
) -> np.ndarray:
    if factor not in table:
        raise ValueError(
            f"ACS dollar source {values.name!r} requires adjustment column {factor!r}."
        )
    amount = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    adjustment = pd.to_numeric(table[factor], errors="coerce").to_numpy(
        dtype=np.float64
    )
    observed = ~np.isnan(amount)
    invalid_amount = observed & ~np.isfinite(amount)
    if invalid_amount.any():
        raise ValueError(f"ACS dollar source {values.name!r} must be finite.")
    invalid = observed & (~np.isfinite(adjustment) | (adjustment <= 0))
    if invalid.any():
        raise ValueError(
            f"ACS {factor} must be finite and positive wherever "
            f"{values.name!r} is observed."
        )
    return amount * (adjustment / _INFLATION_FACTOR_DENOMINATOR)


def _add_native(
    table: pd.DataFrame,
    output: str,
    values: np.ndarray,
    *,
    entity: str,
    source_columns: tuple[str, ...],
    transformation: str,
    register: dict[str, Mapping[str, Any]],
) -> None:
    if output in table:
        raise ValueError(
            f"ACS native mapping refuses to overwrite existing column {output!r}."
        )
    table[output] = values
    missing_rows = int(pd.isna(values).sum())
    register[output] = {
        "entity": entity,
        "source_columns": list(source_columns),
        "transformation": transformation,
        "provenance": "acs_2024_1yr_native",
        "observed_rows": int(len(values) - missing_rows),
        "missing_rows": missing_rows,
    }
