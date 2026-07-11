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

from populace.frame import US_SCHEMA, Frame

__all__ = ["AcsNativeInputResult", "map_acs_native_inputs"]

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


@dataclass(frozen=True)
class AcsNativeInputResult:
    """Mapped ACS frame and JSON-ready native-column provenance."""

    frame: Frame
    native_inputs: Mapping[str, Mapping[str, Any]]


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
    )
    return AcsNativeInputResult(mapped, native)


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
    register[output] = {
        "entity": entity,
        "source_columns": list(source_columns),
        "transformation": transformation,
        "provenance": "acs_2024_1yr_native",
    }
