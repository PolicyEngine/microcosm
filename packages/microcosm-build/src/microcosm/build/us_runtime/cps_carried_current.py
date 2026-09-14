"""Corrected CPS monetary leaves derived from a typed selected-money subset.

Amounts come only from the selected current-money artifact; routing codes come
only from declared raw code columns. A raw dollar column is never an alternate
input here: the derivation refuses one outright, so a corrected leaf cannot
silently fall back to the nominal source value.

The split fractions, the RESNSS reason routing with its age-62 fallback, the
OI_OFF 20/12 partition, the DST code-4 IRA rule and the SPM childcare grain are
the legacy modelling choices, reused unchanged and imported from
``cps_carried``/``alimony`` rather than redeclared. Only the amounts they
consume change: nominal source dollars become the 2024-basis restated amounts
the money recipe already produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alimony import US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS, derive_us_alimony_from_asec
from .asec_current_money import FIELDS
from .asec_current_money_selection import SelectedCurrentMoney
from .cps_carried import (
    LONG_TERM_CAPITAL_GAIN_FRACTION,
    QUALIFIED_DIVIDEND_FRACTION,
    TAXABLE_INTEREST_FRACTION,
    TAXABLE_PENSION_FRACTION,
)

CPS_CARRIED_CURRENT_ROUTING_COLUMNS: tuple[str, ...] = (
    "A_AGE",
    "DST_SC1",
    "DST_SC1_YNG",
    "DST_SC2",
    "DST_SC2_YNG",
    "OI_OFF",
    "RESNSS1",
    "RESNSS2",
)
CPS_CARRIED_CURRENT_MONEY_FIELDS: tuple[str, ...] = (
    "ANN_VAL",
    "CAP_VAL",
    "DIV_VAL",
    "DST_VAL1",
    "DST_VAL1_YNG",
    "DST_VAL2",
    "DST_VAL2_YNG",
    "FRSE_VAL",
    "INT_VAL",
    "OI_VAL",
    "PHIP_VAL",
    "PMED_VAL",
    "PNSN_VAL",
    "POTC_VAL",
    "RNT_VAL",
    "SEMP_VAL",
    "SPM_CHILDCAREXPNS",
    "SS_VAL",
    "UC_VAL",
    "WSAL_VAL",
)
CPS_CARRIED_CURRENT_PERSON_LEAVES: tuple[str, ...] = (
    "age",
    "alimony_income",
    "employment_income_before_lsr",
    "farm_operations_income",
    "health_insurance_premiums_without_medicare_part_b",
    "long_term_capital_gains_before_response",
    "miscellaneous_income",
    "non_qualified_dividend_income",
    "other_medical_expenses",
    "over_the_counter_health_expenses",
    "qualified_dividend_income",
    "rental_income",
    "self_employment_income_before_lsr",
    "short_term_capital_gains",
    "social_security_dependents",
    "social_security_disability",
    "social_security_retirement",
    "social_security_survivors",
    "strike_benefits",
    "tax_exempt_private_pension_income",
    "taxable_interest_income",
    "taxable_ira_distributions",
    "taxable_private_pension_income",
    "unemployment_compensation",
)
CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES: tuple[str, ...] = (
    "spm_unit_pre_subsidy_childcare_expenses",
)
CPS_CARRIED_CURRENT_CONTRACT_SCHEMA = "microcosm.us.cps-carried-current-leaves.v1"
_IRA_DISTRIBUTION_CODE = 4
_SOCIAL_SECURITY_RETIREMENT_AGE_FALLBACK = 62


class CpsCarriedCurrentRefusalError(ValueError):
    """Sanitized refusal; never carries row identifiers or amounts."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CpsCarriedCurrentRefusalError(reason)


def _routing(routing, rows: int) -> dict[str, np.ndarray]:
    _require(type(routing) is pd.DataFrame, "ROUTING_TABLE")
    _require(len(routing) == rows, "ROUTING_ROW_ALIGNMENT")
    columns = tuple(routing.columns)
    _require(len(set(columns)) == len(columns), "ROUTING_ROSTER")
    # A source dollar column can never reach this derivation, even by accident.
    _require(
        not any(name in FIELDS or name.endswith("_VAL") for name in columns),
        "RAW_MONEY_INPUT_REFUSED",
    )
    _require(
        tuple(sorted(columns)) == CPS_CARRIED_CURRENT_ROUTING_COLUMNS, "ROUTING_ROSTER"
    )
    values = {}
    for name in CPS_CARRIED_CURRENT_ROUTING_COLUMNS:
        series = routing[name]
        _require(pd.api.types.is_integer_dtype(series.dtype), "ROUTING_DTYPE")
        _require(not bool(series.isna().any()), "ROUTING_MISSING")
        values[name] = series.to_numpy(dtype="int64", copy=True)
    return values


def _amounts(selected: SelectedCurrentMoney, rows: int) -> dict[str, np.ndarray]:
    values = {}
    for name in CPS_CARRIED_CURRENT_MONEY_FIELDS:
        _require(selected.entity_of(name) == "person", "MONEY_FIELD_ENTITY")
        amounts = selected.amounts(name)
        _require(len(amounts) == rows, "MONEY_ROW_ALIGNMENT")
        _require(bool(np.isfinite(amounts).all()), "MONEY_NONFINITE")
        values[name] = amounts
    return values


def _social_security(amount: np.ndarray, routing) -> dict[str, np.ndarray]:
    """Reuse the legacy RESNSS reason routing and its age-62 fallback exactly."""
    reason_1 = routing["RESNSS1"]
    reason_2 = routing["RESNSS2"]
    age = routing["A_AGE"]
    is_retirement = (reason_1 == 1) | (reason_2 == 1)
    is_disability = (reason_1 == 2) | (reason_2 == 2)
    is_survivor = np.isin(reason_1, [3, 5]) | np.isin(reason_2, [3, 5])
    is_dependent = np.isin(reason_1, [4, 6, 7]) | np.isin(reason_2, [4, 6, 7])
    unclassified = (
        (amount > 0) & ~is_retirement & ~is_disability & ~is_survivor & ~is_dependent
    )
    threshold = _SOCIAL_SECURITY_RETIREMENT_AGE_FALLBACK
    return {
        "social_security_retirement": np.where(
            is_retirement | (unclassified & (age >= threshold)), amount, 0.0
        ),
        "social_security_disability": np.where(
            (is_disability & ~is_retirement) | (unclassified & (age < threshold)),
            amount,
            0.0,
        ),
        "social_security_survivors": np.where(
            is_survivor & ~is_retirement & ~is_disability, amount, 0.0
        ),
        "social_security_dependents": np.where(
            is_dependent & ~is_retirement & ~is_disability & ~is_survivor, amount, 0.0
        ),
    }


def _ira_distributions(amounts, routing, rows: int) -> np.ndarray:
    values = np.zeros(rows, dtype=np.float64)
    for suffix in ("1", "2", "1_YNG", "2_YNG"):
        code = routing[f"DST_SC{suffix}"]
        values += np.where(
            code == _IRA_DISTRIBUTION_CODE, amounts[f"DST_VAL{suffix}"], 0.0
        )
    return values


def _other_income(amount: np.ndarray, routing) -> dict[str, np.ndarray]:
    """Delegate the OI_OFF 20/12 partition to the registered legacy helper."""
    table = pd.DataFrame({"OI_VAL": amount, "OI_OFF": routing["OI_OFF"]})
    split = derive_us_alimony_from_asec(table)
    return {
        name: split[name].to_numpy(dtype="float64", copy=True)
        for name in US_ASEC_OTHER_INCOME_OUTPUT_COLUMNS
    }


def _childcare(amount: np.ndarray, membership: np.ndarray, ids: np.ndarray):
    """Carry the SPM-unit childcare value; members must already agree exactly."""
    _require(
        type(membership) is np.ndarray
        and membership.dtype == np.dtype("int64")
        and membership.shape == amount.shape,
        "SPM_MEMBERSHIP",
    )
    _require(
        type(ids) is np.ndarray and ids.dtype == np.dtype("int64") and ids.ndim == 1,
        "SPM_IDS",
    )
    _require(len(ids) == len(set(ids.tolist())), "SPM_IDS")
    _require(np.array_equal(np.sort(ids), ids), "SPM_IDS")
    _require(np.array_equal(np.unique(membership), ids), "SPM_MEMBERSHIP_COVERAGE")
    positions = np.searchsorted(ids, membership)
    values = np.zeros(len(ids), dtype=np.float64)
    seen = np.zeros(len(ids), dtype=bool)
    for position, value in zip(positions.tolist(), amount.tolist(), strict=True):
        if seen[position]:
            # The source repeats one SPM value on every member; a disagreement
            # is a source contract failure, never something to reduce away.
            _require(values[position] == value, "INCONSISTENT_SPM_AMOUNT")
            continue
        values[position] = value
        seen[position] = True
    return values


@dataclass(frozen=True)
class CpsCarriedCurrentLeaves:
    """Corrected person and SPM-unit leaves, aligned to the selected rows."""

    person: Mapping[str, np.ndarray]
    spm_unit: Mapping[str, np.ndarray]


CPS_CURRENT_PREDICTOR_MONEY_FIELDS = (
    "WSAL_VAL",
    "SEMP_VAL",
    "INT_VAL",
    "DIV_VAL",
    "CAP_VAL",
)
CPS_CURRENT_PREDICTOR_PERSON_LEAVES = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
)


def derive_cps_current_predictor_leaves(amounts):
    """Pure five-field split, sharing the maintained current-money judgments.

    Callers qualify the actual money owner or modeled target artifacts. This
    numerical function grants no source authority and never fills an unknown.
    Inputs are aligned physical float64 arrays, already in the declared dollar
    basis. INT/DIV/CAP fitted for ACS remain modeled, including their splits.
    """
    _require(isinstance(amounts, Mapping), "PREDICTOR_MONEY_MAPPING")
    _require(
        set(amounts) == set(CPS_CURRENT_PREDICTOR_MONEY_FIELDS),
        "PREDICTOR_MONEY_ROSTER",
    )
    lengths = set()
    for value in amounts.values():
        _require(
            type(value) is np.ndarray
            and value.ndim == 1
            and value.dtype == np.dtype("float64"),
            "PREDICTOR_MONEY_TYPE",
        )
        _require(bool(np.isfinite(value).all()), "PREDICTOR_MONEY_UNKNOWN")
        lengths.add(len(value))
    _require(len(lengths) == 1 and next(iter(lengths)) > 0, "PREDICTOR_MONEY_AXIS")
    dividends, gains = amounts["DIV_VAL"], amounts["CAP_VAL"]
    return {
        "employment_income_before_lsr": amounts["WSAL_VAL"].copy(),
        "self_employment_income_before_lsr": amounts["SEMP_VAL"].copy(),
        "taxable_interest_income": amounts["INT_VAL"] * TAXABLE_INTEREST_FRACTION,
        "qualified_dividend_income": dividends * QUALIFIED_DIVIDEND_FRACTION,
        "non_qualified_dividend_income": dividends * (1 - QUALIFIED_DIVIDEND_FRACTION),
        "long_term_capital_gains_before_response": gains
        * LONG_TERM_CAPITAL_GAIN_FRACTION,
        "short_term_capital_gains": gains * (1 - LONG_TERM_CAPITAL_GAIN_FRACTION),
    }


def derive_cps_carried_current_leaves(
    selected: SelectedCurrentMoney,
    *,
    routing: pd.DataFrame,
    spm_membership: np.ndarray,
    spm_ids: np.ndarray,
) -> CpsCarriedCurrentLeaves:
    """Derive the corrected monetary CPS leaves plus ``age`` from selected money."""
    _require(type(selected) is SelectedCurrentMoney, "TYPED_SELECTED_REQUIRED")
    rows = selected.person_rows
    codes = _routing(routing, rows)
    amounts = _amounts(selected, rows)
    # The money recipe already contributes zero for a declared-NIU annuity, so
    # the pension base is the plain sum of the two restated amounts.
    pensions = amounts["PNSN_VAL"] + amounts["ANN_VAL"]
    person: dict[str, np.ndarray] = {
        "age": codes["A_AGE"].astype("float64"),
        **derive_cps_current_predictor_leaves(
            {name: amounts[name] for name in CPS_CURRENT_PREDICTOR_MONEY_FIELDS}
        ),
        "taxable_private_pension_income": pensions * TAXABLE_PENSION_FRACTION,
        "tax_exempt_private_pension_income": pensions * (1 - TAXABLE_PENSION_FRACTION),
        "taxable_ira_distributions": _ira_distributions(amounts, codes, rows),
        "rental_income": amounts["RNT_VAL"],
        "farm_operations_income": amounts["FRSE_VAL"],
        "unemployment_compensation": amounts["UC_VAL"],
        "health_insurance_premiums_without_medicare_part_b": amounts["PHIP_VAL"],
        "other_medical_expenses": amounts["PMED_VAL"],
        "over_the_counter_health_expenses": amounts["POTC_VAL"],
    }
    person.update(_social_security(amounts["SS_VAL"], codes))
    person.update(_other_income(amounts["OI_VAL"], codes))
    spm_unit = {
        "spm_unit_pre_subsidy_childcare_expenses": _childcare(
            amounts["SPM_CHILDCAREXPNS"], spm_membership, spm_ids
        )
    }
    _require(
        tuple(sorted(person)) == CPS_CARRIED_CURRENT_PERSON_LEAVES, "PERSON_LEAF_ROSTER"
    )
    _require(
        tuple(sorted(spm_unit)) == CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
        "SPM_UNIT_LEAF_ROSTER",
    )
    for values in (*person.values(), *spm_unit.values()):
        _require(values.dtype == np.dtype("float64"), "LEAF_DTYPE")
        _require(bool(np.isfinite(values).all()), "LEAF_NONFINITE")
    for name, values in person.items():
        _require(len(values) == rows, "LEAF_ROW_ALIGNMENT")
        del name
    return CpsCarriedCurrentLeaves(person, spm_unit)


def cps_carried_current_leaf_contract() -> dict:
    """Return the recorded derivation contract for these corrected leaves."""
    return {
        "schema": CPS_CARRIED_CURRENT_CONTRACT_SCHEMA,
        "amount_source": "us_asec_selected_current_money",
        "amount_basis": "target_current_2024",
        "raw_dollar_alternates": [],
        "money_fields": list(CPS_CARRIED_CURRENT_MONEY_FIELDS),
        "routing_columns": list(CPS_CARRIED_CURRENT_ROUTING_COLUMNS),
        "person_leaves": list(CPS_CARRIED_CURRENT_PERSON_LEAVES),
        "spm_unit_leaves": list(CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES),
        "dtype": "float64",
        "model_assumptions": {
            "taxable_interest_fraction": TAXABLE_INTEREST_FRACTION,
            "qualified_dividend_fraction": QUALIFIED_DIVIDEND_FRACTION,
            "taxable_pension_fraction": TAXABLE_PENSION_FRACTION,
            "long_term_capital_gain_fraction": LONG_TERM_CAPITAL_GAIN_FRACTION,
            "annuity_niu_contributes_zero": True,
            "signed_losses_retained": ["SEMP_VAL", "RNT_VAL", "FRSE_VAL"],
            "social_security_reason_routing": "RESNSS1/RESNSS2 with age-62 fallback",
            "social_security_age_fallback": _SOCIAL_SECURITY_RETIREMENT_AGE_FALLBACK,
            "other_income_split": "derive_us_alimony_from_asec on OI_VAL/OI_OFF",
            "ira_distribution_code": _IRA_DISTRIBUTION_CODE,
            "spm_childcare_grain": "spm_unit_repeated_on_person, member-consistent",
        },
        "release_eligible": False,
    }


__all__ = [
    "CPS_CARRIED_CURRENT_CONTRACT_SCHEMA",
    "CPS_CARRIED_CURRENT_MONEY_FIELDS",
    "CPS_CARRIED_CURRENT_PERSON_LEAVES",
    "CPS_CARRIED_CURRENT_ROUTING_COLUMNS",
    "CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES",
    "CpsCarriedCurrentLeaves",
    "CpsCarriedCurrentRefusalError",
    "cps_carried_current_leaf_contract",
    "derive_cps_carried_current_leaves",
]
