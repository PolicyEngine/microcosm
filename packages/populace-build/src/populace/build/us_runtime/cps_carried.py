"""CPS ASEC variables carried into PolicyEngine leaf inputs.

This stage translates raw CPS ASEC value columns already present on the
source spine into PolicyEngine input leaves needed by downstream donor stages.
It intentionally does not emit PolicyEngine formula-owned totals such as
``dividend_income`` or ``social_security``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from populace.build.us_runtime.alimony import derive_us_alimony_from_asec
from populace.frame import US_SCHEMA, Frame

__all__ = [
    "CPS_CARRIED_FORMULA_OWNED_COLUMNS",
    "CPS_CARRIED_PERSON_INPUTS",
    "CPS_CARRIED_SPM_UNIT_INPUTS",
    "derive_us_cps_carried_inputs",
]

TAXABLE_INTEREST_FRACTION = 0.680
QUALIFIED_DIVIDEND_FRACTION = 0.448
TAXABLE_PENSION_FRACTION = 0.590
LONG_TERM_CAPITAL_GAIN_FRACTION = 0.880

CPS_CARRIED_FORMULA_OWNED_COLUMNS = frozenset(
    {
        "capital_gains",
        "dividend_income",
        "employment_income",
        "interest_income",
        "long_term_capital_gains",
        "ordinary_dividend_income",
        "self_employment_income",
        "social_security",
        "taxable_pension_income",
    }
)

CPS_CARRIED_PERSON_INPUTS = frozenset(
    {
        "age",
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
        "taxable_interest_income",
        "qualified_dividend_income",
        "non_qualified_dividend_income",
        "short_term_capital_gains",
        "long_term_capital_gains_before_response",
        "social_security_retirement",
        "social_security_disability",
        "social_security_survivors",
        "social_security_dependents",
        "taxable_private_pension_income",
        "tax_exempt_private_pension_income",
        "taxable_ira_distributions",
        "health_insurance_premiums_without_medicare_part_b",
        "other_medical_expenses",
        "over_the_counter_health_expenses",
        "rental_income",
        "farm_operations_income",
        "has_champva_health_coverage_at_interview",
        "has_esi",
        "has_indian_health_service_coverage_at_interview",
        "has_marketplace_health_coverage",
        "has_marketplace_health_coverage_at_interview",
        "has_medicaid_health_coverage_at_interview",
        "has_non_marketplace_direct_purchase_health_coverage_at_interview",
        "has_other_means_tested_health_coverage_at_interview",
        "has_tricare_health_coverage_at_interview",
        "has_va_health_coverage_at_interview",
        "is_female",
        "unemployment_compensation",
        "alimony_income",
        "miscellaneous_income",
    }
)

CPS_CARRIED_SPM_UNIT_INPUTS = frozenset(
    {
        "spm_unit_pre_subsidy_childcare_expenses",
    }
)


def derive_us_cps_carried_inputs(frame: Frame) -> Frame:
    """Carry raw CPS ASEC values onto PE input leaves.

    Existing leaf input columns are preserved, making the transform idempotent.
    The transform refuses to run on a non-US frame and never creates
    formula-owned aggregate variables.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("CPS-carried derivations require the US schema.")
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]

    _fill_missing(person, "age", _source(person, "A_AGE"))
    _fill_bool_missing(person, "is_female", _integer_source(person, "A_SEX") == 2)
    _fill_missing(person, "employment_income_before_lsr", _source(person, "WSAL_VAL"))
    _fill_missing(
        person,
        "self_employment_income_before_lsr",
        _source(person, "SEMP_VAL"),
    )

    interest = _source(person, "INT_VAL")
    _fill_missing(
        person,
        "taxable_interest_income",
        interest * TAXABLE_INTEREST_FRACTION,
    )

    dividends = _source(person, "DIV_VAL")
    _fill_missing(
        person,
        "qualified_dividend_income",
        dividends * QUALIFIED_DIVIDEND_FRACTION,
    )
    _fill_missing(
        person,
        "non_qualified_dividend_income",
        dividends * (1 - QUALIFIED_DIVIDEND_FRACTION),
    )

    capital_gains = _source(person, "CAP_VAL")
    _fill_missing(
        person,
        "long_term_capital_gains_before_response",
        capital_gains * LONG_TERM_CAPITAL_GAIN_FRACTION,
    )
    _fill_missing(
        person,
        "short_term_capital_gains",
        capital_gains * (1 - LONG_TERM_CAPITAL_GAIN_FRACTION),
    )

    _fill_social_security_leaves(person)
    pensions = _source(person, "PNSN_VAL") + _source(person, "ANN_VAL")
    _fill_missing(
        person,
        "taxable_private_pension_income",
        pensions * TAXABLE_PENSION_FRACTION,
    )
    _fill_missing(
        person,
        "tax_exempt_private_pension_income",
        pensions * (1 - TAXABLE_PENSION_FRACTION),
    )
    _fill_missing(person, "taxable_ira_distributions", _ira_distributions(person))

    person = derive_us_alimony_from_asec(person)
    tables["person"] = person

    direct_sources: Mapping[str, str] = {
        "rental_income": "RNT_VAL",
        "farm_operations_income": "FRSE_VAL",
        "unemployment_compensation": "UC_VAL",
        "health_insurance_premiums_without_medicare_part_b": "PHIP_VAL",
        "other_medical_expenses": "PMED_VAL",
        "over_the_counter_health_expenses": "POTC_VAL",
    }
    # PEMCPREM remains on the source frame as evidence only. The corresponding
    # reported Part B leaf has no engine-formula consumers, so the pool must not
    # promote or transfer it onto ACS rows.
    for output, source in direct_sources.items():
        _fill_missing(person, output, _source(person, source))

    _fill_health_coverage_inputs(person)
    _fill_spm_unit_childcare_inputs(person, tables["spm_unit"])

    formula_owned = sorted(CPS_CARRIED_FORMULA_OWNED_COLUMNS.intersection(person))
    if formula_owned:
        raise ValueError(
            "CPS-carried derivations must not carry formula-owned aggregate "
            f"columns: {formula_owned}."
        )

    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _fill_missing(table: pd.DataFrame, column: str, values: np.ndarray) -> None:
    if column not in table.columns:
        table[column] = values.astype("float64")


def _fill_bool_missing(table: pd.DataFrame, column: str, values: np.ndarray) -> None:
    if column not in table.columns:
        table[column] = values.astype(bool)


def _source(person: pd.DataFrame, column: str) -> np.ndarray:
    if column not in person.columns:
        return np.zeros(len(person), dtype=np.float64)
    return (
        pd.to_numeric(person[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _fill_social_security_leaves(person: pd.DataFrame) -> None:
    social_security = _source(person, "SS_VAL")
    reason_1 = _integer_source(person, "RESNSS1")
    reason_2 = _integer_source(person, "RESNSS2")
    age = _integer_source(person, "A_AGE")

    is_retirement = (reason_1 == 1) | (reason_2 == 1)
    is_disability = (reason_1 == 2) | (reason_2 == 2)
    is_survivor = np.isin(reason_1, [3, 5]) | np.isin(reason_2, [3, 5])
    is_dependent = np.isin(reason_1, [4, 6, 7]) | np.isin(reason_2, [4, 6, 7])
    unclassified = (
        (social_security > 0)
        & ~is_retirement
        & ~is_disability
        & ~is_survivor
        & ~is_dependent
    )

    _fill_missing(
        person,
        "social_security_retirement",
        np.where(is_retirement | (unclassified & (age >= 62)), social_security, 0.0),
    )
    _fill_missing(
        person,
        "social_security_disability",
        np.where(
            (is_disability & ~is_retirement) | (unclassified & (age < 62)),
            social_security,
            0.0,
        ),
    )
    _fill_missing(
        person,
        "social_security_survivors",
        np.where(is_survivor & ~is_retirement & ~is_disability, social_security, 0.0),
    )
    _fill_missing(
        person,
        "social_security_dependents",
        np.where(
            is_dependent & ~is_retirement & ~is_disability & ~is_survivor,
            social_security,
            0.0,
        ),
    )


def _fill_health_coverage_inputs(person: pd.DataFrame) -> None:
    marketplace = _yes_code(person, "NOW_MRK")
    _fill_bool_missing(
        person,
        "has_marketplace_health_coverage_at_interview",
        marketplace,
    )
    _fill_bool_missing(person, "has_marketplace_health_coverage", marketplace)

    source_columns: Mapping[str, str] = {
        "has_non_marketplace_direct_purchase_health_coverage_at_interview": "NOW_NONM",
        "has_medicaid_health_coverage_at_interview": "NOW_MCAID",
        "has_esi": "NOW_GRP",
        "has_champva_health_coverage_at_interview": "NOW_CHAMPVA",
        "has_tricare_health_coverage_at_interview": "NOW_MIL",
        "has_va_health_coverage_at_interview": "NOW_VACARE",
        "has_other_means_tested_health_coverage_at_interview": "NOW_OTHMT",
        "has_indian_health_service_coverage_at_interview": "NOW_IHSFLG",
    }
    for output, source in source_columns.items():
        _fill_bool_missing(person, output, _yes_code(person, source))


def _fill_spm_unit_childcare_inputs(
    person: pd.DataFrame,
    spm_unit: pd.DataFrame,
) -> None:
    """Carry ASEC SPM childcare expenses onto the SPM-unit input leaf.

    ``SPM_CHILDCAREXPNS`` is an SPM-unit value replicated on every member's
    person record, so the unit's value is its members' maximum. The engine
    derives person, tax-unit, and CDCC childcare expenses from this leaf;
    a base without it zeroes every CDCC baseline (populace issue #278).
    """

    column = "spm_unit_pre_subsidy_childcare_expenses"
    if column in spm_unit.columns:
        return
    member_values = pd.DataFrame(
        {
            "person_spm_unit_id": person["person_spm_unit_id"],
            column: _source(person, "SPM_CHILDCAREXPNS"),
        }
    )
    unit_values = member_values.groupby("person_spm_unit_id", sort=False)[column].max()
    spm_unit[column] = (
        unit_values.reindex(spm_unit["spm_unit_id"])
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _ira_distributions(person: pd.DataFrame) -> np.ndarray:
    values = np.zeros(len(person), dtype=np.float64)
    for suffix in ("1", "2", "1_YNG", "2_YNG"):
        code = _integer_source(person, f"DST_SC{suffix}")
        amount = _source(person, f"DST_VAL{suffix}")
        values += np.where(code == 4, amount, 0.0)
    return values


def _integer_source(person: pd.DataFrame, column: str) -> np.ndarray:
    return _source(person, column).astype("int64")


def _yes_code(person: pd.DataFrame, column: str) -> np.ndarray:
    return _integer_source(person, column) == 1
