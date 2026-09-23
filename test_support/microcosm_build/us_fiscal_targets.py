# ruff: noqa: F401
import inspect
import json
import re
from hashlib import sha256
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build import nonnegative_columns_gate, target_profile_coverage_gate
from microcosm.build.us_runtime import (
    US_FISCAL_MACRO_REALISM_BANDS,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_REFERENCES,
    US_FISCAL_TARGET_SUPPORT_EXCLUSIONS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_SOI_FISCAL_TARGET_REFERENCES,
    US_STATE_INCOME_TAX_TARGET_REFERENCES,
    SimpleTaxExpenditureReform,
    compile_us_fiscal_target_registry,
)
from microcosm.build.us_runtime.fiscal_targets import (
    US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES,
)

REFERENCE_JCT_TAX_EXPENDITURE_TARGETS = {
    "salt_deduction": "jct.tax_expenditures.cy2024.salt_deduction.revenue_loss",
    "medical_expense_deduction": (
        "jct.tax_expenditures.cy2024.medical_expense_deduction.revenue_loss"
    ),
    "charitable_deduction": (
        "jct.tax_expenditures.cy2024.charitable_deduction.revenue_loss"
    ),
    "deductible_mortgage_interest": (
        "jct.tax_expenditures.cy2024.deductible_mortgage_interest.revenue_loss"
    ),
    "qualified_business_income_deduction": (
        "jct.tax_expenditures.cy2024.qualified_business_income_deduction.revenue_loss"
    ),
    "self_employed_health_insurance_ald": (
        "jct.tax_expenditures.cy2024.self_employed_health_insurance_deduction.revenue_loss"
    ),
    "health_savings_account_ald": (
        "jct.tax_expenditures.cy2024.health_savings_account_deduction.revenue_loss"
    ),
    "student_loan_interest_ald": (
        "jct.tax_expenditures.cy2024.student_loan_interest_deduction.revenue_loss"
    ),
    "self_employed_pension_contribution_ald": (
        "jct.tax_expenditures.cy2024.self_employed_pension_contribution_deduction.revenue_loss"
    ),
    "traditional_ira_contributions": (
        "jct.tax_expenditures.cy2024.traditional_ira_deduction.revenue_loss"
    ),
    "cdcc": (
        "jct.tax_expenditures.cy2024.cdcc_and_employer_child_care_exclusion.revenue_loss"
    ),
}

REFERENCE_PROGRAM_TARGET_ROLES = {
    "federal_income_tax_total",
    "social_security_total",
    "ssi_total",
    "snap_total",
    "unemployment_compensation_total",
    "ssa_retirement_total",
    "ssa_disability_total",
    "ssa_survivors_total",
    "ssa_dependents_total",
    "eitc_total",
    "refundable_ctc_total",
    "ctc_total",
    "aca_spending",
    "aca_enrollment",
    "medicaid_enrollment",
    "medicaid_chip_enrollment",
    "chip_enrollment",
    "medicare_part_b_premium_total",
}

REFERENCE_DEDUCTION_TARGET_ROLES = {
    "itemized_deduction_total",
    "salt_deduction_total",
    "medical_expense_deduction_total",
}

CENSUS_PEP_AGE_GROUPS = (
    "0_to_4",
    "5_to_9",
    "10_to_14",
    "15_to_19",
    "20_to_24",
    "25_to_29",
    "30_to_34",
    "35_to_39",
    "40_to_44",
    "45_to_49",
    "50_to_54",
    "55_to_59",
    "60_to_64",
    "65_to_69",
    "70_to_74",
    "75_to_79",
    "80_to_84",
    "85_plus",
)


def _usda_snap_caseload_fact(
    *,
    measure_id: str,
    value: float,
    record_set_slug: str = "national_average_monthly_households",
    value_id: str = "national_total",
    geography_level: str = "country",
    geography_id: str = "0100000US",
    entity_name: str = "household",
) -> dict[str, object]:
    record_set_id = f"usda_snap.fy2024.{record_set_slug}"
    source_record_id = f"{record_set_id}.{value_id}.{measure_id}"
    return {
        "label": f"Test label for {source_record_id}",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{source_record_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{source_record_id}",
        "legacy_fact_key": f"ledger.fact.v1:{source_record_id}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "fiscal_year", "value": "2024"},
        "entity": {"name": entity_name},
        "aggregation": {"method": "mean"},
        "geography": {
            "level": geography_level,
            "id": geography_id,
            "name": f"Test geography {geography_id}",
        },
        "dimensions": {},
        "dimension_labels": {
            "usda_snap.summary_scope": "SNAP summary scope",
        },
        "dimension_value_labels": {
            "usda_snap.summary_scope": {
                value_id: f"Test SNAP summary scope {value_id}",
            }
        },
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": record_set_id,
            "groupby_dimension": "usda_snap.summary_scope",
            "groupby_dimension_label": "SNAP summary scope",
            "groupby_value_id": value_id,
            "groupby_value_label": f"Test SNAP summary scope {value_id}",
            "measure_id": measure_id,
        },
        "observed_measure": {
            "source_name": "usda_snap",
            "source_table": (
                "SNAP Monthly State Participation and Benefit Summary FY69 to current"
            ),
            "source_measure_id": measure_id,
            "source_concept": f"usda_snap.{measure_id}",
            "unit": "count",
        },
        "source": {
            "source_name": "usda_snap",
            "source_table": (
                "SNAP Monthly State Participation and Benefit Summary FY69 to current"
            ),
            "source_file": "FY24.xlsx",
            "vintage": "fiscal_year_2024",
            "url": "https://www.fns.usda.gov/pd/supplemental-nutrition-assistance-program-snap",
        },
    }


def _ssa_ssi_by_area_fact(
    *,
    measure_id: str,
    value: float,
    record_set_concept: str,
    area_category: str,
    geography_level: str = "country",
    geography_id: str = "0100000US",
) -> dict[str, object]:
    """A synthetic SSA OASDI/SSI ledger fact (source_name ``ssa``).

    ``record_set_concept`` is one of ``ssi_recipients.by_area_category``,
    ``ssi_payments.by_area_category`` or ``oasdi_ssi_payments``;
    ``area_category`` is the fused area+category groupby value id
    (``all_areas_total``, ``alabama_aged``, ``ssi_payments`` …).
    """
    record_set_id = f"ssa_supplement.cy2024.{record_set_concept}"
    source_record_id = f"{record_set_id}.{area_category}.{measure_id}"
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="ssa",
        measure_id=measure_id,
        value=value,
        geography_level=geography_level,
        geography_id=geography_id,
        groupby_dimension="ssa_supplement.area_category",
        groupby_value_id=area_category,
        layout_record_set_id=record_set_id,
    )


def _ssa_ssi_by_age_fact(
    *,
    groupby_value_id: str,
    value: float,
    universe_constraints: list[dict[str, object]],
) -> dict[str, object]:
    record_set_id = "ssa_ssi_monthly.month2024_12.ssi_federal_payment_recipients.by_age"
    return _dynamic_ledger_fact(
        source_record_id=f"{record_set_id}.{groupby_value_id}.recipient_count",
        source_name="ssa",
        measure_id="recipient_count",
        value=value,
        period_value=2024,
        layout_record_set_id=record_set_id,
        groupby_value_id=groupby_value_id,
        universe_constraints=universe_constraints,
    )


def _bea_national_fact(
    *,
    record_set_concept: str,
    measure_id: str,
    groupby_value_id: str,
    value: float,
) -> dict[str, object]:
    record_set_id = f"bea_nipa.cy2024.{record_set_concept}"
    source_record_id = f"{record_set_id}.{groupby_value_id}.{measure_id}"
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="bea",
        measure_id=measure_id,
        value=value,
        groupby_value_id=groupby_value_id,
        layout_record_set_id=record_set_id,
    )


def _ledger_fact_for_reference(reference, *, value: float) -> dict[str, object]:
    selector = dict(reference.ledger_selector)
    dimensions = dict(selector.get("dimensions") or {})
    source_name = str(selector.get("source_name") or reference.family)
    source_table = str(selector.get("source_table") or f"{source_name} table")
    source_measure_id = str(
        selector.get("source_measure_id")
        or selector.get("layout_measure_id")
        or reference.measure
        or reference.name
    )
    period_value = selector.get("period_value") or reference.period
    geography_level = str(selector.get("geography_level") or "country")
    geography_id = str(selector.get("geography_id") or "0100000US")
    entity_name = str(selector.get("entity_name") or reference.entity)
    fact_id = _fact_id(reference.name, period_value)
    return {
        "label": f"Test label for {reference.name}",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{fact_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{fact_id}",
        "legacy_fact_key": f"ledger.fact.v1:{fact_id}",
        "lineage": {"source_record_id": reference_source_record_id(reference)},
        "value": value,
        "period": {
            "type": str(selector.get("period_type") or "tax_year"),
            "value": period_value,
        },
        "entity": {"name": entity_name},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": geography_level,
            "id": geography_id,
            "name": f"Test geography {geography_id}",
        },
        "dimensions": dimensions,
        "layout": {
            "record_set_id": str(
                selector.get("layout_record_set_id") or f"{source_name}.record_set"
            ),
            "groupby_dimension": str(selector.get("layout_groupby_dimension") or ""),
            "groupby_value_id": str(selector.get("layout_groupby_value_id") or "all"),
            "measure_id": source_measure_id,
        },
        "observed_measure": {
            "source_name": source_name,
            "source_table": source_table,
            "source_measure_id": source_measure_id,
            "source_concept": str(selector.get("source_concept") or source_measure_id),
            "unit": "usd",
        },
        "source": {
            "source_name": source_name,
            "source_table": source_table,
            "vintage": str(period_value),
            "url": f"https://example.org/{fact_id}",
        },
    }


def packaged_reference_facts() -> list[dict[str, object]]:
    return [
        _ledger_fact_for_reference(reference, value=index + 1)
        for index, reference in enumerate(US_FISCAL_TARGET_REFERENCES)
    ]


def _soi_eitc_total_fact(
    source_period: int,
    *,
    measure_id: str,
    value: float,
) -> dict[str, object]:
    source_record_id = (
        f"irs_soi.ty{source_period}.table_2_5.eitc_all_returns.total.{measure_id}"
    )
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=source_period,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=f"irs_soi.ty{source_period}.table_2_5.eitc_all_returns",
        groupby_dimension="irs_soi.eitc_return_group",
        groupby_value_id="total",
    )


def _soi_eitc_filing_season_total_fact(
    *,
    measure_id: str,
    value: float,
) -> dict[str, object]:
    source_record_id = (
        "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
        f"earned_income_credit.{measure_id}"
    )
    suffix = "count" if measure_id.endswith("_returns") else "amount"
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=2024,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=(
            f"irs_soi.ty2024.filing_season_week47.eitc_all_returns.{suffix}"
        ),
        groupby_dimension="irs_soi.filing_season_line",
        groupby_value_id="earned_income_credit",
    )


def _soi_eitc_filing_season_agi_fact(
    *,
    measure_id: str,
    value: float,
    income_range: str,
    lower: float,
    upper: float | None,
    geography_level: str = "country",
    geography_id: str = "0100000US",
) -> dict[str, object]:
    source_record_id = (
        f"irs_soi.ty2024.filing_season_week47.eitc_by_agi.{income_range}.{measure_id}"
    )
    suffix = "count" if measure_id.endswith("_returns") else "amount"
    constraints: list[dict[str, object]] = [
        {
            "variable": "adjusted_gross_income",
            "operator": ">=",
            "value": lower,
        }
    ]
    if upper is not None:
        constraints.append(
            {
                "variable": "adjusted_gross_income",
                "operator": "<",
                "value": upper,
            }
        )
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=2024,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"income_range": income_range, "filing_status": "all"},
        universe_constraints=constraints,
        layout_record_set_id=(
            f"irs_soi.ty2024.filing_season_week47.eitc_by_agi.{suffix}"
        ),
        groupby_dimension="us:statutes/26/62#adjusted_gross_income",
        groupby_value_id=income_range,
    )


def _soi_eitc_child_total_facts(
    source_period: int,
    *,
    measure_id: str,
    values: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    if values is None and measure_id == "eitc_returns":
        values = {
            "no_qualifying_children": 5,
            "one_qualifying_child": 10,
            "two_qualifying_children": 15,
            "three_or_more_qualifying_children": 20,
        }
    elif values is None:
        values = {
            "no_qualifying_children": 100,
            "one_qualifying_child": 200,
            "two_qualifying_children": 300,
            "three_or_more_qualifying_children": 400,
        }
    return [
        _dynamic_ledger_fact(
            source_record_id=(
                f"irs_soi.ty{source_period}.table_2_5.eitc_by_agi_children."
                f"{child_group}.total.{measure_id}"
            ),
            source_name="irs_soi",
            measure_id=measure_id,
            value=value,
            period_value=source_period,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id=(
                f"irs_soi.ty{source_period}.table_2_5.eitc_by_agi_children."
                f"{child_group}"
            ),
            groupby_dimension="us.tax.earned_income_credit_qualifying_children",
            groupby_value_id=child_group,
        )
        for child_group, value in values.items()
    ]


def _soi_eitc_child_fact(
    source_period: int,
    *,
    source_record_id: str,
    measure_id: str,
    value: float,
    income_range: str = "25k_to_30k",
    child_group: str = "one_qualifying_child",
    lower: float = 25_000,
    upper: float | None = 30_000,
) -> dict[str, object]:
    constraints: list[dict[str, object]] = [
        {
            "variable": "adjusted_gross_income",
            "operator": ">=",
            "value": lower,
        }
    ]
    if upper is not None:
        constraints.append(
            {
                "variable": "adjusted_gross_income",
                "operator": "<",
                "value": upper,
            }
        )
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=source_period,
        dimensions={"income_range": income_range, "filing_status": "all"},
        universe_constraints=constraints,
        layout_record_set_id=(
            f"irs_soi.ty{source_period}.table_2_5.eitc_by_agi_children.{child_group}"
        ),
        groupby_dimension="us.tax.earned_income_credit_qualifying_children",
        groupby_value_id=child_group,
    )


def _soi_taxable_interest_fact(
    source_period: int,
    *,
    source_record_id: str,
    value: float,
    income_range: str = "all",
    lower: float | None = None,
    upper: float | None = None,
    measure_id: str = "taxable_interest_amount",
    layout_record_set_id: str | None = None,
    geography_level: str = "country",
    geography_id: str = "0100000US",
    filing_status: str = "all",
) -> dict[str, object]:
    constraints: list[dict[str, object]] = []
    if lower is not None:
        constraints.append(
            {
                "variable": "adjusted_gross_income",
                "operator": ">=",
                "value": lower,
            }
        )
    if upper is not None:
        constraints.append(
            {
                "variable": "adjusted_gross_income",
                "operator": "<",
                "value": upper,
            }
        )
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=source_period,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"income_range": income_range, "filing_status": filing_status},
        universe_constraints=constraints,
        layout_record_set_id=layout_record_set_id
        or f"irs_soi.ty{source_period}.historic_table_2.us",
        groupby_dimension="us:statutes/26/62#adjusted_gross_income",
        groupby_value_id=income_range,
    )


def _soi_capital_gains_fact(
    source_period: int,
    *,
    source_record_id: str,
    value: float,
    measure_id: str = "net_capital_gains_amount",
    geography_level: str = "country",
    geography_id: str = "0100000US",
    layout_record_set_id: str | None = None,
) -> dict[str, object]:
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=source_period,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"income_range": "all", "filing_status": "all"},
        layout_record_set_id=layout_record_set_id
        or f"irs_soi.ty{source_period}.historic_table_2.us",
        groupby_dimension="us:statutes/26/62#adjusted_gross_income",
        groupby_value_id="all",
    )


def _soi_congressional_district_fact(
    measure_id: str,
    value: float,
    *,
    groupby_value_id: str = "al_01",
    geography_level: str = "congressional_district",
    geography_id: str = "5001700US0101",
    dimensions: dict[str, object] | None = None,
) -> dict[str, object]:
    fact_dimensions = {"income_range": "all", "filing_status": "all"}
    if dimensions:
        fact_dimensions.update(dimensions)
    return _dynamic_ledger_fact(
        source_record_id=(
            "irs_soi.ty2023.congressional_district_2022.all_returns."
            f"{groupby_value_id}.{measure_id}"
        ),
        source_name="irs_soi",
        measure_id=measure_id,
        value=value,
        period_value=2023,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions=fact_dimensions,
        layout_record_set_id="irs_soi.ty2023.congressional_district_2022.all_returns",
        groupby_dimension="irs_soi.congressional_district",
        groupby_value_id=groupby_value_id,
    )


def _census_acs_population_age_fact(
    *,
    source_record_id: str,
    value: float,
    geography_level: str = "country",
    geography_id: str = "0100000US",
    layout_record_set_id: str = "census_acs.acs1_2024.s0101.national_age",
) -> dict[str, object]:
    return _dynamic_ledger_fact(
        source_record_id=source_record_id,
        source_name="census_acs",
        measure_id="population",
        value=value,
        period_value=2024,
        geography_level=geography_level,
        geography_id=geography_id,
        dimensions={"age": "age_0_to_4"},
        universe_constraints=[
            {
                "variable": "age",
                "operator": ">=",
                "value": 0,
            },
            {
                "variable": "age",
                "operator": "<",
                "value": 5,
            },
        ],
        layout_record_set_id=layout_record_set_id,
        groupby_dimension="age",
        groupby_value_id="age_0_to_4",
    )


def _census_acs_congressional_district_age_fact() -> dict[str, object]:
    return _census_acs_population_age_fact(
        source_record_id=(
            "census_acs.acs1_2024.s0101.congressional_district_age."
            "0101.age_0_to_4.population"
        ),
        value=42_000,
        geography_level="congressional_district",
        geography_id="5001900US0101",
        layout_record_set_id=(
            "census_acs.acs1_2024.s0101.congressional_district_age.0101"
        ),
    )


def _soi_income_tax_fact(source_period: int, *, value: float) -> dict[str, object]:
    source_record_id = (
        f"irs_soi.ty{source_period}.table_3_3.us.all.income_tax_liability_amount"
    )
    return {
        "label": "Income tax liability",
        "aggregate_fact_key": (
            f"ledger.aggregate_fact.v2:income-tax-liability-{source_period}"
        ),
        "semantic_fact_key": (
            f"ledger.semantic_fact.v2:income-tax-liability-{source_period}"
        ),
        "legacy_fact_key": f"ledger.fact.v1:income-tax-liability-{source_period}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "tax_year", "value": source_period},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
        },
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "dimension_labels": {
            "adjusted_gross_income": "Adjusted gross income",
            "filing_status": "Filing status",
            "income_range": "Income range",
        },
        "dimension_value_labels": {
            "adjusted_gross_income": {"all": "All adjusted gross income"},
            "filing_status": {"all": "All filing statuses"},
            "income_range": {"all": "All income ranges"},
        },
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": f"irs_soi.ty{source_period}.table_3_3",
            "groupby_dimension": "adjusted_gross_income",
            "groupby_dimension_label": "Adjusted gross income",
            "groupby_value_id": "all",
            "groupby_value_label": "All adjusted gross income",
            "measure_id": "income_tax_liability_amount",
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 3.3",
            "source_measure_id": "income_tax_liability_amount",
            "source_concept": "irs_soi.income_tax_liability_amount",
            "unit": "usd",
        },
        "source": {
            "source_name": "irs_soi",
            "source_table": "Publication 1304 Table 3.3",
            "source_file": f"{str(source_period)[-2:]}in33ar.xls",
            "vintage": f"tax_year_{source_period}",
            "url": f"https://www.irs.gov/pub/irs-soi/{str(source_period)[-2:]}in33ar.xls",
        },
    }


def _cbo_income_tax_fact(
    source_period: int,
    *,
    value: float,
    measure_id: str = "actual_amount",
) -> dict[str, object]:
    source_record_id = (
        f"cbo.fy{source_period}.revenues.individual_income_taxes.{measure_id}"
    )
    return {
        "label": "Individual income tax receipts",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:cbo-income-tax-{source_period}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:cbo-income-tax-{source_period}",
        "legacy_fact_key": f"ledger.fact.v1:cbo-income-tax-{source_period}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "fiscal_year", "value": source_period},
        "entity": {"name": "household"},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
        },
        "dimensions": {},
        "dimension_labels": {"revenue_source": "Revenue source"},
        "dimension_value_labels": {
            "revenue_source": {"individual_income_taxes": "Individual income taxes"}
        },
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": f"cbo.fy{source_period}.revenues",
            "groupby_dimension": "revenue_source",
            "groupby_dimension_label": "Revenue source",
            "groupby_value_id": "individual_income_taxes",
            "groupby_value_label": "Individual income taxes",
            "measure_id": measure_id,
        },
        "observed_measure": {
            "source_name": "cbo",
            "source_table": "Historical Budget Data",
            "source_measure_id": measure_id,
            "source_concept": "cbo.individual_income_tax_receipts",
            "unit": "usd",
        },
        "source": {
            "source_name": "cbo",
            "source_table": "Historical Budget Data",
            "source_file": "revenue.xlsx",
            "vintage": f"fiscal_year_{source_period}",
            "url": "https://www.cbo.gov/data/budget-economic-data",
        },
    }


def _cms_medicaid_enrollment_fact(
    source_period: str,
    *,
    value: float,
    measure_id: str = "total_medicaid_chip_enrollment",
    geography_level: str = "country",
    geography_id: str = "0100000US",
    geography_slug: str = "us",
) -> dict[str, object]:
    normalized_period = source_period.replace("-", "_")
    source_record_id = (
        f"cms_medicaid.month{normalized_period}.{geography_slug}.{measure_id}"
    )
    return {
        "label": f"Test label for {source_record_id}",
        "aggregate_fact_key": (
            f"ledger.aggregate_fact.v2:cms-medicaid-{normalized_period}"
        ),
        "semantic_fact_key": (
            f"ledger.semantic_fact.v2:cms-medicaid-{normalized_period}"
        ),
        "legacy_fact_key": f"ledger.fact.v1:cms-medicaid-{normalized_period}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "month", "value": source_period},
        "entity": {"name": "person"},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": geography_level,
            "id": geography_id,
            "name": f"Test geography {geography_id}",
        },
        "dimensions": {},
        "dimension_labels": {"program": "Program"},
        "dimension_value_labels": {
            "program": {measure_id: f"Test program {measure_id}"}
        },
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": f"cms_medicaid.month{normalized_period}",
            "groupby_dimension": "program",
            "groupby_dimension_label": "Program",
            "groupby_value_id": measure_id,
            "groupby_value_label": f"Test program {measure_id}",
            "measure_id": measure_id,
        },
        "observed_measure": {
            "source_name": "cms_medicaid",
            "source_table": "Medicaid and CHIP enrollment",
            "source_measure_id": measure_id,
            "source_concept": f"cms.{measure_id}",
            "unit": "people",
        },
        "source": {
            "source_name": "cms_medicaid",
            "source_table": "Medicaid and CHIP enrollment",
            "source_file": f"enrollment_{normalized_period}.csv",
            "vintage": f"month_{normalized_period}",
            "url": "https://data.medicaid.gov/",
        },
    }


def _dynamic_ledger_fact(
    *,
    source_record_id: str,
    source_name: str,
    measure_id: str,
    value: float,
    period_value: int | str = 2024,
    geography_level: str = "country",
    geography_id: str = "0100000US",
    groupby_dimension: str = "",
    groupby_value_id: str = "all",
    layout_record_set_id: str | None = None,
    dimensions: dict[str, object] | None = None,
    universe_constraints: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    fact_id = _fact_id(source_record_id, period_value)
    fact_dimensions = dict(dimensions or {})
    hierarchy_dimensions = dict(fact_dimensions)
    if groupby_dimension and groupby_value_id:
        hierarchy_dimensions.setdefault(groupby_dimension, groupby_value_id)
    dimension_labels = {
        dimension_id: f"Test dimension {dimension_id}"
        for dimension_id in hierarchy_dimensions
    }
    dimension_value_labels = {
        dimension_id: {
            str(dimension_value): (f"Test value {dimension_id}={dimension_value}")
        }
        for dimension_id, dimension_value in hierarchy_dimensions.items()
    }
    return {
        "label": f"Test label for {source_record_id}",
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{fact_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{fact_id}",
        "legacy_fact_key": f"ledger.fact.v1:{fact_id}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "calendar_year", "value": period_value},
        "entity": {"name": "person"},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": geography_level,
            "id": geography_id,
            "name": f"Test geography {geography_id}",
        },
        "dimensions": fact_dimensions,
        "dimension_labels": dimension_labels,
        "dimension_value_labels": dimension_value_labels,
        "universe_constraints": {"constraints": list(universe_constraints or [])},
        "layout": {
            "record_set_id": layout_record_set_id or f"{source_name}.record_set",
            "groupby_dimension": groupby_dimension,
            "groupby_dimension_label": (dimension_labels.get(groupby_dimension, "")),
            "groupby_value_id": groupby_value_id,
            "groupby_value_label": (
                dimension_value_labels.get(groupby_dimension, {}).get(
                    groupby_value_id,
                    "",
                )
            ),
            "measure_id": measure_id,
        },
        "observed_measure": {
            "source_name": source_name,
            "source_table": f"{source_name} table",
            "source_measure_id": measure_id,
            "source_concept": measure_id,
            "unit": "usd",
        },
        "source": {
            "source_name": source_name,
            "source_table": f"{source_name} table",
            "vintage": str(period_value),
            "url": f"https://example.org/{fact_id}",
        },
    }


def _fact_id(name: str, period: object) -> str:
    slug = (
        str(name)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("$", "")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace(".", "_")
    )
    digest = sha256(f"{name}@{period}".encode()).hexdigest()[:12]
    return f"{slug[:72]}_{digest}_{period}"


def reference_source_record_id(reference) -> str:
    return (
        reference.ledger_source_record_id
        or f"source.record:{_fact_id(reference.name, reference.period)}"
    )


def complete_coverage_targets() -> list[dict[str, object]]:
    return [
        federal_income_tax_total_row(),
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_snap_state_rows(),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]


def federal_income_tax_total_row() -> dict[str, object]:
    return {
        "name": "irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount",
        "measure": "irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount",
        "family": "irs_soi",
        "metadata": {"target_role": "federal_income_tax_total"},
    }


def complete_agi_distribution_rows() -> list[dict[str, object]]:
    return [
        {
            "name": f"irs_soi.cy2024.table_1_1.agi_bin_{i}.adjusted_gross_income",
            "measure": f"irs_soi.cy2024.table_1_1.agi_bin_{i}.adjusted_gross_income",
            "family": "irs_soi",
        }
        for i in range(20)
    ]


def complete_income_source_rows() -> list[dict[str, object]]:
    source_measures = [
        "wages_salaries_amount",
        "schedule_c_income_amount",
        "partnership_scorp_income_amount",
        "net_capital_gains_amount",
        "ordinary_dividends_amount",
        "taxable_interest_amount",
        "taxable_pension_income_amount",
        "taxable_social_security_amount",
    ]
    return [
        {
            "name": f"irs_soi.cy2024.table_1_1.agi_bin_{i}.{measure}",
            "measure": f"irs_soi.cy2024.table_1_1.agi_bin_{i}.{measure}",
            "family": "irs_soi",
        }
        for measure in source_measures
        for i in range(100)
    ]


def complete_deduction_amount_rows() -> list[dict[str, object]]:
    return [
        {
            "name": f"irs_soi.ty2024.deductions.us.all.{role}",
            "measure": f"irs_soi.ty2024.deductions.us.all.{role}",
            "family": "irs_soi",
            "metadata": {"target_role": role},
        }
        for role in REFERENCE_DEDUCTION_TARGET_ROLES
    ]


def complete_program_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in REFERENCE_PROGRAM_TARGET_ROLES:
        if role == "federal_income_tax_total":
            continue
        if role.startswith("ssa_") or role in {
            "social_security_total",
            "ssi_total",
        }:
            family = "ssa"
        elif role == "snap_total":
            family = "usda_snap"
        elif role in {"eitc_total", "refundable_ctc_total", "ctc_total"}:
            family = "irs_soi"
        elif role == "unemployment_compensation_total":
            family = "irs_soi"
        elif role in {"aca_spending", "aca_enrollment"}:
            family = "cms_aca"
        elif role in {
            "chip_enrollment",
            "medicaid_enrollment",
            "medicaid_chip_enrollment",
        }:
            family = "cms_medicaid"
        elif role == "medicare_part_b_premium_total":
            family = "cms_medicare"
        else:
            family = "ledger"
        rows.append(
            {
                "name": f"{family}.cy2024.{role}",
                "measure": f"{family}.cy2024.{role}",
                "family": family,
                "metadata": {"target_role": role},
            }
        )
    return rows


def complete_state_income_tax_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "name": f"census_stc.cy2024.state_{i:02d}.individual_income_tax.collections",
            "measure": f"census_stc.cy2024.state_{i:02d}.individual_income_tax.collections",
            "family": "state_income_tax",
            "metadata": {"target_role": "state_income_tax"},
        }
        for i in range(count)
    ]


def complete_snap_state_rows(count: int = 51) -> list[dict[str, object]]:
    # 50 states + DC; state_fips presence is what the snap_state_benefits
    # requirement demands (microcosm #255/#256).
    return [
        {
            "name": f"usda_snap.fy2024.state_benefits.state_{i:02d}.total_benefits",
            "measure": (
                f"usda_snap.fy2024.state_benefits.state_{i:02d}.total_benefits"
            ),
            "family": "usda_snap",
            "metadata": {"target_role": "snap_total", "state_fips": f"{i + 1:02d}"},
        }
        for i in range(count)
    ]


def complete_population_age_rows() -> list[dict[str, object]]:
    national = [
        {
            "name": (
                "census_pep.cy2024.national_resident_population_age."
                f"{age_group}.population"
            ),
            "measure": (
                "census_pep.cy2024.national_resident_population_age."
                f"{age_group}.population"
            ),
            "family": "census_population",
            "metadata": {
                "target_role": "population_age",
                "geography_scope": "national",
            },
        }
        for age_group in CENSUS_PEP_AGE_GROUPS
    ]
    state = [
        {
            "name": (
                "census_pep.v2024.cy2024.state_resident_population."
                f"{state_fips:02d}.{age_group}.population"
            ),
            "measure": (
                "census_pep.v2024.cy2024.state_resident_population."
                f"{state_fips:02d}.{age_group}.population"
            ),
            "family": "census_population",
            "metadata": {
                "target_role": "population_age",
                "geography_scope": "state",
            },
        }
        for state_fips in range(1, 52)
        for age_group in CENSUS_PEP_AGE_GROUPS
    ]
    return [*national, *state]


def complete_jct_rows() -> list[dict[str, object]]:
    return [
        {
            "name": spec.target_name,
            "measure": spec.target_name,
            "family": "jct",
            "kind": spec.kind,
            "output_variable": spec.output_variable,
            "matrix_row": spec.matrix_row,
            "neutralized_variable": spec.neutralized_variable,
        }
        for spec in US_JCT_TAX_EXPENDITURE_REFORMS
    ]


# ---------------------------------------------------------------------------
# Opt-in period aging for US dollar targets (PolicyEngine/microcosm#116, #212).
# ---------------------------------------------------------------------------


def _cbo_income_source_projection_fact(
    source_period: int,
    income_source: str,
    *,
    value: float,
) -> dict[str, object]:
    """A CBO revenue-projection income-by-source fact for one year/series.

    Mirrors the real Ledger consumer-fact shape for
    ``cbo.revenue_projection.tyYYYY.income_by_source.<series>.projected_amount``
    (CBO February 2026 Revenue Projections, sheet 3.Individual Income Tax
    Details), which supplies the aging growth ratios.
    """
    source_record_id = (
        f"cbo.revenue_projection.ty{source_period}.income_by_source."
        f"{income_source}.projected_amount"
    )
    return {
        "label": f"Test CBO projection for {income_source}",
        "assertion": "source_projection",
        "aggregate_fact_key": (
            f"ledger.aggregate_fact.v2:cbo-proj-{income_source}-{source_period}"
        ),
        "semantic_fact_key": (
            f"ledger.semantic_fact.v2:cbo-proj-{income_source}-{source_period}"
        ),
        "legacy_fact_key": f"ledger.fact.v1:cbo-proj-{income_source}-{source_period}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "tax_year", "value": source_period},
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "aggregation": {"method": "sum"},
        "geography": {
            "level": "country",
            "id": "0100000US",
            "name": "United States",
            "vintage": "current",
        },
        "dimensions": {},
        "dimension_labels": {"cbo.income_source": "Income source"},
        "dimension_value_labels": {
            "cbo.income_source": {income_source: f"Test income source {income_source}"}
        },
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": (
                f"cbo.revenue_projection.ty{source_period}.income_by_source."
                f"{income_source}"
            ),
            "groupby_dimension": "cbo.income_source",
            "groupby_dimension_label": "Income source",
            "groupby_value_id": income_source,
            "groupby_value_label": f"Test income source {income_source}",
            "measure_id": "projected_amount",
        },
        "observed_measure": {
            "source_name": "cbo",
            "source_table": "Revenue Projections, by Category, February 2026",
            "source_measure_id": "projected_amount",
            "source_concept": "cbo.adjusted_gross_income",
            "unit": "usd",
        },
        "source": {
            "source_name": "cbo",
            "source_table": "Revenue Projections, by Category, February 2026",
            "source_file": "cbo_revenue_projections_income_by_source_2026_02.csv",
            "vintage": "cbo_2026_02_baseline",
            "url": "https://www.cbo.gov/data/budget-economic-data",
        },
    }


def _aged_spec_by_source_record_id(registry, source_record_id):
    return {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}[
        source_record_id
    ]


def _soi_national_actual_fact(source_period: int, *, value: float) -> dict[str, object]:
    source_record_id = f"irs_soi.ty{source_period}.table_1_1.all.adjusted_gross_income"
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:soi-nat-{source_period}",
        "value": value,
        "period": {"type": "tax_year", "value": source_period},
        "geography": {"level": "country", "id": "0100000US"},
        "entity": {"name": "tax_unit", "role": "filing_unit"},
        "aggregation": {"method": "sum"},
        "observed_measure": {
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "unit": "usd",
        },
        "source": {"source_name": "irs_soi"},
        "lineage": {"source_record_id": source_record_id},
        "layout": {
            "record_set_id": f"irs_soi.ty{source_period}.table_1_1",
            "groupby_value_id": "all",
            "measure_id": "adjusted_gross_income",
        },
    }


__all__ = [name for name in globals() if not name.startswith("__")]
