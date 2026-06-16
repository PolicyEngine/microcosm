import json
from hashlib import sha256
from importlib.resources import files

from populace.build import nonnegative_columns_gate, target_profile_coverage_gate
from populace.build.us import (
    US_FISCAL_MACRO_REALISM_BANDS,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_REFERENCES,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_SOI_FISCAL_TARGET_REFERENCES,
    US_STATE_INCOME_TAX_TARGET_REFERENCES,
    SimpleTaxExpenditureReform,
    compile_us_fiscal_target_registry,
)
from populace.build.us.fiscal_targets import US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES

ECPS_JCT_TAX_EXPENDITURE_TARGETS = {
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
}

ECPS_PARITY_TARGET_ROLES = {
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
    "aca_spending",
    "aca_enrollment",
    "medicaid_spending",
    "medicaid_enrollment",
    "medicare_part_b_premium_total",
}


def test_packaged_us_fiscal_resources_are_value_free() -> None:
    resource = files("populace.build.us").joinpath("fiscal_target_references.json")
    payload = json.loads(resource.read_text())

    assert "target_specs" not in payload
    assert "target_references" in payload
    assert payload["allowed_value_operations"] == ["identity"]
    for reference in payload["target_references"]:
        assert "value" not in reference
        assert "source" not in reference
        assert reference["value_operation"] == "identity"
        assert reference.get("ledger_source_record_id") or reference.get(
            "ledger_selector"
        )
        metadata = reference.get("metadata") or {}
        assert "reference_url" not in metadata
        assert "uprating_factor" not in metadata


def test_legacy_us_fiscal_value_manifest_is_not_packaged() -> None:
    resource = files("populace.build.us").joinpath("fiscal_targets.json")
    assert not resource.is_file()


def test_us_fiscal_target_references_are_declared_registry() -> None:
    assert len(US_FISCAL_TARGET_REFERENCES) == len(ECPS_JCT_TAX_EXPENDITURE_TARGETS)
    assert {reference.family for reference in US_FISCAL_TARGET_REFERENCES} == {"jct"}
    assert len(US_SOI_FISCAL_TARGET_REFERENCES) == 0
    assert len(US_STATE_INCOME_TAX_TARGET_REFERENCES) == 0
    assert len(US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES) == len(
        ECPS_JCT_TAX_EXPENDITURE_TARGETS
    )
    for reference in US_FISCAL_TARGET_REFERENCES:
        assert reference.value_operation == "identity"
        assert reference.ledger_source_record_id or reference.ledger_selector


def test_us_fiscal_references_compile_against_external_ledger_facts() -> None:
    facts = [
        _ledger_fact_for_reference(reference, value=index + 1)
        for index, reference in enumerate(US_FISCAL_TARGET_REFERENCES)
    ]

    registry = compile_us_fiscal_target_registry(facts)

    assert len(registry) == len(US_FISCAL_TARGET_REFERENCES)
    for index, spec in enumerate(registry.specs):
        assert spec.value == index + 1
        assert spec.metadata["ledger_source"] == "policyengine-ledger-data"
        assert spec.metadata["ledger_value_operation"] == "identity"
        assert spec.metadata["ledger_source_record_id"] == reference_source_record_id(
            US_FISCAL_TARGET_REFERENCES[index]
        )


def test_us_fiscal_reference_selectors_are_unique_on_synthetic_fact_surface() -> None:
    facts = [
        _ledger_fact_for_reference(reference, value=index + 1)
        for index, reference in enumerate(US_FISCAL_TARGET_REFERENCES)
    ]

    compile_us_fiscal_target_registry(facts)


def test_jct_tax_expenditure_references_are_simple_income_tax_reforms() -> None:
    assert len(US_JCT_TAX_EXPENDITURE_REFORMS) == len(ECPS_JCT_TAX_EXPENDITURE_TARGETS)
    by_variable = {
        spec.neutralized_variable: spec for spec in US_JCT_TAX_EXPENDITURE_REFORMS
    }
    assert {
        variable: spec.target_name for variable, spec in by_variable.items()
    } == ECPS_JCT_TAX_EXPENDITURE_TARGETS
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert isinstance(spec, SimpleTaxExpenditureReform)
        assert spec.measure == spec.target_name
        assert spec.kind == "neutralize_variable"
        assert spec.output_variable == "income_tax"
        assert spec.matrix_row == "reform_minus_baseline_income_tax"
        assert spec.neutralized_variable


def test_jct_reform_objects_satisfy_their_own_coverage_requirement() -> None:
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        result = target_profile_coverage_gate([spec], [spec.coverage_requirement()])
        assert result.passed


def test_us_fiscal_target_references_pass_issue_40_coverage_gate() -> None:
    result = target_profile_coverage_gate(
        complete_coverage_targets(),
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert result.passed


def test_us_fiscal_target_references_cover_ecps_program_surface() -> None:
    roles = {
        (target.get("metadata") or {}).get("target_role")
        for target in complete_coverage_targets()
        if isinstance(target, dict)
    }
    assert ECPS_PARITY_TARGET_ROLES <= roles


def test_us_fiscal_requirements_include_ecps_program_and_tax_controls() -> None:
    ids = {req.requirement_id for req in US_FISCAL_TARGET_COVERAGE_REQUIREMENTS}
    assert "federal_income_tax_total" in ids
    assert "social_security_total" in ids
    assert "ssi_total" in ids
    assert "snap_total" in ids
    assert "unemployment_compensation_total" in ids
    assert "ssa_social_security_components" in ids
    assert "eitc_total" in ids
    assert "refundable_ctc_total" in ids
    assert "aca_marketplace" in ids
    assert "medicaid" in ids
    assert "irs_agi_distribution" in ids
    assert "state_income_tax" in ids
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert f"jct_tax_expenditure:{spec.neutralized_variable}" in ids


def test_structured_income_tax_positive_does_not_satisfy_total_tax() -> None:
    current_like_targets = [
        {
            "name": "nation/cbo/income_tax_positive",
            "measure": "income_tax",
            "aggregation": "positive_count_or_amount",
        },
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        current_like_targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert any("federal_income_tax_total" in failure for failure in result.failures)


def test_jct_target_name_without_simple_reform_metadata_fails() -> None:
    targets = [
        {
            "name": "irs_soi.cy2024.table_1_1.income_tax_liability_amount",
            "measure": "income_tax",
            "family": "irs_soi",
            "metadata": {"target_role": "federal_income_tax_total"},
        },
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *(spec.target_name for spec in US_JCT_TAX_EXPENDITURE_REFORMS),
    ]
    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert any(
            f"jct_tax_expenditure:{spec.neutralized_variable}" in failure
            for failure in result.failures
        )


def test_state_income_tax_needs_actual_state_surface_not_federal_row() -> None:
    targets = [
        {
            "name": "irs_soi.cy2024.table_1_1.income_tax_liability_amount",
            "measure": "income_tax",
            "family": "irs_soi",
            "metadata": {"target_role": "federal_income_tax_total"},
        },
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(44),
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert any("state_income_tax" in failure for failure in result.failures)


def test_macro_realism_bands_cover_issue_40_backstops() -> None:
    assert "federal_income_tax_to_gdp" in US_FISCAL_MACRO_REALISM_BANDS
    assert "agi_to_gdp" in US_FISCAL_MACRO_REALISM_BANDS
    assert "spm_below_threshold_rate" in US_FISCAL_MACRO_REALISM_BANDS


def test_scf_nonnegative_targets_gate_negative_interest() -> None:
    assert "auto_loan_interest" in US_NONNEGATIVE_SOURCE_OUTPUTS
    result = nonnegative_columns_gate(
        {"auto_loan_interest": [120.0, -9.0]},
        US_NONNEGATIVE_SOURCE_OUTPUTS,
    )
    assert not result.passed
    assert "auto_loan_interest" in result.failures[0]


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
    aggregation = str(selector.get("aggregation_method") or reference.aggregation)
    fact_id = _fact_id(reference.name, period_value)
    return {
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
        "aggregation": {"method": aggregation},
        "geography": {"level": geography_level, "id": geography_id},
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
        {
            "name": "irs_soi.cy2024.table_1_1.income_tax_liability_amount",
            "measure": "income_tax",
            "family": "irs_soi",
            "metadata": {"target_role": "federal_income_tax_total"},
        },
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_jct_rows(),
    ]


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
        for i in range(5)
    ]


def complete_program_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for role in ECPS_PARITY_TARGET_ROLES:
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
        elif role in {"medicaid_spending", "medicaid_enrollment"}:
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
