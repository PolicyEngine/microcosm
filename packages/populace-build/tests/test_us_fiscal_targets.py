from populace.build import nonnegative_columns_gate, target_profile_coverage_gate
from populace.build.us import (
    US_FISCAL_MACRO_REALISM_BANDS,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    SimpleTaxExpenditureReform,
)
from populace.calibrate import TargetSpec


def test_jct_tax_expenditure_specs_are_simple_income_tax_reforms() -> None:
    assert len(US_JCT_TAX_EXPENDITURE_REFORMS) >= 5
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert isinstance(spec, SimpleTaxExpenditureReform)
        assert spec.kind == "neutralize_variable"
        assert spec.output_variable == "income_tax"
        assert spec.matrix_row == "reform_minus_baseline_income_tax"
        assert spec.neutralized_variable


def test_jct_reform_objects_satisfy_their_own_coverage_requirement() -> None:
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        result = target_profile_coverage_gate([spec], [spec.coverage_requirement()])
        assert result.passed


def test_us_fiscal_requirements_include_tax_and_agi_controls() -> None:
    ids = {req.requirement_id for req in US_FISCAL_TARGET_COVERAGE_REQUIREMENTS}
    assert "federal_income_tax_total" in ids
    assert "irs_agi_distribution" in ids
    assert "irs_wages_distribution" in ids
    assert "irs_business_income_distribution" in ids
    assert "irs_partnership_s_corp_distribution" in ids
    assert "irs_capital_gains_distribution" in ids
    assert "irs_dividends_distribution" in ids
    assert "irs_interest_distribution" in ids
    assert "irs_pension_distribution" in ids
    assert "irs_social_security_distribution" in ids
    assert "state_income_tax" in ids
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert f"jct_tax_expenditure:{spec.neutralized_variable}" in ids


def test_current_comparison_like_surface_fails_missing_tax_controls_and_jct_metadata() -> (
    None
):
    current_like_targets = [
        "nation/cbo/income_tax_positive",
        "nation/irs/adjusted gross income/total/AGI in -inf-inf/taxable/All",
        "nation/irs/adjusted gross income/total/AGI in 500k-1m/taxable/All",
        "nation/irs/ordinary dividends/total/AGI in -inf-inf/all returns/All",
        "nation/jct/salt_deduction_expenditure",
        "nation/jct/medical_expense_deduction_expenditure",
        "nation/jct/charitable_deduction_expenditure",
        "nation/jct/interest_deduction_expenditure",
        "nation/jct/qualified_business_income_deduction_expenditure",
    ]
    result = target_profile_coverage_gate(
        current_like_targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert any("federal_income_tax_total" in failure for failure in result.failures)
    assert any("state_income_tax" in failure for failure in result.failures)
    assert any(
        "jct_tax_expenditure:salt_deduction" in failure for failure in result.failures
    )


def test_structured_income_tax_positive_does_not_satisfy_total_tax() -> None:
    current_like_targets = [
        {
            "name": "nation/cbo/income_tax_positive",
            "measure": "income_tax",
            "aggregation": "positive_count_or_amount",
        },
        *[
            f"nation/irs/adjusted gross income/total/AGI in {i}/taxable/All"
            for i in range(20)
        ],
        *complete_income_source_rows(),
        *[f"state/{i:02d}/state_income_tax" for i in range(50)],
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        current_like_targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert result.failures == (
        "federal_income_tax_total: target profile has 0 match(es), needs 1 for "
        "Federal individual income tax total.",
    )


def test_jct_target_name_without_simple_reform_metadata_fails() -> None:
    targets = [
        {"name": "nation/treasury/individual_income_tax", "measure": "income_tax"},
        *[
            f"nation/irs/adjusted gross income/total/AGI in {i}/taxable/All"
            for i in range(20)
        ],
        *complete_income_source_rows(),
        *[f"state/{i:02d}/state_income_tax" for i in range(50)],
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


def complete_income_source_rows() -> list[str]:
    source_stems = [
        "salaries and wages",
        "business net ",
        "partnership and s corp income",
        "capital gains gross",
        "ordinary dividends",
        "taxable interest income",
        "total pension income",
        "total social security",
    ]
    return [
        f"nation/irs/{stem}/total/AGI in {i}/taxable/All"
        for stem in source_stems
        for i in range(5)
    ]


def complete_jct_rows() -> list[TargetSpec]:
    return [
        TargetSpec(
            name=spec.target_name,
            entity="household",
            measure="income_tax_delta",
            value=1.0,
            source=spec.source,
            family="jct",
            metadata={
                "kind": spec.kind,
                "output_variable": spec.output_variable,
                "matrix_row": spec.matrix_row,
                "neutralized_variable": spec.neutralized_variable,
            },
        )
        for spec in US_JCT_TAX_EXPENDITURE_REFORMS
    ]


def test_complete_synthetic_fiscal_surface_passes() -> None:
    targets = [
        {"name": "nation/treasury/individual_income_tax", "measure": "income_tax"},
        *[
            f"nation/irs/adjusted gross income/total/AGI in {i}/taxable/All"
            for i in range(20)
        ],
        *complete_income_source_rows(),
        *[f"state/{i:02d}/state_income_tax" for i in range(50)],
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert result.passed


def test_macro_realism_bands_cover_issue_40_backstops() -> None:
    assert "federal_income_tax_to_gdp" in US_FISCAL_MACRO_REALISM_BANDS
    assert "agi_to_gdp" in US_FISCAL_MACRO_REALISM_BANDS
    assert "spm_poverty_rate" in US_FISCAL_MACRO_REALISM_BANDS


def test_scf_nonnegative_targets_gate_negative_interest() -> None:
    assert "auto_loan_interest" in US_NONNEGATIVE_SOURCE_OUTPUTS
    result = nonnegative_columns_gate(
        {"auto_loan_interest": [120.0, -9.0]},
        US_NONNEGATIVE_SOURCE_OUTPUTS,
    )
    assert not result.passed
    assert "auto_loan_interest" in result.failures[0]
