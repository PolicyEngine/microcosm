from dataclasses import asdict

from populace.build import nonnegative_columns_gate, target_profile_coverage_gate
from populace.build.us import (
    US_FISCAL_LEDGER_PARITY_REGISTRY,
    US_FISCAL_LEDGER_PARITY_REPORT,
    US_FISCAL_MACRO_REALISM_BANDS,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_LEDGER_REFERENCES,
    US_FISCAL_TARGET_REGISTRY,
    US_FISCAL_TARGET_SPECS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_JCT_TAX_EXPENDITURE_TARGET_SPECS,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_SOI_FISCAL_TARGET_SPECS,
    US_STATE_INCOME_TAX_TARGET_SPECS,
    SimpleTaxExpenditureReform,
)
from populace.calibrate import TargetSpec


def test_jct_tax_expenditure_specs_are_simple_income_tax_reforms() -> None:
    assert len(US_JCT_TAX_EXPENDITURE_REFORMS) >= 5
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert isinstance(spec, SimpleTaxExpenditureReform)
        assert spec.value > 0
        assert spec.measure == spec.target_name
        assert spec.kind == "neutralize_variable"
        assert spec.output_variable == "income_tax"
        assert spec.matrix_row == "reform_minus_baseline_income_tax"
        assert spec.neutralized_variable


def test_jct_target_specs_keep_reform_contract_and_values() -> None:
    by_variable = {
        spec.metadata["neutralized_variable"]: spec
        for spec in US_JCT_TAX_EXPENDITURE_TARGET_SPECS
    }
    assert set(by_variable) == {
        "salt_deduction",
        "medical_expense_deduction",
        "charitable_deduction",
        "interest_deduction",
        "qualified_business_income_deduction",
    }
    assert by_variable["salt_deduction"].value == 21.247e9
    assert by_variable["charitable_deduction"].value == 65.301e9
    for target in by_variable.values():
        assert target.measure == target.name
        assert target.metadata["kind"] == "neutralize_variable"
        assert target.metadata["output_variable"] == "income_tax"
        assert target.metadata["matrix_row"] == "reform_minus_baseline_income_tax"
        assert (
            target.metadata["measure_construction"]
            == "income_tax(reform)-income_tax(baseline)"
        )


def test_jct_reform_objects_satisfy_their_own_coverage_requirement() -> None:
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        result = target_profile_coverage_gate([spec], [spec.coverage_requirement()])
        assert result.passed


def test_us_fiscal_target_specs_are_declared_registry() -> None:
    assert US_FISCAL_TARGET_REGISTRY.country == "us"
    assert len(US_FISCAL_TARGET_REGISTRY) == len(US_FISCAL_TARGET_SPECS)
    assert len(US_FISCAL_TARGET_SPECS) >= 450
    assert set(US_FISCAL_TARGET_REGISTRY.families()) == {
        "cbo",
        "irs_soi",
        "jct",
        "state_income_tax",
    }


def test_us_fiscal_ledger_parity_registry_matches_declared_registry() -> None:
    assert US_FISCAL_LEDGER_PARITY_REPORT.passed
    assert not US_FISCAL_LEDGER_PARITY_REPORT.failures
    assert US_FISCAL_LEDGER_PARITY_REGISTRY.version == US_FISCAL_TARGET_REGISTRY.version
    assert len(US_FISCAL_TARGET_LEDGER_REFERENCES) == len(US_FISCAL_TARGET_SPECS)
    assert [asdict(spec) for spec in US_FISCAL_LEDGER_PARITY_REGISTRY.specs] == [
        asdict(spec) for spec in US_FISCAL_TARGET_REGISTRY.specs
    ]


def test_us_fiscal_ledger_references_preserve_model_and_gate_metadata() -> None:
    by_name = {
        reference.name: reference for reference in US_FISCAL_TARGET_LEDGER_REFERENCES
    }
    income_tax = by_name["nation/cbo/individual_income_tax"]
    assert income_tax.ledger_fact_key
    assert income_tax.entity == "household"
    assert income_tax.measure == "income_tax"
    assert income_tax.metadata["target_role"] == "federal_income_tax_total"

    salt = by_name["nation/jct/salt_deduction_expenditure"]
    assert salt.measure == "nation/jct/salt_deduction_expenditure"
    assert salt.metadata["kind"] == "neutralize_variable"
    assert salt.metadata["matrix_row"] == "reform_minus_baseline_income_tax"
    assert salt.metadata["neutralized_variable"] == "salt_deduction"


def test_us_fiscal_target_specs_pass_issue_40_coverage_gate() -> None:
    result = target_profile_coverage_gate(
        US_FISCAL_TARGET_SPECS,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert result.passed


def test_total_income_tax_target_is_not_positive_receipts_surrogate() -> None:
    income_tax = next(
        spec
        for spec in US_FISCAL_TARGET_SPECS
        if spec.name == "nation/cbo/individual_income_tax"
    )
    assert income_tax.measure == "income_tax"
    assert income_tax.value == 2_426_067_000_000
    assert income_tax.metadata["target_role"] == "federal_income_tax_total"


def test_state_income_tax_targets_cover_states_and_dc() -> None:
    assert len(US_STATE_INCOME_TAX_TARGET_SPECS) == 51
    by_state = {
        spec.metadata["state"]: spec for spec in US_STATE_INCOME_TAX_TARGET_SPECS
    }
    assert by_state["CA"].value == 115_845_000_000
    assert by_state["NY"].value == 63_247_000_000
    assert by_state["DC"].value == 3_456_000_000
    for state in ("AK", "FL", "NV", "SD", "TX", "WA", "WY", "NH", "TN"):
        assert by_state[state].value == 0
    for spec in US_STATE_INCOME_TAX_TARGET_SPECS:
        assert spec.measure == spec.name
        assert spec.metadata["target_role"] == "state_income_tax"


def test_soi_fiscal_targets_restore_income_source_surface() -> None:
    assert len(US_SOI_FISCAL_TARGET_SPECS) >= 400
    names = {spec.name for spec in US_SOI_FISCAL_TARGET_SPECS}
    assert any("/irs/adjusted gross income/total/" in name for name in names)
    assert any("/irs/employment income/total/" in name for name in names)
    assert any("/irs/capital gains gross/total/" in name for name in names)
    assert any("/irs/partnership and s corp income/total/" in name for name in names)
    assert any("/irs/ordinary dividends/total/" in name for name in names)
    assert any("/irs/taxable interest income/total/" in name for name in names)
    assert any("/irs/total pension income/total/" in name for name in names)
    assert any("/irs/total social security/total/" in name for name in names)
    for spec in US_SOI_FISCAL_TARGET_SPECS:
        assert spec.measure == spec.name
        assert spec.metadata["target_role"] == "soi_fiscal_distribution"
        assert spec.metadata["taxable_only"] == "true"


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
        *complete_state_income_tax_rows(51),
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
        *complete_state_income_tax_rows(51),
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


def test_jct_target_with_income_tax_measure_fails_even_with_metadata() -> None:
    targets = [
        {"name": "nation/treasury/individual_income_tax", "measure": "income_tax"},
        *[
            f"nation/irs/adjusted gross income/total/AGI in {i}/taxable/All"
            for i in range(20)
        ],
        *complete_income_source_rows(),
        *complete_state_income_tax_rows(51),
        *[
            TargetSpec(
                name=spec.target_name,
                entity="household",
                measure="income_tax",
                value=spec.value,
                source=spec.source,
                family="jct",
                metadata={
                    "kind": spec.kind,
                    "output_variable": spec.output_variable,
                    "matrix_row": spec.matrix_row,
                    "neutralized_variable": spec.neutralized_variable,
                    "measure_construction": ("income_tax(reform)-income_tax(baseline)"),
                },
            )
            for spec in US_JCT_TAX_EXPENDITURE_REFORMS
        ],
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
        {"name": "nation/cbo/individual_income_tax", "measure": "income_tax"},
        *[
            f"nation/irs/adjusted gross income/total/AGI in {i}/taxable/All"
            for i in range(20)
        ],
        *complete_income_source_rows(),
        *complete_state_income_tax_rows(49),
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert result.failures == (
        "state_income_tax: target profile has 49 match(es), needs 51 for "
        "State individual income tax collections.",
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


def complete_state_income_tax_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "name": f"state/{i:02d}/state_income_tax",
            "measure": f"state/{i:02d}/state_income_tax",
            "family": "state_income_tax",
            "metadata": {"target_role": "state_income_tax"},
        }
        for i in range(count)
    ]


def complete_jct_rows() -> list[TargetSpec]:
    return [
        TargetSpec(
            name=spec.target_name,
            entity="household",
            measure=spec.target_name,
            value=spec.value,
            source=spec.source,
            family="jct",
            metadata={
                "kind": spec.kind,
                "output_variable": spec.output_variable,
                "matrix_row": spec.matrix_row,
                "neutralized_variable": spec.neutralized_variable,
                "measure_construction": ("income_tax(reform)-income_tax(baseline)"),
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
        *complete_state_income_tax_rows(51),
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
    assert "spm_below_threshold_rate" in US_FISCAL_MACRO_REALISM_BANDS


def test_scf_nonnegative_targets_gate_negative_interest() -> None:
    assert "auto_loan_interest" in US_NONNEGATIVE_SOURCE_OUTPUTS
    result = nonnegative_columns_gate(
        {"auto_loan_interest": [120.0, -9.0]},
        US_NONNEGATIVE_SOURCE_OUTPUTS,
    )
    assert not result.passed
    assert "auto_loan_interest" in result.failures[0]
