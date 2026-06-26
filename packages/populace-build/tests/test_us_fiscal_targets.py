import json
from hashlib import sha256
from importlib.resources import files

from populace.build import nonnegative_columns_gate, target_profile_coverage_gate
from populace.build.us_runtime import (
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
from populace.build.us_runtime.fiscal_targets import (
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
    "interest_deduction": (
        "jct.tax_expenditures.cy2024.deductible_mortgage_interest.revenue_loss"
    ),
    "qualified_business_income_deduction": (
        "jct.tax_expenditures.cy2024.qualified_business_income_deduction.revenue_loss"
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
    assert len(US_FISCAL_TARGET_REFERENCES) == len(
        REFERENCE_JCT_TAX_EXPENDITURE_TARGETS
    )
    assert {reference.family for reference in US_FISCAL_TARGET_REFERENCES} == {"jct"}
    assert len(US_SOI_FISCAL_TARGET_REFERENCES) == 0
    assert len(US_STATE_INCOME_TAX_TARGET_REFERENCES) == 0
    assert len(US_JCT_TAX_EXPENDITURE_TARGET_REFERENCES) == len(
        REFERENCE_JCT_TAX_EXPENDITURE_TARGETS
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


def test_zero_support_ledger_facts_are_reviewed_exclusions() -> None:
    assert len(US_FISCAL_TARGET_SUPPORT_EXCLUSIONS) == 36
    assert all(
        source_record_id.startswith(("census_stc.", "hhs_acf_tanf.", "irs_soi."))
        for source_record_id in US_FISCAL_TARGET_SUPPORT_EXCLUSIONS
    )
    assert all(US_FISCAL_TARGET_SUPPORT_EXCLUSIONS.values())


def test_reviewed_zero_support_facts_are_not_active_targets() -> None:
    excluded_source_record_id = (
        "hhs_acf_tanf.fy2024.cash_assistance.ar."
        "basic_assistance_excluding_relative_foster_care_and_adoption_guardianship."
        "all_funds"
    )
    control_source_record_id = (
        "hhs_acf_tanf.fy2024.cash_assistance.ca."
        "basic_assistance_excluding_relative_foster_care_and_adoption_guardianship."
        "all_funds"
    )
    facts = [
        *packaged_reference_facts(),
        _dynamic_ledger_fact(
            source_record_id=excluded_source_record_id,
            source_name="hhs_acf_tanf",
            measure_id="all_funds",
            value=123_000_000,
            geography_level="state",
            geography_id="0400000US05",
            groupby_value_id="ar",
        ),
        _dynamic_ledger_fact(
            source_record_id=control_source_record_id,
            source_name="hhs_acf_tanf",
            measure_id="all_funds",
            value=456_000_000,
            geography_level="state",
            geography_id="0400000US06",
            groupby_value_id="ca",
        ),
    ]

    registry = compile_us_fiscal_target_registry(facts)

    by_source_record_id = {
        spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs
    }
    assert excluded_source_record_id in US_FISCAL_TARGET_SUPPORT_EXCLUSIONS
    assert excluded_source_record_id not in by_source_record_id
    assert control_source_record_id in by_source_record_id
    control = by_source_record_id[control_source_record_id]
    assert control.family == "hhs_acf_tanf"
    assert control.metadata["target_role"] == "tanf_total"
    assert control.metadata["base_variable"] == "tanf"
    assert control.metadata["state_fips"] == "06"
    assert control.value == 456_000_000


def test_weight_dependent_medicaid_spending_is_validation_only() -> None:
    source_record_id = (
        "cms_nhe.cy2024.medicaid_title_xix_expenditures."
        "medicaid_title_xix.expenditure_amount"
    )
    facts = [
        *packaged_reference_facts(),
        _dynamic_ledger_fact(
            source_record_id=source_record_id,
            source_name="cms_nhe",
            measure_id="expenditure_amount",
            groupby_value_id="medicaid_title_xix",
            value=931_692_000_000,
        ),
    ]

    registry = compile_us_fiscal_target_registry(facts)

    by_source_record_id = {
        spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs
    }
    assert source_record_id not in by_source_record_id


def test_jct_tax_expenditure_references_are_simple_income_tax_reforms() -> None:
    assert len(US_JCT_TAX_EXPENDITURE_REFORMS) == len(
        REFERENCE_JCT_TAX_EXPENDITURE_TARGETS
    )
    by_variable = {
        spec.neutralized_variable: spec for spec in US_JCT_TAX_EXPENDITURE_REFORMS
    }
    assert {
        variable: spec.target_name for variable, spec in by_variable.items()
    } == REFERENCE_JCT_TAX_EXPENDITURE_TARGETS
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


def test_us_fiscal_target_references_cover_reference_program_surface() -> None:
    roles = {
        (target.get("metadata") or {}).get("target_role")
        for target in complete_coverage_targets()
        if isinstance(target, dict)
    }
    assert REFERENCE_PROGRAM_TARGET_ROLES <= roles


def test_medicaid_chip_enrollment_reference_uses_medicaid_and_chip_support() -> None:
    medicaid_chip_source_record_id = (
        "cms_medicaid.fy2024.us.total_medicaid_chip_enrollment"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *[
                _ledger_fact_for_reference(reference, value=index + 1)
                for index, reference in enumerate(US_FISCAL_TARGET_REFERENCES)
            ],
            {
                "lineage": {
                    "source_record_id": medicaid_chip_source_record_id,
                },
                "value": 90_000_000,
                "period": {"value": 2024},
                "geography": {"level": "country", "id": "0100000US"},
                "aggregation": {"method": "sum"},
                "layout": {"measure_id": "total_medicaid_chip_enrollment"},
                "observed_measure": {
                    "source_name": "cms_medicaid",
                    "source_measure_id": "total_medicaid_chip_enrollment",
                },
            },
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[medicaid_chip_source_record_id]
    assert spec.family == "cms_medicaid"
    assert spec.metadata["target_role"] == "medicaid_chip_enrollment"
    assert spec.metadata["base_variables"] == "medicaid_enrolled,chip_enrolled"
    assert "base_variable" not in spec.metadata


def test_dynamic_us_fiscal_targets_use_builder_target_period() -> None:
    source_record_id = "irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount"

    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_income_tax_fact(2023, value=2_100_000_000_000),
        ],
        target_period=2025,
    )

    specs = {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.period == 2025
    assert spec.metadata["source_period"] == "2023"
    assert spec.metadata["target_period"] == "2025"
    assert spec.metadata["target_role"] == "federal_income_tax_total"


def test_static_jct_targets_use_builder_target_period() -> None:
    registry = compile_us_fiscal_target_registry(
        packaged_reference_facts(),
        target_period=2025,
    )

    specs = {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}
    spec = specs[REFERENCE_JCT_TAX_EXPENDITURE_TARGETS["salt_deduction"]]
    assert spec.period == 2025
    assert spec.metadata["target_period"] == "2025"
    assert spec.metadata["target_role"] == "jct_tax_expenditure"


def test_cbo_individual_income_tax_receipts_do_not_enter_calibration() -> None:
    source_record_id = "cbo.fy2024.revenues.individual_income_taxes.actual_amount"

    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _cbo_income_tax_fact(2024, value=2_426_067_000_000),
        ]
    )

    specs = {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}
    assert source_record_id not in specs


def test_cbo_actual_and_projected_income_tax_receipts_emit_no_hard_target() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _cbo_income_tax_fact(
                2024,
                value=2_426_067_000_000,
                measure_id="actual_amount",
            ),
            _cbo_income_tax_fact(
                2025,
                value=2_656_000_000_000,
                measure_id="projected_amount",
            ),
        ],
        target_period=2025,
    )

    income_tax_specs = [
        spec
        for spec in registry.specs
        if spec.metadata.get("target_role") == "federal_income_tax_total"
    ]
    assert income_tax_specs == []


def test_soi_income_tax_liability_supplies_federal_income_tax_target() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_income_tax_fact(2023, value=2_100_000_000_000),
        ]
    )

    specs = {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}
    spec = specs["irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount"]
    assert spec.family == "irs_soi"
    assert spec.measure == "irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount"
    assert spec.metadata["target_role"] == "federal_income_tax_total"
    assert spec.metadata["variable"] == "income_tax"
    assert spec.metadata["materializer"] == "irs_soi_slice"
    assert spec.metadata["measure_mode"] == "sum"
    assert spec.metadata["base_variable"] == "income_tax"


def test_dynamic_us_fiscal_targets_choose_latest_available_source_period() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_income_tax_fact(2022, value=2_000_000_000_000),
            _soi_income_tax_fact(2023, value=2_100_000_000_000),
        ]
    )

    by_source_record_id = {
        spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs
    }
    assert "irs_soi.ty2022.table_3_3.us.all.income_tax_liability_amount" not in (
        by_source_record_id
    )
    spec = by_source_record_id[
        "irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount"
    ]
    assert spec.value == 2_100_000_000_000
    assert spec.metadata["source_period"] == "2023"
    assert spec.metadata["target_period"] == "2024"


def test_dynamic_us_fiscal_targets_do_not_prefer_future_observed_years() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_income_tax_fact(2023, value=2_100_000_000_000),
            _soi_income_tax_fact(2025, value=2_600_000_000_000),
        ],
        target_period=2024,
    )

    by_source_record_id = {
        spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs
    }
    assert "irs_soi.ty2025.table_3_3.us.all.income_tax_liability_amount" not in (
        by_source_record_id
    )
    spec = by_source_record_id[
        "irs_soi.ty2023.table_3_3.us.all.income_tax_liability_amount"
    ]
    assert spec.value == 2_100_000_000_000
    assert spec.period == 2024


def test_dynamic_us_fiscal_targets_skip_future_only_source_periods() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_income_tax_fact(2025, value=2_600_000_000_000),
        ],
        target_period=2024,
    )

    by_source_record_id = {
        spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs
    }
    assert "irs_soi.ty2025.table_3_3.us.all.income_tax_liability_amount" not in (
        by_source_record_id
    )


def test_dynamic_us_fiscal_targets_do_not_prefer_future_month_periods() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _cms_medicaid_enrollment_fact(
                "2024-12",
                value=80_000_000,
            ),
            _cms_medicaid_enrollment_fact(
                "2025-12",
                value=82_000_000,
            ),
        ],
        target_period=2024,
    )

    by_source_record_id = {
        spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs
    }
    assert (
        "cms_medicaid.month2025_12.us.total_medicaid_chip_enrollment"
        not in by_source_record_id
    )
    spec = by_source_record_id[
        "cms_medicaid.month2024_12.us.total_medicaid_chip_enrollment"
    ]
    assert spec.value == 80_000_000
    assert spec.metadata["source_period"] == "2024-12"
    assert spec.period == 2024


def test_cms_aca_references_use_current_annual_aca_variables() -> None:
    marketplace_source_record_id = (
        "cms_aca.oep2024.state_marketplace.ca.marketplace_enrollment"
    )
    aptc_source_record_id = "cms_aca.oep2024.state_marketplace.ca.aptc_recipients"
    bronze_source_record_id = "cms_aca.oep2024.state_metal.ca.bronze_aptc_consumers"
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=marketplace_source_record_id,
                source_name="cms_aca",
                measure_id="marketplace_enrollment",
                value=1_784_653,
                geography_level="state",
                geography_id="0400000US06",
                groupby_value_id="ca",
            ),
            _dynamic_ledger_fact(
                source_record_id=aptc_source_record_id,
                source_name="cms_aca",
                measure_id="aptc_recipients",
                value=1_568_732,
                geography_level="state",
                geography_id="0400000US06",
                groupby_value_id="ca",
            ),
            _dynamic_ledger_fact(
                source_record_id=bronze_source_record_id,
                source_name="cms_aca",
                measure_id="bronze_aptc_consumers",
                value=1_000_000,
                geography_level="state",
                geography_id="0400000US06",
                groupby_value_id="ca",
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    marketplace = specs[marketplace_source_record_id]
    assert marketplace.family == "cms_aca"
    assert marketplace.metadata["target_role"] == "aca_enrollment"
    assert marketplace.metadata["measure_mode"] == "indicator_sum"
    assert (
        marketplace.metadata["base_variable"]
        == "has_marketplace_health_coverage_at_interview"
    )
    assert marketplace.metadata["state_fips"] == "06"

    aptc = specs[aptc_source_record_id]
    assert aptc.family == "cms_aca"
    assert aptc.metadata["target_role"] == "aca_ptc_recipients"
    assert aptc.metadata["measure_mode"] == "indicator_sum"
    assert aptc.metadata["base_variable"] == "assigned_aca_ptc"
    assert aptc.metadata["indicator_map_to"] == "person"
    assert aptc.metadata["indicator_filter_variable"] == "is_aca_ptc_eligible"
    assert aptc.metadata["state_fips"] == "06"

    bronze = specs[bronze_source_record_id]
    assert bronze.family == "cms_aca"
    assert bronze.metadata["target_role"] == "aca_bronze_aptc_consumers"
    assert bronze.metadata["measure_mode"] == "less_than_indicator_sum"
    assert (
        bronze.metadata["base_variable"] == "selected_marketplace_plan_benchmark_ratio"
    )
    assert bronze.metadata["indicator_less_than"] == "1.0"
    assert bronze.metadata["indicator_filter_variable"] == "assigned_aca_ptc"
    assert bronze.metadata["state_fips"] == "06"


def test_soi_premium_tax_credit_targets_use_annual_assigned_ptc() -> None:
    amount_source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount"
    )
    returns_source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_returns"
    )
    state_amount_source_record_id = (
        "irs_soi.ty2022.historic_table_2.state_broad.ca.all.premium_tax_credit_amount"
    )
    state_returns_source_record_id = (
        "irs_soi.ty2022.historic_table_2.state_broad.ca.all.premium_tax_credit_returns"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=amount_source_record_id,
                source_name="irs_soi",
                measure_id="premium_tax_credit_amount",
                value=53_910_175_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
            _dynamic_ledger_fact(
                source_record_id=returns_source_record_id,
                source_name="irs_soi",
                measure_id="premium_tax_credit_returns",
                value=12_000_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
            _dynamic_ledger_fact(
                source_record_id=state_amount_source_record_id,
                source_name="irs_soi",
                measure_id="premium_tax_credit_amount",
                value=6_000_000_000,
                period_value=2022,
                geography_level="state",
                geography_id="0400000US06",
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
            _dynamic_ledger_fact(
                source_record_id=state_returns_source_record_id,
                source_name="irs_soi",
                measure_id="premium_tax_credit_returns",
                value=1_000_000,
                period_value=2022,
                geography_level="state",
                geography_id="0400000US06",
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    amount = specs[amount_source_record_id]
    assert amount.family == "irs_soi"
    assert amount.metadata["target_role"] == "aca_spending"
    assert amount.metadata["variable"] == "assigned_aca_ptc"
    assert amount.metadata["materializer"] == "irs_soi_slice"
    assert amount.metadata["measure_mode"] == "sum"
    assert amount.metadata["base_variable"] == "assigned_aca_ptc"
    assert "count" not in amount.metadata

    returns = specs[returns_source_record_id]
    assert returns.family == "irs_soi"
    assert returns.metadata["target_role"] == "aca_ptc_returns"
    assert returns.metadata["variable"] == "assigned_aca_ptc"
    assert returns.metadata["materializer"] == "irs_soi_slice"
    assert returns.metadata["measure_mode"] == "indicator_sum"

    state_amount = specs[state_amount_source_record_id]
    assert state_amount.metadata["target_role"] == "aca_spending"
    assert state_amount.metadata["state_fips"] == "06"

    state_returns = specs[state_returns_source_record_id]
    assert state_returns.metadata["target_role"] == "aca_ptc_returns"
    assert state_returns.metadata["state_fips"] == "06"
    assert returns.metadata["base_variable"] == "assigned_aca_ptc"
    assert "count" not in returns.metadata


def test_soi_ctc_targets_expose_allowed_nonrefundable_basis() -> None:
    amount_source_record_id = "irs_soi.ty2022.historic_table_2.us.all.ctc_amount"
    returns_source_record_id = "irs_soi.ty2022.historic_table_2.us.all.ctc_claims"
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=amount_source_record_id,
                source_name="irs_soi",
                measure_id="ctc_amount",
                value=82_863_353_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
            _dynamic_ledger_fact(
                source_record_id=returns_source_record_id,
                source_name="irs_soi",
                measure_id="ctc_claims",
                value=38_068_980,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    amount = specs[amount_source_record_id]
    assert amount.family == "irs_soi"
    assert amount.metadata["target_role"] == "ctc_total"
    assert amount.metadata["variable"] == "ctc"
    assert amount.metadata["materializer"] == "irs_soi_slice"
    assert amount.metadata["measure_mode"] == "sum"
    assert amount.metadata["base_variables"] == "ctc,ctc_limiting_tax_liability"
    assert "base_variable" not in amount.metadata
    assert "count" not in amount.metadata

    returns = specs[returns_source_record_id]
    assert returns.family == "irs_soi"
    assert returns.metadata["variable"] == "ctc"
    assert returns.metadata["materializer"] == "irs_soi_slice"
    assert returns.metadata["measure_mode"] == "indicator_sum"
    assert returns.metadata["base_variables"] == "ctc,ctc_limiting_tax_liability"
    assert "count" not in returns.metadata


def test_soi_eitc_child_record_set_metadata_reaches_compiled_target() -> None:
    source_record_id = (
        "irs_soi.ty2024.table_2_5.eitc_by_agi_children.no_qualifying_children."
        "25k_to_30k.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="eitc_total",
                value=535_000,
                period_value=2024,
                dimensions={"income_range": "25k_to_30k", "filing_status": "all"},
                universe_constraints=[
                    {
                        "variable": "adjusted_gross_income",
                        "operator": ">=",
                        "value": 25_000,
                    },
                    {
                        "variable": "adjusted_gross_income",
                        "operator": "<",
                        "value": 30_000,
                    },
                ],
                layout_record_set_id=(
                    "irs_soi.ty2024.table_2_5.eitc_by_agi_children."
                    "no_qualifying_children"
                ),
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "irs_soi"
    assert spec.metadata["variable"] == "eitc"
    assert spec.metadata["agi_lower_bound"] == "25000.0"
    assert spec.metadata["agi_upper_bound"] == "30000.0"
    assert spec.metadata["ledger_filter_income_range"] == "25k_to_30k"
    assert spec.metadata["ledger_layout_record_set_id"] == (
        "irs_soi.ty2024.table_2_5.eitc_by_agi_children.no_qualifying_children"
    )


def test_cross_period_soi_eitc_decomposition_uprates_to_active_total() -> None:
    amount_source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "25k_to_30k.eitc_total"
    )
    returns_source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "25k_to_30k.eitc_returns"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_total_fact(
                2024,
                measure_id="total_earned_income_credit_amount",
                value=1_200,
            ),
            _soi_eitc_total_fact(
                2024,
                measure_id="total_earned_income_credit_returns",
                value=60,
            ),
            *_soi_eitc_child_total_facts(2023, measure_id="eitc_total"),
            *_soi_eitc_child_total_facts(2023, measure_id="eitc_returns"),
            _soi_eitc_child_fact(
                2023,
                source_record_id=amount_source_record_id,
                measure_id="eitc_total",
                value=25,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=returns_source_record_id,
                measure_id="eitc_returns",
                value=10,
            ),
        ],
        target_period=2024,
    )

    specs = {spec.name: spec for spec in registry.specs}
    amount = specs[amount_source_record_id]
    assert amount.value == 30
    assert amount.metadata["requires_total_eitc_uprating"] == "true"
    assert amount.metadata["uprating_index"] == "total_eitc_amount"
    assert amount.metadata["uprating_from_period"] == "2023"
    assert amount.metadata["uprating_to_period"] == "2024"
    assert amount.metadata["uprating_factor"] == "1.2"

    returns = specs[returns_source_record_id]
    assert returns.value == 12
    assert returns.metadata["requires_total_eitc_uprating"] == "true"
    assert returns.metadata["uprating_index"] == "total_eitc_returns"
    assert returns.metadata["uprating_factor"] == "1.2"

    scaled_child_total = specs[
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
        "one_qualifying_child.total.eitc_total"
    ]
    assert scaled_child_total.value == 240
    assert scaled_child_total.metadata["uprating_factor"] == "1.2"


def test_cross_period_soi_taxable_interest_agi_slice_uprates_to_active_total() -> None:
    source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.200k_to_500k.taxable_interest_amount"
    )
    returns_source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.200k_to_500k.taxable_interest_returns"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=(
                    "irs_soi.ty2022.historic_table_2.us.all.taxable_interest_amount"
                ),
                value=300_000_000_000,
            ),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=(
                    "irs_soi.ty2022.historic_table_2.us.all.taxable_interest_returns"
                ),
                measure_id="taxable_interest_returns",
                value=30_000_000,
            ),
            _soi_taxable_interest_fact(
                2024,
                source_record_id=(
                    "irs_soi.ty2024.state_2022.us.all.taxable_interest_amount"
                ),
                value=360_000_000_000,
                layout_record_set_id="irs_soi.ty2024.state_2022.us",
            ),
            _soi_taxable_interest_fact(
                2024,
                source_record_id=(
                    "irs_soi.ty2024.state_2022.us.all.taxable_interest_returns"
                ),
                measure_id="taxable_interest_returns",
                value=33_000_000,
                layout_record_set_id="irs_soi.ty2024.state_2022.us",
            ),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=source_record_id,
                value=21_000_000_000,
                income_range="200k_to_500k",
                lower=200_000,
                upper=500_000,
            ),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=returns_source_record_id,
                measure_id="taxable_interest_returns",
                value=2_000_000,
                income_range="200k_to_500k",
                lower=200_000,
                upper=500_000,
            ),
        ],
        target_period=2024,
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "irs_soi"
    assert spec.value == 25_200_000_000
    assert spec.metadata["variable"] == "taxable_interest_income"
    assert spec.metadata["measure_mode"] == "sum"
    assert spec.metadata["requires_total_soi_uprating"] == "true"
    assert spec.metadata["uprating_index"] == "total_taxable_interest_amount"
    assert spec.metadata["uprating_from_period"] == "2022"
    assert spec.metadata["uprating_to_period"] == "2024"
    assert spec.metadata["uprating_index_source_period"] == "2024"
    assert spec.metadata["uprating_index_source_record_id"] == (
        "irs_soi.ty2024.state_2022.us.all.taxable_interest_amount"
    )
    assert spec.metadata["uprating_factor"] == "1.2"

    returns = specs[returns_source_record_id]
    assert returns.value == 2_200_000
    assert returns.metadata["variable"] == "taxable_interest_income"
    assert returns.metadata["measure_mode"] == "indicator_sum"
    assert returns.metadata["requires_total_soi_uprating"] == "true"
    assert returns.metadata["uprating_index"] == "total_taxable_interest_returns"
    assert returns.metadata["uprating_factor"] == "1.1"


def test_cross_period_soi_taxable_interest_agi_slice_uses_latest_available_total() -> (
    None
):
    source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.200k_to_500k.taxable_interest_amount"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=(
                    "irs_soi.ty2022.historic_table_2.us.all.taxable_interest_amount"
                ),
                value=300_000_000_000,
            ),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=source_record_id,
                value=21_000_000_000,
                income_range="200k_to_500k",
                lower=200_000,
                upper=500_000,
            ),
        ],
        target_period=2024,
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.value == 21_000_000_000
    assert spec.metadata["uprating_index_source_period"] == "2022"
    assert spec.metadata["uprating_factor"] == "1"


def test_cross_period_soi_taxable_interest_open_ended_agi_slice_without_dimension_is_kept() -> (
    None
):
    source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.500k_plus.taxable_interest_amount"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=(
                    "irs_soi.ty2022.historic_table_2.us.all.taxable_interest_amount"
                ),
                value=300_000_000_000,
            ),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="taxable_interest_amount",
                value=90_000_000_000,
                period_value=2022,
                dimensions={"filing_status": "all"},
                universe_constraints=[
                    {
                        "variable": "adjusted_gross_income",
                        "operator": ">=",
                        "value": 500_000,
                    }
                ],
                layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
                groupby_dimension="us:statutes/26/62#adjusted_gross_income",
                groupby_value_id="500k_plus",
            ),
        ],
        target_period=2024,
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.value == 90_000_000_000
    assert spec.metadata["agi_lower_bound"] == "500000.0"
    assert spec.metadata["agi_upper_bound"] == "inf"
    assert spec.metadata["requires_total_soi_uprating"] == "true"
    assert spec.metadata["uprating_factor"] == "1"


def test_cross_period_soi_taxable_interest_agi_slice_without_total_is_dropped() -> None:
    source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.200k_to_500k.taxable_interest_amount"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_taxable_interest_fact(
                2022,
                source_record_id=source_record_id,
                value=21_000_000_000,
                income_range="200k_to_500k",
                lower=200_000,
                upper=500_000,
            ),
        ],
        target_period=2024,
    )

    assert source_record_id not in {spec.name for spec in registry.specs}


def test_cross_period_soi_eitc_decomposition_uses_latest_source_backed_total() -> None:
    source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "25k_to_30k.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_total_fact(
                2023,
                measure_id="total_earned_income_credit_amount",
                value=66_270_000_000,
            ),
            *_soi_eitc_child_total_facts(
                2023,
                measure_id="eitc_total",
                values={"all_qualifying_children": 66_270_000_000},
            ),
            _soi_eitc_filing_season_total_fact(
                measure_id="total_earned_income_credit_amount",
                value=69_041_649_000,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=source_record_id,
                measure_id="eitc_total",
                value=66_270_000,
            ),
        ],
        target_period=2024,
    )

    spec = {spec.name: spec for spec in registry.specs}[source_record_id]
    assert spec.value == 69_041_649
    assert spec.metadata["uprating_index"] == "total_eitc_amount"
    assert spec.metadata["uprating_index_source_period"] == "2024"
    assert spec.metadata["uprating_index_source_record_id"] == (
        "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
        "earned_income_credit.total_earned_income_credit_amount"
    )
    assert spec.metadata["uprating_factor"] == "1.04182358533273"


def test_cross_period_soi_eitc_decomposition_prefers_agi_specific_factor() -> None:
    amount_source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "25k_to_30k.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_filing_season_total_fact(
                measure_id="total_earned_income_credit_amount",
                value=1_000,
            ),
            _soi_eitc_filing_season_agi_fact(
                measure_id="total_earned_income_credit_amount",
                value=100,
                income_range="25k_to_30k",
                lower=25_000,
                upper=30_000,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    "no_qualifying_children.25k_to_30k.eitc_total"
                ),
                measure_id="eitc_total",
                value=5,
                child_group="no_qualifying_children",
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=amount_source_record_id,
                measure_id="eitc_total",
                value=25,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    "two_qualifying_children.25k_to_30k.eitc_total"
                ),
                measure_id="eitc_total",
                value=15,
                child_group="two_qualifying_children",
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    "three_or_more_qualifying_children.25k_to_30k.eitc_total"
                ),
                measure_id="eitc_total",
                value=5,
                child_group="three_or_more_qualifying_children",
            ),
        ],
        target_period=2024,
    )

    spec = {spec.name: spec for spec in registry.specs}[amount_source_record_id]
    assert spec.value == 50
    assert spec.metadata["uprating_index"] == "agi_eitc_amount"
    assert spec.metadata["uprating_agi_lower_bound"] == "25000"
    assert spec.metadata["uprating_agi_upper_bound"] == "30000"
    assert spec.metadata["uprating_index_source_record_id"] == (
        "irs_soi.ty2024.filing_season_week47.eitc_by_agi."
        "25k_to_30k.total_earned_income_credit_amount"
    )
    assert spec.metadata["uprating_factor"] == "2"


def test_cross_period_soi_eitc_open_ended_decomposition_combines_agi_bins() -> None:
    amount_source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "50k_plus.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_filing_season_total_fact(
                measure_id="total_earned_income_credit_amount",
                value=1_000,
            ),
            _soi_eitc_filing_season_agi_fact(
                measure_id="total_earned_income_credit_amount",
                value=80,
                income_range="50k_to_75k",
                lower=50_000,
                upper=75_000,
            ),
            _soi_eitc_filing_season_agi_fact(
                measure_id="total_earned_income_credit_amount",
                value=20,
                income_range="75k_to_100k",
                lower=75_000,
                upper=100_000,
            ),
            _soi_eitc_filing_season_agi_fact(
                measure_id="total_earned_income_credit_amount",
                value=0,
                income_range="100k_plus",
                lower=100_000,
                upper=None,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    "no_qualifying_children.50k_plus.eitc_total"
                ),
                measure_id="eitc_total",
                value=5,
                income_range="50k_plus",
                child_group="no_qualifying_children",
                lower=50_000,
                upper=None,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=amount_source_record_id,
                measure_id="eitc_total",
                value=25,
                income_range="50k_plus",
                lower=50_000,
                upper=None,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    "two_qualifying_children.50k_plus.eitc_total"
                ),
                measure_id="eitc_total",
                value=10,
                income_range="50k_plus",
                child_group="two_qualifying_children",
                lower=50_000,
                upper=None,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=(
                    "irs_soi.ty2023.table_2_5.eitc_by_agi_children."
                    "three_or_more_qualifying_children.50k_plus.eitc_total"
                ),
                measure_id="eitc_total",
                value=10,
                income_range="50k_plus",
                child_group="three_or_more_qualifying_children",
                lower=50_000,
                upper=None,
            ),
        ],
        target_period=2024,
    )

    spec = {spec.name: spec for spec in registry.specs}[amount_source_record_id]
    assert spec.value == 50
    assert spec.metadata["uprating_index"] == "agi_eitc_amount"
    assert spec.metadata["uprating_agi_lower_bound"] == "50000"
    assert spec.metadata["uprating_agi_upper_bound"] == "inf"
    assert spec.metadata["uprating_index_source_record_ids"] == (
        "irs_soi.ty2024.filing_season_week47.eitc_by_agi."
        "50k_to_75k.total_earned_income_credit_amount,"
        "irs_soi.ty2024.filing_season_week47.eitc_by_agi."
        "75k_to_100k.total_earned_income_credit_amount,"
        "irs_soi.ty2024.filing_season_week47.eitc_by_agi."
        "100k_plus.total_earned_income_credit_amount"
    )
    assert spec.metadata["uprating_factor"] == "2"


def test_cross_period_soi_eitc_uprating_ignores_state_agi_control() -> None:
    amount_source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "25k_to_30k.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_total_fact(
                2023,
                measure_id="total_earned_income_credit_amount",
                value=500,
            ),
            _soi_eitc_filing_season_total_fact(
                measure_id="total_earned_income_credit_amount",
                value=1_000,
            ),
            _soi_eitc_filing_season_agi_fact(
                measure_id="total_earned_income_credit_amount",
                value=10,
                income_range="25k_to_30k",
                lower=25_000,
                upper=30_000,
                geography_level="state",
                geography_id="0400000US06",
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=amount_source_record_id,
                measure_id="eitc_total",
                value=25,
            ),
        ],
        target_period=2024,
    )

    spec = {spec.name: spec for spec in registry.specs}[amount_source_record_id]
    assert spec.value == 50
    assert spec.metadata["uprating_index"] == "total_eitc_amount"
    assert spec.metadata["uprating_factor"] == "2"


def test_cross_period_soi_eitc_uprating_requires_complete_agi_interval() -> None:
    amount_source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "50k_plus.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_total_fact(
                2023,
                measure_id="total_earned_income_credit_amount",
                value=500,
            ),
            _soi_eitc_filing_season_total_fact(
                measure_id="total_earned_income_credit_amount",
                value=1_000,
            ),
            _soi_eitc_filing_season_agi_fact(
                measure_id="total_earned_income_credit_amount",
                value=80,
                income_range="50k_to_75k",
                lower=50_000,
                upper=75_000,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=amount_source_record_id,
                measure_id="eitc_total",
                value=25,
                income_range="50k_plus",
                lower=50_000,
                upper=None,
            ),
        ],
        target_period=2024,
    )

    spec = {spec.name: spec for spec in registry.specs}[amount_source_record_id]
    assert spec.value == 50
    assert spec.metadata["uprating_index"] == "total_eitc_amount"
    assert spec.metadata["uprating_factor"] == "2"


def test_cross_period_soi_eitc_returns_uprate_to_filing_season_total() -> None:
    source_record_id = (
        "irs_soi.ty2023.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "25k_to_30k.eitc_returns"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _soi_eitc_total_fact(
                2023,
                measure_id="total_earned_income_credit_returns",
                value=24_439_936,
            ),
            *_soi_eitc_child_total_facts(
                2023,
                measure_id="eitc_returns",
                values={"all_qualifying_children": 24_439_936},
            ),
            _soi_eitc_filing_season_total_fact(
                measure_id="total_earned_income_credit_returns",
                value=23_837_149,
            ),
            _soi_eitc_child_fact(
                2023,
                source_record_id=source_record_id,
                measure_id="eitc_returns",
                value=24_439.936,
            ),
        ],
        target_period=2024,
    )

    spec = {spec.name: spec for spec in registry.specs}[source_record_id]
    assert spec.value == 23_837.149
    assert spec.metadata["uprating_index"] == "total_eitc_returns"
    assert spec.metadata["uprating_index_source_period"] == "2024"
    assert spec.metadata["uprating_index_source_record_id"] == (
        "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
        "earned_income_credit.total_earned_income_credit_returns"
    )
    assert spec.metadata["uprating_factor"] == "0.975335982876551"


def test_soi_eitc_layout_child_count_filter_reaches_compiled_target() -> None:
    source_record_id = (
        "irs_soi.ty2024.state_2022.us.eitc_three_or_more_children_returns."
        "three_or_more_qualifying_children.return_count"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="return_count",
                value=3_080_790,
                period_value=2024,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id=(
                    "irs_soi.ty2024.state_2022.us.eitc_three_or_more_children_returns"
                ),
                groupby_dimension=("us.tax.earned_income_credit_qualifying_children"),
                groupby_value_id="three_or_more_qualifying_children",
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "irs_soi"
    assert spec.metadata["variable"] == "eitc"
    assert spec.metadata["base_variable"] == "eitc"
    assert spec.metadata["measure_mode"] == "indicator_sum"
    assert "count" not in spec.metadata
    assert (
        spec.metadata["ledger_filter_eitc_child_count"]
        == "three_or_more_qualifying_children"
    )


def test_soi_form_w2_social_security_tips_return_count_targets_tip_income() -> None:
    source_record_id = (
        "irs_soi.ty2024.form_w2_social_security_tips."
        "box_7_social_security_tips.return_count"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="return_count",
                value=6_038_613,
                period_value=2024,
                layout_record_set_id="irs_soi.ty2024.form_w2_social_security_tips",
                groupby_dimension="irs_soi.form_w2_item",
                groupby_value_id="box_7_social_security_tips",
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "irs_soi"
    assert spec.metadata["variable"] == "tip_income"
    assert spec.metadata["base_variable"] == "tip_income"
    assert spec.metadata["measure_mode"] == "indicator_sum"
    assert "count" not in spec.metadata


def test_soi_itemized_deduction_targets_require_itemizing() -> None:
    medical_source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_returns"
    )
    real_estate_source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.all.real_estate_taxes_claims"
    )
    salt_source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_returns"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=medical_source_record_id,
                source_name="irs_soi",
                measure_id="medical_dental_expense_returns",
                value=3_895_410,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
                groupby_dimension="us:statutes/26/62#adjusted_gross_income",
                groupby_value_id="all",
            ),
            _dynamic_ledger_fact(
                source_record_id=real_estate_source_record_id,
                source_name="irs_soi",
                measure_id="real_estate_taxes_claims",
                value=14_258_420,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
                groupby_dimension="us:statutes/26/62#adjusted_gross_income",
                groupby_value_id="all",
            ),
            _dynamic_ledger_fact(
                source_record_id=salt_source_record_id,
                source_name="irs_soi",
                measure_id="limited_state_local_taxes_returns",
                value=14_872_910,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
                groupby_dimension="us:statutes/26/62#adjusted_gross_income",
                groupby_value_id="all",
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    medical_spec = specs[medical_source_record_id]
    assert medical_spec.family == "irs_soi"
    assert medical_spec.metadata["variable"] == "medical_expense_deduction"
    assert medical_spec.metadata["measure_mode"] == "indicator_sum"
    assert medical_spec.metadata["itemized_only"] == "true"
    real_estate_spec = specs[real_estate_source_record_id]
    assert real_estate_spec.family == "irs_soi"
    assert real_estate_spec.metadata["variable"] == "real_estate_taxes"
    assert real_estate_spec.metadata["measure_mode"] == "indicator_sum"
    assert real_estate_spec.metadata["itemized_only"] == "true"
    salt_spec = specs[salt_source_record_id]
    assert salt_spec.family == "irs_soi"
    assert salt_spec.metadata["variable"] == "salt_deduction"
    assert salt_spec.metadata["measure_mode"] == "indicator_sum"
    assert salt_spec.metadata["itemized_only"] == "true"


def test_soi_direct_deduction_amount_targets_expose_model_variables() -> None:
    facts = [
        *packaged_reference_facts(),
        _dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount"
            ),
            source_name="irs_soi",
            measure_id="itemized_deductions_amount",
            value=1_000_000_000_000,
            period_value=2022,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
        ),
        _dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
                "total_itemized_deductions_amount"
            ),
            source_name="irs_soi",
            measure_id="total_itemized_deductions_amount",
            value=1_050_000_000_000,
            period_value=2023,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id="irs_soi.ty2023.table_2_1.itemized_all_returns",
        ),
        _dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2023.table_2_1.itemized_all_returns.all.charitable_amount"
            ),
            source_name="irs_soi",
            measure_id="charitable_amount",
            value=220_000_000_000,
            period_value=2023,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id="irs_soi.ty2023.table_2_1.itemized_all_returns",
        ),
        _dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
                "interest_paid_deduction_amount"
            ),
            source_name="irs_soi",
            measure_id="interest_paid_deduction_amount",
            value=200_000_000_000,
            period_value=2023,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id="irs_soi.ty2023.table_2_1.itemized_all_returns",
        ),
        _dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2022.historic_table_2.us.all."
                "limited_state_local_taxes_amount"
            ),
            source_name="irs_soi",
            measure_id="limited_state_local_taxes_amount",
            value=120_000_000_000,
            period_value=2022,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
        ),
        _dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount"
            ),
            source_name="irs_soi",
            measure_id="medical_dental_expense_amount",
            value=80_000_000_000,
            period_value=2022,
            dimensions={"income_range": "all", "filing_status": "all"},
            layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
        ),
    ]

    registry = compile_us_fiscal_target_registry(facts)

    specs = {spec.name: spec for spec in registry.specs}
    expected = {
        "irs_soi.ty2022.historic_table_2.us.all.itemized_deductions_amount": (
            "itemized_taxable_income_deductions",
            "itemized_deduction_total",
            "true",
        ),
        (
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "total_itemized_deductions_amount"
        ): (
            "itemized_taxable_income_deductions",
            "itemized_deduction_total",
            "true",
        ),
        "irs_soi.ty2023.table_2_1.itemized_all_returns.all.charitable_amount": (
            "charitable_deduction",
            "charitable_deduction_total",
            "true",
        ),
        (
            "irs_soi.ty2023.table_2_1.itemized_all_returns.all."
            "interest_paid_deduction_amount"
        ): (
            "interest_deduction",
            "interest_deduction_total",
            "true",
        ),
        ("irs_soi.ty2022.historic_table_2.us.all.limited_state_local_taxes_amount"): (
            "salt_deduction",
            "salt_deduction_total",
            "true",
        ),
        ("irs_soi.ty2022.historic_table_2.us.all.medical_dental_expense_amount"): (
            "medical_expense_deduction",
            "medical_expense_deduction_total",
            "true",
        ),
    }
    for source_record_id, (variable, role, itemized_only) in expected.items():
        spec = specs[source_record_id]
        assert spec.family == "irs_soi"
        assert spec.metadata["variable"] == variable
        assert spec.metadata["base_variable"] == variable
        assert spec.metadata["measure_mode"] == "sum"
        assert spec.metadata["target_role"] == role
        if itemized_only is None:
            assert "itemized_only" not in spec.metadata
        else:
            assert spec.metadata["itemized_only"] == itemized_only


def test_soi_qbi_historic_table_2_measures_are_not_direct_targets() -> None:
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id="irs_soi.ty2022.historic_table_2.us.all.qbi_amount",
                source_name="irs_soi",
                measure_id="qbi_amount",
                value=31_000_000_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
            ),
            _dynamic_ledger_fact(
                source_record_id="irs_soi.ty2022.historic_table_2.us.all.qbi_claims",
                source_name="irs_soi",
                measure_id="qbi_claims",
                value=3_700_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
                layout_record_set_id="irs_soi.ty2022.historic_table_2.us",
            ),
        ]
    )

    assert "irs_soi.ty2022.historic_table_2.us.all.qbi_amount" not in {
        spec.name for spec in registry.specs
    }
    assert "irs_soi.ty2022.historic_table_2.us.all.qbi_claims" not in {
        spec.name for spec in registry.specs
    }


def test__given_stale_soi_eitc_agi_bucket__then_it_is_not_a_hard_target() -> None:
    source_record_id = (
        "irs_soi.ty2022.table_2_5.eitc_by_agi_children.one_qualifying_child."
        "50k_plus.eitc_total"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="eitc_total",
                value=0,
                period_value=2022,
                dimensions={"income_range": "50k_plus", "filing_status": "all"},
                universe_constraints=[
                    {
                        "variable": "adjusted_gross_income",
                        "operator": ">=",
                        "value": 50_000,
                    },
                ],
                layout_record_set_id=(
                    "irs_soi.ty2022.table_2_5.eitc_by_agi_children.one_qualifying_child"
                ),
            ),
        ],
        target_period=2024,
    )

    specs = {spec.name: spec for spec in registry.specs}
    assert source_record_id not in specs


def test_soi_alias_targets_expose_policyengine_base_variable() -> None:
    source_record_id = (
        "irs_soi.ty2022.historic_table_2.us.all.ordinary_dividends_amount"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="ordinary_dividends_amount",
                value=300_000_000_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "irs_soi"
    assert spec.metadata["variable"] == "ordinary_dividend_income"
    assert spec.metadata["source_variable"] == "ordinary_dividends"
    assert spec.metadata["base_variables"] == (
        "qualified_dividend_income,non_qualified_dividend_income"
    )
    assert "base_variable" not in spec.metadata
    assert spec.metadata["measure_mode"] == "sum"


def test_cbo_net_business_income_uses_source_aligned_policyengine_variable() -> None:
    source_record_id = (
        "cbo.cy2024.income_by_source.net_business_income.projected_amount"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="cbo",
                measure_id="projected_amount",
                groupby_dimension="income_source",
                groupby_value_id="net_business_income",
                value=1_700_000_000_000,
            ),
        ]
    )

    specs = {spec.metadata["ledger_source_record_id"]: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "cbo"
    assert spec.metadata["target_role"] == "cbo_net_business_income"
    assert spec.metadata["base_variable"] == "cbo_net_business_income"
    assert spec.metadata["measure_mode"] == "sum"


def test_soi_return_count_targets_expose_count_mode_without_base_variable() -> None:
    source_record_id = "irs_soi.ty2022.historic_table_2.us.all.return_count"
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="irs_soi",
                measure_id="return_count",
                value=160_000_000,
                period_value=2022,
                dimensions={"income_range": "all", "filing_status": "all"},
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "irs_soi"
    assert spec.metadata["variable"] == "count"
    assert "count" not in spec.metadata
    assert spec.metadata["measure_mode"] == "indicator_sum"
    assert "base_variable" not in spec.metadata
    assert "base_variables" not in spec.metadata


def test_medicare_part_b_premium_reference_uses_gross_premium_income() -> None:
    source_record_id = (
        "cms_medicare.cy2024.part_b_premium_income."
        "premiums_from_enrollees.actual_amount"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=source_record_id,
                source_name="cms_medicare",
                measure_id="actual_amount",
                groupby_value_id="premiums_from_enrollees",
                value=139_837_000_000,
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    spec = specs[source_record_id]
    assert spec.family == "cms_medicare"
    assert spec.metadata["target_role"] == "medicare_part_b_premium_total"
    assert spec.metadata["base_variable"] == "gross_medicare_part_b_premium"
    assert spec.metadata["source_measure_id"] == "actual_amount"


def test_census_pep_population_age_facts_compile_to_count_targets() -> None:
    national_source_record_id = (
        "census_pep.cy2024.national_resident_population_age.0_to_4.population"
    )
    state_source_record_id = (
        "census_pep.v2024.cy2024.state_resident_population.06.5_to_9.population"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=national_source_record_id,
                source_name="census_pep",
                measure_id="population",
                value=18_000_000,
                groupby_value_id="0_to_4",
                universe_constraints=[
                    {"variable": "age", "operator": ">=", "value": 0},
                    {"variable": "age", "operator": "<", "value": 5},
                ],
            ),
            _dynamic_ledger_fact(
                source_record_id=state_source_record_id,
                source_name="census_pep",
                measure_id="population",
                value=2_000_000,
                geography_level="state",
                geography_id="0400000US06",
                groupby_value_id="5_to_9",
                universe_constraints=[
                    {"variable": "age", "operator": ">=", "value": 5},
                    {"variable": "age", "operator": "<", "value": 10},
                ],
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    national = specs[national_source_record_id]
    assert national.family == "census_population"
    assert national.metadata["materializer"] == "population_age"
    assert national.metadata["target_role"] == "population_age"
    assert national.metadata["measure_mode"] == "indicator_sum"
    assert national.metadata["geography_scope"] == "national"
    assert national.metadata["age_lower_bound"] == "0"
    assert national.metadata["age_upper_bound"] == "5"
    assert national.value == 18_000_000

    state = specs[state_source_record_id]
    assert state.family == "census_population"
    assert state.metadata["geography_scope"] == "state"
    assert state.metadata["state_fips"] == "06"
    assert state.metadata["age_lower_bound"] == "5"
    assert state.metadata["age_upper_bound"] == "10"
    assert state.value == 2_000_000


def test_census_pep_population_age_targets_use_latest_source_period() -> None:
    older_source_record_id = (
        "census_pep.cy2023.national_resident_population_age.0_to_4.population"
    )
    latest_source_record_id = (
        "census_pep.cy2024.national_resident_population_age.0_to_4.population"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=older_source_record_id,
                source_name="census_pep",
                measure_id="population",
                value=17_000_000,
                period_value=2023,
                groupby_value_id="0_to_4",
                universe_constraints=[
                    {"variable": "age", "operator": ">=", "value": 0},
                    {"variable": "age", "operator": "<", "value": 5},
                ],
            ),
            _dynamic_ledger_fact(
                source_record_id=latest_source_record_id,
                source_name="census_pep",
                measure_id="population",
                value=18_000_000,
                period_value=2024,
                groupby_value_id="0_to_4",
                universe_constraints=[
                    {"variable": "age", "operator": ">=", "value": 0},
                    {"variable": "age", "operator": "<", "value": 5},
                ],
            ),
        ],
        target_period=2024,
    )

    specs = {spec.name: spec for spec in registry.specs}
    assert older_source_record_id not in specs
    assert specs[latest_source_record_id].value == 18_000_000


def test_census_pep_all_age_population_fact_is_not_age_distribution_target() -> None:
    all_age_source_record_id = (
        "census_pep.cy2024.national_resident_population_age.all.population"
    )
    registry = compile_us_fiscal_target_registry(
        [
            *packaged_reference_facts(),
            _dynamic_ledger_fact(
                source_record_id=all_age_source_record_id,
                source_name="census_pep",
                measure_id="population",
                value=335_000_000,
                groupby_value_id="all",
            ),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs}
    assert all_age_source_record_id not in specs


def test_us_fiscal_requirements_include_reference_program_and_tax_controls() -> None:
    ids = {req.requirement_id for req in US_FISCAL_TARGET_COVERAGE_REQUIREMENTS}
    assert "federal_income_tax_total" in ids
    assert "social_security_total" in ids
    assert "ssi_total" in ids
    assert "snap_total" in ids
    assert "unemployment_compensation_total" in ids
    assert "ssa_social_security_components" in ids
    assert "eitc_total" in ids
    assert "refundable_ctc_total" in ids
    assert "ctc_total" in ids
    assert "aca_marketplace" in ids
    assert "medicaid_spending" not in ids
    assert "medicaid_enrollment" in ids
    assert "medicaid_chip_enrollment" in ids
    assert "irs_agi_distribution" in ids
    assert "state_income_tax" in ids
    assert "population_age_national" in ids
    assert "population_age_state" in ids
    assert REFERENCE_DEDUCTION_TARGET_ROLES <= ids
    for spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        assert f"jct_tax_expenditure:{spec.neutralized_variable}" in ids


def test_structured_income_tax_positive_does_not_satisfy_total_tax() -> None:
    current_like_targets = [
        {
            "name": "nation/cbo/income_tax_positive",
            "measure": "income_tax",
        },
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        current_like_targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert any("federal_income_tax_total" in failure for failure in result.failures)


def test_soi_income_tax_liability_satisfies_total_tax() -> None:
    targets = [
        {
            "name": "irs_soi.cy2024.table_1_1.income_tax_liability_amount",
            "measure": "income_tax",
            "family": "irs_soi",
            "metadata": {"target_role": "federal_income_tax_total"},
        },
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert result.passed


def test_interest_distribution_requirement_rejects_only_broad_state_totals() -> None:
    requirement = next(
        req
        for req in US_FISCAL_TARGET_COVERAGE_REQUIREMENTS
        if req.requirement_id == "irs_interest_distribution"
    )
    state_total_targets = [
        {
            "name": f"irs_soi.ty2024.state_2022.state_{i}.all.taxable_interest_amount",
            "measure": (
                f"irs_soi.ty2024.state_2022.state_{i}.all.taxable_interest_amount"
            ),
            "family": "irs_soi",
            "metadata": {
                "source_measure_id": "taxable_interest_amount",
                "target_role": "soi_fiscal_distribution",
                "agi_lower_bound": "-inf",
                "agi_upper_bound": "inf",
            },
        }
        for i in range(52)
    ]

    result = target_profile_coverage_gate(state_total_targets, [requirement])

    assert not result.passed
    assert any("irs_interest_distribution" in failure for failure in result.failures)


def test_jct_target_name_without_simple_reform_metadata_fails() -> None:
    targets = [
        federal_income_tax_total_row(),
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_population_age_rows(),
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


def test_jct_revenue_loss_targets_do_not_satisfy_deduction_amount_controls() -> None:
    targets = [
        federal_income_tax_total_row(),
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(45),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]

    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )

    assert not result.passed
    for role in REFERENCE_DEDUCTION_TARGET_ROLES:
        assert any(role in failure for failure in result.failures)


def test_state_income_tax_needs_actual_state_surface_not_federal_row() -> None:
    targets = [
        federal_income_tax_total_row(),
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *complete_program_rows(),
        *complete_state_income_tax_rows(43),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]
    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not result.passed
    assert any("state_income_tax" in failure for failure in result.failures)


def test_medicaid_chip_requirement_needs_combined_enrollment_role() -> None:
    targets = [
        federal_income_tax_total_row(),
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *[
            row
            for row in complete_program_rows()
            if row["metadata"]["target_role"] != "medicaid_chip_enrollment"
        ],
        *complete_state_income_tax_rows(45),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]

    result = target_profile_coverage_gate(
        targets,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )

    assert not result.passed
    assert any("medicaid_chip_enrollment" in failure for failure in result.failures)


def test_medicaid_requirement_needs_enrollment_role() -> None:
    base_targets = [
        federal_income_tax_total_row(),
        *complete_agi_distribution_rows(),
        *complete_income_source_rows(),
        *complete_deduction_amount_rows(),
        *complete_state_income_tax_rows(45),
        *complete_population_age_rows(),
        *complete_jct_rows(),
    ]
    program_rows = complete_program_rows()

    without_enrollment = [
        *base_targets,
        *[
            row
            for row in program_rows
            if row["metadata"]["target_role"] != "medicaid_enrollment"
        ],
    ]
    enrollment_result = target_profile_coverage_gate(
        without_enrollment,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    assert not enrollment_result.passed
    assert any(
        "medicaid_enrollment" in failure for failure in enrollment_result.failures
    )


def test_population_age_requirement_needs_complete_national_age_surface() -> None:
    targets = [
        *complete_coverage_targets(),
    ]
    missing_one_national_age = [
        row
        for row in targets
        if row["name"]
        != "census_pep.cy2024.national_resident_population_age.0_to_4.population"
    ]

    result = target_profile_coverage_gate(
        missing_one_national_age,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )

    assert not result.passed
    assert any("population_age_national" in failure for failure in result.failures)


def test_population_age_requirement_needs_complete_state_age_surface() -> None:
    targets = [
        *complete_coverage_targets(),
    ]
    missing_one_state = [
        row
        for row in targets
        if not str(row["name"]).startswith(
            "census_pep.v2024.cy2024.state_resident_population.01."
        )
    ]

    result = target_profile_coverage_gate(
        missing_one_state,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )

    assert not result.passed
    assert any("population_age_state" in failure for failure in result.failures)


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
    aggregation = str(selector.get("aggregation_method") or "sum")
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
        dimensions={"income_range": income_range, "filing_status": "all"},
        universe_constraints=constraints,
        layout_record_set_id=layout_record_set_id
        or f"irs_soi.ty{source_period}.historic_table_2.us",
        groupby_dimension="us:statutes/26/62#adjusted_gross_income",
        groupby_value_id=income_range,
    )


def _soi_income_tax_fact(source_period: int, *, value: float) -> dict[str, object]:
    source_record_id = (
        f"irs_soi.ty{source_period}.table_3_3.us.all.income_tax_liability_amount"
    )
    return {
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
        "geography": {"level": "country", "id": "0100000US"},
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": f"irs_soi.ty{source_period}.table_3_3",
            "groupby_dimension": "adjusted_gross_income",
            "groupby_value_id": "all",
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
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:cbo-income-tax-{source_period}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:cbo-income-tax-{source_period}",
        "legacy_fact_key": f"ledger.fact.v1:cbo-income-tax-{source_period}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "fiscal_year", "value": source_period},
        "entity": {"name": "household"},
        "aggregation": {"method": "sum"},
        "geography": {"level": "country", "id": "0100000US"},
        "dimensions": {},
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": f"cbo.fy{source_period}.revenues",
            "groupby_dimension": "revenue_source",
            "groupby_value_id": "individual_income_taxes",
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
) -> dict[str, object]:
    normalized_period = source_period.replace("-", "_")
    source_record_id = (
        f"cms_medicaid.month{normalized_period}.us.total_medicaid_chip_enrollment"
    )
    return {
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
        "geography": {"level": "country", "id": "0100000US"},
        "dimensions": {},
        "universe_constraints": {"constraints": []},
        "layout": {
            "record_set_id": f"cms_medicaid.month{normalized_period}",
            "groupby_dimension": "program",
            "groupby_value_id": "total_medicaid_chip_enrollment",
            "measure_id": "total_medicaid_chip_enrollment",
        },
        "observed_measure": {
            "source_name": "cms_medicaid",
            "source_table": "Medicaid and CHIP enrollment",
            "source_measure_id": "total_medicaid_chip_enrollment",
            "source_concept": "cms.total_medicaid_chip_enrollment",
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
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{fact_id}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{fact_id}",
        "legacy_fact_key": f"ledger.fact.v1:{fact_id}",
        "lineage": {"source_record_id": source_record_id},
        "value": value,
        "period": {"type": "calendar_year", "value": period_value},
        "entity": {"name": "person"},
        "aggregation": {"method": "sum"},
        "geography": {"level": geography_level, "id": geography_id},
        "dimensions": dict(dimensions or {}),
        "universe_constraints": {"constraints": list(universe_constraints or [])},
        "layout": {
            "record_set_id": layout_record_set_id or f"{source_name}.record_set",
            "groupby_dimension": groupby_dimension,
            "groupby_value_id": groupby_value_id,
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
        elif role in {"medicaid_enrollment", "medicaid_chip_enrollment"}:
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
