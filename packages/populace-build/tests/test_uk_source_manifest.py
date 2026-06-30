"""UK source manifest contract: spec-only resource, full surface."""

from __future__ import annotations

import json
from collections import defaultdict
from importlib.resources import files

import pytest

from populace.build.source_manifest import (
    ALLOWED_SOURCE_OPERATION_KINDS,
    SourceManifest,
    SourceOperationSpec,
    load_source_manifest,
)
from populace.build.uk_runtime import (
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    ROWWISE_GEOGRAPHY_COLUMNS,
    SPI_INCOME_IMPUTATION_COLUMNS,
    UK_SPI_SUPPORT_STAGE_NAME,
)

UK_RESOURCE_ROOT = files("populace.build.uk")
UK_SOURCE_RESOURCE = UK_RESOURCE_ROOT.joinpath("source_stages.json")
UK_SOURCE_MANIFEST = load_source_manifest(UK_SOURCE_RESOURCE)
UK_SOURCE_STAGE_SPECS = UK_SOURCE_MANIFEST.stages
UK_STAGE_NAMES = tuple(stage.stage for stage in UK_SOURCE_STAGE_SPECS)
UK_SOURCE_OUTPUT_STAGES: dict[str, list[str]] = defaultdict(list)
for _stage in UK_SOURCE_STAGE_SPECS:
    for _output in _stage.outputs:
        UK_SOURCE_OUTPUT_STAGES[_output].append(_stage.stage)
UK_SOURCE_OUTPUT_STAGES = dict(UK_SOURCE_OUTPUT_STAGES)
UK_SOURCE_OUTPUTS = set(UK_SOURCE_OUTPUT_STAGES)
UK_NONNEGATIVE_SOURCE_OUTPUTS = {
    output for stage in UK_SOURCE_STAGE_SPECS for output in stage.nonnegative_outputs
}
UK_REWRITTEN_SOURCE_OUTPUT_STAGES = {
    output: tuple(stages)
    for output, stages in UK_SOURCE_OUTPUT_STAGES.items()
    if len(stages) > 1
}


class TestUkSources:
    def test_source_manifest_loads_as_spec_contract(self) -> None:
        assert UK_SOURCE_MANIFEST.country == "uk"
        assert UK_SOURCE_MANIFEST.version == 1
        assert len(UK_SOURCE_STAGE_SPECS) >= 12
        assert "source_stages.json" in _country_package_resources()

    def test_source_specs_align_with_declared_resource_order(self) -> None:
        raw = json.loads(UK_SOURCE_RESOURCE.read_text(encoding="utf-8"))

        assert UK_STAGE_NAMES == tuple(stage["stage"] for stage in raw["stages"])
        assert set(UK_SOURCE_MANIFEST.stage_map()) == set(UK_STAGE_NAMES)
        assert "rowwise_oa_geography" in UK_STAGE_NAMES
        assert "local_geography_weights" not in UK_STAGE_NAMES
        assert "national_calibration" not in UK_STAGE_NAMES

    def test_stage_order_keeps_required_upstream_surfaces_available(self) -> None:
        assert UK_STAGE_NAMES.index("was_wealth") < UK_STAGE_NAMES.index(
            "regional_property_uprating"
        )
        assert UK_STAGE_NAMES.index("was_wealth") < UK_STAGE_NAMES.index(
            "lcfs_consumption"
        )
        assert UK_STAGE_NAMES.index("lcfs_consumption") < UK_STAGE_NAMES.index(
            "bus_public_service_calibration"
        )
        assert UK_STAGE_NAMES.index("etb_public_services") < UK_STAGE_NAMES.index(
            "bus_public_service_calibration"
        )
        assert UK_STAGE_NAMES.index(UK_SPI_SUPPORT_STAGE_NAME) < UK_STAGE_NAMES.index(
            "spi_income"
        )
        assert UK_STAGE_NAMES.index("spi_income") < UK_STAGE_NAMES.index(
            "frs_only_spi_fill"
        )

    def test_source_specs_are_manifest_only_not_python_loaders(self) -> None:
        for spec in UK_SOURCE_STAGE_SPECS:
            assert spec.operations
            for operation in spec.operations:
                assert operation.kind in ALLOWED_SOURCE_OPERATION_KINDS
                assert "module" not in operation.parameters
                assert "function" not in operation.parameters
                assert operation.kind not in {
                    "python_module",
                    "python_function",
                    "import_module",
                }

    def test_ledger_weight_calibration_is_not_declared_as_source_operations(
        self,
    ) -> None:
        operation_kinds = {
            operation.kind
            for stage in UK_SOURCE_STAGE_SPECS
            for operation in stage.operations
        }

        assert "compile_ledger_targets" not in operation_kinds
        assert "calibrate_weights" not in operation_kinds

    def test_raw_source_surface_declares_salient_outputs_from_each_input(
        self,
    ) -> None:
        required_outputs = {
            "employment_sector",
            "sic_industry_division",
            "property_wealth",
            "mortgage_debt",
            "consumer_debt",
            "student_loan_balance",
            "num_vehicles",
            "cash_isa",
            "stocks_and_shares_isa",
            "full_rate_vat_expenditure_rate",
            "food_and_non_alcoholic_beverages_consumption",
            "electricity_consumption",
            "gas_consumption",
            "petrol_spending",
            "diesel_spending",
            "bus_fare_spending",
            "dfe_education_spending",
            "rail_subsidy_spending",
            "bus_subsidy_spending",
            "rail_usage",
            "a_and_e_visits",
            "admitted_patient_visits",
            "outpatient_visits",
            "nhs_spending",
            "gift_aid",
            "charitable_investment_gifts",
            "capital_gains",
            "household_is_capital_gains_clone",
            "pension_contributions_via_salary_sacrifice",
            "student_loan_plan",
            "household_is_spi_synthetic",
            "source_household_key",
        }

        required_outputs.update(SPI_INCOME_IMPUTATION_COLUMNS)
        required_outputs.update(FRS_ONLY_SPI_FILL_PERSON_COLUMNS)
        required_outputs.update(ROWWISE_GEOGRAPHY_COLUMNS)

        assert sorted(required_outputs - UK_SOURCE_OUTPUTS) == []

    def test_nonnegative_surface_covers_key_money_and_count_outputs(self) -> None:
        required_nonnegative = {
            "sic_industry_division",
            "owned_land",
            "property_wealth",
            "mortgage_debt",
            "consumer_debt",
            "student_loan_balance",
            "cash_isa",
            "stocks_and_shares_isa",
            "food_and_non_alcoholic_beverages_consumption",
            "electricity_consumption",
            "gas_consumption",
            "petrol_spending",
            "diesel_spending",
            "bus_fare_spending",
            "bus_subsidy_spending",
            "full_rate_vat_expenditure_rate",
            "a_and_e_visits",
            "nhs_spending",
            "dfe_education_spending",
            "rail_usage",
            "gift_aid",
            "charitable_investment_gifts",
            "capital_gains",
            "pension_contributions_via_salary_sacrifice",
        }

        assert sorted(required_nonnegative - UK_NONNEGATIVE_SOURCE_OUTPUTS) == []
        assert "employment_sector" not in UK_NONNEGATIVE_SOURCE_OUTPUTS
        assert "student_loan_plan" not in UK_NONNEGATIVE_SOURCE_OUTPUTS

    def test_rewritten_outputs_are_explicit_and_have_reviewed_final_writers(
        self,
    ) -> None:
        expected_rewrites = {
            "bus_fare_spending": (
                "lcfs_consumption",
                "bus_public_service_calibration",
            ),
            "bus_subsidy_spending": (
                "etb_public_services",
                "bus_public_service_calibration",
            ),
            "diesel_spending": (
                "lcfs_consumption",
                "road_fuel_energy_calibration",
            ),
            "domestic_energy_consumption": (
                "lcfs_consumption",
                "road_fuel_energy_calibration",
            ),
            "electricity_consumption": (
                "lcfs_consumption",
                "road_fuel_energy_calibration",
            ),
            "gas_consumption": (
                "lcfs_consumption",
                "road_fuel_energy_calibration",
            ),
            "petrol_spending": (
                "lcfs_consumption",
                "road_fuel_energy_calibration",
            ),
            "main_residence_value": (
                "was_wealth",
                "regional_property_uprating",
            ),
            "property_wealth": (
                "was_wealth",
                "regional_property_uprating",
            ),
            "employment_income": (
                "frs_base",
                "spi_income",
            ),
            "private_pension_income": (
                "frs_base",
                "spi_income",
            ),
            "self_employment_income": (
                "frs_base",
                "spi_income",
            ),
            "employee_pension_contributions": (
                "frs_only_spi_fill",
                "frs_salary_sacrifice",
            ),
            "pension_contributions_via_salary_sacrifice": (
                "frs_only_spi_fill",
                "frs_salary_sacrifice",
            ),
            "rail_subsidy_spending": (
                "etb_public_services",
                "rail_public_service_calibration",
            ),
            "rail_usage": (
                "etb_public_services",
                "rail_public_service_calibration",
            ),
        }

        assert UK_REWRITTEN_SOURCE_OUTPUT_STAGES == expected_rewrites
        for output, stages in expected_rewrites.items():
            assert tuple(UK_SOURCE_OUTPUT_STAGES[output]) == stages
            indices = [UK_STAGE_NAMES.index(stage) for stage in stages]
            assert indices == sorted(indices)

    def test_fuel_energy_amount_scaling_is_not_binary_assignment(self) -> None:
        operations = UK_SOURCE_MANIFEST.stage_map()[
            "road_fuel_energy_calibration"
        ].operations
        kinds = [operation.kind for operation in operations]

        assert "calibrate_binary_assignment" not in kinds
        assert "uprate" in kinds
        uprate = operations[kinds.index("uprate")]
        assert tuple(uprate.parameters["variables"]) == (
            "petrol_spending",
            "diesel_spending",
            "electricity_consumption",
            "gas_consumption",
        )
        derive = operations[kinds.index("derive")]
        assert tuple(derive.parameters["outputs"]) == ("domestic_energy_consumption",)

    def test_bus_surface_matches_recent_uk_data_contract(self) -> None:
        specs = UK_SOURCE_MANIFEST.stage_map()
        lcfs_operations = specs["lcfs_consumption"].operations
        lcfs_derive = next(
            operation
            for operation in lcfs_operations
            if operation.kind == "derive"
            and "bus_fare_spending" in operation.parameters["outputs"]
        )
        bus_operations = specs["bus_public_service_calibration"].operations
        kinds = [operation.kind for operation in bus_operations]

        assert tuple(lcfs_derive.parameters["source_codes"]["bus_fare_spending"]) == (
            "c73212",
            "c73213",
            "c73214",
        )
        assert kinds == ["read_table", "uprate"]
        assert tuple(bus_operations[1].parameters["variables"]) == (
            "bus_fare_spending",
            "bus_subsidy_spending",
        )

    def test_wealth_surface_splits_isa_outputs_and_preserves_back_compat(
        self,
    ) -> None:
        stage = UK_SOURCE_MANIFEST.stage_map()["was_wealth"]
        operations = stage.operations
        derive = next(
            operation
            for operation in operations
            if operation.kind == "derive"
            and "cash_isa" in operation.parameters["outputs"]
        )
        folds = [operation for operation in operations if operation.kind == "fold_into"]

        assert {
            "cash_isa",
            "stocks_and_shares_isa",
        } <= set(stage.outputs)
        assert derive.parameters["source_fields"] == {
            "cash_isa": "DVCISAVR8",
            "stocks_and_shares_isa": "DVIISAVR8",
        }
        assert any(
            operation.parameters
            == {
                "target": "corporate_wealth",
                "amount": "stocks_and_shares_isa",
            }
            for operation in folds
        )

    def test_frs_base_carries_employment_sector_and_sic_from_raw_frs(self) -> None:
        stage = UK_SOURCE_MANIFEST.stage_map()["frs_base"]

        assert {"employment_sector", "sic_industry_division"} <= set(stage.outputs)
        assert "sic_industry_division" in stage.nonnegative_outputs

    def test_spi_stage_declares_support_channel_before_income_fit(self) -> None:
        specs = UK_SOURCE_MANIFEST.stage_map()
        spi_kinds = [operation.kind for operation in specs["spi_income"].operations]

        assert spi_kinds.index("read_table") < spi_kinds.index("fit_weighted_qrf")
        assert spi_kinds.index("fit_weighted_qrf") < spi_kinds.index("support_clip")
        assert "household_is_spi_synthetic" in specs[UK_SPI_SUPPORT_STAGE_NAME].outputs

    def test_source_operation_parser_rejects_python_loader_shapes(self) -> None:
        with pytest.raises(ValueError, match="executable-loader"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "python_module",
                    "module": "populace.build.uk.sources",
                    "function": "add_was_wealth",
                }
            )

    def test_source_operation_parser_rejects_old_weight_calibration_ops(self) -> None:
        with pytest.raises(ValueError, match="allowed manifest operation vocabulary"):
            SourceOperationSpec.from_mapping({"kind": "compile_ledger_targets"})

        with pytest.raises(ValueError, match="allowed manifest operation vocabulary"):
            SourceOperationSpec.from_mapping({"kind": "calibrate_weights"})

    def test_source_manifest_parser_rejects_incumbent_package_artifacts(self) -> None:
        with pytest.raises(ValueError, match="forbidden incumbent dependency"):
            SourceManifest.from_mapping(
                {
                    "version": 1,
                    "country": "uk",
                    "policy": "spec only",
                    "stages": [
                        {
                            "stage": "was_wealth",
                            "survey": "Wealth and Assets Survey",
                            "source": "https://example.test/was",
                            "grain": "household",
                            "artifacts": [
                                {
                                    "kind": "derived_dataset",
                                    "locator": "policyengine_" + "uk_data",
                                }
                            ],
                            "operations": [
                                {"kind": "read_table", "table": "was_household"}
                            ],
                            "outputs": ["property_wealth"],
                        }
                    ],
                }
            )


def _country_package_resources() -> set[str]:
    package = json.loads(
        UK_RESOURCE_ROOT.joinpath("country_package.json").read_text(encoding="utf-8")
    )
    return set(package["resources"])
