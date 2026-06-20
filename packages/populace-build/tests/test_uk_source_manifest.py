"""UK raw-source plan declaration: full surface or nothing."""

from __future__ import annotations

import pytest

from populace.build.source_manifest import SourceManifest, SourceOperationSpec
from populace.build.uk import (
    AREA_TYPES,
    FRS_ONLY_SPI_FILL_PERSON_COLUMNS,
    ROWWISE_GEOGRAPHY_COLUMNS,
    SPI_INCOME_IMPUTATION_COLUMNS,
    UK_DONORS,
    UK_NONNEGATIVE_SOURCE_OUTPUTS,
    UK_REWRITTEN_SOURCE_OUTPUT_STAGES,
    UK_SOURCE_MANIFEST,
    UK_SOURCE_OUTPUTS,
    UK_SOURCE_OUTPUT_STAGES,
    UK_SOURCE_STAGE_SPECS,
    UK_SPI_SUPPORT_STAGE_NAME,
    UK_STAGE_NAMES,
    UK_STRUCTURAL_SOURCE_STAGES,
    uk_plan,
)


def _noop_implementations() -> dict:
    return {name: (lambda frame: frame) for name in UK_STAGE_NAMES}


class TestUkPlan:
    def test_assembles_with_all_stages_and_donor_citations(self) -> None:
        plan = uk_plan(_noop_implementations())

        assert tuple(stage.name for stage in plan.stages) == UK_STAGE_NAMES
        donor_stages = dict(plan.donors())
        assert set(donor_stages) == set(UK_DONORS)
        for spec in donor_stages.values():
            assert spec.source.startswith("https://")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        implementations = _noop_implementations()
        del implementations["was_wealth"]

        with pytest.raises(ValueError, match="missing \\['was_wealth'\\]"):
            uk_plan(implementations)

    def test_unknown_stage_is_refused(self) -> None:
        implementations = _noop_implementations()
        implementations["legacy_fill"] = lambda frame: frame

        with pytest.raises(ValueError, match="Unknown stage implementation"):
            uk_plan(implementations)


class TestUkSources:
    def test_source_manifest_loads_as_spec_contract(self) -> None:
        assert UK_SOURCE_MANIFEST.country == "uk"
        assert UK_SOURCE_MANIFEST.version == 1
        assert len(UK_SOURCE_STAGE_SPECS) >= len(UK_DONORS)

    def test_every_donor_stage_has_matching_source_spec(self) -> None:
        specs = UK_SOURCE_MANIFEST.stage_map()
        for stage, donor in UK_DONORS.items():
            assert stage in specs
            assert specs[stage].survey == donor.survey
            assert specs[stage].source == donor.source

    def test_source_specs_align_with_declared_plan(self) -> None:
        source_stage_names = {spec.stage for spec in UK_SOURCE_STAGE_SPECS}

        assert UK_STAGE_NAMES == UK_SOURCE_MANIFEST.plan_stages
        assert set(UK_SOURCE_MANIFEST.stage_map()) == source_stage_names
        assert source_stage_names == set(UK_DONORS) | set(UK_STRUCTURAL_SOURCE_STAGES)
        assert source_stage_names.issubset(UK_STAGE_NAMES)
        assert tuple(spec.stage for spec in UK_SOURCE_STAGE_SPECS) == tuple(
            name for name in UK_STAGE_NAMES if name in source_stage_names
        )
        assert UK_STAGE_NAMES.index("rowwise_oa_geography") < UK_STAGE_NAMES.index(
            "local_geography_weights"
        )

    def test_donor_and_structural_stage_groups_are_manifest_derived(self) -> None:
        donor_stage_names = tuple(
            spec.stage for spec in UK_SOURCE_STAGE_SPECS if spec.role == "donor"
        )
        structural_stage_names = tuple(
            spec.stage for spec in UK_SOURCE_STAGE_SPECS if spec.role != "donor"
        )

        assert tuple(UK_DONORS) == donor_stage_names
        assert UK_STRUCTURAL_SOURCE_STAGES == structural_stage_names
        assert "national_calibration" in UK_STRUCTURAL_SOURCE_STAGES
        assert "local_geography_weights" in UK_STRUCTURAL_SOURCE_STAGES

    def test_stage_order_keeps_required_upstream_surfaces_available(self) -> None:
        assert UK_STAGE_NAMES.index("was_wealth") < UK_STAGE_NAMES.index(
            "regional_property_uprating"
        )
        assert UK_STAGE_NAMES.index("was_wealth") < UK_STAGE_NAMES.index(
            "lcfs_consumption"
        )
        assert UK_STAGE_NAMES.index(UK_SPI_SUPPORT_STAGE_NAME) < UK_STAGE_NAMES.index(
            "spi_income"
        )
        assert UK_STAGE_NAMES.index("spi_income") < UK_STAGE_NAMES.index(
            "frs_only_spi_fill"
        )
        assert UK_STAGE_NAMES.index("local_geography_weights") < UK_STAGE_NAMES.index(
            "rail_public_service_calibration"
        )
        assert UK_STAGE_NAMES.index("local_geography_weights") < UK_STAGE_NAMES.index(
            "road_fuel_energy_calibration"
        )

    def test_source_specs_are_manifest_only_not_python_loaders(self) -> None:
        for spec in UK_SOURCE_STAGE_SPECS:
            assert spec.operations
            for operation in spec.operations:
                assert "module" not in operation.parameters
                assert "function" not in operation.parameters
                assert operation.kind not in {
                    "python_module",
                    "python_function",
                    "import_module",
                }

    def test_weight_calibration_stages_are_manifest_declared(self) -> None:
        specs = UK_SOURCE_MANIFEST.stage_map()
        for stage in ("national_calibration", "local_geography_weights"):
            artifact_kinds = {artifact["kind"] for artifact in specs[stage].artifacts}
            kinds = [operation.kind for operation in specs[stage].operations]
            compile_operation = next(
                operation
                for operation in specs[stage].operations
                if operation.kind == "compile_ledger_targets"
            )

            assert specs[stage].source == "https://github.com/PolicyEngine/arch-data"
            assert artifact_kinds == {"ledger_consumer_facts"}
            assert "target_registry" not in artifact_kinds
            assert "target_tables" not in artifact_kinds
            assert kinds.index("read_table") < kinds.index("compile_ledger_targets")
            assert kinds.index("compile_ledger_targets") < kinds.index(
                "calibrate_weights"
            )
            assert "calibrate_weights" in kinds
            assert compile_operation.parameters["country"] == "uk"

        assert (
            next(
                operation
                for operation in specs["national_calibration"].operations
                if operation.kind == "compile_ledger_targets"
            ).parameters["target_profile"]
            == "uk_national_calibration"
        )
        assert (
            local_compile_operation := next(
                operation
                for operation in specs["local_geography_weights"].operations
                if operation.kind == "compile_ledger_targets"
            )
        ).parameters["target_profile"] == "uk_local_geography"
        assert tuple(local_compile_operation.parameters["area_types"]) == AREA_TYPES

    def test_raw_source_surface_declares_salient_outputs_from_each_input(self) -> None:
        required_outputs = {
            "property_wealth",
            "mortgage_debt",
            "consumer_debt",
            "student_loan_balance",
            "num_vehicles",
            "full_rate_vat_expenditure_rate",
            "food_and_non_alcoholic_beverages_consumption",
            "electricity_consumption",
            "gas_consumption",
            "petrol_spending",
            "diesel_spending",
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
            "local_geography_weight",
        }

        required_outputs.update(SPI_INCOME_IMPUTATION_COLUMNS)
        required_outputs.update(FRS_ONLY_SPI_FILL_PERSON_COLUMNS)
        required_outputs.update(ROWWISE_GEOGRAPHY_COLUMNS)

        assert sorted(required_outputs - UK_SOURCE_OUTPUTS) == []

    def test_nonnegative_surface_covers_key_money_and_count_outputs(self) -> None:
        required_nonnegative = {
            "owned_land",
            "property_wealth",
            "mortgage_debt",
            "consumer_debt",
            "student_loan_balance",
            "food_and_non_alcoholic_beverages_consumption",
            "electricity_consumption",
            "gas_consumption",
            "petrol_spending",
            "diesel_spending",
            "full_rate_vat_expenditure_rate",
            "a_and_e_visits",
            "nhs_spending",
            "dfe_education_spending",
            "rail_usage",
            "gift_aid",
            "charitable_investment_gifts",
            "capital_gains",
            "pension_contributions_via_salary_sacrifice",
            "local_geography_weight",
        }

        assert sorted(required_nonnegative - UK_NONNEGATIVE_SOURCE_OUTPUTS) == []
        assert "student_loan_plan" not in UK_NONNEGATIVE_SOURCE_OUTPUTS

    def test_rewritten_outputs_are_explicit_and_have_reviewed_final_writers(
        self,
    ) -> None:
        expected_rewrites = {
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
            "household_weight": (
                "frs_base",
                "national_calibration",
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

        assert dict(UK_REWRITTEN_SOURCE_OUTPUT_STAGES) == expected_rewrites
        for output, stages in expected_rewrites.items():
            assert UK_SOURCE_OUTPUT_STAGES[output] == stages
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
