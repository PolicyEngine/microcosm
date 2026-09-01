"""The US plan declaration: complete or nothing, every donor cited."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build.source_manifest import (
    SourceManifest,
    SourceOperationSpec,
    SupportSpineSourceSpec,
    SupportSpineSpec,
)
from microcosm.build.us_runtime import (
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES,
    SIPP_HEAD_START_FIT_PARAMETERS,
    SIPP_HEAD_START_READ_PARAMETERS,
    SIPP_SSI_DISABILITY_FIT_PARAMETERS,
    SIPP_SSI_DISABILITY_READ_PARAMETERS,
    SSI_TAKE_UP_SSA_SOURCE_URL,
    US_CHILD_SUPPORT_STAGE_NAME,
    US_CHILDCARE_STAGE_NAME,
    US_DISABILITY_BENEFITS_STAGE_NAME,
    US_DONORS,
    US_EDUCATION_INPUTS_STAGE_NAME,
    US_ENERGY_SUBSIDY_STAGE_NAME,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_OTHER_HEALTH_INSURANCE_STAGE_NAME,
    US_PRIOR_YEAR_INCOME_STAGE_NAME,
    US_PUF_SUPPORT_STAGE_NAME,
    US_QBI_OUTPUT_COLUMNS,
    US_RETIREMENT_CONTRIBUTION_STAGE_NAME,
    US_SIPP_HEAD_START_STAGE_NAME,
    US_SOURCE_MANIFEST,
    US_SOURCE_STAGE_SPECS,
    US_SSI_DISABILITY_CRITERIA_STAGE_NAME,
    US_SSI_TAKE_UP_ANCHOR,
    US_SSI_TAKE_UP_STAGE_NAME,
    US_STAGE_NAMES,
    US_SUPPORT_SPINE_MANIFEST,
    US_SUPPORT_SPINE_SPEC,
    US_VOLUNTARY_FILING_STAGE_NAME,
    US_WEEKS_UNEMPLOYED_STAGE_NAME,
    US_WORKERS_COMPENSATION_STAGE_NAME,
    WEEKS_UNEMPLOYED_DERIVE_PARAMETERS,
    WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS,
    WEEKS_UNEMPLOYED_READ_PARAMETERS,
    BuildConfig,
    us_plan,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
)

ROOT = Path(__file__).resolve().parents[3]


def _noop_implementations() -> dict:
    return {name: (lambda frame: frame) for name in US_STAGE_NAMES}


class TestUsPlan:
    def test_assembles_with_all_stages_and_donor_citations(self) -> None:
        plan = us_plan(_noop_implementations())
        assert tuple(stage.name for stage in plan.stages) == US_STAGE_NAMES
        donor_stages = dict(plan.donors())
        # every declared donor is attached to its stage
        assert set(donor_stages) == set(US_DONORS)
        for spec in donor_stages.values():
            assert spec.source.startswith("https://")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        implementations = _noop_implementations()
        del implementations["org_wages"]
        with pytest.raises(ValueError, match="missing \\['org_wages'\\]"):
            us_plan(implementations)

    def test_unknown_stage_is_refused(self) -> None:
        implementations = _noop_implementations()
        implementations["org_wages_fallback"] = lambda frame: frame
        with pytest.raises(ValueError, match="Unknown stage implementation"):
            us_plan(implementations)

    def test_no_incumbent_dataset_anywhere_in_the_donor_graph(self) -> None:
        """Incumbent production datasets are benchmarks, never build inputs."""
        for spec in US_DONORS.values():
            text = f"{spec.survey} {spec.source} {spec.notes}".lower()
            assert "production dataset" not in text
            assert "benchmark" not in text


class TestBuildConfig:
    def test_manifest_round_trip(self) -> None:
        config = BuildConfig(
            year=2024,
            seed=0,
            max_weight_ratio=50.0,
            registry_path="registry/us-2024.json",
            extra={"scf_vintage": 2022},
        )
        manifest = config.to_manifest()
        assert manifest["max_weight_ratio"] == 50.0
        assert manifest["registry_path"] == "registry/us-2024.json"
        assert manifest["extra"] == {"scf_vintage": 2022}

    def test_bad_knobs_refused(self) -> None:
        with pytest.raises(ValueError, match="max_weight_ratio"):
            BuildConfig(year=2024, max_weight_ratio=0.0)
        with pytest.raises(ValueError, match="mass"):
            BuildConfig(year=2024, mass="leaky")
        with pytest.raises(ValueError, match="survey year"):
            BuildConfig(year=1900)


class TestUsSources:
    def test_source_manifest_loads_as_spec_contract(self) -> None:
        assert US_SOURCE_MANIFEST.country == "us"
        assert US_SOURCE_MANIFEST.version == 1
        assert len(US_SOURCE_STAGE_SPECS) >= len(US_DONORS)

    def test_support_spine_manifest_declares_period_relative_asec_pool(self) -> None:
        assert US_SUPPORT_SPINE_MANIFEST.country == "us"
        assert US_SUPPORT_SPINE_SPEC.stage == "asec_load"
        assert US_SUPPORT_SPINE_SPEC.method == "pool_raw_asec_years"
        assert US_SUPPORT_SPINE_SPEC.target_year_from_build_config is True

        sources = US_SUPPORT_SPINE_SPEC.sources
        assert [source.role for source in sources] == [
            "current_asec",
            "prior_asec",
        ]
        assert [source.source_year_offset for source in sources] == [0, -1]
        assert [source.share for source in sources] == [0.5, 0.5]
        assert [source.resolved_year(2024) for source in sources] == [2024, 2023]
        assert all(source.source.startswith("https://") for source in sources)

    def test_support_spine_parser_rejects_non_period_relative_sources(self) -> None:
        with pytest.raises(ValueError, match="target_year_from_build_config"):
            SupportSpineSpec.from_mapping(
                {
                    "stage": "asec_load",
                    "method": "pool_raw_asec_years",
                    "target_year_from_build_config": False,
                    "sources": [
                        {
                            "role": "current",
                            "survey": "CPS ASEC",
                            "source": "https://www.census.gov/programs-surveys/cps.html",
                            "source_year_offset": 0,
                        }
                    ],
                }
            )

    def test_support_spine_parser_requires_explicit_source_shares(self) -> None:
        with pytest.raises(ValueError, match="explicit shares"):
            SupportSpineSpec.from_mapping(
                {
                    "stage": "asec_load",
                    "method": "pool_raw_asec_years",
                    "target_year_from_build_config": True,
                    "sources": [
                        {
                            "role": "current",
                            "survey": "CPS ASEC",
                            "source": "https://www.census.gov/programs-surveys/cps.html",
                            "source_year_offset": 0,
                        }
                    ],
                }
            )

    def test_support_spine_parser_rejects_boolean_numeric_fields(self) -> None:
        with pytest.raises(ValueError, match="source_year_offset"):
            SupportSpineSourceSpec.from_mapping(
                {
                    "role": "current",
                    "survey": "CPS ASEC",
                    "source": "https://www.census.gov/programs-surveys/cps.html",
                    "source_year_offset": True,
                }
            )
        with pytest.raises(ValueError, match="share"):
            SupportSpineSourceSpec.from_mapping(
                {
                    "role": "current",
                    "survey": "CPS ASEC",
                    "source": "https://www.census.gov/programs-surveys/cps.html",
                    "source_year_offset": 0,
                    "share": True,
                }
            )

    def test_every_donor_stage_has_matching_source_spec(self) -> None:
        specs = US_SOURCE_MANIFEST.stage_map()
        for stage, donor in US_DONORS.items():
            assert stage in specs
            assert specs[stage].survey == donor.survey
            assert specs[stage].source == donor.source

    def test_source_specs_align_with_declared_plan(self) -> None:
        derived_source_specs = {"mortgage_conversion"}
        frame_structural_stages = {US_PUF_SUPPORT_STAGE_NAME}
        assert {spec.stage for spec in US_SOURCE_STAGE_SPECS} == set(
            US_DONORS
        ) | derived_source_specs
        assert set(US_DONORS).issubset(US_STAGE_NAMES)
        assert derived_source_specs.issubset(US_STAGE_NAMES)
        assert frame_structural_stages.issubset(US_STAGE_NAMES)

    def test_puf_support_channel_precedes_puf_detail_donor_stage(self) -> None:
        assert US_STAGE_NAMES.index(
            US_PRIOR_YEAR_INCOME_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        assert US_STAGE_NAMES.index(
            US_RETIREMENT_CONTRIBUTION_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        assert US_STAGE_NAMES.index(US_CHILDCARE_STAGE_NAME) < US_STAGE_NAMES.index(
            US_PUF_SUPPORT_STAGE_NAME
        )
        assert US_STAGE_NAMES.index(
            US_ENERGY_SUBSIDY_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        assert US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME) < US_STAGE_NAMES.index(
            "puf_tax_detail"
        )
        assert US_STAGE_NAMES.index("puf_tax_detail") < US_STAGE_NAMES.index(
            US_CHILD_SUPPORT_STAGE_NAME
        )
        assert US_STAGE_NAMES.index(US_CHILD_SUPPORT_STAGE_NAME) < US_STAGE_NAMES.index(
            US_DISABILITY_BENEFITS_STAGE_NAME
        )
        assert US_STAGE_NAMES.index(
            US_DISABILITY_BENEFITS_STAGE_NAME
        ) < US_STAGE_NAMES.index("education_inputs")
        assert US_STAGE_NAMES.index(
            US_WORKERS_COMPENSATION_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_WEEKS_UNEMPLOYED_STAGE_NAME)
        assert US_STAGE_NAMES.index(
            US_WEEKS_UNEMPLOYED_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_EDUCATION_INPUTS_STAGE_NAME)
        assert US_STAGE_NAMES.index("medicaid_take_up") < US_STAGE_NAMES.index(
            US_OTHER_HEALTH_INSURANCE_STAGE_NAME
        )
        assert US_STAGE_NAMES.index(
            US_OTHER_HEALTH_INSURANCE_STAGE_NAME
        ) < US_STAGE_NAMES.index("export")
        assert US_STAGE_NAMES.index("vehicle_assets") < US_STAGE_NAMES.index(
            US_VOLUNTARY_FILING_STAGE_NAME
        )
        assert US_STAGE_NAMES.index(
            US_VOLUNTARY_FILING_STAGE_NAME
        ) < US_STAGE_NAMES.index("entity_placement")
        assert US_STAGE_NAMES.index("scf_wealth") < US_STAGE_NAMES.index(
            US_SSI_DISABILITY_CRITERIA_STAGE_NAME
        )
        assert US_STAGE_NAMES.index(
            US_SSI_DISABILITY_CRITERIA_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_SIPP_HEAD_START_STAGE_NAME)
        assert US_STAGE_NAMES.index(
            US_SIPP_HEAD_START_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_SSI_TAKE_UP_STAGE_NAME)
        assert US_STAGE_NAMES.index(US_SSI_TAKE_UP_STAGE_NAME) < US_STAGE_NAMES.index(
            "sipp_tips"
        )

    def test_ssi_disability_criteria_stage_pins_sipp_model_contract(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_SSI_DISABILITY_CRITERIA_STAGE_NAME]
        donor = US_DONORS[US_SSI_DISABILITY_CRITERIA_STAGE_NAME]

        assert stage.survey == donor.survey == "Census SIPP"
        assert stage.source == donor.source
        assert stage.grain == "person"
        assert stage.outputs == ("meets_ssi_disability_criteria",)
        assert stage.nonnegative_outputs == ()
        assert [operation.kind for operation in stage.operations] == [
            "read_table",
            "fit_weighted_qrf",
        ]
        assert (
            dict(stage.operations[0].parameters) == SIPP_SSI_DISABILITY_READ_PARAMETERS
        )
        assert (
            dict(stage.operations[1].parameters) == SIPP_SSI_DISABILITY_FIT_PARAMETERS
        )
        assert stage.operations[1].parameters["training_sample_seed"] == (
            8_386_123_572_872_638_692
        )
        assert stage.operations[1].parameters["model_seed"] == 42
        assert stage.operations[1].parameters["seed_from_build_config"] is False
        assert "separately predict PUF-support people" in stage.notes
        assert "ASEC reporter anchor is never copied" in stage.notes

    def test_head_start_stage_pins_measured_sipp_response_contract(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_SIPP_HEAD_START_STAGE_NAME]
        donor = US_DONORS[US_SIPP_HEAD_START_STAGE_NAME]

        assert stage.survey == donor.survey == "Census SIPP"
        assert stage.source == donor.source
        assert stage.grain == "person"
        assert stage.outputs == ("takes_up_head_start_if_eligible",)
        assert stage.nonnegative_outputs == ()
        assert [operation.kind for operation in stage.operations] == [
            "read_table",
            "fit_weighted_qrf",
        ]
        assert dict(stage.operations[0].parameters) == SIPP_HEAD_START_READ_PARAMETERS
        assert dict(stage.operations[1].parameters) == SIPP_HEAD_START_FIT_PARAMETERS
        assert stage.operations[1].parameters["target"] == (
            "takes_up_head_start_if_eligible"
        )
        assert stage.operations[1].parameters["direct_response_filter"] == (
            "AEDHEADST == 1 and EEDHEADST in [1, 2]"
        )
        assert stage.operations[1].parameters["assignment_unit"] == ("person_source_id")
        assert stage.operations[1].parameters["fan_to_support_clones"] is True
        assert any(
            artifact.get("sha256") and artifact.get("size_bytes")
            for artifact in stage.artifacts
        )
        assert "EEDHEADST" in stage.notes
        assert "AEDHEADST" in stage.notes
        assert "measured" in stage.notes.lower()

    def test_ssi_take_up_stage_pins_reporter_and_bernoulli_prior_contract(
        self,
    ) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_SSI_TAKE_UP_STAGE_NAME]
        donor = US_DONORS[US_SSI_TAKE_UP_STAGE_NAME]

        assert (
            stage.survey
            == donor.survey
            == ("CPS ASEC reported SSI + SSA SSI Monthly Statistics December 2024")
        )
        assert stage.source == donor.source == SSI_TAKE_UP_SSA_SOURCE_URL
        assert stage.grain == "person"
        assert stage.outputs == ("takes_up_ssi_if_eligible",)
        assert stage.nonnegative_outputs == ()
        assert [operation.kind for operation in stage.operations] == [
            "read_table",
            "assign_binary_from_rate",
        ]
        assert dict(stage.operations[0].parameters) == {
            "table": "person",
            "weight": "person_weight",
        }
        assert dict(stage.operations[1].parameters) == {
            "output": "takes_up_ssi_if_eligible",
            "draw": "stable_source_person_draw",
            "rate_key": "ssi_age_band_count_prior",
            "rate_column": "ssi_take_up_assignment_prior",
            "reported_true_anchor": f"{US_SSI_TAKE_UP_ANCHOR} > 0",
            "assignment_unit": "person_source_id",
            "fan_to_support_clones": True,
            "age_bands": {
                "under_18": "age < 18",
                "18_64": "18 <= age < 65",
                "65_plus": "age >= 65",
            },
            "rate_derivation": (
                "(band_target - basis_reporter_candidate_floor) / "
                "(basis_candidate_capacity - basis_reporter_candidate_floor) "
                "over uncapped_ssi > 0 candidates, so anchored-plus-drawn "
                "mass expects the target; basis = this frame's weights, or "
                "a prior attempt's delivered-weight us_ssi_take_up.json "
                "diagnostics (microcosm#507/#508); zero once the reporter "
                "floor meets the target; min(basis_reporter_candidate_floor "
                "/ capacity, 1) once capacity cannot subsample the target"
            ),
            "rate_target_role": "ssa_ssi_age_band_recipients",
            "target_source": SSI_TAKE_UP_SSA_SOURCE_URL,
            "target_period": "2024-12",
            "target_measure": "Total with—Federal payment",
        }
        ssa_artifacts = [
            artifact
            for artifact in stage.artifacts
            if artifact.get("source") == SSI_TAKE_UP_SSA_SOURCE_URL
        ]
        assert len(ssa_artifacts) == 1
        # The SSA recipient counts bind only through the ledger-fed
        # calibration registry (microcosm#469/#470) — never hardcoded here.
        assert all("target_values" not in artifact for artifact in stage.artifacts)
        evidence = next(
            artifact
            for artifact in stage.artifacts
            if artifact.get("kind") == "archived_derivation_evidence"
        )
        assert evidence["commit"] == ("42ed5d45c56df80d754fbe24cce21cfeb8d05cbe")
        assert evidence["lines"] == "584,650-657,1497-1499"
        assert evidence["randomness_path_parts"] == [
            "policyengine_",
            "us_data",
            "datasets",
            "cps",
            "takeup.py",
        ]
        assert evidence["randomness_lines"] == "10-35"
        assert evidence["targets_lines"] == "41-74"

    def test_other_health_insurance_donor_matches_manifest(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_OTHER_HEALTH_INSURANCE_STAGE_NAME]
        donor = US_DONORS[US_OTHER_HEALTH_INSURANCE_STAGE_NAME]

        assert donor.survey == stage.survey == "Census CPS ASEC"
        assert donor.source == stage.source
        assert "PHIP_VAL" in donor.notes
        assert "PUF support half" in donor.notes

    def test_prior_year_income_stage_pins_join_fallback_and_joint_qrf(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_PRIOR_YEAR_INCOME_STAGE_NAME]

        assert stage.outputs == (
            "employment_income_last_year",
            "self_employment_income_last_year",
            "previous_year_income_available",
        )
        assert stage.nonnegative_outputs == ("employment_income_last_year",)
        operations = {operation.kind: operation for operation in stage.operations}
        assert tuple(operations) == (
            "read_table",
            "derive_prior_year_income",
            "impute_prior_year_income_to_puf_support",
        )
        derive = operations["derive_prior_year_income"].parameters
        assert derive["employment_allocation_flag"] == "I_ERNVAL"
        assert derive["self_employment_allocation_flag"] == "I_SEVAL"
        assert derive["sentinels"] == [-1, -9999]
        assert derive["no_prior_artifact"] == "leave_defaults"
        qrf = operations["impute_prior_year_income_to_puf_support"].parameters
        assert qrf["max_train_samples"] == 5_000
        assert qrf["weight"] == "person_weight"
        assert qrf["outputs"] == [
            "employment_income_last_year",
            "self_employment_income_last_year",
        ]
        assert "self_employment_income_last_year" not in US_NONNEGATIVE_SOURCE_OUTPUTS

        donor = US_DONORS[US_PRIOR_YEAR_INCOME_STAGE_NAME]
        assert donor.survey == stage.survey
        assert donor.source == stage.source

    def test_weeks_unemployed_stage_pins_direct_source_and_puf_qrf(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_WEEKS_UNEMPLOYED_STAGE_NAME]
        donor = US_DONORS[US_WEEKS_UNEMPLOYED_STAGE_NAME]

        assert stage.survey == donor.survey == "Census CPS ASEC"
        assert stage.source == donor.source
        assert stage.grain == "person"
        assert stage.outputs == ("weeks_unemployed",)
        assert stage.nonnegative_outputs == stage.outputs

        archive = next(
            artifact
            for artifact in stage.artifacts
            if artifact.get("format") == "zip_csv"
        )
        assert archive["vintage"] == "2023 ASEC / 2022 income reference year"
        assert archive["sha256"] == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256
        assert archive["size_bytes"] == ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES
        assert archive["member"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER
        assert (
            archive["member_size_bytes"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES
        )
        assert archive["member_crc32"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32
        assert archive["member_sha256"] == ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256

        operations = {operation.kind: operation for operation in stage.operations}
        assert tuple(operations) == (
            "read_table",
            "derive_weeks_unemployed",
            "impute_weeks_unemployed_to_puf_support",
        )
        assert dict(operations["read_table"].parameters) == (
            WEEKS_UNEMPLOYED_READ_PARAMETERS
        )
        assert dict(operations["derive_weeks_unemployed"].parameters) == (
            WEEKS_UNEMPLOYED_DERIVE_PARAMETERS
        )
        assert (
            operations["impute_weeks_unemployed_to_puf_support"].parameters
            == WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS
        )
        assert "no 2022 source value is filled statistically" in stage.notes

    def test_disability_benefits_stage_pins_direct_formula_and_puf_imputation(
        self,
    ) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_DISABILITY_BENEFITS_STAGE_NAME]

        assert stage.survey == "Census CPS ASEC"
        assert stage.source == "https://www.census.gov/programs-surveys/cps.html"
        assert stage.grain == "person"
        assert stage.outputs == ("disability_benefits",)
        assert stage.nonnegative_outputs == stage.outputs

        operations = {operation.kind: operation for operation in stage.operations}
        assert tuple(operations) == (
            "read_table",
            "derive_disability_benefits",
            "impute_disability_benefits_to_puf_support",
        )
        assert operations["read_table"].parameters == {
            "table": "person",
            "weight": "person_weight",
        }
        assert operations["derive_disability_benefits"].parameters == {
            "first_amount_source": "DIS_VAL1",
            "first_code_source": "DIS_SC1",
            "second_amount_source": "DIS_VAL2",
            "second_code_source": "DIS_SC2",
            "workers_compensation_code": 1,
            "output": "disability_benefits",
        }
        assert operations["impute_disability_benefits_to_puf_support"].parameters == {
            "predictors": [
                "age",
                "is_male",
                "has_esi",
                "tax_unit_is_joint",
                "tax_unit_count_dependents",
                "employment_income",
                "self_employment_income",
                "social_security",
            ],
            "max_train_samples": 5_000,
            "n_estimators": 100,
            "seed_from_build_config": True,
            "weight": "person_weight",
        }

        donor = US_DONORS[US_DISABILITY_BENEFITS_STAGE_NAME]
        assert donor.survey == stage.survey
        assert donor.source == stage.source

    def test_child_support_stage_declares_direct_carry_and_joint_puf_imputation(
        self,
    ) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()[US_CHILD_SUPPORT_STAGE_NAME]

        assert stage.survey == "Census CPS ASEC"
        assert stage.source == "https://www.census.gov/programs-surveys/cps.html"
        assert stage.grain == "person"
        assert stage.outputs == (
            "child_support_received",
            "child_support_expense",
        )
        assert stage.nonnegative_outputs == stage.outputs

        operations = {operation.kind: operation for operation in stage.operations}
        assert tuple(operations) == (
            "read_table",
            "derive_child_support_inputs",
            "impute_child_support_to_puf_support",
        )
        assert operations["read_table"].parameters == {
            "table": "person",
            "weight": "person_weight",
        }
        assert operations["derive_child_support_inputs"].parameters == {
            "received_source": "CSP_VAL",
            "received_output": "child_support_received",
            "expense_source": "CHSP_VAL",
            "expense_output": "child_support_expense",
        }
        assert operations["impute_child_support_to_puf_support"].parameters == {
            "predictors": [
                "age",
                "is_male",
                "has_esi",
                "tax_unit_is_joint",
                "tax_unit_count_dependents",
                "employment_income",
                "self_employment_income",
                "social_security",
            ],
            "max_train_samples": 5_000,
            "n_estimators": 100,
            "seed_from_build_config": True,
            "weight": "person_weight",
        }

        donor = US_DONORS[US_CHILD_SUPPORT_STAGE_NAME]
        assert donor.survey == stage.survey
        assert donor.source == stage.source

    def test_source_specs_are_manifest_only_not_python_loaders(self) -> None:
        for spec in US_SOURCE_STAGE_SPECS:
            assert spec.operations
            for operation in spec.operations:
                assert "module" not in operation.parameters
                assert "function" not in operation.parameters
                assert operation.kind not in {
                    "python_module",
                    "python_function",
                    "import_module",
                }

    def test_source_operation_parser_rejects_python_loader_shapes(self) -> None:
        with pytest.raises(ValueError, match="executable-loader"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "python_module",
                    "module": "microcosm.build.us.sources",
                    "function": "add_scf_wealth",
                }
            )
        with pytest.raises(ValueError, match="executable-loader"):
            SourceOperationSpec.from_mapping({"kind": "custom_loader"})
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "postprocess": [{"callable": "clean_scf"}],
                }
            )
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "callable_path": "microcosm.build.us_runtime.sources:clean",
                }
            )
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "entry-point": "microcosm.build.us_runtime.sources:clean",
                }
            )
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "handler": "clean_scf",
                }
            )
        with pytest.raises(ValueError, match="executable Python entrypoint"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "transform_path": "microcosm.build.us_runtime.sources:clean",
                }
            )

    def test_source_manifest_parser_rejects_python_loader_artifacts(self) -> None:
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceManifest.from_mapping(
                {
                    "version": 1,
                    "country": "us",
                    "policy": "spec only",
                    "stages": [
                        {
                            "stage": "scf_wealth",
                            "survey": "Fed SCF 2022",
                            "source": "https://www.federalreserve.gov/econres/scfindex.htm",
                            "grain": "household",
                            "artifacts": [
                                {
                                    "kind": "public_microdata",
                                    "loader": "microcosm.build.us.sources:add_scf_wealth",
                                }
                            ],
                            "operations": [
                                {"kind": "read_table", "table": "scf_household"}
                            ],
                            "outputs": ["net_worth"],
                        }
                    ],
                }
            )

    def test_legacy_sources_import_path_is_metadata_only(self) -> None:
        from microcosm.build.us_runtime import sources
        from microcosm.build.us_runtime.sources import (
            SCF_TARGETS,
            _support_guard,
            add_scf_wealth,
        )

        assert sources.US_SOURCE_MANIFEST is US_SOURCE_MANIFEST
        with pytest.raises(AttributeError, match="has been removed"):
            sources.__getattr__("add_scf_wealth")
        with pytest.raises(RuntimeError, match="has been removed"):
            add_scf_wealth(None, None, 0, lambda *_args: None)
        with pytest.raises(RuntimeError, match="has been removed"):
            _support_guard([], [], "x", lambda *_args: None)
        with pytest.raises(RuntimeError, match="has been removed"):
            sources.NONNEGATIVE_SCF_TARGETS.__contains__("auto_loan_interest")
        with pytest.raises(RuntimeError, match="has been removed"):
            SCF_TARGETS.__contains__("net_worth")

    def test_scf_nonnegative_requirement_is_source_declared(self) -> None:
        specs = US_SOURCE_MANIFEST.stage_map()
        for column in (
            "auto_loan_balance",
            "auto_loan_interest",
            "qualified_passenger_vehicle_loan_interest",
        ):
            assert column in specs["scf_wealth"].nonnegative_outputs
            assert column in US_NONNEGATIVE_SOURCE_OUTPUTS
        assert "net_worth" not in US_NONNEGATIVE_SOURCE_OUTPUTS

    def test_sipp_vehicle_stage_pins_full_donor_and_no_partial_net_worth(self) -> None:
        spec = US_SOURCE_MANIFEST.stage_map()["vehicle_assets"]
        artifact = spec.artifacts[0]
        assert artifact["vintage"] == "2023"
        assert artifact["sha256"] == (
            "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
        )
        assert artifact["size_bytes"] == 3_726_010_471
        assert "21280dca5995e978d706740a8a4b9b7860cfd7b6" in artifact["locator"]
        model = next(
            operation
            for operation in spec.operations
            if operation.kind == "fit_vehicle_model"
        )
        assert model.parameters["weight"] == "household_weight"
        assert model.parameters["observed_status_values"] == [0, 1, 9]
        assert model.parameters["owned_allocation_flag"] == "AVEH_NUM"
        assert model.parameters["value_allocation_flags"] == [
            "AVEH1VAL",
            "AVEH2VAL",
            "AVEH3VAL",
        ]
        assert model.parameters["seed"] == 42
        assert model.parameters["n_estimators"] == 100
        assert all(operation.kind != "fold_into" for operation in spec.operations)
        assert "does not execute the old placeholder fold" in spec.notes
        assert set(spec.outputs) == {
            "household_vehicles_owned",
            "household_vehicles_value",
        }
        assert set(spec.outputs) <= set(spec.nonnegative_outputs)

    def test_voluntary_filing_stage_pins_measured_sipp_response(self) -> None:
        spec = US_SOURCE_MANIFEST.stage_map()[US_VOLUNTARY_FILING_STAGE_NAME]
        artifact = spec.artifacts[0]
        assert artifact["vintage"] == "2023"
        assert artifact["sha256"] == (
            "5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
        )
        assert artifact["size_bytes"] == 3_726_010_471
        assert "21280dca5995e978d706740a8a4b9b7860cfd7b6" in artifact["locator"]

        read, model = spec.operations
        assert read.kind == "read_table"
        assert read.parameters["month_column"] == "MONTHCODE"
        assert read.parameters["month"] == 12
        assert read.parameters["source_columns"] == [
            "SSUID",
            "PNUM",
            "MONTHCODE",
            "WPFINWGT",
            "TAGE",
            "ESEX",
            "EPNSPOUSE",
            "AFILING",
            "EFILING",
            "AWILLFILE",
            "EWILLFILE",
            "EDEPCLM",
            "TJB1_MSUM",
            "TJB2_MSUM",
            "TJB3_MSUM",
            "TJB4_MSUM",
            "TJB5_MSUM",
            "TJB6_MSUM",
            "TJB7_MSUM",
        ]
        assert model.kind == "fit_weighted_qrf"
        assert model.parameters["predictors"] == [
            "employment_income",
            "reference_age",
            "reference_is_female",
            "reference_is_married",
            "count_under_18",
        ]
        assert model.parameters["target"] == "would_file_taxes_voluntarily"
        assert model.parameters["weight"] == "tax_unit_weight"
        assert model.parameters["response_filter"] == (
            "AFILING == 1 and (EFILING == 1 or (EFILING == 2 and AWILLFILE == 1))"
        )
        assert model.parameters["dependent_exclusion"] == "EDEPCLM == 1"
        assert "reciprocal spouses" in model.parameters["canonical_unit"]
        assert model.parameters["n_estimators"] == 100
        assert model.parameters["seed_from_build_config"] is True
        assert spec.outputs == ("would_file_taxes_voluntarily",)
        assert spec.nonnegative_outputs == ()
        assert "datasets/cps/cps.py lines 726-747" in spec.notes
        assert "parameters/take_up/voluntary_filing.yaml lines 1-43" in spec.notes
        assert "22,313 observed canonical units" in spec.notes
        assert "22,296 have positive finite reference weights" in spec.notes
        assert "0.7603084563 weighted true share" in spec.notes

    def test_puf_stage_declares_partnership_s_corp_leaves_not_aggregate(self) -> None:
        outputs = set(US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"].outputs)
        assert "partnership_income" in outputs
        assert "s_corp_income" in outputs
        assert "partnership_self_employment_net_earnings" in outputs
        assert "qualified_dividend_income" in outputs
        assert "non_qualified_dividend_income" in outputs
        assert "home_mortgage_interest" in outputs
        assert "charitable_cash_donations" in outputs
        assert "charitable_non_cash_donations" in outputs
        assert "partnership_s_corp_income" not in outputs
        assert "partnership_se_income" not in outputs
        assert "partnership_s_corp_loss" not in outputs
        assert "qualified_business_income" not in outputs
        assert "salt_deduction" not in outputs
        assert "medical_expense_deduction" not in outputs
        assert "charitable_deduction" not in outputs
        assert "interest_deduction" not in outputs

    def test_puf_stage_declares_all_materialized_qbi_input_leaves(self) -> None:
        outputs = set(US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"].outputs)

        assert set(US_QBI_OUTPUT_COLUMNS) <= outputs
        assert set(US_QBI_OUTPUT_COLUMNS) <= set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS)
        assert "qualified_business_income" not in outputs

    def test_puf_stage_outputs_match_runtime_defaults(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
        outputs = set(stage.outputs)
        runtime_outputs = set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS) | set(
            PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
        )

        assert outputs == runtime_outputs
        assert "educator_expense" in stage.outputs
        assert "educator_expense" in stage.nonnegative_outputs
        assert "educator_expense" in PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS

    def test_puf_stage_distinguishes_runtime_prefix_from_artifact_lineage(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
        operations = stage.operations
        kinds = [operation.kind for operation in operations]
        assert (
            kinds.index("read_table")
            < kinds.index("derive_puf_policyengine_variables")
            < kinds.index("disaggregate_aggregate_records")
            < kinds.index("uprate")
        )

        derive_operation = operations[kinds.index("derive_puf_policyengine_variables")]
        assert derive_operation.parameters == {
            "ordinary_dividend_source": "E00600",
            "qualified_dividend_source": "E00650",
            "qualified_dividend_output": "qualified_dividend_income",
            "non_qualified_dividend_output": "non_qualified_dividend_income",
            "qualified_tuition_primary_source": "E03230",
            "qualified_tuition_optional_source": "E87530",
            "qualified_tuition_output": "qualified_tuition_expenses",
            "alimony_income_source": "E00800",
            "alimony_income_output": "alimony_income",
            "alimony_expense_source": "E03500",
            "alimony_expense_output": "alimony_expense",
            "casualty_loss_source": "E20500",
            "casualty_loss_output": "casualty_loss",
            "domestic_production_ald_source": "E03240",
            "domestic_production_ald_output": "domestic_production_ald",
            "educator_expense_source": "E03220",
            "educator_expense_output": "educator_expense",
            "unreimbursed_business_employee_expenses_source": "E20400",
            "unreimbursed_business_employee_expenses_output": "unreimbursed_business_employee_expenses",
            "farm_operations_income_source": "E02100",
            "farm_operations_income_output": "farm_operations_income",
            "farm_rent_income_source": "E27200",
            "farm_rent_income_output": "farm_rent_income",
            "investment_income_elected_form_4952_source": "E58990",
            "investment_income_elected_form_4952_output": (
                "investment_income_elected_form_4952"
            ),
            "salt_refund_income_source": "E00700",
            "salt_refund_income_output": "salt_refund_income",
            "collectibles_capital_gain_source": "E24518",
            "collectibles_capital_gain_output": (
                "long_term_capital_gains_on_collectibles"
            ),
            "unrecaptured_section_1250_gain_source": "E24515",
            "unrecaptured_section_1250_gain_output": "unrecaptured_section_1250_gain",
        }

        operation = operations[kinds.index("disaggregate_aggregate_records")]
        assert operation.parameters == {
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999996, 999997, 999998, 999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
            "seed_from_build_config": True,
        }
        assert "use_forbes_top_tail" not in operation.parameters
        artifact = next(
            artifact
            for artifact in stage.artifacts
            if artifact["kind"] == "versioned_derived_microdata"
        )
        assert artifact["lineage"] == {
            "archived_commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
            "generator_path_parts": [
                "policyengine_",
                "us_data",
                "datasets",
                "puf",
                "puf.py",
            ],
            "generator_lines": "1378-1398",
            "disaggregator_path_parts": [
                "policyengine_",
                "us_data",
                "datasets",
                "puf",
                "disaggregate_puf.py",
            ],
            "disaggregator_lines": "55-60,85-123",
            "operation_order": ["uprate", "disaggregate_aggregate_records"],
            "aggregate_disaggregation_seed": 42,
            "forbes_top_tail_enabled": True,
            "forbes_aggregate_recid": 999999,
            "forbes_synthetic_record_count": 3900,
        }
        notes = stage.notes
        assert "uprates the raw TY2015 rows before replacing" in notes
        assert "Forbes backbone for aggregate RECID 999999" in notes
        assert "3,900-record open tail" in notes
        assert "does not request unsupported Forbes synthesis" in notes
        donor_notes = US_DONORS["puf_tax_detail"].notes
        assert "uprates its raw TY2015 rows before" in donor_notes
        assert "Forbes-backed 3,900-record open tail" in donor_notes
        assert "without rerunning Forbes synthesis" in donor_notes

    def test_aca_stage_declares_marketplace_input_surface(self) -> None:
        stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
        outputs = set(stage.outputs)
        assert stage.grain == "tax_unit"
        assert "takes_up_aca_if_eligible" in outputs
        assert "selected_marketplace_plan_benchmark_ratio" in outputs
        assert any(
            operation.kind == "calibrate_binary_assignment"
            and operation.parameters.get("variable") == "takes_up_aca_if_eligible"
            for operation in stage.operations
        )
        assert any(
            operation.kind == "compute_ratio"
            and operation.parameters.get("output")
            == "selected_marketplace_plan_benchmark_ratio"
            for operation in stage.operations
        )

    def test_no_incumbent_data_package_references_in_live_tree(self) -> None:
        forbidden = (
            "policyengine_" + "us_data",
            "policyengine-" + "us-data",
            "policyengine_" + "uk_data",
            "policyengine-" + "uk-data",
        )
        # The eCPS parity reference (microcosm #313) is the launch contract's
        # pinned record of the incumbent it replaces, so those files must NAME
        # the retired package: a sha-locked historical reference, never a live
        # dependency (the loader imports nothing from it). Nothing else in the
        # live tree may reference the retired data packages.
        allowed_incumbent_references = {
            "packages/microcosm-build/src/microcosm/build/us/ecps_parity_reference.json",
            "packages/microcosm-build/src/microcosm/build/us_runtime/parity_reference.py",
            "packages/microcosm-build/tests/test_us_parity_reference.py",
            # UK coverage has the same frozen-incumbent exception: these files
            # name only the immutable, sha-verified enhanced-FRS reference and
            # never import or execute the retired data package.
            "packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json",
            "packages/microcosm-build/src/microcosm/build/uk/frs_release.json",
            "packages/microcosm-build/src/microcosm/build/uk/hmrc_income_source_stages.json",
            # The UK population contract's registry-parity accounting names the
            # retired data package by necessity: 651 rows at pinned ref ebf733c
            # = 609 mapped + 42 signed exclusions + 3 unmapped declarations.
            # A sha-locked historical reference — nothing imported or executed.
            "packages/microcosm-build/src/microcosm/build/uk/uk_population_targets.json",
            "packages/microcosm-build/src/microcosm/build/uk/target_reference_membership.json",
            "packages/microcosm-build/src/microcosm/build/uk_runtime/parity_reference.py",
            # The HMRC source contract pins the licensed SPI/ODS input
            # identities, whose reviewed provenance names the archived data
            # repository the licensed copies were vendored from. Identity
            # strings only — nothing is imported or fetched from it.
            "packages/microcosm-build/src/microcosm/build/uk_runtime/hmrc_source_contract.py",
            "packages/microcosm-build/tests/test_uk_parity_reference.py",
            # The publication contract's June-grandfather test keeps the
            # three semantic-real JSON artifacts from the frozen df82567
            # release. These exact test-only paths preserve historical
            # repository/version provenance; they import or execute nothing.
            "packages/microcosm-data/tests/test_contract.py",
            (
                "packages/microcosm-data/tests/fixtures/uk_june_2023/"
                "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z/"
                "build_manifest.json"
            ),
            (
                "packages/microcosm-data/tests/fixtures/uk_june_2023/"
                "populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z/"
                "calibration_diagnostics.json"
            ),
            "tools/build_uk_efrs_parity_reference.py",
            # The UK local registry incumbent fixture keeps its one-off
            # extractor as provenance for the frozen JSON fixture. It is not
            # imported by package code or live gate/build paths.
            "packages/microcosm-build/src/microcosm/build/uk/"
            "local_registry_parity_fixture_2025.json",
            # The late target-parity inventory is a sha-locked static extract
            # of the archived module surface. Its companion parity register
            # cites those entry ids; neither artifact imports package code.
            "packages/microcosm-build/src/microcosm/build/uk/"
            "uk_data_target_inventory.json",
            "packages/microcosm-build/src/microcosm/build/uk/"
            "uk_data_target_parity.json",
            "tools/extract_uk_local_registry_fixture.py",
            "UK_COVERAGE_PROGRESS.md",
        }
        checked_suffixes = {".py", ".toml", ".md", ".json"}
        offenders: list[tuple[str, str]] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in checked_suffixes:
                continue
            if (
                ".git" in path.parts
                or ".venv" in path.parts
                or ".claude" in path.parts
                or ".codex-work" in path.parts
                or "out" in path.parts
                # Run scaffolding and staged run outputs (launchers, base-rebuild
                # summaries) are not shipped source and may record the incumbent
                # in local-path provenance; the shipped tree under packages/ is
                # still swept. Same non-shipped class as ``out``.
                or "experiments" in path.parts
                # UK runtime scratch (staging builds, replay outputs) is
                # caller-owned run scaffolding, untracked in CI, and records
                # licensed-source provenance; same non-shipped class as ``out``.
                or "uk_runtime" in path.parts
            ):
                continue
            rel = str(path.relative_to(ROOT))
            if rel in allowed_incumbent_references:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    offenders.append((rel, needle))
        assert offenders == []


class TestBaseStageSourceClosure:
    """Every base-stage required source column has a declared provider.

    microcosm#417's sibling failure class: a stage's REQUIRED_SOURCE_COLUMNS
    named a raw ASEC column (``ED_VAL``) that no ingestion path or earlier
    stage provided, and the gap only surfaced 17 stages into a full-scale
    build.  This test closes the class statically: the union of base-stage
    requirements must be covered by the frozen census_cps person columns,
    pool-constructed identity columns, the pinned sidecar restorations, or
    another stage's declared outputs.
    """

    #: Person columns of the frozen census_cps_2022/2023/2024 inputs (union),
    #: pinned so CI can check requirement closure without the data files.
    #: Regenerate via pd.read_hdf(..., "person", stop=1).columns if the
    #: upstream archives are ever re-vendored.
    CENSUS_CPS_PERSON_COLUMNS = frozenset(
        {
            "ACTC_CRD",
            "AGI",
            "ANN_VAL",
            "A_AGE",
            "A_ENRLW",
            "A_EXPRRP",
            "A_FAMREL",
            "A_FAMTYP",
            "A_FNLWGT",
            "A_FTPT",
            "A_HRS1",
            "A_HSCOL",
            "A_LINENO",
            "A_MARITL",
            "A_MJOCC",
            "A_SEX",
            "A_SPOUSE",
            "CAID",
            "CAP_VAL",
            "CENSUS_TAX_ID",
            "CHAMPVA",
            "CHSP_VAL",
            "CSP_VAL",
            "CTC_CRD",
            "DIS_SC1",
            "DIS_SC2",
            "DIS_VAL1",
            "DIS_VAL2",
            "DIV_VAL",
            "DST_SC1",
            "DST_SC1_YNG",
            "DST_SC2",
            "DST_SC2_YNG",
            "DST_VAL1",
            "DST_VAL1_YNG",
            "DST_VAL2",
            "DST_VAL2_YNG",
            "EIT_CRED",
            "FEDTAX_AC",
            "FEDTAX_BC",
            "FRSE_VAL",
            "HRSWK",
            "IHSFLG",
            "INT_VAL",
            "I_ERNVAL",
            "I_SEVAL",
            "LKWEEKS",
            "MARG_TAX",
            "MCARE",
            "MIL",
            "MOOP",
            "NOW_CAID",
            "NOW_CHAMPVA",
            "NOW_COV",
            "NOW_DIR",
            "NOW_GRP",
            "NOW_IHSFLG",
            "NOW_MCAID",
            "NOW_MCARE",
            "NOW_MIL",
            "NOW_MRK",
            "NOW_MRKS",
            "NOW_MRKUN",
            "NOW_NONM",
            "NOW_OTHMT",
            "NOW_PCHIP",
            "NOW_PRIV",
            "NOW_PUB",
            "NOW_VACARE",
            "OI_OFF",
            "OI_VAL",
            "PAW_VAL",
            "PEAFEVER",
            "PECOHAB",
            "PEDISDRS",
            "PEDISEAR",
            "PEDISEYE",
            "PEDISOUT",
            "PEDISPHY",
            "PEDISREM",
            "PEINUSYR",
            "PEIO1COW",
            "PEIOOCC",
            "PEMCPREM",
            "PENATVTY",
            "PEN_SC1",
            "PEN_SC2",
            "PEPAR1",
            "PEPAR2",
            "PERIDNUM",
            "PF_SEQ",
            "PHIP_VAL",
            "PH_SEQ",
            "PMED_VAL",
            "PNSN_VAL",
            "POCCU2",
            "POTC_VAL",
            "PRCITSHP",
            "PRDTHSP",
            "PRDTRACE",
            "PTOTVAL",
            "P_SEQ",
            "RESNSS1",
            "RESNSS2",
            "RESNSSI1",
            "RESNSSI2",
            "RETCB_VAL",
            "RNT_VAL",
            "SEMP_VAL",
            "SPM_ACTC",
            "SPM_BBSUBVAL",
            "SPM_CAPHOUSESUB",
            "SPM_CAPWKCCXPNS",
            "SPM_CHILDCAREXPNS",
            "SPM_CHILDSUPPD",
            "SPM_EITC",
            "SPM_ENGVAL",
            "SPM_EQUIVSCALE",
            "SPM_FAMTYPE",
            "SPM_FEDTAX",
            "SPM_FEDTAXBC",
            "SPM_FICA",
            "SPM_GEOADJ",
            "SPM_HAGE",
            "SPM_HHISP",
            "SPM_HMARITALSTATUS",
            "SPM_HRACE",
            "SPM_ID",
            "SPM_MEDXPNS",
            "SPM_NUMADULTS",
            "SPM_NUMKIDS",
            "SPM_NUMPER",
            "SPM_POOR",
            "SPM_POVTHRESHOLD",
            "SPM_RESOURCES",
            "SPM_SCHLUNCH",
            "SPM_SNAPSUB",
            "SPM_STTAX",
            "SPM_TENMORTSTATUS",
            "SPM_TOTVAL",
            "SPM_WCOHABIT",
            "SPM_WEIGHT",
            "SPM_WFOSTER22",
            "SPM_WICVAL",
            "SPM_WKXPNS",
            "SPM_WNEWHEAD",
            "SPM_WNEWPARENT",
            "SPM_WUI_LT15",
            "SSI_VAL",
            "SSI_YN",
            "SS_VAL",
            "SS_YN",
            "STATETAX_A",
            "STATETAX_B",
            "TAX_ID",
            "TAX_INC",
            "UC_VAL",
            "VET_VAL",
            "WC_VAL",
            "WICYN",
            "WKSWORK",
            "WSAL_VAL",
        }
    )

    #: Columns the pool constructor itself attaches to every person row.
    POOL_CONSTRUCTED_COLUMNS = frozenset(
        {
            "source_year",
            "person_id",
            "person_weight",
            "person_support_channel",
            "person_tax_unit_id",
            "person_household_id",
            "person_family_id",
            "person_spm_unit_id",
            "source_household_id",
            "person_source_id",
        }
    )

    #: Raw columns restored from pinned official sidecars because the frozen
    #: census_cps inputs never carried them (LKWEEKS only for income year
    #: 2022; ED_VAL and PAW_TYP for every pooled year).
    SIDECAR_RESTORED_COLUMNS = frozenset({"LKWEEKS", "ED_VAL", "PAW_TYP"})

    #: Release-time stage constants whose inputs are produced inside the
    #: fiscal-refresh release tool, not the base builder (org_wages consumes
    #: hours/weeks columns the release derives upstream of it).
    RELEASE_TIME_REQUIREMENT_CONSTANTS = frozenset(
        {
            "US_ORG_WAGES_REQUIRED_SOURCE_COLUMNS",
        }
    )

    def test_every_base_stage_requirement_has_a_provider(self) -> None:
        import microcosm.build.us_runtime as us_runtime

        produced: set[str] = set()
        for name in dir(us_runtime):
            if name.endswith("_OUTPUT_COLUMNS"):
                produced.update(getattr(us_runtime, name))
        manifest = json.loads(
            (
                ROOT
                / "packages/microcosm-build/src/microcosm/build/us/source_stages.json"
            ).read_text()
        )
        for stage in manifest.get("stages", []):
            produced.update(stage.get("outputs", ()))
        # The CPS-carried derivation attaches its person/SPM inputs (age,
        # is_female, medical expenses, ...) before any source stage runs.
        from microcosm.build.us_runtime.cps_carried import (
            CPS_CARRIED_FORMULA_OWNED_COLUMNS,
            CPS_CARRIED_PERSON_INPUTS,
            CPS_CARRIED_SPM_UNIT_INPUTS,
        )

        produced.update(CPS_CARRIED_PERSON_INPUTS)
        produced.update(CPS_CARRIED_FORMULA_OWNED_COLUMNS)
        produced.update(CPS_CARRIED_SPM_UNIT_INPUTS)

        providers = (
            self.CENSUS_CPS_PERSON_COLUMNS
            | self.POOL_CONSTRUCTED_COLUMNS
            | self.SIDECAR_RESTORED_COLUMNS
            | produced
        )
        unsourced: dict[str, list[str]] = {}
        for name in dir(us_runtime):
            if not name.endswith("_REQUIRED_SOURCE_COLUMNS"):
                continue
            if name in self.RELEASE_TIME_REQUIREMENT_CONSTANTS:
                continue
            missing = [
                column
                for column in getattr(us_runtime, name)
                if column not in providers
            ]
            if missing:
                unsourced[name] = missing
        assert unsourced == {}, (
            "Stage source columns with no declared provider (wire an ingestion "
            f"carry, a pinned sidecar, or an upstream stage output): {unsourced}"
        )
