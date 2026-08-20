from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.source_manifest import (
    FORBIDDEN_SOURCE_DEPENDENCIES,
    SourceManifest,
)
from microcosm.frame import Frame

ROOT = Path(__file__).resolve().parents[3]
UK_PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build/uk"
FROZEN_SOURCE_STAGES = UK_PACKAGE / "hmrc_income_source_stages.json"
CANONICAL_SOURCE_STAGES = UK_PACKAGE / "source_stages.json"
E3_STAGE_NAMES = [
    "frs_employment",
    "frs_council_tax",
    "frs_disability",
    "frs_education",
    "frs_legacy_proxies",
    "frs_education_grant_split",
]
E4_STAGE_NAMES = [
    "frs_take_up",
    "frs_person_draws",
    "frs_household_draws",
    "frs_brma",
]
E5_STAGE_NAMES = [
    "was_wealth",
    "regional_property_uprating",
]
E7_STAGE_NAMES = [
    "frs_hmrc_spine_leaves",
    "spi_support_channel",
    "hmrc_spi_income_spine",
]
UK_SOURCE_STAGE_NAMES = [
    "frs_spine",
    *E3_STAGE_NAMES,
    *E4_STAGE_NAMES,
    *E5_STAGE_NAMES,
    *E7_STAGE_NAMES,
    "frs_hmrc_retained_leaves",
    "hmrc_spi_income",
]
UK_FRS_SPI_SPINE_DRIVER_STAGE_NAMES = [
    "frs_spine",
    *E3_STAGE_NAMES,
    *E4_STAGE_NAMES,
    *E7_STAGE_NAMES,
]
FROZEN_SOURCE_STAGES_SHA256 = (
    "c0341af7166ae3a85a3c1164e7d9e880c4b4aec122f1a8fa90c73b46c596e1ea"
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(frame: Frame) -> Frame:
    return frame


def _assert_no_forbidden_dependency(value: object) -> None:
    text = json.dumps(value, sort_keys=True).lower()
    for dependency in FORBIDDEN_SOURCE_DEPENDENCIES:
        assert dependency not in text


def _expected_reviewed_source() -> str:
    return (
        "PolicyEngine licensed UKDS mirror (private Hugging Face repository), "
        "spi_2022_23.zip"
    )


def _rephrase_stage2_predictor_note(value: str) -> str:
    return value.replace(
        "policyengine-" + "uk-data frs_only.py",
        "the incumbent UK data build's frs_only.py",
    )


class TestUKSourceStagesManifest:
    def test_source_stages_json_loads_as_shared_manifest(self) -> None:
        manifest = SourceManifest.from_mapping(_load_json(CANONICAL_SOURCE_STAGES))

        assert manifest.country == "uk"
        assert manifest.version == 1
        assert [stage.stage for stage in manifest.stages] == UK_SOURCE_STAGE_NAMES

    def test_country_spec_declares_uk_source_stages(self) -> None:
        spec = load_country_spec("uk")

        assert spec.sources is not None
        assert [stage.stage for stage in spec.sources.stages] == UK_SOURCE_STAGE_NAMES

    def test_e7_block_is_contiguous_before_certified_pair(self) -> None:
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        names = [stage["stage"] for stage in canonical["stages"]]

        assert names[-5:-2] == E7_STAGE_NAMES
        assert names[-2:] == ["frs_hmrc_retained_leaves", "hmrc_spi_income"]

    def test_copy_is_lockstep_with_frozen_original_except_citation_rewrites(
        self,
    ) -> None:
        frozen = _load_json(FROZEN_SOURCE_STAGES)
        canonical = _load_json(CANONICAL_SOURCE_STAGES)
        frozen_stage = frozen["stages"][0]
        stage1, stage2 = canonical["stages"][-2:]

        expected_operations = copy.deepcopy(frozen_stage["operations"])
        predictor_note = expected_operations[6]["reviewed_absent_predictors"][
            "other_investment_income"
        ]
        expected_operations[6]["reviewed_absent_predictors"][
            "other_investment_income"
        ] = _rephrase_stage2_predictor_note(predictor_note)

        assert stage1["operations"] + stage2["operations"] == expected_operations
        _assert_no_forbidden_dependency(
            stage2["operations"][4]["reviewed_absent_predictors"][
                "other_investment_income"
            ]
        )

        expected_artifacts = copy.deepcopy(frozen_stage["artifacts"])
        expected_artifacts[0]["reviewed_source"] = _expected_reviewed_source()
        # Declared output-name correction (licensed-data acceptance finding):
        # the frozen original listed the SPI concept "state_pension", but the
        # stage writes the auxiliary column SPI_HMRC_STATE_PENSION_INCOME_COLUMN
        # ("hmrc_spi_state_pension_income") — the model input state_pension is
        # formula-owned and never a frame column here. Outputs became
        # load-bearing when country_stage_plan compiled them into
        # StagePlan.produces, so the copy declares the persisted truth. The
        # operation payloads keep the concept name unchanged.
        expected_outputs = [
            "hmrc_spi_state_pension_income" if name == "state_pension" else name
            for name in frozen_stage["outputs"]
        ]
        assert stage2["outputs"] == expected_outputs
        assert stage2["grain"] == frozen_stage["grain"]
        assert stage2["artifacts"] == expected_artifacts
        _assert_no_forbidden_dependency(stage2["artifacts"])
        _assert_no_forbidden_dependency(stage2["notes"])

    def test_frozen_original_bytes_are_pinned(self) -> None:
        digest = hashlib.sha256(FROZEN_SOURCE_STAGES.read_bytes()).hexdigest()

        assert digest == FROZEN_SOURCE_STAGES_SHA256

    def test_country_stage_plan_assembles_two_certified_uk_national_stages(
        self,
    ) -> None:
        spec = load_country_spec("uk")
        plan = country_stage_plan(
            spec,
            {
                "frs_hmrc_retained_leaves": _identity,
                "hmrc_spi_income": _identity,
            },
            stage_names=("frs_hmrc_retained_leaves", "hmrc_spi_income"),
        )

        assert [stage.name for stage in plan.stages] == [
            "frs_hmrc_retained_leaves",
            "hmrc_spi_income",
        ]

    def test_country_stage_plan_assembles_fourteen_stage_spine_plan(self) -> None:
        spec = load_country_spec("uk")
        implementations = {name: _identity for name in UK_SOURCE_STAGE_NAMES}
        plan = country_stage_plan(
            spec,
            implementations,
            stage_names=tuple(UK_FRS_SPI_SPINE_DRIVER_STAGE_NAMES),
        )

        assert [stage.name for stage in plan.stages] == (
            UK_FRS_SPI_SPINE_DRIVER_STAGE_NAMES
        )

    @pytest.mark.parametrize(
        "implementations, match",
        [
            ({"frs_hmrc_retained_leaves": _identity}, "missing"),
            (
                {
                    "frs_spine": _identity,
                    "frs_employment": _identity,
                    "frs_council_tax": _identity,
                    "frs_disability": _identity,
                    "frs_education": _identity,
                    "frs_legacy_proxies": _identity,
                    "frs_education_grant_split": _identity,
                    "frs_take_up": _identity,
                    "frs_person_draws": _identity,
                    "frs_household_draws": _identity,
                    "frs_brma": _identity,
                    "was_wealth": _identity,
                    "regional_property_uprating": _identity,
                    "frs_hmrc_spine_leaves": _identity,
                    "spi_support_channel": _identity,
                    "hmrc_spi_income_spine": _identity,
                    "frs_hmrc_retained_leaves": _identity,
                    "hmrc_spi_income": _identity,
                    "hmrc_spi_income_fallback": _identity,
                },
                "Unknown stage implementation",
            ),
        ],
    )
    def test_country_stage_plan_refuses_missing_or_unknown_uk_stage(
        self,
        implementations,
        match: str,
    ) -> None:
        spec = load_country_spec("uk")

        with pytest.raises(ValueError, match=match):
            country_stage_plan(spec, implementations)


class TestDeclaredOutputsAreWrittenColumns:
    """Declared outputs must name columns the stages actually write.

    Outputs are load-bearing (``country_stage_plan`` compiles them into
    ``StagePlan.produces``), and the licensed-data acceptance for this
    migration caught a declared output that was an SPI *concept* rather
    than a persisted column — harmless while nothing read the field,
    refused at full rung once it did. This pins every declared output to
    a named runtime written-column constant so the class cannot recur
    without a licensed build to find it (microcosm#690 review).
    """

    def test_stage1_outputs_are_exactly_the_retained_leaf_columns(self) -> None:
        from microcosm.build.uk_runtime.frs_hmrc_leaves import (
            FRS_HMRC_RETAINED_LEAF_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        stage1 = stages["frs_hmrc_retained_leaves"]
        assert stage1.outputs == tuple(FRS_HMRC_RETAINED_LEAF_COLUMNS)

    def test_e3_outputs_are_backed_by_runtime_written_columns(self) -> None:
        from microcosm.build.uk_runtime.frs_brma import FRS_BRMA_OUTPUT_COLUMNS
        from microcosm.build.uk_runtime.frs_council_tax import (
            FRS_COUNCIL_TAX_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_disability import (
            FRS_DISABILITY_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_education import (
            FRS_EDUCATION_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_education_grants import (
            FRS_EDUCATION_GRANT_OUTPUT_COLUMNS,
            FRS_EDUCATION_GRANT_REWRITES,
        )
        from microcosm.build.uk_runtime.frs_employment import (
            FRS_EMPLOYMENT_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_household_draws import (
            FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_legacy_proxies import (
            FRS_LEGACY_PROXY_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_person_draws import (
            FRS_PERSON_DRAW_NONNEGATIVE_OUTPUT_COLUMNS,
            FRS_PERSON_DRAW_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.frs_take_up import (
            FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS,
            FRS_TAKE_UP_OUTPUT_COLUMNS,
        )
        from microcosm.build.uk_runtime.regional_uprating import (
            UK_REGIONAL_PROPERTY_REWRITES,
        )
        from microcosm.build.uk_runtime.was_wealth import (
            UK_WAS_WEALTH_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_WAS_WEALTH_OUTPUT_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert stages["frs_employment"].outputs == FRS_EMPLOYMENT_OUTPUT_COLUMNS
        assert stages["frs_council_tax"].outputs == FRS_COUNCIL_TAX_OUTPUT_COLUMNS
        assert stages["frs_disability"].outputs == FRS_DISABILITY_OUTPUT_COLUMNS
        assert stages["frs_education"].outputs == FRS_EDUCATION_OUTPUT_COLUMNS
        assert stages["frs_legacy_proxies"].outputs == FRS_LEGACY_PROXY_OUTPUT_COLUMNS
        assert (
            stages["frs_education_grant_split"].outputs
            == FRS_EDUCATION_GRANT_OUTPUT_COLUMNS
        )
        assert (
            stages["frs_education_grant_split"].rewrites == FRS_EDUCATION_GRANT_REWRITES
        )
        assert stages["frs_take_up"].outputs == FRS_TAKE_UP_OUTPUT_COLUMNS
        assert (
            stages["frs_take_up"].nonnegative_outputs
            == FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["frs_person_draws"].outputs == FRS_PERSON_DRAW_OUTPUT_COLUMNS
        assert (
            stages["frs_person_draws"].nonnegative_outputs
            == FRS_PERSON_DRAW_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert (
            stages["frs_household_draws"].outputs == FRS_HOUSEHOLD_DRAW_OUTPUT_COLUMNS
        )
        assert stages["frs_brma"].outputs == FRS_BRMA_OUTPUT_COLUMNS
        assert stages["was_wealth"].outputs == UK_WAS_WEALTH_OUTPUT_COLUMNS
        assert (
            stages["was_wealth"].nonnegative_outputs
            == UK_WAS_WEALTH_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert stages["regional_property_uprating"].outputs == ()
        assert (
            stages["regional_property_uprating"].rewrites
            == UK_REGIONAL_PROPERTY_REWRITES
        )

    def test_e7_outputs_and_rewrites_are_backed_by_runtime_constants(self) -> None:
        from microcosm.build.uk_runtime.spi_spine import (
            UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS,
            UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS,
            UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS,
            UK_SPI_INCOME_SPINE_REWRITE_COLUMNS,
            UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert (
            stages["frs_hmrc_spine_leaves"].outputs
            == UK_FRS_HMRC_SPINE_LEAF_OUTPUT_COLUMNS
        )
        assert (
            stages["spi_support_channel"].outputs
            == UK_SPI_SUPPORT_CHANNEL_OUTPUT_COLUMNS
        )
        income = stages["hmrc_spi_income_spine"]
        assert income.outputs == UK_SPI_INCOME_SPINE_OUTPUT_COLUMNS
        assert income.nonnegative_outputs == (
            UK_SPI_INCOME_SPINE_NONNEGATIVE_OUTPUT_COLUMNS
        )
        assert income.rewrites == UK_SPI_INCOME_SPINE_REWRITE_COLUMNS
        assert not (set(income.outputs) & set(income.rewrites))


class TestE3ManifestLockstep:
    def test_e3_raw_tab_pins_match_spine_artifacts(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        spine_pins = {
            artifact["table"]: (
                artifact["locator"],
                artifact["sha256"],
                artifact["size_bytes"],
            )
            for artifact in stages["frs_spine"].artifacts
        }

        for stage_name in E3_STAGE_NAMES:
            for artifact in stages[stage_name].artifacts:
                assert (
                    artifact["locator"],
                    artifact["sha256"],
                    artifact["size_bytes"],
                ) == spine_pins[artifact["table"]]

    def test_e3_operation_kinds_are_declared_in_order(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert [op.kind for op in stages["frs_employment"].operations] == [
            "read_tables",
            "map_coded_amounts",
        ]
        assert [op.kind for op in stages["frs_council_tax"].operations] == [
            "read_tables",
            "impute_cell_means",
        ]
        assert [op.kind for op in stages["frs_disability"].operations] == [
            "derive",
            "derive",
        ]
        assert [op.kind for op in stages["frs_education"].operations] == [
            "read_tables",
            "derive",
            "impute_cell_means",
        ]
        assert [op.kind for op in stages["frs_legacy_proxies"].operations] == [
            "read_tables",
            "materialize_rules_engine_predictors",
            "derive",
        ]
        assert [op.kind for op in stages["frs_education_grant_split"].operations] == [
            "materialize_rules_engine_predictors",
            "derive",
        ]
        assert [op.kind for op in stages["frs_take_up"].operations] == [
            "aggregate_person_to_benunit",
            "assign_binary_with_anchored_residual",
            "assign_binary_from_rate",
            "assign_binary_with_anchored_residual",
            "assign_binary_with_anchored_residual",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_clipped_normal",
        ]
        assert [op.kind for op in stages["frs_person_draws"].operations] == [
            "assign_binary_from_rate",
            "assign_binary_from_banded_rates",
            "assign_uniform_draw",
        ]
        assert [op.kind for op in stages["frs_household_draws"].operations] == [
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
            "assign_binary_from_rate",
        ]
        assert [op.kind for op in stages["frs_brma"].operations] == [
            "materialize_rules_engine_predictors",
            "sample_categorical_from_count_table",
        ]
        assert [op.kind for op in stages["was_wealth"].operations] == [
            "derive",
            "materialize_rules_engine_predictors",
            "fit_weighted_qrf_chain",
            "fold_into",
            "support_clip",
            "allocate_within_group_waterfall",
        ]
        assert [op.kind for op in stages["regional_property_uprating"].operations] == [
            "uprate_to_regional_reference",
        ]
        assert [op.kind for op in stages["frs_hmrc_spine_leaves"].operations] == [
            "retain_adjudicated_frs_hmrc_leaves",
            "derive",
        ]
        assert [op.kind for op in stages["spi_support_channel"].operations] == [
            "stack_zero_weight_donors",
            "gate_zero_weight_strata",
            "allocate_zero_weight_prior_mass",
        ]
        assert [op.kind for op in stages["hmrc_spi_income_spine"].operations] == [
            "verify_pinned_hmrc_source_pair",
            "strict_read_private_table",
            "fit_weighted_qrf_stage1",
            "fit_weighted_qrf_stage2",
            "redraw_columns_from_fitted_qrf",
            "materialize_hmrc_income_bands_fail_closed",
            "classify_hmrc_income_facts_with_reviewed_fences",
            "gate_distributional_effective_mass",
        ]

    def test_engine_predictor_and_rewrite_constants_match_manifest(self) -> None:
        from microcosm.build.uk_runtime.frs_brma import UK_BRMA_PREDICTORS
        from microcosm.build.uk_runtime.frs_education_grants import (
            FRS_EDUCATION_GRANT_REWRITES,
            UK_EDUCATION_GRANT_CAPACITY_PREDICTORS,
        )
        from microcosm.build.uk_runtime.frs_legacy_proxies import (
            UK_LEGACY_PROXY_PREDICTORS,
        )
        from microcosm.build.uk_runtime.frs_take_up import (
            UK_TAKE_UP_ANCHOR_AGGREGATES,
        )
        from microcosm.build.uk_runtime.was_wealth import (
            UK_WAS_ENGINE_PREDICTORS,
            UK_WAS_WEALTH_PREDICTORS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        legacy_predictors = (
            stages["frs_legacy_proxies"].operations[1].parameters["predictors"]
        )
        grant_predictors = (
            stages["frs_education_grant_split"].operations[0].parameters["predictors"]
        )
        assert tuple(legacy_predictors) == UK_LEGACY_PROXY_PREDICTORS
        assert tuple(grant_predictors) == UK_EDUCATION_GRANT_CAPACITY_PREDICTORS
        assert (
            stages["frs_education_grant_split"].rewrites == FRS_EDUCATION_GRANT_REWRITES
        )
        assert (
            stages["frs_take_up"].operations[0].parameters["aggregates"]
            == UK_TAKE_UP_ANCHOR_AGGREGATES
        )
        assert (
            tuple(stages["frs_brma"].operations[0].parameters["predictors"])
            == UK_BRMA_PREDICTORS
        )
        assert (
            tuple(stages["was_wealth"].operations[1].parameters["predictors"])
            == UK_WAS_ENGINE_PREDICTORS
        )
        assert (
            tuple(stages["was_wealth"].operations[2].parameters["predictors"])
            == UK_WAS_WEALTH_PREDICTORS
        )
        rate_keys = [
            op.parameters["rate_key"]
            for stage_name in (
                "frs_take_up",
                "frs_person_draws",
                "frs_household_draws",
            )
            for op in stages[stage_name].operations
            if "rate_key" in op.parameters
        ]
        assert rate_keys == [
            "child_benefit",
            "child_benefit_opts_out_rate",
            "pension_credit",
            "universal_credit",
            "tax_free_childcare",
            "extended_childcare",
            "universal_childcare",
            "targeted_childcare",
            "marriage_allowance",
            "tv_ownership_rate",
            "tv_licence_evasion_rate",
            "first_time_buyer_rate",
            "property_purchase_rate",
        ]
        scp_bands = stages["frs_person_draws"].operations[1].parameters["bands"]
        assert [band["rate_key"] for band in scp_bands] == [
            "scp_under_6",
            "scp_6_plus",
        ]

    def test_every_e4_stochastic_operation_declares_integer_seed(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        for stage_name in E4_STAGE_NAMES:
            for operation in stages[stage_name].operations:
                if operation.kind in {
                    "assign_binary_with_anchored_residual",
                    "assign_binary_from_rate",
                    "assign_binary_from_banded_rates",
                    "assign_uniform_draw",
                    "assign_clipped_normal",
                    "sample_categorical_from_count_table",
                }:
                    assert isinstance(operation.parameters.get("seed"), int)

    def test_e5_qrf_operation_declares_integer_seed(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        qrf = stages["was_wealth"].operations[2]

        assert qrf.kind == "fit_weighted_qrf_chain"
        assert qrf.parameters["seed"] == 0

    def test_e7_declared_seed_lockstep(self) -> None:
        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}

        assert (
            stages["spi_support_channel"].operations[0].parameters["seed"] == 42
        )
        assert (
            stages["hmrc_spi_income_spine"].operations[2].parameters["seed"] == 42
        )
        assert (
            stages["hmrc_spi_income_spine"].operations[3].parameters["seed"] == 43
        )

    def test_full_uk_source_stage_plan_compiles_with_e4_stages(self) -> None:
        spec = load_country_spec("uk")
        implementations = {name: _identity for name in UK_SOURCE_STAGE_NAMES}

        driver_stage_names = tuple(UK_FRS_SPI_SPINE_DRIVER_STAGE_NAMES)
        plan = country_stage_plan(spec, implementations, stage_names=driver_stage_names)

        assert [stage.name for stage in plan.stages] == list(driver_stage_names)

    def test_internal_disability_carriers_stay_out_of_export_registers(self) -> None:
        from microcosm.build.uk_runtime.frs_disability import (
            UK_INTERNAL_DISABILITY_REPORTED_COLUMNS,
        )
        from microcosm.build.uk_runtime.release_input_coverage import (
            uk_release_input_coverage_required_columns,
        )
        from microcosm.build.uk_runtime.terminal_gates import (
            UK_ALLOWED_EXTRA_EXPORT_COLUMNS,
        )

        gates = _load_json(UK_PACKAGE / "gates.json")
        export_gate = next(
            gate for gate in gates["gates"] if gate["id"] == "uk_export_surface"
        )
        allowed_extra = set(export_gate["parameters"]["allowed_extra_columns"])
        allowed_extra.update(UK_ALLOWED_EXTRA_EXPORT_COLUMNS)
        required = uk_release_input_coverage_required_columns()

        for column in UK_INTERNAL_DISABILITY_REPORTED_COLUMNS:
            assert f"person.{column}" not in allowed_extra
            assert column not in required

    def test_stage2_outputs_are_backed_by_runtime_written_columns(self) -> None:
        from microcosm.build.uk_runtime.spi_support import (
            SPI_HMRC_DERIVED_AUXILIARY_COLUMNS,
            SPI_HMRC_QRF_AUXILIARY_COLUMNS,
            SPI_INCOME_IMPUTATION_COLUMNS,
        )

        spec = load_country_spec("uk")
        stages = {stage.stage: stage for stage in spec.sources.stages}
        stage2 = stages["hmrc_spi_income"]
        written = (
            set(SPI_INCOME_IMPUTATION_COLUMNS)
            | set(SPI_HMRC_QRF_AUXILIARY_COLUMNS)
            | set(SPI_HMRC_DERIVED_AUXILIARY_COLUMNS)
        )
        # The narrow PAY+EPB+TAXTERM employment input is written on SPI rows
        # by the stage even though the QRF output surface excludes it.
        written.add("employment_income")
        unbacked = [name for name in stage2.outputs if name not in written]
        assert unbacked == [], (
            "Declared outputs with no named runtime written-column constant "
            f"backing them: {unbacked}. Either the manifest declares a "
            "concept instead of a persisted column, or the runtime constant "
            "moved without the manifest following."
        )
