"""The US plan declaration: complete or nothing, every donor cited."""

from __future__ import annotations

from pathlib import Path

import pytest

from populace.build.source_manifest import (
    SourceManifest,
    SourceOperationSpec,
    SupportSpineSourceSpec,
    SupportSpineSpec,
)
from populace.build.us_runtime import (
    US_DONORS,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_PUF_SUPPORT_STAGE_NAME,
    US_RETIREMENT_CONTRIBUTION_STAGE_NAME,
    US_SOURCE_MANIFEST,
    US_SOURCE_STAGE_SPECS,
    US_STAGE_NAMES,
    US_SUPPORT_SPINE_MANIFEST,
    US_SUPPORT_SPINE_SPEC,
    BuildConfig,
    us_plan,
)
from populace.build.us_runtime.puf_support import (
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
            US_RETIREMENT_CONTRIBUTION_STAGE_NAME
        ) < US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME)
        assert US_STAGE_NAMES.index(US_PUF_SUPPORT_STAGE_NAME) < US_STAGE_NAMES.index(
            "puf_tax_detail"
        )
        assert US_STAGE_NAMES.index("puf_tax_detail") < US_STAGE_NAMES.index(
            "education_inputs"
        )

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
                    "module": "populace.build.us.sources",
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
                    "callable_path": "populace.build.us_runtime.sources:clean",
                }
            )
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "entry-point": "populace.build.us_runtime.sources:clean",
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
                    "transform_path": "populace.build.us_runtime.sources:clean",
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
                                    "loader": "populace.build.us.sources:add_scf_wealth",
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
        from populace.build.us_runtime import sources
        from populace.build.us_runtime.sources import (
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

    def test_puf_stage_outputs_match_runtime_defaults(self) -> None:
        outputs = set(US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"].outputs)
        runtime_outputs = set(PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS) | set(
            PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
        )

        assert outputs == runtime_outputs

    def test_puf_stage_disaggregates_aggregate_records_before_uprating(self) -> None:
        operations = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"].operations
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
        assert "forbes" not in str(operation.parameters).lower()

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
        # The eCPS parity reference (populace #313) is the launch contract's
        # pinned record of the incumbent it replaces, so those files must NAME
        # the retired package: a sha-locked historical reference, never a live
        # dependency (the loader imports nothing from it). Nothing else in the
        # live tree may reference the retired data packages.
        allowed_incumbent_references = {
            "packages/populace-build/src/populace/build/us/ecps_parity_reference.json",
            "packages/populace-build/src/populace/build/us_runtime/parity_reference.py",
            "packages/populace-build/tests/test_us_parity_reference.py",
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
                or "out" in path.parts
                # Run scaffolding and staged run outputs (launchers, base-rebuild
                # summaries) are not shipped source and may record the incumbent
                # in local-path provenance; the shipped tree under packages/ is
                # still swept. Same non-shipped class as ``out``.
                or "experiments" in path.parts
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
