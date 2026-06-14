"""The US plan declaration: complete or nothing, every donor cited."""

from __future__ import annotations

from pathlib import Path

import pytest

from populace.build.source_manifest import SourceManifest, SourceOperationSpec
from populace.build.us import (
    US_DONORS,
    US_NONNEGATIVE_SOURCE_OUTPUTS,
    US_SOURCE_MANIFEST,
    US_SOURCE_STAGE_SPECS,
    US_STAGE_NAMES,
    BuildConfig,
    us_plan,
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

    def test_every_donor_stage_has_matching_source_spec(self) -> None:
        specs = US_SOURCE_MANIFEST.stage_map()
        for stage, donor in US_DONORS.items():
            assert stage in specs
            assert specs[stage].survey == donor.survey
            assert specs[stage].source == donor.source

    def test_source_specs_align_with_declared_plan(self) -> None:
        derived_source_specs = {"mortgage_conversion"}
        assert {spec.stage for spec in US_SOURCE_STAGE_SPECS} == set(
            US_DONORS
        ) | derived_source_specs
        assert set(US_DONORS).issubset(US_STAGE_NAMES)
        assert derived_source_specs.issubset(US_STAGE_NAMES)

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
        with pytest.raises(ValueError, match="allowed manifest operation"):
            SourceOperationSpec.from_mapping({"kind": "custom_loader"})
        with pytest.raises(ValueError, match="executable-loader key"):
            SourceOperationSpec.from_mapping(
                {
                    "kind": "read_table",
                    "table": "scf_household",
                    "postprocess": [{"callable": "clean_scf"}],
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
        from populace.build.us import sources
        from populace.build.us.sources import (
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
        assert "auto_loan_interest" in specs["scf_wealth"].nonnegative_outputs
        assert "auto_loan_interest" in US_NONNEGATIVE_SOURCE_OUTPUTS
        assert "net_worth" not in US_NONNEGATIVE_SOURCE_OUTPUTS

    def test_no_incumbent_data_package_references_in_live_tree(self) -> None:
        forbidden = (
            "policyengine_" + "us_data",
            "policyengine-" + "us-data",
            "policyengine_" + "uk_data",
            "policyengine-" + "uk-data",
        )
        checked_suffixes = {".py", ".toml", ".md", ".json"}
        offenders: list[tuple[str, str]] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in checked_suffixes:
                continue
            if ".git" in path.parts or ".venv" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                if needle in text:
                    offenders.append((str(path.relative_to(ROOT)), needle))
        assert offenders == []
