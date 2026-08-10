"""The declarative country spec: loading, refusals, and the Belgian package.

Belgium is the first full consumer of the country-spec schema
(microcosm#261): its package declares sources, geography spine, target
references, gates, and release contract as pure data. The golden-file test
pins the loaded spec — stage order, gate selection, release contract, and
the sha256 of every resource — so any byte change to a BE spec is a
reviewed diff against ``tests/golden/be_country_spec.json``, never an
accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from microcosm.build import (
    country_stage_plan,
    load_country_spec,
)
from microcosm.build.trace import canonical_json_bytes

GOLDEN = Path(__file__).parent / "golden" / "be_country_spec.json"


def _write_package(root: Path, files: dict[str, dict]) -> Path:
    package_dir = root / "xx"
    package_dir.mkdir()
    for name, payload in files.items():
        (package_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    return package_dir


def _minimal_package(**overrides) -> dict[str, dict]:
    files = {
        "country_package.json": {
            "schema_version": 1,
            "country": "xx",
            "policy": "spec-only test package",
            "resources": ["gates.json"],
        },
        "gates.json": {
            "version": 1,
            "country": "xx",
            "policy": "test gates",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "fit",
                    "gate": "per_family_fit",
                    "phase": "terminal",
                    "criticality": "release_blocking",
                }
            ],
        },
    }
    files.update(overrides)
    return files


class TestBelgianPackage:
    @pytest.fixture(scope="class")
    def spec(self):
        return load_country_spec("be")

    def test_loads_with_every_declared_resource(self, spec) -> None:
        assert spec.country == "be"
        assert set(spec.resources) == {
            "source_stages.json",
            "geography_spine.json",
            "target_references.json",
            "gates.json",
            "release_contract.json",
        }
        assert set(spec.resource_hashes) == set(spec.resources) | {
            "country_package.json"
        }

    def test_source_stage_declares_the_silc_contract(self, spec) -> None:
        stage = spec.sources.stage_map()["silc_load"]
        assert stage.grain == "person"
        kinds = [operation.kind for operation in stage.operations]
        assert "declare_income_reference_offset" in kinds
        assert "map_columns" in kinds
        offsets = [
            operation.parameters["years"]
            for operation in stage.operations
            if operation.kind == "declare_income_reference_offset"
        ]
        assert offsets == [-1]  # SILC year N carries year N-1 incomes
        assert "belgium_pit_article_23_worker_remuneration" in stage.outputs
        assert "belgium_pit_article_23_worker_remuneration" in stage.nonnegative_outputs

    def test_geography_spine_is_vintage_aware(self, spec) -> None:
        spine = spec.geography_spine.geography_spine
        assert spine.geography_level == "commune"
        assert spine.code_system == "be_nis"
        assert spine.vintage == "2025"
        assert spine.vintage_policy == "error"
        assert spine.collision_avoidance is True
        assert spine.constrain_to_column == "region_nuts1"

    def test_targets_arrive_by_reference_with_no_values(self, spec) -> None:
        names = {reference.name for reference in spec.target_references}
        assert {
            "statbel_population_by_age_sex_region",
            "statbel_fiscal_income_by_commune",
            "spf_finances_pit_total",
            "onss_employee_contribution_total",
            "onem_unemployment_caseload",
            "nbb_household_disposable_income",
        } <= names
        by_name = {reference.name: reference for reference in spec.target_references}
        commune = by_name["statbel_fiscal_income_by_commune"]
        assert commune.metadata["nis_vintage"] == "2025"
        assert commune.metadata["criticality"] == "diagnostic"

    def test_gates_select_no_incumbent_comparison(self, spec) -> None:
        selected = {gate.gate for gate in spec.gates.gates}
        assert "parity" not in selected  # no incumbent: oracles replace it
        assert "export_surface" not in selected
        assert "per_family_fit" in selected
        assert "formula_owned_export" in selected
        blocking = [
            gate.id
            for gate in spec.gates.gates
            if gate.criticality == "release_blocking"
        ]
        assert "target_profile_coverage" in blocking
        diagnostic = [
            gate.id for gate in spec.gates.gates if gate.criticality == "diagnostic"
        ]
        assert "commune_fiscal_income_fit" in diagnostic

    def test_release_contract_is_private_and_ordinal_free(self, spec) -> None:
        contract = spec.release_contract
        assert contract.artifact_repo == "policyengine/populace-be-private"
        assert contract.artifact_repo_private is True
        assert contract.licence_restricted is True
        assert contract.dataset_filename_template == "populace_be_{year}.h5"
        assert "source_coverage.json" in contract.required_release_files
        assert "reform_validation.json" in contract.required_release_files

    def test_gates_declare_their_phase_order(self, spec) -> None:
        assert spec.gates.phases == ("terminal",)
        assert {gate.phase for gate in spec.gates.gates} == {"terminal"}

    def test_fingerprint_is_stable_across_loads(self, spec) -> None:
        assert load_country_spec("be").fingerprint == spec.fingerprint


class TestGoldenBelgianSpec:
    def test_loaded_spec_matches_the_golden_file_byte_for_byte(self) -> None:
        spec = load_country_spec("be")
        summary = {
            "country": spec.country,
            "fingerprint": spec.fingerprint,
            "resources": list(spec.resources),
            "resource_hashes": dict(spec.resource_hashes),
            "stage_names": [stage.stage for stage in spec.sources.stages],
            "geography_spine_stage": spec.geography_spine.geography_spine.stage,
            "target_reference_names": [
                reference.name for reference in spec.target_references
            ],
            "gate_ids": [gate.id for gate in spec.gates.gates],
            "release": {
                "builder": spec.release_contract.builder,
                "artifact_repo": spec.release_contract.artifact_repo,
                "staging_repo": spec.release_contract.staging_repo,
                "dataset_filename_template": (
                    spec.release_contract.dataset_filename_template
                ),
                "required_release_files": list(
                    spec.release_contract.required_release_files
                ),
            },
        }
        rendered = canonical_json_bytes(summary)
        assert GOLDEN.exists(), (
            "Golden file missing. Generate it after reviewing the spec:\n"
            f'  python -c "..." > {GOLDEN}'
        )
        assert rendered == GOLDEN.read_bytes(), (
            "The Belgian country spec changed. If intentional, regenerate "
            "tests/golden/be_country_spec.json from the loaded spec and "
            "review the diff; resource hashes pin every spec byte."
        )


class TestCountryStagePlan:
    def test_compiles_with_noop_implementations_in_declared_order(self) -> None:
        spec = load_country_spec("be")
        names = [stage.stage for stage in spec.sources.stages] + [
            spec.geography_spine.geography_spine.stage
        ]
        plan = country_stage_plan(spec, {name: (lambda frame: frame) for name in names})
        assert [stage.name for stage in plan.stages] == [
            "silc_load",
            "clone_assign_communes",
        ]
        donors = dict(plan.donors())
        assert donors["silc_load"].source.startswith("https://")
        assert donors["clone_assign_communes"].survey.startswith("Statbel")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        spec = load_country_spec("be")
        with pytest.raises(ValueError, match="missing \\['clone_assign_communes'\\]"):
            country_stage_plan(spec, {"silc_load": lambda frame: frame})

    def test_unknown_stage_is_refused(self) -> None:
        spec = load_country_spec("be")
        names = [stage.stage for stage in spec.sources.stages] + [
            spec.geography_spine.geography_spine.stage,
            "silc_load_fallback",
        ]
        with pytest.raises(ValueError, match="Unknown stage implementation"):
            country_stage_plan(spec, {name: (lambda frame: frame) for name in names})


class TestExistingPackagesGeneralize:
    """The loader is country-neutral: the US and UK packages load unchanged."""

    def test_us_package_loads(self) -> None:
        spec = load_country_spec("us")
        assert spec.country == "us"
        assert spec.sources is not None
        assert spec.support_spine is not None
        # US target references live in fiscal_target_references.json (an
        # untyped resource its runtime interprets); the typed
        # target_references.json convention starts with Belgium.
        assert spec.target_references == ()

    def test_uk_package_loads(self) -> None:
        spec = load_country_spec("uk")
        assert spec.country == "uk"
        assert spec.resources == (
            "degenerate_reviewed_exclusions.json",
            "efrs_parity_known_gaps.json",
            "efrs_parity_reference.json",
            "hmrc_income_release_gate_report.json",
            "hmrc_income_replay_report.json",
            "hmrc_income_source_stages.json",
            "input_mass_reviewed_exclusions.json",
            "national_staging_build_record.json",
            "qrf_tail_reviewed_exclusions.json",
            "release_input_coverage_manifest.json",
            "uk_local_target_census.json",
        )


class TestRefusals:
    def test_undeclared_file_on_disk_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        package_dir = _write_package(tmp_path, files)
        (package_dir / "stray.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="stray.json"):
            load_country_spec(package_dir)

    def test_missing_declared_resource_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"] = ["gates.json", "absent.json"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(FileNotFoundError, match="absent.json"):
            load_country_spec(package_dir)

    def test_country_mismatch_in_a_resource_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["country"] = "yy"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="declares country 'yy'"):
            load_country_spec(package_dir)

    def test_unknown_gate_function_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["gate"] = "vibes"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="unknown gate function 'vibes'"):
            load_country_spec(package_dir)

    def test_all_diagnostic_gates_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["criticality"] = "diagnostic"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="release_blocking"):
            load_country_spec(package_dir)

    def test_gate_without_a_phase_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        del files["gates.json"]["gates"][0]["phase"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="phase must be a non-empty string"):
            load_country_spec(package_dir)

    def test_unknown_gate_phase_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["phase"] = "someday"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="unknown phase 'someday'"):
            load_country_spec(package_dir)

    def test_missing_phase_order_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        del files["gates.json"]["phases"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="phases must be a non-empty list"):
            load_country_spec(package_dir)

    def test_gate_phase_outside_the_declared_order_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["phase"] = "preflight"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="not in the declared phase order"):
            load_country_spec(package_dir)

    def test_duplicate_phases_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["phases"] = ["terminal", "terminal"]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="duplicate phase"):
            load_country_spec(package_dir)

    def test_not_applicable_with_parameters_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["not_applicable"] = "no surface yet"
        files["gates.json"]["gates"][0]["parameters"] = {"within": 0.1}
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="mutually exclusive"):
            load_country_spec(package_dir)

    def test_empty_not_applicable_reason_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["not_applicable"] = "  "
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(
            ValueError, match="not_applicable must be a non-empty string"
        ):
            load_country_spec(package_dir)

    def test_target_reference_carrying_a_value_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("target_references.json")
        files["target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "smuggled",
                    "ledger_selector": {"source_name": "somewhere"},
                    "entity": "person",
                    "measure": "people",
                    "value": 123.0,
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="values live in Ledger"):
            load_country_spec(package_dir)

    def test_target_reference_carrying_a_nested_value_is_refused(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("target_references.json")
        files["target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "smuggled_nested",
                    "ledger_selector": {"source_name": "somewhere", "value": 123.0},
                    "entity": "person",
                    "measure": "people",
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="values live in Ledger"):
            load_country_spec(package_dir)

    def test_restricted_licence_requires_a_private_repo(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace-xx",
            "hf": {
                "artifact_repo": "policyengine/populace-xx",
                "private": False,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="requires a private artifact repo"):
            load_country_spec(package_dir)

    def test_ordinal_version_tokens_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace-xx-v2",
            "hf": {
                "artifact_repo": "policyengine/populace-xx-private",
                "private": True,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="ordinal version token"):
            load_country_spec(package_dir)

    def test_embedded_ordinal_version_tokens_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace_xx_v2_staging",
            "hf": {
                "artifact_repo": "policyengine/populace-xx-private",
                "private": True,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="ordinal version token"):
            load_country_spec(package_dir)

    def test_version_like_substrings_without_ordinal_tokens_load(
        self, tmp_path
    ) -> None:
        # "sha-v2x" is not an ordinal token: the digits run into a letter.
        files = _minimal_package()
        files["country_package.json"]["resources"].append("release_contract.json")
        files["release_contract.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "builder": "populace-xx-sha-v2x",
            "hf": {
                "artifact_repo": "policyengine/populace-xx-private",
                "private": True,
                "staging_repo": "policyengine/populace-xx-staging",
            },
            "dataset_filename_template": "microcosm_xx_{year}.h5",
            "required_release_files": ["release_manifest.json"],
            "boundary": {"private": ["microcosm_xx_{year}.h5"], "public": []},
            "licence": {"name": "restricted survey", "restricted": True},
        }
        package_dir = _write_package(tmp_path, files)
        spec = load_country_spec(package_dir)
        assert spec.release_contract is not None

    def test_geography_vintage_policy_must_be_error(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("geography_spine.json")
        files["geography_spine.json"] = {
            "version": 1,
            "country": "xx",
            "policy": "test",
            "geography_spine": {
                "stage": "clone_assign",
                "method": "clone_assign_uniform",
                "geography_level": "area",
                "code_system": "xx_codes",
                "code_column": "area_code",
                "vintage": "2025",
                "vintage_policy": "warn",
                "clones_per_record": 2,
                "collision_avoidance": True,
                "constrain_to_column": "",
                "assignment_source": {
                    "survey": "Test census",
                    "source": "https://example.test/census",
                },
            },
        }
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="vintage_policy"):
            load_country_spec(package_dir)
