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
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from microcosm.build import (
    CountryResourceRow,
    CountrySpec,
    ResolvedCountrySpec,
    country_stage_plan,
    load_country_spec,
)
from microcosm.build.trace import canonical_json_bytes
from microcosm.build.uk_runtime import terminal_gates, weighted_integrity

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
            "spec/bundle.yaml",
            "spec/catalogs.yaml",
            "spec/geography.yaml",
            "spec/sources.yaml",
            "spec/spine.yaml",
            "spec/vintages.yaml",
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

    def test_default_stage_selection_still_requires_all_declared_stages(self) -> None:
        spec = load_country_spec("be")

        with pytest.raises(ValueError, match="missing \\['clone_assign_communes'\\]"):
            country_stage_plan(spec, {"silc_load": lambda frame: frame})

    def test_explicit_stage_subset_uses_manifest_order(self) -> None:
        spec = load_country_spec("be")
        plan = country_stage_plan(
            spec,
            {
                "silc_load": lambda frame: frame,
                "clone_assign_communes": lambda frame: frame,
            },
            stage_names=("clone_assign_communes", "silc_load"),
        )

        assert [stage.name for stage in plan.stages] == [
            "silc_load",
            "clone_assign_communes",
        ]

    def test_explicit_stage_subset_refuses_empty_or_unknown_names(self) -> None:
        spec = load_country_spec("be")
        implementations = {
            "silc_load": lambda frame: frame,
            "clone_assign_communes": lambda frame: frame,
        }

        with pytest.raises(ValueError, match="stage_names must not be empty"):
            country_stage_plan(spec, implementations, stage_names=())

        with pytest.raises(ValueError, match="Unknown stage selection"):
            country_stage_plan(
                spec,
                implementations,
                stage_names=("silc_load", "silc_load_fallback"),
            )


class TestUKCountryPackage:
    def test_spi_spine_adds_no_country_package_resources(self) -> None:
        # The name records the #717 question this was written to answer; what
        # it does now is pin the whole legacy-JSON resource list, so any
        # increment that ships a new country-package resource lands here.
        # spine_swap_signed_differences.json is #686's deliberate addition.
        spec = load_country_spec("uk")

        legacy_rows = tuple(
            row.path for row in spec.resource_rows if row.kind == "legacy_json"
        )
        assert legacy_rows == (
            "cgt_source_stages.json",
            "degenerate_reviewed_exclusions.json",
            "efrs_parity_known_gaps.json",
            "efrs_parity_reference.json",
            "frs_release.json",
            "gates.json",
            "brma_rent_counts.json",
            "calibration_measure_exclusions.json",
            "hmrc_cgt_size_bands.json",
            "advani_summers_capital_gains_distribution.json",
            "salary_sacrifice_anchor.json",
            "slc_liable_stocks.json",
            "cgt_band_donor_support_bounds.json",
            "hmrc_income_release_gate_report.json",
            "hmrc_income_replay_report.json",
            "hmrc_income_source_stages.json",
            "need_energy_targets.json",
            "lcfs_consumption_anchors.json",
            "etb_policy_anchors.json",
            "etb_services_anchors.json",
            "nhs_consumption_by_age_gender.json",
            "ons_age_tail_band_populations.json",
            "lcfs_consumption_support_bounds.json",
            "etb_vat_support_bounds.json",
            "etb_services_support_bounds.json",
            "regional_land_values.json",
            "source_stages.json",
            "take_up_contract.json",
            "input_mass_reviewed_exclusions.json",
            "spine_swap_signed_differences.json",
            "spine_candidate_acceptance.json",
            "ledger_compile_parity_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_production_2023_signed_differences.json",
            "national_staging_build_record.json",
            "parity_fixture_production_2023.json",
            "qrf_tail_reviewed_exclusions.json",
            "release_input_coverage_manifest.json",
            "registry_parity_fixture_2025.json",
            "local_registry_parity_fixture_2025.json",
            "was_wealth_support_bounds.json",
            "local_binding_adjudications.json",
            "uk_local_target_census.json",
            "uk_local_geography_targets.json",
            "uk_firms_targets.json",
            "local_area_crosswalk.json",
            "uk_national_targets.json",
            "target_references.json",
            "target_reference_membership.json",
            "local_target_references.json",
            "local_target_reference_membership.json",
        )

    def test_uk_source_manifest_loads_twenty_seven_stages(self) -> None:
        spec = load_country_spec("uk")

        assert spec.sources is not None
        # 24 spine stages (age_tail is the newest, #747) plus the two
        # certified-pair stages the June path still uses.
        assert len(spec.sources.stages) == 27


class TestExistingPackagesGeneralize:
    """The loader is country-neutral: the US and UK packages load unchanged."""

    def test_us_package_loads(self) -> None:
        spec = load_country_spec("us")
        assert spec.country == "us"
        assert spec.sources is not None
        assert spec.support_spine is not None
        # US target references live in fiscal_target_references.json (an
        # untyped resource its runtime interprets); Belgium and the UK use the
        # typed target_references.json convention.
        assert spec.target_references == ()

    def test_uk_package_loads(self) -> None:
        spec = load_country_spec("uk")
        assert spec.country == "uk"
        assert spec.resources == (
            "spec/bundle.yaml",
            "spec/catalogs.yaml",
            "spec/geography.yaml",
            "spec/sources.yaml",
            "spec/spine.yaml",
            "spec/vintages.yaml",
            "cgt_source_stages.json",
            "degenerate_reviewed_exclusions.json",
            "efrs_parity_known_gaps.json",
            "efrs_parity_reference.json",
            "frs_release.json",
            "gates.json",
            "brma_rent_counts.json",
            "calibration_measure_exclusions.json",
            "hmrc_cgt_size_bands.json",
            "advani_summers_capital_gains_distribution.json",
            "salary_sacrifice_anchor.json",
            "slc_liable_stocks.json",
            "cgt_band_donor_support_bounds.json",
            "hmrc_income_release_gate_report.json",
            "hmrc_income_replay_report.json",
            "hmrc_income_source_stages.json",
            "need_energy_targets.json",
            "lcfs_consumption_anchors.json",
            "etb_policy_anchors.json",
            "etb_services_anchors.json",
            "nhs_consumption_by_age_gender.json",
            "ons_age_tail_band_populations.json",
            "lcfs_consumption_support_bounds.json",
            "etb_vat_support_bounds.json",
            "etb_services_support_bounds.json",
            "regional_land_values.json",
            "source_stages.json",
            "take_up_contract.json",
            "input_mass_reviewed_exclusions.json",
            "spine_swap_signed_differences.json",
            "spine_candidate_acceptance.json",
            "ledger_compile_parity_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_local_incumbent_2025_signed_differences.json",
            "ledger_compile_parity_production_2023_signed_differences.json",
            "national_staging_build_record.json",
            "parity_fixture_production_2023.json",
            "qrf_tail_reviewed_exclusions.json",
            "release_input_coverage_manifest.json",
            "registry_parity_fixture_2025.json",
            "local_registry_parity_fixture_2025.json",
            "was_wealth_support_bounds.json",
            "local_binding_adjudications.json",
            "uk_local_target_census.json",
            "uk_local_geography_targets.json",
            "uk_firms_targets.json",
            "local_area_crosswalk.json",
            "uk_national_targets.json",
            "target_references.json",
            "target_reference_membership.json",
            "local_target_references.json",
            "local_target_reference_membership.json",
        )

    def test_uk_target_references_accept_regenerated_contract_fields(self) -> None:
        spec = load_country_spec("uk")

        references = {reference.name: reference for reference in spec.target_references}
        assert len(references) == 408
        assert references["obr.esa"].value_operation == "sum"
        assert references["dwp.uc.households"].value_operation == (
            "calendar_year_average"
        )
        assert (
            references["obr.income_tax"].assertion_policy == "allow_source_projection"
        )

        fanout = references["hmrc/employment_income_income_band_100_000_to_150_000"]
        assert fanout.metadata == {
            "contract_target_id": (
                "hmrc.spi.employment_income.amount_by_total_income_band"
            ),
            "measure_kind": "prepared_column",
        }
        assert fanout.uprating_from_period == "2023"
        assert fanout.uprating_to_period == 2025


class TestResolvedCountrySpecSeam:
    def test_country_spec_is_the_exact_resolved_alias(self) -> None:
        assert CountrySpec is ResolvedCountrySpec

    def test_generation_one_rows_retain_explicit_legacy_evidence(self) -> None:
        spec = load_country_spec("be")
        assert spec.resources == tuple(row.path for row in spec.resource_rows)
        typed = [row for row in spec.resource_rows if row.kind != "legacy_json"]
        legacy = [row for row in spec.resource_rows if row.kind == "legacy_json"]
        assert {row.kind for row in typed} == {
            "bundle",
            "catalogs",
            "geography",
            "sources",
            "spine",
            "vintages",
        }
        assert all(
            row.kind == "legacy_json" and row.schema_id == "legacy_json"
            for row in legacy
        )
        assert spec.resolved_spec is not None

    def test_resource_rows_are_frozen(self) -> None:
        row = CountryResourceRow(
            path="spec/bundle.yaml",
            kind="bundle",
            schema_id="bundle.schema.json",
        )
        with pytest.raises(FrozenInstanceError):
            row.path = "spec/changed.yaml"

    def test_typed_json_and_yaml_descriptors_load_together(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"] = [
            {
                "path": "gates.json",
                "kind": "legacy_json",
                "schema_id": "legacy_json",
            },
            {
                "path": "spec/bundle.yaml",
                "kind": "bundle",
                "schema_id": "bundle.schema.json",
            },
        ]
        del files["country_package.json"]["policy"]
        package_dir = _write_package(tmp_path, files)
        spec_dir = package_dir / "spec"
        spec_dir.mkdir()
        (spec_dir / "bundle.yaml").write_text(
            "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n",
            encoding="utf-8",
        )

        spec = load_country_spec(package_dir)

        assert spec.resources == ("gates.json", "spec/bundle.yaml")
        assert spec.gates is not None
        assert spec.resource_rows[1] == CountryResourceRow(
            path="spec/bundle.yaml",
            kind="bundle",
            schema_id="bundle.schema.json",
        )
        assert set(spec.resource_hashes) == {
            "country_package.json",
            "gates.json",
            "spec/bundle.yaml",
        }
        assert spec.resolved_spec is not None

    def test_generation_one_manifest_does_not_require_legacy_policy(
        self, tmp_path
    ) -> None:
        package_dir = tmp_path / "xx"
        package_dir.mkdir()
        (package_dir / "country_package.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "country": "xx",
                    "resources": [
                        {
                            "path": "bundle.yaml",
                            "kind": "bundle",
                            "schema_id": "bundle.schema.json",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (package_dir / "bundle.yaml").write_text(
            "country: xx\nidentity_generation: 1\nseed_protocol: legacy-v1\n",
            encoding="utf-8",
        )

        spec = load_country_spec(package_dir)

        assert spec.policy == ""
        assert spec.sources is None
        assert spec.gates is None
        assert spec.resolved_spec is not None
        assert spec.resolved_spec.country == "xx"

    def test_generated_locks_are_admitted_but_excluded_from_authority_hashes(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        files.update(
            {
                "bundle.lock.json": {},
                "engine_abi.lock.json": {},
                "plan.lock.json": {},
            }
        )
        spec = load_country_spec(_write_package(tmp_path, files))

        assert set(spec.resource_hashes) == {"country_package.json", "gates.json"}
        assert all("lock.json" not in resource for resource in spec.resources)

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            (
                {
                    "path": "../escape.yaml",
                    "kind": "bundle",
                    "schema_id": "bundle.schema.json",
                },
                "normalized local POSIX path",
            ),
            (
                {
                    "path": "bundle.yaml",
                    "kind": "executable",
                    "schema_id": "bundle.schema.json",
                },
                "unknown kind",
            ),
            (
                {
                    "path": "bundle.yaml",
                    "kind": "bundle",
                    "schema_id": "bundle.schema.json",
                    "entrypoint": "microcosm.build:run",
                },
                "closed-world",
            ),
            (
                {
                    "path": "engine_abi.lock.json",
                    "kind": "legacy_json",
                    "schema_id": "legacy_json",
                },
                "generated locks cannot be authored",
            ),
        ],
    )
    def test_invalid_typed_resource_rows_are_refused(
        self, tmp_path, row, message
    ) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"] = [row]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match=message):
            load_country_spec(package_dir)

    def test_duplicate_typed_paths_are_refused(self, tmp_path) -> None:
        files = _minimal_package()
        row = {
            "path": "gates.json",
            "kind": "legacy_json",
            "schema_id": "legacy_json",
        }
        files["country_package.json"]["resources"] = [row, dict(row)]
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="duplicate resource path"):
            load_country_spec(package_dir)


class TestUKGatesManifest:
    """The UK battery declared as data (microcosm#611 increment 1).

    Every threshold in ``uk/gates.json`` is pinned against the module
    constant the legacy battery still runs on, so spec and code cannot
    drift apart during the migration window (the constants retire when
    the national build swaps onto the battery executor).
    """

    @pytest.fixture(scope="class")
    def manifest(self):
        return load_country_spec("uk").gates

    def test_declares_the_uk_phases_in_order(self, manifest) -> None:
        assert manifest is not None
        assert manifest.phases == (
            "preflight",
            "assembled",
            "transferred",
            "terminal",
        )

    def test_declares_the_full_june_battery(self, manifest) -> None:
        assert [gate.id for gate in manifest.gates] == [
            "uk_release_input_coverage_manifest_current",
            "uk_release_family_build_stages",
            "uk_ledger_compile_parity_production_2023",
            "uk_ledger_compile_parity_incumbent_2025",
            "uk_stage_was_wealth_support",
            "uk_stage_lcfs_consumption_support",
            "uk_stage_etb_vat_support",
            "uk_stage_etb_services_support",
            "uk_stage_frs_hmrc_spine_leaves_signal",
            "uk_stage_spi_support_channel_mass",
            "uk_stage_hmrc_spi_income_spine_identity",
            "uk_stage_cgt_incidence_clone_mass",
            "uk_stage_cgt_band_donors_support",
            "uk_stage_hmrc_cgt_gains_spine_summary",
            "uk_stage_salary_sacrifice_realization",
            "uk_stage_student_loans_realization",
            "uk_stage_age_tail_targets",
            "uk_ledger_compile_parity_local_incumbent_2025",
            "uk_target_surface_local_default_2025",
            "uk_release_input_coverage",
            "uk_degenerate_release_surface",
            "uk_zero_weight_strata",
            "uk_weight_ess",
            "uk_weight_ratio",
            "uk_weights_audit",
            "uk_nonnegative_columns",
            "uk_support",
            "uk_aggregate_admin",
            "uk_export_surface",
            "uk_take_up_signal",
            "uk_brma_enum_domain",
            "uk_student_loan_plan_enum_domain",
            "uk_calibration_reference_coverage",
            "uk_target_surface",
            "uk_target_fit",
            "uk_input_mass_parity",
            "uk_qrf_tail_concentration",
        ]
        # Legacy behaviour: every evaluated failure raises, so every
        # declared entry blocks release.
        assert all(g.criticality == "release_blocking" for g in manifest.gates)

    def test_ledger_compile_parity_gates_pin_their_fixture_periods(
        self, manifest
    ) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}

        assert (
            params["uk_ledger_compile_parity_production_2023"]["target_period"] == 2023
        )
        assert (
            params["uk_ledger_compile_parity_incumbent_2025"]["target_period"] == 2025
        )
        assert (
            params["uk_ledger_compile_parity_local_incumbent_2025"]["target_period"]
            == 2025
        )
        assert (
            params["uk_ledger_compile_parity_local_incumbent_2025"]["registry_artifact"]
            == "uk_ledger_compiled_local_registries"
        )
        assert (
            params["uk_target_surface_local_default_2025"]["expected"]
            == "local_default_surface"
        )
        assert (
            params["uk_target_surface_local_default_2025"]["registry_artifact"]
            == "uk_ledger_compiled_local_registries"
        )

    def test_strict_absent_evidence_entries_are_declared(self, manifest) -> None:
        # "An absent audit is not a passing audit" — the retired schema-3
        # path blocked every posture on a missing fit-weight audit, and the
        # battery keeps that strictness via the entry flag (#654, #691
        # review). Stage-health gates also block on absent receipts because
        # the spine build cannot silently skip a checkpoint's own evidence.
        flagged = [g.id for g in manifest.gates if g.evidence_absent_blocks]
        assert flagged == [
            "uk_stage_was_wealth_support",
            "uk_stage_lcfs_consumption_support",
            "uk_stage_etb_vat_support",
            "uk_stage_etb_services_support",
            "uk_stage_frs_hmrc_spine_leaves_signal",
            "uk_stage_spi_support_channel_mass",
            "uk_stage_hmrc_spi_income_spine_identity",
            "uk_stage_cgt_incidence_clone_mass",
            "uk_stage_cgt_band_donors_support",
            "uk_stage_hmrc_cgt_gains_spine_summary",
            "uk_stage_salary_sacrifice_realization",
            "uk_stage_student_loans_realization",
            "uk_stage_age_tail_targets",
            "uk_weights_audit",
        ]
        assert all(g.not_applicable is None for g in manifest.gates)

    def test_gate_names_are_country_neutral(self, manifest) -> None:
        by_id = {gate.id: gate.gate for gate in manifest.gates}
        # The two legacy names the bindings re-mint to the shared vocabulary.
        assert by_id["uk_release_input_coverage"] == "release_input_coverage"
        assert by_id["uk_qrf_tail_concentration"] == "tail_concentration"
        assert not any(name.startswith("uk_") for name in by_id.values())

    def test_thresholds_match_the_schema4_manifest(self, manifest) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}
        assert params["uk_weight_ess"]["minimum_ess_fraction"] == 0.01
        assert (
            params["uk_weight_ratio"]["maximum_max_to_median_ratio"]
            == 1_151.2542195939373
        )
        assert params["uk_input_mass_parity"]["relative_tolerance"] == 4.521811483823806
        assert params["uk_input_mass_parity"]["minimum_reference_total"] == 0.0
        assert params["uk_qrf_tail_concentration"]["top_k"] == 100
        assert (
            params["uk_qrf_tail_concentration"]["max_top_share"] == 0.9994670564654868
        )
        assert params["uk_qrf_tail_concentration"]["min_nonzero_records"] == 104
        assert (
            params["uk_target_fit"]["max_abs_relative_error"]
            == terminal_gates.UK_MAX_TARGET_ABS_RELATIVE_ERROR
        )
        assert params["uk_support"]["support_bounds_resources"] == (
            "was_wealth_support_bounds.json",
            "lcfs_consumption_support_bounds.json",
            "etb_vat_support_bounds.json",
            "etb_services_support_bounds.json",
        )
        aggregate = params["uk_aggregate_admin"]
        assert aggregate["default_rtol"] == 0.15
        assert [anchor["name"] for anchor in aggregate["anchors"]] == [
            "need_electricity_mean_spending",
            "need_gas_mean_spending",
            "nhs_spending_total",
        ]

    def test_zero_weight_declarations_match_the_june_strata(self, manifest) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}
        declared = params["uk_zero_weight_strata"]["declarations"]
        strata = terminal_gates.UK_DEFAULT_ZERO_WEIGHT_STRATA
        assert len(declared) == len(strata)
        for entry, stratum in zip(declared, strata, strict=True):
            assert entry["name"] == stratum.name
            assert dict(entry["selector"]) == stratum.selector
            assert entry["maximum_zero_weight_rows"] == (
                stratum.maximum_zero_weight_rows
            )
            assert entry["reason"] == stratum.reason

    def test_export_surface_registers_match_the_reviewed_constants(
        self, manifest
    ) -> None:
        params = {gate.id: gate.parameters for gate in manifest.gates}
        export = params["uk_export_surface"]
        assert (
            export["allowed_extra_columns"]
            == terminal_gates.UK_ALLOWED_EXTRA_EXPORT_COLUMNS
        )
        assert (
            dict(export["reviewed_exclusions"])
            == terminal_gates.UK_REVIEWED_EXPORT_EXCLUSIONS
        )

    def test_input_mass_reference_is_a_declared_pinned_input(self, manifest) -> None:
        # The microcosm#327 rule: a parity gate's reference and exclusion
        # register are declared per-country inputs, never implicit code.
        params = {gate.id: gate.parameters for gate in manifest.gates}
        input_mass = params["uk_input_mass_parity"]
        assert input_mass["reference"] in input_mass["reference_registry"]
        expected_registry = {
            name: descriptor.spec_payload()
            for name, descriptor in (
                weighted_integrity.UK_INPUT_MASS_REFERENCE_REGISTRY.items()
            )
        }
        assert input_mass["reference_registry"] == expected_registry
        assert (
            input_mass["reviewed_exclusions_resource"]
            == weighted_integrity.UK_INPUT_MASS_EXCLUSION_REGISTER_RESOURCE
        )
        qrf = params["uk_qrf_tail_concentration"]
        assert (
            qrf["reviewed_exclusions_resource"]
            == weighted_integrity.UK_QRF_TAIL_EXCLUSION_REGISTER_RESOURCE
        )
        degenerate = params["uk_degenerate_release_surface"]
        assert (
            degenerate["reviewed_exclusions_resource"]
            == weighted_integrity.UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
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

    def test_non_bool_evidence_absent_blocks_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["evidence_absent_blocks"] = "yes"
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="evidence_absent_blocks must be"):
            load_country_spec(package_dir)

    def test_evidence_absent_blocks_on_an_excused_entry_is_refused(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        entry = files["gates.json"]["gates"][0]
        entry.pop("parameters", None)
        entry["not_applicable"] = "reviewed: no surface yet"
        entry["evidence_absent_blocks"] = True
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(ValueError, match="mutually exclusive"):
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

    def test_unknown_gate_entry_key_is_refused(self, tmp_path) -> None:
        files = _minimal_package()
        files["gates.json"]["gates"][0]["paramters"] = {"within": 0.1}
        package_dir = _write_package(tmp_path, files)
        with pytest.raises(
            ValueError, match=r"gate entry 'fit' has unknown keys \['paramters'\]"
        ):
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

    def test_sum_target_reference_roundtrips(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].append("target_references.json")
        files["target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "summed",
                    "ledger_selector": {"source_name": "somewhere"},
                    "value_operation": "sum",
                    "entity": "person",
                    "measure": "people",
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        spec = load_country_spec(package_dir)

        assert spec.target_references[0].value_operation == "sum"

    def test_local_target_reference_roundtrips_with_crosswalk_roster(
        self, tmp_path
    ) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].extend(
            ["local_area_crosswalk.json", "local_target_references.json"]
        )
        files["local_area_crosswalk.json"] = {
            "country": "xx",
            "levels": {
                "constituency": {
                    "expected_vintage": "test_vintage",
                    "area_ids": ["A1"],
                }
            },
        }
        files["local_target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "ons.age.0_10@A1",
                    "ledger_selector": {
                        "source_name": "ons",
                        "source_measure_id": "population",
                        "geography_level": "constituency",
                        "geography_id": "A1",
                    },
                    "value_operation": "sum",
                    "entity": "person",
                    "measure": "age/0_10",
                    "metadata": {"contract_target_id": "ons.age.0_10"},
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        spec = load_country_spec(package_dir)

        assert spec.target_references == ()
        assert len(spec.local_target_references) == 1
        assert spec.local_target_references[0].name == "ons.age.0_10@A1"

    def test_local_target_reference_refuses_unknown_roster_area(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].extend(
            ["local_area_crosswalk.json", "local_target_references.json"]
        )
        files["local_area_crosswalk.json"] = {
            "country": "xx",
            "levels": {
                "constituency": {
                    "expected_vintage": "test_vintage",
                    "area_ids": ["A1"],
                }
            },
        }
        files["local_target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "ons.age.0_10@A2",
                    "ledger_selector": {
                        "source_name": "ons",
                        "source_measure_id": "population",
                        "geography_level": "constituency",
                        "geography_id": "A2",
                    },
                    "entity": "person",
                    "measure": "age/0_10",
                    "metadata": {"contract_target_id": "ons.age.0_10"},
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="test_vintage"):
            load_country_spec(package_dir)

    def test_local_target_reference_refuses_unpinned_name(self, tmp_path) -> None:
        files = _minimal_package()
        files["country_package.json"]["resources"].extend(
            ["local_area_crosswalk.json", "local_target_references.json"]
        )
        files["local_area_crosswalk.json"] = {
            "country": "xx",
            "levels": {
                "constituency": {
                    "expected_vintage": "test_vintage",
                    "area_ids": ["A1"],
                }
            },
        }
        files["local_target_references.json"] = {
            "country": "xx",
            "target_references": [
                {
                    "name": "ons.age.0_10",
                    "ledger_selector": {
                        "source_name": "ons",
                        "source_measure_id": "population",
                        "geography_level": "constituency",
                        "geography_id": "A1",
                    },
                    "entity": "person",
                    "measure": "age/0_10",
                    "metadata": {"contract_target_id": "ons.age.0_10"},
                }
            ],
        }
        package_dir = _write_package(tmp_path, files)

        with pytest.raises(ValueError, match="target_id@geography_id"):
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
