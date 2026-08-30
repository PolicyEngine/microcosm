"""Shared-core country compile proofs for the F0 CountrySpec seam."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.spec_engine import load_bundle
from microcosm.build.spec_engine.compiler_ir import compile_spec
from microcosm.build.spec_engine.resolver import (
    F0_CONTRACT_ONLY_KERNEL_IDS,
    F0_IMPLEMENTED_KERNEL_IDS,
    F0_KERNEL_REGISTRY,
    KernelRegistry,
)

EXPECTED_RESOURCES = {
    "bundle",
    "catalogs",
    "geography",
    "sources",
    "spine",
    "vintages",
}
AM_SPEC_SHA256 = "659b6baf5ebbd71fb7786ec4c4d49df565b2bddabeb868a9385ed226c56880f9"
NZ_SPEC_SHA256 = "7ca62ede964e4e0d5d9d700a136ff2be170b863de3f9fe9e2577394bac12f92f"
NZ_COLUMN_KEYS = {
    "family.best_start_abatement_days",
    "family.best_start_child_care_fraction",
    "family.best_start_entitlement_days",
    "family.best_start_family_scheme_income_for_relationship_period",
    "family.child_tax_credit_for_entitlement_period",
    "family.entitled_to_in_work_tax_credit",
    "family.family_household_id",
    "family.family_id",
    "family.family_tax_credit_eldest_dependent_child_care_units",
    "family.family_tax_credit_entitlement_days",
    "family.family_tax_credit_subsequent_dependent_child_care_units",
    "family.in_work_tax_credit_allowed_children_count",
    "family.in_work_tax_credit_weekly_periods",
    "family.minimum_family_adjusted_income_tax_liability",
    "family.minimum_family_amount_paid",
    "family.minimum_family_amount_received",
    "family.minimum_family_full_time_earner_weeks",
    "family.minimum_family_scheme_income_attributable_to_full_time_weeks",
    "family.minimum_family_tax_credit_weekly_periods",
    "family.parental_tax_credit_additional_abatement",
    "family.parental_tax_credit_for_entitlement_period",
    "family.wff_family_credit_abatement_days",
    "family.wff_family_scheme_income_for_relationship_period",
    "household.household_id",
    "household.household_weight",
    "person.age",
    "person.donor_country_code",
    "person.person_family_id",
    "person.person_household_id",
    "person.person_id",
    "person.region_code",
    "person.sex",
    "person.support_stratum",
}


@pytest.mark.parametrize(
    ("country", "expected_spec_sha256", "expected_columns", "expected_entities"),
    [
        (
            "am",
            AM_SPEC_SHA256,
            {
                "household.household_id",
                "person.age",
                "person.marz_code",
                "person.person_id",
                "person.sex",
            },
            {"household", "person"},
        ),
        (
            "be",
            "7062e38f4d623553fb0604380a8dac0edacb6261c155b6e31fc38ef7c0f1c57c",
            {
                "household.household_id",
                "person.person_id",
                "person.region_nuts1",
            },
            {"household", "person"},
        ),
        (
            "nz",
            NZ_SPEC_SHA256,
            NZ_COLUMN_KEYS,
            {"family", "household", "person"},
        ),
        (
            "uk",
            "cce1c98ea40364a398ae361f4d15790c925d8379d8b9b427076d61059c7d6715",
            {
                "benunit.benunit_id",
                "household.household_id",
                "household.region",
                "person.person_id",
            },
            {"benunit", "household", "person"},
        ),
    ],
)
def test_country_bundle_loads_once_and_compiles_through_the_shared_core(
    country: str,
    expected_spec_sha256: str,
    expected_columns: set[str],
    expected_entities: set[str],
) -> None:
    if country == "us":
        pytest.importorskip(
            "policyengine_us",
            reason="US compile proof reads the live engine ABI",
            exc_type=ModuleNotFoundError,
        )
    direct = load_bundle(country)
    country_spec = load_country_spec(country)

    assert country_spec.resolved_spec is not None
    assert country_spec.resolved_spec.spec_sha256 == direct.spec_sha256
    assert direct.spec_sha256 == expected_spec_sha256

    compiled = compile_spec(direct)
    assert set(compiled.resources_wire()) == EXPECTED_RESOURCES
    assert not compiled.producer_graph.present
    assert compiled.nodes == ()
    assert {column.key for column in direct.columns} == expected_columns
    assert {column.entity.id for column in direct.columns} == expected_entities


def test_country_bundles_exercise_distinct_support_and_geography_kinds() -> None:
    am = compile_spec(load_bundle("am"))
    be = compile_spec(load_bundle("be"))
    nz = compile_spec(load_bundle("nz"))
    uk = compile_spec(load_bundle("uk"))

    assert am.resource("spine")["support_roles"] == [
        {"id": "populace_us_donor_base", "kind": "none"}
    ]
    assert be.resource("spine")["support_roles"] == [
        {"id": "silc_base", "kind": "none"}
    ]
    assert nz.resource("spine")["support_roles"] == [
        {"id": "populace_us_donor_base", "kind": "none"}
    ]
    assert uk.resource("spine")["support_roles"] == [
        {
            "id": "spi_income_support",
            "kind": "synthetic_prior_replacement",
            "clone_index": 1,
        }
    ]
    assert am.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:clone_assign_communities",
        "validate": "kernel:am_community_geography_gate",
    }
    assert be.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:clone_assign_communes",
        "validate": "kernel:be_commune_geography_gate",
    }
    assert nz.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:assign_nz_regions",
        "validate": "kernel:nz_region_geography_gate",
    }
    assert uk.resource("geography")["assignment"]["kernels"] == {
        "assign": "kernel:assign_uk_geography_ladder",
        "validate": "kernel:uk_geography_ladder_gate",
    }


def test_country_kernel_contract_ids_are_closed_in_the_compiler_registry() -> None:
    country_contract_ids = {
        "am_community_geography_gate",
        "assign_am_marz",
        "assign_nz_regions",
        "clone_assign_communities",
        "load_populace_us_support_pool",
        "silc_load",
        "clone_assign_communes",
        "be_commune_geography_gate",
        "load_uk_national_frame",
        "nz_region_geography_gate",
        "prepare_nz_wff_axiom_inputs",
        "assign_uk_geography_ladder",
        "uk_geography_ladder_gate",
    }

    assert country_contract_ids == F0_CONTRACT_ONLY_KERNEL_IDS
    assert country_contract_ids <= F0_KERNEL_REGISTRY.ids
    assert all(F0_KERNEL_REGISTRY.contains(value) for value in country_contract_ids)
    assert not any(
        F0_KERNEL_REGISTRY.has_implementation(value) for value in country_contract_ids
    )


def test_country_contract_ids_do_not_change_the_implementation_digest() -> None:
    implemented_only = KernelRegistry.from_ids(F0_IMPLEMENTED_KERNEL_IDS)

    assert F0_KERNEL_REGISTRY.implemented_ids == implemented_only.ids
    assert (
        F0_KERNEL_REGISTRY.implementation_sha256
        == implemented_only.implementation_sha256
    )


def test_am_generation_zero_views_come_from_the_country_spec_seam() -> None:
    spec = load_country_spec("am")

    assert spec.sources is not None
    assert tuple(spec.sources.stage_map()) == (
        "load_populace_us_support_pool",
        "assign_am_marz",
    )
    assert spec.geography_spine is not None
    assert spec.geography_spine.geography_spine.stage == "clone_assign_communities"
    assert spec.gates is not None
    assert spec.release_contract is not None
    assert not spec.release_contract.artifact_repo_private
    assert {row.path for row in spec.resource_rows if row.kind == "legacy_json"} == {
        "gates.json",
        "geography_spine.json",
        "release_contract.json",
        "source_stages.json",
        "target_references.json",
    }


def test_be_generation_zero_views_still_come_from_the_country_spec_seam() -> None:
    spec = load_country_spec("be")

    assert spec.sources is not None
    assert tuple(spec.sources.stage_map()) == ("silc_load",)
    assert spec.geography_spine is not None
    assert spec.geography_spine.geography_spine.stage == "clone_assign_communes"
    assert spec.gates is not None
    assert spec.release_contract is not None
    assert spec.release_contract.artifact_repo_private
    assert {row.path for row in spec.resource_rows if row.kind == "legacy_json"} == {
        "gates.json",
        "geography_spine.json",
        "release_contract.json",
        "source_stages.json",
        "target_references.json",
    }


def test_nz_generation_zero_views_come_from_the_country_spec_seam() -> None:
    spec = load_country_spec("nz")

    assert spec.sources is not None
    assert tuple(spec.sources.stage_map()) == (
        "load_populace_us_support_pool",
        "prepare_nz_wff_axiom_inputs",
    )
    assert spec.geography_spine is not None
    assert spec.geography_spine.geography_spine.stage == "assign_nz_regions"
    assert spec.gates is not None
    assert spec.release_contract is not None
    assert not spec.release_contract.artifact_repo_private
    assert {row.path for row in spec.resource_rows if row.kind == "legacy_json"} == {
        "export_contract.json",
        "gates.json",
        "geography_spine.json",
        "release_contract.json",
        "source_stages.json",
        "target_references.json",
    }


@pytest.mark.parametrize(
    ("before", "after", "field"),
    [
        (
            "kernel:clone_assign_communes",
            "kernel:assign_uk_geography_ladder",
            "assignment/kernels/assign",
        ),
        ("anchor: commune", "anchor: municipality", "assignment/anchor"),
        ("- commune_nis", "- municipality_nis", "assignment/derive"),
        (
            "commune: vintage:be_nis_2025",
            "commune: vintage:be_nuts1_2025",
            "assignment/layer_vintages",
        ),
        (
            "- geography_vintage_exact",
            "- geography_vintage_warning",
            "assignment/assertions/geography_vintage_exact",
        ),
        (
            "- vintage_refusal",
            "- vintage_warning",
            "assignment/validation/vintage_refusal",
        ),
        (
            "universe: commune_within_nuts1",
            "universe: commune_nationally",
            "assignment/draw/universe",
        ),
        (
            "- source_nuts1_preserved",
            "- source_nuts1_warning",
            "assignment/assertions/source_nuts1_preserved",
        ),
    ],
)
def test_be_typed_geography_drift_from_legacy_evidence_refuses(
    tmp_path: Path,
    before: str,
    after: str,
    field: str,
) -> None:
    package = tmp_path / "be"
    source = Path(__file__).parents[1] / "src/microcosm/build/be"
    shutil.copytree(source, package)
    geography = package / "spec/geography.yaml"
    authored = geography.read_text(encoding="utf-8")
    assert authored.count(before) == 1
    geography.write_text(authored.replace(before, after), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_country_spec(package)
    assert "typed geography compatibility drift" in str(error.value)
    assert field in str(error.value)


def test_be_smoke_build_refuses_without_gated_inputs_and_stage_bindings() -> None:
    """Compile succeeds, but the repository has no honest runnable BE stage map."""

    spec = load_country_spec("be")
    compiled = compile_spec(spec.resolved_spec)
    assert compiled.spec_binding.country == "be"

    silc_stage = spec.sources.stage_map()["silc_load"]
    assert {artifact["kind"] for artifact in silc_stage.artifacts} == {
        "restricted_microdata"
    }
    build_package = Path(__file__).parents[1] / "src/microcosm/build"
    assert not (build_package / "be_runtime").exists()

    with pytest.raises(
        ValueError,
        match=(
            r"missing \['silc_load', 'clone_assign_communes'\].*"
            r"There are no stubs or fallbacks"
        ),
    ):
        country_stage_plan(spec, {})


def test_am_smoke_build_refuses_without_harvests_and_stage_bindings() -> None:
    """Compile succeeds, but engine-free v1 has no runnable AM stage map."""

    spec = load_country_spec("am")
    compiled = compile_spec(spec.resolved_spec)
    assert compiled.spec_binding.country == "am"

    stages = spec.sources.stage_map()
    assert {
        artifact["kind"]
        for artifact in stages["load_populace_us_support_pool"].artifacts
    } == {"public_microdata"}
    assert {artifact["kind"] for artifact in stages["assign_am_marz"].artifacts} == {
        "public_aggregated_counts"
    }
    build_package = Path(__file__).parents[1] / "src/microcosm/build"
    assert not (build_package / "am_runtime").exists()

    with pytest.raises(
        ValueError,
        match=(
            r"missing \['load_populace_us_support_pool', 'assign_am_marz', "
            r"'clone_assign_communities'\].*There are no stubs or fallbacks"
        ),
    ):
        country_stage_plan(spec, {})


def test_nz_smoke_build_refuses_without_evidenced_inputs_and_stage_bindings() -> None:
    """A valid scaffold must not turn missing NZ runtime bindings into stubs."""

    spec = load_country_spec("nz")
    compiled = compile_spec(spec.resolved_spec)
    assert compiled.spec_binding.country == "nz"
    assert not compiled.producer_graph.present
    assert compiled.nodes == ()

    stages = spec.sources.stage_map()
    assert {
        artifact["kind"]
        for artifact in stages["load_populace_us_support_pool"].artifacts
    } == {"public_microdata"}
    assert (
        len(
            stages["prepare_nz_wff_axiom_inputs"].operations[0].parameters["predictors"]
        )
        == 21
    )
    build_package = Path(__file__).parents[1] / "src/microcosm/build"
    assert not (build_package / "nz_runtime").exists()

    with pytest.raises(
        ValueError,
        match=(
            r"missing \['load_populace_us_support_pool', 'prepare_nz_wff_axiom_inputs', "
            r"'assign_nz_regions'\].*There are no stubs or fallbacks"
        ),
    ):
        country_stage_plan(spec, {})
