"""Structure-exact gates for the F0 generation-0 inventory."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import pytest

from microcosm.build.spec_engine.compiler_ir import CompiledSpecIR, compile_spec
from microcosm.build.spec_engine.inventory_coverage import (
    InventoryCoverageError,
    _bundle_home_matches,
    assert_inventory_coverage_complete,
    build_inventory_coverage,
)
from microcosm.build.spec_engine.legacy_adapter import compile_to_legacy_payload
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import ResolvedSpec, ResourceKind, freeze_json

EXPECTED_CHECKS = {
    "acs_group_predictors_exact",
    "acs_person_predictors_exact",
    "capital_gains_tail_contract_exact",
    "conditional_ownership_matrix_exact",
    "early_gap_fill_plan_exact",
    "early_transfer_surface_exact",
    "gap_fill_schedule_receipt_exact",
    "itemization_declared_splits_exact",
    "late_schedule_receipt_exact",
    "late_split_ledger_exact",
    "legacy_adapter_surfaces_exact",
    "post_clone_operator_order_exact",
    "primary_predictor_tuples_exact",
    "producer_dag_order_edges_waves_exact",
    "producer_inputs_exact",
    "producer_outputs_exact",
    "producer_receipt_transition_contract_exact",
    "producer_registry_exact",
    "producer_resource_semantics_exact",
    "producer_virtual_resources_exact",
    "qrf_model_parameters_explicit",
    "release_line_and_regex_exact",
    "release_rungs_exact",
    "seed_inventory_groups_exhaustive",
    "seed_owner_rows_exact",
    "seed_protocol_and_owner_map_digests_exact",
    "seed_protocol_header_streams_exact",
    "seed_site_definitions_exact",
    "seed_site_owner_bindings_exact",
    "source_stage_manifest_exact",
    "stacked_authority_components_exact",
    "stacked_authority_identity_exact",
    "stacked_checkpoint_base_identity_exact",
    "stacked_checkpoint_pool_code_exact",
    "stacked_checkpoint_top_level_exact",
    "stacked_geography_assignment_exact",
    "take_up_identity_exact",
    "take_up_legacy_contract_exact",
    "take_up_pipeline_steps_exact",
    "take_up_program_mechanisms_exact",
    "take_up_program_order_exact",
}

EXPECTED_COUNTS = {
    "adapter_surfaces": 13,
    "authority_components": 9,
    "early_families": 13,
    "early_targets": 48,
    "itemization_batches": 5,
    "itemization_targets": 37,
    "late_groups": 19,
    "late_targets": 70,
    "ownership_rows": 18,
    "primary_effective_predictor_tuples": 65,
    "primary_families": 1,
    "primary_targets": 65,
    "producer_authored_outputs": 92,
    "producer_compiled_outputs": 227,
    "producer_inputs": 2_744,
    "producer_nodes": 38,
    "producer_virtual_resources": 75,
    "release_rungs": 5,
    "resolved_references": 334,
    "seed_owner_bindings": 112,
    "seed_owner_rows": 54,
    "seed_sites": 53,
    "seed_streams": 14,
    "source_operators": 16,
    "source_stages": 37,
    "stacked_checkpoint_full_components": 13,
    "stacked_checkpoint_pool_code_components": 19,
    "stacked_checkpoint_static_components": 10,
    "tail_control_fields": 934,
    "take_up_pipeline_steps": 28,
    "take_up_programs": 17,
    "typed_artifacts": 84,
    "typed_columns": 176,
    "typed_entities": 8,
    "typed_scopes": 7,
}


@pytest.fixture(scope="module")
def resolved_us() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def compiled_us(resolved_us: ResolvedSpec) -> CompiledSpecIR:
    return compile_spec(resolved_us)


@pytest.fixture(scope="module")
def legacy_us(resolved_us: ResolvedSpec) -> dict[str, object]:
    return compile_to_legacy_payload(resolved_us)


def _mutate_domain(
    spec: ResolvedSpec,
    kind: ResourceKind,
    mutation: Callable[[dict[str, Any]], None],
) -> ResolvedSpec:
    resources = list(spec.resources)
    index = next(
        index
        for index, resource in enumerate(resources)
        if resource.descriptor.kind is kind
    )
    resource = resources[index]
    value = copy.deepcopy(resource.domain.to_wire())
    assert isinstance(value, dict)
    mutation(value)
    resources[index] = replace(
        resource,
        domain=replace(resource.domain, value=freeze_json(value)),
    )
    return replace(spec, resources=tuple(resources))


def _assert_named_failure(report: Mapping[str, object], name: str) -> None:
    missing = report["missing_items"]
    assert isinstance(missing, list)
    assert name in missing
    with pytest.raises(InventoryCoverageError, match=name):
        assert_inventory_coverage_complete(report)


def test_us_inventory_is_structure_exact_and_complete(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )

    assert_inventory_coverage_complete(report)
    assert report["required_item_count"] == 41
    assert report["covered_item_count"] == 41
    assert report["missing_item_count"] == 0
    assert report["missing_items"] == []
    assert set(report["items"]) == EXPECTED_CHECKS
    assert report["counts"] == EXPECTED_COUNTS


def test_full_checkpoint_vector_binds_dynamic_inputs_and_scale_controls(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    pytest.importorskip("policyengine_us", exc_type=ModuleNotFoundError)
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    item = report["items"]["stacked_checkpoint_base_identity_exact"]
    assert item["status"] == "covered"
    assert item["observed"]["input_roles"] == ["alpha", "zeta"]
    assert item["observed"]["fraction_token"] == "f025"
    assert item["observed"]["sha256"] == item["expected"]["sha256"]


def test_nonexistent_bundle_home_is_rejected(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    corrupted = copy.deepcopy(report)
    item = corrupted["items"]["early_gap_fill_plan_exact"]
    item["bundle_homes"][0] = "/does/not/exist"

    with pytest.raises(InventoryCoverageError, match="bundle-home match evidence"):
        assert_inventory_coverage_complete(corrupted)


def test_bundle_home_wildcards_require_at_least_one_real_match() -> None:
    domains = {
        "imputation": {
            "nodes": [
                {"inputs": ["person.age"]},
                {"outputs": ["person.income"]},
            ]
        }
    }
    assert _bundle_home_matches(domains, "/imputation/nodes/*/inputs") == (
        ["person.age"],
    )
    assert _bundle_home_matches(domains, "/imputation/nodes/*/missing") == ()


def _mark_inventory_item_missing(report: dict[str, Any]) -> None:
    name = "release_rungs_exact"
    item = report["items"][name]
    item["failures"] = ["fixture missing clause"]
    item["status"] = "missing"
    report["covered_item_count"] = report["required_item_count"] - 1
    report["missing_item_count"] = 1
    report["missing_items"] = [name]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda report: report.__setitem__("required_item_count", 0),
            "required_item_count differs",
        ),
        (
            lambda report: report.__setitem__("unexpected", True),
            "top-level fields differ",
        ),
        (
            lambda report: report["items"].pop("release_rungs_exact"),
            "item registry differs",
        ),
        (
            lambda report: report["counts"].__setitem__("producer_nodes", 0),
            "diagnostic counts differ",
        ),
        (
            _mark_inventory_item_missing,
            "inventory has missing required items",
        ),
    ],
)
def test_inventory_assertion_recomputes_tampered_summaries_and_items(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    corrupted = copy.deepcopy(report)
    mutation(corrupted)

    with pytest.raises(InventoryCoverageError, match=match):
        assert_inventory_coverage_complete(corrupted)


def test_operator_reorder_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    corrupted = copy.deepcopy(legacy_us)
    static = corrupted["stacked_checkpoint_static_components"]
    assert isinstance(static, dict)
    pool_code = static["pool_code"]
    assert isinstance(pool_code, dict)
    order = pool_code["post_clone_source_operator_order"]
    assert isinstance(order, list)
    order[0], order[1] = order[1], order[0]

    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=corrupted,
    )
    _assert_named_failure(report, "post_clone_operator_order_exact")


def test_ownership_cell_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    corrupted = copy.deepcopy(legacy_us)
    imputation = corrupted["imputation"]
    assert isinstance(imputation, dict)
    overlap = imputation["overlap_ownership"]
    assert isinstance(overlap, dict)
    ownership = overlap["ownership"]
    assert isinstance(ownership, list)
    first = ownership[0]
    assert isinstance(first, dict)
    first["final_owner"] = "corrupt:final_owner"

    report = build_inventory_coverage(
        resolved_us,
        compiled=compiled_us,
        legacy_payload=corrupted,
    )
    _assert_named_failure(report, "conditional_ownership_matrix_exact")


def test_take_up_step_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    def mutation(value: dict[str, Any]) -> None:
        tanf = next(row for row in value["programs"] if row["id"] == "tanf")
        tanf["pipeline"][0]["rate"]["value"] = 0.220

    corrupted = _mutate_domain(resolved_us, ResourceKind.TAKE_UP, mutation)
    report = build_inventory_coverage(
        corrupted,
        compiled=compiled_us,
        legacy_payload=legacy_us,
    )
    _assert_named_failure(report, "take_up_pipeline_steps_exact")


def test_seed_owner_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    first = compiled_us.seed_stream_map.sites[0]
    corrupted_first = replace(first, owners=(*first.owners, first.owners[0]))
    corrupted_seed_map = replace(
        compiled_us.seed_stream_map,
        sites=(corrupted_first, *compiled_us.seed_stream_map.sites[1:]),
    )
    corrupted_ir = replace(compiled_us, seed_stream_map=corrupted_seed_map)

    report = build_inventory_coverage(
        resolved_us,
        compiled=corrupted_ir,
        legacy_payload=legacy_us,
    )
    _assert_named_failure(report, "seed_site_owner_bindings_exact")


def test_seed_site_corruption_is_named(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
    legacy_us: dict[str, object],
) -> None:
    first = compiled_us.seed_stream_map.sites[0]
    corrupted_first = replace(first, stream="build_model")
    corrupted_seed_map = replace(
        compiled_us.seed_stream_map,
        sites=(corrupted_first, *compiled_us.seed_stream_map.sites[1:]),
    )
    corrupted_ir = replace(compiled_us, seed_stream_map=corrupted_seed_map)

    report = build_inventory_coverage(
        resolved_us,
        compiled=corrupted_ir,
        legacy_payload=legacy_us,
    )
    _assert_named_failure(report, "seed_site_definitions_exact")
