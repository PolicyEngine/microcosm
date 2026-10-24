"""Structure-exact gates for the F0 generation-0 inventory."""

# ruff: noqa: F401

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import cache
from typing import Any

import pytest

from microcosm.build.spec_engine import inventory_coverage as inventory_coverage_module
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
from microcosm.build.us_runtime import worker_identity as worker_identity_module

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


@pytest.fixture(scope="module", autouse=True)
def _reuse_real_worker_binding_for_pin_checks() -> object:
    """Reuse one real operational binding while testing digests that strip it."""

    original = worker_identity_module.primary_qrf_worker_execution_binding

    @cache
    def cached(
        _fit_jobs: str | None,
        _predict_workers: str | None,
        _cpu_count: int | None,
    ) -> dict[str, object]:
        return original()

    def binding() -> dict[str, object]:
        return copy.deepcopy(
            cached(
                os.environ.get("POPULACE_FIT_N_JOBS"),
                os.environ.get("POPULACE_FIT_PREDICT_WORKERS"),
                os.cpu_count(),
            )
        )

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        worker_identity_module,
        "primary_qrf_worker_execution_binding",
        binding,
    )
    yield
    patcher.undo()


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
    "source_stages": 38,
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


def _mark_inventory_item_missing(report: dict[str, Any]) -> None:
    name = "release_rungs_exact"
    item = report["items"][name]
    item["failures"] = ["fixture missing clause"]
    item["status"] = "missing"
    report["covered_item_count"] = report["required_item_count"] - 1
    report["missing_item_count"] = 1
    report["missing_items"] = [name]


__all__ = [name for name in globals() if not name.startswith("__")]
