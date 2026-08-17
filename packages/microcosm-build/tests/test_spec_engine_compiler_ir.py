"""Exact, fail-closed gates for the F0 typed compiler IR."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    CompiledSpecIR,
    CompilerIRError,
    compile_spec,
)
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import (
    ResolvedSpec,
    ResourceKind,
    freeze_json,
)
from microcosm.build.spec_engine.resolver import F0_CONTRACT_ONLY_KERNEL_IDS

US_SCHEDULE_SHA256 = (
    "b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5"
)


@pytest.fixture(scope="module")
def resolved_us() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def compiled_us(resolved_us: ResolvedSpec) -> CompiledSpecIR:
    return compile_spec(resolved_us)


def _mutate_domain(
    spec: ResolvedSpec,
    kind: ResourceKind,
    mutation,
) -> ResolvedSpec:
    resources = list(spec.resources)
    index = next(
        index
        for index, resource in enumerate(resources)
        if resource.descriptor.kind is kind
    )
    resource = resources[index]
    value = copy.deepcopy(resource.domain.to_wire())
    mutation(value)
    resources[index] = replace(
        resource,
        domain=replace(resource.domain, value=freeze_json(value)),
    )
    return replace(spec, resources=tuple(resources))


def test_us_compiles_exact_stage_dag_and_lossless_producer_graph(
    compiled_us: CompiledSpecIR,
) -> None:
    graph = compiled_us.producer_graph
    assert graph.present
    assert len(graph.nodes) == 38
    assert len(graph.edges) == 71
    assert len(graph.waves) == 6
    assert len(graph.order) == 38
    assert len(set(graph.order)) == 38
    assert len(graph.ownership_matrix) == 18
    assert graph.compiled_output_count == 227
    assert graph.schedule_sha256 == US_SCHEDULE_SHA256
    assert sha256_json(graph.schedule_wire()) == US_SCHEDULE_SHA256

    authored = graph.authored
    assert authored is not None
    assert sum(len(node.source["outputs"]) for node in graph.nodes) == 92
    assert compiled_us.stage_dag.edges == graph.edges
    assert compiled_us.stage_dag.waves == graph.waves
    assert compiled_us.stage_dag.order == graph.order
    assert [node.id for node in compiled_us.stage_dag.nodes] == list(graph.order)
    assert graph.order[0] == "acs_pums_earnings_universe"
    assert graph.order[-1] == "transfer:person/source_operator_education_inputs"


def test_us_ownership_matrix_is_cell_closed(compiled_us: CompiledSpecIR) -> None:
    rows = compiled_us.producer_graph.ownership_matrix
    cell_keys = {
        (row["entity"], row["target"], row["origin"], row["clone_index"])
        for row in rows
    }
    assert len(cell_keys) == len(rows) == 18
    for row in rows:
        final = [
            action["producer"]
            for action in row["producer_actions"]
            if action["owns_final"]
        ]
        assert final == [row["final_owner"]]
    policy = compiled_us.producer_graph.incomparable_node_policy
    assert policy["incomparable_pair_count"] == 587
    assert policy["disjoint_write_pair_count"] == 587


def test_us_seed_stream_map_is_complete_and_owner_typed(
    compiled_us: CompiledSpecIR,
    resolved_us: ResolvedSpec,
) -> None:
    seed_map = compiled_us.seed_stream_map
    assert seed_map.protocol_id == "legacy-v1"
    assert len(seed_map.sites) == 53
    assert len(seed_map.owners) == 54
    assert sum(len(site.owners) for site in seed_map.sites) == 112
    assert {site.id for site in seed_map.sites} == {
        site.id for site in resolved_us.seed_protocol.sites
    }
    assert all(site.owners for site in seed_map.sites)
    assert {owner.kind for owner in seed_map.owners} == {
        "producer_node",
        "source_stage",
        "pipeline_operation",
    }
    for owner in seed_map.owners:
        assert owner.sites
        assert owner.streams


def test_us_node_slices_are_transitive_and_content_attested(
    compiled_us: CompiledSpecIR,
) -> None:
    assert len(compiled_us.nodes) == 38
    assert len({node.node_key for node in compiled_us.nodes}) == 38
    for node in compiled_us.nodes:
        assert sha256_json(node.node_slice_wire()) == node.node_slice_sha256
        transitive_ids = {row.id for row in node.transitive_nodes}
        assert set(node.depends_on) <= transitive_ids
        assert node.kernel_ref.startswith("kernel:")
        assert len(node.kernel_implementation_sha256) == 64


def test_ir_retains_all_typed_resolution_products(compiled_us: CompiledSpecIR) -> None:
    inventory = compiled_us.typed_inventory
    assert len(inventory["entities"]) == 8
    assert len(inventory["columns"]) == 173
    assert len(inventory["artifacts"]) == 84
    assert len(inventory["scopes"]) == 7
    assert len(inventory["references"]) == 318
    assert "engine_abi_lock" in compiled_us.generated_authorities
    assert "records" in compiled_us.vintage_authorities
    assert set(compiled_us.surfaces) == {
        "normative",
        "run_request",
        "execution_profile",
        "operational",
        "chain_state",
        "documentation",
    }
    assert set(compiled_us.resources_wire()) == {
        "battery",
        "bundle",
        "calibration",
        "catalogs",
        "geography",
        "imputation",
        "publication",
        "selection",
        "sources",
        "spine",
        "take_up",
        "vintages",
    }


def test_normative_node_mutation_changes_its_slice_and_descendant_keys(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
) -> None:
    leaf_id = compiled_us.producer_graph.order[-1]

    def mutation(value: dict[str, Any]) -> None:
        node = next(
            row
            for row in value["producer_graph"]["nodes"]
            if row["id"] == leaf_id
        )
        node["capabilities"]["retry_safety"] = "nonretryable"

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    changed = compile_spec(mutated)
    before = {node.id: node.node_key for node in compiled_us.nodes}
    after = {node.id: node.node_key for node in changed.nodes}
    changed_ids = {node_id for node_id in before if before[node_id] != after[node_id]}
    assert changed_ids == {leaf_id}


def test_dangling_compiler_dependency_refuses(resolved_us: ResolvedSpec) -> None:
    def mutation(value: dict[str, Any]) -> None:
        value["producer_graph"]["nodes"][0]["inputs"][0][
            "producing_stage"
        ] = "missing_producer"

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    with pytest.raises(CompilerIRError, match="dangling producer"):
        compile_spec(mutated)


def test_contract_only_kernel_cannot_back_a_producer(
    resolved_us: ResolvedSpec,
) -> None:
    contract_only_kernel = min(F0_CONTRACT_ONLY_KERNEL_IDS)

    def mutation(value: dict[str, Any]) -> None:
        value["producer_graph"]["nodes"][0]["kernel"] = (
            f"kernel:{contract_only_kernel}"
        )

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    with pytest.raises(
        CompilerIRError,
        match=r"contract-only F0 kernel has no producer implementation pin",
    ):
        compile_spec(mutated)


def test_ambiguous_ownership_refuses(resolved_us: ResolvedSpec) -> None:
    def mutation(value: dict[str, Any]) -> None:
        row = value["producer_graph"]["ownership_matrix"][0]
        row["producer_actions"][0]["owns_final"] = True

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    with pytest.raises(CompilerIRError, match="exactly one owns_final"):
        compile_spec(mutated)


def test_bundle_without_imputation_compiles_empty_graph(
    resolved_us: ResolvedSpec,
) -> None:
    minimal = replace(
        resolved_us,
        resources=tuple(
            resource
            for resource in resolved_us.resources
            if resource.descriptor.kind is not ResourceKind.IMPUTATION
        ),
        seed_site_bindings=(),
    )
    compiled = compile_spec(minimal)
    assert not compiled.producer_graph.present
    assert compiled.producer_graph.nodes == ()
    assert compiled.stage_dag.nodes == ()
    assert compiled.nodes == ()
    assert len(compiled.seed_stream_map.sites) == 53
    assert compiled.seed_stream_map.owners == ()
