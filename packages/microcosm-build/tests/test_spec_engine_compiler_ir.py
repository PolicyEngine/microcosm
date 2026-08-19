"""Exact, fail-closed gates for the F0 typed compiler IR."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import pytest

from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import (
    COMPILER_IR_ABI_VERSION,
    EXECUTOR_CONTRACT_ABI,
    ROW_CLASSIFIER_IMPLEMENTATION_DOMAIN,
    CompiledSpecIR,
    CompilerIRError,
    compile_spec,
    row_classifier_contract,
)
from microcosm.build.spec_engine.executor import order_nodes
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import (
    FrozenMap,
    ResolvedSpec,
    ResourceKind,
    freeze_json,
    thaw_json,
)
from microcosm.build.spec_engine.resolver import F0_CONTRACT_ONLY_KERNEL_IDS

US_SCHEDULE_SHA256 = "b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5"


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


def test_operational_source_locators_do_not_enter_node_slices_or_keys(
    resolved_us: ResolvedSpec,
) -> None:
    original = compile_spec(resolved_us)

    def mutate(value: dict[str, object]) -> None:
        stages = value["stages"]
        assert isinstance(stages, list)
        first = stages[0]
        assert isinstance(first, dict)
        first["source"] = "operational://different-host-path"
        artifacts = first["artifacts"]
        assert isinstance(artifacts, list)
        first_artifact = artifacts[0]
        assert isinstance(first_artifact, dict)
        first_artifact["locator"] = "operational://different-locator"

    changed = compile_spec(_mutate_domain(resolved_us, ResourceKind.SOURCES, mutate))
    assert [node.node_key for node in changed.nodes] == [
        node.node_key for node in original.nodes
    ]
    assert [node.node_slice_sha256 for node in changed.nodes] == [
        node.node_slice_sha256 for node in original.nodes
    ]
    source_params = [
        thaw_json(param.value)
        for node in changed.nodes
        for param in node.resolved_params
        if param.path.startswith("/sources/stages/")
    ]
    assert source_params
    assert "operational://" not in str(source_params)


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
    assert len(seed_map.sites) == 72
    assert len(seed_map.owners) == 57
    assert sum(len(site.owners) for site in seed_map.sites) == 131
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


def test_us_nodes_have_exact_effective_seed_grants(
    compiled_us: CompiledSpecIR,
) -> None:
    seeded = [
        node
        for node in compiled_us.nodes
        if node.capabilities["determinism"] == "seeded"
    ]
    deterministic = [
        node
        for node in compiled_us.nodes
        if node.capabilities["determinism"] == "deterministic"
    ]
    assert len(seeded) == 34
    assert len(deterministic) == 4
    assert all(node.seed_sites and node.seed_streams for node in seeded)
    assert all(not node.seed_sites and not node.seed_streams for node in deterministic)

    for node in compiled_us.nodes:
        grant_param = next(
            param
            for param in node.resolved_params
            if param.path == f"/compiled/effective_seed_grant@producer={node.id}"
        )
        grant = thaw_json(grant_param.value)
        assert [site["id"] for site in grant["sites"]] == [
            site.id for site in node.seed_sites
        ]
        assert (
            tuple(dict.fromkeys(site.stream for site in node.seed_sites))
            == node.seed_streams
        )

    housing = next(
        node
        for node in compiled_us.nodes
        if node.kernel_ref == "kernel:impute_us_housing_assistance_to_puf_support"
    )
    housing_grant = thaw_json(
        next(
            param.value
            for param in housing.resolved_params
            if param.path.startswith("/compiled/effective_seed_grant@")
        )
    )
    assert {(row["kind"], row["id"]) for row in housing_grant["grant_sources"]} == {
        ("source_stage", "acs_rent")
    }
    assert "housing_assistance_puf_qrf_model" in {
        row["id"] for row in housing_grant["sites"]
    }
    assert any(
        param.path.startswith("/take_up/programs/") for param in housing.resolved_params
    )
    assert any(
        param.path.startswith("/sources/stages/") for param in housing.resolved_params
    )


def test_compiled_nodes_lift_immutable_executor_contracts_and_bind_them(
    compiled_us: CompiledSpecIR,
) -> None:
    assert COMPILER_IR_ABI_VERSION == 3
    assert EXECUTOR_CONTRACT_ABI == "compiled-node-brokered-contracts-v2"
    producer_by_id = {node.id: node for node in compiled_us.producer_graph.nodes}
    assert tuple(node.execution_rank for node in compiled_us.nodes) == tuple(
        range(len(compiled_us.nodes))
    )
    registry = compiled_us.producer_graph.scope_registry
    assert registry is not None

    for node in compiled_us.nodes:
        producer = producer_by_id[node.id]
        assert node.capabilities == producer.capabilities
        assert node.capabilities == producer.source["capabilities"]
        assert node.mutations == producer.mutations
        assert node.mutations == producer.source["mutations"]
        assert node.write_scopes == producer.write_scopes
        assert node.scope_registry is registry
        expected_classifier = row_classifier_contract(
            compiled_us.compiler_ir_abi, registry
        )
        assert (
            node.row_classifier_ref,
            node.row_classifier_implementation_sha256,
        ) == expected_classifier
        assert node.row_classifier_ref == (f"classifier:{registry['predicate_space']}")
        assert node.to_wire()["row_classifier"] == {
            "ref": node.row_classifier_ref,
            "implementation_sha256": (node.row_classifier_implementation_sha256),
        }
        assert thaw_json(node.compiler_ir_abi) == (
            compiled_us.compiler_ir_abi.to_wire()
        )
        assert (
            node.seed_protocol_sha256
            == compiled_us.seed_stream_map.implementation_sha256
        )
        assert isinstance(node.capabilities["effects"], tuple)
        assert isinstance(node.mutations["entity_keys"], FrozenMap)

        params = {param.path: param for param in node.resolved_params}
        depends_on_param = params[
            f"/compiled/producer_graph/nodes/{node.id}/depends_on"
        ]
        assert thaw_json(depends_on_param.value) == list(node.depends_on)
        inputs_param = params[f"/compiled/producer_graph/nodes/{node.id}/inputs"]
        assert thaw_json(inputs_param.value) == [
            thaw_json(input_row) for input_row in node.inputs
        ]
        outputs_param = params[f"/compiled/producer_graph/nodes/{node.id}/outputs"]
        assert thaw_json(outputs_param.value) == [
            thaw_json(output) for output in node.outputs
        ]
        kernel_param = params[f"/compiled/producer_graph/nodes/{node.id}/kernel"]
        assert thaw_json(kernel_param.value) == {
            "ref": node.kernel_ref,
            "implementation_sha256": node.kernel_implementation_sha256,
        }
        classifier_param = params[
            f"/compiled/producer_graph/nodes/{node.id}/row_classifier"
        ]
        assert thaw_json(classifier_param.value) == {
            "ref": node.row_classifier_ref,
            "implementation_sha256": (node.row_classifier_implementation_sha256),
        }
        scope_param = params["/imputation/producer_graph/scope_registry"]
        assert scope_param.value == registry
        rank_param = params[f"/compiled/producer_graph/nodes/{node.id}/execution_rank"]
        assert rank_param.value == node.execution_rank
        for param in node.resolved_params:
            assert sha256_json(thaw_json(param.value)) == param.value_sha256
        assert sha256_json(node.node_slice_wire()) == node.node_slice_sha256
        assert node.node_key == sha256_json(
            {
                "domain": "microcosm.spec-engine.static-node-key.v1",
                "compiler_ir_abi": thaw_json(node.compiler_ir_abi),
                "node_slice_sha256": node.node_slice_sha256,
                "kernel": {
                    "ref": node.kernel_ref,
                    "implementation_sha256": (node.kernel_implementation_sha256),
                },
                "seed_protocol_sha256": node.seed_protocol_sha256,
            }
        )

        wire_before = node.to_wire()
        slice_before = node.node_slice_wire()
        replacement_capabilities = freeze_json(
            {
                "determinism": "nondeterministic",
                "numeric_reproducibility": "unspecified",
                "effects": ["none"],
                "structural_delta": "none",
                "retry_safety": "nonretryable",
            }
        )
        replacement_mutations = freeze_json({"replacement": True})
        assert isinstance(replacement_capabilities, FrozenMap)
        assert isinstance(replacement_mutations, FrozenMap)
        changed = replace(
            node,
            capabilities=replacement_capabilities,
            mutations=replacement_mutations,
            write_scopes=(),
            scope_registry=replacement_mutations,
        )
        assert changed.to_wire() == wire_before
        assert changed.node_slice_wire() == slice_before
        assert changed.node_slice_sha256 == node.node_slice_sha256
        assert changed.node_key == node.node_key

        tamper_values = {
            "depends_on": (*node.depends_on, "tampered-edge"),
            "inputs": (*node.inputs, freeze_json({"tampered": "input"})),
            "outputs": (*node.outputs, freeze_json({"tampered": "output"})),
        }
        for field, value in tamper_values.items():
            tampered = replace(node, **{field: value})
            assert tampered.node_slice_wire() == slice_before
            assert tampered.node_slice_sha256 == node.node_slice_sha256
            assert tampered.node_key == node.node_key
            bound = params[f"/compiled/producer_graph/nodes/{node.id}/{field}"]
            assert thaw_json(bound.value) != list(value)


def test_row_classifier_digest_recipe_is_versioned_and_registry_exact(
    compiled_us: CompiledSpecIR,
) -> None:
    registry = compiled_us.producer_graph.scope_registry
    assert registry is not None
    expected = sha256_json(
        {
            "domain": ROW_CLASSIFIER_IMPLEMENTATION_DOMAIN,
            "compiler_ir_abi": compiled_us.compiler_ir_abi.to_wire(),
            "scope_registry": thaw_json(registry),
        }
    )
    assert {
        node.row_classifier_implementation_sha256 for node in compiled_us.nodes
    } == {expected}


def test_operational_broker_source_is_not_a_compiler_or_node_identity_input(
    compiled_us: CompiledSpecIR,
) -> None:
    assert all(
        module != "microcosm.build.spec_engine.brokers"
        for module, _digest in compiled_us.compiler_ir_abi.source_inventory
    )


def test_dependency_input_and_output_mutations_change_bound_node_keys(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
) -> None:
    edge_target_id = "source_finalizer"
    data_target_id = "primary_puf_qrf"

    def mutate_dependency(value: dict[str, Any]) -> None:
        node = next(
            row
            for row in value["producer_graph"]["nodes"]
            if row["id"] == edge_target_id
        )
        node["inputs"].pop()

    def mutate_input(value: dict[str, Any]) -> None:
        node = next(
            row
            for row in value["producer_graph"]["nodes"]
            if row["id"] == data_target_id
        )
        node["inputs"].reverse()

    def mutate_output(value: dict[str, Any]) -> None:
        node = next(
            row
            for row in value["producer_graph"]["nodes"]
            if row["id"] == data_target_id
        )
        output = node["outputs"][0]
        output["temporary"] = not output["temporary"]

    before = {node.id: node for node in compiled_us.nodes}
    mutations = (
        (mutate_dependency, edge_target_id, "depends_on"),
        (mutate_input, data_target_id, "inputs"),
        (mutate_output, data_target_id, "outputs"),
    )
    for mutation, target_id, field in mutations:
        changed = compile_spec(
            _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
        )
        after = {node.id: node for node in changed.nodes}
        assert getattr(after[target_id], field) != getattr(before[target_id], field)
        bound = next(
            param
            for param in after[target_id].resolved_params
            if param.path == f"/compiled/producer_graph/nodes/{target_id}/{field}"
        )
        direct_value = getattr(after[target_id], field)
        expected_bound_value = (
            list(direct_value)
            if field == "depends_on"
            else [thaw_json(row) for row in direct_value]
        )
        assert thaw_json(bound.value) == expected_bound_value
        assert after[target_id].node_slice_sha256 != (
            before[target_id].node_slice_sha256
        )
        assert after[target_id].node_key != before[target_id].node_key


def test_us_compiled_nodes_keep_the_executor_total_order(
    compiled_us: CompiledSpecIR,
) -> None:
    ordered = order_nodes(reversed(compiled_us.nodes))
    assert tuple(node.id for node in ordered) == compiled_us.producer_graph.order


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
            row for row in value["producer_graph"]["nodes"] if row["id"] == leaf_id
        )
        node["capabilities"]["retry_safety"] = "nonretryable"

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    changed = compile_spec(mutated)
    before = {node.id: node.node_key for node in compiled_us.nodes}
    after = {node.id: node.node_key for node in changed.nodes}
    changed_ids = {node_id for node_id in before if before[node_id] != after[node_id]}
    assert changed_ids == {leaf_id}


def test_scope_registry_change_invalidates_every_node_slice(
    resolved_us: ResolvedSpec,
    compiled_us: CompiledSpecIR,
) -> None:
    def mutation(value: dict[str, Any]) -> None:
        value["producer_graph"]["scope_registry"]["scopes"].append(
            {"id": "unused_fixture_scope", "atoms": ["receipt:virtual"]}
        )

    changed = compile_spec(
        _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    )
    assert all(
        before.node_slice_sha256 != after.node_slice_sha256
        and before.node_key != after.node_key
        for before, after in zip(compiled_us.nodes, changed.nodes, strict=True)
    )


def test_seeded_node_without_effective_grant_refuses(
    resolved_us: ResolvedSpec,
) -> None:
    def mutation(value: dict[str, Any]) -> None:
        node = next(
            row
            for row in value["producer_graph"]["nodes"]
            if row["id"] == "acs_pums_earnings_universe"
        )
        node["capabilities"]["determinism"] = "seeded"

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    with pytest.raises(CompilerIRError, match="no effective seed-site grant"):
        compile_spec(mutated)


def test_deterministic_node_with_effective_grant_refuses(
    resolved_us: ResolvedSpec,
) -> None:
    def mutation(value: dict[str, Any]) -> None:
        node = next(
            row
            for row in value["producer_graph"]["nodes"]
            if row["id"] == "primary_puf_qrf"
        )
        node["capabilities"]["determinism"] = "deterministic"

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    with pytest.raises(CompilerIRError, match="has an effective seed-site grant"):
        compile_spec(mutated)


def test_dangling_compiler_dependency_refuses(resolved_us: ResolvedSpec) -> None:
    def mutation(value: dict[str, Any]) -> None:
        value["producer_graph"]["nodes"][0]["inputs"][0]["producing_stage"] = (
            "missing_producer"
        )

    mutated = _mutate_domain(resolved_us, ResourceKind.IMPUTATION, mutation)
    with pytest.raises(CompilerIRError, match="dangling producer"):
        compile_spec(mutated)


def test_contract_only_kernel_cannot_back_a_producer(
    resolved_us: ResolvedSpec,
) -> None:
    contract_only_kernel = min(F0_CONTRACT_ONLY_KERNEL_IDS)

    def mutation(value: dict[str, Any]) -> None:
        value["producer_graph"]["nodes"][0]["kernel"] = f"kernel:{contract_only_kernel}"

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
    assert len(compiled.seed_stream_map.sites) == 72
    assert compiled.seed_stream_map.owners == ()
