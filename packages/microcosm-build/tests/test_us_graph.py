"""US post-transfer graph declarations and registry contracts."""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime.graph import us_post_transfer_graph, us_registry
from microcosm.graph import (
    Graph,
    KernelRegistry,
    StructuralDelta,
    compile_graph,
    graph_from_json,
    graph_to_json,
)

pytestmark = pytest.mark.requires_us


_STAGES = (
    "prepare_stacked_tail_derivation",
    "derive_multispine_pool_inputs",
    "seed_multispine_pool_inputs",
    "materialize_multispine_agreement_outputs",
)


class _StubEngine:
    def default_values(self, _names):
        return {}

    def materialize(self, _frame, _variables, _period):
        return {}

    def variable_metadata(self, _name):
        raise AssertionError("the registry test does not execute the engine")

    def variables(self):
        return ()


def _coordinates(node) -> set[tuple[str, str]]:
    return {(owned.entity, owned.column) for owned in node.outputs}


def _input_coordinates(node) -> set[tuple[str, str]]:
    return {
        (slice_.entity, column) for slice_ in node.inputs for column in slice_.columns
    }


def test_us_post_transfer_graph_compiles_to_the_four_stage_chain() -> None:
    graph = us_post_transfer_graph()
    compiled = compile_graph(graph)

    assert isinstance(graph, Graph)
    assert compiled.order == (
        "create_stacked_pool",
        "prepare_stacked_tail_derivation.boundary",
        "prepare_stacked_tail_derivation",
        "derive_multispine_pool_inputs.boundary",
        "derive_multispine_pool_inputs",
        "seed_multispine_pool_inputs.boundary",
        "seed_multispine_pool_inputs",
        "materialize_multispine_agreement_outputs",
    )
    assert tuple(node_id for node_id in compiled.order if node_id in _STAGES) == _STAGES
    assert compiled.order[-1] == "materialize_multispine_agreement_outputs"
    reversed_declaration = Graph(
        graph.country,
        graph.sources,
        tuple(reversed(graph.nodes)),
    )
    assert compile_graph(reversed_declaration).order == compiled.order


def test_us_post_transfer_rewrites_are_opened_by_identity_filters() -> None:
    graph = us_post_transfer_graph()
    create = graph.node("create_stacked_pool")
    prepare = graph.node("prepare_stacked_tail_derivation")
    derive = graph.node("derive_multispine_pool_inputs")
    seed = graph.node("seed_multispine_pool_inputs")
    materialize = graph.node("materialize_multispine_agreement_outputs")

    assert create.structural is StructuralDelta.CREATE
    assert len(create.outputs) == 76
    assert all(not owned.rewrite for owned in create.outputs)

    for stage in _STAGES[:3]:
        boundary = graph.node(f"{stage}.boundary")
        assert boundary.structural is StructuralDelta.FILTER
        assert boundary.kernel == "us.post_transfer.identity@1"
        assert boundary.inputs

    schedule_d = ("person", "schedule_d_capital_gain_distributions")
    assert _coordinates(prepare) == {schedule_d}
    assert all(owned.rewrite for owned in prepare.outputs)
    assert len(derive.outputs) == 17
    assert all(owned.rewrite for owned in derive.outputs)

    seed_by_coordinate = {(owned.entity, owned.column): owned for owned in seed.outputs}
    assert len(seed_by_coordinate) == 17
    assert not seed_by_coordinate[("tax_unit", "takes_up_eitc")].rewrite
    assert all(
        owned.rewrite
        for coordinate, owned in seed_by_coordinate.items()
        if coordinate != ("tax_unit", "takes_up_eitc")
    )
    assert _coordinates(materialize) == {("person", "ssi")}
    assert not materialize.outputs[0].rewrite


def test_us_post_transfer_stages_declare_the_complete_live_surface() -> None:
    graph = us_post_transfer_graph()
    create_coordinates = _coordinates(graph.node("create_stacked_pool"))
    live = set(create_coordinates)

    for stage in _STAGES:
        node = graph.node(stage)
        inputs = _input_coordinates(node)
        outputs = _coordinates(node)
        incumbent_outputs = outputs & live
        assert inputs == live - incumbent_outputs
        assert {slice_.entity for slice_ in node.inputs} == {
            "person",
            "household",
            "tax_unit",
            "spm_unit",
            "family",
            "marital_unit",
        }
        assert node.sources == ("stacked",)
        live.update(outputs)


def test_us_post_transfer_graph_pins_resource_and_engine_contracts() -> None:
    graph = us_post_transfer_graph()
    derive = graph.node("derive_multispine_pool_inputs")
    seed = graph.node("seed_multispine_pool_inputs")
    materialize = graph.node("materialize_multispine_agreement_outputs")

    assert len(str(derive.params["remaining_stage_manifest_sha256"])) == 64
    assert len(str(derive.params["schedule_d_asset_sha256"])) == 64
    assert len(str(seed.params["take_up_resource_sha256"])) == 64
    assert seed.params["engine_ref"] == "policyengine-us"
    assert materialize.params["engine_ref"] == "policyengine-us"
    assert len(str(materialize.params["ssi_dependency_sha256"])) == 64
    assert len(str(materialize.params["engine_input_projection_sha256"])) == 64
    assert len(str(materialize.params["engine_input_defaults_sha256"])) == 64


def test_us_registry_covers_exact_graph_refs_and_hashes_real_modules() -> None:
    graph = us_post_transfer_graph()
    registry = us_registry(engine=_StubEngine())

    assert isinstance(registry, KernelRegistry)
    assert set(registry.refs()) == {node.kernel for node in graph.nodes}
    assert registry.implementation_hash(
        "us.post_transfer.prepare@1"
    ) != registry.implementation_hash("us.post_transfer.derive@1")
    assert all(len(registry.implementation_hash(ref)) == 64 for ref in registry.refs())


def test_us_post_transfer_graph_json_round_trip_is_canonical() -> None:
    graph = us_post_transfer_graph()
    serialized = graph_to_json(graph)

    assert graph_from_json(serialized) == graph
    assert graph_to_json(graph_from_json(serialized)) == serialized
