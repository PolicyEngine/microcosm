"""Canonical compiler metadata for portable graph presentation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from microcosm.graph import (
    GraphError,
    compile_graph,
    graph_to_json,
    save_graph_schema,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.schema import graph_schema, validate_graph_schema
from test_support.microcosm_graph.schema import (
    compiled_default_population_graph,
    compiled_graph,
)


def _field(schema, population, column):
    return next(
        field
        for field in schema["fields"]
        if field["population"] == population and field["column"] == column
    )


def test_schema_is_derived_from_the_compiled_graph():
    compiled = compiled_graph()
    schema = graph_schema(compiled)

    assert set(schema) == {
        "protocol",
        "country",
        "graph_sha256",
        "graph",
        "compiled",
        "fields",
        "input_bindings",
        "extensions",
    }
    assert schema["protocol"] == "microcosm.graph.schema.v1"
    assert schema["country"] == "invented"
    assert schema["compiled"] == {
        "order": list(compiled.order),
        "versions": dict(sorted(compiled.versions.items())),
        "predecessors": {
            node_id: list(compiled.predecessors[node_id])
            for node_id in sorted(compiled.predecessors)
        },
        "owners": [
            [*coordinate, owner]
            for coordinate, owner in sorted(compiled.owners.items())
        ],
    }
    assert validate_graph_schema(schema) == schema


def test_embedded_graph_keeps_descriptions_and_citations():
    schema = graph_schema(compiled_graph())
    operations = {operation["id"]: operation for operation in schema["graph"]["nodes"]}

    assert operations["base"]["description"] == "Load the declared source tables"
    assert operations["rewrite"]["description"] == "Replace age for selected people"
    assert operations["rewrite"]["citation"] == "Example method"
    assert [source["description"] for source in schema["graph"]["sources"]] == [
        "Recorded survey tables"
    ]


def test_saved_schema_is_canonical_and_binds_the_graph_bytes(tmp_path):
    compiled = compiled_graph()

    saved = save_graph_schema(compiled, tmp_path)

    assert saved.read_bytes() == canonical_json(graph_schema(compiled))
    assert (
        json.loads(saved.read_bytes())["graph_sha256"]
        == hashlib.sha256(graph_to_json(compiled.graph).encode()).hexdigest()
    )


def test_fields_record_carriage_rewrites_and_nearest_declaration():
    schema = graph_schema(compiled_graph())

    assert _field(schema, "base", "age") == {
        "population": "base",
        "entity": "person",
        "column": "age",
        "dtype": "int64",
        "provider": "base",
        "declared_in": "base",
        "rows": "all",
        "ownership": "produced",
        "rewrite": False,
    }
    assert _field(schema, "filtered", "age") == {
        "population": "filtered",
        "entity": "person",
        "column": "age",
        "dtype": "int64",
        "provider": "rewrite",
        "declared_in": "rewrite",
        "rows": "keep",
        "ownership": "produced",
        "rewrite": True,
    }
    assert _field(schema, "filtered", "keep")["provider"] == "filtered"
    assert _field(schema, "filtered", "keep")["declared_in"] == "base"
    assert _field(schema, "expanded", "age")["provider"] == "expanded"
    assert _field(schema, "expanded", "age")["declared_in"] == "rewrite"
    assert _field(schema, "expanded", "clone") == {
        "population": "expanded",
        "entity": "person",
        "column": "clone",
        "dtype": "int64",
        "provider": "expanded.claim",
        "declared_in": "expanded.claim",
        "rows": "all",
        "ownership": "produced",
        "rewrite": False,
    }
    assert _field(schema, "reweighted", "age")["provider"] == "reweighted"
    assert _field(schema, "reweighted", "clone")["declared_in"] == "expanded.claim"
    assert len(schema["fields"]) == 10


def test_input_bindings_preserve_each_declared_read_role():
    bindings = graph_schema(compiled_graph())["input_bindings"]
    rewrite = [binding for binding in bindings if binding["node"] == "rewrite"]

    assert rewrite == [
        {
            "node": "rewrite",
            "population": "filtered",
            "entity": "person",
            "column": "age",
            "provider": "filtered",
            "declared_in": "base",
            "kind": "slice",
            "rows": "keep",
        },
        {
            "node": "rewrite",
            "population": "filtered",
            "entity": "person",
            "column": "keep",
            "provider": "filtered",
            "declared_in": "base",
            "kind": "slice",
            "rows": "keep",
        },
        {
            "node": "rewrite",
            "population": "filtered",
            "entity": "person",
            "column": "keep",
            "provider": "filtered",
            "declared_in": "base",
            "kind": "slice_mask",
            "rows": "all",
        },
        {
            "node": "rewrite",
            "population": "filtered",
            "entity": "person",
            "column": "age",
            "provider": "filtered",
            "declared_in": "base",
            "kind": "rewrite_incumbent",
            "rows": "keep",
        },
        {
            "node": "rewrite",
            "population": "filtered",
            "entity": "person",
            "column": "keep",
            "provider": "filtered",
            "declared_in": "base",
            "kind": "output_mask",
            "rows": "all",
        },
    ]
    assert (
        next(binding for binding in bindings if binding["node"] == "expanded")[
            "provider"
        ]
        == "rewrite"
    )
    assert next(
        binding for binding in bindings if binding["node"] == "expanded.claim"
    ) == {
        "node": "expanded.claim",
        "population": "expanded",
        "entity": "person",
        "column": "clone",
        "provider": "expanded",
        "declared_in": "expanded.claim",
        "kind": "materialized_expand_output",
        "rows": "all",
    }


@pytest.mark.parametrize(
    ("expand_cells", "message"),
    [
        ((), "does not declare materialized output person.clone"),
        (
            (("person", "clone", "float64"),),
            "declares materialized output person.clone as 'float64'",
        ),
    ],
)
def test_materialized_expand_binding_matches_the_expand_declaration(
    expand_cells, message
):
    graph = compiled_graph().graph
    changed = replace(
        graph,
        nodes=tuple(
            replace(node, params={**node.params, "expand_cells": expand_cells})
            if node.id == "expanded"
            else node
            for node in graph.nodes
        ),
    )

    with pytest.raises(GraphError, match=message):
        graph_schema(compile_graph(changed))


def test_omitted_population_uses_the_compiler_resolved_version():
    compiled = compiled_default_population_graph()
    schema = graph_schema(compiled)

    assert schema["graph"]["nodes"][1]["population"] is None
    assert schema["compiled"]["versions"]["consumer"] == "base"
    assert schema["input_bindings"] == [
        {
            "node": "consumer",
            "population": "base",
            "entity": "person",
            "column": "age",
            "provider": "base",
            "declared_in": "base",
            "kind": "slice",
            "rows": "all",
        }
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value["compiled"]["order"].reverse(),
        lambda value: value["compiled"]["versions"].update(rewrite="base"),
        lambda value: value["compiled"]["predecessors"]["rewrite"].clear(),
        lambda value: value["compiled"]["owners"].clear(),
        lambda value: value["fields"][0].update(provider="rewrite"),
        lambda value: value["fields"][0].update(provider="unknown"),
        lambda value: value["input_bindings"][0].update(provider="rewrite"),
        lambda value: value["input_bindings"][0].update(kind="invented"),
    ],
)
def test_validation_recompiles_and_refuses_core_metadata_changes(mutation):
    schema = graph_schema(compiled_graph())
    mutation(schema)
    with pytest.raises(ValueError):
        validate_graph_schema(schema)


def test_validation_rejects_equal_values_with_different_json_types():
    schema = graph_schema(compiled_graph())
    schema["fields"][0]["rewrite"] = 0

    with pytest.raises(ValueError, match="core metadata"):
        validate_graph_schema(schema)


def test_extensions_are_detached_bounded_json():
    extension = {"producer": {"labels": ["one", 2**80]}}
    schema = graph_schema(compiled_graph(), extensions=extension)
    extension["producer"]["labels"].append("changed")
    assert schema["extensions"] == {"producer": {"labels": ["one", 2**80]}}

    copied = validate_graph_schema(schema)
    copied["extensions"]["producer"]["labels"].append("output change")
    assert schema["extensions"] == {"producer": {"labels": ["one", 2**80]}}

    invalid = copy.deepcopy(schema)
    invalid["extensions"] = {"bad": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        validate_graph_schema(invalid)


def test_serialized_operation_contracts_survive_the_schema_round_trip():
    compiled = compiled_graph()
    # Full UK gate declarations exceed 64 KiB; spine-only gates are smaller.
    contract = "x" * 70_000
    original = compiled.graph.nodes[0]
    graph = replace(
        compiled.graph,
        nodes=(
            replace(original, params={**original.params, "contract": contract}),
            *compiled.graph.nodes[1:],
        ),
    )
    schema = graph_schema(compile_graph(graph))
    assert validate_graph_schema(schema) == schema
    operation = next(
        operation
        for operation in schema["graph"]["nodes"]
        if operation["id"] == original.id
    )
    assert operation["params"]["contract"] == contract

    oversized = replace(
        graph,
        nodes=(
            replace(original, params={"contract": "x" * (128 * 1024 + 1)}),
            *graph.nodes[1:],
        ),
    )
    with pytest.raises(ValueError, match="length limit"):
        graph_schema(compile_graph(oversized))
