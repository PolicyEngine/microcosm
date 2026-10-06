"""Orrery presentation over canonical compiler metadata."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from microcosm.graph import (
    compile_graph,
    graph_from_json,
    graph_schema,
    graph_to_json,
    orrery_document,
    orrery_json,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.orrery import (
    main,
    orrery_document_from_schema,
    orrery_json_from_schema,
)
from test_support.microcosm_graph.schema import (
    compiled_default_population_graph,
    compiled_graph,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-graph")
_ORRERY_VERIFY = _TEST_PATHS.repository / "tools" / "orrery-contract" / "verify.mjs"


def _parts(identity):
    return json.loads(identity)


def test_document_is_complete_static_metadata_with_no_runtime_claims():
    schema = graph_schema(compiled_graph())
    document = orrery_document_from_schema(schema)

    assert document["schemaVersion"] == "graph-explorer/v1"
    assert document["metadata"]["microcosm"] == schema
    assert document["metadata"]["scope"] == "complete_compiler_schema"
    assert document["metadata"]["truncated"] is False
    assert len([node for node in document["nodes"] if node["kind"] == "operation"]) == 6
    assert len([node for node in document["nodes"] if node["kind"] == "source"]) == 1
    assert (
        not {
            "activities",
            "receipts",
            "artifacts",
            "assessments",
        }
        & document.keys()
    )
    assert all("statuses" not in node for node in document["nodes"])
    assert {edge["category"] for edge in document["edges"]} == {
        "dependency",
        "provenance",
    }
    identities = {node["id"] for node in document["nodes"]}
    assert {edge["source"] for edge in document["edges"]} | {
        edge["target"] for edge in document["edges"]
    } <= identities


def test_descriptions_are_promoted_and_declarations_remain_available():
    document = orrery_document_from_schema(graph_schema(compiled_graph()))
    base = next(
        node
        for node in document["nodes"]
        if _parts(node["id"]) == ["operation", "base"]
    )
    rewrite = next(
        node
        for node in document["nodes"]
        if _parts(node["id"]) == ["operation", "rewrite"]
    )
    source = next(node for node in document["nodes"] if node["kind"] == "source")

    assert base["description"] == "Load the declared source tables"
    assert rewrite["description"] == "Replace age for selected people"
    assert rewrite["data"]["declaration"]["citation"] == "Example method"
    assert source["description"] == "Recorded survey tables"


def test_rewrite_incumbent_has_a_distinct_auxiliary_field():
    document = orrery_document_from_schema(graph_schema(compiled_graph()))
    ages = [
        node
        for node in document["nodes"]
        if node["kind"] == "field"
        and node["data"]["population"] == "filtered"
        and node["data"]["column"] == "age"
    ]
    assert len(ages) == 2
    incumbent = next(node for node in ages if node["data"]["provider"] == "filtered")
    final = next(node for node in ages if node["data"]["provider"] == "rewrite")
    assert incumbent["data"]["visible_in_schema"] is False
    assert final["data"]["visible_in_schema"] is True

    target = json.dumps(["operation", "rewrite"], separators=(",", ":"))
    reads = [
        edge
        for edge in document["edges"]
        if edge["target"] == target and edge["kind"] == "declared_read"
    ]
    assert {edge["data"]["read_kind"] for edge in reads} == {
        "slice",
        "slice_mask",
        "output_mask",
        "rewrite_incumbent",
    }
    rewrite_read = next(
        edge for edge in reads if edge["data"]["read_kind"] == "rewrite_incumbent"
    )
    assert rewrite_read["source"] == incumbent["id"]


def test_materialized_expand_output_has_a_distinct_input_field():
    document = orrery_document_from_schema(graph_schema(compiled_graph()))
    clones = [
        node
        for node in document["nodes"]
        if node["kind"] == "field"
        and node["data"]["population"] == "expanded"
        and node["data"]["column"] == "clone"
    ]
    assert len(clones) == 2
    materialized = next(
        node for node in clones if node["data"]["provider"] == "expanded"
    )
    claimed = next(
        node for node in clones if node["data"]["provider"] == "expanded.claim"
    )
    assert materialized["data"]["declared_in"] == "expanded.claim"
    assert materialized["data"]["visible_in_schema"] is False
    assert claimed["data"]["visible_in_schema"] is True

    target = json.dumps(["operation", "expanded.claim"], separators=(",", ":"))
    read = next(
        edge
        for edge in document["edges"]
        if edge["target"] == target
        and edge["kind"] == "declared_read"
        and edge["data"]["read_kind"] == "materialized_expand_output"
    )
    assert read["source"] == materialized["id"]


def test_dependencies_and_provenance_are_separate_categories():
    document = orrery_document_from_schema(graph_schema(compiled_graph()))
    dependency_kinds = {
        edge["kind"] for edge in document["edges"] if edge["category"] == "dependency"
    }
    provenance_kinds = {
        edge["kind"] for edge in document["edges"] if edge["category"] == "provenance"
    }
    assert dependency_kinds == {
        "compiled_predecessor",
        "declared_read",
        "artifact_input",
    }
    assert provenance_kinds == {
        "provided_field",
        "structural_input",
        "declared_source",
    }


def test_legal_omitted_population_uses_compiler_metadata():
    document = orrery_document_from_schema(
        graph_schema(compiled_default_population_graph())
    )
    consumer = next(
        node
        for node in document["nodes"]
        if _parts(node["id"]) == ["operation", "consumer"]
    )
    assert consumer["data"]["declaration"]["population"] is None
    assert consumer["data"]["population"] == "base"


def test_json_is_deterministic_detached_and_transports_large_numbers():
    schema = graph_schema(compiled_graph(), extensions={"large": 2**80, "value": 1e100})
    before = copy.deepcopy(schema)

    first = orrery_json_from_schema(schema)
    second = orrery_json_from_schema(schema)
    assert first == second and first.endswith("\n")
    document = json.loads(first)
    assert document["metadata"]["microcosm"]["extensions"] == {
        "large": {"integer_literal": str(2**80)},
        "value": {"float_literal": "1e+100"},
    }
    document["metadata"]["microcosm"]["graph"]["nodes"].clear()
    assert schema == before


def test_revisions_ignore_mapping_order_but_bind_descriptions():
    schema = graph_schema(compiled_graph())
    reordered = json.loads(
        json.dumps(schema, sort_keys=True),
        object_pairs_hook=lambda pairs: dict(reversed(pairs)),
    )
    assert orrery_json_from_schema(schema) == orrery_json_from_schema(reordered)

    edited_graph = copy.deepcopy(schema["graph"])
    edited_graph["nodes"][0]["description"] = "Changed description"
    edited = graph_schema(
        compile_graph(graph_from_json(canonical_json(edited_graph).decode()))
    )
    assert (
        orrery_document_from_schema(schema)["revision"]
        != orrery_document_from_schema(edited)["revision"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(extra=True),
        lambda value: value["compiled"]["order"].reverse(),
        lambda value: value["compiled"]["owners"].clear(),
        lambda value: value["fields"][0].update(provider="unknown"),
        lambda value: value["input_bindings"][0].update(kind="invented"),
    ],
)
def test_invalid_schema_fails_before_producing_a_partial_document(mutation):
    schema = graph_schema(compiled_graph())
    mutation(schema)
    with pytest.raises(ValueError):
        orrery_document_from_schema(schema)


def test_output_complexity_and_size_limits_are_enforced(monkeypatch):
    schema = graph_schema(compiled_graph())
    from microcosm.graph import orrery

    monkeypatch.setattr(orrery, "_MAX_NODES", 2)
    with pytest.raises(ValueError, match="node limit"):
        orrery_document_from_schema(schema)

    monkeypatch.setattr(orrery, "_MAX_NODES", 20_000)
    monkeypatch.setattr(orrery, "_MAX_OUTPUT_BYTES", len(canonical_json(schema)) + 64)
    with pytest.raises(ValueError, match="metadata size"):
        orrery_document_from_schema(schema)


def test_schema_embeds_the_exact_canonical_graph_declaration():
    compiled = compiled_graph()
    document = orrery_document_from_schema(graph_schema(compiled))
    embedded = document["metadata"]["microcosm"]["graph"]
    assert canonical_json(embedded).decode() == graph_to_json(compiled.graph)


def test_direct_graph_compiled_and_saved_schema_paths_are_identical():
    compiled = compiled_graph()
    schema = graph_schema(compiled, extensions={"producer": "test"})

    expected = orrery_json_from_schema(schema, title="Invented build")
    assert (
        orrery_json(
            compiled,
            title="Invented build",
            extensions={"producer": "test"},
        )
        == expected
    )
    assert (
        orrery_json(
            compiled.graph,
            title="Invented build",
            extensions={"producer": "test"},
        )
        == expected
    )
    assert orrery_document(compiled)["schemaVersion"] == "graph-explorer/v1"


def test_public_orrery_parser_accepts_generated_document():
    result = subprocess.run(
        ["node", str(_ORRERY_VERIFY)],
        input=orrery_json(
            compiled_graph(),
            extensions={"large": 2**80, "negative": -(2**80), "float": 1e100},
        ),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "Orrery accepted the Microcosm graph document.\n"


def test_cli_requires_an_explicit_input_type_and_matches_direct_output(tmp_path):
    compiled = compiled_graph()
    graph_path = tmp_path / "graph.json"
    schema_path = tmp_path / "schema.json"
    graph_output = tmp_path / "graph-output.json"
    schema_output = tmp_path / "schema-output.json"
    graph_path.write_text(graph_to_json(compiled.graph))
    schema_path.write_bytes(canonical_json(graph_schema(compiled)))

    assert (
        main(
            [
                "--graph",
                str(graph_path),
                "--output",
                str(graph_output),
                "--title",
                "Invented build",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--schema",
                str(schema_path),
                "--output",
                str(schema_output),
                "--title",
                "Invented build",
            ]
        )
        == 0
    )
    expected = orrery_json(compiled, title="Invented build")
    assert graph_output.read_text() == expected
    assert schema_output.read_text() == expected

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--graph",
                str(graph_path),
                "--schema",
                str(schema_path),
                "--output",
                str(tmp_path / "ambiguous.json"),
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("raw", ['{"country":1,"country":2}', '{"x":NaN}', "[]"])
def test_cli_refuses_invalid_json_without_output(tmp_path, raw):
    source = tmp_path / "input.json"
    output = tmp_path / "output.json"
    source.write_text(raw)
    with pytest.raises(SystemExit) as error:
        main(["--graph", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_cli_preserves_existing_output_and_input(tmp_path):
    compiled = compiled_graph()
    source = tmp_path / "graph.json"
    raw = graph_to_json(compiled.graph)
    source.write_text(raw)
    with pytest.raises(SystemExit) as error:
        main(["--graph", str(source), "--output", str(source)])
    assert error.value.code == 2
    assert source.read_text() == raw


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO")
def test_cli_refuses_fifo_without_waiting_for_a_writer(tmp_path):
    source = tmp_path / "graph.json"
    output = tmp_path / "output.json"
    os.mkfifo(source)
    with pytest.raises(SystemExit) as error:
        main(["--graph", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_module_cli_starts_without_an_import_order_warning():
    result = subprocess.run(
        [sys.executable, "-m", "microcosm.graph.orrery", "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--graph" in result.stdout
    assert "RuntimeWarning" not in result.stderr
