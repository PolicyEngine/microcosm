"""Presentation contracts over invented declaration metadata; no population run."""

import copy
import hashlib
import json
import os

import pytest

from microcosm.graph import (
    Graph,
    Node,
    Owned,
    Ownership,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    explorer,
    graph_to_json,
)
from microcosm.graph.canonical import canonical_json


def test_cli_exports_exact_metadata_and_large_integer(tmp_path):
    schema = snapshot()
    schema["audit_integer"] = 2**60 + 1
    source = tmp_path / "schema.json"
    output = tmp_path / "report" / "graph.json"
    source.write_text(json.dumps(schema), encoding="utf-8")
    assert (
        explorer.main(
            ["--input", str(source), "--output", str(output), "--title", "Microcosm"]
        )
        == 0
    )
    assert output.read_text() == explorer.graph_explorer_json(schema, title="Microcosm")
    doc = json.loads(output.read_text())
    assert doc["metadata"]["microcosm"]["audit_integer"] == {
        "integer_literal": str(2**60 + 1)
    }


@pytest.mark.parametrize("raw", ['{"protocol":1,"protocol":2}', '{"x":NaN}', "[]"])
def test_cli_refuses_invalid_json_without_output(tmp_path, raw):
    source, output = tmp_path / "schema.json", tmp_path / "graph.json"
    source.write_text(raw)
    with pytest.raises(SystemExit) as error:
        explorer.main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_cli_preserves_existing_output_and_input(tmp_path):
    source = tmp_path / "schema.json"
    raw = json.dumps(snapshot())
    source.write_text(raw)
    with pytest.raises(SystemExit) as error:
        explorer.main(["--input", str(source), "--output", str(source)])
    assert error.value.code == 2
    assert source.read_text() == raw


def test_cli_refuses_oversized_input_before_decoding(tmp_path, monkeypatch):
    source, output = tmp_path / "schema.json", tmp_path / "graph.json"
    source.write_bytes(b"!" * 33)
    monkeypatch.setattr(explorer, "_MAX_INPUT_BYTES", 32)
    with pytest.raises(SystemExit) as error:
        explorer.main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO")
def test_cli_refuses_fifo_without_waiting_for_writer(tmp_path):
    source, output = tmp_path / "schema.json", tmp_path / "graph.json"
    os.mkfifo(source)
    with pytest.raises(SystemExit) as error:
        explorer.main(["--input", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def snapshot():
    graph = Graph(
        "invented",
        sources=(SourceRef("survey", "unused@1"), SourceRef("unused", "unused@1")),
        nodes=(
            Node(
                "base",
                "unused.create@1",
                structural=StructuralDelta.CREATE,
                sources=("survey",),
                outputs=(
                    Owned("person", "age", "int64"),
                    Owned("person", "mask", "boolean"),
                    Owned("person", "missing", "float64", ownership=Ownership.ABSENT),
                ),
            ),
            Node(
                "filtered",
                "unused.filter@1",
                structural=StructuralDelta.FILTER,
                base="base",
                inputs=(Slice("person", ("age",)),),
            ),
            Node(
                "rewrite",
                "unused.rewrite@1",
                population="filtered",
                inputs=(Slice("person", ("age", "mask"), rows="mask"),),
                outputs=(Owned("person", "age", "int64", rows="mask", rewrite=True),),
            ),
            Node(
                "consumer",
                "unused.consume@1",
                population="filtered",
                inputs=(Slice("person", ("age",)),),
            ),
        ),
    )
    compiled = compile_graph(graph)
    raw = graph_to_json(graph)
    fields = []
    for population in ("base", "filtered"):
        for owned in graph.nodes[0].outputs:
            rewrite = population == "filtered" and owned.column == "age"
            fields.append(
                {
                    "population": population,
                    "entity": owned.entity,
                    "column": owned.column,
                    "dtype": owned.dtype,
                    "producer": "rewrite" if rewrite else population,
                    "declared_in": "rewrite" if rewrite else "base",
                    "rows": "mask" if rewrite else "all",
                    "ownership": owned.ownership.value,
                    "rewrite": rewrite,
                }
            )

    def read(
        node,
        column,
        *,
        population="filtered",
        producer="filtered",
        declared_in="base",
        kind="slice",
        rows="all",
    ):
        return {
            "node": node,
            "entity": "person",
            "column": column,
            "population": population,
            "producer": producer,
            "declared_in": declared_in,
            "kind": kind,
            "rows": rows,
        }

    return {
        "protocol": "microcosm.graph.schema.v1",
        "country": "invented",
        "graph_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "graph": json.loads(raw),
        "compiled": {
            "order": list(compiled.order),
            "versions": dict(compiled.versions),
            "predecessors": {
                key: list(value) for key, value in compiled.predecessors.items()
            },
            "owners": [
                list(key) + [owner] for key, owner in sorted(compiled.owners.items())
            ],
        },
        "schema": fields,
        "input_bindings": [
            read("filtered", "age", population="base", producer="base"),
            read("rewrite", "age", rows="mask"),
            read("rewrite", "mask", rows="mask"),
            read("rewrite", "mask", kind="slice_mask"),
            read("rewrite", "mask", kind="output_mask"),
            read("rewrite", "age", kind="rewrite_incumbent", rows="mask"),
            read("consumer", "age", producer="rewrite", declared_in="rewrite"),
        ],
    }


def update_graph_digest(doc):
    doc["graph_sha256"] = hashlib.sha256(canonical_json(doc["graph"])).hexdigest()


def test_full_snapshot_and_no_runtime_claims():
    source = snapshot()
    doc = explorer.graph_explorer_document(source)
    assert doc["schemaVersion"] == "graph-explorer/v1"
    assert doc["metadata"]["microcosm"] == source
    assert doc["metadata"]["scope"] == "complete_supplied_schema"
    assert doc["metadata"]["truncated"] is False
    assert len([node for node in doc["nodes"] if node["kind"] == "operation"]) == 4
    assert len([node for node in doc["nodes"] if node["kind"] == "source"]) == 2
    assert len([node for node in doc["nodes"] if node["kind"] == "field"]) == 7
    assert not {"activities", "receipts", "artifacts", "assessments"} & doc.keys()
    assert all(
        "statuses" not in node and "parentId" not in node for node in doc["nodes"]
    )
    assert len(
        [edge for edge in doc["edges"] if edge["kind"] == "compiled_predecessor"]
    ) == sum(map(len, source["compiled"]["predecessors"].values()))
    assert {edge["source"] for edge in doc["edges"]} | {
        edge["target"] for edge in doc["edges"]
    } <= {node["id"] for node in doc["nodes"]}


def test_rewrite_incumbent_is_distinct_and_masks_retain_roles():
    doc = explorer.graph_explorer_document(snapshot())
    ages = [
        n
        for n in doc["nodes"]
        if n["kind"] == "field"
        and n["data"]["population"] == "filtered"
        and n["data"]["column"] == "age"
    ]
    assert len(ages) == 2 and ages[0]["id"] != ages[1]["id"]
    incumbent = next(n for n in ages if n["data"]["producer"] == "filtered")
    final = next(n for n in ages if n["data"]["producer"] == "rewrite")
    assert incumbent["data"]["visible_in_schema"] is False
    assert final["data"]["visible_in_schema"] is True
    reads = [e for e in doc["edges"] if e["kind"] == "declared_read"]
    assert {e["data"]["read_kind"] for e in reads} == {
        "slice",
        "slice_mask",
        "output_mask",
        "rewrite_incumbent",
    }
    incoming = [
        e
        for e in reads
        if json.loads(e["target"])[1] == "rewrite"
        and e["data"]["read_kind"] == "rewrite_incumbent"
    ]
    assert len(incoming) == 1 and incoming[0]["source"] == incumbent["id"]
    assert incoming[0]["data"]["rows"] == "mask"
    assert not any(
        e["source"] == final["id"] and json.loads(e["target"])[1] == "rewrite"
        for e in reads
    )
    structural = [e for e in doc["edges"] if e["kind"] == "structural_input"]
    assert len(structural) == 3
    assert all(json.loads(e["source"])[1] == "base" for e in structural)


def test_typed_artifact_edge_and_uninterpreted_metadata_survive():
    source = snapshot()
    base, _, _, consumer = source["graph"]["nodes"]
    type_ = {"name": "invented.summary", "schema_version": 2}
    base["artifact_outputs"] = [{"name": "summary", "type": type_}]
    consumer["artifact_inputs"] = [
        {
            "name": "local_alias",
            "producer": "base",
            "artifact": "summary",
            "type": type_,
        }
    ]
    source["compiled"]["predecessors"]["consumer"].append("base")
    source["graph"]["mass_partition"] = ["person", "period"]
    update_graph_digest(source)
    doc = explorer.graph_explorer_document(source)
    edge = next(e for e in doc["edges"] if e["kind"] == "artifact")
    assert edge["data"] == consumer["artifact_inputs"][0]
    assert edge["category"] == "dependency"
    assert doc["metadata"]["microcosm"]["graph"]["mass_partition"] == [
        "person",
        "period",
    ]


def test_deterministic_detached_json_and_exact_large_number_transport():
    source = snapshot()
    source["graph"]["nodes"][0]["params"] = {
        "large": 2**80 + 1,
        "negative": -(2**80 + 1),
        "safe": 2**53 - 1,
        "float": 1e100,
        "flag": True,
    }
    update_graph_digest(source)
    before = copy.deepcopy(source)
    text = explorer.graph_explorer_json(source)
    assert text == explorer.graph_explorer_json(source) and text.endswith("\n")
    doc = json.loads(text)
    params = doc["metadata"]["microcosm"]["graph"]["nodes"][0]["params"]
    assert params == {
        "large": {"integer_literal": str(2**80 + 1)},
        "negative": {"integer_literal": str(-(2**80 + 1))},
        "safe": 2**53 - 1,
        "float": {"float_literal": "1e+100"},
        "flag": True,
    }
    doc["metadata"]["microcosm"]["graph"]["nodes"].clear()
    assert source == before


def test_document_revision_binds_compiled_metadata_not_only_graph():
    source = snapshot()
    left = explorer.graph_explorer_document(source)
    source["compiled"]["extra_declaration"] = "caller supplied, not verified"
    right = explorer.graph_explorer_document(source)
    assert left["revision"] != right["revision"]
    assert (
        left["metadata"]["microcosm"]["graph_sha256"]
        == right["metadata"]["microcosm"]["graph_sha256"]
    )
    assert left["nodes"] == right["nodes"]


def test_named_mask_read_is_preserved_separately_from_compiler_predecessors():
    graph = Graph(
        "invented",
        sources=(SourceRef("survey", "unused@1"),),
        nodes=(
            Node(
                "base",
                "unused.create@1",
                structural=StructuralDelta.CREATE,
                sources=("survey",),
                outputs=(Owned("person", "age", "int64"),),
            ),
            Node(
                "a_mask",
                "unused.mask@1",
                population="base",
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "mask", "boolean"),),
            ),
            Node(
                "z_consumer",
                "unused.consume@1",
                population="base",
                inputs=(Slice("person", ("age", "mask"), rows="mask"),),
            ),
        ),
    )
    compiled = compile_graph(graph)
    source = snapshot()
    source["graph"] = json.loads(graph_to_json(graph))
    update_graph_digest(source)
    source["compiled"] = {
        "order": list(compiled.order),
        "versions": dict(compiled.versions),
        "predecessors": {
            key: list(value) for key, value in compiled.predecessors.items()
        },
        "owners": [list(key) + [value] for key, value in compiled.owners.items()],
    }
    source["schema"] = [
        {
            "population": "base",
            "entity": "person",
            "column": column,
            "dtype": dtype,
            "producer": owner,
            "declared_in": owner,
            "rows": "all",
            "ownership": "produced",
            "rewrite": False,
        }
        for column, dtype, owner in (
            ("age", "int64", "base"),
            ("mask", "boolean", "a_mask"),
        )
    ]
    source["input_bindings"] = [
        {
            "node": node,
            "population": "base",
            "entity": "person",
            "column": column,
            "producer": owner,
            "declared_in": owner,
            "kind": kind,
            "rows": rows,
        }
        for node, column, owner, kind, rows in (
            ("a_mask", "age", "base", "slice", "all"),
            ("z_consumer", "age", "base", "slice", "mask"),
            ("z_consumer", "mask", "a_mask", "slice", "mask"),
            ("z_consumer", "mask", "a_mask", "slice_mask", "all"),
        )
    ]
    doc = explorer.graph_explorer_document(source)
    target = json.dumps(["operation", "z_consumer"], separators=(",", ":"))
    incoming = [edge for edge in doc["edges"] if edge["target"] == target]
    compiled_parents = {
        json.loads(edge["source"])[1]
        for edge in incoming
        if edge["kind"] == "compiled_predecessor"
    }
    assert compiled_parents == set(compiled.predecessors["z_consumer"])
    mask = next(
        edge
        for edge in incoming
        if edge["kind"] == "declared_read" and edge["data"]["read_kind"] == "slice_mask"
    )
    assert json.loads(mask["source"])[4] == "a_mask"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda doc: doc.update(protocol="future"),
        lambda doc: doc.update(graph_sha256="0" * 64),
        lambda doc: doc["compiled"]["order"].reverse(),
        lambda doc: doc["schema"].append(copy.deepcopy(doc["schema"][0])),
        lambda doc: doc["schema"][0].update(dtype="invented"),
        lambda doc: doc["input_bindings"][0].update(producer="unknown"),
        lambda doc: doc["input_bindings"][0].update(kind="invented"),
        lambda doc: doc["input_bindings"].clear(),
        lambda doc: doc["input_bindings"].append(
            {
                **doc["input_bindings"][-1],
                "column": "mask",
                "producer": "filtered",
                "declared_in": "base",
            }
        ),
    ],
)
def test_invalid_references_fail_without_partial_document(mutation):
    source = snapshot()
    mutation(source)
    with pytest.raises(ValueError):
        explorer.graph_explorer_document(source)


def test_oversize_or_non_json_metadata_fails(monkeypatch):
    source = snapshot()
    source["extra"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        explorer.graph_explorer_document(source)
    source["extra"] = source
    with pytest.raises(ValueError, match="complexity"):
        explorer.graph_explorer_document(source)
    source = snapshot()
    monkeypatch.setattr(explorer, "_MAX_NODES", 3)
    with pytest.raises(ValueError, match="node limit"):
        explorer.graph_explorer_document(source)


def test_expanded_output_budget_is_checked_during_construction(monkeypatch):
    source = snapshot()
    budget = len(canonical_json(source)) + 65_536 + 2_000
    monkeypatch.setattr(explorer, "_MAX_OUTPUT_BYTES", budget)

    # The input fits, but presentation records must stop while being added.
    # A late-only final serialization check would visit the field loop first.
    def unexpected_field(_field):
        pytest.fail("expanded records reached field construction before refusal")

    monkeypatch.setattr(explorer, "_field_id", unexpected_field)
    with pytest.raises(ValueError, match="serialized metadata size"):
        explorer.graph_explorer_document(source)
