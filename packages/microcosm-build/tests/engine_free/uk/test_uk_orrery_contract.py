"""UK supplies grouping and pinned checkpoint references to shared contracts."""

import json

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
from microcosm.build.uk_runtime.graph_evidence import add_uk_spine_gate_nodes
from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
from microcosm.build.uk_runtime.orrery_contract import (
    checkpoint_references,
    save_uk_graph_schema,
    uk_graph_presentation,
)
from microcosm.graph import (
    Graph,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    orrery_document,
)
from microcosm.graph.evidence import sha256
from microcosm.graph.orrery import orrery_document_from_schema
from microcosm.graph.presentation import PRESENTATION_EXTENSION


def checkpoint_graph():
    return Graph(
        "uk",
        (SourceRef("uk_spine", "raw-bytes-v1"),),
        (
            Node(
                "uk.full.spine_checkpoint",
                "fixture@1",
                sources=("uk_spine",),
                structural=StructuralDelta.CREATE,
                outputs=(Owned("person", "age", "int64"),),
            ),
        ),
    )


def test_spine_groups_keep_composite_execution_boundaries():
    graph = uk_spine_graph(source_mode="split")
    contract = uk_graph_presentation(graph, scope="raw_spine")
    document = orrery_document(graph, extensions={PRESENTATION_EXTENSION: contract})
    was = next(node for node in document["nodes"] if node["label"] == "was_wealth")
    assert was["data"]["composite"]["execution_unit"] == "composite"
    assert "coupled" in was["data"]["composite"]["coupling"]
    assert sum(node["kind"] == "operation" for node in document["nodes"]) == len(
        graph.nodes
    )
    assert not any(
        node["kind"] == "operation" and ".fit" in node["label"]
        for node in document["nodes"]
    )


@pytest.mark.parametrize("households", [None, 100])
def test_complete_dense_and_size_declarations_fit_existing_export_limits(households):
    spine = add_uk_spine_gate_nodes(
        uk_spine_graph(source_mode="split"),
        spec=load_country_spec("uk"),
        engine_identity="declaration-only",
    )
    full = uk_full_graph(
        UKFullBuildConfig(
            calibration_year=2025,
            geography_assignment="legacy",
            calibration=UKGraphCalibrationConfig(dataset_households=households),
        ),
        spine=spine,
    )
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=uk_spine_endpoint(spine).stage_names,
        engine_identity="declaration-only",
        review_date="2026-10-08",
    )
    contract = uk_graph_presentation(graph, scope="full_dense")
    document = orrery_document(graph, extensions={PRESENTATION_EXTENSION: contract})
    assert sum(node["kind"] == "operation" for node in document["nodes"]) == len(
        graph.nodes
    )
    assert len(document["nodes"]) < 20_000
    assert len(document["edges"]) < 100_000
    assert document["metadata"]["presentation_scope"]["id"] == "full_dense"
    for node_id in ("uk.full.gates.preflight", "uk.full.gates.calibrated"):
        operation = next(node for node in document["nodes"] if node["label"] == node_id)
        assert (
            operation["data"]["declaration"]["params"]["gate_manifest"]
            == (graph.node(node_id).params["gate_manifest"])
        )


def test_old_checkpoint_has_an_explicit_missing_boundary(tmp_path):
    sidecar = tmp_path / "old.build.json"
    sidecar.write_text("{}")
    contract = uk_graph_presentation(
        checkpoint_graph(), scope="checkpoint_dense", sidecar_path=sidecar
    )
    boundary = contract["scope"]["boundaries"][0]
    assert boundary["upstream"] == []
    assert boundary["missing"]
    document = orrery_document(
        checkpoint_graph(), extensions={PRESENTATION_EXTENSION: contract}
    )
    node = next(node for node in document["nodes"] if node["kind"] == "operation")
    assert node["statuses"][0]["label"] == "Upstream evidence missing"


def test_checkpoint_links_use_recorded_bytes_and_survive_output_relocation(tmp_path):
    sidecar = tmp_path / "spine.build.json"
    graph = tmp_path / "original.graph.json"
    manifest = tmp_path / "original.manifest.json"
    graph.write_bytes(b'{"original":true}')
    manifest.write_bytes(b'{"attempt":"recorded"}')
    sidecar.write_text(
        json.dumps(
            {
                "graph_declaration": {
                    "path": str(graph),
                    "sha256": sha256(graph.read_bytes()),
                },
                "graph_manifest": {
                    "path": str(manifest),
                    "sha256": sha256(manifest.read_bytes()),
                },
            }
        )
    )
    saved = save_uk_graph_schema(
        checkpoint_graph(), tmp_path / "out", scope="national", sidecar_path=sidecar
    )
    relocated = tmp_path / "relocated"
    saved.parent.rename(relocated)
    saved = relocated / saved.name
    document = orrery_document_from_schema(json.loads(saved.read_bytes()))
    operation = next(node for node in document["nodes"] if node["kind"] == "operation")
    assert len(operation["sources"]) == 2
    for ref in operation["sources"]:
        assert sha256((saved.parent / ref["url"]).read_bytes()) == ref["sha256"]
        assert ref["url"].startswith("upstream-")
    from microcosm.graph.orrery import main

    output = tmp_path / "preview" / "graph.orrery.json"
    assert main(["--schema", str(saved), "--output", str(output)]) == 0
    exported = json.loads(output.read_bytes())
    relocated_node = next(
        node for node in exported["nodes"] if node["kind"] == "operation"
    )
    for ref in relocated_node["sources"]:
        assert sha256((output.parent / ref["url"]).read_bytes()) == ref["sha256"]
    graph.write_bytes(b'{"original":false}')
    with pytest.raises(ValueError, match="digest mismatch"):
        checkpoint_references(sidecar)
