"""UK supplies grouping and pinned checkpoint references to shared contracts."""

import json

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig, uk_full_graph
from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
from microcosm.build.uk_runtime.graph_evidence import add_uk_spine_gate_nodes
from microcosm.build.uk_runtime.graph_national import (
    NATIONAL_GATES_NODE,
    NATIONAL_PROBLEM_NODE,
    NATIONAL_READBACK_NODE,
    NATIONAL_TARGETS_NODE,
)
from microcosm.build.uk_runtime.graph_terminal import append_uk_full_gate_nodes
from microcosm.build.uk_runtime.orrery_contract import (
    checkpoint_references,
    save_uk_graph_schema,
    uk_graph_presentation,
    uk_operation_group,
    uk_spine_stage_ids,
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
    assert json.loads(document["id"]) == ["microcosm", "uk", "full_dense"]
    assert document["title"] == "UK · full dense"
    # Every maintained operation is explicitly rostered; nothing falls through.
    groups = {group["id"]: group for group in contract["groups"]}
    assert "other" not in groups
    assert set(groups) == {
        "sources",
        "enrichment",
        "population",
        "geography",
        "targets",
        "calibration",
        "checks",
    }
    assert all(group["description"] for group in groups.values())
    assert set(groups["enrichment"]["operations"]) == {
        node.id for node in spine.nodes if not node.id.startswith("spine.gates.")
    }
    assert {"uk.full.pool", "uk.full.expand", "uk.full.expand.owned"} <= set(
        groups["population"]["operations"]
    )
    assert set(groups["checks"]["operations"]) >= {
        "spine.gates.assembled",
        "uk.full.gates.preflight",
        "uk.full.gates.calibrated",
        "uk.full.holdout",
    }
    if households is not None:
        assert {"uk.full.size_search", "uk.full.selected"} <= set(
            groups["calibration"]["operations"]
        )
    fields = [node for node in document["nodes"] if node["kind"] == "field"]
    assert all(json.loads(node["parentId"])[0] == "operation" for node in fields)
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


def test_national_and_export_operations_are_rostered():
    spine = uk_spine_stage_ids()
    assert uk_operation_group("create_uk_frs", spine_stages=spine) == "enrichment"
    assert uk_operation_group("was_wealth", spine_stages=spine) == "enrichment"
    assert uk_operation_group("frs_brma.checkpoint", spine_stages=spine) == "enrichment"
    assert uk_operation_group(NATIONAL_TARGETS_NODE, spine_stages=spine) == "targets"
    assert uk_operation_group(NATIONAL_PROBLEM_NODE, spine_stages=spine) == "targets"
    assert uk_operation_group(NATIONAL_GATES_NODE, spine_stages=spine) == "checks"
    assert uk_operation_group(NATIONAL_READBACK_NODE, spine_stages=spine) == "export"
    for node_id in (
        "uk.full.export.prepare",
        "uk.full.export.readback",
        "uk.full.package",
    ):
        assert uk_operation_group(node_id, spine_stages=spine) == "export"
    assert uk_operation_group("uk.full.spine_checkpoint", spine_stages=spine) == (
        "population"
    )
    assert uk_operation_group("uk.full.invented", spine_stages=spine) == "other"
    contract = uk_graph_presentation(checkpoint_graph(), scope="national")
    assert [group["id"] for group in contract["groups"]] == ["sources", "population"]


def test_unavailable_checkpoint_bytes_are_named_not_linked(tmp_path):
    sidecar = tmp_path / "spine.build.json"
    graph = tmp_path / "spine.graph-declaration.json"
    graph.write_bytes(b'{"original":true}')
    evidence = tmp_path / "execution.evidence.json"
    evidence.write_bytes(b'{"protocol":"x","runs":[]}')
    sidecar.write_text(
        json.dumps(
            {
                "graph_declaration": {
                    "path": str(graph),
                    "sha256": sha256(graph.read_bytes()),
                },
                "graph_manifest": {
                    "path": str(tmp_path / "gone.json"),
                    "sha256": "a" * 64,
                },
                "graph_execution_evidence": {
                    "path": str(evidence),
                    "sha256": sha256(evidence.read_bytes()),
                },
            }
        )
    )
    contract = uk_graph_presentation(
        checkpoint_graph(), scope="checkpoint_dense", sidecar_path=sidecar
    )
    boundary = contract["scope"]["boundaries"][0]
    assert [ref["label"] for ref in boundary["upstream"]] == [
        "Producing graph declaration",
        "Producing execution evidence index",
    ]
    assert "bytes unavailable" in boundary["missing"]
    assert "a" * 64 in boundary["missing"]
    assert not any("gone.json" in json.dumps(ref) for ref in boundary["upstream"])
    saved = save_uk_graph_schema(
        checkpoint_graph(), tmp_path / "out", scope="national", sidecar_path=sidecar
    )
    document = orrery_document_from_schema(json.loads(saved.read_bytes()))
    node = next(node for node in document["nodes"] if node["kind"] == "operation")
    assert node["statuses"][0]["label"] == "Upstream evidence missing"
    assert all(ref["url"].startswith("upstream-") for ref in node["sources"])
    assert "gone.json" not in saved.read_text()
