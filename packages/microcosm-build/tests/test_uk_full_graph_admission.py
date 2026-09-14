"""Real full-graph gates refuse synthetic evidence before any solver or export."""

import json
from dataclasses import replace

import pytest
from test_uk_full_population_graph import Source, graph_and_registry
from test_uk_full_target_graph import target_inputs as target_inputs
from test_uk_ladder_rowwise_clone import toy_ladder as toy_ladder

from microcosm.build.uk_runtime import graph_calibration
from microcosm.build.uk_runtime.full_build_cli import _through
from microcosm.build.uk_runtime.full_certification import (
    append_uk_full_certification_node,
    register_uk_full_certification_kernel,
)
from microcosm.build.uk_runtime.graph_build import (
    SPINE_PROVENANCE_TYPE,
    UKFullBuildConfig,
    register_uk_full_kernels,
    uk_full_graph,
)
from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
from microcosm.build.uk_runtime.graph_terminal import (
    FULL_GATE_REPORT_TYPE,
    add_uk_export_continuation,
    add_uk_export_preparation,
    append_uk_full_gate_nodes,
    decode_full_gate_report,
    register_uk_full_gate_kernels,
    register_uk_terminal_kernels,
)
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ContentStore,
    Graph,
    KernelRegistry,
    KernelResult,
    NodeRejectedError,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json


class IncompleteSource(Source):
    ref = "uk.test.full-incomplete-source@1"

    def run(self, context):
        return KernelResult(
            frame=super().run(context).frame,
            artifacts={
                "provenance": canonical_json(
                    {
                        "stages": ["synthetic_fixture"],
                        "stage_evidence": {},
                        "fit_weight_records": {},
                    }
                )
            },
        )


@pytest.mark.requires_uk
def test_real_full_graph_preflight_replays_and_blocks_dense_export_and_certification(
    target_inputs, toy_ladder, tmp_path, monkeypatch
):
    """Source adapters are synthetic; gate, solver and terminal owners are real.

    A tiny fixture lacks the approved national/local reference surface and
    canonical source roster. Its proper scientific outcome is refusal, never
    fabricated passing release evidence to drive a successful filesystem test.
    """
    _, ladder_path = toy_ladder
    primitive, _ = graph_and_registry(1)
    source = replace(
        primitive.node("source"),
        kernel=IncompleteSource.ref,
        artifact_outputs=(ArtifactOutput("provenance", SPINE_PROVENANCE_TYPE),),
    )
    base = Graph(
        "uk",
        tuple(item for item in primitive.sources if item.name == "fixture"),
        (source,),
    )
    full = uk_full_graph(
        UKFullBuildConfig(
            calibration_year=2026,
            time_period="2023",
            source_year=2023,
            seed=7,
            calibration=UKGraphCalibrationConfig(epochs=2, seed=7),
        ),
        spine=base,
        spine_population="source",
    )
    # Keep maintained K so the tiny fixture can reach every selected area;
    # zero-support targets still refuse before gates at deliberately smaller K.
    provenance = ArtifactInput(
        "spine_provenance", "source", "provenance", SPINE_PROVENANCE_TYPE
    )
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=("synthetic_fixture",),
        spine_provenance=provenance,
        engine_identity="fixture-engine",
        review_date="2026-09-10",
    )
    final_gate = ArtifactInput(
        "full_gates", "uk.full.gates.calibrated", "gate_report", FULL_GATE_REPORT_TYPE
    )
    graph = add_uk_export_preparation(
        graph,
        population=full.calibration.population,
        bindings={"target_scope": "all", "n_clones": full.config.n_clones},
        artifact_inputs=(final_gate,),
    )
    graph = add_uk_export_continuation(
        graph,
        population=full.calibration.population,
        manifest_binding={"kind": "fixture", "target_scope": "all"},
        artifact_inputs=(final_gate,),
    )
    graph = append_uk_full_certification_node(
        graph, population=full.calibration.population, spine_provenance=provenance
    )
    compiled = compile_graph(graph)
    assert "uk.full.package" in compiled.predecessors["uk.full.certification"]
    assert "uk.full.gates.calibrated" in compiled.predecessors["uk.full.export.prepare"]
    assert "uk.full.gates.preflight" in compiled.predecessors["uk.full.dense"]

    def registry():
        result = KernelRegistry()
        result.register(IncompleteSource())
        register_uk_full_kernels(result)
        register_uk_full_gate_kernels(
            result, coverage_engine=object(), engine_identity="fixture-engine"
        )
        register_uk_terminal_kernels(result)
        register_uk_full_certification_kernel(result)
        return result

    sources = {
        "fixture": ladder_path,
        "uk_ladder": ladder_path,
        "uk_ledger_facts": ladder_path,
    }
    store = ContentStore(tmp_path / "store")
    checkpoint = compile_graph(_through(graph, "uk.full.gates.preflight"))
    cold = run_graph(checkpoint, sources=sources, store=store, kernels=registry())
    key = cold.nodes["uk.full.gates.preflight"].opaque_artifacts["gate_report"]
    payload = store.load_bytes(key)
    report, enforcement = decode_full_gate_report(payload)
    assert report.phase == "preflight"
    assert not enforcement["artifact_permitted"]
    assert "uk_release_family_build_stages" in enforcement["structural_failures"]
    selection = json.loads(payload)["selection_receipt"]
    assert selection["selector"]["geography_levels"] is None
    assert {row["geography_level"] for row in selection["included"]} >= {
        "country",
        "constituency",
        "la",
    }
    warm = run_graph(
        checkpoint,
        sources=sources,
        store=store,
        kernels=registry(),
        resume="require",
    )
    assert all(receipt.hit for receipt in warm.nodes.values())
    assert warm.nodes["uk.full.gates.preflight"].opaque_artifacts["gate_report"] == key

    monkeypatch.setattr(
        graph_calibration,
        "calibrate",
        lambda *args, **kwargs: pytest.fail(
            "A failed source preflight reached the solver"
        ),
    )
    with pytest.raises(
        NodeRejectedError, match="Dense calibration refused by the source preflight"
    ):
        run_graph(
            compile_graph(_through(graph, "uk.full.dense")),
            sources=sources,
            store=store,
            kernels=registry(),
        )
    # The authenticated failure survives the rejected continuation. No later
    # gate, export, or certification result has been created by this attempt.
    assert store.load_bytes(key) == payload
    assert not any(
        json.loads(path.read_text()).get("node_id")
        in {"uk.full.export.prepare", "uk.full.package", "uk.full.certification"}
        for path in (store.root / "objects").glob("*/*/payload.json")
    )
