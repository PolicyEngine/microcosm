"""Spine evidence survives cached execution and a completely fresh process."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.graph import (
    uk_registry,
    uk_spine_endpoint,
    uk_spine_graph,
    uk_spine_operation_inventory,
)
from microcosm.build.uk_runtime.graph_evidence import (
    add_uk_spine_gate_nodes,
    load_spine_stage_artifacts,
    spine_sidecar_evidence,
)
from microcosm.graph import ContentStore, compile_graph, load_source, run_graph


def test_spine_evidence_replays_without_instantiating_original_transform(tmp_path):
    country = load_country_spec("uk")
    country = replace(
        country, sources=replace(country.sources, stages=country.sources.stages[:1])
    )
    graph = uk_spine_graph(country)
    compiled = compile_graph(graph)
    source = (
        Path(__file__).resolve().parents[2]
        / "microcosm-graph/tests/fixtures/parity/uk_spine/sources"
    )

    class Root:
        sampling = {"fraction": 0.1, "seed": 7}
        fit_weight_records = (
            SimpleNamespace(fit_name="fixture-fit", weight_kind="design"),
        )

        def run_with_sources(self, frame, sources):
            return load_source("csv-tables", sources["frs"])

        def checkpoint_metadata(self):
            return {
                "evidence": {"rows": 8},
                "replay_payload": {"classification": "fixture"},
            }

    store = ContentStore(tmp_path / "store")
    cold = run_graph(
        compiled,
        sources={"frs": source},
        store=store,
        kernels=uk_registry({"frs_spine": Root()}, graph=graph),
        resume="forbid",
    )
    expected = load_spine_stage_artifacts(cold, store, stage_names=("frs_spine",))
    warm = run_graph(
        compiled,
        sources={"frs": source},
        store=store,
        kernels=uk_registry(graph=graph),
        resume="require",
    )
    assert all(receipt.store_hit for receipt in warm.nodes.values())
    assert (
        load_spine_stage_artifacts(warm, store, stage_names=("frs_spine",)) == expected
    )
    path = tmp_path / "manifest.json"
    warm.save(path)
    script = """
import json, sys
from microcosm.graph import ContentStore, RunManifest
from microcosm.build.uk_runtime.graph_evidence import load_spine_stage_artifacts
store = ContentStore(sys.argv[2])
manifest = RunManifest.load(sys.argv[1], store)
print(json.dumps(load_spine_stage_artifacts(manifest, store, stage_names=('frs_spine',)), sort_keys=True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(store.root)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == expected
    sidecar = spine_sidecar_evidence(expected)
    assert sidecar["sampling"] == {"fraction": 0.1, "seed": 7}
    assert sidecar["fit_weight_records"]["frs_spine"][0]["weight_kind"] == "design"


def test_spine_inventory_is_roster_derived_and_gates_bind_checkpoint_versions():
    country = load_country_spec("uk")
    spine = uk_spine_graph(country)
    endpoint = uk_spine_endpoint(spine)
    inventory = uk_spine_operation_inventory(spine, country)
    assert tuple(row["stage"] for row in inventory) == endpoint.stage_names
    assert inventory[0]["node"] == "create_uk_frs"
    assert "normalization" in inventory[0]["coupling"]
    wealth = next(row for row in inventory if row["stage"] == "was_wealth")
    assert "Donor and recipient" in wealth["coupling"]
    graph = add_uk_spine_gate_nodes(spine, spec=country, engine_identity="test-engine")
    compiled = compile_graph(graph)
    assembled = graph.node("spine.gates.assembled")
    transferred = graph.node("spine.gates.transferred")
    assert assembled.population == compiled.versions["frs_brma"]
    assert assembled.population != compiled.versions["was_wealth"]
    assert transferred.population == endpoint.population
    assert {edge.producer for edge in transferred.artifact_inputs} >= {
        "spine.gates.assembled",
        "was_wealth",
    }
    assert "spine.gates.assembled" in compiled.predecessors["frs_brma.checkpoint"]
    assert "spine.gates.assembled" in compiled.predecessors["was_wealth"]
    assert graph.node("was_wealth").params["spine_gate_phase"] == "assembled"


def _assembled_report(status):
    from microcosm.build.gate_battery import (
        GateOutcome,
        GatePhaseReport,
        GateStatus,
        gate_phase_report_payload,
    )
    from microcosm.build.gates import GateResult
    from microcosm.build.uk_runtime.graph_evidence import uk_spine_gate_manifest

    gates = uk_spine_gate_manifest(load_country_spec("uk"))
    entries = tuple(entry for entry in gates.gates if entry.phase == "assembled")
    selected = next(
        entry
        for entry in entries
        if entry.criticality == "release_blocking" and not entry.evidence_absent_blocks
    )
    outcomes = []
    for entry in entries:
        state = status if entry == selected else GateStatus.PASSED
        evaluated = state in (GateStatus.PASSED, GateStatus.FAILED)
        outcomes.append(
            GateOutcome(
                entry,
                state,
                result=GateResult(
                    entry.gate,
                    state is GateStatus.PASSED,
                    () if state is GateStatus.PASSED else ("fixture failure",),
                )
                if evaluated
                else None,
                reason=None if evaluated else "fixture missing reference",
            )
        )
    return gate_phase_report_payload(
        GatePhaseReport("assembled", tuple(outcomes)), gates=gates
    ), selected.id


def test_assembled_admission_refuses_before_model_and_preserves_development_policy(
    tmp_path,
):
    from microcosm.build.gate_battery import GateStatus
    from microcosm.build.uk_runtime.graph_evidence import (
        require_uk_spine_gate_admission,
    )
    from microcosm.build.uk_runtime.graph_kernels import UKStageKernel

    report, failed = _assembled_report(GateStatus.FAILED)
    path = tmp_path / "stored-phase.json"
    path.write_text(json.dumps(report))
    context = SimpleNamespace(
        artifacts={"spine_gate": SimpleNamespace(payload=path.read_bytes())},
        params={"spine_gate_phase": "assembled", "spine_gate_release_candidate": False},
    )

    class UntouchedModel:
        def __call__(self, frame):
            raise AssertionError("A blocked assembled spine must not run a donor model")

    with pytest.raises(ValueError, match=failed):
        UKStageKernel("was_wealth", UntouchedModel()).run(context)
    assert json.loads(path.read_bytes()) == report
    report, _ = _assembled_report(GateStatus.EVIDENCE_ABSENT)
    context.artifacts["spine_gate"].payload = json.dumps(report).encode()
    require_uk_spine_gate_admission(context)
    context.params["spine_gate_release_candidate"] = True
    with pytest.raises(ValueError, match="block downstream"):
        require_uk_spine_gate_admission(context)


def test_full_gate_replay_carries_spine_failure_and_checks_bound_policy():
    from microcosm.build.gate_battery import (
        GateOutcome,
        GatePhaseReport,
        GateStatus,
        gate_phase_report_payload,
    )
    from microcosm.build.gates import GateResult
    from microcosm.build.uk_runtime.full_gates import uk_full_gate_manifest
    from microcosm.build.uk_runtime.graph_terminal import (
        _full_gate_enforcement,
        decode_full_gate_report,
    )

    gates = uk_full_gate_manifest()
    report = GatePhaseReport(
        "preflight",
        tuple(
            GateOutcome(entry, GateStatus.PASSED, GateResult(entry.gate, True))
            for entry in gates.gates
            if entry.phase == "preflight"
        ),
    )
    previous, failed = _assembled_report(GateStatus.FAILED)
    document = {
        "schema_version": 1,
        "kind": "uk_full_gate_report",
        "selection_receipt": None,
        "sample_fraction": 0.01,
        "release_candidate": False,
        "report": gate_phase_report_payload(report, gates=gates),
        "upstream_phase_reports": {"spine_assembled": previous},
    }
    document["enforcement"] = _full_gate_enforcement(document, report)
    restored, enforcement = decode_full_gate_report(document)
    assert restored.phase == "preflight"
    assert enforcement["artifact_permitted"] is False
    assert failed in enforcement["structural_failures"]
    document["upstream_phase_reports"] = {}
    with pytest.raises(ValueError, match="enforcement"):
        decode_full_gate_report(document)
