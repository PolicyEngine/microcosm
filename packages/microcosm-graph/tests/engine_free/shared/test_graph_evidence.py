"""Country-neutral binding, phase history and existing Orrery surfaces."""

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from microcosm.graph import (
    collect_execution_evidence,
    graph_schema,
    load_run_evidence,
    orrery_document,
    orrery_json,
    record_run_binding,
    save_run_evidence,
)
from microcosm.graph.orrery import main
from microcosm.graph.presentation import PRESENTATION_EXTENSION, PRESENTATION_PROTOCOL
from microcosm.graph.store import StoreCorrupt, StoreMiss
from test_support.paths import paths_for

_PATHS = paths_for("microcosm-graph")
if "_toy" not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location("_toy", _PATHS.tests / "_toy.py")
    _TOY = importlib.util.module_from_spec(_SPEC)
    sys.modules["_toy"] = _TOY
    _SPEC.loader.exec_module(_TOY)
toy = sys.modules["_toy"]


def presentation(compiled):
    return {
        "protocol": PRESENTATION_PROTOCOL,
        "scope": {"id": "fixture", "label": "Synthetic country"},
        "groups": [
            {"id": "sources", "label": "Sources", "sources": ["survey"]},
            {"id": "work", "label": "Processing", "operations": list(compiled.order)},
        ],
        "operations": {"survey": {"composite": {"execution_unit": "single"}}},
        "sources": {
            "survey": {
                "references": [
                    {"label": "Source definition", "url": "https://example.org/survey"}
                ]
            }
        },
    }


@pytest.fixture
def fresh(tmp_path):
    return toy.run_toy(toy.full_graph(), tmp_path)


def test_fresh_and_cache_phases_keep_same_ids_and_independent_states(fresh, tmp_path):
    cached = toy.run_toy(
        fresh.compiled.graph,
        tmp_path,
        sources=fresh.sources,
        registry=fresh.registry,
        store=fresh.store,
    )
    assert fresh.manifest.key == cached.manifest.key
    first = record_run_binding(
        fresh.compiled, fresh.manifest, attempt_id="attempt", phase="numerical"
    )
    second = record_run_binding(
        cached.compiled, cached.manifest, attempt_id="attempt", phase="final"
    )
    assert first.binding["manifest_sha256"] != second.binding["manifest_sha256"]
    assert first.binding["source_identities"]
    schema = graph_schema(
        fresh.compiled,
        extensions={PRESENTATION_EXTENSION: presentation(fresh.compiled)},
    )
    evidence = collect_execution_evidence(
        schema, runs=[first, second], store=fresh.store
    )
    before = toy.total_calls(fresh.registry)
    document = orrery_document(
        fresh.compiled, extensions=schema["extensions"], execution=evidence
    )
    assert toy.total_calls(fresh.registry) == before
    static = orrery_document(fresh.compiled)
    assert {node["id"] for node in static["nodes"]} <= {
        node["id"] for node in document["nodes"]
    }
    assert static["edges"] == document["edges"]
    operation = next(node for node in document["nodes"] if node["label"] == "survey")
    assert operation["parentId"] == '["group","work"]'
    history = operation["data"]["execution_history"]
    assert [record["cache"] for record in history] == ["miss", "hit"]
    assert {record["execution"] for record in history} == {"completed"}
    assert len(document["activities"]) == 2 * len(fresh.compiled.order)
    assert document["artifacts"]
    assert "receipts" not in document
    assert document == orrery_document(
        fresh.compiled, extensions=schema["extensions"], execution=evidence
    )
    assert all(node["revision"] for node in document["nodes"])


def test_gate_failure_is_separate_from_successful_execution(tmp_path):
    run = toy.run_toy(toy.full_graph(gate_low=1e20, gate_high=1e21), tmp_path)
    binding = record_run_binding(
        run.compiled, run.manifest, attempt_id="failed", phase="gates"
    )
    evidence = collect_execution_evidence(
        graph_schema(run.compiled), runs=[binding], store=run.store
    )
    gate = evidence["phases"][0]["operations"]["gate_tax"]
    assert gate["gate"] == "fail"
    assert gate["execution"] == "completed"


def test_changed_implementation_reuses_only_unaffected_nodes(fresh, tmp_path):
    changed = toy.run_toy(
        fresh.compiled.graph,
        tmp_path,
        sources=fresh.sources,
        store=fresh.store,
        registry=toy.toy_registry(
            variants={fresh.compiled.graph.node("resources").kernel: "changed"}
        ),
    )
    overlay = collect_execution_evidence(
        graph_schema(fresh.compiled),
        runs=[
            record_run_binding(
                fresh.compiled, fresh.manifest, attempt_id="first", phase="p"
            ),
            record_run_binding(
                changed.compiled, changed.manifest, attempt_id="second", phase="p"
            ),
        ],
        store=fresh.store,
    )
    first, second = (phase["operations"] for phase in overlay["phases"])
    assert second["survey"]["cache"] == "hit"
    assert first["survey"]["node_key"] == second["survey"]["node_key"]
    assert changed.misses() and changed.hits()
    for node_id in changed.misses():
        assert second[node_id]["cache"] == "miss"
        assert second[node_id]["execution"] == "completed"
        assert first[node_id]["node_key"] != second[node_id]["node_key"]


def test_rejects_another_graph_even_with_same_operation_names(fresh):
    node = fresh.compiled.graph.node("resources")
    changed = toy.replace_node(
        fresh.compiled.graph, replace(node, params={**node.params, "scale": 9.0})
    )
    from microcosm.graph import compile_graph

    with pytest.raises(ValueError, match="receipt does not match graph"):
        record_run_binding(
            compile_graph(changed), fresh.manifest, attempt_id="a", phase="p"
        )


def test_detached_manifest_cannot_invent_missing_source_binding(fresh):
    from microcosm.graph import RunManifest

    detached = RunManifest.from_json(fresh.manifest.to_json())
    with pytest.raises(ValueError, match="source identities"):
        record_run_binding(fresh.compiled, detached, attempt_id="a", phase="p")


def test_incomplete_history_is_explicit_and_raw_receipts_are_excluded(fresh):
    receipts = dict(fresh.manifest.nodes)
    receipts.pop(fresh.compiled.order[-1])
    receipts["survey"] = replace(
        receipts["survey"], receipt={"private_household_ids": [123456789]}
    )
    partial = replace(fresh.manifest, nodes=receipts)
    binding = record_run_binding(
        fresh.compiled, partial, attempt_id="a", phase="partial"
    )
    evidence = collect_execution_evidence(
        graph_schema(fresh.compiled), runs=[binding], store=fresh.store
    )
    assert (
        evidence["phases"][0]["operations"][fresh.compiled.order[-1]]["execution"]
        == "missing"
    )
    assert "absent earlier computation is unknown" in evidence["history"]
    assert "private_household_ids" not in json.dumps(evidence)


def test_saved_evidence_roundtrip_and_cli(fresh, tmp_path):
    directory = tmp_path / "evidence"
    index = save_run_evidence(
        fresh.compiled,
        fresh.manifest,
        store=fresh.store,
        directory=directory,
        attempt_id="a",
        phase="p",
    )
    assert not list(directory.rglob("*.orrery.json"))
    runs = load_run_evidence(index, store=fresh.store)
    assert runs[0].binding["source_identities"] == fresh.manifest.source_identities
    schema = graph_schema(
        fresh.compiled,
        extensions={PRESENTATION_EXTENSION: presentation(fresh.compiled)},
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    output = tmp_path / "orrery.json"
    # Read-only export must not recreate a writer's temporary directory.
    (fresh.store.root / "tmp").rmdir()
    before = toy.total_calls(fresh.registry)
    assert (
        main(
            [
                "--schema",
                str(schema_path),
                "--evidence",
                str(index),
                "--store",
                str(fresh.store.root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert toy.total_calls(fresh.registry) == before
    assert not (fresh.store.root / "tmp").exists()
    document = json.loads(output.read_text())
    assert document["activities"]
    assert all(len(artifact["sha256"]) == 64 for artifact in document["artifacts"])
    assert next(node for node in document["nodes"] if node["label"] == "survey")[
        "sources"
    ]
    with pytest.raises(SystemExit):
        main(
            [
                "--schema",
                str(schema_path),
                "--evidence",
                str(index),
                "--output",
                str(output),
            ]
        )


def test_corrupt_evidence_and_store_payloads_fail_closed(fresh, tmp_path):
    index = save_run_evidence(
        fresh.compiled,
        fresh.manifest,
        store=fresh.store,
        directory=tmp_path / "evidence",
        attempt_id="a",
        phase="p",
    )
    values = json.loads(index.read_text())
    graph_path = index.parent / values["runs"][0]["graph"]["path"]
    graph_path.write_bytes(graph_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="file digest mismatch"):
        load_run_evidence(index, store=fresh.store)
    binding = record_run_binding(
        fresh.compiled, fresh.manifest, attempt_id="a", phase="p"
    )
    artifact = next(iter(fresh.manifest.nodes["resources"].artifacts.values()))
    metadata = fresh.store.metadata(artifact)
    payload = fresh.store.object_path(artifact) / next(iter(metadata["payloads"]))
    payload.write_bytes(b"corrupt")
    with pytest.raises(StoreCorrupt):
        collect_execution_evidence(
            graph_schema(fresh.compiled), runs=[binding], store=fresh.store
        )


def test_offline_evidence_uses_the_recorded_platform(tmp_path, monkeypatch):
    import microcosm.graph.evidence as evidence_module
    import microcosm.graph.keys as keys_module
    from microcosm.graph import Numeric

    registry = toy.toy_registry()
    kernel = registry.get("source.csv@1")
    kernel.capabilities = replace(kernel.capabilities, numeric=Numeric.PLATFORM_BITWISE)
    monkeypatch.setattr(
        keys_module, "platform_fingerprint", lambda: "original-platform"
    )
    monkeypatch.setattr(
        evidence_module, "platform_fingerprint", lambda: "original-platform"
    )
    run = toy.run_toy(toy.small_graph(), tmp_path, registry=registry)
    index = save_run_evidence(
        run.compiled,
        run.manifest,
        store=run.store,
        directory=tmp_path / "evidence",
        attempt_id="a",
        phase="p",
    )
    monkeypatch.setattr(keys_module, "platform_fingerprint", lambda: "viewer-platform")
    loaded = load_run_evidence(index, store=run.store)
    overlay = collect_execution_evidence(
        graph_schema(run.compiled), runs=loaded, store=run.store
    )
    assert (
        overlay["phases"][0]["operations"]["survey"]["node_key"] == run.keys()["survey"]
    )


def test_artifact_producer_metadata_must_match_the_receipt(fresh):
    key = next(iter(fresh.manifest.nodes["resources"].artifacts.values()))
    path = fresh.store.object_path(key) / "meta.json"
    metadata = json.loads(path.read_bytes())
    metadata["node_key"] = "0" * 64
    path.write_text(json.dumps(metadata))
    binding = record_run_binding(
        fresh.compiled, fresh.manifest, attempt_id="a", phase="p"
    )
    with pytest.raises(ValueError, match="artifact producer identity mismatch"):
        collect_execution_evidence(
            graph_schema(fresh.compiled), runs=[binding], store=fresh.store
        )


def test_missing_required_store_artifact_is_actionable(fresh, tmp_path, capsys):
    index = save_run_evidence(
        fresh.compiled,
        fresh.manifest,
        store=fresh.store,
        directory=tmp_path / "evidence",
        attempt_id="a",
        phase="p",
    )
    key = next(iter(fresh.manifest.nodes["resources"].artifacts.values()))
    fresh.store.object_path(key).rename(tmp_path / "removed-object")
    binding = record_run_binding(
        fresh.compiled, fresh.manifest, attempt_id="a", phase="p"
    )
    with pytest.raises(StoreMiss, match=key):
        collect_execution_evidence(
            graph_schema(fresh.compiled), runs=[binding], store=fresh.store
        )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(graph_schema(fresh.compiled)))
    output = tmp_path / "missing.orrery.json"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--schema",
                str(schema_path),
                "--evidence",
                str(index),
                "--store",
                str(fresh.store.root),
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert key in capsys.readouterr().err
    assert not output.exists()


def test_shared_contract_modules_have_no_country_dispatch_or_imports():
    directory = _PATHS.package / "src" / "microcosm" / "graph"
    for name in ("orrery.py", "evidence.py", "presentation.py", "orrery_evidence.py"):
        tree = ast.parse((directory / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "uk" not in (node.module or "").split(".")
                assert "uk_runtime" not in (node.module or "").split(".")
            elif isinstance(node, ast.Import):
                assert all("uk_runtime" not in alias.name for alias in node.names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"uk", "us"}
                assert not node.value.startswith(("uk.full.", "hmrc_"))


@pytest.mark.parametrize(
    "change", ["member", "cycle", "duplicate", "unsafe_url", "boundary"]
)
def test_invalid_presentation_contracts_fail(fresh, change):
    contract = presentation(fresh.compiled)
    if change == "member":
        contract["groups"][1]["operations"].append("invented")
    elif change == "cycle":
        contract["groups"][0]["parent"] = "sources"
    elif change == "duplicate":
        contract["groups"][0]["operations"] = ["survey"]
    elif change == "unsafe_url":
        contract["sources"]["survey"]["references"][0]["url"] = "javascript:alert(1)"
    else:
        contract["scope"]["boundaries"] = [
            {"operation": "survey", "kind": "checkpoint"}
        ]
    with pytest.raises(ValueError):
        orrery_document(fresh.compiled, extensions={PRESENTATION_EXTENSION: contract})


def test_execution_overlay_graph_binding_is_checked(fresh):
    run = record_run_binding(fresh.compiled, fresh.manifest, attempt_id="a", phase="p")
    evidence = collect_execution_evidence(
        graph_schema(fresh.compiled), runs=[run], store=fresh.store
    )
    changed = copy.deepcopy(evidence)
    changed["graph_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="overlay graph binding mismatch"):
        orrery_document(fresh.compiled, execution=changed)


def test_public_orrery_parser_accepts_groups_and_execution(fresh):
    run = record_run_binding(fresh.compiled, fresh.manifest, attempt_id="a", phase="p")
    extension = {PRESENTATION_EXTENSION: presentation(fresh.compiled)}
    evidence = collect_execution_evidence(
        graph_schema(fresh.compiled), runs=[run], store=fresh.store
    )
    payload = orrery_json(fresh.compiled, extensions=extension, execution=evidence)
    script = _PATHS.repository / "tools" / "orrery-contract" / "verify-execution.mjs"
    result = subprocess.run(
        ["node", str(script)], input=payload, text=True, capture_output=True, check=True
    )
    assert "accepted" in result.stdout
