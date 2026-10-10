"""Country-neutral binding, phase history and existing Orrery surfaces."""

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from microcosm.graph import (
    checkpoint_references,
    collect_execution_evidence,
    graph_schema,
    load_run_evidence,
    orrery_document,
    orrery_json,
    publish_run_evidence,
    record_run_binding,
    save_graph_schema,
    save_run_evidence,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.evidence import sha256
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
    assert {record["kernel_role"] for record in history} == {"compute"}
    assert history[0]["artifacts"]
    # Cards show two badges: the execution state and an attempt-level cache
    # summary that names the computing phase before counting replays.
    assert [badge["label"] for badge in operation["statuses"]] == [
        "Execution: completed",
        "Computed: numerical · reused 1×",
    ]
    gate = next(node for node in document["nodes"] if node["label"] == "gate_tax")
    assert [badge["label"] for badge in gate["statuses"]][:2] == [
        "Gate: pass",
        "Execution: completed",
    ]
    assert not any(
        badge["label"].startswith("Gate:")
        for node in document["nodes"]
        if node["kind"] == "operation" and node["label"] != "gate_tax"
        for badge in node["statuses"]
    )
    for node in document["nodes"]:
        if node["kind"] == "field":
            assert json.loads(node["parentId"])[0] == "operation"
        if node["kind"] == "operation":
            assert "sources" not in node
    assert len(document["activities"]) == 2 * len(fresh.compiled.order)
    assert {activity["agent"]["name"] for activity in document["activities"]} == {
        node.kernel for node in fresh.compiled.graph.nodes
    }
    assert document["artifacts"]
    labels = [artifact["label"] for artifact in document["artifacts"]]
    assert all(label.endswith(("file)", "files)")) for label in labels)
    assert len(document["artifacts"]) == len(
        {
            artifact["key"]
            for phase in evidence["phases"]
            for artifact in phase["artifacts"]
        }
    )
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
    run_files = [artifact for artifact in document["artifacts"] if artifact.get("uri")]
    assert {artifact["label"] for artifact in run_files} == {
        f"p: {name}" for name in ("graph", "manifest", "binding", "summaries")
    }
    assert all(len(artifact["sha256"]) == 64 for artifact in run_files)
    assert all(
        {artifact["id"] for artifact in run_files} <= set(activity["artifactIds"])
        for activity in document["activities"]
    )
    # A multi-file store object has no single byte digest to declare; its
    # per-file digests stay in metadata.execution.phases[].artifacts.
    for artifact in document["artifacts"]:
        if artifact["label"].endswith("files)"):
            assert "sha256" not in artifact
    assert all(
        {payload["sha256"] for payload in artifact["payloads"].values()}
        for phase in document["metadata"]["execution"]["phases"]
        for artifact in phase["artifacts"]
    )
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


def test_publish_run_evidence_merges_attempts_and_refuses_corrupt_files(
    fresh, tmp_path
):
    out = tmp_path / "out"
    first = save_run_evidence(
        fresh.compiled,
        fresh.manifest,
        store=fresh.store,
        directory=tmp_path / "attempt-1",
        attempt_id="attempt-1",
        phase="numerical",
    )
    publish_run_evidence(first, out)
    cached = toy.run_toy(
        fresh.compiled.graph,
        tmp_path,
        sources=fresh.sources,
        registry=fresh.registry,
        store=fresh.store,
    )
    second = save_run_evidence(
        cached.compiled,
        cached.manifest,
        store=fresh.store,
        directory=tmp_path / "attempt-2",
        attempt_id="attempt-2",
        phase="final",
    )
    published = publish_run_evidence(second, out)
    index = json.loads(published.read_bytes())
    assert [(run["attempt_id"], run["phase"]) for run in index["runs"]] == [
        ("attempt-1", "numerical"),
        ("attempt-2", "final"),
    ]
    assert all(
        Path(run[field]["path"]).parent == Path(".")
        for run in index["runs"]
        for field in ("graph", "manifest", "binding", "summaries")
    )
    runs = load_run_evidence(published, store=fresh.store)
    overlay = collect_execution_evidence(
        graph_schema(fresh.compiled), runs=runs, store=fresh.store
    )
    assert [phase["operations"]["survey"]["cache"] for phase in overlay["phases"]] == [
        "miss",
        "hit",
    ]
    document = orrery_document(fresh.compiled, execution=overlay)
    survey = next(node for node in document["nodes"] if node["label"] == "survey")
    assert survey["statuses"][1]["label"] == "Computed: numerical · reused 1×"
    # Republishing the same attempt replaces its entry instead of duplicating it.
    publish_run_evidence(second, out)
    assert len(json.loads(published.read_bytes())["runs"]) == 2
    damaged = out / index["runs"][0]["graph"]["path"]
    damaged.write_bytes(damaged.read_bytes() + b" ")
    with pytest.raises(ValueError, match="no longer matches"):
        publish_run_evidence(second, out)
    damaged.unlink()
    with pytest.raises(ValueError, match="is missing"):
        publish_run_evidence(second, out)


def test_checkpoint_references_link_only_verifiable_bytes(tmp_path):
    graph = tmp_path / "graph.json"
    manifest = tmp_path / "manifest.json"
    evidence = tmp_path / "execution.evidence.json"
    for path in (graph, manifest, evidence):
        path.write_bytes(b'{"recorded":true}')
    record = {
        "graph_declaration": {
            "path": "graph.json",
            "sha256": sha256(graph.read_bytes()),
        },
        "graph_manifest": {
            "path": str(manifest),
            "sha256": sha256(manifest.read_bytes()),
        },
        "graph_execution_evidence": {
            "path": str(evidence),
            "sha256": sha256(evidence.read_bytes()),
        },
    }
    refs, missing = checkpoint_references(record, base=tmp_path)
    assert [ref["label"] for ref in refs] == [
        "Producing graph declaration",
        "Producing run manifest",
        "Producing execution evidence index",
    ]
    assert all(Path(ref["url"]).is_absolute() for ref in refs)
    assert missing is None
    manifest.unlink()
    refs, missing = checkpoint_references(record, base=tmp_path)
    assert [ref["label"] for ref in refs] == [
        "Producing graph declaration",
        "Producing execution evidence index",
    ]
    assert "Producing run manifest bytes unavailable" in missing
    assert record["graph_manifest"]["sha256"] in missing
    refs, missing = checkpoint_references({}, base=tmp_path)
    assert refs == [] and "Producing graph declaration" in missing
    graph.write_bytes(b'{"recorded":false}')
    with pytest.raises(ValueError, match="digest mismatch"):
        checkpoint_references(record, base=tmp_path)


def test_save_graph_schema_copies_upstream_bytes_and_relinks(fresh, tmp_path):
    from microcosm.graph.presentation import PRESENTATION_EXTENSION

    upstream = tmp_path / "spine.graph.json"
    upstream.write_bytes(b'{"spine":true}')
    contract = presentation(fresh.compiled)
    contract["scope"]["boundaries"] = [
        {
            "operation": "survey",
            "kind": "checkpoint",
            "upstream": [
                {
                    "label": "Producing graph declaration",
                    "url": str(upstream),
                    "sha256": sha256(upstream.read_bytes()),
                },
                {"label": "Published method", "url": "https://example.org/method"},
            ],
        }
    ]
    saved = save_graph_schema(fresh.compiled, tmp_path / "out", presentation=contract)
    schema = json.loads(saved.read_bytes())
    refs = schema["extensions"][PRESENTATION_EXTENSION]["scope"]["boundaries"][0][
        "upstream"
    ]
    assert refs[0]["url"] == f"upstream-{refs[0]['sha256']}.json"
    assert (saved.parent / refs[0]["url"]).read_bytes() == upstream.read_bytes()
    assert refs[1]["url"] == "https://example.org/method"
    assert canonical_json(schema) == saved.read_bytes()
    upstream.write_bytes(b'{"spine":false}')
    with pytest.raises(ValueError, match="changed during capture"):
        save_graph_schema(fresh.compiled, tmp_path / "again", presentation=contract)


@pytest.mark.parametrize(
    "composite",
    [
        "single",
        {"execution_unit": "paired"},
        {"operations": [{"parameters": {}}]},
        {"coupling": 7},
    ],
)
def test_malformed_composite_metadata_is_refused(fresh, composite):
    contract = presentation(fresh.compiled)
    contract["operations"]["survey"]["composite"] = composite
    with pytest.raises(ValueError, match="composite"):
        orrery_document(fresh.compiled, extensions={PRESENTATION_EXTENSION: contract})


def test_item_heavy_native_evidence_is_read_hashed_and_copied(fresh, tmp_path):
    """A real spine manifest holds over a million small JSON items in 13 MB."""
    from microcosm.graph.evidence import read_bytes, read_json
    from microcosm.graph.presentation import PRESENTATION_EXTENSION

    heavy = tmp_path / "manifest.json"
    heavy.write_bytes(json.dumps({"receipts": [[0, 1]] * 1_500_000}).encode())
    assert heavy.stat().st_size < 32 * 1024 * 1024
    value, payload = read_json(heavy)
    assert len(value["receipts"]) == 1_500_000
    assert sha256(payload) == sha256(read_bytes(heavy))
    record = {
        "graph_declaration": {"path": str(heavy), "sha256": sha256(payload)},
        "graph_manifest": {"path": str(heavy), "sha256": sha256(payload)},
    }
    refs, missing = checkpoint_references(record, base=tmp_path)
    assert len(refs) == 2 and missing is None
    contract = presentation(fresh.compiled)
    contract["scope"]["boundaries"] = [
        {"operation": "survey", "kind": "checkpoint", "upstream": refs}
    ]
    saved = save_graph_schema(fresh.compiled, tmp_path / "out", presentation=contract)
    copied = json.loads(saved.read_bytes())["extensions"][PRESENTATION_EXTENSION]
    url = copied["scope"]["boundaries"][0]["upstream"][0]["url"]
    assert (saved.parent / url).read_bytes() == payload


@pytest.mark.parametrize(
    "text, message",
    [
        ("[NaN]", "finite"),
        ("[" * 70 + "]" * 70, "nesting"),
        ('{"a": 1, "a": 2}', "duplicate"),
    ],
)
def test_native_evidence_reader_refuses_malformed_json(tmp_path, text, message):
    from microcosm.graph.evidence import read_json

    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_json(path)
