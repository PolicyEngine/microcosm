"""Actual graph producers, clone lineage and fresh zero-call replay, invented only."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from microcosm.build.us_runtime import graph_native_household_origin as stage
from microcosm.build.us_runtime import native_household_origin as native
from microcosm.build.us_runtime.graph_combined_clone import (
    USCombinedSurveyCloneClaimKernel,
    USCombinedSurveyCloneExpandKernel,
    us_combined_survey_clone_nodes,
)
from microcosm.build.us_runtime.graph_native_origin_implementation import (
    implementation_hash,
)
from microcosm.build.us_runtime.graph_sources import (
    frame_column_declarations,
    us_source_codecs,
)
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import load_frame_store


def _peer(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _restore(root):
    meta = json.loads((root / "fixture.json").read_text())
    native.acs._ARCHIVE_PINS = tuple(tuple(pin) for pin in meta["acs_pins"])
    native._ASEC_MEMBER_PINS = tuple(
        native.AsecNativeMemberPin(**pin) for pin in meta["asec_pins"]
    )
    return meta


def _graph_run(
    root, *, resume="forbid", forbid_calls=False, omit_column=False, mutate_column=False
):
    meta = _restore(root)
    columns = tuple(Owned(*entry) for entry in meta["columns"])
    binding_columns = tuple(
        c for c in columns if not (omit_column and c.column == "source_record_note")
    )
    clones = us_combined_survey_clone_nodes(
        columns, base="source", source_channels=("acs", "asec")
    )
    create = _peer("test_us_graph_combined_clone")._Create
    graph = Graph(
        country="us",
        sources=(SourceRef("population", "frame-store"), *stage.SOURCES),
        nodes=(
            Node(
                "source",
                create.ref,
                sources=("population",),
                structural=StructuralDelta.CREATE,
                outputs=columns,
            ),
            *stage.native_origin_source_nodes(population="source"),
            stage.native_origin_binding_node(
                binding_columns, population="source", node_id="native_binding"
            ),
            *clones,
            stage.native_origin_binding_node(
                binding_columns,
                population=clones[0].id,
                node_id="clone_binding",
                parent_binding_node="native_binding",
            ),
        ),
    )
    classes = [
        create,
        stage.NativeACSOriginKernel,
        stage.NativeASECOriginKernel,
        stage.PopulationOriginBindingKernel,
        USCombinedSurveyCloneExpandKernel,
        USCombinedSurveyCloneClaimKernel,
    ]
    registry = KernelRegistry()
    calls = []
    for cls in classes:
        kernel = cls()
        original = kernel.run

        def run(context, original=original, ref=kernel.ref):
            calls.append(ref)
            if forbid_calls:
                raise BaseException("REQUIRED replay executed " + ref)
            return original(context)

        kernel.run = run
        registry.register(kernel)
    codecs = us_source_codecs()
    codecs.register("frame-store", load_frame_store)
    store = ContentStore(root / "store", codecs=codecs)
    sources = {
        "population": Path(meta["population"]),
        stage.ACS_SOURCE: Path(meta["acs_source"]),
    }
    sources.update(
        {
            name: Path(meta["asec_paths"][str(year)])
            for year, name in zip((2022, 2023, 2024), stage.ASEC_SOURCES, strict=True)
        }
    )
    manifest = run_graph(
        compile_graph(graph),
        store=store,
        sources=sources,
        kernels=registry,
        resume=resume,
    )
    payload = store.load_bytes(
        manifest.nodes["clone_binding"].opaque_artifacts["binding"]
    )
    if mutate_column:
        manifest.population(clones[0].id).person.loc[0, "source_record_note"] = (
            "invented mutation"
        )
    bound = stage.verify_materialized_population_origins(
        manifest,
        store,
        population_node=clones[0].id,
        binding_node="clone_binding",
        parent_binding_node="native_binding",
    )
    assert bound.payload == payload
    return {
        "pid": os.getpid(),
        "calls": calls,
        "nodes": {
            name: {"key": value.key, "cache_hit": value.hit}
            for name, value in manifest.nodes.items()
        },
        "binding_sha256": hashlib.sha256(payload).hexdigest(),
        "summary": json.loads(payload)["summary"],
        "build_file": str(Path(native.__file__).resolve()),
    }


def _fixture_graph(tmp_path, monkeypatch):
    direct = _peer("test_us_native_household_origin")
    frame, _, paths, acs_fixture = direct.inputs.__wrapped__(tmp_path, monkeypatch)
    frame.person["source_record_note"] = _peer("test_us_graph_combined_clone")._string(
        [f"invented source row {i}" for i in range(frame.n("person"))]
    )
    root = tmp_path.resolve() / "graph"
    root.mkdir()
    population = ContentStore(root / "inputs").put_frame(
        hashlib.sha256(b"invented-origin-pool").hexdigest(), frame
    )
    meta = {
        "population": str(population),
        "acs_source": str(acs_fixture.source_dir),
        "acs_pins": list(native.acs._ARCHIVE_PINS),
        "asec_pins": [asdict(pin) for pin in native._ASEC_MEMBER_PINS],
        "asec_paths": {str(y): str(p) for y, p in paths.items()},
        "columns": [
            [c.entity, c.column, c.dtype] for c in frame_column_declarations(frame)
        ],
    }
    (root / "fixture.json").write_text(json.dumps(meta))
    return root


def test_actual_graph_clone_and_fresh_required_zero_call(tmp_path, monkeypatch):
    root = _fixture_graph(tmp_path, monkeypatch)
    cold = _graph_run(root)
    assert len(cold["calls"]) == 7
    assert cold["summary"]["households"] == 10
    assert cold["summary"]["original_household_origins"] == 5
    script = """
import importlib.util,json,sys
from pathlib import Path
path=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('origin_replay',path)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
result=module._graph_run(Path(sys.argv[2]),resume='require',forbid_calls=True)
Path(sys.argv[3]).write_text(json.dumps(result,sort_keys=True))
"""
    output = root / "warm.json"
    subprocess.run(
        [sys.executable, "-c", script, __file__, str(root), str(output)],
        check=True,
        timeout=180,
    )
    warm = json.loads(output.read_text())
    assert warm["pid"] != cold["pid"]
    assert warm["calls"] == []
    assert all(node["cache_hit"] for node in warm["nodes"].values())
    assert {k: v["key"] for k, v in warm["nodes"].items()} == {
        k: v["key"] for k, v in cold["nodes"].items()
    }
    assert warm["binding_sha256"] == cold["binding_sha256"]
    assert warm["build_file"] == str(Path(native.__file__).resolve())


@pytest.mark.parametrize("mutation", ["omit_column", "mutate_column"])
def test_materialized_boundary_rejects_incomplete_or_changed_projection(
    tmp_path, monkeypatch, mutation
):
    root = _fixture_graph(tmp_path, monkeypatch)
    with pytest.raises(native.NativeOriginError, match="POPULATION_CONTENT"):
        _graph_run(root, **{mutation: True})


def test_registry_changes_only_extension_identity(monkeypatch):
    from microcosm.build.us_runtime.graph_implementation import implementation_manifest

    before = implementation_manifest("composed_asec_binding_v1")
    original = implementation_hash()
    # The unused production registry is part of the new source identity; its
    # change cannot alter any byte or manifest in the nine-node base scope.
    monkeypatch.setattr(
        native, "asec_member_registry", lambda: [{"invented": "new-scope"}]
    )
    assert implementation_hash() != original
    assert implementation_manifest("composed_asec_binding_v1") == before
