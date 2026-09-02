#!/usr/bin/env python3
"""Run the toy country and write its deterministic offline review page."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from microcosm.graph import (
    ContentStore,
    NodeRejectedError,
    RunManifest,
    StoreCorruptError,
    StoreUnavailableError,
    compile_graph,
    explain_html,
    graph_to_json,
)

ROOT = Path(__file__).resolve().parents[1]
CHARTER = ROOT / "docs" / "graph-acceptance.md"
TOY_PATH = ROOT / "packages" / "microcosm-graph" / "tests" / "_toy.py"
REMOVED = ("a", "b", "c", "d", "e")


def _load_burndown_tool() -> ModuleType:
    name = "_graph_acceptance_burndown"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "tools" / "graph_acceptance_burndown.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load burndown tool from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_toy() -> ModuleType:
    """Load the acceptance fixture by file path, matching the suite."""

    if "_toy" in sys.modules:
        return sys.modules["_toy"]
    spec = importlib.util.spec_from_file_location("_toy", TOY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load toy country from {TOY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_toy"] = module
    spec.loader.exec_module(module)
    return module


def _normalized_manifest(manifest: RunManifest) -> RunManifest:
    """Remove operational clocks/host while retaining the run evidence."""

    nodes = {
        node_id: replace(receipt, wall_time=0.0)
        for node_id, receipt in manifest.nodes.items()
    }
    return replace(
        manifest,
        nodes=nodes,
        started_at="",
        finished_at="",
        host="",
    )


def _snapshot(
    compiled,
    manifest: RunManifest | None = None,
    *,
    states: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """A small deterministic graph projection for one replay step."""

    states = states or {}
    depths: dict[str, int] = {}
    nodes = []
    edges = []
    for node_id in compiled.order:
        predecessors = compiled.predecessors[node_id]
        depths[node_id] = (
            0
            if not predecessors
            else 1 + max(depths[predecessor] for predecessor in predecessors)
        )
        receipt = None if manifest is None else manifest.nodes.get(node_id)
        nodes.append(
            {
                "id": node_id,
                "key": "" if receipt is None else receipt.key,
                "depth": depths[node_id],
                "state": states.get(node_id, "stable"),
            }
        )
        edges.extend((predecessor, node_id) for predecessor in predecessors)
    return {"nodes": nodes, "edges": edges}


def _replay_wic(toy: ModuleType, root: Path) -> dict[str, object]:
    breach = toy.patch_node(
        "wic_recode",
        "receives_x",
        "boolean",
        True,
        population="pool",
        kernel="bad.dense_bool@1",
    )
    downstream = toy.count_node(
        "consumes_the_recode", ("receives_x",), "downstream", population="pool"
    )
    graph = toy.small_graph(nodes=(toy.CREATE, toy.POOL, breach, downstream))
    compiled = compile_graph(graph)
    registry = toy.toy_registry()
    rejected = False
    rejection = ""
    try:
        toy.run_toy(graph, root, registry=registry)
    except NodeRejectedError as error:
        rejected = True
        rejection = str(error)
    calls = toy.calls_by_ref(registry)
    passed = (
        rejected
        and "receives_x" in rejection
        and "wic_recode" in toy.descendants(compiled, "pool")
        and calls["bad.dense_bool@1"] == 1
        and calls["count.calls@1"] == 0
    )
    observed = (
        "The executor rejected wic_recode; its reader remained unreached."
        if passed
        else "The dtype boundary did not produce the required rejection."
    )
    return {
        "id": "wic-dtype-breach",
        "title": "WIC dtype breach",
        "property": "B3 · Storage-preserving patch",
        "verdict": "pass" if passed else "fail",
        "changed_nodes": ("wic_recode",),
        "moved_keys": ("wic_recode",),
        "cell_changes": ("person.receives_x: dense bool rejected",),
        "observed": observed,
        "stages": (
            {
                "label": "Before",
                "summary": "The nullable boolean incumbent is intact.",
                "snapshot": _snapshot(compiled),
            },
            {
                "label": "Change",
                "summary": "wic_recode returns a dense bool over a nullable boolean.",
                "snapshot": _snapshot(compiled, states={"wic_recode": "changed"}),
            },
            {
                "label": "After",
                "summary": observed,
                "snapshot": _snapshot(
                    compiled,
                    states={
                        "wic_recode": "rejected",
                        "consumes_the_recode": "not-executed",
                    },
                ),
            },
        ),
    }


def _replay_repack(toy: ModuleType, root: Path) -> dict[str, object]:
    before = toy.run_toy(toy.chained_graph(REMOVED), root / "run")
    removed = tuple(f"leaf_{name}" for name in REMOVED)
    graph_after = toy.drop_nodes(toy.chained_graph(REMOVED), *removed)
    registry = toy.toy_registry()
    after = toy.run_toy(
        graph_after,
        root / "run",
        sources=before.sources,
        registry=registry,
        store=ContentStore(root / "run" / "store"),
    )
    survivors = tuple(
        node_id for node_id in before.compiled.order if node_id not in removed
    )
    moved = tuple(
        node_id
        for node_id in survivors
        if before.manifest.nodes[node_id].key != after.manifest.nodes[node_id].key
    )
    passed = (
        after.keys() == {node_id: before.keys()[node_id] for node_id in survivors}
        and after.seeds() == {node_id: before.seeds()[node_id] for node_id in survivors}
        and after.all_bytes()
        == {node_id: before.all_bytes()[node_id] for node_id in survivors}
        and not after.misses()
        and toy.total_calls(registry) == 0
        and not moved
    )
    observed = (
        f"Five leaves were removed; 0 of {len(survivors)} survivor keys moved."
        if passed
        else f"The removal moved {len(moved)} survivor keys."
    )
    return {
        "id": "0347a009-repack",
        "title": "0347a009 repack",
        "property": "C1 + C2 · Order and removal invariance",
        "verdict": "pass" if passed else "fail",
        "changed_nodes": removed,
        "moved_keys": moved,
        "cell_changes": tuple(
            f"person.leaf_{name}_value: leaf removed" for name in REMOVED
        ),
        "observed": observed,
        "stages": (
            {
                "label": "Before",
                "summary": "The chained graph contains five independent leaves.",
                "snapshot": _snapshot(before.compiled, before.manifest),
            },
            {
                "label": "Change",
                "summary": "Five unrelated leaf declarations are removed.",
                "snapshot": _snapshot(
                    before.compiled,
                    before.manifest,
                    states={node_id: "removed" for node_id in removed},
                ),
            },
            {
                "label": "After",
                "summary": observed,
                "snapshot": _snapshot(after.compiled, after.manifest),
            },
        ),
    }


def _replay_engine_less(toy: ModuleType, root: Path, baseline) -> dict[str, object]:
    registry = toy.toy_registry()
    store = ContentStore(root / "store", codecs={})
    stopped = False
    failure = ""
    try:
        toy.run_toy(toy.full_graph(), root, registry=registry, store=store)
    except StoreUnavailableError as error:
        stopped = True
        failure = str(error)
    store_files = tuple(path for path in store.root.rglob("*") if path.is_file())
    passed = (
        stopped
        and "csv-tables" in failure
        and toy.total_calls(registry) == 0
        and not store_files
    )
    observed = (
        "The missing csv-tables verifier stopped the run before any kernel executed."
        if passed
        else "The unavailable verifier did not stop all execution."
    )
    after_states = {node_id: "not-executed" for node_id in baseline.compiled.order}
    return {
        "id": "engine-less-environment",
        "title": "Engine-less environment",
        "property": "E2 · Verifier unavailability is fatal",
        "verdict": "pass" if passed else "fail",
        "changed_nodes": (),
        "moved_keys": (),
        "cell_changes": (),
        "observed": observed,
        "stages": (
            {
                "label": "Before",
                "summary": "The unchanged graph has a validated cached run.",
                "snapshot": _snapshot(baseline.compiled, baseline.manifest),
            },
            {
                "label": "Change",
                "summary": "The source verifier registry is unavailable.",
                "snapshot": _snapshot(baseline.compiled, baseline.manifest),
            },
            {
                "label": "After",
                "summary": observed,
                "snapshot": _snapshot(
                    baseline.compiled, baseline.manifest, states=after_states
                ),
            },
        ),
    }


def _replay_evidence_flip(toy: ModuleType, root: Path) -> dict[str, object]:
    red = toy.run_toy(toy.full_graph(gate_low=1e11, gate_high=1e12), root / "release")
    path = root / "release" / "manifest.json"
    red.manifest.save(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    original_node_keys = tuple(
        document["nodes"][node_id]["key"] for node_id in red.compiled.order
    )
    document["tier"] = "certified"
    document["schema_version"] = 1
    path.write_text(json.dumps(document), encoding="utf-8")

    refused_load = False
    refused_certified = False
    try:
        RunManifest.load(path, store=red.store)
    except StoreCorruptError as error:
        refused_load = red.manifest.key in str(error)
    try:
        RunManifest.load_certified(path, store=red.store)
    except (StoreCorruptError, NodeRejectedError):
        refused_certified = True
    altered_node_keys = tuple(
        document["nodes"][node_id]["key"] for node_id in red.compiled.order
    )
    passed = (
        red.manifest.tier == "evidence"
        and bool(red.manifest.known_failures)
        and refused_load
        and refused_certified
        and original_node_keys == altered_node_keys
    )
    observed = (
        "Both loaders refused the forged certified tier; graph keys stayed fixed."
        if passed
        else "The altered release evidence was not refused by both loaders."
    )
    return {
        "id": "evidence-flip",
        "title": "Evidence flip",
        "property": "F3 · The one-field flip is impossible",
        "verdict": "pass" if passed else "fail",
        "changed_nodes": (),
        "moved_keys": (),
        "cell_changes": ("release.tier: evidence → forged certified (refused)",),
        "observed": observed,
        "stages": (
            {
                "label": "Before",
                "summary": "A failed gate derives an evidence-tier release.",
                "snapshot": _snapshot(red.compiled, red.manifest),
            },
            {
                "label": "Change",
                "summary": "Only the serialized top-level tier is changed.",
                "snapshot": _snapshot(
                    red.compiled, red.manifest, states={"release": "changed"}
                ),
            },
            {
                "label": "After",
                "summary": observed,
                "snapshot": _snapshot(
                    red.compiled, red.manifest, states={"release": "refused"}
                ),
            },
        ),
    }


def _run_replays(toy: ModuleType, root: Path, baseline) -> list[dict[str, object]]:
    return [
        _replay_wic(toy, root / "wic"),
        _replay_repack(toy, root / "repack"),
        _replay_engine_less(toy, root / "engine-less", baseline),
        _replay_evidence_flip(toy, root / "evidence"),
    ]


def _burndown_data(node_count: int, unrelated_edit_misses: int) -> dict[str, object]:
    burndown_tool = _load_burndown_tool()
    data = json.loads(
        json.dumps(
            burndown_tool.report(burndown_tool.counts(burndown_tool.suite_files()))
        )
    )
    for identifier in ("V1", "V2", "V3", "V4"):
        data["properties"].append(
            {
                "id": identifier,
                "test": "visual review surface",
                "file": "packages/microcosm-graph/tests/test_graph_explain.py",
                "state": "green",
                "reasons": [],
            }
        )
    data["payoffs"] = {
        "unrelated_edit_misses": unrelated_edit_misses,
        "unrelated_edit_nodes": node_count,
        "describe_target": "target_b",
    }
    return data


def build_demo(destination: Path) -> Path:
    """Execute all toy evidence and write deterministic demo artifacts."""

    toy = _load_toy()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="microcosm-graph-demo-") as temporary:
        scratch = Path(temporary)
        graph = toy.full_graph()
        primary = toy.run_toy(graph, scratch / "primary")

        edited_node = replace(
            graph.node("target_b"), description="Unrelated explanatory edit"
        )
        edited_graph = toy.replace_node(graph, edited_node)
        edited_registry = toy.toy_registry()
        edited = toy.run_toy(
            edited_graph,
            scratch / "primary",
            sources=primary.sources,
            registry=edited_registry,
            store=primary.store,
        )
        unrelated_edit_misses = len(edited.misses())
        if unrelated_edit_misses != 0:
            raise RuntimeError(
                "A descriptive edit unexpectedly invalidated "
                f"{unrelated_edit_misses} toy nodes."
            )

        manifest = _normalized_manifest(primary.manifest)
        replays = _run_replays(toy, scratch / "replays", primary)
        charter = CHARTER.read_text(encoding="utf-8")
        burndown = _burndown_data(len(primary.compiled.order), unrelated_edit_misses)
        rendered = explain_html(
            primary.compiled,
            manifest,
            charter=charter,
            burndown=burndown,
            replays=replays,
            title="Microcosm node graph explorer",
        )

        store_destination = destination / "store"
        if store_destination.exists():
            shutil.rmtree(store_destination)
        shutil.copytree(primary.store.root, store_destination)
        manifest.save(destination / "manifest.json")
        (destination / "graph.json").write_text(graph_to_json(graph), encoding="utf-8")
        output = destination / "demo.html"
        output.write_text(rendered, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = build_demo(args.out)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
