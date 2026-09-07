"""Actual small survey graph proof; source inputs are code-owned inventions."""

import hashlib
import json
import subprocess
from dataclasses import replace

from survey_lineage_prep import invented
from us_survey_lineage_graph import (
    CLONE_BINDING,
    CLONE_POPULATION,
    NATIVE_BINDING,
    NATIVE_POPULATION,
    integrated_graph,
    integrated_registry,
)

from microcosm.build.cd_benchmark_io import source_adapters as io
from microcosm.build.us_runtime import graph_cd_source_evidence as stage
from microcosm.build.us_runtime import graph_composed_population as composed
from microcosm.build.us_runtime.graph_sources import us_source_codecs
from microcosm.graph import (
    ContentStore,
    Owned,
    compile_graph,
    run_graph,
)
from microcosm.graph.keys import node_key, source_content_key


def setup(root, checkout, *, create):
    fixture = root / "fixture"
    if create:
        fixture.mkdir()
        invented.make(fixture, checkout)
    invented.restore(fixture, checkout)
    projection = json.loads((fixture / "projection-identity.json").read_text())
    graph = integrated_graph(
        tuple(
            Owned(*row) for row in json.loads((fixture / "columns.json").read_text())
        ),
        sample_fraction=1.0,
        sample_seed=17,
        projection_sha256=projection["projection_sha256"],
        rows=projection["rows"],
        cohort_rows=tuple(tuple(row) for row in projection["cohort_rows"]),
    )
    inventory = {}
    compiled = compile_graph(graph)
    for node_id in compiled.order:
        for cell in graph.node(node_id).outputs:
            inventory[cell.entity, cell.column] = Owned(
                cell.entity, cell.column, cell.dtype
            )
    extra = stage.evidence_nodes(
        tuple(inventory[key] for key in sorted(inventory)),
        native_population=NATIVE_POPULATION,
        current_population=CLONE_POPULATION,
        native_binding=NATIVE_BINDING,
        clone_binding=CLONE_BINDING,
        prepared_node=composed.CREATE_NODE,
    )
    extended = replace(graph, nodes=(*graph.nodes, *extra))
    registry = integrated_registry()
    stage.register(registry)
    store = ContentStore(root / "store", codecs=us_source_codecs())
    return graph, extended, registry, store, invented.sources(fixture)


def registry_calls(registry, calls, *, fatal):
    for kernel in registry._kernels.values():
        original = kernel.run

        def wrapped(context, _original=original, _ref=kernel.ref):
            calls.append(_ref)
            if fatal:
                raise RuntimeError("REQUIRED_KERNEL_FORBIDDEN")
            return _original(context)

        kernel.run = wrapped


def verified(root, checkout, manifest, store, graph, observations, expected):
    head = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    return stage.verify_materialized(
        manifest,
        store,
        graph,
        observations,
        expected_node_keys=expected,
        native_population=NATIVE_POPULATION,
        clone_population=CLONE_POPULATION,
        current_population=CLONE_POPULATION,
        native_binding_node=NATIVE_BINDING,
        clone_binding_node=CLONE_BINDING,
        source_git_head=head,
    )


def run(mode, root, checkout):
    initial, graph, registry, store, sources = setup(
        root, checkout, create=mode == "cold"
    )
    # Independently derive the complete roster before executing or reading its
    # node receipts. Hashes here cover only the owned invented source files.
    source_keys = {
        name: source_content_key(name, path) for name, path in sources.items()
    }
    compiled = compile_graph(graph)
    expected = {}
    implementations = {}
    for name in compiled.order:
        kernel = registry.get(graph.node(name).kernel)
        if kernel.ref not in implementations:
            implementations[kernel.ref] = kernel.implementation_hash()
        implementation = implementations[kernel.ref]
        expected[name] = node_key(
            compiled,
            name,
            expected,
            implementation,
            source_keys,
            kernel_capabilities=kernel.capabilities,
        )
    calls, observed = [], {}

    def observe(node_id, population):
        observed[node_id] = population
        observed[population.version] = population

    registry_calls(registry, calls, fatal=mode != "cold")
    before = None
    if mode == "cold":
        before = run_graph(
            compile_graph(initial),
            sources=sources,
            store=store,
            kernels=registry,
            resume="forbid",
            _population_observer=observe,
        )
        old_keys = {name: node.key for name, node in before.nodes.items()}
        calls.clear()
    manifest = run_graph(
        compile_graph(graph),
        sources=sources,
        store=store,
        kernels=registry,
        resume="auto" if mode == "cold" else "require",
        _population_observer=observe,
    )
    keys = {name: node.key for name, node in manifest.nodes.items()}
    key_number = len(list(root.glob(mode + "-key-proof*.json")))
    (root / (mode + f"-key-proof-{key_number}.json")).write_text(
        json.dumps(
            {
                "keys": keys,
                "upstream_keys": old_keys if before is not None else None,
                "implementation_hashes": {
                    name: node.kernel_impl_hash for name, node in manifest.nodes.items()
                },
                "hits": {name: node.hit for name, node in manifest.nodes.items()},
                "calls": calls,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if before is not None:
        assert all(
            manifest.node(name).key == key and manifest.node(name).hit
            for name, key in old_keys.items()
        )
        assert len(calls) == 3
    else:
        old_keys = json.loads((root / "graph-result.json").read_text())["keys"]
        assert (
            keys == old_keys
            and not calls
            and all(node.hit for node in manifest.nodes.values())
        )
    assert keys == expected
    result = verified(root, checkout, manifest, store, graph, observed, expected)
    payloads = {
        name: hashlib.sha256(
            store.load_bytes(
                manifest.node(name).opaque_artifacts[
                    "projection" if name == stage.ACS_NODE else "evidence"
                ]
            )
        ).hexdigest()
        for name in (stage.ACS_NODE, stage.UNIVERSE_NODE, stage.GEOGRAPHY_NODE)
    }
    summary = {"keys": keys, "payloads": payloads, "result": result}
    if mode == "cold":
        (root / "graph-result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        control, completion, seal = io.write_verifier_control(
            root / "external-control",
            result=result,
            run_id="invented-cd-source",
            proposal_sha256="1" * 64,
            verifier_implementation_sha256=stage.implementation_hash(),
        )
        slot_pins = io.write_slot_summaries(
            root / "candidate-bundle", control_pin=control
        )
        adapters = io.producer_adapters(
            control_pin=control,
            completion_pin=completion,
            seal_pin=seal,
            expected_run_id="invented-cd-source",
            expected_proposal_sha256="1" * 64,
            expected_verifier_implementation_sha256=stage.implementation_hash(),
            candidate_files=slot_pins,
        )
        for slot in slot_pins:
            from microcosm.build.cd_benchmark_io.artifacts import document

            docs = {name: document(p) for name, p in slot_pins.items()}
            assert (
                dict(adapters[slot](slot, docs[slot], docs).identity)
                == result["identity"]
            )
    else:
        previous = json.loads((root / "graph-result.json").read_text())
        assert summary == previous
    if mode == "checks":
        checks = invented.peer(checkout, "test_us_cd_source_evidence")
        check_root = root / (
            "checks-attempt-" + str(len(list(root.glob("checks-attempt-*"))))
        )
        check_root.mkdir()
        return checks.exercise(
            check_root, checkout, manifest, store, graph, observed, result, keys
        )
    return {
        "nodes": len(keys),
        "calls": calls,
        "hits": sum(node.hit for node in manifest.nodes.values()),
        "new_payload_sha256": payloads,
        "materialized_result_sha256": hashlib.sha256(
            json.dumps(result, sort_keys=True).encode()
        ).hexdigest(),
        "original_anchor_roster": sorted(observed[CLONE_POPULATION].design_weights),
        "row_counts": result["slots"][io.UNIVERSE]["evidence"]["entity_rows"],
        "initial17_preserved": before is not None or keys == old_keys,
    }
