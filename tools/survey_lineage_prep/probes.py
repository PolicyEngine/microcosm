"""Targeted adversarial cases on actual integrated ContentStore populations."""

import hashlib
from dataclasses import replace


def refusals(root, manifest, store, graph, registry, sources):
    from us_survey_lineage_graph import (
        CLONE_BINDING,
        CLONE_POPULATION,
        NATIVE_BINDING,
        NATIVE_POPULATION,
        integrated_registry,
    )

    from microcosm.build.us_runtime import (
        graph_composed_asec_demographics as demographic,
    )
    from microcosm.build.us_runtime import graph_native_household_origin as origin
    from microcosm.build.us_runtime import native_household_origin as native
    from microcosm.build.us_runtime.support_provenance import (
        support_channel_column,
        support_clone_index_column,
    )
    from microcosm.graph import NodeRejectedError, compile_graph, run_graph

    results = {}

    def check(label, reason, function, exception=native.NativeOriginError):
        try:
            function()
        except exception as error:
            actual = (
                str(error.__cause__)
                if isinstance(error, NodeRejectedError)
                else str(error)
            )
            assert reason in actual, (label, actual)
            results[label] = {"status": "REFUSED", "reason": reason}
        else:
            raise AssertionError(label + " was admitted")

    node = graph.node(demographic.SOURCE_NODE)
    stale = replace(node, params={**node.params, "projection_sha256": "0" * 64})
    changed = replace(
        graph, nodes=tuple(stale if n.id == node.id else n for n in graph.nodes)
    )
    check(
        "stale_demographic_source_binding",
        "DEMOGRAPHIC_DECLARED_PROJECTION_DIGEST",
        lambda: run_graph(
            compile_graph(changed),
            sources=sources,
            store=store,
            kernels=integrated_registry(),
            resume="auto",
        ),
        NodeRejectedError,
    )

    # A genuine omitted-column case: execute a binding with PERIDNUM absent
    # from its declaration while that column remains in the actual stored Frame.
    node = graph.node(NATIVE_BINDING)
    column = "PERIDNUM"
    assert any(column in part.columns for part in node.inputs)
    partial = replace(
        node,
        inputs=tuple(
            replace(part, columns=tuple(c for c in part.columns if c != column))
            for part in node.inputs
        ),
        params={
            **node.params,
            "columns": tuple(row for row in node.params["columns"] if row[1] != column),
        },
    )
    native_graph = replace(
        graph,
        nodes=tuple(
            partial if n.id == node.id else n
            for n in graph.nodes
            if n.id
            not in (CLONE_POPULATION, CLONE_POPULATION + ".owned", CLONE_BINDING)
        ),
    )
    omitted = run_graph(
        compile_graph(native_graph),
        sources=sources,
        store=store,
        kernels=integrated_registry(),
        resume="auto",
    )
    check(
        "omitted_carried_source_column",
        "POPULATION_CONTENT",
        lambda: origin.verify_materialized_population_origins(
            omitted,
            store,
            population_node=NATIVE_POPULATION,
            binding_node=NATIVE_BINDING,
        ),
    )

    after = manifest.population(CLONE_POPULATION)
    acs_index = after.person.index[
        after.person[support_channel_column("person")].eq("acs")
    ][0]
    old = after.person.loc[acs_index, "is_female"]
    after.person.loc[acs_index, "is_female"] = not bool(old)
    check(
        "changed_acs_cell",
        "POPULATION_CONTENT",
        lambda: origin.verify_materialized_population_origins(
            manifest,
            store,
            population_node=CLONE_POPULATION,
            binding_node=CLONE_BINDING,
            parent_binding_node=NATIVE_BINDING,
        ),
    )
    after.person.loc[acs_index, "is_female"] = old

    def payload(node, output):
        return store.load_bytes(manifest.nodes[node].opaque_artifacts[output])

    # Original ASEC weights are strictly positive. ACS WGTP=0 placeholders
    # become positive current PWGTP in composition. Exercise downstream zero
    # weights in a separate seven-node invented Frame boundary, using the
    # actual complete integrated native Frame and the real clone/origin kernels.
    zero_manifest, zero_store = zero_weight_graph(
        root, manifest.population(NATIVE_POPULATION), sources
    )
    zero_binding = origin.verify_materialized_population_origins(
        zero_manifest,
        zero_store,
        population_node=CLONE_POPULATION,
        binding_node=CLONE_BINDING,
        parent_binding_node=NATIVE_BINDING,
    )
    assert zero_binding.summary["zero_weight_households_with_lineage"] == 4
    results["zero_weight_graph"] = {
        "status": "PASS",
        "nodes": len(zero_manifest.nodes),
        "summary": zero_binding.summary,
        "boundary": "invented downstream zero weights in both arms",
    }
    after = zero_manifest.population(CLONE_POPULATION)
    parent = native.PopulationOriginBinding(
        zero_store.load_bytes(
            zero_manifest.nodes[NATIVE_BINDING].opaque_artifacts["binding"]
        ),
        _token=native._BOUND_TOKEN,
    )
    source_values = [
        native.AuthenticatedNativeOriginSource(
            payload(n, "origins"), _token=native._TOKEN
        )
        for n in (origin.ACS_NODE, origin.ASEC_NODE)
    ]
    hh = after.household
    omitted_id = hh.loc[
        (after.weights_for("household").values == 0)
        & hh[support_clone_index_column("household")].eq(1),
        "household_id",
    ].iloc[0]
    selected = after.select(after.person.person_household_id != omitted_id)
    assert (
        selected.weights_for("household").values.sum()
        == after.weights_for("household").values.sum()
    )
    assert selected.n("household") == after.n("household") - 1
    store.put_frame(
        hashlib.sha256(b"invented-zero-clone-omission").hexdigest(), selected
    )
    check(
        "zero_weight_clone_household_omission",
        "CLONE_PAIR_COVERAGE",
        lambda: native.bind_population_origins(
            selected, sources=source_values, parent=parent
        ),
    )
    # The unmodified full candidate must still pass after all counterprobes.
    origin.verify_materialized_population_origins(
        manifest,
        store,
        population_node=CLONE_POPULATION,
        binding_node=CLONE_BINDING,
        parent_binding_node=NATIVE_BINDING,
    )
    return results


def zero_weight_graph(root, before, sources):
    from pathlib import Path

    from survey_lineage_prep.invented import peer
    from us_survey_lineage_graph import CLONE_BINDING, CLONE_POPULATION, NATIVE_BINDING

    from microcosm.build.us_runtime import graph_combined_clone as clone
    from microcosm.build.us_runtime import graph_native_household_origin as origin
    from microcosm.build.us_runtime.graph_sources import (
        frame_column_declarations,
        us_source_codecs,
    )
    from microcosm.build.us_runtime.support_provenance import support_channel_column
    from microcosm.frame import Frame, Weights
    from microcosm.graph import (
        ContentStore,
        Graph,
        KernelRegistry,
        Node,
        SourceRef,
        StructuralDelta,
        compile_graph,
        run_graph,
    )

    weights = before.weights_for("household").values.copy()
    for arm in ("acs", "asec"):
        index = before.household.index[
            before.household[support_channel_column("household")].eq(arm)
        ][0]
        weights[index] = 0
    frame = Frame(
        {e: before.table(e).copy(deep=True) for e in before.entities},
        before.schema,
        {"household": Weights(weights, before.weights_for("household").kind)},
        before.strata.copy(),
    )
    from microcosm.graph.codecs import load_frame_store

    codecs = us_source_codecs()
    codecs.register("frame-store", load_frame_store)
    store = ContentStore(root / "zero-store", codecs=codecs)
    key = hashlib.sha256(b"invented-downstream-zero-weights").hexdigest()
    population_path = store.put_frame(key, frame)
    columns = frame_column_declarations(frame)
    create = peer(
        Path(__file__).resolve().parents[2], "test_us_graph_combined_clone"
    )._Create
    nodes = (
        Node(
            "zero_native",
            create.ref,
            sources=("population",),
            structural=StructuralDelta.CREATE,
            outputs=columns,
        ),
        *origin.native_origin_source_nodes(population="zero_native"),
        origin.native_origin_binding_node(
            columns, population="zero_native", node_id=NATIVE_BINDING
        ),
        *clone.us_combined_survey_clone_nodes(
            columns, base="zero_native", source_channels=("acs", "asec")
        ),
        origin.native_origin_binding_node(
            columns,
            population=CLONE_POPULATION,
            node_id=CLONE_BINDING,
            parent_binding_node=NATIVE_BINDING,
        ),
    )
    graph = Graph(
        country="us",
        sources=(SourceRef("population", "frame-store"), *origin.SOURCES),
        nodes=nodes,
    )
    registry = KernelRegistry()
    registry.register(create())
    origin.register_native_origin_kernels(registry)
    clone.register_us_combined_survey_clone_kernels(registry)
    paths = {s.name: sources[s.name] for s in origin.SOURCES}
    paths["population"] = population_path
    manifest = run_graph(
        compile_graph(graph),
        sources=paths,
        store=store,
        kernels=registry,
        resume="forbid",
    )
    return manifest, store
