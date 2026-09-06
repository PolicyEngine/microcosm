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

    parent = native.PopulationOriginBinding(
        payload(NATIVE_BINDING, "binding"), _token=native._BOUND_TOKEN
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
