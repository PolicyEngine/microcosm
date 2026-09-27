"""Genuine invented source -> categorical artifact -> report graph and replay.

No country model, full financial graph, actual survey records, canonical
beneficiary columns or claimed scientific/release qualification.
"""

import hashlib
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from ss_report_source_fixture import ss_report_source_arguments

from microcosm.build.us_runtime import graph_current_survey_ss_completion as graph
from microcosm.fit import categorical
from microcosm.graph import (
    ArtifactValue,
    ContentStore,
    Graph,
    KernelContext,
    Numeric,
    NumericScope,
    SourceRef,
    compile_graph,
    platform_fingerprint,
    run_graph,
)
from microcosm.graph.keys import opaque_artifact_key

values = graph.values
CONFIG = categorical.CategoricalConfig(max_iter=3, min_samples_leaf=1)


def compiled(boundary):
    return compile_graph(
        Graph(
            "us",
            sources=(SourceRef(graph.SOURCE_NAME, "frame-store"),),
            nodes=boundary.nodes,
        )
    )


def probability_binding(manifest, store, q):
    fit = manifest.node(graph.FIT_NODE)
    metadata = graph.codec.decode_json(
        store.load_bytes(fit.opaque_artifacts["model_metadata"])
    )
    probability = manifest.node(graph.PROBABILITY_NODE)
    payload = store.load_bytes(probability.opaque_artifacts["probabilities"])
    return {
        "model_sha256": hashlib.sha256(
            store.load_bytes(fit.opaque_artifacts["model"])
        ).hexdigest(),
        "model_producer_key": fit.key,
        "training_id": metadata["training"]["training_id"],
        "donor_source_sha256": graph.codec.sha(q.donor_projection),
        "matrix_sha256": graph.codec.sha(q.matrix),
        "matrix_producer_key": manifest.node(graph.MATRIX_NODE).key,
        "source_sha256": graph.codec.sha(q.recipient_projection),
        "source_producer_key": manifest.node(graph.SOURCE_NODE).key,
        "classes": list(values.basis.COMPONENTS),
        "probability_producer_key": probability.key,
        "probability_sha256": graph.codec.sha(payload),
    }


@pytest.fixture(scope="module")
def genuine_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("ss-report-graph")
    with pytest.MonkeyPatch.context() as patch:
        arguments = ss_report_source_arguments(root, patch)
        preparation = values.source.prepare_authenticated_survey_population(**arguments)
        boundary = graph._ReportBoundary(preparation, seed=29, config=CONFIG)
        yield root, boundary


def test_genuine_seven_node_report_cold_required_and_exact_source_preservation(
    genuine_graph,
):
    root, boundary = genuine_graph
    boundary.validate()
    q = boundary.qualified
    pd.testing.assert_series_equal(
        q.source_frame.person.age, q.full_source.asec_frame.person.age
    )
    assert [(output.entity, output.column) for output in boundary.nodes[0].outputs] == [
        ("person", "age")
    ]
    seal = values.qualified_seal(q)
    before = values.source._frame_identity(boundary.entry[2].frame)
    store = ContentStore(root / "graph-store")
    refs = {
        graph.SOURCE_NAME: store.put_frame(
            graph.codec.sha(q.donor_projection), q.source_frame
        )
    }
    registry = graph.ss_report_kernel_registry(boundary)
    observed = {}
    cold = run_graph(
        compiled(boundary),
        sources=refs,
        store=store,
        kernels=registry,
        _population_observer=lambda name, population: observed.setdefault(
            name, population
        ),
    )
    assert len(cold.nodes) == 7 and not any(n.hit for n in cold.nodes.values())
    donor = observed[graph.DONOR_NODE].frame
    assert donor.person.person_id.tolist() == [105, 109, 110, 111]
    assert tuple(sorted(donor.person[values.LABEL])) == tuple(
        sorted(values.basis.COMPONENTS)
    )
    assert donor.resolve_weights("person").kind.value == "design"
    np.testing.assert_array_equal(
        donor.resolve_weights("person").values, [2552.12] * 3 + [100.0]
    )
    assert (
        105
        not in q.originals.loc[
            q.originals.source.eq("asec"), "native_person_id"
        ].tolist()
    )
    payload = store.load_bytes(cold.node(graph.REPORT_NODE).opaque_artifacts["report"])
    expected = probability_binding(cold, store, q)
    completed, probabilities = graph.read_report(
        payload, qualified=q, expected_model_binding=expected
    )
    _, requested = values._report_selection(q.originals)
    assert set(q.originals.loc[requested, "source"]) == {"asec", "acs"}
    assert len(probabilities) == requested.sum() > 1
    assert np.isfinite(probabilities.to_numpy()).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-12)
    known = q.originals.source_reporting_universe.to_numpy()
    np.testing.assert_array_equal(
        completed.loc[known].sum(axis=1),
        q.originals.loc[known, "social_security_source_total"],
    )
    assert values._bits(completed.loc[~requested].to_numpy()) == values._bits(
        q.originals.loc[~requested, list(values.basis.COMPONENTS)].to_numpy()
    )
    ambiguous_id = q.originals.index[
        q.originals.source.eq("asec") & q.originals.native_person_id.eq(107)
    ][0]
    assert probabilities.loc[ambiguous_id, "social_security_retirement"] > 0
    assert completed.loc[ambiguous_id, "social_security_retirement"] == 0
    np.testing.assert_allclose(
        completed.loc[ambiguous_id, list(values.basis.COMPONENTS[1:])].to_numpy(),
        probabilities.loc[ambiguous_id, list(values.basis.COMPONENTS[1:])].to_numpy()
        / probabilities.loc[ambiguous_id, list(values.basis.COMPONENTS[1:])].sum()
        * 3000,
        rtol=0,
        atol=1e-10,
    )
    # No graph node writes canonical beneficiary fields or changes receiving data.
    assert not any(
        output.column in values.basis.COMPONENTS
        for node in boundary.nodes
        for output in node.outputs
    )
    assert values.source._frame_identity(boundary.entry[2].frame) == before
    boundary.validate()
    # A hit-only run must not call the estimator; retain any supervising profile.
    previous = sys.getprofile()
    fits = []

    def no_fit(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "call" and frame.f_code is categorical.fit_categorical.__code__:
            fits.append(True)
            raise AssertionError("required replay entered a fit")

    try:
        sys.setprofile(no_fit)
        warm = run_graph(
            compiled(boundary),
            sources=refs,
            store=store,
            kernels=registry,
            resume="require",
        )
    finally:
        sys.setprofile(previous)
    assert not fits and len(warm.nodes) == 7 and all(n.hit for n in warm.nodes.values())
    assert cold.key == warm.key
    assert (
        cold.node(graph.REPORT_NODE).opaque_artifacts
        == warm.node(graph.REPORT_NODE).opaque_artifacts
    )
    assert (
        store.load_bytes(warm.node(graph.REPORT_NODE).opaque_artifacts["report"])
        == payload
    )
    assert values.qualified_seal(q) == seal
    boundary.validate()
    boundary.pure()


@pytest.mark.parametrize("change", ["basis", "matrix", "config"])
def test_borrowed_boundary_refuses_mutation_before_cached_replay(genuine_graph, change):
    _, boundary = genuine_graph
    q, old_config = boundary.qualified, boundary.config
    basis = q.originals.copy(deep=True)
    matrix = q.matrix
    try:
        if change == "basis":
            q.originals.iloc[
                0, q.originals.columns.get_loc("social_security_source_total")
            ] = 777
        elif change == "matrix":
            boundary.qualified = replace(q, matrix=matrix + b" ")
        else:
            boundary.config = categorical.CategoricalConfig(
                max_iter=4, min_samples_leaf=1
            )
        with pytest.raises(ValueError, match="QUALIFIED_OR_CONFIGURATION_CHANGED"):
            boundary.pure()
    finally:
        q.originals.loc[:, :] = basis
        boundary.qualified = q
        boundary.config = old_config
    boundary.pure()


@pytest.mark.parametrize("change", ["body", "binding", "trailing"])
def test_report_decoder_refuses_changed_report_or_expected_binding(
    genuine_graph, change
):
    _, boundary = genuine_graph
    q = boundary.qualified
    p = pd.DataFrame(
        0.25,
        index=q.recipient_features.index,
        columns=values.basis.COMPONENTS,
        dtype="float64",
    )
    payload = graph.encode_report(q, p, {"fixture": "numeric-codec-only"})
    expected = {"fixture": "numeric-codec-only"}
    if change == "binding":
        expected = {"fixture": "changed"}
    elif change == "trailing":
        payload += b"x"
    else:
        payload = payload[:-1] + bytes([payload[-1] ^ 1])
    with pytest.raises(ValueError):
        graph.read_report(payload, qualified=q, expected_model_binding=expected)


def test_empty_source_branch_needs_no_class_support_or_model(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        arguments = ss_report_source_arguments(tmp_path, patch, no_recipients=True)
        preparation = values.source.prepare_authenticated_survey_population(**arguments)
        boundary = graph._ReportBoundary(preparation, seed=29, config=CONFIG)
        q = boundary.qualified
        assert q.matrix is None and len(q.recipient_features) == 0
        assert q.evidence["resolved_positive_donors"] == 0
        assert set(q.evidence["class_design_weight_mass"].values()) == {0.0}
        assert [node.id for node in boundary.nodes] == [
            graph.SOURCE_NODE,
            graph.REPORT_NODE,
        ]
        previous = sys.getprofile()
        fits = []

        def no_fit(frame, event, result):
            if previous is not None:
                previous(frame, event, result)
            if event == "call" and frame.f_code is categorical.fit_categorical.__code__:
                fits.append(True)
                raise AssertionError("empty branch entered a fit")

        boundary.validate()
        store = ContentStore(tmp_path / "empty-store")
        refs = {
            graph.SOURCE_NAME: store.put_frame(
                graph.codec.sha(q.donor_projection), q.source_frame
            )
        }
        try:
            sys.setprofile(no_fit)
            cold = run_graph(
                compiled(boundary),
                sources=refs,
                store=store,
                kernels=graph.ss_report_kernel_registry(boundary),
            )
            warm = run_graph(
                compiled(boundary),
                sources=refs,
                store=store,
                kernels=graph.ss_report_kernel_registry(boundary),
                resume="require",
            )
        finally:
            sys.setprofile(previous)
        assert not fits and len(cold.nodes) == len(warm.nodes) == 2
        assert not any(n.hit for n in cold.nodes.values()) and all(
            n.hit for n in warm.nodes.values()
        )
        assert cold.key == warm.key
        payload = store.load_bytes(
            cold.node(graph.REPORT_NODE).opaque_artifacts["report"]
        )
        assert payload == store.load_bytes(
            warm.node(graph.REPORT_NODE).opaque_artifacts["report"]
        )
        completed, probabilities = graph.read_report(
            payload, qualified=q, expected_model_binding={}
        )
        assert probabilities.empty
        assert values._bits(completed.to_numpy()) == values._bits(
            q.originals.loc[:, list(values.basis.COMPONENTS)].to_numpy()
        )
        assert q.originals.social_security_source_total.isna().any()
        assert q.originals.social_security_source_total.eq(0).any()
        boundary.validate()
        boundary.pure()


@pytest.mark.parametrize("mutate", [False, True])
def test_late_boundary_callback_cannot_change_detached_kernel_columns(
    genuine_graph, mutate
):
    _, boundary = genuine_graph
    q = boundary.qualified
    context = columns_context(boundary)
    previous = sys.getprofile()
    visited = []

    def callback(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        caller = frame.f_back
        if (
            event == "return"
            and frame.f_code is graph._ReportBoundary.pure.__code__
            and caller is not None
            and caller.f_code is graph._Kernel.run.__code__
            and "result" in caller.f_locals
            and not visited
        ):
            visited.append(True)
            if mutate:
                caller.f_locals["result"].columns[("person", values.FEATURES[0])].iloc[
                    0
                ] += 1

    try:
        sys.setprofile(callback)
        if mutate:
            with pytest.raises(ValueError, match="FINAL_RESULT_CHANGED"):
                graph.ColumnsKernel(boundary).run(context)
        else:
            result = graph.ColumnsKernel(boundary).run(context)
            pd.testing.assert_series_equal(
                result.columns[("person", values.FEATURES[0])],
                q.donor_columns[values.FEATURES[0]],
            )
    finally:
        sys.setprofile(previous)
    assert visited == [True]
    boundary.pure()


def columns_context(boundary):
    node = next(n for n in boundary.nodes if n.id == graph.COLUMNS_NODE)
    q = boundary.qualified
    # Direct kernel refusal probe. The real graph/replay test separately binds
    # actual producer keys; this typed context confers no source authority.
    producer = "a" * 64
    return KernelContext(
        node,
        {"person": q.source_frame.person[["person_id", "age"]].copy(deep=True)},
        {},
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(0),
        artifacts={
            "donor_projection": ArtifactValue(
                q.donor_projection,
                graph.PROJECTION_TYPE,
                opaque_artifact_key(producer, "donor_projection"),
                producer,
                NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
            )
        },
    )


@pytest.mark.parametrize("defect", ["value", "dtype", "absent"])
def test_model_columns_refuse_changed_declared_source_age(genuine_graph, defect):
    _, boundary = genuine_graph
    context = columns_context(boundary)
    person = context.tables["person"].copy(deep=True)
    if defect == "value":
        person.loc[person.index[0], "age"] += 1
    elif defect == "dtype":
        person["age"] = person.age.astype(
            "float32" if person.age.dtype != np.dtype("float32") else "float64"
        )
    else:
        person = person.drop(columns="age")
    with pytest.raises(ValueError, match="CONTEXT_SOURCE_AGE"):
        graph.ColumnsKernel(boundary).run(replace(context, tables={"person": person}))
    boundary.pure()
