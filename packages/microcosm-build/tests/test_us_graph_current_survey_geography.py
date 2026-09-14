"""Observed-geography graph contracts over actual issuers and invented originals."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments

from microcosm.build.us_runtime import current_survey_geography as qualifier
from microcosm.build.us_runtime import graph_current_survey_geography as graph
from microcosm.build.us_runtime import graph_survey_population as population
from microcosm.build.us_runtime import survey_population_preparation as source
from microcosm.graph import ArtifactType, StructuralDelta, compile_graph, run_graph


def _executed(tmp_path, monkeypatch, *, unknown=False):
    """Construct privately pinned invented originals for a complete graph run."""
    arguments = _demographic_arguments(tmp_path, monkeypatch, unknown=unknown)
    return _execute(arguments, store_root=tmp_path / "store")


def _execute(arguments, *, store_root):
    """Keep real preparation/artifact owners and capture a real kernel context."""
    prefix = population.run_authenticated_survey_population(
        **arguments,
        store_root=store_root,
        clones=False,
        return_values=True,
    )
    qualified = qualifier.qualify_current_survey_geography(prefix.preparation)
    node = graph.current_survey_geography_node(
        preparation_sha256=population._sha(prefix.preparation.payload),
        projection_receipt_sha256=population._sha(qualified.receipt),
        population=population.ALLOCATION_NODE,
    )
    compiled = compile_graph(
        replace(prefix.compiled.graph, nodes=(*prefix.compiled.graph.nodes, node))
    )
    kernel = graph.CurrentSurveyGeographyKernel(prefix.preparation)
    prefix.kernels.register(kernel)
    contexts = []

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is kernel.run.__code__:
            contexts.append(frame.f_locals["context"])

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        manifest = run_graph(
            compiled,
            sources=prefix.sources,
            store=prefix.store,
            kernels=prefix.kernels,
        )
    finally:
        sys.setprofile(previous)
    assert len(contexts) == 1
    return SimpleNamespace(
        arguments=arguments,
        prefix=prefix,
        qualified=qualified,
        node=node,
        compiled=compiled,
        kernel=kernel,
        context=contexts[0],
        manifest=manifest,
    )


@pytest.fixture(scope="module")
def known_run(tmp_path_factory):
    """Share one real execution while its invented registry pins remain active."""
    with pytest.MonkeyPatch.context() as patch:
        run = _executed(tmp_path_factory.mktemp("survey-geography-known"), patch)
        yield run
        # All consumers of this fixture must leave its issued source untouched.
        source.verify_survey_population_preparation(run.prefix.preparation)


@pytest.fixture
def fresh_source_run(known_run, tmp_path):
    """Reissue independent mutable Frames over the shared immutable originals."""
    run = _execute(known_run.arguments, store_root=tmp_path / "fresh-source-store")
    assert run.prefix.preparation is not known_run.prefix.preparation
    assert run.prefix.preparation.frame is not known_run.prefix.preparation.frame
    yield run
    source.verify_survey_population_preparation(known_run.prefix.preparation)


def _detached_context(context):
    """Each negative gets independent tables, weights and container mappings."""
    return replace(
        context,
        tables={name: table.copy(deep=True) for name, table in context.tables.items()},
        weights={
            name: replace(value, values=value.values.copy())
            for name, value in context.weights.items()
        },
        strata=context.strata.copy(deep=True),
        params=dict(context.params),
        sources=dict(context.sources),
        artifacts=dict(context.artifacts),
        tolerances=dict(context.tolerances),
        numerics=dict(context.numerics),
    )


@pytest.mark.parametrize("unknown", (False, True))
def test_real_projection_graph_execution_and_replay_preserve_inputs(
    tmp_path, monkeypatch, unknown
):
    run = _executed(tmp_path, monkeypatch, unknown=unknown)
    assert run.compiled.order == (
        population.CREATE_NODE,
        population.ALLOCATION_NODE,
        graph.NODE,
    )
    assert run.node.structural is StructuralDelta.NONE
    assert run.node.sources == run.node.artifact_outputs == ()
    assert run.node.weights is None
    assert tuple(o.column for o in run.node.outputs) == qualifier.COLUMNS
    assert all(
        o.entity == "household" and o.dtype == "string" and not o.rewrite
        for o in run.node.outputs
    )
    assert run.node.artifact_inputs[0].producer == population.CREATE_NODE
    assert run.node.artifact_inputs[0].type == population.PREPARATION_TYPE
    assert not run.manifest.node(graph.NODE).hit
    original = run.prefix.allocated_population.frame
    version = run.compiled.versions[graph.NODE]
    projected = run.manifest.population(version)
    for entity in original.entities:
        pd.testing.assert_frame_equal(
            projected.table(entity).loc[:, original.table(entity).columns],
            original.table(entity),
            check_exact=True,
        )
    assert (
        projected.weights_for("household").kind
        is original.weights_for("household").kind
    )
    assert (
        projected.weights_for("household").values.tobytes()
        == original.weights_for("household").values.tobytes()
    )
    assert (
        run.manifest.mass_ledger(version) == run.prefix.allocated_population.mass_ledger
    )
    table = projected.table("household").set_index("household_id")
    actual = table.loc[:, list(graph.OUTPUT_COLUMNS)]
    pd.testing.assert_frame_equal(actual, run.qualified.household, check_exact=True)
    assert len(actual) == 6
    assert actual.survey_observed_state.isna().sum() == int(unknown)
    assert actual.survey_observed_puma.notna().sum() == 4
    unknown_asec = actual.survey_geography_origin_key.eq('["asec",2024,2025,"00008"]')
    assert unknown_asec.sum() == 1
    if unknown:
        assert actual.loc[unknown_asec, "survey_observed_state"].isna().all()
    else:
        assert actual.loc[unknown_asec, "survey_observed_state"].tolist() == ["36"]
    receipt = run.manifest.node(graph.NODE).receipt
    assert receipt["projection_receipt_sha256"] == population._sha(
        run.qualified.receipt
    )
    assert receipt["source_projection"]["households"] == 6
    assert (
        not receipt["population_admission_issued"] and not receipt["release_eligible"]
    )

    warm_calls = []

    def trace(frame, event, arg):
        if event == "call" and frame.f_code is run.kernel.run.__code__:
            warm_calls.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        warm = run_graph(
            run.compiled,
            sources=run.prefix.sources,
            store=run.prefix.store,
            kernels=run.prefix.kernels,
            resume="require",
        )
    finally:
        sys.setprofile(previous)
    assert warm_calls == []
    assert all(node.hit for node in warm.nodes.values())
    assert warm.key == run.manifest.key
    population._same_frame(projected, warm.population(version))
    # Cached graph output is not source admission. Requalify the retained
    # preparation independently, as the country runner must on every replay.
    current = qualifier.qualify_current_survey_geography(run.prefix.preparation)
    pd.testing.assert_frame_equal(current.household, actual, check_exact=True)
    assert current.receipt == run.qualified.receipt


@pytest.mark.parametrize(
    "change,reason",
    (
        ("preparation_digest", "DECLARATION"),
        ("projection_digest", "PROJECTION_RECEIPT"),
        ("artifact_payload", "ARTIFACT_BINDING"),
        ("artifact_type", "ARTIFACT_BINDING"),
        ("rewrite_declaration", "DECLARATION"),
        ("source_roster", "CONTEXT_ROSTER"),
    ),
)
def test_mismatched_declaration_or_typed_evidence_refuses(
    known_run, tmp_path, change, reason
):
    run = known_run
    context = _detached_context(run.context)
    if change in {"preparation_digest", "projection_digest"}:
        field = (
            "preparation_sha256"
            if change == "preparation_digest"
            else "projection_receipt_sha256"
        )
        node = replace(context.node, params={**context.node.params, field: "f" * 64})
        context = replace(context, node=node, params=node.params)
    elif change in {"artifact_payload", "artifact_type"}:
        artifact = context.artifacts["preparation"]
        artifact = (
            replace(artifact, payload=artifact.payload + b" ")
            if change == "artifact_payload"
            else replace(
                artifact, type=ArtifactType(population.PREPARATION_TYPE.name, 99)
            )
        )
        context = replace(context, artifacts={"preparation": artifact})
    elif change == "rewrite_declaration":
        first, *rest = context.node.outputs
        context = replace(
            context,
            node=replace(context.node, outputs=(replace(first, rewrite=True), *rest)),
        )
    else:
        context = replace(context, sources={"undeclared": tmp_path})
    with pytest.raises(ValueError, match=reason):
        run.kernel.run(context)


def test_same_household_ids_with_wrong_native_origin_refuse(known_run):
    run = known_run
    context = _detached_context(run.context)
    table = context.tables["household"]
    column = population.spine_source_id_column("household")
    table.loc[table.index[0], column] += 1
    with pytest.raises(ValueError, match="HOUSEHOLD_ORIGIN"):
        run.kernel.run(context)


def test_returned_series_are_detached_and_do_not_authorize_reuse(known_run):
    run = known_run
    before = source._frame_identity(run.prefix.preparation.frame)
    result = run.kernel.run(_detached_context(run.context))
    column = ("household", "survey_observed_state")
    result.columns[column].iloc[0] = "99"
    result.receipt["source_projection"]["households"] = 999
    fresh = run.kernel.run(_detached_context(run.context))
    assert fresh.receipt["source_projection"]["households"] == 6
    for name in graph.OUTPUT_COLUMNS:
        pd.testing.assert_series_equal(
            fresh.columns[("household", name)],
            run.qualified.household[name],
            check_exact=True,
        )
    assert source._frame_identity(run.prefix.preparation.frame) == before
    assert (
        fresh.frame
        is fresh.keep
        is fresh.expand
        is fresh.weights
        is fresh.strata
        is None
    )
    assert not fresh.artifacts


def _assert_final_mutation_refuses(run, change):
    context = _detached_context(run.context)
    state = source._ISSUED[id(run.prefix.preparation)][2]
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code
            is source.AuthenticatedSurveyPopulationPreparation._checked.__code__
            and frame.f_back.f_code is run.kernel.run.__code__
            and "result" in frame.f_back.f_locals
            and not fired
        ):
            fired.append(True)
            result = frame.f_back.f_locals["result"]
            if change == "source_frame":
                state.frame.table("household")["state_fips"] = 99
            elif change == "returned_column":
                result.columns[("household", "survey_observed_state")].iloc[0] = "99"
            else:
                result.receipt["phase"] = "changed"

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError):
            run.kernel.run(context)
    finally:
        sys.setprofile(previous)
    assert fired == [True]


@pytest.mark.parametrize("change", ("returned_column", "receipt"))
def test_mutation_at_final_owner_return_refuses(known_run, change):
    _assert_final_mutation_refuses(known_run, change)


def test_source_frame_mutation_at_final_owner_return_refuses(fresh_source_run):
    _assert_final_mutation_refuses(fresh_source_run, "source_frame")


def test_changed_qualifier_return_cannot_hide_behind_unchanged_receipt(
    known_run,
):
    run = known_run
    context = _detached_context(run.context)
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is qualifier.qualify_current_survey_geography.__code__
            and frame.f_back.f_code is run.kernel.run.__code__
            and not fired
        ):
            fired.append(True)
            assert isinstance(json.loads(arg.receipt)["projection_sha256"], str)
            arg.household.loc[arg.household.index[0], "survey_observed_state"] = "99"

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError, match="QUALIFIED_PROJECTION"):
            run.kernel.run(context)
    finally:
        sys.setprofile(previous)
    assert fired == [True]
