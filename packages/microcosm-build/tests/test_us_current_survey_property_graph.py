"""Invented source owners and the actual property fragment; no native/engine."""

import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_property_income_sources import actual  # noqa: F401

from microcosm.build.us_runtime import graph_current_survey_property as graph
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.frame import WeightKind
from microcosm.graph import compile_graph, run_graph
from microcosm.graph import population as population_ops

financial = graph.financial
shared = graph.shared
OPTIONS = dict(scales=(1.0, 1.0, 1.0, 1.0), atol=1e-10, rtol=1e-12, n_estimators=2)


def test_explicit_frozen_options_and_complete_owned_roster():
    options = graph.PropertyIncomeOptions(**OPTIONS)
    assert graph.codec.decode_json(options.to_bytes()) == graph.codec.decode_json(
        graph.codec.encode_json(OPTIONS)
    )
    with pytest.raises(FrozenInstanceError):
        options.atol = 1
    assert (
        len(graph.owned_columns())
        == len({o.column for o in graph.owned_columns()})
        == 24
    )
    assert not set(shared.OUTPUTS) & {o.column for o in graph.owned_columns()}
    with pytest.raises(TypeError):
        graph.PropertyIncomeOptions()


@pytest.mark.parametrize(
    "key,value",
    [
        ("scales", (1.0, 1.0, 1.0)),
        ("scales", (1.0, 1.0, 1.0, 0.0)),
        ("scales", (1.0, 1.0, 1.0, float("inf"))),
        ("atol", -1.0),
        ("rtol", float("nan")),
        ("atol", 1),
        ("n_estimators", 0),
        ("n_estimators", True),
    ],
)
def test_invalid_options_refuse(key, value):
    with pytest.raises((ValueError, TypeError)):
        graph.PropertyIncomeOptions(**{**OPTIONS, key: value})


def _pins(live):
    result = {}
    for edge in graph.host.current_survey_host_edges():
        record = live.manifest.node(edge.producer)
        key = record.opaque_artifacts[edge.artifact]
        result[edge.name] = {
            "producer_key": record.key,
            "artifact_key": key,
            "payload_sha256": graph.codec.sha(live.store.load_bytes(key)),
        }
    return result


@pytest.fixture(scope="module")
def complete(actual, tmp_path_factory):  # noqa: F811
    live, qualified, _ = actual
    pins = _pins(live)
    options = graph.PropertyIncomeOptions(**OPTIONS)
    legacy_nodes = financial.current_survey_predictor_nodes(
        qualified.shared_predictors,
        live.clone_population.frame,
        host_pins=pins,
        n_estimators=2,
    )
    nodes = graph.current_survey_property_nodes(
        qualified, live.clone_population.frame, host_pins=pins, options=options
    )
    compiled = compile_graph(
        replace(
            live.compiled.graph,
            nodes=(*live.compiled.graph.nodes, *legacy_nodes, *nodes),
        )
    )
    for cls in (
        financial.CurrentSurveyPredictorProjectionKernel,
        financial.CurrentSurveyPredictorDonorFilterKernel,
        financial.CurrentSurveyPredictorDonorColumnsKernel,
        financial.CurrentSurveyPredictorAttachKernel,
    ):
        live.kernels.register(
            cls(
                live.preparation,
                live.allocated_population,
                live.clone_population,
                host_pins=pins,
                n_estimators=2,
            )
        )
    live.kernels.register(LegacyQRFTrainKernel())
    live.kernels.register(LegacyQRFApplyMatrixKernel())
    graph.register_property_kernels(
        live.kernels,
        live.preparation,
        live.allocated_population,
        live.clone_population,
        host_pins=pins,
        options=options,
    )
    observed = {}
    manifest = run_graph(
        compiled,
        sources=live.sources,
        store=live.store,
        kernels=live.kernels,
        _population_observer=lambda node_id, p: observed.__setitem__(node_id, p),
    )
    loaded = {
        (node_id, name): live.store.load_bytes(key)
        for node_id, record in manifest.nodes.items()
        for name, key in record.opaque_artifacts.items()
    }
    results = graph.reconstruct_property_results(
        qualified,
        live.clone_population.frame,
        host_pins=pins,
        options=options,
        artifacts=loaded,
        legacy_matrix_producer_key=manifest.node(financial.PROJECTION_NODE).key,
    )
    again_observed = {}
    again = run_graph(
        compiled,
        sources=live.sources,
        store=live.store,
        kernels=live.kernels,
        resume="require",
        _population_observer=lambda node_id, p: again_observed.__setitem__(node_id, p),
    )
    assert again.key == manifest.key and all(n.hit for n in again.nodes.values())
    metadata = {
        "scope": "invented full source qualification plus actual legacy10/property16 graph; no native/country engine",
        "nodes": len(compiled.order),
        "property_nodes": len(nodes),
        "cold_property_hits": {n.id: manifest.node(n.id).hit for n in nodes},
        "required_hits": len(again.nodes),
        "source_people": len(qualified.origins),
        "clone_people": len(live.clone_population.frame.person),
        "donor_people": len(qualified.donor_columns),
        "recipient_people": len(qualified.recipient_columns),
    }
    (
        tmp_path_factory.mktemp("property-fragment-metadata") / "acceptance.json"
    ).write_text(json.dumps(metadata, indent=2))
    return (
        live,
        qualified,
        pins,
        options,
        compiled,
        manifest,
        observed,
        loaded,
        results,
        again_observed,
    )


def test_sixteen_actual_nodes_replay_and_reconstructed_populations(complete):
    (
        live,
        qualified,
        pins,
        options,
        compiled,
        manifest,
        observed,
        loaded,
        results,
        again,
    ) = complete
    assert len(results) == 8
    assert sum(n.id.startswith("survey_property.") for n in compiled.graph.nodes) == 16
    assert financial.ATTACH_NODE in compiled.predecessors[graph.ATTACH_NODE]
    assert set(shared.OUTPUTS).issubset(
        next(
            s.columns
            for s in compiled.graph.node(graph.ATTACH_NODE).inputs
            if s.entity == "person"
        )
    )
    expected = {}
    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        if node_id in results:
            parent = (
                expected[node.base]
                if node.structural is graph.StructuralDelta.FILTER
                else expected.get(compiled.versions[node_id])
            )
            if parent is None:
                # The first property operation in the pre-existing donor or
                # clone version receives the actual independently checked prefix.
                parent = (
                    observed[financial.ATTACH_NODE]
                    if node_id == graph.ATTACH_NODE
                    else observed[financial.DONOR_NODE]
                )
            result = results[node_id]
            if node.structural is graph.StructuralDelta.FILTER:
                mask = result.keep.reindex(parent.frame.person.person_id).to_numpy(
                    dtype=bool
                )
                result = replace(result, frame=parent.frame.select(mask), keep=None)
            current = population_ops.patch(parent, node, result)
            graph.replay.same_replayed_population(current, observed[node_id])
            expected[compiled.versions[node_id]] = current
        elif node_id.startswith(graph.PREFIX + "."):
            assert node_id in {
                f"{graph.PREFIX}.{kind}.{i:03d}"
                for kind in ("fit", "apply")
                for i in range(4)
            }
        else:
            expected[compiled.versions[node_id]] = observed[node_id]
        graph.replay.same_replayed_population(observed[node_id], again[node_id])
    final = observed[graph.ATTACH_NODE]
    receipt = graph.verify_materialized_property_income(
        live.preparation,
        live.allocated_population,
        live.clone_population,
        legacy_population=observed[financial.ATTACH_NODE],
        population=final,
        host_pins=pins,
        options=options,
        artifacts=loaded,
        legacy_matrix_producer_key=manifest.node(financial.PROJECTION_NODE).key,
    )
    assert (
        receipt["tax_split_rebased"] is False
        and receipt["capital_gains"] == graph.CAP_LIMITATION
    )


def test_exact_clone_pairs_unknowns_and_asec_qualified_values(complete):
    live, qualified, _, _, _, _, observed, *_ = complete
    final = observed[graph.ATTACH_NODE].frame
    legacy = observed[financial.ATTACH_NODE].frame
    for entity in legacy.entities:
        pd.testing.assert_frame_equal(
            final.table(entity)[list(legacy.table(entity))], legacy.table(entity)
        )
    assert final.metadata == legacy.metadata and final.mass_log == legacy.mass_log
    for entity in legacy.weighted_entities:
        assert final.weights_for(entity).kind is legacy.weights_for(entity).kind
        np.testing.assert_array_equal(
            final.weights_for(entity).values, legacy.weights_for(entity).values
        )
    ids = final.person[shared.provenance.support_source_id_column("person")].to_numpy()
    columns = [o.column for o in graph.owned_columns()]
    for origin in qualified.origins.index:
        pair = final.person.loc[ids == origin, columns].reset_index(drop=True)
        pd.testing.assert_series_equal(pair.iloc[0], pair.iloc[1], check_names=False)
    basis = qualified.donor_basis.person
    for origin in basis.index:
        pair = final.person.loc[ids == origin]
        for name in (*graph.PROPERTY_COMPONENTS, graph.PROPERTY_REPORTED_TOTAL):
            np.testing.assert_array_equal(
                pair[name].to_numpy(), np.repeat(basis.loc[origin, name], 2)
            )
        assert pair[list(graph.PROPERTY_DRAW_COLUMNS)].isna().all().all()
    excluded = qualified.recipient_diagnostics.index[
        ~qualified.recipient_diagnostics.eligible_recipient
    ]
    mask = np.isin(ids, excluded)
    assert mask.sum() > 0
    assert (
        final.person.loc[
            mask, [*graph.PROPERTY_COMPONENTS, graph.PROPERTY_REPORTED_TOTAL]
        ]
        .isna()
        .all()
        .all()
    )
    assert (
        not final.person.loc[
            mask, ["property_anchor_known", "property_components_known"]
        ]
        .to_numpy()
        .any()
    )
    assert (
        observed[graph.RECIPIENT_NODE].frame.person.person_id.to_numpy()
        == qualified.recipient_columns.index.to_numpy()
    ).all()
    assert (
        observed[graph.DONOR_NODE].frame.resolve_weights("person").kind
        is WeightKind.DESIGN
    )
    assert (
        observed[graph.DONOR_NODE].frame.person.person_id.to_numpy()
        == qualified.donor_columns.index.to_numpy()
    ).all()


@pytest.mark.parametrize(
    "which", ["raw", "state", "summary", "reconciliation", "projection"]
)
def test_tampered_graph_artifacts_refuse(complete, which):
    live, q, pins, options, _, manifest, _, artifacts, *_ = complete
    copied = dict(artifacts)
    key = {
        "raw": (graph.PREFIX + ".apply.001", "raw_draw"),
        "state": (graph.PREFIX + ".apply.001", "apply_state"),
        "summary": (graph.PREFIX + ".draws", "summary"),
        "reconciliation": (graph.PREFIX + ".reconcile", "summary"),
        "projection": (graph.PROJECTION_NODE, "projection"),
    }[which]
    if which == "raw":
        raw = graph.codec.read_raw_target(
            copied[key],
            target=graph.PROPERTY_COMPONENTS[1],
            index=q.recipient_frame.person.index,
        ).copy()
        raw[0] += 1
        copied[key] = graph.codec.encode_raw_target(
            raw,
            target=graph.PROPERTY_COMPONENTS[1],
            index=q.recipient_frame.person.index,
        )
    else:
        data = graph.codec.decode_json(copied[key])
        data["unexpected"] = True
        copied[key] = graph.codec.encode_json(data)
    with pytest.raises((ValueError, KeyError)):
        graph.reconstruct_property_results(
            q,
            live.clone_population.frame,
            host_pins=pins,
            options=options,
            artifacts=copied,
            legacy_matrix_producer_key=manifest.node(financial.PROJECTION_NODE).key,
        )


def test_scales_change_only_reconciliation_and_attachment_declarations(complete):
    live, q, pins, options, *_ = complete
    old = graph.current_survey_property_nodes(
        q, live.clone_population.frame, host_pins=pins, options=options
    )
    new = graph.current_survey_property_nodes(
        q,
        live.clone_population.frame,
        host_pins=pins,
        options=replace(options, scales=(2.0, 1.0, 1.0, 1.0)),
    )
    assert [a.id for a, b in zip(old, new, strict=True) if a != b] == [
        graph.PREFIX + ".reconcile",
        graph.ATTACH_NODE,
    ]


def test_origin_identity_refusal_and_order_independence(complete):
    live, q, pins, options, _, manifest, observed, artifacts, *_ = complete
    ids, axis = graph._clone_lookup(q, live.clone_population.frame)
    reordered = replace(q, origins=q.origins.iloc[::-1])
    assert np.array_equal(
        graph._clone_lookup(reordered, live.clone_population.frame)[0], ids
    )
    bad = q.origins.copy(deep=True)
    bad.iloc[0, bad.columns.get_loc("native_person_id")] += 1
    with pytest.raises(ValueError, match="CLONE_ORIGIN_IDENTITY"):
        graph._clone_lookup(replace(q, origins=bad), live.clone_population.frame)
    assert axis.dtype == np.dtype("int64")


@pytest.mark.parametrize("target", ["donor_frame", "recipient_frame"])
def test_empty_branch_refuses_without_model_defaults(complete, target):
    live, q, pins, options, *_ = complete
    with pytest.raises(ValueError, match="EMPTY_MODEL_BRANCH"):
        graph.current_survey_property_nodes(
            replace(q, **{target: None}),
            live.clone_population.frame,
            host_pins=pins,
            options=options,
        )


def test_declaration_without_source_execution(monkeypatch):
    from microcosm.frame import US_SCHEMA, Frame, Weights

    tables = {
        e: pd.DataFrame(
            {
                US_SCHEMA.entity_id_column(e): np.array([1], dtype="int64"),
                "fixture_" + e: np.array([1.0]),
            }
        )
        for e in US_SCHEMA.entities
    }
    for e in US_SCHEMA.group_entities:
        tables["person"][US_SCHEMA.membership_column(e)] = np.array([1], dtype="int64")
    tables["person"]["employment_income"] = 5.0
    frame = Frame(
        tables, US_SCHEMA, {"household": Weights(np.array([1.0]), WeightKind.DESIGN)}
    )
    features = (*shared.FEATURES, graph.PROPERTY_REPORTED_TOTAL)
    q = SimpleNamespace(
        source_frame=frame,
        donor_frame=frame,
        recipient_frame=frame,
        donor_columns=pd.DataFrame(columns=(*features, *graph.PROPERTY_COMPONENTS)),
        recipient_columns=pd.DataFrame(columns=features),
        shared_predictors=SimpleNamespace(
            donor_frame=frame,
            demographic_conditioning=False,
            geography_config_payload=None,
        ),
    )
    # Isolate declaration shape only. No fixture object claims source authority;
    # the full graph fixture below uses the actual source qualifiers unchanged.
    monkeypatch.setattr(graph, "_params", lambda *args: {"protocol": graph.PROTOCOL})
    nodes = graph.current_survey_property_nodes(
        q, frame, host_pins={}, options=graph.PropertyIncomeOptions(**OPTIONS)
    )
    assert len(nodes) == 16
    assert set(nodes[-1].params) == {"protocol", *OPTIONS}

    context_tables = {}
    for entity in frame.entities:
        structural = [US_SCHEMA.entity_id_column(entity)]
        if entity == "person":
            structural.extend(
                US_SCHEMA.membership_column(e) for e in US_SCHEMA.group_entities
            )
        context_tables[entity] = frame.table(entity)[
            structural + [c for c in frame.table(entity) if c not in structural]
        ].copy()
    for name in shared.OUTPUTS:
        context_tables["person"][name] = 0.0
    context = graph.KernelContext(
        node=nodes[-1],
        tables=context_tables,
        weights={e: frame.resolve_weights(e) for e in frame.entities},
        strata=frame.strata,
        params=nodes[-1].params,
        rng=None,
    )
    graph._clone_context(context, frame)
    assert set(shared.OUTPUTS).issubset(
        next(x.columns for x in nodes[-1].inputs if x.entity == "person")
    )


@pytest.mark.parametrize("field", ["property_reported_total", shared.OUTPUTS[0]])
def test_final_verifier_refuses_changed_owned_or_retained_values(complete, field):
    from test_us_current_survey_puf_transfer import _copy_population

    live, _, pins, options, _, manifest, observed, loaded, *_ = complete
    changed = _copy_population(observed[graph.ATTACH_NODE])
    table = changed.frame.person
    row = table.index[np.flatnonzero(np.isfinite(table[field].to_numpy()))[0]]
    table.loc[row, field] += 1
    with pytest.raises((ValueError, AssertionError)):
        graph.verify_materialized_property_income(
            live.preparation,
            live.allocated_population,
            live.clone_population,
            legacy_population=observed[financial.ATTACH_NODE],
            population=changed,
            host_pins=pins,
            options=options,
            artifacts=loaded,
            legacy_matrix_producer_key=manifest.node(financial.PROJECTION_NODE).key,
        )
