"""Invented host-only bridge; ordinary pytest, no donor/fit/engine/target access.

The actual source test deliberately retains normal readiness and inventory gates.
Its proposal has not been executed. Fixture private file pins follow the existing
invented source helpers; no source/issuer function or money.ready is replaced.
"""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_survey_population import authenticated_arguments
from test_us_puf_detail_transfer import host as legacy_host
from test_us_survey_population_preparation import fixture as source_fixture

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import graph_combined_clone as clone
from microcosm.build.us_runtime import graph_puf_diagnostic_consumer as bridge
from microcosm.build.us_runtime import graph_survey_population as survey
from microcosm.build.us_runtime import puf_detail_transfer as detail
from microcosm.build.us_runtime import puf_diagnostic_consumer as consumer
from microcosm.build.us_runtime import support_provenance as provenance
from microcosm.build.us_runtime import survey_population_preparation as owner
from microcosm.build.us_runtime import survey_population_replay as replay
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit.model_input import decode_recipient_matrix
from microcosm.frame import Frame, Weights
from microcosm.graph import NodeRejected, compile_graph, executor, run_graph
from microcosm.graph.keys import opaque_artifact_key


def _copy_frame(frame):
    return Frame(
        {e: frame.table(e).copy(deep=True) for e in frame.entities},
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata.copy(deep=True),
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )


def _legacy_clone():
    return clone.clone_us_frame_for_puf_support(legacy_host())


def test_existing_matrix_default_and_explicit_projection_have_identical_bytes():
    frame = _legacy_clone()
    old, old_mask = detail.recipient_matrix(frame)
    people = frame.person
    values = [
        row["asec_reported_wage_income_2024_price"]
        if row[provenance.support_channel_column("person")] == "asec"
        else row["employment_income_before_lsr"]
        for _, row in people.iterrows()
    ]
    wages = pd.Series(
        values, index=pd.Index(people.person_id.to_numpy()), dtype="float64"
    )
    new, new_mask = detail.recipient_matrix(frame, person_wages=wages)
    assert new == old
    np.testing.assert_array_equal(new_mask, old_mask)
    # Independent hand expectations: actual JOINT spouse included, dependent
    # wages 999/800 excluded, zero-weight single unit retained.
    assert sorted(
        decode_recipient_matrix(new).features.reported_wage_proxy.tolist()
    ) == [10.0, 20.0, 30.0, 170.0]
    with pytest.raises(ValueError, match="HOST_WAGE_ROW_AXIS"):
        detail.recipient_matrix(frame, person_wages=wages.iloc[::-1])
    wrong = wages.copy()
    role = people[provenance.support_clone_index_column("person")].to_numpy()
    wrong.iloc[np.flatnonzero(role == 1)[0]] += 1
    with pytest.raises(ValueError, match="HOST_WAGE_PAIR_BYTES"):
        detail.recipient_matrix(frame, person_wages=wrong)
    missing = wages.copy()
    first = people.iloc[0][provenance.support_source_id_column("person")]
    pair = people[provenance.support_source_id_column("person")].eq(first).to_numpy()
    missing.iloc[np.flatnonzero(pair)] = np.nan
    with pytest.raises(ValueError, match="HOST_MISSING_FEATURE"):
        detail.recipient_matrix(frame, person_wages=missing)
    with pytest.raises(ValueError, match="HOST_WAGE_ROW_AXIS"):
        detail.recipient_matrix(frame, person_wages=wages.astype("object"))


def test_current_route_is_explicit_and_legacy_declarations_are_separate():
    columns = frame_column_declarations(_legacy_clone())
    current = bridge.current_survey_host_nodes(
        columns,
        preparation_sha256="1" * 64,
        allocation_sha256="2" * 64,
        clone_sha256="3" * 64,
        host_pins={
            e.name: {
                "producer_key": "4" * 64,
                "artifact_key": "5" * 64,
                "payload_sha256": "6" * 64,
            }
            for e in bridge.current_survey_host_edges()
        },
    )
    assert len(current) == 2
    assert all(not n.outputs and not n.sources for n in current)
    assert [e.name for e in current[0].artifact_inputs] == [
        "preparation",
        "allocation",
        "frame_context",
    ]
    assert [e.name for e in current[1].artifact_inputs] == ["projection"]
    assert current[0].params["host_route"] == consumer.CURRENT_SURVEY_ROUTE
    assert len(bridge.host_edges(native_binding="native", clone_binding="clones")) == 11
    assert not any(
        word in n.kernel
        for n in current
        for word in ("donor", "fit", "attach", "apply")
    )


def test_unissued_byte_equal_preparation_refuses_before_projection():
    fake = object.__new__(owner.AuthenticatedSurveyPopulationPreparation)
    object.__setattr__(fake, "payload", b"{}")
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="UNISSUED_OR_CHANGED"
    ):
        bridge.qualify_current_survey_host(fake, None, None)


@pytest.mark.parametrize("year,field", [(2024, "ANN_VAL"), (2022, "WSAL_VAL")])
def test_current_host_preserves_full_parent_readiness(
    tmp_path, monkeypatch, year, field
):
    arguments = source_fixture(tmp_path, monkeypatch, missing_asec_money=(year, field))
    live = survey.run_authenticated_survey_population(
        **arguments,
        store_root=tmp_path / "graph-store",
        clones=True,
        return_values=True,
    )
    # Reaching this point establishes actual preparation and native issuance;
    # an earlier SOURCE_CHANGED error is not the readiness result under test.
    preparation_entry = owner._ISSUED[id(live.preparation)]
    native = preparation_entry[2].native[1]
    parent = owner.asec_native._ISSUED[id(native)][2].parent
    decoded = money.classify_asec_money(parent.views(), parent.spec, parent.scope)
    years = np.asarray(parent.scope.person_years)
    absent = ~decoded.field(field).validity.astype(bool)
    assert int(absent.sum()) == 1
    assert years[absent].tolist() == [year]
    assert decoded.field("WSAL_VAL").validity[years == 2024].all()
    with pytest.raises(money.MoneyRefusalError) as caught:
        bridge.qualify_current_survey_host(
            live.preparation, live.allocated_population, live.clone_population
        )
    assert caught.value.reason == "MISSING_REQUIRED_AMOUNT"
    assert caught.value.field == field


def test_actual_current_sources_cold_and_replayed_host(tmp_path, monkeypatch):
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    live = survey.run_authenticated_survey_population(
        **arguments, clones=True, return_values=True
    )
    preparation, allocated, expanded = (
        live.preparation,
        live.allocated_population,
        live.clone_population,
    )
    before = owner._frame_identity(expanded.frame)
    projection, matrix, mask, evidence = bridge.qualify_current_survey_host(
        preparation, allocated, expanded
    )
    document = codec.decode_json(projection)
    assert evidence["release_eligible"] is False
    assert evidence["source_period_equivalence_claim"] is False
    assert len(document["asec"]["rows"]) == 4
    assert [r[2] for r in document["asec"]["rows"]] == [2024] * 4
    assert [
        np.frombuffer(bytes.fromhex(r[3]), dtype="<f8")[0]
        for r in document["asec"]["rows"]
    ] == [60000.0, 4800.0, 60000.0, 4800.0]
    assert all(r[4:] == [1, 1, 0] for r in document["asec"]["rows"])
    selected = expanded.frame.table("tax_unit").loc[mask]
    decoded = decode_recipient_matrix(matrix)
    asec_rows = (
        selected[provenance.support_channel_column("tax_unit")].eq("asec").to_numpy()
    )
    assert decoded.features.reported_wage_proxy.to_numpy()[asec_rows].tolist() == [
        60000.0,
        60000.0,
    ]
    assert len(decoded.features) == int(mask.sum())
    assert np.any(expanded.frame.resolve_weights("tax_unit").values[mask] == 0)
    assert owner._frame_identity(expanded.frame) == before

    # These mutations hit the receiving decoder after a successful real owner
    # projection. They cannot pass because a substituted issuer rejected earlier.
    for column, value, reason in (
        (1, 999999, "SURVEY_WAGE_ORIGIN"),
        (2, 2023, "SURVEY_WAGE_ROW"),
        (5, 0, "SURVEY_WAGE_ROW"),
        (4, 0, "INVALID_SLOT"),
        (6, 1, "ZERO_ORIGIN_ENCODING"),
    ):
        changed = copy.deepcopy(document)
        changed["asec"]["rows"][0][column] = value
        with pytest.raises(ValueError, match=reason):
            consumer.current_survey_recipient_matrix(
                expanded.frame, codec.encode_json(changed)
            )
    changed = copy.deepcopy(document)
    changed["asec"]["rows"] = changed["asec"]["rows"][:-1]
    with pytest.raises(ValueError, match="SURVEY_WAGE_ROSTER"):
        consumer.current_survey_recipient_matrix(
            expanded.frame, codec.encode_json(changed)
        )
    altered_frame = _copy_frame(expanded.frame)
    people = altered_frame.person
    index = np.flatnonzero(
        people[provenance.support_clone_index_column("person")].eq(1).to_numpy()
    )[0]
    people.iloc[index, people.columns.get_loc("age")] += 1
    with pytest.raises(ValueError, match="HOST_COPIED_SOURCE_CELL"):
        consumer.current_survey_recipient_matrix(altered_frame, projection)
    with pytest.raises(ValueError):
        bridge.qualify_current_survey_host(preparation, expanded, allocated)
    # A real byte-identical but unissued preparation remains inadmissible.
    fake = object.__new__(owner.AuthenticatedSurveyPopulationPreparation)
    object.__setattr__(fake, "payload", preparation.payload)
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="UNISSUED_OR_CHANGED"
    ):
        bridge.qualify_current_survey_host(fake, allocated, expanded)

    pins = {}
    for edge in bridge.current_survey_host_edges():
        parent = live.manifest.node(edge.producer)
        key = parent.opaque_artifacts[edge.artifact]
        pins[edge.name] = {
            "producer_key": parent.key,
            "artifact_key": key,
            "payload_sha256": codec.sha(live.store.load_bytes(key)),
        }
    nodes = bridge.current_survey_host_nodes(
        frame_column_declarations(expanded.frame),
        host_pins=pins,
        **{
            k: document[k]
            for k in ("preparation_sha256", "allocation_sha256", "clone_sha256")
        },
    )
    # Use the actual executor projection over this same issued population.
    # No additional preparation/graph run or replacement issuer is needed.
    context = executor._project_context(
        nodes[0],
        expanded,
        key="a" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    assert any(
        not expanded.frame.table(entity).columns.identical(table.columns)
        for entity, table in context.tables.items()
    )  # Establish the original representation mismatch, not just a helper call.
    bridge._current_context_frame(context, expanded.frame)

    first = nodes[0].inputs[0]
    assert len(first.columns) > 1
    for changed_slice in (
        replace(first, columns=first.columns[:-1]),
        replace(first, columns=first.columns[::-1]),
        replace(first, columns=(*first.columns, "__extra_input__")),
        # A declared column lets Node construction succeed; this direct
        # contract check must reject non-ALL rows before any mask evaluation.
        replace(first, rows=first.columns[0]),
    ):
        bad_node = replace(nodes[0], inputs=(changed_slice, *nodes[0].inputs[1:]))
        with pytest.raises(ValueError, match="^SURVEY_HOST_INPUTS$"):
            bridge._current_context_frame(
                replace(context, node=bad_node), expanded.frame
            )
    for field, values, reason in (
        ("tables", context.tables, "SURVEY_HOST_TABLES"),
        ("weights", context.weights, "SURVEY_HOST_WEIGHTS"),
    ):
        missing = dict(values)
        del missing[next(iter(missing))]
        extra = {**values, "__extra_entity__": next(iter(values.values()))}
        for changed_values in (missing, extra):
            with pytest.raises(ValueError, match="^" + reason + "$"):
                bridge._current_context_frame(
                    replace(context, **{field: changed_values}), expanded.frame
                )

    person = context.tables[expanded.frame.schema.person_entity]
    changed_tables = dict(context.tables)
    changed_tables[expanded.frame.schema.person_entity] = person.iloc[:, ::-1]
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_REPLAY_AXIS$"):
        bridge._current_context_frame(
            replace(context, tables=changed_tables), expanded.frame
        )
    changed_tables[expanded.frame.schema.person_entity] = person.rename_axis(
        "__changed_columns__", axis="columns"
    )
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_REPLAY_AXIS$"):
        bridge._current_context_frame(
            replace(context, tables=changed_tables), expanded.frame
        )
    changed_person = person.copy(deep=True)
    changed_person.iloc[0, changed_person.columns.get_loc("age")] += 1
    changed_tables[expanded.frame.schema.person_entity] = changed_person
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_REPLAY_NATIVE_BITS$"):
        bridge._current_context_frame(
            replace(context, tables=changed_tables), expanded.frame
        )

    # Positive and negative zero are both valid weights; only their exact bits
    # differ. This must reach the replay comparison rather than Frame refusal.
    entity = next(e for e, w in context.weights.items() if np.any(w.values == 0))
    value = context.weights[entity]
    changed_values = value.values.copy()
    zero = int(np.flatnonzero(changed_values == 0)[0])
    changed_values[zero] = 0.0 if np.signbit(changed_values[zero]) else -0.0
    changed_weights = {**context.weights, entity: Weights(changed_values, value.kind)}
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_REPLAY_WEIGHT_BYTES$"):
        bridge._current_context_frame(
            replace(context, weights=changed_weights), expanded.frame
        )
    with pytest.raises(ValueError, match="^SURVEY_HOST_STRATA_NAME$"):
        bridge._current_context_frame(
            replace(context, strata=context.strata.rename("__changed_name__")),
            expanded.frame,
        )
    with pytest.raises(ValueError, match="^SURVEY_POPULATION_REPLAY_AXIS$"):
        bridge._current_context_frame(
            replace(context, strata=context.strata.rename_axis("__changed_index__")),
            expanded.frame,
        )
    assert owner._frame_identity(expanded.frame) == before

    graph = replace(live.compiled.graph, nodes=(*live.compiled.graph.nodes, *nodes))
    compiled = compile_graph(graph)
    assert len(compiled.graph.nodes) == 6
    live.kernels.register(
        bridge.CurrentSurveyHostProjectionKernel(preparation, allocated, expanded)
    )
    live.kernels.register(bridge.CurrentSurveyMatrixKernel())
    outputs = []
    for resume in ("auto", "require"):
        observed = {}
        manifest = run_graph(
            compiled,
            store=live.store,
            kernels=live.kernels,
            sources=live.sources,
            resume=resume,
            _population_observer=lambda name, population, observed=observed: (
                observed.__setitem__(name, population)
            ),
        )
        actual = []
        for node_id, name, type_, expected, kernel in (
            (
                bridge.CURRENT_PROJECTION_NODE,
                "projection",
                bridge.CURRENT_PROJECTION_TYPE,
                projection,
                bridge.CurrentSurveyHostProjectionKernel,
            ),
            (
                bridge.CURRENT_MATRIX_NODE,
                "matrix",
                bridge.model_input.RECIPIENT_MATRIX_TYPE,
                matrix,
                bridge.CurrentSurveyMatrixKernel,
            ),
        ):
            survey._final_artifact(
                manifest,
                live.store,
                node_id=node_id,
                name=name,
                type_=type_,
                payload=expected,
                capabilities=kernel.capabilities,
            )
            receipt = manifest.node(node_id)
            assert receipt.opaque_artifacts[name] == opaque_artifact_key(
                receipt.key, name
            )
            actual.append(live.store.load_bytes(receipt.opaque_artifacts[name]))
            replay.same_replayed_population(expanded, observed[node_id])
        checked = bridge.verify_materialized_current_survey_host(
            preparation,
            allocated,
            expanded,
            population=observed[bridge.CURRENT_MATRIX_NODE],
            projection=actual[0],
            matrix=actual[1],
        )
        assert checked == evidence
        outputs.append(tuple(actual))
    assert outputs[0] == outputs[1] == (projection, matrix)
    assert owner._frame_identity(expanded.frame) == before
    wrong_pins = copy.deepcopy(pins)
    wrong_pins["preparation"]["producer_key"] = "f" * 64
    bad_projection_node = replace(
        nodes[0],
        params={
            **dict(nodes[0].params),
            "host_edges": codec.encode_json(wrong_pins).decode(),
        },
    )
    bad_graph = replace(
        graph, nodes=(*live.compiled.graph.nodes, bad_projection_node, nodes[1])
    )
    with pytest.raises(NodeRejected, match="SURVEY_HOST_EDGE_PIN"):
        run_graph(
            compile_graph(bad_graph),
            store=live.store,
            kernels=live.kernels,
            sources=live.sources,
            resume="auto",
        )

    # A well-formed amount mutation can pass the pure value parser; the live
    # materialized verifier must reject it against the retained current owner.
    changed = copy.deepcopy(document)
    changed["asec"]["rows"][0][3] = np.asarray([60001.0], dtype="<f8").tobytes().hex()
    changed_bytes = codec.encode_json(changed)
    changed_matrix, _, _ = consumer.current_survey_recipient_matrix(
        expanded.frame, changed_bytes
    )
    with pytest.raises(ValueError, match="SURVEY_HOST_MATERIALIZED_PROJECTION"):
        bridge.verify_materialized_current_survey_host(
            preparation,
            allocated,
            expanded,
            population=expanded,
            projection=changed_bytes,
            matrix=changed_matrix,
        )
