"""Descriptive schema controls and actual four-node replay on tiny real owners.

Detached fixtures are explicitly unissued and used only for pure declarations,
recodes and bindings. The separate actual_graph fixture prepares invented
members through the real issuer before any kernel is admitted or executed.
"""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_survey_person_status import acs, asec

from microcosm.build.us_runtime import graph_current_survey_person_status as graph
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import ArtifactInput, ArtifactType, ArtifactValue, NumericScope
from microcosm.graph import population as populations
from microcosm.graph.keys import opaque_artifact_key


def _frame(people):
    people = people.copy()
    tables = {"person": people}
    for entity in US_SCHEMA.group_entities:
        people[US_SCHEMA.membership_column(entity)] = people.person_id.to_numpy()
        tables[entity] = pd.DataFrame(
            {US_SCHEMA.entity_id_column(entity): np.sort(people.person_id.to_numpy())}
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(people)), WeightKind.DESIGN)},
    )


def detached(arms=("asec", "acs"), changes=None):
    """No owner callback: these values cannot pass actual kernel admission."""
    rows = [
        asec(PERIDNUM=str(10**21 + i)) if arm == "asec" else acs(SPORDER=str(i + 1))
        for i, arm in enumerate(arms)
    ]
    if changes:
        for row, arm in zip(rows, arms, strict=True):
            row.update(changes.get(arm, {}))
    index = pd.Index(
        [2**53 + 1 + 2 * i for i in range(len(rows))], name="person_id", dtype="int64"
    )
    origins = pd.DataFrame(
        {"source": arms, "native_person_id": index.to_numpy() + 100}, index=index
    )
    frame = _frame(
        pd.DataFrame(
            {
                "person_id": index.to_numpy(),
                "A_AGE": [
                    int(r["A_AGE" if a == "asec" else "AGEP"])
                    for a, r in zip(arms, rows, strict=True)
                ],
            }
        )
    )
    selected, keys = {"asec": {}, "acs": {}}, {}
    for arm, row, pid in zip(arms, rows, index, strict=True):
        key = graph.source.original._key(row, arm)
        selected[arm][key] = row
        keys[arm, key] = int(pid)
    raw, observed = graph.source._combine(frame, origins, keys, selected)
    return graph.source.QualifiedSurveyPersonStatus(
        frame, origins, raw, observed, b'{"invented_unissued":true}'
    )


def receiving(qualified):
    pairs = [(int(pid), clone) for pid in qualified.origins.index for clone in (0, 1)][
        ::-1
    ]
    people = pd.DataFrame(
        {
            "person_id": np.arange(1, len(pairs) + 1, dtype="int64"),
            graph.provenance.support_source_id_column("person"): np.array(
                [p for p, _ in pairs], dtype="int64"
            ),
            graph.provenance.support_clone_index_column("person"): np.array(
                [c for _, c in pairs], dtype="int64"
            ),
            graph.provenance.spine_source_id_column("person"): np.array(
                [qualified.origins.at[p, "native_person_id"] for p, _ in pairs],
                dtype="int64",
            ),
            graph.provenance.support_channel_column("person"): pd.array(
                [qualified.origins.at[p, "source"] for p, _ in pairs],
                dtype=graph.source.STRING,
            ),
            "unrelated_amount": np.arange(len(pairs), dtype="float64"),
        }
    )
    return _frame(people)


def declarations(qualified, frame=None):
    frame = receiving(qualified) if frame is None else frame
    edge = ArtifactInput(
        "ordering", "invented.after", "evidence", ArtifactType("invented.ordering", 1)
    )
    pins = {
        edge.name: {
            "producer_key": "1" * 64,
            "artifact_key": "2" * 64,
            "payload_sha256": hashlib.sha256(b"invented ordering").hexdigest(),
        }
    }
    nodes = graph.current_survey_person_status_nodes(
        qualified,
        frame,
        receiving_version="invented.postclone",
        after=edge,
        host_edges=(edge,),
        host_pins=pins,
    )
    return nodes, edge, pins


@pytest.mark.parametrize("arms", [("acs",), ("asec",), ("asec", "acs")])
def test_fixed_schema_including_all_null_opposite_survey_codes(arms):
    qualified = detached(arms)
    raw, observed = graph.status_tables(qualified)
    assert tuple(raw) == tuple(n for n, _ in graph.LITERAL_TOKENS)
    assert tuple(observed) == tuple(n for n, _ in graph.RECODE_TOKENS)
    for table, schema in ((raw, graph.LITERAL_TOKENS), (observed, graph.RECODE_TOKENS)):
        assert tuple(
            populations.token_for_dtype(table[n].dtype) for n, _ in schema
        ) == tuple(t for _, t in schema)
    nodes, _, _ = declarations(qualified)
    assert tuple(n.id for n in nodes) == graph.NODE_IDS
    assert tuple(o.column for o in nodes[-1].outputs) == graph.BIND_COLUMNS
    assert not {
        "is_blind",
        "is_disabled",
        "is_full_time_college_student",
        "is_household_head",
        "is_female",
        "prior_wages",
    } & set(graph.BIND_COLUMNS)
    assert not any(c.startswith("person_status_source_") for c in graph.BIND_COLUMNS)
    missing = "PEDISEYE" if arms == ("acs",) else "DEYE" if arms == ("asec",) else None
    if missing:
        assert observed["person_status_source_" + missing + "__code"].isna().all()


@pytest.mark.parametrize(
    "changes,expected,known",
    [
        ({"DEYE": "1", "DEAR": ""}, True, True),
        ({"DEYE": ""}, None, False),
        ({"DEYE": "b"}, None, False),
        ({"DEYE": "2"}, False, True),
    ],
)
def test_exact_partial_and_unknown_semantics_survive_binding(changes, expected, known):
    qualified = detached(("acs",), {"acs": changes})
    frame = receiving(qualified)
    nodes, _, _ = declarations(qualified, frame)
    result = graph._result(qualified, nodes[-1], frame.person)
    value = result.columns["person", "survey_vision_difficulty"]
    assert value.isna().all() if expected is None else value.eq(expected).all()
    assert result.columns["person", "survey_vision_difficulty__known"].eq(known).all()
    assert (
        result.columns["person", "survey_full_time_college_student_last_week"]
        .isna()
        .all()
    )
    assert (
        result.columns["person", "survey_student_reference_period"]
        .eq(graph.status.ACS_STUDENT_PERIOD)
        .all()
    )


def test_large_integer_private_artifact_and_both_exact_clone_identities():
    qualified = detached()
    before = graph.person_status_seal(qualified)
    frame = receiving(qualified)
    nodes, _, _ = declarations(qualified, frame)
    bound = graph._result(qualified, nodes[-1], frame.person)
    payload = json.loads(graph.projection_payload(qualified))
    assert payload["origins"]["index"] == [str(v) for v in qualified.origins.index]
    assert payload["origins"]["rows"][0][1] == {"integer_literal": str(2**53 + 101)}
    assert payload["raw"]["rows"][0][graph.status.RAW_COLUMNS.index("PERIDNUM")] == str(
        10**21
    )
    originals = frame.person[graph.provenance.support_source_id_column("person")]
    assert (
        bound.columns["person", "survey_status_source"].tolist()
        == qualified.origins.source.reindex(originals).tolist()
    )
    assert graph.person_status_seal(qualified) == before
    public = json.dumps(dict(bound.receipt))
    assert str(10**21) not in public and "native_person_id" not in public
    columns_doc = json.loads(graph._columns_payload(graph.status_tables(qualified)[0]))
    assert set(columns_doc) == {"protocol", "rows", "columns", "table_sha256"}
    assert isinstance(columns_doc["rows"], int)


@pytest.mark.parametrize(
    "defect",
    ["float_original", "float_clone", "native", "arm", "pair", "missing", "collision"],
)
def test_binder_refuses_inexact_foreign_missing_and_duplicate_joins(defect):
    qualified = detached()
    frame = receiving(qualified)
    nodes, _, _ = declarations(qualified, frame)
    people = frame.person.copy()
    original, clone, native, channel = graph._identities()
    if defect == "float_original":
        people[original] = people[original].astype(float)
    elif defect == "float_clone":
        people[clone] = people[clone].astype(float)
    elif defect == "native":
        people.loc[0, native] += 1
    elif defect == "arm":
        people.loc[0, channel] = "foreign"
    elif defect == "pair":
        people.loc[0, clone] = people.loc[1, clone]
    elif defect == "missing":
        people = people.iloc[:-1].copy()
    else:
        people["survey_vision_difficulty"] = pd.array(
            [False] * len(people), dtype="boolean"
        )
    with pytest.raises(ValueError):
        graph._result(qualified, nodes[-1], people)


@pytest.mark.parametrize("column", ["survey_vision_difficulty", "unrelated_amount"])
def test_complete_materialization_refuses_status_or_untouched_cell_drift(column):
    qualified = detached()
    frame = receiving(qualified)
    nodes, _, _ = declarations(qualified, frame)
    incoming = populations.Population.from_frame(frame, nodes[-1].population)
    result = graph._result(qualified, nodes[-1], frame.person)
    output = populations.patch(incoming, nodes[-1], result)
    graph.verify_materialized_person_status(
        output, incoming, qualified=qualified, node=nodes[-1]
    )
    output.frame.person.loc[0, column] = True if column.startswith("survey_") else 999.0
    with pytest.raises(ValueError):
        graph.verify_materialized_person_status(
            output, incoming, qualified=qualified, node=nodes[-1]
        )


@pytest.mark.parametrize(
    "defect", ["raw", "recode", "source_arm", "host_pin", "artifact"]
)
def test_recode_inputs_and_exact_typed_artifacts_are_checked(defect):
    qualified = detached()
    nodes, edge, pins = declarations(qualified)
    projected = graph._projection_frame(qualified)
    literal = graph._result(qualified, nodes[1], projected.person)
    incoming = populations.patch(
        populations.Population.from_frame(projected, nodes[0].id), nodes[1], literal
    )
    if defect == "raw":
        incoming.frame.person.loc[0, "person_status_source_PEDISEYE"] = "1"
        with pytest.raises(ValueError, match="RECODE_SOURCE_SLICE"):
            graph._result(qualified, nodes[2], incoming.frame.person)
    elif defect == "source_arm":
        incoming.frame.person.loc[0, "person_status_source_arm"] = "acs"
        with pytest.raises(ValueError, match="RECODE_SOURCE_ARM"):
            graph._result(qualified, nodes[2], incoming.frame.person)
    elif defect == "recode":
        qualified.observations.iloc[
            0, qualified.observations.columns.get_loc("survey_vision_difficulty")
        ] = True
        with pytest.raises(ValueError, match="RECODE_QUALIFIER_DISAGREEMENT"):
            graph.status_tables(qualified)
    else:
        pin = pins[edge.name]
        value = ArtifactValue(
            b"invented ordering",
            edge.type,
            pin["artifact_key"],
            pin["producer_key"],
            NumericScope(),
        )
        if defect == "host_pin":
            value = replace(value, producer_key="3" * 64)
        else:
            value = replace(value, payload=b"changed")
        with pytest.raises(ValueError, match="HOST_EDGE_PIN"):
            graph._check_artifacts(nodes[0], {edge.name: value}, qualified)


def test_pure_declarations_do_not_issue_retained_authority():
    qualified = detached()
    frame = receiving(qualified)
    _, edge, pins = declarations(qualified, frame)
    with pytest.raises(ValueError, match="RETAINED_OWNER_REQUIRED"):
        graph.person_status_kernels(
            qualified,
            frame,
            receiving_version="invented.postclone",
            after=edge,
            host_edges=(edge,),
            host_pins=pins,
            require_current=lambda: None,
        )


@pytest.fixture(scope="module")
def actual_graph(tmp_path_factory):
    from test_us_current_survey_person_status_source import _source_arguments
    from test_us_graph_atomic_survey_population import _support_payload

    from microcosm.build.us_runtime import graph_atomic_survey_population as atomic
    from microcosm.graph import compile_graph, run_graph
    from microcosm.graph.executor import _all_node_keys, _source_paths_and_keys

    root = tmp_path_factory.mktemp("person-status-actual-graph")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _source_arguments(root, patch)
        payload, source_ids = _support_payload()
        support = root / "invented-block-support.npz"
        support.write_bytes(payload)
        config = atomic.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        prefix = atomic.run_atomic_survey_population(
            **arguments,
            store_root=root / "store",
            geography_config=config,
            return_values=True,
        )
        qualified = graph.qualify_current_survey_person_status(prefix.preparation)
        # The actual host supplies the same four real typed ancestor pins.
        from microcosm.build.us_runtime import (
            graph_current_survey_predictors as financial,
        )

        edges = (
            *financial.host.current_survey_host_edges(),
            financial._geography_edge(),
        )
        pins = {}
        for edge in edges:
            record = prefix.manifest.node(edge.producer)
            pins[edge.name] = {
                "producer_key": record.key,
                "artifact_key": record.opaque_artifacts[edge.artifact],
                "payload_sha256": graph._sha(
                    prefix.store.load_bytes(record.opaque_artifacts[edge.artifact])
                ),
            }
        seal = graph.person_status_seal(qualified)
        state = SimpleNamespace(calls=0, mutate=None)

        def current():
            assert graph.person_status_seal(qualified) == seal
            state.calls += 1
            if state.mutate is not None:
                state.mutate(state.calls)

        nodes, kernels = graph.person_status_kernels(
            qualified,
            prefix.clone_population.frame,
            receiving_version=prefix.clone_population.version,
            after=edges[-1],
            host_edges=edges,
            host_pins=pins,
            require_current=current,
        )
        compiled = compile_graph(
            replace(prefix.compiled.graph, nodes=(*prefix.compiled.graph.nodes, *nodes))
        )
        for kernel in kernels:
            prefix.kernels.register(kernel)
        _, source_keys = _source_paths_and_keys(compiled, prefix.sources, prefix.store)
        keys, _ = _all_node_keys(compiled, prefix.kernels, source_keys)
        outputs = []
        for resume in ("auto", "require"):
            observed = {}
            manifest = run_graph(
                compiled,
                sources=prefix.sources,
                store=prefix.store,
                kernels=prefix.kernels,
                resume=resume,
                _population_observer=lambda n, p, target=observed: target.__setitem__(
                    n, p
                ),
            )
            qualified.validate()
            assert graph.person_status_seal(qualified) == seal
            outputs.append((manifest, observed))
        yield SimpleNamespace(
            prefix=prefix,
            qualified=qualified,
            nodes=nodes,
            kernels=kernels,
            compiled=compiled,
            keys=keys,
            outputs=outputs,
            state=state,
        )
        qualified.validate()


def test_actual_thirteen_node_cold_required_replay_and_private_source_schema(
    actual_graph,
):
    case = actual_graph
    payload = graph.projection_payload(case.qualified)
    assert 0 < len(payload) <= 16 * 1024 * len(case.qualified.origins)
    assert len(payload) <= graph.MAX_ARTIFACT_BYTES
    assert len(case.compiled.order) == 13
    assert case.outputs[0][0].key == case.outputs[1][0].key
    assert all(n.hit for n in case.outputs[1][0].nodes.values())
    for manifest, observed in case.outputs:
        assert set(observed) == set(case.compiled.order)
        graph.verify_materialized_person_status(
            observed[graph.BIND_NODE],
            case.prefix.clone_population,
            qualified=case.qualified,
            node=case.nodes[-1],
        )
        current = {}
        for node in case.nodes:
            artifacts = {}
            for edge in node.artifact_inputs:
                record = manifest.node(edge.producer)
                artifacts[edge.name] = ArtifactValue(
                    case.prefix.store.load_bytes(
                        record.opaque_artifacts[edge.artifact]
                    ),
                    edge.type,
                    opaque_artifact_key(case.keys[edge.producer], edge.artifact),
                    case.keys[edge.producer],
                    NumericScope(),
                )
            incoming = (
                None
                if node.id == graph.SOURCE_NODE
                else case.prefix.clone_population
                if node.id == graph.BIND_NODE
                else current[graph.SOURCE_NODE]
            )
            expected, result = graph.expected_person_status_population(
                incoming, qualified=case.qualified, node=node, artifacts=artifacts
            )
            graph.replay.same_replayed_population(expected, observed[node.id])
            current[expected.version] = expected
            for name, payload in result.artifacts.items():
                assert (
                    case.prefix.store.load_bytes(
                        manifest.node(node.id).opaque_artifacts[name]
                    )
                    == payload
                )
    assert (
        case.state.calls == 8
    )  # Two pure owner boundaries per executed fragment node.


@pytest.mark.parametrize("boundary", [1, 2])
@pytest.mark.parametrize(
    "mutation",
    [
        "raw",
        "observations",
        "receipt",
        "callback",
        "owner",
        "code",
        "defaults",
        "closure",
        "constant",
        "student",
    ],
)
def test_actual_kernel_first_and_final_owner_mutation_refuses(
    actual_graph, boundary, mutation
):
    from microcosm.graph.executor import _project_context

    case = actual_graph
    qualified = case.qualified
    kernel = case.kernels[0]
    node = case.nodes[0]
    manifest = case.outputs[0][0]
    artifacts = {}
    for edge in node.artifact_inputs:
        record = manifest.node(edge.producer)
        artifacts[edge.name] = ArtifactValue(
            case.prefix.store.load_bytes(record.opaque_artifacts[edge.artifact]),
            edge.type,
            record.opaque_artifacts[edge.artifact],
            record.key,
            NumericScope(),
        )
    context = _project_context(
        node,
        None,
        key=case.keys[node.id],
        sources=case.prefix.sources,
        tolerances={},
        numerics={},
        artifacts=artifacts,
    )
    pid = qualified.raw.index[0]
    raw, observed, receipt, callback = (
        qualified.raw.copy(deep=True),
        qualified.observations.copy(deep=True),
        qualified.receipt,
        kernel.require_current,
    )
    prep_payload = case.prefix.preparation.payload
    source_closure = dict(
        zip(
            qualified._revalidate.__code__.co_freevars,
            (c.cell_contents for c in qualified._revalidate.__closure__),
            strict=True,
        )
    )
    controls = source_closure["controls"]
    controls_body = controls._body
    callback_code, callback_defaults = callback.__code__, callback.__defaults__
    closure_cell = dict(
        zip(callback.__code__.co_freevars, callback.__closure__, strict=True)
    )["seal"]
    closure_value, protocol = closure_cell.cell_contents, graph.status.PROTOCOL
    start = case.state.calls

    def mutate(count):
        if count != start + boundary:
            return
        if mutation == "raw":
            qualified.raw.loc[pid, "DEYE"] = "changed"
        elif mutation == "observations":
            qualified.observations.loc[pid, "survey_vision_difficulty"] = True
        elif mutation == "receipt":
            object.__setattr__(qualified, "receipt", b"{}")
        elif mutation == "callback":
            kernel.require_current = lambda: None
        elif mutation == "code":
            callback.__code__ = callback.__code__.replace(co_name="changed_callback")
        elif mutation == "defaults":
            callback.__defaults__ = ("changed",)
        elif mutation == "closure":
            closure_cell.cell_contents = ("changed",)
        elif mutation == "constant":
            graph.status.PROTOCOL = "changed"
        elif mutation == "student":
            object.__setattr__(controls, "_body", controls_body + b"changed")
        else:
            object.__setattr__(case.prefix.preparation, "payload", b"{}")

    case.state.mutate = mutate
    try:
        with pytest.raises((ValueError, AssertionError)):
            kernel.run(context)
    finally:
        for column in raw:
            qualified.raw[column] = raw[column].copy(deep=True)
        for column in observed:
            qualified.observations[column] = observed[column].copy(deep=True)
        object.__setattr__(qualified, "receipt", receipt)
        kernel.require_current = callback
        callback.__code__, callback.__defaults__ = callback_code, callback_defaults
        closure_cell.cell_contents, graph.status.PROTOCOL = closure_value, protocol
        object.__setattr__(case.prefix.preparation, "payload", prep_payload)
        object.__setattr__(controls, "_body", controls_body)
        case.state.mutate = None
    qualified.validate()
