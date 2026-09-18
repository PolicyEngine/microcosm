"""Invented numerical/graph contracts; these fixtures issue no survey authority."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import graph_joint_empirical as graph
from microcosm.fit import joint_empirical as empirical
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    Owned,
    SeedSource,
    SourceRef,
    StoreCorruptError,
    StructuralDelta,
    compile_graph,
    keyed_uniform,
    load_source,
    platform_fingerprint,
    run_graph,
)
from microcosm.graph.keys import node_key, opaque_artifact_key

INPUT_TYPE = ArtifactType("test.joint_empirical_source", 1)
DONOR_SOURCE = b"invented-full-donor-projection"
RECIPIENT_SOURCE = b"invented-original-recipient-projection"
STREAM = ("sha256-u53-v1", "invented-empirical", 2, 7)
SUFFIX = ("child-property-v1",)
COORDINATES = ("survey", "native_id")
SCENARIO = "a" * 64


def policy():
    threshold = empirical.SupportThreshold(1, 1, 1.0, 1.0)
    return empirical.SupportRequirements(
        "invented-test-only", "test", threshold, threshold, threshold, (0, 1, 2, 3)
    )


def transport():
    return empirical.JointTransport(1.0, (1.0, 1.0))


def donor_table():
    return pd.DataFrame(
        dict(
            person_id=np.array([2**53 + 1 + 2 * i for i in range(5)], dtype=np.int64),
            survey=pd.array(["asec"] * 5, dtype="string"),
            native_id=np.array([2**53 + 101 + i for i in range(5)], dtype=np.int64),
            original_household=pd.array(["01", "02", "03", "04", "05"], dtype="string"),
            O=[0.0, 10.0, 0.0, 3.0, 30.0],
            D=[0.0, 0.0, 20.0, 7.0, 70.0],
            original_weight=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
    )


def recipient_table():
    return pd.DataFrame(
        dict(
            person_id=np.array(
                [2**53 + 501 + 2 * i for i in range(12)], dtype=np.int64
            ),
            survey=pd.array(["acs"] * 12, dtype="string"),
            native_id=np.array([2**53 + 701 + i for i in range(12)], dtype=np.int64),
        )
    )


def fit_node(**changes):
    options = dict(
        population="donors",
        entity="person",
        targets=("O", "D"),
        donor_key_columns=COORDINATES,
        household_key_columns=("survey", "original_household"),
        weight_column="original_weight",
        support=policy(),
        source_projection=ArtifactInput(
            "source_projection", "donors", "projection", INPUT_TYPE
        ),
        source_sha256=graph._sha(DONOR_SOURCE),
    )
    options.update(changes)
    return graph.joint_empirical_fit_node("fit", **options)


def draw_node(**changes):
    options = dict(
        population="recipients",
        entity="person",
        recipient_key_columns=COORDINATES,
        coordinate_suffix=SUFFIX,
        stream=STREAM,
        support=policy(),
        transport=transport(),
        scenario_sha256=SCENARIO,
        donor_source_sha256=graph._sha(DONOR_SOURCE),
        source_sha256=graph._sha(RECIPIENT_SOURCE),
        model_producer="fit",
        source_projection=ArtifactInput(
            "source_projection", "recipients", "projection", INPUT_TYPE
        ),
    )
    options.update(changes)
    return graph.joint_empirical_draw_node("draw", **options)


def artifact(payload, type_, output, producer="b" * 64, *, numeric=Numeric.BITWISE):
    scope = NumericScope(
        numeric,
        platform=platform_fingerprint()
        if numeric is Numeric.PLATFORM_BITWISE
        else None,
    )
    return ArtifactValue(
        payload, type_, opaque_artifact_key(producer, output), producer, scope
    )


def ctx(node, table, artifacts):
    return KernelContext(
        node,
        {"person": table},
        {},
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(911),
        artifacts=artifacts,
    )


def fitted(table=None):
    table = donor_table() if table is None else table
    return graph.JointEmpiricalFitKernel().run(
        ctx(
            fit_node(),
            table,
            {"source_projection": artifact(DONOR_SOURCE, INPUT_TYPE, "projection")},
        )
    )


def draw_context(table=None, node=None, fit=None):
    table = recipient_table() if table is None else table
    fit = fitted() if fit is None else fit
    return ctx(
        draw_node() if node is None else node,
        table,
        {
            "source_projection": artifact(
                RECIPIENT_SOURCE, INPUT_TYPE, "projection", "c" * 64
            ),
            "model": artifact(
                fit.artifacts["model"],
                graph.MODEL_TYPE,
                "model",
                numeric=Numeric.PLATFORM_BITWISE,
            ),
            "model_metadata": artifact(
                fit.artifacts["model_metadata"],
                graph.MODEL_METADATA_TYPE,
                "model_metadata",
                numeric=Numeric.PLATFORM_BITWISE,
            ),
        },
    )


def decoded(payload, table, node=None):
    p = (draw_node() if node is None else node).params
    model = empirical.JointEmpiricalModel.from_bytes(fitted().artifacts["model"])
    return graph.read_joint_empirical_draw(
        payload,
        recipient_ids=pd.Index(table.person_id, name="person_id"),
        recipient_keys=graph._keys(table, COORDINATES, unique=True),
        model_sha256=model.sha256,
        support=policy(),
        donor_source_sha256=p["donor_source_sha256"],
        source_sha256=p["source_sha256"],
        scenario_sha256=p["scenario_sha256"],
        transport=graph._transport(p["transport_json"]),
        stream=p["stream"],
        coordinate_suffix=p["coordinate_suffix"],
    )


def test_real_fit_matches_explicit_weight_law_and_private_model_is_permutation_stable():
    table = donor_table()
    before = table.copy(deep=True)
    result = fitted(table)
    model = empirical.JointEmpiricalModel.from_bytes(result.artifacts["model"])
    np.testing.assert_allclose(
        model.diagnostics["pattern_probabilities"], np.array([1, 2, 3, 9]) / 15
    )
    assert model.diagnostics["weight_kind"] == "explicit"
    assert result.artifacts == fitted(table.iloc[::-1]).artifacts
    pd.testing.assert_frame_equal(table, before)
    metadata = json.loads(result.artifacts["model_metadata"])
    assert str(2**53 + 101) not in result.artifacts["model_metadata"].decode()
    assert str(2**53 + 101) in result.artifacts["model"].decode()
    assert metadata["support_sha256"] == graph._sha(graph._json(policy().document()))
    assert not result.columns and result.frame is None


def test_real_keyed_draw_matches_numerical_operator_and_leaves_sequential_rng_untouched():
    context = draw_context()
    before = context.tables["person"].copy(deep=True)
    state = json.dumps(context.rng.bit_generator.state, sort_keys=True)
    result = graph.JointEmpiricalDrawKernel().run(context)
    parsed = decoded(result.artifacts["draws"], before)
    keys = graph._keys(before, COORDINATES, unique=True)
    model = empirical.JointEmpiricalModel.from_bytes(context.artifacts["model"].payload)
    expected = empirical.draw_joint_empirical(
        model,
        pattern_uniforms=keyed_uniform(
            stream=STREAM, keys=[(*key, *SUFFIX, "pattern") for key in keys]
        ),
        donor_uniforms=keyed_uniform(
            stream=STREAM, keys=[(*key, *SUFFIX, "donor") for key in keys]
        ),
        transport=transport(),
    )
    np.testing.assert_array_equal(parsed.values, expected.values)
    assert parsed.donor_keys == expected.donor_keys
    assert not parsed.values.flags.writeable and not parsed.patterns.flags.writeable
    assert json.dumps(context.rng.bit_generator.state, sort_keys=True) == state
    pd.testing.assert_frame_equal(context.tables["person"], before)
    assert not result.columns and result.frame is None
    assert graph.JointEmpiricalDrawKernel.capabilities.seed_source is SeedSource.KEYED


@pytest.mark.parametrize(
    "change", ["permutation", "partition", "addition", "side_ids", "scenario"]
)
def test_original_draw_coordinates_ignore_packing_and_scenario_revision(change):
    table = recipient_table()
    baseline = graph.JointEmpiricalDrawKernel().run(draw_context(table))
    original = decoded(baseline.artifacts["draws"], table)
    node = draw_node()
    changed = table.copy(deep=True)
    if change == "permutation":
        changed = changed.iloc[::-1]
    elif change == "partition":
        changed = changed.iloc[::2]
    elif change == "addition":
        extra = changed.iloc[:1].copy()
        extra["person_id"] += 1000
        extra["native_id"] += 1000
        changed = pd.concat([changed, extra], ignore_index=True)
    elif change == "side_ids":
        changed["person_id"] += 10000
    elif change == "scenario":
        node = draw_node(scenario_sha256="f" * 64)
    result = graph.JointEmpiricalDrawKernel().run(draw_context(changed, node))
    parsed = decoded(result.artifacts["draws"], changed, node)
    by_key = dict(
        zip(graph._keys(table, COORDINATES, unique=True), original.values, strict=True)
    )
    for key, pair in zip(
        graph._keys(changed, COORDINATES, unique=True), parsed.values, strict=True
    ):
        if key in by_key:
            np.testing.assert_array_equal(pair, by_key[key])
    if change == "permutation":
        assert result.artifacts == baseline.artifacts


def test_integer_string_coordinates_remain_distinct_above_2_to_53():
    table = recipient_table()
    strings = table.copy()
    strings["native_id"] = pd.array(strings.native_id.map(str), dtype="string")
    ints = graph._keys(table, COORDINATES, unique=True)
    texts = graph._keys(strings, COORDINATES, unique=True)
    assert graph._coordinate(ints[0]) != graph._coordinate(texts[0])
    assert not np.array_equal(
        keyed_uniform(stream=STREAM, keys=ints),
        keyed_uniform(stream=STREAM, keys=texts),
    )
    payload = (
        graph.JointEmpiricalDrawKernel().run(draw_context(strings)).artifacts["draws"]
    )
    decoded(payload, strings)
    with pytest.raises(ValueError, match="DRAW_COORDINATE"):
        decoded(payload, table)


@pytest.mark.parametrize(
    "defect",
    [
        "key_float",
        "key_bool",
        "missing_key",
        "duplicate_key",
        "side_float",
        "weight_nullable",
        "negative_weight",
        "unknown_amount",
    ],
)
def test_fit_refuses_unsupported_identity_weights_and_amounts(defect):
    table = donor_table()
    if defect == "key_float":
        table["native_id"] = table.native_id.astype(float)
    elif defect == "key_bool":
        table["native_id"] = True
    elif defect == "missing_key":
        table.loc[0, "survey"] = pd.NA
    elif defect == "duplicate_key":
        table.loc[1, "native_id"] = table.loc[0, "native_id"]
    elif defect == "side_float":
        table["person_id"] = table.person_id.astype(float)
    elif defect == "weight_nullable":
        table["original_weight"] = pd.array(table.original_weight, dtype="Float64")
    elif defect == "negative_weight":
        table.loc[0, "original_weight"] = -1.0
    else:
        table.loc[0, "O"] = np.nan
    with pytest.raises(ValueError):
        fitted(table)


@pytest.mark.parametrize(
    "defect",
    [
        "params",
        "source",
        "type",
        "artifact_key",
        "model",
        "valid_model_stale_metadata",
        "metadata",
        "metadata_count_float",
        "producer",
        "scope",
        "schema",
    ],
)
def test_draw_rejects_changed_declarations_or_typed_artifact_bindings(defect):
    context = draw_context()
    artifacts = dict(context.artifacts)
    if defect == "params":
        context = replace(
            context, params={**context.params, "support_sha256": "f" * 64}
        )
    elif defect == "source":
        artifacts["source_projection"] = replace(
            artifacts["source_projection"], payload=b"different"
        )
    elif defect == "type":
        artifacts["model"] = replace(artifacts["model"], type=INPUT_TYPE)
    elif defect == "artifact_key":
        artifacts["model"] = replace(artifacts["model"], key="0" * 64)
    elif defect == "model":
        artifacts["model"] = replace(artifacts["model"], payload=b"{}")
    elif defect == "valid_model_stale_metadata":
        changed = donor_table()
        changed.loc[4, "O"] = 31.0
        valid_model = fitted(changed).artifacts["model"]
        empirical.JointEmpiricalModel.from_bytes(valid_model)
        artifacts["model"] = replace(artifacts["model"], payload=valid_model)
    elif defect == "metadata_count_float":
        doc = json.loads(artifacts["model_metadata"].payload)

        def float_counts(value):
            if type(value) is int:
                return float(value)
            if type(value) is dict:
                return {key: float_counts(item) for key, item in value.items()}
            if type(value) is list:
                return [float_counts(item) for item in value]
            return value

        doc["diagnostics"] = float_counts(doc["diagnostics"])
        payload = graph._json(doc)
        assert payload != artifacts["model_metadata"].payload
        artifacts["model_metadata"] = replace(
            artifacts["model_metadata"], payload=payload
        )
    elif defect == "metadata":
        doc = json.loads(artifacts["model_metadata"].payload)
        doc["donor_source_sha256"] = "0" * 64
        artifacts["model_metadata"] = replace(
            artifacts["model_metadata"], payload=graph._json(doc)
        )
    elif defect == "producer":
        artifacts["model_metadata"] = artifact(
            artifacts["model_metadata"].payload,
            graph.MODEL_METADATA_TYPE,
            "model_metadata",
            "e" * 64,
            numeric=Numeric.PLATFORM_BITWISE,
        )
    elif defect == "scope":
        artifacts["model"] = replace(artifacts["model"], numerics=NumericScope())
    else:
        context = replace(context, node=replace(context.node, inputs=()))
    with pytest.raises(ValueError):
        graph.JointEmpiricalDrawKernel().run(replace(context, artifacts=artifacts))


@pytest.mark.parametrize(
    "defect",
    ["side_id", "coordinate", "pattern", "donor", "value", "duplicate", "hash"],
)
def test_private_reader_checks_exact_original_join_and_payload_shape(defect):
    table = recipient_table()
    payload = (
        graph.JointEmpiricalDrawKernel().run(draw_context(table)).artifacts["draws"]
    )
    doc = json.loads(payload)
    row = doc["rows"][0]
    if defect == "side_id":
        row["support_id"] = "1"
    elif defect == "coordinate":
        row["coordinate"][-1][1] = "99999"
    elif defect == "pattern":
        row["pattern"] = (row["pattern"] + 1) % 4
    elif defect == "donor":
        row["donor_key"] = None if row["donor_key"] is not None else [["int", "5"]]
    elif defect == "value":
        row["values"][0] = -1.0
    elif defect == "duplicate":
        doc["rows"][-1] = doc["rows"][0]
    else:
        doc["source_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        decoded(graph._json(doc), table)


def test_zero_stress_and_empty_recipient_projection_are_explicit():
    node = draw_node(transport=empirical.JointTransport(0.0, (2.0, 3.0)))
    table = recipient_table()
    result = graph.JointEmpiricalDrawKernel().run(draw_context(table, node))
    parsed = decoded(result.artifacts["draws"], table, node)
    assert not parsed.values.any() and all(key is None for key in parsed.donor_keys)
    empty = table.iloc[:0]
    result = graph.JointEmpiricalDrawKernel().run(draw_context(empty))
    assert decoded(result.artifacts["draws"], empty).values.shape == (0, 2)


@pytest.mark.parametrize("replacement", [2.0, True, 0.0, False])
def test_private_reader_refuses_noninteger_stream_encodings(replacement):
    replicate = 2 if replacement == 2.0 else int(replacement)
    node = draw_node(stream=(STREAM[0], STREAM[1], replicate, STREAM[3]))
    table = recipient_table()
    result = graph.JointEmpiricalDrawKernel().run(draw_context(table, node))
    document = json.loads(result.artifacts["draws"])
    document["stream"][2] = replacement
    with pytest.raises(ValueError, match="DRAW_BINDING"):
        decoded(graph._json(document), table, node)


@pytest.mark.parametrize(
    "defect", ["nullable", "integer", "null", "duplicate_unselected"]
)
def test_eligibility_is_a_complete_boolean_source_input(defect):
    table = recipient_table()
    table["eligible"] = False
    if defect == "nullable":
        table["eligible"] = pd.array(table.eligible, dtype="boolean")
    elif defect == "integer":
        table["eligible"] = 0
    elif defect == "null":
        table["eligible"] = None
    else:
        table.loc[1, "native_id"] = table.loc[0, "native_id"]
    with pytest.raises(ValueError):
        graph.JointEmpiricalDrawKernel().run(
            draw_context(table, draw_node(eligibility_column="eligible"))
        )


class PrivateSupportKernel(KernelBase):
    ref = "test.joint_empirical_support@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        payload = DONOR_SOURCE if context.node.id == "donors" else RECIPIENT_SOURCE
        return KernelResult(
            frame=load_source("frame-store", context.sources[context.node.id]),
            artifacts={"projection": payload},
        )


def support_frame(table):
    people = table.copy(deep=True)
    people["person_household_id"] = people.person_id.to_numpy(copy=True)
    return Frame(
        {
            "person": people,
            "household": pd.DataFrame(
                {"household_id": people.person_id.to_numpy(copy=True)}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(len(people)), WeightKind.DESIGN)},
        metadata={"private_invented_support": True},
    )


@pytest.mark.parametrize("eligibility", ["all", "some", "none"])
def test_actual_cold_and_required_replay_and_scenario_key_change(tmp_path, eligibility):
    tables = {"donors": donor_table(), "recipients": recipient_table()}
    if eligibility != "all":
        tables["recipients"]["eligible"] = (
            np.arange(len(tables["recipients"])) % 2 == 0
            if eligibility == "some"
            else False
        )
    draw = draw_node(eligibility_column=None if eligibility == "all" else "eligible")
    sources = []
    creates = []
    for name, table in tables.items():
        sources.append(SourceRef(name, "frame-store"))
        creates.append(
            Node(
                name,
                PrivateSupportKernel.ref,
                sources=(name,),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned("person", column, str(table[column].dtype))
                    for column in table
                    if column != "person_id"
                ),
                artifact_outputs=(ArtifactOutput("projection", INPUT_TYPE),),
            )
        )
    declarations = (*creates, fit_node(), draw)
    compiled = compile_graph(
        Graph(country="us", sources=tuple(sources), nodes=declarations)
    )
    registry = KernelRegistry()
    for kernel in (
        PrivateSupportKernel(),
        graph.JointEmpiricalFitKernel(),
        graph.JointEmpiricalDrawKernel(),
    ):
        registry.register(kernel)
    store = ContentStore(tmp_path)
    refs = {
        name: store.put_frame(
            hashlib.sha256(name.encode()).hexdigest(), support_frame(table)
        )
        for name, table in tables.items()
    }
    observed = {}
    cold = run_graph(
        compiled,
        sources=refs,
        store=store,
        kernels=registry,
        _population_observer=lambda name, pop: observed.update({name: pop}),
    )
    assert len(cold.nodes) == 4 and not any(node.hit for node in cold.nodes.values())
    warm = run_graph(
        compiled,
        sources=refs,
        store=ContentStore(tmp_path),
        kernels=registry,
        resume="require",
    )
    assert cold.key == warm.key and all(node.hit for node in warm.nodes.values())
    assert cold.node("draw").opaque_artifacts == warm.node("draw").opaque_artifacts
    payload = store.load_bytes(cold.node("draw").opaque_artifacts["draws"])
    selected = (
        tables["recipients"]
        if eligibility == "all"
        else tables["recipients"].loc[tables["recipients"].eligible]
    )
    assert decoded(payload, selected, draw).values.shape == (len(selected), 2)
    assert cold.node("draw").receipt["candidate_rows"] == len(tables["recipients"])
    for name, table in tables.items():
        pd.testing.assert_frame_equal(observed[name].frame.person[table.columns], table)
    changed_draw = draw_node(
        scenario_sha256="f" * 64,
        eligibility_column=None if eligibility == "all" else "eligible",
    )
    changed = compile_graph(
        Graph(
            country="us",
            sources=tuple(sources),
            nodes=(*creates, fit_node(), changed_draw),
        )
    )
    rerun = run_graph(changed, sources=refs, store=store, kernels=registry)
    assert rerun.node("fit").hit and not rerun.node("draw").hit
    assert rerun.node("draw").key != cold.node("draw").key
    new = store.load_bytes(rerun.node("draw").opaque_artifacts["draws"])
    np.testing.assert_array_equal(
        decoded(payload, selected, draw).values,
        decoded(new, selected, changed_draw).values,
    )
    # Only this test's invented cache bytes are changed. Required replay must
    # refuse the corrupted artifact, including a legitimate zero-draw payload.
    corrupt = (
        store.object_path(cold.node("draw").opaque_artifacts["draws"]) / "payload.bin"
    )
    corrupt.write_bytes(payload + b" ")
    with pytest.raises(StoreCorruptError):
        run_graph(
            compiled, sources=refs, store=store, kernels=registry, resume="require"
        )


def _key_case(*, fit=None, draw=None):
    creates = tuple(
        Node(
            name,
            PrivateSupportKernel.ref,
            sources=(name,),
            structural=StructuralDelta.CREATE,
            outputs=tuple(
                Owned("person", column, str(table[column].dtype))
                for column in table
                if column != "person_id"
            ),
            artifact_outputs=(ArtifactOutput("projection", INPUT_TYPE),),
        )
        for name, table in (
            ("donors", donor_table()),
            ("recipients", recipient_table()),
        )
    )
    return compile_graph(
        Graph(
            "us",
            tuple(SourceRef(name, "frame-store") for name in ("donors", "recipients")),
            (
                *creates,
                fit_node() if fit is None else fit,
                draw_node() if draw is None else draw,
            ),
        )
    )


def _keys_for(compiled, draw_implementation):
    keys = {}
    for name in compiled.order:
        kernel = (
            graph.JointEmpiricalFitKernel()
            if name == "fit"
            else graph.JointEmpiricalDrawKernel()
            if name == "draw"
            else PrivateSupportKernel()
        )
        keys[name] = node_key(
            compiled,
            name,
            keys,
            draw_implementation if name == "draw" else kernel.implementation_hash(),
            {"donors": "d" * 64, "recipients": "e" * 64},
            kernel_capabilities=kernel.capabilities,
        )
    return keys


def test_actual_canonical_source_bytes_change_implementation_and_draw_key(monkeypatch):
    compiled = _key_case()
    kernel = graph.JointEmpiricalDrawKernel()
    before = kernel.implementation_hash()
    key_before = _keys_for(compiled, before)["draw"]
    target = Path(graph.canonical.__file__).resolve()
    original = Path.read_bytes
    reads = []

    def changed(path):
        data = original(path)
        if path.resolve() == target:
            reads.append(path)
            return data + b"\n# invented canonical implementation revision\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", changed)
    after = kernel.implementation_hash()
    assert reads and before != after
    assert key_before != _keys_for(compiled, after)["draw"]


@pytest.mark.parametrize(
    "change", ["donor_source", "recipient_source", "support", "transport", "stream"]
)
def test_actual_compiler_keys_bind_each_model_and_draw_revision(change):
    baseline = _key_case()
    fit, draw = fit_node(), draw_node()
    if change == "donor_source":
        fit, draw = (
            fit_node(source_sha256="f" * 64),
            draw_node(donor_source_sha256="f" * 64),
        )
    elif change == "recipient_source":
        draw = draw_node(source_sha256="f" * 64)
    elif change == "support":
        support = replace(policy(), policy_id="another-explicit-test-policy")
        fit, draw = fit_node(support=support), draw_node(support=support)
    elif change == "transport":
        draw = draw_node(transport=empirical.JointTransport(0.5, (2.0, 3.0)))
    else:
        draw = draw_node(stream=(*STREAM[:3], STREAM[3] + 1))
    implementation = graph.JointEmpiricalDrawKernel().implementation_hash()
    before, after = (
        _keys_for(baseline, implementation),
        _keys_for(_key_case(fit=fit, draw=draw), implementation),
    )
    assert before["draw"] != after["draw"]
    assert (before["fit"] != after["fit"]) == (change in ("donor_source", "support"))


def test_recipient_ceiling_refuses_at_a_patched_down_value_and_follows_the_rule(
    monkeypatch,
):
    """MAX_RECIPIENTS bounds the recipient support table, one row per stacked person.

    The US producer builds that table from every stacked person of the
    selection (3,565,013 at full source), so the shipped value is four times
    that count rounded up to the next whole million, the same number its two
    upstream siblings take. The draw kernel checks it before reading the model,
    so a table one row over a patched-down ceiling refuses with RECIPIENT_COUNT
    and a table at the ceiling draws; no full-source table is allocated here.
    """
    assert graph.MAX_RECIPIENTS == 15_000_000
    assert graph.MAX_RECIPIENTS == -(-4 * 3_565_013 // 1_000_000) * 1_000_000
    table = recipient_table()
    monkeypatch.setattr(graph, "MAX_RECIPIENTS", len(table) - 1)
    with pytest.raises(ValueError, match="RECIPIENT_COUNT"):
        graph.JointEmpiricalDrawKernel().run(draw_context(table))
    monkeypatch.setattr(graph, "MAX_RECIPIENTS", len(table))
    result = graph.JointEmpiricalDrawKernel().run(draw_context(table))
    assert decoded(result.artifacts["draws"], table).values.shape[0] == len(table)
