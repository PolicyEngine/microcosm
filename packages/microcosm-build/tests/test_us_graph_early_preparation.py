"""Measured hours retain the real source boundary and mandatory gate."""

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import multispine_pool as pool
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_early_preparation import (
    HOURS_PHASE,
    US_HOURS_EVIDENCE_TYPE,
    USHoursGateKernel,
    register_us_hours_kernels,
    us_hours_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
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
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.errors import NodeRejectedError


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "_hours_pool_fixtures", Path(__file__).with_name("test_us_multispine_pool.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures()


def _frame():
    asec = fixtures._real_pre_clone_source_frame()
    asec = Frame(
        {entity: asec.table(entity) for entity in asec.entities},
        asec.schema,
        {"household": Weights(np.array([1.0, 3.0]), WeightKind.DESIGN)},
    )
    frame = assemble_spines(
        {
            "asec": asec,
            "acs": fixtures._source_frame(offset=100.0),
        },
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for table in tables.values():
        for column in table:
            if table[column].dtype == object or str(table[column].dtype) == "str":
                table[column] = table[column].astype("string")
    for column in ("HRSWK", "A_HRS1", "WKSWORK"):
        tables["person"][column] = pd.to_numeric(tables["person"][column]).astype(
            "float64"
        )
    tables["person"].index = pd.Index(
        np.arange(frame.n("person")) * 3 + 101, name="original_row"
    )
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata.set_axis(tables["person"].index),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


class _Create(KernelBase):
    ref = "test.hours.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["population"]),
            artifacts={"frame_context": context.params["frame_context"].encode()},
        )


class _Consumer(KernelBase):
    ref = "test.hours.consumer@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        return KernelResult(
            receipt={
                "received_context": bool(context.artifacts["frame_context"].payload)
            }
        )


class _StructuralConsumer(KernelBase):
    ref = "test.hours.structural_consumer@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        structural=StructuralDelta.FILTER,
    )

    def run(self, context):
        return KernelResult(
            keep=pd.Series(
                True,
                index=pd.Index(context.tables["person"].person_id, name="person_id"),
                dtype=bool,
            )
        )


def _setup(tmp_path, frame):
    store = ContentStore(tmp_path / "store")
    source = ContentStore(tmp_path / "inputs").put_frame("a" * 64, frame)
    columns = frame_column_declarations(frame)
    nodes = us_hours_nodes(columns, base="source", context_producer="source")
    graph = Graph(
        country="us",
        sources=(SourceRef("population", "frame-store"),),
        nodes=(
            Node(
                "source",
                _Create.ref,
                sources=("population",),
                structural=StructuralDelta.CREATE,
                outputs=columns,
                params={"frame_context": encode_us_frame_context(frame).decode()},
                artifact_outputs=(
                    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ),
            ),
            *nodes,
            Node(
                "after",
                _Consumer.ref,
                population=f"{HOURS_PHASE}.boundary",
                artifact_inputs=(
                    ArtifactInput(
                        "frame_context",
                        f"{HOURS_PHASE}.gate",
                        "frame_context",
                        US_FRAME_CONTEXT_TYPE,
                    ),
                ),
            ),
            Node(
                "after_boundary",
                _StructuralConsumer.ref,
                base=f"{HOURS_PHASE}.boundary",
                structural=StructuralDelta.FILTER,
                inputs=(Slice("person", ("weekly_hours_worked_before_lsr",)),),
                artifact_inputs=(
                    ArtifactInput(
                        "frame_context",
                        f"{HOURS_PHASE}.gate",
                        "frame_context",
                        US_FRAME_CONTEXT_TYPE,
                    ),
                ),
            ),
        ),
    )
    kernels = KernelRegistry()
    kernels.register(_Create())
    kernels.register(_Consumer())
    kernels.register(_StructuralConsumer())
    register_us_hours_kernels(kernels)
    return compile_graph(graph), {"population": source}, store, kernels


def _direct(frame):
    return pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=("with_us_hours_worked_inputs",),
        operators={
            "with_us_hours_worked_inputs": pool._with_gated_us_hours_worked_inputs
        },
    )


def test_actual_direct_helper_graph_parity_and_required_resume(tmp_path):
    frame = _frame()
    expected = _direct(frame)
    observed = pool.observe_multispine_hours_worked_inputs(frame)
    assert observed.receipt == expected.receipt
    args, sources, store, kernels = _setup(tmp_path, frame)
    first = run_graph(args, sources=sources, store=store, kernels=kernels)
    actual = first.population(f"{HOURS_PHASE}.boundary")
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), expected.frame.table(entity)
        )
    assert "weeks_worked" not in actual.person
    pd.testing.assert_series_equal(actual.strata, frame.strata)
    for entity in frame.weighted_entities:
        assert actual.weights_for(entity).kind == frame.weights_for(entity).kind
        assert (
            actual.weights_for(entity).values.tobytes()
            == frame.weights_for(entity).values.tobytes()
        )
    gate = first.node(f"{HOURS_PHASE}.gate")
    assert canonical_json(gate.receipt["source_preparation"]) == canonical_json(
        expected.receipt
    )
    summary = expected.receipt["suboperators"][0]["kernel_receipt"][
        "hours_worked_signal_gate"
    ]["details"]
    assert summary["mean_weekly_hours_workers"] == 36.25
    assert summary["worked_share"] == 0.5
    context = store.load_bytes(gate.opaque_artifacts["frame_context"])
    assert json.loads(context) == json.loads(encode_us_frame_context(expected.frame))
    again = run_graph(
        args,
        sources=sources,
        store=ContentStore(store.root),
        kernels=kernels,
        resume="require",
    )
    assert again.key == first.key
    assert all(node.hit for node in again.nodes.values())
    assert again.node("after").receipt["received_context"]
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            again.population(f"{HOURS_PHASE}.boundary").table(entity),
            expected.frame.table(entity),
        )
    assert (
        store.load_bytes(
            again.node(f"{HOURS_PHASE}.gate").opaque_artifacts["frame_context"]
        )
        == context
    )


@pytest.mark.parametrize("failure", ("implausible", "constant"))
def test_gate_failure_preserves_evidence_and_blocks_dependents_on_cold_warm_require(
    tmp_path,
    failure,
):
    frame = _frame()
    cps = frame.person.PERIDNUM.notna()
    frame.person.loc[cps, "HRSWK"] = [60, 65, 70, 75]
    if failure == "constant":
        for column in ("HRSWK", "A_HRS1", "WKSWORK"):
            frame.person.loc[cps, column] = 0.0
    with pytest.raises(ValueError) as old:
        _direct(frame)
    args, sources, store, kernels = _setup(tmp_path, frame)
    for resume in ("forbid", "auto", "require"):
        manifest = run_graph(
            args, sources=sources, store=store, kernels=kernels, resume=resume
        )
        gate = manifest.node(f"{HOURS_PHASE}.gate")
        assert gate.receipt["outcome"] == "fail"
        assert str(old.value) == gate.receipt["evidence"]["message"]
        assert not gate.opaque_artifacts
        assert manifest.node("after").receipt["execution"]["state"] == "unreached"
        assert (
            manifest.node("after_boundary").receipt["execution"]["state"] == "unreached"
        )
        report = manifest.node(HOURS_PHASE).receipt["source_preparation"]
        evidence = report["suboperators"][0]["kernel_receipt"][
            "hours_worked_signal_gate"
        ]
        assert not evidence["passed"] and evidence["failures"]


@pytest.mark.parametrize("mutation", ("missing_context", "wrong_ids", "clone"))
def test_boundary_rejects_invalid_authority(tmp_path, mutation):
    frame = _frame()
    if mutation == "clone":
        frame.person.loc[frame.person.index[0], "person_support_clone_index"] = 1
    args, sources, store, kernels = _setup(tmp_path, frame)
    if mutation != "clone":
        document = json.loads(args.graph.nodes[0].params["frame_context"])
        if mutation == "missing_context":
            document["metadata"] = {}
        else:
            document["entities"]["person"]["ordered_ids_sha256"] = "0" * 64
        source = replace(
            args.graph.nodes[0], params={"frame_context": json.dumps(document)}
        )
        args = compile_graph(replace(args.graph, nodes=(source, *args.graph.nodes[1:])))
    with pytest.raises(Exception, match="(assembly|identity|clone|provenance)"):
        run_graph(args, sources=sources, store=store, kernels=kernels)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_legacy_fixture_payload_receipt_and_context_fingerprints(tmp_path):
    # Captured by executing the archived c71e3165 hours/pool implementations.
    # Every stored table/index/dtype/weight/stratum/legacy mass payload is covered;
    # metadata travels separately in the authenticated graph context.
    output = _direct(_frame())
    store = ContentStore(tmp_path)
    store.put_frame("b" * 64, output.frame)
    assert _digest(store.metadata("b" * 64)["payloads"]) == (
        "66d17ebf3cc2377a6a0ee9f9fec10b32b37f59df5c60d981d1fa8d2d21dc5620"
    )
    assert _digest(output.receipt) == (
        "0224f0b8f3614f21fc7e597f9c335ea81bd1ce92409fa6cf6289cbf6cd8065a7"
    )
    assert hashlib.sha256(encode_us_frame_context(output.frame)).hexdigest() == (
        "4762725b037710882a2b12a4047d86c8fcb06feccb8efd300bc9ad3867047cbc"
    )


@pytest.mark.parametrize(
    "surface", ("all_null", "partial", "pass_through", "zero_clip")
)
def test_real_hours_presence_projection_and_native_peers(tmp_path, surface):
    frame = _frame()
    person = frame.person
    cps = person.PERIDNUM.notna()
    if surface == "zero_clip":
        person.loc[cps, "HRSWK"] = [40, -1, 35, np.nan]
        person.loc[cps, "A_HRS1"] = [42, -1, 30, np.nan]
        person.loc[cps, "WKSWORK"] = [52, -4, 100, np.nan]
    else:
        person["weekly_hours_worked_before_lsr"] = 99.0
        person["hours_worked_last_week"] = 98.0
        values = [np.nan] * 4 if surface == "all_null" else [1, np.nan, 2, np.nan]
        person.loc[cps, "weekly_hours_worked_before_lsr"] = values
        person.loc[cps, "hours_worked_last_week"] = values
        if surface == "pass_through":
            person.loc[cps, "weekly_hours_worked_before_lsr"] = [20, 0, 45, 0]
            person.loc[cps, "hours_worked_last_week"] = [10, 0, 20, 0]
            person["weeks_worked"] = 17.0
            person.loc[cps, "weeks_worked"] = [40, 0, 48, 0]
            person.drop(columns=["HRSWK", "A_HRS1", "WKSWORK"], inplace=True)
    expected = _direct(frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    actual = manifest.population(f"{HOURS_PHASE}.boundary")
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            actual.table(entity), expected.frame.table(entity)
        )
    receipt = manifest.node(f"{HOURS_PHASE}.gate").receipt["source_preparation"]
    assert canonical_json(receipt) == canonical_json(expected.receipt)
    if surface != "zero_clip":
        assert actual.person.loc[~cps, "weekly_hours_worked_before_lsr"].tolist() == [
            99,
            99,
        ]
        assert actual.person.loc[~cps, "hours_worked_last_week"].tolist() == [98, 98]
    if surface == "pass_through":
        # The legacy pool merge leaves an already-present full-pool weeks column
        # alone; it does not persist a newly derived transient weeks column.
        pd.testing.assert_series_equal(actual.person.weeks_worked, person.weeks_worked)
    else:
        assert "weeks_worked" not in actual.person


@pytest.mark.parametrize("weight_source", ("person", "tax_unit"))
def test_counts_and_gate_use_actual_memberships_and_canonical_weights(
    tmp_path, weight_source
):
    frame = _frame()
    frame = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {weight_source: frame.resolve_weights(weight_source)},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    expected = _direct(frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    actual = manifest.node(f"{HOURS_PHASE}.gate").receipt["source_preparation"]
    assert canonical_json(actual) == canonical_json(expected.receipt)
    rows = actual["suboperators"][0]["operator_input_rows"]
    assert rows == {
        "person": 4,
        "household": 2,
        "tax_unit": 2,
        "spm_unit": 2,
        "family": 2,
        "marital_unit": 4,
    }


@pytest.mark.parametrize(
    "invalid", ("missing_raw", "blank_evidence", "zero_selected_weight")
)
def test_unavailable_source_inputs_keep_direct_refusal(tmp_path, invalid):
    frame = _frame()
    if invalid == "missing_raw":
        frame.person.drop(columns="WKSWORK", inplace=True)
    elif invalid == "blank_evidence":
        frame.person["PERIDNUM"] = pd.Series(
            "  ", index=frame.person.index, dtype="string"
        )
    else:
        values = frame.resolve_weights("person").values.copy()
        values[frame.person.PERIDNUM.notna()] = 0.0
        frame = Frame(
            {entity: frame.table(entity) for entity in frame.entities},
            frame.schema,
            {"person": Weights(values, WeightKind.IMPORTANCE)},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
    with pytest.raises(Exception) as direct:
        _direct(frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    with pytest.raises(NodeRejectedError) as graph:
        run_graph(compiled, sources=sources, store=store, kernels=kernels)
    assert type(graph.value.__cause__) is type(direct.value)
    assert str(graph.value.__cause__) == str(direct.value)


def test_extra_explicit_weight_vectors_are_not_silently_unchecked(tmp_path):
    frame = _frame()
    frame = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": frame.weights_for("household"),
            "person": frame.resolve_weights("person"),
        },
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    assert _direct(frame).receipt["suboperators"][0]["kernel_receipt"][
        "hours_worked_signal_gate"
    ]["passed"]
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    with pytest.raises(NodeRejectedError, match="one explicit weight source"):
        run_graph(compiled, sources=sources, store=store, kernels=kernels)


def test_omitting_an_available_raw_field_cannot_change_healing_semantics(tmp_path):
    frame = _frame()
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    nodes = tuple(
        replace(
            node,
            inputs=(
                Slice(
                    "person",
                    tuple(
                        column
                        for column in node.inputs[0].columns
                        if column != "WKSWORK"
                    ),
                ),
            ),
        )
        if node.id == HOURS_PHASE
        else node
        for node in compiled.graph.nodes
    )
    compiled = compile_graph(replace(compiled.graph, nodes=nodes))
    with pytest.raises(NodeRejectedError, match="omits available raw or incumbent"):
        run_graph(compiled, sources=sources, store=store, kernels=kernels)


@pytest.mark.parametrize(
    "tamper", ("pass_flag", "missing_count", "context", "producer", "schema")
)
def test_gate_rejects_forged_or_misbound_producer_observations(tamper):
    frame = _frame()
    observed = pool.observe_multispine_hours_worked_inputs(frame)
    payload = encode_us_frame_context(observed.frame)
    evidence = {
        "schema_version": 1,
        "frame_context_sha256": hashlib.sha256(payload).hexdigest(),
        "source_preparation": observed.receipt,
    }
    report = evidence["source_preparation"]["suboperators"][0]["kernel_receipt"][
        "hours_worked_signal_gate"
    ]
    if tamper == "pass_flag":
        report["passed"] = False
    elif tamper == "missing_count":
        del report["details"]["unique_counts"]["weeks_worked"]
    elif tamper == "context":
        evidence["frame_context_sha256"] = "0" * 64
    elif tamper == "schema":
        evidence["schema_version"] = 2
    node = us_hours_nodes(
        frame_column_declarations(frame), base="source", context_producer="source"
    )[-1]
    scope = NumericScope(Numeric.BITWISE)
    context = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series(dtype="string"),
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts={
            "frame_context": ArtifactValue(
                payload,
                US_FRAME_CONTEXT_TYPE,
                "a" * 64,
                "b" * 64 if tamper == "producer" else "c" * 64,
                scope,
            ),
            "hours_evidence": ArtifactValue(
                canonical_json(evidence),
                US_HOURS_EVIDENCE_TYPE,
                "d" * 64,
                "c" * 64,
                scope,
            ),
        },
    )
    with pytest.raises(ValueError, match="(evidence|summary|producer)"):
        USHoursGateKernel().run(context)
