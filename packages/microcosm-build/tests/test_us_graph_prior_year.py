"""Pre-clone prior-year joins retain the actual CPS/hours source sequence."""

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import multispine_pool as pool
from microcosm.build.us_runtime import prior_year_income as prior
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    _mass_records,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_cps_carried import (
    register_us_cps_carried_kernels,
    us_cps_carried_nodes,
)
from microcosm.build.us_runtime.graph_early_preparation import (
    register_us_hours_kernels,
    us_hours_nodes,
)
from microcosm.build.us_runtime.graph_prior_year import (
    _PRIOR_IMPLEMENTATION_MODULES,
    PRIOR_YEAR_DEPENDENCIES,
    PRIOR_YEAR_NODE,
    USPriorYearSourceKernel,
    register_us_prior_year_source_kernels,
    us_prior_year_source_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.errors import NodeRejectedError


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "_prior_cps_fixtures", Path(__file__).with_name("test_us_graph_cps_carried.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cps = _fixtures()
OPERATORS = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
    "with_us_prior_year_income_inputs",
)


def _operators():
    return {
        OPERATORS[0]: pool.derive_us_cps_carried_inputs,
        OPERATORS[1]: pool._with_gated_us_hours_worked_inputs,
        OPERATORS[2]: lambda frame: prior.with_us_prior_year_income_inputs(
            frame, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
        ),
    }


def _direct(frame, operators=OPERATORS):
    return pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=operators,
        operators={name: _operators()[name] for name in operators},
    )


def _setup(tmp_path, frame, *, compose=True):
    store = ContentStore(tmp_path / "store")
    source = ContentStore(tmp_path / "inputs").put_frame("a" * 64, frame)
    columns = frame_column_declarations(frame)
    nodes = [
        Node(
            "source",
            cps.hours._Create.ref,
            sources=("population",),
            structural=StructuralDelta.CREATE,
            outputs=columns,
            params={"frame_context": encode_us_frame_context(frame).decode()},
            artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
        )
    ]
    base = context_producer = "source"
    if compose:
        for factory in (us_cps_carried_nodes, us_hours_nodes):
            produced = factory(columns, base=base, context_producer=context_producer)
            nodes.extend(produced)
            inventory = {(value.entity, value.column): value for value in columns}
            for node in produced:
                for value in node.outputs:
                    inventory[value.entity, value.column] = value
            columns = tuple(inventory.values())
            base, context_producer = produced[0].id, produced[-1].id
    nodes.extend(
        us_prior_year_source_nodes(
            columns, base=base, context_producer=context_producer
        )
    )
    nodes.append(
        Node(
            "after",
            cps.hours._Consumer.ref,
            population=f"{PRIOR_YEAR_NODE}.boundary",
            artifact_inputs=(
                ArtifactInput(
                    "frame_context",
                    PRIOR_YEAR_NODE,
                    "frame_context",
                    US_FRAME_CONTEXT_TYPE,
                ),
            ),
        )
    )
    registry = KernelRegistry()
    registry.register(cps.hours._Create())
    registry.register(cps.hours._Consumer())
    register_us_cps_carried_kernels(registry)
    register_us_hours_kernels(registry)
    register_us_prior_year_source_kernels(registry)
    return (
        compile_graph(
            Graph(
                country="us",
                sources=(SourceRef("population", "frame-store"),),
                nodes=tuple(nodes),
            )
        ),
        {"population": source},
        store,
        registry,
    )


def _forbid_qrf(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The pre-clone source join must never construct QRF")

    monkeypatch.setattr(prior, "QRF", forbidden)


def test_actual_cps_hours_prior_composition_and_exact_required_replay(
    tmp_path, monkeypatch
):
    _forbid_qrf(monkeypatch)
    assert pool.POOL_SOURCE_OPERATOR_ORDER[:3] == OPERATORS
    frame = cps._frame()
    expected = _direct(frame)
    individual = []
    current = frame
    for name in OPERATORS:
        step = _direct(current, (name,))
        individual.append(step.receipt)
        current = step.frame
    cps._assert_equal(current, expected.frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    for resume in ("forbid", "auto", "require"):
        manifest = run_graph(
            compiled, sources=sources, store=store, kernels=kernels, resume=resume
        )
        actual = manifest.population(f"{PRIOR_YEAR_NODE}.boundary")
        cps._assert_equal(actual, expected.frame)
        for node_id, receipt in zip(
            (
                "early_preparation.cps_carried",
                "early_preparation.hours.gate",
                PRIOR_YEAR_NODE,
            ),
            individual,
            strict=True,
        ):
            assert canonical_json(
                manifest.node(node_id).receipt["source_preparation"]
            ) == canonical_json(receipt)
        node = manifest.node(PRIOR_YEAR_NODE)
        context_payload = store.load_bytes(node.opaque_artifacts["frame_context"])
        assert context_payload == encode_us_frame_context(expected.frame)
        document = json.loads(context_payload)
        restored = Frame(
            {entity: actual.table(entity) for entity in actual.entities},
            actual.schema,
            {entity: actual.weights_for(entity) for entity in actual.weighted_entities},
            actual.strata,
            metadata=document["metadata"],
            mass_log=_mass_records(document["mass_log"]),
        )
        output_store = ContentStore(tmp_path / f"restored-{resume}")
        output_store.put_frame("b" * 64, restored)
        output_store.put_frame("c" * 64, expected.frame)
        assert (
            output_store.metadata("b" * 64)["payloads"]
            == output_store.metadata("c" * 64)["payloads"]
        )
        assert manifest.node("after").receipt["received_context"]
        if resume != "forbid":
            assert all(node.hit for node in manifest.nodes.values())
    receipt = individual[-1]
    assert receipt["transient_outputs_carried_through_clone"] == {
        "person": ["employment_income_last_year"]
    }
    assert receipt["suboperators"][0]["formula_owned_outputs_removed"] == {}
    assert "employment_income_last_year" in actual.person
    mask = actual.person.PERIDNUM.notna()
    assert actual.person.loc[mask, "employment_income_last_year"].tolist() == [
        0.0,
        0.0,
        40000.0,
        0.0,
    ]
    assert actual.person.loc[mask, "previous_year_income_available"].tolist() == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.parametrize(
    "surface", ("all_null", "partial", "pass_through", "shuffled_person", "signed_loss")
)
def test_actual_prior_source_presence_null_authority_and_order(
    tmp_path, monkeypatch, surface
):
    _forbid_qrf(monkeypatch)
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    mask = frame.person.PERIDNUM.notna()
    if surface == "shuffled_person":
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = frame.person.iloc[[3, 0, 5, 2, 1, 4]].copy()
        frame = Frame(
            tables,
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata.reindex(tables["person"].index),
            metadata=frame.metadata,
            mass_log=frame.mass_log,
        )
    elif surface == "signed_loss":
        frame.person.loc[mask, "SEMP_VAL"] = [-500.0, 0.0, 700.0, 0.0]
    else:
        for name in prior.US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS:
            frame.person[name] = (
                pd.Series(pd.NA, index=frame.person.index, dtype="boolean")
                if name == "previous_year_income_available"
                else np.nan
            )
        frame.person.loc[~mask, "employment_income_last_year"] = [120.0, 340.0]
        frame.person.loc[~mask, "previous_year_income_available"] = [True, False]
        if surface != "all_null":
            frame.person.loc[frame.person.index[0], "employment_income_last_year"] = (
                999.0
            )
            frame.person.loc[
                frame.person.index[0], "self_employment_income_last_year"
            ] = 777.0
            frame.person.loc[
                frame.person.index[0], "previous_year_income_available"
            ] = True
            frame.person.drop(columns="I_SEVAL", inplace=True)
        if surface == "pass_through":
            frame.person.loc[mask, "employment_income_last_year"] = [5.0, 6.0, 7.0, 8.0]
            frame.person.loc[mask, "self_employment_income_last_year"] = [
                1.0,
                2.0,
                3.0,
                4.0,
            ]
            frame.person.loc[mask, "previous_year_income_available"] = [
                True,
                False,
                True,
                False,
            ]
    expected = _direct(frame, OPERATORS[2:])
    args, sources, store, kernels = _setup(tmp_path, frame, compose=False)
    result = run_graph(args, sources=sources, store=store, kernels=kernels)
    cps._assert_equal(result.population(f"{PRIOR_YEAR_NODE}.boundary"), expected.frame)
    assert canonical_json(
        result.node(PRIOR_YEAR_NODE).receipt["source_preparation"]
    ) == canonical_json(expected.receipt)
    assert store.load_bytes(
        result.node(PRIOR_YEAR_NODE).opaque_artifacts["frame_context"]
    ) == encode_us_frame_context(expected.frame)


def test_registered_person_observer_uses_actual_selected_weights_and_ids(monkeypatch):
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    expected = _direct(frame, OPERATORS[2:])
    observed = []
    original = pool.prepare_us_prior_year_person

    def capture(person, *, weights, **kwargs):
        observed.append((person.copy(), weights))
        return original(person, weights=weights, **kwargs)

    monkeypatch.setattr(pool, "prepare_us_prior_year_person", capture)
    result, receipt = pool.observe_multispine_prior_year_person(
        frame.person,
        weights=frame.resolve_weights("person"),
        metadata=frame.metadata,
        pool_rows={entity: frame.n(entity) for entity in frame.entities},
    )
    pd.testing.assert_frame_equal(result, expected.frame.person)
    assert canonical_json(receipt) == canonical_json(expected.receipt)
    selected, weights = observed[0]
    mask = frame.person.PERIDNUM.notna()
    assert selected.person_id.tolist() == frame.person.loc[mask, "person_id"].tolist()
    assert selected.index.equals(frame.person.index[mask])
    assert weights.kind == frame.resolve_weights("person").kind
    assert (
        weights.values.tobytes()
        == frame.resolve_weights("person").values[mask].tobytes()
    )
    assert len(set(weights.values)) > 1


@pytest.mark.parametrize(
    "failure",
    (
        "missing_raw",
        "duplicate_join_key",
        "invalid_year",
        "negative_wages",
        "support_context",
    ),
)
def test_original_source_refusal_or_preclone_misuse_refuses(
    tmp_path, monkeypatch, failure
):
    _forbid_qrf(monkeypatch)
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    mask = frame.person.PERIDNUM.notna()
    if failure == "missing_raw":
        frame.person.drop(columns="I_SEVAL", inplace=True)
    elif failure == "duplicate_join_key":
        ids = frame.person.index[mask]
        frame.person.loc[ids[1], "PERIDNUM"] = frame.person.loc[ids[0], "PERIDNUM"]
    elif failure == "invalid_year":
        frame.person.loc[frame.person.index[mask][0], "source_year"] = "invalid"
    elif failure == "negative_wages":
        frame.person.loc[frame.person.index[mask][0], "WSAL_VAL"] = -5.0
    else:
        from microcosm.build.us_runtime.support_provenance import (
            support_clone_index_column,
        )

        frame.person.loc[
            frame.person.index[0], support_clone_index_column("person")
        ] = 1
    with pytest.raises((ValueError, SourceRuntimeError)) as direct:
        _direct(frame, OPERATORS[2:])
    args, sources, store, kernels = _setup(tmp_path, frame, compose=False)
    with pytest.raises(NodeRejectedError) as graph:
        run_graph(args, sources=sources, store=store, kernels=kernels)
    assert str(direct.value) in str(graph.value)


def test_prior_depends_on_accepted_hours_context_not_pending(tmp_path):
    args, _, _, _ = _setup(tmp_path, cps._frame())
    boundary = next(
        node for node in args.graph.nodes if node.id == f"{PRIOR_YEAR_NODE}.boundary"
    )
    assert boundary.artifact_inputs[0].producer == "early_preparation.hours.gate"
    changed = replace(
        boundary,
        artifact_inputs=(
            replace(boundary.artifact_inputs[0], producer="early_preparation.hours"),
        ),
    )
    with pytest.raises(ValueError, match="type"):
        compile_graph(
            replace(
                args.graph,
                nodes=tuple(
                    changed if node.id == boundary.id else node
                    for node in args.graph.nodes
                ),
            )
        )


def test_failed_actual_hours_gate_leaves_prior_unreached_on_every_resume(
    tmp_path, monkeypatch
):
    _forbid_qrf(monkeypatch)
    frame = cps._frame()
    mask = frame.person.PERIDNUM.notna()
    frame.person.loc[mask, ["HRSWK", "A_HRS1", "WKSWORK"]] = 0.0
    args, sources, store, kernels = _setup(tmp_path, frame)
    for resume in ("forbid", "auto", "require"):
        result = run_graph(
            args, sources=sources, store=store, kernels=kernels, resume=resume
        )
        assert result.node("early_preparation.hours.gate").receipt["outcome"] == "fail"
        for name in (f"{PRIOR_YEAR_NODE}.boundary", PRIOR_YEAR_NODE, "after"):
            assert result.node(name).receipt["execution"]["state"] == "unreached"
            assert result.node(name).opaque_artifacts == {}


def test_present_raw_read_and_output_ownership_cannot_be_omitted(tmp_path):
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    args, sources, store, kernels = _setup(tmp_path, frame, compose=False)
    derive = next(node for node in args.graph.nodes if node.id == PRIOR_YEAR_NODE)
    for changed in (
        replace(
            derive,
            inputs=tuple(
                replace(
                    value,
                    columns=tuple(name for name in value.columns if name != "I_SEVAL"),
                )
                for value in derive.inputs
            ),
        ),
        replace(derive, outputs=derive.outputs[:-1]),
        replace(derive, params={"phase": "post_clone"}),
    ):
        compiled = compile_graph(
            replace(
                args.graph,
                nodes=tuple(
                    changed if node.id == derive.id else node
                    for node in args.graph.nodes
                ),
            )
        )
        with pytest.raises(
            NodeRejectedError,
            match="omits available|must own exactly|authored outer phase",
        ):
            run_graph(compiled, sources=sources, store=store, kernels=kernels)


@pytest.mark.parametrize("field", ("identity", "counts", "weights"))
def test_malformed_full_context_refuses(tmp_path, field):
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    args, sources, store, kernels = _setup(tmp_path, frame, compose=False)
    source = args.graph.nodes[0]
    doc = json.loads(source.params["frame_context"])
    if field == "identity":
        doc["entities"]["person"]["ordered_ids_sha256"] = "0" * 64
    elif field == "counts":
        doc["entities"]["household"]["rows"] += 1
    else:
        doc["weight_sources"]["person"] = "design"
    source = replace(source, params={"frame_context": canonical_json(doc).decode()})
    changed = compile_graph(replace(args.graph, nodes=(source, *args.graph.nodes[1:])))
    with pytest.raises(
        NodeRejectedError,
        match="identity differs|counts differ|one explicit weight source",
    ):
        run_graph(changed, sources=sources, store=store, kernels=kernels)


@pytest.mark.parametrize("dtype", ("bool", "float64"))
def test_incompatible_boolean_incumbent_refuses_before_execution(tmp_path, dtype):
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    frame.person["previous_year_income_available"] = pd.Series(
        False if dtype == "bool" else np.nan, index=frame.person.index, dtype=dtype
    )
    with pytest.raises(ValueError, match="requires nullable boolean"):
        _setup(tmp_path, frame, compose=False)


@pytest.mark.parametrize("module", _PRIOR_IMPLEMENTATION_MODULES)
def test_actual_module_bytes_bind_prior_implementation(monkeypatch, module):
    kernel = USPriorYearSourceKernel()
    before = kernel.implementation_hash()
    target = Path(import_module(module).__file__).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b"\n# independent revision\n"
            if path.resolve() == target
            else original(path)
        ),
    )
    assert kernel.implementation_hash() != before


@pytest.mark.parametrize("dependency", PRIOR_YEAR_DEPENDENCIES)
def test_dependency_versions_bind_prior_implementation(monkeypatch, dependency):
    from microcosm.graph import kernel as graph_kernel

    kernel = USPriorYearSourceKernel()
    before = kernel.implementation_hash()
    original = graph_kernel.importlib_metadata.version
    monkeypatch.setattr(
        graph_kernel.importlib_metadata,
        "version",
        lambda name: (
            original(name) + ".probe" if name == dependency else original(name)
        ),
    )
    assert kernel.implementation_hash() != before


def test_source_resource_and_live_projection_bind_actual_prior_receipt(monkeypatch):
    from importlib.resources import files

    kernel = USPriorYearSourceKernel()
    before = kernel.implementation_hash()
    target = Path(files("microcosm.build.us").joinpath("source_stages.json")).resolve()
    original = Path.read_bytes
    with monkeypatch.context() as patch:
        patch.setattr(
            Path,
            "read_bytes",
            lambda path: (
                original(path) + b" " if path.resolve() == target else original(path)
            ),
        )
        assert kernel.implementation_hash() != before
    frame = _direct(cps._frame(), OPERATORS[:2]).frame

    def observe():
        return pool.observe_multispine_prior_year_person(
            frame.person,
            weights=frame.resolve_weights("person"),
            metadata=frame.metadata,
            pool_rows={entity: frame.n(entity) for entity in frame.entities},
        )[1]

    old_receipt = observe()
    monkeypatch.setattr(
        pool,
        "_FORMULA_OWNED_SOURCE_OUTPUTS",
        {
            **pool._FORMULA_OWNED_SOURCE_OUTPUTS,
            "person": frozenset(
                set(pool._FORMULA_OWNED_SOURCE_OUTPUTS["person"])
                - {"employment_income_last_year"}
            ),
        },
    )
    assert (
        observe()["transient_outputs_carried_through_clone"]
        != old_receipt["transient_outputs_carried_through_clone"]
    )
    assert kernel.implementation_hash() != before


def test_live_output_registry_binds_prior_implementation(monkeypatch):
    kernel = USPriorYearSourceKernel()
    before = kernel.implementation_hash()
    monkeypatch.setitem(
        pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
        "prior_year_income",
        {"person": frozenset({"previous_year_income_available"})},
    )
    assert kernel.implementation_hash() != before


def test_full_legacy_composition_retains_pre_edit_byte_fingerprints(tmp_path):
    output = _direct(cps._frame())
    store = ContentStore(tmp_path)
    store.put_frame("b" * 64, output.frame)
    actual = {
        "payload": hashlib.sha256(
            canonical_json(store.metadata("b" * 64)["payloads"])
        ).hexdigest(),
        "receipt": hashlib.sha256(canonical_json(output.receipt)).hexdigest(),
        "context": hashlib.sha256(encode_us_frame_context(output.frame)).hexdigest(),
    }
    # Measured on unmodified helper successor 763926d2, before graph/pool edits.
    assert actual == {
        "payload": "0fd587d2a606351c04f7c86853b4067143f61888ccc0da706ad9cae1c3f210fe",
        "receipt": "fc9efd13de468e71c6a1d88c39d90a8be56a7f247b3eed44aa089a228fbeb127",
        "context": "bec4179534957997a7df1a06a5a22654f72b992d2568e0dd869bf6963add0c51",
    }


def test_multiple_explicit_weight_sources_refuse(tmp_path):
    from microcosm.frame import Weights

    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    frame = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": frame.weights_for("household"),
            "person": Weights(
                frame.resolve_weights("person").values,
                frame.resolve_weights("person").kind,
            ),
        },
        frame.strata,
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )
    args, sources, store, kernels = _setup(tmp_path, frame, compose=False)
    with pytest.raises(NodeRejectedError, match="one explicit weight source"):
        run_graph(args, sources=sources, store=store, kernels=kernels)


def test_registered_projection_rejects_helper_identity_change(monkeypatch):
    frame = _direct(cps._frame(), OPERATORS[:2]).frame
    original = pool.prepare_us_prior_year_person

    def invalid(person, **kwargs):
        output = original(person, **kwargs).copy()
        output.loc[output.index[0], "person_id"] = -111
        return output

    monkeypatch.setattr(pool, "prepare_us_prior_year_person", invalid)
    with pytest.raises(ValueError, match="identity|IDs|id"):
        pool.observe_multispine_prior_year_person(
            frame.person,
            weights=frame.resolve_weights("person"),
            metadata=frame.metadata,
            pool_rows={entity: frame.n(entity) for entity in frame.entities},
        )
