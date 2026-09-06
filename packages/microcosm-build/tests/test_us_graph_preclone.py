"""All six nominal pre-clone operators, including the actual rent fit and gate."""

import hashlib
import importlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import graph_early_preparation as hours
from microcosm.build.us_runtime.graph_person_preparation import ELIGIBILITY_NODE
from microcosm.graph import (
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
from microcosm.graph.codecs import SourceCodecRegistry, load_frame_store


def _housing_fixture():
    spec = importlib.util.spec_from_file_location(
        "_preclone_housing_fixture",
        Path(__file__).with_name("test_us_graph_housing_source.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


h = _housing_fixture()
GOLDEN = Path(__file__).with_name("golden") / "us_preclone_source_receipt_f019.json"
STAGES = (
    "early_preparation.cps_carried",
    "early_preparation.hours.gate",
    "early_preparation.prior_year",
    "early_preparation.relationship",
    h.HOUSING_SOURCE_ACCEPTED_CONTEXT,
    ELIGIBILITY_NODE,
)


def _composition():
    return importlib.import_module("microcosm.build.us_runtime.graph_preclone")


def _key(frame):
    document = h._snapshot_frame(frame)
    document["strata_dtype"] = str(frame.strata.dtype)
    document["strata_name"] = frame.strata.name
    return hashlib.sha256(canonical_json(document)).hexdigest()


def _setup(tmp_path, frame, *, transform=None):
    composition = _composition()
    donor_path = h._donor(tmp_path)
    donor_sha = h._sha(donor_path)
    descriptor = h.rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=donor_sha
    )
    columns = h.frame_column_declarations(frame)
    source = Node(
        "source",
        h._Create.ref,
        sources=("population",),
        structural=StructuralDelta.CREATE,
        outputs=columns,
        params={"frame_context": h.encode_us_frame_context(frame).decode()},
        artifact_outputs=(ArtifactOutput("frame_context", h.US_FRAME_CONTEXT_TYPE),),
    )
    nodes = (
        source,
        *composition.us_preclone_source_nodes(
            columns,
            base="source",
            context_producer="source",
            donor_descriptor=descriptor,
            n_estimators=2,
        ),
    )
    if transform is not None:
        nodes = transform(nodes)
    codecs = SourceCodecRegistry()
    codecs.register("frame-store", load_frame_store)
    codecs.register(
        h.rent_source.RENT_SOURCE_CODEC,
        partial(h.rent_source.load_graph_acs_rent_source, expected_sha256=donor_sha),
    )
    kernels = KernelRegistry()
    kernels.register(h._Create())
    composition.register_us_preclone_source_kernels(kernels, source_codecs=codecs)
    store = ContentStore(tmp_path / "store", codecs=codecs)
    population = ContentStore(tmp_path / "inputs").put_frame(_key(frame), frame)
    compiled = compile_graph(
        Graph(
            country="us",
            sources=(SourceRef("population", "frame-store"), h.rent_source.RENT_SOURCE),
            nodes=nodes,
        )
    )
    sources = {"population": population, h.rent_source.RENT_SOURCE_NAME: donor_path}
    return compiled, sources, store, kernels, nodes, donor_path


def _run(tmp_path, frame, *, resume="forbid", transform=None):
    compiled, sources, store, kernels, nodes, donor = _setup(
        tmp_path, frame, transform=transform
    )
    manifest = run_graph(
        compiled, sources=sources, store=store, kernels=kernels, resume=resume
    )
    return manifest, store, nodes, donor


def _restored(manifest, store):
    context = h._payload(manifest, store, ELIGIBILITY_NODE, "frame_context")
    return h._restore(
        manifest.population(f"{ELIGIBILITY_NODE}.boundary"), context
    ), context


def _direct(frame, donor):
    operators = h._prefix_operators()
    operators["with_us_housing_inputs"] = lambda current: h.pool.with_us_housing_inputs(
        current,
        seed=h.SEED,
        time_period=h.pool.POOL_TIME_PERIOD,
        acs_rent_donor=donor,
        n_estimators=2,
    )
    operators["with_us_eligibility_inputs"] = lambda current: (
        h.pool.with_us_eligibility_inputs(
            current,
            seed=h.SEED,
            time_period=h.pool.POOL_TIME_PERIOD,
        )
    )
    return h.pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=tuple(operators),
        operators=operators,
    )


def _complete_receipt(manifest):
    receipts = [manifest.nodes[node].receipt["source_preparation"] for node in STAGES]
    evidence = receipts[0]["cps_source_evidence"]
    assert all(receipt["cps_source_evidence"] == evidence for receipt in receipts)
    order = h.pool.POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
    suboperators = []
    for index, (name, receipt) in enumerate(zip(order, receipts, strict=True)):
        assert receipt["phase"] == "pre_clone"
        assert tuple(receipt["operator_order"]) == (name,)
        assert len(receipt["suboperators"]) == 1
        suboperators.append({**receipt["suboperators"][0], "order_index": index})
    return h.pool._source_chain_receipt(
        phase="pre_clone",
        operator_names=order,
        evidence_rows=evidence["person_rows"],
        receipts=suboperators,
        output_families=h.pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
    )


def test_full_nominal_preclone_cold_auto_require_matches_authenticated_original(
    tmp_path,
):
    frame = h._pool_frame()
    golden_frame = json.loads(h.GOLDEN.read_text())
    original = json.loads(GOLDEN.read_text())
    assert h._snapshot_frame(frame) == golden_frame["pool"]
    assert (
        original["runtime_source_commit"] == "f01964c7c71e1ae575ac4c16766dbdaed1a342d5"
    )
    for resume in ("forbid", "auto", "require"):
        manifest, store, nodes, donor_path = _run(tmp_path, frame, resume=resume)
        actual, context = _restored(manifest, store)
        donor = h.housing_inputs.load_acs_2022_rent_donor(
            donor_path, expected_sha256=h._sha(donor_path)
        )
        direct = _direct(frame, donor)
        h._assert_frames_equal(actual, direct.frame)
        assert context == h.encode_us_frame_context(direct.frame)
        assert h._snapshot_frame(actual) == golden_frame["eligibility"]["frame"]
        canonical_frame = json.dumps(
            h._snapshot_frame(actual),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        assert (
            hashlib.sha256(canonical_frame).hexdigest()
            == original["final_frame_canonical_sha256"]
        )
        assert canonical_json(_complete_receipt(manifest)) == canonical_json(
            original["receipt"]
        )
        assert canonical_json(_complete_receipt(manifest)) == canonical_json(
            direct.receipt
        )
        assert manifest.nodes[STAGES[1]].receipt["outcome"] == "pass"
        assert (
            manifest.nodes[h.HOUSING_SOURCE_ACCEPTED_CONTEXT].receipt["outcome"]
            == "pass"
        )
        assert manifest.nodes[ELIGIBILITY_NODE].receipt["source_preparation"]
        assert not any("seam" in node.id for node in nodes)
        if resume != "forbid":
            assert all(node.hit for node in manifest.nodes.values())


def test_composition_uses_real_accepted_predecessors_and_full_pool_after_housing(
    tmp_path,
):
    _, _, _, _, nodes, _ = _setup(tmp_path, h._pool_frame())
    by_id = {node.id: node for node in nodes}
    expected = {
        "early_preparation.prior_year.boundary": "early_preparation.hours.gate",
        "early_preparation.relationship.boundary": "early_preparation.prior_year",
        h.HOUSING_SOURCE_POPULATION: "early_preparation.relationship",
        f"{ELIGIBILITY_NODE}.boundary": h.HOUSING_SOURCE_ACCEPTED_CONTEXT,
    }
    for node_id, producer in expected.items():
        edge = next(
            edge
            for edge in by_id[node_id].artifact_inputs
            if edge.name == "frame_context"
        )
        assert edge.producer == producer
        assert edge.type == h.US_FRAME_CONTEXT_TYPE
    assert (
        by_id[f"{ELIGIBILITY_NODE}.boundary"].base == h.HOUSING_SOURCE_MERGE_POPULATION
    )
    assert by_id[ELIGIBILITY_NODE].kernel == "us.production.eligibility@1"
    assert f"{h.rent.RENT_PREFIX}.fit.000" in by_id


@pytest.mark.parametrize("mutation", ["omit", "reorder"])
def test_composition_refuses_drift_from_registered_six_stage_order(
    tmp_path, monkeypatch, mutation
):
    order = h.pool.POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
    changed = (
        order[:-1] if mutation == "omit" else (*order[:3], order[4], order[3], order[5])
    )
    monkeypatch.setattr(h.pool, "POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER", changed)
    with pytest.raises(ValueError, match="operator order"):
        _setup(tmp_path, h._pool_frame())


@pytest.mark.parametrize("gate", ["hours", "housing"])
def test_failed_actual_gate_withholds_later_stages_on_all_replay_modes(
    tmp_path, monkeypatch, gate
):
    kernel = (
        hours.USHoursGateKernel
        if gate == "hours"
        else h.graph_housing_source.USHousingSourceGateKernel
    )
    original = kernel.run

    def failed(self, context):
        original(self, context)
        raise ValueError("injected complete-chain acceptance failure")

    monkeypatch.setattr(kernel, "run", failed)
    for resume in ("forbid", "auto", "require"):
        manifest, _, _, _ = _run(tmp_path, h._pool_frame(), resume=resume)
        failed_id = STAGES[1] if gate == "hours" else h.HOUSING_SOURCE_ACCEPTED_CONTEXT
        assert manifest.nodes[failed_id].receipt["outcome"] == "fail"
        downstream = STAGES[2:] if gate == "hours" else (ELIGIBILITY_NODE,)
        h._assert_unreached(manifest, downstream)


@pytest.mark.parametrize("mutation", ["age", "person_order"])
def test_recipient_changes_recompute_all_later_stages_and_reuse_real_donor_fit(
    tmp_path, monkeypatch, mutation
):
    calls = []
    fit = h.graph_legacy_train.fit_target

    def counted(*args, **kwargs):
        calls.append(1)
        return fit(*args, **kwargs)

    monkeypatch.setattr(h.graph_legacy_train, "fit_target", counted)
    frame = h._pool_frame()
    first, _, _, _ = _run(tmp_path, frame)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    if mutation == "age":
        row = person.index[person.PERIDNUM.notna()][0]
        person.loc[row, "A_AGE"] = "51"
    else:
        tables["person"] = person.iloc[[7, 3, 0, 6, 5, 1, 4, 2]]
    changed = h.Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata.reindex(tables["person"].index),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    second, store, _, donor_path = _run(tmp_path, changed, resume="auto")
    assert first.nodes["source"].key != second.nodes["source"].key
    for node_id in (
        *STAGES,
        f"{h.rent.RENT_PREFIX}.recipient",
        f"{h.rent.RENT_PREFIX}.apply.000",
    ):
        assert first.nodes[node_id].key != second.nodes[node_id].key, node_id
        assert not second.nodes[node_id].hit, node_id
    fit_id = f"{h.rent.RENT_PREFIX}.fit.000"
    assert calls == [1]
    assert first.nodes[fit_id].key == second.nodes[fit_id].key
    assert second.nodes[fit_id].hit
    assert second.nodes[f"{h.rent.RENT_PREFIX}.donor"].hit
    actual, _ = _restored(second, store)
    donor = h.housing_inputs.load_acs_2022_rent_donor(
        donor_path, expected_sha256=h._sha(donor_path)
    )
    h._assert_frames_equal(actual, _direct(changed, donor).frame)
    np.testing.assert_array_equal(actual.person.person_id, changed.person.person_id)
