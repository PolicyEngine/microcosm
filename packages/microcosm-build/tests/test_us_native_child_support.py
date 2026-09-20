"""Invented source, received-only model and explicit canonical rewrite checks."""

import json
import shutil
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_child_support_source as source
from microcosm.build.us_runtime import current_survey_amounts as amounts
from microcosm.build.us_runtime import graph_us_survey_enrichment as graph
from microcosm.fit import model_input


@pytest.mark.parametrize("option", [None, 0, 1, "true", [], {}])
def test_child_original_option_is_strict_before_owner(option):
    with pytest.raises(ValueError, match="FULL_ORIGINAL_OPTION"):
        source.qualify_current_asec_child_support(object(), full_original=option)


def test_child_group_requires_explicit_full_original_support():
    with pytest.raises(ValueError, match="CHILD_FULL_ORIGINAL_REQUIRED"):
        amounts.qualify_current_survey_amounts(object(), groups=("child_support",))
    spec = amounts.selected_groups(("child_support",))[0]
    assert spec.fields == (("CSP_VAL", "child_support_received"),)
    assert len(spec.targets) == 1
    assert "CHSP" not in spec.targets[0]


def _child_family():
    from test_us_native_workers_compensation import _invented_wc_family

    qualified, receiving = _invented_wc_family()
    spec = amounts.selected_groups(("child_support",))[0]
    native = qualified.native.rename(
        columns={"workers_compensation": "child_support_received"}
    )
    native["child_support_expense"] = [np.nan, np.nan, 2000.0, np.nan]
    reports = qualified.reports.rename(
        columns={"survey_current_WC_VAL_origin": "survey_current_CSP_VAL_origin"}
    )
    reports["survey_current_CHSP_VAL_origin"] = pd.Series(
        [
            "unresolved",
            "unresolved",
            "observed_positive_payment",
            "declared_niu_amount",
        ],
        index=native.index,
        dtype="string",
    )
    old = qualified.groups[0]
    cols = old.donor_columns.rename(columns={old.spec.targets[0]: spec.targets[0]})
    group = replace(old, spec=spec, donor_columns=cols)
    return replace(
        qualified, native=native, reports=reports, groups=(group,)
    ), receiving


def _draw(qualified, value=120.0):
    group = qualified.groups[0]
    return {
        "child_support": pd.DataFrame(
            {group.spec.targets[0]: [value]},
            index=model_input.decode_recipient_matrix(group.matrix).features.index,
            dtype="float64",
        )
    }


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("carried", [120.0, -999.0])
def test_rewrite_replaces_both_outputs_without_trusting_carried_values(dtype, carried):
    qualified, receiving = _child_family()
    for name in amounts.child_mapper.US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        receiving.person[name] = pd.Series(
            carried, index=receiving.person.index, dtype=dtype
        )
    before = receiving.person.copy(deep=True)
    source_before = qualified.source_frame.person.copy(deep=True)
    weights = receiving.weights_for("household").values.tobytes()
    columns = amounts.attach_columns(qualified, receiving, _draw(qualified))
    received = columns["person", "child_support_received"]
    paid = columns["person", "child_support_expense"]
    assert (
        received.loc[2]
        == received.loc[12]
        == received.loc[1]
        == received.loc[11]
        == 120
    )
    assert received.loc[3] == received.loc[13] == 0
    assert np.isnan(received.loc[0]) and np.isnan(received.loc[10])
    assert paid.loc[2] == paid.loc[12] == 2000
    assert paid.drop([2, 12]).isna().all()
    assert str(received.dtype) == str(paid.dtype) == dtype
    pd.testing.assert_frame_equal(receiving.person, before, check_exact=True)
    pd.testing.assert_frame_equal(
        qualified.source_frame.person, source_before, check_exact=True
    )
    assert receiving.weights_for("household").values.tobytes() == weights
    owned = {
        o.column: o
        for o in graph.amount_nodes(
            qualified, receiving, parent_digest="a" * 64, n_estimators=2
        )[-1].outputs
    }
    for name in amounts.child_mapper.US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        assert owned[name].rewrite and owned[name].dtype == dtype


@pytest.mark.parametrize(
    "dtype", ["int64", "Int64", "bool", "boolean", "string", "Float64"]
)
def test_unsupported_child_rewrite_storage_refuses(dtype):
    qualified, receiving = _child_family()
    receiving.person["child_support_received"] = pd.Series(
        "1" if dtype == "string" else 1, index=receiving.person.index, dtype=dtype
    )
    with pytest.raises(ValueError, match="CHILD_REWRITE_DTYPE"):
        amounts.attach_columns(qualified, receiving, _draw(qualified))
    with pytest.raises(ValueError, match="CHILD_REWRITE_DTYPE"):
        graph.amount_nodes(qualified, receiving, parent_digest="a" * 64, n_estimators=2)


def test_lossy_child_rewrite_refuses_and_other_collisions_still_refuse():
    qualified, receiving = _child_family()
    receiving.person["child_support_received"] = np.zeros(
        len(receiving.person), dtype="float32"
    )
    with pytest.raises(ValueError, match="CHILD_REWRITE_LOSS"):
        amounts.attach_columns(qualified, receiving, _draw(qualified, 0.1))
    receiving.person["survey_current_CSP_VAL_origin"] = "inherited"
    with pytest.raises(ValueError, match="ATTACH_OWNERSHIP_COLLISION"):
        amounts.attach_columns(qualified, receiving, _draw(qualified))


def _full_original_child_source_and_donor(tmp_path, monkeypatch):
    import test_us_survey_population_preparation as preparation_fixture
    from test_us_current_asec_child_support_source import child_support_arguments

    original_fixture = preparation_fixture.fixture
    monkeypatch.setattr(
        preparation_fixture,
        "fixture",
        lambda *args, **kwargs: original_fixture(*args, **kwargs, zero=False),
    )
    args = child_support_arguments(tmp_path, monkeypatch)
    selected_dir = tmp_path / "partial"
    shutil.copytree(args["source_dir"], selected_dir)
    request = json.loads((selected_dir / "selection-request.json").read_bytes())
    request["fraction"], request["seed"] = [2, 3], 41
    (selected_dir / "selection-request.json").write_text(
        json.dumps(request, sort_keys=True, separators=(",", ":"))
    )
    snapshots = tmp_path / "partial-captures"
    snapshots.mkdir()
    preparation = source.routing.source.prepare_authenticated_survey_population(
        source_dir=selected_dir,
        snapshot_root=snapshots,
        fraction=Fraction(2, 3),
        seed=41,
    )
    selected = source.qualify_current_asec_child_support(preparation)
    full = source.qualify_current_asec_child_support(preparation, full_original=True)
    assert 105 not in selected.person.native_person_id.values
    assert full.person.index.tolist() == full.person.native_person_id.tolist()
    assert set(full.person.index) == {105, 106, 107, 108}
    pd.testing.assert_frame_equal(
        full.asec_literals, selected.asec_literals, check_exact=True
    )
    donor = amounts.full_donor.qualify_full_original_amount_donor(
        preparation, amounts.selected_groups(("child_support",))
    )
    assert donor.amounts.columns.tolist() == ["CSP_VAL"]
    np.testing.assert_array_equal(
        donor.amounts.CSP_VAL.to_numpy(), [1200, np.nan, 0, np.nan]
    )
    assert donor.frame.weights_for("household").values.tolist() == [2552.12, 100]
    assert full.evidence["selected_rows"] == len(selected.person)
    assert full.evidence["returned_rows"] == 4
    state = preparation._checked()[2]
    origins = state.frame.person
    ids = pd.Index(origins.person_id.to_numpy(), name="person_id")
    mask = (
        origins[amounts.provenance.support_channel_column("person")]
        .eq("asec")
        .to_numpy()
    )
    native_ids = origins.loc[mask, amounts.provenance.spine_source_id_column("person")]
    _, native, reports = amounts._child_projection(preparation, ids, mask, native_ids)
    assert native.child_support_expense.isna().all()  # Positive payer was omitted.
    assert reports.loc[ids[~mask], "survey_child_CSP_VAL_amount_known"].isna().all()
    assert (
        reports.loc[ids[~mask], "survey_current_CHSP_VAL_origin"].eq("unresolved").all()
    )
    assert not full.person.loc[107, "CHSP_VAL_amount_known"]
    assert full.person.loc[107, "CHSP_VAL_reporting_status"] == "declared_niu_amount"
    return donor


def _run_child_source_graph(tmp_path, monkeypatch, projection, carried):
    from types import SimpleNamespace

    from microcosm.fit import model_input
    from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
    from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
    from microcosm.graph import (
        ArtifactInput,
        ArtifactOutput,
        Capabilities,
        ContentStore,
        Determinism,
        Graph,
        KernelBase,
        KernelRegistry,
        KernelResult,
        Node,
        Numeric,
        Owned,
        SourceRef,
        StructuralDelta,
        compile_graph,
        run_graph,
    )
    from microcosm.graph import population as population_ops

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    qualified, receiving = _child_family()
    spec = qualified.groups[0].spec
    full = projection.frame
    keep = np.isfinite(projection.features.to_numpy()).all(axis=1) & np.isfinite(
        projection.amounts.CSP_VAL.to_numpy()
    )
    columns = projection.features.loc[keep].copy()
    columns[spec.targets[0]] = projection.amounts.loc[keep, "CSP_VAL"]
    recipient = columns.loc[:, list(projection.features)].iloc[:1].copy()
    recipient.index = pd.Index([1], name="person_id")
    matrix = model_input.encode_recipient_matrix(
        recipient, entity="person", entity_ids=np.array([1], dtype="<i8")
    )
    qualified = replace(qualified, features=tuple(projection.features))
    if carried is not None:
        for name in amounts.child_mapper.US_CHILD_SUPPORT_OUTPUT_COLUMNS:
            receiving.person[name] = np.full(
                len(receiving.person), carried, dtype="float64"
            )
    source_before = qualified.source_frame.person.copy(deep=True)
    omitted = 105
    group = amounts.GroupValues(spec, full.select(keep), columns, matrix, keep)
    qualified = replace(qualified, full_donor_source_frame=full, groups=(group,))
    assert omitted not in qualified.source_frame.person.person_id.values
    assert group.donor_columns.loc[omitted, spec.targets[0]] > 0
    before = {e: receiving.table(e).copy(deep=True) for e in receiving.entities}
    weight = receiving.weights_for("household")
    weight_bytes, weight_kind = weight.values.tobytes(), weight.kind
    nodes = graph.amount_nodes(
        qualified, receiving, parent_digest="a" * 64, n_estimators=2
    )
    assert sum(n.id == graph.FULL_DONOR_SOURCE_NODE for n in nodes) == 1
    donor_id = graph._ids(group)[0]
    assert (
        next(n for n in nodes if n.id == donor_id).base == graph.FULL_DONOR_SOURCE_NODE
    )

    class Receiving(KernelBase):
        ref = "test.full-amount.receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "b" * 64

        def run(self, context):
            return KernelResult(frame=receiving)

    class Finalization(KernelBase):
        ref = "test.full-amount.finalization@1"
        capabilities = Capabilities(Determinism.DETERMINISTIC)

        def implementation_hash(self):
            return "c" * 64

        def run(self, context):
            return KernelResult(
                artifacts={"finalization": b"invented; no native parent authority"}
            )

    class ReceivingFilter(KernelBase):
        ref = "test.child.receiving_filter@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
        )

        def implementation_hash(self):
            return "d" * 64

        def run(self, context):
            return KernelResult(
                keep=pd.Series(
                    True,
                    index=pd.Index(receiving.person.person_id, name="person_id"),
                    dtype=bool,
                )
            )

    create = Node(
        "test.child.source",
        Receiving.ref,
        sources=(graph.health_graph.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(
                selection.entity,
                c,
                population_ops.token_for_dtype(
                    receiving.table(selection.entity)[c].dtype
                ),
            )
            for selection in graph.predictor_graph._inputs(receiving)
            for c in selection.columns
        ),
    )
    receiving_node = Node(
        graph.parent.attach.FILTER_NODE,
        ReceivingFilter.ref,
        base=create.id,
        structural=StructuralDelta.FILTER,
        mass="free",
        inputs=graph.predictor_graph._inputs(receiving),
    )
    finalization = Node(
        graph.parent.attach.ATTACH_NODE,
        Finalization.ref,
        population=receiving_node.id,
        artifact_outputs=(
            ArtifactOutput("finalization", graph.parent.attach.FINALIZATION_TYPE),
        ),
    )

    class Later(KernelBase):
        ref = "test.child.later@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
        )

        def implementation_hash(self):
            return "e" * 64

        def run(self, context):
            return KernelResult(
                columns={
                    ("person", "test_late"): pd.Series(
                        receiving.person.person_id.to_numpy() * 2,
                        index=pd.Index(receiving.person.person_id, name="person_id"),
                        dtype="int64",
                    )
                }
            )

    later = Node(
        "test.child.later",
        Later.ref,
        population=receiving_node.id,
        outputs=(Owned("person", "test_late", "int64"),),
        artifact_inputs=(
            ArtifactInput(
                "amounts", graph.ATTACH_NODE, "attachment", graph.ATTACHMENT_TYPE
            ),
        ),
    )
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(graph.health_graph.SOURCE_NAME, "raw-bytes-v1"),),
            (create, receiving_node, finalization, *nodes, later),
        )
    )
    original_seal = amounts.seal(qualified)

    def pure():
        assert amounts.seal(qualified) == original_seal

    def context(value):
        pure()
        assert value.node in nodes
        return qualified

    boundary = SimpleNamespace(
        qualified=qualified,
        context=context,
        pure=pure,
        run=SimpleNamespace(population=SimpleNamespace(frame=receiving)),
        parent_view=SimpleNamespace(digest="a" * 64),
        n_estimators=2,
        compiled=compiled,
    )
    registry = KernelRegistry()
    for kernel in (
        Receiving(),
        ReceivingFilter(),
        Finalization(),
        Later(),
        graph.CurrentSurveyAmountProjectionKernel(boundary),
        graph.CurrentSurveyFullAmountSourceKernel(boundary),
        graph.CurrentSurveyAmountDonorKernel(boundary),
        graph.CurrentSurveyAmountColumnsKernel(boundary),
        graph.CurrentSurveyAmountAttachKernel(boundary),
        graph.CurrentSurveyChildVersionKernel(boundary),
        graph.CurrentSurveyChildAttachKernel(boundary),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        registry.register(kernel)
    path = tmp_path / "invented.txt"
    path.write_bytes(b"invented full-original amount graph")
    store = ContentStore(tmp_path / "store")
    kwargs = dict(
        sources={graph.health_graph.SOURCE_NAME: path}, store=store, kernels=registry
    )
    cold = run_graph(compiled, **kwargs)
    observed = {}
    warm = run_graph(
        compiled,
        **kwargs,
        resume="require",
        _population_observer=lambda node, population: observed.__setitem__(
            node, population
        ),
    )
    assert cold.key == warm.key and all(n.hit for n in warm.nodes.values())
    fitted = warm.population(donor_id)
    assert fitted.person.person_id.tolist() == columns.index.tolist()
    assert fitted.person[spec.targets[0]].tolist() == columns[spec.targets[0]].tolist()
    assert omitted in fitted.person.person_id.values
    assert fitted.person.person_id.tolist() == [105, 107]
    assert fitted.person[spec.targets[0]].tolist() == [1200, 0]
    assert fitted.weights_for("household").values.tolist() == [2552.12, 100]
    assert not any("CHSP" in node.id for node in compiled.graph.nodes)
    assert (
        sum(node.kernel == LegacyQRFTrainKernel.ref for node in compiled.graph.nodes)
        == 1
    )
    output = warm.population(graph.CHILD_VERSION_NODE)
    np.testing.assert_array_equal(output.person.test_late, output.person.person_id * 2)
    assert graph.CHILD_ATTACH_NODE in cold.nodes
    assert compiled.versions[graph.PROJECTION_NODE] == receiving_node.id
    assert compiled.versions[graph.CHILD_ATTACH_NODE] == graph.CHILD_VERSION_NODE
    base_population = population_ops.Population.from_frame(
        warm.population(receiving_node.id),
        receiving_node.id,
        mass_ledger=warm.mass_ledger(receiving_node.id),
    )
    reconstructed = graph._child_version_population(
        base_population, compiled.graph.node(graph.CHILD_VERSION_NODE)
    )
    for entity in output.entities:
        pd.testing.assert_frame_equal(
            reconstructed.frame.table(entity),
            warm.population(receiving_node.id).table(entity),
            check_exact=True,
        )
    assert reconstructed.mass_ledger[:-1] == base_population.mass_ledger
    assert reconstructed.mass_ledger[-1].policy == "conserve"
    for e in receiving.entities:
        unrelated = [
            c
            for c in before[e]
            if e != "person"
            or c not in amounts.child_mapper.US_CHILD_SUPPORT_OUTPUT_COLUMNS
        ]
        pd.testing.assert_frame_equal(
            output.table(e)[unrelated], before[e][unrelated], check_exact=True
        )
    assert output.weights_for("household").kind is weight_kind
    assert output.weights_for("household").values.tobytes() == weight_bytes
    # Native observed cells survive; only the eligible ACS original gets a draw.
    expected = output.person.set_index("person_id").child_support_received
    assert expected.loc[2] == expected.loc[12] == 120
    assert expected.loc[3] == expected.loc[13] == 0
    assert np.isnan(expected.loc[0]) and np.isnan(expected.loc[10])
    assert expected.loc[1] == expected.loc[11]
    assert expected.loc[1] in (0.0, 1200.0)
    paid = output.person.set_index("person_id").child_support_expense
    assert paid.loc[2] == paid.loc[12] == 2000
    assert paid.drop([2, 12]).isna().all()
    pd.testing.assert_frame_equal(
        qualified.source_frame.person, source_before, check_exact=True
    )
    for e in receiving.entities:
        pd.testing.assert_frame_equal(receiving.table(e), before[e], check_exact=True)
    loaded = {
        (node.id, name): store.load_bytes(key)
        for node in compiled.graph.nodes
        for name, key in warm.node(node.id).opaque_artifacts.items()
    }
    version_node = compiled.graph.node(graph.CHILD_VERSION_NODE)
    replayed_version = graph._child_version_population(observed[later.id], version_node)
    graph.physical.replay.same_replayed_population(
        replayed_version, observed[graph.CHILD_VERSION_NODE]
    )
    child_node = compiled.graph.node(graph.CHILD_ATTACH_NODE)
    child_result = graph._attachment_result(
        boundary,
        graph.parent._loaded_values(boundary, warm, loaded, child_node),
        child_only=True,
    )
    assert (
        loaded[graph.CHILD_ATTACH_NODE, "attachment"]
        == child_result.artifacts["attachment"]
    )
    replayed_child = population_ops.patch(replayed_version, child_node, child_result)
    graph.physical.replay.same_replayed_population(
        replayed_child, observed[graph.CHILD_ATTACH_NODE]
    )
    graph._verify_models(
        boundary,
        loaded,
        {donor_id: population_ops.Population.from_frame(fitted, donor_id)},
    )
    assert model_input.decode_recipient_matrix(
        group.matrix
    ).features.index.tolist() == [1]
    pure()


@pytest.mark.parametrize("carried", [None, 120.0, -999.0])
def test_genuine_child_donor_real_graph_cold_required_and_rewrite(
    tmp_path, monkeypatch, carried
):
    donor = _full_original_child_source_and_donor(tmp_path, monkeypatch)
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    _run_child_source_graph(graph_dir, monkeypatch, donor, carried)


@pytest.mark.parametrize(
    "name",
    [
        "current_asec_child_support_source.py",
        "current_asec_amount_donor.py",
        "current_survey_amounts.py",
        "graph_us_survey_enrichment.py",
    ],
)
def test_child_changed_modules_keep_source_spine_contract(name):
    from test_us_spine_blindness import _US_RUNTIME, _non_owner_source_spine_accesses

    assert not _non_owner_source_spine_accesses(name, (_US_RUNTIME / name).read_text())


def test_child_source_lookup_and_cached_callable_are_live_sealed(monkeypatch):
    original = graph._live()

    class ChangedLookup(dict):
        def __getitem__(self, key):
            return (999,)

    with monkeypatch.context() as patch:
        patch.setattr(
            source, "RESPONSE_ENTRIES", ChangedLookup(source.RESPONSE_ENTRIES)
        )
        assert graph._live() != original
    wrapped = source._cached_amount_entries_json.__wrapped__
    with monkeypatch.context() as patch:
        patch.setattr(wrapped, "__code__", (lambda: b"changed").__code__)
        assert graph._live() != original


def test_keep_all_child_version_refuses_pruned_orphan_group():
    from microcosm.graph import population as population_ops

    qualified, receiving = _child_family()
    node = next(
        n
        for n in graph.amount_nodes(
            qualified, receiving, parent_digest="a" * 64, n_estimators=2
        )
        if n.id == graph.CHILD_VERSION_NODE
    )
    population = population_ops.Population.from_frame(receiving, node.base)
    membership = receiving.schema.membership_column("household")
    first, second = receiving.person[membership].drop_duplicates().iloc[:2]
    receiving.person.loc[receiving.person[membership].eq(first), membership] = second
    with pytest.raises(ValueError, match="CHILD_VERSION_AXIS_CHANGED"):
        graph._child_version_population(population, node)
