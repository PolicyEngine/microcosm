"""Full original donor projections and tiny amount graphs over invented inputs."""

import json
import shutil
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_amount_donor as donors
from microcosm.build.us_runtime import current_survey_amounts as amounts
from microcosm.build.us_runtime import graph_us_survey_enrichment as graph


@pytest.mark.parametrize("option", [None, 1, "true", [], {}])
def test_full_original_option_refuses_before_owner_access(option):
    with pytest.raises(ValueError, match="FULL_ORIGINAL_DONORS_OPTION"):
        amounts.qualify_current_survey_amounts(object(), full_original_donors=option)
    with pytest.raises(ValueError, match="FULL_ORIGINAL_DONORS_OPTION"):
        graph.Boundary(
            object(),
            groups=("unemployment",),
            n_estimators=2,
            full_original_amount_donors=option,
        )
    with pytest.raises(ValueError, match="FULL_ORIGINAL_OPTION"):
        donors.receipts._qualify_receipt_amount(
            object(), family="unemployment", full_original=option
        )


def test_detached_frame_does_not_supply_full_donor_authority():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        donors.qualify_full_original_amount_donor(
            object(), amounts.selected_groups(("workers_compensation",))
        )


def test_same_items_state_mapping_replacement_changes_live_seal(monkeypatch):
    class ChangedMembership(dict):
        def __contains__(self, key):
            return True

    before = graph._live()
    original = donors.US_STATE_NUMERIC_FIPS_TO_POSTAL
    changed = ChangedMembership(original)
    assert tuple(changed.items()) == tuple(original.items())
    assert 999 in changed and 999 not in original
    monkeypatch.setattr(donors, "US_STATE_NUMERIC_FIPS_TO_POSTAL", changed)
    assert graph._live() != before


def test_default_scope_is_selected_and_additional_source_is_opt_in():
    import inspect

    from test_us_native_workers_compensation import _invented_wc_family

    qualified, receiving = _invented_wc_family()
    assert qualified.full_donor_source_frame is None
    assert qualified.donor_source_frame is qualified.source_frame
    nodes = graph.amount_nodes(
        qualified, receiving, parent_digest="a" * 64, n_estimators=2
    )
    assert graph.FULL_DONOR_SOURCE_NODE not in {n.id for n in nodes}
    for function in (graph.run_us_survey_enrichment, graph._construct, graph.Boundary):
        assert (
            inspect.signature(function)
            .parameters["full_original_amount_donors"]
            .default
            is False
        )
    assert (
        inspect.signature(amounts.qualify_current_survey_amounts)
        .parameters["full_original_donors"]
        .default
        is False
    )
    assert (
        inspect.signature(donors.receipts._qualify_receipt_amount)
        .parameters["full_original"]
        .default
        is False
    )


@pytest.mark.parametrize(
    "function,target",
    [(graph.run_us_survey_enrichment, "_construct"), (graph._construct, "Boundary")],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_optional_scope_forwarding(function, target, enabled, monkeypatch):
    class StopError(Exception):
        pass

    def observed(*args, **kwargs):
        assert kwargs["full_original_amount_donors"] is enabled
        raise StopError

    monkeypatch.setattr(graph, target, observed)
    with pytest.raises(StopError):
        function(
            object(),
            full_original_amount_donors=enabled,
            groups=("unemployment",),
            n_estimators=2,
        )


def test_genuine_full_donor_survives_receiving_selection(tmp_path, monkeypatch):
    from test_us_native_workers_compensation import _wc_source_arguments

    args = _wc_source_arguments(tmp_path, monkeypatch)
    partial = tmp_path / "partial"
    shutil.copytree(args["source_dir"], partial)
    request = json.loads((partial / "selection-request.json").read_bytes())
    request["fraction"], request["seed"] = [2, 3], 41
    (partial / "selection-request.json").write_text(
        json.dumps(request, sort_keys=True, separators=(",", ":"))
    )
    captures = tmp_path / "partial-captures"
    captures.mkdir()
    full = donors.source.prepare_authenticated_survey_population(**args)
    selected = donors.source.prepare_authenticated_survey_population(
        source_dir=partial, snapshot_root=captures, fraction=Fraction(2, 3), seed=41
    )
    specs = amounts.selected_groups(
        ("unemployment", "health_costs", "workers_compensation")
    )
    complete = donors.qualify_full_original_amount_donor(
        full, specs, demographic_conditioning=True
    )
    subset = donors.qualify_full_original_amount_donor(
        selected, specs, demographic_conditioning=True
    )
    native_selected = set(selected._checked()[2].source_frames[1].person.person_id)
    assert native_selected < set(subset.features.index)
    assert 105 not in native_selected
    assert subset.amounts.loc[105, "WC_VAL"] == 120
    assert len(subset.features) == 4
    assert subset.frame.weights_for("household").kind.value == "design"
    assert subset.frame.weights_for("household").values.tolist() == [2552.12, 100.0]
    for entity in complete.frame.entities:
        pd.testing.assert_frame_equal(
            complete.frame.table(entity), subset.frame.table(entity), check_exact=True
        )
    pd.testing.assert_frame_equal(complete.features, subset.features, check_exact=True)
    pd.testing.assert_frame_equal(complete.amounts, subset.amounts, check_exact=True)
    np.testing.assert_array_equal(
        subset.amounts.loc[[105, 106, 107, 108], "WC_VAL"], [120.0, 0.0, np.nan, np.nan]
    )
    assert set(subset.features) == set(amounts.predictors.DEMOGRAPHIC_FEATURES)
    assert not any("last_year" in c or "prior" in c for c in subset.frame.person)
    before = donors.source._frame_identity(selected._checked()[2].frame)
    selected_receipt = donors.receipts._qualify_receipt_amount(
        selected, family="workers_compensation"
    )
    full_receipt = donors.receipts._qualify_receipt_amount(
        selected, family="workers_compensation", full_original=True
    )
    assert len(selected_receipt[0]) == 2 and len(full_receipt[0]) == 4
    assert "donor_scope" not in selected_receipt[1]
    assert full_receipt[1]["donor_scope"] == "full_original_current_asec"
    assert donors.source._frame_identity(selected._checked()[2].frame) == before
    # Carry the genuine full-source projection into the real numerical nodes.
    # The receiving family is explicitly an invented component fixture, not a
    # replacement for a genuine full financial/enrichment host acceptance run.
    _run_full_source_graph(tmp_path, monkeypatch, projection=subset)


def test_actual_full_source_filter_fit_apply_and_clone_replay(tmp_path, monkeypatch):
    _run_full_source_graph(tmp_path, monkeypatch)


def _run_full_source_graph(tmp_path, monkeypatch, *, projection=None):
    from types import SimpleNamespace

    from test_us_current_survey_health_coverage import _frame
    from test_us_native_workers_compensation import _invented_wc_family

    from microcosm.fit import model_input
    from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
    from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
    from microcosm.graph import (
        ArtifactOutput,
        Capabilities,
        ContentStore,
        Determinism,
        Graph,
        KernelBase,
        KernelRegistry,
        KernelResult,
        Node,
        Owned,
        SourceRef,
        StructuralDelta,
        compile_graph,
        run_graph,
    )
    from microcosm.graph import population as population_ops

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    qualified, receiving = _invented_wc_family()
    spec = qualified.groups[0].spec
    full = _frame(pd.DataFrame({"person_id": [2, 3, 999], "age": [40, 40, 80]}))
    columns = pd.DataFrame(
        {
            qualified.features[0]: [40.0, 40.0, 80.0],
            spec.targets[0]: [120.0, 0.0, 900.0],
        },
        index=pd.Index([2, 3, 999], name="person_id"),
    )
    keep = np.ones(3, dtype=bool)
    matrix = qualified.groups[0].matrix
    omitted = 999
    if projection is not None:
        full = projection.frame
        keep = np.isfinite(projection.features.to_numpy()).all(axis=1) & np.isfinite(
            projection.amounts.WC_VAL.to_numpy()
        )
        columns = projection.features.loc[keep].copy()
        columns[spec.targets[0]] = projection.amounts.loc[keep, "WC_VAL"]
        recipient = columns.loc[:, list(projection.features)].iloc[:1].copy()
        recipient.index = pd.Index([1], name="person_id")
        matrix = model_input.encode_recipient_matrix(
            recipient, entity="person", entity_ids=np.array([1], dtype="<i8")
        )
        qualified = replace(qualified, features=tuple(projection.features))
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

    create = Node(
        graph.parent.attach.FILTER_NODE,
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
    finalization = Node(
        graph.parent.attach.ATTACH_NODE,
        Finalization.ref,
        population=create.id,
        artifact_outputs=(
            ArtifactOutput("finalization", graph.parent.attach.FINALIZATION_TYPE),
        ),
    )
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(graph.health_graph.SOURCE_NAME, "raw-bytes-v1"),),
            (create, finalization, *nodes),
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
        Finalization(),
        graph.CurrentSurveyAmountProjectionKernel(boundary),
        graph.CurrentSurveyFullAmountSourceKernel(boundary),
        graph.CurrentSurveyAmountDonorKernel(boundary),
        graph.CurrentSurveyAmountColumnsKernel(boundary),
        graph.CurrentSurveyAmountAttachKernel(boundary),
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
    warm = run_graph(compiled, **kwargs, resume="require")
    assert cold.key == warm.key and all(n.hit for n in warm.nodes.values())
    fitted = warm.population(donor_id)
    assert fitted.person.person_id.tolist() == columns.index.tolist()
    assert fitted.person[spec.targets[0]].tolist() == columns[spec.targets[0]].tolist()
    assert omitted in fitted.person.person_id.values
    output = warm.population(create.id)
    for e in receiving.entities:
        pd.testing.assert_frame_equal(
            output.table(e)[before[e].columns], before[e], check_exact=True
        )
    assert output.weights_for("household").kind is weight_kind
    assert output.weights_for("household").values.tobytes() == weight_bytes
    # Native observed cells survive; only the eligible ACS original gets a draw.
    expected = output.person.set_index("person_id").workers_compensation
    assert expected.loc[2] == expected.loc[12] == 120
    assert expected.loc[3] == expected.loc[13] == 0
    assert np.isnan(expected.loc[0]) and np.isnan(expected.loc[10])
    assert expected.loc[1] == expected.loc[11]
    loaded = {
        (node.id, name): store.load_bytes(key)
        for node in compiled.graph.nodes
        for name, key in warm.node(node.id).opaque_artifacts.items()
    }
    graph._verify_models(
        boundary,
        loaded,
        {donor_id: population_ops.Population.from_frame(fitted, donor_id)},
    )
    assert model_input.decode_recipient_matrix(
        group.matrix
    ).features.index.tolist() == [1]
    pure()


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("axis", "FULL_DONOR_TABLE"),
        ("value", "FULL_DONOR_TABLE"),
        ("weight", "FULL_DONOR_WEIGHT_VALUES"),
        ("weight_kind", "FULL_DONOR_WEIGHT_VALUES"),
        ("strata", "FULL_DONOR_STRATA"),
        ("tables", "FULL_DONOR_TABLES"),
        ("declaration", "FULL_DONOR_INPUTS"),
    ],
)
def test_full_donor_context_checks_exact_declared_projection(mutation, reason):
    from types import SimpleNamespace

    from test_us_current_survey_health_coverage import _frame

    from microcosm.frame import WeightKind, Weights

    frame = _frame(pd.DataFrame({"person_id": [2, 9], "age": [40, 80]}))
    inputs = graph._full_donor_inputs(frame)
    assert len(inputs) == 1 and inputs[0].entity == "person"
    columns = (
        ["person_id"]
        + [frame.schema.membership_column(e) for e in frame.schema.group_entities]
        + ["age"]
    )
    original_weights = frame.resolve_weights("person")
    context = SimpleNamespace(
        node=SimpleNamespace(inputs=inputs),
        tables={"person": frame.person.loc[:, columns].copy(deep=True)},
        weights={"person": original_weights},
        strata=frame.strata.copy(deep=True),
    )
    graph._full_donor_context(context, frame)
    if mutation == "axis":
        context.tables["person"] = context.tables["person"].iloc[::-1]
    elif mutation == "value":
        context.tables["person"].loc[0, "age"] += 1
    elif mutation in {"weight", "weight_kind"}:
        changed = original_weights.values.copy()
        kind = original_weights.kind
        if mutation == "weight":
            changed[0] += 1
        else:
            kind = WeightKind.IMPORTANCE
        context.weights["person"] = Weights(changed, kind)
    elif mutation == "strata":
        context.strata.iloc[0] = "changed"
    elif mutation == "tables":
        context.tables["household"] = frame.table("household")
    else:
        context.node.inputs = ()
    with pytest.raises(ValueError, match=reason):
        graph._full_donor_context(context, frame)


@pytest.mark.parametrize(
    "name",
    [
        "current_asec_amount_donor.py",
        "current_asec_unemployment_source.py",
        "current_survey_amounts.py",
        "graph_us_survey_enrichment.py",
    ],
)
def test_full_donor_changed_modules_preserve_source_classification(name):
    import test_us_spine_blindness as guard

    assert not guard._non_owner_source_spine_accesses(
        name, (guard._US_RUNTIME / name).read_text()
    )
    if name == "current_asec_amount_donor.py":
        assert name in guard._US_LAUNCH_GRAPH_RUNTIME_MODULES
        assert name not in guard._SOURCE_SPINE_PROVENANCE_OWNERS
        assert guard._non_owner_source_spine_accesses(
            name, 'def wrong(f):\n return f.person["person_spine_source_id"]\n'
        )
