"""Real weighted QRF and signed reconciliation over invented person branches."""

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.graph_property_income import (
    PROPERTY_COMPONENTS,
    PROPERTY_DRAW_COLUMNS,
    PROPERTY_REPORTED_TOTAL,
    PropertyIncomeDrawColumnsKernel,
    PropertyIncomeTrainKernel,
    property_income_nodes,
)
from microcosm.fit import QRF, graph_legacy_train
from microcosm.fit.graph_legacy_qrf import LegacyQRFApplyKernel
from microcosm.fit.graph_signed_reconciliation import SignedReconciliationKernel
from microcosm.fit.signed_reconciliation import reconcile_signed_total
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)

FEATURES = (PROPERTY_REPORTED_TOTAL, "age", "earnings")


def test_base_training_implementation_is_part_of_cache_identity(monkeypatch):
    seen = []
    original = graph_legacy_train.source_hash

    def observe(*items, **kwargs):
        seen.append(items)
        return original(*items, **kwargs)

    monkeypatch.setattr(graph_legacy_train, "source_hash", observe)
    PropertyIncomeTrainKernel().implementation_hash()
    assert any(items[0] is graph_legacy_train.LegacyQRFTrainKernel for items in seen)


@pytest.fixture(autouse=True)
def one_worker(monkeypatch):
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


def frame(values, *, kind=WeightKind.DESIGN, donor=False):
    values = values.copy()
    ids = np.arange(len(values), dtype=np.int64) + 2**53 + 21
    groups = np.arange(len(values), dtype=np.int64) // 2 + 301
    values.insert(0, "person_id", ids)
    values.insert(1, "person_household_id", groups)
    households = np.unique(groups)
    weights = np.arange(len(households), dtype=float) + 1
    weights[-1] = 0
    return Frame(
        {"person": values, "household": pd.DataFrame({"household_id": households})},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(weights, kind)},
        metadata={
            "invented": True,
            "branch": "donor" if donor else "recipient",
            "keep": {"n": 7},
        },
    )


def frames():
    r = np.arange(44)
    donor = pd.DataFrame(
        {
            "age": 25.0 + r % 20,
            "earnings": 1000.0 * (r % 7),
            PROPERTY_COMPONENTS[0]: 5.0 * (r % 4),
            PROPERTY_COMPONENTS[1]: 10.0 * (r % 3),
            PROPERTY_COMPONENTS[2]: 15.0 * (r % 2),
            PROPERTY_COMPONENTS[3]: 20.0 * (r % 5 - 2),
        }
    )
    donor[PROPERTY_REPORTED_TOTAL] = donor.loc[:, list(PROPERTY_COMPONENTS)].sum(axis=1)
    recipient = pd.DataFrame(
        {
            PROPERTY_REPORTED_TOTAL: [-60.0, 0.0, 37.5, 120.0],
            "age": [22.0, 55.0, 67.0, 32.0],
            "earnings": [2500.0, 0.0, 1200.0, 7600.0],
            "retained_note": ["alpha", "beta", "gamma", "delta"],
        },
        index=pd.Index([9, 2, 17, 41], name="original_row"),
    )
    recipient["retained_note"] = recipient.retained_note.astype("string")
    return frame(donor, donor=True), frame(recipient)


class SourceKernel(KernelBase):
    ref = "test.property_income.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, ctx):
        return KernelResult(
            frame=load_source("frame-store", ctx.sources[ctx.node.sources[0]])
        )


class RecordingDrawKernel(PropertyIncomeDrawColumnsKernel):
    context = None

    def run(self, context):
        self.context = context
        return super().run(context)


def source_node(name, source):
    return Node(
        name,
        SourceKernel.ref,
        sources=(name,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, c, str(table[c].dtype))
            for entity in source.entities
            for table in (source.table(entity),)
            for c in table
            if not c.endswith("_id")
        ),
    )


def source_path(store, source):
    parts = [
        source.table(entity).to_json(orient="table").encode()
        for entity in source.entities
    ]
    parts.extend(
        source.weights_for(entity).values.tobytes()
        for entity in source.weighted_entities
    )
    return store.put_frame(hashlib.sha256(b"".join(parts)).hexdigest(), source)


def setup(tmp_path, *, scales=(1.0, 1.0, 1.0, 1.0), recipient=None):
    donor, default_recipient = frames()
    recipient = default_recipient if recipient is None else recipient
    nodes = property_income_nodes(
        "property",
        donor_population="donor",
        recipient_population="recipient",
        features=FEATURES,
        seed=811,
        n_estimators=2,
        scales=scales,
        atol=1e-10,
        rtol=1e-12,
    )
    graph = Graph(
        country="us",
        sources=(
            SourceRef("donor", "frame-store"),
            SourceRef("recipient", "frame-store"),
        ),
        nodes=(
            source_node("donor", donor),
            source_node("recipient", recipient),
            *nodes,
        ),
    )
    kernels = KernelRegistry()
    draw = RecordingDrawKernel()
    for kernel in (
        SourceKernel(),
        PropertyIncomeTrainKernel(),
        LegacyQRFApplyKernel(),
        draw,
        SignedReconciliationKernel(),
    ):
        kernels.register(kernel)
    store = ContentStore(tmp_path)
    sources = {
        "donor": source_path(store, donor),
        "recipient": source_path(store, recipient),
    }
    return compile_graph(graph), store, kernels, sources, donor, recipient, draw


def execute(setup_values, **kwargs):
    compiled, store, kernels, sources, *_ = setup_values
    return run_graph(compiled, store=store, kernels=kernels, sources=sources, **kwargs)


def assert_preserved(source, result):
    for entity in source.entities:
        pd.testing.assert_frame_equal(
            result.table(entity)[list(source.table(entity))], source.table(entity)
        )
    assert result.metadata == source.metadata
    assert result.schema == source.schema
    for entity in source.weighted_entities:
        assert result.weights_for(entity).kind is source.weights_for(entity).kind
        np.testing.assert_array_equal(
            result.weights_for(entity).values, source.weights_for(entity).values
        )


def test_actual_joint_fit_projection_and_required_replay(tmp_path):
    values = setup(tmp_path)
    compiled, store, kernels, sources, donor, recipient, _ = values
    manifest = execute(values)
    result = manifest.population("recipient")
    assert len(manifest.nodes) == 12
    assert_preserved(donor, manifest.population("donor"))
    assert_preserved(recipient, result)
    model = QRF(seed=811, n_estimators=2, zero_atol=0)
    state = model.start_chain(
        donor, list(FEATURES), list(PROPERTY_COMPONENTS), weights="design"
    )
    raw = pd.DataFrame(index=recipient.person.index)
    for target, draw_name in zip(
        PROPERTY_COMPONENTS, PROPERTY_DRAW_COLUMNS, strict=True
    ):
        next_ = model.fit_draw_next(
            donor, recipient, raw, state=state, weights="design"
        )
        raw[target] = next_.raw_draw
        state = next_.state
        np.testing.assert_array_equal(result.person[draw_name], next_.raw_draw)
    projected = reconcile_signed_total(
        raw.to_numpy(),
        recipient.person[PROPERTY_REPORTED_TOTAL].to_numpy(),
        components=PROPERTY_COMPONENTS,
        nonnegative=np.array([True, True, True, False]),
        scales=np.ones(4),
        atol=1e-10,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        result.person[list(PROPERTY_COMPONENTS)], projected.values
    )
    np.testing.assert_allclose(
        result.person[list(PROPERTY_COMPONENTS)].sum(axis=1),
        recipient.person[PROPERTY_REPORTED_TOTAL],
        rtol=1e-12,
        atol=1e-10,
    )
    assert (result.person[list(PROPERTY_COMPONENTS[:3])] >= 0).all().all()
    assert result.person[PROPERTY_COMPONENTS[-1]].iloc[0] < 0
    assert result.person.person_id.dtype == np.dtype("int64")
    again = run_graph(
        compiled,
        store=ContentStore(store.root),
        kernels=kernels,
        sources=sources,
        resume="require",
    )
    assert again.key == manifest.key
    assert all(node.hit for node in again.nodes.values())
    pd.testing.assert_frame_equal(result.person, again.population("recipient").person)


def test_changed_scales_reuse_fit_draws_and_only_recompute_projection(tmp_path):
    first = execute(setup(tmp_path))
    second = execute(setup(tmp_path, scales=(4.0, 2.0, 3.0, 1.0)))
    assert sum(n.hit for n in second.nodes.values()) == 11
    assert not second.node("property.reconcile").hit
    for c in PROPERTY_DRAW_COLUMNS:
        np.testing.assert_array_equal(
            first.population("recipient").person[c],
            second.population("recipient").person[c],
        )


def test_changed_recipient_anchor_reuses_all_donor_fits(tmp_path):
    execute(setup(tmp_path))
    _, recipient = frames()
    data = recipient.person.copy()
    data.loc[data.index[0], PROPERTY_REPORTED_TOTAL] = -75.0
    changed = Frame(
        {"person": data, "household": recipient.table("household")},
        recipient.schema,
        {"household": recipient.weights_for("household")},
        metadata=recipient.metadata,
    )
    result = execute(setup(tmp_path, recipient=changed))
    assert sum(n.hit for n in result.nodes.values()) == 5
    assert result.node("donor").hit
    assert all(result.node(f"property.fit.{i:03d}").hit for i in range(4))
    assert not result.node("property.reconcile").hit


@pytest.mark.parametrize("kind", [WeightKind.IMPORTANCE, WeightKind.CALIBRATED])
def test_postallocation_weight_kinds_refuse_before_fitting(kind):
    donor, _ = frames()
    node = property_income_nodes(
        "p",
        donor_population="d",
        recipient_population="r",
        features=FEATURES,
        seed=2,
        n_estimators=2,
        scales=(1.0,) * 4,
        atol=1e-10,
        rtol=1e-12,
    )[0]
    ctx = KernelContext(
        node,
        {"person": donor.person},
        {"person": Weights(donor.resolve_weights("person").values, kind)},
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(2),
    )
    with pytest.raises(ValueError, match="ORIGINAL_DESIGN_WEIGHTS"):
        PropertyIncomeTrainKernel().run(ctx)


@pytest.mark.parametrize(
    "defect",
    ["unknown", "nonfinite", "negative_interest", "inconsistent_total", "nullable"],
)
def test_ineligible_donor_basis_refuses_before_fitting(defect):
    donor, _ = frames()
    table = donor.person.copy()
    if defect == "unknown":
        table.loc[0, PROPERTY_COMPONENTS[0]] = np.nan
    elif defect == "nonfinite":
        table.loc[0, PROPERTY_COMPONENTS[3]] = np.inf
    elif defect == "negative_interest":
        table.loc[0, PROPERTY_COMPONENTS[0]] = -5.0
        table.loc[0, PROPERTY_REPORTED_TOTAL] -= 5.0
    elif defect == "inconsistent_total":
        table.loc[0, PROPERTY_REPORTED_TOTAL] += 5.0
    else:
        table[PROPERTY_COMPONENTS[0]] = pd.array(
            table[PROPERTY_COMPONENTS[0]], dtype="Float64"
        )
    node = property_income_nodes(
        "p",
        donor_population="d",
        recipient_population="r",
        features=FEATURES,
        seed=2,
        n_estimators=2,
        scales=(1.0,) * 4,
        atol=1e-10,
        rtol=1e-12,
    )[0]
    ctx = KernelContext(
        node,
        {"person": table},
        {"person": donor.resolve_weights("person")},
        pd.Series(dtype=str),
        node.params,
        np.random.default_rng(2),
    )
    with pytest.raises(
        ValueError, match="KNOWN_DONOR_VALUES|RESOLVED_DONOR_COMPONENTS"
    ):
        PropertyIncomeTrainKernel().run(ctx)


@pytest.mark.parametrize(
    "defect", ["raw_history", "producer", "row_index", "entity_ids", "seed", "roster"]
)
def test_draw_placement_rejects_unbound_history_and_row_identity(tmp_path, defect):
    values = setup(tmp_path)
    execute(values)
    ctx = values[-1].context
    if defect in ("raw_history", "producer"):
        artifacts = dict(ctx.artifacts)
        entry = artifacts["raw_0"]
        artifacts["raw_0"] = replace(
            entry,
            **(
                {
                    "payload": entry.payload[:-8]
                    + np.array([999.0], dtype="<f8").tobytes()
                }
                if defect == "raw_history"
                else {"producer_key": "1" * 64}
            ),
        )
        ctx = replace(ctx, artifacts=artifacts)
    elif defect in ("row_index", "entity_ids"):
        table = ctx.tables["person"].copy()
        if defect == "row_index":
            table.index = table.index[::-1]
        else:
            table["person_id"] = table.person_id.astype(float)
        ctx = replace(ctx, tables={"person": table})
    elif defect == "seed":
        ctx = replace(ctx, params={**ctx.params, "seed": 99})
    else:
        ctx = replace(ctx, node=replace(ctx.node, outputs=ctx.node.outputs[:-1]))
    with pytest.raises(ValueError):
        PropertyIncomeDrawColumnsKernel().run(ctx)


@pytest.mark.parametrize(
    "changes",
    [
        {"features": ("age", "earnings")},
        {"features": (PROPERTY_REPORTED_TOTAL, PROPERTY_COMPONENTS[0])},
        {"donor_population": "recipient"},
        {"seed": True},
        {"n_estimators": 0},
    ],
)
def test_invalid_fragment_declarations_refuse(changes):
    arguments = dict(
        donor_population="donor",
        recipient_population="recipient",
        features=FEATURES,
        seed=811,
        n_estimators=2,
        scales=(1.0,) * 4,
        atol=1e-10,
        rtol=1e-12,
    )
    with pytest.raises(ValueError):
        property_income_nodes("property", **{**arguments, **changes})
