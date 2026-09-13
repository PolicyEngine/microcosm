"""Deterministic tax splits on invented components; no native source or engine."""

import hashlib
import importlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.property_income_constants import PROPERTY_REPORTED_TOTAL
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.schema import LinkSpec
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


@pytest.fixture
def owner():
    return importlib.import_module(
        "microcosm.build.us_runtime.graph_property_tax_leaves"
    )


def people(owner):
    # First three rows have negative, zero-net, and positive anchors, each with
    # positive ordinary interest and an offsetting signed broad-property amount.
    data = pd.DataFrame(
        {
            "person_id": np.arange(8, dtype=np.int64) + 2**53 + 21,
            "person_household_id": [301, 301, 302, 302, 303, 303, 304, 304],
            owner.PROPERTY_COMPONENTS[0]: [
                100.0,
                100.0,
                100.0,
                np.nan,
                20.0,
                0.0,
                40.0,
                0.0,
            ],
            owner.PROPERTY_COMPONENTS[1]: [
                40.0,
                40.0,
                40.0,
                5.0,
                np.nan,
                0.0,
                np.nan,
                0.0,
            ],
            owner.PROPERTY_COMPONENTS[2]: [
                50.0,
                50.0,
                50.0,
                30.0,
                np.nan,
                0.0,
                80.0,
                0.0,
            ],
            owner.PROPERTY_COMPONENTS[3]: [
                -250.0,
                -190.0,
                -90.0,
                np.nan,
                5.0,
                0.0,
                np.nan,
                0.0,
            ],
            "unrelated_nullable": pd.array(
                [None, 1.0, 2.0, None, 0.0, 4.0, 5.0, 6.0], dtype="Int64"
            ),
            "geography": pd.array(["001"] * 8, dtype="string"),
            "property_components_known": [
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                True,
            ],
        }
    )
    data[PROPERTY_REPORTED_TOTAL] = [
        -60.0,
        0.0,
        100.0,
        np.nan,
        np.nan,
        0.0,
        np.nan,
        0.0,
    ]
    for column in owner.TAX_LEAF_COLUMNS:
        data[column] = 999.0
    data.index = pd.Index([13, 4, 19, 2, 7, 22, 1, 31], name="original_row")
    return data


def test_split_uses_individual_knownness_and_declared_complements(owner):
    data = people(owner)
    before = data.copy(deep=True)
    result = owner.split_property_tax_leaves(data)
    np.testing.assert_allclose(
        result.iloc[0], [68.0, 32.0, 22.4, 27.6], rtol=0, atol=1e-14
    )
    assert result.iloc[3, :2].isna().all()
    assert result.iloc[3, 2:].notna().all()
    assert result.iloc[4, :2].notna().all()
    assert result.iloc[4, 2:].isna().all()
    assert result.iloc[5].eq(0).all()
    assert result.iloc[6].notna().all()  # R and broad receipts unknown.
    assert result.index.tolist() == data.person_id.tolist()
    assert result.index.name == "person_id"
    pd.testing.assert_frame_equal(data, before, check_exact=True)
    assert result.iloc[:3].eq(result.iloc[0]).all().all()


@pytest.mark.parametrize("column", [0, 2])
@pytest.mark.parametrize("bad", [-1.0, np.inf, -np.inf])
def test_invalid_component_refuses(owner, column, bad):
    data = people(owner)
    data.iloc[0, data.columns.get_loc(owner.PROPERTY_COMPONENTS[column])] = bad
    with pytest.raises(ValueError):
        owner.split_property_tax_leaves(data)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_id",
        "float_id",
        "object_component",
        "float32_component",
        "missing_component",
    ],
)
def test_identity_and_dtype_refuse(owner, defect):
    data = people(owner)
    if defect == "duplicate_id":
        data.iloc[1, data.columns.get_loc("person_id")] = data.person_id.iloc[0]
    elif defect == "float_id":
        data["person_id"] = data.person_id.astype(float)
    elif defect == "object_component":
        data[owner.PROPERTY_COMPONENTS[0]] = data[owner.PROPERTY_COMPONENTS[0]].astype(
            object
        )
    elif defect == "float32_component":
        data[owner.PROPERTY_COMPONENTS[0]] = data[owner.PROPERTY_COMPONENTS[0]].astype(
            "float32"
        )
    else:
        data = data.drop(columns=owner.PROPERTY_COMPONENTS[2])
    with pytest.raises(ValueError):
        owner.split_property_tax_leaves(data)


def test_permutation_empty_and_exact_operation_order(owner):
    data = people(owner)
    result = owner.split_property_tax_leaves(data)
    shuffled = owner.split_property_tax_leaves(data.iloc[::-1])
    pd.testing.assert_frame_equal(shuffled.loc[result.index], result, check_exact=True)
    assert owner.split_property_tax_leaves(data.iloc[:0]).empty
    ordinary = data[owner.PROPERTY_COMPONENTS[0]].to_numpy()
    dividends = data[owner.PROPERTY_COMPONENTS[2]].to_numpy()
    expected = np.column_stack(
        (
            ordinary * 0.680,
            ordinary - ordinary * 0.680,
            dividends * 0.448,
            dividends - dividends * 0.448,
        )
    )
    np.testing.assert_array_equal(
        result.to_numpy().view("uint64"), expected.view("uint64")
    )


PROJECTION = ArtifactType("test.property_projection", 1)
RECONCILIATION = ArtifactType("test.property_reconciliation", 1)


class SourceKernel(KernelBase):
    ref = "test.property_tax_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["invented"]),
            artifacts={
                "projection": b'{"invented":true}',
                "reconciliation": b'{"numerical_fixture":true}',
            },
        )


def declaration(owner, frame):
    return owner.property_tax_leaf_nodes(
        frame,
        population="source",
        projection=ArtifactInput("projection", "source", "projection", PROJECTION),
        reconciliation=ArtifactInput(
            "reconciliation", "source", "reconciliation", RECONCILIATION
        ),
        atol=1e-10,
        rtol=1e-12,
    )


@pytest.fixture
def actual(owner, tmp_path, request):
    data = people(owner)
    if getattr(request, "param", False):
        # A separate fully known invented input, not production completion.
        for component in (owner.PROPERTY_COMPONENTS[0], owner.PROPERTY_COMPONENTS[2]):
            data[component] = data[component].fillna(10.0)
    frame = Frame(
        {
            "person": data,
            "household": pd.DataFrame(
                {
                    "household_id": [301, 302, 303, 304],
                    "retained_group_value": [3, 4, 5, 6],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights([1.0, 2.0, 0.0, 0.0], WeightKind.DESIGN)},
        metadata={"invented": True, "keep": {"nested": (1, "retained")}},
    )
    store = ContentStore(tmp_path / "store")
    path = store.put_frame(
        hashlib.sha256(b"invented-property-tax-frame").hexdigest(), frame
    )
    source = Node(
        "source",
        SourceKernel.ref,
        sources=("invented",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, column, str(frame.table(entity)[column].dtype))
            for entity in frame.entities
            for column in frame.table(entity)
            if not column.endswith("_id")
        ),
        artifact_outputs=(
            ArtifactOutput("projection", PROJECTION),
            ArtifactOutput("reconciliation", RECONCILIATION),
        ),
    )
    receiving, rebase, gate = declaration(owner, frame)
    graph = Graph(
        country="us",
        nodes=(source, receiving, rebase, gate),
        sources=(SourceRef("invented", "frame-store"),),
    )
    compiled = compile_graph(graph)
    contexts = {}

    class RecordingReceiving(owner.PropertyTaxReceivingKernel):
        def run(self, context):
            contexts["receiving"] = context
            return super().run(context)

    class RecordingSplit(owner.PropertyTaxLeavesKernel):
        def run(self, context):
            contexts["split"] = context
            return super().run(context)

    class RecordingGate(owner.PropertyTaxLeafGateKernel):
        def run(self, context):
            contexts["gate"] = context
            return super().run(context)

    registry = KernelRegistry()
    for kernel in (
        SourceKernel(),
        RecordingReceiving(),
        RecordingSplit(),
        RecordingGate(),
    ):
        registry.register(kernel)
    cold = run_graph(
        compiled,
        store=store,
        kernels=registry,
        sources={"invented": path},
        resume="auto",
    )
    warm = run_graph(
        compiled,
        store=store,
        kernels=registry,
        sources={"invented": path},
        resume="require",
    )
    return frame, compiled, cold, warm, contexts


def test_actual_graph_cold_required_preserves_full_frame_and_unknowns(owner, actual):
    before, compiled, cold, warm, contexts = actual
    after = cold.population(owner.RECEIVING_NODE)
    assert len(compiled.order) == 4
    assert not any(node.hit for node in cold.nodes.values())
    assert all(node.hit for node in warm.nodes.values())
    assert cold.key == warm.key
    for entity in before.entities:
        retained = [c for c in before.table(entity) if c not in owner.TAX_LEAF_COLUMNS]
        pd.testing.assert_frame_equal(
            after.table(entity)[retained],
            before.table(entity)[retained],
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            after.table(entity),
            warm.population(owner.RECEIVING_NODE).table(entity),
            check_exact=True,
        )
    assert after.schema == before.schema and after.links == before.links
    assert after.metadata == before.metadata and after.mass_log == before.mass_log
    pd.testing.assert_series_equal(after.strata, before.strata)
    np.testing.assert_array_equal(
        after.weights_for("household").values, before.weights_for("household").values
    )
    assert after.weights_for("household").kind is WeightKind.DESIGN
    original_ledger = cold.mass_ledger("source")
    receiving_ledger = cold.mass_ledger(owner.RECEIVING_NODE)
    assert receiving_ledger[:-1] == original_ledger
    transition = receiving_ledger[-1]
    assert transition.node_id == owner.RECEIVING_NODE
    assert transition.operation == "filter"
    assert transition.policy == "conserve"
    assert transition.before_total == transition.after_total
    assert receiving_ledger == warm.mass_ledger(owner.RECEIVING_NODE)
    result = owner.PropertyTaxLeafGateKernel().run(contexts["gate"])
    doc = json.loads(result.artifacts["verification"])
    assert doc["complete"] is False
    assert doc["numeric_verified"] is True
    assert doc["unknown_interest_persons"] == doc["unknown_dividend_persons"] == 1
    assert result.receipt["outcome"] == "evidence_absent"
    assert doc["source_authority"] is False
    assert doc["clone_pairing"] == "host_required"
    assert doc["whole_frame_verification"] == "host_required"


@pytest.mark.parametrize("actual", [True], indirect=True)
def test_complete_gate_does_not_require_retirement_or_broad_components(owner, actual):
    context = actual[-1]["gate"]
    assert context.tables["person"][owner.PROPERTY_COMPONENTS[1]].isna().any()
    result = owner.PropertyTaxLeafGateKernel().run(context)
    doc = json.loads(result.artifacts["verification"])
    assert doc["complete"] is True
    assert doc["unknown_interest_persons"] == doc["unknown_dividend_persons"] == 0
    assert result.receipt["outcome"] == "pass"


@pytest.mark.parametrize(
    "defect",
    [
        "leaf_bit",
        "unknown_leaf",
        "retirement_component",
        "person_id",
        "diagnostic_payload",
        "producer_key",
        "missing_artifact",
        "params",
    ],
)
def test_actual_gate_rejects_changed_values_or_receipt(owner, actual, defect):
    context = actual[-1]["gate"]
    table = context.tables["person"].copy(deep=True)
    artifacts = dict(context.artifacts)
    node = context.node
    if defect == "leaf_bit":
        name = owner.TAX_LEAF_COLUMNS[0]
        table.iloc[0, table.columns.get_loc(name)] = np.nextafter(
            table[name].iloc[0], np.inf
        )
    elif defect == "unknown_leaf":
        table.iloc[3, table.columns.get_loc(owner.TAX_LEAF_COLUMNS[0])] = 0.0
    elif defect == "retirement_component":
        table.iloc[0, table.columns.get_loc(owner.PROPERTY_COMPONENTS[1])] += 1.0
    elif defect == "person_id":
        table.iloc[0, table.columns.get_loc("person_id")] += 1
    elif defect == "diagnostic_payload":
        old = artifacts["rebase"]
        artifacts["rebase"] = replace(old, payload=b'{"complete":true}')
    elif defect == "producer_key":
        old = artifacts["rebase"]
        artifacts["rebase"] = ArtifactValue(
            old.payload, old.type, old.key, "0" * 64, old.numerics
        )
    elif defect == "missing_artifact":
        artifacts.pop("rebase")
    else:
        node = replace(node, params={**node.params, "unknown": "zero"})
    changed = replace(
        context,
        node=node,
        params=node.params,
        tables={"person": table},
        artifacts=artifacts,
    )
    with pytest.raises(ValueError):
        owner.PropertyTaxLeafGateKernel().run(changed)


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan"), True])
def test_tolerance_contract_refuses(owner, bad):
    with pytest.raises(ValueError):
        owner.property_tax_leaf_nodes(
            _supported_frame(owner),
            population="source",
            projection=ArtifactInput("projection", "source", "projection", PROJECTION),
            reconciliation=ArtifactInput(
                "reconciliation", "source", "reconciliation", RECONCILIATION
            ),
            atol=bad,
            rtol=1e-12,
        )


def _supported_frame(owner, *, defect=None):
    groups = pd.DataFrame(
        {"household_id": [301, 302, 303, 304], "retained": [1, 2, 3, 4]}
    )
    weights = [1.0, 1.0, 0.0, 0.0]
    if defect == "group_index":
        groups.index = pd.Index([4, 3, 2, 1], name="keep_index")
    elif defect == "id_only":
        groups = groups[["household_id"]]
    tables = {"person": people(owner), "household": groups}
    links = ()
    if defect == "links":
        links = (LinkSpec("connections", "person", "household"),)
        tables["connections"] = pd.DataFrame(
            {"person_id": [tables["person"].person_id.iloc[0]], "household_id": [301]}
        )
    frame = Frame(
        tables,
        EntitySchema(group_entities=("household",), links=links),
        {"household": Weights(weights, WeightKind.DESIGN)},
    )
    if defect == "orphan":
        # Constructor validation already refuses orphans. Exercise the factory's
        # defensive check after an invented post-construction source mutation.
        frame.person.loc[
            frame.person.person_household_id == 304, "person_household_id"
        ] = 303
    return frame


@pytest.mark.parametrize("defect", ["orphan", "group_index", "id_only", "links"])
def test_receiving_refuses_shapes_that_select_would_change_or_cannot_inspect(
    owner, defect
):
    source = _supported_frame(owner, defect=defect)
    before = source.table("household").copy(deep=True)
    with pytest.raises(ValueError):
        declaration(owner, source)
    pd.testing.assert_frame_equal(source.table("household"), before, check_exact=True)


def test_receiving_rechecks_membership_after_declaration(owner, actual):
    context = actual[-1]["receiving"]
    tables = dict(context.tables)
    tables["person"] = tables["person"].copy(deep=True)
    person = tables["person"]
    person.loc[person.person_household_id == 304, "person_household_id"] = 303
    with pytest.raises(ValueError, match="ORPHAN_GROUP:household"):
        owner.PropertyTaxReceivingKernel().run(replace(context, tables=tables))
