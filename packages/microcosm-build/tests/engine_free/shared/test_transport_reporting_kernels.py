"""Executed reporting classifiers over invented benefit-unit populations."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from microcosm.build.transport.artifact_types import INCOME_CELLS_TYPE
from microcosm.build.transport.compose import compose_transport_graph
from microcosm.build.transport.reporting_kernels import (
    REPORTING_CARRY_INCOME_DECILE,
    REPORTING_INCOME_DECILE,
    REPORTING_RECEIPT_GROUP,
    decode_income_cells,
    register_reporting_kernels,
)
from microcosm.build.transport.takeup_kernels import reporting_cell_declarations
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
    NumericScope,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_entitlement import make_entitlement_fixture
from test_support.paths import paths_for

_ENTITIES = ("person", "household", "family")
_PATHS = paths_for("microcosm-build")
_CELL_TYPES = ("single", "couple", "sole_parent", "couple_with_children")


def _income_spec():
    return {
        "entity": "family",
        "person_entity": "person",
        "unit_id_column": "family_id",
        "membership_column": "person_family_id",
        "weight_entity": "household",
        "expected_weight_kind": "calibrated",
        "tie_key": "family_id",
        "boundary_scenario": "S0",
        "n_bins": 10,
        "rank_method": "weighted_midpoint",
        "components": [
            {
                "entity": "person",
                "column": "market",
                "kind": "market",
                "tax_basis": "before_tax",
                "annualisation_factor": 1,
            },
            {
                "entity": "person",
                "column": "main_amount",
                "kind": "benefit",
                "tax_basis": "before_tax",
                "annualisation_factor": 52,
                "receipt_column": "receives_main",
            },
            {
                "entity": "person",
                "column": "pension_amount",
                "kind": "benefit",
                "tax_basis": "before_tax",
                "annualisation_factor": 52,
                "receipt_column": "receives_pension",
            },
        ],
        "excluded_columns": [{"entity": "family", "column": "housing_support"}],
    }


def _receipt_spec():
    return {
        "entity": "family",
        "person_entity": "person",
        "unit_id_column": "family_id",
        "membership_column": "person_family_id",
        "predicates": {
            "superannuitant": {
                "entity": "person",
                "columns": ["receives_pension", "receives_veteran"],
                "reduction": "any",
            },
            "beneficiary": {
                "entity": "person",
                "columns": ["receives_main"],
                "reduction": "any",
            },
        },
        "priority": ["superannuitant", "beneficiary"],
        "default": "other",
    }


def _frame(*, weight_kind=WeightKind.CALIBRATED):
    units = np.arange(40, dtype=np.int64)
    person_units = np.repeat(units, 2)
    first = np.tile([True, False], len(units))
    second = ~first
    person = pd.DataFrame(
        {
            "person_id": np.arange(80, dtype=np.int64),
            "person_household_id": person_units,
            "person_family_id": person_units,
            "market": np.where(first, person_units * 10000, 0).astype(np.float64),
            "main_amount": np.full(80, 10, dtype=np.float64),
            "pension_amount": np.full(80, 20, dtype=np.float64),
            "receives_main": first & np.isin(person_units % 4, [0, 3]),
            "receives_pension": second & np.isin(person_units % 4, [1, 3]),
            "receives_veteran": np.zeros(80, dtype=bool),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": units,
            # Clones sharing a source identity remain distinct unit observations.
            "source_household_id": np.repeat(np.arange(20, dtype=np.int64), 2),
        }
    )
    family = pd.DataFrame(
        {
            "family_id": units,
            "family_household_id": units,
            "family_type": pd.Series(np.tile(_CELL_TYPES, 10), dtype="string"),
            "housing_support": np.full(40, 1000000, dtype=np.float64),
        }
    )
    return Frame(
        {"person": person, "household": household, "family": family},
        EntitySchema(group_entities=("household", "family")),
        {"household": Weights(np.ones(40), weight_kind)},
        pd.Series(["all"] * 80, name="stratum"),
    )


def _node(kernel, *, population="S0", scenario="S0", spec=None):
    receipt = kernel is REPORTING_RECEIPT_GROUP
    spec = (_receipt_spec() if receipt else _income_spec()) if spec is None else spec
    carry = kernel is REPORTING_CARRY_INCOME_DECILE
    params = {
        "classifier": canonical_json(spec).decode(),
        "classifier_sha256": sha256(canonical_json(spec)).hexdigest(),
        "scenario": scenario,
    }
    if not receipt:
        params["entities"] = _ENTITIES
    person_columns = (
        ("receives_main", "receives_pension", "receives_veteran")
        if receipt
        else (
            "market",
            "main_amount",
            "pension_amount",
            "receives_main",
            "receives_pension",
        )
    )
    return Node(
        f"{population}.{kernel.ref.split('.')[1].split('@')[0]}",
        kernel.ref,
        population=population,
        inputs=(
            Slice("person", person_columns),
            Slice("household", ("source_household_id",)),
            Slice("family", ("family_household_id", "family_type")),
        ),
        outputs=(
            Owned(
                "family",
                "receipt_group" if receipt else "income_decile",
                "string" if receipt else "int64",
            ),
        ),
        params=params,
        artifact_outputs=(ArtifactOutput("baseline_cells", INCOME_CELLS_TYPE),)
        if kernel is REPORTING_INCOME_DECILE
        else (),
        artifact_inputs=(
            ArtifactInput(
                "baseline_cells",
                "S0.income_decile",
                "baseline_cells",
                INCOME_CELLS_TYPE,
            ),
        )
        if carry
        else (),
    )


def _artifact(payload, *, kind=INCOME_CELLS_TYPE):
    return ArtifactValue(payload, kind, "a" * 64, "b" * 64, NumericScope())


def _context(node, *, frame=None, artifacts=None):
    frame = _frame() if frame is None else frame
    return KernelContext(
        node=node,
        tables={entity: frame.table(entity).copy() for entity in _ENTITIES},
        weights={"household": frame.resolve_weights("household")},
        strata=frame.strata.copy(),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={},
        artifacts={} if artifacts is None else artifacts,
    )


def _baseline_payload():
    return REPORTING_INCOME_DECILE.run(
        _context(_node(REPORTING_INCOME_DECILE))
    ).artifacts["baseline_cells"]


def test_baseline_uses_annual_assigned_income_calibrated_unit_weights_and_reports_shares():
    result = REPORTING_INCOME_DECILE.run(_context(_node(REPORTING_INCOME_DECILE)))
    expected = np.repeat(np.arange(1, 11, dtype=np.int64), 4)
    np.testing.assert_array_equal(result.columns["family", "income_decile"], expected)
    assert result.receipt["weighted_shares"] == pytest.approx([0.1] * 10)
    assert result.receipt["scenario"] == "S0"
    cells = decode_income_cells(result.artifacts["baseline_cells"])
    assert cells["unit_ids"] == list(range(40))
    assert cells["deciles"] == expected.tolist()


def test_receipt_any_member_priority_and_mixed_count():
    result = REPORTING_RECEIPT_GROUP.run(_context(_node(REPORTING_RECEIPT_GROUP)))
    assert (
        result.columns["family", "receipt_group"].tolist()
        == ["beneficiary", "superannuitant", "other", "superannuitant"] * 10
    )
    assert str(result.columns["family", "receipt_group"].dtype) == "string"
    assert result.receipt["mixed_units"] == 10
    assert result.receipt["group_counts"] == {
        "superannuitant": 20,
        "beneficiary": 10,
        "other": 10,
    }


def test_carry_retains_baseline_cells_after_weights_income_and_receipt_change():
    node = _node(REPORTING_CARRY_INCOME_DECILE, scenario="V3")
    context = _context(
        node, artifacts={"baseline_cells": _artifact(_baseline_payload())}
    )
    context.weights["household"] = Weights(np.arange(1, 41), WeightKind.CALIBRATED)
    context.tables["person"]["market"] *= -1
    context.tables["person"]["receives_main"] = False
    result = REPORTING_CARRY_INCOME_DECILE.run(context)
    np.testing.assert_array_equal(
        result.columns["family", "income_decile"],
        np.repeat(np.arange(1, 11, dtype=np.int64), 4),
    )
    assert result.receipt["weighted_shares"] == pytest.approx(
        [sum(range(4 * i + 1, 4 * i + 5)) / 820 for i in range(10)]
    )


def test_veteran_receipt_on_another_member_uses_superannuitant_group():
    context = _context(_node(REPORTING_RECEIPT_GROUP))
    context.tables["person"].loc[5, "receives_veteran"] = True
    result = REPORTING_RECEIPT_GROUP.run(context)
    assert result.columns["family", "receipt_group"].loc[2] == "superannuitant"
    assert result.receipt["group_counts"]["superannuitant"] == 21
    assert result.receipt["mixed_units"] == 10


def test_income_benefit_amount_uses_assigned_receipt_and_declared_annualisation():
    spec = _income_spec()
    spec["n_bins"] = 40
    node = _node(REPORTING_INCOME_DECILE, spec=spec)
    context = _context(node)
    context.tables["person"].loc[0, "market"] = 5000
    context.tables["person"].loc[0, "main_amount"] = 300
    assigned = REPORTING_INCOME_DECILE.run(context).columns["family", "income_decile"]
    assert assigned.loc[0] == 3
    context.tables["person"].loc[0, "receives_main"] = False
    unassigned = REPORTING_INCOME_DECILE.run(context).columns["family", "income_decile"]
    assert unassigned.loc[0] == 1


@pytest.mark.parametrize("bad", [None, 1, "yes"])
def test_receipt_invalid_flags_refused_even_when_higher_priority_matches(bad):
    context = _context(_node(REPORTING_RECEIPT_GROUP))
    context.tables["person"]["receives_main"] = context.tables["person"][
        "receives_main"
    ].astype(object)
    context.tables["person"].loc[2, "receives_main"] = bad
    with pytest.raises(ValueError, match="boolean flags"):
        REPORTING_RECEIPT_GROUP.run(context)


def test_receipt_missing_flag_is_refused():
    context = _context(_node(REPORTING_RECEIPT_GROUP))
    del context.tables["person"]["receives_veteran"]
    with pytest.raises(ValueError, match="receives_veteran.*missing"):
        REPORTING_RECEIPT_GROUP.run(context)


@pytest.mark.parametrize(
    "kernel",
    [REPORTING_INCOME_DECILE, REPORTING_CARRY_INCOME_DECILE, REPORTING_RECEIPT_GROUP],
)
def test_reporting_refuses_wrong_owned_output_dtype(kernel):
    node = _node(kernel)
    node = replace(node, outputs=(replace(node.outputs[0], dtype="float64"),))
    artifacts = (
        {"baseline_cells": _artifact(_baseline_payload())}
        if kernel is REPORTING_CARRY_INCOME_DECILE
        else None
    )
    with pytest.raises(ValueError, match="must own exactly"):
        kernel.run(_context(node, artifacts=artifacts))


@pytest.mark.parametrize(
    "kernel", [REPORTING_INCOME_DECILE, REPORTING_CARRY_INCOME_DECILE]
)
@pytest.mark.parametrize("kind", [None, WeightKind.DESIGN])
def test_income_refuses_missing_or_noncalibrated_source_weights(kernel, kind):
    node = _node(kernel)
    context = _context(
        node,
        artifacts={"baseline_cells": _artifact(_baseline_payload())}
        if kernel is REPORTING_CARRY_INCOME_DECILE
        else None,
    )
    context.weights.clear()
    if kind is not None:
        context.weights["household"] = Weights(np.ones(40), kind)
    with pytest.raises(ValueError, match="effective weights|calibrated"):
        kernel.run(context)


def test_recomputing_deciles_on_variant_is_refused():
    with pytest.raises(ValueError, match="only computes.*baseline"):
        REPORTING_INCOME_DECILE.run(
            _context(_node(REPORTING_INCOME_DECILE, scenario="V1"))
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_scenario", "S1"),
        ("entity", "household"),
        ("unit_id_column", "household_id"),
        ("n_bins", 11),
        ("classifier_sha256", "c" * 64),
    ],
)
def test_carry_refuses_artifact_classifier_mismatch(field, value):
    document = decode_income_cells(_baseline_payload())
    document[field] = value
    with pytest.raises(ValueError, match="disagree.*classifier"):
        REPORTING_CARRY_INCOME_DECILE.run(
            _context(
                _node(REPORTING_CARRY_INCOME_DECILE, scenario="V1"),
                artifacts={"baseline_cells": _artifact(canonical_json(document))},
            )
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing_id",
        "different_id",
        "duplicate_id",
        "out_of_range",
        "float_label",
        "bad_digest",
    ],
)
def test_carry_refuses_invalid_cells_and_changed_population(change):
    document = decode_income_cells(_baseline_payload())
    if change == "missing_id":
        document["unit_ids"].pop()
        document["deciles"].pop()
    elif change == "different_id":
        document["unit_ids"][-1] = 400
    elif change == "duplicate_id":
        document["unit_ids"][-1] = 38
    elif change == "out_of_range":
        document["deciles"][0] = 0
    elif change == "float_label":
        document["deciles"][0] = 1.0
    else:
        document["classifier_sha256"] = "invalid"
    with pytest.raises(ValueError, match="same unit ids|unique sorted|digest"):
        REPORTING_CARRY_INCOME_DECILE.run(
            _context(
                _node(REPORTING_CARRY_INCOME_DECILE, scenario="V1"),
                artifacts={"baseline_cells": _artifact(canonical_json(document))},
            )
        )


def test_carry_refuses_wrong_typed_artifact():
    node = _node(REPORTING_CARRY_INCOME_DECILE, scenario="V1")
    with pytest.raises(ValueError, match="type"):
        REPORTING_CARRY_INCOME_DECILE.run(
            _context(
                node,
                artifacts={
                    "baseline_cells": _artifact(
                        _baseline_payload(), kind=ArtifactType("other", 1)
                    )
                },
            )
        )


class _Create(KernelBase):
    ref = "test.reporting_create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        declaration = json.loads(context.sources["fixture"].read_text())
        if declaration != {"n_units": 40}:
            raise ValueError("Unsupported reporting fixture population.")
        return KernelResult(frame=_frame(weight_kind=WeightKind.DESIGN))


class _Reweight(KernelBase):
    ref = "test.reporting_reweight@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def run(self, context):
        values = context.weights["household"].values.copy()
        if context.params["variant"]:
            values *= np.arange(1, len(values) + 1)
        return KernelResult(weights=Weights(values, WeightKind.CALIBRATED))


class _Redraw(KernelBase):
    ref = "test.reporting_redraw@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        person = context.tables["person"]
        ids = pd.Index(person["person_id"], name="person_id")
        return KernelResult(
            columns={
                ("person", "market"): pd.Series(
                    -person["market"].to_numpy(), index=ids
                ),
                ("person", "receives_main"): pd.Series(
                    np.zeros(len(person), dtype=bool), index=ids
                ),
            }
        )


def test_native_cross_branch_artifact_preserves_baseline_cells_for_every_scenario(
    tmp_path,
):
    declaration = json.loads(
        (_PATHS.package / "src/microcosm/build/nz/scenarios.json").read_text()
    )
    frame = _frame(weight_kind=WeightKind.DESIGN)
    structural = {
        ("person", "person_id"),
        ("person", "person_household_id"),
        ("person", "person_family_id"),
        ("household", "household_id"),
        ("family", "family_id"),
    }
    create = Node(
        "open",
        _Create.ref,
        sources=("fixture",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(entity, column, str(frame.table(entity)[column].dtype))
            for entity in _ENTITIES
            for column in frame.table(entity)
            if (entity, column) not in structural
        ),
    )
    nodes = [create]
    for row in declaration["scenarios"]:
        name = row["id"]
        variant = row["tier"] == "calibration"
        nodes.append(
            Node(
                name,
                _Reweight.ref,
                structural=StructuralDelta.REWEIGHT,
                base="open",
                inputs=(Slice("household", ("source_household_id",)),),
                params={"variant": variant},
                weights=WeightTransition("household", "calibrated", mass="free"),
                mass="free",
            )
        )
        if variant:
            nodes.append(
                Node(
                    f"{name}.redraw",
                    _Redraw.ref,
                    population=name,
                    inputs=(Slice("person", ("market", "receives_main")),),
                    outputs=(
                        Owned("person", "market", "float64", rewrite=True),
                        Owned("person", "receives_main", "bool", rewrite=True),
                    ),
                )
            )
        nodes.extend(
            [
                _node(
                    REPORTING_INCOME_DECILE
                    if name == "S0"
                    else REPORTING_CARRY_INCOME_DECILE,
                    population=name,
                    scenario=name,
                ),
                _node(REPORTING_RECEIPT_GROUP, population=name, scenario=name),
            ]
        )
    registry = register_reporting_kernels(KernelRegistry())
    for kernel in (_Create(), _Reweight(), _Redraw()):
        registry.register(kernel)
    graph = Graph(
        country="toy",
        sources=(SourceRef("fixture", "raw-bytes-v1"),),
        nodes=tuple(nodes),
    )
    fixture_source = tmp_path / "population.json"
    fixture_source.write_bytes(canonical_json({"n_units": 40}))
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compile_graph(graph),
        sources={"fixture": fixture_source},
        store=store,
        kernels=registry,
    )
    baseline = manifest.populations["S0"].table("family").set_index("family_id")
    cells = reporting_cell_declarations(
        declaration["gap_conventions"]["reporting_cells"]
    )
    assert len(cells) == 43
    assert (
        len(set(zip(baseline.income_decile, baseline.family_type, strict=True))) == 40
    )
    assert set(baseline.receipt_group) == {"beneficiary", "superannuitant", "other"}
    assert set(manifest.populations) == {
        "open",
        *[row["id"] for row in declaration["scenarios"]],
    }
    for row in declaration["scenarios"]:
        name = row["id"]
        current = manifest.populations[name].table("family").set_index("family_id")
        pd.testing.assert_series_equal(current.income_decile, baseline.income_decile)
        receipt = manifest.nodes[f"{name}.receipt_group"].receipt
        assert receipt["scenario"] == name
        assert receipt["mixed_units"] == (0 if row["tier"] == "calibration" else 10)
        if row["tier"] == "calibration":
            weights = manifest.populations[name].resolve_weights("family").values
            np.testing.assert_array_equal(weights, np.arange(1, 41))
            shares = manifest.nodes[f"{name}.carry_income_decile"].receipt[
                "weighted_shares"
            ]
            assert shares == pytest.approx(
                [sum(range(4 * i + 1, 4 * i + 5)) / 820 for i in range(10)]
            )
            assert shares != pytest.approx([0.1] * 10)
            assert set(current.receipt_group) == {"superannuitant", "other"}


def test_reporting_registration_and_declarative_entitlement_composer(tmp_path):
    registry = register_reporting_kernels(KernelRegistry())
    register_reporting_kernels(registry)
    for kernel in (
        REPORTING_INCOME_DECILE,
        REPORTING_CARRY_INCOME_DECILE,
        REPORTING_RECEIPT_GROUP,
    ):
        assert registry.get(kernel.ref) is kernel
        assert kernel.implementation_hash() == kernel.implementation_hash()
    fixture = make_entitlement_fixture(tmp_path / "fixture", variants=False)
    for kernel in (
        REPORTING_INCOME_DECILE,
        REPORTING_CARRY_INCOME_DECILE,
        REPORTING_RECEIPT_GROUP,
    ):
        assert fixture.registry.kernels.get(kernel.ref) is kernel
    spec = copy.deepcopy(fixture.spec)
    classifier = _receipt_spec()
    classifier["predicates"]["superannuitant"]["columns"] = ["receives_pension"]
    spec["resources"]["entitlement_graph"]["nodes"].append(
        {
            "id": "toy.reporting_receipt",
            "kernel": REPORTING_RECEIPT_GROUP.ref,
            "population": "nz.scn.S0",
            "inputs": [
                {"entity": "person", "columns": ["receives_main", "receives_pension"]}
            ],
            "outputs": [
                {"entity": "family", "column": "receipt_group", "dtype": "string"}
            ],
            "params": {
                "classifier": classifier,
                "classifier_sha256": sha256(canonical_json(classifier)).hexdigest(),
                "scenario": "S0",
            },
        }
    )
    graph = compose_transport_graph(spec, fixture.config)
    assert graph.node("toy.reporting_receipt").kernel == REPORTING_RECEIPT_GROUP.ref
