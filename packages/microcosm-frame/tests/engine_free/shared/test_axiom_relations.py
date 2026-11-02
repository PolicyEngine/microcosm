"""Relation declaration and native API contract tests; no substitute evaluator.

The recording program only returns preselected output arrays and captures the
arguments supplied to the documented dense API. It does not implement Axiom
formulas. Real dense execution remains conditional on the installed extension.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import (
    AxiomEngine,
    AxiomRelation,
    _source_relation_arguments,
    axiom_engine_ref,
)
from microcosm.frame.kernels import SimulateRulesKernel, missing_relation_inputs
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateRulesByRefKernel,
    SimulateSolveZeroKernel,
)
from microcosm.frame.schema import EntitySchema
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    GraphError,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejected,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from test_support.paths import paths_for

ROOT = paths_for("microcosm-frame").tests / "fixtures/rulespec-zz"
MODULE = ROOT / "zz/policies/tests/axiom_toy_relations.yaml"
SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))


def _relation(**updates):
    values = dict(
        predicate="toy_member",
        key="toy_member:0:1",
        current_slot=0,
        related_slot=1,
        arguments=("TaxUnit", "Person"),
        root_entity="tax_unit",
        related_entity="person",
        membership_column="person_tax_unit_id",
        related_inputs={"toy_income": "income"},
        cardinality="many",
        period="2025",
    )
    values.update(updates)
    return AxiomRelation(**values)


def _adapter(*relations, **updates):
    values = dict(
        module=MODULE,
        rulespec_roots=(ROOT,),
        schema=SCHEMA,
        entity_names={
            "person": "Person",
            "household": "Household",
            "tax_unit": "TaxUnit",
        },
        nesting={"tax_unit": "household"},
        relations=relations or (_relation(),),
    )
    values.update(updates)
    return AxiomEngine(**values)


def _frame(ids=(3, 1, 2), groups=(11, 10, 10), income=None, roles=None):
    group_ids = sorted(set(groups))
    people = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": np.ones(len(ids), dtype=np.int64),
            "person_tax_unit_id": groups,
            "income": np.ones(len(ids)) if income is None else income,
            "role": ["a"] * len(ids) if roles is None else roles,
        }
    )
    return Frame(
        tables={
            "person": people,
            "household": pd.DataFrame({"household_id": [1]}),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": group_ids, "scale": np.ones(len(group_ids))}
            ),
        },
        schema=SCHEMA,
        weights={"household": Weights(np.ones(1), WeightKind.DESIGN)},
    )


@dataclass
class _ContractBatch:
    offsets: np.ndarray
    inputs: dict[str, np.ndarray]
    related_row_count: int | None = None


class _RecordingProgram:
    """Records the native signature; never evaluates a RuleSpec expression."""

    root_entity = "TaxUnit"
    root_inputs = ["scale"]

    def __init__(self, relations=None):
        self.relations = (
            relations
            if relations is not None
            else [
                SimpleNamespace(
                    name="toy_member",
                    key="toy_member:0:1",
                    current_slot=0,
                    related_slot=1,
                    related_inputs=("toy_income",),
                )
            ]
        )
        self.calls = []

    def execute(self, **arguments):
        self.calls.append(arguments)
        return {"outputs": {"toy_total": np.zeros(len(arguments["inputs"]["scale"]))}}

    execute_f64 = execute


def _wire(adapter, monkeypatch, program=None):
    program = program or _RecordingProgram()
    monkeypatch.setattr(
        adapter,
        "_program",
        lambda entity, **kw: program if entity == "tax_unit" else None,
    )
    monkeypatch.setattr(
        adapter,
        "_derived_metadata",
        lambda: {
            "toy_total": SimpleNamespace(
                entity="TaxUnit", dtype="decimal", period="Year"
            )
        },
    )
    dense = ModuleType("axiom_rules_engine.dense")
    dense.DenseRelationBatch = _ContractBatch
    monkeypatch.setitem(sys.modules, "axiom_rules_engine.dense", dense)
    return program


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"predicate": ""}, "non-empty"),
        ({"current_slot": True}, "binary coordinates"),
        ({"related_slot": 0}, "binary coordinates"),
        ({"arguments": ("TaxUnit",)}, "two engine"),
        ({"arguments": "ab"}, "arguments must be a sequence"),
        ({"arguments": {"TaxUnit": 0, "Person": 1}}, "arguments must be a sequence"),
        ({"key": "toy_member:1:0"}, "key must agree"),
        ({"cardinality": "optional"}, "cardinality"),
        ({"related_inputs": {"toy_income": ""}}, "related_inputs"),
        ({"role_column": "role"}, "supplied together"),
        ({"role_column": "role", "roles": ()}, "unique non-empty"),
        ({"role_column": "role", "roles": "head"}, "roles must be a sequence"),
        ({"role_column": "role", "roles": {"head": 1}}, "roles must be a sequence"),
        ({"role_column": "role", "roles": ("a", "a")}, "unique non-empty"),
    ],
)
def test_manifest_shape_refuses_invalid_declarations(updates, match):
    with pytest.raises(ValueError, match=match):
        _relation(**updates)


@pytest.mark.parametrize(
    "relation,match",
    [
        (_relation(root_entity="household"), "partition"),
        (_relation(arguments=("Person", "TaxUnit")), "argument slots"),
    ],
)
def test_manifest_matches_physical_partition_and_entity_slots(relation, match):
    with pytest.raises(ValueError, match=match):
        _adapter(relation)


def test_duplicate_binding_and_undeclared_period_refused():
    from microcosm.frame.adapters.axiom import AxiomPeriod

    with pytest.raises(ValueError, match="Duplicate relation"):
        _adapter(_relation(), _relation())
    with pytest.raises(ValueError, match="period.*not declared"):
        _adapter(periods={"2026": AxiomPeriod("2026-01-01", "2026-12-31", "year")})
    with pytest.raises(TypeError, match="AxiomRelation"):
        _adapter(relations=({},))


def test_source_tuple_and_compiled_coordinates_are_verified(monkeypatch, tmp_path):
    adapter = _adapter()
    program = _wire(adapter, monkeypatch)
    adapter.assert_declared_relations("tax_unit")
    assert _source_relation_arguments(MODULE, (ROOT,), "toy_member") == (
        "TaxUnit",
        "Person",
    )
    assert _source_relation_arguments(
        MODULE, (ROOT,), "zz:policies/tests/axiom_toy_relations#relation.toy_member"
    ) == ("TaxUnit", "Person")
    for field, value, match in [
        ("current_slot", 1, "slots disagree"),
        ("related_slot", 0, "slots disagree"),
        ("name", "other", "slots disagree"),
        ("key", "other:0:1", "explicit relation declaration"),
        ("related_inputs", ("other",), "input union"),
    ]:
        previous = getattr(program.relations[0], field)
        setattr(program.relations[0], field, value)
        with pytest.raises((ValueError, NotImplementedError), match=match):
            adapter.assert_declared_relations("tax_unit")
        setattr(program.relations[0], field, previous)
    bad = tmp_path / "rulespec-zz"
    bad.mkdir()
    source = bad / "x.yaml"
    source.write_text(
        MODULE.read_text().replace("[TaxUnit, Person]", "[Person, TaxUnit]")
    )
    scoped = bad / "zz/policies/tests/alias.yaml"
    scoped.parent.mkdir(parents=True)
    scoped.write_text(
        MODULE.read_text().replace("predicate: toy_member", "predicate: authored_alias")
    )
    assert _source_relation_arguments(
        scoped, (bad,), "zz:policies/tests/alias#relation.toy_member"
    ) == ("TaxUnit", "Person")
    adapter._module, adapter._rulespec_roots = source, (bad,)
    with pytest.raises(ValueError, match="source arguments"):
        adapter.assert_declared_relations("tax_unit")


@pytest.mark.parametrize(
    "problem,match",
    [
        ("arity", "source must be binary"),
        ("outside", "escapes"),
        ("import_type", "imports must name"),
        ("missing", "missing or ambiguous"),
        ("ambiguous", "missing or ambiguous"),
        ("conflicting", "conflicting source"),
    ],
)
def test_source_relation_declaration_integrity(tmp_path, problem, match):
    root = tmp_path / "rulespec-zz"
    root.mkdir()
    source = root / "x.yaml"
    source.write_text(MODULE.read_text())
    roots = (root,)
    if problem == "arity":
        source.write_text(source.read_text().replace("arity: 2", "arity: 3"))
    elif problem == "outside":
        roots = (tmp_path / "other",)
    elif problem == "import_type":
        source.write_text(source.read_text() + "\nimports: [123]\n")
    elif problem == "missing":
        source.write_text(source.read_text() + "\nimports: [missing.yaml]\n")
    elif problem == "ambiguous":
        other = tmp_path / "other"
        (other / "zz").mkdir(parents=True)
        (root / "zz").mkdir()
        (root / "zz/import.yaml").write_text("{}")
        (other / "zz/import.yaml").write_text("{}")
        roots = (root, other)
        source.write_text(source.read_text() + "\nimports: [zz:import]\n")
    elif problem == "conflicting":
        (root / "other.yaml").write_text(
            source.read_text().replace("[TaxUnit, Person]", "[Person, TaxUnit]")
        )
        source.write_text(source.read_text() + "\nimports: [other.yaml]\n")
    with pytest.raises(ValueError, match=match):
        _source_relation_arguments(source, roots, "toy_member")


def test_extra_manifest_and_undeclared_relation_refuse(monkeypatch):
    adapter = _adapter()
    _wire(adapter, monkeypatch, _RecordingProgram(relations=[]))
    with pytest.raises(ValueError, match="compiled keys"):
        adapter.assert_declared_relations("tax_unit")
    adapter = _adapter(relations=None)
    _wire(adapter, monkeypatch)
    with pytest.raises(NotImplementedError, match="explicit relation declaration"):
        adapter.materialize(_frame(), ["toy_total"], "2025")


def test_related_input_union_and_graph_dependencies(monkeypatch):
    relation = _relation(
        related_inputs={"toy_income": "income", "toy_eligible": "eligible"},
        role_column="role",
        roles=("a",),
    )
    adapter = _adapter(relation)
    program = _wire(adapter, monkeypatch)
    program.relations.append(
        SimpleNamespace(
            name="toy_member",
            key="toy_member:0:1",
            current_slot=0,
            related_slot=1,
            related_inputs=("toy_eligible",),
        )
    )
    assert adapter.variables() == ["scale", "toy_eligible", "toy_income"]
    assert adapter.relation_dependencies("tax_unit") == (
        ("person", "eligible"),
        ("person", "income"),
        ("person", "person_id"),
        ("person", "person_tax_unit_id"),
        ("person", "role"),
        ("tax_unit", "tax_unit_id"),
    )
    frame = _frame()
    frame._tables["person"]["eligible"] = True
    adapter.materialize(frame, ["toy_total"], "2025")
    batch = program.calls[-1]["relations"]["toy_member:0:1"]
    assert set(batch.inputs) == {"toy_income", "toy_eligible"}
    assert batch.offsets.tolist() == [0, 2, 3]
    assert batch.inputs["toy_eligible"].tolist() == [True, True, True]


def test_relation_without_related_inputs_states_native_row_count(monkeypatch):
    adapter = _adapter(_relation(related_inputs={}))
    program = _RecordingProgram()
    program.relations[0].related_inputs = ()
    _wire(adapter, monkeypatch, program)
    adapter.materialize(_frame(), ["toy_total"], "2025")
    batch = program.calls[-1]["relations"]["toy_member:0:1"]
    assert batch.inputs == {}
    assert batch.offsets.tolist() == [0, 2, 3]
    assert batch.related_row_count == 3


@pytest.mark.parametrize("mode", ("override", "solve"))
@pytest.mark.parametrize("column", ("toy_income", "income"))
def test_bridge_aliased_related_input_override_refused(monkeypatch, mode, column):
    adapter = _adapter()
    program = _wire(adapter, monkeypatch)
    kernel = SimulateCounterfactualKernel({"fixture": adapter})
    component = {
        "engine_ref": "fixture",
        "variables": ["toy_total"],
        "period": "2025",
        "input_overrides": [],
    }
    options = {}
    if mode == "override":
        component["input_overrides"] = [
            {"entity": "person", "column": column, "value": 100}
        ]
    else:
        options = {
            "trial": ("tax_unit", np.asarray([100.0, 200.0])),
            "solve_input": {
                "engine_ref": "fixture",
                "entity": "person",
                "column": column,
            },
        }
    with pytest.raises(NotImplementedError, match="aliased related Axiom inputs"):
        kernel._evaluate(_frame(), [component], [], **options)
    assert program.calls == []


@pytest.mark.parametrize("mode", ("override", "solve"))
def test_bridge_unaliased_related_input_override_reaches_native_batch(
    monkeypatch, mode
):
    adapter = _adapter(_relation(related_inputs={"toy_income": "toy_income"}))
    program = _wire(adapter, monkeypatch)
    frame = _frame()
    frame._tables["person"]["toy_income"] = frame.table("person")["income"]
    kernel = SimulateCounterfactualKernel({"fixture": adapter})
    component = {
        "engine_ref": "fixture",
        "variables": ["toy_total"],
        "period": "2025",
        "input_overrides": [],
    }
    options = {}
    if mode == "override":
        component["input_overrides"] = [
            {"entity": "person", "column": "toy_income", "value": 100}
        ]
        expected = [100.0, 100.0, 100.0]
    else:
        options = {
            "trial": ("tax_unit", np.asarray([100.0, 200.0])),
            "solve_input": {
                "engine_ref": "fixture",
                "entity": "person",
                "column": "toy_income",
            },
        }
        expected = [100.0, 100.0, 200.0]
    kernel._evaluate(frame, [component], [], **options)
    batch = program.calls[-1]["relations"]["toy_member:0:1"]
    assert batch.inputs["toy_income"].tolist() == expected


def _bridge_options(mode, entity, column):
    component = {
        "engine_ref": "fixture",
        "variables": ["toy_total"],
        "period": "2025",
        "input_overrides": [],
    }
    if mode == "override":
        component["input_overrides"] = [
            {"entity": entity, "column": column, "value": 100}
        ]
        return component, {}
    return component, {
        "trial": ("tax_unit", np.asarray([100.0, 200.0])),
        "solve_input": {"engine_ref": "fixture", "entity": entity, "column": column},
    }


@pytest.mark.parametrize("mode", ("override", "solve"))
@pytest.mark.parametrize(
    ("related_inputs", "entity", "column"),
    (
        ({"toy_income": "income"}, "tax_unit", "toy_income"),
        ({"toy_income": "income"}, "household", "toy_income"),
        ({"toy_income": "toy_income"}, "tax_unit", "toy_income"),
    ),
)
def test_bridge_override_on_another_entity_refused(
    monkeypatch, mode, related_inputs, entity, column
):
    """An override on a table no batch reads would change nothing, silently."""
    adapter = _adapter(_relation(related_inputs=related_inputs))
    program = _wire(adapter, monkeypatch)
    component, options = _bridge_options(mode, entity, column)
    kernel = SimulateCounterfactualKernel({"fixture": adapter})
    with pytest.raises(ValueError, match="belongs to 'person'"):
        kernel._evaluate(_frame(), [component], [], **options)
    assert program.calls == []


@pytest.mark.parametrize("mode", ("override", "solve"))
def test_bridge_root_input_sharing_a_related_name_stays_overridable(monkeypatch, mode):
    adapter = _adapter()
    program = _wire(adapter, monkeypatch)
    program.root_inputs = ["scale", "toy_income"]
    frame = _frame()
    frame._tables["tax_unit"]["toy_income"] = 1.0
    component, options = _bridge_options(mode, "tax_unit", "toy_income")
    kernel = SimulateCounterfactualKernel({"fixture": adapter})
    kernel._evaluate(frame, [component], [], **options)
    assert set(program.calls[-1]["inputs"]["toy_income"].tolist()) != {1.0}
    batch = program.calls[-1]["relations"]["toy_member:0:1"]
    assert batch.inputs["toy_income"].tolist() == [1.0, 1.0, 1.0]


class _RelationPopulation(KernelBase):
    """Three people, two units, and a mask that keeps one member of the couple."""

    ref = "test.relation_population@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.CREATE,
        consumes_se=False,
    )

    def run(self, context):
        frame = _frame(ids=(1, 2, 3), groups=(10, 10, 11), income=(10.0, 20.0, 30.0))
        frame.table("person").drop(columns="role", inplace=True)
        frame.table("person")["selected"] = np.array([True, False, True])
        return KernelResult(frame=frame)


_STRUCTURAL = (
    ("person", "person_id"),
    ("person", "person_tax_unit_id"),
    ("tax_unit", "tax_unit_id"),
)


def _relation_node(*, rows="all", omitted=(), kernel=SimulateRulesByRefKernel.ref):
    person = ["selected"] if ("person", "income") in omitted else ["income", "selected"]
    return Node(
        "relations",
        kernel,
        population="create",
        inputs=(
            Slice("person", tuple(person), rows=rows),
            Slice("tax_unit", ("scale",)),
        ),
        outputs=(Owned("tax_unit", "toy_total", "float64"),),
        params={"engine_ref": "fixture", "variables": ("toy_total",), "period": "2025"},
    )


def _relation_graph(node):
    create = Node(
        "create",
        _RelationPopulation.ref,
        sources=("raw",),
        structural=StructuralDelta.CREATE,
        outputs=(
            Owned("person", "income", "float64"),
            Owned("person", "selected", "bool"),
            Owned("tax_unit", "scale", "float64"),
        ),
    )
    return Graph("relations", (SourceRef("raw", "raw-bytes-v1"),), (create, node))


def _graph_with_identity_slice():
    node = _relation_node()
    return _relation_graph(
        replace(node, inputs=(Slice("person", ("person_id",)), *node.inputs[1:]))
    )


def _run_relation_graph(tmp_path, adapter, node):
    registry = KernelRegistry()
    registry.register(_RelationPopulation())
    registry.register(SimulateRulesByRefKernel({"fixture": adapter}))
    raw = tmp_path / "raw.txt"
    raw.write_text("source")
    return run_graph(
        compile_graph(_relation_graph(node)),
        sources={"raw": raw},
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
        resume="forbid",
        decisions=(),
    )


def test_graph_refuses_a_row_mask_that_drops_a_relation_member(monkeypatch, tmp_path):
    """Both units keep a member here, so only the declared mask shows the loss."""
    adapter = _adapter()
    program = _wire(adapter, monkeypatch)
    with pytest.raises((NodeRejected, ValueError), match="whole-table relation inputs"):
        _run_relation_graph(tmp_path, adapter, _relation_node(rows="selected"))
    assert program.calls == []


def test_graph_gives_the_engine_every_member_of_a_whole_slice(monkeypatch, tmp_path):
    adapter = _adapter()
    program = _wire(adapter, monkeypatch)
    _run_relation_graph(tmp_path, adapter, _relation_node())
    batch = program.calls[-1]["relations"]["toy_member:0:1"]
    assert batch.offsets.tolist() == [0, 2, 3]
    assert batch.inputs["toy_income"].tolist() == [10.0, 20.0, 30.0]


def _entry_points(adapter):
    bridge = {
        "input_overrides": "[]",
        "output_coefficients": "[]",
    }
    solve = {
        **bridge,
        "solve_input": "{}",
        "bracket": (0.0, 1.0),
        "tolerance": 1.0,
        "iterations": 1,
    }
    return {
        "rules": (SimulateRulesKernel("fixture", adapter), {}),
        "by_ref": (SimulateRulesByRefKernel({"fixture": adapter}), {}),
        "counterfactual": (SimulateCounterfactualKernel({"fixture": adapter}), bridge),
        "solve": (SimulateSolveZeroKernel({"fixture": adapter}), solve),
    }


@pytest.mark.parametrize("entry", ("rules", "by_ref", "counterfactual", "solve"))
@pytest.mark.parametrize(
    "problem",
    ({"rows": "selected"}, {"omitted": (("person", "income"),)}),
    ids=("masked", "undeclared"),
)
def test_every_rules_entry_point_refuses_a_partial_relation(
    monkeypatch, entry, problem
):
    adapter = _adapter()
    program = _wire(adapter, monkeypatch)
    kernel, params = _entry_points(adapter)[entry]
    node = _relation_node(**problem)
    node = replace(node, params={**node.params, **params})
    frame = _frame(ids=(1, 3), groups=(10, 11))
    context = KernelContext(
        node=node,
        tables={entity: frame.table(entity) for entity in frame.entities},
        weights={},
        strata=pd.Series(dtype=object),
        params=node.params,
        rng=np.random.default_rng(0),
        sources={},
    )
    with pytest.raises(ValueError, match="whole-table relation inputs"):
        kernel.run(context)
    assert program.calls == []


@settings(max_examples=40, deadline=None, database=None)
@given(
    entity=st.sampled_from(("person", "tax_unit", "household")),
    masked=st.booleans(),
    declared=st.booleans(),
)
def test_a_node_passes_only_with_every_relation_dependency_whole(
    entity, masked, declared
):
    """Any mask on a dependency's entity or an undeclared data column refuses."""
    adapter = _adapter()
    inputs = [
        Slice("person", ("income",) if declared else ("selected",)),
        Slice("tax_unit", ("scale",)),
        Slice("household", ("region",)),
    ]
    if masked:
        inputs = [
            replace(item, columns=(*item.columns, "keep"), rows="keep")
            if item.entity == entity
            else item
            for item in inputs
        ]
    node = replace(_relation_node(), inputs=tuple(inputs))
    if not declared or (masked and entity != "household"):
        with pytest.raises(ValueError, match="whole-table relation inputs"):
            SimulateRulesKernel._require_relation_inputs(node, adapter)
    else:
        SimulateRulesKernel._require_relation_inputs(node, adapter)


def test_structural_columns_need_no_slice_because_no_graph_can_declare_them():
    """A table carries its id and memberships; compile refuses a slice of them."""
    assert missing_relation_inputs(_STRUCTURAL, (), SCHEMA) == []
    assert missing_relation_inputs(
        _STRUCTURAL, (Slice("person", ("income",), rows="selected"),), SCHEMA
    ) == [("person", "person_id"), ("person", "person_tax_unit_id")]
    graph = _graph_with_identity_slice()
    with pytest.raises(GraphError, match="which no node owns"):
        compile_graph(graph)


def test_absent_relation_declaration_needs_no_extra_slice():
    adapter = _adapter(relations=None)
    node = _relation_node(rows="selected", omitted=(("person", "income"),))
    SimulateRulesKernel._require_relation_inputs(node, adapter)


@settings(max_examples=40, deadline=None, database=None)
@given(
    st.lists(
        st.integers(min_value=1, max_value=500), min_size=1, max_size=20, unique=True
    ),
    st.data(),
)
def test_related_rows_order_by_identity_and_preserve_equal_values(ids, data):
    permutation = data.draw(st.permutations(ids))
    groups = [10 + value % 3 for value in permutation]
    adapter = _adapter()
    # Capture the native API batch without implementing the engine evaluator.
    with pytest.MonkeyPatch.context() as patch:
        program = _wire(adapter, patch)
        frame = _frame(permutation, groups, income=np.asarray(permutation, dtype=float))
        adapter.materialize(frame, ["toy_total"], "2025")
        batch = program.calls[-1]["relations"]["toy_member:0:1"]
        expected = [
            identity
            for group in sorted(set(groups))
            for identity in sorted(ids)
            if 10 + identity % 3 == group
        ]
        assert batch.inputs["toy_income"].tolist() == expected
        assert batch.offsets.tolist() == [
            0,
            *np.cumsum([groups.count(group) for group in sorted(set(groups))]).tolist(),
        ]
        assert len(batch.inputs["toy_income"]) == len(set(ids))
        equal = _frame(permutation, groups)
        adapter.materialize(equal, ["toy_total"], "2025")
        batch = program.calls[-1]["relations"]["toy_member:0:1"]
        assert batch.inputs["toy_income"].tolist() == [1.0] * len(ids)


def test_role_selection_empty_sets_cardinality_and_arithmetic(monkeypatch):
    adapter = _adapter(_relation(role_column="role", roles=("b",)), arithmetic="f64")
    program = _wire(adapter, monkeypatch)
    adapter.materialize(_frame(roles=["b", "a", "a"]), ["toy_total"], "2025")
    batch = program.calls[-1]["relations"]["toy_member:0:1"]
    assert batch.offsets.tolist() == [0, 0, 1]
    adapter = _adapter(_relation(cardinality="one"))
    _wire(adapter, monkeypatch)
    with pytest.raises(ValueError, match="exactly one"):
        adapter.materialize(_frame(), ["toy_total"], "2025")


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "missing related columns"),
        ("dangling", "dangling membership"),
        ("duplicate", "unique non-null"),
        ("period", "requested period"),
        ("cross_household", "nest in exactly one"),
    ],
)
def test_relation_data_integrity_refused(monkeypatch, problem, match):
    adapter = _adapter()
    _wire(adapter, monkeypatch)
    frame = _frame()
    if problem == "missing":
        frame._tables["person"].drop(columns="income", inplace=True)
    elif problem == "dangling":
        frame._tables["person"].loc[0, "person_tax_unit_id"] = 999
    elif problem == "duplicate":
        frame._tables["person"].loc[0, "person_id"] = 1
    elif problem == "cross_household":
        frame._tables["person"].loc[1, "person_household_id"] = 2
    with pytest.raises(ValueError, match=match):
        adapter.materialize(
            frame, ["toy_total"], "2026" if problem == "period" else "2025"
        )


def test_absent_relation_identity_and_call_are_unchanged(monkeypatch):
    absent = _adapter(relations=None)
    explicit = _adapter(relations=())
    pins = dict(
        engine_commit="a" * 40,
        wheel_sha256="b" * 64,
        rulespec_root=ROOT,
        rulespec_commit="c" * 40,
    )
    reference = json.loads(axiom_engine_ref(absent, **pins))
    assert "relations" not in reference
    assert json.loads(axiom_engine_ref(explicit, **pins))["relations"] == []
    program = _wire(absent, monkeypatch, _RecordingProgram(relations=[]))
    absent.materialize(_frame(), ["toy_total"], "2025")
    assert "relations" not in program.calls[-1]
    declared = json.loads(axiom_engine_ref(_adapter(), **pins))
    assert declared["relations"] == [_relation().to_dict()]
    assert "roles" not in declared["relations"][0]


def test_relation_manifest_mapping_is_immutable():
    source = {"toy_income": "income"}
    relation = _relation(related_inputs=source)
    source["toy_income"] = "changed"
    assert relation.related_inputs["toy_income"] == "income"
    with pytest.raises(TypeError):
        relation.related_inputs["toy_income"] = "changed"
    assert replace(relation, role_column="role", roles=("a",)).to_dict()["roles"] == [
        "a"
    ]


_ENGINE = importlib.util.find_spec("axiom_rules_engine")
if _ENGINE is not None:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram
else:
    NativeCompiledDenseProgram = None


@pytest.mark.skipif(
    NativeCompiledDenseProgram is None,
    reason="Axiom dense native extension is not installed",
)
def test_real_dense_relation_fixture():
    predicate = "zz:policies/tests/axiom_toy_relations#relation.toy_member"
    adapter = _adapter(
        _relation(
            predicate=predicate,
            key=f"{predicate}:0:1",
            related_inputs={"toy_income": "toy_income"},
        )
    )
    frame = _frame()
    frame._tables["person"].rename(columns={"income": "toy_income"}, inplace=True)
    result = adapter.materialize(frame, ["toy_total"], "2025")
    assert result["toy_total"].tolist() == [2.0, 1.0]
