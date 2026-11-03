"""Binding-local view contract tests for the Axiom adapter, engine-free.

A fake engine compiles fake dense programs per engine entity and evaluates a
fixed row-local function ``f``; it implements no RuleSpec formula. Real dense
execution of a view is the skip-gated differential (b) at the end.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import (
    AxiomEngine,
    AxiomPeriod,
    AxiomRelation,
    AxiomView,
    assert_no_relations,
    axiom_engine_ref,
)
from microcosm.frame.schema import EntitySchema
from test_support.paths import paths_for

ROOT = paths_for("microcosm-frame").tests / "fixtures/rulespec-zz"
MODULE = ROOT / "zz/policies/tests/axiom_toy_relations.yaml"
SCHEMA = EntitySchema(group_entities=("household",))
NAMES = {"person": "Person", "household": "Household"}
PINS = dict(
    engine_commit="a" * 40,
    wheel_sha256="b" * 64,
    rulespec_root=ROOT,
    rulespec_commit="c" * 40,
)
OUTSIDE = {"child_amount": -7.5, "child_test": 0, "child_flag": False}


def f(age: np.ndarray, income: np.ndarray) -> dict[str, np.ndarray]:
    """The fake program's row-local function; integer-valued decimals."""
    return {
        "child_amount": age * 10.0 + income,
        "child_test": np.where(age < 18, 1, -1).astype(np.int8),
        "child_flag": age % 2 == 0,
    }


def _metadata(name, entity, dtype):
    return SimpleNamespace(name=name, entity=entity, dtype=dtype, period="Year")


CHILD_METADATA = [
    _metadata("child_amount", "Child", "decimal"),
    _metadata("child_test", "Child", "judgment"),
    _metadata("child_flag", "Child", "bool"),
]


class _Program:
    """A fake dense program: records each call and evaluates ``f``."""

    def __init__(self, entity, root_inputs, metadata, *, relations=(), function=f):
        self.root_entity = entity
        self.root_inputs = list(root_inputs)
        self.relations = list(relations)
        self.derived_metadata = list(metadata)
        self.function = function
        self.calls = []

    def execute(self, **arguments):
        self.calls.append(arguments)
        inputs = arguments["inputs"]
        values = self.function(*(inputs[name] for name in self.root_inputs))
        return {"outputs": {name: values[name] for name in arguments["outputs"]}}

    execute_f64 = execute


class _FakeEngine:
    """Stands in for ``axiom_rules_engine``: compiles fake programs by entity."""

    def __init__(self, programs):
        self.programs = programs
        self.compiled = []
        self.CompiledDenseProgram = self

    def from_file(self, module, *, rulespec_roots, entity):
        self.compiled.append(entity)
        if entity not in self.programs:
            raise ValueError(
                f"dense compilation could not find derived outputs for entity `{entity}`"
            )
        return self.programs[entity]


def _child_program(**updates):
    return _Program("Child", ("age", "income"), CHILD_METADATA, **updates)


def _view(**updates):
    values = dict(logical_entity="Child", selector="is_child", outside=OUTSIDE)
    values.update(updates)
    return AxiomView(**values)


def _adapter(monkeypatch, programs=None, *, views=(), **updates):
    values = dict(
        module=MODULE,
        rulespec_roots=(ROOT,),
        schema=SCHEMA,
        entity_names=NAMES,
        output_dtypes="graph",
        views=views if views != () else (_view(),),
    )
    values.update(updates)
    adapter = AxiomEngine(**values)
    engine = _FakeEngine({"Child": _child_program()} if programs is None else programs)
    monkeypatch.setattr(adapter, "_import_engine", lambda: engine)
    return adapter, engine


def _frame(selected, age=None, income=None, ids=None):
    n = len(selected)
    people = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1) if ids is None else ids,
            "person_household_id": np.ones(n, dtype=np.int64),
            "is_child": np.asarray(selected, dtype=bool),
            "age": np.arange(n, dtype=np.int64) if age is None else age,
            "income": np.arange(n, dtype=np.float64) * 3.0
            if income is None
            else income,
        }
    )
    return Frame(
        tables={"person": people, "household": pd.DataFrame({"household_id": [1]})},
        schema=SCHEMA,
        weights={"household": Weights(np.ones(1), WeightKind.DESIGN)},
    )


def _materialize(adapter, selected, variables, age=None, income=None):
    """Materialize through a Frame; an empty person table skips the Frame.

    A Frame refuses a table without rows (its weights cannot be empty), so a
    zero-row person table reaches the view through ``_materialize_view``
    with the same graph dtypes ``materialize`` passes it.
    """
    if len(selected):
        return adapter.materialize(_frame(selected, age, income), variables, "2025")
    table = pd.DataFrame(
        {
            "person_id": np.asarray([], dtype=np.int64),
            "person_household_id": np.asarray([], dtype=np.int64),
            "is_child": np.asarray([], dtype=bool),
            "age": np.asarray([], dtype=np.int64),
            "income": np.asarray([], dtype=np.float64),
        }
    )
    derived = adapter._derived_metadata()
    return adapter._materialize_view(
        SimpleNamespace(table=lambda entity: table),
        adapter._views[0],
        variables,
        {name: derived[name].dtype for name in variables},
        dict(period_kind="calendar_year", start="2025-01-01", end="2025-12-31"),
    )


VARIABLES = ["child_amount", "child_test", "child_flag"]


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"logical_entity": ""}, "logical_entity"),
        ({"selector": ""}, "selector"),
        ({"outside": {}}, "outside"),
        ({"outside": {"x": "zero"}}, "finite bool"),
        ({"outside": {"x": float("nan")}}, "finite bool"),
        ({"outside": {"": 0}}, "non-empty"),
    ],
)
def test_view_declaration_shape_is_refused(updates, match):
    with pytest.raises(ValueError, match=match):
        _view(**updates)


def test_view_declaration_is_immutable_and_serializes_flat():
    source = {"child_amount": 1.0}
    view = _view(outside=source)
    source["child_amount"] = 2.0
    assert view.outside["child_amount"] == 1.0
    with pytest.raises(TypeError):
        view.outside["child_amount"] = 3.0
    assert view.to_dict() == {
        "logical_entity": "Child",
        "selector": {"column": "is_child"},
        "outside": {"child_amount": 1.0},
    }


@pytest.mark.parametrize(
    "updates,error,match",
    [
        ({"views": []}, ValueError, "at least one view"),
        ({"views": [{}]}, TypeError, "AxiomView"),
        ({"views": [_view(logical_entity="Person")]}, ValueError, "maps to"),
        ({"views": [_view(), _view()]}, ValueError, "Two views"),
        ({"views": [_view(selector="person_id")]}, ValueError, "structural"),
        ({"views": [_view(selector="person_household_id")]}, ValueError, "structural"),
        ({"views": [_view(selector="child_flag")]}, ValueError, "view output"),
        ({"output_dtypes": "native"}, ValueError, "output_dtypes='graph'"),
    ],
)
def test_adapter_refuses_an_unexecutable_view(monkeypatch, updates, error, match):
    with pytest.raises(error, match=match):
        _adapter(monkeypatch, **updates)


def test_adapter_refuses_views_with_relations(monkeypatch):
    relation = AxiomRelation(
        predicate="toy_member",
        key="toy_member:0:1",
        current_slot=0,
        related_slot=1,
        arguments=("Household", "Person"),
        root_entity="household",
        related_entity="person",
        membership_column="person_household_id",
        related_inputs={"toy_income": "income"},
        cardinality="many",
        period="2025",
    )
    with pytest.raises(ValueError, match="views or relations"):
        _adapter(monkeypatch, relations=(relation,))


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def test_a_view_runs_the_logical_entity_on_selected_rows_only(monkeypatch):
    adapter, engine = _adapter(monkeypatch)
    frame = _frame([True, False, True, True, False])
    result = adapter.materialize(frame, VARIABLES, "2025")
    program = engine.programs["Child"]
    (call,) = program.calls
    assert set(call) == {"period_kind", "start", "end", "inputs", "outputs"}
    assert call["inputs"]["age"].tolist() == [0, 2, 3]
    assert call["outputs"] == VARIABLES
    assert result["child_amount"].tolist() == [0.0, -7.5, 26.0, 39.0, -7.5]
    assert result["child_amount"].dtype == np.float64
    assert result["child_test"].tolist() == [1, 0, 1, 1, 0]
    assert result["child_test"].dtype == np.int64
    assert result["child_flag"].tolist() == [True, False, True, False, False]
    assert engine.compiled.count("Child") == 1


def test_a_view_output_resolves_to_the_person_entity(monkeypatch):
    adapter, _ = _adapter(monkeypatch)
    metadata = adapter.variable_metadata("child_amount")
    assert (metadata.entity, metadata.dtype, metadata.period) == (
        "person",
        "float",
        "year",
    )
    assert adapter.graph_dtype("child_test") == "int64"
    assert adapter.variables() == ["age", "income"]
    assert adapter.view_root_inputs() == {"Child": ("age", "income")}


def test_a_child_output_without_a_view_keeps_todays_message(monkeypatch):
    adapter = AxiomEngine(
        MODULE,
        SCHEMA,
        rulespec_roots=(ROOT,),
        entity_names=NAMES,
        output_dtypes="graph",
    )
    engine = _FakeEngine(
        {
            "Person": _Program(
                "Person",
                ("income",),
                [_metadata("adult_amount", "Person", "decimal"), *CHILD_METADATA],
            )
        }
    )
    monkeypatch.setattr(adapter, "_import_engine", lambda: engine)
    with pytest.raises(ValueError, match="which no frame entity is mapped to"):
        adapter.variable_metadata("child_amount")
    with pytest.raises(ValueError, match="which no frame entity is mapped to"):
        adapter.materialize(_frame([True]), ["child_amount"], "2025")


def test_a_child_only_module_materializes(monkeypatch):
    adapter, engine = _adapter(monkeypatch)
    assert set(engine.compiled) <= {"Child"}
    result = adapter.materialize(_frame([False, True]), ["child_amount"], "2025")
    assert result["child_amount"].tolist() == [-7.5, 13.0]
    assert "Person" in engine.compiled and "Household" in engine.compiled


@pytest.mark.parametrize(
    "selector,match",
    [
        (pd.array([True, None], dtype="boolean"), "null"),
        (np.asarray([1, 0]), "bool column"),
        (np.asarray(["y", "n"], dtype=object), "bool column"),
        (None, "not a column"),
    ],
)
def test_a_null_absent_or_non_bool_selector_refuses(monkeypatch, selector, match):
    adapter, engine = _adapter(monkeypatch)
    frame = _frame([True, False])
    if selector is None:
        frame.table("person").drop(columns="is_child", inplace=True)
    else:
        frame.table("person")["is_child"] = selector
    with pytest.raises(ValueError, match=match):
        adapter.materialize(frame, VARIABLES, "2025")
    assert engine.programs["Child"].calls == []


def test_an_empty_selection_makes_no_engine_call(monkeypatch):
    adapter, engine = _adapter(monkeypatch)
    result = adapter.materialize(_frame([False, False, False]), VARIABLES, "2025")
    assert engine.programs["Child"].calls == []
    assert result["child_amount"].tolist() == [-7.5] * 3
    assert result["child_test"].tolist() == [0] * 3
    empty = _materialize(adapter, [], VARIABLES)
    assert {name: values.shape for name, values in empty.items()} == {
        name: (0,) for name in VARIABLES
    }


def test_the_batch_length_must_equal_the_selected_count(monkeypatch):
    """A length-1 engine array would broadcast through ``out[mask] = values``."""

    def one_value(age, income):
        return {name: values[:1] for name, values in f(age, income).items()}

    adapter, engine = _adapter(
        monkeypatch, {"Child": _child_program(function=one_value)}
    )
    with pytest.raises(ValueError, match="selects 3 row"):
        adapter.materialize(_frame([True, True, True]), ["child_amount"], "2025")
    (call,) = engine.programs["Child"].calls
    assert len(call["inputs"]["age"]) == 3


@pytest.mark.parametrize(
    "outside,match",
    [
        ({"child_amount": 1.0}, "no outside value"),
        ({**OUTSIDE, "child_test": 2}, "codes outside"),
        ({**OUTSIDE, "child_test": 0.5}, "losslessly"),
        ({**OUTSIDE, "child_test": True}, "losslessly"),
        ({**OUTSIDE, "child_flag": 0}, "losslessly"),
        ({**OUTSIDE, "child_amount": True}, "losslessly"),
        ({**OUTSIDE, "child_amount": 2**60 + 1}, "losslessly"),
    ],
)
def test_a_missing_or_uncastable_outside_value_refuses(monkeypatch, outside, match):
    adapter, engine = _adapter(monkeypatch, views=(_view(outside=outside),))
    with pytest.raises(ValueError, match=match):
        adapter.materialize(_frame([True, False]), VARIABLES, "2025")
    assert engine.programs["Child"].calls == []


def test_an_integer_outside_value_stands_for_an_exact_decimal(monkeypatch):
    adapter, _ = _adapter(
        monkeypatch, views=(_view(outside={**OUTSIDE, "child_amount": 4}),)
    )
    result = adapter.materialize(_frame([False]), ["child_amount"], "2025")
    assert result["child_amount"].dtype == np.float64
    assert result["child_amount"].tolist() == [4.0]


def test_a_relation_on_the_view_program_refuses(monkeypatch):
    relation = SimpleNamespace(name="toy_member", key="toy_member:0:1")
    adapter, _ = _adapter(monkeypatch, {"Child": _child_program(relations=[relation])})
    with pytest.raises(NotImplementedError, match="view entity 'Child'"):
        adapter.materialize(_frame([True]), ["child_amount"], "2025")
    adapter, _ = _adapter(monkeypatch, {"Child": _child_program(relations=[relation])})
    with pytest.raises(NotImplementedError, match="gathers no related rows"):
        assert_no_relations(adapter, "Child")


def test_a_view_compile_refuses_after_the_pinned_tree_moves(monkeypatch, tmp_path):
    root = tmp_path / "rulespec-zz"
    module = root / "zz/policies/tests/axiom_toy_relations.yaml"
    module.parent.mkdir(parents=True)
    module.write_bytes(MODULE.read_bytes())
    adapter, engine = _adapter(monkeypatch, module=module, rulespec_roots=(root,))
    axiom_engine_ref(adapter, **{**PINS, "rulespec_root": root})
    (root / "zz/added.yaml").write_text("{}\n")
    with pytest.raises(ValueError, match="changed after this adapter's engine_ref"):
        adapter.view_root_inputs()
    assert "Child" not in engine.compiled


def test_a_reference_refuses_an_adapter_whose_view_already_compiled(monkeypatch):
    adapter, _ = _adapter(monkeypatch)
    adapter.view_root_inputs()
    with pytest.raises(ValueError, match="before the adapter compiles"):
        axiom_engine_ref(adapter, **PINS)


def test_view_and_frame_programs_cache_under_separate_keys(monkeypatch):
    """A module with a Person program and a Child view runs each on its own rows."""
    person = _Program(
        "Person",
        ("income",),
        [_metadata("adult_amount", "Person", "decimal")],
        function=lambda income: {"adult_amount": income + 0.5},
    )
    adapter, engine = _adapter(
        monkeypatch, {"Person": person, "Child": _child_program()}
    )
    frame = _frame([True, False, True])
    result = adapter.materialize(frame, ["adult_amount", "child_amount"], "2025")
    assert result["adult_amount"].tolist() == [0.5, 3.5, 6.5]
    assert result["child_amount"].tolist() == [0.0, -7.5, 26.0]
    assert len(person.calls[-1]["inputs"]["income"]) == 3
    assert len(engine.programs["Child"].calls[-1]["inputs"]["income"]) == 2
    assert engine.compiled.count("Person") == 1
    assert engine.compiled.count("Child") == 1
    assert adapter._programs["person"] is person
    assert adapter._view_programs["Child"] is engine.programs["Child"]
    assert adapter.variable_metadata("adult_amount").entity == "person"


def test_the_selector_is_a_whole_table_person_dependency(monkeypatch):
    adapter, _ = _adapter(monkeypatch)
    assert adapter.relation_dependencies("person") == (("person", "is_child"),)
    assert adapter.relation_dependencies("household") == ()


def test_the_reference_carries_views_in_declaration_order(monkeypatch):
    second = AxiomView("Vehicle", "has_vehicle", {"vehicle_tax": 0.0})
    adapter, _ = _adapter(monkeypatch, views=(_view(), second))
    document = json.loads(axiom_engine_ref(adapter, **PINS))
    assert document["views"] == [_view().to_dict(), second.to_dict()]
    absent = AxiomEngine(
        MODULE,
        SCHEMA,
        rulespec_roots=(ROOT,),
        entity_names=NAMES,
        output_dtypes="graph",
    )
    assert "views" not in json.loads(axiom_engine_ref(absent, **PINS))


# ---------------------------------------------------------------------------
# Properties P1 to P6
# ---------------------------------------------------------------------------


@st.composite
def _population(draw, *, min_rows=0, max_rows=60):
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    selected = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    ages = draw(st.lists(st.integers(0, 90), min_size=n, max_size=n))
    incomes = draw(st.lists(st.integers(-1000, 1000), min_size=n, max_size=n))
    outside = draw(st.integers(-1000, 1000).filter(lambda value: value != 0).map(float))
    return (
        np.asarray(selected, dtype=bool),
        np.asarray(ages, dtype=np.int64),
        np.asarray(incomes, dtype=np.float64),
        outside,
    )


def _run(population, *, function=f):
    selected, ages, incomes, outside = population
    with pytest.MonkeyPatch.context() as patch:
        adapter, engine = _adapter(
            patch,
            {"Child": _child_program(function=function)},
            views=(_view(outside={**OUTSIDE, "child_amount": outside}),),
        )
        result = _materialize(adapter, selected, ["child_amount"], ages, incomes)
    return result["child_amount"], engine.programs["Child"]


@settings(max_examples=60, deadline=None, database=None)
@given(_population())
def test_p1_one_output_per_row(population):
    """A sentinel per selected row shows each index written exactly once."""
    selected, ages, incomes, outside = population
    out, program = _run(population)
    assert len(out) == len(selected)
    sentinel = np.full(len(selected), np.nan)
    sentinel[selected] = f(ages[selected], incomes[selected])["child_amount"]
    sentinel[~selected] = outside
    assert np.array_equal(out, sentinel)
    assert len(program.calls) == int(selected.any())


@settings(max_examples=60, deadline=None, database=None)
@given(_population())
def test_p2_outside_rows_hold_the_outside_value(population):
    selected, _, _, outside = population
    out, _ = _run(population)
    assert out.dtype == np.float64
    assert (out[~selected] == np.float64(outside)).all()


@settings(max_examples=60, deadline=None, database=None)
@given(_population())
def test_p3_conservation_is_exact_for_decimals(population):
    selected, ages, incomes, outside = population
    out, _ = _run(population)
    view_total = f(ages[selected], incomes[selected])["child_amount"].sum()
    assert out.sum() == view_total + outside * int((~selected).sum())


_RELATION = AxiomRelation(
    predicate="toy_member",
    key="toy_member:0:1",
    current_slot=0,
    related_slot=1,
    arguments=("Household", "Person"),
    root_entity="household",
    related_entity="person",
    membership_column="person_household_id",
    related_inputs={"toy_income": "income"},
    cardinality="many",
    period="2025",
)


@settings(max_examples=40, deadline=None, database=None)
@given(
    relations=st.sampled_from((None, (), (_RELATION,))),
    arithmetic=st.sampled_from(("decimal", "f64")),
    output_dtypes=st.sampled_from(("native", "graph")),
    population=_population(min_rows=1, max_rows=12),
)
def test_p4_no_views_no_change(relations, arithmetic, output_dtypes, population):
    """``views=None`` leaves the call, dependencies and variables as before."""
    selected, ages, incomes, _ = population
    calls = []
    for spelled in (False, True):
        program = _Program(
            "Household",
            ("scale",),
            [_metadata("toy_total", "Household", "decimal")],
            relations=[]
            if not relations
            else [
                SimpleNamespace(
                    name="toy_member",
                    key="toy_member:0:1",
                    current_slot=0,
                    related_slot=1,
                    related_inputs=("toy_income",),
                )
            ],
            function=lambda scale: {"toy_total": scale * 1.0},
        )
        options = dict(
            relations=relations, arithmetic=arithmetic, output_dtypes=output_dtypes
        )
        if spelled:
            options["views"] = None
        adapter = AxiomEngine(
            MODULE, SCHEMA, rulespec_roots=(ROOT,), entity_names=NAMES, **options
        )
        engine = _FakeEngine({"Household": program})
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(adapter, "_import_engine", lambda engine=engine: engine)
            reference = json.loads(axiom_engine_ref(adapter, **PINS))
            patch.setattr(
                "microcosm.frame.adapters.axiom._source_relation_arguments",
                lambda *args: ("Household", "Person"),
            )
            patch.setattr(
                "microcosm.frame.adapters.axiom.AxiomEngine._relation_batches",
                lambda self, bundle, entity, period: {"toy_member:0:1": "batch"},
            )
            frame = _frame(selected, ages, incomes)
            frame.table("household")["scale"] = np.ones(1)
            adapter.materialize(frame, ["toy_total"], "2025")
            calls.append(
                (
                    sorted(program.calls[-1]),
                    {k: v.tolist() for k, v in program.calls[-1]["inputs"].items()},
                    adapter.relation_dependencies("person"),
                    adapter.relation_dependencies("household"),
                    adapter.variables(),
                    reference,
                )
            )
        keys = calls[-1][0]
        assert "views" not in calls[-1][5]
        assert ("relations" in keys) == bool(relations)
    assert calls[0] == calls[1]


@settings(max_examples=60, deadline=None, database=None)
@given(_population(), st.randoms(use_true_random=False))
def test_p5_shuffling_rows_shuffles_outputs(population, generator):
    selected, ages, incomes, outside = population
    order = list(range(len(selected)))
    generator.shuffle(order)
    out, _ = _run(population)
    shuffled, _ = _run((selected[order], ages[order], incomes[order], outside))
    assert np.array_equal(shuffled, out[order])


@settings(max_examples=60, deadline=None, database=None)
@given(_population())
def test_p6_each_selected_output_is_f_of_that_row_alone(population):
    selected, ages, incomes, _ = population
    out, _ = _run(population)
    for index in np.flatnonzero(selected):
        alone = f(ages[index : index + 1], incomes[index : index + 1])["child_amount"]
        assert out[index] == alone[0]


def _batch_dependent(age, income):
    """Varies with the batch: each output adds the batch's position."""
    values = f(age, income)
    values["child_amount"] = values["child_amount"] + np.arange(len(age))
    return values


def test_p6_detects_an_engine_whose_output_depends_on_other_rows():
    population = (
        np.asarray([True, True]),
        np.asarray([3, 4]),
        np.asarray([1.0, 2.0]),
        5.0,
    )
    out, _ = _run(population, function=_batch_dependent)
    alone = f(np.asarray([4]), np.asarray([2.0]))["child_amount"][0]
    assert out[1] != alone


# ---------------------------------------------------------------------------
# Differential (a): the mapped-entity path against the view path, engine-free
# ---------------------------------------------------------------------------

CHILD_SCHEMA = EntitySchema(person_entity="child", group_entities=("household",))


@settings(max_examples=40, deadline=None, database=None)
@given(_population(min_rows=1, max_rows=30).filter(lambda item: item[0].any()))
def test_differential_a_mapped_child_entity_against_the_view(population):
    """An adapter whose person entity maps to Child, and the view, agree per child."""
    selected, ages, incomes, outside = population
    with pytest.MonkeyPatch.context() as patch:
        view, _ = _adapter(
            patch,
            views=(_view(outside={**OUTSIDE, "child_amount": outside}),),
        )
        through_view = view.materialize(
            _frame(selected, ages, incomes), VARIABLES, "2025"
        )
        mapped = AxiomEngine(
            MODULE,
            CHILD_SCHEMA,
            rulespec_roots=(ROOT,),
            entity_names={"child": "Child", "household": "Household"},
            output_dtypes="graph",
        )
        engine = _FakeEngine({"Child": _child_program()})
        patch.setattr(mapped, "_import_engine", lambda: engine)
        count = int(selected.sum())
        children = Frame(
            tables={
                "child": pd.DataFrame(
                    {
                        "child_id": np.arange(1, count + 1),
                        "child_household_id": np.ones(count, dtype=np.int64),
                        "age": ages[selected],
                        "income": incomes[selected],
                    }
                ),
                "household": pd.DataFrame({"household_id": [1]}),
            },
            schema=CHILD_SCHEMA,
            weights={"household": Weights(np.ones(1), WeightKind.DESIGN)},
        )
        direct = mapped.materialize(children, VARIABLES, "2025")
    for name in VARIABLES:
        assert through_view[name][selected].tolist() == direct[name].tolist()
        assert through_view[name].dtype == direct[name].dtype


# ---------------------------------------------------------------------------
# Placement: view code stays in axiom.py (source_hash does not follow imports)
# ---------------------------------------------------------------------------


def test_axiom_first_party_imports_are_pinned():
    """A view helper moved to a new module would sit in no kernel's source hash."""
    import ast

    source = (
        paths_for("microcosm-frame").package / "src/microcosm/frame/adapters/axiom.py"
    ).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    first_party = {
        name
        for name in imported
        if name.startswith("microcosm") or name.startswith(".")
    }
    assert first_party == {
        "microcosm.frame.bundle",
        "microcosm.frame.concept_mapping",
        "microcosm.frame.materialize",
        "microcosm.frame.rules",
        "microcosm.frame.schema",
        "microcosm.graph.canonical",
    }


# ---------------------------------------------------------------------------
# Differential (b): the real engine, skip-gated
# ---------------------------------------------------------------------------

_ENGINE = importlib.util.find_spec("axiom_rules_engine")
if _ENGINE is not None:
    from axiom_rules_engine.dense import NativeCompiledDenseProgram
else:
    NativeCompiledDenseProgram = None

#: rulespec-be at 530849e: ``be.json`` module 61 (0-based), 107 Child roots.
_BE_ROOT = os.environ.get("POPULACE_RULESPEC_BE")
_BE_MODULE = "be/statutes/family_benefits/regional_routing.yaml"
_BE_MODULE_SHA256 = "408e73587e60d91136f80a71612802cb8c111f7ea5baf5abaeecb772c43c2413"
_GRAPH_KINDS = ("bool", "integer", "decimal", "judgment")


def _real_inputs(names, n, generator):
    """Typed random inputs by name: RuleSpec inputs carry no dtype (axiom#62).

    Counts, ages and years become integers, money becomes decimals, the
    region becomes one of the three region codes, and every other input a
    bool. The run on the pinned engine checks the guess: the engine refuses
    an input of the wrong type by name.
    """
    columns = {}
    for name in names:
        if name.endswith(("_years", "_count", "_months", "_rank")):
            columns[name] = generator.integers(0, 26, n).astype(np.int64)
        elif name.endswith(("_income", "_amount", "_eur")):
            columns[name] = generator.integers(0, 60_000, n).astype(np.float64)
        elif name.endswith("_region"):
            columns[name] = generator.integers(1, 4, n).astype(np.int64)
        else:
            columns[name] = generator.integers(0, 2, n).astype(bool)
    return columns


@pytest.mark.skipif(
    NativeCompiledDenseProgram is None or not _BE_ROOT,
    reason=(
        "Axiom dense native extension is not installed, or POPULACE_RULESPEC_BE "
        "does not name a rulespec-be 530849e export"
    ),
)
def test_differential_b_real_child_program_against_the_view():
    """The real Child program agrees with the view, and is row-local per batch.

    Shuffled, single-row and random-subset batches must give each row the
    value the full batch gives it: the only test that can falsify
    row-locality inside the engine.
    """
    import hashlib

    from axiom_rules_engine import CompiledDenseProgram

    root = Path(_BE_ROOT).resolve()
    module = root / _BE_MODULE
    assert hashlib.sha256(module.read_bytes()).hexdigest() == _BE_MODULE_SHA256
    program = CompiledDenseProgram.from_file(
        module, rulespec_roots=(root,), entity="Child"
    )
    assert not list(program.relations)
    dtypes = {
        item.name: item.dtype
        for item in program.derived_metadata
        if item.entity == "Child" and item.dtype in _GRAPH_KINDS
    }
    outputs = sorted(dtypes)
    assert outputs
    generator = np.random.default_rng(530849)
    n = 40
    batch = _real_inputs(program.root_inputs, n, generator)
    bounds = dict(period_kind="calendar_year", start="2026-01-01", end="2026-12-31")
    full = program.execute(**bounds, inputs=batch, outputs=outputs)["outputs"]
    for order in (
        generator.permutation(n),
        np.asarray([0]),
        np.flatnonzero(generator.integers(0, 2, n)),
    ):
        part = program.execute(
            **bounds,
            inputs={name: values[order] for name, values in batch.items()},
            outputs=outputs,
        )["outputs"]
        for name in outputs:
            assert (
                np.asarray(part[name]).tolist()
                == np.asarray(full[name])[order].tolist()
            )
    # The view over persons: the children are the batch's rows, in order,
    # among adults the selector leaves out.
    selected = np.zeros(2 * n, dtype=bool)
    selected[generator.choice(2 * n, n, replace=False)] = True
    people = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n + 1),
            "person_household_id": np.ones(2 * n, dtype=np.int64),
            "is_child": selected,
        }
    )
    for name, values in batch.items():
        column = np.zeros(2 * n, dtype=values.dtype)
        column[selected] = values
        people[name] = column
    outside = {
        name: {"bool": False, "integer": 0, "judgment": 0}.get(dtypes[name], 0.0)
        for name in outputs
    }
    adapter = AxiomEngine(
        module,
        SCHEMA,
        rulespec_roots=(root,),
        entity_names=NAMES,
        output_dtypes="graph",
        periods={"2026": AxiomPeriod("2026-01-01", "2026-12-31", "calendar_year")},
        views=(AxiomView("Child", "is_child", outside),),
    )
    frame = Frame(
        tables={"person": people, "household": pd.DataFrame({"household_id": [1]})},
        schema=SCHEMA,
        weights={"household": Weights(np.ones(1), WeightKind.DESIGN)},
    )
    through_view = adapter.materialize(frame, outputs, "2026")
    for name in outputs:
        expected = np.asarray(full[name])
        assert through_view[name][selected].tolist() == expected.tolist()
        assert (through_view[name][~selected] == outside[name]).all()
        if dtypes[name] == "decimal":
            # P3 on the real engine.
            assert through_view[name].sum() == expected.astype(
                np.float64
            ).sum() + outside[name] * int((~selected).sum())
