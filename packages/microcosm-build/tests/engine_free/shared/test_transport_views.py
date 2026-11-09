"""Binding-local views through the transport registry, compose and closure.

A fake ``axiom_rules_engine`` compiles fake dense programs per engine
entity; it evaluates no RuleSpec formula. The registry builds real
:class:`AxiomEngine` adapters over it, so preparation, references and the
closure partition run their own code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.transport.compose as compose_module
from microcosm.build.transport.evidence_contract import (
    binding_view_entities,
    closure_partition,
)
from microcosm.build.transport.evidence_kernels import engine_root_inputs
from microcosm.build.transport.registry import (
    binding_view_dependencies,
    build_transport_registry,
)
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import AxiomEngine, AxiomPeriod, AxiomView
from microcosm.frame.concept_mapping import ConceptMapping
from microcosm.frame.concepts import CONCEPT_ENTITIES, CONCEPTS
from microcosm.frame.input_closure import ClosureEntry, InputClosure, UndeterminedPolicy
from microcosm.graph import Node, Owned, Slice
from test_support.microcosm_build.transport_composed import make_composed_fixture
from test_support.microcosm_build.transport_population import UNIT_RULE

PERIOD = "2026-27"
CHILD_OUTPUTS = ("child_amount", "child_test")


def _metadata(name, entity, dtype):
    return SimpleNamespace(name=name, entity=entity, dtype=dtype, period="Year")


class _Program:
    def __init__(self, entity, root_inputs, metadata, function, relations=()):
        self.root_entity = entity
        self.root_inputs = list(root_inputs)
        self.relations = list(relations)
        self.derived_metadata = list(metadata)
        self.function = function
        self.calls = []

    def execute(self, **arguments):
        self.calls.append(arguments)
        values = self.function(
            *(arguments["inputs"][name] for name in self.root_inputs)
        )
        return {"outputs": {name: values[name] for name in arguments["outputs"]}}

    execute_f64 = execute


class _FakeEngine:
    def __init__(self, programs_by_module):
        self.programs_by_module = programs_by_module
        self.CompiledDenseProgram = self

    def from_file(self, module, *, rulespec_roots, entity):
        programs = self.programs_by_module[Path(module).name]
        if entity not in programs:
            raise ValueError(
                f"dense compilation could not find derived outputs for entity `{entity}`"
            )
        return programs[entity]


def _child_program(relations=()):
    return _Program(
        "Child",
        ("age", "child_income"),
        [
            _metadata("child_amount", "Child", "decimal"),
            _metadata("child_test", "Child", "judgment"),
        ],
        lambda age, income: {
            "child_amount": age * 10.0 + income,
            "child_test": np.where(age < 12, 1, -1).astype(np.int8),
        },
        relations,
    )


def _person_program(root_inputs=("income",), extra=()):
    return _Program(
        "Person",
        root_inputs,
        [_metadata("adult_amount", "Person", "decimal"), *extra],
        lambda *columns: {"adult_amount": columns[0] + 0.5},
    )


def _view(**updates):
    view = {
        "logical_entity": "Child",
        "selector": {"column": "is_child"},
        "outside": {"child_amount": 0.0, "child_test": 0},
    }
    view.update(updates)
    return view


def _fixture(tmp_path, monkeypatch, *, programs=None, views=None, variables=None):
    root = tmp_path / "rulespec"
    root.mkdir(exist_ok=True)
    payload = b"rules: []\n"
    (root / "child.yaml").write_bytes(payload)
    row = {
        "id": "child_benefit",
        "rulespec_path": "child.yaml",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "entity": "person",
        "engine_entity": "Person",
        "period": PERIOD,
        "variables": list(variables or CHILD_OUTPUTS),
        "views": [_view()] if views is None else views,
    }
    document = {
        "engine": {"commit": "a" * 40, "wheel_sha256": "b" * 64},
        "rulespec": {"commit": "c" * 40},
        "entity_names": {
            "person": "Person",
            "household": "Household",
            "family": "Family",
        },
        "periods": {
            PERIOD: {"kind": "tax_year", "start": "2026-04-01", "end": "2027-03-31"}
        },
        "bindings": [row],
    }
    engine = _FakeEngine(
        {"child.yaml": {"Child": _child_program()} if programs is None else programs}
    )
    monkeypatch.setattr(AxiomEngine, "_import_engine", lambda self: engine)
    return root, document, engine


def _prepared(tmp_path, monkeypatch, **options):
    root, document, engine = _fixture(tmp_path, monkeypatch, **options)
    registry = build_transport_registry(document, root, unit_rule=UNIT_RULE)
    adapter = registry.engines[registry.engine_refs["child_benefit"]]
    return registry, adapter, document, engine


def _frame(selected, ages):
    n = len(selected)
    people = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1),
            "person_household_id": np.ones(n, dtype=np.int64),
            "person_family_id": np.ones(n, dtype=np.int64),
            "is_child": np.asarray(selected, dtype=bool),
            "age": np.asarray(ages, dtype=np.int64),
            "child_income": np.zeros(n),
            "income": np.full(n, 2.0),
        }
    )
    return Frame(
        tables={
            "person": people,
            "household": pd.DataFrame({"household_id": [1]}),
            "family": pd.DataFrame({"family_id": [1]}),
        },
        schema=UNIT_RULE_SCHEMA,
        weights={"household": Weights(np.ones(1), WeightKind.DESIGN)},
    )


UNIT_RULE_SCHEMA = None


@pytest.fixture(autouse=True)
def _schema():
    global UNIT_RULE_SCHEMA
    from microcosm.build.transport.registry import transport_entity_schema

    UNIT_RULE_SCHEMA = transport_entity_schema(UNIT_RULE)[0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_a_child_only_module_prepares_through_the_registry(tmp_path, monkeypatch):
    registry, adapter, document, _ = _prepared(tmp_path, monkeypatch)
    reference = json.loads(registry.engine_refs["child_benefit"])
    assert reference["views"] == [_view()]
    assert adapter._views == (
        AxiomView("Child", "is_child", {"child_amount": 0.0, "child_test": 0}),
    )
    assert adapter.variable_metadata("child_amount").entity == "person"
    result = adapter.materialize(
        _frame([True, False, True], [3, 40, 15]), list(CHILD_OUTPUTS), PERIOD
    )
    assert result["child_amount"].tolist() == [30.0, 0.0, 150.0]
    assert result["child_test"].tolist() == [1, 0, -1]


def test_view_dependencies_agree_with_the_adapter(tmp_path, monkeypatch):
    registry, adapter, document, _ = _prepared(tmp_path, monkeypatch)
    binding = document["bindings"][0]
    expected = (("person", "is_child"),)
    assert (
        binding_view_dependencies(
            binding, entity_names=document["entity_names"], schema=registry.schema
        )
        == expected
    )
    assert adapter.relation_dependencies("person") == expected
    assert {
        dependency
        for entity in registry.schema.entities
        for dependency in adapter.relation_dependencies(entity)
    } == set(expected)
    binding.pop("views")
    assert (
        binding_view_dependencies(
            binding, entity_names=document["entity_names"], schema=registry.schema
        )
        == ()
    )


def test_absent_views_stay_absent_from_the_reference(tmp_path, monkeypatch):
    root, document, _ = _fixture(
        tmp_path,
        monkeypatch,
        programs={"Person": _person_program()},
        variables=["adult_amount"],
    )
    document["bindings"][0].pop("views")
    registry = build_transport_registry(document, root, unit_rule=UNIT_RULE)
    reference = json.loads(registry.engine_refs["child_benefit"])
    assert "views" not in reference
    assert registry.engines[registry.engine_refs["child_benefit"]]._views is None


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda row: row.update(views=[]), ValueError, "at least one view"),
        (lambda row: row.update(views={}), TypeError, "list of view"),
        (
            lambda row: row["views"][0]["outside"].pop("child_test"),
            ValueError,
            "outside keys",
        ),
        (
            lambda row: row["views"][0]["outside"].update(other=0.0),
            ValueError,
            "non-binding variables",
        ),
        (
            lambda row: row["views"][0].update(logical_entity="Person"),
            ValueError,
            "mapped engine entity",
        ),
        (
            lambda row: row["views"][0].update(logical_entity="Family"),
            ValueError,
            "mapped engine entity",
        ),
        (
            lambda row: row["views"].append(copy.deepcopy(row["views"][0])),
            ValueError,
            "repeats a logical entity",
        ),
        (
            lambda row: row["views"][0]["selector"].update(equals=True),
            ValueError,
            "exactly one bool person column",
        ),
        (
            lambda row: row["views"][0].update(selector={"expression": "age < 18"}),
            ValueError,
            "exactly one bool person column",
        ),
        (
            lambda row: row["views"][0].update(selector={"rows": [1, 2]}),
            ValueError,
            "exactly one bool person column",
        ),
        (
            lambda row: row["views"][0].update(selector={"column": "person_id"}),
            ValueError,
            "structural",
        ),
        (
            lambda row: row["views"][0].update(selector={"column": "person_family_id"}),
            ValueError,
            "structural",
        ),
        (
            lambda row: row["views"][0].update(selector={"column": "child_amount"}),
            ValueError,
            "output of the binding",
        ),
        (
            lambda row: row["views"][0].update(id="child"),
            ValueError,
            "exactly logical_entity, selector and outside",
        ),
        (
            lambda row: row.update(entity="family", engine_entity="Family"),
            ValueError,
            "person entity",
        ),
        (
            lambda row: row.update(relations=[]),
            ValueError,
            "relations cannot share",
        ),
    ],
)
def test_the_registry_refuses_an_unsupported_view(
    tmp_path, monkeypatch, change, error, match
):
    root, document, _ = _fixture(tmp_path, monkeypatch)
    change(document["bindings"][0])
    with pytest.raises(error, match=match):
        build_transport_registry(document, root, unit_rule=UNIT_RULE)


def test_the_registry_refuses_a_relation_on_the_view_program(tmp_path, monkeypatch):
    relation = SimpleNamespace(name="toy_member", key="toy_member:0:1")
    root, document, _ = _fixture(
        tmp_path, monkeypatch, programs={"Child": _child_program([relation])}
    )
    with pytest.raises(NotImplementedError, match="view entity 'Child'"):
        build_transport_registry(document, root, unit_rule=UNIT_RULE)


def test_the_registry_refuses_a_view_input_named_like_a_person_root(
    tmp_path, monkeypatch
):
    root, document, _ = _fixture(
        tmp_path,
        monkeypatch,
        programs={"Person": _person_program(("age",)), "Child": _child_program()},
        variables=["adult_amount", *CHILD_OUTPUTS],
    )
    with pytest.raises(ValueError, match=r"inputs \['age'\] share their names"):
        build_transport_registry(document, root, unit_rule=UNIT_RULE)


def test_a_mixed_module_prepares_with_distinct_person_and_view_inputs(
    tmp_path, monkeypatch
):
    registry, adapter, _, _ = _prepared(
        tmp_path,
        monkeypatch,
        programs={"Person": _person_program(), "Child": _child_program()},
        variables=["adult_amount", *CHILD_OUTPUTS],
    )
    result = adapter.materialize(
        _frame([True, False], [5, 50]), ["adult_amount", "child_amount"], PERIOD
    )
    assert result["adult_amount"].tolist() == [2.5, 2.5]
    assert result["child_amount"].tolist() == [50.0, 0.0]


def test_a_child_output_without_a_view_is_refused(tmp_path, monkeypatch):
    root, document, _ = _fixture(
        tmp_path,
        monkeypatch,
        programs={
            "Person": _person_program(extra=_child_program().derived_metadata),
            "Child": _child_program(),
        },
        variables=["adult_amount", *CHILD_OUTPUTS],
    )
    document["bindings"][0].pop("views")
    with pytest.raises(ValueError, match="which no frame entity is mapped to"):
        build_transport_registry(document, root, unit_rule=UNIT_RULE)


def test_an_injected_python_engine_with_views_is_refused(tmp_path, monkeypatch):
    root, document, _ = _fixture(tmp_path, monkeypatch)

    class _Python:
        rulespec_root = root

        def entity_schema(self):
            return UNIT_RULE_SCHEMA

        def cache_identity(self):
            return {}

        def assert_no_relations(self, entity):
            return None

        def variable_metadata(self, name):
            raise AssertionError("unused")

        def variables(self):
            return ()

        def materialize(self, bundle, variables, period):
            raise AssertionError("unused")

        def export_contract(self):
            raise AssertionError("unused")

        def write_dataset(self, bundle, path, period):
            raise AssertionError("unused")

    with pytest.raises(TypeError, match="views require an AxiomEngine"):
        build_transport_registry(
            document,
            root,
            unit_rule=UNIT_RULE,
            engines_by_binding={"child_benefit": _Python()},
        )
    with pytest.raises(TypeError, match="only an AxiomEngine runs"):
        engine_root_inputs(_Python(), document["bindings"][0], document["entity_names"])


@pytest.mark.parametrize(
    "selector,outside",
    [
        ("is_kid", {"child_amount": 0.0, "child_test": 0}),
        ("is_child", {"child_amount": 0, "child_test": 0}),
    ],
)
def test_an_injected_axiom_engine_must_carry_the_binding_views(
    tmp_path, monkeypatch, selector, outside
):
    root, document, _ = _fixture(tmp_path, monkeypatch)
    schema = UNIT_RULE_SCHEMA
    periods = {PERIOD: AxiomPeriod("2026-04-01", "2027-03-31", "tax_year")}

    def adapter(views):
        return AxiomEngine(
            root.resolve() / "child.yaml",
            schema,
            rulespec_roots=(root.resolve(),),
            periods=periods,
            entity_names=document["entity_names"],
            nesting={"family": "household"},
            output_dtypes="graph",
            views=views,
        )

    with pytest.raises(ValueError, match="views do not match"):
        build_transport_registry(
            document,
            root,
            unit_rule=UNIT_RULE,
            engines_by_binding={
                "child_benefit": adapter((AxiomView("Child", selector, outside),))
            },
        )
    prepared = build_transport_registry(
        document,
        root,
        unit_rule=UNIT_RULE,
        engines_by_binding={
            "child_benefit": adapter(
                (
                    AxiomView(
                        "Child", "is_child", {"child_amount": 0.0, "child_test": 0}
                    ),
                )
            )
        },
    )
    assert json.loads(prepared.engine_refs["child_benefit"])["views"] == [_view()]


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


def _closure_document(document, entries):
    row = document["bindings"][0]
    return {
        "entries": [
            {
                "rulespec_path": row["rulespec_path"],
                "entity": entity,
                "input": name,
                **fields,
            }
            for entity, name, fields in entries
        ]
    }


_ENTRIES = (
    ("Child", "age", {"class": "encoded", "encoded_by": "concept_binding"}),
    ("Child", "child_income", {"class": "defaulted", "value": 0.0}),
    ("Child", "child_supplement", {"class": "encoded", "encoded_by": "graph_node"}),
    ("Person", "income", {"class": "encoded", "encoded_by": "concept_binding"}),
)


def _spec(document, entries=_ENTRIES):
    return {
        "country": "zz",
        "resources": {
            "axiom_rules_bindings": document,
            "benefit_unit_rule": UNIT_RULE.to_dict(),
            "axiom_input_closure": _closure_document(document, entries),
        },
    }


def _rules_node(person_columns=("is_child", "age", "child_income"), rows="all"):
    return Node(
        "child_benefit_node",
        "simulate.rules_by_ref@1",
        population="create",
        inputs=(Slice("person", tuple(person_columns), rows=rows),),
        outputs=tuple(Owned("person", name, "float64") for name in CHILD_OUTPUTS),
        params={},
    )


def _config(registry):
    return SimpleNamespace(engine_refs=dict(registry.engine_refs))


def test_compose_binds_a_view_node_that_slices_its_selector_and_inputs(
    tmp_path, monkeypatch
):
    registry, _, document, _ = _prepared(tmp_path, monkeypatch)
    node = compose_module.transport_rules_node(
        _spec(document), _config(registry), "child_benefit", _rules_node()
    )
    assert node.params["engine_ref"] == registry.engine_refs["child_benefit"]
    assert node.params["variables"] == CHILD_OUTPUTS


@pytest.mark.parametrize(
    ("node", "match"),
    [
        (_rules_node(("age", "child_income")), "whole-table relation inputs"),
        (
            _rules_node(("is_child", "age", "child_income"), rows="is_child"),
            "whole-table relation inputs",
        ),
        (_rules_node(("is_child", "child_income")), r"closed inputs \['age'\]"),
        (_rules_node(("is_child", "age")), r"closed inputs \['child_income'\]"),
    ],
    ids=("no-selector", "masked", "encoded-missing", "defaulted-missing"),
)
def test_compose_refuses_an_incomplete_view_node(tmp_path, monkeypatch, node, match):
    registry, _, document, _ = _prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=match):
        compose_module.transport_rules_node(
            _spec(document), _config(registry), "child_benefit", node
        )


def test_compose_leaves_graph_node_inputs_to_the_node_that_writes_them(
    tmp_path, monkeypatch
):
    """``child_supplement`` is closed through a graph node, so the slice may omit it."""
    registry, _, document, _ = _prepared(tmp_path, monkeypatch)
    compose_module._require_relation_inputs(
        _spec(document), "child_benefit", _rules_node().inputs
    )


def test_compose_refuses_a_view_binding_in_a_spec_without_a_closure(
    tmp_path, monkeypatch
):
    registry, _, document, _ = _prepared(tmp_path, monkeypatch)
    spec = _spec(document)
    del spec["resources"]["axiom_input_closure"]
    with pytest.raises(ValueError, match="no 'axiom_input_closure' resource"):
        compose_module.transport_rules_node(
            spec, _config(registry), "child_benefit", _rules_node()
        )


def test_a_rules_binding_row_refuses_a_view_binding_without_a_closure(
    tmp_path, monkeypatch
):
    registry, _, document, _ = _prepared(tmp_path, monkeypatch)
    spec = _spec(document)
    del spec["resources"]["axiom_input_closure"]
    row = {
        "id": "child_benefit_node",
        "kernel": "simulate.rules_by_ref@1",
        "rules_binding": "child_benefit",
        "population": "create",
        "inputs": [
            {"entity": "person", "columns": ["is_child", "age", "child_income"]}
        ],
        "outputs": [
            {"entity": "person", "column": name, "dtype": "float64"}
            for name in CHILD_OUTPUTS
        ],
    }
    with pytest.raises(ValueError, match="no 'axiom_input_closure' resource"):
        compose_module._node(spec, row, _config(registry), {})


_RELATION = {
    "predicate": "toy_members",
    "key": "toy_members:0:1",
    "current_slot": 0,
    "related_slot": 1,
    "arguments": ["Family", "Person"],
    "root_entity": "family",
    "related_entity": "person",
    "membership_column": "person_family_id",
    "related_inputs": {"toy_income": "income"},
    "cardinality": "many",
    "period": PERIOD,
    "role_column": "person_family_role",
    "roles": ["head", "partner"],
}


@pytest.mark.parametrize(
    ("fields", "columns"),
    (
        ({}, ("is_child", "age", "child_income")),
        ({"relations": []}, ("is_child", "age", "child_income")),
        (
            {"entity": "family", "engine_entity": "Family", "relations": [_RELATION]},
            ("income", "person_family_role"),
        ),
    ),
    ids=("plain", "relations-only", "real-relation"),
)
def test_a_binding_without_views_needs_no_closure_at_compose(
    tmp_path, monkeypatch, fields, columns
):
    _, _, document, _ = _prepared(tmp_path, monkeypatch)
    document["bindings"][0].pop("views")
    document["bindings"][0].update(fields)
    spec = _spec(document)
    del spec["resources"]["axiom_input_closure"]
    compose_module._require_relation_inputs(
        spec, "child_benefit", _rules_node(columns).inputs
    )


def _toy_spec_without_a_closure(tmp_path):
    """The composed toy spec, with its closure renamed out of ``_has_resource``'s reach.

    A node selects the closure, so renaming keeps activation satisfied where
    deleting the resource wouldn't.
    """

    fixture = make_composed_fixture(tmp_path / "fixture")
    text = json.dumps(fixture.spec).replace("axiom_input_closure", "other_closure")
    return json.loads(text), fixture.config


@pytest.mark.parametrize(
    "binding",
    ("family_housing", "main_benefit"),
    ids=("no-node-names-it", "a-rules-node-names-it"),
)
def test_composition_refuses_a_view_binding_in_a_spec_without_a_closure(
    tmp_path, binding
):
    spec, config = _toy_spec_without_a_closure(tmp_path)
    row = next(
        row
        for row in spec["resources"]["axiom_rules_bindings"]["bindings"]
        if row["id"] == binding
    )
    row["views"] = [_view()]
    with pytest.raises(
        ValueError,
        match=rf"Rules bindings \['{binding}'\] declare views, but the spec has no",
    ):
        compose_module.compose_transport_graph(spec, config)


def test_composition_needs_no_closure_when_no_binding_declares_views(tmp_path):
    spec, config = _toy_spec_without_a_closure(tmp_path)
    assert compose_module.compose_transport_graph(spec, config).nodes


def _closure_node(person_columns):
    return Node(
        "closure",
        compose_module.INPUT_CLOSURE_REF,
        population="create",
        inputs=(Slice("person", tuple(person_columns)),),
        params={"bindings": json.dumps({"bindings": None})},
    )


def test_the_closure_producer_refuses_a_person_slice_without_the_selector(
    tmp_path, monkeypatch
):
    _, _, document, _ = _prepared(tmp_path, monkeypatch)
    spec = _spec(document)

    def graph(columns):
        node = _closure_node(columns)
        node = Node(
            node.id,
            node.kernel,
            population=node.population,
            inputs=node.inputs,
            params={"bindings": json.dumps(document)},
        )
        return SimpleNamespace(nodes=(node,))

    with pytest.raises(ValueError, match=r"\['is_child'\]"):
        compose_module._check_closure_relation_slices(spec, graph(("age",)))
    compose_module._check_closure_relation_slices(spec, graph(("is_child", "age")))


# ---------------------------------------------------------------------------
# Closure
# ---------------------------------------------------------------------------


def _empty_mapping() -> ConceptMapping:
    return ConceptMapping(
        engine="none",
        engine_version="none",
        entity_correspondence={entity: entity for entity in CONCEPT_ENTITIES},
        input_declaration="usage_inferred",
        bindings=(),
        unmapped={item.id: "Synthetic: unmapped." for item in CONCEPTS},
    )


def _closure(document, entries):
    row = document["bindings"][0]
    return InputClosure(
        country="zz",
        content_basis="transport",
        mapping_engine="none",
        mapping_engine_version="none",
        surface_rulespec_commit="synthetic",
        surface_engine_repository="engine",
        surface_engine_commit="synthetic",
        modules={row["rulespec_path"]: row["sha256"]},
        engine_optional_evidence="Synthetic.",
        undetermined=UndeterminedPolicy(
            action="refuse", reason="Synthetic.", alternatives=(), new_default=False
        ),
        knobs=(),
        entries=tuple(
            ClosureEntry(
                module=row["rulespec_path"],
                entity=entity,
                input=name,
                closure_class=closure_class,
                value=value,
                reason="Synthetic.",
                citation=None if value is None else "Synthetic provision.",
            )
            for entity, name, closure_class, value in entries
        ),
    )


_DEFAULTED = (
    ("Child", "age", "defaulted", 4),
    ("Child", "child_income", "engine_optional", None),
)


def _partition(document, entries=_DEFAULTED, *, roots=None, frame_columns=None):
    rows = document["bindings"]
    closure = _closure(document, entries)
    return closure_partition(
        closure,
        _empty_mapping(),
        rows,
        roots or {row["id"]: {"Child": ("age", "child_income")} for row in rows},
        module_sha256={row["id"]: row["sha256"] for row in rows},
        module_imports={row["id"]: False for row in rows},
        mapped_entities=document["entity_names"].values(),
        frame_columns=frame_columns,
        entity_names=document["entity_names"],
        person_entity="person",
    )


def test_the_closure_reads_view_roots_on_the_logical_entity(tmp_path, monkeypatch):
    registry, adapter, document, _ = _prepared(
        tmp_path,
        monkeypatch,
        programs={"Person": _person_program(), "Child": _child_program()},
        variables=["adult_amount", *CHILD_OUTPUTS],
    )
    roots = engine_root_inputs(
        adapter, document["bindings"][0], document["entity_names"]
    )
    assert roots == {"Child": ("age", "child_income"), "Person": ("income",)}
    assert binding_view_entities(document["bindings"][0]) == {"Child"}
    assert binding_view_entities({"id": "x"}) == frozenset()


def test_a_view_binding_partitions_its_child_roots_on_child(tmp_path, monkeypatch):
    _, document, _ = _fixture(tmp_path, monkeypatch)
    (module,) = _partition(document)["modules"]
    assert module["roots"] == ["Child.age", "Child.child_income"]
    assert module["unmapped_entity"] == []
    assert [item["input"] for item in module["defaulted"]] == ["Child.age"]
    assert module["engine_optional"] == ["Child.child_income"]
    assert module["gap"] == []


def test_without_the_view_child_roots_stay_unmapped(tmp_path, monkeypatch):
    """The mutant that drops the per-binding view entity leaves Child unmapped."""
    _, document, _ = _fixture(tmp_path, monkeypatch)
    document["bindings"][0].pop("views")
    (module,) = _partition(document)["modules"]
    assert module["roots"] == []
    assert module["unmapped_entity"] == ["Child.age", "Child.child_income"]


def test_child_roots_placed_on_person_do_not_close(tmp_path, monkeypatch):
    """Child roots reported on Person miss their Child entries and become gaps."""
    _, document, _ = _fixture(tmp_path, monkeypatch)
    result = _partition(
        document,
        roots={"child_benefit": {"Person": ("age", "child_income")}},
    )
    (module,) = result["modules"]
    assert module["gap"] == ["Person.age", "Person.child_income"]
    assert module["unexpected"] == ["Child.age", "Child.child_income"]


def test_an_unclosed_child_root_is_a_gap(tmp_path, monkeypatch):
    _, document, _ = _fixture(tmp_path, monkeypatch)
    result = _partition(document, entries=_DEFAULTED[:1])
    (module,) = result["modules"]
    assert module["gap"] == ["Child.child_income"]
    assert any("child_income" in str(problem) for problem in result["check"])


def test_a_defaulted_child_root_on_the_person_table_is_not_frame_missing(
    tmp_path, monkeypatch
):
    """PR 2 writes Child defaults as person columns; the person slice carries them."""
    _, document, _ = _fixture(tmp_path, monkeypatch)
    carried = _partition(
        document, frame_columns={"person": frozenset({"is_child", "age"})}
    )
    assert carried["modules"][0]["frame_missing"] == []
    missing = _partition(document, frame_columns={"person": frozenset({"is_child"})})
    assert missing["modules"][0]["frame_missing"] == ["Child.age"]


def test_several_view_bindings_keep_their_roots_in_order(tmp_path, monkeypatch):
    _, document, _ = _fixture(tmp_path, monkeypatch)
    second = copy.deepcopy(document["bindings"][0])
    second["id"] = "child_supplement"
    document["bindings"].append(second)
    result = _partition(document)
    assert [module["binding"] for module in result["modules"]] == [
        "child_benefit",
        "child_supplement",
    ]
    assert all(
        module["roots"] == ["Child.age", "Child.child_income"]
        for module in result["modules"]
    )


def test_the_closure_no_longer_refuses_a_view_binding(tmp_path, monkeypatch):
    _, document, _ = _fixture(tmp_path, monkeypatch)
    _partition(document)


# ---------------------------------------------------------------------------
# Downstream: a view output is an ordinary person column
# ---------------------------------------------------------------------------


def test_downstream_reads_a_view_output_as_a_person_variable(tmp_path, monkeypatch):
    """Composition sites key on (binding entity, variable): both are person."""
    registry, adapter, document, _ = _prepared(tmp_path, monkeypatch)
    row = document["bindings"][0]
    assert {
        (adapter.variable_metadata(name).entity, name) for name in row["variables"]
    } == {(row["entity"], name) for name in row["variables"]}
    assert adapter.graph_dtype("child_amount") == "float64"
    assert adapter.graph_dtype("child_test") == "int64"


def test_the_support_roster_reads_a_view_default_as_a_person_column(
    tmp_path, monkeypatch
):
    """PR 2 writes Child closure defaults on the person table."""
    _, document, _ = _fixture(tmp_path, monkeypatch)
    defaults = compose_module._closure_defaults(_spec(document))
    assert defaults == {("person", "child_income"): "child.yaml Child.child_income"}
    document["bindings"][0].pop("views")
    assert compose_module._closure_defaults(_spec(document)) == {
        (None, "child_income"): "child.yaml Child.child_income"
    }


def test_shared_path_closure_does_not_depend_on_binding_order(tmp_path, monkeypatch):
    _, document, _ = _fixture(tmp_path, monkeypatch)
    second = copy.deepcopy(document["bindings"][0])
    second["id"] = "no_view"
    second.pop("views")
    document["bindings"].append(second)
    forward = _partition(document)
    document["bindings"].reverse()
    backward = _partition(document)
    assert forward["check"] == backward["check"] == []
    assert {row["binding"]: row for row in forward["modules"]} == {
        row["binding"]: row for row in backward["modules"]
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_path_refuses_different_module_hashes(tmp_path, monkeypatch, reverse):
    _, document, _ = _fixture(tmp_path, monkeypatch)
    second = copy.deepcopy(document["bindings"][0])
    second.update(id="no_view", sha256="d" * 64)
    second.pop("views")
    document["bindings"].append(second)
    if reverse:
        document["bindings"].reverse()
    with pytest.raises(
        ValueError, match=r"(?:child_benefit.*no_view|no_view.*child_benefit)"
    ):
        _partition(document)


def test_one_adapter_exposes_roots_from_two_logical_views(tmp_path, monkeypatch):
    vehicle = _Program(
        "Vehicle",
        ("vehicle_value",),
        [_metadata("vehicle_amount", "Vehicle", "decimal")],
        lambda value: {"vehicle_amount": value * 0.1},
    )
    _, adapter, document, _ = _prepared(
        tmp_path,
        monkeypatch,
        programs={"Child": _child_program(), "Vehicle": vehicle},
        views=[
            _view(),
            _view(
                logical_entity="Vehicle",
                selector={"column": "has_vehicle"},
                outside={"vehicle_amount": 0.0},
            ),
        ],
        variables=[*CHILD_OUTPUTS, "vehicle_amount"],
    )
    expected = {"Child": ("age", "child_income"), "Vehicle": ("vehicle_value",)}
    assert adapter.view_root_inputs() == expected
    assert {"age", "child_income", "vehicle_value"} <= set(adapter.variables())
    assert (
        engine_root_inputs(adapter, document["bindings"][0], document["entity_names"])
        == expected
    )


def test_outside_cannot_name_a_binding_output_on_person(tmp_path, monkeypatch):
    root, document, _ = _fixture(
        tmp_path,
        monkeypatch,
        programs={"Person": _person_program(), "Child": _child_program()},
        variables=["adult_amount", *CHILD_OUTPUTS],
    )
    document["bindings"][0]["views"][0]["outside"]["adult_amount"] = 0.0
    with pytest.raises(ValueError, match="outside keys"):
        build_transport_registry(document, root, unit_rule=UNIT_RULE)


@pytest.mark.parametrize(
    "name,value,message",
    [
        (
            "child_amount",
            True,
            "View outside value .* does not cast losslessly to its 'decimal' output",
        ),
        ("child_test", 2, "Judgment output 'child_test' holds codes outside -1/0/1"),
        (
            "child_test",
            0.5,
            "View outside value .* does not cast losslessly to its 'judgment' output",
        ),
    ],
)
def test_registry_refuses_outside_values_that_cannot_cast(
    tmp_path, monkeypatch, name, value, message
):
    root, document, _ = _fixture(tmp_path, monkeypatch)
    document["bindings"][0]["views"][0]["outside"][name] = value
    with pytest.raises(ValueError, match=message):
        build_transport_registry(document, root, unit_rule=UNIT_RULE)
