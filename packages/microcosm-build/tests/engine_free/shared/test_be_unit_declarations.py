"""Belgium's declared family benefit unit (W3-R1), still inactive.

``build/be/unit_declarations.json`` is a ``microcosm.multi_unit_rule.v1``
document for ``transport.create@2``. Belgium has no transport graph, so
nothing builds units from it yet. These tests pin what the declaration says,
that the bindings' ``entity_names`` matches its schema as the registry
requires, that a ``transport.create@2`` node can select it, that no income
default still ties the unit to another pull request, and what it does to a
few households and, as a property, to every valid frame. Every value is a
METHOD-CARD-BE proposal under d1902.
"""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

from microcosm.build.transport.compose import (
    load_transport_spec,
    validate_transport_activation,
)
from microcosm.build.transport.registry import transport_entity_schema
from microcosm.build.transport.unit_declarations import (
    CREATE_V1_REF,
    CREATE_V2_REF,
    create_unit_declaration,
)
from microcosm.frame.adapters.axiom import axiom_concept_mapping
from microcosm.frame.adapters.axiom_input_surface import load_axiom_input_surface
from microcosm.frame.unit_construction import build_declared_units
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.paths import paths_for

_BE = paths_for("microcosm-build").package / "src/microcosm/build/be"
_SURFACE = paths_for("microcosm-frame").tests / "fixtures/axiom_input_surfaces/be.json"
UNITS_PATH = _BE / "unit_declarations.json"
ENTITY, ENGINE = "family_benefit_unit", "FamilyBenefitUnit"


def _document() -> dict:
    return json.loads(UNITS_PATH.read_text(encoding="utf-8"))


def _declaration():
    return create_unit_declaration(CREATE_V2_REF, _document())


def test_the_document_is_one_benefit_unit_for_create_2() -> None:
    declaration = _declaration()
    assert [(unit.kind, unit.entity) for unit in declaration.units] == [
        ("benefit_unit", ENTITY)
    ]
    # Strict format: the file is exactly what the rule serializes to, so no
    # value rests on a parser default.
    assert declaration.to_dict() == _document()
    assert UNITS_PATH.read_text(encoding="ascii") == (
        json.dumps(_document(), indent=2) + "\n"
    )
    # create@1 takes one family rule, never this document.
    with pytest.raises(ValueError):
        create_unit_declaration(CREATE_V1_REF, _document())


def test_the_rule_states_the_method_card_proposal() -> None:
    rule = _declaration().units[0].rule
    child = rule.dependent_child
    assert child.max_age == 24
    assert child.financial_independence is None
    assert child.parent_independence_min_age is None
    assert rule.unparented_child.value == "reference_person_unit"
    assert rule.split_parents.value == "first_parent_unit"
    assert rule.refuse_dependent_reference_person is False
    # The status reads as a proposal until the card records Max's ruling.
    status = rule.provenance["status"]
    assert "d1902" in status
    assert "METHOD-CARD-BE" in status
    assert "is a proposal" in status
    assert "until that card records his ruling" in status
    # Each declared option carries its reason.
    assert {
        "max_age",
        "financial_independence",
        "parent_independence_min_age",
        "unparented_child",
        "split_parents",
        "refuse_dependent_reference_person",
    } <= set(rule.provenance)


def test_entity_names_cover_exactly_the_declared_schema() -> None:
    # build_transport_registry's own check, on the committed files.
    schema, nesting = transport_entity_schema(_declaration())
    names = json.loads((_BE / "axiom_rules_bindings.json").read_text())["entity_names"]
    assert set(names) == set(schema.entities) == {"person", "household", ENTITY}
    assert dict(nesting) == {ENTITY: "household"}
    assert len(set(names.values())) == len(names)
    assert names[ENTITY] == ENGINE
    assert "Child" not in names.values()


def test_the_engine_name_attaches_no_rule_and_no_binding() -> None:
    mapping = axiom_concept_mapping("be")
    surface = load_axiom_input_surface(_SURFACE)
    assert ENGINE not in mapping.entity_correspondence.values()
    assert ENGINE not in mapping.person_role_entities()
    assert all(binding.engine_entity != ENGINE for binding in mapping.bindings)
    assert all(ENGINE not in module.inputs for module in surface.modules.values())


def test_a_create_2_node_can_select_the_declaration() -> None:
    spec = load_transport_spec(_BE)
    with pytest.raises(ValueError, match="not activated") as before:
        validate_transport_activation(spec)
    # No create@2 node yet, so activation still asks for the create@1 resource.
    assert "benefit_unit_rule.json" in str(before.value)
    stub = copy.deepcopy(spec)
    stub["resources"]["transport_graph"] = {
        "schema_version": 1,
        "country": "be",
        "nodes": [
            {
                "id": "be.create",
                "kernel": CREATE_V2_REF,
                "structural": "create",
                "params": {
                    "unit_rule": {"resource": "unit_declarations", "encoding": "json"}
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="not activated") as after:
        validate_transport_activation(stub)
    assert "benefit_unit_rule.json" not in str(after.value)
    assert "unit declarations" not in str(after.value)


def test_no_income_default_ties_the_unit_to_another_pull_request() -> None:
    closure = json.loads((_BE / "axiom_input_closure.json").read_text())
    reasons = [entry["reason"] for entry in closure["entries"] if "reason" in entry]
    assert not [reason for reason in reasons if "PR 3" in reason]
    assert not [reason for reason in reasons if "legal unit" in reason]
    naming = [reason for reason in reasons if UNITS_PATH.name in reason]
    # Four Brussels, three Flemish and three Walloon income reasons.
    assert len(naming) == 10
    assert all("a proposed stand-in" in reason for reason in naming)


NA = pd.NA
# (person_id, household, age, partner, parent_1, parent_2, unit, role)
PEOPLE = [
    (101, 1, 44, 102, NA, NA, 101, "head"),
    (102, 1, 42, 101, NA, NA, 101, "partner"),
    (103, 1, 3, NA, 101, 102, 101, "dependent_child"),
    (104, 1, 13, NA, 101, 102, 101, "dependent_child"),
    (105, 1, 20, NA, 101, 102, 101, "dependent_child"),
    (107, 1, 24, NA, 101, 102, 101, "dependent_child"),  # the last age inside
    (106, 1, 25, NA, 101, 102, 106, "head"),  # past the age limit
    (201, 2, 70, NA, NA, NA, 201, "head"),  # three generations: two units
    (202, 2, 22, NA, 201, NA, 202, "head"),
    (203, 2, 2, NA, 202, NA, 202, "dependent_child"),
    (301, 3, 50, NA, NA, NA, 301, "head"),
    (302, 3, 10, NA, NA, NA, 301, "dependent_child"),  # no co-resident parent
    (401, 4, 30, NA, NA, NA, 401, "head"),
    (402, 4, 22, NA, NA, NA, 401, "dependent_child"),  # the disclosed approximation
    (501, 5, 35, NA, NA, NA, 501, "head"),
    (502, 5, 36, NA, NA, NA, 502, "head"),
    (503, 5, 6, NA, 501, 502, 501, "dependent_child"),  # non-partner co-parents
    (601, 6, 23, NA, NA, NA, 601, "head"),  # a young reference person alone
    (701, 7, 17, NA, NA, NA, 701, "head"),  # a minor parent anchors a unit
    (702, 7, 1, NA, 701, NA, 701, "dependent_child"),
    (801, 8, 21, 802, 803, NA, 802, "partner"),  # partnered: never a dependent
    (802, 8, 23, 801, NA, NA, 802, "head"),  # the older adult of the couple
    (803, 8, 55, NA, NA, NA, 803, "head"),  # the parent, reference person
]
REFERENCE = {1: 101, 2: 201, 3: 301, 4: 401, 5: 501, 6: 601, 7: 701, 8: 803}


def test_the_declaration_builds_the_stated_units() -> None:
    person = pd.DataFrame(
        {
            "person_id": [row[0] for row in PEOPLE],
            "person_household_id": [row[1] for row in PEOPLE],
            "age": [row[2] for row in PEOPLE],
            "partner_person_id": pd.array([row[3] for row in PEOPLE], dtype="Int64"),
            "parent_1_person_id": pd.array([row[4] for row in PEOPLE], dtype="Int64"),
            "parent_2_person_id": pd.array([row[5] for row in PEOPLE], dtype="Int64"),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": list(REFERENCE),
            "reference_person_id": list(REFERENCE.values()),
        }
    )
    # No usual_weekly_hours column: the rule declares no hours test.
    (built,) = build_declared_units(person, household, _declaration()).values()
    assert built.person_unit.name == f"person_{ENTITY}_id"
    assert list(built.units.columns) == [
        f"{ENTITY}_id",
        f"{ENTITY}_household_id",
        f"{ENTITY}_head_person_id",
    ]
    assert built.person_unit.tolist() == [row[6] for row in PEOPLE]
    assert built.person_role.tolist() == [row[7] for row in PEOPLE]


def _row_of(person: pd.DataFrame, pointer: pd.Series) -> np.ndarray:
    """Each pointer's row position in ``person``, or -1 when it is absent."""
    rows = np.full(len(pointer), -1, dtype=np.int64)
    present = pointer.notna().to_numpy()
    ids = pd.Index(person["person_id"].to_numpy())
    rows[present] = ids.get_indexer(pointer[present].to_numpy(dtype=np.int64))
    return rows


@settings(max_examples=150, deadline=None, database=None)
@given(tables=concept_frames(max_households=4, max_members=6))
def test_the_declaration_decides_dependency_the_stated_way_on_any_frame(tables) -> None:
    person, household = tables["person"], tables["household"]
    # The rule refuses nothing on membership, so every valid frame builds.
    (built,) = build_declared_units(person, household, _declaration()).values()
    unit = built.person_unit.to_numpy()
    child = built.person_role.eq("dependent_child").to_numpy()
    parent = _row_of(person, person["parent_1_person_id"])
    is_parent = person["person_id"].isin(
        set(person["parent_1_person_id"].dropna())
        | set(person["parent_2_person_id"].dropna())
    )
    reference = pd.Series(
        household["reference_person_id"].to_numpy(dtype=np.int64),
        index=household["household_id"].to_numpy(),
    ).reindex(person["person_household_id"].to_numpy())
    is_reference = person["person_id"].to_numpy() == reference.to_numpy()
    # A dependent child is exactly a person aged 24 or under with no partner
    # and no co-resident child, who has a co-resident parent or is not the
    # household's reference person. No hours test and no parent age enter.
    expected = (
        (person["age"].to_numpy() <= 24)
        & person["partner_person_id"].isna().to_numpy()
        & ~is_parent.to_numpy()
        & ((parent >= 0) | ~is_reference)
    )
    assert np.array_equal(child, expected)
    # A dependent child sits with parent 1, or with the reference person when
    # no parent lives in the household.
    with_parent = child & (parent >= 0)
    assert np.array_equal(unit[with_parent], unit[parent[with_parent]])
    without = child & (parent < 0)
    reference_row = _row_of(person, pd.Series(reference.to_numpy(), dtype="Int64"))
    assert np.array_equal(unit[without], unit[reference_row[without]])
