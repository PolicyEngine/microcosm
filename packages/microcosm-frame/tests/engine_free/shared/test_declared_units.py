"""Shared declared partitions: compatibility, identity, lineage and refusals."""

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.concepts import SOURCE_HOUSEHOLD_ID_COLUMN
from microcosm.frame.unit_construction import (
    MULTI_UNIT_RULE_FORMAT,
    BenefitUnitRule,
    DependentChildRule,
    MultiUnitRule,
    TaxUnitRule,
    UnitDeclaration,
    _build_partition,
    _Pointers,
    benefit_unit_membership,
    build_benefit_units,
    build_declared_units,
    declared_unit_attributes,
    prepare_roster_counts,
    reciprocal_partner_rows,
    units_per_household,
)
from test_support.microcosm_frame.concept_frames import concept_frames
from test_support.microcosm_frame.unit_rules import unit_rules

PROPERTY = settings(max_examples=70, deadline=None)
FRAMES = concept_frames(max_households=4, max_members=5, min_id=-(10**8))
TAX = TaxUnitRule(
    entity="tax_unit",
    pairing_eligibility_column="joint_assessment_eligible",
    provenance={
        "eligibility": "Explicit upstream eligibility result, no implicit legal test."
    },
)
BENEFIT = BenefitUnitRule(
    entity="family",
    dependent_child=DependentChildRule(max_age=17, financial_independence=None),
    unparented_child="own_unit",
    split_parents="first_parent_unit",
    provenance={"test": "fixture"},
)
DECLARATION = MultiUnitRule(
    units=(
        UnitDeclaration(kind="benefit_unit", rule=BENEFIT),
        UnitDeclaration(kind="tax_unit", rule=TAX),
    )
)


def _tables():
    person = pd.DataFrame(
        {
            "person_id": [10, 20, 30],
            "person_household_id": [1, 1, 1],
            "age": [45, 40, 10],
            "partner_person_id": pd.array([20, 10, None], dtype="Int64"),
            "parent_1_person_id": pd.array([None, None, 10], dtype="Int64"),
            "parent_2_person_id": pd.array([None, None, 20], dtype="Int64"),
            "joint_assessment_eligible": [True, True, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [1],
            "reference_person_id": pd.array([20], dtype="Int64"),
            SOURCE_HOUSEHOLD_ID_COLUMN: [901],
        }
    )
    return person, household


@PROPERTY
@given(tables=FRAMES, rule=unit_rules())
def test_benefit_kind_is_byte_identical_to_legacy(tables, rule):
    person, household = tables["person"], tables["household"]
    legacy_units, legacy_membership = build_benefit_units(person, household, rule)
    legacy = benefit_unit_membership(person, legacy_units, legacy_membership, rule)
    built = build_declared_units(
        person, household, MultiUnitRule.from_benefit_unit(rule)
    )[rule.entity]
    for left, right in (
        (legacy.units, built.units),
        (legacy.person_unit, built.person_unit),
        (legacy.person_role, built.person_role),
    ):
        assert left.to_json() == right.to_json()
        if isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        else:
            pd.testing.assert_series_equal(left, right)


@PROPERTY
@given(tables=FRAMES, data=st.data())
def test_every_declared_partition_nests_and_preserves_identity(tables, data):
    person, household = tables["person"].copy(), tables["household"].copy()
    person[TAX.pairing_eligibility_column] = data.draw(
        st.lists(st.booleans(), min_size=len(person), max_size=len(person))
    )
    household[SOURCE_HOUSEHOLD_ID_COLUMN] = household["household_id"] + 10**9
    built = build_declared_units(person, household, DECLARATION)
    shuffled = person.iloc[list(data.draw(st.permutations(range(len(person)))))].copy()
    shuffled["employment_income"] = data.draw(st.integers(0, 10**8))
    again = build_declared_units(shuffled, household, DECLARATION)
    for item in DECLARATION.units:
        group, other = built[item.entity], again[item.entity]
        rule = item.rule
        ids = group.units[rule.id_column]
        assert group.person_unit.index.equals(person.index)
        assert group.person_unit.notna().all()
        assert ids.is_unique
        assert set(group.person_unit) == set(ids)
        nested = group.units.set_index(rule.id_column)[rule.household_column]
        assert (
            nested.loc[group.person_unit.to_numpy()].tolist()
            == person["person_household_id"].tolist()
        )
        assert rule.source_household_column in group.units
        assert all("weight" not in column for column in group.units)
        pd.testing.assert_frame_equal(group.units, other.units)
        pd.testing.assert_series_equal(
            group.person_unit.sort_index(), other.person_unit.sort_index()
        )
        pd.testing.assert_series_equal(
            group.person_role.sort_index(), other.person_role.sort_index()
        )
    tax = built[TAX.entity]
    assert tax.person_role.isin(["head", "partner"]).all()
    assert tax.person_unit.value_counts().max() <= 2


@PROPERTY
@given(tables=FRAMES, data=st.data())
def test_unit_weights_inherit_household_once(tables, data):
    person, household = tables["person"].copy(), tables["household"].copy()
    person[TAX.pairing_eligibility_column] = True
    weights = np.asarray(
        data.draw(
            st.lists(
                st.integers(1, 10000), min_size=len(household), max_size=len(household)
            )
        ),
        dtype=float,
    )
    groups = build_declared_units(person, household, DECLARATION)
    tables = {"person": person, "household": household}
    for entity, group in groups.items():
        person[group.person_unit.name] = group.person_unit
        tables[entity] = group.units
    schema, _ = DECLARATION.entity_schema()
    order = np.argsort(household["household_id"].to_numpy())
    tables["household"] = household.iloc[order].reset_index(drop=True)
    frame = Frame(
        tables=tables,
        schema=schema,
        weights={"household": Weights(values=weights[order], kind=WeightKind.DESIGN)},
    )
    for item in DECLARATION.units:
        resolved = frame.resolve_weights(item.entity).values
        counts = units_per_household(household, groups[item.entity].units, item.rule)
        assert float(resolved.sum()) == float(np.dot(weights, counts))
        hh_weight = dict(zip(household["household_id"], weights, strict=True))
        assert resolved.tolist() == [
            hh_weight[value]
            for value in groups[item.entity].units[item.rule.household_column]
        ]


@PROPERTY
@given(eligible=st.tuples(st.booleans(), st.booleans()))
def test_pairing_requires_explicit_eligibility_on_both_members(eligible):
    person, household = _tables()
    person[TAX.pairing_eligibility_column] = [*eligible, False]
    group = build_declared_units(person, household, DECLARATION)[TAX.entity]
    paired = all(eligible)
    assert group.person_unit.tolist() == [10, 10 if paired else 20, 30]
    assert group.person_role.tolist() == [
        "head",
        "partner" if paired else "head",
        "head",
    ]


@PROPERTY
@given(pointer=st.sampled_from([None, 10, 30]))
def test_a_nonreciprocal_or_self_pointer_never_forms_a_pair(pointer):
    person, household = _tables()
    person.loc[1, "partner_person_id"] = pointer
    if pointer == 10:
        # Make the first pointer self-referential instead.
        person.loc[0, "partner_person_id"] = 10
    with pytest.raises(ValueError, match="asymmetric|self"):
        reciprocal_partner_rows(person, household)
    with pytest.raises(ValueError, match="asymmetric|self"):
        build_declared_units(person, household, DECLARATION)


def test_serialized_declaration_is_strict_and_roundtrips():
    encoded = DECLARATION.to_dict()
    assert encoded["format"] == MULTI_UNIT_RULE_FORMAT
    assert MultiUnitRule.from_dict(encoded).to_dict() == encoded
    assert MultiUnitRule.from_spec(encoded).to_dict() == encoded
    schema, nesting = DECLARATION.entity_schema()
    assert schema.group_entities == ("household", "family", "tax_unit")
    assert nesting == {"family": "household", "tax_unit": "household"}
    assert isinstance(nesting, MappingProxyType)
    assert MultiUnitRule.from_benefit_unit(BENEFIT).units[0].rule is BENEFIT


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda row: row.update(extra=True), "unexpected"),
        (lambda row: row["units"][0].update(extra=True), "unexpected"),
        (lambda row: row.update(format="unknown"), "format"),
        (lambda row: row.update(units=[]), "non-empty"),
        (lambda row: row.update(units={}), "Declared units must be a list\\."),
        (lambda row: row["units"].append(row["units"][0]), "repeat"),
        (lambda row: row["units"][0].update(kind="unknown"), "Unknown unit kind"),
        (lambda row: row["units"][1]["rule"].update(extra=True), "unexpected"),
        (lambda row: row["units"][1]["rule"].update(format="unknown"), "format"),
        (
            lambda row: row["units"][1]["rule"].update(entity="person"),
            "cannot be named",
        ),
        (lambda row: row["units"][1]["rule"].update(entity="TaxUnit"), "snake_case"),
        (
            lambda row: row["units"][1]["rule"].update(
                pairing_eligibility_column="bad column"
            ),
            "snake_case",
        ),
        (
            lambda row: row["units"][1]["rule"].update(provenance={"missing": ""}),
            "provenance",
        ),
    ],
)
def test_bad_serialized_declarations_refuse(mutate, message):
    encoded = DECLARATION.to_dict()
    mutate(encoded)
    with pytest.raises(ValueError, match=message):
        MultiUnitRule.from_dict(encoded)


@pytest.mark.parametrize("units", [None, "tax_unit", [], [TAX]])
def test_bad_declaration_lists_refuse(units):
    with pytest.raises(ValueError, match="non-empty"):
        MultiUnitRule(units=units)


def test_kind_and_rule_must_agree():
    with pytest.raises(ValueError, match="requires BenefitUnitRule"):
        UnitDeclaration(kind="benefit_unit", rule=TAX)
    with pytest.raises(ValueError, match="Unknown unit kind"):
        UnitDeclaration(kind="unknown", rule=TAX)
    with pytest.raises(TypeError, match="MultiUnitRule"):
        person, household = _tables()
        build_declared_units(person, household, BENEFIT)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [True, None, False],
        pd.array([True, None, False], dtype="boolean"),
        [1, 1, 0],
        ["true"] * 3,
    ],
)
def test_missing_or_malformed_eligibility_refuses(bad):
    person, household = _tables()
    if bad is None:
        person = person.drop(columns=TAX.pairing_eligibility_column)
    else:
        person[TAX.pairing_eligibility_column] = bad
    with pytest.raises(ValueError, match="eligibility"):
        build_declared_units(person, household, DECLARATION)


@pytest.mark.parametrize(
    "heads, message",
    [
        (np.array([0, 0]), "exactly one"),
        (np.array([-1, 0, 0]), "exactly one"),
        (np.array([0, 0, 3]), "exactly one"),
        (np.array([0.0, 0.0, 0.0]), "exactly one"),
        (np.array([1, 0, 2]), "own unit"),
    ],
)
def test_registered_constructor_must_return_a_valid_partition(heads, message):
    person, household = _tables()
    rows = _Pointers.read(person, household, TAX)
    with pytest.raises(ValueError, match=message):
        _build_partition(person, household, TAX, rows, heads)


def test_registered_constructor_cannot_cross_households():
    person, household = _tables()
    person.loc[2, "person_household_id"] = 2
    person.loc[2, ["parent_1_person_id", "parent_2_person_id"]] = None
    household.loc[1] = [2, 30, 902]
    rows = _Pointers.read(person, household, TAX)
    with pytest.raises(ValueError, match="one household"):
        _build_partition(person, household, TAX, rows, np.array([0, 0, 0]))


def test_changed_eligibility_changes_partition_identity():
    person, household = _tables()
    before = build_declared_units(person, household, DECLARATION)[TAX.entity]
    person[TAX.pairing_eligibility_column] = False
    after = build_declared_units(person, household, DECLARATION)[TAX.entity]
    # Structural membership is part of population identity: no legal proxy is
    # hidden in prepared state, even when the head id remains the same.
    assert before.person_unit.to_json() != after.person_unit.to_json()


def test_support_clones_keep_physical_ids_and_share_original_lineage():
    person, household = _tables()
    clone = person.copy()
    clone["person_id"] += 100
    clone["person_household_id"] += 1
    for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
        clone[column] += 100
    cloned_household = household.copy()
    cloned_household["household_id"] += 1
    cloned_household["reference_person_id"] += 100
    both = build_declared_units(
        pd.concat([person, clone], ignore_index=True),
        pd.concat([household, cloned_household], ignore_index=True),
        DECLARATION,
    )
    for item in DECLARATION.units:
        group, rule = both[item.entity], item.rule
        assert group.units[rule.id_column].is_unique
        assert set(group.units[rule.household_column]) == {1, 2}
        assert set(group.units[rule.source_household_column]) == {901}
        assert set(group.person_unit[:3]).isdisjoint(set(group.person_unit[3:]))


def test_partner_pointer_input_surface_is_required():
    person, household = _tables()
    with pytest.raises(ValueError, match="need columns"):
        reciprocal_partner_rows(person.drop(columns="partner_person_id"), household)


def test_tax_membership_does_not_need_dependency_facts():
    person, household = _tables()
    person = person.drop(columns=["age", "parent_1_person_id", "parent_2_person_id"])
    declaration = UnitDeclaration(kind="tax_unit", rule=TAX)
    group = build_declared_units(
        person, household, MultiUnitRule(units=(declaration,))
    )[TAX.entity]
    attributes = declared_unit_attributes(
        person, group.units, group.person_unit, declaration
    )
    assert attributes["n_members"].tolist() == [2, 1]
    assert attributes["n_dependent_children"].tolist() == [0, 0]
    assert attributes["youngest_dependent_child_age"].isna().all()


@pytest.mark.parametrize(
    "defect,message",
    [
        ("entries", "must be a list"),
        ("entry", "must be an object"),
        ("declaration", "declared identities"),
        ("identity", "strings or integers"),
        ("duplicate", "duplicate membership identities"),
        ("orphan", "orphan membership"),
        ("empty", "empty households"),
    ],
)
def test_roster_preparation_is_validated_by_the_unit_framework(defect, message):
    tables = {
        "household": pd.DataFrame({"household_id": [10, 20]}),
        "person": pd.DataFrame({"person_id": [1, 2], "person_household_id": [10, 20]}),
    }
    required = {entity: list(table.columns) for entity, table in tables.items()}
    entries = [
        {
            "entity": "household",
            "member_entity": "person",
            "name": "size",
            "entity_id": "household_id",
            "member_id": "person_id",
            "membership": "person_household_id",
            "representation": "one row per physical person",
        }
    ]
    if defect == "entries":
        entries = {}
    elif defect == "entry":
        entries = [None]
    elif defect == "declaration":
        entries[0]["representation"] = ""
    elif defect == "identity":
        tables["person"]["person_id"] = [True, False]
    elif defect == "duplicate":
        tables["person"]["person_id"] = [1, 1]
    elif defect == "orphan":
        tables["person"]["person_household_id"] = [10, 30]
    else:
        tables["person"]["person_household_id"] = [10, 10]
    with pytest.raises(ValueError, match=message):
        prepare_roster_counts(tables, entries, required)
