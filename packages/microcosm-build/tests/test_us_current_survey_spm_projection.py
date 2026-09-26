"""Pure invented clone transport; detached fixtures confer no source authority."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
from microcosm.build.us_runtime import current_survey_spm_projection as projection
from microcosm.build.us_runtime import support_provenance as provenance
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def fixture():
    people = pd.DataFrame(
        {
            "person_id": [10, 11, 20, 21, 30],
            "person_spm_unit_id": [100, 100, 200, 200, 300],
            "person_household_id": [1, 1, 2, 2, 3],
        }
    )
    source = SimpleNamespace(person=people)
    for entity, ids, channels in (
        ("person", [10, 11, 20, 21, 30], ["acs", "acs", "asec", "asec", "acs"]),
        ("spm_unit", [100, 200, 300], ["acs", "asec", "acs"]),
        ("household", [1, 2, 3], ["acs", "asec", "acs"]),
    ):
        table = people if entity == "person" else pd.DataFrame({entity + "_id": ids})
        table[provenance.spine_source_id_column(entity)] = np.array(ids) + 1000
        table[provenance.support_channel_column(entity)] = channels
        setattr(source, entity, table)
    receiving = SimpleNamespace()
    for entity in ("person", "spm_unit", "household"):
        original = getattr(source, entity)
        clones = []
        for clone in (0, 1):
            table = original.copy(deep=True)
            table[provenance.support_source_id_column(entity)] = original[
                entity + "_id"
            ]
            table[provenance.support_clone_index_column(entity)] = clone
            table[entity + "_id"] += 10000 * (clone + 1)
            if entity == "person":
                table.person_spm_unit_id += 10000 * (clone + 1)
                table.person_household_id += 10000 * (clone + 1)
            clones.append(table)
        setattr(
            receiving, entity, pd.concat(clones, ignore_index=True).iloc[::-1].copy()
        )
    origins = pd.DataFrame(
        {
            "source": source.person[
                provenance.support_channel_column("person")
            ].to_numpy(),
            "native_person_id": source.person[
                provenance.spine_source_id_column("person")
            ].to_numpy(),
            "source_year": 2024,
            "survey_year": [2024, 2024, 2025, 2025, 2024],
        },
        index=pd.Index(source.person.person_id, name="person_id"),
    )
    roles = pd.Series(
        [True, False, True, False, None], dtype="boolean", index=origins.index
    )
    status = pd.Series(
        ["INCLUDED", "UNRESOLVED", "OUTSIDE"],
        index=pd.Index(source.spm_unit.spm_unit_id, name="spm_unit_id"),
        dtype="str",
    )
    return source, origins, roles, status, receiving


def run(args, **kwargs):
    return projection.project_spm_inputs(
        *args,
        source_year=kwargs.pop("source_year", 2024),
        year=kwargs.pop("year", 2024),
        outside_role_placeholder=kwargs.pop("outside_role_placeholder", False),
        **kwargs,
    )


@pytest.mark.parametrize("sides", [(0,), (4,), (0, 4)])
def test_generic_entity_tables_on_real_frames_match_table_views(sides):
    args = list(fixture())
    for side in (0, 4):
        for entity in ("person", "spm_unit", "household"):
            table = getattr(args[side], entity)
            setattr(
                args[side],
                entity,
                table.sort_values(entity + "_id").reset_index(drop=True),
            )
    expected = run(args)
    for side in sides:
        view = args[side]
        people = view.person.copy(deep=True)
        tables = {"person": people}
        for entity in US_SCHEMA.group_entities:
            if hasattr(view, entity):
                tables[entity] = getattr(view, entity).copy(deep=True)
            else:
                people[US_SCHEMA.membership_column(entity)] = people.person_id
                tables[entity] = pd.DataFrame({entity + "_id": people.person_id})
        args[side] = Frame(
            tables,
            US_SCHEMA,
            {
                "household": Weights(
                    np.ones(len(tables["household"])), WeightKind.DESIGN
                )
            },
        )
    observed = run(args)
    assert projection.spm_projection_seal(observed) == projection.spm_projection_seal(
        expected
    )


def test_exact_units_and_both_outside_representations_preserve_inputs():
    args = fixture()
    before = {
        (side, entity): getattr(args[side], entity).copy(deep=True)
        for side in (0, 4)
        for entity in ("person", "spm_unit", "household")
    }
    original_roles = args[2].copy(deep=True)
    false, true = run(args), run(args, outside_role_placeholder=True)
    assert set(false.columns) == {("person", ROLE_INPUT), ("spm_unit", UNIVERSE_INPUT)}
    expected = args[4].person.person_id.to_list()
    assert false.columns["person", ROLE_INPUT].index.to_list() == expected
    assert false.columns["person", ROLE_INPUT].dtype == np.dtype("bool")
    assert false.placeholder_person_ids == (20030, 10030)
    differing = (
        false.columns["person", ROLE_INPUT] != true.columns["person", ROLE_INPUT]
    )
    assert differing[differing].index.tolist() == [20030, 10030]
    pd.testing.assert_series_equal(false.nullable_source_roles, original_roles)
    assert false.nullable_source_roles is not args[2]
    assert false.year == 2024
    assert false.unit_mapping == (
        (20300, 300, 1),
        (20200, 200, 1),
        (20100, 100, 1),
        (10300, 300, 0),
        (10200, 200, 0),
        (10100, 100, 0),
    )
    for (side, entity), table in before.items():
        pd.testing.assert_frame_equal(
            table, getattr(args[side], entity), check_exact=True
        )
    pd.testing.assert_series_equal(args[2], original_roles)


@pytest.mark.parametrize("status", ["INCLUDED", "UNRESOLVED"])
def test_unknown_role_cannot_be_filled_inside_or_unresolved(status):
    args = fixture()
    args[3].loc[300] = status
    with pytest.raises(ValueError, match="NULL_ROLE_REQUIRES_OUTSIDE"):
        run(args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"year": 2025},
        {"source_year": 2023},
        {"year": True},
        {"outside_role_placeholder": 0},
    ],
)
def test_no_annual_carry_or_implicit_representation(kwargs):
    with pytest.raises(ValueError):
        run(fixture(), **kwargs)


@pytest.mark.parametrize("entity", ["person", "spm_unit", "household"])
@pytest.mark.parametrize("kind", ["native", "channel", "clone", "source"])
def test_changed_ancestry_refuses(entity, kind):
    args = fixture()
    table = getattr(args[4], entity)
    column = {
        "native": provenance.spine_source_id_column(entity),
        "channel": provenance.support_channel_column(entity),
        "clone": provenance.support_clone_index_column(entity),
        "source": provenance.support_source_id_column(entity),
    }[kind]
    row = table.index[0]
    table.loc[row, column] = "asec" if kind == "channel" else 9999
    with pytest.raises(ValueError):
        run(args)


@pytest.mark.parametrize("link", ["person_spm_unit_id", "person_household_id"])
def test_moved_member_even_with_complete_person_roster_refuses(link):
    args = fixture()
    people = args[4].person
    people.loc[people.person_id.eq(10011), link] = 10200 if "spm" in link else 10002
    with pytest.raises(ValueError, match="MEMBERSHIP"):
        run(args)


def test_mixing_clone_members_refuses():
    args = fixture()
    people = args[4].person
    people.loc[people.person_id.eq(10010), "person_spm_unit_id"] = 20100
    people.loc[people.person_id.eq(20010), "person_spm_unit_id"] = 10100
    with pytest.raises(ValueError, match="MEMBERSHIP"):
        run(args)


def test_unrepresented_empty_unit_refuses():
    args = fixture()
    args[4].spm_unit = pd.concat(
        [args[4].spm_unit, args[4].spm_unit.iloc[:1]], ignore_index=True
    )
    with pytest.raises(ValueError):
        run(args)


@pytest.mark.parametrize(
    "entity,column", [("person", ROLE_INPUT), ("spm_unit", UNIVERSE_INPUT)]
)
def test_existing_outputs_never_overwritten(entity, column):
    args = fixture()
    getattr(args[4], entity)[column] = False if entity == "person" else "INCLUDED"
    with pytest.raises(ValueError, match="COLLISION"):
        run(args)


def test_source_role_and_status_axes_are_exact():
    args = list(fixture())
    args[2] = args[2].iloc[::-1]
    with pytest.raises(ValueError, match="ROLE_AXIS"):
        run(args)
    args = list(fixture())
    args[3] = args[3].iloc[::-1]
    with pytest.raises(ValueError, match="STATUS_AXIS"):
        run(args)


@pytest.mark.parametrize("entity", ["spm_unit", "household"])
def test_group_source_must_match_its_people(entity):
    args = fixture()
    column = provenance.support_channel_column(entity)
    getattr(args[0], entity).loc[:, column] = "asec"
    getattr(args[4], entity).loc[:, column] = "asec"
    with pytest.raises(ValueError, match="GROUP_PERSON_SOURCE"):
        run(args)


@pytest.mark.parametrize("column", ["native_person_id", "source_year", "survey_year"])
def test_origin_identifiers_and_periods_are_not_coerced(column):
    args = fixture()
    args[1][column] = args[1][column].astype(float)
    with pytest.raises(ValueError, match="IDENTIFIER_TYPE"):
        run(args)


@pytest.mark.parametrize(
    "position,reason", [(1, "ORIGIN_AXIS"), (2, "ROLE_AXIS"), (3, "STATUS_AXIS")]
)
def test_source_axes_cannot_be_coerced_from_float(position, reason):
    args = fixture()
    args[position].index = args[position].index.astype(float)
    with pytest.raises(ValueError, match=reason):
        run(args)


def test_final_projection_validation_preserves_unchanged_result():
    result = run(fixture())
    calls = []
    assert (
        projection.validate_spm_projection(result, lambda: calls.append(True)) is result
    )
    assert calls == [True]


@pytest.mark.parametrize(
    "field",
    [
        "role",
        "status",
        "nullable",
        "nullable_name",
        "dtype",
        "index_name",
        "index_ids",
        "mapping_key",
        "placeholders",
        "unit_mapping",
        "year",
    ],
)
def test_projection_mutation_during_final_validation_refuses(field):
    from types import MappingProxyType

    result = run(fixture())

    def mutate():
        role = result.columns["person", ROLE_INPUT]
        if field == "role":
            role.iloc[0] = not bool(role.iloc[0])
        elif field == "status":
            result.columns["spm_unit", UNIVERSE_INPUT].iloc[0] = "INCLUDED"
        elif field == "nullable":
            result.nullable_source_roles.iloc[0] = False
        elif field == "nullable_name":
            assert result.nullable_source_roles.name is None
            result.nullable_source_roles.name = 0
        elif field == "index_name":
            role.index.name = "wrong"
        elif field == "index_ids":
            role.index = role.index + 1
        elif field in ("dtype", "mapping_key"):
            columns = dict(result.columns)
            if field == "dtype":
                columns["person", ROLE_INPUT] = role.astype("boolean")
            else:
                columns["person", "wrong"] = columns.pop(("person", ROLE_INPUT))
            object.__setattr__(result, "columns", MappingProxyType(columns))
        elif field == "placeholders":
            object.__setattr__(result, "placeholder_person_ids", ())
        elif field == "unit_mapping":
            object.__setattr__(result, "unit_mapping", result.unit_mapping[::-1])
        else:
            object.__setattr__(result, "year", 2025)

    with pytest.raises(ValueError, match="SPM_PROJECTION_RESULT_CHANGED"):
        projection.validate_spm_projection(result, mutate)
