"""Frame.place: the single entity-placement operation builds rely on.

Covers all three directions (person -> group aggregation, group -> person
broadcast/head-carry, nested group -> group moves) and every refusal:
structural columns, direction-invalid ``how``, dtype-invalid ``how``,
ambiguous head flags, and non-nested group placement. Placement always MOVES
the column — the flattening rule (globally unique column names) forbids the
same name on two tables, so there is no keep-source variant.
"""

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


@pytest.fixture
def nested_schema() -> EntitySchema:
    """person + tax_unit nested inside household."""
    return EntitySchema(group_entities=("household", "tax_unit"))


@pytest.fixture
def nested_frame(nested_schema) -> Frame:
    """Five persons, two households, three tax units nested in households.

    Household 1 holds tax units 10 (persons 0, 1) — household 2 holds tax
    units 20 (persons 2, 3) and 21 (person 4). Values are hand-computable.
    """
    person = pd.DataFrame(
        {
            "person_id": np.arange(5, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 2, 2], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 10, 20, 20, 21], dtype="int64"),
            "is_head": np.asarray([True, False, True, False, False]),
            "pension_ald": np.asarray([100.0, 25.0, 0.0, 50.0, 7.0]),
            "is_disabled": np.asarray([False, True, False, False, False]),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([1, 2], dtype="int64"),
            "rent": np.asarray([12000.0, 0.0]),
        }
    )
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": np.asarray([10, 20, 21], dtype="int64"),
            "premium": np.asarray([5.0, 7.0, 11.0]),
        }
    )
    return Frame(
        {"person": person, "household": household, "tax_unit": tax_unit},
        nested_schema,
        {
            "household": Weights(
                values=np.asarray([100.0, 200.0]), kind=WeightKind.DESIGN
            )
        },
    )


class TestPersonToGroup:
    def test_sum_moves_column_to_group(self, nested_frame):
        placed = nested_frame.place("pension_ald", "tax_unit")
        np.testing.assert_allclose(
            placed.table("tax_unit")["pension_ald"].to_numpy(),
            [125.0, 50.0, 7.0],
        )
        assert "pension_ald" not in placed.person.columns
        # source frame untouched (place returns a new frame)
        assert "pension_ald" in nested_frame.person.columns

    def test_any_aggregates_booleans(self, nested_frame):
        placed = nested_frame.place("is_disabled", "household", how="any")
        np.testing.assert_array_equal(
            placed.table("household")["is_disabled"].to_numpy(), [True, False]
        )

    def test_max_takes_group_maximum(self, nested_frame):
        placed = nested_frame.place("pension_ald", "household", how="max")
        np.testing.assert_allclose(
            placed.table("household")["pension_ald"].to_numpy(), [100.0, 50.0]
        )

    def test_first_takes_first_member_value(self, nested_frame):
        placed = nested_frame.place("pension_ald", "tax_unit", how="first")
        np.testing.assert_allclose(
            placed.table("tax_unit")["pension_ald"].to_numpy(),
            [100.0, 0.0, 7.0],
        )

    def test_sum_refuses_boolean_column(self, nested_frame):
        with pytest.raises(ValueError, match="needs a numeric column"):
            nested_frame.place("is_disabled", "household", how="sum")

    def test_any_refuses_numeric_column(self, nested_frame):
        with pytest.raises(ValueError, match="needs a boolean column"):
            nested_frame.place("pension_ald", "household", how="any")


class TestGroupToPerson:
    def test_broadcast_gives_every_member_the_value(self, nested_frame):
        placed = nested_frame.place("rent", "person", how="broadcast")
        np.testing.assert_allclose(
            placed.person["rent"].to_numpy(),
            [12000.0, 12000.0, 0.0, 0.0, 0.0],
        )
        assert "rent" not in placed.table("household").columns

    def test_head_carry_with_flag(self, nested_frame):
        placed = nested_frame.place("rent", "person", how="head", head_flag="is_head")
        np.testing.assert_allclose(
            placed.person["rent"].to_numpy(),
            [12000.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_head_carry_first_member_when_no_flag(self, nested_frame):
        placed = nested_frame.place("premium", "person", how="head")
        np.testing.assert_allclose(
            placed.person["premium"].to_numpy(),
            [5.0, 0.0, 7.0, 0.0, 11.0],
        )

    def test_head_flag_must_name_exactly_one_member(self, nested_frame):
        # is_disabled flags one member of household 1 and nobody in
        # household 2 — not a valid head designation for households.
        with pytest.raises(ValueError, match="exactly one member"):
            nested_frame.place("rent", "person", how="head", head_flag="is_disabled")

    def test_sum_is_invalid_downward(self, nested_frame):
        with pytest.raises(ValueError, match="broadcast' or 'head'"):
            nested_frame.place("rent", "person", how="sum")


class TestGroupToGroup:
    def test_nested_sum_moves_tax_unit_to_household(self, nested_frame):
        placed = nested_frame.place("premium", "household")
        np.testing.assert_allclose(
            placed.table("household")["premium"].to_numpy(), [5.0, 18.0]
        )
        assert "premium" not in placed.table("tax_unit").columns

    def test_non_nested_placement_is_refused(self, nested_schema):
        # tax unit 10 spans both households -> household does not contain it.
        person = pd.DataFrame(
            {
                "person_id": np.arange(2, dtype="int64"),
                "person_household_id": np.asarray([1, 2], dtype="int64"),
                "person_tax_unit_id": np.asarray([10, 10], dtype="int64"),
            }
        )
        household = pd.DataFrame({"household_id": np.asarray([1, 2], dtype="int64")})
        tax_unit = pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10], dtype="int64"),
                "premium": np.asarray([5.0]),
            }
        )
        frame = Frame(
            {"person": person, "household": household, "tax_unit": tax_unit},
            nested_schema,
            {
                "household": Weights(
                    values=np.asarray([10.0, 20.0]), kind=WeightKind.DESIGN
                )
            },
        )
        with pytest.raises(ValueError, match="does not nest"):
            frame.place("premium", "household")


class TestRefusals:
    def test_unknown_destination(self, nested_frame):
        with pytest.raises(ValueError, match="Unknown destination entity"):
            nested_frame.place("pension_ald", "spm_unit")

    def test_structural_columns_refused(self, nested_frame):
        with pytest.raises(ValueError, match="structural"):
            nested_frame.place("person_tax_unit_id", "tax_unit")

    def test_same_entity_is_identity(self, nested_frame):
        assert nested_frame.place("pension_ald", "person") is nested_frame

    def test_unknown_how_refused(self, nested_frame):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            nested_frame.place("pension_ald", "tax_unit", how="mean")


class TestInvariantsPreserved:
    def test_weights_strata_mass_log_pass_through(self, nested_frame):
        placed = nested_frame.place("pension_ald", "tax_unit")
        np.testing.assert_allclose(
            placed.weights_for("household").values, [100.0, 200.0]
        )
        assert placed.mass_log == nested_frame.mass_log
        assert tuple(placed.strata) == tuple(nested_frame.strata)


class TestDtypeContracts:
    def test_first_carries_object_dtype(self, nested_frame):
        frame = nested_frame
        person = frame.person.copy()
        person["occupation"] = ["nurse", "", "clerk", "farmer", "pilot"]
        frame = Frame(
            {
                "person": person,
                "household": frame.table("household"),
                "tax_unit": frame.table("tax_unit"),
            },
            frame.schema,
            {"household": frame.weights_for("household")},
        )
        placed = frame.place("occupation", "tax_unit", how="first")
        assert list(placed.table("tax_unit")["occupation"]) == [
            "nurse",
            "clerk",
            "pilot",
        ]

    def test_head_flag_with_non_head_how_is_refused(self, nested_frame):
        with pytest.raises(ValueError, match='only meaningful with how="head"'):
            nested_frame.place("rent", "person", how="broadcast", head_flag="is_head")
