"""WeightedBundle accessors, broadcast, weights resolution, concat, select."""

import numpy as np
import pandas as pd
import pytest
from microframe import (
    DEFAULT_STRATUM,
    EntitySchema,
    WeightedBundle,
    WeightKind,
    Weights,
    wsum,
)


class TestAccessors:
    def test_person_and_table_and_n(self, make_bundle) -> None:
        bundle = make_bundle()
        assert bundle.n("person") == 5
        assert bundle.n("household") == 2
        assert bundle.entities == ("person", "household")
        assert list(bundle.person.columns) == [
            "person_id",
            "person_household_id",
            "age",
            "income",
        ]
        assert bundle.table("household")["state"].tolist() == ["CA", "NY"]

    def test_unknown_entity_is_named(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="tax_unit"):
            bundle.table("tax_unit")

    def test_weights_for_missing_entity_names_weighted_entities(
        self, make_bundle
    ) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match=r"weighted entities: \['household'\]"):
            bundle.weights_for("person")
        assert bundle.weighted_entities == ("household",)

    def test_default_stratum_is_assigned(self, make_bundle) -> None:
        bundle = make_bundle()
        assert set(bundle.strata.unique()) == {DEFAULT_STRATUM}

    def test_tables_are_copied_from_caller(self, simple_schema) -> None:
        person = pd.DataFrame(
            {"person_id": [0], "person_household_id": [1], "x": [1.0]}
        )
        household = pd.DataFrame({"household_id": [1]})
        weights = {
            "household": Weights(values=np.array([2.0]), kind=WeightKind.DESIGN)
        }
        bundle = WeightedBundle(
            {"person": person, "household": household}, simple_schema, weights
        )
        person.loc[0, "x"] = 99.0  # caller mutation must not reach the bundle
        assert bundle.person["x"].tolist() == [1.0]

    def test_missing_entity_table_is_named(self, simple_schema) -> None:
        person = pd.DataFrame({"person_id": [0], "person_household_id": [1]})
        weights = {
            "household": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)
        }
        with pytest.raises(ValueError, match="household"):
            WeightedBundle({"person": person}, simple_schema, weights)

    def test_unknown_entity_table_is_named(self, simple_schema, make_bundle) -> None:
        bundle = make_bundle()
        tables = {
            "person": bundle.person,
            "household": bundle.table("household"),
            "spaceship": pd.DataFrame({"spaceship_id": [1]}),
        }
        with pytest.raises(ValueError, match="spaceship"):
            WeightedBundle(
                tables, simple_schema, {"household": bundle.weights_for("household")}
            )

    def test_weights_for_unknown_entity_key_rejected(self, make_bundle) -> None:
        bundle = make_bundle()
        weights = {
            "household": bundle.weights_for("household"),
            "galaxy": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN),
        }
        with pytest.raises(ValueError, match="galaxy"):
            WeightedBundle(
                {"person": bundle.person, "household": bundle.table("household")},
                bundle.schema,
                weights,
            )

    def test_weights_must_not_be_empty(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="at least one entity"):
            WeightedBundle(
                {"person": bundle.person, "household": bundle.table("household")},
                bundle.schema,
                {},
            )

    def test_strata_must_not_contain_missing_labels(self, make_bundle) -> None:
        bundle = make_bundle()
        strata = pd.Series(["a", "a", None, "b", "b"], index=bundle.person.index)
        with pytest.raises(ValueError, match="missing"):
            WeightedBundle(
                {"person": bundle.person, "household": bundle.table("household")},
                bundle.schema,
                {"household": bundle.weights_for("household")},
                strata=strata,
            )

    def test_unsorted_group_ids_rejected(self, simple_schema) -> None:
        person = pd.DataFrame(
            {"person_id": [0, 1], "person_household_id": [2, 1]}
        )
        household = pd.DataFrame({"household_id": [2, 1]})
        weights = {
            "household": Weights(values=np.array([1.0, 1.0]), kind=WeightKind.DESIGN)
        }
        with pytest.raises(ValueError, match="sorted"):
            WeightedBundle(
                {"person": person, "household": household}, simple_schema, weights
            )


class TestBroadcast:
    def test_group_column_broadcasts_to_persons(self, make_bundle) -> None:
        bundle = make_bundle()
        state = bundle.broadcast("state")
        assert state.tolist() == ["CA", "CA", "NY", "NY", "NY"]
        assert state.index.equals(bundle.person.index)

    def test_person_column_broadcasts_to_itself(self, make_bundle) -> None:
        bundle = make_bundle()
        assert bundle.broadcast("age").tolist() == [40, 8, 30, 28, 1]

    def test_unknown_column_is_named(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="'nope'"):
            bundle.broadcast("nope")

    def test_only_person_target_supported(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="person"):
            bundle.broadcast("age", to="household")

    def test_column_entity_resolves_owner(self, make_bundle) -> None:
        bundle = make_bundle()
        assert bundle.column_entity("state") == "household"
        assert bundle.column_entity("income") == "person"


class TestEffectiveWeights:
    """Weight resolution for accounting: explicit, broadcast, or collapsed."""

    @staticmethod
    def _nested_bundle(person_weights: bool = False) -> WeightedBundle:
        """person + household + tax_unit, tax units nested within households."""
        schema = EntitySchema(group_entities=("household", "tax_unit"))
        person = pd.DataFrame(
            {
                "person_id": range(4),
                "person_household_id": [1, 1, 2, 2],
                "person_tax_unit_id": [1, 1, 2, 3],
                "income": [10.0, 0.0, 7.0, 3.0],
            }
        )
        household = pd.DataFrame({"household_id": [1, 2]})
        tax_unit = pd.DataFrame(
            {"tax_unit_id": [1, 2, 3], "tax_unit_income": [10.0, 7.0, 3.0]}
        )
        weights: dict[str, Weights] = {
            "household": Weights(values=np.array([5.0, 11.0]), kind=WeightKind.DESIGN)
        }
        if person_weights:
            weights["person"] = Weights(
                values=np.array([1.0, 1.0, 2.0, 2.0]), kind=WeightKind.DESIGN
            )
        return WeightedBundle(
            {"person": person, "household": household, "tax_unit": tax_unit},
            schema,
            weights,
        )

    def test_group_entity_weights_derive_from_member_constant_person_weights(
        self,
    ) -> None:
        bundle = self._nested_bundle()
        # tax units 1 (hh 1, weight 5), 2 and 3 (hh 2, weight 11):
        # wsum = 5*10 + 11*7 + 11*3 = 160
        assert wsum(bundle, "tax_unit_income") == 160.0

    def test_explicit_person_weights_take_precedence(self) -> None:
        bundle = self._nested_bundle(person_weights=True)
        # person weights [1,1,2,2]: wsum income = 10 + 0 + 14 + 6 = 30
        assert wsum(bundle, "income") == 30.0

    def test_ambiguous_person_weights_are_refused(self) -> None:
        schema = EntitySchema(group_entities=("household", "tax_unit"))
        bundle = self._nested_bundle()
        with_tax_unit_weights = bundle.with_weights(
            "tax_unit",
            Weights(values=np.array([1.0, 1.0, 1.0]), kind=WeightKind.DESIGN),
        )
        assert with_tax_unit_weights.schema == schema
        with pytest.raises(ValueError, match="weighted group entities"):
            wsum(with_tax_unit_weights, "income")

    def test_non_constant_member_weights_are_refused_for_group_collapse(
        self,
    ) -> None:
        bundle = self._nested_bundle(person_weights=True)
        # Tax unit 1 spans persons with weights... here they are constant; make
        # them unequal by replacing person weights (design -> design allowed).
        uneven = bundle.with_weights(
            "person",
            Weights(values=np.array([1.0, 3.0, 2.0, 2.0]), kind=WeightKind.DESIGN),
        )
        with pytest.raises(ValueError, match="unequal person-level weights"):
            wsum(uneven, "tax_unit_income")


class TestStratumMass:
    def test_mass_per_stratum_via_household_broadcast(self, make_bundle) -> None:
        strata = pd.Series(["cps", "cps", "syn", "syn", "syn"], index=range(5))
        bundle = make_bundle(strata=strata)
        mass = bundle.stratum_mass()
        assert mass["cps"] == 200.0  # two persons in household 1 (weight 100)
        assert mass["syn"] == 600.0  # three persons in household 2 (weight 200)


class TestWithWeights:
    def test_returns_new_bundle_and_keeps_original(self, make_bundle) -> None:
        bundle = make_bundle()
        replacement = Weights(
            values=np.array([1.0, 2.0]), kind=WeightKind.CALIBRATED
        )
        updated = bundle.with_weights("household", replacement)
        assert bundle.weights_for("household").kind is WeightKind.DESIGN
        assert updated.weights_for("household").kind is WeightKind.CALIBRATED

    def test_adding_weights_to_unweighted_entity(self, make_bundle) -> None:
        bundle = make_bundle()
        person_weights = Weights(values=np.ones(5), kind=WeightKind.IMPORTANCE)
        updated = bundle.with_weights("person", person_weights)
        assert set(updated.weighted_entities) == {"person", "household"}

    def test_require_mass_without_existing_weights_is_an_error(
        self, make_bundle
    ) -> None:
        bundle = make_bundle()
        person_weights = Weights(values=np.ones(5), kind=WeightKind.DESIGN)
        with pytest.raises(ValueError, match="existing weights"):
            bundle.with_weights("person", person_weights, require_mass=True)

    def test_length_mismatch_is_an_error(self, make_bundle) -> None:
        bundle = make_bundle()
        wrong = Weights(values=np.array([1.0]), kind=WeightKind.CALIBRATED)
        with pytest.raises(ValueError, match="length 1"):
            bundle.with_weights("household", wrong)


class TestConcat:
    def test_column_set_mismatch_is_named(self, make_bundle) -> None:
        a = make_bundle()
        b = make_bundle(strata=pd.Series("other", index=range(5)))
        extra = b.person.copy()
        extra["bonus"] = 1.0
        b_extra = WeightedBundle(
            {"person": extra, "household": b.table("household")},
            b.schema,
            {"household": b.weights_for("household")},
            strata=b.strata,
        )
        with pytest.raises(ValueError, match="bonus"):
            a.concat(b_extra)

    def test_weighted_entity_mismatch_is_named(self, make_bundle) -> None:
        a = make_bundle()
        b = make_bundle(strata=pd.Series("other", index=range(5)))
        b = b.with_weights("person", Weights(values=np.ones(5), kind=WeightKind.DESIGN))
        with pytest.raises(ValueError, match="weighted entities differ"):
            a.concat(b)

    def test_schema_mismatch_is_an_error(self, make_bundle) -> None:
        a = make_bundle()
        other_schema = EntitySchema(group_entities=("family",))
        person = pd.DataFrame({"person_id": [0], "person_family_id": [1]})
        family = pd.DataFrame({"family_id": [1]})
        b = WeightedBundle(
            {"person": person, "family": family},
            other_schema,
            {"family": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)},
        )
        with pytest.raises(ValueError, match="schema"):
            a.concat(b)

    def test_id_shift_preserves_membership_structure(self, make_bundle) -> None:
        a = make_bundle()
        b = make_bundle(strata=pd.Series("other", index=range(5)))
        union = a.concat(b)
        person = union.person
        other_half = person.iloc[5:]
        # Shifted household ids still partition the second bundle's persons
        # into two households of sizes 2 and 3.
        sizes = other_half.groupby("person_household_id").size().tolist()
        assert sorted(sizes) == [2, 3]
        assert union.table("household")["household_id"].tolist() == [1, 2, 3, 4]


class TestSelect:
    def test_series_mask_must_be_index_aligned(self, make_bundle) -> None:
        bundle = make_bundle()
        misaligned = pd.Series([True] * 5, index=range(10, 15))
        with pytest.raises(ValueError, match="index-aligned"):
            bundle.select(misaligned)

    def test_non_boolean_mask_is_rejected(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="boolean"):
            bundle.select(np.array([1, 0, 1, 0, 1]))

    def test_wrong_length_mask_is_rejected(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="one element per person"):
            bundle.select(np.array([True, False]))

    def test_select_slices_strata_and_weights_together(self, make_bundle) -> None:
        strata = pd.Series(["a", "a", "b", "b", "b"], index=range(5))
        bundle = make_bundle(strata=strata)
        selected = bundle.select(np.array([False, False, True, True, True]))
        assert set(selected.strata.unique()) == {"b"}
        assert selected.weights_for("household").values.tolist() == [200.0]
        assert selected.stratum_mass()["b"] == 600.0
