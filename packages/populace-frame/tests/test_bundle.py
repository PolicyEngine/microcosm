"""Frame accessors, broadcast, weights resolution, concat, select."""

import numpy as np
import pandas as pd
import pytest

from populace.frame import (
    DEFAULT_STRATUM,
    EntitySchema,
    Frame,
    MassChange,
    WeightKind,
    Weights,
    wsum,
)

#: A declared mass change with unspecified magnitude, for tests that replace
#: weights to exercise non-mass behavior (kind transitions, length, collapse).
_FREE_MASS = MassChange(factor=None, reason="test fixture: mass not under test")


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
        weights = {"household": Weights(values=np.array([2.0]), kind=WeightKind.DESIGN)}
        bundle = Frame(
            {"person": person, "household": household}, simple_schema, weights
        )
        person.loc[0, "x"] = 99.0  # caller mutation must not reach the bundle
        assert bundle.person["x"].tolist() == [1.0]

    def test_missing_entity_table_is_named(self, simple_schema) -> None:
        person = pd.DataFrame({"person_id": [0], "person_household_id": [1]})
        weights = {"household": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)}
        with pytest.raises(ValueError, match="household"):
            Frame({"person": person}, simple_schema, weights)

    def test_unknown_entity_table_is_named(self, simple_schema, make_bundle) -> None:
        bundle = make_bundle()
        tables = {
            "person": bundle.person,
            "household": bundle.table("household"),
            "spaceship": pd.DataFrame({"spaceship_id": [1]}),
        }
        with pytest.raises(ValueError, match="spaceship"):
            Frame(tables, simple_schema, {"household": bundle.weights_for("household")})

    def test_weights_for_unknown_entity_key_rejected(self, make_bundle) -> None:
        bundle = make_bundle()
        weights = {
            "household": bundle.weights_for("household"),
            "galaxy": Weights(values=np.array([1.0]), kind=WeightKind.DESIGN),
        }
        with pytest.raises(ValueError, match="galaxy"):
            Frame(
                {"person": bundle.person, "household": bundle.table("household")},
                bundle.schema,
                weights,
            )

    def test_weights_must_not_be_empty(self, make_bundle) -> None:
        bundle = make_bundle()
        with pytest.raises(ValueError, match="at least one entity"):
            Frame(
                {"person": bundle.person, "household": bundle.table("household")},
                bundle.schema,
                {},
            )

    def test_strata_must_not_contain_missing_labels(self, make_bundle) -> None:
        bundle = make_bundle()
        strata = pd.Series(["a", "a", None, "b", "b"], index=bundle.person.index)
        with pytest.raises(ValueError, match="missing"):
            Frame(
                {"person": bundle.person, "household": bundle.table("household")},
                bundle.schema,
                {"household": bundle.weights_for("household")},
                strata=strata,
            )

    def test_stray_weight_column_is_reserved_by_the_kernel(self, simple_schema) -> None:
        """C1: a {entity}_weight column the bundle doesn't own as typed weights
        is a reserved name the kernel materializes at export — it is refused."""
        person = pd.DataFrame(
            {"person_id": [0, 1], "person_household_id": [1, 1], "x": [1.0, 2.0]}
        )
        household = pd.DataFrame({"household_id": [1], "household_weight": [42.0]})
        # household carries a household_weight COLUMN but only person carries
        # typed weights — the column is an orphan the engine would consume.
        weights = {
            "person": Weights(values=np.array([3.0, 3.0]), kind=WeightKind.DESIGN)
        }
        with pytest.raises(ValueError, match="reserved weight-column name"):
            Frame({"person": person, "household": household}, simple_schema, weights)

    def test_unsorted_group_ids_rejected(self, simple_schema) -> None:
        person = pd.DataFrame({"person_id": [0, 1], "person_household_id": [2, 1]})
        household = pd.DataFrame({"household_id": [2, 1]})
        weights = {
            "household": Weights(values=np.array([1.0, 1.0]), kind=WeightKind.DESIGN)
        }
        with pytest.raises(ValueError, match="sorted"):
            Frame({"person": person, "household": household}, simple_schema, weights)


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
    def _nested_bundle(person_weights: bool = False) -> Frame:
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
        return Frame(
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
            mass=_FREE_MASS,
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
            mass=_FREE_MASS,
        )
        with pytest.raises(ValueError, match="unequal person-level weights"):
            wsum(uneven, "tax_unit_income")


class TestResolveWeights:
    """Public kind-preserving weight resolution (the fit-on-a-grouped-frame fix).

    ``resolve_weights`` returns a typed :class:`Weights` for *any* entity,
    resolving inheritance and carrying the source entity's kind — unlike
    ``weights_for``, which only returns an entity's own stored vector. This is
    what lets a person-level fit run on a household-weighted frame.
    """

    def test_person_resolve_on_household_weighted_frame_carries_kind_and_values(
        self, make_bundle
    ) -> None:
        """A person resolve inherits the household's kind and broadcast values.

        ``weights_for("person")`` raises (no stored person weights); the new
        ``resolve_weights("person")`` instead returns the household design
        weights broadcast through membership, *as a Weights of kind design* —
        not a bare ndarray that has dropped the kind.
        """
        bundle = make_bundle(weight_values=(100.0, 200.0))
        # The old accessor still refuses an entity without its own weights.
        with pytest.raises(ValueError, match="No weights stored"):
            bundle.weights_for("person")

        resolved = bundle.resolve_weights("person")
        assert isinstance(resolved, Weights)
        # Kind is preserved from the source (household) entity.
        assert resolved.kind is WeightKind.DESIGN
        # Values are the household weights broadcast onto the 5 persons:
        # persons 0-1 in hh 1 (100), persons 2-4 in hh 2 (200).
        assert resolved.values.tolist() == [100.0, 100.0, 200.0, 200.0, 200.0]
        # And exactly the effective-weight values accounting uses.
        np.testing.assert_array_equal(
            resolved.values, bundle._effective_weights("person")
        )

    def test_calibrated_household_resolves_to_calibrated_person(
        self, make_bundle
    ) -> None:
        """A calibrated household frame resolves person weights as calibrated.

        The kind moves forward with the source: when the household weights are
        calibrated, a person inherits ``calibrated``, so a fit that demands the
        kind match sees ``calibrated`` (not a kind the inherited vector lost).
        """
        bundle = make_bundle(weight_values=(100.0, 200.0), kind=WeightKind.CALIBRATED)
        resolved = bundle.resolve_weights("person")
        assert resolved.kind is WeightKind.CALIBRATED
        assert resolved.values.tolist() == [100.0, 100.0, 200.0, 200.0, 200.0]

    def test_entity_with_its_own_weights_returns_them_as_is(self, make_bundle) -> None:
        """When the entity stores weights, resolve returns that exact object."""
        bundle = make_bundle(weight_values=(100.0, 200.0))
        # The household stores its own weights: returned identically.
        assert bundle.resolve_weights("household") is bundle.weights_for("household")

    def test_ambiguity_still_raises(self) -> None:
        """Zero or multiple weighted group entities make a person resolve ambiguous.

        ``resolve_weights`` keeps the same ambiguity guard ``_effective_weights``
        has: with two weighted group entities there is no single source kind to
        carry, so it refuses.
        """
        schema = EntitySchema(group_entities=("household", "tax_unit"))
        person = pd.DataFrame(
            {
                "person_id": range(4),
                "person_household_id": [1, 1, 2, 2],
                "person_tax_unit_id": [1, 1, 2, 3],
            }
        )
        household = pd.DataFrame({"household_id": [1, 2]})
        tax_unit = pd.DataFrame({"tax_unit_id": [1, 2, 3]})
        bundle = Frame(
            {"person": person, "household": household, "tax_unit": tax_unit},
            schema,
            {
                "household": Weights(
                    values=np.array([5.0, 11.0]), kind=WeightKind.DESIGN
                ),
                "tax_unit": Weights(
                    values=np.array([1.0, 2.0, 3.0]), kind=WeightKind.DESIGN
                ),
            },
        )
        with pytest.raises(ValueError, match="weighted group entities"):
            bundle.resolve_weights("person")

    def test_unknown_entity_is_named(self, make_bundle) -> None:
        """Resolving an undeclared entity raises, naming the schema's entities."""
        bundle = make_bundle()
        with pytest.raises(ValueError, match="Unknown entity"):
            bundle.resolve_weights("firm")

    def test_group_resolve_on_a_person_only_weighted_frame(self) -> None:
        """A group entity derives its weights from the person weights.

        Regression: when only the person entity is weighted (no weighted group
        entity — the shape the fit suite's own fixtures build), resolving a
        group entity must derive its weights from the person weights, exactly
        as ``_effective_weights`` does, not raise "0 weighted group entities".
        Because all accounting routes through ``resolve_weights``, the earlier
        version broke ``wsum``/``wmean``/etc. on group columns of such frames.
        """
        schema = EntitySchema(group_entities=("household",))
        person = pd.DataFrame(
            {"person_id": range(4), "person_household_id": [1, 1, 2, 2]}
        )
        household = pd.DataFrame({"household_id": [1, 2], "hh_value": [10.0, 20.0]})
        frame = Frame(
            {"person": person, "household": household},
            schema,
            {
                "person": Weights(
                    values=np.array([1.0, 1.0, 2.0, 2.0]), kind=WeightKind.DESIGN
                )
            },
        )
        resolved = frame.resolve_weights("household")
        assert resolved.kind is WeightKind.DESIGN  # from the person source
        assert resolved.values.tolist() == [1.0, 2.0]  # member-constant collapse
        # accounting on a group column works (it raised before the fix)
        assert wsum(frame, "hh_value") == 1.0 * 10.0 + 2.0 * 20.0  # 50.0

    def test_mixed_kind_resolve_tags_values_with_their_source_kind(self) -> None:
        """A resolved kind names the source the *values* came from.

        person calibrated + household design; resolving ``tax_unit`` derives
        its values from the person (calibrated) weights, so the kind must be
        calibrated too — not the sibling household's design. The earlier
        version tagged person-derived values with the group's kind, which would
        let ``resolve_fit_weights(..., "design")`` silently fit on calibrated
        weights — the exact discipline this API exists to enforce.
        """
        schema = EntitySchema(group_entities=("household", "tax_unit"))
        person = pd.DataFrame(
            {
                "person_id": range(4),
                "person_household_id": [1, 1, 2, 2],
                "person_tax_unit_id": [1, 1, 2, 3],
            }
        )
        household = pd.DataFrame({"household_id": [1, 2]})
        tax_unit = pd.DataFrame({"tax_unit_id": [1, 2, 3]})
        frame = Frame(
            {"person": person, "household": household, "tax_unit": tax_unit},
            schema,
            {
                "person": Weights(
                    values=np.array([1.0, 1.0, 2.0, 2.0]),
                    kind=WeightKind.CALIBRATED,
                ),
                "household": Weights(
                    values=np.array([5.0, 11.0]), kind=WeightKind.DESIGN
                ),
            },
        )
        resolved = frame.resolve_weights("tax_unit")
        np.testing.assert_array_equal(
            resolved.values, frame._effective_weights("tax_unit")
        )
        assert resolved.kind is WeightKind.CALIBRATED  # the value source, not design


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
        replacement = Weights(values=np.array([1.0, 2.0]), kind=WeightKind.CALIBRATED)
        updated = bundle.with_weights("household", replacement, mass=_FREE_MASS)
        assert bundle.weights_for("household").kind is WeightKind.DESIGN
        assert updated.weights_for("household").kind is WeightKind.CALIBRATED

    def test_adding_weights_to_unweighted_entity(self, make_bundle) -> None:
        bundle = make_bundle()
        person_weights = Weights(values=np.ones(5), kind=WeightKind.IMPORTANCE)
        updated = bundle.with_weights("person", person_weights, mass=_FREE_MASS)
        assert set(updated.weighted_entities) == {"person", "household"}

    def test_conserve_without_existing_weights_is_an_error(self, make_bundle) -> None:
        bundle = make_bundle()
        person_weights = Weights(values=np.ones(5), kind=WeightKind.DESIGN)
        with pytest.raises(ValueError, match="existing weights"):
            bundle.with_weights("person", person_weights, mass="conserve")

    def test_length_mismatch_is_an_error(self, make_bundle) -> None:
        bundle = make_bundle()
        wrong = Weights(values=np.array([1.0]), kind=WeightKind.CALIBRATED)
        with pytest.raises(ValueError, match="length 1"):
            bundle.with_weights("household", wrong, mass=_FREE_MASS)


class TestMetadata:
    def test_metadata_is_deeply_immutable_and_propagates(self, make_bundle) -> None:
        source = make_bundle()
        frame = Frame(
            {entity: source.table(entity) for entity in source.entities},
            source.schema,
            {entity: source.weights_for(entity) for entity in source.weighted_entities},
            source.strata,
            metadata={
                "assembly": {
                    "channels": ["asec", "acs"],
                    "counts": {"asec": 2, "acs": 1},
                }
            },
        )

        with pytest.raises(TypeError):
            frame.metadata["assembly"]["counts"]["asec"] = 99
        replacement = Weights(
            values=frame.weights_for("household").values,
            kind=WeightKind.IMPORTANCE,
        )
        updated = frame.with_weights("household", replacement, mass="conserve")
        assert updated.metadata == frame.metadata


class TestConcat:
    def test_column_set_mismatch_is_named(self, make_bundle) -> None:
        a = make_bundle()
        b = make_bundle(strata=pd.Series("other", index=range(5)))
        extra = b.person.copy()
        extra["bonus"] = 1.0
        b_extra = Frame(
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
        b = b.with_weights(
            "person",
            Weights(values=np.ones(5), kind=WeightKind.DESIGN),
            mass=_FREE_MASS,
        )
        with pytest.raises(ValueError, match="weighted entities differ"):
            a.concat(b)

    def test_schema_mismatch_is_an_error(self, make_bundle) -> None:
        a = make_bundle()
        other_schema = EntitySchema(group_entities=("family",))
        person = pd.DataFrame({"person_id": [0], "person_family_id": [1]})
        family = pd.DataFrame({"family_id": [1]})
        b = Frame(
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
