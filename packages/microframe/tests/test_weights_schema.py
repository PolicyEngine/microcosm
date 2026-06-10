"""Weights and schema primitives."""

import numpy as np
import pytest
from microframe import (
    EntitySchema,
    VariableMetadata,
    WeightKind,
    Weights,
    assert_kind_transition,
)


class TestWeights:
    def test_total_and_len(self) -> None:
        weights = Weights(values=np.array([1.5, 2.5]), kind=WeightKind.DESIGN)
        assert weights.total == 4.0
        assert len(weights) == 2

    def test_values_are_float64_copies(self) -> None:
        source = np.array([1, 2, 3], dtype="int64")
        weights = Weights(values=source, kind=WeightKind.DESIGN)
        assert weights.values.dtype == np.float64
        source[0] = 99  # the stored vector is decoupled from the caller's
        assert weights.values[0] == 1.0

    def test_with_values_revalidates(self) -> None:
        weights = Weights(values=np.array([1.0]), kind=WeightKind.DESIGN)
        replacement = weights.with_values(
            np.array([2.0]), kind=WeightKind.CALIBRATED
        )
        assert replacement.kind is WeightKind.CALIBRATED
        assert replacement.total == 2.0
        with pytest.raises(ValueError, match="non-negative"):
            weights.with_values(np.array([-1.0]), kind=WeightKind.DESIGN)

    def test_kind_must_be_weight_kind(self) -> None:
        with pytest.raises(TypeError, match="WeightKind"):
            Weights(values=np.array([1.0]), kind="design")

    def test_multidimensional_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="one-dimensional"):
            Weights(values=np.ones((2, 2)), kind=WeightKind.DESIGN)

    def test_mass_conservation_helper_carries_both_totals(self) -> None:
        a = Weights(values=np.array([10.0]), kind=WeightKind.DESIGN)
        b = Weights(values=np.array([12.0]), kind=WeightKind.CALIBRATED)
        with pytest.raises(ValueError) as excinfo:
            a.assert_mass_conserved(b)
        assert "10.0" in str(excinfo.value)
        assert "12.0" in str(excinfo.value)
        a.assert_mass_conserved(
            Weights(values=np.array([10.0 + 1e-12]), kind=WeightKind.CALIBRATED)
        )

    def test_kind_transition_authority(self) -> None:
        assert_kind_transition(WeightKind.DESIGN, WeightKind.DESIGN)
        assert_kind_transition(WeightKind.DESIGN, WeightKind.CALIBRATED)
        assert_kind_transition(WeightKind.IMPORTANCE, WeightKind.CALIBRATED)
        with pytest.raises(ValueError, match="backward"):
            assert_kind_transition(WeightKind.CALIBRATED, WeightKind.IMPORTANCE)


class TestEntitySchema:
    def test_linkage_column_conventions(self) -> None:
        schema = EntitySchema(group_entities=("household", "tax_unit"))
        assert schema.person_entity == "person"
        assert schema.entities == ("person", "household", "tax_unit")
        assert schema.person_id_column == "person_id"
        assert schema.membership_column("tax_unit") == "person_tax_unit_id"
        assert schema.id_column("tax_unit") == "tax_unit_id"

    def test_unknown_group_is_named(self) -> None:
        schema = EntitySchema(group_entities=("household",))
        with pytest.raises(ValueError, match="benunit"):
            schema.membership_column("benunit")
        with pytest.raises(ValueError, match="benunit"):
            schema.id_column("benunit")


class TestVariableMetadata:
    def test_valid_metadata(self) -> None:
        metadata = VariableMetadata(
            name="employment_income", entity="person", dtype="float", period="year"
        )
        assert metadata.dtype == "float"
        assert metadata.period == "year"

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"dtype": "double"}, id="bad-dtype"),
            pytest.param({"period": "week"}, id="bad-period"),
            pytest.param({"name": ""}, id="empty-name"),
            pytest.param({"entity": ""}, id="empty-entity"),
        ],
    )
    def test_invalid_metadata_rejected(self, kwargs: dict) -> None:
        base = {
            "name": "age",
            "entity": "person",
            "dtype": "int",
            "period": "point",
        }
        with pytest.raises(ValueError):
            VariableMetadata(**{**base, **kwargs})
