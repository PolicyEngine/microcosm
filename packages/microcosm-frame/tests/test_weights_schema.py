"""Weights and schema primitives."""

import copy
import pickle
from functools import partial

import numpy as np
import pytest

from microcosm.frame import (
    EntitySchema,
    LinkSpec,
    MassChange,
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
        replacement = weights.with_values(np.array([2.0]), kind=WeightKind.CALIBRATED)
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


def _trusted_pickle_roundtrip(value, *, protocol=pickle.DEFAULT_PROTOCOL):
    # Only bytes constructed from this test's invented value are loaded.
    return pickle.loads(pickle.dumps(value, protocol=protocol))


_COPY_OPERATIONS = (
    pytest.param(copy.copy, id="shallow"),
    pytest.param(copy.deepcopy, id="deep"),
    pytest.param(_trusted_pickle_roundtrip, id="trusted-pickle"),
)
_ROUNDTRIP_OPERATIONS = (
    *_COPY_OPERATIONS[:2],
    *(
        pytest.param(
            partial(_trusted_pickle_roundtrip, protocol=protocol),
            id=f"trusted-pickle-{protocol}",
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
    ),
)


@pytest.fixture
def cold_weights_namespace():
    # copyreg lazily caches slot names even on a dataclass without slots. Reset
    # only that cache to exercise a first copy, and restore its original state.
    missing = object()
    previous = vars(Weights).get("__slotnames__", missing)
    if previous is not missing:
        delattr(Weights, "__slotnames__")
    try:
        yield
    finally:
        if "__slotnames__" in vars(Weights):
            delattr(Weights, "__slotnames__")
        if previous is not missing:
            Weights.__slotnames__ = previous


class _PlainWeights(Weights):
    pass


class _StatefulWeights(Weights):
    def __init__(self, values, kind, *, label):
        super().__init__(values, kind)
        object.__setattr__(self, "label", label)


class _SlottedWeights(Weights):
    __slots__ = ("label",)

    def __init__(self, values, kind, *, label):
        super().__init__(values, kind)
        object.__setattr__(self, "label", label)


class _CustomReducedWeights(_StatefulWeights):
    def __reduce__(self):
        return _restore_custom_weights, (self.values, self.kind, self.label)


def _restore_custom_weights(values, kind, label):
    return _CustomReducedWeights(values, kind, label=label)


class TestWeightsCopy:
    @pytest.mark.parametrize("operation", _ROUNDTRIP_OPERATIONS)
    @pytest.mark.parametrize("kind", tuple(WeightKind))
    def test_preserves_exact_values_and_constructor_invariants(self, operation, kind):
        original = Weights(
            np.array([0.0, -0.0, np.nextafter(0.0, 1.0), 1.25, 2.0**53 + 2.0]),
            kind,
        )
        copied = operation(original)
        assert type(copied) is Weights
        assert copied is not original
        assert copied.kind is kind
        assert copied.values.dtype == np.float64
        np.testing.assert_array_equal(
            copied.values.view(np.uint64), original.values.view(np.uint64)
        )
        assert not copied.values.flags.writeable
        assert not np.shares_memory(copied.values, original.values)
        with pytest.raises(ValueError, match="read-only"):
            copied.values[0] = 4.0
        assert not original.values.flags.writeable

    @pytest.mark.parametrize("operation", _ROUNDTRIP_OPERATIONS)
    def test_copy_does_not_mutate_class_namespace(
        self, operation, cold_weights_namespace
    ):
        original = Weights(np.array([1.0]), WeightKind.DESIGN)
        before = {name: id(value) for name, value in vars(Weights).items()}
        operation(original)
        # On Python 3.14 this namespace is also captured by the deferred
        # annotation function, so a new __slotnames__ entry changes live seals.
        assert {name: id(value) for name, value in vars(Weights).items()} == before
        assert "__slotnames__" not in vars(Weights)

    @pytest.mark.parametrize("operation", _COPY_OPERATIONS)
    @pytest.mark.parametrize(
        ("field", "value", "error", "match"),
        [
            ("values", np.array([-1.0]), ValueError, "non-negative"),
            ("kind", "design", TypeError, "WeightKind"),
        ],
    )
    def test_reconstruction_revalidates_corrupted_objects(
        self, operation, field, value, error, match
    ):
        original = Weights(np.array([1.0]), WeightKind.DESIGN)
        object.__setattr__(original, field, value)
        with pytest.raises(error, match=match):
            operation(original)

    @pytest.mark.parametrize("operation", _COPY_OPERATIONS)
    @pytest.mark.parametrize(
        "subclass",
        [_PlainWeights, _StatefulWeights, _SlottedWeights, _CustomReducedWeights],
    )
    def test_existing_subclass_state_and_protocol_are_preserved(
        self, operation, subclass
    ):
        kwargs = {} if subclass is _PlainWeights else {"label": ["source"]}
        original = subclass(np.array([0.0, 2.5]), WeightKind.IMPORTANCE, **kwargs)
        copied = operation(original)
        assert type(copied) is subclass
        assert copied is not original
        assert copied.kind is original.kind
        np.testing.assert_array_equal(copied.values, original.values)
        if kwargs:
            assert copied.label == original.label
            if operation is copy.copy:
                assert copied.label is original.label
            else:
                assert copied.label is not original.label
        # Subclasses retain their own/default protocol, including extra state
        # and constructors that do not accept the base class's two arguments.
        if subclass is _CustomReducedWeights:
            assert not copied.values.flags.writeable


class TestMassChange:
    def test_factor_and_reason_are_recorded(self) -> None:
        change = MassChange(factor=2.0, reason="oversampled the rare stratum")
        assert change.factor == 2.0
        assert change.reason == "oversampled the rare stratum"

    def test_unspecified_factor_is_allowed(self) -> None:
        change = MassChange(factor=None, reason="importance resampling")
        assert change.factor is None

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"factor": 1.0, "reason": ""}, "reason", id="empty-reason"),
            pytest.param({"factor": 1.0, "reason": "   "}, "reason", id="blank-reason"),
            pytest.param({"factor": 0.0, "reason": "x"}, "positive", id="zero-factor"),
            pytest.param(
                {"factor": -1.0, "reason": "x"}, "positive", id="negative-factor"
            ),
            pytest.param(
                {"factor": float("inf"), "reason": "x"},
                "positive",
                id="infinite-factor",
            ),
        ],
    )
    def test_invalid_mass_change_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            MassChange(**kwargs)


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

    def test_entity_id_column_resolves_person_and_groups(self) -> None:
        schema = EntitySchema(group_entities=("household",))
        assert schema.entity_id_column("person") == "person_id"
        assert schema.entity_id_column("household") == "household_id"
        with pytest.raises(ValueError, match="firm"):
            schema.entity_id_column("firm")


class TestLinkSpec:
    def test_declares_an_association(self) -> None:
        jobs = LinkSpec(name="jobs", left_entity="person", right_entity="firm")
        schema = EntitySchema(group_entities=("household", "firm"), links=(jobs,))
        assert schema.links == (jobs,)

    @pytest.mark.parametrize("field_name", ["name", "left_entity", "right_entity"])
    def test_fields_must_be_non_empty(self, field_name: str) -> None:
        kwargs = {"name": "jobs", "left_entity": "person", "right_entity": "firm"}
        kwargs[field_name] = ""
        with pytest.raises(ValueError, match=field_name):
            LinkSpec(**kwargs)


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
