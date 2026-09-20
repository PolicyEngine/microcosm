"""Explicit formula metadata without country imports or consumer qualification."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_fiscal_refresh_builder import (
    _install_multi_reform_fakes,
    _load_builder_module,
    _multi_reform_frame,
)

from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine


class RecordingMetadata:
    def __init__(self, formula_columns=()):
        self.formula_columns = set(formula_columns)
        self.calls = []

    def _engine_computed_columns(self, tables, *, period):
        self.calls.append((tables, period))
        return self.formula_columns


@pytest.fixture
def builder(monkeypatch):
    original_import = builtins.__import__

    def no_country_import(name, *args, **kwargs):
        if (
            name == "policyengine_us" or name.startswith("policyengine_us.")
        ) and not isinstance(sys.modules.get(name), SimpleNamespace):
            raise AssertionError("test must not import the country engine")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_country_import)
    return _load_builder_module()


def test_default_keeps_one_cached_import_free_index(builder, monkeypatch, small_frame):
    index = RecordingMetadata()
    constructions = []

    def make_index():
        constructions.append(True)
        return index

    monkeypatch.setattr(builder, "PolicyEngineUSVariableMetadataIndex", make_index)
    builder._assert_no_formula_owned_columns(small_frame)
    builder._assert_no_formula_owned_columns(small_frame)
    assert len(constructions) == 1
    assert len(index.calls) == 2
    assert builder._FORMULA_OWNED_GATE_ADAPTER is index


def test_explicit_provider_does_not_replace_or_construct_default(
    builder, monkeypatch, small_frame
):
    provider = RecordingMetadata()

    def forbidden_index():
        raise AssertionError("explicit metadata must not load the index")

    monkeypatch.setattr(builder, "PolicyEngineUSVariableMetadataIndex", forbidden_index)
    builder._assert_no_formula_owned_columns(small_frame, formula_metadata=provider)
    assert builder._FORMULA_OWNED_GATE_ADAPTER is None
    default = RecordingMetadata({"income"})
    monkeypatch.setattr(builder, "_FORMULA_OWNED_GATE_ADAPTER", default)
    builder._assert_no_formula_owned_columns(small_frame, formula_metadata=provider)
    with pytest.raises(ValueError, match="Formula-owned.*income"):
        builder._assert_no_formula_owned_columns(small_frame)
    assert builder._FORMULA_OWNED_GATE_ADAPTER is default


def test_falsey_provider_is_used_with_exact_tables_and_period(builder):
    class FalseyMetadata(RecordingMetadata):
        def __bool__(self):
            return False

    frame = _multi_reform_frame(builder)
    provider = FalseyMetadata()
    builder._assert_no_formula_owned_columns(frame, formula_metadata=provider)
    tables, period = provider.calls.pop()
    assert period == builder.PERIOD
    assert set(tables) == set(frame.entities)
    for entity in frame.entities:
        pd.testing.assert_frame_equal(tables[entity], frame.table(entity))


def test_provider_failure_never_falls_back(builder, monkeypatch, small_frame):
    class BrokenMetadata:
        def _engine_computed_columns(self, tables, *, period):
            raise RuntimeError("invented metadata unavailable")

    default = RecordingMetadata()
    monkeypatch.setattr(builder, "_FORMULA_OWNED_GATE_ADAPTER", default)
    with pytest.raises(RuntimeError, match="invented metadata unavailable"):
        builder._assert_no_formula_owned_columns(
            small_frame, formula_metadata=BrokenMetadata()
        )
    assert not default.calls


@pytest.mark.parametrize(
    "helper",
    ["dataset", "aca", "reform", "materialize", "dense", "sparse", "validation"],
)
def test_supplied_provider_refuses_before_engine_or_weight_work(
    builder, small_frame, helper
):
    metadata = RecordingMetadata({"income"})
    before = {
        entity: small_frame.table(entity).copy(deep=True)
        for entity in small_frame.entities
    }
    weights = small_frame.weights_for("household").values.copy()
    calls = {
        "dataset": lambda: builder._dataset_from_frame(
            small_frame, formula_metadata=metadata
        ),
        "aca": lambda: builder._aca_source_tax_unit_table_batched(
            small_frame,
            {},
            microsimulation_cls=None,
            maximum_microsim_batch_size=None,
            formula_metadata=metadata,
        ),
        "reform": lambda: builder._reform_household_income_tax(
            base_frame=small_frame,
            reform_spec=None,
            system=None,
            microsimulation_cls=None,
            n_households=2,
            batch_size=None,
            formula_metadata=metadata,
        ),
        "materialize": lambda: builder._materialize_target_frame(
            small_frame, (), formula_metadata=metadata
        ),
        "dense": lambda: builder._with_calibrated_weights(
            small_frame, np.asarray([3.0, 4.0]), formula_metadata=metadata
        ),
        "sparse": lambda: builder._with_l0_refit_weights(
            small_frame, None, formula_metadata=metadata
        ),
        "validation": lambda: (
            builder._batched_reform_validation_simulate_factory_from_frame(
                small_frame, formula_metadata=metadata
            )
        ),
    }
    with pytest.raises(ValueError, match="Formula-owned.*income"):
        calls[helper]()
    assert len(metadata.calls) == 1
    for entity, expected in before.items():
        pd.testing.assert_frame_equal(small_frame.table(entity), expected)
    np.testing.assert_array_equal(small_frame.weights_for("household").values, weights)


@pytest.mark.parametrize("sparse", [False, True])
def test_explicit_metadata_preserves_existing_weight_attachment(
    builder, small_frame, sparse
):
    provider = RecordingMetadata()
    if sparse:
        result = SimpleNamespace(
            weight_entity="household",
            selected_entity_ids=np.asarray([2]),
            weights=np.asarray([3333.0]),
        )
        exported = builder._with_l0_refit_weights(
            small_frame, result, formula_metadata=provider
        )
        assert exported.table("person")["person_id"].tolist() == [2, 3]
        expected = [3333.0]
    else:
        exported = builder._with_calibrated_weights(
            small_frame, np.asarray([900.0, 2100.0]), formula_metadata=provider
        )
        for entity in small_frame.entities:
            pd.testing.assert_frame_equal(
                exported.table(entity), small_frame.table(entity)
            )
        expected = [900.0, 2100.0]
    assert len(provider.calls) == 1
    np.testing.assert_array_equal(exported.weights_for("household").values, expected)
    np.testing.assert_array_equal(
        small_frame.weights_for("household").values, [1000, 2000]
    )


@pytest.mark.parametrize("reject", [False, True])
def test_checkpoint_hit_checks_supplied_metadata_before_read(
    builder, monkeypatch, small_frame, tmp_path, reject
):
    events = []

    class Metadata(RecordingMetadata):
        def _engine_computed_columns(self, tables, *, period):
            events.append("check")
            return super()._engine_computed_columns(tables, period=period)

    provider = Metadata({"income"} if reject else ())
    cached = (small_frame, object(), {"cached": True})

    def read(*args, **kwargs):
        events.append("read")
        return cached

    monkeypatch.setattr(builder, "_read_target_frame_checkpoint", read)
    kwargs = dict(
        target_frame_checkpoint_path=tmp_path / "invented",
        target_frame_checkpoint_identity={"invented": True},
        formula_metadata=provider,
    )
    if reject:
        with pytest.raises(ValueError, match="Formula-owned.*income"):
            builder._load_or_materialize_target_frame(small_frame, (), **kwargs)
        assert events == ["check"]
    else:
        assert (
            builder._load_or_materialize_target_frame(small_frame, (), **kwargs)
            is cached
        )
        assert events == ["check", "read"]


def test_materializer_receives_same_provider(builder, monkeypatch, small_frame):
    provider = RecordingMetadata()
    seen = []

    def materialize(frame, specs, **kwargs):
        seen.append(kwargs["formula_metadata"])
        return frame, object(), {}

    monkeypatch.setattr(builder, "_materialize_target_frame", materialize)
    builder._load_or_materialize_target_frame(
        small_frame, (), formula_metadata=provider
    )
    assert seen == [provider]


def test_materialization_and_reforms_forward_provider_to_each_batch(
    builder, monkeypatch
):
    """Exercise the maintained materializer with only invented model outputs."""
    frame = _multi_reform_frame(builder)
    provider = RecordingMetadata()
    real_gate = builder._assert_no_formula_owned_columns
    reform_calls = []
    _, targets = _install_multi_reform_fakes(
        builder,
        monkeypatch,
        reforms=(("jct_invented", "invented_credit"),),
        reform_income_tax_by_id={"invented_credit": {10: 90.0, 20: 25.0, 30: 40.0}},
        reform_sim_calls=reform_calls,
    )
    fake_dataset = builder._dataset_from_frame
    datasets = []

    def recording_dataset(frame_arg, *, formula_metadata, **kwargs):
        datasets.append((frame_arg.n("household"), formula_metadata))
        return fake_dataset(frame_arg, **kwargs)

    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", real_gate)
    monkeypatch.setattr(builder, "_dataset_from_frame", recording_dataset)
    target_frame, _, _ = builder._materialize_target_frame(
        frame, targets, maximum_microsim_batch_size=1, formula_metadata=provider
    )
    assert datasets == [(2, provider), (1, provider), (1, provider)]
    assert reform_calls == ["invented_credit"]
    assert len(provider.calls) == 2
    assert builder._FORMULA_OWNED_GATE_ADAPTER is None
    np.testing.assert_array_equal(
        target_frame.table("household")["jct_invented"], [-15.0, -30.0]
    )


def test_validation_factory_forwards_provider_to_maintained_dataset_helper(
    builder, monkeypatch
):
    frame = _multi_reform_frame(builder)
    provider = RecordingMetadata()
    seen = []

    class Simulation:
        def __init__(self, *, dataset):
            pass

        def calculate(self, measure, period):
            return np.asarray([1.0])

    def dataset(
        frame_arg,
        *,
        assert_no_formula_owned_columns,
        formula_metadata,
        dataset_cls=None,
    ):
        seen.append((frame_arg.n("household"), formula_metadata))
        assert (
            not assert_no_formula_owned_columns
        )  # Whole-frame check precedes batching.
        return object()

    monkeypatch.setattr(builder, "_dataset_from_frame", dataset)
    factory = builder._batched_reform_validation_simulate_factory_from_frame(
        frame,
        maximum_microsim_batch_size=1,
        microsimulation_cls=Simulation,
        formula_metadata=provider,
    )
    assert factory(None).calculate("invented_measure", 2024).sum() == 2.0
    assert seen == [(1, provider), (1, provider)]
    assert len(provider.calls) == 1


def test_aca_forwards_provider_after_whole_frame_check(builder, monkeypatch):
    frame = _multi_reform_frame(builder)
    provider = RecordingMetadata()
    seen = []

    class ReachedDatasetError(Exception):
        pass

    def dataset(
        frame_arg,
        *,
        assert_no_formula_owned_columns,
        formula_metadata,
        dataset_cls=None,
    ):
        seen.append((frame_arg.n("household"), formula_metadata))
        assert not assert_no_formula_owned_columns
        raise ReachedDatasetError

    monkeypatch.setattr(builder, "_dataset_from_frame", dataset)
    with pytest.raises(ReachedDatasetError):
        builder._aca_source_tax_unit_table_batched(
            frame,
            {},
            microsimulation_cls=None,
            maximum_microsim_batch_size=1,
            formula_metadata=provider,
        )
    assert seen == [(1, provider)]
    assert len(provider.calls) == 1


def test_dataset_with_explicit_provider_preserves_selected_cells(builder, monkeypatch):
    frame = _multi_reform_frame(builder)
    provider = RecordingMetadata()
    monkeypatch.setitem(
        sys.modules,
        "policyengine_us.data",
        SimpleNamespace(USSingleYearDataset=lambda **kwargs: kwargs),
    )
    dataset = builder._dataset_from_frame(frame, formula_metadata=provider)
    assert dataset["time_period"] == builder.PERIOD
    for entity in frame.entities:
        original = frame.table(entity)
        pd.testing.assert_frame_equal(dataset[entity][original.columns], original)
        assert dataset[entity] is not original
    np.testing.assert_array_equal(
        dataset["household"]["household_weight"], frame.weights_for("household").values
    )
    assert "household_weight" not in frame.table("household")
    assert len(provider.calls) == 1


def test_default_checkpoint_path_keeps_existing_behavior(
    builder, monkeypatch, small_frame, tmp_path
):
    def forbidden_gate(*args, **kwargs):
        raise AssertionError("legacy checkpoint hit must retain its existing behavior")

    cached = (small_frame, object(), {})
    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", forbidden_gate)
    monkeypatch.setattr(
        builder, "_read_target_frame_checkpoint", lambda *a, **k: cached
    )
    assert (
        builder._load_or_materialize_target_frame(
            small_frame,
            (),
            target_frame_checkpoint_path=tmp_path / "invented",
            target_frame_checkpoint_identity={"invented": True},
        )
        is cached
    )


def test_maintained_live_adapter_formula_method_accepts_invented_system(
    builder, small_frame
):
    """Run the real method, overriding only its two engine metadata sources."""

    class Variable:
        adds = None
        subtracts = None

        def __init__(self, formula_period):
            self.formula_period = formula_period

        def get_formula(self, period):
            return object() if period == self.formula_period else None

    class InventedSystemAdapter(PolicyEngineUSEngine):
        def _tax_benefit_system(self):
            return SimpleNamespace(
                variables={
                    "income": Variable("2024"),
                    "source_fallback": Variable("2024"),
                    "later_formula": Variable("2025"),
                }
            )

        def _dataset_source_inputs(self):
            return frozenset({"source_fallback"})

    provider = InventedSystemAdapter()
    small_frame.table("person")["source_fallback"] = 1.0
    small_frame.table("person")["later_formula"] = 2.0
    with pytest.raises(ValueError, match=r"\['income'\]"):
        builder._assert_no_formula_owned_columns(small_frame, formula_metadata=provider)
    # The source fallback formula is an allowed source primitive; the 2025
    # formula is not owned in the builder's 2024 period. This is metadata
    # mechanics only, not an assertion that these invented columns are valid.
    small_frame.table("person").drop(columns="income", inplace=True)
    builder._assert_no_formula_owned_columns(small_frame, formula_metadata=provider)
