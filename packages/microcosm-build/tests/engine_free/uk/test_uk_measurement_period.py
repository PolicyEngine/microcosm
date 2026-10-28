"""Plain-column values are measured at their binding's period (microcosm#1324).

The frame stores its inputs at its own period (2024); the calibration solves
at 2025. A binding's value variable that the frame stores is computed by the
measure provider at the measurement period, like every engine-computed
variable, unless the binding declares the frame's own period. Filters,
conditions and bands keep their stored reads. A provider binding reads its
columns as the frame holds them, so a period it declares must be one each of
its value reads holds.
"""

from __future__ import annotations

import json
import sys
from importlib import resources as importlib_resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.target_materialization import (
    ENGINE_PERIOD_COLUMN_PREFIX,
    MeasurementPeriodError,
    _refuse_ignored_measurement_period,
    default_provider_registry,
    drop_injected_measure_inputs,
    inject_measure_inputs,
    materialize_target_bindings,
    measure_input_column,
    resolve_target_measures,
)
from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.hmrc_income import (
    LOSS_RELIEF_COLUMN,
    self_employment_income_after_loss_relief,
)
from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter
from microcosm.build.uk_runtime.measure_simulation import (
    UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF,
    UK_TOTAL_INCOME_AFTER_LOSS_RELIEF,
    UKMeasureResolver,
    compute_uk_measure_input,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

STORED = np.array([1_000.0, 0.0, 500.0])
#: The engine's 2025 values. The middle person has none stored but a positive
#: engine value, so a filter that read the engine would let them in.
ENGINE_2025 = np.array([1_053.0, 7.0, 526.5])


class PeriodFrameStub:
    """A frame stub with the period label and the surface the adapter reads."""

    entities = ("person", "benunit", "household")
    links = ()

    def __init__(self, time_period: str = "2024"):
        self.metadata = {"time_period": time_period}
        self._tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "person_benunit_id": [10, 10, 20],
                    "person_household_id": [100, 100, 200],
                    "employment_income": STORED.copy(),
                    "self_employment_income": [10_000.0, 2_000.0, 5_000.0],
                    LOSS_RELIEF_COLUMN: [3_000.0, 5_000.0, 0.0],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
            "household": pd.DataFrame({"household_id": [100, 200]}),
        }

    def table(self, entity):
        return self._tables[entity]

    def link(self, name):
        raise KeyError(name)


class SimulationStub:
    def __init__(self, values):
        self.values = values
        self.tax_benefit_system = SimpleNamespace(
            variables={
                name: SimpleNamespace(
                    entity=SimpleNamespace(key=entity), value_type=float
                )
                for name, (entity, _) in values.items()
            }
        )
        self.calls = []

    def calculate(self, variable, year, map_to=None):
        self.calls.append((variable, year))
        return self.values[variable][1]


def _resolver(monkeypatch, tmp_path, frame, sim, year=2025):
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="2.134.4", Microsimulation=lambda **kw: sim),
    )
    return UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=year,
        frame=frame,
    )


def _registry(bindings, *, fact_period=None, entity="person"):
    metadata = {} if fact_period is None else {"ledger_fact_period": fact_period}
    return (
        TargetRegistry(
            [
                TargetSpec(
                    name=name,
                    entity=entity,
                    measure=f"test_{name}",
                    value=1,
                    source="test",
                    metadata={"contract_target_id": name, **metadata},
                )
                for name in bindings
            ],
            country="uk",
        ),
        {name: {"bindings": {"policyengine": b}} for name, b in bindings.items()},
    )


def _resolve_and_materialize(monkeypatch, tmp_path, bindings, *, period=2025):
    frame = PeriodFrameStub()
    sim = SimulationStub({"employment_income": ("person", ENGINE_2025)})
    resolver = _resolver(monkeypatch, tmp_path, frame, sim)
    registry, contract = _registry(bindings)
    resolver.contract_targets = contract
    resolution = resolve_target_measures(
        lambda: UKFrameTargetAdapter(frame), registry, resolver, period=period
    )
    adapter = UKFrameTargetAdapter(frame)
    original_columns = {name: set(t.columns) for name, t in adapter.tables.items()}
    inject_measure_inputs(adapter, resolution.measure_inputs)
    materialized = materialize_target_bindings(
        adapter, registry, contract, period=period
    )
    return frame, sim, adapter, materialized, resolution, original_columns


def test_a_stored_value_is_measured_at_the_calibration_period(monkeypatch, tmp_path):
    positive = [{"variable": "employment_income", "operator": ">", "value": 0}]
    frame, sim, adapter, materialized, resolution, original = _resolve_and_materialize(
        monkeypatch,
        tmp_path,
        {
            "at_calibration": {
                "value_variable": "employment_income",
                "filters": positive,
            },
            "at_frame": {
                "value_variable": "employment_income",
                "measurement_period": 2024,
            },
        },
    )

    assert not materialized.skipped
    person = adapter.tables["person"]
    # The value is the engine's 2025 value; who is counted is the stored filter.
    np.testing.assert_allclose(person["test_at_calibration"], [1_053.0, 0.0, 526.5])
    # A binding that declares the frame's period reads the stored column.
    np.testing.assert_allclose(person["test_at_frame"], STORED)
    assert {call for call in sim.calls} == {("employment_income", 2025)}
    assert resolution.receipt["attached"]["person.employment_income"] == "native"
    # The stored column is never overwritten; the engine value has its own name.
    np.testing.assert_allclose(person["employment_income"], STORED)
    assert ENGINE_PERIOD_COLUMN_PREFIX + "employment_income" in person
    np.testing.assert_allclose(frame.table("person")["employment_income"], STORED)

    drop_injected_measure_inputs(adapter, resolution.measure_inputs, original)
    assert ENGINE_PERIOD_COLUMN_PREFIX + "employment_income" not in person
    np.testing.assert_allclose(person["employment_income"], STORED)


def test_a_value_expression_term_is_measured_at_the_calibration_period(
    monkeypatch, tmp_path
):
    _, _, adapter, materialized, _, _ = _resolve_and_materialize(
        monkeypatch,
        tmp_path,
        {"sum": {"value_expression": "employment_income + employment_income"}},
    )

    assert not materialized.skipped
    np.testing.assert_allclose(adapter.tables["person"]["test_sum"], 2 * ENGINE_2025)


def test_a_declared_period_the_route_cannot_honour_refuses(monkeypatch, tmp_path):
    with pytest.raises(MeasurementPeriodError, match="2023"):
        _resolve_and_materialize(
            monkeypatch,
            tmp_path,
            {
                "stale": {
                    "value_variable": "employment_income",
                    "measurement_period": 2023,
                }
            },
        )


@pytest.mark.parametrize(
    "binding",
    [
        {"value_variable": "income_tax", "measurement_period": 2024},
        {
            "value_expression": "employment_income + income_tax",
            "measurement_period": 2024,
        },
    ],
)
def test_the_frame_period_on_an_engine_value_refuses(binding):
    # The frame stores no income_tax, so the engine would compute it at the
    # calibration period and the row would be read there under 2024.
    registry, contract = _registry({"engine_value": binding})
    adapter = UKFrameTargetAdapter(PeriodFrameStub())

    with pytest.raises(MeasurementPeriodError, match="person.income_tax"):
        materialize_target_bindings(adapter, registry, contract, period=2025)


def test_the_resolver_refuses_the_frame_period_on_an_engine_value(
    monkeypatch, tmp_path
):
    with pytest.raises(MeasurementPeriodError, match="person.income_tax"):
        _resolve_and_materialize(
            monkeypatch,
            tmp_path,
            {
                "engine_value": {
                    "value_variable": "income_tax",
                    "measurement_period": 2024,
                }
            },
        )


def test_a_cgt_measure_dated_by_its_name_holds_its_own_year():
    adapter = UKFrameTargetAdapter(PeriodFrameStub())

    assert adapter.value_period("person", "employment_income") == "2024"
    assert adapter.value_period("person", "cgt_2024_tax") == "2024"
    assert adapter.value_period("person", "cgt_2025_gains") == "2025"
    assert adapter.value_period("person", "income_tax") is None

    # The guard lets the dated measure through: the provider computes it at
    # the year in its name. With no provider here, the read reports it missing.
    registry, contract = _registry(
        {"dated": {"value_variable": "cgt_2024_tax", "measurement_period": 2024}}
    )
    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)
    assert [skip.name for skip in materialized.skipped] == ["dated"]

    registry, contract = _registry(
        {"misdated": {"value_variable": "cgt_2025_tax", "measurement_period": 2024}}
    )
    with pytest.raises(MeasurementPeriodError, match="person.cgt_2025_tax"):
        materialize_target_bindings(
            UKFrameTargetAdapter(PeriodFrameStub()), registry, contract, period=2025
        )


def test_a_run_at_the_frame_period_declares_any_value_without_refusal():
    # Calibrating at the frame's own period, the engine computes every value
    # at that period, so the declaration is honoured whatever the value.
    registry, contract = _registry(
        {"engine_value": {"value_variable": "income_tax", "measurement_period": 2024}}
    )
    adapter = UKFrameTargetAdapter(PeriodFrameStub())

    materialized = materialize_target_bindings(adapter, registry, contract, period=2024)

    assert [skip.name for skip in materialized.skipped] == ["engine_value"]


def test_an_entity_count_holds_no_period_at_the_frames_period():
    # A count is ones whatever the period, so the frame's period is honoured
    # for it as for a stored value.
    registry, contract = _registry(
        {"count": {"value_variable": "person_count", "measurement_period": 2024}}
    )
    adapter = UKFrameTargetAdapter(PeriodFrameStub())

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert not materialized.skipped
    np.testing.assert_allclose(adapter.tables["person"]["test_count"], [1.0] * 3)


# Provider rows ---------------------------------------------------------------

#: A provider row reading an engine value under a declared 2023: as the value,
#: the flag, a member reduction, ``count_of`` and the gated variable.
_UNDATED_PROVIDERS = {
    "value": {
        "kind": "baseline_flag_crosstab",
        "from_entity": "person",
        "affected_flag_variable": "is_SP_age",
        "value_variable": "income_tax",
    },
    "flag": {
        "kind": "baseline_flag_crosstab",
        "from_entity": "person",
        "affected_flag_variable": "income_tax",
        "value_variable": "person_count",
    },
    "count_of": {
        "kind": "baseline_flag_crosstab",
        "from_entity": "person",
        "affected_flag_variable": "is_SP_age",
        "count_of": "income_tax",
    },
    "reduction": {
        "kind": "baseline_flag_crosstab",
        "affected_flag_variable": "is_SP_age",
        "value_variable": "person_count",
        "value_reduction": {"entity": "person", "variable": "income_tax"},
    },
    "gated": {
        "kind": "parameter_gated_threshold",
        "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
        "gated_variable": "income_tax",
        "value_variable": "person_count",
    },
}


def _frame(metadata=True):
    frame = PeriodFrameStub()
    if not metadata:
        frame.metadata = {}
    return frame


@pytest.mark.parametrize("metadata", [True, False])
@pytest.mark.parametrize("case", sorted(_UNDATED_PROVIDERS))
def test_a_provider_refuses_an_engine_value_under_another_period(case, metadata):
    """A provider reads its columns as the frame holds them, whatever period it
    is passed, so an engine value (injected at the calibration period) under a
    declared 2023 is refused before the provider runs."""

    adapter = UKFrameTargetAdapter(_frame(metadata))
    adapter.set_column("person", "is_SP_age", [True, False, True])
    adapter.set_column("household", "is_SP_age", [True, False])
    adapter.set_column("person", "income_tax", [1.0, 2.0, 3.0])
    registry, contract = _registry(
        {
            "provider": {
                **_UNDATED_PROVIDERS[case],
                "measurement_period": 2023,
                "require_matching_fact_period": True,
            }
        },
        fact_period="2023",
    )
    with pytest.raises(MeasurementPeriodError, match="income_tax"):
        materialize_target_bindings(adapter, registry, contract, period=2025)


@pytest.mark.parametrize("metadata", [True, False])
def test_a_provider_reads_values_held_at_its_period(metadata):
    """The CGT providers' shape: a gate on a measure dated to the declared
    year, counted or summed; and a stored value on the declared frame period."""

    adapter = UKFrameTargetAdapter(_frame(metadata))
    adapter.set_column("person", "cgt_2024_gains", [0.0, 5_000.0, 50_000.0])
    gate = {
        "kind": "parameter_gated_threshold",
        "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
        "gated_variable": "cgt_2024_gains",
        "from_entity": "person",
        "measurement_period": 2024,
        "require_matching_fact_period": True,
    }
    registry, contract = _registry(
        {
            "count": {**gate, "value_variable": "person_count"},
            "amount": {**gate, "value_variable": "cgt_2024_gains"},
            "stored": {**gate, "value_variable": "employment_income"},
        },
        fact_period="2024",
    )

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert not materialized.skipped
    people = adapter.tables["person"]
    np.testing.assert_allclose(people["test_count"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(people["test_amount"], [0.0, 5_000.0, 50_000.0])
    np.testing.assert_allclose(people["test_stored"], [0.0, 0.0, 500.0])


def test_a_provider_on_a_frame_of_another_period_refuses_its_stored_values():
    """A stored column holds the frame's period: a provider declaring the
    calibration period on a 2024 frame would read 2024's values."""

    adapter = UKFrameTargetAdapter(PeriodFrameStub())
    registry, contract = _registry(
        {
            "stored": {
                "kind": "baseline_flag_crosstab",
                "from_entity": "person",
                "affected_flag_variable": "employment_income",
                "value_variable": "person_count",
                "measurement_period": 2025,
            }
        }
    )
    with pytest.raises(MeasurementPeriodError, match="held at 2024"):
        materialize_target_bindings(adapter, registry, contract, period=2025)


@pytest.mark.parametrize(
    "kind", ["input_substitution_counterfactual", "caller_defined"]
)
def test_a_provider_whose_reads_are_unknown_declares_only_the_calibration_period(
    kind,
):
    adapter = UKFrameTargetAdapter(PeriodFrameStub())
    binding = {"kind": kind, "metric_name": "x/delta", "from_entity": "person"}
    registry, contract = _registry({"other": {**binding, "measurement_period": 2023}})
    with pytest.raises(MeasurementPeriodError, match="reads are not known here"):
        materialize_target_bindings(adapter, registry, contract, period=2025)

    adapter.set_column("person", "counterfactual_delta__x_delta", [1.0, 2.0, 3.0])
    registry, contract = _registry({"other": {**binding, "measurement_period": 2025}})
    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)
    if kind == "caller_defined":
        assert "unsupported binding kind" in materialized.skipped[0].reason
    else:
        assert not materialized.skipped


@pytest.mark.parametrize("metadata", [True, False])
@pytest.mark.parametrize("period", [2024, 2025, 2026])
def test_every_packaged_provider_row_declares_a_period_its_reads_hold(metadata, period):
    # The CGT rows read measures dated by their name (cgt_2024_*), which the
    # resolver refuses to find stored, and counts.
    contract = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    dated = {}
    for target in contract["targets"]:
        binding = (target.get("bindings") or {}).get("policyengine") or {}
        if binding.get("kind") and "measurement_period" in binding:
            dated[target["target_id"]] = binding
    assert dated
    adapter = UKFrameTargetAdapter(_frame(metadata))
    for name, binding in dated.items():
        spec = SimpleNamespace(name=name, entity="person")
        _refuse_ignored_measurement_period(spec, binding, adapter, period)


@pytest.mark.parametrize("metadata", [True, False])
def test_a_member_reduction_reads_a_count_as_the_frame_stores_it(metadata):
    """A count read through the adapter's column is ones, but a member
    reduction sums the member table as stored: a stored person_count holds
    the frame's period, or is read as it is on a frame without one."""

    frame = _frame(metadata)
    frame.table("person")["person_count"] = [7.0, 9.0, 4.0]
    binding = {
        "kind": "baseline_flag_crosstab",
        "affected_flag_variable": "household_count",
        "value_variable": "person_count",
        "value_reduction": {"entity": "person", "variable": "person_count"},
    }
    years = (2024,) if metadata else (2023, 2024)
    for declared in years:
        adapter = UKFrameTargetAdapter(frame)
        registry, contract = _registry(
            {"members": {**binding, "measurement_period": declared}},
            entity="household",
        )
        materialized = materialize_target_bindings(
            adapter, registry, contract, period=2025
        )
        assert not materialized.skipped
        np.testing.assert_allclose(
            adapter.tables["household"]["test_members"], [16.0, 4.0]
        )
    if metadata:
        registry, contract = _registry(
            {"members": {**binding, "measurement_period": 2023}}, entity="household"
        )
        with pytest.raises(MeasurementPeriodError, match="person.person_count"):
            materialize_target_bindings(
                UKFrameTargetAdapter(frame), registry, contract, period=2025
            )


#: The values a generated provider row may read. The fixture below stores the
#: first three on the frame (the counts with values other than one, as a
#: member reduction would read them) and injects the rest.
_PROVIDER_READS = (
    "employment_income",
    "person_count",
    "household_count",
    "income_tax",  # engine-computed: injected at the calibration period
    "cgt_2023_gains",  # dated by name
    "cgt_2024_gains",
    "cgt_2025_gains",
)
_STORED_READS = frozenset(_PROVIDER_READS[:3])


def _held(variable, *, through_column, metadata, calibration):
    """The period the fixture holds one read at; None for any period."""

    if through_column and variable.endswith("_count"):
        return None  # the adapter's column reads a count as ones
    if variable in _STORED_READS:
        return "2024" if metadata else None
    if variable.startswith("cgt_"):
        return variable.split("_")[1]
    return str(calibration)


def _provider_adapter(metadata, adapter_class=UKFrameTargetAdapter):
    frame = _frame(metadata)
    frame.table("person")["person_count"] = [7.0, 9.0, 4.0]
    frame.table("person")["household_count"] = [2.0, 2.0, 1.0]
    frame.table("household")["household_count"] = [3.0, 5.0]
    adapter = adapter_class(frame)
    for variable in _PROVIDER_READS[3:]:
        adapter.set_column("person", variable, [0.0, 5_000.0, 50_000.0])
    adapter.set_column("household", "income_tax", [1.0, 0.0])
    return adapter


class _ReadRecorder(UKFrameTargetAdapter):
    """Records each column a provider reads, and through which path."""

    def __init__(self, frame):
        super().__init__(frame)
        self.reads = []

    def column(self, entity, variable):
        self.reads.append((entity, variable, True))
        return super().column(entity, variable)

    def entity_reduction(self, reduction):
        entity = str(reduction.get("entity") or "person")
        self.reads.append((entity, str(reduction["variable"]), False))
        return super().entity_reduction(reduction)


@st.composite
def _provider_rows(draw):
    read = st.sampled_from(_PROVIDER_READS)
    shape = draw(st.sampled_from(("gated", "value", "count_of", "reduction")))
    if shape == "gated":
        binding = {
            "kind": "parameter_gated_threshold",
            "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
            "from_entity": "person",
            "gated_variable": draw(read),
        }
        value = draw(st.none() | read)
        if value is not None:
            binding["value_variable"] = value
        return binding
    if shape == "reduction":
        return {
            "kind": "baseline_flag_crosstab",
            "affected_flag_variable": draw(
                st.sampled_from(("household_count", "income_tax"))
            ),
            "value_variable": "person_count",
            "value_reduction": {"entity": "person", "variable": draw(read)},
        }
    binding = {
        "kind": "baseline_flag_crosstab",
        "from_entity": "person",
        "affected_flag_variable": draw(read),
    }
    if shape == "value":
        binding["value_variable"] = draw(read)
    else:
        binding["count_of"] = draw(read | st.sampled_from(("person", "household")))
    return binding


@settings(max_examples=300, deadline=None)
@given(
    binding=_provider_rows(),
    declared=st.sampled_from((2023, 2024, 2025)),
    calibration=st.sampled_from((2024, 2025)),
    metadata=st.booleans(),
)
def test_a_provider_row_refuses_exactly_when_a_read_is_off_its_period(
    binding, declared, calibration, metadata
):
    """Differential: the guard's static reads against the provider's own.

    The provider runs on an adapter that records every column it reads; the
    row must refuse exactly when one of those is held at a period other than
    the declared one. A count read through the adapter's column, or a stored
    column on a frame without period metadata, holds any period.
    """

    recorder = _provider_adapter(metadata, _ReadRecorder)
    # The reads do not depend on the period; 2024 has a reviewed AEA.
    default_provider_registry()[binding["kind"]](recorder, binding, 2024)
    off = {
        f"{entity}.{variable}"
        for entity, variable, through_column in recorder.reads
        if _held(
            variable,
            through_column=through_column,
            metadata=metadata,
            calibration=calibration,
        )
        not in (None, str(declared))
    }
    registry, contract = _registry({"row": {**binding, "measurement_period": declared}})
    adapter = _provider_adapter(metadata)
    if off:
        with pytest.raises(MeasurementPeriodError) as refused:
            materialize_target_bindings(adapter, registry, contract, period=calibration)
        assert all(read in str(refused.value) for read in off)
    else:
        materialize_target_bindings(adapter, registry, contract, period=calibration)


def test_without_a_provider_the_frame_is_read_as_it_is():
    frame = PeriodFrameStub()
    registry, contract = _registry(
        {"frame_only": {"value_variable": "employment_income"}}
    )
    adapter = UKFrameTargetAdapter(frame)

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert not materialized.skipped
    np.testing.assert_allclose(adapter.tables["person"]["test_frame_only"], STORED)


def test_only_stored_variables_take_the_reserved_injection_name():
    adapter = UKFrameTargetAdapter(PeriodFrameStub())

    assert measure_input_column(adapter, "person", "employment_income") == (
        ENGINE_PERIOD_COLUMN_PREFIX + "employment_income"
    )
    assert measure_input_column(adapter, "person", "income_tax") == "income_tax"
    # An adapter that declares no stored columns keeps every name, as before.
    assert measure_input_column(SimpleNamespace(), "person", "employment_income") == (
        "employment_income"
    )


@pytest.mark.parametrize(
    "year,route_prefix", [(2024, "frame_columns:"), (2025, "engine_components:")]
)
def test_the_profit_after_loss_relief_reads_the_frame_only_at_its_period(
    year, route_prefix
):
    frame = PeriodFrameStub(time_period="2024")
    profit = np.array([10_700.0, 2_140.0, 5_350.0])
    relief = np.array([3_000.0, 5_000.0, 0.0])
    sim = SimulationStub(
        {
            "self_employment_income": ("person", profit),
            LOSS_RELIEF_COLUMN: ("person", relief),
        }
    )

    values, route = compute_uk_measure_input(
        frame, sim, "person", UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF, year
    )

    assert route.startswith(route_prefix)
    stored = frame.table("person")
    expected = (
        self_employment_income_after_loss_relief(
            stored["self_employment_income"].to_numpy(dtype=float),
            stored[LOSS_RELIEF_COLUMN].to_numpy(dtype=float),
        )
        if year == 2024
        else self_employment_income_after_loss_relief(profit, relief)
    )
    np.testing.assert_allclose(values, expected)


@pytest.mark.parametrize("year", [2024, 2025])
def test_the_named_loss_relief_terms_keep_the_resolvers_floats(year):
    """The terms give the floats the resolver gave by name (microcosm#1095).

    Before the terms were declared, the profit after the relief read the
    stored columns at the frame's period and the engine's components at any
    other, and total income after the relief always read the engine's.
    """

    frame = PeriodFrameStub(time_period="2024")
    person = frame.table("person")
    # Amounts no float32 holds exactly, so a read from the wrong source shows.
    person["self_employment_income"] = [10_700.37, 2_140.11, 5_350.93]
    person[LOSS_RELIEF_COLUMN] = [3_000.29, 5_000.0, 0.0]
    profit = np.array([11_235.389, 2_247.1155, 5_618.4765])
    relief = np.array([3_000.29, 5_000.0, 0.0])
    total = np.array([31_235.389, 9_247.1155, 5_618.4765])
    sim = SimulationStub(
        {
            "self_employment_income": ("person", profit),
            LOSS_RELIEF_COLUMN: ("person", relief),
            "total_income": ("person", total),
        }
    )

    after, after_route = compute_uk_measure_input(
        frame, sim, "person", UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF, year
    )
    banded, banded_route = compute_uk_measure_input(
        frame, sim, "person", UK_TOTAL_INCOME_AFTER_LOSS_RELIEF, year
    )

    def former_after(stored_profit, stored_relief):
        return np.maximum(
            np.asarray(stored_profit, dtype=float)
            - np.asarray(stored_relief, dtype=float),
            0.0,
        )

    if year == 2024:
        expected_after = former_after(
            person["self_employment_income"].to_numpy(dtype=float),
            person[LOSS_RELIEF_COLUMN].to_numpy(dtype=float),
        )
        assert after_route == "frame_columns:self_employment_income_after_loss_relief"
    else:
        expected_after = former_after(profit, relief)
        assert after_route == (
            "engine_components:self_employment_income_after_loss_relief"
        )
    assert banded_route == "engine_components:total_income_after_loss_relief"
    assert np.array_equal(after, expected_after)
    assert np.array_equal(
        banded,
        total.astype(float) - profit.astype(float) + former_after(profit, relief),
    )
    # The annual simulation's inputs are not the frame's columns, so the
    # terms read the engine there at every period.
    _, route = compute_uk_measure_input(
        frame,
        sim,
        "person",
        UK_SELF_EMPLOYMENT_INCOME_AFTER_LOSS_RELIEF,
        year,
        frame_columns=False,
    )
    assert route.startswith("engine_components:")


def test_the_child_benefit_children_count_each_units_children():
    # HMRC bands its families in payment by these children (microcosm#1095).
    frame = PeriodFrameStub(time_period="2024")
    sim = SimulationStub(
        {
            "is_child_or_qualifying_young_person_for_child_benefit": (
                "person",
                np.array([True, True, False]),
            )
        }
    )
    name = measure_simulation.UK_CHILD_BENEFIT_CHILD_COUNT

    values, route = compute_uk_measure_input(frame, sim, "benunit", name, 2025)

    assert values.tolist() == [2.0, 0.0]
    assert route == (
        "engine_members:is_child_or_qualifying_young_person_for_child_benefit"
    )
    with pytest.raises(KeyError, match="benunit-only"):
        compute_uk_measure_input(frame, sim, "person", name, 2025)


def test_a_frame_without_a_period_label_keeps_the_frame_route():
    frame = PeriodFrameStub()
    frame.metadata = {}

    assert measure_simulation._measured_at_frame_period(frame, 2025)


#: Every filter, household condition and band variable a plain-column binding
#: reads. They keep their stored reads (microcosm#1324): who a row counts is
#: fixed at the frame. Each is engine-computed (measured at the calibration
#: period through the provider) or a stored input that does not change between
#: the frame's period and the calibration period (ages and sex are static; sign
#: filters keep their sign under uprating, the student-loan plan and the
#: private-school flag are stored at the calibration year). The DfE childcare
#: rows read the four spring-term measures (dfe_spring_term.py), which are
#: never stored (the resolver refuses them as columns), so they are computed
#: at the calibration period. A new name here is
#: a stored read to review: route it, or measure the row at the frame period.
REVIEWED_PLAIN_COLUMN_POPULATION_VARIABLES = frozenset(
    {
        "adult_dependants_grant",
        "age",
        "attendance_allowance",
        "attends_private_school",
        "benefit_cap_reduction",
        "benunit_tenure_type",
        # Engine-computed at the calibration period: the Child Benefit paid and
        # the children it is paid for (microcosm#1095).
        "child_benefit",
        "child_benefit_child_count",
        "council_tax_band",
        # Engine-computed from the household's adults (pe-uk#1864; microcosm#953).
        "council_tax_discount_rate",
        "country",
        "dfe_age_at_31_december",
        "dfe_spring_early_learning_2",
        "dfe_spring_universal",
        "dfe_spring_working_parent",
        "dividend_income",
        "eldest_adult_age",
        "employment_income",
        "employment_sector",
        "esa",
        "esa_contrib",
        "esa_income",
        "gas_consumption",
        "gender",
        "guarantee_credit",
        "household_num_benunits",
        "housing_benefit",
        "income_tax",
        # Engine-computed at the calibration period: the frame stores age, not
        # this flag, and the engine reads State Pension age as reached by 6
        # October of the period it is asked for (microcosm#1069).
        "is_SP_age",
        "is_child",
        "is_child_receiving_tax_free_childcare",
        "jsa_contrib",
        "main_residence_value",
        "maintenance_loan",
        "ons_household_type",
        "parents_learning_allowance",
        "pension_contributions_via_salary_sacrifice",
        "pension_credit",
        "private_pension_income",
        "property_income",
        "region",
        "relation_type",
        "savings_credit",
        # The stored gross interest less its ISA part, read at the frame by the
        # resolver; a sign filter, so its sign holds under uprating
        # (microcosm#1095).
        "savings_interest_income_outside_isas",
        "self_employment_income_after_loss_relief",
        "state_pension",
        "state_pension_type",
        "student_loan_plan",
        "student_loan_repayments",
        "tax_band",
        "tenure_type",
        "total_income_after_loss_relief",
        "uc_LCWRA_element",
        "uc_benefit_cap_reduction_after_floor",
        "uc_calibration_administrative_family_type",
        "uc_calibration_child_count",
        "uc_calibration_has_child_under_one",
        "uc_carer_element",
        "uc_childcare_element",
        "uc_deductions",
        "uc_housing_costs_element",
        "universal_credit",
        # The annual income measures: the frame stores none of them, and the
        # resolver computes each on its annual simulation at the calibration
        # period.
        *measure_simulation.UK_ANNUAL_INCOME_MEASURES,
    }
)


def test_plain_column_population_variables_are_reviewed():
    contract = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text()
    )
    used: set[str] = set()
    for target in contract["targets"]:
        binding = (target.get("bindings") or {}).get("policyengine") or {}
        if binding.get("kind"):
            continue
        for predicate in (
            *binding.get("filters", ()),
            *binding.get("household_conditions", ()),
        ):
            if "reduce" in predicate and predicate in binding.get("filters", ()):
                continue
            used.add(str(predicate.get("variable") or predicate.get("concept")))
        if binding.get("groupby_variable"):
            used.add(str(binding["groupby_variable"]))

    unreviewed = sorted(used - REVIEWED_PLAIN_COLUMN_POPULATION_VARIABLES)
    assert not unreviewed, (
        f"plain-column bindings read {unreviewed} as a filter, condition or band; "
        "these keep their stored reads, so confirm each is engine-computed or "
        "does not change between the frame's period and the calibration period "
        "(microcosm#1324), then add it here."
    )
