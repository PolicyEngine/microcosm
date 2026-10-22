"""Plain-column values are measured at their binding's period (microcosm#1324).

The frame stores its inputs at its own period (2024); the calibration solves
at 2025. A binding's value variable that the frame stores is computed by the
measure provider at the measurement period, like every engine-computed
variable, unless the binding declares the frame's own period. Filters,
conditions and bands keep their stored reads.
"""

from __future__ import annotations

import json
import sys
from importlib import resources as importlib_resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.target_materialization import (
    ENGINE_PERIOD_COLUMN_PREFIX,
    MeasurementPeriodError,
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


def _registry(bindings):
    return (
        TargetRegistry(
            [
                TargetSpec(
                    name=name,
                    entity="person",
                    measure=f"test_{name}",
                    value=1,
                    source="test",
                    metadata={"contract_target_id": name},
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


def test_a_frame_without_a_period_label_keeps_the_frame_route():
    frame = PeriodFrameStub()
    frame.metadata = {}

    assert measure_simulation._measured_at_frame_period(frame, 2025)


#: Every filter, household condition and band variable a plain-column binding
#: reads. They keep their stored reads (microcosm#1324): who a row counts is
#: fixed at the frame. Each is engine-computed (measured at the calibration
#: period through the provider) or a stored input that does not change between
#: the frame's period and the calibration period (ages and sex are static, and
#: so is current_education, which the DfE childcare rows read beside age to
#: leave out reception; sign filters keep their sign under uprating, the
#: student-loan plan and the private-school flag are stored at the calibration
#: year). A new name here is
#: a stored read to review: route it, or measure the row at the frame period.
REVIEWED_PLAIN_COLUMN_POPULATION_VARIABLES = frozenset(
    {
        "adult_dependants_grant",
        "age",
        "attendance_allowance",
        "attends_private_school",
        "benefit_cap_reduction",
        "benunit_tenure_type",
        "council_tax_band",
        "country",
        "current_education",
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
        "is_child",
        "is_child_receiving_extended_childcare",
        "is_child_receiving_targeted_childcare",
        "is_child_receiving_tax_free_childcare",
        "is_child_receiving_universal_childcare",
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
        "savings_interest_income",
        "self_employment_income_after_loss_relief",
        "state_pension",
        "state_pension_type",
        "student_loan_plan",
        "student_loan_repayments",
        "tax_band",
        "tenure_type",
        "total_income_after_loss_relief",
        "uc_LCWRA_element",
        "uc_calibration_administrative_family_type",
        "uc_calibration_child_count",
        "uc_calibration_has_child_under_one",
        "uc_carer_element",
        "uc_childcare_element",
        "uc_deductions",
        "uc_housing_costs_element",
        "universal_credit",
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
