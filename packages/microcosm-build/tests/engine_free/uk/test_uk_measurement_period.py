"""Plain-column values are measured at their binding's period (microcosm#1324).

The frame stores its inputs at its own period (2024); the calibration solves
at 2025. A binding's value variable that the frame stores is computed by the
measure provider at the measurement period, like every engine-computed
variable, unless the binding declares the frame's own period. Filters,
conditions and bands keep their stored reads.
"""

from __future__ import annotations

import copy
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


def _bound_rows_refused_at_their_declared_period(
    exclusions=None, membership=None, floor_targets=None, targets=None
) -> list[str]:
    """The national rows the solve would refuse to measure, or would lose.

    The rows are the ones the national compiles (``graph_national``): the
    targets on the national line, less the rows signed out of a fan-out and
    the rows a per-row exclusion holds, except held rows in the support
    floor's scope, which are compiled as probes to measure their support
    (``floor_targets`` stands in for the floor's targets). On them it runs
    the solve's two checks in its order (``target_materialization``): on
    every binding kind, a row that requires its fact at the measurement
    period resolves to that period; then ``_refuse_ignored_measurement_period``
    over each plain-column target that declares a period, with the spine's
    stored columns and the release's frame period standing in for the frame:
    a value is held at the frame's period only as a stored column or a CGT
    dated measure (``UKFrameTargetAdapter.value_period``). The plain-column
    check also covers the national-geography targets on the local line only,
    which the dense build compiles through the same check (the eighth
    re-review of microcosm#1420).

    One check is this test's own: no build step refuses a required row that a
    re-pin defers, or drops from the membership. The national compiles
    ``uk/target_references.json``, the active references only, so the row
    leaves it silently. This is the check tied to the period requirement (the
    seventh and eighth re-reviews); ``test_uk_cgt_targets.py`` also pins the
    CGT rows' active status.
    """

    from microcosm.build.target_materialization import (
        MeasurementPeriodError,
        _refuse_ignored_measurement_period,
    )
    from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
    from microcosm.build.uk_runtime.graph import uk_spine_endpoint, uk_spine_graph
    from microcosm.build.uk_runtime.ledger_targets import (
        UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS,
        uk_contract_target_calibration_lines,
    )
    from microcosm.build.uk_runtime.measure_simulation import (
        load_uk_calibration_measure_exclusions,
    )
    from microcosm.build.uk_runtime.support_floor import (
        load_uk_support_floor_declaration,
        uk_support_floor_from_payload,
    )

    stored: dict[str, set[str]] = {}
    for cells in uk_spine_endpoint(uk_spine_graph()).inputs:
        stored.setdefault(str(cells.entity), set()).update(map(str, cells.columns))
    release = load_uk_frs_release()
    frame = SimpleNamespace(
        stored_columns={entity: frozenset(names) for entity, names in stored.items()},
        frame_period=str(release.time_period),
    )
    adapter = SimpleNamespace(
        frame_period=frame.frame_period,
        value_period=lambda entity, variable: UKFrameTargetAdapter.value_period(
            frame, entity, variable
        ),
    )

    def resource(name: str):
        return json.loads(
            importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
        )

    held = {
        entry["name"]
        for entry in (
            load_uk_calibration_measure_exclusions()
            if exclusions is None
            else exclusions
        )
    }
    signed = {
        entry["target_id"]
        for entry in resource("target_reference_signed_exclusions.json")["exclusions"]
        if "row" not in entry
    }
    if membership is None:
        membership = resource("target_reference_membership.json")["targets"]
    if floor_targets is None:
        declaration = load_uk_support_floor_declaration()
        floor_targets = (
            frozenset()
            if declaration is None
            else uk_support_floor_from_payload(declaration).contract_target_ids
        )
    if targets is None:
        targets = resource("uk_population_targets.json")["targets"]
    refused = []
    for target in targets:
        target_id = target["target_id"]
        # National geography, as ``_uk_contract_targets(national_only=True)``.
        national = (
            set(target.get("geography_levels") or ())
            <= UK_NATIONAL_TARGET_GEOGRAPHY_LEVELS
        )
        if not national or target_id in signed:
            continue
        on_national_line = "national" in uk_contract_target_calibration_lines(target)
        binding = (target.get("bindings") or {}).get("policyengine") or {}
        candidates = list(membership.get(target_id, {}).get("candidates", ()))
        rows = [
            candidate
            for candidate in candidates
            if candidate.get("status") != "signed_excluded"
            and (candidate.get("name") not in held or target_id in floor_targets)
        ]
        if candidates and not rows:
            continue
        # The fact-period check runs before the solve branches on kind, so it
        # covers the CGT provider rows too (the fifth and sixth re-reviews of
        # microcosm#1420, 4 and 2).
        if on_national_line and binding.get("require_matching_fact_period"):
            expected = str(binding.get("measurement_period", release.calibration_year))
            if not rows or any(
                row.get("status") != "active"
                or str(row.get("resolved_fact_period")) != expected
                for row in rows
            ):
                refused.append(target_id)
                continue
        if binding.get("kind") or binding.get("measurement_period") is None:
            continue
        spec = SimpleNamespace(name=target_id, entity=target["measurement"]["entity"])
        try:
            _refuse_ignored_measurement_period(
                spec, binding, adapter, release.calibration_year
            )
        except MeasurementPeriodError:
            refused.append(target_id)
    return refused


def test_every_bound_row_with_a_declared_period_can_be_measured_there():
    """The fourth re-review of microcosm#1420, 1: the solve refuses a bound
    row whose declared period the frame does not hold, and stops the whole
    run at the national problem, after the spine and engine stages. Binding
    the betting and gaming row passed CI and refused on the first national;
    this runs the same check over the packaged contract."""

    assert _bound_rows_refused_at_their_declared_period() == []
    # Unheld, the row bound by commit 16 is the one the build refused.
    exclusions = [
        entry
        for entry in measure_simulation.load_uk_calibration_measure_exclusions()
        if entry["name"] != "obr.betting_and_gaming_duties"
    ]
    assert _bound_rows_refused_at_their_declared_period(exclusions) == [
        "obr.betting_and_gaming_duties"
    ]


def _committed_membership() -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("target_reference_membership.json")
        .read_text()
    )["targets"]


def _first_active(membership: dict, target_id: str) -> dict:
    (candidate, *_) = [
        candidate
        for candidate in membership[target_id]["candidates"]
        if candidate.get("status") == "active"
    ]
    return candidate


def test_the_fact_period_check_covers_every_binding_kind():
    """The sixth re-review of microcosm#1420, 2: the solve checks a row's fact
    period before it branches on kind, so a re-pin that resolves a CGT
    provider row to a later fact stops the national after the spine. One that
    defers a required row, or drops it from the membership, removes the row
    from the national without a refusal (the seventh re-review, 3): no build
    step refuses it, and this is the check tied to the period requirement.
    All three fail here."""

    membership = _committed_membership()
    later = copy.deepcopy(membership)
    _first_active(later, "hmrc.cgt.gains_by_region")["resolved_fact_period"] = "2025"
    assert _bound_rows_refused_at_their_declared_period(membership=later) == [
        "hmrc.cgt.gains_by_region"
    ]
    deferred = copy.deepcopy(membership)
    _first_active(deferred, "hmrc.cgt.gains_by_age_band")["status"] = (
        "no_fact_at_or_before_period"
    )
    assert _bound_rows_refused_at_their_declared_period(membership=deferred) == [
        "hmrc.cgt.gains_by_age_band"
    ]
    missing = {
        name: entry
        for name, entry in membership.items()
        if name != "hmrc.cgt.residential_property_gains"
    }
    assert _bound_rows_refused_at_their_declared_period(membership=missing) == [
        "hmrc.cgt.residential_property_gains"
    ]


def test_the_period_checks_read_the_rows_the_national_compiles():
    """The seventh re-review of microcosm#1420, 2: the national drops a row a
    per-row exclusion holds, unless the support floor's scope compiles it as
    a probe, and never compiles a row signed out of a fan-out."""

    membership = _committed_membership()
    later = copy.deepcopy(membership)
    moved = _first_active(later, "hmrc.cgt.gains_by_region")
    moved["resolved_fact_period"] = "2025"
    exclusions = [
        *measure_simulation.load_uk_calibration_measure_exclusions(),
        {"name": moved["name"]},
    ]
    # Held on its own, the moved row leaves the national: nothing to refuse.
    assert (
        _bound_rows_refused_at_their_declared_period(exclusions, membership=later) == []
    )
    # In the floor's scope the held row is compiled as a probe, and checked.
    assert _bound_rows_refused_at_their_declared_period(
        exclusions,
        membership=later,
        floor_targets=frozenset({"hmrc.cgt.gains_by_region"}),
    ) == ["hmrc.cgt.gains_by_region"]
    # Signed out row by row, the whole fan-out compiles nothing.
    for candidate in later["hmrc.cgt.gains_by_region"]["candidates"]:
        candidate["status"] = "signed_excluded"
    assert _bound_rows_refused_at_their_declared_period(membership=later) == []


def test_the_plain_column_check_covers_the_local_line_national_targets():
    """The eighth re-review of microcosm#1420: the dense build compiles the
    national-geography targets on the local line only through the same
    plain-column check, so a period declared on one of them is checked too."""

    from microcosm.build.uk_runtime.ledger_targets import (
        uk_contract_target_calibration_lines,
    )

    targets = copy.deepcopy(
        json.loads(
            importlib_resources.files("microcosm.build.uk")
            .joinpath("uk_population_targets.json")
            .read_text()
        )["targets"]
    )
    (local,) = [
        target
        for target in targets
        if target["target_id"] == "microcosm.high_value_homes.over_1m_by_region"
    ]
    assert uk_contract_target_calibration_lines(local) == {"local"}
    assert _bound_rows_refused_at_their_declared_period(targets=targets) == []
    local["bindings"]["policyengine"]["measurement_period"] = 2023
    assert _bound_rows_refused_at_their_declared_period(targets=targets) == [
        "microcosm.high_value_homes.over_1m_by_region"
    ]
