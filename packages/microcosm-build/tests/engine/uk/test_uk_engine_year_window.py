"""policyengine-uk simulations in the build hold no year after the one read.

Given a one-year dataset, policyengine-uk projects it to 2032 and every
constructor runs the Universal Credit rebalancing loop over 2026-2032. The
spine adapter and the measure resolver's scratch engines read only the
frame's year and the year they measure, so they stop the projection and the
loop there (``uk_dataset_through``, ``uk_rebalancing_years_through``).

These tests build each simulation both ways on a synthetic frame where the
Universal Credit health element is paid, and compare every requested output
and every array the simulation and its baseline hold for a period up to the
year read. Release scoring (a direct H5) keeps the whole projection.

Two guards go with the cut. A scratch resolver refuses a measure dated after
its year, which policyengine-uk would answer from the last year's inputs. A
direct H5 is built only where no cut of the loop is in force
(``uk_whole_rebalancing_years``).
"""

from __future__ import annotations

import hashlib
import inspect
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind
from microcosm.frame.adapters import policyengine_uk as adapter_module
from microcosm.frame.adapters.policyengine_uk import (
    PolicyEngineUKEngine,
    uk_dataset_through,
    uk_whole_rebalancing_years,
)
from test_support.microcosm_build.uk_engine_window import stub_scratch_engine_window

#: The policyengine-uk sources the rebalancing cut rests on, hashed as
#: microcosm#1538's engine population declaration hashes its pins (the text
#: ``inspect.getsource`` returns). A new digest means policyengine-uk changed
#: the loop or the formula that reads its years: re-check the cut, then re-pin.
_REVIEWED_UC_REFORM_SOURCES = {
    "add_universal_credit_reform": (
        "ca13079bb2043c8965dd2cc9397c13c58b7f98f8e58b2c678ab7602f72a55935"
    ),
    "_add_inactive_health_element_formula": (
        "8f20a36b13458b140caeb8d6f06f1fdad4bf0ba28d17a04f7f5b4a325b2ad907"
    ),
}

#: (the frame's year, the year read): the spine reads 2024 data at 2024 and
#: at 2025, and the measure step reads a 2024 or 2025 frame at 2025.
_CASES = [(2024, 2024), (2024, 2025), (2025, 2025)]

_OUTPUTS = (
    "universal_credit",
    "uc_LCWRA_element",
    "income_tax",
    "national_insurance",
    "capital_gains_tax",
    "pension_credit",
    "child_benefit",
    "state_pension",
    "household_net_income",
)


def _frame(data_year: int):
    """Eight people in seven households, invented for the test.

    Two working-age claimants have limited capability for work-related
    activity, so their Universal Credit awards carry the health element the
    rebalancing loop rewrites from 2026. The others pay income tax, capital
    gains tax and National Insurance, draw pensions and Pension Credit, and
    receive Child Benefit.
    """

    people = 8
    person = pd.DataFrame(
        {
            "person_id": np.arange(people, dtype=np.int64),
            "person_benunit_id": np.asarray([0, 1, 2, 3, 3, 5, 6, 7], dtype=np.int64),
            "person_household_id": np.asarray([0, 1, 2, 3, 3, 5, 6, 7], dtype=np.int64),
            "age": [45.0, 38.0, 33.0, 29.0, 6.0, 72.0, 80.0, 51.0],
            "employment_income": [
                0.0,
                6_000.0,
                32_000.0,
                11_000.0,
                0.0,
                0.0,
                0.0,
                125_000.0,
            ],
            "self_employment_income": [0.0] * people,
            "private_pension_income": [0.0] * 5 + [500.0, 38_000.0, 0.0],
            "state_pension_reported": [0.0] * 5 + [7_000.0, 11_500.0, 0.0],
            "uc_limited_capability_for_WRA": [True, True] + [False] * 6,
            "capital_gains": [0.0, 0.0, 18_000.0] + [0.0] * 4 + [60_000.0],
            "savings_interest_income": [0.0, 0.0, 1_200.0, 0.0, 0.0, 300.0]
            + [2_000.0, 4_000.0],
            "dividend_income": [0.0] * 7 + [15_000.0],
            "pension_contributions_via_salary_sacrifice": [0.0, 0.0, 2_000.0]
            + [0.0] * 4
            + [10_000.0],
            "is_benunit_head": [True] * 4 + [False] + [True] * 3,
            "is_parent": [False] * 3 + [True] + [False] * 4,
            # Each adult's whole year at today's rates (the annual measures).
            "employment_income_annual_share": [1.0] * people,
            "employment_income_other_spells": [0.0] * people,
            "self_employment_income_annual_share": [1.0] * people,
            "self_employment_income_other_spells": [0.0] * people,
            "private_pension_income_annual_share": [1.0] * people,
            "contributory_benefits_annual_share": [1.0] * people,
        }
    )
    benunits = [0, 1, 2, 3, 5, 6, 7]
    benunit = pd.DataFrame(
        {
            "benunit_id": np.asarray(benunits, dtype=np.int64),
            "would_claim_uc": [True] * 4 + [False] * 3,
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray(benunits, dtype=np.int64),
            "household_weight": [1.0] * len(benunits),
            "region": [
                "NORTH_EAST",
                "WALES",
                "LONDON",
                "SCOTLAND",
                "SOUTH_WEST",
                "NORTH_WEST",
                "SOUTH_EAST",
            ],
            "council_tax": [1_400.0, 1_500.0, 1_700.0, 1_300.0, 1_900.0]
            + [2_100.0, 3_200.0],
            "rent": [5_200.0, 0.0, 0.0, 5_600.0, 0.0, 0.0, 0.0],
            "tenure_type": [
                "RENT_FROM_COUNCIL",
                "OWNED_OUTRIGHT",
                "OWNED_WITH_MORTGAGE",
                "RENT_FROM_HA",
                "OWNED_OUTRIGHT",
                "OWNED_OUTRIGHT",
                "OWNED_OUTRIGHT",
            ],
            "main_residence_value": [0.0, 180_000.0, 420_000.0, 0.0]
            + [260_000.0, 310_000.0, 900_000.0],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=str(data_year),
        weight_kind=WeightKind.DESIGN,
    )


def _held(simulation, last_year: int) -> dict[tuple[str, str], np.ndarray]:
    """Every array the simulation's default branch holds up to ``last_year``."""

    from policyengine_core import periods

    held = {}
    for population in simulation.populations.values():
        for name, holder in population._holders.items():
            for key, array in holder._memory_storage._arrays.items():
                branch, period_text = key.split(":", 1)
                if branch != "default":
                    continue
                if periods.period(period_text).start.year > last_year:
                    continue
                held[(name, period_text)] = np.asarray(array)
    return held


def _assert_same_up_to(today, cut, last_year: int, label: str) -> set:
    """The two simulations hold the same arrays up to ``last_year``.

    A key one of them holds and the other does not is a value the other has
    not computed yet; computed there, it must be the same. Returns those keys.
    """

    held_today = _held(today, last_year)
    held_cut = _held(cut, last_year)
    assert held_today and held_cut, label
    for key in sorted(held_today.keys() & held_cut.keys()):
        np.testing.assert_array_equal(
            held_cut[key], held_today[key], err_msg=f"{label}: {key}"
        )
    only_one = held_today.keys() ^ held_cut.keys()
    for key in sorted(only_one):
        held, other = (held_today, cut) if key in held_today else (held_cut, today)
        np.testing.assert_array_equal(
            np.asarray(other.calculate(*key)), held[key], err_msg=f"{label}: {key}"
        )
    return only_one


def _today(monkeypatch) -> None:
    """Restore policyengine-uk's own projection and loop, as before the cut."""

    monkeypatch.setattr(
        adapter_module,
        "uk_dataset_through",
        lambda dataset, end_year, parameters: dataset,
    )
    monkeypatch.setattr(
        adapter_module, "uk_rebalancing_years_through", lambda end_year: nullcontext()
    )


def _adapter_run(frame, data_year: int, read_year: int):
    engine = PolicyEngineUKEngine()
    simulations = []
    build = engine._read_year_simulation

    def recording(microsimulation_class, dataset, period):
        simulation = build(microsimulation_class, dataset, period)
        simulations.append(simulation)
        return simulation

    engine._read_year_simulation = recording
    outputs = engine.materialize(
        frame,
        [name for name in _OUTPUTS if name in engine._tax_benefit_system().variables],
        read_year,
        dataset_period=None if data_year == read_year else data_year,
    )
    (simulation,) = simulations
    return outputs, simulation


@pytest.mark.parametrize(("data_year", "read_year"), _CASES)
def test_the_adapter_cut_holds_what_the_whole_projection_holds(
    monkeypatch, data_year, read_year
) -> None:
    frame = _frame(data_year)
    outputs, cut = _adapter_run(frame, data_year, read_year)
    with monkeypatch.context() as patched:
        _today(patched)
        today_outputs, today = _adapter_run(frame, data_year, read_year)

    assert sorted(cut.dataset.years) == list(range(data_year, read_year + 1))
    assert sorted(today.dataset.years)[-1] > read_year
    # The health element is paid, so the loop had awards to rewrite, and
    # each programme the frame is built to reach pays someone.
    assert (np.asarray(outputs["uc_LCWRA_element"]) > 0).sum() == 2
    for name in ("capital_gains_tax", "pension_credit", "child_benefit"):
        assert np.asarray(outputs[name]).any(), name
    assert outputs.keys() == today_outputs.keys()
    for name, values in outputs.items():
        np.testing.assert_array_equal(values, today_outputs[name], err_msg=name)
    extra = _assert_same_up_to(today, cut, read_year, "simulation")
    extra |= _assert_same_up_to(today.baseline, cut.baseline, read_year, "baseline")
    assert len(extra) <= 1, sorted(extra)


def _resolver_run(frame, read_year: int, scratch):
    resolver = UKMeasureResolver(
        simulation_source=None, scratch_dir=scratch, year=read_year, frame=frame
    )
    measures = {
        name: np.asarray(resolver.compute("person", name)[0])
        for name in ("income_tax", "employment_income_annual")
    }
    measures |= {
        name: np.asarray(resolver.compute("benunit", name)[0])
        for name in ("universal_credit", "uc_LCWRA_element")
    }
    counterfactual = resolver._counterfactual_simulation(
        "pension_contributions_via_salary_sacrifice", "employment_income"
    )
    simulations = {
        "main": resolver.simulation,
        "annual income": resolver._annual_income_simulation(),
        "counterfactual": counterfactual,
    }
    return measures, simulations


@pytest.mark.parametrize(("data_year", "read_year"), _CASES)
def test_the_resolver_scratch_cut_holds_what_the_whole_projection_holds(
    monkeypatch, tmp_path, data_year, read_year
) -> None:
    from policyengine_uk.data import UKMultiYearDataset

    frame = _frame(data_year)
    measures, cut = _resolver_run(frame, read_year, tmp_path / "cut")
    scratch = tmp_path / "cut" / "simulation-input.h5"
    assert UKMultiYearDataset.validate_file_path(str(scratch))
    assert sorted(UKMultiYearDataset(str(scratch)).years) == list(
        range(data_year, read_year + 1)
    )
    with monkeypatch.context() as patched:
        stub_scratch_engine_window(patched)
        today_measures, today = _resolver_run(frame, read_year, tmp_path / "today")

    assert (measures["uc_LCWRA_element"] > 0).sum() == 2
    for name, values in measures.items():
        np.testing.assert_array_equal(values, today_measures[name], err_msg=name)
    extra = set()
    for label, simulation in cut.items():
        assert sorted(simulation.dataset.years) == list(range(data_year, read_year + 1))
        assert sorted(today[label].dataset.years)[-1] > read_year
        extra |= _assert_same_up_to(today[label], simulation, read_year, label)
        extra |= _assert_same_up_to(
            today[label].baseline, simulation.baseline, read_year, f"{label} baseline"
        )
    assert len(extra) <= 1, sorted(extra)


def test_a_direct_h5_keeps_the_whole_projection_and_loop(tmp_path) -> None:
    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk.data.economic_assumptions import PROJECTION_END_YEAR
    from policyengine_uk.scenarios import uc_reform

    frame = _frame(2024)
    path = tmp_path / "release.h5"
    measure_simulation.write_uk_national_frame(frame, path)
    assert UKSingleYearDataset.validate_file_path(str(path), False)
    resolver = UKMeasureResolver(
        simulation_source=path, scratch_dir=tmp_path / "scratch", year=2025, frame=frame
    )

    assert sorted(resolver.simulation.dataset.years)[-1] == PROJECTION_END_YEAR
    held = _held(resolver.simulation, PROJECTION_END_YEAR)
    assert all(
        ("uc_LCWRA_element", str(year)) in held for year in uc_reform.REBALANCING_YEARS
    )
    # It holds the years after the resolver's, so it measures them.
    gains, route = resolver.compute("person", "cgt_2026_gains")
    assert route.startswith("engine_period:2026:")
    assert np.asarray(gains).sum() > 0


def test_a_scratch_resolver_refuses_a_measure_dated_after_its_year(tmp_path) -> None:
    """policyengine-uk answers a year the scratch engine does not hold from the
    last year's inputs, with no error, so the resolver refuses the read."""

    frame = _frame(2024)
    resolver = UKMeasureResolver(
        simulation_source=None, scratch_dir=tmp_path, year=2025, frame=frame
    )

    for name in ("cgt_2026_gains", "company_car_benefit_2026"):
        with pytest.raises(ValueError, match="holds no year after 2025"):
            resolver.compute("person", name)
    # The frame's year and the resolver's are held, and measured.
    people = len(frame.table("person"))
    for name in ("cgt_2024_gains", "cgt_2025_gains", "company_car_benefit_2024"):
        values, _route = resolver.compute("person", name)
        assert np.asarray(values).shape == (people,), name
    assert np.asarray(resolver.compute("person", "cgt_2024_gains")[0]).sum() > 0


def test_a_direct_h5_is_refused_while_a_rebalancing_cut_is_left(
    monkeypatch, tmp_path
) -> None:
    """A release is scored on the whole loop: a cut that anything left in the
    process would shorten it with no error."""

    from policyengine_uk.scenarios import uc_reform

    frame = _frame(2024)
    path = tmp_path / "release.h5"
    measure_simulation.write_uk_national_frame(frame, path)
    first = uc_reform.FIRST_REBALANCING_YEAR
    monkeypatch.setattr(uc_reform, "REBALANCING_YEARS", range(first, first))

    with pytest.raises(RuntimeError, match="a cut was left"):
        UKMeasureResolver(
            simulation_source=path,
            scratch_dir=tmp_path / "scratch",
            year=2025,
            frame=frame,
        )


def test_uk_dataset_through_builds_the_data_year_to_the_end_year() -> None:
    engine = PolicyEngineUKEngine()
    parameters = engine._tax_benefit_system().parameters
    dataset = engine._build_dataset(_frame(2024), 2024)

    assert sorted(uk_dataset_through(dataset, 2024, parameters).years) == [2024]
    assert sorted(uk_dataset_through(dataset, 2025, parameters).years) == [2024, 2025]
    assert sorted(uk_dataset_through(dataset, 2027, parameters).years) == [
        2024,
        2025,
        2026,
        2027,
    ]


def test_the_rebalancing_years_are_still_a_range_to_the_projection_end() -> None:
    from policyengine_uk.data.economic_assumptions import PROJECTION_END_YEAR
    from policyengine_uk.scenarios import uc_reform

    assert isinstance(uc_reform.REBALANCING_YEARS, range)
    assert uc_reform.REBALANCING_YEARS == range(
        uc_reform.FIRST_REBALANCING_YEAR, PROJECTION_END_YEAR + 1
    )
    # The whole-loop block reads both years off the loop's own module.
    assert uc_reform.PROJECTION_END_YEAR == PROJECTION_END_YEAR
    with uk_whole_rebalancing_years():
        pass


@pytest.mark.parametrize("name", sorted(_REVIEWED_UC_REFORM_SOURCES))
def test_the_uc_reform_sources_are_the_reviewed_ones(name) -> None:
    from policyengine_uk.scenarios import uc_reform

    source = inspect.getsource(getattr(uc_reform, name))
    assert (
        hashlib.sha256(source.encode("utf-8")).hexdigest()
        == _REVIEWED_UC_REFORM_SOURCES[name]
    ), f"policyengine-uk changed {name}; re-check the rebalancing cut, then re-pin."
