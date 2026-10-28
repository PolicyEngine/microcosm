"""SPI band rows bound at their own tax year (option 1 of the pensioner review).

The four SPI 2023-24 pension band rows (State Pension and private pension,
count and amount by total income band) bind the published values at 2023-24
through measures dated by their name, as the CGT rows bind 2024-25. These
tests hold the engine-free parts: the names, the population factor, the
carry-back arithmetic, the measurement-period guard and the contract rows.
"""

from __future__ import annotations

import json
import math
import sys
from importlib import resources as importlib_resources
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.target_materialization import (
    MeasurementPeriodError,
    _binding_mask,
    materialize_target_bindings,
)
from microcosm.build.uk_runtime.hmrc_income import HMRC_SPI_INCOME_BAND_LOWER_BOUNDS
from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.spi_dated_measures import (
    UK_SPI_ADDITIVE_MEASURES,
    UK_SPI_CLASSIFIER_MEASURES,
    UK_SPI_POPULATION_AGE_BANDS,
    back_projected_inputs,
    back_projection_factors,
    ons_single_year_populations,
    population_factor_for_people,
    population_factors,
    round_trip_mismatches,
    uk_spi_dated_measure_kind,
    uk_spi_dated_measure_year,
    uk_spi_dated_measures_named,
)
from microcosm.calibrate import TargetRegistry, TargetSpec

PILOT_ROWS = (
    "hmrc.spi.state_pension.count_by_total_income_band",
    "hmrc.spi.state_pension.amount_by_total_income_band",
    "hmrc.spi.private_pension_income.count_by_total_income_band",
    "hmrc.spi.private_pension_income.amount_by_total_income_band",
)
_SEXES = ("FEMALE", "MALE")


def _uk_resource(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def _binding_names(binding: dict) -> list[str]:
    names = [binding.get("value_variable"), binding.get("groupby_variable")]
    names += [predicate["variable"] for predicate in binding.get("filters", ())]
    return [name for name in names if name]


# Names ---------------------------------------------------------------------


def test_dated_names_carry_their_year_and_measure():
    assert uk_spi_dated_measure_year("spi_2023_state_pension") == 2023
    assert uk_spi_dated_measure_kind("spi_2023_state_pension") == "state_pension"
    assert uk_spi_dated_measure_year("spi_2024_income_tax") == 2024
    for name in (
        "state_pension",
        "spi_2023_dividend_income",
        "spi_23_state_pension",
        "cgt_2024_gains",
        "spi_2023_state_pension_annual",
    ):
        assert uk_spi_dated_measure_year(name) is None
    # Every measure is either additive (carries the population factor) or a
    # classifier (does not), never both.
    kinds = set(UK_SPI_ADDITIVE_MEASURES) | set(UK_SPI_CLASSIFIER_MEASURES)
    assert not set(UK_SPI_ADDITIVE_MEASURES) & set(UK_SPI_CLASSIFIER_MEASURES)
    for kind in kinds:
        assert uk_spi_dated_measure_kind(f"spi_2023_{kind}") == kind


# Population factor ---------------------------------------------------------


def test_the_bands_are_the_ons_bands_the_calibration_binds():
    """The factor's bands are the sex-by-age bands the weights are solved to."""

    contract = _uk_resource("uk_population_targets.json")
    bound = set()
    for target in contract["targets"]:
        target_id = target["target_id"]
        for sex in ("male", "female"):
            prefix = f"ons.population.{sex}_"
            if target_id.startswith(prefix):
                bound.add(target_id.removeprefix(prefix))
    declared = {
        "90_plus"
        if lower == 90
        else (f"{lower}" if lower == upper else f"{lower}_{upper}")
        for lower, upper in UK_SPI_POPULATION_AGE_BANDS
    }
    assert declared == bound
    # The bands tile 0 to 90 exactly once.
    ages = [
        age
        for lower, upper in UK_SPI_POPULATION_AGE_BANDS
        for age in range(lower, upper + 1)
    ]
    assert ages == list(range(91))


_POPULATION = st.floats(min_value=1.0, max_value=5e6, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(
    then=st.lists(_POPULATION, min_size=182, max_size=182),
    now=st.lists(_POPULATION, min_size=182, max_size=182),
)
def test_a_band_weighted_to_the_calibration_population_sums_to_the_fact_years(
    then, now
):
    """Invariant: where the weights meet ONS's mid-2025 band populations, the
    band's population factor sums to its population at 1 October 2023."""

    keys = [(sex, age) for sex in _SEXES for age in range(91)]
    populations = {
        2023: dict(zip(keys, then, strict=True)),
        2025: dict(zip(keys, now, strict=True)),
    }
    factors = population_factors(2023, 2025, populations=populations)
    for sex in _SEXES:
        for lower, upper in UK_SPI_POPULATION_AGE_BANDS:
            ages = range(lower, upper + 1)
            mid_2023 = sum(populations[2023][(sex, a)] for a in ages)
            mid_2025 = sum(populations[2025][(sex, a)] for a in ages)
            october_2023 = mid_2023 + (3 / 24) * (mid_2025 - mid_2023)
            assert factors[(sex, lower)] > 0
            assert math.isclose(
                mid_2025 * factors[(sex, lower)], october_2023, rel_tol=1e-9
            )


def test_an_unchanged_population_has_unit_factors():
    keys = [(sex, age) for sex in _SEXES for age in range(91)]
    flat = {key: 1000.0 + key[1] for key in keys}
    factors = population_factors(2023, 2025, populations={2023: flat, 2025: flat})
    assert all(math.isclose(value, 1.0) for value in factors.values())


def test_the_vendored_estimates_give_the_pilot_factors():
    mid_2023 = ons_single_year_populations(2023)
    mid_2025 = ons_single_year_populations(2025)
    factors = population_factors(2023, 2025)
    # Recomputed by hand for one band from the vendored rows.
    then = sum(mid_2023[("FEMALE", age)] for age in range(70, 75))
    now = sum(mid_2025[("FEMALE", age)] for age in range(70, 75))
    assert math.isclose(factors[("FEMALE", 70)], (then + (now - then) / 8) / now)
    # Fewer people of State Pension age in 2023-24 than in mid-2025 in all,
    # though a band can be larger then: the post-war cohort ageing past 75
    # leaves 70 to 74 smaller in mid-2025 than in October 2023.
    pensioners = [
        (sex, lower, upper)
        for sex in _SEXES
        for lower, upper in UK_SPI_POPULATION_AGE_BANDS
        if lower >= 65
    ]
    now_total = then_total = 0.0
    for sex, lower, upper in pensioners:
        band = sum(mid_2025[(sex, age)] for age in range(lower, upper + 1))
        now_total += band
        then_total += band * factors[(sex, lower)]
    assert then_total < now_total
    with pytest.raises(ValueError, match="No vendored ONS mid-2024"):
        ons_single_year_populations(2024)
    with pytest.raises(ValueError, match="must precede"):
        population_factors(2025, 2025)


def test_each_person_takes_their_whole_year_band():
    factors = {
        (sex, lower): lower + (0.5 if sex == "MALE" else 0.0)
        for sex in _SEXES
        for lower, _ in UK_SPI_POPULATION_AGE_BANDS
    }
    values = population_factor_for_people(
        [65.75, 66.0, 15.9, 16.0, 97.0, 0.0],
        ["MALE", "female", "FEMALE", "MALE", "FEMALE", "MALE"],
        factors,
    )
    np.testing.assert_allclose(values, [65.5, 65.0, 15.0, 16.5, 90.0, 0.5])
    with pytest.raises(ValueError, match="Unknown sex"):
        population_factor_for_people([40.0], ["UNKNOWN"], factors)
    with pytest.raises(ValueError, match="non-negative age"):
        population_factor_for_people([-1.0], ["MALE"], factors)


# Carry-back arithmetic -----------------------------------------------------


_GROWTH = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False)


@settings(max_examples=80, deadline=None)
@given(
    growth_a=st.lists(_GROWTH, min_size=3, max_size=3),
    growth_b=st.lists(_GROWTH, min_size=3, max_size=3),
    pay=st.lists(
        st.floats(min_value=-1e6, max_value=1e7, allow_nan=False),
        min_size=1,
        max_size=8,
    ),
    years_back=st.integers(min_value=1, max_value=3),
)
def test_carrying_back_then_forward_returns_the_stored_inputs(
    growth_a, growth_b, pay, years_back
):
    """Invariant: the engine's forward index step undoes the carry-back."""

    dataset_year = 2024
    year = dataset_year - years_back
    growth = {
        "index.a": dict(zip(range(2022, 2025), growth_a, strict=True)),
        "index.b": dict(zip(range(2022, 2025), growth_b, strict=True)),
    }
    index_map = {"index.a": ["pay", "pay"], "index.b": ["rent"]}
    factors = back_projection_factors(
        index_map, growth, from_year=dataset_year, to_year=year
    )
    stored = {
        "person": pd.DataFrame(
            {
                "pay": pay,
                "age": np.arange(len(pay), dtype=float),
                "unused_pension_annual_allowance": 1.0,
            }
        ),
        "household": pd.DataFrame({"rent": [500.0], "region": ["LONDON"]}),
    }
    back = back_projected_inputs(
        stored, factors, dropped=["unused_pension_annual_allowance"]
    )
    assert "unused_pension_annual_allowance" not in back["person"]
    # Held columns are untouched; indexed ones divide.
    np.testing.assert_array_equal(back["person"]["age"], stored["person"]["age"])
    assert list(back["household"]["region"]) == ["LONDON"]
    forward = {name: table.copy() for name, table in back.items()}
    for step in range(year + 1, dataset_year + 1):
        forward["person"]["pay"] *= 1 + growth["index.a"][step]
        forward["household"]["rent"] *= 1 + growth["index.b"][step]
    assert round_trip_mismatches(stored, forward, ["pay", "rent"]) == {}
    # The originals are not mutated.
    assert "unused_pension_annual_allowance" in stored["person"]


def test_a_broken_round_trip_is_reported():
    stored = {"person": pd.DataFrame({"pay": [100.0, 200.0]})}
    forward = {"person": pd.DataFrame({"pay": [100.0, 201.0]})}
    assert round_trip_mismatches(stored, forward, ["pay"]) == {"pay": 1.0}
    assert round_trip_mismatches(stored, {"person": pd.DataFrame()}, ["pay"]) == {
        "pay": math.inf
    }


@settings(max_examples=60, deadline=None)
@given(
    values=st.lists(
        st.one_of(st.floats(-1e7, 1e7, allow_nan=False), st.just(math.nan)),
        min_size=1,
        max_size=12,
    ),
    data=st.data(),
)
def test_a_missing_value_matches_only_a_missing_value(values, data):
    """Invariant: a value the forward step loses, or invents, is a miss."""

    stored = {"person": pd.DataFrame({"pay": values})}
    assert (
        round_trip_mismatches(stored, {"person": stored["person"].copy()}, ["pay"])
        == {}
    )
    row = data.draw(st.integers(0, len(values) - 1))
    flipped = list(values)
    flipped[row] = 1.0 if math.isnan(values[row]) else math.nan
    forward = {"person": pd.DataFrame({"pay": flipped})}
    assert round_trip_mismatches(stored, forward, ["pay"]) == {"pay": math.inf}


def test_an_amount_reset_with_its_label_is_compared_where_the_label_held():
    """pe-uk zeroes a repayment exactly where it assigns a new loan plan."""

    stored = {
        "person": pd.DataFrame(
            {"plan": ["PLAN_1", "PLAN_2", "NONE"], "repaid": [500.0, 800.0, 0.0]}
        )
    }
    reassigned = {
        "person": pd.DataFrame(
            {"plan": ["PLAN_2", "PLAN_2", "NONE"], "repaid": [0.0, 800.0, 0.0]}
        )
    }
    where_held = {"repaid": "plan"}
    assert round_trip_mismatches(stored, reassigned, ["repaid"]) == {"repaid": 500.0}
    assert (
        round_trip_mismatches(
            stored, reassigned, ["repaid"], compared_where_unchanged=where_held
        )
        == {}
    )
    moved = {"person": reassigned["person"].assign(repaid=[0.0, 801.0, 0.0])}
    assert round_trip_mismatches(
        stored, moved, ["repaid"], compared_where_unchanged=where_held
    ) == {"repaid": 1.0}
    unlabelled = {"person": reassigned["person"].drop(columns="plan")}
    assert round_trip_mismatches(
        stored, unlabelled, ["repaid"], compared_where_unchanged=where_held
    ) == {"repaid": math.inf}
    # A reassigned row the forward step did not zero is not the reset rule:
    # the label condition must not hide it.
    not_reset = {"person": reassigned["person"].assign(repaid=[3.0, 800.0, 0.0])}
    assert round_trip_mismatches(
        stored, not_reset, ["repaid"], compared_where_unchanged=where_held
    ) == {"repaid": math.inf}
    nan_reset = {"person": reassigned["person"].assign(repaid=[np.nan, 800.0, 0.0])}
    assert round_trip_mismatches(
        stored, nan_reset, ["repaid"], compared_where_unchanged=where_held
    ) == {"repaid": math.inf}


def test_an_input_on_two_indices_or_a_later_year_refuses():
    growth = {"a": {2024: 0.1}, "b": {2024: 0.2}}
    with pytest.raises(ValueError, match="uprated by both"):
        back_projection_factors(
            {"a": ["pay"], "b": ["pay"]}, growth, from_year=2024, to_year=2023
        )
    with pytest.raises(ValueError, match="earlier year"):
        back_projection_factors({"a": ["pay"]}, growth, from_year=2024, to_year=2024)
    with pytest.raises(ValueError, match="not numeric"):
        back_projected_inputs({"person": pd.DataFrame({"pay": ["x"]})}, {"pay": 1.1})


# Bands ---------------------------------------------------------------------


class _Columns:
    def __init__(self, columns):
        self.columns = columns

    def column(self, entity, variable):
        return np.asarray(self.columns[variable], dtype=float)


_INCOME = st.one_of(
    st.floats(min_value=-1e5, max_value=3e6, allow_nan=False),
    st.sampled_from([float(edge) for edge in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS]),
)


@settings(max_examples=80, deadline=None)
@given(
    income=st.lists(_INCOME, min_size=1, max_size=40),
    pension=st.lists(st.floats(min_value=0, max_value=1e5), min_size=40, max_size=40),
)
def test_the_published_bands_count_every_taxpayer_once(income, pension):
    """Invariant: band counts sum to the table total over the published edges.

    The bands partition [GBP 12,570, inf) on the dated total income, so the
    banded values of any row add up to its unbanded value above the lowest
    edge: no one is dropped at an edge or counted in two bands.
    """

    values = np.asarray(pension[: len(income)], dtype=float)
    adapter = _Columns({"spi_2023_total_income": income})
    binding = {"groupby_variable": "spi_2023_total_income"}
    edges = [float(edge) for edge in HMRC_SPI_INCOME_BAND_LOWER_BOUNDS]
    bands = list(zip(edges, [*edges[1:], math.inf], strict=True))
    banded = [
        np.where(
            _binding_mask(adapter, "person", binding, values, band=band), values, 0.0
        )
        for band in bands
    ]
    membership = sum(
        _binding_mask(adapter, "person", binding, values, band=band).astype(int)
        for band in bands
    )
    above = np.asarray(income) >= edges[0]
    np.testing.assert_array_equal(membership, above.astype(int))
    np.testing.assert_allclose(sum(banded), np.where(above, values, 0.0))


# The measurement-period guard ----------------------------------------------


class PeriodFrameStub:
    entities = ("person", "benunit", "household")
    links = ()

    def __init__(self):
        self.metadata = {"time_period": "2024"}
        self._tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "person_benunit_id": [10, 10, 20],
                    "person_household_id": [100, 100, 200],
                    "employment_income": [1_000.0, 0.0, 500.0],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
            "household": pd.DataFrame({"household_id": [100, 200]}),
        }

    def table(self, entity):
        return self._tables[entity]

    def link(self, name):
        raise KeyError(name)


def _registry(bindings, *, fact_period="2023"):
    return (
        TargetRegistry(
            [
                TargetSpec(
                    name=name,
                    entity="person",
                    measure=f"test_{name}",
                    value=1,
                    source="test",
                    metadata={
                        "contract_target_id": name,
                        "ledger_fact_period": fact_period,
                    },
                )
                for name in bindings
            ],
            country="uk",
        ),
        {name: {"bindings": {"policyengine": b}} for name, b in bindings.items()},
    )


def _dated_binding(**overrides):
    binding = {
        "value_variable": "spi_2023_state_pension",
        "filters": [{"variable": "spi_2023_income_tax", "operator": ">", "value": 0}],
        "measurement_period": 2023,
        "require_matching_fact_period": True,
    }
    binding.update(overrides)
    return binding


def test_a_row_dated_throughout_binds_its_own_year():
    adapter = UKFrameTargetAdapter(PeriodFrameStub())
    assert adapter.value_period("person", "spi_2023_state_pension") == "2023"
    assert adapter.value_period("person", "spi_2023_person_count") == "2023"
    adapter.set_column("person", "spi_2023_state_pension", [100.0, 200.0, 300.0])
    adapter.set_column("person", "spi_2023_income_tax", [0.0, 50.0, 10.0])
    registry, contract = _registry({"dated": _dated_binding()})

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert not materialized.skipped
    np.testing.assert_allclose(
        adapter.tables["person"]["test_dated"], [0.0, 200.0, 300.0]
    )


@pytest.mark.parametrize(
    "overrides, missing",
    [
        # A 2025 filter would count 2023's pensions by 2025's taxpayers.
        (
            {
                "filters": [
                    {
                        "variable": "income_tax_before_winter_fuel_payment_charge_annual",
                        "operator": ">",
                        "value": 0,
                    }
                ]
            },
            "filter",
        ),
        ({"groupby_variable": "total_income_after_loss_relief_annual"}, "band"),
        ({"value_variable": "spi_2024_state_pension"}, "value"),
        ({"value_variable": "person_count"}, "count"),
    ],
)
def test_a_row_mixing_years_refuses(overrides, missing):
    del missing
    registry, contract = _registry({"mixed": _dated_binding(**overrides)})
    with pytest.raises(MeasurementPeriodError, match="every variable of the row"):
        materialize_target_bindings(
            UKFrameTargetAdapter(PeriodFrameStub()), registry, contract, period=2025
        )


def test_a_dated_row_refuses_a_fact_of_another_year():
    adapter = UKFrameTargetAdapter(PeriodFrameStub())
    adapter.set_column("person", "spi_2023_state_pension", [1.0, 1.0, 1.0])
    adapter.set_column("person", "spi_2023_income_tax", [1.0, 1.0, 1.0])
    registry, contract = _registry({"dated": _dated_binding()}, fact_period="2024")

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert [skip.name for skip in materialized.skipped] == ["dated"]
    assert "does not match declared measurement period 2023" in (
        materialized.skipped[0].reason
    )


class _SimulationStub:
    tax_benefit_system = SimpleNamespace(variables={})

    def calculate(self, variable, year, map_to=None):
        raise AssertionError("no engine read is expected")


@pytest.mark.parametrize(
    "binding",
    [
        _dated_binding(measurement_period=2024),
        _dated_binding(require_matching_fact_period=False),
    ],
)
def test_the_resolver_refuses_a_dated_measure_bound_at_another_year(
    monkeypatch, tmp_path, binding
):
    from microcosm.build.uk_runtime import measure_simulation

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="2.162.8", Microsimulation=lambda **kw: _SimulationStub()
        ),
    )
    monkeypatch.setattr(
        measure_simulation, "validate_uc_claimant_input", lambda *a: None
    )
    monkeypatch.setattr(
        measure_simulation,
        "_uk_contract_targets",
        lambda: {"pilot": {"bindings": {"policyengine": binding}}},
    )
    with pytest.raises(ValueError, match="SPI dated measure period mismatch: pilot"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=PeriodFrameStub(),
        )


@pytest.mark.parametrize(
    "binding, refusal",
    [
        (
            _dated_binding(
                filters=[
                    {
                        "variable": "income_tax_before_winter_fuel_payment_charge_annual",
                        "operator": ">",
                        "value": 0,
                    }
                ]
            ),
            "mixes dated SPI measures with variables of another year",
        ),
        (
            {
                **_dated_binding(),
                "value_variable": None,
                "value_expression": "spi_2023_state_pension + employment_income",
            },
            "mixes dated SPI measures with variables of another year",
        ),
        (
            _dated_binding(kind="parameter_gated_threshold"),
            "binds dated SPI measures \\['spi_2023_income_tax', "
            "'spi_2023_state_pension'\\] through provider 'parameter_gated_threshold'",
        ),
        # The interpreter reads a filter's concept when it names no variable.
        (
            _dated_binding(
                filters=[{"concept": "is_SP_age", "operator": "==", "value": True}]
            ),
            "mixes dated SPI measures with variables of another year",
        ),
        # A provider's own fields: no plain-column field names the measure.
        (
            {
                "kind": "count_affected",
                "affected_flag_variable": "spi_2023_state_pension",
                "measurement_period": 2023,
            },
            "binds dated SPI measures \\['spi_2023_state_pension'\\] through "
            "provider 'count_affected'",
        ),
    ],
)
def test_the_resolver_refuses_a_dated_row_it_could_not_read_at_one_year(
    monkeypatch, tmp_path, binding, refusal
):
    """A frame without period metadata does not exempt a row from the check."""

    binding = {key: value for key, value in binding.items() if value is not None}
    with pytest.raises(ValueError, match=f"pilot {refusal}"):
        _resolver_on(monkeypatch, tmp_path, {"bindings": {"policyengine": binding}})


def test_the_resolver_refuses_a_dated_row_that_is_also_projected(monkeypatch, tmp_path):
    target = {
        "uprating_index": "ons.population",
        "bindings": {"policyengine": _dated_binding()},
    }
    with pytest.raises(ValueError, match="pilot measures at 2023 but also projects"):
        _resolver_on(monkeypatch, tmp_path, target)


def _resolver_on(monkeypatch, tmp_path, target):
    """A resolver on a stub engine whose contract is the one target."""

    from microcosm.build.uk_runtime import measure_simulation

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="2.162.8", Microsimulation=lambda **kw: _SimulationStub()
        ),
    )
    monkeypatch.setattr(
        measure_simulation, "validate_uc_claimant_input", lambda *a: None
    )
    monkeypatch.setattr(
        measure_simulation, "_uk_contract_targets", lambda: {"pilot": target}
    )
    frame = PeriodFrameStub()
    frame.metadata = {}
    return UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "filters": [
                {
                    "variable": "income_tax_before_winter_fuel_payment_charge_annual",
                    "operator": ">",
                    "value": 0,
                }
            ]
        },
        {"value_variable": "spi_2024_state_pension"},
    ],
)
def test_a_frame_without_period_metadata_still_refuses_a_mixed_row(overrides):
    frame = PeriodFrameStub()
    frame.metadata = {}
    adapter = UKFrameTargetAdapter(frame)
    registry, contract = _registry({"mixed": _dated_binding(**overrides)})
    with pytest.raises(MeasurementPeriodError, match="reads variables of other years"):
        materialize_target_bindings(adapter, registry, contract, period=2025)


def test_a_frame_without_period_metadata_binds_a_row_dated_throughout():
    frame = PeriodFrameStub()
    frame.metadata = {}
    adapter = UKFrameTargetAdapter(frame)
    assert adapter.frame_period is None
    adapter.set_column("person", "spi_2023_state_pension", [100.0, 200.0, 300.0])
    adapter.set_column("person", "spi_2023_income_tax", [0.0, 50.0, 10.0])
    registry, contract = _registry({"dated": _dated_binding()})

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert not materialized.skipped
    np.testing.assert_allclose(
        adapter.tables["person"]["test_dated"], [0.0, 200.0, 300.0]
    )


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize(
    "binding",
    [
        {"value_variable": "income_tax"},
        {"value_expression": "employment_income + income_tax"},
    ],
)
def test_a_frame_without_period_metadata_refuses_an_engine_value_no_name_dates(
    binding, injected
):
    """The engine computes an unstored value at the calibration period, so a
    row declaring another period cannot read it. A column the resolver
    injected is still the engine's, not the frame's."""

    frame = PeriodFrameStub()
    frame.metadata = {}
    adapter = UKFrameTargetAdapter(frame)
    if injected:
        adapter.set_column("person", "income_tax", [1.0, 2.0, 3.0])
    registry, contract = _registry({"undated": {**binding, "measurement_period": 2023}})
    with pytest.raises(MeasurementPeriodError, match="neither stored nor dated"):
        materialize_target_bindings(adapter, registry, contract, period=2025)


@pytest.mark.parametrize(
    "binding, expected",
    [
        ({"value_variable": "employment_income"}, [1_000.0, 0.0, 500.0]),
        (
            {
                "value_variable": "person_count",
                "filters": [
                    {"variable": "employment_income", "operator": ">", "value": 0}
                ],
            },
            [1.0, 0.0, 1.0],
        ),
        # Who a row counts is read at the frame, as at the frame's own period.
        (
            {
                "value_variable": "employment_income",
                "filters": [{"concept": "is_SP_age", "operator": "==", "value": True}],
            },
            [1_000.0, 0.0, 0.0],
        ),
    ],
)
def test_a_frame_without_period_metadata_reads_its_own_values_as_they_are(
    binding, expected
):
    frame = PeriodFrameStub()
    frame.metadata = {}
    adapter = UKFrameTargetAdapter(frame)
    adapter.set_column("person", "is_SP_age", [True, False, False])
    registry, contract = _registry({"stored": {**binding, "measurement_period": 2023}})

    materialized = materialize_target_bindings(adapter, registry, contract, period=2025)

    assert not materialized.skipped
    np.testing.assert_allclose(adapter.tables["person"]["test_stored"], expected)


def test_the_dated_measures_a_binding_names_are_found_in_any_field():
    binding = {
        "kind": "count_affected",
        "affected_flag_variable": "spi_2023_state_pension",
        "value_reduction": {"spi_2023_income_tax": "sum"},
        "terms": [["x", "spi_2023_total_income * 2"]],
        "note": "not spi_2023_state_pensions, nor x_spi_2023_income_tax",
        "count": 3,
    }
    assert uk_spi_dated_measures_named(binding) == {
        "spi_2023_state_pension",
        "spi_2023_income_tax",
        "spi_2023_total_income",
    }
    assert uk_spi_dated_measures_named({"value_variable": "state_pension"}) == set()


def test_a_persisted_dated_measure_refuses(monkeypatch, tmp_path):
    from microcosm.build.uk_runtime import measure_simulation

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="2.162.8", Microsimulation=lambda **kw: _SimulationStub()
        ),
    )
    frame = PeriodFrameStub()
    frame.table("person")["spi_2023_state_pension"] = 1.0
    with pytest.raises(ValueError, match="must not be persisted"):
        measure_simulation.UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
        )


# The contract rows ---------------------------------------------------------


def test_the_pilot_rows_bind_2023_24_as_published():
    """Invariant: no target-side projection on these rows."""

    targets = {
        target["target_id"]: target
        for target in _uk_resource("uk_population_targets.json")["targets"]
    }
    for target_id in PILOT_ROWS:
        target = targets[target_id]
        binding = target["bindings"]["policyengine"]
        assert "uprating_index" not in target, target_id
        assert binding["measurement_period"] == 2023
        assert binding["require_matching_fact_period"] is True
        assert target["ledger_selector"]["period_value"] == 2023
        assert target["ledger_selector"]["period_type"] == "tax_year"
        names = _binding_names(binding)
        assert all(uk_spi_dated_measure_year(name) == 2023 for name in names), names
        assert binding["groupby_variable"] == "spi_2023_total_income"
        assert {"variable": "spi_2023_income_tax", "operator": ">", "value": 0} in (
            binding["filters"]
        )
    references = [
        reference
        for reference in _uk_resource("target_references.json")["target_references"]
        if reference["metadata"].get("contract_target_id") in PILOT_ROWS
    ]
    # 13 published bands for each of the four tables.
    assert len(references) == 4 * len(HMRC_SPI_INCOME_BAND_LOWER_BOUNDS)
    for reference in references:
        assert "uprating_index" not in reference, reference["name"]
        assert reference["metadata"]["measurement_period"] == "2023"
        assert reference["metadata"]["source_period_policy"] == "exact_observation"
