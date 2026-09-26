"""Invented native-integration controls for the Table 1.1 compiler-only port."""

import pytest
from test_us_fiscal_targets import (
    _cbo_income_source_projection_fact,
    _soi_table_1_1_fact,
    _soi_table_1_1_totals,
    packaged_reference_facts,
)

from microcosm.build.us_runtime.fiscal_targets import compile_us_fiscal_target_registry


def _compile(*facts, age_targets=False):
    registry = compile_us_fiscal_target_registry(
        [*packaged_reference_facts(), *facts],
        target_period=2024,
        age_targets=age_targets,
        allow_unaged_dollar_targets=not age_targets,
    )
    return {spec.name: spec for spec in registry.specs}


def _name(fact):
    return fact["lineage"]["source_record_id"]


def test_native_soi_distribution_preserves_source_and_alignment_years():
    amount = _soi_table_1_1_fact(
        2022, income_range="10m_plus", lower=10_000_000, value=900_000_000_000
    )
    count = _soi_table_1_1_fact(
        2022,
        income_range="10m_plus",
        lower=10_000_000,
        measure_id="return_count",
        value=30_000,
    )
    specs = _compile(
        *_soi_table_1_1_totals(2022, returns=150_000_000, agi=15_000_000_000_000),
        *_soi_table_1_1_totals(2023, returns=160_000_000, agi=16_500_000_000_000),
        amount,
        count,
        _cbo_income_source_projection_fact(
            2023, "adjusted_gross_income", value=20_000_000_000_000
        ),
        _cbo_income_source_projection_fact(
            2024, "adjusted_gross_income", value=24_000_000_000_000
        ),
        age_targets=True,
    )
    target = specs[_name(amount)]
    assert target.value == pytest.approx(900_000_000_000 * 16.5 / 15 * 24 / 20)
    assert target.name == _name(amount)
    assert target.period == 2024
    assert target.metadata["source_period"] == "2023"
    assert target.metadata["ledger_fact_period"] == "2022"
    assert target.metadata["uprating_from_period"] == "2022"
    assert target.metadata["uprating_to_period"] == "2023"
    assert target.metadata["target_period"] == "2024"
    assert target.metadata["materializer"] == "irs_soi_slice"
    assert target.metadata["filing_status"] == "All"
    assert target.metadata["agi_lower_bound"] == "10000000.0"
    assert target.metadata["agi_upper_bound"] == "inf"
    assert specs[_name(count)].value == pytest.approx(30_000 * 160 / 150)
    assert specs[_name(count)].metadata["source_variable"] == "count"


def test_future_control_and_class_cannot_displace_eligible_vintage():
    current = _soi_table_1_1_fact(
        2023, income_range="10m_plus", lower=10_000_000, value=900_000_000_000
    )
    future = _soi_table_1_1_fact(
        2025, income_range="10m_plus", lower=10_000_000, value=2_000_000_000_000
    )
    specs = _compile(
        *_soi_table_1_1_totals(2023, returns=160_000_000, agi=16_000_000_000_000),
        *_soi_table_1_1_totals(2025, returns=200_000_000, agi=30_000_000_000_000),
        current,
        future,
    )
    assert specs[_name(current)].value == 900_000_000_000
    assert specs[_name(current)].metadata["uprating_to_period"] == "2023"
    assert _name(future) not in specs


@pytest.mark.parametrize("source_total", [None, 0])
def test_newer_control_cannot_supply_a_missing_or_zero_source_denominator(source_total):
    old_class = _soi_table_1_1_fact(
        2022, income_range="10m_plus", lower=10_000_000, value=900_000_000_000
    )
    source = (
        [] if source_total is None else [_soi_table_1_1_fact(2022, value=source_total)]
    )
    specs = _compile(
        *source,
        *_soi_table_1_1_totals(2023, returns=160_000_000, agi=16_000_000_000_000),
        old_class,
    )
    assert _name(old_class) not in specs


@pytest.mark.parametrize("measure", ["return_count", "adjusted_gross_income"])
@pytest.mark.parametrize(
    "other_control",
    [
        {"filing_status": "single"},
        {"geography_level": "state", "geography_id": "0400000US06"},
        {"table": "table_1_2"},
    ],
)
def test_unrelated_newer_control_does_not_rebase_all_returns(measure, other_control):
    band = _soi_table_1_1_fact(
        2022,
        income_range="10m_plus",
        lower=10_000_000,
        measure_id=measure,
        value=30_000 if measure == "return_count" else 900_000_000_000,
    )
    wrong_control = _soi_table_1_1_fact(
        2023,
        measure_id=measure,
        value=30_000_000_000_000,
        **other_control,
    )
    specs = _compile(
        *_soi_table_1_1_totals(2022, returns=150_000_000, agi=15_000_000_000_000),
        band,
        wrong_control,
    )
    assert specs[_name(band)].value == band["value"]
    assert specs[_name(band)].metadata["uprating_to_period"] == "2022"


def test_other_table_cannot_activate_through_size_distribution_rescue():
    # The held-out top-rate comparison has no new target path in this change.
    other = _soi_table_1_1_fact(
        2023,
        table="table_3_4",
        income_range="10m_plus",
        lower=10_000_000,
        value=900_000_000_000,
    )
    specs = _compile(
        *_soi_table_1_1_totals(2023, returns=160_000_000, agi=16_000_000_000_000),
        other,
    )
    assert _name(other) not in specs
