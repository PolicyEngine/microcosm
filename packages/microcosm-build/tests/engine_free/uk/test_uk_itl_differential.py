"""HMRC's Income Tax liabilities projections as a differential check.

The invariants: a cell's expected count is the calendar window of its two
tax years; the cells, and each model count, do not depend on the order of
the rows or the people; the band cells and the area cells partition what
their totals count; a cell is within tolerance exactly when its relative
miss is no larger than the tolerance fixed before any file is read.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime.hmrc_uprating import ITL_BANDS
from microcosm.build.uk_runtime.itl_differential import (
    ALL_TAXPAYERS,
    BAND_COUNT,
    BELOW_STATE_PENSION_AGE,
    NATIONAL_TOLERANCE,
    REGIONAL_TOLERANCE,
    RESIDENTS,
    UNITED_KINGDOM,
    ITLPeople,
    calendar_window,
    differential,
    expected_cells,
    model_count,
    residual,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource
from microcosm.calibrate.geography_constants import UK_REGION_TIER

AREAS = tuple(code for _level, code in UK_REGION_TIER)
_counts = st.floats(min_value=1.0, max_value=5e7, allow_nan=False)


def _row(table, measure, geography, year, value, band=None):
    dimensions = {}
    if band is not None:
        dimensions["total_income_lower_bound"] = band[0]
        if band[1] is not None:
            dimensions["total_income_upper_bound"] = band[1]
    return {
        "source_record_id": f"{table}.{measure}.{geography}.{band}.{year}",
        "measure_id": measure,
        "geography": {"id": geography},
        "period": {"type": "tax_year", "value": year},
        "layout": {
            "record_set_id": f"hmrc.itl_2026.table_{table.replace('.', '_')}.ty{year}"
        },
        "dimensions": dimensions,
        "value": value,
    }


def _rows(*, ages=True, value=1_000.0):
    rows = []
    for year in (2023, 2024, 2025, 2026):
        scale = 1.0 + (year - 2023) / 10
        rows.append(
            _row(
                "2.5", "total_taxpayer_count", UNITED_KINGDOM, year, 40 * value * scale
            )
        )
        for index, band in enumerate(ITL_BANDS):
            rows.append(
                _row(
                    "2.5",
                    "total_taxpayer_count",
                    UNITED_KINGDOM,
                    year,
                    (index + 1) * value * scale,
                    band,
                )
            )
        if ages:
            for measure in ("state_pension_age", "age_65_and_over"):
                rows.append(
                    _row("2.1", measure, UNITED_KINGDOM, year, 13 * value * scale)
                )
                for index, area in enumerate(AREAS):
                    rows.append(
                        _row(
                            "2.2", measure, area, year, (index + 1) * value * scale / 6
                        )
                    )
    return rows


@given(then=_counts, now=_counts)
def test_the_window_weights_the_two_tax_years(then, now) -> None:
    rows = [
        _row("2.1", "state_pension_age", UNITED_KINGDOM, 2024, then),
        _row("2.1", "state_pension_age", UNITED_KINGDOM, 2025, now),
    ]
    ((value, sources),) = calendar_window(rows, year=2025).values()
    assert value == math.fsum((0.25 * then, 0.75 * now))
    assert len(sources) == 2


@settings(max_examples=25, deadline=None)
@given(order=st.permutations(range(len(_rows()))))
def test_the_cells_do_not_depend_on_the_order_of_the_rows(order) -> None:
    rows = _rows()
    assert expected_cells(
        [rows[index] for index in order], year=2025
    ) == expected_cells(rows, year=2025)


def test_a_duplicate_or_half_window_is_refused() -> None:
    rows = _rows()
    with pytest.raises(ValueError, match="two rows"):
        calendar_window([*rows, rows[0]], year=2024)
    lonely = _row("2.1", "males", UNITED_KINGDOM, 2025, 1.0)
    with pytest.raises(ValueError, match="window needs"):
        calendar_window([*rows, lonely], year=2025)


def test_every_table_2_5_band_and_the_all_band_count_are_required() -> None:
    rows = _rows(ages=False)
    for dropped in (ITL_BANDS[3], None):
        kept = [
            row
            for row in rows
            if not (
                row["measure_id"] == "total_taxpayer_count" and _band_of(row) == dropped
            )
        ]
        with pytest.raises(ValueError, match="no Income Tax liabilities rows"):
            expected_cells(kept, year=2025)


def _band_of(row):
    dimensions = row["dimensions"]
    if "total_income_lower_bound" not in dimensions:
        return None
    return dimensions["total_income_lower_bound"], dimensions.get(
        "total_income_upper_bound"
    )


def test_a_table_2_2_measure_must_carry_every_area() -> None:
    rows = [row for row in _rows() if row["geography"]["id"] != AREAS[4]]
    with pytest.raises(ValueError, match="Table 2.2 state_pension_age has no rows"):
        expected_cells(rows, year=2025)


def test_the_age_tables_enter_only_when_present() -> None:
    without = expected_cells(_rows(ages=False), year=2025)
    assert {cell.table for cell in without} == {"2.5"}
    assert len(without) == len(ITL_BANDS) + 1
    with_ages = expected_cells(_rows(), year=2025)
    # Per age measure: the UK row, twelve areas and their sum; and below SPA.
    assert len(with_ages) == len(without) + 2 * (1 + len(AREAS) + 1) + 1


def test_the_derived_cells_are_their_components() -> None:
    cells = {
        (cell.table, cell.measure, cell.key): cell
        for cell in expected_cells(_rows(), year=2025)
    }
    all_taxpayers = cells[("2.5", ALL_TAXPAYERS, UNITED_KINGDOM)].expected
    state_pension_age = cells[("2.1", "state_pension_age", UNITED_KINGDOM)].expected
    assert (
        cells[("2.1", BELOW_STATE_PENSION_AGE, UNITED_KINGDOM)].expected
        == all_taxpayers - state_pension_age
    )
    for measure in ("state_pension_age", "age_65_and_over"):
        areas = [cells[("2.2", measure, area)].expected for area in AREAS]
        assert cells[("2.2", measure, RESIDENTS)].expected == math.fsum(areas)


def test_tolerances_and_in_sample_flags() -> None:
    bound = {
        "hmrc.itl.taxpayers_by_total_income_band",
        "hmrc.itl.state_pension_age_taxpayers",
    }
    for cell in expected_cells(_rows(), year=2025, bound_targets=bound):
        regional = cell.table == "2.2" and cell.area != RESIDENTS
        assert cell.tolerance == (
            REGIONAL_TOLERANCE if regional else NATIONAL_TOLERANCE
        )
        bound_family = (cell.table, cell.measure) in {
            ("2.5", BAND_COUNT),
            ("2.2", "state_pension_age"),
        }
        assert cell.in_sample is bound_family, cell


@st.composite
def _people(draw):
    size = draw(st.integers(min_value=1, max_value=60))
    floats = st.floats(min_value=0.0, max_value=3e6, allow_nan=False)
    return ITLPeople(
        weight=np.array(
            draw(st.lists(st.floats(0.0, 5e3), min_size=size, max_size=size))
        ),
        taxpayer=np.array(draw(st.lists(st.booleans(), min_size=size, max_size=size))),
        total_income=np.array(draw(st.lists(floats, min_size=size, max_size=size))),
        state_pension_age=np.array(
            draw(st.lists(st.booleans(), min_size=size, max_size=size))
        ),
        age=np.array(
            draw(st.lists(st.integers(16, 100), min_size=size, max_size=size)),
            dtype=float,
        ),
        area=np.array(
            draw(st.lists(st.sampled_from([*AREAS, ""]), min_size=size, max_size=size))
        ),
    )


def _shuffled(people, order):
    return ITLPeople(
        **{
            name: np.asarray(getattr(people, name))[order]
            for name in people.__dataclass_fields__
        }
    )


@settings(max_examples=60, deadline=None)
@given(people=_people(), data=st.data())
def test_model_counts_do_not_depend_on_the_order_of_the_people(people, data) -> None:
    order = np.array(data.draw(st.permutations(range(len(people.weight)))))
    shuffled = _shuffled(people, order)
    for cell in expected_cells(_rows(), year=2025):
        assert model_count(cell, shuffled) == model_count(cell, people)


@settings(max_examples=60, deadline=None)
@given(people=_people())
def test_the_bands_and_the_areas_partition_their_totals(people) -> None:
    cells = {
        (cell.table, cell.measure, cell.key): cell
        for cell in expected_cells(_rows(), year=2025)
    }
    weight = people.weight
    below_first_edge = math.fsum(
        weight[people.taxpayer & (people.total_income < ITL_BANDS[0][0])].tolist()
    )
    bands = [
        model_count(cell, people) for key, cell in cells.items() if key[1] == BAND_COUNT
    ]
    total = model_count(cells[("2.5", ALL_TAXPAYERS, UNITED_KINGDOM)], people)
    assert math.fsum([*bands, below_first_edge]) == pytest.approx(
        total, rel=1e-12, abs=1e-9
    )
    for measure in ("state_pension_age", "age_65_and_over"):
        areas = [model_count(cells[("2.2", measure, area)], people) for area in AREAS]
        residents = model_count(cells[("2.2", measure, RESIDENTS)], people)
        assert math.fsum(areas) == pytest.approx(residents, rel=1e-12, abs=1e-9)
    spa = model_count(cells[("2.1", "state_pension_age", UNITED_KINGDOM)], people)
    below = model_count(cells[("2.1", BELOW_STATE_PENSION_AGE, UNITED_KINGDOM)], people)
    assert math.fsum([spa, below]) == pytest.approx(total, rel=1e-12, abs=1e-9)


@given(model=st.floats(0.0, 1e8), expected=_counts)
def test_the_verdict_is_the_tolerance(model, expected) -> None:
    people = ITLPeople(
        weight=np.array([model]),
        taxpayer=np.array([True]),
        total_income=np.array([20_000.0]),
        state_pension_age=np.array([False]),
        age=np.array([40.0]),
        area=np.array([AREAS[0]]),
    )
    rows = [
        row
        for row in _rows(ages=False)
        if row["dimensions"].get("total_income_lower_bound") != 20_000
    ]
    rows += [
        _row(
            "2.5",
            "total_taxpayer_count",
            UNITED_KINGDOM,
            year,
            expected,
            (20_000, 30_000),
        )
        for year in (2024, 2025)
    ]
    report = differential(expected_cells(rows, year=2025), people)
    (cell,) = [row for row in report["cells"] if row["key"] == "20000-30000"]
    assert cell["residual"] == residual(model, expected)
    assert cell["within_tolerance"] is (
        abs(model / expected - 1.0) <= NATIONAL_TOLERANCE
    )
    assert (cell["key"] in " ".join(report["misses"])) is not cell["within_tolerance"]


def test_the_report_is_deterministic() -> None:
    people = ITLPeople(
        weight=np.arange(1.0, 25.0),
        taxpayer=np.arange(24) % 3 != 0,
        total_income=np.linspace(5_000.0, 2_500_000.0, 24),
        state_pension_age=np.arange(24) % 2 == 0,
        age=np.linspace(20.0, 90.0, 24),
        area=np.array([AREAS[index % len(AREAS)] for index in range(24)]),
    )
    first = json.dumps(
        differential(expected_cells(_rows(), year=2025), people), sort_keys=True
    )
    second = json.dumps(
        differential(expected_cells(_rows()[::-1], year=2025), people), sort_keys=True
    )
    assert first == second


def test_the_vendored_table_2_5_rows_give_every_band_at_calendar_2025() -> None:
    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
    cells = {
        (cell.table, cell.measure, cell.key): cell
        for cell in expected_cells(rows, year=2025)
    }
    total = cells[("2.5", ALL_TAXPAYERS, UNITED_KINGDOM)]
    # Table 2.5's all-band count: 38.6m in 2024-25 and 39.8m in 2025-26.
    assert total.expected == pytest.approx(0.25 * 38_600_000 + 0.75 * 39_800_000)
    assert sum(1 for key in cells if key[1] == BAND_COUNT) == len(ITL_BANDS)
