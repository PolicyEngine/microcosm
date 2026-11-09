"""HMRC's Income Tax liabilities projections as a differential check.

The invariants: a cell's expected count is the calendar window of its two
tax years; the cells, and each model count, do not depend on the order of
the rows or the people; the band cells, with the taxpayers below the first
edge, partition all taxpayers, and the area cells partition residents; a sum
is in sample only when its in-sample parts fix its model count; a cell is
within tolerance exactly when its relative miss is no larger than the
tolerance fixed before any file is read. The in-sample flags read the
targets by the names the contract's references and a run's diagnostics give
them, so Table 2.2's areas, bound since microcosm#1069, are in sample. Each
bound cell counts exactly the people the build's own materialization of its
target counts: the binding's filters, the band the build reads from the
compiled spec, and the reference's geography. So an edit to a binding's
filter, band variable or band reading fails these tests until the check
follows it, and an edit to any other key that counts fails the test that
holds the bindings to what the check reads.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import re
from fractions import Fraction
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import load_country_spec
from microcosm.build.ledger_targets import (
    CALENDAR_YEAR_WINDOW_WEIGHTS,
    _ledger_metadata,
    _reference_metadata,
)
from microcosm.build.target_materialization import (
    WHOLE_PENCE_UPPER_CLOSED,
    materialize_target_bindings,
)
from microcosm.build.uk_runtime import itl_differential
from microcosm.build.uk_runtime.hmrc_uprating import (
    ITL_BANDS,
    UK_HMRC_ITL_UK_GEOGRAPHY_ID,
)
from microcosm.build.uk_runtime.itl_differential import (
    AGE,
    AGE_MEASURES,
    ALL_TAXPAYERS,
    BAND_COUNT,
    BELOW_STATE_PENSION_AGE,
    BOUND_BY,
    IS_SP_AGE,
    NATIONAL_TOLERANCE,
    REGIONAL_TOLERANCE,
    RESIDENTS,
    TAX,
    TOTAL_INCOME,
    ITLPeople,
    calendar_window,
    cell_mask,
    differential,
    expected_cells,
    itl_band_key,
    model_count,
    residual,
    resolved_people,
    table_of,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource
from microcosm.build.uk_runtime.ledger_targets import UKFrameTargetAdapter
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.geography_constants import (
    UK_GEOGRAPHY_ID_TO_LABEL,
    UK_REGION_TIER,
    UK_REGION_TIER_ENUM,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from test_support.paths import paths_for

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
                "2.5",
                "total_taxpayer_count",
                UK_HMRC_ITL_UK_GEOGRAPHY_ID,
                year,
                40 * value * scale,
            )
        )
        for index, band in enumerate(ITL_BANDS):
            rows.append(
                _row(
                    "2.5",
                    "total_taxpayer_count",
                    UK_HMRC_ITL_UK_GEOGRAPHY_ID,
                    year,
                    (index + 1) * value * scale,
                    band,
                )
            )
        if ages:
            for measure in ("state_pension_age", "age_65_and_over"):
                rows.append(
                    _row(
                        "2.1",
                        measure,
                        UK_HMRC_ITL_UK_GEOGRAPHY_ID,
                        year,
                        13 * value * scale,
                    )
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
        _row("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2024, then),
        _row("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2025, now),
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
    lonely = _row("2.1", "males", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2025, 1.0)
    with pytest.raises(ValueError, match="window at 2025 needs"):
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
    all_taxpayers = cells[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)].expected
    state_pension_age = cells[
        ("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)
    ].expected
    assert (
        cells[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)].expected
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
            # Every part bound: the all-band count's bands, the UK cell's
            # areas, and the two counts the one below State Pension age is
            # the difference of. ``differential`` keeps the mark only when the
            # file holds nobody outside the parts.
            ("2.5", ALL_TAXPAYERS),
            ("2.1", "state_pension_age"),
            ("2.1", BELOW_STATE_PENSION_AGE),
        }
        assert cell.in_sample is bound_family, cell


def test_the_all_band_count_is_in_sample_only_with_its_bands() -> None:
    unbound = expected_cells(
        _rows(), year=2025, bound_targets={"hmrc.itl.state_pension_age_taxpayers"}
    )
    assert not any(cell.in_sample for cell in unbound if cell.table == "2.5")


def test_a_family_is_in_sample_cell_by_cell() -> None:
    """A band or area the run left out is out of sample, and so is each sum
    that counts it; its bound siblings stay in."""

    band_family = "hmrc.itl.taxpayers_by_total_income_band"
    age_family = "hmrc.itl.state_pension_age_taxpayers"
    left_out_band, left_out_area = ITL_BANDS[3], AREAS[5]
    bound = {
        *(f"{band_family}.{itl_band_key(b)}" for b in ITL_BANDS if b != left_out_band),
        *(f"{age_family}@{a}" for a in AREAS if a != left_out_area),
    }
    cells = expected_cells(_rows(), year=2025, bound_targets=bound)
    flags = {(cell.table, cell.measure, cell.key): cell.in_sample for cell in cells}
    for band in ITL_BANDS:
        key = ("2.5", BAND_COUNT, cells_key_of_band(cells, band))
        assert flags[key] is (band != left_out_band)
    for area in AREAS:
        assert flags[("2.2", "state_pension_age", area)] is (area != left_out_area)
    assert not flags[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
    assert not flags[("2.2", "state_pension_age", RESIDENTS)]
    assert not flags[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]

    every = {
        *(f"{band_family}.{itl_band_key(b)}" for b in ITL_BANDS),
        *(f"{age_family}@{a}" for a in AREAS),
    }
    whole = expected_cells(_rows(), year=2025, bound_targets=every)
    family = expected_cells(_rows(), year=2025, bound_targets={band_family, age_family})
    assert whole == family


def _person(**overrides):
    """One weighted taxpayer of State Pension age in the first area and band."""

    person = {
        "weight": [1.0],
        "taxpayer": [True],
        "total_income": [float(ITL_BANDS[0][0])],
        "state_pension_age": [True],
        "age": [70.0],
        "area": [AREAS[0]],
    }
    person.update(overrides)
    return {name: np.asarray(values) for name, values in person.items()}


def _flags(report):
    return {(row["table"], row["measure"], row["key"]): row for row in report["cells"]}


def test_a_sum_is_in_sample_only_when_its_parts_count_everyone_it_counts() -> None:
    """Binding every band fixes the all-band count only if the file holds no
    taxpayer below the lowest edge; binding every area fixes the UK count
    only if everyone it counts lives in one of the twelve."""

    bound = {
        "hmrc.itl.taxpayers_by_total_income_band",
        "hmrc.itl.state_pension_age_taxpayers",
    }
    cells = expected_cells(_rows(), year=2025, bound_targets=bound)
    everyone_in_a_part = ITLPeople(**_person())
    flags = _flags(differential(cells, everyone_in_a_part))
    for key in (
        ("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID),
        ("2.2", "state_pension_age", RESIDENTS),
        ("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID),
        ("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID),
    ):
        assert flags[key]["in_sample"], key
        assert flags[key]["outside_parts"] == 0.0, key

    def merged(*people):
        return ITLPeople(
            **{
                name: np.concatenate([person[name] for person in people])
                for name in people[0]
            }
        )

    below_the_first_edge = merged(
        _person(), _person(total_income=[10_000.0], weight=[2.5])
    )
    flags = _flags(differential(cells, below_the_first_edge))
    assert not flags[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["in_sample"]
    assert (
        flags[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["outside_parts"]
        == 2.5
    )
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]
    assert flags[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["in_sample"]

    in_no_area = merged(_person(), _person(area=[""], weight=[4.0]))
    flags = _flags(differential(cells, in_no_area))
    assert not flags[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]
    assert (
        flags[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
            "outside_parts"
        ]
        == 4.0
    )
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]
    # The residents' count is exactly its areas', whoever else the file holds.
    assert flags[("2.2", "state_pension_age", RESIDENTS)]["in_sample"]
    assert flags[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["in_sample"]


def test_a_family_is_in_sample_only_when_every_cell_is() -> None:
    """The family summary counts its cells in sample; leaving out the lowest
    band or a higher one gives the same summary."""

    family = "hmrc.itl.taxpayers_by_total_income_band"
    for left_out in (ITL_BANDS[0], ITL_BANDS[3]):
        bound = {f"{family}.{itl_band_key(b)}" for b in ITL_BANDS if b != left_out}
        report = differential(
            expected_cells(_rows(), year=2025, bound_targets=bound),
            ITLPeople(**_person(total_income=[20_000.0])),
        )
        summary = report["families"][f"2.5:{BAND_COUNT}"]
        assert summary["cells"] == len(ITL_BANDS)
        assert summary["in_sample_cells"] == len(ITL_BANDS) - 1
        assert summary["in_sample"] is False


def cells_key_of_band(cells, band):
    (key,) = {cell.key for cell in cells if cell.band == band}
    return key


def test_itl_band_key_names_the_band_as_the_targets_do() -> None:
    assert itl_band_key((12_570, 15_000)) == "band_12570_15000"
    assert itl_band_key((2_000_000, None)) == "band_2000000_plus"
    assert len({itl_band_key(band) for band in ITL_BANDS}) == len(ITL_BANDS)


def test_rows_of_another_geography_are_not_read() -> None:
    """Table 2.5's and 2.1's cells read the UK; Table 2.2's the twelve areas.
    A row elsewhere, even one that would refuse the window, changes nothing."""

    elsewhere = [
        # One tax year only, and a band no cell has: either refuses if read.
        _row("2.5", "total_taxpayer_count", AREAS[0], 2025, 1.0),
        _row("2.5", "total_taxpayer_count", AREAS[0], 2025, 1.0, (15_000, 17_500)),
        _row("2.1", "state_pension_age", AREAS[0], 2025, 1.0),
        _row("2.2", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2025, 1.0),
        _row("2.2", "age_65_and_over", "E06000001", 2025, 1.0),
    ]
    assert expected_cells([*_rows(), *elsewhere], year=2025) == expected_cells(
        _rows(), year=2025
    )


def test_table_of_names_the_table_or_none() -> None:
    assert table_of(
        _row("2.5", "total_taxpayer_count", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2025, 1)
    ) == ("2.5")
    assert table_of(_row("2.2", "males", AREAS[0], 2025, 1)) == "2.2"
    assert table_of({"layout": {"record_set_id": "hmrc.spi.table_3_6"}}) == "3.6"
    assert table_of({"layout": {"record_set_id": "ons.mid_year_2025"}}) is None
    assert table_of({}) is None


def test_a_band_outside_itl_bands_is_refused() -> None:
    rows = _rows(ages=False)
    rows += [
        _row(
            "2.5",
            "total_taxpayer_count",
            UK_HMRC_ITL_UK_GEOGRAPHY_ID,
            year,
            1.0,
            (15_000, 17_500),
        )
        for year in (2024, 2025)
    ]
    with pytest.raises(ValueError, match=r"bands \['\(15000, 17500\)'\]"):
        expected_cells(rows, year=2025)


def test_rows_no_cell_reads_do_not_refuse_the_window() -> None:
    """A measure the check does not read may carry one tax year only."""

    lonely = [
        _row("2.1", "males", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2025, 1.0),
        _row("2.5", "total_income_amount", UK_HMRC_ITL_UK_GEOGRAPHY_ID, 2025, 1.0),
    ]
    assert expected_cells([*_rows(), *lonely], year=2025) == expected_cells(
        _rows(), year=2025
    )


def test_the_vendored_table_2_5_bands_sum_to_its_all_band_count() -> None:
    """The bands partition Table 2.5's taxpayers to within HMRC's rounding."""

    rows = [
        row
        for row in load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
        if table_of(row) == "2.5" and row["measure_id"] == "total_taxpayer_count"
    ]
    years = sorted({row["period"]["value"] for row in rows})
    assert years
    for year in years:
        of_year = [row for row in rows if row["period"]["value"] == year]
        bands = {_band_of(row): row["value"] for row in of_year if _band_of(row)}
        (total,) = [row["value"] for row in of_year if _band_of(row) is None]
        assert set(bands) == set(ITL_BANDS)
        assert math.fsum(bands.values()) == pytest.approx(total, rel=0.005), year


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
    total = model_count(
        cells[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)], people
    )
    assert math.fsum([*bands, below_first_edge]) == pytest.approx(
        total, rel=1e-12, abs=1e-9
    )
    for measure in ("state_pension_age", "age_65_and_over"):
        areas = [model_count(cells[("2.2", measure, area)], people) for area in AREAS]
        residents = model_count(cells[("2.2", measure, RESIDENTS)], people)
        assert math.fsum(areas) == pytest.approx(residents, rel=1e-12, abs=1e-9)
    spa = model_count(
        cells[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)], people
    )
    below = model_count(
        cells[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)], people
    )
    assert math.fsum([spa, below]) == pytest.approx(total, rel=1e-12, abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(people=_people(), every_band=st.booleans())
def test_a_sum_in_sample_is_fixed_by_bound_parts(people, every_band) -> None:
    """Whenever a sum is in sample, its model count is its in-sample parts':
    the sum of the bands, the areas, or the difference of the two counts."""

    bands = ITL_BANDS if every_band else ITL_BANDS[1:]
    bound = {
        *(f"hmrc.itl.taxpayers_by_total_income_band.{itl_band_key(b)}" for b in bands),
        "hmrc.itl.state_pension_age_taxpayers",
    }
    report = differential(
        expected_cells(_rows(), year=2025, bound_targets=bound), people
    )
    rows = _flags(report)
    model = {key: row["model"] for key, row in rows.items()}
    total = rows[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
    if total["in_sample"]:
        assert every_band
        band_models = [m for (t, ms, _k), m in model.items() if ms == BAND_COUNT]
        assert math.fsum(band_models) == pytest.approx(
            total["model"], rel=1e-12, abs=1e-9
        )
    spa = rows[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
    if spa["in_sample"]:
        areas = [model[("2.2", "state_pension_age", area)] for area in AREAS]
        assert math.fsum(areas) == pytest.approx(spa["model"], rel=1e-12, abs=1e-9)
    below = rows[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
    if below["in_sample"]:
        assert total["in_sample"] and spa["in_sample"]
    families: dict[str, list] = {}
    for row in report["cells"]:
        families.setdefault(f"{row['table']}:{row['measure']}", []).append(row)
    assert sorted(families) == sorted(report["families"])
    for name, members in families.items():
        summary = report["families"][name]
        assert summary["cells"] == len(members)
        assert summary["in_sample_cells"] == sum(row["in_sample"] for row in members)
        assert summary["in_sample"] is all(row["in_sample"] for row in members)
        assert summary["misses"] == sum(not row["within_tolerance"] for row in members)
        assert summary["largest_abs_residual"] == max(
            abs(row["residual"]) for row in members
        )


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
            UK_HMRC_ITL_UK_GEOGRAPHY_ID,
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
    total = cells[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
    # Table 2.5's all-band count: 38.6m in 2024-25 and 39.8m in 2025-26.
    assert total.expected == pytest.approx(0.25 * 38_600_000 + 0.75 * 39_800_000)
    assert sum(1 for key in cells if key[1] == BAND_COUNT) == len(ITL_BANDS)


#: Numbers as the tolerance note spells them.
_NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
_MEASURE_WORDS = {
    "state_pension_age": "State Pension age",
    "age_65_and_over": "aged 65 and over",
}


def _tolerance_note() -> str:
    """The ``#:`` note above ``NATIONAL_TOLERANCE`` in the module's source,
    as one line."""

    lines = Path(itl_differential.__file__).read_text(encoding="utf-8").splitlines()
    end = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("NATIONAL_TOLERANCE = ")
    )
    start = end
    while lines[start - 1].startswith("#:"):
        start -= 1
    return " ".join(line.removeprefix("#:").strip() for line in lines[start:end])


def test_the_tolerance_note_cites_the_vendored_counts() -> None:
    """The note on ``NATIONAL_TOLERANCE`` says what the vendored rows give at
    the calendar-2025 window, and this test reads the note: how many bands
    hold under 400,000 taxpayers and, in band order, each one's count; how
    many of the top bands hold fewer than the smallest area cell; and that
    cell's area, measure and count. A change to the rows fails here until
    the note follows it. Every band takes the national tolerance all the
    same: geography, not size, decides."""

    note = _tolerance_note()
    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
    cells = expected_cells(rows, year=2025)
    bands = [cell for cell in cells if cell.measure == BAND_COUNT]
    assert [cell.band for cell in bands] == list(ITL_BANDS)
    under = [cell.expected for cell in bands if cell.expected < 400_000]
    assert (
        f"{_NUMBER_WORDS[len(under)]} of the {_NUMBER_WORDS[len(bands)]} bands "
        "hold under 400,000 taxpayers"
    ) in note
    assert re.findall(r"([\d,]+) at £", note) == [f"{count:,.0f}" for count in under]
    smallest = min(
        (cell for cell in cells if cell.tolerance == REGIONAL_TOLERANCE),
        key=lambda cell: cell.expected,
    )
    fewer = [cell for cell in bands if cell.expected < smallest.expected]
    assert fewer == bands[len(bands) - len(fewer) :]
    assert (
        f"the top {_NUMBER_WORDS[len(fewer)]} hold fewer than the smallest area cell"
    ) in note
    assert (
        f"({UK_GEOGRAPHY_ID_TO_LABEL[smallest.key]}'s "
        f"{_MEASURE_WORDS[smallest.measure]} count, {smallest.expected:,.0f})"
    ) in note
    assert {cell.tolerance for cell in bands} == {NATIONAL_TOLERANCE}


# The tool ------------------------------------------------------------------


@pytest.fixture(scope="module")
def tool():
    path = paths_for("microcosm-build").repository / "tools/check_uk_itl_projections.py"
    spec = importlib.util.spec_from_file_location("uk_itl_projections_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_run_binds_the_targets_its_diagnostics_weight(tool, tmp_path) -> None:
    """Each weighted row of the year binds by its name without the period;
    the cells then read the run's own names, one band or area at a time."""

    band_family = "hmrc.itl.taxpayers_by_total_income_band"
    age_family = "hmrc.itl.state_pension_age_taxpayers"
    rows = [
        {"name": f"{band_family}.{itl_band_key(band)}@2025", "target_loss_weight": 1}
        for band in ITL_BANDS[1:]
    ]
    rows += [
        # Dropped from the loss: out of sample however many siblings bind.
        {
            "name": f"{band_family}.{itl_band_key(ITL_BANDS[0])}@2025",
            "target_loss_weight": 0,
        },
        # No weight recorded: bound.
        {"name": f"{age_family}@{AREAS[0]}@2025"},
        {
            "name": "hmrc.spi.state_pension.count_by_total_income_band.band_x@2025",
            "target_loss_weight": 0.5,
        },
        # Structured identity, as the diagnostics write it.
        {
            "name": "anything",
            "target_name": f"{age_family}@{AREAS[1]}",
            "period": 2025,
            "target_loss_weight": 1,
        },
        # Another year, or none, binds nothing at 2025.
        {"name": f"{age_family}@{AREAS[2]}@2024", "target_loss_weight": 1},
        {
            "name": f"{age_family}@{AREAS[3]}@2025",
            "target_name": f"{age_family}@{AREAS[3]}",
            "period": 2024,
        },
        {"name": "unperioded", "target_loss_weight": 1},
    ]
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(json.dumps({"targets": rows}))
    bound = tool.run_bound_targets(diagnostics, 2025)
    assert bound == {
        *(f"{band_family}.{itl_band_key(band)}" for band in ITL_BANDS[1:]),
        f"{age_family}@{AREAS[0]}",
        f"{age_family}@{AREAS[1]}",
        "hmrc.spi.state_pension.count_by_total_income_band.band_x",
    }
    assert tool.run_bound_targets(diagnostics, 2024) == {
        f"{age_family}@{AREAS[2]}",
        f"{age_family}@{AREAS[3]}",
    }
    flags = {
        (cell.table, cell.measure, cell.band, cell.area): cell.in_sample
        for cell in expected_cells(_rows(), year=2025, bound_targets=bound)
    }
    assert [flags[("2.5", BAND_COUNT, band, None)] for band in ITL_BANDS] == [
        False,
        *[True] * (len(ITL_BANDS) - 1),
    ]
    assert not flags[("2.5", ALL_TAXPAYERS, None, None)]
    assert flags[("2.2", "state_pension_age", None, AREAS[0])]
    assert flags[("2.2", "state_pension_age", None, AREAS[1])]
    assert not flags[("2.2", "state_pension_age", None, AREAS[2])]


class _Resolver:
    """Person columns a calibrated file would give, without the engine."""

    def __init__(self, columns):
        self.columns = columns

    def compute(self, entity, variable):
        assert entity == "person"
        return np.asarray(self.columns[variable]), "stub"


def _consistent_rows():
    """Table 2.5 rows whose bands sum to the all-band count in every year."""

    rows = []
    for year in (2023, 2024, 2025):
        for index, band in enumerate(ITL_BANDS):
            rows.append(
                _row(
                    "2.5",
                    "total_taxpayer_count",
                    UK_HMRC_ITL_UK_GEOGRAPHY_ID,
                    year,
                    (index + 1) * 1_000.0,
                    band,
                )
            )
        total = math.fsum((index + 1) * 1_000.0 for index in range(len(ITL_BANDS)))
        rows.append(
            _row(
                "2.5", "total_taxpayer_count", UK_HMRC_ITL_UK_GEOGRAPHY_ID, year, total
            )
        )
    return rows


def _band_people(weight):
    """One taxpayer at each band's lower edge."""

    size = len(ITL_BANDS)
    edges = np.array([float(band[0]) for band in ITL_BANDS])
    return _Resolver(
        {
            "household_weight": weight,
            TAX: np.ones(size),
            TOTAL_INCOME: edges,
            IS_SP_AGE: np.zeros(size, dtype=bool),
            AGE: np.full(size, 40.0),
            "region": np.array(["LONDON"] * size),
        }
    )


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("off_by", [1.0, 1.2])
def test_strict_exits_one_exactly_on_a_miss(
    tool, tmp_path, monkeypatch, strict, off_by
) -> None:
    weight = np.array([(index + 1) * 1_000.0 for index in range(len(ITL_BANDS))])
    weight[3] *= off_by
    monkeypatch.setattr(tool, "measure_resolver", lambda *a, **k: _band_people(weight))
    h5 = tmp_path / "national.h5"
    h5.write_bytes(b"not read")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"rows": _consistent_rows()}))
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "hmrc.itl.taxpayers_by_total_income_band."
                        f"{itl_band_key(band)}@2025",
                        "target_loss_weight": 1.0,
                        # The solve's own estimate: 1% above the file's count.
                        "final_estimate": 1.01 * weight[index],
                    }
                    for index, band in enumerate(ITL_BANDS)
                ]
            }
        )
    )
    out = tmp_path / "report.json"
    argv = [
        "--h5",
        str(h5),
        "--out",
        str(out),
        "--itl-rows",
        str(rows),
        "--diagnostics",
        str(diagnostics),
    ]
    code = tool.main([*argv, "--strict"] if strict else argv)

    receipt = json.loads(out.read_text())
    missed = off_by != 1.0
    assert code == (1 if strict and missed else 0)
    assert bool(receipt["misses"]) is missed
    assert (
        receipt["in_sample_basis"]
        == f"{diagnostics} sha256:{tool._sha256(diagnostics)}"
    )
    assert all(cell["in_sample"] for cell in receipt["cells"])
    bands = [cell for cell in receipt["cells"] if cell["measure"] == BAND_COUNT]
    assert all(
        cell["model_vs_run_estimate"] == pytest.approx(1 / 1.01 - 1) for cell in bands
    )
    assert all(
        cell["model_minus_run_estimate"]
        == pytest.approx(cell["model"] - cell["run_estimate"])
        for cell in bands
    )
    (total,) = [cell for cell in receipt["cells"] if cell["measure"] == ALL_TAXPAYERS]
    assert total["reference"] is None and total["run_estimate"] is None
    assert receipt["run_estimates"]["cells"] == len(ITL_BANDS)


# The names the contract and a run give the bound targets ----------------------


def _resource(name):
    return json.loads(
        files("microcosm.build.uk").joinpath(name).read_text(encoding="utf-8")
    )


def _bound_references():
    """Every target reference of a family ``BOUND_BY`` reads, by its target."""

    references: dict[str, set[str]] = {}
    for row in _resource("target_references.json")["target_references"]:
        target = (row.get("metadata") or {}).get("contract_target_id")
        if target in BOUND_BY.values():
            references.setdefault(target, set()).add(row["name"])
    return references


def test_bound_by_names_the_contract_targets_and_their_references() -> None:
    """Each family ``BOUND_BY`` reads is a contract target, and its references
    are named as a cell reads them: a band ``<target>.band_<lower>_<upper>``,
    an area ``<target>@<GSS code>``. A run's diagnostics name each row by its
    reference (``target_name``), so a rename the cells did not follow would
    leave bound cells out of sample."""

    contract = {
        row["target_id"]: row
        for row in _resource("uk_population_targets.json")["targets"]
    }
    family = {"2.5": "hmrc_itl", "2.2": "hmrc_itl_age"}
    references = _bound_references()
    for (table, measure), target in BOUND_BY.items():
        assert contract[target]["family"] == family[table], target
        if table == "2.5":
            assert measure == BAND_COUNT
            names = {f"{target}.{itl_band_key(band)}" for band in ITL_BANDS}
        else:
            names = {f"{target}@{area}" for area in AREAS}
        assert references[target] == names, target
    assert {measure for table, measure in BOUND_BY if table == "2.2"} == set(
        AGE_MEASURES
    )


def test_the_references_mark_the_bound_tables_in_sample(tool) -> None:
    """On the vendored rows at calendar 2025 the tree's references bind every
    Table 2.5 band and every Table 2.2 area, so each is in sample, and so is
    each sum of them until a file holds someone outside its parts. Table
    2.1's United Kingdom cells are in sample through their areas only: no
    target binds them. The references bind at 2025 alone, so at another year
    nothing is in sample."""

    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
    for year in (2024, 2026):
        other = expected_cells(
            rows, year=year, bound_targets=tool.reference_bound_targets(None, year)
        )
        assert not any(cell.in_sample for cell in other), year
    bound = tool.reference_bound_targets(None, 2025)
    assert {cell.reference for cell in expected_cells(rows, year=2025)} - {None} <= (
        bound
    )
    cells = expected_cells(rows, year=2025, bound_targets=bound)
    by_table: dict[str, list] = {}
    for cell in cells:
        by_table.setdefault(cell.table, []).append(cell)
    assert len(by_table["2.5"]) == len(ITL_BANDS) + 1
    assert len(by_table["2.2"]) == len(AGE_MEASURES) * (len(AREAS) + 1)
    assert len(by_table["2.1"]) == len(AGE_MEASURES) + 1
    assert all(cell.in_sample for cell in cells)
    assert all(cell.parts for cell in by_table["2.1"])
    assert not any(cell.parts for cell in by_table["2.2"] if cell.area != RESIDENTS)


def test_the_bound_cells_expect_what_the_build_compiles() -> None:
    """Two implementations of the calendar window agree: each bound cell's
    expected count is the value the build's Ledger compile gives its target
    at 2025, as the committed parity receipt records it (compiled from the
    pinned Chronicle feed the vendored rows were copied from)."""

    receipt = _resource("ledger_compile_parity_incumbent_2025_signed_differences.json")
    compiled = {
        row["name"]: row["ledger_value"]
        for row in receipt["differences"]
        if row.get("period") == 2025 and row.get("ledger_value") is not None
    }
    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
    checked = []
    for cell in expected_cells(rows, year=2025):
        target = BOUND_BY.get((cell.table, cell.measure))
        if target is None or cell.parts:
            assert cell.reference is None
            continue
        name = cell.reference
        assert cell.expected == pytest.approx(compiled[name], rel=1e-12), name
        checked.append(name)
    assert sorted(checked) == sorted(
        name for names in _bound_references().values() for name in names
    )


# Whom the contract's bindings count -------------------------------------------


def _contract():
    """The contract's targets by id."""

    return {
        row["target_id"]: row
        for row in _resource("uk_population_targets.json")["targets"]
    }


#: The keys of a ``policyengine`` binding that change no count: ``notes``
#: documents the row, and ``metric_name`` names a counterfactual binding's
#: column, which these plain bindings do not have.
_INERT_KEYS = frozenset({"metric_name", "notes"})


def test_the_check_reads_what_the_contract_binds() -> None:
    """The three families ``BOUND_BY`` reads bind exactly what the check
    counts: ``person_count`` at person grain, mapped to the household (the
    check gives each person its household's weight), of people whose ``TAX``
    exceeds zero. The bands group on ``TOTAL_INCOME`` along the
    ``total_income`` filter dimension and declare no band semantics, upper
    bound or period factor, so the build reads each band as ``[lower,
    upper)`` between the vendored lower edges. The age families add
    ``IS_SP_AGE`` true or ``AGE`` at least 65, and band nothing. Every key
    but ``notes`` and ``metric_name`` is held, so a binding that gains or
    changes any other key fails here. That covers the keys whose effect the
    differential below cannot see on these rows: ``band_filter_dimension``
    beside a single band-like filter, ``measurement_period`` on a frame
    without period metadata, and ``require_matching_fact_period``."""

    taxpayer = {"variable": TAX, "operator": ">", "value": 0}
    counted = {
        "value_variable": "person_count",
        "from_entity": "person",
        "map_to": "household",
    }
    expected = {
        ("2.5", BAND_COUNT): {
            **counted,
            "filters": [taxpayer],
            "groupby_variable": TOTAL_INCOME,
            "band_filter_dimension": "total_income",
        },
        ("2.2", "state_pension_age"): {
            **counted,
            "filters": [taxpayer, {"variable": IS_SP_AGE, "equals": True}],
        },
        ("2.2", "age_65_and_over"): {
            **counted,
            "filters": [taxpayer, {"variable": AGE, "operator": ">=", "value": 65}],
        },
    }
    assert set(expected) == set(BOUND_BY)
    contract = _contract()
    for family, target in BOUND_BY.items():
        binding = contract[target]["bindings"]["policyengine"]
        held = {key: value for key, value in binding.items() if key not in _INERT_KEYS}
        assert held == expected[family], target


def _edge_people():
    """Engine columns for people at every edge the bindings read, and each
    person's household region: tax either side of zero; total income at each
    band edge, a penny either side of it and 0.4p below it (where a reading
    in whole pence moves a person across the edge); either State Pension age
    flag; ages either side of 65; in each of the twelve region-tier areas
    and in none."""

    edges = [float(lower) for lower, _upper in ITL_BANDS]
    steps = (-0.01, -0.004, 0.0, 0.01)
    incomes = sorted(
        {0.0, 5_000_000.0, *(edge + step for edge in edges for step in steps)}
    )
    grid = itertools.product(
        (*UK_REGION_TIER_ENUM.values(), "UNKNOWN"),
        (-1.0, 0.0, 0.01, 1.0),
        incomes,
        (False, True),
        (64.0, 64.99, 65.0, 65.01, 90.0),
    )
    region, tax, income, sp_age, age = (
        np.array(column) for column in zip(*grid, strict=True)
    )
    return {TAX: tax, TOTAL_INCOME: income, IS_SP_AGE: sp_age, AGE: age}, region


def _edge_frame(engine, region):
    """The edge people as a frame, each in a household of their own."""

    ids = np.arange(len(region), dtype="int64")
    return Frame(
        {
            "person": pd.DataFrame(
                {"person_id": ids, "person_household_id": ids, **engine}
            ),
            "household": pd.DataFrame({"household_id": ids, "region": region}),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )


def _window_facts(rows, reference):
    """The vendored facts ``reference`` selects in the calendar-2025 window:
    its geography, measure and dimension values, at each tax year the
    window weights."""

    selector = reference.ledger_selector
    wanted = {
        key: str(value)
        for key, value in (selector.get("dimension_values") or {}).items()
    }
    years = {2025 + offset for offset in CALENDAR_YEAR_WINDOW_WEIGHTS}
    return [
        row
        for row in rows
        if row["geography"]["id"] == selector["geography_id"]
        and row["measure_id"] == selector["source_measure_id"]
        and row["period"]["value"] in years
        and all(
            str(row["dimensions"].get(key)) == value for key, value in wanted.items()
        )
    ]


def _bound_registry():
    """The compiled spec of each bound reference, stamped as the Ledger
    compile stamps it: the compile's Ledger metadata of a fact the reference
    selects (the fact's dimensions give the band's edges, as
    ``ledger_filter_total_income_lower_bound``), under the compile's
    metadata of the reference (its contract target and geography predicate
    among it). The window's facts carry the same dimensions, so either one
    stamps the spec."""

    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
    specs = []
    for reference in load_country_spec("uk").target_references:
        if reference.metadata.get("contract_target_id") not in BOUND_BY.values():
            continue
        facts = _window_facts(rows, reference)
        assert len(facts) == len(CALENDAR_YEAR_WINDOW_WEIGHTS), reference.name
        assert all(fact["dimensions"] == facts[0]["dimensions"] for fact in facts), (
            reference.name
        )
        metadata = {
            **_ledger_metadata(facts[0], fact_key=facts[0]["source_record_id"]),
            **_reference_metadata(reference),
        }
        specs.append(
            TargetSpec(
                name=reference.name,
                entity=reference.entity,
                measure=reference.measure,
                value=1.0,
                period=reference.period,
                source="vendored",
                metadata=metadata,
            )
        )
    return TargetRegistry(specs, country="uk")


def _readings(contract):
    """Each bound cell (a Table 2.5 band or a Table 2.2 area) on
    ``_edge_people``, with what each person counts for in it: the check's
    reading, and what ``materialize_target_bindings`` writes for the cell's
    reference from ``contract`` on a ``UKFrameTargetAdapter`` (None when it
    skips the reference). The build reads each band from the compiled spec
    and scopes each area by its reference's geography predicate, as a UK
    calibration does."""

    engine, region = _edge_people()
    size = len(region)
    people = resolved_people(
        _Resolver({**engine, "household_weight": np.ones(size), "region": region})
    )
    registry = _bound_registry()
    adapter = UKFrameTargetAdapter(_edge_frame(engine, region))
    result = materialize_target_bindings(adapter, registry, contract, period=2025)
    skipped = {skip.name for skip in result.skipped}
    specs = {spec.name: spec for spec in registry.specs}
    for cell in expected_cells(_rows(), year=2025):
        if cell.reference is None:
            continue
        build = None
        if cell.reference not in skipped:
            spec = specs[cell.reference]
            build = np.asarray(adapter.column(spec.entity, spec.measure), dtype=float)
        yield cell, cell_mask(cell, people).astype(float), build


def test_each_bound_cell_counts_whom_its_binding_counts() -> None:
    """Two readings of each bound family agree: on people at every edge, each
    Table 2.5 band and Table 2.2 area counts, person by person, what the
    build's own materialization of its contract target counts, band edges
    and geography included; the build skips none of them; and each counts
    someone and leaves someone out."""

    readings = list(_readings(_contract()))
    assert len(readings) == len(ITL_BANDS) + len(AGE_MEASURES) * len(AREAS)
    for cell, check, build in readings:
        assert build is not None, cell.reference
        assert np.array_equal(check, build), cell.reference
        assert 0 < check.sum() < len(check), cell.reference


_BAND_TARGET = BOUND_BY[("2.5", BAND_COUNT)]
_STATE_PENSION_AGE_TARGET = BOUND_BY[("2.2", "state_pension_age")]
_AGED_65_AND_OVER_TARGET = BOUND_BY[("2.2", "age_65_and_over")]
_TOP_BAND = f"{_BAND_TARGET}.{itl_band_key(ITL_BANDS[-1])}"

#: Edits a later change could make to a binding, by name: the binding's
#: target, the edit, and the references it turns red (None: every reference
#: of the target).
_BINDING_EDITS = {
    "taxpayers-from-zero": (
        _BAND_TARGET,
        lambda binding: binding["filters"][0].update(operator=">="),
        None,
    ),
    "another-tax-variable": (
        _STATE_PENSION_AGE_TARGET,
        lambda binding: binding["filters"][0].update(variable="income_tax"),
        None,
    ),
    "bands-on-tax": (
        _BAND_TARGET,
        lambda binding: binding.update(groupby_variable=TAX),
        None,
    ),
    "bands-on-another-variable": (
        _BAND_TARGET,
        lambda binding: binding.update(groupby_variable="total_income"),
        None,
    ),
    "no-band-variable": (
        _BAND_TARGET,
        lambda binding: binding.pop("groupby_variable"),
        None,
    ),
    "bands-in-whole-pence": (
        _BAND_TARGET,
        lambda binding: binding.update(band_semantics=WHOLE_PENCE_UPPER_CLOSED),
        None,
    ),
    "a-capped-top-band": (
        _BAND_TARGET,
        lambda binding: binding.update(
            band_upper_bound=3_000_000, band_upper_bound_inclusive=False
        ),
        {_TOP_BAND},
    ),
    "monthly-band-edges": (
        _BAND_TARGET,
        lambda binding: binding.update(band_period_factor=12),
        None,
    ),
    "a-provider-kind": (
        _BAND_TARGET,
        lambda binding: binding.update(kind="parameter_gated_threshold"),
        None,
    ),
    "below-state-pension-age": (
        _STATE_PENSION_AGE_TARGET,
        lambda binding: binding["filters"][1].update(equals=False),
        None,
    ),
    "another-state-pension-age-flag": (
        _STATE_PENSION_AGE_TARGET,
        lambda binding: binding["filters"][1].update(variable="is_female"),
        None,
    ),
    "aged-66-and-over": (
        _AGED_65_AND_OVER_TARGET,
        lambda binding: binding["filters"][1].update(value=66),
        None,
    ),
    "aged-over-65": (
        _AGED_65_AND_OVER_TARGET,
        lambda binding: binding["filters"][1].update(operator=">"),
        None,
    ),
    "any-age": (
        _AGED_65_AND_OVER_TARGET,
        lambda binding: binding["filters"].pop(),
        None,
    ),
    "an-added-filter": (
        _STATE_PENSION_AGE_TARGET,
        lambda binding: binding["filters"].append(
            {"variable": AGE, "operator": ">=", "value": 70}
        ),
        None,
    ),
}


@pytest.mark.parametrize("edit", sorted(_BINDING_EDITS))
def test_an_edited_binding_turns_its_cells_red(edit) -> None:
    """The differential reads the bindings themselves, through the build: an
    edit to a filter, the band variable, the band reading (whole pence, an
    upper bound or a period factor) or the binding's kind makes exactly the
    cells it moves read differently from the check, or the build skip them.
    That is every cell of the binding's family, except for an upper bound
    above the top band's lower edge, which moves the top band alone."""

    target, change, red = _BINDING_EDITS[edit]
    contract = copy.deepcopy(_contract())
    change(contract[target]["bindings"]["policyengine"])
    if red is None:
        red = {
            cell.reference
            for cell in expected_cells(_rows(), year=2025)
            if cell.reference is not None
            and BOUND_BY[(cell.table, cell.measure)] == target
        }
    assert {
        cell.reference
        for cell, check, build in _readings(contract)
        if build is None or not np.array_equal(check, build)
    } == red


def _diagnostics(tmp_path, *, left_out=frozenset()):
    """A run's calibration_diagnostics.json as a UK build writes its target
    rows: each bound reference at 2025, a reference in ``left_out`` with no
    loss weight."""

    rows = [
        {
            "name": f"{name}@2025",
            "target_name": name,
            "period": 2025,
            "target_loss_weight": 0.0 if name in left_out else 1.0,
        }
        for names in _bound_references().values()
        for name in sorted(names)
    ]
    path = tmp_path / "calibration_diagnostics.json"
    path.write_text(json.dumps({"targets": rows}))
    return path


def _area_people(*extra):
    """A taxpayer of State Pension age in each area, one younger taxpayer,
    all in one band, and ``extra`` people."""

    people = [
        _person(area=[area], total_income=[20_000.0], weight=[1.0 + index])
        for index, area in enumerate(AREAS)
    ]
    people.append(
        _person(
            area=[AREAS[0]],
            total_income=[40_000.0],
            state_pension_age=[False],
            age=[40.0],
        )
    )
    people.extend(extra)
    return ITLPeople(
        **{
            name: np.concatenate([person[name] for person in people])
            for name in people[0]
        }
    )


def test_a_run_that_binds_table_2_2_marks_its_cells_in_sample(tool, tmp_path) -> None:
    """With every band and area weighted in the run, a file whose taxpayers
    all live in an area and sit in a band has every cell in sample. A
    taxpayer in no area takes Table 2.1's United Kingdom counts out, and the
    count below State Pension age with them, but not the residents; an area
    the run left out takes itself and each sum over it out, and leaves the
    other age group in."""

    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]

    def report(people, **left_out):
        bound = tool.run_bound_targets(_diagnostics(tmp_path, **left_out), 2025)
        return differential(
            expected_cells(rows, year=2025, bound_targets=bound), people
        )

    whole = report(_area_people())
    assert all(row["in_sample"] for row in whole["cells"])
    assert all(
        row["outside_parts"] == 0.0
        for row in whole["cells"]
        if row["outside_parts"] is not None
    )
    assert all(family["in_sample"] for family in whole["families"].values())

    abroad = _person(area=[""], total_income=[20_000.0], weight=[3.0])
    flags = _flags(report(_area_people(abroad)))
    for measure in AGE_MEASURES:
        assert not flags[("2.1", measure, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["in_sample"]
        assert (
            flags[("2.1", measure, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["outside_parts"] == 3.0
        )
        assert flags[("2.2", measure, RESIDENTS)]["in_sample"]
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]
    assert flags[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["in_sample"]

    area = AREAS[6]
    target = BOUND_BY[("2.2", "state_pension_age")]
    left = report(_area_people(), left_out=frozenset({f"{target}@{area}"}))
    flags = _flags(left)
    assert not flags[("2.2", "state_pension_age", area)]["in_sample"]
    assert not flags[("2.2", "state_pension_age", RESIDENTS)]["in_sample"]
    assert not flags[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]
    assert all(
        flags[("2.2", "state_pension_age", other)]["in_sample"]
        for other in AREAS
        if other != area
    )
    assert left["families"]["2.2:age_65_and_over"]["in_sample"]
    assert not left["families"]["2.2:state_pension_age"]["in_sample"]
    assert flags[("2.1", "age_65_and_over", UK_HMRC_ITL_UK_GEOGRAPHY_ID)]["in_sample"]


def test_resolved_people_reads_the_area_from_the_household_region() -> None:
    """A region-tier enum name, with or without its class prefix, gives the
    area's GSS code; any other value gives no area."""

    people = resolved_people(
        _Resolver(
            {
                "household_weight": np.array([1.0, 2.0, 3.0]),
                TAX: np.array([1.0, 0.0, 5.0]),
                TOTAL_INCOME: np.array([20_000.0, 5_000.0, 60_000.0]),
                IS_SP_AGE: np.array([True, False, False]),
                AGE: np.array([70.0, 30.0, 50.0]),
                "region": np.array(["Region.LONDON", "WALES", "UNKNOWN"]),
            }
        )
    )
    assert people.area.tolist() == ["E12000007", "W92000004", ""]
    assert people.taxpayer.tolist() == [True, False, True]
    assert people.weight.tolist() == [1.0, 2.0, 3.0]


# The guards and edges ---------------------------------------------------------


def test_without_table_2_2_the_united_kingdom_counts_have_no_parts() -> None:
    """Rows that carry Table 2.1 but not Table 2.2 (another ``--itl-rows``
    feed) give the United Kingdom counts no parts, so they are out of sample
    even with the age families bound, and the count below State Pension age
    with them."""

    rows = [row for row in _rows() if table_of(row) != "2.2"]
    cells = expected_cells(rows, year=2025, bound_targets=set(BOUND_BY.values()))
    by_key = {(cell.table, cell.measure, cell.key): cell for cell in cells}
    for measure in AGE_MEASURES:
        national = by_key[("2.1", measure, UK_HMRC_ITL_UK_GEOGRAPHY_ID)]
        assert national.parts == ()
        assert not national.in_sample
    assert not by_key[
        ("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)
    ].in_sample
    assert by_key[("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)].in_sample
    flags = _flags(differential(cells, ITLPeople(**_person())))
    assert not flags[("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]


def test_a_sum_given_without_one_of_its_parts_is_out_of_sample() -> None:
    """``differential`` handed a sum but not every part (one band left out of
    the cells) cannot show that the parts fix it."""

    bound = {
        "hmrc.itl.taxpayers_by_total_income_band",
        "hmrc.itl.state_pension_age_taxpayers",
    }
    cells = expected_cells(_rows(), year=2025, bound_targets=bound)
    people = ITLPeople(**_person())
    total = ("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID)
    assert _flags(differential(cells, people))[total]["in_sample"]
    without = [cell for cell in cells if cell.band != ITL_BANDS[3]]
    flags = _flags(differential(without, people))
    assert not flags[total]["in_sample"]
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID)][
        "in_sample"
    ]


def test_the_cells_read_the_bindings_at_their_edges() -> None:
    """Each mask is its binding at the edges, pinned by hand: a band holds its
    lower edge and not its upper; 65 is in the count aged 65 and over and
    64.9 is not; only taxpayers count; State Pension age is the engine's
    flag, not an age; the residents are the twelve areas and no other code.
    ``test_each_bound_cell_counts_whom_its_binding_counts`` reads the
    bindings themselves, through the build's materialization."""

    people = ITLPeople(
        weight=np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        taxpayer=np.array([True, True, True, True, False, True, True]),
        total_income=np.array(
            [12_570.0, 15_000.0, 12_569.99, 20_000.0, 20_000.0, 20_000.0, 20_000.0]
        ),
        state_pension_age=np.array([False, False, False, True, True, False, True]),
        age=np.array([65.0, 64.9, 70.0, 60.0, 70.0, 66.0, 70.0]),
        area=np.array(
            [AREAS[0], AREAS[0], AREAS[0], AREAS[1], AREAS[1], "", "E92000001"]
        ),
    )
    cells = {
        (cell.table, cell.measure, cell.key): cell
        for cell in expected_cells(_rows(), year=2025)
    }

    def count(*key):
        return model_count(cells[key], people)

    assert count("2.5", BAND_COUNT, "12570-15000") == 1.0
    assert count("2.5", BAND_COUNT, "15000-20000") == 2.0
    assert count("2.5", BAND_COUNT, "20000-30000") == 8.0 + 32.0 + 64.0
    assert count("2.5", ALL_TAXPAYERS, UK_HMRC_ITL_UK_GEOGRAPHY_ID) == 111.0
    assert (
        count("2.1", "age_65_and_over", UK_HMRC_ITL_UK_GEOGRAPHY_ID)
        == 1.0 + 4.0 + 32.0 + 64.0
    )
    assert count("2.1", "state_pension_age", UK_HMRC_ITL_UK_GEOGRAPHY_ID) == 8.0 + 64.0
    assert count("2.1", BELOW_STATE_PENSION_AGE, UK_HMRC_ITL_UK_GEOGRAPHY_ID) == 39.0
    assert count("2.2", "age_65_and_over", AREAS[0]) == 1.0 + 4.0
    assert count("2.2", "state_pension_age", AREAS[1]) == 8.0
    assert count("2.2", "state_pension_age", RESIDENTS) == 8.0
    assert count("2.2", "age_65_and_over", RESIDENTS) == 1.0 + 4.0


@given(model=st.floats(0.0, 1e8), expected=_counts)
def test_no_residual_lands_on_a_tolerance(model, expected) -> None:
    """The verdict never turns on ``<`` against ``<=``: every residual is a
    multiple of 2**-53 (exact below 0.5 in magnitude, and the float spacing
    is at least that above), and neither tolerance is one."""

    for tolerance in (NATIONAL_TOLERANCE, REGIONAL_TOLERANCE):
        assert Fraction(tolerance).denominator > 2**53
    miss = residual(model, expected)
    assert (Fraction(miss) * 2**53).denominator == 1
    assert abs(miss) not in (NATIONAL_TOLERANCE, REGIONAL_TOLERANCE)


# The tool's inputs ------------------------------------------------------------


def _consumer_fact(row):
    """The Chronicle consumer fact a vendored row copies."""

    kept = (
        "entity",
        "geography",
        "period",
        "period_coverage",
        "layout",
        "dimensions",
        "source",
        "value",
    )
    return {
        "aggregate_fact_key": row["aggregate_fact_key"],
        "semantic_fact_key": row["semantic_fact_key"],
        "lineage": {"source_record_id": row["source_record_id"]},
        "observed_measure": {
            "source_concept": row["concept"],
            "source_measure_id": row["measure_id"],
            "unit": row["unit"],
        },
        "aggregation": {"method": row["aggregation"]},
        **{key: row[key] for key in kept},
    }


def test_a_jsonl_of_consumer_facts_reads_as_the_vendored_rows(tool, tmp_path) -> None:
    """``--itl-rows`` with a JSONL of the consumer facts the vendored rows copy
    gives those rows exactly; blank and whitespace-only lines are skipped."""

    rows = load_vendored_resource("hmrc_itl_taxpayer_counts.json")["rows"]
    lines = [json.dumps(_consumer_fact(row)) for row in rows]
    path = tmp_path / "facts.jsonl"
    path.write_text("\n".join([*lines[:3], "   ", "", *lines[3:]]) + "\n")
    read, source = tool.load_rows(path)
    assert read == rows
    assert source == f"{path} sha256:{tool._sha256(path)}"


def test_a_zero_run_estimate_still_shows_its_difference(tool) -> None:
    """A run that estimates a bound cell at zero gives it no relative
    difference, but its count difference and the summary keep it."""

    cells = [
        {"reference": "a", "model": 10.0},
        {"reference": "b", "model": 0.0},
        {"reference": "c", "model": 12.0},
        {"reference": "d", "model": 7.0},
        {"reference": None, "model": 29.0},
    ]
    estimates = {"a": 0.0, "b": 0.0, "c": 10.0}
    report = tool.with_run_estimates({"cells": cells}, estimates)
    by_reference = {cell["reference"]: cell for cell in report["cells"]}
    assert by_reference["a"]["model_minus_run_estimate"] == 10.0
    assert by_reference["a"]["model_vs_run_estimate"] is None
    assert by_reference["b"]["model_minus_run_estimate"] == 0.0
    assert by_reference["c"]["model_vs_run_estimate"] == pytest.approx(0.2)
    for reference in ("d", None):
        assert by_reference[reference]["run_estimate"] is None
        assert by_reference[reference]["model_minus_run_estimate"] is None
    assert report["run_estimates"] == {
        "cells": 3,
        "zero_estimates": 2,
        "largest_abs_difference": pytest.approx(0.2),
        "largest_abs_count_difference": 10.0,
    }

    # The reviewer's case: the only estimate is zero beside a model count of 10.
    report = tool.with_run_estimates({"cells": cells[:1]}, estimates)
    assert report["run_estimates"] == {
        "cells": 1,
        "zero_estimates": 1,
        "largest_abs_difference": None,
        "largest_abs_count_difference": 10.0,
    }


_count = st.floats(min_value=0, max_value=1e9, allow_nan=False)


@settings(max_examples=60, deadline=None)
@given(
    models=st.lists(_count, max_size=12),
    estimates=st.dictionaries(st.integers(0, 15).map(str), _count, max_size=12),
)
def test_every_estimated_cell_counts_in_the_run_summary(
    tool, models, estimates
) -> None:
    cells = [
        {"reference": str(index), "model": model} for index, model in enumerate(models)
    ]
    report = tool.with_run_estimates({"cells": cells}, estimates)
    estimated = [cell for cell in report["cells"] if cell["run_estimate"] is not None]
    assert [cell["reference"] for cell in estimated] == [
        cell["reference"] for cell in cells if cell["reference"] in estimates
    ]
    for cell in estimated:
        assert cell["model_minus_run_estimate"] == cell["model"] - cell["run_estimate"]
        assert (cell["model_vs_run_estimate"] is None) is (cell["run_estimate"] == 0)
    summary = report["run_estimates"]
    assert summary["cells"] == len(estimated)
    assert summary["zero_estimates"] == sum(
        cell["run_estimate"] == 0 for cell in estimated
    )
    assert summary["largest_abs_count_difference"] == max(
        (abs(cell["model_minus_run_estimate"]) for cell in estimated), default=None
    )


def test_rows_without_a_loss_weight_bind_and_are_named(tool, tmp_path) -> None:
    """A row the diagnostics writer left without a loss weight binds, and the
    tool names it so the in-sample basis can say so."""

    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "targets": [
                    {"name": "a@2025", "target_loss_weight": 1.0, "final_estimate": 3},
                    {"name": "b@2025"},
                    {"name": "c@2025", "target_loss_weight": 0.0},
                    {"name": "d@2024", "final_estimate": 5},
                ]
            }
        )
    )
    assert tool.run_bound_targets(diagnostics, 2025) == {"a", "b"}
    assert tool.run_unweighted_targets(diagnostics, 2025) == {"b"}
    assert tool.run_estimates(diagnostics, 2025) == {"a": 3.0}


def test_the_basis_reports_rows_bound_without_a_weight(
    tool, tmp_path, monkeypatch
) -> None:
    weight = np.array([(index + 1) * 1_000.0 for index in range(len(ITL_BANDS))])
    monkeypatch.setattr(tool, "measure_resolver", lambda *a, **k: _band_people(weight))
    h5 = tmp_path / "national.h5"
    h5.write_bytes(b"not read")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"rows": _consistent_rows()}))
    diagnostics = tmp_path / "calibration_diagnostics.json"
    names = [
        f"hmrc.itl.taxpayers_by_total_income_band.{itl_band_key(band)}@2025"
        for band in ITL_BANDS
    ]
    diagnostics.write_text(
        json.dumps(
            {
                "targets": [
                    {"name": names[0]},
                    *({"name": name, "target_loss_weight": 1.0} for name in names[1:]),
                ]
            }
        )
    )
    out = tmp_path / "report.json"
    argv = ["--h5", str(h5), "--out", str(out), "--itl-rows", str(rows)]
    assert tool.main([*argv, "--diagnostics", str(diagnostics)]) == 0
    receipt = json.loads(out.read_text())
    assert receipt["in_sample_basis"].endswith(
        "; 1 rows at 2025 carry no target_loss_weight and bind"
    )
    assert all(cell["in_sample"] for cell in receipt["cells"])
    assert receipt["run_estimates"] == {
        "cells": 0,
        "zero_estimates": 0,
        "largest_abs_difference": None,
        "largest_abs_count_difference": None,
    }

    references = tmp_path / "target_references.json"
    references.write_text(json.dumps({"target_references": []}))
    with pytest.raises(SystemExit):
        tool.main(
            [*argv, "--diagnostics", str(diagnostics), "--references", str(references)]
        )
    assert tool.main([*argv, "--references", str(references)]) == 0
    receipt = json.loads(out.read_text())
    assert receipt["in_sample_basis"] == f"target references {references} at 2025"
    assert not any(cell["in_sample"] for cell in receipt["cells"])
    assert "run_estimates" not in receipt


def test_the_current_income_route_needs_a_basis(
    tool, tmp_path, monkeypatch, capsys
) -> None:
    """``--current-income-for-annual-measures`` reads a file built before the
    annual income descriptors, whose calibration this tree's references do
    not describe: on their basis every band would read in sample. Without
    ``--diagnostics`` or ``--references`` the tool refuses the option before
    it reads anything, and it refuses an empty reason, which would turn the
    route off; with a basis it passes the reason on, and the receipt records
    it. Without the option, this tree's references stay the default basis
    and the receipt records no reason."""

    reasons = []
    weight = np.array([(index + 1) * 1_000.0 for index in range(len(ITL_BANDS))])

    def resolver(*args, reason=None, **kwargs):
        reasons.append(reason)
        return _band_people(weight)

    monkeypatch.setattr(tool, "measure_resolver", resolver)
    h5 = tmp_path / "incumbent.h5"
    h5.write_bytes(b"not read")
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps({"rows": _consistent_rows()}))
    out = tmp_path / "report.json"
    argv = ["--h5", str(h5), "--out", str(out), "--itl-rows", str(rows)]
    incumbent = [*argv, "--current-income-for-annual-measures", "incumbent"]

    references = tmp_path / "target_references.json"
    references.write_text(json.dumps({"target_references": []}))
    for refused_argv, message in (
        (incumbent, "needs --diagnostics or --references"),
        (
            [
                *argv,
                "--current-income-for-annual-measures",
                " ",
                "--references",
                str(references),
            ],
            "needs a reason",
        ),
    ):
        with pytest.raises(SystemExit) as refused:
            tool.main(refused_argv)
        assert refused.value.code == 2
        assert message in capsys.readouterr().err
    assert reasons == [] and not out.exists()

    assert tool.main(argv) == 0
    receipt = json.loads(out.read_text())
    assert receipt["in_sample_basis"] == "target references of this tree at 2025"
    assert receipt["current_income_for_annual_measures"] is None
    assert all(cell["in_sample"] for cell in receipt["cells"])

    assert tool.main([*incumbent, "--references", str(references)]) == 0
    receipt = json.loads(out.read_text())
    assert receipt["in_sample_basis"] == f"target references {references} at 2025"
    assert receipt["current_income_for_annual_measures"] == "incumbent"
    assert not any(cell["in_sample"] for cell in receipt["cells"])
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(json.dumps({"targets": []}))
    assert tool.main([*incumbent, "--diagnostics", str(diagnostics)]) == 0
    assert not any(cell["in_sample"] for cell in json.loads(out.read_text())["cells"])
    assert reasons == [None, "incumbent", "incumbent"]
