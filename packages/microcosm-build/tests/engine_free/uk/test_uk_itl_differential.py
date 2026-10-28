"""HMRC's Income Tax liabilities projections as a differential check.

The invariants: a cell's expected count is the calendar window of its two
tax years; the cells, and each model count, do not depend on the order of
the rows or the people; the band cells and the area cells partition what
their totals count; a cell is within tolerance exactly when its relative
miss is no larger than the tolerance fixed before any file is read.
"""

from __future__ import annotations

import importlib.util
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
    SINGLE_TAX_YEAR,
    TAX,
    TOTAL_INCOME,
    UNITED_KINGDOM,
    ITLPeople,
    calendar_window,
    differential,
    expected_cells,
    itl_band_key,
    model_count,
    residual,
    table_of,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource
from microcosm.calibrate.geography_constants import UK_REGION_TIER
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
            # The bands partition the all-band count, so binding them binds it.
            ("2.5", ALL_TAXPAYERS),
            ("2.2", "state_pension_age"),
            # The areas sum to the UK cell, and with the all-band count to
            # the count below State Pension age.
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
    assert not flags[("2.5", ALL_TAXPAYERS, UNITED_KINGDOM)]
    assert not flags[("2.2", "state_pension_age", RESIDENTS)]
    assert not flags[("2.1", "state_pension_age", UNITED_KINGDOM)]
    assert not flags[("2.1", BELOW_STATE_PENSION_AGE, UNITED_KINGDOM)]

    every = {
        *(f"{band_family}.{itl_band_key(b)}" for b in ITL_BANDS),
        *(f"{age_family}@{a}" for a in AREAS),
    }
    whole = expected_cells(_rows(), year=2025, bound_targets=every)
    family = expected_cells(_rows(), year=2025, bound_targets={band_family, age_family})
    assert whole == family


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
        _row("2.2", "state_pension_age", UNITED_KINGDOM, 2025, 1.0),
        _row("2.2", "age_65_and_over", "E06000001", 2025, 1.0),
    ]
    assert expected_cells([*_rows(), *elsewhere], year=2025) == expected_cells(
        _rows(), year=2025
    )


def test_table_of_names_the_table_or_none() -> None:
    assert table_of(_row("2.5", "total_taxpayer_count", UNITED_KINGDOM, 2025, 1)) == (
        "2.5"
    )
    assert table_of(_row("2.2", "males", AREAS[0], 2025, 1)) == "2.2"
    assert table_of({"layout": {"record_set_id": "hmrc.spi.table_3_6"}}) == "3.6"
    assert table_of({"layout": {"record_set_id": "ons.mid_year_2025"}}) is None
    assert table_of({}) is None


def test_a_band_outside_itl_bands_is_refused() -> None:
    rows = _rows(ages=False)
    rows += [
        _row("2.5", "total_taxpayer_count", UNITED_KINGDOM, year, 1.0, (15_000, 17_500))
        for year in (2024, 2025)
    ]
    with pytest.raises(ValueError, match=r"bands \['\(15000, 17500\)'\]"):
        expected_cells(rows, year=2025)


def test_rows_no_cell_reads_do_not_refuse_the_window() -> None:
    """A measure the check does not read may carry one tax year only."""

    lonely = [
        _row("2.1", "males", UNITED_KINGDOM, 2025, 1.0),
        _row("2.5", "total_income_amount", UNITED_KINGDOM, 2025, 1.0),
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
        cells = expected_cells(rows, year=year, weights=SINGLE_TAX_YEAR)
        bands = [cell.expected for cell in cells if cell.measure == BAND_COUNT]
        (total,) = [cell.expected for cell in cells if cell.measure == ALL_TAXPAYERS]
        assert len(bands) == len(ITL_BANDS)
        assert math.fsum(bands) == pytest.approx(total, rel=0.005), year


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


def test_one_tax_year_reads_its_own_column() -> None:
    """The dated check reads Table 2.5's 2023-24 outturn as published."""

    rows = _rows(ages=False)
    cells = expected_cells(rows, year=2023, weights=SINGLE_TAX_YEAR)
    published = {
        _band_of(row): row["value"] for row in rows if row["period"]["value"] == 2023
    }
    for cell in cells:
        assert cell.expected == published[cell.band]
        assert len(cell.sources) == 1


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
    """Each weighted row binds by its name without the period; the cells
    then read the run's own names, one band or area at a time."""

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
        {"name": "unperioded", "target_loss_weight": 1},
    ]
    diagnostics = tmp_path / "calibration_diagnostics.json"
    diagnostics.write_text(json.dumps({"targets": rows}))
    bound = tool.run_bound_targets(diagnostics)
    assert bound == {
        *(f"{band_family}.{itl_band_key(band)}" for band in ITL_BANDS[1:]),
        f"{age_family}@{AREAS[0]}",
        "hmrc.spi.state_pension.count_by_total_income_band.band_x",
        "unperioded",
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
    assert not flags[("2.2", "state_pension_age", None, AREAS[1])]


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
                    UNITED_KINGDOM,
                    year,
                    (index + 1) * 1_000.0,
                    band,
                )
            )
        total = math.fsum((index + 1) * 1_000.0 for index in range(len(ITL_BANDS)))
        rows.append(_row("2.5", "total_taxpayer_count", UNITED_KINGDOM, year, total))
    return rows


def _band_people(weight):
    """One taxpayer at each band's lower edge, in 2025 and in 2023."""

    size = len(ITL_BANDS)
    edges = np.array([float(band[0]) for band in ITL_BANDS])
    return _Resolver(
        {
            "household_weight": weight,
            TAX: np.ones(size),
            TOTAL_INCOME: edges,
            "is_SP_age": np.zeros(size, dtype=bool),
            "age": np.full(size, 40.0),
            "region": np.array(["LONDON"] * size),
            "spi_2023_person_count": np.ones(size),
            "spi_2023_income_tax": np.ones(size),
            "spi_2023_total_income": edges,
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
                    }
                    for band in ITL_BANDS
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
        "--dated-year",
        "2023",
    ]
    code = tool.main([*argv, "--strict"] if strict else argv)

    receipt = json.loads(out.read_text())
    missed = off_by != 1.0
    assert code == (1 if strict and missed else 0)
    assert bool(receipt["misses"]) is missed
    assert bool(receipt["dated"]["misses"]) is missed
    assert receipt["in_sample_basis"].startswith(f"{diagnostics} sha256:")
    assert all(cell["in_sample"] for cell in receipt["cells"])
    assert not any(cell["in_sample"] for cell in receipt["dated"]["cells"])
