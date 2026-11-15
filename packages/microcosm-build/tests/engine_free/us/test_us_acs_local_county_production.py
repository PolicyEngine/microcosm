"""The production county rung: ACS families, Kalawao, support and grades.

microcosm#922. Fixtures follow the pinned county facts' projected row shape:
ACS rows carry ``dimensions.statistic`` (estimate or margin_of_error, ids
ending E or M), B01003 splits its counties across two record sets, B19013
lacks a county, and IRS SOI county rows are tax-year all-return totals.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.calibrate import TargetRegistry, TargetSpec
from test_support.paths import paths_for

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_SETTINGS = settings(max_examples=40, deadline=None)


# The SOI fixtures of test_us_acs_local_county_surface.py, copied rather than
# imported: that module imports microcosm.build.us_runtime, whose package
# imports policyengine_us, and these tests need no engine.
def _state_spec(measure="return_count", value=100.0, *, state="01", **metadata):
    variable = "count" if measure == "return_count" else measure
    name = f"state.{state}.{measure}"
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="fixture state SOI",
        family="irs_soi",
        signed=value < 0,
        metadata={
            "ledger_geography_level": "state",
            "ledger_layout_record_set_spec_id": "irs_soi.historic_table_2.state_broad_totals.v1",
            "state_fips": state,
            "source_measure_id": measure,
            "source_period": "2022",
            "ledger_measure_unit": "count" if measure == "return_count" else "usd",
            "variable": variable,
            "source_variable": variable,
            "soi_variable": variable,
            "measure_mode": "sum",
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            **metadata,
        },
    )


def _fact(county, measure, value):
    return {
        "geography": {
            "id": f"0500000US{county}",
            "level": "county",
            "name": f"County {county}",
            "vintage": "2022",
        },
        "period": {"type": "tax_year", "value": 2022},
        "observed_measure": {
            "source_measure_id": measure,
            "source_concept": (
                "irs_soi.individual_income_tax_returns"
                if measure == "return_count"
                else "irs_soi.adjusted_gross_income"
            ),
            "source_name": "irs_soi",
            "unit": "count" if measure == "return_count" else "usd",
        },
        "layout": {
            "record_set_id": "irs_soi.ty2022.county_totals",
            "record_set_spec_id": "irs_soi.county_totals.v1",
            "measure_id": measure,
        },
        "dimensions": {"filing_status": "all", "income_range": "all"},
        "value": value,
    }


@lru_cache
def _county():
    """The county surface module alone.

    Loading it through the release tool imports policyengine_us, which these
    engine-free tests never need. The release imports the same module by the
    same name, so either route gives one module object.
    """
    tools = str(paths_for("microcosm-build").repository / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("us_acs_local_county_surface")


_TABLE_DIMENSIONS = {
    "B11001_001": {"household_type": "all"},
    "B11001_002": {"household_type": "family"},
    "B11001_007": {"household_type": "nonfamily"},
    "B19051_001": {"earnings_status": "all"},
    "B19051_002": {"earnings_status": "with_earnings"},
    "B19051_003": {"earnings_status": "without_earnings"},
    "B19001_001": {"income_range": "all"},
}


def _dimensions(measure):
    stem = measure[:-1]
    if stem in _TABLE_DIMENSIONS:
        return dict(_TABLE_DIMENSIONS[stem])
    for bracket, label, _lower, _upper in _county().B19001_BRACKETS:
        if bracket[:-1] == stem:
            return {"income_range": label}
    return {}


def _acs_fact(county, measure, value, *, group=0, name=None):
    """One projected ACS county row, in the pinned file's shape."""
    table = measure.split("_")[0].lower()
    statistic = "estimate" if measure.endswith("E") else "margin_of_error"
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{county}{measure}",
        "schema_version": "chronicle.consumer_fact.v5",
        "geography": {
            "id": f"0500000US{county}",
            "level": "county",
            "name": name or f"Fixture County {county}, Fixture State",
            "vintage": "2024",
        },
        "observed_measure": {
            "source_concept": f"{measure} ({statistic})",
            "source_measure_id": measure,
            "source_name": "census_acs",
            "source_table": f"ACS 2020-2024 5-year {table.upper()}",
            "unit": "USD" if table in {"b19013", "b19025", "b19061"} else "count",
        },
        "dimensions": {**_dimensions(measure), "statistic": statistic},
        "period": {"type": "calendar_year", "value": 2024},
        "value": value,
        "value_type": "integer",
        "layout": {
            "groupby_dimension": "census_acs.county",
            "groupby_value_id": county,
            "measure_id": measure,
            "record_set_id": f"census_acs.acs5_2020_2024.{table}.county.group_{group}",
            "record_set_spec_id": f"census_acs.acs5.{table}.county.v1",
            "source_column_id": measure,
            "source_row_id": county,
            "table_record_kind": "total",
        },
    }


def _county_values(code, seed=0):
    rng = np.random.default_rng(int(code) + 7919 * seed)
    households = int(rng.integers(50, 5_000))
    brackets = rng.multinomial(households, np.full(16, 1 / 16))
    return {
        "households": households,
        "earning_households": int(households * rng.uniform(0.5, 0.9)),
        "earnings": int(households * rng.uniform(20_000, 90_000)),
        "income": int(households * rng.uniform(30_000, 120_000)),
        "brackets": [int(value) for value in brackets],
        "median": int(rng.uniform(30_000, 120_000)),
        "population": int(households * rng.uniform(2.0, 3.0)),
    }


def _acs_rows(counties, *, values=None, without_median=(), b01003_group_1=()):
    """All seven ACS county tables for ``counties`` (five-digit strings).

    Margins of error ride beside most estimates; B01003 estimates sit in
    record set group_0 without margins, except ``b01003_group_1`` counties,
    which carry both in group_1. ``without_median`` counties have no B19013.
    """
    values = values or {}
    rows = []
    for county in counties:
        data = values.get(county) or _county_values(county)
        cells = {
            "B11001_001": data["households"],
            "B11001_002": int(data["households"] * 0.6),
            "B11001_007": data["households"] - int(data["households"] * 0.6),
            "B19051_001": data["households"],
            "B19051_002": data["earning_households"],
            "B19051_003": data["households"] - data["earning_households"],
            "B19061_001": data["earnings"],
            "B19025_001": data["income"],
            "B19001_001": sum(data["brackets"]),
        }
        for (bracket, *_rest), value in zip(
            _county().B19001_BRACKETS, data["brackets"], strict=True
        ):
            cells[bracket[:-1]] = value
        for stem, value in cells.items():
            rows.append(_acs_fact(county, f"{stem}E", value))
            rows.append(_acs_fact(county, f"{stem}M", max(1, int(value * 0.05))))
        if county not in without_median:
            rows.append(_acs_fact(county, "B19013_001E", data["median"], group=1))
            rows.append(_acs_fact(county, "B19013_001M", 900, group=1))
        if county in b01003_group_1:
            rows.append(_acs_fact(county, "B01003_001E", data["population"], group=1))
            rows.append(_acs_fact(county, "B01003_001M", 40, group=1))
        else:
            rows.append(_acs_fact(county, "B01003_001E", data["population"], group=0))
    return rows


def _soi_rows(values):
    """IRS SOI county rows, ``{county: (returns, agi)}``."""
    rows = []
    for county, (returns, agi) in values.items():
        for measure, value in (
            ("return_count", returns),
            ("adjusted_gross_income", agi),
        ):
            row = _fact(county, measure, value)
            row["layout"]["groupby_value_id"] = f"county_{county}"
            rows.append(row)
    return rows


def _codes(counties):
    return [f"{int(code):05d}" for code in counties]


# --- ACS fact parsing ---------------------------------------------------------


def test_acs_values_train_on_estimates_and_carry_optional_margins() -> None:
    county = _county()
    rows = _acs_rows(["01001", "01003"], b01003_group_1=["01003"])
    # Drop one margin: an estimate never needs its MOE row.
    rows = [
        row
        for row in rows
        if not (
            row["observed_measure"]["source_measure_id"] == "B11001_001M"
            and row["geography"]["id"].endswith("01001")
        )
    ]
    values = county.acs_county_values(
        rows, ["B11001_001E", "B01003_001E"], required={1001: 1.0, 1003: 1.0}
    )
    households = values["B11001_001E"]
    assert households[1001][1] is None
    assert households[1003][1] is not None
    assert households[1003][0] != households[1003][1]
    # B01003 reaches every county only across its two record sets.
    assert set(values["B01003_001E"]) == {1001, 1003}
    assert values["B01003_001E"][1001][1] is None
    assert values["B01003_001E"][1003][1] == 40


@pytest.mark.parametrize(
    "case",
    ["missing", "unexpected", "duplicate", "negative", "statistic", "layout", "pr"],
)
def test_acs_county_inventory_refusals(case) -> None:
    county = _county()
    rows = _acs_rows(["01001", "01003"])
    estimate = next(
        row
        for row in rows
        if row["observed_measure"]["source_measure_id"] == "B11001_001E"
    )
    required = {1001: 1.0, 1003: 1.0}
    if case == "missing":
        rows.remove(estimate)
        match = "missing=\\['01001'\\]"
    elif case == "unexpected":
        rows += _acs_rows(["01005"])
        match = "unexpected=\\['01005'\\]"
    elif case == "duplicate":
        rows.append(copy.deepcopy(estimate))
        match = "Duplicate"
    elif case == "negative":
        estimate["value"] = -1
        match = "nonnegative"
    elif case == "statistic":
        estimate["dimensions"]["statistic"] = "margin_of_error"
        match = "statistic"
    elif case == "layout":
        estimate["layout"]["groupby_value_id"] = "01003"
        match = "disagree"
    else:
        rows += _acs_rows(["72001"])
        match = "Puerto Rico"
    with pytest.raises(ValueError, match=match):
        county.acs_county_values(rows, ["B11001_001E"], required=required)


def test_a_puerto_rico_soi_row_reaching_the_rung_is_refused() -> None:
    county = _county()
    rows = _soi_rows({"01001": (10, 100), "72001": (5, 50)})
    with pytest.raises(ValueError, match="Puerto Rico"):
        county.state_county_soi_surface(
            [_state_spec(), _state_spec("adjusted_gross_income")],
            rows,
            {1001: 10.0},
        )


def test_median_income_is_a_diagnostic_and_a_missing_county_is_listed() -> None:
    county = _county()
    counties = ["35001", "35011"]
    design = county.county_validation_design(
        _acs_rows(counties, without_median=["35011"]), {35001: 10.0, 35011: 5.0}
    )
    assert design["median_income_diagnostic"]["missing"] == ["35011"]
    # Materialize writes the design as strict JSON (open bracket ends: null).
    assert (
        json.loads(json.dumps(design, allow_nan=False))["brackets"][0]["lower"] is None
    )
    measures = {
        measure
        for family in design["families"].values()
        for measure in family["source_measure_ids"]
    }
    assert "B19013_001E" not in measures


def test_validation_design_refuses_a_zero_state_denominator_and_bad_brackets() -> None:
    county = _county()
    values = {code: _county_values(code) for code in ["01001", "01003"]}
    for data in values.values():
        data["earnings"] = 0
    with pytest.raises(ValueError, match="share denominator"):
        county.county_validation_design(
            _acs_rows(list(values), values=values), {1001: 1.0, 1003: 1.0}
        )
    rows = _acs_rows(["01001"])
    for row in rows:
        if row["observed_measure"]["source_measure_id"] == "B19001_003E":
            row["dimensions"]["income_range"] = "10000_to_19999"
    with pytest.raises(ValueError, match="bracket"):
        county.county_validation_design(rows, {1001: 1.0})


# --- ACS households (trained) -------------------------------------------------


@st.composite
def _acs_layouts(draw):
    states = draw(
        st.lists(
            st.sampled_from([1, 6, 12, 15, 36]), min_size=1, max_size=3, unique=True
        )
    )
    counties = {}
    for state in states:
        for index in range(draw(st.integers(1, 5))):
            code = state * 1000 + 2 * index + 1
            counties[code] = float(draw(st.integers(1, 10_000_000)))
    values = {
        f"{code:05d}": {
            **_county_values(f"{code:05d}"),
            "households": draw(st.integers(0, 2_000_000)),
        }
        for code in counties
    }
    for state in states:
        codes = [f"{code:05d}" for code in counties if code // 1000 == state]
        if not any(values[code]["households"] for code in codes):
            values[codes[0]]["households"] = 1
    return counties, values


@_SETTINGS
@given(_acs_layouts())
def test_acs_household_children_sum_to_their_new_state_parent(layout) -> None:
    county = _county()
    populations, values = layout
    surface = county.acs_household_surface(
        _acs_rows(list(values), values=values), populations
    )
    parents = {
        spec.metadata["state_fips"]: spec
        for spec in surface.specs
        if spec.metadata["geography_level"] == "state"
    }
    for state, parent in parents.items():
        public = math.fsum(
            data["households"] for code, data in values.items() if code[:2] == state
        )
        assert parent.value == pytest.approx(public, rel=1e-12)
        assert parent.family == "acs_households"
        children = [
            spec
            for spec in surface.specs
            if spec.metadata.get("state_county_parent_target_name") == parent.name
        ]
        assert {child.metadata["county_fips"] for child in children} == {
            code for code in values if code[:2] == state
        }
        assert math.fsum(child.value for child in children) == pytest.approx(
            parent.value, rel=1e-12, abs=1e-9
        )
    assert surface.receipt["new_state_targets"] == len(parents)


def test_acs_household_specs_share_one_label_per_geography_with_other_targets() -> None:
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_NAME

    county = _county()
    populations = {1001: 80.0, 1003: 120.0}
    acs = county.acs_household_surface(_acs_rows(["01001", "01003"]), populations)
    for spec in acs.specs:
        geography = spec.hierarchy.geography
        if spec.metadata["geography_level"] == "state":
            assert geography.id == "0400000US01"
            # The label the release's state population targets carry.
            assert geography.label == US_STATE_FIPS_TO_NAME["01"]
    county_population = [
        county.population_spec(f"pop_county_{code:05d}", value)
        for code, value in populations.items()
    ]
    TargetRegistry((*county_population, *acs.specs, _state_spec()), country="us")


def test_connecticut_planning_regions_are_county_equivalents() -> None:
    county = _county()
    regions = [f"091{index}0" for index in range(1, 10)]
    populations = {int(code): 100.0 + index for index, code in enumerate(regions)}
    surface = county.acs_household_surface(_acs_rows(regions), populations)
    (parent,) = [
        spec for spec in surface.specs if spec.metadata["geography_level"] == "state"
    ]
    assert parent.metadata["state_fips"] == "09"
    children = [spec for spec in surface.specs if spec is not parent]
    assert [child.metadata["county_fips"] for child in children] == regions
    households = _households(
        {"county_fips": code, "state_fips": 9, "household_spine": "asec_puf"}
        for code in regions
    )
    rows = county.acs_household_rows(households, surface.specs).toarray()
    at = surface.specs.index(parent)
    child_rows = rows[[index for index in range(len(rows)) if index != at]]
    np.testing.assert_array_equal(child_rows.sum(axis=0), rows[at])
    np.testing.assert_array_equal(child_rows, np.eye(9))
    # A legacy Connecticut county code is not in the planning-region inventory.
    with pytest.raises(ValueError, match="09001"):
        county.acs_county_values(
            _acs_rows([*regions, "09001"]),
            ["B11001_001E"],
            required=populations,
        )


def _households(rows):
    frame = pd.DataFrame(rows)
    frame["household_id"] = np.arange(1, len(frame) + 1)
    return frame


def test_household_universe_excludes_acs_group_quarters_only() -> None:
    county = _county()
    households = pd.DataFrame(
        {
            "household_spine": ["acs_2024_1yr"] * 3 + ["asec_puf"],
            "TYPEHUGQ": [1, 2, 3, np.nan],
        }
    )
    np.testing.assert_array_equal(county.household_universe(households), [1, 0, 0, 1])
    for bad in (
        households.assign(TYPEHUGQ=[1, 2, np.nan, np.nan]),
        households.assign(TYPEHUGQ=[1, 2, 4, np.nan]),
        households.assign(TYPEHUGQ=[1, 2, 3, 1]),
        households.drop(columns="TYPEHUGQ"),
    ):
        with pytest.raises(ValueError):
            county.household_universe(bad)


@_SETTINGS
@given(
    st.lists(
        st.tuples(
            st.sampled_from([1001, 1003, 6001, 6003, 6005]),
            st.sampled_from(["acs_2024_1yr", "asec_puf"]),
            st.sampled_from([1, 2, 3]),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_acs_household_rows_partition_each_state_row(draws) -> None:
    county = _county()
    households = _households(
        {
            "county_fips": code,
            "state_fips": code // 1000,
            "household_spine": spine,
            "TYPEHUGQ": kind if spine == "acs_2024_1yr" else np.nan,
        }
        for code, spine, kind in draws
    )
    populations = {code: 10.0 for code in (1001, 1003, 6001, 6003, 6005)}
    specs = county.acs_household_surface(
        _acs_rows(_codes(populations)), populations
    ).specs
    rows = county.acs_household_rows(households, specs)
    universe = county.household_universe(households)
    for index, spec in enumerate(specs):
        if spec.metadata["geography_level"] != "state":
            continue
        children = [
            child
            for child, other in enumerate(specs)
            if other.metadata.get("state_county_parent_target_name") == spec.name
        ]
        parent = rows[[index]].toarray()[0]
        np.testing.assert_array_equal(rows[children].toarray().sum(axis=0), parent)
        # Each household counts in at most one county row, and only if it is
        # a household (not group quarters).
        assert (rows[children].toarray() > 0).sum(axis=0).max(initial=0) <= 1
        in_state = households["state_fips"].to_numpy() == int(
            spec.metadata["state_fips"]
        )
        np.testing.assert_array_equal(parent, universe * in_state)


# --- Kalawao and declared combined SOI cells ---------------------------------

_HAWAII = {15001: 200.0, 15003: 1000.0, 15005: 80.0, 15007: 70.0, 15009: 160.0}


def _hawaii_parents():
    return [
        _state_spec(state="15", value=500.0),
        _state_spec("adjusted_gross_income", 50_000.0, state="15"),
    ]


def _hawaii_soi(without=(15005,)):
    return _soi_rows(
        {
            f"{code:05d}": (int(population), int(population * 100))
            for code, population in _HAWAII.items()
            if code not in without
        }
    )


def test_kalawao_shares_a_declared_soi_cell_with_maui() -> None:
    county = _county()
    surface = county.state_county_soi_surface(_hawaii_parents(), _hawaii_soi(), _HAWAII)
    maui = [spec for spec in surface.specs if spec.metadata["county_fips"] == "15009"]
    assert len(maui) == 2
    for spec in maui:
        assert spec.metadata["state_county_cell_members"] == "15005,15009"
        assert "fewer than 20 returns" in spec.metadata["state_county_cell_basis"]
        published = sum(
            population for code, population in _HAWAII.items() if code != 15005
        )
        assert float(spec.metadata["state_county_share"]) == pytest.approx(
            160.0 / published
        )
        assert float(spec.metadata["county_population"]) == 80.0 + 160.0
    assert not [
        spec for spec in surface.specs if spec.metadata["county_fips"] == "15005"
    ]
    (merge,) = surface.receipt["combined_cells"]
    assert merge["members"] == ["15005", "15009"]
    assert merge["absent_from_soi"] == "15005"
    assert set(merge["concepts"]) == {
        "return_count|2022",
        "adjusted_gross_income|2022",
    }
    assert "22incydocguide" in merge["source"]


def test_an_undeclared_absent_county_is_still_refused() -> None:
    county = _county()
    with pytest.raises(ValueError, match="missing=\\['15007'\\]"):
        county.state_county_soi_surface(
            _hawaii_parents(), _hawaii_soi(without=(15005, 15007)), _HAWAII
        )
    with pytest.raises(ValueError, match="missing=\\['15005'\\]"):
        county.state_county_soi_surface(
            _hawaii_parents(), _hawaii_soi(), _HAWAII, combined_cells={}
        )
    # A host that is itself absent cannot carry the absent county.
    with pytest.raises(ValueError, match="missing"):
        county.state_county_soi_surface(
            _hawaii_parents(), _hawaii_soi(without=(15005, 15009)), _HAWAII
        )


@pytest.mark.parametrize(
    ("cells", "match"),
    [
        ({15005: 6001}, "one state"),
        ({15005: 15005}, "distinct"),
        ({15005: 15009, 15009: 15001}, "itself merged"),
    ],
)
def test_declared_combined_cells_must_be_well_formed(cells, match) -> None:
    with pytest.raises(ValueError, match=match):
        _county().state_county_soi_surface(
            _hawaii_parents(), _hawaii_soi(), _HAWAII, combined_cells=cells
        )


def test_a_present_declared_county_keeps_its_own_soi_row() -> None:
    surface = _county().state_county_soi_surface(
        _hawaii_parents(), _hawaii_soi(without=()), _HAWAII
    )
    assert surface.receipt["combined_cells"] == []
    assert {spec.metadata["county_fips"] for spec in surface.specs} == set(
        _codes(_HAWAII)
    )
    assert "state_county_cell_members" not in surface.specs[0].metadata


@_SETTINGS
@given(
    st.lists(st.sampled_from(sorted(_HAWAII)), min_size=1, max_size=30),
    st.lists(st.integers(-1000, 1000), min_size=30, max_size=30),
)
def test_combined_cells_partition_the_parent_row_exactly(codes, values) -> None:
    county = _county()
    surface = county.state_county_soi_surface(_hawaii_parents(), _hawaii_soi(), _HAWAII)
    households = _households({"county_fips": code, "state_fips": 15} for code in codes)
    parents = _hawaii_parents()
    data = np.asarray(values[: len(codes)], dtype=np.float32)
    parent_matrix = sparse.csr_array(np.vstack([np.ones(len(codes)), data]))
    children = county.split_state_rows(
        parents, parent_matrix, surface.specs, households
    )
    for index, parent in enumerate(parents):
        rows = [
            row
            for row, spec in enumerate(surface.specs)
            if spec.metadata["state_county_parent_target_name"] == parent.name
        ]
        dense = children[rows].toarray()
        np.testing.assert_array_equal(
            dense.sum(axis=0), parent_matrix[[index]].toarray()[0]
        )
        # Every household is in exactly one SOI cell of each concept.
        np.testing.assert_array_equal(
            (children[rows].toarray() != 0).sum(axis=0) <= 1, True
        )
        kalawao = households["county_fips"].to_numpy() == 15005
        maui_row = next(
            position
            for position, row in enumerate(rows)
            if surface.specs[row].metadata["county_fips"] == "15009"
        )
        np.testing.assert_array_equal(
            dense[maui_row][kalawao], parent_matrix[[index]].toarray()[0][kalawao]
        )
    assert math.fsum(
        spec.value
        for spec in surface.specs
        if spec.metadata["state_county_parent_target_name"] == parents[0].name
    ) == pytest.approx(parents[0].value)


def test_split_rows_refuse_overlapping_member_sets() -> None:
    county = _county()
    surface = county.state_county_soi_surface(_hawaii_parents(), _hawaii_soi(), _HAWAII)
    specs = list(surface.specs)
    tampered = [
        spec
        if spec.metadata["county_fips"] != "15001"
        else type(spec)(
            **{
                **{field: getattr(spec, field) for field in spec.__dataclass_fields__},
                "metadata": {
                    **spec.metadata,
                    "state_county_cell_members": "15001,15005",
                },
            }
        )
        for spec in specs
    ]
    households = _households({"county_fips": 15001, "state_fips": 15} for _ in range(2))
    with pytest.raises(ValueError, match="overlap"):
        county.split_state_rows(
            _hawaii_parents(),
            sparse.csr_array(np.ones((2, 2), np.float32)),
            tampered,
            households,
        )


def test_any_all_return_soi_concept_with_a_broad_parent_gets_county_rows() -> None:
    county = _county()
    parents = [
        _state_spec(),
        _state_spec("adjusted_gross_income", 1000.0),
        _state_spec(
            "wages_and_salaries_amount",
            600.0,
            ledger_measure_unit="usd",
        ),
    ]
    facts = _soi_rows({"01001": (10, 100), "01003": (30, 300)})
    for code, value in (("01001", 40.0), ("01003", 120.0)):
        row = _fact(code, "wages_and_salaries_amount", value)
        row["observed_measure"]["unit"] = "usd"
        facts.append(row)
    surface = county.state_county_soi_surface(parents, facts, {1001: 1.0, 1003: 1.0})
    wages = {
        spec.metadata["county_fips"]: spec.value
        for spec in surface.specs
        if spec.metadata["source_measure_id"] == "wages_and_salaries_amount"
    }
    assert wages == pytest.approx({"01001": 150.0, "01003": 450.0})
    assert "wages_and_salaries_amount" in surface.receipt["concepts"]


# --- minimum support ----------------------------------------------------------


def _support_households(cells, *, tract=(), gq=(), disagree=()):
    """Households from ``[(puma, county), ...]``; flags index into ``cells``."""
    rows = []
    for index, (puma, code) in enumerate(cells):
        acs = index % 2 == 0
        rows.append(
            {
                "household_id": 1000 + index,
                "state_fips": code // 1000,
                "county_fips": f"{code:05d}",
                "county_fips_2023": f"{code + 2 if index in disagree else code:05d}",
                "puma": f"{puma:07d}",
                "household_spine": "acs_2024_1yr" if acs else "asec_puf",
                "household_source_id": index,
                "TYPEHUGQ": (2.0 if index in gq else 1.0) if acs else np.nan,
                "tract_geoid": f"{code:05d}000100" if index in tract else None,
                "size_marker": index,
            }
        )
    return pd.DataFrame(rows)


def test_minimum_support_moves_exactly_k_same_puma_households() -> None:
    county = _county()
    cells = [(100001, 1001)] * 8 + [(100001, 1003)] * 4 + [(100002, 1007)] * 5
    households = _support_households(cells, tract=(0,), gq=(2,), disagree=(4,))
    populations = {1001: 100.0, 1003: 50.0, 1005: 12.0, 1007: 60.0}
    overlaps = county.puma_overlaps(
        [100001, 100001, 100001, 100002, 100002],
        [1001, 1003, 1005, 1005, 1007],
        [90.0, 50.0, 10.0, 2.0, 58.0],
    )
    moves, receipt = county.plan_minimum_support(
        households, populations, overlaps, per_county=3
    )
    assert receipt["unsupported_before"] == ["01005"]
    assert len(moves) == 3
    assert {move["to_county"] for move in moves} == {"01005"}
    assert {move["puma"] for move in moves} == {"0100001"}
    moved = set(move["household_id"] for move in moves)
    # Ineligible households (tract, group quarters, disagreeing columns) stay.
    assert moved.isdisjoint({1000, 1002, 1004})
    # The donor county keeps at least k households.
    remaining = households.loc[
        ~households["household_id"].isin(moved), "county_fips"
    ].value_counts()
    assert remaining.min() >= 3
    applied = county.apply_support_moves(households, moves)
    changed = applied["household_id"].isin(moved)
    assert (applied.loc[changed, "county_fips"] == "01005").all()
    assert (applied.loc[changed, "county_fips_2023"] == "01005").all()
    untouched = [
        column
        for column in households.columns
        if column not in {"county_fips", "county_fips_2023"}
    ]
    pd.testing.assert_frame_equal(applied[untouched], households[untouched])
    pd.testing.assert_frame_equal(applied[~changed], households[~changed])


def test_minimum_support_is_all_or_nothing_and_names_the_county() -> None:
    county = _county()
    cells = [(100001, 1001)] * 4
    households = _support_households(cells)
    overlaps = county.puma_overlaps([100001, 100001], [1001, 1003], [90.0, 10.0])
    moves, receipt = county.plan_minimum_support(
        households, {1001: 90.0, 1003: 10.0}, overlaps, per_county=3
    )
    assert moves == []
    assert receipt["counties"]["01003"]["status"] == "insufficient_eligible_donors"
    assert receipt["counties"]["01003"]["moved"] == 0


def test_group_quarters_records_are_not_county_support() -> None:
    county = _county()
    counties = [1001, 1003, 1003, 1003, 1003, 1007, 1007]
    kinds = [1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0]
    households = _support_households([(100001, code) for code in counties])
    households["household_spine"] = "acs_2024_1yr"
    households["TYPEHUGQ"] = kinds
    # 01003 has one housing unit beside three group-quarters records; 01007
    # has group quarters only; 01005 has nothing.
    populations = {1001: 10.0, 1003: 40.0, 1005: 5.0, 1007: 8.0, 1009: 30.0}
    overlaps = county.puma_overlaps(
        [100001] * 5, sorted(populations), [10.0, 40.0, 5.0, 8.0, 30.0]
    )
    moves, receipt = county.plan_minimum_support(
        households, populations, overlaps, per_county=1
    )
    assert receipt["unsupported_before"] == ["01005", "01007"]
    # 01003's lone housing unit would leave it with group quarters only.
    assert moves == []
    assert {entry["status"] for entry in receipt["counties"].values()} == {
        "insufficient_eligible_donors"
    }
    richer = pd.concat(
        [
            households,
            _support_households([(100001, 1009)] * 3).assign(
                household_id=[2001, 2002, 2003],
                household_spine="acs_2024_1yr",
                TYPEHUGQ=1.0,
            ),
        ],
        ignore_index=True,
    )
    moves, receipt = county.plan_minimum_support(
        richer, populations, overlaps, per_county=1
    )
    assert {move["to_county"] for move in moves} == {"01005", "01007"}
    assert {move["from_county"] for move in moves} == {"01009"}


@pytest.mark.parametrize("soi_mode", ["state_county", "state"])
def test_a_state_county_run_refuses_group_quarters_only_counties_before_the_engine(
    tmp_path, soi_mode
) -> None:
    county = _county()
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "county_fips": ["01001", "01003"],
            "state_fips": [1, 1],
            "household_spine": ["acs_2024_1yr", "acs_2024_1yr"],
            "TYPEHUGQ": [1.0, 3.0],
        }
    )
    frame = SimpleNamespace(table=lambda _entity: households)
    args = SimpleNamespace(
        checkpoint_dir=tmp_path,
        county_column="county_fips",
        county_min_support=0,
        soi_mode=soi_mode,
        allow_partial_geography=False,
        sample_fraction=1.0,
    )
    populations = {"county": {1001: 10.0, 1003: 20.0}}
    if soi_mode == "state_county":
        with pytest.raises(SystemExit, match="01003"):
            county.check_support(frame, populations, args)
    else:
        assert county.check_support(frame, populations, args) is frame
    report = json.loads((tmp_path / county.SUPPORT_FILENAME).read_text())
    strict = soi_mode == "state_county"
    assert report["unsupported_counties"] == (["01003"] if strict else [])
    assert report["counties"]["01003"]["supporting_households"] == (0 if strict else 1)


def test_a_fully_supported_frame_is_returned_unchanged() -> None:
    county = _county()
    households = _support_households([(100001, 1001)] * 3 + [(100001, 1003)] * 3)
    moves, receipt = county.plan_minimum_support(
        households, {1001: 1.0, 1003: 1.0}, {}, per_county=3
    )
    assert moves == [] and receipt["unsupported_before"] == []
    assert county.apply_support_moves(households, moves) is households


@st.composite
def _support_layouts(draw):
    states = draw(
        st.lists(st.sampled_from([1, 6]), min_size=1, max_size=2, unique=True)
    )
    overlaps_rows, cells, populations = [], [], {}
    for state in states:
        pumas = [
            state * 100_000 + index + 1 for index in range(draw(st.integers(1, 2)))
        ]
        counties = [
            state * 1000 + 2 * index + 1 for index in range(draw(st.integers(2, 5)))
        ]
        for code in counties:
            populations[code] = float(draw(st.integers(0, 1000)))
            for puma in draw(
                st.lists(st.sampled_from(pumas), min_size=1, max_size=2, unique=True)
            ):
                overlaps_rows.append((puma, code, float(draw(st.integers(1, 500)))))
                cells += [(puma, code)] * draw(st.integers(0, 7))
    if not cells:
        cells = [(overlaps_rows[0][0], overlaps_rows[0][1])]
    flags = draw(
        st.lists(
            st.sampled_from(["", "tract", "gq", "disagree"]),
            min_size=len(cells),
            max_size=len(cells),
        )
    )
    per_county = draw(st.integers(1, 4))
    order = draw(st.permutations(range(len(cells))))
    return overlaps_rows, cells, populations, flags, per_county, order


@_SETTINGS
@given(_support_layouts())
def test_minimum_support_invariants(layout) -> None:
    county = _county()
    overlap_rows, cells, populations, flags, per_county, order = layout
    households = _support_households(
        cells,
        tract={i for i, flag in enumerate(flags) if flag == "tract"},
        gq={i for i, flag in enumerate(flags) if flag == "gq"},
        disagree={i for i, flag in enumerate(flags) if flag == "disagree"},
    )
    overlaps = county.puma_overlaps(*zip(*overlap_rows, strict=True))
    moves, receipt = county.plan_minimum_support(
        households, populations, overlaps, per_county=per_county
    )
    before = households.set_index("household_id")
    # Support is a household in the household universe: ACS group quarters
    # (TYPEHUGQ 2 here) support no households-family target.
    in_universe = households["TYPEHUGQ"].ne(2.0)
    supported = set(households.loc[in_universe, "county_fips"].astype(int))
    destinations = {}
    for move in moves:
        destination = int(move["to_county"])
        destinations[destination] = destinations.get(destination, 0) + 1
        source = before.loc[move["household_id"]]
        # Never into a county with support; state and PUMA unchanged.
        assert destination not in supported
        assert populations[destination] > 0
        assert int(move["puma"]) == int(source["puma"])
        assert destination // 1000 == int(source["state_fips"])
        assert int(move["puma"]) in {puma for puma, _ in overlaps[destination]}
        # Only eligible households move.
        assert pd.isna(source["tract_geoid"])
        assert source["county_fips"] == source["county_fips_2023"]
        assert source["TYPEHUGQ"] != 2.0
    # The count moved into a county is exact, and the receipt says so.
    assert set(destinations.values()) <= {per_county}
    for code, entry in receipt["counties"].items():
        assert entry["moved"] == destinations.get(int(code), 0)
        assert entry["status"] == (
            "moved" if entry["moved"] else "insufficient_eligible_donors"
        )
    assert {int(code) for code in receipt["unsupported_before"]} == {
        code
        for code, population in populations.items()
        if population > 0 and code not in supported
    }
    after = county.apply_support_moves(households, moves)
    # Donor counties keep at least per_county supporting households.
    counts = after.loc[in_universe, "county_fips"].astype(int).value_counts()
    for code in {int(move["from_county"]) for move in moves}:
        assert counts[code] >= per_county
    for column in households.columns:
        if column not in {"county_fips", "county_fips_2023"}:
            pd.testing.assert_series_equal(after[column], households[column])
    # Deterministic, and independent of row order.
    shuffled = households.iloc[list(order)].reset_index(drop=True)
    again, _receipt = county.plan_minimum_support(
        shuffled, populations, overlaps, per_county=per_county
    )
    assert sorted(again, key=json.dumps) == sorted(moves, key=json.dumps)
    assert receipt["moves_sha256"] == county._moves_digest(moves)


def test_recorded_moves_reapply_to_a_fresh_frame_and_refuse_a_changed_plan(
    tmp_path,
) -> None:
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    county = _county()
    cells = [(100001, 1001)] * 8
    households = _support_households(cells)
    overlaps = county.puma_overlaps([100001, 100001], [1001, 1003], [90.0, 10.0])
    moves, receipt = county.plan_minimum_support(
        households, {1001: 90.0, 1003: 10.0}, overlaps, per_county=3
    )
    ids = households["household_id"].to_numpy()
    tables = {
        "household": households,
        "person": pd.DataFrame(
            {
                "person_id": ids,
                "person_household_id": ids,
                **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
            }
        ),
    }
    for group in US_SCHEMA.group_entities:
        tables.setdefault(group, pd.DataFrame({f"{group}_id": ids}))
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )
    (tmp_path / county.SUPPORT_FILENAME).write_text(
        json.dumps({"minimum_support": receipt})
    )
    identity = {
        "county_column": "county_fips",
        "county_support": {
            "moves": len(moves),
            "moves_sha256": receipt["moves_sha256"],
            "rewritten_columns": receipt["rewritten_columns"],
        },
    }
    moved = county.apply_recorded_support(frame, identity, tmp_path)
    expected = county.apply_support_moves(households, moves)
    pd.testing.assert_frame_equal(
        moved.table("household").reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert county.apply_recorded_support(frame, {}, tmp_path) is frame
    tampered = copy.deepcopy(receipt)
    tampered["moves"][0]["to_county"] = "01001"
    (tmp_path / county.SUPPORT_FILENAME).write_text(
        json.dumps({"minimum_support": tampered})
    )
    with pytest.raises(SystemExit, match="no longer holds"):
        county.apply_recorded_support(frame, identity, tmp_path)
    stale = county.apply_support_moves(households, moves)
    with pytest.raises(ValueError, match="no longer matches"):
        county.apply_support_moves(stale, moves)


def test_check_support_applies_minimum_support_before_refusing(
    monkeypatch, tmp_path
) -> None:
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    county = _county()
    households = _support_households([(100001, 1001)] * 8)
    ids = households["household_id"].to_numpy()
    tables = {
        "household": households,
        "person": pd.DataFrame(
            {
                "person_id": ids,
                "person_household_id": ids,
                **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
            }
        ),
    }
    for group in US_SCHEMA.group_entities:
        tables.setdefault(group, pd.DataFrame({f"{group}_id": ids}))
    frame = Frame(
        tables, US_SCHEMA, {"household": Weights(np.ones(8), WeightKind.DESIGN)}
    )
    monkeypatch.setattr(
        county,
        "county_puma_overlaps",
        lambda _path: county.puma_overlaps(
            [100001, 100001], [1001, 1003], [90.0, 10.0]
        ),
    )
    populations = {"county": {1001: 90.0, 1003: 10.0}}
    args = SimpleNamespace(
        county_column="county_fips",
        county_min_support=3,
        ladder=tmp_path / "ladder.npz",
        checkpoint_dir=tmp_path,
        allow_partial_geography=False,
        sample_fraction=1.0,
    )
    result = county.check_support(frame, populations, args)
    report = json.loads((tmp_path / county.SUPPORT_FILENAME).read_text())
    assert report["unsupported_counties"] == []
    assert report["counties"]["01003"]["supporting_households"] == 3
    assert len(report["minimum_support"]["moves"]) == 3
    assert args.county_support_receipt["moved_counties"] == ["01003"]
    assert (result.table("household")["county_fips"] == "01003").sum() == 3
    # Without the step the same frame is refused.
    with pytest.raises(SystemExit, match="no supporting household"):
        county.check_support(
            frame,
            populations,
            SimpleNamespace(**{**vars(args), "county_min_support": 0}),
        )


# --- validation scoring and the reliability grade ----------------------------


def _validation_inputs(counties, households_per_county, *, seed=0):
    county = _county()
    populations = {int(code): 100.0 + int(code) % 97 for code in counties}
    design = county.county_validation_design(_acs_rows(counties), populations)
    rng = np.random.default_rng(seed)
    codes = np.repeat(
        np.asarray([int(code) for code in counties]), households_per_county
    )
    n = len(codes)
    households = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "county_fips": codes,
            "state_fips": codes // 1000,
            "household_spine": np.where(
                np.arange(n) % 2 == 0, "acs_2024_1yr", "asec_puf"
            ),
            "household_source_id": np.arange(n),
        }
    )
    earnings = rng.uniform(0, 100_000, n) * (rng.uniform(size=n) > 0.2)
    measures = {
        "household_id": households["household_id"].to_numpy(),
        "in_household_universe": (rng.uniform(size=n) > 0.05).astype(float),
        "earnings": earnings,
        "household_income_approx": earnings + rng.uniform(-5_000, 60_000, n),
    }
    return design, measures, households, rng.uniform(0.5, 3.0, n)


def test_validation_targets_redistribute_each_state_total_by_public_shares() -> None:
    county = _county()
    counties = ["01001", "01003", "01005", "06001", "06003"]
    design, measures, households, weights = _validation_inputs(
        counties, [5, 3, 4, 6, 2]
    )
    result = county.score_county_validation(
        design,
        measures,
        households,
        final_weights=weights,
        design_weights=np.ones(len(weights)),
        cap=10.0,
    )
    assert result["never_in_loss"] is True
    json.dumps(result, allow_nan=False)
    for family in county.VALIDATION_FAMILIES:
        if family.household_measure == "income_bracket":
            continue
        for state in ("01", "06"):
            cells = [
                result["counties"][code][family.name]
                for code in counties
                if code.startswith(state)
            ]
            assert math.fsum(cell["target"] for cell in cells) == pytest.approx(
                math.fsum(cell["estimate"] for cell in cells), rel=1e-9
            )
            assert math.fsum(cell["pro_rata"] for cell in cells) == pytest.approx(
                math.fsum(cell["estimate"] for cell in cells), rel=1e-9
            )
    # Scaling every weight leaves every error unchanged (M_s scales too).
    scaled = county.score_county_validation(
        design,
        measures,
        households,
        final_weights=weights * 7.5,
        design_weights=np.ones(len(weights)),
        cap=10.0,
    )
    for code in counties:
        for family in county.GRADED_VALIDATION_FAMILIES:
            assert scaled["counties"][code][family]["abs_rel_error"] == pytest.approx(
                result["counties"][code][family]["abs_rel_error"], rel=1e-9, abs=1e-12
            )


def test_validation_scores_zero_when_estimates_match_public_shares() -> None:
    county = _county()
    counties = ["01001", "01003"]
    design, measures, households, _weights = _validation_inputs(counties, [1, 1])
    public = design["families"]["acs_earnings"]["public"]["B19061_001E"]
    measures = {
        **measures,
        "in_household_universe": np.ones(2),
        "earnings": np.asarray([public["01001"][0], public["01003"][0]]),
    }
    result = county.score_county_validation(
        design,
        measures,
        households,
        final_weights=np.ones(2),
        design_weights=np.ones(2),
        cap=1.0,
    )
    for code in counties:
        assert result["counties"][code]["acs_earnings"]["abs_rel_error"] == (
            pytest.approx(0.0, abs=1e-12)
        )


def test_every_one_of_3144_counties_gets_a_cell_and_only_unsupported_are_u() -> None:
    county = _county()
    states = [state for state in range(1, 57) if state not in {3, 7, 14, 43, 52}]
    counties = []
    for index in range(3_144):
        state = states[index % len(states)]
        counties.append(f"{state * 1000 + 1 + 2 * (index // len(states)):05d}")
    assert len(set(counties)) == 3_144
    per_county = [0 if index % 997 == 0 else 2 for index in range(len(counties))]
    supported = [
        code for code, count in zip(counties, per_county, strict=True) if count
    ]
    design, measures, households, weights = _validation_inputs(counties, per_county)
    validation = county.score_county_validation(
        design,
        measures,
        households,
        final_weights=weights,
        design_weights=np.ones(len(weights)),
        cap=10.0,
    )
    report = county.reliability_report(
        households,
        [],
        sparse.csr_array((0, len(households))),
        final_weights=weights,
        validation=validation,
        county_populations=design["county_populations"],
        county_names=design["county_names"],
    )
    assert set(report["counties"]) == set(counties)
    grades = {code: cell["grade"] for code, cell in report["counties"].items()}
    assert {code for code, grade in grades.items() if grade == "U"} == (
        set(counties) - set(supported)
    )
    for code in supported:
        cell = report["counties"][code]
        assert cell["validation"]["n_families"] == len(
            county.GRADED_VALIDATION_FAMILIES
        )
        assert cell["grade_driver"].startswith("validation:")
        assert cell["county_name"]
    csv_text = county.reliability_csv(report)
    lines = csv_text.splitlines()
    assert lines[0].split(",")[:5] == [
        "county_fips",
        "state_fips",
        "county_name",
        "status",
        "grade",
    ]
    assert len(lines) == 1 + 3_144


def test_minimum_support_counties_are_graded_and_marked() -> None:
    county = _county()
    counties = ["01001", "01003"]
    design, measures, households, weights = _validation_inputs(counties, [5, 3])
    validation = county.score_county_validation(
        design,
        measures,
        households,
        final_weights=weights,
        design_weights=np.ones(len(weights)),
        cap=10.0,
    )
    report = county.reliability_report(
        households,
        [],
        sparse.csr_array((0, len(households))),
        final_weights=weights,
        validation=validation,
        minimum_support_counties=["01003"],
    )
    assert report["counties"]["01003"]["status"] == "minimum_support"
    assert report["counties"]["01003"]["grade"] in {"A", "B", "C", "D"}
    assert report["counties"]["01001"]["status"] == "supported"


def test_combined_soi_cell_errors_count_toward_both_member_counties() -> None:
    county = _county()
    households = pd.DataFrame(
        {
            "county_fips": [15005, 15009],
            "state_fips": [15, 15],
            "household_spine": ["asec_puf", "asec_puf"],
            "household_source_id": [1, 2],
        }
    )
    records = [
        {
            "name": "county_15009__returns",
            "geography_level": "county",
            "county_fips": "15009",
            "state_county_cell_members": "15005,15009",
            "value": 10.0,
            "role": "holdout",
        }
    ]
    report = county.reliability_report(
        households,
        records,
        sparse.csr_array(np.asarray([[1.0, 1.0]])),
        final_weights=np.asarray([4.0, 4.0]),
    )
    for code in ("15005", "15009"):
        assert report["counties"][code]["holdout"]["n_targets"] == 1
        assert report["counties"][code]["holdout"]["max_abs_rel_error"] == (
            pytest.approx(0.2)
        )
        assert report["counties"][code]["grade"] == "D"
    # With no Kalawao household, Kalawao is U: Maui's error is not its own.
    only_maui = county.reliability_report(
        households.iloc[[1]],
        records,
        sparse.csr_array(np.asarray([[1.0]])),
        final_weights=np.asarray([8.0]),
        county_populations={15005: 80.0, 15009: 160.0},
    )
    kalawao = only_maui["counties"]["15005"]
    assert kalawao["status"] == "no_support"
    assert kalawao["holdout"]["n_targets"] == 1
    assert (kalawao["grade"], kalawao["grade_error"]) == ("U", None)
    assert only_maui["counties"]["15009"]["grade"] == "D"


@_SETTINGS
@given(st.floats(0.0, 1.0), st.floats(1.0, 2_000.0), st.floats(1.0, 2_000.0))
def test_grades_stay_monotone_in_ess_at_fixed_validation_error(
    error, low, high
) -> None:
    county = _county()
    order = "UABCD"
    low, high = sorted((low, high))
    grade_low = county.reliability_grade(low, error)
    grade_high = county.reliability_grade(high, error)
    assert "ABCD".index(grade_high) <= "ABCD".index(grade_low)
    assert grade_low in order


def test_household_validation_measures_count_semp_once_and_exclude_gq() -> None:
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    county = _county()
    ids = np.asarray([1, 2, 3])
    households = pd.DataFrame(
        {
            "household_id": ids,
            "household_spine": ["acs_2024_1yr", "acs_2024_1yr", "asec_puf"],
            "TYPEHUGQ": [1.0, 3.0, np.nan],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 1, 2, 3],
            **{
                f"person_{group}_id": [1, 1, 2, 3] for group in US_SCHEMA.group_entities
            },
            "employment_income_before_lsr": [100.0, 50.0, 10.0, 1000.0],
            # ACS SEMP routed wholly to the SSTB leaf after QBI reconciliation.
            "self_employment_income_before_lsr": [0.0, 0.0, 0.0, 20.0],
            "sstb_self_employment_income_before_lsr": [30.0, np.nan, 0.0, 5.0],
            "farm_operations_income": [0.0, 0.0, 0.0, 7.0],
            "taxable_interest_income": [1.0, 0.0, 0.0, 2.0],
            "partnership_income": [99.0, 0.0, 0.0, 11.0],
            # Donor-only earnings; zero on ACS rows after the identity gate,
            # and ignored there even when not.
            "partnership_self_employment_net_earnings": [40.0, 0.0, 0.0, 9.0],
        }
    )
    tables = {"household": households, "person": person}
    for group in US_SCHEMA.group_entities:
        tables.setdefault(group, pd.DataFrame({f"{group}_id": ids}))
    frame = Frame(
        tables, US_SCHEMA, {"household": Weights(np.ones(3), WeightKind.DESIGN)}
    )
    arrays, report = county.household_validation_measures(frame)
    # The donor household adds its partnership SE earnings (9) to earnings.
    np.testing.assert_array_equal(arrays["earnings"], [180.0, 10.0, 1041.0])
    # Income starts from the four leaves, then adds the carried inputs and
    # donor-only partnership income, which holds the partnership earnings.
    np.testing.assert_array_equal(
        arrays["household_income_approx"], [181.0, 10.0, 1045.0]
    )
    assert report["donor_only_earnings_columns_carried"] == [
        "partnership_self_employment_net_earnings"
    ]
    np.testing.assert_array_equal(arrays["in_household_universe"], [1.0, 0.0, 1.0])
    assert report["group_quarters_households"] == 1
    with pytest.raises(ValueError, match="earnings inputs"):
        county.household_validation_measures(
            Frame(
                {**tables, "person": person.drop(columns="farm_operations_income")},
                US_SCHEMA,
                {"household": Weights(np.ones(3), WeightKind.DESIGN)},
            )
        )


def test_packaged_csv_must_be_the_flattening_of_the_bound_report(tmp_path) -> None:
    county = _county()
    counties = ["01001", "01003"]
    design, measures, households, weights = _validation_inputs(counties, [3, 2])
    validation = county.score_county_validation(
        design,
        measures,
        households,
        final_weights=weights,
        design_weights=np.ones(len(weights)),
        cap=10.0,
    )
    identity = {"geographies": "state,county", "county_column": "county_fips"}
    stamp = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "run_identity_sha256": stamp,
        "weights_sha256": "w",
        "county_reliability": county.reliability_report(
            households,
            [],
            sparse.csr_array((0, len(households))),
            final_weights=weights,
            validation=validation,
        ),
    }
    (tmp_path / "calibration_summary.json").write_text(json.dumps(summary))
    county.write_reports(tmp_path, summary, identity=identity)
    files = {}
    extra = county.package_reports(files, tmp_path, identity)
    assert set(files) == {county.RELIABILITY_FILENAME}
    assert extra == {
        county.RELIABILITY_CSV_FILENAME: (
            tmp_path / county.RELIABILITY_CSV_FILENAME
        ).read_text()
    }
    csv_path = tmp_path / county.RELIABILITY_CSV_FILENAME
    csv_path.write_text(csv_path.read_text().replace(",D,", ",A,", 1) + "\n")
    with pytest.raises(SystemExit, match="flattening"):
        county.package_reports({}, tmp_path, identity)


def test_acs_household_rows_have_their_own_state_budgets_in_the_loss() -> None:
    county = _county()
    populations = {1001: 80.0, 1003: 120.0, 6001: 50.0}
    specs = county.acs_household_surface(
        _acs_rows(_codes(populations)), populations
    ).specs
    mapping = county.loss_row_mapping(specs)
    assert mapping.mapping_id == "us_acs_local_county.v2"
    keys = [mapping.concept_budget_key(spec) for spec in specs]
    states = [
        key
        for key, spec in zip(keys, specs, strict=True)
        if spec.metadata["geography_level"] == "state"
    ]
    assert len(set(states)) == 2
    counties = {
        spec.metadata["state_fips"]: key
        for key, spec in zip(keys, specs, strict=True)
        if spec.metadata["geography_level"] == "county"
    }
    assert len(set(counties.values())) == 2
    assert all(mapping.value_basis(spec) == "count" for spec in specs)
    weights = county.target_loss_weights(specs, {})
    assert np.isfinite(weights).all() and (weights > 0).all()
