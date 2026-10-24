"""County calibration pilot for the LWI County Files (microcosm#922).

The pilot's pure logic lives in ``experiments/us-county-pilot-20261024/
county_targets.py``: public county source readers, the Connecticut planning
region mapping, the household measures, the county target design and the run
metrics. These tests run it on small committed cuts of the pinned sources.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
FIXTURES = _TEST_PATHS.tests / "fixtures" / "us_county_pilot"
PILOT = _TEST_PATHS.repository / "experiments" / "us-county-pilot-20261024"
CT_REGIONS = {f"09{code}" for code in range(110, 200, 10)}
CT_LEGACY = {f"090{code:02d}" for code in range(1, 16, 2)}
FIXTURE_STATES = {"09", "15", "44"}


def _load_county_targets():
    spec = importlib.util.spec_from_file_location(
        "us_county_pilot_targets", PILOT / "county_targets.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ct = _load_county_targets()


@pytest.fixture(scope="module")
def sources() -> dict[str, bytes]:
    out = {
        path.stem: path.read_bytes()
        for path in sorted(FIXTURES.glob("acs5_2024_*_county.json"))
    }
    out["pep_county_2024"] = (FIXTURES / "pep_county_2024_subset.csv").read_bytes()
    out["soi_county_2022"] = (FIXTURES / "soi_county_2022_subset.csv").read_bytes()
    return out


@pytest.fixture(scope="module")
def ct_tables():
    return ct.ct_planning_region_tables(
        (FIXTURES / "ct_cou_to_cousub_crosswalk.txt").read_text(encoding="utf-8-sig"),
        (FIXTURES / "ct_block_mcd_first_block_per_tract_town.txt").read_text(),
        (FIXTURES / "ct_tract_puma_2020.txt").read_text(encoding="utf-8-sig"),
        (FIXTURES / "ct_tract_population_2020.json").read_text(),
    )


def test_sources_share_one_2023_county_set(sources) -> None:
    acs = {
        key: ct.read_acs_county(sources[f"acs5_2024_{table}_county"], variable)
        for key, (table, variable) in ct.ACS_CELLS.items()
    }
    pep = ct.read_pep_county(sources["pep_county_2024"])
    soi = ct.read_soi_county(sources["soi_county_2022"], "A00200")
    expected = set(acs["households"].index)
    # Puerto Rico rows are in the ACS cut and dropped by the reader.
    assert {code[:2] for code in expected} == FIXTURE_STATES
    assert {code for code in expected if code[:2] == "09"} == CT_REGIONS
    assert not CT_LEGACY & expected
    for series in [*acs.values(), pep]:
        assert set(series.index) == expected
    assert set(soi.index) == expected - {"15005"}
    for key in ("households", "aggregate_earnings", "earning_households"):
        assert acs[key].notna().all() and (acs[key] >= 0).all()
    assert (acs["earning_households"] <= acs["households"]).all()


def test_acs_reader_drops_puerto_rico_and_sentinels() -> None:
    raw = json.dumps(
        [
            ["NAME", "B19013_001E", "state", "county"],
            ["A", "51000", "44", "001"],
            ["B", "-666666666", "44", "003"],
            ["C", None, "44", "005"],
            ["D", "22000", "72", "001"],
        ]
    ).encode()
    series = ct.read_acs_county(raw, "B19013_001E")
    assert list(series.index) == ["44001", "44003", "44005"]
    assert series["44001"] == 51000.0
    assert series[["44003", "44005"]].isna().all()


def test_ct_cells_nest_in_one_region_but_one(ct_tables) -> None:
    tract_region, cells = ct_tables
    assert set(tract_region.values()) == CT_REGIONS
    split = {key: regions for key, regions in cells.items() if len(regions) > 1}
    assert set(split) == {"0920301|09013"}
    population = sum(sum(regions.values()) for regions in cells.values())
    assert population == 3_605_944  # Connecticut, 2020 census
    assert set().union(*(regions.keys() for regions in cells.values())) == CT_REGIONS


def test_ct_assignment_is_seeded_and_keeps_single_region_cells(ct_tables) -> None:
    _, cells = ct_tables
    puma = np.array(["0920301"] * 4000 + ["0920100"] * 10)
    county = np.array(["09013"] * 4000 + ["09005"] * 10)
    first, report = ct.assign_ct_regions(puma, county, cells, seed=0)
    second, _ = ct.assign_ct_regions(puma, county, cells, seed=0)
    assert (first == second).all()
    assert set(first[4000:]) == {"09160"}
    split = cells["0920301|09013"]
    expected = split["09150"] / sum(split.values())
    assert abs((first[:4000] == "09150").mean() - expected) < 0.02
    assert report["drawn"] == 4000
    with pytest.raises(ValueError, match="not in the PUMA/county table"):
        ct.assign_ct_regions(np.array(["0999999"]), np.array(["09001"]), cells)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["01", "02", "06", "09"]),
            # published counts and dollars: zero or at least a thousandth
            st.one_of(st.just(0.0), st.floats(min_value=1e-3, max_value=1e9)),
        ),
        min_size=1,
        max_size=40,
    ),
    st.floats(min_value=1.0, max_value=1e12, allow_nan=False),
)
def test_share_values_preserve_state_totals(cells, scale) -> None:
    index = [f"{state}{k:03d}" for k, (state, _) in enumerate(cells)]
    public = pd.Series([value for _, value in cells], index=index, dtype=np.float64)
    states = sorted({state for state, _ in cells})
    totals = {state: scale * (1 + k) for k, state in enumerate(states)}
    values = ct.within_state_share_values(public, totals)
    assert (values.dropna() >= 0).all()
    for state in states:
        part = public[public.index.str[:2] == state]
        if part.sum() > 0:
            assert values[part.index].sum() == pytest.approx(totals[state], rel=1e-9)
            ratio = values[part.index] / part
            assert np.allclose(ratio[part > 0], ratio[part > 0].iloc[0], rtol=1e-9)


def _people(rows: list[dict]) -> pd.DataFrame:
    columns = {
        *ct.EARNINGS_COLUMNS,
        *ct.DONOR_ONLY_EARNINGS_COLUMNS,
        *ct.OTHER_INCOME_COLUMNS,
        *ct.DONOR_ONLY_OTHER_INCOME_COLUMNS,
        "farm_income",
    }
    frame = pd.DataFrame(rows)
    for column in columns - set(frame.columns):
        frame[column] = 0.0
    return frame


def test_household_measures_add_donor_only_income_to_donor_records_only() -> None:
    household = pd.DataFrame(
        {"household_id": [10, 20], "household_spine": [ct.ACS_SPINE, ct.DONOR_SPINE]}
    )
    business = {
        "employment_income_before_lsr": 1_000.0,
        "self_employment_income_before_lsr": 200.0,
        "sstb_self_employment_income_before_lsr": 30.0,
        "farm_operations_income": 4.0,
        "partnership_income": 50.0,
        "s_corp_income": 60.0,
        "farm_income": 7_000.0,
        "taxable_interest_income": 0.5,
    }
    person = _people(
        [
            {"person_household_id": 10, **business},
            {"person_household_id": 20, **business},
            {"person_household_id": 20, "employment_income_before_lsr": 9.0},
        ]
    )
    measures, report = ct.household_measures(household, person, np.array([2.0, 3.0]))
    acs, donor = measures.iloc[0], measures.iloc[1]
    assert acs["earnings"] == 1_230.0
    assert donor["earnings"] == 1_243.0
    assert acs["household_income_approx"] == 1_230.5
    assert donor["household_income_approx"] == 1_243.0 + 110.0 + 0.5
    assert acs["n_persons"] == 1 and donor["n_persons"] == 2
    assert "farm_income" not in report["earnings_columns"]
    assert "farm_income" not in report["other_income_columns_carried"]
    totals = report["weighted_component_totals_by_spine"]
    assert totals["farm_operations_income"] == {ct.ACS_SPINE: 0.0, ct.DONOR_SPINE: 12.0}
    assert totals["sstb_self_employment_income_before_lsr"][ct.ACS_SPINE] == 60.0


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(0, 5),
            st.floats(-1e5, 1e6, allow_nan=False, allow_subnormal=False),
            st.floats(-1e5, 1e6, allow_nan=False, allow_subnormal=False),
        ),
        min_size=1,
        max_size=30,
    )
)
def test_acs_earnings_ignore_donor_only_leaves(people) -> None:
    household = pd.DataFrame(
        {
            "household_id": np.arange(6),
            "household_spine": [ct.ACS_SPINE] * 3 + [ct.DONOR_SPINE] * 3,
        }
    )
    rows = [
        {
            "person_household_id": owner,
            "employment_income_before_lsr": wage,
            "farm_operations_income": farm,
            "partnership_income": farm,
        }
        for owner, wage, farm in people
    ]
    measures, _ = ct.household_measures(household, _people(rows), np.ones(6))
    zeroed = [{**row, "farm_operations_income": 0.0} for row in rows]
    zeroed = [{**row, "partnership_income": 0.0} for row in zeroed]
    without, _ = ct.household_measures(household, _people(zeroed), np.ones(6))
    acs = slice(0, 3)
    assert np.array_equal(
        measures["earnings"].to_numpy()[acs], without["earnings"].to_numpy()[acs]
    )
    assert np.array_equal(
        measures["household_income_approx"].to_numpy()[acs],
        without["household_income_approx"].to_numpy()[acs],
    )


def test_household_measures_refuse_unknown_spines_and_orphans() -> None:
    household = pd.DataFrame({"household_id": [1], "household_spine": ["other"]})
    person = _people([{"person_household_id": 1}])
    with pytest.raises(ValueError, match="unexpected household spines"):
        ct.household_measures(household, person, np.ones(1))
    household["household_spine"] = ct.ACS_SPINE
    orphan = _people([{"person_household_id": 2}])
    with pytest.raises(ValueError, match="missing from the table"):
        ct.household_measures(household, orphan, np.ones(1))


def _synthetic_households(counties: list[str], n: int = 6_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    county = rng.choice(counties, size=n)
    earnings = np.where(rng.random(n) < 0.75, rng.lognormal(10.8, 1.0, n), 0.0)
    return pd.DataFrame(
        {
            "household_id": np.arange(n),
            "weight": rng.uniform(50, 150, n),
            "county_fips_2023": county,
            "congressional_district_geoid": [
                int(c[:2]) * 100 + int(rng.integers(1, 3)) for c in county
            ],
            "earnings": earnings,
            "employment_income_before_lsr": earnings * 0.9,
            "n_persons": rng.integers(1, 6, n).astype(float),
            "household_income_approx": earnings + rng.uniform(0, 20_000, n),
        }
    )


def test_design_rows_share_out_state_totals(sources) -> None:
    counties = sorted(
        ct.read_acs_county(sources["acs5_2024_B11001_county"], "B11001_001E").index
    )
    households = _synthetic_households(counties)
    w0 = households["weight"].to_numpy()
    design = ct.build_design(households, sources)
    table = ct.fit_table(design["records"], design["matrix"], w0, w0)
    county = households["county_fips_2023"].to_numpy()
    everywhere = np.ones(len(households), dtype=bool)
    # SOI publishes no Kalawao row, so its rows share out Hawaii without it.
    soi_covered = county != "15005"
    for family, measure, inside in (
        ("county_households", np.ones(len(households)), everywhere),
        ("county_population", households["n_persons"].to_numpy(), everywhere),
        ("county_earnings", households["earnings"].to_numpy(), everywhere),
        (
            "county_soi_wages",
            households["employment_income_before_lsr"].to_numpy(),
            soi_covered,
        ),
    ):
        rows = table[table["family"] == family]
        truth = (
            pd.Series((w0 * measure)[inside]).groupby(county[inside].astype("U2")).sum()
        )
        assert np.allclose(rows.groupby("state")["value"].sum()[truth.index], truth)
        assert np.allclose(rows.groupby("state")["before"].sum()[truth.index], truth)
    holds = table[table["family"].str.startswith("cd_")]
    assert np.allclose(holds["before"], holds["value"])
    roles = table.groupby("family")["role"].agg(set)
    for family in ("county_earning_households", "county_soi_wages", "county_income"):
        assert roles[family] == {"holdout"}
    assert design["dropped"] == {"county_soi_wages": ["15005"]}
    soi = table[table["family"] == "county_soi_wages"]
    assert "15005" not in set(soi["geography"])


def test_design_marks_counties_without_records_unsupported(sources) -> None:
    counties = sorted(
        ct.read_acs_county(sources["acs5_2024_B11001_county"], "B11001_001E").index
    )
    households = _synthetic_households([c for c in counties if c != "15005"])
    design = ct.build_design(households, sources)
    roles = {r["name"]: r["role"] for r in design["records"]}
    assert roles["county_households:15005"] == "unsupported"
    assert roles["county_households:44001"] == "train"
    targets = ct.target_set(design["records"], design["matrix"], len(households))
    assert "county_households:15005" not in {t.name for t in targets}


@settings(max_examples=150, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["a", "b", "c"]),
            st.floats(0.01, 1e4, allow_nan=False, allow_subnormal=False),
            st.integers(1, 4),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_source_merged_ess_is_invariant_to_cloning(households) -> None:
    groups, weights, sources = [], [], []
    clone_groups, clone_weights, clone_sources = [], [], []
    for index, (group, weight, clones) in enumerate(households):
        groups.append(group)
        weights.append(weight)
        sources.append(f"s{index}")
        parts = np.arange(1, clones + 1, dtype=np.float64)
        for part in parts / parts.sum() * weight:
            clone_groups.append(group)
            clone_weights.append(part)
            clone_sources.append(f"s{index}")
    original = ct.kish_ess(np.array(weights), np.array(groups), np.array(sources))
    cloned = ct.kish_ess(
        np.array(clone_weights), np.array(clone_groups), np.array(clone_sources)
    )
    assert np.allclose(original["ess"], cloned["ess"][original.index], rtol=1e-9)
    by_row = ct.kish_ess(np.array(clone_weights), np.array(clone_groups))
    assert (by_row["ess"] >= cloned["ess"] * (1 - 1e-9)).all()
    assert ct.total_ess(np.array(weights), np.array(sources)) == pytest.approx(
        ct.total_ess(np.array(clone_weights), np.array(clone_sources)), rel=1e-9
    )


def test_grades_use_provisional_lower_bounds() -> None:
    assert [ct.grade(x) for x in (300.0, 299.9, 100.0, 30.0, 29.9, 0.0)] == [
        "A",
        "B",
        "B",
        "C",
        "D",
        "D",
    ]


def _metric_table() -> pd.DataFrame:
    rows = []
    for state, drift in (("01", 0.10), ("02", -0.02), ("04", 0.04)):
        for county in ("001", "003"):
            code = state + county
            rows.append(
                {
                    "family": "county_households",
                    "role": "train",
                    "state": state,
                    "geography": code,
                    "public_value": 900.0,
                    "value": 1_000.0,
                    "before": 800.0,
                    "after": 800.0 * (1 + drift),
                }
            )
            rows.append(
                {
                    "family": "county_earning_households",
                    "role": "holdout",
                    "state": state,
                    "geography": code,
                    "public_value": 450.0,
                    "value": 500.0,
                    "before": 400.0,
                    "after": 480.0,
                }
            )
            rows.append(
                {
                    "family": "county_soi_wages",
                    "role": "holdout",
                    "state": state,
                    "geography": code,
                    "public_value": 2_000.0,
                    "value": 2.2e6,
                    "before": 2.0e6,
                    "after": 2.1e6,
                }
            )
    return pd.DataFrame(rows)


def test_state_total_drift_scores_trained_families_only() -> None:
    drift = ct.state_total_drift(_metric_table())
    assert set(drift) == {"county_households"}
    entry = drift["county_households"]
    assert entry["median_abs"] == pytest.approx(0.04)
    assert entry["max_abs"] == pytest.approx(0.10)
    assert entry["max_state"] == "01" and entry["max_signed"] == pytest.approx(0.10)


def test_holdout_baseline_rescales_release_by_driver_ratio() -> None:
    table = _metric_table()
    out = ct.holdout_baselines(table)
    earning = out["county_earning_households"]
    # baseline = 400 * (1000 / 800) = 500, the target exactly
    assert earning["baseline"]["median_abs_rel_error"] == pytest.approx(0.0)
    assert earning["calibrated"]["median_abs_rel_error"] == pytest.approx(0.04)
    assert earning["release"]["median_abs_rel_error"] == pytest.approx(0.2)
    # no county_earnings rows: the wages baseline has no driver to rescale by
    wages = out["county_soi_wages"]
    assert np.isnan(wages["baseline"]["median_abs_rel_error"])


def test_fit_vs_published_scales_soi_thousands() -> None:
    out = ct.fit_vs_published(_metric_table())
    wages = out["county_soi_wages"]
    assert wages["before"]["median_abs_rel_error"] == pytest.approx(0.0)
    assert wages["after"]["median_abs_rel_error"] == pytest.approx(0.05)
    households = out["county_households"]
    assert households["before"]["median_abs_rel_error"] == pytest.approx(1 / 9)
