"""Tests for the county pilot's sources, CT mapping and target design.

The source tests read the public files fetched for the pilot and skip when
they are absent. Run with::

    uv run --no-project --with numpy --with pandas --with scipy \
        --with pytest --with hypothesis \
        pytest experiments/us-county-pilot-20261024/test_county_targets.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import county_targets as ct  # noqa: E402

ARTIFACTS = Path("/Users/maxghenis/PolicyEngine/_build_artifacts/lwi-county-20261024")
CHRONICLE = Path("/Users/maxghenis/PolicyEngine/chronicle")
CT_REGIONS = {f"09{code}" for code in range(110, 200, 10)}
CT_LEGACY = {f"090{code:02d}" for code in range(1, 16, 2)}

needs_sources = pytest.mark.skipif(
    not (ARTIFACTS / "census_api_acs5_2024_county").is_dir() or not CHRONICLE.is_dir(),
    reason="pilot source files not present",
)


def _chronicle(member: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(CHRONICLE), "show", f"origin/main:{member}"],
        check=True,
        capture_output=True,
    ).stdout


@pytest.fixture(scope="module")
def sources() -> dict[str, bytes]:
    acs = ARTIFACTS / "census_api_acs5_2024_county"
    out = {path.stem: path.read_bytes() for path in acs.glob("acs5_2024_*.json")}
    out["pep_county_2024"] = _chronicle(
        "db/data/census/pep_county_2024/co-est2024-alldata.csv"
    )
    out["soi_county_2022"] = _chronicle(
        "db/data/irs_soi/county_2022/22incyallnoagi.csv"
    )
    return out


@pytest.fixture(scope="module")
def ct_tables():
    base = ARTIFACTS / "ct_sources"
    return ct.ct_planning_region_tables(
        (base / "ct_cou_to_cousub_crosswalk.txt").read_text(encoding="utf-8-sig"),
        (base / "BlockAssign_ST09_CT_MCD.txt").read_text(),
        (base / "tract_puma" / "2020_Census_Tract_to_2020_PUMA.txt").read_text(
            encoding="utf-8-sig"
        ),
        (base / "tract_pop_2020" / "dec_pl_2020_P1_001N_tract_ct.json").read_text(),
    )


@needs_sources
def test_sources_cover_the_2023_county_set(sources):
    acs = {
        key: ct.read_acs_county(sources[f"acs5_2024_{table}_county"], variable)
        for key, (table, variable) in ct.ACS_CELLS.items()
    }
    pep = ct.read_pep_county(sources["pep_county_2024"])
    soi = ct.read_soi_county(sources["soi_county_2022"], "A00200")
    expected = set(acs["households"].index)
    assert len(expected) == 3144
    assert CT_REGIONS <= expected and not (CT_LEGACY & expected)
    for series in [*acs.values(), pep]:
        assert set(series.index) == expected
    # SOI publishes no row for Kalawao County, HI (about 80 residents)
    assert set(soi.index) == expected - {"15005"}
    for key in ("households", "aggregate_earnings", "earning_households"):
        assert acs[key].notna().all() and (acs[key] >= 0).all()
    assert (acs["earning_households"] <= acs["households"]).all()
    assert 3.3e8 < pep.sum() < 3.5e8


@needs_sources
def test_ct_cells_nest_but_one(ct_tables):
    tract_region, cells = ct_tables
    assert set(tract_region.values()) == CT_REGIONS
    split = {key: regions for key, regions in cells.items() if len(regions) > 1}
    assert set(split) == {"0920301|09013"}
    population = sum(sum(regions.values()) for regions in cells.values())
    assert population == 3_605_944  # CT, 2020 census
    every_region = set().union(*(regions.keys() for regions in cells.values()))
    assert every_region == CT_REGIONS


@needs_sources
def test_ct_assignment_is_seeded_and_respects_single_region_cells(ct_tables):
    _, cells = ct_tables
    puma = np.array(["0920301"] * 4000 + ["0920100"] * 10)
    county = np.array(["09013"] * 4000 + ["09005"] * 10)
    first, report = ct.assign_ct_regions(puma, county, cells, seed=0)
    second, _ = ct.assign_ct_regions(puma, county, cells, seed=0)
    assert (first == second).all()
    assert set(first[4000:]) == {"09160"}
    share = (first[:4000] == "09150").mean()
    assert abs(share - 785 / (785 + 11472)) < 0.02
    assert report["drawn"] == 4000


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["01", "02", "06", "09"]),
            st.floats(
                min_value=0.0, max_value=1e9, allow_nan=False, allow_subnormal=False
            ),
        ),
        min_size=1,
        max_size=40,
    ),
    st.floats(min_value=1.0, max_value=1e12, allow_nan=False),
)
def test_share_values_preserve_state_totals(cells, scale):
    index = [f"{state}{k:03d}" for k, (state, _) in enumerate(cells)]
    public = pd.Series([value for _, value in cells], index=index, dtype=np.float64)
    states = sorted({state for state, _ in cells})
    totals = {state: scale * (1 + k) for k, state in enumerate(states)}
    values = ct.within_state_share_values(public, totals)
    assert (values.dropna() >= 0).all()
    for state in states:
        part = public[public.index.str[:2] == state]
        if part.sum() > 0:
            got = values[part.index].sum()
            assert got == pytest.approx(totals[state], rel=1e-9)
            # proportional to the public figure within the state
            ratio = values[part.index] / part
            assert np.allclose(ratio[part > 0], ratio[part > 0].iloc[0], rtol=1e-9)


def _synthetic_households(counties: list[str], n: int = 30_000, seed: int = 1):
    rng = np.random.default_rng(seed)
    county = rng.choice(counties, size=n)
    earnings = np.where(rng.random(n) < 0.75, rng.lognormal(10.8, 1.0, n), 0.0)
    return pd.DataFrame(
        {
            "household_id": np.arange(n),
            "weight": rng.uniform(50, 150, n),
            "county_fips_2023": county,
            "congressional_district_geoid": [
                int(c[:2]) * 100 + int(rng.integers(1, 4)) for c in county
            ],
            "earnings": earnings,
            "employment_income_before_lsr": earnings * 0.9,
            "n_persons": rng.integers(1, 6, n).astype(float),
            "household_income_approx": earnings + rng.uniform(0, 20_000, n),
        }
    )


@needs_sources
def test_design_rows_sum_to_state_totals_and_holds_are_exact(sources):
    counties = sorted(
        ct.read_acs_county(sources["acs5_2024_B11001_county"], "B11001_001E").index
    )
    households = _synthetic_households(counties)
    design = ct.build_design(households, sources)
    table = ct.fit_table(
        design["records"],
        design["matrix"],
        households["weight"].to_numpy(),
        households["weight"].to_numpy(),
    )
    w0 = households["weight"].to_numpy()
    state = households["county_fips_2023"].str[:2]
    for family, measure in (
        ("county_households", np.ones(len(households))),
        ("county_population", households["n_persons"].to_numpy()),
        ("county_earnings", households["earnings"].to_numpy()),
    ):
        rows = table[table["family"] == family]
        by_state = rows.groupby("state")["value"].sum()
        truth = pd.Series(w0 * measure).groupby(state.to_numpy()).sum()
        assert np.allclose(by_state.reindex(truth.index), truth, rtol=1e-9)
        # the before estimates also add up to the state totals
        assert np.allclose(
            rows.groupby("state")["before"].sum().reindex(truth.index), truth
        )
    holds = table[table["family"].str.startswith("cd_")]
    assert np.allclose(holds["before"], holds["value"])
    roles = table.groupby("family")["role"].agg(set)
    assert roles["county_soi_wages"] == {"holdout"}
