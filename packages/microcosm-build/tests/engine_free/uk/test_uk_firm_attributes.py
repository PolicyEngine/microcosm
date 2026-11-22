"""Region and legal status draws for UK firms from ONS Tables 8 and 13."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_attributes import (
    UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS,
    UK_FIRM_LEGAL_STATUS_TURNOVER_FILE,
    UK_FIRM_REGION_TURNOVER_FILE,
    UKFirmAttributeSourceData,
    assign_uk_firm_attributes,
    industry_group_divisions,
    parse_turnover_band_lowers,
    read_uk_firm_attribute_source_data,
    turnover_band_index,
    uk_firm_attribute_fallback_counts,
)
from test_support.paths import paths_for

FIXTURES = paths_for("microcosm-build").package / "tests" / "fixtures" / "uk"
ATTRIBUTE_FIXTURES = FIXTURES / "firm_attributes_2025"

REGIONS = ("E12000001", "E12000007", "W92000004")
REGION_BANDS = ("0-49", "50-99", "100+")
#: Table 8 counts by division, then band, then region. Division 2's "50-99"
#: cell is empty, so its firms fall back to division 2's row total.
REGION_COUNTS = {
    "01": {"0-49": (600, 300, 100), "50-99": (100, 100, 800), "100+": (50, 900, 50)},
    "02": {"0-49": (100, 0, 0), "50-99": (0, 0, 0), "100+": (0, 0, 100)},
    "45": {"0-49": (200, 200, 600), "50-99": (500, 500, 0), "100+": (0, 1000, 0)},
}
STATUSES = (1, 2, 5)
STATUS_BANDS = ("0-49", "50-149", "150+")
#: Table 13 counts by industry group, then band, then legal status.
STATUS_COUNTS = {
    "01-03": {
        "0-49": (100, 800, 100),
        "50-149": (500, 500, 0),
        "150+": (900, 50, 50),
    },
    "45": {"0-49": (300, 700, 0), "50-149": (0, 0, 0), "150+": (1000, 0, 0)},
}


def _write_source(directory: Path) -> Path:
    pd.DataFrame(
        [
            (division, region, band, count)
            for division, bands in REGION_COUNTS.items()
            for band, counts in bands.items()
            for region, count in zip(REGIONS, counts, strict=True)
        ],
        columns=["sic_division", "region_code", "turnover_band", "enterprises"],
    ).to_csv(directory / UK_FIRM_REGION_TURNOVER_FILE, index=False)
    pd.DataFrame(
        [
            (group, status, band, count)
            for group, bands in STATUS_COUNTS.items()
            for band, counts in bands.items()
            for status, count in zip(STATUSES, counts, strict=True)
        ],
        columns=["industry_group", "legal_status", "turnover_band", "enterprises"],
    ).to_csv(directory / UK_FIRM_LEGAL_STATUS_TURNOVER_FILE, index=False)
    return directory


@pytest.fixture
def source(tmp_path: Path) -> UKFirmAttributeSourceData:
    return read_uk_firm_attribute_source_data(_write_source(tmp_path))


def _firms(division: list[int], turnover: list[float], **extra) -> pd.DataFrame:
    n = len(division)
    frame = pd.DataFrame(
        {
            "firm_id": np.arange(n, dtype=np.int64),
            "sic_code": [str(d).zfill(5) for d in division],
            "annual_turnover_k": np.asarray(turnover, dtype=np.float32),
            "employment": np.ones(n, dtype=np.int64),
            "firm_weight": np.ones(n, dtype=np.float32),
            "base_weight": np.ones(n, dtype=np.float64),
            "in_frame": np.ones(n, dtype=bool),
            "vat_scope": np.ones(n, dtype=bool),
        }
    )
    for name, values in extra.items():
        frame[name] = values
    return frame


def _large_frame(seed: int = 7) -> pd.DataFrame:
    """Firms in division 1 and 45, turnover spread over every band."""

    rng = np.random.default_rng(seed)
    n = 60_000
    division = rng.choice([1, 45], size=n)
    turnover = rng.choice([10.0, 75.0, 120.0, 400.0], size=n)
    frame = _firms(division.tolist(), turnover.tolist())
    frame["base_weight"] = rng.uniform(0.5, 1.5, size=n)
    return frame


# Parsing ------------------------------------------------------------------


def test_band_lowers_parse_closed_and_open_labels() -> None:
    lowers = parse_turnover_band_lowers(["50-99", "0-49", "5000+", "100-4999"])
    assert lowers == {"0-49": 0.0, "50-99": 50.0, "100-4999": 100.0, "5000+": 5000.0}


@pytest.mark.parametrize(
    "labels",
    [
        ["0-49", "50-99", "Total", "100+"],
        ["0-49", "50-99"],
        ["0-49", "100+", "200+"],
        ["0-49", "60-99", "100+"],
        ["5-49", "50+"],
    ],
)
def test_band_lowers_refuse_malformed_labels(labels) -> None:
    with pytest.raises(ValueError):
        parse_turnover_band_lowers(labels)


def test_band_index_uses_contiguous_lower_edges() -> None:
    lowers = parse_turnover_band_lowers(["0-49", "50-99", "100-249", "250+"])
    turnover = np.array([0.1, 49.999, 50.0, 99.5, 100.0, 249.9, 250.0, 1e9])
    assert turnover_band_index(turnover, lowers).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_industry_groups_are_parsed_from_their_keys() -> None:
    mapping = industry_group_divisions(["01-03", "45", "05-09"])
    assert mapping == {
        1: "01-03",
        2: "01-03",
        3: "01-03",
        45: "45",
        5: "05-09",
        6: "05-09",
        7: "05-09",
        8: "05-09",
        9: "05-09",
    }
    with pytest.raises(ValueError, match="division 3"):
        industry_group_divisions(["01-03", "03-05"])
    with pytest.raises(ValueError):
        industry_group_divisions(["Total"])


def test_source_data_refuses_missing_columns(tmp_path: Path) -> None:
    directory = _write_source(tmp_path)
    frame = pd.read_csv(directory / UK_FIRM_REGION_TURNOVER_FILE)
    with pytest.raises(ValueError, match="missing columns"):
        UKFirmAttributeSourceData(
            region_turnover=frame.drop(columns="region_code"),
            legal_status_turnover=pd.read_csv(
                directory / UK_FIRM_LEGAL_STATUS_TURNOVER_FILE
            ),
        )


def test_source_data_refuses_a_legal_status_outside_one_to_seven(
    tmp_path: Path,
) -> None:
    directory = _write_source(tmp_path)
    statuses = pd.read_csv(directory / UK_FIRM_LEGAL_STATUS_TURNOVER_FILE)
    statuses.loc[0, "legal_status"] = 8
    with pytest.raises(ValueError, match="1 to 7"):
        UKFirmAttributeSourceData(
            region_turnover=pd.read_csv(directory / UK_FIRM_REGION_TURNOVER_FILE),
            legal_status_turnover=statuses,
        )


# Draws --------------------------------------------------------------------


def test_other_columns_are_unchanged_and_input_is_not_mutated(source) -> None:
    firms = _firms([1, 2, 45, 99], [10.0, 75.0, 400.0, 20.0])
    firms.index = pd.Index([10, 11, 12, 13])
    before = firms.copy(deep=True)
    result = assign_uk_firm_attributes(firms, source, seed=3)
    pd.testing.assert_frame_equal(firms, before)
    assert list(result.columns) == [
        *firms.columns,
        "ons_region_code",
        "ons_legal_status",
    ]
    pd.testing.assert_frame_equal(result[firms.columns], before)


def test_output_dtypes(source) -> None:
    result = assign_uk_firm_attributes(
        _firms([1, 2, 45], [10.0, 75.0, 400.0]), source, seed=0
    )
    assert result["ons_region_code"].dtype == object
    assert all(isinstance(code, str) for code in result["ons_region_code"])
    assert set(result["ons_region_code"]) <= set(REGIONS)
    assert result["ons_legal_status"].dtype == np.int64
    assert result["ons_legal_status"].between(1, 7).all()


def test_draws_are_reproducible_by_seed(source) -> None:
    firms = _large_frame()
    first = assign_uk_firm_attributes(firms, source, seed=11)
    again = assign_uk_firm_attributes(firms, source, seed=11)
    other = assign_uk_firm_attributes(firms, source, seed=12)
    pd.testing.assert_frame_equal(first, again)
    assert not first["ons_region_code"].equals(other["ons_region_code"])


def test_zero_count_categories_are_never_drawn(source) -> None:
    # Division 1, "100+": region shares 50/900/50; group 45 "150+" is all
    # status 1; division 45 "100+" is all London.
    firms = _firms([45] * 500, [400.0] * 500)
    result = assign_uk_firm_attributes(firms, source, seed=5)
    assert set(result["ons_region_code"]) == {"E12000007"}
    assert set(result["ons_legal_status"]) == {1}


def _shares(codes: pd.Series, weight: np.ndarray, categories) -> np.ndarray:
    totals = np.array(
        [weight[(codes == category).to_numpy()].sum() for category in categories]
    )
    return totals / totals.sum()


@pytest.mark.parametrize(
    ("division", "turnover", "region_band", "status_group", "status_band"),
    [
        (1, 10.0, "0-49", "01-03", "0-49"),
        (1, 75.0, "50-99", "01-03", "50-149"),
        (1, 120.0, "100+", "01-03", "50-149"),
        (1, 400.0, "100+", "01-03", "150+"),
        (45, 10.0, "0-49", "45", "0-49"),
        (45, 75.0, "50-99", "45", "0-49"),
    ],
)
def test_base_weighted_shares_track_source_cells(
    source, division, turnover, region_band, status_group, status_band
) -> None:
    firms = _large_frame()
    result = assign_uk_firm_attributes(firms, source, seed=21)
    in_cell = (
        (firms["sic_code"].astype(int) == division)
        & (firms["annual_turnover_k"] == turnover)
    ).to_numpy()
    weight = firms["base_weight"].to_numpy()[in_cell]
    expected_region = np.array(
        REGION_COUNTS[str(division).zfill(2)][region_band], dtype=float
    )
    np.testing.assert_allclose(
        _shares(result["ons_region_code"][in_cell], weight, REGIONS),
        expected_region / expected_region.sum(),
        atol=0.025,
    )
    if turnover == 75.0 and division == 45:
        # Group 45's "50-149" cell is empty: the group row total applies.
        expected_status = np.sum(list(STATUS_COUNTS["45"].values()), axis=0)
    else:
        expected_status = np.array(STATUS_COUNTS[status_group][status_band])
    expected_status = np.asarray(expected_status, dtype=float)
    np.testing.assert_allclose(
        _shares(result["ons_legal_status"][in_cell], weight, STATUSES),
        expected_status / expected_status.sum(),
        atol=0.025,
    )


# Fallbacks ----------------------------------------------------------------


def _levels(source, firms: pd.DataFrame) -> dict[tuple[str, str], int]:
    counts = uk_firm_attribute_fallback_counts(firms, source)
    return {
        (row.attribute, row.level): row.firms
        for row in counts.itertuples(index=False)
        if row.firms
    }


def test_fallback_levels_cover_every_path(source) -> None:
    firms = _firms(
        # cell; empty cell (division 2, "50-99") -> row; unknown division -> UK;
        # out of frame (in_frame False) -> row; zero turnover -> row;
        # out of frame with an unknown division -> UK.
        [1, 2, 99, 1, 45, 99],
        [10.0, 75.0, 20.0, 30.0, 0.0, 0.0],
        in_frame=[True, True, True, False, True, False],
    )
    assert _levels(source, firms) == {
        ("ons_region_code", "cell"): 1,
        ("ons_region_code", "row"): 3,
        ("ons_region_code", "uk"): 2,
        # Division 2 is in group "01-03", whose "50-149" cell is populated.
        ("ons_legal_status", "cell"): 2,
        ("ons_legal_status", "row"): 2,
        ("ons_legal_status", "uk"): 2,
    }


def test_fallback_counts_report_every_level_with_base_weight(source) -> None:
    firms = _firms([1, 99], [10.0, 10.0])
    firms["base_weight"] = [2.5, 4.0]
    counts = uk_firm_attribute_fallback_counts(firms, source)
    assert list(counts.columns) == ["attribute", "level", "firms", "weight"]
    assert len(counts) == 2 * len(UK_FIRM_ATTRIBUTE_FALLBACK_LEVELS)
    region = counts[counts["attribute"] == "ons_region_code"].set_index("level")
    assert region.loc["cell", "weight"] == 2.5
    assert region.loc["row", "weight"] == 0.0
    assert region.loc["uk", "weight"] == 4.0


def test_row_and_uk_fallbacks_draw_from_their_totals(source) -> None:
    n = 40_000
    # Division 2 in frame at 75 (empty cell) -> division 2 row: 100 / 0 / 100.
    row = assign_uk_firm_attributes(_firms([2] * n, [75.0] * n), source, seed=1)
    np.testing.assert_allclose(
        _shares(row["ons_region_code"], np.ones(n), REGIONS),
        [0.5, 0.0, 0.5],
        atol=0.02,
    )
    # Unknown division -> UK total over every division and band.
    uk = assign_uk_firm_attributes(_firms([99] * n, [20.0] * n), source, seed=1)
    uk_counts = np.sum(
        [counts for bands in REGION_COUNTS.values() for counts in bands.values()],
        axis=0,
    ).astype(float)
    np.testing.assert_allclose(
        _shares(uk["ons_region_code"], np.ones(n), REGIONS),
        uk_counts / uk_counts.sum(),
        atol=0.02,
    )
    # Out of frame in division 1 -> division 1 row total, whatever the turnover.
    out = assign_uk_firm_attributes(
        _firms([1] * n, [10.0] * n, in_frame=np.zeros(n, dtype=bool)),
        source,
        seed=1,
    )
    row_counts = np.sum(list(REGION_COUNTS["01"].values()), axis=0).astype(float)
    np.testing.assert_allclose(
        _shares(out["ons_region_code"], np.ones(n), REGIONS),
        row_counts / row_counts.sum(),
        atol=0.02,
    )


def test_nan_turnover_is_refused(source) -> None:
    with pytest.raises(ValueError, match="NaN"):
        assign_uk_firm_attributes(_firms([1], [float("nan")]), source, seed=0)


# Real fixtures ------------------------------------------------------------


def test_real_fixtures_cover_every_ons_division() -> None:
    source = read_uk_firm_attribute_source_data(ATTRIBUTE_FIXTURES)
    turnover = pd.read_csv(
        FIXTURES / "firm_2024_25" / "ons_firm_turnover.csv", dtype={"SIC Code": str}
    )
    divisions = [
        int(code) for code in turnover["SIC Code"].dropna() if code.strip().isdigit()
    ]
    groups = industry_group_divisions(
        source.legal_status_turnover["industry_group"].astype(str).str.strip()
    )
    assert sorted(set(divisions) - set(groups)) == []
    region_divisions = set(
        source.region_turnover["sic_division"].astype(str).str.strip().astype(int)
    )
    assert sorted(set(divisions) - region_divisions) == []
    # Table 8 uses the generator's seven ONS bands, labelled alike.
    assert list(
        parse_turnover_band_lowers(source.region_turnover["turnover_band"])
    ) == [label for label in turnover.columns if label[0].isdigit()]
    # Divisions Table 8 counts no enterprise in (05, 07, 12, ...) have none in
    # the generator's frame either, so only out-of-frame traders reach the UK
    # total; every division with enterprises resolves to a cell or its row.
    populated = (
        turnover.loc[turnover["SIC Code"].str.strip().str.isdigit(), "Total"]
        .astype(float)
        .gt(0)
        .to_numpy()
    )
    populated_divisions = [
        d for d, kept in zip(divisions, populated, strict=True) if kept
    ]
    firms = _firms(
        populated_divisions,
        np.linspace(1.0, 60_000.0, num=len(populated_divisions)).tolist(),
    )
    counts = uk_firm_attribute_fallback_counts(firms, source)
    assert set(counts.loc[counts["firms"] > 0, "level"]) <= {"cell", "row"}
