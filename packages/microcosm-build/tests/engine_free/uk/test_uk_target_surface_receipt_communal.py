"""The communal-establishment receipt sizes the age-universe change (microcosm#1278).

Synthetic facts only: the receipt reads census residence-type counts by area
and age band, so the tests write a few such facts and check the shares, the
band mapping, the region composition and the approximate Northern Ireland path.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_tool():
    path = _TEST_PATHS.repository / "tools/uk_target_surface_receipt.py"
    spec = importlib.util.spec_from_file_location("uk_target_surface_receipt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _ew(level, area, age, residence, value, sex="All persons"):
    return {
        "observed_measure": {"source_concept": "ons.census2021_residents"},
        "geography": {"level": level, "id": area},
        "dimensions": {
            "C2021_AGE_24_NAME": age,
            "C2021_RESTYPE_3_NAME": residence,
            "C_SEX_NAME": sex,
        },
        "value": value,
    }


def _scotland(area, age, residence, value):
    return {
        "observed_measure": {"source_concept": "nrs.census2022_residents"},
        "geography": {"level": "local_authority", "id": area},
        "dimensions": {
            "Age": age,
            "Residence Type Indicator": residence,
            "Sex": "All people",
        },
        "value": value,
    }


def _facts(tmp_path):
    household, communal = "Lives in a household", "Lives in a communal establishment"
    rows = []
    for level, area in (
        ("constituency", "E14000001"),
        ("local_authority", "E06000001"),
    ):
        rows += [
            _ew(level, area, "Aged 20 to 24 years", household, 600.0),
            _ew(level, area, "Aged 20 to 24 years", communal, 300.0),
            _ew(level, area, "Aged 25 to 29 years", household, 100.0),
            _ew(level, area, "Aged 18 to 19 years", household, 50.0),
            _ew(level, area, "Aged 18 to 19 years", communal, 50.0),
            _ew(level, area, "Aged 2 years and under", household, 90.0),
            _ew(level, area, "Aged 85 years and over", household, 30.0),
            _ew(level, area, "Aged 85 years and over", communal, 10.0),
            _ew(level, area, "Total", household, 870.0),
            _ew(level, area, "Total", communal, 360.0),
            # Sex-split rows are ignored; the all-persons rows carry the area.
            _ew(level, area, "Aged 20 to 24 years", communal, 999.0, sex="Female"),
        ]
    rows += [
        _scotland("S12000005", "20 - 24", household, 800.0),
        _scotland("S12000005", "25 - 29", communal, 200.0),
    ]
    rows += [
        {
            "observed_measure": {
                "source_concept": "nisra.census2021_communal_residents"
            },
            "geography": {"level": "local_authority", "id": "N09000003"},
            "dimensions": {"Age": "16-24 years", "Sex": sex, "position": "resident"},
            "value": 45.0,
        }
        for sex in ("Female", "Male")
    ]
    rows += [
        {
            "observed_measure": {"source_concept": "ons.mid_year_population_estimate"},
            "geography": {
                "level": "local_authority",
                "id": "N09000003",
                "vintage": "lgd_2014",
            },
            "period": {"type": "calendar_year", "value": 2025},
            "dimensions": {"age": age, "sex": "female"},
            "value": 100.0,
        }
        for age in [*range(16, 30), "90_plus"]
    ]
    path = tmp_path / "consumer_facts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_bands_are_the_local_ten_year_bands_and_80_plus() -> None:
    tool = _load_tool()
    assert [tool._band_key(band) for band in tool.COMMUNAL_AGE_BANDS] == [
        "0_10",
        "10_20",
        "20_30",
        "30_40",
        "40_50",
        "50_60",
        "60_70",
        "70_80",
        "80_plus",
    ]
    assert tool._band_of(tool._age_range("Aged 2 years and under")) == "0_10"
    assert tool._band_of(tool._age_range("Aged 18 to 19 years")) == "10_20"
    assert tool._band_of(tool._age_range("85 and over")) == "80_plus"
    # A category that straddles two bands lands in neither.
    assert tool._band_of(tool._age_range("16-24 years")) is None


def test_shares_by_band_and_the_sizing_counts(tmp_path) -> None:
    tool = _load_tool()
    receipt = tool._communal_validation(
        _facts(tmp_path),
        region_by_area={
            "E06000001": "E12000001",
            "S12000005": "S92000003",
            "N09000003": "N92000002",
        },
    )
    # 300 communal of 1,000 residents aged 20-29 in each English area; the
    # Scottish council (local-authority grain only) sits at exactly 20%.
    for level, at_least in (
        ("constituency", {"0.20": 1, "0.25": 1}),
        ("local_authority", {"0.20": 2, "0.25": 1}),
    ):
        band = receipt[level]["bands"]["20_30"]
        assert band["areas_at_or_above"] == at_least
        top = band["areas_at_or_above_lowest"][0]
        assert top["communal_share"] == pytest.approx(0.3)
        assert receipt[level]["bands"]["80_plus"]["median_share"] == pytest.approx(0.25)
        assert receipt[level]["bands"]["0_10"]["median_share"] == 0.0
    # The 18-24 student rule is unchanged: 350 of 1,000 residents aged 18-24.
    (student,) = receipt["constituency"]["student_areas"]
    assert student["communal_share_18_24"] == pytest.approx(350 / 1000)
    # Scotland's council: 200 communal of 1,000 aged 20-29.
    scottish = receipt["local_authority"]["bands"]["20_30"]["areas_at_or_above_lowest"]
    assert {row["area"] for row in scottish} == {"E06000001", "S12000005"}
    region = receipt["region"]["bands"]["20_30"]
    assert region["E12000001"] == pytest.approx(0.3)
    assert region["S92000003"] == pytest.approx(0.2)
    # Northern Ireland: 90 communal aged 16-24 spread over nine single years,
    # so 50 fall in 20-29 against 1,000 residents (100 at each age 20-29).
    ni = receipt["northern_ireland_local_authority"]
    assert ni["approximate"] is True
    assert ni["bands"]["20_30"]["median_share"] == pytest.approx(50 / 1000)
    assert region["N92000002"] == pytest.approx(50 / 1000)


def test_communal_only_writes_the_receipt_without_a_ladder(tmp_path) -> None:
    tool = _load_tool()
    out = tmp_path / "communal.json"
    assert (
        tool.main(
            [
                "--ledger-facts",
                str(_facts(tmp_path)),
                "--communal-only",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    receipt = json.loads(out.read_text())
    assert receipt["ledger_facts"] == "consumer_facts.jsonl"
    assert receipt["communal_validation"]["sizing_shares"] == [0.2, 0.25]
    with pytest.raises(SystemExit):
        tool.main(["--ledger-facts", str(out), "--out", str(out)])
