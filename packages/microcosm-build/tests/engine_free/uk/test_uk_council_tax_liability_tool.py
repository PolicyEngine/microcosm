"""The CT-C comparison of the engine's council tax with the stored bill (microcosm#1095)."""

from __future__ import annotations

from itertools import count

import pandas as pd

from tools.compare_uk_council_tax_liability import summarise

_IDS = count(1)


def _households(
    code: str,
    band: str,
    sources: list[int],
    *,
    stored: float = 2_000.0,
    simulated: float = 2_400.0,
    less_discounts: float = 1_800.0,
    region: str = "SOUTH_EAST",
    country: str = "ENGLAND",
    weight: float = 10.0,
) -> list[dict[str, object]]:
    return [
        {
            "household_id": next(_IDS),
            "household_weight": weight,
            "local_authority": f"AUTHORITY_{code}",
            "council_tax_band": band,
            "country": country,
            "region": region,
            "council_tax": stored,
            "simulated_council_tax": simulated,
            "simulated_council_tax_less_discounts": less_discounts,
            "la_code_oa": code,
            "household_source_id": source,
        }
        for source in sources
    ]


def _cell(report: dict, code: str, band: str) -> dict:
    return next(
        cell
        for cell in report["cells"]
        if cell["la_code_oa"] == code and cell["council_tax_band"] == band
    )


def test_a_supported_cell_gives_means_and_ratios_and_a_thin_one_is_withheld() -> None:
    table = pd.DataFrame(
        _households("E07000110", "D", list(range(100, 112)))
        # Two source households: under the floor of three.
        + _households("E07000110", "H", [200, 201], simulated=4_800.0)
    )

    report = summarise(table)

    supported = _cell(report, "E07000110", "D")
    assert supported["source_households"] == 12
    assert supported["means"] == {
        "council_tax": 2_000.0,
        "simulated_council_tax": 2_400.0,
        "simulated_council_tax_less_discounts": 1_800.0,
    }
    assert supported["ratios"] == {
        "simulated_council_tax_over_council_tax": 1.2,
        "simulated_council_tax_less_discounts_over_council_tax": 0.9,
    }
    # A cell never gives a total or a weighted count, so a published total
    # less the published cells cannot give a withheld cell back.
    assert "totals" not in supported
    assert _cell(report, "E07000110", "H") == {
        "la_code_oa": "E07000110",
        "local_authority": "AUTHORITY_E07000110",
        "council_tax_band": "H",
        "withheld": True,
    }
    assert report["great_britain"]["source_households"] == 14
    assert report["great_britain"]["totals"]["council_tax"] == 14 * 10.0 * 2_000.0


def test_clones_of_one_source_household_count_once() -> None:
    # Five rows, but two source households: under the floor of three.
    sources = [300, 300, 300, 301, 301]
    table = pd.DataFrame(_households("E09000033", "C", sources, region="LONDON"))

    report = summarise(table)

    assert _cell(report, "E09000033", "C")["withheld"]
    assert report["by_region"]["LONDON"] == {"withheld": True}


def test_northern_ireland_is_left_out_and_a_thin_region_is_withheld() -> None:
    table = pd.DataFrame(
        _households("E07000110", "D", list(range(400, 412)))
        + _households(
            "W06000015",
            "B",
            [500, 501],
            region="WALES",
            country="WALES",
        )
        + _households(
            "N09000003",
            "D",
            list(range(600, 615)),
            region="NORTHERN_IRELAND",
            country="NORTHERN_IRELAND",
            simulated=0.0,
            less_discounts=0.0,
            stored=0.0,
        )
    )

    report = summarise(table)

    assert {cell["la_code_oa"] for cell in report["cells"]} == {
        "E07000110",
        "W06000015",
    }
    assert "NORTHERN_IRELAND" not in report["by_country"]
    assert report["by_country"]["WALES"] == {"withheld": True}
    assert report["by_region"]["SOUTH_EAST"]["source_households"] == 12
    assert report["great_britain"]["source_households"] == 14


def test_an_authority_the_engine_charges_nothing_is_named() -> None:
    table = pd.DataFrame(
        _households("E07000110", "D", list(range(700, 712)))
        + _households(
            "E07000048",
            "D",
            [800],
            simulated=0.0,
            less_discounts=0.0,
        )
    )

    assert summarise(table)["authorities_without_band_d"] == ["E07000048"]
