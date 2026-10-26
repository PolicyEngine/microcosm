"""The council-tax band constants and the views derived from them agree."""

from __future__ import annotations

from microcosm.build.uk_runtime import frs_spine, local_targets
from microcosm.build.uk_runtime.council_tax_constants import (
    UK_COUNCIL_TAX_BANDS,
    UK_COUNCIL_TAX_BANDS_BY_NATION,
    UK_COUNCIL_TAX_BANDS_BY_REGION,
    UK_COUNCIL_TAX_PLACEHOLDER_BAND,
    UK_NATION_BY_REGION,
)
from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM


def test_each_nations_bands_are_engine_bands() -> None:
    assert UK_COUNCIL_TAX_BANDS == tuple("ABCDEFGHI")
    for bands in UK_COUNCIL_TAX_BANDS_BY_NATION.values():
        assert set(bands) <= set(UK_COUNCIL_TAX_BANDS)
    assert UK_COUNCIL_TAX_BANDS_BY_NATION["WALES"] == UK_COUNCIL_TAX_BANDS
    assert "I" not in UK_COUNCIL_TAX_BANDS_BY_NATION["ENGLAND"]
    assert "I" not in UK_COUNCIL_TAX_BANDS_BY_NATION["SCOTLAND"]
    assert UK_COUNCIL_TAX_BANDS_BY_NATION["NORTHERN_IRELAND"] == (
        UK_COUNCIL_TAX_PLACEHOLDER_BAND,
    )


def test_region_views_follow_the_region_tier() -> None:
    assert set(UK_NATION_BY_REGION) == set(UK_REGION_TIER_ENUM.values())
    assert set(UK_NATION_BY_REGION.values()) == set(UK_COUNCIL_TAX_BANDS_BY_NATION)
    assert sum(nation == "ENGLAND" for nation in UK_NATION_BY_REGION.values()) == 9
    for region, nation in UK_NATION_BY_REGION.items():
        expected = UK_COUNCIL_TAX_BANDS_BY_NATION[nation]
        assert UK_COUNCIL_TAX_BANDS_BY_REGION[region] == expected


def test_the_spine_and_the_local_metrics_use_the_shared_bands() -> None:
    assert frs_spine.COUNCIL_TAX_BAND_MAP == {
        code: band for code, band in enumerate(UK_COUNCIL_TAX_BANDS, start=1)
    }
    council_tax_metrics = [
        name
        for name in local_targets.metric_names("la")
        if name.startswith("council_tax/")
    ]
    # The bands, then England's single adult discount (microcosm#1095).
    assert council_tax_metrics == [
        *(f"council_tax/band_{band.lower()}" for band in UK_COUNCIL_TAX_BANDS),
        "council_tax/single_adult_discount",
    ]
