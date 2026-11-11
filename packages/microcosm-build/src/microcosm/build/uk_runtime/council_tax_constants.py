"""UK council-tax bands: the engine's set and the bands each nation uses."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM

#: The engine's ``council_tax_band`` members in enum order. Bands A-H are the
#: England and Scotland systems; band I exists in Wales only (the 2005
#: revaluation). Appended last, so the positional indices of A-H never move.
UK_COUNCIL_TAX_BANDS: tuple[str, ...] = tuple("ABCDEFGHI")

#: The band the FRS spine stores for a household the survey gives no band:
#: every Northern Ireland household, which pays domestic rates rather than
#: council tax, and any other unbanded ``ctband`` code.
UK_COUNCIL_TAX_PLACEHOLDER_BAND = "D"

#: The bands each nation's households can hold, by the engine's ``country``
#: member names. Northern Ireland holds only the placeholder.
UK_COUNCIL_TAX_BANDS_BY_NATION: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ENGLAND": UK_COUNCIL_TAX_BANDS[:8],
        "WALES": UK_COUNCIL_TAX_BANDS,
        "SCOTLAND": UK_COUNCIL_TAX_BANDS[:8],
        "NORTHERN_IRELAND": (UK_COUNCIL_TAX_PLACEHOLDER_BAND,),
    }
)

#: The nation of each region-tier GSS code, by its first letter.
_NATION_BY_GSS_PREFIX: Mapping[str, str] = MappingProxyType(
    {"E": "ENGLAND", "W": "WALES", "S": "SCOTLAND", "N": "NORTHERN_IRELAND"}
)

#: The spine's ``region`` member -> its nation, derived from the region tier so
#: the twelve regions cannot drift from the roster calibration fans out over.
UK_NATION_BY_REGION: Mapping[str, str] = MappingProxyType(
    {
        region: _NATION_BY_GSS_PREFIX[code[0]]
        for code, region in UK_REGION_TIER_ENUM.items()
    }
)

#: The spine's ``region`` member -> the bands its households can hold.
UK_COUNCIL_TAX_BANDS_BY_REGION: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        region: UK_COUNCIL_TAX_BANDS_BY_NATION[nation]
        for region, nation in UK_NATION_BY_REGION.items()
    }
)

__all__ = [
    "UK_COUNCIL_TAX_BANDS",
    "UK_COUNCIL_TAX_BANDS_BY_NATION",
    "UK_COUNCIL_TAX_BANDS_BY_REGION",
    "UK_COUNCIL_TAX_PLACEHOLDER_BAND",
    "UK_NATION_BY_REGION",
]
