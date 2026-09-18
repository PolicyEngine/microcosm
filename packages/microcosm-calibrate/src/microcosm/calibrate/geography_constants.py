"""Shared country geography identifiers and display-code mappings.

These mappings are static code-system facts used by both calibration
diagnostics and country build logic.  Keep their literal definitions here so
consumers import one reviewed value instead of maintaining module-local copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "UK_GEOGRAPHY_ID_TO_LABEL",
    "UK_LADDER_NATION_REGION_CODES",
    "UK_REGION_TIER",
    "UK_REGION_TIER_ENUM",
    "US_STATE_FIPS_TO_POSTAL",
    "US_STATE_NUMERIC_FIPS_TO_POSTAL",
    "US_STATE_POSTAL_TO_NUMERIC_FIPS",
]


UK_GEOGRAPHY_ID_TO_LABEL: Mapping[str, str] = MappingProxyType(
    {
        "K02000001": "United Kingdom",
        "K03000001": "Great Britain",
        "E92000001": "England",
        "W92000004": "Wales",
        "S92000003": "Scotland",
        "N92000002": "Northern Ireland",
        # The nine English regions (ONS statistical regions, E12 codes), the
        # roster UK national references may pin at region level.
        "E12000001": "North East",
        "E12000002": "North West",
        "E12000003": "Yorkshire and The Humber",
        "E12000004": "East Midlands",
        "E12000005": "West Midlands",
        "E12000006": "East of England",
        "E12000007": "London",
        "E12000008": "South East",
        "E12000009": "South West",
    }
)


#: The twelve-area region tier of the national calibration surface, in the
#: order the FRS ``gvtregno`` coding and the incumbent's regional rows use:
#: the nine English regions at Chronicle's ``region`` level, then Wales,
#: Scotland and Northern Ireland, which Chronicle stamps at ``country`` (their
#: GSS codes are country codes) but which sit at the same ITL1 tier as the
#: English regions. Two-level (country + region) contract targets fan out
#: over this roster (microcosm#905); every code is a key of
#: ``UK_GEOGRAPHY_ID_TO_LABEL``, so the schema-8 hierarchy can label it.
UK_REGION_TIER: tuple[tuple[str, str], ...] = (
    ("region", "E12000001"),  # North East
    ("region", "E12000002"),  # North West
    ("region", "E12000003"),  # Yorkshire and The Humber
    ("region", "E12000004"),  # East Midlands
    ("region", "E12000005"),  # West Midlands
    ("region", "E12000006"),  # East of England
    ("region", "E12000007"),  # London
    ("region", "E12000008"),  # South East
    ("region", "E12000009"),  # South West
    ("country", "W92000004"),  # Wales
    ("country", "S92000003"),  # Scotland
    ("country", "N92000002"),  # Northern Ireland
)

#: Region-tier GSS code -> the spine's ``region`` enum name (``REGION_MAP`` in
#: ``frs_spine``), the value a fan-out row's geography predicate compares
#: against. Kept beside the roster so the two cannot drift apart.
UK_REGION_TIER_ENUM: Mapping[str, str] = MappingProxyType(
    {
        "E12000001": "NORTH_EAST",
        "E12000002": "NORTH_WEST",
        "E12000003": "YORKSHIRE",
        "E12000004": "EAST_MIDLANDS",
        "E12000005": "WEST_MIDLANDS",
        "E12000006": "EAST_OF_ENGLAND",
        "E12000007": "LONDON",
        "E12000008": "SOUTH_EAST",
        "E12000009": "SOUTH_WEST",
        "W92000004": "WALES",
        "S92000003": "SCOTLAND",
        "N92000002": "NORTHERN_IRELAND",
    }
)

#: The ladder's nation pseudo region codes -> the GSS country code the region
#: tier uses for the same area, so an OA-derived region membership can be
#: expressed in tier codes.
UK_LADDER_NATION_REGION_CODES: Mapping[str, str] = MappingProxyType(
    {
        "W99999999": "W92000004",
        "S99999999": "S92000003",
        "N99999999": "N92000002",
    }
)

# Two-character Census state FIPS code -> USPS postal abbreviation.  This
# project treats the District of Columbia as a state-level geography.
US_STATE_FIPS_TO_POSTAL: Mapping[str, str] = MappingProxyType(
    {
        "01": "AL",
        "02": "AK",
        "04": "AZ",
        "05": "AR",
        "06": "CA",
        "08": "CO",
        "09": "CT",
        "10": "DE",
        "11": "DC",
        "12": "FL",
        "13": "GA",
        "15": "HI",
        "16": "ID",
        "17": "IL",
        "18": "IN",
        "19": "IA",
        "20": "KS",
        "21": "KY",
        "22": "LA",
        "23": "ME",
        "24": "MD",
        "25": "MA",
        "26": "MI",
        "27": "MN",
        "28": "MS",
        "29": "MO",
        "30": "MT",
        "31": "NE",
        "32": "NV",
        "33": "NH",
        "34": "NJ",
        "35": "NM",
        "36": "NY",
        "37": "NC",
        "38": "ND",
        "39": "OH",
        "40": "OK",
        "41": "OR",
        "42": "PA",
        "44": "RI",
        "45": "SC",
        "46": "SD",
        "47": "TN",
        "48": "TX",
        "49": "UT",
        "50": "VT",
        "51": "VA",
        "53": "WA",
        "54": "WV",
        "55": "WI",
        "56": "WY",
    }
)


# Numeric views are derived from the canonical zero-padded mapping so a caller
# that reads integer HDF columns does not maintain another copy of the codes.
US_STATE_NUMERIC_FIPS_TO_POSTAL: Mapping[int, str] = MappingProxyType(
    {int(fips): postal for fips, postal in US_STATE_FIPS_TO_POSTAL.items()}
)
US_STATE_POSTAL_TO_NUMERIC_FIPS: Mapping[str, int] = MappingProxyType(
    {postal: fips for fips, postal in US_STATE_NUMERIC_FIPS_TO_POSTAL.items()}
)
