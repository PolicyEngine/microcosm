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
    "US_STATE_FIPS_TO_POSTAL",
    "US_STATE_NUMERIC_FIPS_TO_POSTAL",
    "US_STATE_POSTAL_TO_NUMERIC_FIPS",
    "US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION",
]


UK_GEOGRAPHY_ID_TO_LABEL: Mapping[str, str] = MappingProxyType(
    {
        "K02000001": "United Kingdom",
        "K03000001": "Great Britain",
        "E92000001": "England",
        "W92000004": "Wales",
        "S92000003": "Scotland",
        "N92000002": "Northern Ireland",
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


# Census region codes, https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf
_US_CENSUS_REGION_POSTAL_CODES = {
    1: ("CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"),
    2: ("IN", "IL", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    3: (
        "DE",
        "DC",
        "FL",
        "GA",
        "MD",
        "NC",
        "SC",
        "VA",
        "WV",
        "AL",
        "KY",
        "MS",
        "TN",
        "AR",
        "LA",
        "OK",
        "TX",
    ),
    4: ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"),
}
US_STATE_NUMERIC_FIPS_TO_CENSUS_REGION: Mapping[int, int] = MappingProxyType(
    {
        US_STATE_POSTAL_TO_NUMERIC_FIPS[postal]: region
        for region, states in _US_CENSUS_REGION_POSTAL_CODES.items()
        for postal in states
    }
)
