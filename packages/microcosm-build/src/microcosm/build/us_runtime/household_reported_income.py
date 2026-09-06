"""Pure comparable household reported money income for the ACS and ASEC arms.

Both arms measure the *publisher's own* household total. The ACS arm reads
``HINCP``; the prepared ASEC arm reads ``asec_HTOTVAL``. Neither arm sums
components: the ACS person roster carried by the executed spine holds six of
the eight published ACS income types, so a component sum would be an
incomplete total wearing a complete total's name.

The module runs on explicitly supplied projections of native source columns.
It constructs no Frame, loads no archive, imports no country model, and reads
no engine output. It states periods and universes rather than reconciling
them: the ASEC calendar-year and ACS rolling-twelve-month reference periods
stay visible in every artifact, and no combined two-arm value is produced.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import metadata, resources
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.quality_aggregates import DistributionBin, DistributionSpec

__all__ = [
    "ACS_2024_HINCP_ORIGIN",
    "ACS_CARRIED_REPORTED_INCOME",
    "ACS_UNCARRIED_INCOME_TYPES",
    "ASEC_HTOTVAL_ORIGIN",
    "ASEC_MEASUREMENT_REQUIRES_INDEPENDENT_HU_MASK",
    "AVAILABILITY",
    "B19001_2024_BIN_DEFINITION_ID",
    "B19001_2024_METADATA",
    "CONTRACT",
    "HouseholdReportedIncome",
    "QUALITY",
    "ReportedIncomeOrigin",
    "ReportedIncomeRefusalError",
    "align_household_values",
    "b19001_household_income_distribution_spec",
    "measure_acs_household_reported_income",
    "measure_asec_household_reported_income",
    "two_arm_comparability",
]

CONTRACT = "us_household_reported_money_income_v1"

#: Why a household has no measured amount. ``out_of_universe`` is only used
#: where the publisher's own dictionary declares the field not applicable.
AVAILABILITY: Mapping[str, int] = MappingProxyType(
    {
        "observed": 0,
        "unavailable_out_of_universe": 1,
        "unavailable_in_universe": 2,
    }
)
#: What the publisher says about the observed amount itself.
QUALITY: Mapping[str, int] = MappingProxyType(
    {
        "exact_publisher_total": 0,
        "declared_open_ended_bound": 1,
        "not_applicable": 2,
    }
)

#: The ASEC arm's raw total carries no occupied-housing-unit determination of
#: its own: `measure_asec_household_reported_income` never derives an
#: out-of-universe code, because HTOTVAL's printed universe is All Households
#: with no declared N/A pattern. A published household count excluding
#: group-quarters residents does not mean the underlying file has no
#: group-quarters rows -- the prepared ASEC household table retains the raw
#: fields `asec_housing_universe` classifies -- so a B19001-style consumer
#: must bind that independently carried housing-universe mask before treating
#: an ASEC household as occupied-HU in scope. This module does not compute or
#: expose that mask; it only states the requirement.
ASEC_MEASUREMENT_REQUIRES_INDEPENDENT_HU_MASK = (
    "unbound_by_this_measurement: the ASEC arm's raw asec_HTOTVAL total does "
    "not by itself establish occupied-housing-unit membership; a "
    "B19001-style consumer must bind the independently carried ASEC "
    "housing-universe mask before counting an ASEC household in an "
    "occupied-HU universe"
)

_ACS_DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
    "data_dict/PUMS_Data_Dictionary_2024.pdf"
)
_ACS_DICTIONARY_SHA256 = (
    "929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff"
)
_ACS_USER_GUIDE_URL = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/"
    "2024ACS_PUMS_User_Guide.pdf"
)
_ACS_USER_GUIDE_SHA256 = (
    "70d485604dfd89534eef7d491a219ab7d336f98e724732985a0a859eac5d9787"
)
_ACS_SUBJECT_DEFINITIONS_URL = (
    "https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/"
    "2024_ACSSubjectDefinitions.pdf"
)
_ACS_SUBJECT_DEFINITIONS_SHA256 = (
    "6e0c5b1a5d3ebf7e300e583ac6424cbd5f09de50d69e7ef03aa6d070d9c4a7a8"
)

#: The published ACS person income types the executed spine already carries,
#: mapped to the reported quantity each one is adjusted into. This is six of
#: the publisher's eight types: ``PAP`` and ``OIP`` are not read, which is why
#: no component sum can stand in for ``HINCP``.
ACS_CARRIED_REPORTED_INCOME: Mapping[str, str] = MappingProxyType(
    {
        "WAGP": "employment_income_before_lsr",
        "SEMP": "self_employment_income_before_lsr",
        "SSIP": "ssi_reported",
        "SSP": "acs_social_security_income",
        "RETP": "acs_retirement_income",
        "INTP": "acs_interest_dividend_rental_income",
    }
)
#: The two published types the executed spine does not read.
ACS_UNCARRIED_INCOME_TYPES = ("PAP", "OIP")

#: The single ADJINC code the pinned 2024 1-year dictionary enumerates.
ACS_2024_DECLARED_ADJINC = 1_015_250
ACS_ADJUSTMENT_DENOMINATOR = 1_000_000
#: ``HINCP`` printed range: the bottom code is the declared open-ended loss.
ACS_HINCP_MINIMUM = -59_999
ACS_HINCP_MAXIMUM = 9_999_999
#: Housing-record ``TYPEHUGQ``: 1 housing unit, 2/3 group-quarters placeholder.
ACS_HOUSING_UNIT_KIND = 1
ACS_GROUP_QUARTERS_KINDS = (2, 3)

_ASEC_TECHDOC_URL = "https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf"
_ASEC_TECHDOC_SHA256 = (
    "7f0cb9f791f2737ad12fd552b8cb201f4d1de266eac160322597344e73499fb4"
)

_ASEC_PRICE_BASIS_RESOURCE = "asec_current_money_price_basis_v1.json"
_ASEC_PRICE_BASIS_SHA256 = (
    "88cec207a3c73e5de93d677acf16e8db9ddae9761c644c06da92cacf0feb3f63"
)
#: The Census workbook's own bytes, as recorded in the packaged resource's
#: immutable ``workbook`` object -- distinct from ``_ASEC_PRICE_BASIS_SHA256``,
#: which authenticates the packaged JSON itself, not the workbook it restates.
_ASEC_PRICE_BASIS_WORKBOOK_URL = (
    "https://www2.census.gov/programs-surveys/demo/tables/p60/"
    "286/annual-index-value_annual-percent-change.xls"
)
_ASEC_PRICE_BASIS_WORKBOOK_SHA256 = (
    "8a5ef078b02c48c9e6414f3ca0cbdfe4a013ce6acf1a3fda42e6bd9c7da49e48"
)
_ASEC_PRICE_BASIS_WORKBOOK_SIZE_BYTES = 41_472
ASEC_HTOTVAL_MINIMUM = -999_999
ASEC_HTOTVAL_MAXIMUM = 99_999_999
ASEC_INCOME_YEARS = (2022, 2023, 2024)
#: Income year -> (survey year, dictionary SHA-256, 1-based dictionary page).
ASEC_HTOTVAL_VINTAGES: Mapping[int, tuple[int, str, int]] = MappingProxyType(
    {
        2022: (
            2023,
            "66bd6e3fe516233ab63b75c60573451222b3b3d3235d61cfed96f64608de2117",
            5,
        ),
        2023: (
            2024,
            "761c67ea53f5c3264329b3e9ddbdd802826ba4a859122b8a8a2f29ab85b3a840",
            5,
        ),
        2024: (
            2025,
            "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f",
            12,
        ),
    }
)
_ASEC_DICTIONARY_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/cps/datasets/"
    "{survey_year}/march/asec{survey_year}_ddl_pub_full.pdf"
)


class ReportedIncomeRefusalError(ValueError):
    """Value-free refusal reason for an unsupported reported-income input."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ReportedIncomeRefusalError(reason)


@dataclass(frozen=True, eq=False)
class ReportedIncomeOrigin:
    """Where one arm's household total comes from and what it means."""

    arm: str
    publisher_column: str
    concept: str
    universe_as_printed: str
    observation_period: str
    reference_period: str
    source_price_period: str
    target_price_period: str
    transformation: str
    quoted_locator: str
    authority: tuple[Mapping[str, object], ...]

    def as_json(self) -> dict[str, object]:
        """Return an independent JSON-safe copy of this origin."""

        return json.loads(json.dumps(_origin_body(self), sort_keys=True))


def _origin_body(origin: ReportedIncomeOrigin) -> dict[str, object]:
    return {
        "arm": origin.arm,
        "publisher_column": origin.publisher_column,
        "concept": origin.concept,
        "universe_as_printed": origin.universe_as_printed,
        "observation_period": origin.observation_period,
        "reference_period": origin.reference_period,
        "source_price_period": origin.source_price_period,
        "target_price_period": origin.target_price_period,
        "transformation": origin.transformation,
        "quoted_locator": origin.quoted_locator,
        "authority": [dict(item) for item in origin.authority],
    }


ACS_2024_HINCP_ORIGIN = ReportedIncomeOrigin(
    arm="acs_2024_1yr",
    publisher_column="HINCP",
    concept="Household income (past 12 months)",
    universe_as_printed="Occupied housing units; N/A (GQ/vacant)",
    observation_period="rolling_past_twelve_months",
    reference_period="twelve months preceding each interview month in 2024",
    source_price_period="rolling_twelve_months_ending_in_interview_month",
    target_price_period="calendar_2024_dollars_census_adjinc",
    transformation="HINCP * ADJINC / 1_000_000, applied exactly once",
    quoted_locator=(
        "HINCP Numeric 7 / Household income (past 12 months, use ADJINC to "
        "adjust HINCP to constant dollars) / bbbbbbb .N/A (GQ/vacant) / "
        "0 .No household income / -59999 .Loss of -$59,999 or more / "
        "-59998..-1 .Loss of $1 to -$59,998 / 1..9999999 .Total household "
        "income in dollars (Components are rounded)"
    ),
    authority=(
        MappingProxyType(
            {
                "document": "2024 ACS PUMS Data Dictionary (October 16, 2025)",
                "url": _ACS_DICTIONARY_URL,
                "sha256": _ACS_DICTIONARY_SHA256,
                "pdf_page_1based": 15,
                "establishes": "HINCP concept, universe, sentinel and dollars",
            }
        ),
        MappingProxyType(
            {
                "document": "2024 ACS PUMS Data Dictionary (October 16, 2025)",
                "url": _ACS_DICTIONARY_URL,
                "sha256": _ACS_DICTIONARY_SHA256,
                "pdf_page_1based": 2,
                "establishes": (
                    "housing-record ADJINC, one enumerated 2024 factor "
                    "1015250 (1.015250), six implied decimal places"
                ),
            }
        ),
        MappingProxyType(
            {
                "document": "2024 ACS PUMS User Guide, section F",
                "url": _ACS_USER_GUIDE_URL,
                "sha256": _ACS_USER_GUIDE_SHA256,
                "pdf_page_1based": 16,
                "establishes": (
                    "Divide ADJINC by 1,000,000 and multiply it to the PUMS "
                    "variable value to adjust it to 2024 dollars; the Housing "
                    "Unit file variables requiring ADJINC are FINCP and HINCP"
                ),
            }
        ),
        MappingProxyType(
            {
                "document": "2024 ACS Subject Definitions",
                "url": _ACS_SUBJECT_DEFINITIONS_URL,
                "sha256": _ACS_SUBJECT_DEFINITIONS_SHA256,
                "printed_page": 96,
                "establishes": (
                    "Income of Households covers the householder and all other "
                    "individuals 15 years old and over in the household; income "
                    "covers the past 12 months while household composition "
                    "refers to the time of interview"
                ),
            }
        ),
        MappingProxyType(
            {
                "document": "2024 ACS Subject Definitions",
                "url": _ACS_SUBJECT_DEFINITIONS_URL,
                "sha256": _ACS_SUBJECT_DEFINITIONS_SHA256,
                "printed_page": 98,
                "establishes": (
                    "Income components were reported for the 12 months "
                    "preceding the interview month and inflation-adjusted to a "
                    "reference calendar year with monthly CPI factors"
                ),
            }
        ),
    ),
)

ASEC_HTOTVAL_ORIGIN = ReportedIncomeOrigin(
    arm="asec_prepared",
    publisher_column="asec_HTOTVAL",
    concept="Total household income",
    universe_as_printed="All Households",
    observation_period="preceding_calendar_year",
    reference_period="the calendar year before the March survey",
    source_price_period="nominal_income_year",
    target_price_period="calendar_2024_dollars_census_ccpiu",
    transformation=(
        "parse C-CPI-U cells to float64, divide target by source, multiply "
        "each nonzero amount once; the 2024 identity year copies bytes"
    ),
    quoted_locator=(
        "HTOTVAL 8 108 (-999999:99999999) / total household income / "
        "Values: 0 = none, negative dollar amount, positive dollar amount / "
        "Universe: All Households"
    ),
    authority=(
        MappingProxyType(
            {
                "document": "CPS ASEC 2025 public-use data dictionary",
                "url": _ASEC_DICTIONARY_URL_TEMPLATE.format(survey_year=2025),
                "sha256": ASEC_HTOTVAL_VINTAGES[2024][1],
                "pdf_page_1based": ASEC_HTOTVAL_VINTAGES[2024][2],
                "establishes": "HTOTVAL concept, printed range and universe",
            }
        ),
        MappingProxyType(
            {
                "document": "CPS ASEC 2025 public-use data dictionary",
                "url": _ASEC_DICTIONARY_URL_TEMPLATE.format(survey_year=2025),
                "sha256": ASEC_HTOTVAL_VINTAGES[2024][1],
                "pdf_page_1based": 48,
                "establishes": (
                    "PTOTVAL total persons income, universe All Persons aged "
                    "15+; the household total's members are the 15+ universe, "
                    "and this module never sums PTOTVAL into HTOTVAL"
                ),
            }
        ),
        MappingProxyType(
            {
                "document": "CPS March 2025 technical documentation, glossary",
                "url": _ASEC_TECHDOC_URL,
                "sha256": _ASEC_TECHDOC_SHA256,
                "pdf_page_1based": 112,
                "establishes": (
                    "CPS income is asked of each person 15 years old and over "
                    "for money income received in the preceding calendar year "
                    "from eleven listed sources; the ACS asks eight"
                ),
            }
        ),
        MappingProxyType(
            {
                "document": "CPS March 2025 technical documentation, glossary",
                "url": _ASEC_TECHDOC_URL,
                "sha256": _ASEC_TECHDOC_SHA256,
                "pdf_page_1based": 112,
                "establishes": (
                    "CPS data cover money income received exclusive of certain "
                    "money receipts such as capital gains, so the household "
                    "total excludes them, and the count of households excludes "
                    "persons living in group quarters"
                ),
            }
        ),
        MappingProxyType(
            {
                "document": (
                    "Census P60-286 annual index value / annual percent "
                    "change workbook, restated by packaged microcosm "
                    "resource asec_current_money_price_basis_v1.json "
                    "(census_ccpiu_annual_2024)"
                ),
                "url": _ASEC_PRICE_BASIS_WORKBOOK_URL,
                "sha256": _ASEC_PRICE_BASIS_WORKBOOK_SHA256,
                "size_bytes": _ASEC_PRICE_BASIS_WORKBOOK_SIZE_BYTES,
                "resource": _ASEC_PRICE_BASIS_RESOURCE,
                "resource_sha256": _ASEC_PRICE_BASIS_SHA256,
                "establishes": "the authoritative ASEC current-money restatement",
            }
        ),
    ),
)

B19001_2024_BIN_DEFINITION_ID = "acs_2024_B19001_household_income_bins_v1"
_B19001_2024_BOUNDS = (
    ("B19001_002", "Less than $10,000", None, 10_000.0),
    ("B19001_003", "$10,000 to $14,999", 10_000.0, 15_000.0),
    ("B19001_004", "$15,000 to $19,999", 15_000.0, 20_000.0),
    ("B19001_005", "$20,000 to $24,999", 20_000.0, 25_000.0),
    ("B19001_006", "$25,000 to $29,999", 25_000.0, 30_000.0),
    ("B19001_007", "$30,000 to $34,999", 30_000.0, 35_000.0),
    ("B19001_008", "$35,000 to $39,999", 35_000.0, 40_000.0),
    ("B19001_009", "$40,000 to $44,999", 40_000.0, 45_000.0),
    ("B19001_010", "$45,000 to $49,999", 45_000.0, 50_000.0),
    ("B19001_011", "$50,000 to $59,999", 50_000.0, 60_000.0),
    ("B19001_012", "$60,000 to $74,999", 60_000.0, 75_000.0),
    ("B19001_013", "$75,000 to $99,999", 75_000.0, 100_000.0),
    ("B19001_014", "$100,000 to $124,999", 100_000.0, 125_000.0),
    ("B19001_015", "$125,000 to $149,999", 125_000.0, 150_000.0),
    ("B19001_016", "$150,000 to $199,999", 150_000.0, 200_000.0),
    ("B19001_017", "$200,000 or more", 200_000.0, None),
)
B19001_2024_BINS = tuple(
    DistributionBin(label, lower, upper)
    for _variable, label, lower, upper in _B19001_2024_BOUNDS
)
B19001_2024_METADATA: Mapping[str, object] = MappingProxyType(
    {
        "table": "B19001",
        "dataset": "acs/acs1",
        "year": 2024,
        "concept": (
            "Household Income in the Past 12 Months (in 2024 "
            "Inflation-Adjusted Dollars)"
        ),
        "groups_url": "https://api.census.gov/data/2024/acs/acs1/groups/B19001.json",
        "groups_sha256": (
            "d73a050438fb75f43c1b7738d0aa42c3c7ad17e4fb62b2661294652fbd05aafa"
        ),
        "denominator": "B19001_001",
        "bin_variables": tuple(item[0] for item in _B19001_2024_BOUNDS),
        "boundary_rule": (
            "lower inclusive, upper exclusive; null lower means minus infinity, "
            "null upper plus infinity"
        ),
        "reservation": "reserved_same_source_holdout",
        "targets_allowed": False,
        "tuning_allowed": False,
        "note": (
            "ACS tabulations and ACS PUMS share a source. These bins are "
            "holdout observations for diagnosis only; this module activates no "
            "national or congressional-district target and scores no candidate."
        ),
    }
)


# Identity, not value, equality: the measured arrays make a field-by-field
# comparison raise rather than answer, and two measurements are never compared.
@dataclass(frozen=True, eq=False)
class HouseholdReportedIncome:
    """One arm's measured household total plus its complete provenance."""

    origin: ReportedIncomeOrigin
    household_keys: tuple[str, ...]
    values: np.ndarray
    availability: np.ndarray
    quality: np.ndarray
    _raw_values: pd.Series
    _evidence_json: str

    @property
    def raw_values(self) -> pd.Series:
        """Return an independent copy of the literal publisher amounts."""

        return self._raw_values.copy()

    @property
    def evidence(self) -> dict[str, object]:
        """Return an independent JSON-safe copy of the measurement evidence."""

        return json.loads(self._evidence_json)

    @property
    def source_price_period(self) -> str:
        """The price period the publisher's raw amounts are stated in."""

        return self.origin.source_price_period

    @property
    def target_price_period(self) -> str:
        """The price period this measurement's values are stated in."""

        return self.origin.target_price_period

    @property
    def transformation(self) -> str:
        """The single declared transformation from raw amount to value."""

        return self.origin.transformation


@cache
def _definition_identity() -> str:
    """Return this contract's identity, computed once per process.

    Module bytes and installed dependency versions cannot change under a live
    interpreter, so recomputing this per measurement only costs time.
    """

    module = resources.files(__package__).joinpath("household_reported_income.py")
    body = {
        "contract": CONTRACT,
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "availability": dict(AVAILABILITY),
        "quality": dict(QUALITY),
        "origins": {
            "acs": _origin_body(ACS_2024_HINCP_ORIGIN),
            "asec": _origin_body(ASEC_HTOTVAL_ORIGIN),
        },
        "b19001": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in B19001_2024_METADATA.items()
        },
        "dependencies": {name: metadata.version(name) for name in ("numpy", "pandas")},
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _keys(household_keys: Sequence[str]) -> tuple[str, ...]:
    _require(not isinstance(household_keys, str), "HOUSEHOLD_KEYS_NOT_A_SEQUENCE")
    keys = tuple(household_keys)
    _require(len(keys) > 0, "EMPTY_SAMPLE")
    # Native identifiers keep their exact spelling, including leading zeros.
    # An integer key would silently drop them, so it is refused outright.
    _require(all(type(key) is str for key in keys), "HOUSEHOLD_KEY_NOT_STRING")
    _require(all(key and key == key.strip() for key in keys), "HOUSEHOLD_KEY_SPELLING")
    _require(len(set(keys)) == len(keys), "DUPLICATE_HOUSEHOLD_KEY")
    return keys


def _integer_column(values: object, rows: int, label: str) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    _require(len(series) == rows, f"{label}_LENGTH")
    if series.dtype == np.dtype("int64"):
        series = series.astype("Int64")
    # Integer dollars only. A float column is refused here rather than
    # rounded, which is what stops an already-adjusted amount being re-adjusted.
    _require(series.dtype == pd.Int64Dtype(), f"{label}_DTYPE")
    return series


def measure_acs_household_reported_income(
    *,
    household_keys: Sequence[str],
    hincp: object,
    adjinc: object,
    housing_unit_kind: object,
) -> HouseholdReportedIncome:
    """Measure the ACS publisher household total in 2024 dollars.

    ``hincp``, ``adjinc`` and ``housing_unit_kind`` are the literal native
    ``HINCP``, housing-record ``ADJINC`` and ``TYPEHUGQ`` columns as pandas
    nullable ``Int64``. The adjustment factor is applied exactly once, here.
    """

    keys = _keys(household_keys)
    rows = len(keys)
    raw = _integer_column(hincp, rows, "HINCP")
    factor = _integer_column(adjinc, rows, "ADJINC")
    kind = _integer_column(housing_unit_kind, rows, "TYPEHUGQ")

    _require(kind.notna().all(), "TYPEHUGQ_MISSING")
    supported = (ACS_HOUSING_UNIT_KIND, *ACS_GROUP_QUARTERS_KINDS)
    _require(bool(kind.isin(supported).all()), "TYPEHUGQ_DOMAIN")
    in_universe = (kind == ACS_HOUSING_UNIT_KIND).to_numpy(dtype=bool)
    observed = raw.notna().to_numpy(dtype=bool)
    # The dictionary declares HINCP not applicable for GQ and vacant records.
    # A value there is a source-contract violation, not a measurable total.
    _require(not bool((~in_universe & observed).any()), "OUT_OF_UNIVERSE_AMOUNT")
    # ADJINC is a single enumerated code in the pinned 2024 1-year dictionary.
    # Requiring it exactly refuses a caller that already divided or multiplied.
    _require(factor.notna().all(), "ADJINC_MISSING")
    _require(
        bool((factor == ACS_2024_DECLARED_ADJINC).all()),
        "ADJINC_NOT_THE_DECLARED_2024_FACTOR",
    )
    amounts = raw.to_numpy(dtype="float64", na_value=np.nan)
    present = amounts[observed]
    _require(
        bool(((present >= ACS_HINCP_MINIMUM) & (present <= ACS_HINCP_MAXIMUM)).all()),
        "HINCP_RANGE",
    )

    availability = np.where(
        observed,
        AVAILABILITY["observed"],
        np.where(
            in_universe,
            AVAILABILITY["unavailable_in_universe"],
            AVAILABILITY["unavailable_out_of_universe"],
        ),
    ).astype("uint8")
    quality = np.where(
        ~observed,
        QUALITY["not_applicable"],
        np.where(
            amounts == ACS_HINCP_MINIMUM,
            QUALITY["declared_open_ended_bound"],
            QUALITY["exact_publisher_total"],
        ),
    ).astype("uint8")
    ratio = float(ACS_2024_DECLARED_ADJINC) / float(ACS_ADJUSTMENT_DENOMINATOR)
    values = np.full(rows, np.nan, dtype="float64")
    values[observed] = amounts[observed] * ratio
    evidence = {
        "contract": CONTRACT,
        "arm": ACS_2024_HINCP_ORIGIN.arm,
        "definition_identity": _definition_identity(),
        "origin": _origin_body(ACS_2024_HINCP_ORIGIN),
        "rows": rows,
        "source_columns": ["HINCP", "ADJINC", "TYPEHUGQ"],
        "join_key": "SERIALNO",
        "adjustment": {
            "declared_adjinc": ACS_2024_DECLARED_ADJINC,
            "denominator": ACS_ADJUSTMENT_DENOMINATOR,
            "factor_expression": f"{ACS_2024_DECLARED_ADJINC}/{ACS_ADJUSTMENT_DENOMINATOR}",
            "factor_bits_le": struct.pack("<d", ratio).hex(),
            "applications": 1,
        },
        "availability_counts": _counts(availability, AVAILABILITY),
        "quality_counts": _counts(quality, QUALITY),
        "negative_value_rows": int((values[observed] < 0).sum()),
        "component_reconstruction": (
            "none; the executed ACS person roster holds six of the eight "
            "published income types "
            + ", ".join(ACS_CARRIED_REPORTED_INCOME)
            + ", leaving "
            + " and ".join(ACS_UNCARRIED_INCOME_TYPES)
            + " unread, so no component sum is claimed and no residual is "
            "labelled other income"
        ),
    }
    return HouseholdReportedIncome(
        origin=ACS_2024_HINCP_ORIGIN,
        household_keys=keys,
        values=_readonly(values),
        availability=_readonly(availability),
        quality=_readonly(quality),
        _raw_values=raw.copy(),
        _evidence_json=json.dumps(evidence, sort_keys=True),
    )


def measure_asec_household_reported_income(
    *,
    household_keys: Sequence[str],
    htotval: object,
    income_year: object,
) -> HouseholdReportedIncome:
    """Measure the prepared-ASEC publisher household total in 2024 dollars.

    ``htotval`` is the literal ``asec_HTOTVAL`` observation. The restatement is
    the packaged, byte-pinned C-CPI-U current-money basis; income year 2024 is
    the identity year and copies its normalized amount without arithmetic.
    """

    keys = _keys(household_keys)
    rows = len(keys)
    raw = _integer_column(htotval, rows, "HTOTVAL")
    years = _integer_column(income_year, rows, "INCOME_YEAR")
    _require(years.notna().all(), "INCOME_YEAR_MISSING")
    _require(bool(years.isin(ASEC_INCOME_YEARS).all()), "INCOME_YEAR_DOMAIN")

    observed = raw.notna().to_numpy(dtype=bool)
    amounts = raw.to_numpy(dtype="float64", na_value=np.nan)
    present = amounts[observed]
    _require(
        bool(
            (
                (present >= ASEC_HTOTVAL_MINIMUM) & (present <= ASEC_HTOTVAL_MAXIMUM)
            ).all()
        ),
        "HTOTVAL_RANGE",
    )

    basis = _asec_price_basis()
    target_year = int(basis["target_year"])
    cells = basis["cells"]
    values = np.full(rows, np.nan, dtype="float64")
    factors: dict[str, dict[str, str]] = {}
    year_values = years.to_numpy(dtype="int64")
    for year in sorted(set(int(value) for value in year_values)):
        selected = observed & (year_values == year)
        if year == target_year:
            # The identity year copies normalized bytes; no arithmetic runs.
            values[selected] = amounts[selected]
            factors[str(year)] = {"expression": "identity", "bits_le": ""}
            continue
        ratio = float(cells[str(target_year)]) / float(cells[str(year)])
        nonzero = selected & (amounts != 0)
        values[selected & (amounts == 0)] = 0.0
        values[nonzero] = amounts[nonzero] * ratio
        factors[str(year)] = {
            "expression": f"{cells[str(target_year)]}/{cells[str(year)]}",
            "bits_le": struct.pack("<d", ratio).hex(),
        }

    availability = np.where(
        observed,
        AVAILABILITY["observed"],
        # HTOTVAL's printed universe is All Households: the publisher declares
        # no out-of-universe state, so absence is an in-universe gap.
        AVAILABILITY["unavailable_in_universe"],
    ).astype("uint8")
    quality = np.where(
        observed, QUALITY["exact_publisher_total"], QUALITY["not_applicable"]
    ).astype("uint8")
    evidence = {
        "contract": CONTRACT,
        "arm": ASEC_HTOTVAL_ORIGIN.arm,
        "definition_identity": _definition_identity(),
        "origin": _origin_body(ASEC_HTOTVAL_ORIGIN),
        "rows": rows,
        "source_columns": ["asec_HTOTVAL"],
        "income_years": sorted({int(value) for value in year_values}),
        "price_basis": {
            "price_basis": basis["price_basis"],
            "algorithm": basis["algorithm"],
            "target_year": target_year,
            "cells": {key: cells[key] for key in sorted(cells)},
            "resource": _ASEC_PRICE_BASIS_RESOURCE,
            "resource_sha256": _ASEC_PRICE_BASIS_SHA256,
            "factors": factors,
            "applications": 1,
        },
        "availability_counts": _counts(availability, AVAILABILITY),
        "quality_counts": _counts(quality, QUALITY),
        "negative_value_rows": int((values[observed] < 0).sum()),
        "component_reconstruction": (
            "none; HTOTVAL is the publisher's own household total and is "
            "never rebuilt from admitted engine roots. CPS money income is "
            "published exclusive of certain receipts such as capital gains, so "
            "the separate ASEC CAP_VAL field is outside this total; it is "
            "neither the ACS all-other-income component OIP nor an unallocated "
            "residual"
        ),
    }
    return HouseholdReportedIncome(
        origin=ASEC_HTOTVAL_ORIGIN,
        household_keys=keys,
        values=_readonly(values),
        availability=_readonly(availability),
        quality=_readonly(quality),
        _raw_values=raw.copy(),
        _evidence_json=json.dumps(evidence, sort_keys=True),
    )


def align_household_values(
    measurement: HouseholdReportedIncome,
    household_keys: Sequence[str],
) -> np.ndarray:
    """Return measured values in ``household_keys`` order, by exact key.

    Keys are matched as native strings so a leading-zero identifier keeps its
    spelling. Positional alignment is never used and unknown keys refuse.
    """

    keys = _keys(household_keys)
    positions = {key: index for index, key in enumerate(measurement.household_keys)}
    missing = [key for key in keys if key not in positions]
    _require(not missing, "UNKNOWN_HOUSEHOLD_KEY")
    order = np.asarray([positions[key] for key in keys], dtype=np.int64)
    return _readonly(np.asarray(measurement.values)[order].copy())


def b19001_household_income_distribution_spec(
    variable: str,
    *,
    entity: str = "household",
    label: str = "Household income in the past 12 months (2024 dollars)",
) -> DistributionSpec:
    """Return the reserved B19001 holdout bin scheme as a distribution spec.

    The bins reproduce the pinned Census group metadata exactly. They are
    holdout observations: building this spec activates no target and scores no
    candidate.
    """

    _require(isinstance(variable, str) and variable.strip() == variable, "VARIABLE")
    return DistributionSpec(
        variable=variable,
        label=label,
        entity=entity,
        unit="2024_inflation_adjusted_USD",
        bin_definition_id=B19001_2024_BIN_DEFINITION_ID,
        bins=B19001_2024_BINS,
    )


def two_arm_comparability(
    acs: HouseholdReportedIncome,
    asec: HouseholdReportedIncome,
) -> dict[str, object]:
    """Return both arms' stated differences without combining any value.

    This is the whole two-arm surface: the consumer receives each arm's origin
    and the explicit list of source-dependent differences. No pooled total,
    ratio or reconciliation is produced here.
    """

    _require(acs.origin is ACS_2024_HINCP_ORIGIN, "FIRST_ARGUMENT_MUST_BE_ACS")
    _require(asec.origin is ASEC_HTOTVAL_ORIGIN, "SECOND_ARGUMENT_MUST_BE_ASEC")
    return {
        "contract": CONTRACT,
        "arms": {"acs": acs.evidence["origin"], "asec": asec.evidence["origin"]},
        "shared": {
            "concept": "publisher-reported household total money income",
            "target_price_year": 2024,
            "component_sum": "never; both arms use the publisher aggregate",
        },
        "asec_universe_binding": ASEC_MEASUREMENT_REQUIRES_INDEPENDENT_HU_MASK,
        "differences": [
            {
                "axis": "reference period",
                "acs": "rolling twelve months preceding each interview month",
                "asec": "the preceding calendar year",
                "consequence": (
                    "the two arms describe different windows of time; a "
                    "difference in level is not by itself an error in either"
                ),
            },
            {
                "axis": "price restatement",
                "acs": (
                    "Census ADJINC to 2024 dollars: one enumerated annual "
                    "factor on the 1-year PUMS file"
                ),
                "asec": "Census C-CPI-U annual restatement to 2024",
                "consequence": (
                    "both land on 2024 dollars through different published "
                    "indices, and neither is applied twice; the published "
                    "tabulations use month-of-interview CPI factors the PUMS "
                    "file does not expose, so a PUMS-adjusted amount "
                    "approximates rather than reproduces a tabulated one"
                ),
            },
            {
                "axis": "household universe",
                "acs": "occupied housing units; GQ and vacant records are N/A",
                "asec": (
                    "All Households as printed; the CPS household count "
                    "excludes persons living in group quarters"
                ),
                "consequence": (
                    "both source files carry group-quarters-flagged records: "
                    "the prepared ASEC household table retains the raw "
                    "group-quarters fields a separate classifier keys on, so "
                    "a published household count excluding GQ does not mean "
                    "the file has no GQ rows. This measurement derives an "
                    "out-of-universe availability code only from the ACS "
                    "TYPEHUGQ field, because only the ACS dictionary "
                    "declares an N/A pattern for HINCP; the ASEC arm's raw "
                    "asec_HTOTVAL total does not by itself establish "
                    "occupied-housing-unit membership, so a B19001-style "
                    "consumer must bind the independently carried ASEC "
                    "housing-universe mask before treating an ASEC household "
                    "as in scope"
                ),
            },
            {
                "axis": "reported income sources",
                "acs": "eight published income types",
                "asec": "eleven listed CPS money-income sources",
                "consequence": (
                    "the two totals are built from differently partitioned "
                    "question sets; both exclude capital gains, and neither "
                    "total is reconstructed from components here"
                ),
            },
            {
                "axis": "declared bounds",
                "acs": "-59999 is the declared open-ended loss floor",
                "asec": "no open-ended bound is printed for HTOTVAL",
                "consequence": (
                    "only the ACS arm ever reports declared_open_ended_bound"
                ),
            },
        ],
    }


def _asec_price_basis() -> dict[str, object]:
    payload = resources.files(__package__).joinpath(_ASEC_PRICE_BASIS_RESOURCE)
    raw = payload.read_bytes()
    _require(
        hashlib.sha256(raw).hexdigest() == _ASEC_PRICE_BASIS_SHA256,
        "ASEC_PRICE_BASIS_RESOURCE_CHANGED",
    )
    basis = json.loads(raw)
    _require(basis["price_basis"] == "census_ccpiu_annual_2024", "ASEC_PRICE_BASIS")
    _require(int(basis["target_year"]) == 2024, "ASEC_PRICE_TARGET_YEAR")
    _require(
        set(basis["cells"]) == {str(year) for year in ASEC_INCOME_YEARS},
        "ASEC_PRICE_CELLS",
    )
    return basis


def _counts(codes: np.ndarray, codebook: Mapping[str, int]) -> dict[str, int]:
    return {
        name: int((codes == value).sum()) for name, value in sorted(codebook.items())
    }


def _readonly(values: np.ndarray) -> np.ndarray:
    values.flags.writeable = False
    return values
