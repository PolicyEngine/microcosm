"""HMRC's company car statistics, typed for the ``company_cars`` stage.

HMRC's Benefits in kind statistics (July 2026 release) count the employees taxed
on a company car in 2024-25, provisionally, by the car's CO2 emissions band
(Table 2A) and by its fuel (Table 2C), with the taxable value, income tax and
NICs on the benefit (Table 1). They publish no list price and no electric range,
and Tables 2A and 2C are margins: no table crosses CO2 with fuel. The stage reads
the two margins to draw what the FRS does not record; Table 1 and the income
table are measured against, never bound (María's ruling of 28 October 2026 for
the ISA facts, applied here: binding the cells a stage draws to would be
circular).

The rows come from the Chronicle feed (PolicyEngine/uk-autumn-budget-dashboard-2026#316)
through ``tools/vendor_uk_ledger_facts.py``. Until the feed carries them the
resource is absent, and the loader says so.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

HMRC_COMPANY_CAR_STATISTICS_RESOURCE = "hmrc_benefits_in_kind_facts.json"
HMRC_COMPANY_CAR_STATISTICS_ARTIFACT_ROLE = "benefits_in_kind_facts"
#: The tax year the stage reads (2024-25, provisional in the July 2026 release),
#: the build's period.
HMRC_COMPANY_CAR_TAX_YEAR = 2024

#: Table 2A's CO2 bands from 2020-21, in g/km: (id, lowest, highest). The top
#: band is open; every figure in it is at or above the 2024-25 and 2030-31
#: maximum percentages' thresholds (160 and 155 g/km), so where in it a car sits
#: moves no benefit.
HMRC_CO2_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("zero", 0, 0),
    ("1_50", 1, 50),
    ("51_54", 51, 54),
    ("55_59", 55, 59),
    ("60_64", 60, 64),
    ("65_69", 65, 69),
    ("70_74", 70, 74),
    ("75_79", 75, 79),
    ("80_84", 80, 84),
    ("85_89", 85, 89),
    ("90_94", 90, 94),
    ("95_104", 95, 104),
    ("105_114", 105, 114),
    ("115_124", 115, 124),
    ("125_134", 125, 134),
    ("135_144", 135, 144),
    ("145_154", 145, 154),
    ("155_164", 155, 164),
    ("165_169", 165, 169),
    ("170_plus", 170, None),
)
HMRC_CO2_BAND_IDS = tuple(band for band, _, _ in HMRC_CO2_BANDS)
#: Table 2C's fuels: diesel and petrol each include their hybrids.
HMRC_FUEL_TYPE_IDS = ("electric", "petrol", "diesel")


class HMRCCompanyCarStatisticsError(ValueError):
    """Raised when the vendored facts are absent, incomplete or inconsistent."""


def co2_band_bounds(band: str) -> tuple[int, int | None]:
    for band_id, low, high in HMRC_CO2_BANDS:
        if band_id == band:
            return low, high
    raise HMRCCompanyCarStatisticsError(f"Unknown CO2 band {band!r}.")


def _require_counts(
    values: Mapping[str, float], expected: tuple[str, ...], label: str
) -> dict[str, float]:
    if tuple(values) != expected:
        raise HMRCCompanyCarStatisticsError(
            f"{label} must carry exactly {list(expected)} in that order, not "
            f"{list(values)}."
        )
    counts = {key: float(value) for key, value in values.items()}
    if any(not value >= 0.0 for value in counts.values()):
        raise HMRCCompanyCarStatisticsError(f"{label} has a negative or missing count.")
    if not sum(counts.values()) > 0.0:
        raise HMRCCompanyCarStatisticsError(f"{label} counts nobody.")
    return counts


@dataclass(frozen=True)
class HMRCCompanyCarStatistics:
    """The 2024-25 company car facts the stage draws from and reports against.

    Counts are recipients in thousands, as published; amounts are GBP million.
    A band HMRC prints as below 0.5 thousand is carried as 0.25 thousand and
    named in ``suppressed_co2_bands``.
    """

    tax_year: int
    co2_band_recipients: Mapping[str, float]
    fuel_type_recipients: Mapping[str, float]
    car_recipients: float
    car_taxable_value_gbp_m: float
    car_income_tax_gbp_m: float
    car_nics_gbp_m: float
    fuel_recipients: float
    fuel_taxable_value_gbp_m: float
    suppressed_co2_bands: tuple[str, ...] = ()
    source_commit: str | None = None
    resource_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "co2_band_recipients",
            _require_counts(self.co2_band_recipients, HMRC_CO2_BAND_IDS, "Table 2A"),
        )
        object.__setattr__(
            self,
            "fuel_type_recipients",
            _require_counts(self.fuel_type_recipients, HMRC_FUEL_TYPE_IDS, "Table 2C"),
        )
        if self.tax_year != HMRC_COMPANY_CAR_TAX_YEAR:
            raise HMRCCompanyCarStatisticsError(
                f"The stage reads {HMRC_COMPANY_CAR_TAX_YEAR}-"
                f"{(HMRC_COMPANY_CAR_TAX_YEAR + 1) % 100:02d}'s facts, not "
                f"{self.tax_year}'s."
            )
        unknown = sorted(set(self.suppressed_co2_bands) - set(HMRC_CO2_BAND_IDS))
        if unknown:
            raise HMRCCompanyCarStatisticsError(f"Unknown suppressed bands {unknown}.")

    def nonzero_co2_shares(self) -> dict[str, float]:
        """Each band's share of the cars that emit CO2 (every band but zero)."""

        bands = {k: v for k, v in self.co2_band_recipients.items() if k != "zero"}
        total = sum(bands.values())
        return {band: value / total for band, value in bands.items()}

    def fuel_type_shares(self) -> dict[str, float]:
        total = sum(self.fuel_type_recipients.values())
        return {
            fuel: value / total for fuel, value in self.fuel_type_recipients.items()
        }

    def evidence(self) -> dict[str, Any]:
        return {
            "resource": HMRC_COMPANY_CAR_STATISTICS_RESOURCE,
            "resource_sha256": self.resource_sha256,
            "source_commit": self.source_commit,
            "tax_year": self.tax_year,
            "suppressed_co2_bands": list(self.suppressed_co2_bands),
        }


def check_company_car_statistics_declaration(
    parameters: Mapping[str, Any], *, stage: str, period: int
) -> None:
    """The runtime arm of a stage's ``verify_vendored_fact_resource`` operation."""

    expected = {
        "artifact_role": HMRC_COMPANY_CAR_STATISTICS_ARTIFACT_ROLE,
        "resource": HMRC_COMPANY_CAR_STATISTICS_RESOURCE,
        "mapped_build_period": HMRC_COMPANY_CAR_TAX_YEAR,
    }
    declared = {key: parameters.get(key) for key in expected}
    if declared != expected:
        raise HMRCCompanyCarStatisticsError(
            f"{stage}'s verify_vendored_fact_resource must declare {expected}, "
            f"not {declared}."
        )
    if int(period) != HMRC_COMPANY_CAR_TAX_YEAR:
        raise HMRCCompanyCarStatisticsError(
            f"{stage} maps HMRC's company car facts to build period "
            f"{HMRC_COMPANY_CAR_TAX_YEAR}, but the frame's period is {period}."
        )


def _resource_bytes() -> bytes:
    resource = files("microcosm.build.uk").joinpath(
        HMRC_COMPANY_CAR_STATISTICS_RESOURCE
    )
    if not resource.is_file():
        raise HMRCCompanyCarStatisticsError(
            f"{HMRC_COMPANY_CAR_STATISTICS_RESOURCE} is not vendored yet: the pinned "
            "Chronicle feed must carry HMRC's Benefits in kind statistics (July 2026) "
            "before tools/vendor_uk_ledger_facts.py can write it. Pass the facts "
            "explicitly until then."
        )
    return resource.read_bytes()


def load_hmrc_company_car_statistics(
    *,
    payload: Mapping[str, Any] | None = None,
    verify_feed_identity: bool = True,
) -> HMRCCompanyCarStatistics:
    """Type the vendored company car facts.

    The record ids are the Chronicle package's; they are bound here when the
    package lands, and until then only the absence of the resource is reported.
    """

    if payload is None:
        digest = hashlib.sha256(_resource_bytes()).hexdigest()
        raise HMRCCompanyCarStatisticsError(
            f"{HMRC_COMPANY_CAR_STATISTICS_RESOURCE} ({digest[:12]}) has no reader "
            "for its record ids yet."
        )
    raise HMRCCompanyCarStatisticsError(
        "The vendored company car facts have no reader for their record ids yet."
    )


__all__ = [
    "HMRC_CO2_BANDS",
    "HMRC_CO2_BAND_IDS",
    "HMRC_COMPANY_CAR_STATISTICS_ARTIFACT_ROLE",
    "HMRC_COMPANY_CAR_STATISTICS_RESOURCE",
    "HMRC_COMPANY_CAR_TAX_YEAR",
    "HMRC_FUEL_TYPE_IDS",
    "HMRCCompanyCarStatistics",
    "HMRCCompanyCarStatisticsError",
    "check_company_car_statistics_declaration",
    "co2_band_bounds",
    "load_hmrc_company_car_statistics",
]
