"""HMRC's company car statistics, typed for the ``company_cars`` stage.

HMRC's Benefits in kind statistics (July 2026 release) count the employees taxed
on a company car in 2024-25, provisionally, by the car's CO2 emissions band
(Table 2A) and by its fuel (Table 2C), with the taxable value, income tax and
Class 1A NICs on the car and fuel benefits (Table 1), and recipients and taxable
value by range of total income (Tables 3A and 3B) and by region (Table 4). They
publish no list price and no electric range, and Tables 2A and 2C are margins: no
table crosses CO2 with fuel. The stage draws fuel and CO2 to the two margins; the
rest is measured against, never bound (María's ruling of 28 October 2026 for the
ISA facts, applied here: binding the cells a stage draws to would be circular).

The rows come from the Chronicle package ``hmrc-benefits-in-kind-july-2026``
(PolicyEngine/chronicle#399, for PolicyEngine/uk-autumn-budget-dashboard-2026#316)
through ``tools/vendor_uk_ledger_facts.py``. Chronicle carries each value in
persons and pounds, and emits no fact for a cell HMRC marks as suppressed, so a
2024-25 cell the stage needs and the feed lacks is refused, never read as zero.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    load_vendored_resource,
    verify_vendored_resource_feed_identity,
)

HMRC_COMPANY_CAR_STATISTICS_RESOURCE = "hmrc_benefits_in_kind_facts.json"
HMRC_COMPANY_CAR_STATISTICS_ARTIFACT_ROLE = "benefits_in_kind_facts"
#: The tax year the stage reads (2024-25, provisional in the July 2026 release),
#: the build's period.
HMRC_COMPANY_CAR_TAX_YEAR = 2024
_PREFIX = "hmrc.benefits_in_kind_july_2026."
_YEAR = f"ty{HMRC_COMPANY_CAR_TAX_YEAR}"

#: Table 2A's CO2 bands from 2020-21, by Chronicle's row ids, in g/km: (id,
#: lowest, highest). The top band is open; every figure in it is at or above the
#: 2024-25 and 2030-31 maximum percentages' thresholds (160 and 155 g/km), so
#: where in it a car sits moves no benefit.
HMRC_CO2_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("co2_0", 0, 0),
    ("co2_1_to_50", 1, 50),
    ("co2_51_to_54", 51, 54),
    ("co2_55_to_59", 55, 59),
    ("co2_60_to_64", 60, 64),
    ("co2_65_to_69", 65, 69),
    ("co2_70_to_74", 70, 74),
    ("co2_75_to_79", 75, 79),
    ("co2_80_to_84", 80, 84),
    ("co2_85_to_89", 85, 89),
    ("co2_90_to_94", 90, 94),
    ("co2_95_to_104", 95, 104),
    ("co2_105_to_114", 105, 114),
    ("co2_115_to_124", 115, 124),
    ("co2_125_to_134", 125, 134),
    ("co2_135_to_144", 135, 144),
    ("co2_145_to_154", 145, 154),
    ("co2_155_to_164", 155, 164),
    ("co2_165_to_169", 165, 169),
    ("co2_170_plus", 170, None),
)
HMRC_CO2_BAND_IDS = tuple(band for band, _, _ in HMRC_CO2_BANDS)
HMRC_ZERO_EMISSION_BAND = "co2_0"
HMRC_LOW_EMISSION_BAND = "co2_1_to_50"
#: Table 2C's fuels: diesel and petrol each include their hybrids.
HMRC_FUEL_TYPE_IDS = (
    "diesel_including_hybrids",
    "petrol_including_hybrids",
    "fully_electric",
)
#: The 2024-25 record sets the stage's declaration names and the loader reads.
HMRC_COMPANY_CAR_RECORD_SETS = tuple(
    f"{_PREFIX}{name}.{_YEAR}"
    for name in (
        "table1.recipients",
        "table1.taxable_value",
        "table1.income_tax_liability",
        "table1.class_1a_nic_liability",
        "table2a.recipients",
        "table2c.recipients",
        "table3a.recipients",
        "table3a.taxable_value",
        "table3b.recipients",
        "table3b.taxable_value",
        "table4.recipients",
        "table4.taxable_value",
    )
)
#: How far each table's own total may sit from Table 1's 2024-25 car recipients,
#: given each sheet's rounding: Table 1 to the nearest 10,000, Table 2A's twenty
#: bands to whole thousands, Table 2C's three fuels to the nearest 5,000.
_TABLE_2A_TOLERANCE = 5_000 + 20 * 500
_TABLE_2C_TOLERANCE = 5_000 + 3 * 2_500
#: How far Table 2C's fully electric cars may sit from Table 2A's 0 g/km band.
_ELECTRIC_TOLERANCE = 2_500 + 500


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

    Counts are recipients and amounts are pounds, as Chronicle carries them.
    """

    tax_year: int
    co2_band_recipients: Mapping[str, float]
    fuel_type_recipients: Mapping[str, float]
    car_recipients: float
    car_taxable_value_gbp: float
    car_income_tax_gbp: float
    car_class_1a_nics_gbp: float
    fuel_recipients: float
    fuel_taxable_value_gbp: float
    fuel_income_tax_gbp: float
    fuel_class_1a_nics_gbp: float
    income_range_recipients: Mapping[str, float] = field(default_factory=dict)
    income_range_taxable_value_gbp: Mapping[str, float] = field(default_factory=dict)
    region_recipients: Mapping[str, float] = field(default_factory=dict)
    region_taxable_value_gbp: Mapping[str, float] = field(default_factory=dict)
    source_commit: str | None = None
    resource_sha256: str | None = None
    #: The rows the facts were typed from, which the stage's fact fence counts.
    row_count: int | None = None

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

    def nonzero_co2_shares(self) -> dict[str, float]:
        """Each band's share of the cars that emit CO2 (every band but 0 g/km)."""

        bands = {
            band: value
            for band, value in self.co2_band_recipients.items()
            if band != HMRC_ZERO_EMISSION_BAND
        }
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
            "row_count": self.row_count,
        }


def check_company_car_statistics_declaration(
    parameters: Mapping[str, Any], *, stage: str, period: int
) -> None:
    """The runtime arm of a stage's ``verify_vendored_fact_resource`` operation."""

    expected = {
        "artifact_role": HMRC_COMPANY_CAR_STATISTICS_ARTIFACT_ROLE,
        "resource": HMRC_COMPANY_CAR_STATISTICS_RESOURCE,
        "record_sets": list(HMRC_COMPANY_CAR_RECORD_SETS),
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


def _load_payload(verify_feed_identity: bool) -> tuple[dict[str, Any], str]:
    payload = load_vendored_resource(HMRC_COMPANY_CAR_STATISTICS_RESOURCE)
    if verify_feed_identity:
        verify_vendored_resource_feed_identity(payload)
    digest = hashlib.sha256(
        files("microcosm.build.uk")
        .joinpath(HMRC_COMPANY_CAR_STATISTICS_RESOURCE)
        .read_bytes()
    ).hexdigest()
    return payload, digest


def _row_values(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        record_id = str(row["source_record_id"])
        if record_id in values:
            raise HMRCCompanyCarStatisticsError(
                f"Duplicate vendored company car fact {record_id!r}."
            )
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HMRCCompanyCarStatisticsError(
                f"Vendored company car fact {record_id!r} is not numeric."
            )
        values[record_id] = float(value)
    return values


def _take(values: Mapping[str, float], record_id: str) -> float:
    try:
        return values[record_id]
    except KeyError:
        raise HMRCCompanyCarStatisticsError(
            f"The vendored company car facts lack {record_id!r}; re-vendor from the "
            "pinned feed."
        ) from None


def _rows_of(
    values: Mapping[str, float], record_set: str, measure: str
) -> dict[str, float]:
    """Every row a record set carries for one measure, by row id, in feed order."""

    prefix = f"{_PREFIX}{record_set}.{_YEAR}."
    suffix = f".{measure}"
    return {
        record_id[len(prefix) : -len(suffix)]: value
        for record_id, value in values.items()
        if record_id.startswith(prefix) and record_id.endswith(suffix)
    }


def load_hmrc_company_car_statistics(
    *,
    payload: Mapping[str, Any] | None = None,
    verify_feed_identity: bool = True,
) -> HMRCCompanyCarStatistics:
    """Type the vendored company car facts, refusing a gap or a drifted table.

    ``payload`` replaces the packaged resource (tests and the synthetic fixture);
    the packaged resource is checked against the committed feed pin first.
    """

    if payload is None:
        payload, digest = _load_payload(verify_feed_identity)
    else:
        if verify_feed_identity:
            verify_vendored_resource_feed_identity(payload)
        digest = "supplied"
    rows = list(payload.get("rows", ()))
    values = _row_values(rows)

    def table1(row: str, measure: str) -> float:
        return _take(values, f"{_PREFIX}table1.{measure}.{_YEAR}.{row}.{measure}")

    statistics = HMRCCompanyCarStatistics(
        tax_year=HMRC_COMPANY_CAR_TAX_YEAR,
        co2_band_recipients={
            band: _take(
                values, f"{_PREFIX}table2a.recipients.{_YEAR}.{band}.recipients"
            )
            for band in HMRC_CO2_BAND_IDS
        },
        fuel_type_recipients={
            fuel: _take(
                values, f"{_PREFIX}table2c.recipients.{_YEAR}.{fuel}.recipients"
            )
            for fuel in HMRC_FUEL_TYPE_IDS
        },
        car_recipients=table1("car", "recipients"),
        car_taxable_value_gbp=table1("car", "taxable_value"),
        car_income_tax_gbp=table1("car", "income_tax_liability"),
        car_class_1a_nics_gbp=table1("car", "class_1a_nic_liability"),
        fuel_recipients=table1("fuel", "recipients"),
        fuel_taxable_value_gbp=table1("fuel", "taxable_value"),
        fuel_income_tax_gbp=table1("fuel", "income_tax_liability"),
        fuel_class_1a_nics_gbp=table1("fuel", "class_1a_nic_liability"),
        income_range_recipients=_rows_of(values, "table3a.recipients", "recipients"),
        income_range_taxable_value_gbp=_rows_of(
            values, "table3a.taxable_value", "taxable_value"
        ),
        region_recipients=_rows_of(values, "table4.recipients", "recipients"),
        region_taxable_value_gbp=_rows_of(
            values, "table4.taxable_value", "taxable_value"
        ),
        source_commit=dict(payload.get("source_fact_feed", {})).get("source_commit"),
        resource_sha256=digest,
        row_count=len(rows),
    )
    _check_published_identities(statistics)
    return statistics


def _check_published_identities(statistics: HMRCCompanyCarStatistics) -> None:
    """Refuse a resource whose tables disagree beyond each sheet's rounding."""

    cars = statistics.car_recipients
    for label, total, tolerance in (
        ("Table 2A", sum(statistics.co2_band_recipients.values()), _TABLE_2A_TOLERANCE),
        (
            "Table 2C",
            sum(statistics.fuel_type_recipients.values()),
            _TABLE_2C_TOLERANCE,
        ),
    ):
        if abs(total - cars) > tolerance:
            raise HMRCCompanyCarStatisticsError(
                f"{label} counts {total:,.0f} company cars, Table 1 {cars:,.0f}: more "
                "apart than their rounding allows."
            )
    electric = statistics.fuel_type_recipients["fully_electric"]
    zero = statistics.co2_band_recipients[HMRC_ZERO_EMISSION_BAND]
    if abs(electric - zero) > _ELECTRIC_TOLERANCE:
        raise HMRCCompanyCarStatisticsError(
            f"Table 2C's fully electric cars ({electric:,.0f}) and Table 2A's 0 g/km "
            f"band ({zero:,.0f}) are more apart than their rounding allows."
        )


__all__ = [
    "HMRC_CO2_BANDS",
    "HMRC_CO2_BAND_IDS",
    "HMRC_COMPANY_CAR_RECORD_SETS",
    "HMRC_COMPANY_CAR_STATISTICS_ARTIFACT_ROLE",
    "HMRC_COMPANY_CAR_STATISTICS_RESOURCE",
    "HMRC_COMPANY_CAR_TAX_YEAR",
    "HMRC_FUEL_TYPE_IDS",
    "HMRC_LOW_EMISSION_BAND",
    "HMRC_ZERO_EMISSION_BAND",
    "HMRCCompanyCarStatistics",
    "HMRCCompanyCarStatisticsError",
    "check_company_car_statistics_declaration",
    "co2_band_bounds",
    "load_hmrc_company_car_statistics",
]
