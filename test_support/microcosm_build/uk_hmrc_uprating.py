"""The HMRC SPI uprating appliers (PolicyEngine/chronicle#280 lane)."""

# ruff: noqa: F401

from __future__ import annotations

import pytest

from microcosm.build.ledger_targets import (
    CALENDAR_YEAR_WINDOW_WEIGHTS,
    LedgerTargetReference,
    TargetRegistry,
    TargetSpec,
)
from microcosm.build.uk_runtime.hmrc_uprating import (
    SPI_BAND_TO_ITL_BANDS,
    UK_ENGINE_INDEX_CONCEPTS,
    UK_ENGINE_INDEX_PARAMETERS,
    UK_ENGINE_PARAMETER_INDEX_PREFIX,
    UK_HMRC_ITL_UK_GEOGRAPHY_ID,
    UK_HMRC_STATE_PENSION_AGE_GROWTH_INDEX_CONCEPT,
    UK_HMRC_TAXPAYER_GROWTH_INDEX_CONCEPT,
    UK_SPI_SAVINGS_INTEREST_HISTORY_PARAMETER,
    UK_SPI_SAVINGS_INTEREST_INDEX_CONCEPT,
    UK_SPI_SAVINGS_INTEREST_PROJECTION_PARAMETER,
    UK_SPI_STATE_PENSION_AGE_AMOUNT_INDEX_CONCEPT,
    UK_SPI_STATE_PENSION_RATE_PARAMETER,
    align_hmrc_amount_by_recipient_growth_and_rate,
    align_hmrc_count_row_by_taxpayer_growth,
    align_hmrc_row_by_engine_index,
    align_hmrc_row_by_history_then_projection,
    align_hmrc_row_by_resident_state_pension_age_growth,
    engine_parameter_value,
    hmrc_uprating_appliers,
)
from microcosm.build.uk_runtime.ledger_targets import UK_UPRATING_APPLIERS
from microcosm.calibrate.geography_constants import UK_REGION_TIER

EARNINGS = "gov.economic_assumptions.indices.obr.average_earnings"
EARNINGS_INDEX = f"{UK_ENGINE_PARAMETER_INDEX_PREFIX}{EARNINGS}"
#: An amount index no contract declares, composing Table 2.5's band taxpayer
#: growth (the amount applier's default count index) with a rate, so the
#: generic amount applier is tested apart from the State Pension age index.
TABLE_2_5_AMOUNT_INDEX = "test.taxpayer_count_growth_times_rate_by_total_income_band"


def _spec(
    *,
    name: str,
    value: float,
    lower_bound: int | None,
    period: str = "2023",
    value_id: str | None = None,
):
    metadata = {
        "ledger_fact_period": period,
        "ledger_period_type": "tax_year",
        "contract_target_id": "hmrc.spi.employment_income.amount_by_total_income_band",
    }
    if lower_bound is not None:
        metadata["ledger_filter_total_income_lower_bound"] = str(lower_bound)
    if value_id is not None:
        metadata["ledger_layout_groupby_value_id"] = value_id
    return TargetSpec(
        name=name,
        entity="person",
        measure="hmrc/employment_income_income_band",
        value=value,
        period=2025,
        family="hmrc_spi",
        source="HMRC SPI 2023-24 Table 3.6 (test fixture)",
        metadata=metadata,
    )


def _reference(index: str, **overrides) -> LedgerTargetReference:
    values = {
        "name": "hmrc.spi.employment_income.amount_by_total_income_band",
        "ledger_selector": {"source_name": "hmrc"},
        "entity": "person",
        "measure": "hmrc/employment_income_income_band",
        "family": "hmrc_spi",
        "period": 2025,
        "uprating_index": index,
    }
    values.update(overrides)
    return LedgerTargetReference(**values)


def _fake_parameter(path: str, instant: str) -> float:
    assert path == EARNINGS
    return {"2023-01-01": 1.5, "2025-01-01": 1.65}[instant]


def _count_row(*, year: int, lower: int, upper: int | None, value: float):
    return {
        "concept": "hmrc.spi_taxpayer_count",
        "measure_id": "total_taxpayer_count",
        "period": {"type": "tax_year", "value": year},
        "dimensions": {
            "total_income_lower_bound": lower,
            "total_income_upper_bound": upper,
        },
        "value": value,
        "source_record_id": f"hmrc.itl_2026.table_2_5.ty{year}.band_{lower}_{upper}.total_taxpayer_count",
    }


def _count_rows():
    rows = []
    # 30,000-50,000: 2023 10.0m, 2024 11.0m, 2025 12.0m -> window 11.75m -> factor 1.175
    for year, value in ((2023, 10.0e6), (2024, 11.0e6), (2025, 12.0e6)):
        rows.append(_count_row(year=year, lower=30_000, upper=50_000, value=value))
    # 1m-2m and 2m+: the SPI 1m+ band sums both
    for year, one, two in (
        (2023, 20_000, 6_000),
        (2024, 22_000, 6_500),
        (2025, 24_000, 7_000),
    ):
        rows.append(_count_row(year=year, lower=1_000_000, upper=2_000_000, value=one))
        rows.append(_count_row(year=year, lower=2_000_000, upper=None, value=two))
    return rows


def _age_row(
    *, year: int, geography_id: str, value: float, measure_id: str = "state_pension_age"
):
    return {
        "concept": "hmrc.spi_taxpayer_count",
        "measure_id": measure_id,
        "period": {"type": "tax_year", "value": year},
        "geography": {"id": geography_id},
        "dimensions": {"age_band": measure_id.removeprefix("age_")},
        "value": value,
        "source_record_id": f"hmrc.itl_2026.{geography_id}.ty{year}.{measure_id}",
    }


#: Fixture State Pension age taxpayers: the United Kingdom row of Table 2.1
#: and twelve region-tier areas of Table 2.2 at 0.6m each in 2023-24 (7.2m),
#: rising 8% to 2024-25 and 3% more to 2025-26.
STATE_PENSION_AGE_UK = {2023: 8.0e6, 2024: 8.6e6, 2025: 8.9e6}
STATE_PENSION_AGE_AREA = {2023: 0.6e6, 2024: 0.648e6, 2025: 0.66744e6}


def _state_pension_age_rows(uk=None, area=None):
    uk = STATE_PENSION_AGE_UK if uk is None else uk
    area = STATE_PENSION_AGE_AREA if area is None else area
    rows = [
        _age_row(year=year, geography_id=UK_HMRC_ITL_UK_GEOGRAPHY_ID, value=value)
        for year, value in uk.items()
    ]
    for _, code in UK_REGION_TIER:
        rows.extend(
            _age_row(year=year, geography_id=code, value=value)
            for year, value in area.items()
        )
    # Rows the applier must ignore: another measure, England's own row and a
    # Table 2.5 band count of the same concept.
    rows.append(
        _age_row(
            year=2025,
            geography_id="E92000001",
            value=1.0,
            measure_id="state_pension_age",
        )
    )
    rows.append(
        _age_row(
            year=2025,
            geography_id=UK_HMRC_ITL_UK_GEOGRAPHY_ID,
            value=1.0,
            measure_id="age_65_and_over",
        )
    )
    rows.extend(_count_rows())
    return rows


__all__ = [name for name in globals() if not name.startswith("__")]
