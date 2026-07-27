"""Published SOI component shares for the PUF E19200 interest deduction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np

_SOURCE_ASSET = "soi_table_2_1_interest_components_ty2015.json"
_AMOUNT_COLUMNS = {
    "total_interest_paid_amount": "CF",
    "home_mortgage_interest_amount": "CH",
    "deductible_points_amount": "CN",
    "qualified_mortgage_insurance_premiums_amount": "CP",
    "investment_interest_amount": "CR",
}


@dataclass(frozen=True)
class PufE19200InterestComponents:
    """One published SOI Table 2.1 interest-paid component row.

    Amounts retain the source table's unit, thousands of US dollars. The
    ratios used to decompose donor records are therefore unitless.
    """

    source_row: int
    total_interest_paid_amount: int
    home_mortgage_interest_amount: int
    deductible_points_amount: int
    qualified_mortgage_insurance_premiums_amount: int
    investment_interest_amount: int

    @property
    def non_mortgage_interest_amount(self) -> int:
        """Return the conserving residual represented by the engine leaf."""

        return self.total_interest_paid_amount - self.home_mortgage_interest_amount

    @property
    def source_cells(self) -> dict[str, str]:
        """Return the literal official-workbook cell for every carried amount."""

        return {
            name: f"{column}{self.source_row}"
            for name, column in _AMOUNT_COLUMNS.items()
        }


@dataclass(frozen=True)
class PufE19200AgiBand(PufE19200InterestComponents):
    """One half-open adjusted-gross-income band and its component amounts."""

    label: str
    lower_bound: float | None
    upper_bound: float | None

    @property
    def home_mortgage_share(self) -> float:
        """Return the published mortgage share of total E19200."""

        return self.home_mortgage_interest_amount / self.total_interest_paid_amount


def _components(raw: dict[str, Any]) -> PufE19200InterestComponents:
    return PufE19200InterestComponents(
        source_row=int(raw["source_row"]),
        total_interest_paid_amount=int(raw["total_interest_paid_amount"]),
        home_mortgage_interest_amount=int(raw["home_mortgage_interest_amount"]),
        deductible_points_amount=int(raw["deductible_points_amount"]),
        qualified_mortgage_insurance_premiums_amount=int(
            raw["qualified_mortgage_insurance_premiums_amount"]
        ),
        investment_interest_amount=int(raw["investment_interest_amount"]),
    )


def _band(raw: dict[str, Any]) -> PufE19200AgiBand:
    components = _components(raw)
    return PufE19200AgiBand(
        source_row=components.source_row,
        total_interest_paid_amount=components.total_interest_paid_amount,
        home_mortgage_interest_amount=components.home_mortgage_interest_amount,
        deductible_points_amount=components.deductible_points_amount,
        qualified_mortgage_insurance_premiums_amount=(
            components.qualified_mortgage_insurance_premiums_amount
        ),
        investment_interest_amount=components.investment_interest_amount,
        label=str(raw["label"]),
        lower_bound=(
            None if raw["lower_bound"] is None else float(raw["lower_bound"])
        ),
        upper_bound=(
            None if raw["upper_bound"] is None else float(raw["upper_bound"])
        ),
    )


def _load_source_asset() -> tuple[
    PufE19200InterestComponents,
    tuple[PufE19200AgiBand, ...],
]:
    payload = json.loads(
        files("populace.build.us").joinpath(_SOURCE_ASSET).read_text()
    )
    source = payload.get("source", {})
    if (
        source.get("tax_year") != 2015
        or source.get("table") != "2.1"
        or source.get("units") != "thousands_of_us_dollars"
    ):
        raise ValueError(f"{_SOURCE_ASSET} has unexpected source metadata.")
    source_columns = {
        name: details.get("workbook_column")
        for name, details in source.get("columns", {}).items()
    }
    if source_columns != _AMOUNT_COLUMNS:
        raise ValueError(f"{_SOURCE_ASSET} has unexpected workbook columns.")

    all_returns = _components(payload["all_returns"])
    bands = tuple(_band(raw) for raw in payload["agi_bands"])
    if len(bands) != 22 or [band.source_row for band in bands] != list(range(11, 33)):
        raise ValueError(f"{_SOURCE_ASSET} must carry Table 2.1 rows 11 through 32.")
    if bands[0].lower_bound is not None or bands[-1].upper_bound is not None:
        raise ValueError(f"{_SOURCE_ASSET} must cover the full real AGI line.")
    for previous, following in zip(bands[:-1], bands[1:], strict=True):
        if previous.upper_bound != following.lower_bound:
            raise ValueError(f"{_SOURCE_ASSET} AGI bands are not contiguous.")
    for row in (all_returns, *bands):
        amounts = (
            row.total_interest_paid_amount,
            row.home_mortgage_interest_amount,
            row.deductible_points_amount,
            row.qualified_mortgage_insurance_premiums_amount,
            row.investment_interest_amount,
        )
        if any(amount < 0 for amount in amounts) or row.total_interest_paid_amount == 0:
            raise ValueError(f"{_SOURCE_ASSET} carries an invalid amount.")
        component_sum = sum(amounts[1:])
        if abs(row.total_interest_paid_amount - component_sum) > 1:
            raise ValueError(
                f"{_SOURCE_ASSET} row {row.source_row} violates the published "
                "component identity beyond $1,000 source rounding."
            )
    return all_returns, bands


(
    US_PUF_E19200_ALL_RETURNS_COMPONENTS,
    US_PUF_E19200_AGI_BANDS,
) = _load_source_asset()

_AGI_UPPER_BOUNDS = np.asarray(
    [
        band.upper_bound
        for band in US_PUF_E19200_AGI_BANDS
        if band.upper_bound is not None
    ],
    dtype=np.float64,
)
_HOME_MORTGAGE_SHARES = np.asarray(
    [band.home_mortgage_share for band in US_PUF_E19200_AGI_BANDS],
    dtype=np.float64,
)


def split_us_puf_e19200_by_agi_band(
    total_interest_paid: Any,
    adjusted_gross_income: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Split each E19200 record by its published TY2015 SOI AGI-band share.

    PolicyEngine has one non-mortgage interest input,
    ``investment_interest_expense``. The residual routed there is broader
    than the table's investment-interest column: it also carries deductible
    points and qualified mortgage-insurance premiums. Computing it as
    ``total - mortgage`` preserves E19200 exactly despite published component
    rounding.
    """

    total = np.asarray(total_interest_paid, dtype=np.float64)
    agi = np.asarray(adjusted_gross_income, dtype=np.float64)
    if total.ndim != 1 or agi.ndim != 1:
        raise ValueError("E19200 and adjusted_gross_income must be one-dimensional.")
    if len(total) != len(agi):
        raise ValueError(
            "E19200 and adjusted_gross_income must have the same record count."
        )
    if not np.isfinite(total).all():
        raise ValueError("E19200 must contain only finite values.")
    if not np.isfinite(agi).all():
        raise ValueError("adjusted_gross_income must contain only finite values.")
    if (total < 0).any():
        raise ValueError("E19200 must be nonnegative.")

    band_index = np.searchsorted(_AGI_UPPER_BOUNDS, agi, side="right")
    mortgage = total * _HOME_MORTGAGE_SHARES[band_index]
    non_mortgage = total - mortgage
    return mortgage, non_mortgage


__all__ = [
    "PufE19200AgiBand",
    "PufE19200InterestComponents",
    "US_PUF_E19200_AGI_BANDS",
    "US_PUF_E19200_ALL_RETURNS_COMPONENTS",
    "split_us_puf_e19200_by_agi_band",
]
