"""Held-out Medicare IRMAA payers and premiums, never a calibration target.

The US release reports this diagnostic beside the SSI holdout. The 2026
Medicare Trustees Report's Part B and Part D income-related premium tables are
comparators for the model's payers and income-related premiums, not fitted
rows or a release gate (microcosm#1206).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "MEDICARE_IRMAA_HOLDOUT_VARIABLES",
    "MEDICARE_IRMAA_TRUSTEES_2026",
    "MedicareIrmaaBenchmark",
    "TRUSTEES_REPORT_2026_PDF_SHA256",
    "TRUSTEES_REPORT_2026_TITLE",
    "TRUSTEES_REPORT_2026_URL",
    "medicare_irmaa_holdout_from_sim",
    "medicare_irmaa_holdout_payload",
    "medicare_irmaa_holdout_unavailable",
    "write_medicare_irmaa_holdout",
]


@dataclass(frozen=True)
class MedicareIrmaaBenchmark:
    """One calendar-year row of a Trustees income-related premium table."""

    part: str
    label: str
    table: str
    printed_page: int
    pdf_page: int
    # "Beneficiaries affected (millions)" and "Aggregate premiums (billions)",
    # published to one decimal place and stored here in persons and dollars.
    beneficiaries_affected: int
    aggregate_premiums: int
    aggregate_premiums_definition: str

    @property
    def average_monthly_premium_per_payer(self) -> float:
        return self.aggregate_premiums / self.beneficiaries_affected / 12


TRUSTEES_REPORT_2026_TITLE = (
    "2026 Annual Report of the Boards of Trustees of the Federal Hospital "
    "Insurance and Federal Supplementary Medical Insurance Trust Funds"
)
TRUSTEES_REPORT_2026_URL = "https://www.cms.gov/oact/tr/2026"
#: SHA-256 of the report PDF the rows below were read from (271 pages).
TRUSTEES_REPORT_2026_PDF_SHA256 = (
    "ffa56b9137006872300b0346149eae1613d09a172b6ba118aad48e66dfc48fa8"
)
#: Rounding unit of the published counts (0.1 million) and dollars (0.1 billion).
_PUBLISHED_PAYER_UNIT = 100_000
_PUBLISHED_PREMIUM_UNIT = 100_000_000

#: Tables V.E3 and V.E4, "Historical data", calendar year 2024. Fixed held-out
#: observations, never passed to the calibration target registry.
MEDICARE_IRMAA_TRUSTEES_2026 = MappingProxyType(
    {
        2024: MappingProxyType(
            {
                "part_b": MedicareIrmaaBenchmark(
                    part="part_b",
                    label="Medicare Part B income-related premium",
                    table="V.E3",
                    printed_page=209,
                    pdf_page=215,
                    beneficiaries_affected=4_800_000,
                    aggregate_premiums=12_800_000_000,
                    aggregate_premiums_definition=(
                        "Represents the total amount paid by affected "
                        "beneficiaries in excess of the Part B standard premium."
                    ),
                ),
                "part_d": MedicareIrmaaBenchmark(
                    part="part_d",
                    label="Medicare Part D income-related premium",
                    table="V.E4",
                    printed_page=212,
                    pdf_page=218,
                    beneficiaries_affected=4_200_000,
                    aggregate_premiums=2_200_000_000,
                    aggregate_premiums_definition=(
                        "Represents the total amount paid by affected "
                        "beneficiaries in excess of the Part D plan premium."
                    ),
                ),
            }
        ),
    }
)

_ENROLLED = "medicare_enrolled"
_PART_B_GROSS = "gross_medicare_part_b_premium_if_enrolled"
_PART_B_BASE = "base_part_b_premium"
_PART_D_SURCHARGE = "income_adjusted_part_d_premium_surcharge"
#: The person-level variables the release scorer computes, in request order.
MEDICARE_IRMAA_HOLDOUT_VARIABLES: tuple[str, ...] = (
    _ENROLLED,
    _PART_B_GROSS,
    _PART_B_BASE,
    _PART_D_SURCHARGE,
)
_MODEL_MEASURES = MappingProxyType(
    {
        "part_b": (
            f"{_PART_B_GROSS} minus {_PART_B_BASE}, for persons with {_ENROLLED}"
        ),
        "part_d": f"{_PART_D_SURCHARGE}, for persons with {_ENROLLED}",
    }
)
#: Every income-related premium is withheld when no enrollee pays one.
_NO_PAYERS_REASON = (
    "No weighted Medicare enrollee pays an income-related premium, so the "
    "Trustees comparisons are withheld rather than reported as -100%. The "
    "premium for a year rests on modified adjusted gross income from two years "
    "earlier. policyengine-us 2.2.1 computes that income from the earlier "
    "year's inputs, which a release H5 carrying only the release year's inputs "
    "does not supply; PolicyEngine/policyengine-us#10106 adds a fallback to "
    "the earliest input year."
)
_NO_ENROLLEES_REASON = (
    "No weighted person is enrolled in Medicare, so the Trustees comparisons "
    "are withheld."
)
_GATES = (
    f"{_ENROLLED} covers Part A and/or Part B, so the Part B measure also "
    "counts any Part A-only enrollee the engine charges. The Part D measure is "
    "the surcharge as the engine gates it: every Medicare enrollee before "
    "PolicyEngine/policyengine-us#10115, Part D enrollees after it. A "
    "surcharge on a person outside Medicare enrollment is excluded and "
    "reported as outside_enrollment_weighted_amount."
)


def medicare_irmaa_holdout_unavailable(
    error: Exception, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """An observational receipt when plan discovery or scoring is unavailable."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "source_url": TRUSTEES_REPORT_2026_URL,
        "error": f"{type(error).__name__}: {error}",
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def _relative_difference(model: float, benchmark: float) -> float:
    return model / benchmark - 1


def _part_row(
    part: str,
    amounts: np.ndarray,
    weights: np.ndarray,
    benchmark: MedicareIrmaaBenchmark | None,
    *,
    compare: bool,
) -> dict[str, Any]:
    payers = amounts > 0
    weighted_payers = float(weights[payers].sum())
    weighted_premiums = float((weights * amounts).sum())
    row: dict[str, Any] = {
        "part": part,
        "model_measure": _MODEL_MEASURES[part],
        "weighted_payers": weighted_payers,
        "weighted_aggregate_premiums": weighted_premiums,
        "average_monthly_premium_per_payer": (
            weighted_premiums / weighted_payers / 12 if weighted_payers else None
        ),
        "benchmark_payers": None,
        "benchmark_aggregate_premiums": None,
        "benchmark_average_monthly_premium_per_payer": None,
        "payers_relative_difference": None,
        "aggregate_premiums_relative_difference": None,
    }
    if benchmark is None:
        return row
    row.update(
        {
            "label": benchmark.label,
            "benchmark_table": benchmark.table,
            "benchmark_printed_page": benchmark.printed_page,
            "benchmark_pdf_page": benchmark.pdf_page,
            "benchmark_aggregate_premiums_definition": (
                benchmark.aggregate_premiums_definition
            ),
            "benchmark_payers": benchmark.beneficiaries_affected,
            "benchmark_aggregate_premiums": benchmark.aggregate_premiums,
            "benchmark_average_monthly_premium_per_payer": (
                benchmark.average_monthly_premium_per_payer
            ),
        }
    )
    if compare:
        row["payers_relative_difference"] = _relative_difference(
            weighted_payers, benchmark.beneficiaries_affected
        )
        row["aggregate_premiums_relative_difference"] = _relative_difference(
            weighted_premiums, benchmark.aggregate_premiums
        )
    return row


def medicare_irmaa_holdout_payload(
    enrolled: np.ndarray,
    part_b_gross_if_enrolled: np.ndarray,
    part_b_base: np.ndarray,
    part_d_surcharge: np.ndarray,
    weights: np.ndarray,
    *,
    period: int,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Weighted IRMAA payers and annual income-related premiums by Part.

    Both Parts count ``medicare_enrolled`` people only. A person's Part B
    income-related premium is their enrollment-gated gross Part B premium
    less the standard premium; ``base_part_b_premium`` is defined for every
    Medicare-eligible person, so a non-enrollee's base premium never counts.
    The Part D amount is the engine's surcharge; any surcharge on a
    non-enrollee is excluded and reported. A payer is a person with a positive
    annual amount. Each person carries their household's calibrated weight,
    as supplied by the written-H5 scorer's person results; the function reads
    its inputs and never writes them.

    The status is ``available`` when some weighted enrollee pays. With no
    weighted enrollee (``no_medicare_enrollees``) or no weighted payer in
    either Part (``no_payers``), the model totals are still reported but the
    Trustees comparisons are withheld and a reason is given.
    """
    enrolled_values = np.asarray(enrolled)
    part_b_gross, part_b_base, part_d_surcharge, weights = (
        np.asarray(values, dtype=np.float64)
        for values in (part_b_gross_if_enrolled, part_b_base, part_d_surcharge, weights)
    )
    if enrolled_values.ndim != 1 or any(
        values.shape != enrolled_values.shape
        for values in (part_b_gross, part_b_base, part_d_surcharge, weights)
    ):
        raise ValueError(
            "Medicare IRMAA holdout needs aligned one-dimensional person arrays."
        )
    if enrolled_values.dtype != np.bool_:
        numeric = enrolled_values.astype(np.float64)
        if not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError(f"{_ENROLLED} must be boolean.")
        enrolled_values = numeric == 1.0
    if any(
        not np.isfinite(values).all()
        for values in (part_b_gross, part_b_base, part_d_surcharge, weights)
    ) or np.any(weights < 0):
        raise ValueError(
            "Medicare IRMAA holdout needs finite values and nonnegative weights."
        )
    if np.any(part_d_surcharge < 0):
        raise ValueError(f"{_PART_D_SURCHARGE} must be nonnegative.")
    part_b_amount = np.where(enrolled_values, part_b_gross - part_b_base, 0.0)
    if np.any(part_b_amount < 0):
        raise ValueError(
            f"{_PART_B_GROSS} is below {_PART_B_BASE} for an enrolled person."
        )
    part_d_amount = np.where(enrolled_values, part_d_surcharge, 0.0)
    part_d_outside = np.where(enrolled_values, 0.0, part_d_surcharge)

    weighted_enrollees = float(weights[enrolled_values].sum())
    any_payer = bool(weights[(part_b_amount > 0) | (part_d_amount > 0)].sum() > 0)
    if weighted_enrollees <= 0:
        status, reason = "no_medicare_enrollees", _NO_ENROLLEES_REASON
    elif not any_payer:
        status, reason = "no_payers", _NO_PAYERS_REASON
    else:
        status, reason = "available", None
    compare = status == "available"

    benchmarks = MEDICARE_IRMAA_TRUSTEES_2026.get(int(period))
    parts = [
        _part_row(
            "part_b",
            part_b_amount,
            weights,
            None if benchmarks is None else benchmarks["part_b"],
            compare=compare,
        ),
        _part_row(
            "part_d",
            part_d_amount,
            weights,
            None if benchmarks is None else benchmarks["part_d"],
            compare=compare,
        ),
    ]
    parts[1]["outside_enrollment_weighted_amount"] = float(
        (weights * part_d_outside).sum()
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "measure": "Medicare income-related premium (IRMAA) payers and amounts",
        "weight_basis": "person's calibrated household_weight",
        "payer": "a Medicare enrollee with a positive annual income-related premium",
        "weighted_medicare_enrollees": weighted_enrollees,
        "benchmark": {
            "source": TRUSTEES_REPORT_2026_TITLE,
            "source_url": TRUSTEES_REPORT_2026_URL,
            "source_pdf_sha256": TRUSTEES_REPORT_2026_PDF_SHA256,
            "tables": ["V.E3", "V.E4"],
            "columns": [
                "Beneficiaries affected (millions)",
                "Aggregate premiums (billions)",
            ],
            "year": None if benchmarks is None else int(period),
            "basis": None if benchmarks is None else "Historical data",
            "published_rounding": {
                "payers": _PUBLISHED_PAYER_UNIT,
                "aggregate_premiums": _PUBLISHED_PREMIUM_UNIT,
            },
            "available_years": sorted(MEDICARE_IRMAA_TRUSTEES_2026),
        },
        "comparability": (
            "The Trustees report the number of beneficiaries affected and the "
            "aggregate additional premium amounts collected in a calendar "
            "year; the tables do not state whether the count is an average "
            "month, a year-end or an ever-in-year count. The model counts "
            "Medicare enrollees with a positive annual income-related premium "
            f"in {int(period)} and sums those annual amounts. {_GATES} "
            "This is a held-out diagnostic, not a calibration target or a "
            "release gate."
        ),
        "parts": parts,
    }
    if reason is not None:
        payload["reason"] = reason
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def medicare_irmaa_holdout_from_sim(
    simulation: Any, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """Measure aligned person results from the household-batched H5 scorer."""
    results = {
        variable: simulation.calculate(variable, period)
        for variable in MEDICARE_IRMAA_HOLDOUT_VARIABLES
    }
    return medicare_irmaa_holdout_payload(
        np.asarray(results[_ENROLLED]),
        np.asarray(results[_PART_B_GROSS]),
        np.asarray(results[_PART_B_BASE]),
        np.asarray(results[_PART_D_SURCHARGE]),
        np.asarray(results[_ENROLLED].weights),
        period=period,
        release_id=release_id,
    )


def write_medicare_irmaa_holdout(payload: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path
