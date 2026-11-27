"""UK paybill anchors: national accounts wages and salaries and PAYE RTI pay (report-only).

The opt-in paybill (``firm_paybill.py``) is headcount times ASHE section mean
pay, and its binding guard is the ratio to ASHE annual jobs times mean pay
(0.85-1.25). That guard shares ASHE's means with the paybill itself, so only
the jobs count is independent. The anchors here are two published paybill
totals that do not depend on ASHE:

- ONS DTWL, UK wages and salaries (ESA D.11), calendar 2024, whole economy;
- ONS/HMRC PAYE RTI aggregate pay, tax year 2024-25 (sum of twelve months,
  not seasonally adjusted), with the mean payrolled employee count.

:func:`uk_firm_paybill_anchor_ratios` divides a build's weighted paybill and
employees by them. The ratios are reported, never used as a target or a
guard. Neither anchor is the firm table's concept: both include employees of
every employer (the firm table's public sector is partial), D.11 includes
wages in kind, and the RTI count is of payrolled employees (persons) in each month, averaged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

UK_FIRM_PAYBILL_ANCHOR_FILE = "paybill_anchors.csv"
UK_FIRM_PAYBILL_ANCHORS = (
    "ons_d11_wages_and_salaries",
    "paye_rti_aggregate_pay",
    "paye_rti_payrolled_employees",
)


@dataclass(frozen=True)
class UKFirmPaybillAnchors:
    """Anchor values by name (pounds, or persons for the employee count)."""

    values: Mapping[str, float]
    sha256: str


def read_uk_firm_paybill_anchors(path: str | Path) -> UKFirmPaybillAnchors:
    """Read ``paybill_anchors.csv``; refuse a file whose sha256 differs from provenance."""

    directory = Path(path)
    payload = (directory / UK_FIRM_PAYBILL_ANCHOR_FILE).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["files"][UK_FIRM_PAYBILL_ANCHOR_FILE]["sha256"] != digest:
        raise ValueError(
            f"{UK_FIRM_PAYBILL_ANCHOR_FILE} sha256 differs from provenance.json."
        )
    frame = pd.read_csv(directory / UK_FIRM_PAYBILL_ANCHOR_FILE)
    if tuple(frame["anchor"]) != UK_FIRM_PAYBILL_ANCHORS:
        raise ValueError(f"paybill anchors must be {list(UK_FIRM_PAYBILL_ANCHORS)}.")
    values = dict(zip(frame["anchor"], frame["value"].astype(np.float64), strict=True))
    if not all(v > 0 for v in values.values()):
        raise ValueError("paybill anchors must be positive.")
    return UKFirmPaybillAnchors(values=MappingProxyType(values), sha256=digest)


def uk_firm_paybill_anchor_ratios(
    paybill: np.ndarray,
    employees: np.ndarray,
    weight: np.ndarray,
    anchors: UKFirmPaybillAnchors,
) -> dict[str, float | str]:
    """Weighted paybill and employees over each anchor (report-only)."""

    weight = np.asarray(weight, np.float64)
    total_pay = float((np.asarray(paybill, np.float64) * weight).sum())
    total_employees = float((np.asarray(employees, np.float64) * weight).sum())
    v = anchors.values
    return {
        "weighted_paybill_gbp": total_pay,
        "weighted_employees": total_employees,
        "paybill_over_ons_d11": total_pay / v["ons_d11_wages_and_salaries"],
        "paybill_over_paye_rti": total_pay / v["paye_rti_aggregate_pay"],
        "employees_over_paye_rti": total_employees / v["paye_rti_payrolled_employees"],
        "status": "report_only",
    }


__all__ = [
    "UK_FIRM_PAYBILL_ANCHORS",
    "UK_FIRM_PAYBILL_ANCHOR_FILE",
    "UKFirmPaybillAnchors",
    "read_uk_firm_paybill_anchors",
    "uk_firm_paybill_anchor_ratios",
]
