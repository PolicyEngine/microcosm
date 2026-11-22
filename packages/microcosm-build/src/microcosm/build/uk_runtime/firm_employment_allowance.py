"""UK firm Employment Allowance flags: assumed rules from legal status and size.

policyengine-uk decides Employment Allowance eligibility
(``employment_allowance_eligible``) from four firm inputs. Two of them have
nothing to read in the experimental UK firm generator, and their defaults let
every employer claim:

- ``firm_claims_employment_allowance`` (default true): false for a firm that
  does not claim, including a connected company that is not its group's
  claimant;
- ``firm_is_single_director_company`` (default false): true for a company
  whose only employee paid Class 1 earnings is a director (NICA 2014
  s.2(4A)).

This module sets both from the ONS legal status code and the ONS employment
headcount, which counts working proprietors as well as employees. The rules
are assumptions, not observations:

- a sole proprietor (ONS status 2) with employment of at most 1 is the
  proprietor alone: no employee, so no claim;
- a partnership (status 3) with employment of at most 2 is taken to be its
  partners alone: no claim;
- a company (status 1) with employment 1 is taken to be a single-director
  company: its one person is a director. A company whose director employs a
  spouse below the secondary threshold is also excluded in law and is missed.

policyengine-uk documents ``firm_claims_employment_allowance`` as "this firm
is its connected group's claimant"; this module reuses it for "has no
employees to claim for". Nothing else changes: every other firm keeps the
policyengine-uk defaults (claims, not a single-director company). Connected
groups and the de minimis state aid test are not modelled. The 2024-25
£100,000 prior-year liability limit (s.2(4B)) is applied by policyengine-uk:
``firm_prior_year_secondary_class_1_liabilities`` is left unset, and its
formula falls back to the firm's current-year liabilities.

The flags change only the allowance. Proprietor-only firms still carry an
assumed paybill (``firm_paybill.py``) and so employer NIC; with the flags
they lose the allowance against it, which raises modelled employer NIC.

The rules read no random numbers, so the generator stream and every other
column are unchanged. HMRC's tax relief statistics (January 2026: 1,223,000
claimants and a cost of £3,300 million in 2024-25) are a held-out check in
``tests/fixtures/uk/firm_employment_allowance_2024_25/``, never a target.
``uk_firm_employment_allowance_comparison`` sets a policyengine-uk run beside
them; no build or tool calls it.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "UK_FIRM_EMPLOYMENT_ALLOWANCE_CHECK_FILE",
    "UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS",
    "UK_FIRM_EMPLOYMENT_ALLOWANCE_METHOD",
    "UK_FIRM_EMPLOYMENT_ALLOWANCE_STATUS",
    "UK_FIRM_PARTNERSHIP_PARTNERS_ONLY_MAX_EMPLOYMENT",
    "UKFirmEmploymentAllowanceCheck",
    "assign_uk_firm_employment_allowance",
    "read_uk_firm_employment_allowance_check",
    "uk_firm_employment_allowance_comparison",
    "uk_firm_employment_allowance_flags",
]

import numpy as np
import pandas as pd

#: Status of the two flags, in ``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS`` terms.
UK_FIRM_EMPLOYMENT_ALLOWANCE_STATUS = "assumed"

#: The flags this module sets, in export order.
UK_FIRM_EMPLOYMENT_ALLOWANCE_COLUMNS = (
    "firm_claims_employment_allowance",
    "firm_is_single_director_company",
)

#: How the flags are set, for build receipts.
UK_FIRM_EMPLOYMENT_ALLOWANCE_METHOD = (
    "Assumed from ONS legal status and ONS employment (which counts working "
    "proprietors): sole proprietors with employment of at most 1 and "
    "partnerships with employment of at most 2 do not claim; companies with employment 1 are "
    "single-director companies. Every other firm keeps the policyengine-uk "
    "defaults."
)

#: ONS legal status codes the rules read.
_COMPANY, _SOLE_PROPRIETOR, _PARTNERSHIP = 1, 2, 3
#: Largest employment at which a partnership is taken to be its partners alone.
UK_FIRM_PARTNERSHIP_PARTNERS_ONLY_MAX_EMPLOYMENT = 2

#: The held-out HMRC figures' file name in the fixture directory.
UK_FIRM_EMPLOYMENT_ALLOWANCE_CHECK_FILE = "hmrc_employment_allowance.csv"
_CHECK_COLUMNS = ("tax_year", "claimants", "cost_gbp")


def uk_firm_employment_allowance_flags(
    legal_status: np.ndarray | pd.Series, employment: np.ndarray | pd.Series
) -> dict[str, np.ndarray]:
    """The two flags for ONS ``legal_status`` codes and ``employment`` counts."""

    status = np.asarray(legal_status, dtype=np.int64)
    headcount = np.asarray(employment, dtype=np.int64)
    if status.shape != headcount.shape:
        raise ValueError("legal_status and employment must have the same shape.")
    if np.any(headcount < 0):
        raise ValueError("UK firm employment must be non-negative.")
    proprietor_alone = (status == _SOLE_PROPRIETOR) & (headcount <= 1)
    partners_alone = (status == _PARTNERSHIP) & (
        headcount <= UK_FIRM_PARTNERSHIP_PARTNERS_ONLY_MAX_EMPLOYMENT
    )
    return {
        "firm_claims_employment_allowance": ~(proprietor_alone | partners_alone),
        "firm_is_single_director_company": (status == _COMPANY) & (headcount == 1),
    }


def assign_uk_firm_employment_allowance(firms: pd.DataFrame) -> pd.DataFrame:
    """``firms`` with the two flags added from ``ons_legal_status`` and ``employment``."""

    missing = [c for c in ("ons_legal_status", "employment") if c not in firms]
    if missing:
        raise ValueError(
            f"UK firm Employment Allowance flags need {missing}; assign the ONS "
            "attributes first."
        )
    return firms.assign(
        **uk_firm_employment_allowance_flags(
            firms["ons_legal_status"].to_numpy(), firms["employment"].to_numpy()
        )
    )


@dataclass(frozen=True)
class UKFirmEmploymentAllowanceCheck:
    """HMRC's Employment Allowance claimants and cost for one tax year."""

    tax_year: str
    claimants: float
    cost_gbp: float
    sha256: str


def read_uk_firm_employment_allowance_check(
    directory: str | Path,
) -> UKFirmEmploymentAllowanceCheck:
    """Read the held-out HMRC figures for 2024-25 from ``directory``."""

    path = Path(directory) / UK_FIRM_EMPLOYMENT_ALLOWANCE_CHECK_FILE
    data = path.read_bytes()
    rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
    if not rows or tuple(rows[0]) != _CHECK_COLUMNS:
        raise ValueError(f"{path} must have columns {list(_CHECK_COLUMNS)}.")
    by_year = {row["tax_year"]: row for row in rows}
    if "2024-25" not in by_year:
        raise ValueError(f"{path} has no 2024-25 row.")
    row = by_year["2024-25"]
    return UKFirmEmploymentAllowanceCheck(
        tax_year="2024-25",
        claimants=float(row["claimants"]),
        cost_gbp=float(row["cost_gbp"]),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def uk_firm_employment_allowance_comparison(
    allowance: np.ndarray, weight: np.ndarray, check: UKFirmEmploymentAllowanceCheck
) -> Mapping[str, float]:
    """Weighted model claimants and cost beside HMRC's, with ratios.

    ``allowance`` is policyengine-uk's ``employment_allowance`` per firm. A
    diagnostic only: the HMRC figures are held out.
    """

    allowance = np.asarray(allowance, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    claimants = float(weight[allowance > 0].sum())
    cost = float((allowance * weight).sum())
    return {
        "model_claimants": claimants,
        "hmrc_claimants": check.claimants,
        "claimants_ratio": claimants / check.claimants,
        "model_cost_gbp": cost,
        "hmrc_cost_gbp": check.cost_gbp,
        "cost_ratio": cost / check.cost_gbp,
        "model_cost_per_claimant_gbp": cost / claimants if claimants else 0.0,
        "hmrc_cost_per_claimant_gbp": check.cost_gbp / check.claimants,
    }
