"""Employer pension contributions from ASHE contribution-rate bands (microcosm#1069 R6).

The FRS records whether a person belongs to an employer's pension scheme and of
what kind (``PENPROV.STEMPPEN`` 1 to 4: group personal, occupational, group
stakeholder, other or not known) but not what the employer pays. The incumbent
estimate set employer contributions to three times the employee's, which gives
nothing to members of non-contributory schemes and the same multiple to a
defined-benefit public-sector scheme as to an auto-enrolment minimum.

This module draws each member's employer contribution rate instead, from the
ONS ASHE 2024 pension tables: the share of jobs in each employer contribution
band (as a percentage of full pay) by pension type and sector (Table P10, all
industries). An occupational scheme is defined benefit or defined contribution
with the probabilities ASHE's job counts give within the member's sector (Table
P2); the other employer types map one to one, and a scheme of unknown type uses
the all-types distribution. The rate is drawn uniformly within its band (the
open top band is read as 20% to 30%; the largest defined-benefit employer rates
sit just under 30%), and the contribution is that rate times the person's
employment income. Salary sacrifice stays its own column; ASHE and DWP count it
as an employer contribution, so the bound DWP employer total adds it back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms

ASHE_EMPLOYER_RATE_RESOURCE = "ashe_employer_pension_contribution_rates.json"
ASHE_EMPLOYER_RATE_PERIOD = "2024"
#: ASHE's employer contribution bands as fractions of full pay; the open top
#: band is closed at 30%.
ASHE_EMPLOYER_RATE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Under 4%", 0.0, 0.04),
    ("4% to < 8%", 0.04, 0.08),
    ("8% to < 10%", 0.08, 0.10),
    ("10% to < 12%", 0.10, 0.12),
    ("12% to < 15%", 0.12, 0.15),
    ("15% to < 20%", 0.15, 0.20),
    ("20% and over", 0.20, 0.30),
)
#: FRS PENPROV.STEMPPEN codes for employer schemes (FRS 2024-25 variable
#: listing) and the ASHE pension type each maps to; "occupational" splits into
#: defined benefit or defined contribution by sector.
FRS_EMPLOYER_SCHEME_TYPES: Mapping[int, str] = {
    1: "group_personal_pension",
    2: "occupational",
    3: "group_stakeholder_pension",
    4: "all",
}
#: A member of several employer schemes takes the first type in this order.
FRS_EMPLOYER_SCHEME_PRIORITY = (2, 1, 3, 4)
#: FRS JOB.JOBSECT codes (1 private, 2 public).
FRS_JOB_SECTORS: Mapping[int, str] = {1: "Private", 2: "Public"}
EMPLOYER_RATE_SEED = 0
EMPLOYER_RATE_SALT = "employer_pension_contribution_rate"
EMPLOYER_TYPE_SALT = "employer_pension_occupational_type"
EMPLOYER_RATE_METHOD = (
    "ASHE 2024 Table P10 employer contribution band shares by pension type and "
    "sector (full pay, all industries); occupational schemes split into defined "
    "benefit and defined contribution by ASHE Table P2 job counts in the sector; "
    "the rate is uniform within its band (top band 20% to 30%) and multiplies "
    "employment income"
)


@dataclass(frozen=True)
class AsheEmployerRates:
    """Band shares by (pension type, sector) and the occupational DB share."""

    shares: Mapping[tuple[str, str], np.ndarray]
    defined_benefit_share: Mapping[str, float]

    def distribution(self, pension_type: str, sector: str) -> np.ndarray:
        for key in ((pension_type, sector), (pension_type, "all")):
            if key in self.shares:
                return self.shares[key]
        raise ValueError(
            f"ASHE has no employer rate distribution for {pension_type!r}."
        )


def ashe_employer_rates_from_rows(
    rows: list[Mapping[str, object]],
) -> AsheEmployerRates:
    """Build the rate model from vendored ASHE rows (P10 shares, P2 job counts)."""

    bands = [band for band, _, _ in ASHE_EMPLOYER_RATE_BANDS]
    shares: dict[tuple[str, str], np.ndarray] = {}
    collected: dict[tuple[str, str], dict[str, float]] = {}
    jobs: dict[tuple[str, str], float] = {}
    for row in rows:
        dimensions = row.get("dimensions") or {}
        concept = row.get("concept")
        value = float(row["value"])
        if concept == "ons.workplace_pension_employer_contribution_band_share":
            if dimensions.get("contribution_basis") != "full_pay":
                continue
            if dimensions.get("sic2007_section", "all") != "all":
                continue
            key = (
                str(dimensions["pension_type"]),
                str(dimensions.get("sector", "all")),
            )
            collected.setdefault(key, {})[
                str(dimensions["employer_contribution_rate_band"])
            ] = value
        elif concept == "ons.workplace_pension_jobs_by_type":
            if dimensions.get("sic2007_section", "all") != "all":
                continue
            if dimensions.get("weekly_earnings_band", "all") != "all":
                continue
            jobs[
                (str(dimensions["pension_type"]), str(dimensions.get("sector", "all")))
            ] = value
    for key, values in collected.items():
        if set(values) != set(bands):
            # ASHE suppresses some sector cells; those fall back to the type's
            # all-sector distribution.
            continue
        vector = np.asarray([values[band] for band in bands], dtype=np.float64)
        if vector.sum() <= 0.0:
            continue
        shares[key] = vector / vector.sum()
    # ASHE publishes the all-industry bands by sector only: a type's all-sector
    # distribution is its sector distributions mixed by the sector's jobs of
    # that type (Table P2), the "all" type's by the sector's jobs of every type
    # with an employer scheme.
    scheme_types = {
        pension_type
        for (pension_type, _sector) in jobs
        if pension_type not in ("no_pension_provision", "unknown_pension_provision")
    }
    for pension_type in {key[0] for key in shares}:
        if (pension_type, "all") in shares:
            continue
        parts = []
        for sector in ("Public", "Private"):
            if (pension_type, sector) not in shares:
                continue
            if pension_type == "all":
                weight = sum(jobs.get((kind, sector), 0.0) for kind in scheme_types)
            else:
                weight = jobs.get((pension_type, sector), 0.0)
            if weight > 0.0:
                parts.append((weight, shares[(pension_type, sector)]))
        if parts:
            total = sum(weight for weight, _ in parts)
            mixed = sum(weight * vector for weight, vector in parts) / total
            shares[(pension_type, "all")] = mixed / mixed.sum()
    defined_benefit_share: dict[str, float] = {}
    for sector in ("Public", "Private"):
        db = jobs.get(("defined_benefit", sector))
        dc = jobs.get(("defined_contribution", sector))
        if db is not None and dc is not None and db + dc > 0.0:
            defined_benefit_share[sector] = db / (db + dc)
    db_all = sum(
        jobs.get(("defined_benefit", sector), 0.0) for sector in ("Public", "Private")
    )
    dc_all = sum(
        jobs.get(("defined_contribution", sector), 0.0)
        for sector in ("Public", "Private")
    )
    if db_all + dc_all > 0.0:
        defined_benefit_share["all"] = db_all / (db_all + dc_all)
    for required in ("defined_benefit", "defined_contribution", "all"):
        if (required, "all") not in shares:
            raise ValueError(
                f"ASHE employer rates lack the {required!r} all-sector row."
            )
    if "all" not in defined_benefit_share:
        raise ValueError("ASHE job counts lack the DB and DC rows.")
    return AsheEmployerRates(shares=shares, defined_benefit_share=defined_benefit_share)


def load_ashe_employer_rates() -> AsheEmployerRates:
    """Read the vendored ASHE rows (refused if they lag the feed pin)."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return ashe_employer_rates_from_rows(
        vendored_rows(
            ASHE_EMPLOYER_RATE_RESOURCE, period_value=ASHE_EMPLOYER_RATE_PERIOD
        )
    )


def employer_scheme_and_sector(
    person_ids: np.ndarray,
    penprov: pd.DataFrame,
    job: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Each person's employer scheme STEMPPEN code (0 if none) and job sector.

    ``penprov`` and ``job`` carry ``source_person_id`` (the raw sernum/person
    identity) with ``stemppen`` and ``jobsect`` respectively; a person's sector
    is that of their first job with a stated sector, "all" otherwise.
    """

    scheme = np.zeros(len(person_ids), dtype=np.int64)
    codes = pd.to_numeric(penprov["stemppen"], errors="coerce").fillna(0).astype(int)
    employer = penprov.loc[codes.isin(FRS_EMPLOYER_SCHEME_TYPES)].assign(code=codes)
    rank = {
        code: position for position, code in enumerate(FRS_EMPLOYER_SCHEME_PRIORITY)
    }
    if len(employer):
        employer = employer.assign(rank=employer["code"].map(rank))
        best = employer.sort_values("rank").drop_duplicates("source_person_id")
        lookup = best.set_index("source_person_id")["code"]
        scheme = pd.Series(person_ids).map(lookup).fillna(0).astype(np.int64).to_numpy()
    sectors = pd.to_numeric(job["jobsect"], errors="coerce")
    stated = job.loc[sectors.isin(FRS_JOB_SECTORS)].assign(
        sector=sectors.map(FRS_JOB_SECTORS)
    )
    sector = np.full(len(person_ids), "all", dtype=object)
    if len(stated):
        first = stated.drop_duplicates("source_person_id").set_index("source_person_id")
        sector = (
            pd.Series(person_ids)
            .map(first["sector"])
            .fillna("all")
            .to_numpy(dtype=object)
        )
    return scheme, sector


def draw_employer_pension_contributions(
    person_ids: np.ndarray,
    *,
    employment_income: np.ndarray,
    scheme: np.ndarray,
    sector: np.ndarray,
    rates: AsheEmployerRates,
) -> tuple[np.ndarray, dict[str, object]]:
    """Employer contributions for members of an employer scheme with earnings."""

    income = np.asarray(employment_income, dtype=np.float64)
    if not np.isfinite(income).all():
        raise ValueError("employment_income must be finite.")
    scheme = np.asarray(scheme, dtype=np.int64)
    members = (scheme > 0) & (income > 0.0)
    type_draws = stable_identity_uniforms(
        person_ids, seed=EMPLOYER_RATE_SEED, salt=EMPLOYER_TYPE_SALT
    )
    rate_draws = stable_identity_uniforms(
        person_ids, seed=EMPLOYER_RATE_SEED, salt=EMPLOYER_RATE_SALT
    )
    rate = np.zeros(len(person_ids), dtype=np.float64)
    ashe_type = np.full(len(person_ids), "", dtype=object)
    edges = np.asarray([[low, high] for _, low, high in ASHE_EMPLOYER_RATE_BANDS])
    counts: dict[str, int] = {}
    for index in np.flatnonzero(members):
        mapped = FRS_EMPLOYER_SCHEME_TYPES[int(scheme[index])]
        sector_label = str(sector[index])
        if mapped == "occupational":
            db_share = rates.defined_benefit_share.get(
                sector_label, rates.defined_benefit_share["all"]
            )
            mapped = (
                "defined_benefit"
                if type_draws[index] < db_share
                else "defined_contribution"
            )
        distribution = rates.distribution(mapped, sector_label)
        cumulative = np.cumsum(distribution)
        position = min(
            int(np.searchsorted(cumulative, rate_draws[index], side="right")),
            len(distribution) - 1,
        )
        lower = cumulative[position - 1] if position else 0.0
        within = (rate_draws[index] - lower) / max(distribution[position], 1e-12)
        low, high = edges[position]
        rate[index] = low + min(max(within, 0.0), 1.0) * (high - low)
        ashe_type[index] = mapped
        counts[mapped] = counts.get(mapped, 0) + 1
    amounts = rate * np.where(members, income, 0.0)
    receipt = {
        "method": EMPLOYER_RATE_METHOD,
        "members_with_earnings": int(members.sum()),
        "members_by_ashe_type": dict(sorted(counts.items())),
        "mean_rate_of_members": float(rate[members].mean()) if members.any() else None,
        "unweighted_amount": float(amounts.sum()),
    }
    return amounts, receipt


__all__ = [
    "ASHE_EMPLOYER_RATE_BANDS",
    "ASHE_EMPLOYER_RATE_RESOURCE",
    "AsheEmployerRates",
    "EMPLOYER_RATE_METHOD",
    "FRS_EMPLOYER_SCHEME_TYPES",
    "ashe_employer_rates_from_rows",
    "draw_employer_pension_contributions",
    "employer_scheme_and_sector",
    "load_ashe_employer_rates",
]
