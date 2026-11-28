"""UK firm R&D: an opt-in, assumed allocation of HMRC R&D tax credit claims.

policyengine-uk's R&D relief reads, per firm and year:

- ``firm_qualifying_rd_expenditure`` (£, no default): qualifying R&D
  expenditure already deducted in computing trading profit. Only a
  ``COMPANY`` (``firm_legal_form``) gets relief. Claimants are drawn from
  ``ons_legal_status`` 1; public corporations (status 4), which also export
  as ``COMPANY``, are never drawn, except that rows with only
  ``firm_legal_form`` count every ``COMPANY``;
- ``firm_is_sme`` (bool, default false): SME for R&D relief, which the
  variable documents as fewer than 500 staff and either turnover of no more
  than 100 million euros or a balance sheet of no more than 86 million euros,
  with linked and partner enterprises;
- ``firm_total_relevant_expenditure`` (£): the R&D intensity denominator.
  From April 2024 (``merged_scheme_in_effect``) a company claims SME relief
  (enhanced R&D intensive support) only if it is an SME, R&D intensive and
  loss-making; every other company with expenditure gets the merged R&D
  expenditure credit (20%).

The experimental UK firm generator sets none of these. This module sets the
first two from HMRC's *Research and Development Tax Credits Statistics:
September 2026* main tables (financial year 2024-25, provisional and
uplifted), extracted by ``tools/extract_uk_firm_rd_tax_credits_tables.py`` to
``tests/fixtures/uk/firm_rd_tax_credits_2024_25/``. Its status is
``assumed``. The steps:

1. Strata. Per SIC 2007 section, RD6's large-company claims are its RDEC by
   large companies plus merged RDEC by large companies (a suppressed or N/A
   cell counts as 0, so the large count is a lower bound); SME claims are the
   section's total claims less that.
2. Claims are not companies: RD1 gives 40,325 claims but 37,545 companies
   in 2024-25. Both strata targets are scaled by companies over claims, so a
   drawn company stands for one claiming company.
3. Claimants. Among companies of the section with turnover, the large
   stratum is taken top-down by headcount (ties in the order of a uniform
   from the module's own ``SeedSequence`` stream, assigned in ``firm_id``
   order) until summed ``firm_weight`` is nearest its target; the SME stratum
   is then drawn uniformly at random from the rest, in the same uniform's
   order, until its weight is nearest its target. A stratum whose pool runs
   out is capped and reported.
4. Expenditure. A section by size table is seeded with RD6 claims times the
   national expenditure per claim by size (RD4 over RD1: large is RDEC plus
   merged RDEC by large companies, SME the rest) and raked to RD6 section
   expenditure and to RD4's large and SME totals (£30,245m and £20,725m),
   the latter scaled to the sections' sum, since RD6's "Unknown or other"
   row (95 claims, £135m) has no section and is left unallocated. Each
   cell is shared over its drawn claimants in proportion to turnover, with
   expenditure never above turnover; what the cap leaves, or a cell with no
   claimant, is reported unallocated. RD6's cost columns (relief paid) are
   not used.
5. ``firm_is_sme``: a large-stratum claimant is False and an SME-stratum
   claimant True; every other row has the headcount limb only (fewer than 500
   employees). The euro turnover and balance sheet limbs and linked
   enterprises are not applied: the generator has no balance sheet.

``firm_total_relevant_expenditure`` stays unset, so no company is R&D
intensive and, under policyengine-uk's merged scheme, every claimant gets the
20% expenditure credit. Modelled relief is therefore not calibrated to HMRC's
cost of support.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

UK_FIRM_RD_SECTOR_FILE = "hmrc_rd_rd6_sector_2024_25.csv"
UK_FIRM_RD_CLAIMS_FILE = "hmrc_rd_rd1_claims.csv"
UK_FIRM_RD_EXPENDITURE_FILE = "hmrc_rd_rd4_expenditure.csv"
UK_FIRM_RD_FILES = (
    UK_FIRM_RD_SECTOR_FILE,
    UK_FIRM_RD_CLAIMS_FILE,
    UK_FIRM_RD_EXPENDITURE_FILE,
)
UK_FIRM_RD_YEAR = "2024-25"
UK_FIRM_RD_SECTIONS = tuple("ABCDEFGHIJKLMNOPQRS")
UK_FIRM_RD_UNKNOWN_SECTOR = "Unknown or other"
UK_FIRM_RD_STATUS = "assumed"
UK_FIRM_RD_METHOD = (
    "per SIC section, HMRC R&D Tax Credits Statistics September 2026 RD6 "
    "2024-25 claims split into large (RDEC and merged RDEC by large companies) "
    "and SME strata and scaled by RD1 companies over claims; large claimants "
    "taken top-down by headcount and SME claimants drawn uniformly among the "
    "rest; expenditure raked to RD6 section and RD4 size totals and shared by "
    "turnover, capped at turnover; firm_is_sme from the stratum for claimants "
    "and from headcount under 500 otherwise"
)
#: policyengine-uk's SME headcount limit for R&D relief (``firm_is_sme``).
UK_FIRM_RD_SME_EMPLOYEE_LIMIT = 500
#: Columns the assignment reads (one of each tuple).
UK_FIRM_RD_INPUT_COLUMNS = (
    ("firm_id",),
    ("firm_weight", "weights"),
    ("ons_employment", "firm_employees", "employment"),
    ("firm_turnover", "annual_turnover_k"),
    ("firm_sic_section", "sic_code"),
    ("ons_legal_status", "firm_legal_form"),
)
#: Second word of the draw's ``SeedSequence``: ASCII "RDTC".
UK_FIRM_RD_SEED_TAG = 0x52445443
UK_FIRM_RD_RAKE_ITERATIONS = 200

_COMPANY_STATUS = 1
_COMPANY_FORM = "COMPANY"
_SME_CLAIM_COLUMNS = (
    "sme_scheme_claims",
    "sme_intensive_eris_claims",
    "rdec_sme_claims",
    "merged_rdec_sme_claims",
)
_LARGE_CLAIM_COLUMNS = ("rdec_large_claims", "merged_rdec_large_claims")
#: RD1 / RD4 2024-25 columns of large-company claims and expenditure.
_RD1_LARGE = ("rdec_large", "merged_rdec_large")
_RD1_SME = ("sme_total", "rdec_sme", "merged_rdec_sme")
_RD4_LARGE = ("rdec_large", "merged_rdec_large")
_RD4_SME = ("sme_scheme", "sme_intensive_eris", "rdec_sme", "merged_rdec_sme")


def _published(value: str) -> float | None:
    """A published cell as a number; None when suppressed, N/A or blank."""

    text = value.strip()
    if not text or text == "N/A" or text.startswith("<"):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        raise ValueError(f"HMRC R&D cell {value!r} is not a number.") from None


@dataclass(frozen=True)
class UKFirmRDSection:
    """RD6 2024-25 for one SIC section: claims and expenditure (£)."""

    claims: float
    expenditure_gbp: float
    #: RDEC plus merged RDEC claims by large companies, suppressed cells as 0.
    large_claims: float
    large_cells_suppressed: int
    #: Unsuppressed SME claim cells summed, and how many were suppressed.
    sme_claims_published: float
    sme_cells_suppressed: int

    @property
    def sme_claims(self) -> float:
        return self.claims - self.large_claims


@dataclass(frozen=True)
class UKFirmRDSource:
    sections: Mapping[str, UKFirmRDSection]
    unknown_claims: float
    unknown_expenditure_gbp: float
    national_claims: float
    national_companies: float
    national_large_claims: float
    national_sme_claims: float
    #: RD4 2024-25 expenditure (£) by scheme column.
    national_expenditure_gbp: Mapping[str, float]
    sha256: Mapping[str, str]

    @property
    def national_large_expenditure_gbp(self) -> float:
        return float(sum(self.national_expenditure_gbp[c] for c in _RD4_LARGE))

    @property
    def national_sme_expenditure_gbp(self) -> float:
        return float(sum(self.national_expenditure_gbp[c] for c in _RD4_SME))

    @property
    def companies_per_claim(self) -> float:
        return self.national_companies / self.national_claims


def _read(directory: Path, name: str) -> tuple[pd.DataFrame, str]:
    path = directory / name
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame, hashlib.sha256(payload).hexdigest()


def _year_row(frame: pd.DataFrame, name: str) -> pd.Series:
    rows = frame[frame["financial_year"] == UK_FIRM_RD_YEAR]
    if len(rows) != 1:
        raise ValueError(f"{name} has {len(rows)} rows for {UK_FIRM_RD_YEAR}.")
    return rows.iloc[0]


def _required(value: str, label: str) -> float:
    number = _published(value)
    if number is None or not np.isfinite(number) or number < 0:
        raise ValueError(f"HMRC R&D {label} must be a published number, not {value!r}.")
    return number


def read_uk_firm_rd_source(path: str | Path) -> UKFirmRDSource:
    """Read the three extracted HMRC R&D CSVs from a fixture directory."""

    directory = Path(path)
    sector, sector_sha = _read(directory, UK_FIRM_RD_SECTOR_FILE)
    claims, claims_sha = _read(directory, UK_FIRM_RD_CLAIMS_FILE)
    spend, spend_sha = _read(directory, UK_FIRM_RD_EXPENDITURE_FILE)
    labels = sector["sector"].tolist()
    if [label[:1] for label in labels[:-1]] != list(UK_FIRM_RD_SECTIONS) or (
        labels[-1] != UK_FIRM_RD_UNKNOWN_SECTOR
    ):
        raise ValueError(f"HMRC R&D RD6 sector rows {labels} are not A-S + unknown.")
    sections: dict[str, UKFirmRDSection] = {}
    for record in sector.iloc[:-1].to_dict("records"):
        letter = record["sector"][:1]
        sme = [_published(record[c]) for c in _SME_CLAIM_COLUMNS]
        large = [_published(record[c]) for c in _LARGE_CLAIM_COLUMNS]
        total = _required(record["total_claims"], f"{letter} total claims")
        large_claims = float(sum(v for v in large if v is not None))
        if large_claims > total:
            raise ValueError(f"HMRC R&D {letter} large claims exceed total claims.")
        sections[letter] = UKFirmRDSection(
            claims=total,
            expenditure_gbp=1e6
            * _required(record["total_expenditure_gbp_m"], f"{letter} expenditure"),
            large_claims=large_claims,
            large_cells_suppressed=sum(v is None for v in large),
            sme_claims_published=float(sum(v for v in sme if v is not None)),
            sme_cells_suppressed=sum(v is None for v in sme),
        )
    unknown = sector.iloc[-1]
    year_claims = _year_row(claims, UK_FIRM_RD_CLAIMS_FILE)
    year_spend = _year_row(spend, UK_FIRM_RD_EXPENDITURE_FILE)
    national_expenditure = {
        column: 1e6 * _required(year_spend[column], f"RD4 {column}")
        for column in spend.columns
        if column not in ("financial_year", "status")
    }
    return UKFirmRDSource(
        sections=MappingProxyType(sections),
        unknown_claims=_required(unknown["total_claims"], "unknown claims"),
        unknown_expenditure_gbp=1e6
        * _required(unknown["total_expenditure_gbp_m"], "unknown expenditure"),
        national_claims=_required(year_claims["total_claims"], "RD1 total claims"),
        national_companies=_required(
            year_claims["total_companies"], "RD1 total companies"
        ),
        national_large_claims=float(
            sum(_required(year_claims[c], f"RD1 {c}") for c in _RD1_LARGE)
        ),
        national_sme_claims=float(
            sum(_required(year_claims[c], f"RD1 {c}") for c in _RD1_SME)
        ),
        national_expenditure_gbp=MappingProxyType(national_expenditure),
        sha256=MappingProxyType(
            {
                UK_FIRM_RD_SECTOR_FILE: sector_sha,
                UK_FIRM_RD_CLAIMS_FILE: claims_sha,
                UK_FIRM_RD_EXPENDITURE_FILE: spend_sha,
            }
        ),
    )


def uk_firm_rd_expenditure_table(source: UKFirmRDSource) -> pd.DataFrame:
    """Section by size expenditure (£), raked to RD6 sections and RD4 sizes.

    Columns ``large`` and ``sme``, one row per section A-S. The seed is RD6
    claims times RD4 over RD1 expenditure per claim by size; RD4's size totals
    are scaled to the sections' expenditure sum before raking.
    """

    letters = list(source.sections)
    per_claim = np.array(
        [
            source.national_large_expenditure_gbp / source.national_large_claims,
            source.national_sme_expenditure_gbp / source.national_sme_claims,
        ]
    )
    table = (
        np.array(
            [
                [source.sections[s].large_claims, source.sections[s].sme_claims]
                for s in letters
            ]
        )
        * per_claim
    )
    rows = np.array([source.sections[s].expenditure_gbp for s in letters])
    columns = np.array(
        [source.national_large_expenditure_gbp, source.national_sme_expenditure_gbp]
    )
    columns = columns * rows.sum() / columns.sum()
    for _ in range(UK_FIRM_RD_RAKE_ITERATIONS):
        table *= (rows / np.where(table.sum(1) > 0, table.sum(1), 1.0))[:, None]
        table *= columns / table.sum(0)
    return pd.DataFrame(table, index=letters, columns=["large", "sme"])


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"UK firm R&D needs one of {list(names)}.")


def _inputs(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    ids = frame[_first_column(frame, ("firm_id",))].to_numpy()
    if not pd.Index(ids).is_unique:
        raise ValueError("UK firm R&D needs one row per firm_id.")
    weight = frame[_first_column(frame, ("firm_weight", "weights"))].to_numpy(
        np.float64
    )
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("UK firm weights must be finite and non-negative.")
    employees = frame[
        _first_column(frame, ("ons_employment", "firm_employees", "employment"))
    ].to_numpy(np.float64)
    if not np.all(np.isfinite(employees)) or np.any(employees < 0):
        raise ValueError("UK firm employees must be finite and non-negative.")
    turnover_column = _first_column(frame, ("firm_turnover", "annual_turnover_k"))
    turnover = frame[turnover_column].to_numpy(np.float64)
    if turnover_column == "annual_turnover_k":
        turnover = turnover * 1000.0
    if not np.all(np.isfinite(turnover)) or np.any(turnover < 0):
        raise ValueError("UK firm turnover must be finite and non-negative.")
    if "firm_sic_section" in frame.columns:
        section = frame["firm_sic_section"].astype(str).to_numpy(object)
    elif "sic_code" in frame.columns:
        from microcosm.build.uk_runtime.firm_table import (  # noqa: PLC0415
            uk_firm_sic_sections,
        )

        section = uk_firm_sic_sections(frame["sic_code"]).astype(object)
    else:
        raise ValueError("UK firm R&D needs firm_sic_section or sic_code.")
    if "ons_legal_status" in frame.columns:
        company = pd.to_numeric(frame["ons_legal_status"]).to_numpy() == (
            _COMPANY_STATUS
        )
    elif "firm_legal_form" in frame.columns:
        company = frame["firm_legal_form"].astype(str).to_numpy(object) == (
            _COMPANY_FORM
        )
    else:
        raise ValueError("UK firm R&D needs ons_legal_status or firm_legal_form.")
    return {
        "ids": ids,
        "weight": weight,
        "employees": employees,
        "turnover": turnover,
        "section": section,
        "company": company,
    }


def _ordered_uniforms(ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), UK_FIRM_RD_SEED_TAG])
    )
    uniforms = np.empty(len(ids), dtype=np.float64)
    uniforms[np.argsort(ids, kind="stable")] = rng.random(len(ids))
    return uniforms


def _take_nearest(order: np.ndarray, weight: np.ndarray, target: float) -> np.ndarray:
    """The leading rows of ``order`` whose summed weight is nearest ``target``."""

    cumulative = np.concatenate([[0.0], np.cumsum(weight[order])])
    return order[: int(np.argmin(np.abs(cumulative - target)))]


def _share(
    rows: np.ndarray, weight: np.ndarray, turnover: np.ndarray, target: float
) -> tuple[np.ndarray, float]:
    """Expenditure of ``rows`` by turnover, at most turnover, and the remainder.

    Proportional shares with a cap at turnover reduce to one factor
    ``min(target / weighted turnover, 1)``: no row is capped unless all are.
    """

    capacity = float(np.dot(weight[rows], turnover[rows]))
    if capacity <= 0:
        return np.zeros(rows.size), target
    factor = min(target / capacity, 1.0)
    return turnover[rows] * factor, max(target - capacity, 0.0)


@dataclass(frozen=True)
class UKFirmRDAssignment:
    qualifying_rd_expenditure: np.ndarray
    is_sme: np.ndarray
    claimant: np.ndarray
    diagnostics: Mapping[str, Any]


def draw_uk_firm_rd_tax_credits(
    firms: pd.DataFrame, source: UKFirmRDSource, *, seed: int
) -> UKFirmRDAssignment:
    """Draw claimants, their size stratum and expenditure (module docstring)."""

    data = _inputs(firms)
    weight, employees, turnover = data["weight"], data["employees"], data["turnover"]
    section, company = data["section"], data["company"]
    uniforms = _ordered_uniforms(data["ids"], seed)
    is_sme = employees < UK_FIRM_RD_SME_EMPLOYEE_LIMIT
    claimant = np.zeros(len(firms), dtype=bool)
    expenditure = np.zeros(len(firms), dtype=np.float64)
    table = uk_firm_rd_expenditure_table(source)
    ratio = source.companies_per_claim
    rows: list[dict[str, Any]] = []
    unallocated = 0.0
    capped_rows = 0
    for letter, spec in source.sections.items():
        pool = np.flatnonzero(
            company & (section == letter) & (weight > 0) & (turnover > 0)
        )
        large_target = spec.large_claims * ratio
        sme_target = spec.sme_claims * ratio
        by_size = pool[np.lexsort((uniforms[pool], -employees[pool]))]
        large = _take_nearest(by_size, weight, large_target)
        rest = np.setdiff1d(pool, large, assume_unique=True)
        rest = rest[np.argsort(uniforms[rest], kind="stable")]
        sme = _take_nearest(rest, weight, sme_target)
        claimant[large] = True
        claimant[sme] = True
        is_sme[large] = False
        is_sme[sme] = True
        row: dict[str, Any] = {
            "section": letter,
            "hmrc_claims": spec.claims,
            "large_cells_suppressed": spec.large_cells_suppressed,
            "pool_weight": float(weight[pool].sum()),
            "pool_rows": int(pool.size),
        }
        for stratum, chosen, target, available in (
            ("large", large, large_target, float(weight[pool].sum())),
            ("sme", sme, sme_target, float(weight[rest].sum())),
        ):
            amounts, left = _share(chosen, weight, turnover, table.at[letter, stratum])
            expenditure[chosen] = amounts
            unallocated += left
            at_cap = int(np.sum(amounts >= turnover[chosen])) if left > 0 else 0
            capped_rows += at_cap
            drawn = float(weight[chosen].sum())
            row.update(
                {
                    f"{stratum}_target_companies": target,
                    f"{stratum}_weighted_claimants": drawn,
                    f"{stratum}_claimant_rows": int(chosen.size),
                    f"{stratum}_draw_capped": available < target,
                    f"{stratum}_target_expenditure_gbp": float(
                        table.at[letter, stratum]
                    ),
                    f"{stratum}_model_expenditure_gbp": float(
                        np.dot(weight[chosen], amounts)
                    ),
                    f"{stratum}_unallocated_gbp": left,
                    f"{stratum}_rows_at_turnover_cap": at_cap,
                }
            )
        row["hmrc_expenditure_gbp"] = spec.expenditure_gbp
        row["model_expenditure_gbp"] = (
            row["large_model_expenditure_gbp"] + row["sme_model_expenditure_gbp"]
        )
        rows.append(row)
    large_claimant = claimant & ~is_sme
    sme_claimant = claimant & is_sme
    diagnostics = {
        "status": UK_FIRM_RD_STATUS,
        "method": UK_FIRM_RD_METHOD,
        "year": UK_FIRM_RD_YEAR,
        "seed": int(seed),
        "seed_tag": UK_FIRM_RD_SEED_TAG,
        "companies_per_claim": ratio,
        "hmrc_national_claims": source.national_claims,
        "hmrc_national_companies": source.national_companies,
        "hmrc_unallocated_claims": source.unknown_claims,
        "hmrc_unallocated_expenditure_gbp": source.unknown_expenditure_gbp,
        "hmrc_large_expenditure_gbp": source.national_large_expenditure_gbp,
        "hmrc_sme_expenditure_gbp": source.national_sme_expenditure_gbp,
        "hmrc_section_expenditure_gbp": float(
            sum(s.expenditure_gbp for s in source.sections.values())
        ),
        "weighted_claimants": float(weight[claimant].sum()),
        "weighted_large_claimants": float(weight[large_claimant].sum()),
        "weighted_sme_claimants": float(weight[sme_claimant].sum()),
        "model_expenditure_gbp": float(np.dot(weight, expenditure)),
        "model_large_expenditure_gbp": float(
            np.dot(weight[large_claimant], expenditure[large_claimant])
        ),
        "model_sme_expenditure_gbp": float(
            np.dot(weight[sme_claimant], expenditure[sme_claimant])
        ),
        "unallocated_expenditure_gbp": unallocated,
        "rows_at_turnover_cap": capped_rows,
        "draw_capped_strata": [
            f"{row['section']}:{stratum}"
            for row in rows
            for stratum in ("large", "sme")
            if row[f"{stratum}_draw_capped"]
        ],
        "sections": rows,
        "sha256": dict(source.sha256),
    }
    return UKFirmRDAssignment(
        qualifying_rd_expenditure=expenditure,
        is_sme=is_sme,
        claimant=claimant,
        diagnostics=MappingProxyType(diagnostics),
    )


def assign_uk_firm_rd_tax_credits(
    firms: pd.DataFrame, source: UKFirmRDSource, *, seed: int
) -> pd.DataFrame:
    """A copy of ``firms`` with ``firm_qualifying_rd_expenditure`` (£) and
    ``firm_is_sme`` set; rows, order, index and other columns unchanged.

    ``firms`` needs one column of each tuple in ``UK_FIRM_RD_INPUT_COLUMNS``.
    """

    draw = draw_uk_firm_rd_tax_credits(firms, source, seed=seed)
    result = firms.copy()
    result["firm_qualifying_rd_expenditure"] = draw.qualifying_rd_expenditure
    result["firm_is_sme"] = draw.is_sme
    return result


__all__ = [
    "UK_FIRM_RD_CLAIMS_FILE",
    "UK_FIRM_RD_EXPENDITURE_FILE",
    "UK_FIRM_RD_FILES",
    "UK_FIRM_RD_INPUT_COLUMNS",
    "UK_FIRM_RD_METHOD",
    "UK_FIRM_RD_SECTIONS",
    "UK_FIRM_RD_SECTOR_FILE",
    "UK_FIRM_RD_SEED_TAG",
    "UK_FIRM_RD_SME_EMPLOYEE_LIMIT",
    "UK_FIRM_RD_STATUS",
    "UK_FIRM_RD_UNKNOWN_SECTOR",
    "UK_FIRM_RD_YEAR",
    "UKFirmRDAssignment",
    "UKFirmRDSection",
    "UKFirmRDSource",
    "assign_uk_firm_rd_tax_credits",
    "draw_uk_firm_rd_tax_credits",
    "read_uk_firm_rd_source",
    "uk_firm_rd_expenditure_table",
]
