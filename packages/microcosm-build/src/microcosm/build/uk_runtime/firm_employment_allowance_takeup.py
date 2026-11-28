"""UK firm Employment Allowance claims: an opt-in draw at HMRC take-up by size.

``firm_employment_allowance`` sets ``firm_claims_employment_allowance`` by
rule: every firm with an employee to claim for claims. HMRC counts 1,223,000
employers benefitting in 2024-25 (*Employment Allowance take-up statistics:
2024 to 2025 tax year estimates*, published 25 June 2025, committed in
``tests/fixtures/uk/firm_employment_allowance_takeup_2024_25/``). A firm table
with more eligible employers than that over-claims. This module, opt-in
(``build_uk_firm_dataset``'s ``employment_allowance_takeup_dir``), thins the
rule flag towards HMRC's claimant count in each employee size band. It
reaches the count only in a band with at least as many weighted eligible
firms as HMRC claimants: see the shortfall below.

Eligibility (all must hold; each is an assumption on the firm table):

- the rule flag claims (``firm_claims_employment_allowance``);
- ``firm_employees`` (employees, not working proprietors) is at least one;
- not a public body (``firm_is_public_body``) and not a single-director
  company (``firm_is_single_director_company``);
- a PROXY employer NIC before the allowance is below the £100,000 limit.
  The proxy is ``UK_FIRM_EA_PROXY_SECONDARY_RATE`` times pay above the
  secondary threshold, with every employee's pay taken above it unless
  ``firm_paybill_below_secondary_threshold`` and
  ``firm_employees_below_secondary_threshold`` are present, in which case
  the pay of employees below the threshold is removed first. It is
  current-year and paybill based; the law (NICA 2014 s.2(4B)) tests the
  prior year's secondary Class 1 liabilities, which policyengine-uk
  computes itself. A firm near the limit can be denied here and allowed
  there, or the reverse.

Size bands are HMRC's (PAYE employees at the end of the tax year), with
lower edges read from the fixture's labels. HMRC publishes 250+ as ``<500``,
a suppressed marker: no number is invented for it. Bands from
``UK_FIRM_EA_TAKEUP_POOLED_FROM`` up are pooled into one whose count is the
sum of their published numbers, so the pooled 50+ band takes the 50-249
figure (7,000) and the 250+ claimants (fewer than 500) are folded in and
ignored. The bands' published counts sum to 1,222,000, not the 1,223,000
total, from HMRC's rounding; neither is rescaled.

Per band, propensity = min(1, HMRC claimants / weighted eligible firms); a
band with fewer weighted eligible firms than claimants is capped at one and
its shortfall reported. Each firm draws one uniform from its own stream,
``np.random.default_rng(SeedSequence([seed, UK_FIRM_EA_TAKEUP_SEED_TAG]))``,
in ``firm_id`` order, so the draw does not depend on row order and no other
stream moves. The new ``firm_claims_employment_allowance`` is
eligible-and-drawn, a subset of the rule flag. Ineligible firms that the
rule flag let claim are switched off; policyengine-uk would deny most of
them the allowance anyway, except those the proxy denies.

Under this opt-in the band totals, and so the 1,223,000 total, are read
by the draw: they no longer test the firm table. They are not met exactly.
Every firm pays its section's mean (``firm_paybill.py``), so the proxy NIC
passes the £100,000 limit at a few dozen employees and almost no firm in
the pooled 50+ band is eligible: on the thinned 2024-25 test build that
band has about 10 weighted eligible firms against HMRC's 7,000, its
propensity is capped at one and its ``cap_shortfall`` is about 6,990, so
the drawn total is about 1.21 million. The proxy is kept rather than
dropped there because policyengine-uk applies the same limit and would deny
those claims anyway; HMRC's 50+ claimants are low-paying employers the
section-mean paybill cannot represent. HMRC's sector and region
tables stay held out (``uk_firm_employment_allowance_takeup_held_out``), as
does the £3.3bn 2024-25 cost in
``tests/fixtures/uk/firm_employment_allowance_2024_25/``. Status: assumed.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.calibrate.geography_constants import (
    UK_LADDER_NATION_REGION_CODES,
    UK_REGION_TIER_ENUM,
)

__all__ = [
    "UK_FIRM_EA_LIABILITY_LIMIT_GBP",
    "UK_FIRM_EA_PROXY_SECONDARY_RATE",
    "UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP",
    "UK_FIRM_EA_TAKEUP_FILES",
    "UK_FIRM_EA_TAKEUP_METHOD",
    "UK_FIRM_EA_TAKEUP_POOLED_FROM",
    "UK_FIRM_EA_TAKEUP_SEED_TAG",
    "UK_FIRM_EA_TAKEUP_STATUS",
    "UKFirmEATakeupBand",
    "UKFirmEATakeupDraw",
    "UKFirmEATakeupTables",
    "assign_uk_firm_employment_allowance_takeup",
    "draw_uk_firm_employment_allowance_takeup",
    "read_uk_firm_employment_allowance_takeup",
    "uk_firm_employer_nic_proxy",
    "uk_firm_employment_allowance_eligible",
    "uk_firm_employment_allowance_takeup_bands",
    "uk_firm_employment_allowance_takeup_held_out",
]

#: Status of the drawn flag, in ``UK_FIRM_POLICYENGINE_UK_INPUT_STATUS`` terms.
UK_FIRM_EA_TAKEUP_STATUS = "assumed"

#: 2024-25 secondary Class 1 NIC rate (SSCBA 1992 s.9; HMRC rates and
#: thresholds for employers 2024 to 2025). Used only for the proxy below.
UK_FIRM_EA_PROXY_SECONDARY_RATE = 0.138
#: 2024-25 secondary threshold, £175 a week, £9,100 a year (HMRC rates and
#: thresholds for employers 2024 to 2025). Used only for the proxy below.
UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP = 9_100.0
#: An employer whose secondary Class 1 liabilities reached £100,000 in the
#: prior tax year cannot claim (NICA 2014 s.2(4B), 2024-25).
UK_FIRM_EA_LIABILITY_LIMIT_GBP = 100_000.0

#: Lower edge (employees) from which HMRC's size bands are pooled into one.
UK_FIRM_EA_TAKEUP_POOLED_FROM = 50

#: Second word of the draw's ``SeedSequence``: ASCII "EATU".
UK_FIRM_EA_TAKEUP_SEED_TAG = 0x45415455

#: How the flag is set, for build receipts.
UK_FIRM_EA_TAKEUP_METHOD = (
    "Opt-in Bernoulli draw over firms eligible by rule flag, employees, public "
    "body, single-director company and a proxy employer NIC below GBP 100,000; "
    "propensity per HMRC size band (pooled from 50 employees; the suppressed "
    "250+ count is ignored) = min(1, HMRC 2024-25 claimants / weighted "
    "eligible firms)."
)

#: The fixture's files, by table.
UK_FIRM_EA_TAKEUP_FILES = {
    "size": "hmrc_ea_takeup_by_size.csv",
    "sector": "hmrc_ea_takeup_by_sector.csv",
    "region": "hmrc_ea_takeup_by_region.csv",
}
_COLUMNS = {
    "size": ("business_size", "employers_benefitting", "share_percent"),
    "sector": (
        "sic_section",
        "sic_description",
        "employers_benefitting",
        "share_percent",
    ),
    "region": (
        "region_code",
        "region_description",
        "employers_benefitting",
        "share_percent",
    ),
}
_SUPPRESSED = re.compile(r"^<\d+$")
_BAND_LABEL = re.compile(r"^(\d+)(?:-(\d+)|\+)$")


def _count(value: str) -> float | None:
    """A published count, or None for a suppressed marker such as ``<500``."""

    if _SUPPRESSED.match(value):
        return None
    if not value.isdigit():
        raise ValueError(f"expected a count or a '<n' marker, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class UKFirmEATakeupTables:
    """HMRC's 2024-25 Employment Allowance claimants by size, sector, region.

    Each row is ``(code, description, claimants, marker)``: ``claimants`` is
    None where HMRC suppressed the count, and ``marker`` keeps the published
    text (``"<500"``) there, else is None. The size table's description is
    its label. ``total`` is the published total; ``sha256`` is per file.
    """

    size: tuple[tuple[str, str, float | None, str | None], ...]
    sector: tuple[tuple[str, str, float | None, str | None], ...]
    region: tuple[tuple[str, str, float | None, str | None], ...]
    total: float
    sha256: Mapping[str, str]


def read_uk_firm_employment_allowance_takeup(
    directory: str | Path,
) -> UKFirmEATakeupTables:
    """Read the three HMRC take-up tables from ``directory``."""

    tables: dict[str, tuple[tuple[str, str, float | None, str | None], ...]] = {}
    totals: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for key, name in UK_FIRM_EA_TAKEUP_FILES.items():
        path = Path(directory) / name
        data = path.read_bytes()
        rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
        if not rows or tuple(rows[0]) != _COLUMNS[key]:
            raise ValueError(f"{path} must have columns {list(_COLUMNS[key])}.")
        # The size table's label doubles as its description.
        code, description = _COLUMNS[key][0], _COLUMNS[key][-3]
        body = []
        for row in rows:
            value = row["employers_benefitting"]
            claimants = _count(value)
            if row[code] == "Total":
                if claimants is None:
                    raise ValueError(f"{path}: the Total is suppressed.")
                totals[key] = claimants
                continue
            body.append(
                (
                    row[code],
                    row[description],
                    claimants,
                    value if claimants is None else None,
                )
            )
        if key not in totals:
            raise ValueError(f"{path} has no Total row.")
        tables[key] = tuple(body)
        hashes[name] = hashlib.sha256(data).hexdigest()
    if len(set(totals.values())) != 1:
        raise ValueError(f"The tables' totals differ: {totals}.")
    return UKFirmEATakeupTables(
        size=tables["size"],
        sector=tables["sector"],
        region=tables["region"],
        total=totals["size"],
        sha256=hashes,
    )


@dataclass(frozen=True)
class UKFirmEATakeupBand:
    """One modelled size band: employees from ``lower`` up to the next band's."""

    lower: int
    label: str
    hmrc_claimants: float
    #: Published bands pooled into this one.
    pooled: tuple[str, ...]
    #: Pooled bands whose count is suppressed and so not in ``hmrc_claimants``,
    #: with their markers.
    ignored_suppressed: tuple[tuple[str, str], ...]


def uk_firm_employment_allowance_takeup_bands(
    tables: UKFirmEATakeupTables,
    pooled_from: int = UK_FIRM_EA_TAKEUP_POOLED_FROM,
) -> tuple[UKFirmEATakeupBand, ...]:
    """Modelled size bands from the size table's labels, pooled from ``pooled_from``.

    A suppressed band may only sit inside the pooled top band, where its count
    is left out and listed in ``ignored_suppressed``.
    """

    parsed = []
    for label, _, claimants, marker in tables.size:
        match = _BAND_LABEL.match(label)
        if match is None:
            raise ValueError(f"Unexpected size band label {label!r}.")
        parsed.append((int(match.group(1)), label, claimants, marker))
    lowers = [lower for lower, *_ in parsed]
    if lowers != sorted(lowers) or len(set(lowers)) != len(lowers):
        raise ValueError(f"Size bands are not in increasing order: {lowers}.")
    if pooled_from not in lowers:
        raise ValueError(f"{pooled_from} is not a published band edge {lowers}.")
    bands = []
    for lower, label, claimants, marker in parsed:
        if lower >= pooled_from:
            continue
        if claimants is None:
            raise ValueError(
                f"Size band {label!r} is suppressed ({marker}) below the pooled "
                "band; no number is invented for it."
            )
        bands.append(UKFirmEATakeupBand(lower, label, claimants, (label,), ()))
    top = [row for row in parsed if row[0] >= pooled_from]
    known = [row for row in top if row[2] is not None]
    if not known:
        raise ValueError("Every pooled size band is suppressed.")
    bands.append(
        UKFirmEATakeupBand(
            lower=pooled_from,
            label=f"{pooled_from}+",
            hmrc_claimants=float(sum(row[2] for row in known)),
            pooled=tuple(row[1] for row in top),
            ignored_suppressed=tuple((row[1], row[3]) for row in top if row[2] is None),
        )
    )
    return tuple(bands)


def uk_firm_employer_nic_proxy(
    paybill: np.ndarray | pd.Series,
    employees: np.ndarray | pd.Series,
    paybill_below_threshold: np.ndarray | pd.Series | None = None,
    employees_below_threshold: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    """A PROXY for 2024-25 employer NIC before the allowance, in pounds.

    Rate times pay above the secondary threshold, summed over employees with
    every employee's pay taken above it, or, given the below-threshold
    columns, over the employees above it only. Floored at zero. Not
    policyengine-uk's calculation.
    """

    pay = np.asarray(paybill, dtype=np.float64)
    heads = np.asarray(employees, dtype=np.float64)
    if (paybill_below_threshold is None) != (employees_below_threshold is None):
        raise ValueError("Give both below-threshold columns or neither.")
    if paybill_below_threshold is not None:
        pay = pay - np.asarray(paybill_below_threshold, dtype=np.float64)
        heads = heads - np.asarray(employees_below_threshold, dtype=np.float64)
    above = np.maximum(pay - heads * UK_FIRM_EA_PROXY_SECONDARY_THRESHOLD_GBP, 0.0)
    return UK_FIRM_EA_PROXY_SECONDARY_RATE * above


_REQUIRED = (
    "firm_employees",
    "firm_claims_employment_allowance",
    "firm_is_single_director_company",
    "firm_is_public_body",
    "firm_paybill",
)
_BELOW = (
    "firm_paybill_below_secondary_threshold",
    "firm_employees_below_secondary_threshold",
)


def uk_firm_employment_allowance_eligible(
    firms: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Eligibility and its parts: ``eligible`` and one failing mask per reason.

    The reasons are ``no_rule_claim``, ``no_employees``, ``public_body``,
    ``single_director_company`` and ``proxy_nic_over_limit``; they overlap.
    """

    missing = [column for column in _REQUIRED if column not in firms]
    if missing:
        raise ValueError(f"UK firm Employment Allowance take-up needs {missing}.")
    employees = firms["firm_employees"].to_numpy(np.float64)
    if np.any(employees < 0):
        raise ValueError("UK firm firm_employees must be non-negative.")
    below = (
        [firms[column].to_numpy() for column in _BELOW]
        if all(column in firms for column in _BELOW)
        else [None, None]
    )
    nic = uk_firm_employer_nic_proxy(
        firms["firm_paybill"].to_numpy(), employees, *below
    )
    reasons = {
        "no_rule_claim": ~firms["firm_claims_employment_allowance"].to_numpy(bool),
        "no_employees": ~(employees >= 1),
        "public_body": firms["firm_is_public_body"].to_numpy(bool),
        "single_director_company": firms["firm_is_single_director_company"].to_numpy(
            bool
        ),
        "proxy_nic_over_limit": ~(nic < UK_FIRM_EA_LIABILITY_LIMIT_GBP),
    }
    failing = np.zeros(len(firms), dtype=bool)
    for mask in reasons.values():
        failing |= mask
    return {"eligible": ~failing, "proxy_employer_nic_gbp": nic, **reasons}


@dataclass(frozen=True)
class UKFirmEATakeupDraw:
    """The drawn flag, positional with the input frame, and diagnostics."""

    claims: np.ndarray
    eligible: np.ndarray
    propensity: np.ndarray
    diagnostics: Mapping[str, Any]


def _ordered_uniforms(ids: np.ndarray, seed: int) -> np.ndarray:
    ordered = np.sort(ids, kind="stable")
    if np.any(ordered[1:] == ordered[:-1]):
        raise ValueError("UK firm ids must be unique for the take-up draw.")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), UK_FIRM_EA_TAKEUP_SEED_TAG])
    )
    uniforms = np.empty(len(ids), dtype=np.float64)
    uniforms[np.argsort(ids, kind="stable")] = rng.random(len(ids))
    return uniforms


def draw_uk_firm_employment_allowance_takeup(
    firms: pd.DataFrame,
    tables: UKFirmEATakeupTables,
    *,
    seed: int,
    id_column: str = "firm_id",
    weight_column: str = "firm_weight",
) -> UKFirmEATakeupDraw:
    """Draw each eligible firm's claim at its size band's propensity."""

    for column in (id_column, weight_column):
        if column not in firms:
            raise ValueError(f"UK firm Employment Allowance take-up needs {column!r}.")
    weight = firms[weight_column].to_numpy(np.float64)
    if np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("UK firm weights must be finite and non-negative.")
    parts = uk_firm_employment_allowance_eligible(firms)
    eligible = parts["eligible"]
    bands = uk_firm_employment_allowance_takeup_bands(tables)
    edges = np.array([band.lower for band in bands], dtype=np.float64)
    employees = firms["firm_employees"].to_numpy(np.float64)
    band = np.searchsorted(edges, employees, side="right") - 1
    in_band = eligible & (band >= 0)
    band_propensity = np.zeros(len(bands), dtype=np.float64)
    band_rows = []
    for index, spec in enumerate(bands):
        members = in_band & (band == index)
        weighted = float(weight[members].sum())
        if weighted > 0:
            band_propensity[index] = min(1.0, spec.hmrc_claimants / weighted)
        band_rows.append(
            {
                "band": spec.label,
                "lower_employees": spec.lower,
                "pooled_hmrc_bands": list(spec.pooled),
                "ignored_suppressed": [list(item) for item in spec.ignored_suppressed],
                "hmrc_claimants": spec.hmrc_claimants,
                "weighted_eligible": weighted,
                "eligible_rows": int(members.sum()),
                "propensity": float(band_propensity[index]),
                "capped": weighted < spec.hmrc_claimants,
                "cap_shortfall": max(0.0, spec.hmrc_claimants - weighted),
            }
        )
    propensity = np.where(in_band, band_propensity[np.clip(band, 0, None)], 0.0)
    uniforms = _ordered_uniforms(firms[id_column].to_numpy(), seed)
    claims = in_band & (uniforms < propensity)
    for index, row in enumerate(band_rows):
        row["weighted_claimants"] = float(weight[claims & (band == index)].sum())
    rule = ~parts["no_rule_claim"]
    switched_off = {
        reason: float(weight[rule & parts[reason]].sum())
        for reason in (
            "no_employees",
            "public_body",
            "single_director_company",
            "proxy_nic_over_limit",
        )
    }
    diagnostics = {
        "status": UK_FIRM_EA_TAKEUP_STATUS,
        "seed": int(seed),
        "seed_tag": UK_FIRM_EA_TAKEUP_SEED_TAG,
        "proxy_nic_uses_below_threshold_columns": all(c in firms for c in _BELOW),
        "bands": band_rows,
        "hmrc_band_sum": float(sum(band.hmrc_claimants for band in bands)),
        "hmrc_total": tables.total,
        "weighted_rule_claimants": float(weight[rule].sum()),
        "weighted_eligible": float(weight[in_band].sum()),
        "weighted_claimants": float(weight[claims].sum()),
        "rule_claimants_ineligible_weighted": switched_off,
        "rule_claimants_ineligible_weighted_total": float(
            weight[rule & ~eligible].sum()
        ),
        "sha256": dict(tables.sha256),
    }
    return UKFirmEATakeupDraw(
        claims=claims, eligible=eligible, propensity=propensity, diagnostics=diagnostics
    )


def assign_uk_firm_employment_allowance_takeup(
    firms: pd.DataFrame,
    tables: UKFirmEATakeupTables,
    *,
    seed: int,
    id_column: str = "firm_id",
    weight_column: str = "firm_weight",
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """``firms`` with the drawn ``firm_claims_employment_allowance``, and diagnostics."""

    draw = draw_uk_firm_employment_allowance_takeup(
        firms, tables, seed=seed, id_column=id_column, weight_column=weight_column
    )
    return firms.assign(firm_claims_employment_allowance=draw.claims), draw.diagnostics


def _region_names(code: str) -> str | None:
    """policyengine-uk ``Region`` name for an HMRC region code, if any."""

    return UK_REGION_TIER_ENUM.get(UK_LADDER_NATION_REGION_CODES.get(code, code))


def _compare(
    labels: np.ndarray,
    weight: np.ndarray,
    rows: Sequence[tuple[str, str, float | None, str | None]],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    by_code: dict[str, float] = {}
    unmapped: dict[str, float] = {}
    for label in pd.unique(labels):
        mass = float(weight[labels == label].sum())
        code = mapping.get(str(label))
        if code is None:
            unmapped[str(label)] = mass
        else:
            by_code[code] = by_code.get(code, 0.0) + mass
    out = []
    for code, description, claimants, marker in rows:
        model = by_code.get(code, 0.0)
        out.append(
            {
                "hmrc_code": code,
                "description": description,
                "model_weighted_claimants": model,
                "hmrc_claimants": claimants,
                "hmrc_marker": marker,
                "ratio": model / claimants if claimants else None,
            }
        )
    return {
        "rows": out,
        "unmapped_model_labels": unmapped,
        "unmapped_hmrc_codes": sorted(
            code for code, *_ in rows if code not in set(mapping.values())
        ),
    }


def uk_firm_employment_allowance_takeup_held_out(
    firms: pd.DataFrame,
    claims: np.ndarray,
    tables: UKFirmEATakeupTables,
    *,
    weight_column: str = "firm_weight",
) -> dict[str, Any]:
    """Weighted drawn claimants by SIC section and region beside HMRC's.

    Held out: the draw never reads these tables. ``firm_sic_section`` letters
    map to HMRC's sections, with B, D, E, O and U onto HMRC's pooled
    ``B,D,E,O,U``; ``UNKNOWN`` is unmapped (HMRC apportioned its unknown
    sectors). ``firm_region`` names map through the region's GSS code, with
    HMRC's ``W99999999``, ``S99999999`` and ``N99999999`` taken as Wales,
    Scotland and Northern Ireland. A table whose column is absent is skipped.
    """

    claims = np.asarray(claims, dtype=bool)
    weight = np.where(claims, firms[weight_column].to_numpy(np.float64), 0.0)
    out: dict[str, Any] = {"model_weighted_claimants": float(weight.sum())}
    if "firm_sic_section" in firms:
        sector_map = {
            letter: code for code, *_ in tables.sector for letter in code.split(",")
        }
        out["sector"] = _compare(
            firms["firm_sic_section"].astype(str).to_numpy(object),
            weight,
            tables.sector,
            sector_map,
        )
    if "firm_region" in firms:
        region_map = {
            name: code
            for code, *_ in tables.region
            if (name := _region_names(code)) is not None
        }
        out["region"] = _compare(
            firms["firm_region"].astype(str).to_numpy(object),
            weight,
            tables.region,
            region_map,
        )
    return out
