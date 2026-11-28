"""UK firm taxable profit: an opt-in draw from HMRC's liability bands.

policyengine-uk's ``firm_taxable_trading_profit`` is unset by the experimental
UK firm build, so every company's corporation tax is 0. This module imputes
it for companies (``ons_legal_status`` 1, policyengine-uk ``COMPANY``) from
one published table, HMRC *Corporation Tax Statistics 2026*, Tables 9A and 9B
(*Tax liability by band*), financial year 2024-25: the number of companies and
the total liability in each of thirteen liability bands, and the number with
no liability. The extracted CSV and its provenance sit in
``tests/fixtures/uk/firm_ct_2024/`` (``tools/extract_uk_firm_ct_tables.py``
re-creates it). ``build_uk_firm_dataset(profit_dir=...)`` and
``tools/build_uk_firms.py --profit-dir`` turn it on; the status of the
result is ``assumed``.

**Inversion.** With no associated companies, no exempt distributions and a
twelve-month accounting period inside the financial year, policyengine-uk's
``corporation_tax_before_credits`` on taxable total profits ``p`` is

    T(p) = s p                        p <= L
         = m p - f (U - p)            L < p <= U
         = m p                        p > U

with main rate ``m``, small profits rate ``s``, marginal relief fraction ``f``
and lower and upper limits ``L``, ``U`` (``UK_FIRM_PROFIT_CT_SCHEDULE``; the
values are policyengine-uk's parameters for the financial year 2024). ``T`` is
continuous and strictly increasing, so a liability ``t`` has one profit:

    p = t / s                         t <= s L
      = (t + f U) / (m + f)           s L < t <= m U
      = t / m                         t > m U

(``uk_corporation_tax_profit``). Each liability band's edges map through it to
a taxable-profit band (``uk_firm_profit_bands``).

**Assignment.** Companies are ranked by turnover within their SIC section
(weighted, highest first; equal turnovers are ordered by a uniform draw from
this module's own random stream, seeded by ``UK_FIRM_PROFIT_SEED`` and drawn
in ``firm_id`` order, never the generator's stream). The within-section
weighted rank fractions are pooled, and bands are allotted from the top of
the pooled ranking: the highest band takes the first rows whose cumulative
weight reaches the band's published number of companies, and so on down; the
companies left over have no liability and a taxable profit of 0. A row
belongs to the band holding the midpoint of its weight, so a band's weighted
count differs from the table's by at most about one row's weight at each edge.

Within a band, liabilities rise with rank and their weighted total is the
band's published total: in a closed band ``(a, b]`` the liability is
``a + (b - a) x`` with ``x`` a power of the within-band rank fraction, the
power set by the band mean and ``x`` then shifted to make the mean exact; in
the open top band the excess over the floor is exponential in shape, scaled to
the mean. Where the published total over the build's weighted count falls
outside the band (the build holds too few or too many companies there), the
table's own band mean is used instead and the diagnostics say so.

Assumptions beyond the source, all of them:

- the tables' "tax liability" is Corporation Tax *payable* (after double
  taxation relief, with ring fence tax and the supplementary charge), read
  here as ``T(p)``;
- the imputed amount is profits chargeable to corporation tax, after capital
  allowances, reliefs and losses, standing in for
  ``firm_taxable_trading_profit`` while policyengine-uk's capital allowances,
  non-trading profits, losses and R&D inputs are unset; a company with no
  liability gets 0, never a loss;
- profit rises with turnover rank inside every SIC section, and every
  section has the same share of companies in each band: the sector pattern
  of liability (Tables 6A and 6B, held out in ``ct_sic_sections.csv``) is
  not used, so the financial sector's share of liability is not reproduced;
- HMRC counts 3.3 million companies filing a return, many of them dormant
  or outside the VAT and PAYE frame; the build's companies are ONS frame
  enterprises. Positive-liability band counts are matched in absolute terms
  and the zero band is whatever the build has left over;
- sole traders, partnerships and other bodies get 0 (policyengine-uk charges
  corporation tax on ``COMPANY`` only);
- public corporations (``ons_legal_status`` 4) export as ``COMPANY``, so they
  are liable to corporation tax, but they get 0 profit: HMRC's liability
  bands are allotted to status 1 only. Their corporation tax is therefore 0,
  an understatement. Rows with only ``firm_legal_form`` (no
  ``ons_legal_status``) count every ``COMPANY``, status 4 included.

HMRC's sector tables do not isolate banks (the finest split is SIC division
64, *Financial service activities, except insurance and pension funding*), so
``firm_is_bank`` stays unset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.firm_table import uk_firm_sic_sections

#: The extracted liability-band table's file name in a profit source directory.
UK_FIRM_PROFIT_FILE = "ct_liability_bands.csv"
#: Its columns, in order.
UK_FIRM_PROFIT_SOURCE_COLUMNS = (
    "band",
    "lower_gbp",
    "upper_gbp",
    "companies",
    "liability_gbp_million",
)
#: ``band`` of the rows that are not a positive-liability band.
UK_FIRM_PROFIT_ZERO_BAND = "0"
UK_FIRM_PROFIT_POSITIVE_ROW = "POSITIVE"
UK_FIRM_PROFIT_ALL_RETURNS_ROW = "ALL_RETURNS"
#: The financial year the table describes (1 April 2024 to 31 March 2025).
UK_FIRM_PROFIT_FINANCIAL_YEAR = "2024-25"
#: The receipt status and method of the imputed profit.
UK_FIRM_PROFIT_STATUS = "assumed"
UK_FIRM_PROFIT_METHOD = (
    "HMRC Corporation Tax Statistics 2026 Tables 9A/9B liability bands "
    "(2024-25), inverted through the corporation tax rate schedule and "
    "allotted to companies by turnover rank within SIC section"
)
#: Seed of this module's own random stream (tie-breaks between equal
#: turnovers). It is not the generator's seed and shares no state with it.
UK_FIRM_PROFIT_SEED = 20_240_904
#: ONS legal status code of a company (policyengine-uk ``COMPANY``).
UK_FIRM_PROFIT_COMPANY_LEGAL_STATUS = 1
_COMPANY_LEGAL_FORM = "COMPANY"
_GBP_PER_MILLION = 1_000_000.0


@dataclass(frozen=True)
class UKCorporationTaxSchedule:
    """The corporation tax rate schedule on taxable total profits."""

    main_rate: float
    small_profits_rate: float
    marginal_relief_fraction: float
    lower_limit: float
    upper_limit: float

    def __post_init__(self) -> None:
        low = self.small_profits_rate * self.lower_limit
        high = self.main_rate * self.lower_limit - self.marginal_relief_fraction * (
            self.upper_limit - self.lower_limit
        )
        if not (
            0 < self.small_profits_rate <= self.main_rate
            and self.marginal_relief_fraction >= 0
            and 0 < self.lower_limit < self.upper_limit
        ):
            raise ValueError("UK corporation tax schedule is not ordered.")
        if not np.isclose(low, high, rtol=1e-12, atol=1e-9):
            raise ValueError(
                "UK corporation tax schedule is not continuous at the lower limit."
            )


#: policyengine-uk's ``gov.hmrc.corporation_tax`` parameters for the financial
#: year 2024 (values dated 2023-04-01): ``main_rate`` (Finance Act 2021
#: s. 6(2)(b)), ``small_profits_rate`` (s. 7(2)(a)), ``marginal_relief.
#: fraction`` (s. 7(2)(b), 3/200ths), ``marginal_relief.lower_limit`` and
#: ``upper_limit`` (Corporation Tax Act 2010 s. 18D). Read from the
#: policyengine-uk checkout at 57a99a67ac26 on 2026-11-22.
UK_FIRM_PROFIT_CT_SCHEDULE = UKCorporationTaxSchedule(
    main_rate=0.25,
    small_profits_rate=0.19,
    marginal_relief_fraction=0.015,
    lower_limit=50_000.0,
    upper_limit=250_000.0,
)


def uk_corporation_tax_liability(
    profit: np.ndarray | float,
    schedule: UKCorporationTaxSchedule = UK_FIRM_PROFIT_CT_SCHEDULE,
) -> np.ndarray:
    """Corporation tax (pounds) on taxable total profits ``profit``.

    The small profits rate up to the lower limit, the main rate less marginal
    relief up to the upper limit, the main rate above it; 0 on no profit.
    One company, no associates, no exempt distributions, a full year.
    """

    p = np.maximum(np.asarray(profit, dtype=np.float64), 0.0)
    relief = schedule.marginal_relief_fraction * (schedule.upper_limit - p)
    return np.select(
        [p <= schedule.lower_limit, p <= schedule.upper_limit],
        [p * schedule.small_profits_rate, p * schedule.main_rate - relief],
        default=p * schedule.main_rate,
    )


def uk_corporation_tax_profit(
    liability: np.ndarray | float,
    schedule: UKCorporationTaxSchedule = UK_FIRM_PROFIT_CT_SCHEDULE,
) -> np.ndarray:
    """Taxable total profits (pounds) whose corporation tax is ``liability``.

    The inverse of ``uk_corporation_tax_liability`` on non-negative amounts.
    """

    t = np.maximum(np.asarray(liability, dtype=np.float64), 0.0)
    small_top = schedule.small_profits_rate * schedule.lower_limit
    main_floor = schedule.main_rate * schedule.upper_limit
    taper = (t + schedule.marginal_relief_fraction * schedule.upper_limit) / (
        schedule.main_rate + schedule.marginal_relief_fraction
    )
    return np.select(
        [t <= small_top, t <= main_floor],
        [t / schedule.small_profits_rate, taper],
        default=t / schedule.main_rate,
    )


@dataclass(frozen=True)
class UKFirmProfitBand:
    """One positive-liability band: ``lower`` exclusive, ``upper`` inclusive
    (``inf`` for the open band), pounds; ``liability`` in pounds."""

    label: str
    lower: float
    upper: float
    companies: float
    liability: float


@dataclass(frozen=True)
class UKFirmProfitSource:
    """HMRC liability bands, lowest first, with the table's own totals."""

    bands: tuple[UKFirmProfitBand, ...]
    zero_companies: float
    positive_companies: float
    positive_liability: float
    all_returns: float
    file: str
    sha256: str


def _number(text: str, where: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(
            f"UK firm profit source {where} {text!r} is not a number."
        ) from None
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"UK firm profit source {where} must be non-negative.")
    return value


def read_uk_firm_profit_source(path: str | Path) -> UKFirmProfitSource:
    """Read the liability-band table from a directory or the CSV itself.

    Raises ``ValueError`` unless the file has exactly
    ``UK_FIRM_PROFIT_SOURCE_COLUMNS``; a ``0`` row, contiguous bands rising
    from 0 to one open band, then ``POSITIVE`` and ``ALL_RETURNS`` rows; a
    positive count and liability in every band; and each band's mean
    liability inside the band.
    """

    path = Path(path)
    if path.is_dir():
        path = path / UK_FIRM_PROFIT_FILE
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != UK_FIRM_PROFIT_SOURCE_COLUMNS:
        raise ValueError(
            f"UK firm profit source columns {list(frame.columns)} are not "
            f"{list(UK_FIRM_PROFIT_SOURCE_COLUMNS)}."
        )
    records = {str(row.band).strip(): row for row in frame.itertuples(index=False)}
    if len(records) != len(frame):
        raise ValueError("UK firm profit source repeats a band.")
    special = (
        UK_FIRM_PROFIT_ZERO_BAND,
        UK_FIRM_PROFIT_POSITIVE_ROW,
        UK_FIRM_PROFIT_ALL_RETURNS_ROW,
    )
    for key in special:
        if key not in records:
            raise ValueError(f"UK firm profit source has no {key!r} row.")
    bands: list[UKFirmProfitBand] = []
    edge = 0.0
    for label, row in records.items():
        if label in special:
            continue
        lower = _number(row.lower_gbp, f"band {label} lower edge")
        if lower != edge or not np.isfinite(edge):
            raise ValueError(f"UK firm profit band {label} does not follow on.")
        upper = (
            _number(row.upper_gbp, f"band {label} upper edge")
            if row.upper_gbp.strip()
            else np.inf
        )
        companies = _number(row.companies, f"band {label} companies")
        liability = (
            _number(row.liability_gbp_million, f"band {label} liability")
            * _GBP_PER_MILLION
        )
        if not upper > lower or companies <= 0 or liability <= 0:
            raise ValueError(f"UK firm profit band {label} is empty or inverted.")
        if not lower < liability / companies < upper:
            raise ValueError(
                f"UK firm profit band {label} mean liability is outside the band."
            )
        bands.append(UKFirmProfitBand(label, lower, upper, companies, liability))
        edge = upper
    if not bands or np.isfinite(edge):
        raise ValueError("UK firm profit source must end with one open band.")
    positive = records[UK_FIRM_PROFIT_POSITIVE_ROW]
    return UKFirmProfitSource(
        bands=tuple(bands),
        zero_companies=_number(
            records[UK_FIRM_PROFIT_ZERO_BAND].companies, "zero-band companies"
        ),
        positive_companies=_number(positive.companies, "positive companies"),
        positive_liability=_number(positive.liability_gbp_million, "positive liability")
        * _GBP_PER_MILLION,
        all_returns=_number(
            records[UK_FIRM_PROFIT_ALL_RETURNS_ROW].companies, "all returns"
        ),
        file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def uk_firm_profit_bands(
    source: UKFirmProfitSource,
    schedule: UKCorporationTaxSchedule = UK_FIRM_PROFIT_CT_SCHEDULE,
) -> pd.DataFrame:
    """Each liability band with its taxable-profit band, lowest first.

    ``profit_lower_gbp`` and ``profit_upper_gbp`` are the liability edges
    through ``uk_corporation_tax_profit``; ``mean_liability_gbp`` and
    ``mean_profit_gbp`` describe a company at the band's mean liability.
    """

    lower = np.array([band.lower for band in source.bands])
    upper = np.array([band.upper for band in source.bands])
    companies = np.array([band.companies for band in source.bands])
    liability = np.array([band.liability for band in source.bands])
    finite_upper = np.where(np.isfinite(upper), upper, 0.0)
    return pd.DataFrame(
        {
            "band": [band.label for band in source.bands],
            "liability_lower_gbp": lower,
            "liability_upper_gbp": upper,
            "profit_lower_gbp": uk_corporation_tax_profit(lower, schedule),
            "profit_upper_gbp": np.where(
                np.isfinite(upper),
                uk_corporation_tax_profit(finite_upper, schedule),
                np.inf,
            ),
            "companies": companies,
            "liability_gbp": liability,
            "mean_liability_gbp": liability / companies,
            "mean_profit_gbp": uk_corporation_tax_profit(
                liability / companies, schedule
            ),
        }
    )


@dataclass(frozen=True)
class _Allotment:
    """Per row, in row order: band index (-1 for none) and liability."""

    band: np.ndarray
    liability: np.ndarray
    company: np.ndarray
    weight: np.ndarray
    sections: np.ndarray
    table_mean_bands: tuple[int, ...]


def _columns(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Company flag, weight, turnover, section and ``firm_id`` per row."""

    names = set(frame.columns)
    if "firm_id" not in names:
        raise ValueError("UK firm profit needs firm_id.")
    firm_id = frame["firm_id"].to_numpy()
    if not pd.Index(firm_id).is_unique:
        raise ValueError("UK firm profit needs one row per firm_id.")
    if "ons_legal_status" in names:
        company = (
            frame["ons_legal_status"].to_numpy() == UK_FIRM_PROFIT_COMPANY_LEGAL_STATUS
        )
    elif "firm_legal_form" in names:
        company = (
            frame["firm_legal_form"].astype(str).to_numpy(object) == _COMPANY_LEGAL_FORM
        )
    else:
        raise ValueError("UK firm profit needs ons_legal_status or firm_legal_form.")
    turnover_column = next(
        (name for name in ("annual_turnover_k", "firm_turnover") if name in names),
        None,
    )
    if turnover_column is None:
        raise ValueError("UK firm profit needs annual_turnover_k or firm_turnover.")
    turnover = frame[turnover_column].to_numpy(np.float64)
    if "firm_sic_section" in names:
        sections = frame["firm_sic_section"].astype(str).to_numpy(object)
    elif "sic_code" in names:
        sections = uk_firm_sic_sections(frame["sic_code"])
    else:
        raise ValueError("UK firm profit needs firm_sic_section or sic_code.")
    weight = (
        frame["firm_weight"].to_numpy(np.float64)
        if "firm_weight" in names
        else np.ones(len(frame), dtype=np.float64)
    )
    if not np.all(np.isfinite(turnover)) or not np.all(np.isfinite(weight)):
        raise ValueError("UK firm profit needs finite turnover and weights.")
    if np.any(weight < 0):
        raise ValueError("UK firm profit needs non-negative weights.")
    return company, weight, turnover, sections, firm_id


def _band_liabilities(
    weight: np.ndarray, band: UKFirmProfitBand
) -> tuple[np.ndarray, bool]:
    """Liabilities of one band's rows, given top rank first.

    Returns the liabilities and whether the table's band mean stood in for
    the published total over the weighted count.
    """

    total = weight.sum()
    mean = band.liability / total if total > 0 else np.inf
    table_mean = not band.lower < mean < band.upper
    if table_mean:
        mean = band.liability / band.companies
    if total <= 0:
        return np.full(len(weight), mean), table_mean
    # Rank fraction in (0, 1): near 1 at the top of the band.
    fraction = 1.0 - (np.cumsum(weight) - weight / 2.0) / total
    fraction = np.clip(fraction, 1e-12, 1.0 - 1e-12)
    if np.isfinite(band.upper):
        target = (mean - band.lower) / (band.upper - band.lower)
        shape = fraction ** ((1.0 - target) / target)
        got = np.average(shape, weights=weight)
        if got >= target:
            shape = shape * (target / got)
        else:
            shape = 1.0 - (1.0 - shape) * ((1.0 - target) / (1.0 - got))
        return band.lower + (band.upper - band.lower) * shape, table_mean
    excess = -np.log1p(-fraction)
    excess = excess / np.average(excess, weights=weight)
    return band.lower + (mean - band.lower) * excess, table_mean


def _allot(frame: pd.DataFrame, source: UKFirmProfitSource, seed: int) -> _Allotment:
    company, weight, turnover, sections, firm_id = _columns(frame)
    n = len(frame)
    # One draw per row in firm_id order, so row order never changes a draw.
    draws = np.empty(n, dtype=np.float64)
    draws[np.argsort(firm_id, kind="stable")] = np.random.default_rng(seed).random(n)

    rows = np.flatnonzero(company & (weight > 0))
    band_index = np.full(n, -1, dtype=np.int64)
    liability = np.zeros(n, dtype=np.float64)
    table_mean: list[int] = []
    if rows.size:
        # Weighted rank fraction within section, 0 at the highest turnover.
        codes = pd.factorize(sections[rows], sort=True)[0]
        order = np.lexsort((draws[rows], -turnover[rows], codes))
        ordered_weight = weight[rows][order]
        ordered_codes = codes[order]
        running = np.cumsum(ordered_weight)
        section_total = np.bincount(ordered_codes, weights=ordered_weight)
        before = np.concatenate(([0.0], np.cumsum(section_total)[:-1]))
        rank = np.empty(rows.size, dtype=np.float64)
        rank[order] = (
            running - ordered_weight / 2.0 - before[ordered_codes]
        ) / section_total[ordered_codes]
        # Pool the sections and allot bands from the top by cumulative weight.
        pooled = np.lexsort((draws[rows], rank))
        pooled_weight = weight[rows][pooled]
        midpoint = np.cumsum(pooled_weight) - pooled_weight / 2.0
        top_down = source.bands[::-1]
        edges = np.cumsum([band.companies for band in top_down])
        position = np.searchsorted(edges, midpoint, side="right")
        n_bands = len(top_down)
        for step, band in enumerate(top_down):
            members = np.flatnonzero(position == step)
            if not members.size:
                continue
            index = n_bands - 1 - step
            values, used_table_mean = _band_liabilities(pooled_weight[members], band)
            target = rows[pooled[members]]
            band_index[target] = index
            liability[target] = values
            if used_table_mean:
                table_mean.append(index)
    return _Allotment(
        band=band_index,
        liability=liability,
        company=company,
        weight=weight,
        sections=np.asarray(sections, dtype=object),
        table_mean_bands=tuple(sorted(table_mean)),
    )


def assign_uk_firm_taxable_profit(
    table_or_firms: pd.DataFrame,
    source: UKFirmProfitSource,
    *,
    schedule: UKCorporationTaxSchedule = UK_FIRM_PROFIT_CT_SCHEDULE,
    seed: int = UK_FIRM_PROFIT_SEED,
) -> np.ndarray:
    """Taxable profit in pounds per row, in row order.

    ``table_or_firms`` is an extended policyengine-uk firm table or the
    generator's rows after the attribute draw: ``firm_id``, a legal form
    (``ons_legal_status`` or ``firm_legal_form``), a turnover
    (``annual_turnover_k`` or ``firm_turnover``; only its order matters), a
    section (``firm_sic_section`` or ``sic_code``) and, optionally,
    ``firm_weight`` (ones without it). A row that is not a company, or has
    no weight, or falls below the lowest band, gets 0. The result does not
    depend on row order.
    """

    allotment = _allot(table_or_firms, source, seed)
    return np.where(
        allotment.band >= 0,
        uk_corporation_tax_profit(allotment.liability, schedule),
        0.0,
    )


def uk_firm_profit_diagnostics(
    table_or_firms: pd.DataFrame,
    source: UKFirmProfitSource,
    *,
    schedule: UKCorporationTaxSchedule = UK_FIRM_PROFIT_CT_SCHEDULE,
    seed: int = UK_FIRM_PROFIT_SEED,
) -> pd.DataFrame:
    """The draw against the table, one row per band, lowest first.

    The first row is the ``0`` band (companies with no liability; the table's
    count there covers every company filing a return, the build's only its
    own companies). ``weighted_liability_gbp`` is the rate schedule applied
    to the assigned profits, so it checks the inversion as well as the
    allotment; ``weighted_profit_gbp`` is the weighted assigned profit.
    ``table_mean_used`` marks a band whose published total over the build's
    weighted count fell outside the band, so the table's band mean was used.
    """

    allotment = _allot(table_or_firms, source, seed)
    profit = np.where(
        allotment.band >= 0,
        uk_corporation_tax_profit(allotment.liability, schedule),
        0.0,
    )
    tax = uk_corporation_tax_liability(profit, schedule)
    n_bands = len(source.bands)
    # Slot 0 is the zero band; companies only.
    slot = allotment.band[allotment.company] + 1
    weight = allotment.weight[allotment.company]
    weighted = np.bincount(slot, weights=weight, minlength=n_bands + 1)
    liability = np.bincount(
        slot, weights=weight * tax[allotment.company], minlength=n_bands + 1
    )
    profits = np.bincount(
        slot, weights=weight * profit[allotment.company], minlength=n_bands + 1
    )
    table_companies = np.array(
        [source.zero_companies, *(band.companies for band in source.bands)]
    )
    table_liability = np.array([0.0, *(band.liability for band in source.bands)])
    used = np.zeros(n_bands + 1, dtype=bool)
    used[[index + 1 for index in allotment.table_mean_bands]] = True
    return pd.DataFrame(
        {
            "band": [UK_FIRM_PROFIT_ZERO_BAND, *(b.label for b in source.bands)],
            "table_companies": table_companies,
            "weighted_companies": weighted,
            "table_liability_gbp": table_liability,
            "weighted_liability_gbp": liability,
            "weighted_profit_gbp": profits,
            "table_mean_used": used,
        }
    )


def uk_firm_profit_section_liability(
    table_or_firms: pd.DataFrame,
    source: UKFirmProfitSource,
    *,
    schedule: UKCorporationTaxSchedule = UK_FIRM_PROFIT_CT_SCHEDULE,
    seed: int = UK_FIRM_PROFIT_SEED,
) -> pd.DataFrame:
    """Weighted companies, companies with a liability, and liability per
    ``SICSection`` name, for a check against HMRC's held-out sector table."""

    allotment = _allot(table_or_firms, source, seed)
    tax = uk_corporation_tax_liability(
        uk_corporation_tax_profit(allotment.liability, schedule), schedule
    )
    tax = np.where(allotment.band >= 0, tax, 0.0)
    frame = pd.DataFrame(
        {
            "sic_section": allotment.sections[allotment.company],
            "weighted_companies": allotment.weight[allotment.company],
            "weighted_companies_with_liability": (
                allotment.weight * (allotment.band >= 0)
            )[allotment.company],
            "weighted_liability_gbp": (allotment.weight * tax)[allotment.company],
        }
    )
    return frame.groupby("sic_section", as_index=False, sort=True).sum()


__all__ = [
    "UK_FIRM_PROFIT_ALL_RETURNS_ROW",
    "UK_FIRM_PROFIT_COMPANY_LEGAL_STATUS",
    "UK_FIRM_PROFIT_CT_SCHEDULE",
    "UK_FIRM_PROFIT_FILE",
    "UK_FIRM_PROFIT_FINANCIAL_YEAR",
    "UK_FIRM_PROFIT_METHOD",
    "UK_FIRM_PROFIT_POSITIVE_ROW",
    "UK_FIRM_PROFIT_SEED",
    "UK_FIRM_PROFIT_SOURCE_COLUMNS",
    "UK_FIRM_PROFIT_STATUS",
    "UK_FIRM_PROFIT_ZERO_BAND",
    "UKCorporationTaxSchedule",
    "UKFirmProfitBand",
    "UKFirmProfitSource",
    "assign_uk_firm_taxable_profit",
    "read_uk_firm_profit_source",
    "uk_corporation_tax_liability",
    "uk_corporation_tax_profit",
    "uk_firm_profit_bands",
    "uk_firm_profit_diagnostics",
    "uk_firm_profit_section_liability",
]
