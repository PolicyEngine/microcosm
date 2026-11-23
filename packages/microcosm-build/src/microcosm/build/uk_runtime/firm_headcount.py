"""UK firm headcount: working proprietors, employees and a small-firm redraw.

The experimental UK firm generator draws ``employment`` from the ONS *UK
business: activity, size and location* employment bands. ONS employment is
employees plus working proprietors, and inside the ONS "0-4" band the
generator spreads headcount evenly over 1 to 4 whatever the legal status. This
module adds three opt-in pieces, none of which runs by default:

- ``uk_firm_assumed_proprietors`` and ``uk_firm_employees`` split employment
  into working proprietors and employees by ONS legal status: a sole
  proprietorship (status 2) has one working proprietor, an ordinary
  partnership (status 3) two (``min(2, employment)``), a company (status 1)
  none, because the DBT *Business population estimates* (BPE) count a
  company's working directors as PAYE employees; public corporations, central
  and local government and non-profit bodies (statuses 4 to 7) none. These are
  assumptions: a partnership may have more than two partners and a sole
  trader's spouse may work unpaid. ``uk_firm_employee_size_bands`` labels
  employees with the BPE employee size bands (``UK_FIRM_EMPLOYEE_SIZE_BANDS``).
- ``redraw_uk_firm_small_headcount`` redraws ``employment`` for company, sole
  proprietorship and partnership rows in the ONS 0-4 band (employment 1 to 4)
  from P(employment = k | legal status, 0-4 band) built from BPE 2025 Table 3
  (UK private sector by legal status and employee size band, start 2025).
  Rows of any other status or band are left alone, and no row leaves its
  band.
- ``uk_firm_small_headcount_diagnostics`` and
  ``uk_firm_bpe_size_band_reconciliation`` report what the redraw did and how
  modelled firms, employment and employees by BPE employee band compare with
  BPE Table 2 (private sector and whole economy).

How the target is built (``uk_firm_small_headcount_targets``). Each BPE Table
3 row of *registered* businesses ("With no employees (registered)", "1",
"2 to 4"; the unregistered row is excluded because the ONS frame holds VAT- or
PAYE-registered enterprises only) is turned into an employment support by the
proprietor assumption:

- companies: the no-employee row is employment 1 (BPE notes that a company
  whose one PAYE employee is its working director is counted with no
  employees); "2 to 4" is employment 2 to 4; the "1" row is empty by that
  same reclassification;
- sole proprietorships: employees plus one, so no-employee is 1, "1" is 2 and
  "2 to 4" is 3 to 5;
- partnerships: employees plus two, so no-employee is 2, "1" is 3 and
  "2 to 4" is 4 to 6. A partnership never has employment 1 under the
  two-partner assumption, so its k = 1 share is zero.

A row whose support has several values is spread over it by a truncated
geometric distribution, p(j) proportional to ``ratio ** (j - low)``, whose
mean equals the row's published mean employment (employment thousands times
1,000 over business number; BPE employment already counts working
proprietors, so the proprietor assumption shifts the support, not the mean).
The ratio is solved numerically and may exceed one. A mean outside the open
support is clipped just inside it and the row is flagged ``clipped``. The
fitted distribution is then cut to the ONS band (employment 1 to 4) and only
the in-band part of the row's business count is kept: a sole trader with four
employees, or a partnership with three or four, has employment 5 or 6 and
belongs to the ONS 5-9 band, so that share (``in_band_share`` below one in the
fit table) is dropped and disclosed rather than piled onto k = 4.

Caveats a caller should know before opting in:

- the BPE Table 3 split of small firms by legal status becomes an in-sample
  fact: a later comparison of the redrawn frame with that table is not an
  independent check;
- BPE describes the business population at the start of 2025; the ONS band
  counts are the generator's own vintage, so the two can disagree on the
  number of firms in a band. The redraw keeps every row's band and weight and
  changes only the shape inside the 0-4 band;
- the draw ignores industry and turnover, and the paybill, employees and
  any column computed from ``employment`` before the redraw are stale.

Every number comes from the fixture
(``tests/fixtures/uk/firm_bpe_2025/``, re-created by
``tools/extract_uk_firm_bpe_tables.py``); none is written here.
"""

from __future__ import annotations

import hashlib
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

#: The extracted BPE Table 3 CSV in a BPE source directory.
UK_FIRM_BPE_LEGAL_STATUS_FILE = "bpe_legal_status_bands.csv"
#: The extracted BPE Tables 1 and 2 CSV in a BPE source directory.
UK_FIRM_BPE_SIZE_BANDS_FILE = "bpe_size_bands.csv"
UK_FIRM_BPE_LEGAL_STATUS_COLUMNS = (
    "legal_status",
    "size_band",
    "business_number",
    "business_number_flag",
    "employment_thousand",
    "employment_flag",
)
UK_FIRM_BPE_SIZE_BANDS_COLUMNS = (
    "table",
    "sector",
    "size_band",
    "business_number",
    "employment_thousand",
    "employees_thousand",
    "working_proprietors_thousand",
    "turnover_million",
)
#: The flag BPE prints for a disclosive cell; the fixture leaves it empty.
UK_FIRM_BPE_SUPPRESSED = "[c]"

#: The receipt's status and method for the redraw.
UK_FIRM_HEADCOUNT_STATUS = "assumed"
UK_FIRM_HEADCOUNT_METHOD = (
    "employment of ONS legal status 1-3 rows in the ONS 0-4 band redrawn from "
    "P(employment | legal status, 0-4 band) built from DBT Business population "
    "estimates 2025 Table 3 registered rows, mapped to employment by the "
    "proprietor assumption (company 0, sole proprietor 1, partnership 2 "
    "working proprietors) with a mean-matched truncated geometric inside "
    "multi-value rows, cut to the band"
)

#: BPE employee size bands, 0 and 1 kept apart, in order.
UK_FIRM_EMPLOYEE_SIZE_BANDS: tuple[str, ...] = (
    "0",
    "1",
    "2-4",
    "5-9",
    "10-19",
    "20-49",
    "50-99",
    "100-199",
    "200-249",
    "250-499",
    "500-999",
    "1000+",
)
#: The lowest employee count of each band in ``UK_FIRM_EMPLOYEE_SIZE_BANDS``.
_EMPLOYEE_BAND_LOWER_EDGES = np.asarray(
    [int(re.match(r"\d+", band).group()) for band in UK_FIRM_EMPLOYEE_SIZE_BANDS],
    dtype=np.int64,
)

#: ONS legal status -> BPE Table 3 section the redraw reads.
UK_FIRM_SMALL_HEADCOUNT_STATUSES: Mapping[int, str] = MappingProxyType(
    {1: "companies", 2: "sole_proprietorships", 3: "ordinary_partnerships"}
)
#: Working proprietors assumed per ONS legal status (capped at employment).
UK_FIRM_ASSUMED_PROPRIETORS: Mapping[int, int] = MappingProxyType(
    {1: 0, 2: 1, 3: 2, 4: 0, 5: 0, 6: 0, 7: 0}
)
#: The ONS "0-4" employment band as the generator draws it (employment 1-4).
UK_FIRM_ONS_SMALL_BAND: tuple[int, int] = (1, 4)
#: BPE Table 3 registered rows the targets read, with the employee range.
_BPE_NO_EMPLOYEES_REGISTERED = "With no employees (registered)"
_BPE_SMALL_ROWS: tuple[tuple[str, int, int], ...] = (
    (_BPE_NO_EMPLOYEES_REGISTERED, 0, 0),
    ("1", 1, 1),
    ("2 to 4", 2, 4),
)
_BPE_UNREGISTERED = "With no employees (unregistered)"
#: Distinguishes this redraw's random stream from any other use of ``seed``.
_SEED_TAG = zlib.crc32(b"microcosm.uk_firm_headcount.small_redraw")
#: Status codes the BPE private sector covers (companies include public
#: corporations, BPE Table 3 heading).
_PRIVATE_SECTOR_STATUSES = (1, 2, 3, 4)


@dataclass(frozen=True)
class UKFirmBPESource:
    """The extracted BPE 2025 tables.

    ``legal_status`` is Table 3 and ``size_bands`` Tables 1 and 2, as read
    from the CSVs, numbers as floats (NaN where suppressed); ``sha256`` maps
    each file name to its hash.
    """

    legal_status: pd.DataFrame
    size_bands: pd.DataFrame
    sha256: Mapping[str, str]


def _number_column(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    text = frame[column].str.strip()
    values = pd.to_numeric(text.replace("", np.nan), errors="coerce")
    bad = text.ne("") & values.isna()
    if bad.any():
        raise ValueError(f"UK firm BPE {name} {column} has non-numbers.")
    if (values.dropna() < 0).any():
        raise ValueError(f"UK firm BPE {name} {column} has negative values.")
    return values.astype(np.float64)


def read_uk_firm_bpe_source(path: str | Path) -> UKFirmBPESource:
    """Read the BPE 2025 CSVs from their directory.

    Raises ``ValueError`` unless each file has exactly its columns, every
    number parses, and in Table 3 a value is empty exactly when its flag is
    ``[c]``.
    """

    directory = Path(path)
    frames = {}
    hashes = {}
    for name, columns in (
        (UK_FIRM_BPE_LEGAL_STATUS_FILE, UK_FIRM_BPE_LEGAL_STATUS_COLUMNS),
        (UK_FIRM_BPE_SIZE_BANDS_FILE, UK_FIRM_BPE_SIZE_BANDS_COLUMNS),
    ):
        file = directory / name
        payload = file.read_bytes()
        frame = pd.read_csv(file, dtype=str, keep_default_na=False)
        if tuple(frame.columns) != columns:
            raise ValueError(
                f"UK firm BPE {name} columns {list(frame.columns)} are not "
                f"{list(columns)}."
            )
        frames[name] = frame
        hashes[name] = hashlib.sha256(payload).hexdigest()
    legal = frames[UK_FIRM_BPE_LEGAL_STATUS_FILE]
    for value, flag in (
        ("business_number", "business_number_flag"),
        ("employment_thousand", "employment_flag"),
    ):
        flags = legal[flag].str.strip()
        if not flags.isin(["", UK_FIRM_BPE_SUPPRESSED]).all():
            raise ValueError(f"UK firm BPE Table 3 {flag} must be empty or [c].")
        if not (
            legal[value].str.strip().eq("") == flags.eq(UK_FIRM_BPE_SUPPRESSED)
        ).all():
            raise ValueError(
                f"UK firm BPE Table 3 {value} must be empty exactly where "
                f"{flag} is [c]."
            )
        legal[value] = _number_column(legal, value, UK_FIRM_BPE_LEGAL_STATUS_FILE)
    size = frames[UK_FIRM_BPE_SIZE_BANDS_FILE]
    for column in UK_FIRM_BPE_SIZE_BANDS_COLUMNS[3:]:
        size[column] = _number_column(size, column, UK_FIRM_BPE_SIZE_BANDS_FILE)
    return UKFirmBPESource(
        legal_status=legal,
        size_bands=size,
        sha256=MappingProxyType(hashes),
    )


def _table_3(source: UKFirmBPESource, section: str, band: str) -> tuple[float, float]:
    """``(business number, employment in people)`` of one Table 3 row."""

    frame = source.legal_status
    rows = frame[(frame["legal_status"] == section) & (frame["size_band"] == band)]
    if len(rows) != 1:
        raise ValueError(f"UK firm BPE Table 3 {section} {band!r} found {len(rows)}x.")
    businesses = float(rows["business_number"].iloc[0])
    employment = float(rows["employment_thousand"].iloc[0]) * 1000.0
    if not (np.isfinite(businesses) and np.isfinite(employment)):
        raise ValueError(f"UK firm BPE Table 3 {section} {band!r} is suppressed.")
    return businesses, employment


# -- proprietors and employees -------------------------------------------------


def _status_array(legal_status: Any) -> np.ndarray:
    status = np.asarray(legal_status)
    if status.dtype.kind not in "iu":
        if status.dtype.kind == "f" and np.all(status == np.round(status)):
            status = status.astype(np.int64)
        else:
            raise ValueError("UK firm ons_legal_status must be integers.")
    status = status.astype(np.int64)
    if not np.isin(status, list(UK_FIRM_ASSUMED_PROPRIETORS)).all():
        raise ValueError(
            "UK firm ons_legal_status must be ONS legal status codes "
            f"{sorted(UK_FIRM_ASSUMED_PROPRIETORS)}."
        )
    return status


def _employment_array(employment: Any) -> np.ndarray:
    values = np.asarray(employment, dtype=np.float64)
    if (
        not np.all(np.isfinite(values))
        or np.any(values != np.round(values))
        or np.any(values < 0)
    ):
        raise ValueError("UK firm employment must be non-negative whole numbers.")
    return values.astype(np.int64)


def uk_firm_assumed_proprietors(legal_status: Any, employment: Any) -> np.ndarray:
    """Working proprietors per firm (int64) under the legal status assumption.

    Sole proprietorship (ONS status 2) one, ordinary partnership (3) two,
    company (1) and statuses 4 to 7 none; never more than ``employment``.
    """

    status = _status_array(legal_status)
    employment = _employment_array(employment)
    if status.shape != employment.shape:
        raise ValueError("legal_status and employment differ in shape.")
    per_status = np.zeros(max(UK_FIRM_ASSUMED_PROPRIETORS) + 1, dtype=np.int64)
    for code, owners in UK_FIRM_ASSUMED_PROPRIETORS.items():
        per_status[code] = owners
    return np.minimum(per_status[status], employment)


def uk_firm_employees(legal_status: Any, employment: Any) -> np.ndarray:
    """Employees per firm (int64): employment less assumed proprietors, >= 0."""

    employment_values = _employment_array(employment)
    proprietors = uk_firm_assumed_proprietors(legal_status, employment_values)
    return np.maximum(employment_values - proprietors, 0)


def uk_firm_employee_size_bands(employees: Any) -> np.ndarray:
    """BPE employee size band label of each employee count (str array)."""

    values = _employment_array(employees)
    index = np.searchsorted(_EMPLOYEE_BAND_LOWER_EDGES, values, side="right") - 1
    return np.asarray(UK_FIRM_EMPLOYEE_SIZE_BANDS, dtype=object)[index].astype(str)


# -- the target ----------------------------------------------------------------


def _geometric_pmf(low: int, high: int, ratio: float) -> np.ndarray:
    steps = np.arange(high - low + 1, dtype=np.float64)
    log = steps * np.log(ratio)
    weights = np.exp(log - log.max())
    return weights / weights.sum()


def fit_truncated_geometric(
    low: int, high: int, mean: float
) -> tuple[float, float, bool]:
    """``(ratio, fitted mean, clipped)`` of p(j) ~ ratio ** (j - low) on low..high.

    The ratio is found by bisection on its logarithm (the mean rises with
    it). A mean outside the open interval (low, high) is clipped to within
    1e-6 of the nearer end and ``clipped`` is True.
    """

    if high <= low:
        raise ValueError("a geometric fit needs at least two support values")
    edge = 1e-6
    clipped = not (low + edge <= mean <= high - edge)
    target = min(max(mean, low + edge), high - edge)
    support = np.arange(low, high + 1, dtype=np.float64)
    lower, upper = -60.0, 60.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if float(support @ _geometric_pmf(low, high, np.exp(middle))) < target:
            lower = middle
        else:
            upper = middle
    ratio = float(np.exp(0.5 * (lower + upper)))
    return ratio, float(support @ _geometric_pmf(low, high, ratio)), clipped


def _support(status: int, employees_low: int, employees_high: int) -> tuple[int, int]:
    """Employment range of a BPE employee range under the proprietor rule."""

    if status == 1 and employees_high == 0:
        # BPE counts a company whose one PAYE employee is its working
        # director with no employees: employment 1.
        return 1, 1
    shift = UK_FIRM_ASSUMED_PROPRIETORS[status]
    return employees_low + shift, employees_high + shift


def _fits(source: UKFirmBPESource) -> pd.DataFrame:
    band_low, band_high = UK_FIRM_ONS_SMALL_BAND
    rows = []
    for status, section in UK_FIRM_SMALL_HEADCOUNT_STATUSES.items():
        for band, employees_low, employees_high in _BPE_SMALL_ROWS:
            rows.append(
                _fit_row(source, status, section, band, employees_low, employees_high)
            )
    return pd.DataFrame(rows)


def uk_firm_small_headcount_fits(source: UKFirmBPESource) -> pd.DataFrame:
    """One row per (status, BPE registered row): its support, fit and mass.

    Columns: ``ons_legal_status``, ``bpe_legal_status``, ``bpe_size_band``,
    ``businesses``, ``employment``, ``mean_employment`` (published),
    ``support_low``/``support_high`` (employment), ``ratio`` (NaN for one
    value), ``fitted_mean``, ``clipped``, ``in_band_share`` and
    ``in_band_businesses``.
    """

    return _fits(source).drop(columns="_pmf")


def _fit_row(
    source: UKFirmBPESource,
    status: int,
    section: str,
    band: str,
    employees_low: int,
    employees_high: int,
) -> dict[str, Any]:
    band_low, band_high = UK_FIRM_ONS_SMALL_BAND
    businesses, employment = _table_3(source, section, band)
    low, high = _support(status, employees_low, employees_high)
    mean = employment / businesses if businesses > 0 else np.nan
    if high > low and businesses > 0:
        ratio, fitted, clipped = fit_truncated_geometric(low, high, mean)
        pmf = _geometric_pmf(low, high, ratio)
    else:
        ratio, fitted, clipped = np.nan, float(low), False
        pmf = np.ones(high - low + 1) / (high - low + 1)
    values = np.arange(low, high + 1)
    inside = (values >= band_low) & (values <= band_high)
    share = float(pmf[inside].sum())
    return {
        "ons_legal_status": status,
        "bpe_legal_status": section,
        "bpe_size_band": band,
        "businesses": businesses,
        "employment": employment,
        "mean_employment": mean,
        "support_low": low,
        "support_high": high,
        "ratio": ratio,
        "fitted_mean": fitted,
        "clipped": clipped,
        "in_band_share": share,
        "in_band_businesses": businesses * share,
        "_pmf": dict(zip(values.tolist(), (businesses * pmf).tolist(), strict=True)),
    }


def uk_firm_small_headcount_targets(source: UKFirmBPESource) -> pd.DataFrame:
    """P(employment = k | ONS legal status, 0-4 band) from BPE Table 3.

    One row per status (1, 2, 3) and k (1 to 4): ``ons_legal_status``,
    ``employment``, ``businesses`` (BPE businesses assigned to k) and
    ``share`` (sums to one within a status).
    """

    fits = _fits(source)
    band_low, band_high = UK_FIRM_ONS_SMALL_BAND
    rows = []
    for status in UK_FIRM_SMALL_HEADCOUNT_STATUSES:
        mass = dict.fromkeys(range(band_low, band_high + 1), 0.0)
        for pmf in fits.loc[fits["ons_legal_status"] == status, "_pmf"]:
            for value, businesses in pmf.items():
                if value in mass:
                    mass[value] += businesses
        total = sum(mass.values())
        if not total > 0:
            raise ValueError(f"BPE gives no businesses in band for status {status}.")
        rows.extend(
            {
                "ons_legal_status": status,
                "employment": value,
                "businesses": businesses,
                "share": businesses / total,
            }
            for value, businesses in mass.items()
        )
    return pd.DataFrame(rows)


# -- the redraw ----------------------------------------------------------------


def _firm_ids(firms: pd.DataFrame, id_column: str | None) -> np.ndarray:
    if id_column is None:
        id_column = next(
            (c for c in ("source_firm_id", "firm_id") if c in firms.columns), None
        )
    ids = firms.index.to_numpy() if id_column is None else firms[id_column].to_numpy()
    if np.asarray(ids).dtype.kind not in "iu":
        raise ValueError("UK firm ids must be integers.")
    ordered = np.sort(np.asarray(ids, dtype=np.int64))
    if ordered.size and np.any(np.diff(ordered) == 0):
        raise ValueError("UK firm ids repeat.")
    return np.asarray(ids, dtype=np.int64)


def _redrawn(
    firms: pd.DataFrame,
    source: UKFirmBPESource,
    seed: int,
    id_column: str | None,
    employment_column: str,
) -> np.ndarray:
    for column in ("ons_legal_status", employment_column):
        if column not in firms.columns:
            raise ValueError(f"UK firm headcount redraw needs column {column!r}.")
    status = _status_array(firms["ons_legal_status"].to_numpy())
    employment = _employment_array(firms[employment_column].to_numpy())
    ids = _firm_ids(firms, id_column)
    band_low, band_high = UK_FIRM_ONS_SMALL_BAND
    eligible = (
        np.isin(status, list(UK_FIRM_SMALL_HEADCOUNT_STATUSES))
        & (employment >= band_low)
        & (employment <= band_high)
    )
    after = employment.copy()
    rows = np.flatnonzero(eligible)
    rows = rows[np.argsort(ids[rows], kind="stable")]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), _SEED_TAG]))
    uniform = rng.random(rows.size)
    targets = uk_firm_small_headcount_targets(source)
    for code in UK_FIRM_SMALL_HEADCOUNT_STATUSES:
        target = targets[targets["ons_legal_status"] == code].sort_values("employment")
        cdf = np.cumsum(target["share"].to_numpy())
        cdf[-1] = 1.0
        values = target["employment"].to_numpy(np.int64)
        mine = status[rows] == code
        pick = np.searchsorted(cdf, uniform[mine], side="right")
        after[rows[mine]] = values[np.minimum(pick, values.size - 1)]
    redrawn = after[rows]
    if np.any((redrawn < band_low) | (redrawn > band_high)):
        raise AssertionError("UK firm headcount redraw left the 0-4 band")
    return after


def redraw_uk_firm_small_headcount(
    firms: pd.DataFrame,
    source: UKFirmBPESource,
    *,
    seed: int,
    id_column: str | None = None,
    employment_column: str = "employment",
) -> pd.DataFrame:
    """A copy of ``firms`` with small-firm employment redrawn from BPE.

    Rows with ``ons_legal_status`` 1, 2 or 3 and employment 1 to 4 draw a
    new employment from ``uk_firm_small_headcount_targets``; every other row,
    the index, the order and every other column are unchanged. Draws come from
    ``np.random.default_rng(SeedSequence([seed, tag]))`` in order of firm id
    (``id_column``; by default ``source_firm_id``, else ``firm_id``, else the
    index), so the result does not depend on row order.
    """

    after = _redrawn(firms, source, seed, id_column, employment_column)
    result = firms.copy()
    result[employment_column] = after.astype(firms[employment_column].dtype)
    return result


# -- diagnostics ---------------------------------------------------------------


def uk_firm_small_headcount_diagnostics(
    before: pd.DataFrame,
    after: pd.DataFrame,
    source: UKFirmBPESource,
    *,
    weight_col: str = "firm_weight",
    employment_column: str = "employment",
) -> pd.DataFrame:
    """Weighted counts by status (1-3) and employment 1-4, before and after.

    ``before`` and ``after`` are the frames either side of the redraw, rows
    aligned. Columns: ``ons_legal_status``, ``employment``,
    ``weight_before``, ``weight_after``, ``share_before``, ``share_after``
    (within status and band) and ``target_share``.
    """

    if len(before) != len(after):
        raise ValueError("before and after differ in length.")
    weight = before[weight_col].to_numpy(dtype=np.float64)
    status = _status_array(before["ons_legal_status"].to_numpy())
    targets = uk_firm_small_headcount_targets(source)
    rows = []
    for label, frame in (("before", before), ("after", after)):
        employment = _employment_array(frame[employment_column].to_numpy())
        for _, target in targets.iterrows():
            code, value = int(target["ons_legal_status"]), int(target["employment"])
            rows.append(
                {
                    "ons_legal_status": code,
                    "employment": value,
                    "when": label,
                    "weight": float(
                        weight[(status == code) & (employment == value)].sum()
                    ),
                }
            )
    table = (
        pd.DataFrame(rows)
        .pivot_table(
            index=["ons_legal_status", "employment"], columns="when", values="weight"
        )
        .reset_index()
        .rename(columns={"before": "weight_before", "after": "weight_after"})
    )
    table.columns.name = None
    for when in ("before", "after"):
        totals = table.groupby("ons_legal_status")[f"weight_{when}"].transform("sum")
        table[f"share_{when}"] = np.where(
            totals > 0, table[f"weight_{when}"] / totals.where(totals > 0, 1.0), 0.0
        )
    table = table.merge(
        targets[["ons_legal_status", "employment", "share"]].rename(
            columns={"share": "target_share"}
        ),
        on=["ons_legal_status", "employment"],
    )
    return table[
        [
            "ons_legal_status",
            "employment",
            "weight_before",
            "weight_after",
            "share_before",
            "share_after",
            "target_share",
        ]
    ]


def _lower_edge(label: str) -> int:
    if label.startswith("Zero"):
        return 0
    match = re.match(r"\d+", label)
    if match is None:
        raise ValueError(f"BPE band {label!r} has no lower edge.")
    return int(match.group())


def uk_firm_bpe_size_band_reconciliation(
    firms: pd.DataFrame,
    source: UKFirmBPESource,
    weight_col: str = "firm_weight",
    *,
    employment_column: str = "employment",
) -> pd.DataFrame:
    """Modelled firms, employment and employees by BPE band against BPE.

    Bands are BPE Table 2's ("Zero or 1 employee", "2 to 4", ...,
    "1000 or more", and "All"), on employees from ``uk_firm_employees``.
    ``private_sector`` compares statuses 1 to 4 with Table 2's private
    sector; ``whole_economy`` compares every row with Table 2's whole
    economy. BPE counts unregistered businesses the ONS frame does not hold;
    ``bpe_registered_businesses`` and ``bpe_registered_employment`` take
    Table 1's unregistered row out of the "Zero or 1 employee" and "All"
    bands (every unregistered business is private), and the ratios divide by
    them. BPE counts people, start 2025; the model counts weighted firms.
    """

    status = _status_array(firms["ons_legal_status"].to_numpy())
    employment = _employment_array(firms[employment_column].to_numpy())
    weight = firms[weight_col].to_numpy(dtype=np.float64)
    employees = uk_firm_employees(status, employment)
    table_2 = source.size_bands[source.size_bands["table"] == "Table 2"]
    table_1 = source.size_bands[source.size_bands["table"] == "Table 1"]
    unregistered = table_1[table_1["size_band"] == _BPE_UNREGISTERED]
    if len(unregistered) != 1:
        raise ValueError("BPE Table 1 lacks the unregistered row.")
    unregistered_businesses = float(unregistered["business_number"].iloc[0])
    unregistered_employment = float(unregistered["employment_thousand"].iloc[0]) * 1000
    rows = []
    for scope, keep in (
        ("private_sector", np.isin(status, _PRIVATE_SECTOR_STATUSES)),
        ("whole_economy", np.ones(status.size, dtype=bool)),
    ):
        published = table_2[table_2["sector"] == scope]
        bands = [band for band in published["size_band"] if band != "All"]
        edges = np.asarray([_lower_edge(band) for band in bands], dtype=np.int64)
        index = np.searchsorted(edges, employees, side="right") - 1
        for position, band in enumerate([*bands, "All"]):
            rows_in = keep & (index == position) if band != "All" else keep
            bpe = published[published["size_band"] == band].iloc[0]
            businesses = float(bpe["business_number"])
            employment_people = float(bpe["employment_thousand"]) * 1000
            first = band == "All" or position == 0
            registered_businesses = businesses - (
                unregistered_businesses if first else 0.0
            )
            registered_employment = employment_people - (
                unregistered_employment if first else 0.0
            )
            modelled_businesses = float(weight[rows_in].sum())
            modelled_employment = float(weight[rows_in] @ employment[rows_in])
            rows.append(
                {
                    "scope": scope,
                    "size_band": band,
                    "modelled_businesses": modelled_businesses,
                    "modelled_employment": modelled_employment,
                    "modelled_employees": float(weight[rows_in] @ employees[rows_in]),
                    "bpe_businesses": businesses,
                    "bpe_employment": employment_people,
                    "bpe_employees": float(bpe["employees_thousand"]) * 1000,
                    "bpe_registered_businesses": registered_businesses,
                    "bpe_registered_employment": registered_employment,
                    "businesses_ratio": modelled_businesses / registered_businesses
                    if registered_businesses > 0
                    else np.nan,
                    "employment_ratio": modelled_employment / registered_employment
                    if registered_employment > 0
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "UK_FIRM_ASSUMED_PROPRIETORS",
    "UK_FIRM_BPE_LEGAL_STATUS_COLUMNS",
    "UK_FIRM_BPE_LEGAL_STATUS_FILE",
    "UK_FIRM_BPE_SIZE_BANDS_COLUMNS",
    "UK_FIRM_BPE_SIZE_BANDS_FILE",
    "UK_FIRM_BPE_SUPPRESSED",
    "UK_FIRM_EMPLOYEE_SIZE_BANDS",
    "UK_FIRM_HEADCOUNT_METHOD",
    "UK_FIRM_HEADCOUNT_STATUS",
    "UK_FIRM_ONS_SMALL_BAND",
    "UK_FIRM_SMALL_HEADCOUNT_STATUSES",
    "UKFirmBPESource",
    "fit_truncated_geometric",
    "read_uk_firm_bpe_source",
    "redraw_uk_firm_small_headcount",
    "uk_firm_assumed_proprietors",
    "uk_firm_bpe_size_band_reconciliation",
    "uk_firm_employee_size_bands",
    "uk_firm_employees",
    "uk_firm_small_headcount_diagnostics",
    "uk_firm_small_headcount_fits",
    "uk_firm_small_headcount_targets",
]
