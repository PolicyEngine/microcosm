"""UK firm public sector employment: an opt-in rescale to ONS headcounts.

The experimental UK firm generator draws ``ons_legal_status`` from the ONS
*UK business: activity, size and location* tables, which count VAT- and
PAYE-registered enterprises. Central government (status 5) and local
authority (status 6) rows are therefore few and small: on the full 2024-25
build they carry about 0.66 million weighted employees (0.47 million and 0.19
million), where ONS counts about six million in central and local government. Employer NIC and the apprenticeship levy,
which follow the paybill, come out low for public bodies.

``scale_uk_firm_public_sector_employment`` multiplies the headcount of the
existing status 5 and status 6 rows so that their weighted employment equals
the ONS *Public sector employment, UK: March 2025* Table 1 headcounts for
central government (series G6NQ) and local government (series G6NT). The
extracted CSV and its provenance sit in
``tests/fixtures/uk/firm_public_sector_2025/``
(``tools/extract_uk_firm_public_sector_table.py`` re-creates it).

Nothing calls this by default. It is an assumption, not an estimate, and it
has costs a caller should know before opting in:

- no row is added and no weight moves: the same few bodies each grow by one
  factor per status, so their headcounts leave the generator's size bands
  (a status 5 row can reach tens of thousands of employees) while turnover
  and every other column stay as drawn;
- the ONS public sector total covers the National Health Service,
  education, police and HM Forces (the workbook's Table 2), whose staff the
  generator may already hold under other statuses (companies, non-profit
  bodies) through the business counts; those rows are not reduced, so total
  weighted employment rises by about the amount added and passes the ASHE
  jobs count;
- public corporations (status 4) are left alone;
- the headcount is the seasonally adjusted March 2025 quarter, the last
  quarter of tax year 2024-25, not a tax-year average;
- a paybill computed before the rescale is stale: assign the paybill after.

The rescale is to a published employment count, never to a tax receipt.
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

#: The extracted ONS table's file name in a public sector source directory.
UK_FIRM_PUBLIC_SECTOR_FILE = "ons_public_sector_employment.csv"
#: Its columns, in order.
UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS = (
    "sector",
    "ons_legal_status",
    "series_id",
    "period",
    "headcount_thousand",
)
#: ONS legal status of the rows each sector's headcount is spread over.
UK_FIRM_PUBLIC_SECTOR_STATUSES = MappingProxyType(
    {"central_government": 5, "local_government": 6}
)
#: ONS series identifiers of the two Table 1 columns.
UK_FIRM_PUBLIC_SECTOR_SERIES = MappingProxyType(
    {"central_government": "G6NQ", "local_government": "G6NT"}
)
#: The quarter the headcounts describe and the build vintage it stands for.
UK_FIRM_PUBLIC_SECTOR_PERIOD = "Mar 2025"
UK_FIRM_PUBLIC_SECTOR_TAX_YEAR = "2024-25"
#: The receipt's status and method for the rescale.
UK_FIRM_PUBLIC_SECTOR_STATUS = "assumed"
UK_FIRM_PUBLIC_SECTOR_METHOD = (
    "employment of ONS legal status 5 and 6 rows scaled so weighted employment "
    "equals ONS Public sector employment, UK: March 2025 Table 1 headcount "
    "(central government G6NQ, local government G6NT)"
)

_REQUIRED_COLUMNS = ("firm_weight", "ons_legal_status")


@dataclass(frozen=True)
class UKFirmPublicSectorSource:
    """ONS public sector headcount (people) per ONS legal status.

    ``headcount`` maps ONS legal status (5, 6) to the published headcount in
    people; ``series`` maps the same statuses to the ONS series identifier.
    ``sha256`` is the CSV's hash.
    """

    headcount: Mapping[int, float]
    series: Mapping[int, str]
    period: str
    file: str
    sha256: str


def read_uk_firm_public_sector_source(path: str | Path) -> UKFirmPublicSectorSource:
    """Read the ONS public sector headcounts from a directory or the CSV.

    Raises ``ValueError`` unless the file has exactly
    ``UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS``, one row per sector in
    ``UK_FIRM_PUBLIC_SECTOR_STATUSES`` with that sector's legal status and
    series identifier, the ``UK_FIRM_PUBLIC_SECTOR_PERIOD`` quarter and a
    positive finite headcount.
    """

    path = Path(path)
    if path.is_dir():
        path = path / UK_FIRM_PUBLIC_SECTOR_FILE
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS:
        raise ValueError(
            f"UK firm public sector source columns {list(frame.columns)} are not "
            f"{list(UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS)}."
        )
    sectors = frame["sector"].str.strip().tolist()
    if sorted(sectors) != sorted(UK_FIRM_PUBLIC_SECTOR_STATUSES):
        raise ValueError(
            f"UK firm public sector source rows {sectors} are not "
            f"{list(UK_FIRM_PUBLIC_SECTOR_STATUSES)}, once each."
        )
    headcount: dict[int, float] = {}
    series: dict[int, str] = {}
    for record in frame.itertuples(index=False):
        sector = record.sector.strip()
        status = UK_FIRM_PUBLIC_SECTOR_STATUSES[sector]
        if record.ons_legal_status.strip() != str(status):
            raise ValueError(
                f"UK firm public sector {sector} must have ons_legal_status "
                f"{status}, not {record.ons_legal_status!r}."
            )
        if record.series_id.strip() != UK_FIRM_PUBLIC_SECTOR_SERIES[sector]:
            raise ValueError(
                f"UK firm public sector {sector} must be series "
                f"{UK_FIRM_PUBLIC_SECTOR_SERIES[sector]}, not {record.series_id!r}."
            )
        if record.period.strip() != UK_FIRM_PUBLIC_SECTOR_PERIOD:
            raise ValueError(
                f"UK firm public sector {sector} period {record.period!r} is not "
                f"{UK_FIRM_PUBLIC_SECTOR_PERIOD!r}."
            )
        try:
            thousand = float(record.headcount_thousand)
        except ValueError:
            raise ValueError(
                f"UK firm public sector {sector} headcount "
                f"{record.headcount_thousand!r} is not a number."
            ) from None
        if not np.isfinite(thousand) or thousand <= 0:
            raise ValueError(
                f"UK firm public sector {sector} headcount must be positive."
            )
        headcount[status] = thousand * 1000.0
        series[status] = record.series_id.strip()
    return UKFirmPublicSectorSource(
        headcount=MappingProxyType(dict(sorted(headcount.items()))),
        series=MappingProxyType(dict(sorted(series.items()))),
        period=UK_FIRM_PUBLIC_SECTOR_PERIOD,
        file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _columns(
    firms: pd.DataFrame, employment_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Headcount, weight and legal status of ``firms``, checked."""

    missing = [
        column
        for column in (employment_column, *_REQUIRED_COLUMNS)
        if column not in firms.columns
    ]
    if missing:
        raise ValueError(f"UK firm public sector rescale needs columns {missing}.")
    employment = firms[employment_column].to_numpy(dtype=np.float64)
    weight = firms["firm_weight"].to_numpy(dtype=np.float64)
    status = firms["ons_legal_status"].to_numpy()
    if (
        not np.all(np.isfinite(employment))
        or np.any(employment != np.round(employment))
        or np.any(employment < 0)
    ):
        raise ValueError(
            f"UK firm {employment_column} must be non-negative whole numbers."
        )
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("UK firm firm_weight must be non-negative and finite.")
    return employment, weight, status


def _scaled_headcounts(
    employment: np.ndarray, weight: np.ndarray, target: float
) -> tuple[np.ndarray, float]:
    """Whole headcounts of one status whose weighted sum is nearest ``target``.

    Each row is scaled by ``target / sum(weight * employment)`` and floored,
    never below one. Rows are then rounded up in order of falling fractional
    part (row order breaks ties) for as long as that brings the weighted sum
    closer to ``target``, so the result is deterministic and the miss is at
    most half of one row's weight unless the floor of one binds.
    """

    before = float(np.dot(weight, employment))
    if not before > 0:
        raise ValueError("weighted employment is not positive")
    factor = target / before
    scaled = employment * factor
    floor = np.floor(scaled)
    base = np.maximum(floor, 1.0)
    fraction = np.where(floor >= 1.0, scaled - floor, 0.0)
    shortfall = target - float(np.dot(weight, base))
    order = np.argsort(-fraction, kind="stable")
    order = order[(fraction[order] > 0) & (weight[order] > 0)]
    if shortfall > 0 and order.size:
        added = np.concatenate([[0.0], np.cumsum(weight[order])])
        take = int(np.argmin(np.abs(shortfall - added)))
        base[order[:take]] += 1.0
    return base, factor


def _rescale(
    firms: pd.DataFrame, source: UKFirmPublicSectorSource, employment_column: str
) -> tuple[np.ndarray, dict[str, Any]]:
    employment, weight, status = _columns(firms, employment_column)
    after = employment.copy()
    sectors: dict[str, dict[str, Any]] = {}
    for sector, code in UK_FIRM_PUBLIC_SECTOR_STATUSES.items():
        if code not in source.headcount:
            raise ValueError(f"UK firm public sector source has no status {code}.")
        rows = status == code
        if not rows.any():
            raise ValueError(
                f"UK firm public sector rescale found no rows with "
                f"ons_legal_status {code} ({sector})."
            )
        target = float(source.headcount[code])
        try:
            after[rows], factor = _scaled_headcounts(
                employment[rows], weight[rows], target
            )
        except ValueError as error:
            raise ValueError(
                f"UK firm public sector rescale cannot scale ons_legal_status "
                f"{code} ({sector}): {error}."
            ) from None
        sectors[sector] = {
            "ons_legal_status": code,
            "series_id": source.series[code],
            "rows": int(rows.sum()),
            "weight": float(weight[rows].sum()),
            "target_weighted_employment": target,
            "weighted_employment_before": float(np.dot(weight[rows], employment[rows])),
            "weighted_employment_after": float(np.dot(weight[rows], after[rows])),
            "scale_factor": factor,
            "largest_headcount_before": int(employment[rows].max()),
            "largest_headcount_after": int(after[rows].max()),
        }
    diagnostics = {
        "status": UK_FIRM_PUBLIC_SECTOR_STATUS,
        "method": UK_FIRM_PUBLIC_SECTOR_METHOD,
        "period": source.period,
        "source_file": source.file,
        "source_sha256": source.sha256,
        "employment_column": employment_column,
        "rows": len(firms),
        "total_weighted_employment_before": float(np.dot(weight, employment)),
        "total_weighted_employment_after": float(np.dot(weight, after)),
        "sectors": sectors,
    }
    return after, diagnostics


def scale_uk_firm_public_sector_employment(
    firms: pd.DataFrame,
    source: UKFirmPublicSectorSource,
    *,
    employment_column: str = "employment",
) -> pd.DataFrame:
    """A copy of ``firms`` with public authority headcounts scaled to ONS.

    Rows with ``ons_legal_status`` 5 are scaled to ``source``'s central
    government headcount and rows with status 6 to its local government
    headcount, each by one factor, so that ``firm_weight`` times
    ``employment_column`` sums to the published total. Headcounts stay whole
    and at least one; the weighted total misses by at most half of one row's
    weight (see ``_scaled_headcounts``). Rows, their order, the index, the
    weights and every other column are unchanged, public corporations
    (status 4) included.

    Raises ``ValueError`` when a column is missing, a headcount is not a
    non-negative whole number, or a status has no rows or no weighted
    employment to scale.
    """

    after, _ = _rescale(firms, source, employment_column)
    result = firms.copy()
    result[employment_column] = after.astype(firms[employment_column].dtype)
    return result


def uk_firm_public_sector_diagnostics(
    firms: pd.DataFrame,
    source: UKFirmPublicSectorSource,
    *,
    employment_column: str = "employment",
) -> dict[str, Any]:
    """What ``scale_uk_firm_public_sector_employment`` does to ``firms``.

    ``firms`` is the frame before the rescale. The dict holds the method,
    the source file and hash, total weighted employment before and after,
    and per sector the legal status, series, rows, weight, target, weighted
    employment before and after, scale factor and largest headcount before
    and after.
    """

    _, diagnostics = _rescale(firms, source, employment_column)
    return diagnostics


__all__ = [
    "UK_FIRM_PUBLIC_SECTOR_FILE",
    "UK_FIRM_PUBLIC_SECTOR_METHOD",
    "UK_FIRM_PUBLIC_SECTOR_PERIOD",
    "UK_FIRM_PUBLIC_SECTOR_SERIES",
    "UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS",
    "UK_FIRM_PUBLIC_SECTOR_STATUS",
    "UK_FIRM_PUBLIC_SECTOR_STATUSES",
    "UK_FIRM_PUBLIC_SECTOR_TAX_YEAR",
    "UKFirmPublicSectorSource",
    "read_uk_firm_public_sector_source",
    "scale_uk_firm_public_sector_employment",
    "uk_firm_public_sector_diagnostics",
]
