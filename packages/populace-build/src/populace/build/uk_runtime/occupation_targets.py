"""UK occupation employment calibration targets from ONS ASHE and the APS.

Two source families are supported, each parsed from a tidy CSV produced from
the published release (the raw workbooks/API pulls live outside the repo):

* **ASHE Table 14** — employee-job counts by four-digit SOC2020 unit group,
  from the Annual Survey of Hours and Earnings occupation tables
  (Table 14.7a, "Annual pay - Gross"). Tidy columns: ``soc_code``,
  ``soc_title``, ``employment_jobs``, ``median_annual_pay``. Only the job
  counts become targets: a median is not a sum constraint, so
  ``median_annual_pay`` is carried for provenance/diagnostics only.
* **Annual Population Survey via Nomis** (dataset ``NM_17_1``) — employment
  by SOC2020 sub-major occupation group (table ``T09b``) and, separately,
  employment by age band (table ``T01``). The APS publishes no
  occupation-by-age cross-tabulation, so the two margins are declared as
  independent target families. Tidy columns: ``soc_code``, ``soc_title``,
  ``employment``, ``period``, ``geography`` (occupation) and ``age_band``,
  ``employment``, ``period``, ``geography`` (age).

Every builder returns an :class:`OccupationTargetBuild`: the compiled
:class:`~populace.calibrate.TargetSet` plus the rows that could not become
targets, each with a reason — bad rows are skipped and reported, never
silently dropped. Structural problems (missing columns, empty files) raise.

Measures are prepared indicator/count columns on the ``person`` entity:
``employee_jobs/occupation/<soc4>`` for ASHE job counts (a person may hold
more than one job, so the column is a job count, not a 0/1 flag),
``employment/occupation_submajor/<soc2>`` and ``employment/age/<band>`` for
the APS person counts.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from populace.calibrate import Target, TargetSet

__all__ = [
    "ASHE_TABLE14_SOURCE",
    "APS_NOMIS_SOURCE",
    "PACKAGED_APS_AGE_CSV",
    "PACKAGED_APS_OCCUPATION_CSV",
    "PACKAGED_ASHE_TABLE14_CSV",
    "OccupationTargetBuild",
    "SkippedTargetRow",
    "aps_age_band_employment_targets",
    "aps_occupation_employment_targets",
    "ashe_occupation_employment_targets",
    "packaged_occupation_csv_path",
]

#: Provenance for the ASHE Table 14 vintage the tidy CSV was parsed from.
ASHE_TABLE14_SOURCE = (
    "ONS Annual Survey of Hours and Earnings (ASHE) Table 14.7a "
    "'Annual pay - Gross', occupation by four-digit SOC2020, United Kingdom, "
    "2025 provisional edition, released 23 October 2025, ons.gov.uk"
)

#: Provenance for the APS pulls from the Nomis API (dataset NM_17_1).
APS_NOMIS_SOURCE = (
    "ONS Annual Population Survey via Nomis API dataset NM_17_1 (www.nomisweb.co.uk)"
)

#: The all-ages roll-up band published alongside the disjoint APS age bands.
APS_TOTAL_AGE_BAND = "16+"

#: Filenames of the tidy CSVs committed under ``populace/build/uk/data``.
PACKAGED_ASHE_TABLE14_CSV = "ashe_table14_2025_soc4.csv"
PACKAGED_APS_OCCUPATION_CSV = "nomis_aps_occupation_employment.csv"
PACKAGED_APS_AGE_CSV = "nomis_aps_age_employment.csv"


def packaged_occupation_csv_path(filename: str) -> Path:
    """Filesystem path of a tidy CSV committed with the package.

    Args:
        filename: One of the ``PACKAGED_*_CSV`` constants.

    Returns:
        The path of the packaged tidy CSV under ``populace/build/uk/data``.

    Raises:
        FileNotFoundError: If ``filename`` is not a packaged tidy CSV.
    """

    path = Path(str(files("populace.build.uk").joinpath("data", filename)))
    if not path.is_file():
        raise FileNotFoundError(
            f"No packaged tidy CSV {filename!r} under populace/build/uk/data."
        )
    return path


@dataclass(frozen=True)
class SkippedTargetRow:
    """One tidy-CSV row that could not become a calibration target.

    Attributes:
        row_number: 1-based data-row position in the CSV (header excluded).
        reason: Human-readable explanation naming the culprit field.
        row: The offending row, as read.
    """

    row_number: int
    reason: str
    row: Mapping[str, str]


@dataclass(frozen=True)
class OccupationTargetBuild:
    """A compiled occupation target set plus its skipped-row report.

    Attributes:
        target_set: The targets that compiled cleanly.
        skipped: Rows excluded from ``target_set``, each with a reason.
            Callers deciding whether a build is usable must consult this —
            an empty tuple means every row became a target.
    """

    target_set: TargetSet
    skipped: tuple[SkippedTargetRow, ...]


def ashe_occupation_employment_targets(
    csv_path: str | Path,
    *,
    period: int | str = 2025,
    source: str = ASHE_TABLE14_SOURCE,
) -> OccupationTargetBuild:
    """Employee-job count targets by four-digit SOC2020 from ASHE Table 14.

    Rows with a suppressed or non-numeric ``employment_jobs`` (ASHE marks
    suppressed cells ``x``; the tidy CSV leaves them blank) and rows whose
    ``soc_code`` is not a four-digit code are skipped and reported.

    Args:
        csv_path: Tidy CSV with columns ``soc_code``, ``soc_title``,
            ``employment_jobs``, ``median_annual_pay``.
        period: Period tag for the targets (the ASHE survey year).
        source: Provenance string recorded on every target.

    Returns:
        The compiled build: one ``person``-entity target per usable unit
        group, measured by the prepared job-count column
        ``employee_jobs/occupation/<soc_code>``.

    Raises:
        ValueError: If the CSV is missing required columns or has no rows.
    """

    rows = _read_rows(csv_path, required=("soc_code", "soc_title", "employment_jobs"))
    targets: list[Target] = []
    skipped: list[SkippedTargetRow] = []
    seen: dict[str, int] = {}
    for row_number, row in rows:
        soc_code = row["soc_code"].strip()
        if not (len(soc_code) == 4 and soc_code.isdigit()):
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"soc_code {soc_code!r} is not a four-digit SOC2020 "
                    "unit-group code",
                    row,
                )
            )
            continue
        if soc_code in seen:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"duplicate soc_code {soc_code!r} (first seen at data row "
                    f"{seen[soc_code]})",
                    row,
                )
            )
            continue
        jobs = _parse_count(row.get("employment_jobs", ""))
        if jobs is None:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    "employment_jobs "
                    f"{row.get('employment_jobs', '')!r} is missing or not a "
                    "non-negative number (ASHE suppression is blank in the "
                    "tidy CSV)",
                    row,
                )
            )
            continue
        seen[soc_code] = row_number
        targets.append(
            Target(
                name=f"ons/ashe14/occupation/{soc_code}/employment_jobs",
                entity="person",
                measure=f"employee_jobs/occupation/{soc_code}",
                value=jobs,
                period=period,
                source=f"{source}; unit group {soc_code} {row['soc_title']}",
            )
        )
    return _finish(csv_path, targets, skipped)


def aps_occupation_employment_targets(
    csv_path: str | Path,
    *,
    source: str = APS_NOMIS_SOURCE,
) -> OccupationTargetBuild:
    """Employment targets by SOC2020 sub-major group from the APS (Nomis).

    Args:
        csv_path: Tidy CSV with columns ``soc_code`` (two-digit sub-major
            group), ``soc_title``, ``employment``, ``period`` (e.g.
            ``"Jan 2025-Dec 2025"``); an optional ``geography`` column is
            appended to the provenance.
        source: Provenance prefix recorded on every target.

    Returns:
        The compiled build: one ``person``-entity target per usable
        sub-major group, measured by the prepared indicator column
        ``employment/occupation_submajor/<soc_code>``, with the row's APS
        period string as the target period.

    Raises:
        ValueError: If the CSV is missing required columns or has no rows.
    """

    rows = _read_rows(
        csv_path, required=("soc_code", "soc_title", "employment", "period")
    )
    targets: list[Target] = []
    skipped: list[SkippedTargetRow] = []
    seen: dict[str, int] = {}
    for row_number, row in rows:
        soc_code = row["soc_code"].strip()
        if not (len(soc_code) == 2 and soc_code.isdigit()):
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"soc_code {soc_code!r} is not a two-digit SOC2020 "
                    "sub-major group code",
                    row,
                )
            )
            continue
        if soc_code in seen:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"duplicate soc_code {soc_code!r} (first seen at data row "
                    f"{seen[soc_code]})",
                    row,
                )
            )
            continue
        employment = _parse_count(row.get("employment", ""))
        if employment is None:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"employment {row.get('employment', '')!r} is missing or "
                    "not a non-negative number",
                    row,
                )
            )
            continue
        period = row["period"].strip()
        if not period:
            skipped.append(SkippedTargetRow(row_number, "period is blank", row))
            continue
        seen[soc_code] = row_number
        targets.append(
            Target(
                name=f"ons/aps/occupation_submajor/{soc_code}/employment",
                entity="person",
                measure=f"employment/occupation_submajor/{soc_code}",
                value=employment,
                period=period,
                source=_aps_row_source(source, row, f"table T09b, {period}")
                + f"; sub-major group {soc_code} {row['soc_title']}",
            )
        )
    return _finish(csv_path, targets, skipped)


def aps_age_band_employment_targets(
    csv_path: str | Path,
    *,
    include_total: bool = False,
    source: str = APS_NOMIS_SOURCE,
) -> OccupationTargetBuild:
    """In-employment person-count targets by age band from the APS (Nomis).

    The APS publishes no occupation-by-age cross-tabulation, so this age
    margin complements :func:`aps_occupation_employment_targets` as an
    independent family. The published ``16+`` roll-up duplicates the sum of
    the disjoint bands; it is excluded by default and reported as skipped so
    the exclusion is visible.

    Args:
        csv_path: Tidy CSV with columns ``age_band`` (``16-19`` … ``65+``,
            plus the ``16+`` total), ``employment``, ``period``; an optional
            ``geography`` column is appended to the provenance.
        include_total: Keep the redundant ``16+`` roll-up as a target.
        source: Provenance prefix recorded on every target.

    Returns:
        The compiled build: one ``person``-entity target per usable band,
        measured by the prepared indicator column
        ``employment/age/<band>`` (band with ``-`` as ``_`` and ``+`` as
        ``plus``, e.g. ``employment/age/16_19``, ``employment/age/65plus``).

    Raises:
        ValueError: If the CSV is missing required columns or has no rows.
    """

    rows = _read_rows(csv_path, required=("age_band", "employment", "period"))
    targets: list[Target] = []
    skipped: list[SkippedTargetRow] = []
    seen: dict[str, int] = {}
    for row_number, row in rows:
        age_band = row["age_band"].strip()
        if not age_band:
            skipped.append(SkippedTargetRow(row_number, "age_band is blank", row))
            continue
        if age_band == APS_TOTAL_AGE_BAND and not include_total:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"age_band {APS_TOTAL_AGE_BAND!r} is the roll-up of the "
                    "disjoint bands; excluded by default (pass "
                    "include_total=True to keep it)",
                    row,
                )
            )
            continue
        if age_band in seen:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"duplicate age_band {age_band!r} (first seen at data row "
                    f"{seen[age_band]})",
                    row,
                )
            )
            continue
        employment = _parse_count(row.get("employment", ""))
        if employment is None:
            skipped.append(
                SkippedTargetRow(
                    row_number,
                    f"employment {row.get('employment', '')!r} is missing or "
                    "not a non-negative number",
                    row,
                )
            )
            continue
        period = row["period"].strip()
        if not period:
            skipped.append(SkippedTargetRow(row_number, "period is blank", row))
            continue
        seen[age_band] = row_number
        band_token = age_band.replace("-", "_").replace("+", "plus")
        targets.append(
            Target(
                name=f"ons/aps/employment/age/{band_token}",
                entity="person",
                measure=f"employment/age/{band_token}",
                value=employment,
                period=period,
                source=_aps_row_source(source, row, f"table T01, {period}")
                + f"; in employment, aged {age_band}",
            )
        )
    return _finish(csv_path, targets, skipped)


def _read_rows(
    csv_path: str | Path, *, required: Sequence[str]
) -> list[tuple[int, dict[str, str]]]:
    """Read a tidy CSV, enforcing the structural contract.

    Returns ``(data_row_number, row)`` pairs; raises on missing columns or an
    empty file — structural breakage is an error, not a skippable row.
    """

    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s) {missing}; "
                f"found {list(fieldnames)}."
            )
        rows = [
            (row_number, {key: (value or "") for key, value in row.items()})
            for row_number, row in enumerate(reader, start=1)
        ]
    if not rows:
        raise ValueError(f"{csv_path}: no data rows.")
    return rows


def _parse_count(text: str) -> float | None:
    """Parse a non-negative count; ``None`` marks the row skippable."""

    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _aps_row_source(source: str, row: Mapping[str, str], detail: str) -> str:
    geography = row.get("geography", "").strip()
    suffix = f", {geography}" if geography else ""
    return f"{source}; {detail}{suffix}"


def _finish(
    csv_path: str | Path,
    targets: list[Target],
    skipped: list[SkippedTargetRow],
) -> OccupationTargetBuild:
    if not targets:
        reasons = "; ".join(
            f"row {entry.row_number}: {entry.reason}" for entry in skipped[:5]
        )
        raise ValueError(
            f"{csv_path}: every row was skipped, no targets built. "
            f"First reasons: {reasons}"
        )
    return OccupationTargetBuild(target_set=TargetSet(targets), skipped=tuple(skipped))
