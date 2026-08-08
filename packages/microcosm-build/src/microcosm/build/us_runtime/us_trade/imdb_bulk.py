"""Census monthly bulk imports database (IMDB) ingest — the primary source.

Census publishes the complete *U.S. Imports of Merchandise* database every
month as a public, no-auth ZIP::

    https://www.census.gov/trade/downloads/{YEAR}/Merch/im_m/IMDB{YY}{MM}.ZIP

Each archive carries the full-detail fixed-width file ``IMP_DETL.txt`` —
HTS-10 × country × country-subcode × district of entry × district of
unlading × rate provision, with monthly customs value, dutiable value,
calculated duty, charges, CIF value, quantities, and air/vessel/
containerized transport splits — plus the publisher's own control-total
files (``imp_CTY.txt`` by country, ``IMP_COMM.txt`` by commodity,
``IMP_DE.txt`` by district of entry) and the record layouts under
``Documentation/*.lay``. Eighteen GETs replace thousands of per-chapter
API queries; the API ingest (:mod:`microcosm.build.us_runtime.us_trade.census_imports`)
remains as an independent cross-check leg only.

Field positions below are transcribed from the archives' own
``Documentation/IMP_DETL.lay`` / ``IMP_CTY.lay`` / ``IMP_COMM.lay`` /
``IMP_DE.lay`` (verified against IMDB2501.ZIP, retrieved 2026-08-05). The
same fixed-width geometry is parsed by Yale Budget Lab's tariff model
(``src/io/build_import_weights.R``), whose six parsed positions (hs10,
cty_code, year, month, con_val_mo, gen_val_mo) match these layouts exactly.

Semantics notes (all source-verified, none inferred):

- Rate-provision codes classify free vs dutiable status only
  (census.gov/foreign-trade/reference/rpcodes.html); they carry **no**
  informal/mail/postal marker, so the files cannot ground a postal or
  entry-type split.
- ``cty_subco`` is the Country SubCode: the special trade-agreement /
  preference-program claim under which the records entered ("0" = no
  program; e.g. "A" = GSP, "S" = USMCA — enumerated at
  census.gov/foreign-trade/reference/codes/csc.html). It is carried
  through to the detail artifact and aggregated over for the margins
  table.
- ``cards_mo`` is defined by the published file structure as "Number of
  Detailed Records, Current Month" — a count of detail records, **not**
  of entry summaries — and is therefore never used as an entry-count
  anchor.
- District of entry ``70`` is Schedule D "LOW-VALUED IMPORTS AND EXPORTS"
  (census.gov/foreign-trade/schedules/d/distcode.html): the publisher's
  own low-valued-shipments aggregation rides the district margins.

Every archive byte-stream is recorded in a retrieval manifest (URL, sha256,
size, retrieval timestamp). Reconciliation is exact-integer against the
publisher's own control totals; any difference fails the ingest.
"""

from __future__ import annotations

import hashlib
import time as time_module
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microcosm.build.us_runtime.us_trade.census_country_bridge import (
    CensusCountryBridge,
)

__all__ = [
    "IMDB_URL_TEMPLATE",
    "IMDB_DETAIL_MEMBER",
    "IMDB_MARGIN_MEASURES",
    "ImdbBulkAssembly",
    "ImdbMonth",
    "ImdbMonthSummary",
    "assemble_bulk_margins",
    "ensure_imdb_archive",
    "imdb_archive_name",
    "imdb_archive_url",
    "latest_available_imdb_month",
    "load_imdb_month",
    "summarize_imdb_month",
]

IMDB_URL_TEMPLATE = (
    "https://www.census.gov/trade/downloads/{year}/Merch/im_m/IMDB{yy}{mm}.ZIP"
)

#: Inner archive members (names as published; the country file's case
#: varies across vintages, so members are located case-insensitively).
IMDB_DETAIL_MEMBER = "IMP_DETL.txt"
_CONTROL_CTY_MEMBER = "IMP_CTY.txt"
_CONTROL_COMM_MEMBER = "IMP_COMM.txt"
_CONTROL_DE_MEMBER = "IMP_DE.txt"

#: ``Documentation/IMP_DETL.lay`` monthly-measure positions (1-indexed
#: inclusive start-end, converted to 0-indexed half-open slices below).
#: Year-to-date mirrors (columns 359-688) are derivable and not parsed.
_DETAIL_KEYS = (
    ("hts10", 1, 10),
    ("cty_code", 11, 14),
    ("cty_subco", 15, 16),
    ("dist_entry", 17, 18),
    ("dist_unlad", 19, 20),
    ("rate_prov", 21, 22),
    ("year", 23, 26),
    ("month", 27, 28),
)
_DETAIL_MEASURES = (
    ("cards_mo", 29, 43),
    ("con_qy1_mo", 44, 58),
    ("con_qy2_mo", 59, 73),
    ("con_val_mo", 74, 88),
    ("dut_val_mo", 89, 103),
    ("cal_dut_mo", 104, 118),
    ("con_cha_mo", 119, 133),
    ("con_cif_mo", 134, 148),
    ("gen_qy1_mo", 149, 163),
    ("gen_qy2_mo", 164, 178),
    ("gen_val_mo", 179, 193),
    ("gen_cha_mo", 194, 208),
    ("gen_cif_mo", 209, 223),
    ("air_val_mo", 224, 238),
    ("air_wgt_mo", 239, 253),
    ("air_cha_mo", 254, 268),
    ("ves_val_mo", 269, 283),
    ("ves_wgt_mo", 284, 298),
    ("ves_cha_mo", 299, 313),
    ("cnt_val_mo", 314, 328),
    ("cnt_wgt_mo", 329, 343),
    ("cnt_cha_mo", 344, 358),
)

#: ``IMP_CTY.lay`` (country control totals): every monthly measure the
#: country file publishes (all detail measures except unit quantities).
_CTY_FIELDS = (
    ("cty_code", 1, 4),
    ("year", 35, 38),
    ("month", 39, 40),
    ("cards_mo", 41, 55),
    ("con_val_mo", 56, 70),
    ("dut_val_mo", 71, 85),
    ("cal_dut_mo", 86, 100),
    ("con_cha_mo", 101, 115),
    ("con_cif_mo", 116, 130),
    ("gen_val_mo", 131, 145),
    ("gen_cha_mo", 146, 160),
    ("gen_cif_mo", 161, 175),
    ("air_val_mo", 176, 190),
    ("air_wgt_mo", 191, 205),
    ("air_cha_mo", 206, 220),
    ("ves_val_mo", 221, 235),
    ("ves_wgt_mo", 236, 250),
    ("ves_cha_mo", 251, 265),
    ("cnt_val_mo", 266, 280),
    ("cnt_wgt_mo", 281, 295),
    ("cnt_cha_mo", 296, 310),
)

#: ``IMP_COMM.lay`` (commodity control totals + units of quantity): every
#: monthly measure the detail carries.
_COMM_FIELDS = (
    ("hts10", 1, 10),
    ("unit_qy1", 61, 63),
    ("unit_qy2", 64, 66),
    ("year", 67, 70),
    ("month", 71, 72),
    ("cards_mo", 73, 87),
    ("con_qy1_mo", 88, 102),
    ("con_qy2_mo", 103, 117),
    ("con_val_mo", 118, 132),
    ("dut_val_mo", 133, 147),
    ("cal_dut_mo", 148, 162),
    ("con_cha_mo", 163, 177),
    ("con_cif_mo", 178, 192),
    ("gen_qy1_mo", 193, 207),
    ("gen_qy2_mo", 208, 222),
    ("gen_val_mo", 223, 237),
    ("gen_cha_mo", 238, 252),
    ("gen_cif_mo", 253, 267),
    ("air_val_mo", 268, 282),
    ("air_wgt_mo", 283, 297),
    ("air_cha_mo", 298, 312),
    ("ves_val_mo", 313, 327),
    ("ves_wgt_mo", 328, 342),
    ("ves_cha_mo", 343, 357),
    ("cnt_val_mo", 358, 372),
    ("cnt_wgt_mo", 373, 387),
    ("cnt_cha_mo", 388, 402),
)

#: ``IMP_DE.lay`` (district-of-entry control totals): every monthly
#: measure the district file publishes.
_DE_FIELDS = (
    ("dist_entry", 1, 2),
    ("dist_name", 3, 32),
    ("year", 33, 36),
    ("month", 37, 38),
    ("cards_mo", 39, 53),
    ("con_val_mo", 54, 68),
    ("dut_val_mo", 69, 83),
    ("cal_dut_mo", 84, 98),
    ("con_cha_mo", 99, 113),
    ("con_cif_mo", 114, 128),
    ("gen_val_mo", 129, 143),
    ("gen_cha_mo", 144, 158),
    ("gen_cif_mo", 159, 173),
    ("air_val_mo", 174, 188),
    ("air_wgt_mo", 189, 203),
    ("air_cha_mo", 204, 218),
    ("ves_val_mo", 219, 233),
    ("ves_wgt_mo", 234, 248),
    ("ves_cha_mo", 249, 263),
    ("cnt_val_mo", 264, 278),
    ("cnt_wgt_mo", 279, 293),
    ("cnt_cha_mo", 294, 308),
)

#: Every measure a control file carries is reconciled — a silent
#: corruption of any admitted measure must fail some gate.
_RECONCILE_CTY = tuple(name for name, _, _ in _CTY_FIELDS if name.endswith("_mo"))
_RECONCILE_COMM = tuple(name for name, _, _ in _COMM_FIELDS if name.endswith("_mo"))
_RECONCILE_DE = tuple(name for name, _, _ in _DE_FIELDS if name.endswith("_mo"))

#: Margin-table dollar/quantity measure columns emitted at the
#: HTS10 × country grain (the API-compatible core four first).
IMDB_MARGIN_MEASURES = (
    "con_val_mo",
    "gen_val_mo",
    "cal_dut_mo",
    "dut_val_mo",
    "con_qy1_mo",
    "con_qy2_mo",
    "gen_qy1_mo",
    "gen_qy2_mo",
    "cards_mo",
    "con_cha_mo",
    "con_cif_mo",
    "gen_cha_mo",
    "gen_cif_mo",
    "air_val_mo",
    "air_wgt_mo",
    "air_cha_mo",
    "ves_val_mo",
    "ves_wgt_mo",
    "ves_cha_mo",
    "cnt_val_mo",
    "cnt_wgt_mo",
    "cnt_cha_mo",
)

_MIN_COUNTRY_CODE = "1000"
_MAX_COUNTRY_CODE = "7999"
_MAX_DOWNLOAD_ATTEMPTS = 4
_DOWNLOAD_TIMEOUT_SECONDS = 1800.0
_DOWNLOAD_BACKOFF_SECONDS = 15.0


@dataclass(frozen=True)
class ImdbMonth:
    """One parsed monthly archive: detail plus reconciliation report.

    ``reconciliation_evidence`` is the machine-readable record of every
    comparison the gate ran — per axis: key-set sizes, duplicate-key
    verdicts, and per-measure compared/matched cell counts with the exact
    integer totals on both sides — so a build's "zero failures" claim is
    recomputable and auditable, not prose.
    """

    month: str
    detail: pd.DataFrame
    control_cty: pd.DataFrame
    control_comm: pd.DataFrame
    control_de: pd.DataFrame
    manifest_entry: dict[str, object]
    reconciliation_failures: tuple[str, ...]
    reconciliation_evidence: dict[str, object]


@dataclass(frozen=True)
class ImdbMonthSummary:
    """A month reduced to what assembly needs, detail released.

    A December detail table is ~3.5M rows × 22 int64 measures; holding all
    18 months costs multiple GB. The summary keeps the (hts10, cty_code)
    margin cells (already an order of magnitude smaller) and the control
    tables, so the caller can write the detail artifact and free the full
    table before parsing the next month.
    """

    month: str
    margin_cells: pd.DataFrame
    control_comm: pd.DataFrame
    control_de: pd.DataFrame
    manifest_entry: dict[str, object]
    reconciliation_failures: tuple[str, ...]
    reconciliation_evidence: dict[str, object]


def summarize_imdb_month(month: ImdbMonth) -> ImdbMonthSummary:
    """Reduce a parsed month to its assembly inputs (drops the detail)."""
    margin_cells = month.detail.groupby(["hts10", "cty_code"], as_index=False)[
        list(IMDB_MARGIN_MEASURES)
    ].sum()
    return ImdbMonthSummary(
        month=month.month,
        margin_cells=margin_cells,
        control_comm=month.control_comm,
        control_de=month.control_de,
        manifest_entry=dict(month.manifest_entry),
        reconciliation_failures=month.reconciliation_failures,
        reconciliation_evidence=dict(month.reconciliation_evidence),
    )


@dataclass(frozen=True)
class ImdbBulkAssembly:
    """Assembled multi-month bulk ingest, mirroring the API pull's shape."""

    months: tuple[ImdbMonthSummary, ...]
    margins: pd.DataFrame
    census_totals: pd.DataFrame
    district_entry: pd.DataFrame

    @property
    def manifest_entries(self) -> tuple[dict[str, object], ...]:
        return tuple(month.manifest_entry for month in self.months)

    @property
    def reconciliation_failures(self) -> tuple[str, ...]:
        return tuple(
            failure
            for month in self.months
            for failure in month.reconciliation_failures
        )


def imdb_archive_name(month: str) -> str:
    """``IMDB{yy}{mm}.ZIP`` archive basename for a ``YYYY-MM`` month."""
    year, month_number = _parse_month(month)
    return f"IMDB{year % 100:02d}{month_number:02d}.ZIP"


def imdb_archive_url(month: str) -> str:
    """Published download URL for a ``YYYY-MM`` month."""
    year, month_number = _parse_month(month)
    return IMDB_URL_TEMPLATE.format(
        year=f"{year:04d}", yy=f"{year % 100:02d}", mm=f"{month_number:02d}"
    )


def latest_available_imdb_month(
    *,
    now: datetime | None = None,
    max_probes: int = 6,
    head: object | None = None,
) -> str:
    """Most recent month with a published IMDB archive, probed via HEAD.

    Census publishes with roughly a two-month lag (FT-900 schedule); this
    probes backward from the current month so the caller never hard-codes
    a publication calendar. ``head`` is a test seam returning an HTTP
    status for a URL.
    """
    moment = now or datetime.now(UTC)
    year, month_number = moment.year, moment.month
    for _ in range(max_probes):
        candidate = f"{year:04d}-{month_number:02d}"
        if _head_status(imdb_archive_url(candidate), head=head) == 200:
            return candidate
        month_number -= 1
        if month_number == 0:
            year, month_number = year - 1, 12
    raise RuntimeError(
        f"No published IMDB archive found in the last {max_probes} months; "
        "the bulk publication should never lag that far."
    )


def _head_status(url: str, *, head: object | None) -> int:
    if head is not None:
        return int(head(url))  # type: ignore[operator]
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "populace-us-trade-ingest"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)


def ensure_imdb_archive(
    month: str,
    archive_dir: str | Path,
    *,
    retrieved_at_by_sha: dict[tuple[str, str], str] | None = None,
    fetch: object | None = None,
) -> tuple[Path, dict[str, object]]:
    """Return the verified archive path + manifest entry, downloading if absent.

    A pre-downloaded archive is adopted after a ZIP-integrity check (its
    central directory must parse and list the detail member); the sha256 is
    always recomputed from the bytes on disk, never trusted from a sidecar.
    ``retrieved_at_by_sha`` optionally supplies original retrieval
    timestamps for adopted files keyed by ``(filename, sha256)`` — the
    timestamp is used only when the recorded hash matches the bytes on
    disk, so a swapped file can never inherit another download's
    provenance. ``http_status`` is recorded only for downloads this call
    actually performed.
    """
    archive_path = Path(archive_dir) / imdb_archive_name(month)
    url = imdb_archive_url(month)
    downloaded = not archive_path.exists()
    if downloaded:
        _download_archive(url, archive_path, fetch=fetch)
    _verify_zip_lists_detail(archive_path, month)
    raw = archive_path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    verified_at = datetime.now(UTC).isoformat(timespec="seconds")
    retrieved_at: str | None
    if downloaded:
        retrieved_at = verified_at
        retrieval_note = "downloaded by this build"
    else:
        supplied = (retrieved_at_by_sha or {}).get((archive_path.name, sha256))
        retrieved_at = supplied
        retrieval_note = (
            "pre-downloaded archive adopted; retrieval timestamp from the "
            "download manifest (sha-matched)"
            if supplied
            else (
                "pre-downloaded archive adopted with no download-manifest "
                "match; the retrieval time is unknown and deliberately not "
                "recorded (verified_at records only this build's "
                "verification of the bytes)"
            )
        )
    entry: dict[str, object] = {
        "source_name": "census_imdb_bulk",
        "dataset": "US Imports of Merchandise monthly database (IMDB)",
        "endpoint": IMDB_URL_TEMPLATE,
        "url": url,
        "month": month,
        "filename": archive_path.name,
        "sha256": sha256,
        "size_bytes": len(raw),
        "verified_at": verified_at,
        "retrieval_note": retrieval_note,
    }
    if retrieved_at is not None:
        entry["retrieved_at"] = retrieved_at
    if downloaded:
        entry["http_status"] = 200
    return archive_path, entry


def load_imdb_month(
    archive_path: str | Path,
    month: str,
    manifest_entry: dict[str, object],
) -> ImdbMonth:
    """Parse one monthly archive and reconcile against its control totals.

    Every detail row's embedded statistical (year, month) must equal the
    archive's month: the monthly archives are period snapshots (verified
    empirically on the 2025-01 archive: 727,251 rows, zero off-month), and
    a row claiming another period would silently corrupt the margin series,
    so any such row fails the ingest rather than being dropped.
    """
    archive = Path(archive_path)
    period_year, period_month = _parse_month(month)
    with zipfile.ZipFile(archive) as bundle:
        detail = _read_fixed_width(
            bundle,
            IMDB_DETAIL_MEMBER,
            _DETAIL_KEYS + _DETAIL_MEASURES,
            measure_names=[name for name, _, _ in _DETAIL_MEASURES],
        )
        control_cty = _read_fixed_width(
            bundle,
            _CONTROL_CTY_MEMBER,
            _CTY_FIELDS,
            measure_names=[name for name, _, _ in _CTY_FIELDS if name.endswith("_mo")],
        )
        control_comm = _read_fixed_width(
            bundle,
            _CONTROL_COMM_MEMBER,
            _COMM_FIELDS,
            measure_names=[name for name, _, _ in _COMM_FIELDS if name.endswith("_mo")],
        )
        control_de = _read_fixed_width(
            bundle,
            _CONTROL_DE_MEMBER,
            _DE_FIELDS,
            measure_names=[name for name, _, _ in _DE_FIELDS if name.endswith("_mo")],
        )
    failures: list[str] = []
    for frame, label in (
        (detail, IMDB_DETAIL_MEMBER),
        (control_cty, _CONTROL_CTY_MEMBER),
        (control_comm, _CONTROL_COMM_MEMBER),
        (control_de, _CONTROL_DE_MEMBER),
    ):
        off_period = frame[
            (frame["year"] != f"{period_year:04d}")
            | (frame["month"] != f"{period_month:02d}")
        ]
        if len(off_period):
            failures.append(
                f"{month} {label}: {len(off_period)} rows carry a statistical "
                f"period other than {month} (first: year="
                f"{off_period.iloc[0]['year']!r} month="
                f"{off_period.iloc[0]['month']!r}); the archive is expected "
                "to be a single-period snapshot."
            )
    bad_codes = detail[
        (detail["cty_code"] < _MIN_COUNTRY_CODE)
        | (detail["cty_code"] > _MAX_COUNTRY_CODE)
    ]
    if len(bad_codes):
        failures.append(
            f"{month} {IMDB_DETAIL_MEMBER}: {len(bad_codes)} rows carry "
            f"country codes outside Schedule C {_MIN_COUNTRY_CODE}–"
            f"{_MAX_COUNTRY_CODE} (first: {bad_codes.iloc[0]['cty_code']!r})."
        )
    for frame, label, column, pattern in (
        (detail, IMDB_DETAIL_MEMBER, "hts10", r"\d{10}"),
        (detail, IMDB_DETAIL_MEMBER, "cty_code", r"\d{4}"),
        (detail, IMDB_DETAIL_MEMBER, "dist_entry", r"\d{2}"),
        (control_comm, _CONTROL_COMM_MEMBER, "hts10", r"\d{10}"),
        (control_cty, _CONTROL_CTY_MEMBER, "cty_code", r"\d{4}"),
        (control_de, _CONTROL_DE_MEMBER, "dist_entry", r"\d{2}"),
    ):
        malformed = frame[~frame[column].str.fullmatch(pattern)]
        if len(malformed):
            failures.append(
                f"{month} {label}: {len(malformed)} rows carry a malformed "
                f"{column} (expected /{pattern}/; first: "
                f"{malformed.iloc[0][column]!r})."
            )
    reconcile_failures, evidence = _reconcile_month(
        month, detail, control_cty, control_comm, control_de
    )
    failures.extend(reconcile_failures)
    return ImdbMonth(
        month=month,
        detail=detail,
        control_cty=control_cty,
        control_comm=control_comm,
        control_de=control_de,
        manifest_entry=dict(manifest_entry),
        reconciliation_failures=tuple(failures),
        reconciliation_evidence=evidence,
    )


def assemble_bulk_margins(
    months: tuple[ImdbMonthSummary, ...],
    bridge: CensusCountryBridge,
) -> ImdbBulkAssembly:
    """Aggregate month summaries into the HTS10 × country × month margins table.

    The margins table carries the API-compatible core columns (period,
    hts10, chapter, cty_code, iso2, country_name, the four dollar measures,
    first-unit quantities, unit codes) plus the bulk-only measures
    (second-unit quantities, card counts, charges, CIF, and the air/vessel/
    containerized transport splits). ``census_totals`` mirrors the API
    ingest's publisher-total table from the archives' per-commodity control
    file. ``district_entry`` is the per-district publisher control table
    with names, reconciled against the detail.
    """
    margin_frames: list[pd.DataFrame] = []
    totals_frames: list[pd.DataFrame] = []
    district_frames: list[pd.DataFrame] = []
    for month in months:
        # The monthly archives are year-to-date cell unions: a cell active
        # in any earlier month of the statistical year persists with
        # all-zero monthly measures (verified on the 2025-12 archive —
        # 3.5M rows, ~0.8M with monthly activity). Margins keep one row
        # per cell with monthly activity, matching the API leg; the full
        # union stays in the per-month detail artifact as published.
        grouped = month.margin_cells.loc[
            month.margin_cells[list(IMDB_MARGIN_MEASURES)].any(axis="columns")
        ].reset_index(drop=True)
        units = month.control_comm.set_index("hts10")[["unit_qy1", "unit_qy2"]]
        grouped["unit_qy1"] = grouped["hts10"].map(units["unit_qy1"]).fillna("")
        grouped["unit_qy2"] = grouped["hts10"].map(units["unit_qy2"]).fillna("")
        grouped.insert(0, "period", month.month)
        margin_frames.append(grouped)

        totals = month.control_comm.copy()
        totals.insert(0, "period", month.month)
        totals_frames.append(
            totals[
                [
                    "period",
                    "hts10",
                    "con_val_mo",
                    "gen_val_mo",
                    "cal_dut_mo",
                    "dut_val_mo",
                    "con_qy1_mo",
                    "gen_qy1_mo",
                    "unit_qy1",
                ]
            ]
        )

        district = month.control_de.copy()
        district.insert(0, "period", month.month)
        district_frames.append(
            district[
                [
                    "period",
                    "dist_entry",
                    "dist_name",
                    "con_val_mo",
                    "gen_val_mo",
                    "cal_dut_mo",
                    "dut_val_mo",
                    "air_val_mo",
                    "ves_val_mo",
                    "cnt_val_mo",
                ]
            ]
        )

    margins = pd.concat(margin_frames, ignore_index=True)
    margins["iso2"] = margins["cty_code"].map(lambda code: bridge.iso2(str(code)))
    margins["country_name"] = margins["cty_code"].map(
        lambda code: bridge.name_by_census_code.get(str(code), "")
    )
    margins.insert(2, "chapter", margins["hts10"].str[:2])
    ordered = [
        "period",
        "hts10",
        "chapter",
        "cty_code",
        "iso2",
        "country_name",
        *IMDB_MARGIN_MEASURES,
        "unit_qy1",
        "unit_qy2",
    ]
    margins = margins[ordered]
    dtypes: dict[str, str] = {
        column: "string"
        for column in (
            "period",
            "hts10",
            "chapter",
            "cty_code",
            "iso2",
            "country_name",
            "unit_qy1",
            "unit_qy2",
        )
    }
    dtypes.update({measure: "int64" for measure in IMDB_MARGIN_MEASURES})
    margins = margins.astype(dtypes).sort_values(
        ["period", "hts10", "cty_code"], ignore_index=True
    )

    census_totals = pd.concat(totals_frames, ignore_index=True)
    census_totals = census_totals.astype(
        {
            "period": "string",
            "hts10": "string",
            "con_val_mo": "int64",
            "gen_val_mo": "int64",
            "cal_dut_mo": "int64",
            "dut_val_mo": "int64",
            "con_qy1_mo": "int64",
            "gen_qy1_mo": "int64",
            "unit_qy1": "string",
        }
    ).sort_values(["period", "hts10"], ignore_index=True)

    district_entry = pd.concat(district_frames, ignore_index=True)
    district_entry = district_entry.astype(
        {
            "period": "string",
            "dist_entry": "string",
            "dist_name": "string",
            "con_val_mo": "int64",
            "gen_val_mo": "int64",
            "cal_dut_mo": "int64",
            "dut_val_mo": "int64",
            "air_val_mo": "int64",
            "ves_val_mo": "int64",
            "cnt_val_mo": "int64",
        }
    ).sort_values(["period", "dist_entry"], ignore_index=True)

    return ImdbBulkAssembly(
        months=months,
        margins=margins,
        census_totals=census_totals,
        district_entry=district_entry,
    )


def _reconcile_month(
    month: str,
    detail: pd.DataFrame,
    control_cty: pd.DataFrame,
    control_comm: pd.DataFrame,
    control_de: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    """Exact-integer detail-vs-control comparison on every published measure.

    Three independent gates per axis (country, commodity, district of
    entry):

    - **Control-key uniqueness**: a control file must publish exactly one
      row per key. A duplicated control row — even a byte-identical one —
      fails the gate outright, because a duplicate-keyed control table
      cannot serve as a reconciliation oracle (a set comparison and an
      index join would both accept it silently).
    - **Key-set equality**: the detail's key set must equal the control
      file's key set exactly — a key on either side only is reported by
      name, never silently compared against zero.
    - **Value equality on every measure the control file publishes** (all
      22 detail measures on the commodity axis; everything but unit
      quantities on the country and district axes). Measures are published
      as explicit integers, so any difference means dropped, duplicated,
      or misparsed detail.

    Returns the failure list plus the machine-readable evidence of every
    comparison run: per axis, the key-set sizes, duplicate verdict, and
    per-measure compared/matched cell counts with both sides' integer
    totals.

    Certification limit, stated honestly: a detail row that is all-zero on
    every monthly measure (a year-to-date union carrier) contributes
    nothing to any control total — including ``cards_mo``, which counts
    *current-month* records — so its presence cannot be certified against
    the control files. Such rows carry no monthly information; they are
    preserved verbatim in the detail artifact and excluded from margins.
    Any row with any current-month activity is protected by the value
    gates (``cards_mo`` reconciliation counts the active records
    themselves).
    """
    failures: list[str] = []
    evidence: dict[str, object] = {"month": month, "axes": {}}
    comparisons = (
        ("cty_code", control_cty, _RECONCILE_CTY, _CONTROL_CTY_MEMBER),
        ("hts10", control_comm, _RECONCILE_COMM, _CONTROL_COMM_MEMBER),
        ("dist_entry", control_de, _RECONCILE_DE, _CONTROL_DE_MEMBER),
    )
    for key, control, measures, label in comparisons:
        axis_evidence: dict[str, object] = {"key": key, "control_member": label}
        duplicated = control.loc[control[key].duplicated(), key]
        duplicate_keys = sorted(set(duplicated))
        axis_evidence["control_rows"] = int(len(control))
        axis_evidence["duplicate_control_keys"] = duplicate_keys[:20]
        axis_evidence["duplicate_control_key_count"] = len(duplicate_keys)
        if duplicate_keys:
            failures.append(
                f"{month} {label}: {len(duplicate_keys)} duplicated control "
                f"key(s) for {key} (first: {duplicate_keys[0]!r}); a "
                "control table with duplicate keys cannot serve as a "
                "reconciliation oracle, so value reconciliation on this "
                "axis was not attempted."
            )
            axis_evidence["value_comparison"] = "skipped_duplicate_control_keys"
            evidence["axes"][label] = axis_evidence
            continue
        detail_keys = set(detail[key].unique())
        control_keys = set(control[key].unique())
        missing_keys = sorted(control_keys - detail_keys)
        extra_keys = sorted(detail_keys - control_keys)
        axis_evidence["detail_key_count"] = len(detail_keys)
        axis_evidence["control_key_count"] = len(control_keys)
        axis_evidence["keys_matched"] = len(detail_keys & control_keys)
        axis_evidence["control_only_keys"] = missing_keys[:20]
        axis_evidence["control_only_key_count"] = len(missing_keys)
        axis_evidence["detail_only_keys"] = extra_keys[:20]
        axis_evidence["detail_only_key_count"] = len(extra_keys)
        for missing in missing_keys:
            failures.append(
                f"{month} {label} {key}={missing}: present in the control "
                "file but absent from the detail."
            )
        for extra in extra_keys:
            failures.append(
                f"{month} {label} {key}={extra}: present in the detail but "
                "absent from the control file."
            )
        summed = detail.groupby(key)[list(measures)].sum()
        published = control.set_index(key)[list(measures)]
        joined = summed.join(
            published, how="inner", lsuffix="_detail", rsuffix="_published"
        )
        measures_evidence: dict[str, object] = {}
        for measure in measures:
            detail_column = f"{measure}_detail"
            published_column = f"{measure}_published"
            detail_values = joined[detail_column].astype("int64")
            published_values = joined[published_column].astype("int64")
            mismatched = joined[detail_values != published_values]
            measures_evidence[measure] = {
                "cells_compared": int(len(joined)),
                "cells_matched": int(len(joined) - len(mismatched)),
                "detail_total": int(detail_values.sum()),
                "published_total": int(published_values.sum()),
            }
            for key_value, row in mismatched.iterrows():
                failures.append(
                    f"{month} {label} {key}={key_value} {measure}: detail "
                    f"sums to {int(row[detail_column])}, published control "
                    f"total is {int(row[published_column])}."
                )
        axis_evidence["measures"] = measures_evidence
        evidence["axes"][label] = axis_evidence
    evidence["failure_count"] = len(failures)
    evidence["failures"] = list(failures[:50])
    return failures, evidence


def _read_fixed_width(
    bundle: zipfile.ZipFile,
    member_name: str,
    fields: tuple[tuple[str, int, int], ...],
    *,
    measure_names: list[str],
) -> pd.DataFrame:
    """Parse a fixed-width member per its layout into typed columns.

    Positions are the layouts' 1-indexed inclusive ranges. Lines shorter
    than the last needed position fail loudly (a layout change must never
    silently zero-fill). Measure columns are parsed as int64; everything
    else is kept as stripped strings.
    """
    member = _locate_member(bundle, member_name)
    colspecs = [(start - 1, end) for _, start, end in fields]
    names = [name for name, _, _ in fields]
    minimum_width = max(end for _, _, end in fields)
    with bundle.open(member) as stream:
        frame = pd.read_fwf(
            stream,
            colspecs=colspecs,
            names=names,
            dtype=str,
            header=None,
            encoding="latin-1",
        )
    if frame.empty:
        raise ValueError(f"IMDB member {member_name} parsed to zero rows.")
    # An all-space slice reads as NaN; for string fields that is a
    # legitimately blank cell (e.g. unit_qy2 for unitless lines). A
    # truncated or re-laid-out line instead leaves *measure* cells blank,
    # which the integer parse below refuses — measures are published as
    # explicit integers (zeros included) in every row.
    for name in names:
        if name in measure_names:
            continue
        frame[name] = frame[name].fillna("").str.strip()
    for name in measure_names:
        column = frame[name]
        if column.isna().any() or (column.str.strip() == "").any():
            raise ValueError(
                f"IMDB member {member_name} has blank {name} cells (layout "
                f"width {minimum_width}); either the line is truncated or "
                "the published layout changed — refusing to guess."
            )
        frame[name] = column.astype("int64")
    return frame


def _locate_member(bundle: zipfile.ZipFile, member_name: str) -> str:
    matches = [
        name
        for name in bundle.namelist()
        if "/" not in name and name.lower() == member_name.lower()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"IMDB archive must carry exactly one top-level {member_name} "
            f"member; found {matches!r}."
        )
    return matches[0]


def _verify_zip_lists_detail(archive_path: Path, month: str) -> None:
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            _locate_member(bundle, IMDB_DETAIL_MEMBER)
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"IMDB archive for {month} at {archive_path} is not a readable "
            f"ZIP: {error}. Delete it to re-download."
        ) from error


def _download_archive(url: str, destination: Path, *, fetch: object | None) -> None:
    """Stream one archive to disk with bounded retries and a ZIP check."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        if attempt:
            time_module.sleep(_DOWNLOAD_BACKOFF_SECONDS * attempt)
        try:
            if fetch is not None:
                part_path.write_bytes(fetch(url))  # type: ignore[operator]
            else:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "populace-us-trade-ingest"}
                )
                with (
                    urllib.request.urlopen(
                        request, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                    ) as response,
                    part_path.open("wb") as sink,
                ):
                    while chunk := response.read(1 << 20):
                        sink.write(chunk)
            with zipfile.ZipFile(part_path):
                pass
            part_path.rename(destination)
            return
        except (
            urllib.error.URLError,
            TimeoutError,
            zipfile.BadZipFile,
            OSError,
        ) as error:
            last_error = error
            part_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not download a valid IMDB archive from {url} after "
        f"{_MAX_DOWNLOAD_ATTEMPTS} attempts: {last_error!r}."
    )


def _parse_month(month: str) -> tuple[int, int]:
    parts = month.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Month {month!r} is not in YYYY-MM form.")
    year, month_number = int(parts[0]), int(parts[1])
    if not 1 <= month_number <= 12:
        raise ValueError(f"Month {month!r} is not in YYYY-MM form.")
    return year, month_number
