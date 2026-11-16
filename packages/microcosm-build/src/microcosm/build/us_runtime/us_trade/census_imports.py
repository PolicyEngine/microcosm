"""Census International Trade API ingest for US import-entry margins.

Source dataset: ``timeseries/intltrade/imports/hs`` on api.census.gov — the
official monthly US merchandise import statistics by Harmonized System
commodity and Schedule C country (the same publication as USA Trade Online).
Variable identities were verified against the dataset's published variable
catalog (``variables.json``): imports-for-consumption total value
(``CON_VAL_MO``), general-imports total value (``GEN_VAL_MO``), calculated
duty (``CAL_DUT_MO``), dutiable value (``DUT_VAL_MO``), and first-unit
quantities (``CON_QY1_MO``/``GEN_QY1_MO`` with ``UNIT_QY1``).

Pull geometry: the API returns 204 for an unconstrained full-month HS10
query, so the ingest iterates HS chapters (``I_COMMODITY=<NN>*``) per month.
The server cannot always materialize a full chapter in one response — the
largest chapters (vehicles, machinery, electronics) time out server-side
with HTTP 500 after ~150 seconds — so a chapter that stalls or returns a
compute-timeout status is split into its ten 3-digit prefixes (and 4-digit
if ever needed). Prefixes partition the HTS10 space, so the split changes
request geometry only, never coverage; every fetched prefix gets its own
cache file and manifest entry, and the aborted chapter attempt is recorded
with ``superseded_by_split`` so a resumed pull skips straight to the
sub-prefixes.

Response rows carry a ``SUMMARY_LVL`` marker: ``DET`` rows are the detail
atoms (Schedule C countries plus the ``-`` all-country total row) and
``CGP`` rows are published country-group rollups. The ingest keeps DET
country rows as margins, keeps the DET ``-`` rows as the publisher's own
totals for an exact reconciliation gate, and drops CGP rollups (derivable,
would double count).

Every fetched byte stream is cached verbatim and recorded in a retrieval
manifest (endpoint, parameters, retrieval timestamp, SHA-256, row count).
The API key is never written into manifests or recorded URLs.

This module owns retrieval, parsing, and margin-table assembly only; fact
emission for the ledger consumer contract lives in
:mod:`microcosm.build.us_runtime.us_trade.import_entry_facts`.
"""

from __future__ import annotations

import hashlib
import json
import time as time_module
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microcosm.build.us_runtime.us_trade.census_country_bridge import (
    CensusCountryBridge,
)

__all__ = [
    "CENSUS_IMPORTS_HS_ENDPOINT",
    "CENSUS_IMPORTS_MEASURES",
    "CENSUS_IMPORTS_QUANTITY_COLUMNS",
    "CensusImportsMonth",
    "CensusImportsPull",
    "HS_CHAPTERS",
    "assemble_margins_table",
    "fetch_imports_month",
    "latest_published_month",
    "month_range",
    "parse_imports_response",
]

CENSUS_IMPORTS_HS_ENDPOINT = (
    "https://api.census.gov/data/timeseries/intltrade/imports/hs"
)

#: Dollar-valued measure columns, as published (integer USD).
CENSUS_IMPORTS_MEASURES = (
    "CON_VAL_MO",
    "GEN_VAL_MO",
    "CAL_DUT_MO",
    "DUT_VAL_MO",
)

#: First-unit quantity columns; units vary by HTS line (``UNIT_QY1``).
CENSUS_IMPORTS_QUANTITY_COLUMNS = ("CON_QY1_MO", "GEN_QY1_MO")

_GET_VARIABLES = (
    "CTY_CODE",
    "SUMMARY_LVL",
    *CENSUS_IMPORTS_MEASURES,
    *CENSUS_IMPORTS_QUANTITY_COLUMNS,
    "UNIT_QY1",
)

#: HS chapters 01–97 are the commodity nomenclature; 98 covers US special
#: classification provisions (e.g. returned goods), which do report import
#: value. Chapter 77 is reserved and chapter 99 headings are duty overlays,
#: not commodity classifications; both return no data and are recorded as
#: empty pulls rather than skipped a priori.
HS_CHAPTERS = tuple(f"{chapter:02d}" for chapter in range(1, 100))

_TOTAL_COUNTRY_CODE = "-"
_DETAIL_SUMMARY_LEVEL = "DET"
_MIN_COUNTRY_CODE = "1000"
_MAX_COUNTRY_CODE = "7999"
_RETRY_STATUSES = frozenset({429, 502, 503})
_TERMINAL_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
#: Statuses the server emits when a prefix is too large to materialize
#: (observed: HTTP 500 after ~150s on the biggest chapters).
_SPLIT_STATUSES = frozenset({500, 504})
_MAX_ATTEMPTS = 5
_BACKOFF_BASE_SECONDS = 5.0
#: A chapter that has not answered in this long is almost certainly headed
#: for a server-side compute timeout; abort and split instead of waiting.
_CHAPTER_STALL_SECONDS = 75.0
_DEEP_TIMEOUT_SECONDS = 300.0
_MAX_PREFIX_LENGTH = 4


class _ServerOverloadError(Exception):
    """The server cannot materialize this prefix; split it finer.

    ``cached_entry`` carries the on-disk split marker when the overload
    verdict came from the cache rather than a live request, so resumes
    reuse the original marker — its ``retrieved_at`` is the moment the
    live overload was actually observed, and reusing a cache must never
    rewrite it.
    """

    def __init__(
        self,
        prefix: str,
        reason: str,
        cached_entry: dict[str, object] | None = None,
    ):
        super().__init__(f"Prefix {prefix}* overloads the imports API: {reason}")
        self.prefix = prefix
        self.reason = reason
        self.cached_entry = cached_entry


@dataclass(frozen=True)
class _FetchResult:
    raw: bytes
    url: str
    status: int
    retrieved_at: str


@dataclass(frozen=True)
class CensusImportsMonth:
    """One month's raw pulls plus parsed rows and reconciliation report."""

    month: str
    country_rows: tuple[dict[str, object], ...]
    total_rows: tuple[dict[str, object], ...]
    manifest_entries: tuple[dict[str, object], ...]
    reconciliation_failures: tuple[str, ...]


@dataclass(frozen=True)
class CensusImportsPull:
    """A completed multi-month ingest: margins plus retrieval manifest."""

    months: tuple[CensusImportsMonth, ...]
    margins: pd.DataFrame
    census_totals: pd.DataFrame

    @property
    def manifest_entries(self) -> tuple[dict[str, object], ...]:
        return tuple(entry for month in self.months for entry in month.manifest_entries)

    @property
    def reconciliation_failures(self) -> tuple[str, ...]:
        return tuple(
            failure
            for month in self.months
            for failure in month.reconciliation_failures
        )


def month_range(start: str, end: str) -> tuple[str, ...]:
    """Inclusive ``YYYY-MM`` month sequence from ``start`` through ``end``."""
    start_year, start_month = _parse_month(start)
    end_year, end_month = _parse_month(end)
    if (start_year, start_month) > (end_year, end_month):
        raise ValueError(f"Month range start {start!r} is after end {end!r}.")
    months: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return tuple(months)


def latest_published_month(
    api_key: str,
    *,
    now: datetime | None = None,
    probe_chapter: str = "31",
    max_probes: int = 6,
    fetch: object | None = None,
) -> str:
    """Most recent month the imports dataset has published, by probing.

    Census publishes with roughly a two-month lag (FT-900 schedule); this
    probes backward from the current month until a chapter query returns
    rows, so the caller never hard-codes a publication calendar.
    """
    moment = now or datetime.now(UTC)
    year, month = moment.year, moment.month
    for _ in range(max_probes):
        candidate = f"{year:04d}-{month:02d}"
        rows = _request_rows(
            _build_url(candidate, probe_chapter, api_key),
            allow_no_content=True,
            fetch=fetch,
        )
        if rows is not None and len(rows) > 1:
            return candidate
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    raise RuntimeError(
        f"No published imports month found in the last {max_probes} months; "
        "the Census imports dataset should never lag that far."
    )


def fetch_imports_month(
    month: str,
    api_key: str,
    *,
    cache_dir: str | Path,
    chapters: tuple[str, ...] = HS_CHAPTERS,
    fetch: object | None = None,
    throttle_seconds: float = 0.2,
    max_workers: int = 1,
) -> CensusImportsMonth:
    """Fetch, cache, and parse one month of HS10 × country import margins.

    Already-cached chapter files are reused byte-for-byte (the manifest
    entry then records the cached file's retrieval metadata sidecar), so an
    interrupted multi-month pull resumes without re-downloading.
    ``max_workers`` bounds concurrent chapter requests; results are
    assembled in chapter order regardless of completion order.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def fetch_one(chapter: str) -> list[tuple[bytes | None, dict[str, object]]]:
        return _fetch_prefix_tree(
            month,
            chapter,
            api_key,
            cache_path,
            fetch=fetch,
            throttle_seconds=throttle_seconds,
        )

    if max_workers > 1 and fetch is None:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fetched = list(pool.map(fetch_one, chapters))
    else:
        fetched = [fetch_one(chapter) for chapter in chapters]

    country_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    manifest_entries: list[dict[str, object]] = []
    for results in fetched:
        for raw, entry in results:
            manifest_entries.append(entry)
            if raw is None:
                continue
            parsed_countries, parsed_totals = parse_imports_response(
                raw, month, str(entry["prefix"])
            )
            country_rows.extend(parsed_countries)
            total_rows.extend(parsed_totals)
    failures = _reconcile_against_census_totals(country_rows, total_rows, month)
    return CensusImportsMonth(
        month=month,
        country_rows=tuple(country_rows),
        total_rows=tuple(total_rows),
        manifest_entries=tuple(manifest_entries),
        reconciliation_failures=failures,
    )


def parse_imports_response(
    raw: bytes,
    month: str,
    prefix: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Parse one API response into country detail rows and ``-`` total rows.

    ``prefix`` is the commodity constraint the response answered — a whole
    chapter (``"87"``) or a split sub-prefix (``"870"``). Returns
    ``(country_rows, total_rows)``. CGP country-group rollups are dropped.
    Rows echo their constraint columns; the parse validates the echoed
    month and commodity prefix against the request so a mislabeled cache
    file cannot silently cross wires.
    """
    table = json.loads(raw.decode("utf-8"))
    if not isinstance(table, list) or not table or not isinstance(table[0], list):
        raise ValueError(
            f"Census imports response for {month} prefix {prefix} is not "
            "a JSON row table."
        )
    header = table[0]
    index = {name: position for position, name in enumerate(header)}
    required = {*_GET_VARIABLES, "I_COMMODITY", "time"}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(
            f"Census imports response for {month} prefix {prefix} is "
            f"missing expected columns {missing}; header was {header}."
        )
    country_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    for values in table[1:]:
        row_time = values[index["time"]]
        if row_time != month:
            raise ValueError(
                f"Census imports response row echoes month {row_time!r} for "
                f"a {month} request (prefix {prefix})."
            )
        hts10 = values[index["I_COMMODITY"]]
        if len(hts10) != 10 or not hts10.isdigit() or not hts10.startswith(prefix):
            raise ValueError(
                f"Census imports response row echoes commodity {hts10!r} for "
                f"a prefix {prefix} HS10 request ({month})."
            )
        if values[index["SUMMARY_LVL"]] != _DETAIL_SUMMARY_LEVEL:
            continue
        cty_code = values[index["CTY_CODE"]]
        row = {
            "period": month,
            "hts10": hts10,
            "cty_code": cty_code,
            "con_val_mo": _required_int(values, index, "CON_VAL_MO", month, hts10),
            "gen_val_mo": _required_int(values, index, "GEN_VAL_MO", month, hts10),
            "cal_dut_mo": _required_int(values, index, "CAL_DUT_MO", month, hts10),
            "dut_val_mo": _required_int(values, index, "DUT_VAL_MO", month, hts10),
            "con_qy1_mo": _optional_int(values, index, "CON_QY1_MO"),
            "gen_qy1_mo": _optional_int(values, index, "GEN_QY1_MO"),
            "unit_qy1": values[index["UNIT_QY1"]] or "",
        }
        if cty_code == _TOTAL_COUNTRY_CODE:
            total_rows.append(row)
        elif _MIN_COUNTRY_CODE <= cty_code <= _MAX_COUNTRY_CODE:
            country_rows.append(row)
        else:
            raise ValueError(
                f"Census imports DET row has unexpected country code "
                f"{cty_code!r} ({month}, HTS {hts10}); detail rows should be "
                "Schedule C countries (1000–7999) or the '-' total."
            )
    return country_rows, total_rows


def assemble_margins_table(
    months: tuple[CensusImportsMonth, ...],
    bridge: CensusCountryBridge,
) -> CensusImportsPull:
    """Assemble parsed months into the tidy HTS10 × country × month table.

    Every country margin row is bridged to ISO-2 fail-closed: an unmapped
    Schedule C code aborts the assembly rather than shipping a mislabeled
    or dropped country.
    """
    country_records: list[dict[str, object]] = []
    for month in months:
        for row in month.country_rows:
            cty_code = str(row["cty_code"])
            country_records.append(
                {
                    **row,
                    "iso2": bridge.iso2(cty_code),
                    "country_name": bridge.name_by_census_code.get(cty_code, ""),
                }
            )
    total_records = [dict(row) for month in months for row in month.total_rows]
    margins = pd.DataFrame(country_records)
    census_totals = pd.DataFrame(total_records)
    if not margins.empty:
        margins = margins.astype(
            {
                "period": "string",
                "hts10": "string",
                "cty_code": "string",
                "iso2": "string",
                "country_name": "string",
                "con_val_mo": "int64",
                "gen_val_mo": "int64",
                "cal_dut_mo": "int64",
                "dut_val_mo": "int64",
                "con_qy1_mo": "Int64",
                "gen_qy1_mo": "Int64",
                "unit_qy1": "string",
            }
        )
        margins.insert(2, "chapter", margins["hts10"].str[:2].astype("string"))
        margins = margins.sort_values(
            ["period", "hts10", "cty_code"], ignore_index=True
        )
    if not census_totals.empty:
        census_totals = census_totals.drop(columns=["cty_code"]).astype(
            {
                "period": "string",
                "hts10": "string",
                "con_val_mo": "int64",
                "gen_val_mo": "int64",
                "cal_dut_mo": "int64",
                "dut_val_mo": "int64",
                "con_qy1_mo": "Int64",
                "gen_qy1_mo": "Int64",
                "unit_qy1": "string",
            }
        )
        census_totals = census_totals.sort_values(
            ["period", "hts10"], ignore_index=True
        )
    return CensusImportsPull(
        months=months, margins=margins, census_totals=census_totals
    )


def _reconcile_against_census_totals(
    country_rows: list[dict[str, object]],
    total_rows: list[dict[str, object]],
    month: str,
) -> tuple[str, ...]:
    """Check per-HTS10 country sums against the publisher's ``-`` totals.

    The dollar measures are published as integer USD, so the country detail
    must sum exactly to the published total; any difference means dropped,
    duplicated, or misparsed detail and fails the ingest. Coverage is part
    of the check: iterating only the published total keys would let active
    detail pass unreconciled whenever its ``-`` row is missing (a totals
    slice holding nothing but unrelated zero carriers reconciles nothing),
    so an HTS10 with active detail and no published total is itself a
    failure. Zero-carrier detail without a total stays agreement-by-absence.
    """
    dollar_measures = ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo")
    quantity_measures = ("con_qy1_mo", "gen_qy1_mo")
    sums: dict[tuple[str, str], int] = {}
    detail_active: dict[str, bool] = {}
    for row in country_rows:
        hts10 = str(row["hts10"])
        active = detail_active.get(hts10, False)
        for measure in dollar_measures:
            key = (hts10, measure)
            value = int(row[measure])  # type: ignore[arg-type]
            sums[key] = sums.get(key, 0) + value
            active = active or value != 0
        for measure in quantity_measures:
            quantity = row.get(measure)
            active = active or (quantity is not None and int(quantity) != 0)  # type: ignore[arg-type]
        detail_active[hts10] = active
    failures: list[str] = []
    total_keys = set()
    for row in total_rows:
        hts10 = str(row["hts10"])
        total_keys.add(hts10)
        for measure in dollar_measures:
            published = int(row[measure])  # type: ignore[arg-type]
            summed = sums.get((hts10, measure), 0)
            if summed != published:
                failures.append(
                    f"{month} HTS {hts10} {measure}: country detail sums to "
                    f"{summed}, published total is {published}."
                )
    for hts10 in sorted(detail_active):
        if detail_active[hts10] and hts10 not in total_keys:
            failures.append(
                f"{month} HTS {hts10}: active country detail has no "
                "published '-' total row to reconcile against."
            )
    return tuple(failures)


def _fetch_prefix_tree(
    month: str,
    prefix: str,
    api_key: str,
    cache_dir: Path,
    *,
    fetch: object | None,
    throttle_seconds: float,
) -> list[tuple[bytes | None, dict[str, object]]]:
    """Fetch one commodity prefix, splitting finer when the server cannot.

    Returns ``(raw, manifest_entry)`` pairs for every request made in the
    subtree, including the ``superseded_by_split`` marker for an aborted
    prefix attempt (recorded so resumes skip the doomed request and the
    manifest keeps the full retrieval story).
    """
    try:
        return [
            _fetch_prefix_with_cache(
                month,
                prefix,
                api_key,
                cache_dir,
                fetch=fetch,
                throttle_seconds=throttle_seconds,
            )
        ]
    except _ServerOverloadError as overload:
        if len(prefix) >= _MAX_PREFIX_LENGTH:
            raise RuntimeError(
                f"Census imports API cannot materialize {month} prefix "
                f"{prefix}* even at the maximum split depth: {overload.reason}"
            ) from overload
        # A cached overload verdict reuses its on-disk marker verbatim —
        # reuse is not retrieval, so no fresh timestamp and no sidecar
        # rewrite (a read-only cache must resume cleanly). Only a live
        # overload mints and records a new marker.
        marker = overload.cached_entry or _split_marker(
            month, prefix, overload.reason, cache_dir
        )
        results: list[tuple[bytes | None, dict[str, object]]] = [(None, marker)]
        for digit in "0123456789":
            results.extend(
                _fetch_prefix_tree(
                    month,
                    prefix + digit,
                    api_key,
                    cache_dir,
                    fetch=fetch,
                    throttle_seconds=throttle_seconds,
                )
            )
        return results


def _cache_paths(cache_dir: Path, month: str, prefix: str) -> tuple[Path, Path]:
    stem = f"ch{prefix}" if len(prefix) == 2 else f"p{prefix}"
    data_path = cache_dir / f"imports_hs10_{month}_{stem}.json"
    return data_path, data_path.with_suffix(".meta.json")


def _split_marker(
    month: str, prefix: str, reason: str, cache_dir: Path
) -> dict[str, object]:
    """Record (and cache) that a prefix attempt was superseded by a split."""
    _, meta_path = _cache_paths(cache_dir, month, prefix)
    marker: dict[str, object] = {
        "source_name": "census_intltrade",
        "dataset": "timeseries/intltrade/imports/hs",
        "endpoint": CENSUS_IMPORTS_HS_ENDPOINT,
        "month": month,
        "prefix": prefix,
        "chapter": prefix[:2],
        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "http_status": None,
        "sha256": None,
        "size_bytes": 0,
        "row_count": 0,
        "superseded_by_split": True,
        "split_reason": reason,
        "filename": None,
    }
    meta_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return marker


def _fetch_prefix_with_cache(
    month: str,
    prefix: str,
    api_key: str,
    cache_dir: Path,
    *,
    fetch: object | None,
    throttle_seconds: float,
) -> tuple[bytes | None, dict[str, object]]:
    data_path, meta_path = _cache_paths(cache_dir, month, prefix)
    if meta_path.exists():
        entry = json.loads(meta_path.read_text())
        # Chapter-level caches written before prefix splitting existed
        # carry only the "chapter" key.
        entry.setdefault("prefix", entry.get("chapter", prefix))
        if entry.get("superseded_by_split"):
            raise _ServerOverloadError(
                prefix,
                str(entry.get("split_reason", "cached")),
                cached_entry=entry,
            )
        if entry.get("http_status") == 204:
            return None, entry
        raw = data_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(
                f"Cached Census pull {data_path} does not match its recorded "
                f"hash; delete the pair to re-fetch."
            )
        return raw, entry
    url = _build_url(month, prefix, api_key)
    if throttle_seconds:
        time_module.sleep(throttle_seconds)
    result = _request(
        url,
        allow_no_content=True,
        fetch=fetch,
        splittable=len(prefix) < _MAX_PREFIX_LENGTH,
        prefix=prefix,
    )
    entry: dict[str, object] = {
        "source_name": "census_intltrade",
        "dataset": "timeseries/intltrade/imports/hs",
        "endpoint": CENSUS_IMPORTS_HS_ENDPOINT,
        "url": _elide_key(result.url),
        "month": month,
        "prefix": prefix,
        "chapter": prefix[:2],
        "retrieved_at": result.retrieved_at,
        "http_status": result.status,
        "filename": data_path.name,
    }
    if result.status == 204:
        entry.update({"sha256": None, "size_bytes": 0, "row_count": 0})
        meta_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        return None, entry
    raw = result.raw
    row_count = max(len(json.loads(raw.decode("utf-8"))) - 1, 0)
    entry.update(
        {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "row_count": row_count,
        }
    )
    data_path.write_bytes(raw)
    meta_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
    return raw, entry


def _build_url(month: str, prefix: str, api_key: str) -> str:
    query = urllib.parse.urlencode(
        {
            "get": ",".join(_GET_VARIABLES),
            "COMM_LVL": "HS10",
            "I_COMMODITY": f"{prefix}*",
            "time": month,
            "key": api_key,
        }
    )
    return f"{CENSUS_IMPORTS_HS_ENDPOINT}?{query}"


def _elide_key(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    pairs = [
        (name, value)
        for name, value in urllib.parse.parse_qsl(parsed.query)
        if name != "key"
    ]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(pairs)))


def _request_rows(
    url: str,
    *,
    allow_no_content: bool,
    fetch: object | None,
) -> list[list[str]] | None:
    result = _request(url, allow_no_content=allow_no_content, fetch=fetch)
    if result.status == 204:
        return None
    return json.loads(result.raw.decode("utf-8"))


def _request(
    url: str,
    *,
    allow_no_content: bool,
    fetch: object | None,
    splittable: bool = False,
    prefix: str = "",
) -> _FetchResult:
    """GET with bounded retries; ``fetch`` is a test seam for canned bytes.

    On a splittable request, a compute-timeout status (500/504) or a stalled
    connection raises :class:`_ServerOverloadError` immediately — those failures
    are a property of the response size, and splitting the prefix is both
    faster and kinder to the API than retrying the doomed request. At the
    maximum split depth they retry with backoff like any transient error.
    """
    if fetch is not None:
        status, raw = fetch(url)  # type: ignore[operator]
        if splittable and status in _SPLIT_STATUSES:
            raise _ServerOverloadError(prefix, f"HTTP {status}")
        return _FetchResult(
            raw=raw,
            url=url,
            status=status,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
    timeout = _CHAPTER_STALL_SECONDS if splittable else _DEEP_TIMEOUT_SECONDS
    retry_statuses = _RETRY_STATUSES if splittable else _TERMINAL_RETRY_STATUSES
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        if attempt:
            time_module.sleep(_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "populace-us-trade-ingest"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as error:
            if splittable and error.code in _SPLIT_STATUSES:
                raise _ServerOverloadError(prefix, f"HTTP {error.code}") from None
            if error.code in retry_statuses:
                last_error = error
                continue
            raise
        except TimeoutError as error:
            if splittable:
                raise _ServerOverloadError(
                    prefix, f"no response within {timeout:.0f}s"
                ) from None
            last_error = error
            continue
        except urllib.error.URLError as error:
            if splittable and isinstance(error.reason, TimeoutError):
                raise _ServerOverloadError(
                    prefix, f"no response within {timeout:.0f}s"
                ) from None
            last_error = error
            continue
        if status == 204 and allow_no_content:
            return _FetchResult(
                raw=b"",
                url=url,
                status=204,
                retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        if status == 200:
            return _FetchResult(
                raw=raw,
                url=url,
                status=200,
                retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        raise RuntimeError(
            f"Census imports API returned HTTP {status} for {_elide_key(url)}; "
            "a redirect here means a missing or unactivated API key."
        )
    raise RuntimeError(
        f"Census imports API request failed after {_MAX_ATTEMPTS} attempts "
        f"for {_elide_key(url)}: {last_error!r}."
    )


def _required_int(
    values: list[str],
    index: dict[str, int],
    column: str,
    month: str,
    hts10: str,
) -> int:
    text = values[index[column]]
    if text is None or text == "":
        raise ValueError(
            f"Census imports row ({month}, HTS {hts10}) is missing required "
            f"measure {column}."
        )
    return int(text)


def _optional_int(values: list[str], index: dict[str, int], column: str) -> int | None:
    text = values[index[column]]
    if text is None or text == "":
        return None
    return int(text)


def _parse_month(month: str) -> tuple[int, int]:
    parts = month.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Month {month!r} is not in YYYY-MM form.")
    year, month_number = int(parts[0]), int(parts[1])
    if not 1 <= month_number <= 12:
        raise ValueError(f"Month {month!r} is not in YYYY-MM form.")
    return year, month_number
