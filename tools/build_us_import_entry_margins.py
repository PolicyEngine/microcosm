"""Build the US import-entry margin artifacts (populace#615 P1).

Primary source: the Census monthly bulk *U.S. Imports of Merchandise*
database (IMDB) archives — one public no-auth ZIP per month carrying the
full HTS10 × country × district × rate-provision detail (customs value,
dutiable value, calculated duty, charges, CIF, quantities, and air/vessel/
containerized transport splits) plus the publisher's own control-total
files. Each archive is verified, hashed into the retrieval manifest, parsed
per the archives' own record layouts, and reconciled exact-integer against
the in-archive control totals (by country, by commodity, and by district of
entry). The Census International Trade API is not used by this build; it
remains an independent cross-check leg
(``tools/crosscheck_us_import_margins_api.py``).

Emits under ``--out-dir``:

- ``margins_hts10_country_month.parquet`` — the tidy HTS10 × country ×
  month margins table (the P2 generator input; API-compatible core columns
  plus the bulk-only measures),
- ``census_totals_hts10_month.parquet`` — the publisher's per-commodity
  control totals (reconciliation oracle),
- ``district_entry_month.parquet`` — the publisher's district-of-entry
  totals with names,
- ``detail/period=YYYY-MM.parquet`` — the complete monthly detail at
  publication grain (HTS10 × country × subcode × districts × rate
  provision, all monthly measures),
- ``consumer_artifact/`` — the ledger-contract fact feed at the national,
  chapter, country, chapter × country, and district-of-entry grains,
- ``build_report.json`` — window, counts, reconciliation status, artifact
  hashes.

Archives cache under ``--archive-dir`` and are never re-downloaded once
present and valid. ``--download-manifest`` optionally points at a JSONL
manifest (rows with ``file``/``retrieved_at_utc``) recording when
pre-downloaded archives were actually retrieved.

Example::

    uv run python tools/build_us_import_entry_margins.py \
        --start 2025-01 --end 2026-06 \
        --archive-dir ~/.cache/populace/us-trade/imdb \
        --out-dir out/us-import-entry-margins

Exit code: 1 on any reconciliation failure or empty result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.us_runtime.us_trade import (  # noqa: E402
    CBP_TRADE_STATS_URL,
    build_cbp_entry_fact_rows,
    build_district_entry_fact_rows,
    build_import_entry_fact_rows,
    default_generator_block,
    ensure_imdb_archive,
    latest_available_imdb_month,
    load_census_country_bridge,
    load_imdb_month,
    month_range,
    parse_cbp_trade_stats,
    summarize_imdb_month,
    write_consumer_artifact,
)
from populace.build.us_runtime.us_trade.imdb_bulk import assemble_bulk_margins  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01", help="First month (YYYY-MM).")
    parser.add_argument(
        "--end",
        default=None,
        help="Last month (YYYY-MM); default = latest published archive, probed.",
    )
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--download-manifest",
        default=None,
        type=Path,
        help=(
            "JSONL manifest from the download loop (rows with file/"
            "retrieved_at_utc) supplying retrieval timestamps for "
            "pre-downloaded archives."
        ),
    )
    parser.add_argument(
        "--skip-cbp",
        action="store_true",
        help="Skip the CBP page archive (facts then omit the entry anchors).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_at = datetime.now(UTC).isoformat(timespec="seconds")

    end = args.end or latest_available_imdb_month()
    months = month_range(args.start, end)
    print(f"[margins] window {months[0]} .. {months[-1]} ({len(months)} months)")

    retrieved_at_by_sha = _load_download_manifest(args.download_manifest)
    bridge = load_census_country_bridge()

    # A rerun must never leave artifacts from an earlier window or a prior
    # failed pass beside fresh outputs: stale detail partitions and a stale
    # success report would misrepresent this build.
    detail_dir = out_dir / "detail"
    if detail_dir.exists():
        for stale in detail_dir.glob("period=*.parquet"):
            stale.unlink()
    detail_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "build_report.json").unlink(missing_ok=True)

    parsed = []
    detail_row_count = 0
    detail_paths: dict[str, Path] = {}
    for month in months:
        archive_path, manifest_entry = ensure_imdb_archive(
            month,
            args.archive_dir,
            retrieved_at_by_sha=retrieved_at_by_sha,
        )
        month_data = load_imdb_month(archive_path, month, manifest_entry)
        detail_path = detail_dir / f"period={month}.parquet"
        month_data.detail.to_parquet(detail_path, index=False)
        detail_paths[month] = detail_path
        detail_row_count += len(month_data.detail)
        print(
            f"[margins] {month}: {len(month_data.detail)} detail rows, "
            f"{len(month_data.control_cty)} country controls, "
            f"{len(month_data.control_comm)} commodity controls, "
            f"{len(month_data.reconciliation_failures)} reconciliation failures"
        )
        # Keep only the assembly-grain summary; the full detail (3.5M rows
        # in late-year archives) is on disk and must not accumulate in
        # memory across 18 months.
        parsed.append(summarize_imdb_month(month_data))
        del month_data

    failures = [
        failure for month in parsed for failure in month.reconciliation_failures
    ]
    if failures:
        for failure in failures[:20]:
            print(f"[margins] RECONCILIATION FAIL: {failure}", file=sys.stderr)
        print(
            f"[margins] FATAL: {len(failures)} reconciliation failures",
            file=sys.stderr,
        )
        return 1

    assembly = assemble_bulk_margins(tuple(parsed), bridge)
    if assembly.margins.empty:
        print("[margins] FATAL: empty margins table", file=sys.stderr)
        return 1

    margins_path = out_dir / "margins_hts10_country_month.parquet"
    totals_path = out_dir / "census_totals_hts10_month.parquet"
    district_path = out_dir / "district_entry_month.parquet"
    assembly.margins.to_parquet(margins_path, index=False)
    assembly.census_totals.to_parquet(totals_path, index=False)
    assembly.district_entry.to_parquet(district_path, index=False)

    retrievals = list(assembly.manifest_entries)
    fact_rows = build_import_entry_fact_rows(
        assembly.margins,
        retrieval_manifest=retrievals,
        extracted_at=extracted_at,
    )
    district_rows = build_district_entry_fact_rows(
        assembly.district_entry,
        retrieval_manifest=retrievals,
        extracted_at=extracted_at,
    )
    fact_rows.extend(district_rows)

    cbp_facts = 0
    if not args.skip_cbp:
        raw_html, cbp_entry = _archive_cbp_page(out_dir, extracted_at)
        retrievals.append(cbp_entry)
        stats = parse_cbp_trade_stats(raw_html)
        cbp_rows = build_cbp_entry_fact_rows(
            stats,
            page_sha256=str(cbp_entry["sha256"]),
            retrieved_at=extracted_at,
        )
        cbp_facts = len(cbp_rows)
        fact_rows.extend(cbp_rows)

    generator = {
        **default_generator_block(months=months),
        # The Schedule C -> ISO-2 bridge determines the country dimensions
        # and therefore fact identity; pin the exact table used.
        "reference_inputs": {"census_iso_bridge_sha256": bridge.sha256},
    }
    manifest = write_consumer_artifact(
        out_dir / "consumer_artifact",
        fact_rows,
        retrieval_manifest=retrievals,
        generator=generator,
    )

    report = {
        "source": "census_imdb_bulk",
        "window": {"start": months[0], "end": months[-1], "months": len(months)},
        "detail_rows": int(detail_row_count),
        "margin_rows": int(len(assembly.margins)),
        "census_total_rows": int(len(assembly.census_totals)),
        "district_rows": int(len(assembly.district_entry)),
        "distinct_hts10": int(assembly.margins["hts10"].nunique()),
        "distinct_countries": int(assembly.margins["iso2"].nunique()),
        "distinct_districts": int(assembly.district_entry["dist_entry"].nunique()),
        "fact_rows": len(fact_rows),
        "district_fact_rows": len(district_rows),
        "cbp_fact_rows": cbp_facts,
        "facts_sha256": manifest["facts_sha256"],
        "margins_parquet_sha256": _sha256(margins_path),
        "census_totals_parquet_sha256": _sha256(totals_path),
        "district_parquet_sha256": _sha256(district_path),
        "detail_parquet_sha256": {
            month: _sha256(path) for month, path in sorted(detail_paths.items())
        },
        "reconciliation_failures": 0,
        "extracted_at": extracted_at,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "detail_parquet_sha256"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_download_manifest(path: Path | None) -> dict[tuple[str, str], str]:
    """Retrieval timestamps keyed by (filename, sha256) from the download loop.

    Binding the timestamp to the recorded hash means a file swapped after
    download can never inherit the original retrieval provenance.
    """
    if path is None or not path.exists():
        return {}
    timestamps: dict[tuple[str, str], str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        name = row.get("file")
        sha256 = row.get("sha256")
        retrieved = row.get("retrieved_at_utc") or row.get("retrieved_at")
        if name and sha256 and retrieved:
            timestamps[(str(name), str(sha256))] = str(retrieved)
    return timestamps


def _archive_cbp_page(out_dir: Path, extracted_at: str) -> tuple[bytes, dict]:
    request = urllib.request.Request(
        CBP_TRADE_STATS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "populace-us-trade-ingest"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    archive_path = out_dir / "cbp_newsroom_stats_trade.html"
    archive_path.write_bytes(raw)
    entry = {
        "source_name": "cbp_trade_stats",
        "endpoint": CBP_TRADE_STATS_URL,
        "url": CBP_TRADE_STATS_URL,
        "retrieved_at": extracted_at,
        "http_status": 200,
        "filename": archive_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    return raw, entry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
