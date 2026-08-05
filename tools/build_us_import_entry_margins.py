"""Build the US import-entry margin artifacts (populace#615 P1).

Pulls official Census monthly import statistics (HS10 × country: customs
value, calculated duty, dutiable value, quantities) from 2025-01 through the
latest published month, archives every response byte-for-byte with a
retrieval manifest, archives the CBP trade-statistics page for the
fiscal-year entry-summary anchors, and emits:

- ``margins_hts10_country_month.parquet`` — the full-grain tidy margins
  table (the P2 generator input),
- ``census_totals_hts10_month.parquet`` — the publisher's own ``-`` total
  rows (reconciliation oracle),
- ``consumer_artifact/`` — the ledger-contract fact feed
  (``manifest.json`` + ``consumer_facts.jsonl``) at the national, chapter,
  country, and chapter × country grains,
- ``build_report.json`` — counts, months, reconciliation status, and the
  artifact hashes.

The pull is resumable: response bytes cache under ``--cache-dir`` and cached
chapters are never re-fetched. The Census API key comes from
``--census-key`` or ``CENSUS_API_KEY`` and is never written to disk.

Example::

    CENSUS_API_KEY=... uv run python tools/build_us_import_entry_margins.py \
        --start 2025-01 \
        --cache-dir ~/.cache/populace/us-trade/census-imports-hs \
        --out-dir out/us-import-entry-margins

Exit code: 1 on any reconciliation failure or empty pull.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.us_trade import (  # noqa: E402
    CBP_TRADE_STATS_URL,
    build_cbp_entry_fact_rows,
    build_import_entry_fact_rows,
    default_generator_block,
    fetch_imports_month,
    latest_published_month,
    load_census_country_bridge,
    month_range,
    parse_cbp_trade_stats,
    write_consumer_artifact,
)
from populace.build.us_trade.census_imports import assemble_margins_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01", help="First month (YYYY-MM).")
    parser.add_argument(
        "--end",
        default=None,
        help="Last month (YYYY-MM); default = latest published month, probed.",
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--census-key",
        default=None,
        help="Census API key; default $CENSUS_API_KEY.",
    )
    parser.add_argument(
        "--skip-cbp",
        action="store_true",
        help="Skip the CBP page archive (facts then omit the entry anchors).",
    )
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.2,
        help="Pause between uncached API calls.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent chapter requests per month.",
    )
    args = parser.parse_args()

    api_key = args.census_key or os.environ.get("CENSUS_API_KEY")
    if not api_key:
        parser.error("A Census API key is required (--census-key or CENSUS_API_KEY).")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_at = datetime.now(UTC).isoformat(timespec="seconds")

    end = args.end or latest_published_month(api_key)
    months = month_range(args.start, end)
    print(f"[margins] window {months[0]} .. {months[-1]} ({len(months)} months)")

    bridge = load_census_country_bridge()
    pulled = []
    for month in months:
        pulled_month = fetch_imports_month(
            month,
            api_key,
            cache_dir=args.cache_dir,
            throttle_seconds=args.throttle_seconds,
            max_workers=args.max_workers,
        )
        pulled.append(pulled_month)
        print(
            f"[margins] {month}: {len(pulled_month.country_rows)} country rows, "
            f"{len(pulled_month.total_rows)} census totals, "
            f"{len(pulled_month.reconciliation_failures)} reconciliation failures"
        )
    pull = assemble_margins_table(tuple(pulled), bridge)

    if pull.margins.empty:
        print("[margins] FATAL: empty margins table", file=sys.stderr)
        return 1
    failures = pull.reconciliation_failures
    if failures:
        for failure in failures[:20]:
            print(f"[margins] RECONCILIATION FAIL: {failure}", file=sys.stderr)
        print(
            f"[margins] FATAL: {len(failures)} reconciliation failures",
            file=sys.stderr,
        )
        return 1

    margins_path = out_dir / "margins_hts10_country_month.parquet"
    totals_path = out_dir / "census_totals_hts10_month.parquet"
    pull.margins.to_parquet(margins_path, index=False)
    pull.census_totals.to_parquet(totals_path, index=False)

    fact_rows = build_import_entry_fact_rows(
        pull.margins,
        retrieval_manifest=pull.manifest_entries,
        extracted_at=extracted_at,
    )
    retrievals = list(pull.manifest_entries)

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

    manifest = write_consumer_artifact(
        out_dir / "consumer_artifact",
        fact_rows,
        retrieval_manifest=retrievals,
        generator=default_generator_block(months=months),
    )

    report = {
        "window": {"start": months[0], "end": months[-1], "months": len(months)},
        "margin_rows": int(len(pull.margins)),
        "census_total_rows": int(len(pull.census_totals)),
        "distinct_hts10": int(pull.margins["hts10"].nunique()),
        "distinct_countries": int(pull.margins["iso2"].nunique()),
        "fact_rows": len(fact_rows),
        "cbp_fact_rows": cbp_facts,
        "facts_sha256": manifest["facts_sha256"],
        "margins_parquet_sha256": _sha256(margins_path),
        "census_totals_parquet_sha256": _sha256(totals_path),
        "reconciliation_failures": 0,
        "extracted_at": extracted_at,
    }
    (out_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


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
