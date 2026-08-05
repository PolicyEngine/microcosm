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

Publication is atomic at the directory level: everything is built into a
hidden staging sibling of ``--out-dir`` and the destination directory is
replaced only after every artifact, gate, and report has been produced.
On macOS and Linux the replacement is a single-syscall directory exchange
(``renamex_np(RENAME_SWAP)`` / ``renameat2(RENAME_EXCHANGE)``), so a
reader holding the published path never observes it missing and no crash
point leaves ``--out-dir`` absent. Filesystems without an exchange fall
back to a two-rename swap guarded by a durable recovery marker: the
marker records both directory names before the first rename, and the next
build rolls an interrupted swap forward (staged set intact) or back
(previous publication restored) before doing anything else. A failed
build removes its staging directory and leaves any previously published
artifact set byte-for-byte untouched — there is no state in which old and
new artifacts coexist under ``--out-dir``.

Publishes under ``--out-dir``:

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
- ``reconciliation/period=YYYY-MM.json`` — the machine-readable record of
  every reconciliation comparison run for the month (key-set sizes,
  duplicate-key verdicts, per-measure compared/matched cell counts with
  both sides' integer totals),
- ``source_manifest.jsonl`` — one retrieval row per source byte-stream
  (archives + the CBP page): URL, sha256, size, retrieval provenance,
- ``consumer_artifact/`` — the ledger-contract fact feed at the national,
  chapter, country, chapter × country, and district-of-entry grains,
- ``cbp_newsroom_stats_trade.html`` — the archived CBP page bytes,
- ``build_report.json`` — window, counts, reconciliation totals, artifact
  hashes (including the consumer manifest's own sha256).

Archives cache under ``--archive-dir`` and are never re-downloaded once
present and valid. ``--download-manifest`` optionally points at a JSONL
manifest (rows with ``file``/``sha256``/``retrieved_at_utc``) recording
when pre-downloaded archives were actually retrieved; adopted archives
without a manifest match carry no ``retrieved_at`` at all, and the
consumer-artifact writer refuses to publish facts without retrieval
provenance, so supplying the manifest is required in practice for
pre-downloaded archives.

Example::

    uv run python tools/build_us_import_entry_margins.py \
        --start 2025-01 --end 2026-06 \
        --archive-dir ~/.cache/populace/us-trade/imdb \
        --out-dir out/us-import-entry-margins

Exit code: 1 on any reconciliation failure or empty result (nothing is
published in that case).
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import uuid
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
from populace.build.us_runtime.us_trade.imdb_bulk import (  # noqa: E402
    assemble_bulk_margins,
)


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
            "sha256/retrieved_at_utc) supplying retrieval timestamps for "
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
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    recovery = _recover_interrupted_publication(out_dir)
    if recovery:
        print(f"[margins] recovered interrupted publication: {recovery}")
    # All work happens in a staging sibling (same filesystem, so the final
    # exchange/rename is atomic); the destination is only ever a complete
    # artifact set. A failed build leaves any previous publication untouched.
    staging = out_dir.parent / f".{out_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        exit_code = _build(args, staging)
        if exit_code == 0:
            _publish_atomically(staging, out_dir)
            print(f"[margins] published atomically to {out_dir}")
        return exit_code
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _build(args: argparse.Namespace, staging: Path) -> int:
    extracted_at = datetime.now(UTC).isoformat(timespec="seconds")

    end = args.end or latest_available_imdb_month()
    months = month_range(args.start, end)
    print(f"[margins] window {months[0]} .. {months[-1]} ({len(months)} months)")

    retrieved_at_by_sha = _load_download_manifest(args.download_manifest)
    bridge = load_census_country_bridge()

    detail_dir = staging / "detail"
    detail_dir.mkdir(parents=True)
    reconciliation_dir = staging / "reconciliation"
    reconciliation_dir.mkdir(parents=True)

    parsed = []
    detail_row_count = 0
    detail_paths: dict[str, Path] = {}
    reconciliation_paths: dict[str, Path] = {}
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
        reconciliation_path = reconciliation_dir / f"period={month}.json"
        reconciliation_path.write_text(
            json.dumps(month_data.reconciliation_evidence, indent=2, sort_keys=True)
            + "\n"
        )
        reconciliation_paths[month] = reconciliation_path
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
            f"[margins] FATAL: {len(failures)} reconciliation failures; "
            "nothing was published",
            file=sys.stderr,
        )
        return 1

    assembly = assemble_bulk_margins(tuple(parsed), bridge)
    if assembly.margins.empty:
        print(
            "[margins] FATAL: empty margins table; nothing was published",
            file=sys.stderr,
        )
        return 1

    margins_path = staging / "margins_hts10_country_month.parquet"
    totals_path = staging / "census_totals_hts10_month.parquet"
    district_path = staging / "district_entry_month.parquet"
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
        raw_html, cbp_entry = _archive_cbp_page(staging)
        retrievals.append(cbp_entry)
        stats = parse_cbp_trade_stats(raw_html)
        cbp_rows = build_cbp_entry_fact_rows(
            stats,
            page_sha256=str(cbp_entry["sha256"]),
            # The facts carry the manifest's own retrieval moment — captured
            # at the HTTP read, not the build start.
            retrieved_at=str(cbp_entry["retrieved_at"]),
            source_file=str(cbp_entry["filename"]),
        )
        cbp_facts = len(cbp_rows)
        fact_rows.extend(cbp_rows)

    source_manifest_path = staging / "source_manifest.jsonl"
    source_manifest_path.write_text(
        "".join(
            json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
            for entry in retrievals
        )
    )

    generator = {
        **default_generator_block(months=months),
        # The Schedule C -> ISO-2 bridge determines the country dimensions
        # and therefore fact identity; pin the exact table used.
        "reference_inputs": {"census_iso_bridge_sha256": bridge.sha256},
    }
    manifest = write_consumer_artifact(
        staging / "consumer_artifact",
        fact_rows,
        retrieval_manifest=retrievals,
        generator=generator,
    )

    reconciliation_totals = {
        month.month: month.reconciliation_evidence.get("failure_count", 0)
        for month in parsed
    }
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
        "consumer_manifest_sha256": _sha256(
            staging / "consumer_artifact" / "manifest.json"
        ),
        "margins_parquet_sha256": _sha256(margins_path),
        "census_totals_parquet_sha256": _sha256(totals_path),
        "district_parquet_sha256": _sha256(district_path),
        "detail_parquet_sha256": {
            month: _sha256(path) for month, path in sorted(detail_paths.items())
        },
        "reconciliation_evidence_sha256": {
            month: _sha256(path) for month, path in sorted(reconciliation_paths.items())
        },
        "source_manifest_sha256": _sha256(source_manifest_path),
        "reconciliation_failures": 0,
        "reconciliation_failures_by_month": reconciliation_totals,
        "extracted_at": extracted_at,
    }
    (staging / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("detail_parquet_sha256", "reconciliation_evidence_sha256")
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _publish_atomically(staging: Path, out_dir: Path) -> None:
    """Replace ``out_dir`` with the staged artifact set in one transition.

    Consumers only ever observe the previous complete publication or the
    new complete publication, never a mixture. Where the OS offers an
    atomic directory exchange the swap is a single syscall — there is no
    instant at which ``out_dir`` does not exist, so a reader can never see
    ``ENOENT`` and no crash point strands either artifact set. Filesystems
    without an exchange fall back to a marker-guarded two-rename swap
    whose interruption states are all recovered by
    :func:`_recover_interrupted_publication` at the next build.
    """
    _recover_interrupted_publication(out_dir)
    if not out_dir.exists():
        # First publication: nothing can be reading the path yet and a
        # plain same-directory rename is atomic on its own.
        staging.rename(out_dir)
        return
    if _exchange_directories(staging, out_dir):
        # Single-syscall swap succeeded; ``staging`` now holds the
        # previous publication.
        shutil.rmtree(staging)
        return
    _publish_with_recovery_marker(staging, out_dir)


#: macOS <sys/stdio.h> RENAME_SWAP and Linux <linux/fs.h> RENAME_EXCHANGE:
#: both request "exchange the two names atomically" from rename-with-flags.
_RENAME_SWAP_DARWIN = 0x00000002
_RENAME_EXCHANGE_LINUX = 2
_AT_FDCWD_LINUX = -100


def _exchange_directories(staging: Path, out_dir: Path) -> bool:
    """Atomically exchange two sibling directories in one syscall.

    Uses macOS ``renamex_np(RENAME_SWAP)`` or Linux
    ``renameat2(RENAME_EXCHANGE)``. Returns ``False`` when the platform,
    libc, or filesystem cannot exchange (the caller then uses the
    marker-guarded fallback); real failures on a supporting filesystem
    raise.
    """
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return False
    source = os.fsencode(staging)
    target = os.fsencode(out_dir)
    if sys.platform == "darwin":
        exchange = getattr(libc, "renamex_np", None)
        if exchange is None:
            return False
        result = exchange(source, target, _RENAME_SWAP_DARWIN)
    elif sys.platform.startswith("linux"):
        exchange = getattr(libc, "renameat2", None)
        if exchange is None:
            return False
        result = exchange(
            _AT_FDCWD_LINUX, source, _AT_FDCWD_LINUX, target, _RENAME_EXCHANGE_LINUX
        )
    else:
        return False
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.ENOTSUP, errno.EINVAL, errno.ENOSYS):
        return False
    raise OSError(error, os.strerror(error), str(out_dir))


def _publish_marker_path(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.publish-recovery.json"


def _publish_with_recovery_marker(staging: Path, out_dir: Path) -> None:
    """Two-rename swap guarded by a durable recovery marker.

    Without an atomic exchange there is a window between the two renames
    in which ``out_dir`` does not exist, and a crash inside it strands the
    previous publication under its ``.previous-*`` name. The marker
    records all three directory names before the first rename and is
    removed only after the swap completes, so
    :func:`_recover_interrupted_publication` can roll any interruption
    forward or back; an in-process failure of the second rename restores
    the previous publication immediately.
    """
    previous = out_dir.parent / f".{out_dir.name}.previous-{uuid.uuid4().hex}"
    marker = _publish_marker_path(out_dir)
    marker.write_text(
        json.dumps(
            {
                "out": out_dir.name,
                "staging": staging.name,
                "previous": previous.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    out_dir.rename(previous)
    try:
        staging.rename(out_dir)
    except BaseException:
        previous.rename(out_dir)
        marker.unlink()
        raise
    shutil.rmtree(previous)
    marker.unlink()


def _recover_interrupted_publication(out_dir: Path) -> str | None:
    """Complete or roll back a marker-guarded swap that was interrupted.

    Returns a description of the action taken, or ``None`` when there is
    no marker. The staged set was complete before the swap began (nothing
    is ever swapped in earlier), so an interruption with the staged set
    still on disk rolls *forward*; only when the staged set is gone is the
    previous publication restored.
    """
    marker = _publish_marker_path(out_dir)
    if not marker.exists():
        return None
    names = json.loads(marker.read_text())
    staging = out_dir.parent / str(names["staging"])
    previous = out_dir.parent / str(names["previous"])
    if out_dir.exists():
        # Interrupted after the swap completed but before cleanup.
        if previous.exists():
            shutil.rmtree(previous)
        if staging.exists():
            shutil.rmtree(staging)
        action = "removed leftover swap directories"
    elif staging.exists():
        staging.rename(out_dir)
        if previous.exists():
            shutil.rmtree(previous)
        action = "completed the interrupted publication from staging"
    elif previous.exists():
        previous.rename(out_dir)
        action = "restored the previous publication"
    else:
        raise RuntimeError(
            f"Publication marker {marker} names directories that no longer "
            "exist; the publication cannot be recovered automatically."
        )
    marker.unlink()
    return action


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


def _archive_cbp_page(staging: Path) -> tuple[bytes, dict]:
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
        # Retrieval provenance is the moment the bytes were read, not the
        # build start; the same value flows into the facts.
        retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    archive_path = staging / "cbp_newsroom_stats_trade.html"
    archive_path.write_bytes(raw)
    entry = {
        "source_name": "cbp_trade_stats",
        "endpoint": CBP_TRADE_STATS_URL,
        "url": CBP_TRADE_STATS_URL,
        "retrieved_at": retrieved_at,
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
