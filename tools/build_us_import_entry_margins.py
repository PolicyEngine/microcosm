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
hidden staging sibling of ``--out-dir`` and the destination is replaced
only after every artifact, gate, and report has been produced. On macOS
and Linux a real-directory layout is replaced by a single-syscall
directory exchange (``renamex_np(RENAME_SWAP)`` /
``renameat2(RENAME_EXCHANGE)``), so a reader holding the published path
never observes it missing and no crash point leaves ``--out-dir`` absent.
Where no exchange exists the layout is symlink-based instead: the public
name is a symlink to a hidden versioned set directory, and publication
retargets it with one atomic ``rename`` of a prepared sibling link — the
public name never vacates on this path either, on any POSIX filesystem.
The only windowed transition is the one-time migration of a legacy real
directory on an exchange-less filesystem (plain renames cannot atomically
replace a populated directory); it installs the symlink layout so it
never recurs. Every destructive step is preceded by a durable (fsynced)
recovery marker naming the directories involved, and the next build rolls
any interruption forward (staged set intact) or back (previous
publication restored) before doing anything else — including reclaiming a
previous set stranded between an exchange and its cleanup. The staged
tree itself is fsynced file by file before publication begins, so the
recovery assumption — a staged set on disk is complete — holds through
power loss. An advisory ``flock`` on a persistent lockfile serializes
publishers toward the same ``--out-dir``: concurrent publication is
refused, and a dead publisher's lock is released by the kernel with its
process, so there is no staleness protocol to race. The public name is
only ever operated on when it is a real directory or a symlink this
publisher installed (a ``.<name>.set-*`` sibling); any other symlink is
refused untouched. A failed build removes its staging directory and
leaves any previously published artifact set byte-for-byte untouched.
There is no state in which old and new artifacts coexist under
``--out-dir``.

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
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import socket
import stat
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
    new complete publication, never a mixture — and, outside the one-time
    legacy migration below, never a missing path. Steady states:

    - real-directory layout where the OS can exchange (macOS/Linux): a
      single-syscall directory exchange;
    - symlink layout (what exchange-less publication installs): the
      public name is a symlink to a versioned set directory, retargeted
      by one atomic rename of a prepared sibling link — windowless on
      any POSIX filesystem;
    - first publication: one rename (or symlink install) onto the vacant
      name, layout chosen by probing whether this filesystem exchanges.

    A legacy real directory on a filesystem without exchange cannot be
    replaced windowlessly by plain renames, so it is migrated once —
    marker-guarded, crash-recovered — to the symlink layout, after which
    the window never recurs. Every destructive step writes a durable
    (fsynced) recovery marker first, the staged tree is fsynced file by
    file before anything moves, and the publisher lock refuses concurrent
    publication toward the same destination. A public name that is a
    symlink the publisher did not install is refused untouched — its
    target is not this publisher's to destroy.
    """
    with _publisher_lock(out_dir):
        _recover_unlocked(out_dir)
        _fsync_tree(staging)
        if not out_dir.exists() and not out_dir.is_symlink():
            # First publication: nothing can be reading the path yet, so
            # an install onto the vacant name is atomic on its own. The
            # layout is chosen for the filesystem so that exchange-less
            # deployments start symlink-based and never need migrating.
            if _exchange_supported(out_dir.parent):
                staging.rename(out_dir)
                _fsync_dir(out_dir.parent)
            else:
                _publish_symlink_retarget(staging, out_dir, old_target=None)
            return
        if out_dir.is_symlink():
            _publish_symlink_retarget(
                staging,
                out_dir,
                old_target=_owned_sibling_name(
                    out_dir, os.readlink(out_dir), kind="set"
                ),
            )
            return
        if _publish_by_exchange(staging, out_dir):
            return
        _migrate_real_dir_to_symlink_layout(staging, out_dir)


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


def _fsync_dir(path: Path) -> None:
    """Flush a directory's entry table so completed renames survive power loss."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(root: Path) -> None:
    """Fsync every file and directory under ``root``, leaves first.

    Recovery assumes a staged set found on disk is complete, so the
    staged *contents* must be durable before the first marker-guarded
    rename — the parent-directory fsyncs elsewhere persist names, not
    bytes. Without this, a power loss can leave the new name durable and
    the new tree hollow while the old set has already been reclaimed.
    """

    def refuse_traversal_failure(error: OSError) -> None:
        # Silently skipping an unreadable subtree would publish a set the
        # durability pass never saw; abort before any marker is written.
        raise error

    for dirpath, _dirnames, filenames in os.walk(
        root, topdown=False, onerror=refuse_traversal_failure
    ):
        for filename in filenames:
            fd = os.open(os.path.join(dirpath, filename), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(dirpath))


def _write_all(fd: int, data: bytes) -> None:
    """``os.write`` until every byte lands — a short write is legal, loss is not."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _owned_sibling_name(out_dir: Path, name: object, *, kind: str) -> str:
    """Admit only names this publisher itself mints beside ``out_dir``.

    Everything that reaches a destructive operation *by name* — a symlink
    target read back from the public path, a directory name read from a
    recovery marker — must be a plain basename in the publisher's own
    ``.<out>.<kind>-…`` namespace. Anything else (absolute paths,
    parent-relative escapes, foreign basenames, the public name itself)
    was not created by this publisher and is refused untouched: joining
    an absolute target under ``parent`` would resolve *to that target*,
    handing it to ``rmtree``.
    """
    text = str(name)
    expected = f".{out_dir.name}.{kind}-"
    if (
        os.sep in text
        or (os.altsep is not None and os.altsep in text)
        or not text.startswith(expected)
    ):
        raise RuntimeError(
            f"{text!r} is not a {kind} name owned by the publisher of "
            f"{out_dir} (expected a plain {expected}* basename); refusing "
            "to operate on it."
        )
    return text


def _write_marker_durably(out_dir: Path, payload: dict[str, object]) -> None:
    """Write the recovery marker so it is on disk before anything moves.

    The marker is the recovery source of truth, so it must be durable
    *before* the destructive step it guards: file bytes fully written and
    fsynced, then renamed into the marker name, then the directory entry
    fsynced. A power loss after any subsequent rename therefore always
    finds the complete marker.
    """
    marker = _publish_marker_path(out_dir)
    tmp = marker.with_name(f"{marker.name}.tmp-{uuid.uuid4().hex}")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        _write_all(fd, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.rename(marker)
    _fsync_dir(marker.parent)


def _remove_marker(out_dir: Path) -> None:
    """Withdraw the marker — the commit point that says "no recovery needed".

    Every mutation performed since the marker was written must be on
    disk before its removal can be: the parent is fsynced BEFORE the
    unlink (bounding all prior renames, deletions, and link cleanups —
    a cleanup that evaporates in a power loss after a durable marker
    removal would leave a markerless orphan), and again after it
    (bounding the removal itself). This makes every call site correct
    by construction rather than by per-site discipline.
    """
    _fsync_dir(out_dir.parent)
    _publish_marker_path(out_dir).unlink(missing_ok=True)
    _fsync_dir(out_dir.parent)


def _publisher_lock_path(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.publish-lock"


@contextlib.contextmanager
def _publisher_lock(out_dir: Path):
    """Serialize publishers of one destination with an advisory ``flock``.

    Two builds publishing toward the same ``out_dir`` would interleave
    renames and corrupt each other's recovery state, so concurrent
    publication is refused rather than attempted. The kernel owns
    liveness: the lock dies with its holder's open file description, so
    there is no staleness record to inspect, no takeover step to race
    (deciding a path is stale and then unlinking it are two operations —
    two contenders can each pass the first and then destroy each other's
    fresh lock), and no orphan to clean by hand. The lockfile persists
    between runs and is never unlinked — removing it would let a later
    opener lock a fresh inode while a prior holder still holds the
    orphaned one. Its contents are diagnostics for the human reading a
    refusal, not protocol state.
    """
    lock_path = _publisher_lock_path(out_dir)
    try:
        # O_NOFOLLOW: a symlink planted at the lock name must not hand
        # this publisher an arbitrary file to truncate and overwrite.
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o644)
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.EMLINK):
            raise RuntimeError(
                f"{lock_path} is a symlink; refusing to lock through it — "
                "the lockfile must be a regular file the publisher owns."
            ) from None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise RuntimeError(
            f"{lock_path} is not a regular file; refusing to use it as "
            "the publisher lock."
        )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = ""
            with contextlib.suppress(OSError):
                holder = " ".join(lock_path.read_text().split())
            raise RuntimeError(
                f"Another publisher holds {lock_path}"
                + (f" ({holder})" if holder else "")
                + f"; refusing concurrent publication toward {out_dir}."
            ) from None
        os.ftruncate(fd, 0)
        _write_all(
            fd,
            (
                json.dumps(
                    {
                        "host": socket.gethostname(),
                        "locked_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "pid": os.getpid(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        yield
    finally:
        os.close(fd)


def _exchange_supported(parent: Path) -> bool:
    """Probe whether this directory's filesystem can exchange directories."""
    probe_a = parent / f".exchange-probe-a-{uuid.uuid4().hex}"
    probe_b = parent / f".exchange-probe-b-{uuid.uuid4().hex}"
    probe_a.mkdir()
    probe_b.mkdir()
    try:
        return _exchange_directories(probe_a, probe_b)
    finally:
        probe_a.rmdir()
        probe_b.rmdir()


def _publish_by_exchange(staging: Path, out_dir: Path) -> bool:
    """Single-syscall exchange publish of a real-directory layout.

    The marker records the staging name *before* the exchange: the
    syscall parks the displaced previous publication under that name, so
    a hard exit between the exchange and its cleanup leaves a marker
    pointing at the orphan and recovery reclaims it. Returns ``False``
    with both directories untouched when the platform or filesystem
    cannot exchange.
    """
    _write_marker_durably(
        out_dir,
        {"mode": "exchange", "out": out_dir.name, "staging": staging.name},
    )
    try:
        exchanged = _exchange_directories(staging, out_dir)
    except BaseException:
        # An asynchronous exception can arrive AFTER the syscall
        # committed, so "moved nothing" cannot be assumed here. Make
        # whatever happened durable and leave the marker in place: the
        # exchange recovery handles both sides of the ambiguity (exactly
        # one complete set answers to the public name; the staging name
        # holds the other for disposal).
        _fsync_dir(out_dir.parent)
        raise
    if not exchanged:
        # The syscall itself reported no exchange: nothing moved, and
        # the marker guards no destructive step any more.
        _remove_marker(out_dir)
        return False
    _fsync_dir(out_dir.parent)
    shutil.rmtree(staging)
    _remove_marker(out_dir)
    return True


def _publish_symlink_retarget(
    staging: Path, out_dir: Path, *, old_target: str | None
) -> None:
    """Windowless publish on the symlink layout: one rename retargets it.

    The staged set is parked under a versioned ``.set-*`` name, a sibling
    temp link is pointed at it, and the only operation ever performed on
    the public name is the atomic ``os.rename`` of that link over it —
    there is no instant at which ``out_dir`` is missing, on any POSIX
    filesystem. With ``old_target=None`` the same steps install the
    layout onto a vacant name (first publication). A crash at any point
    leaves the prior publication readable; recovery finishes the retarget
    from the marker and disposes of the superseded set.
    """
    parent = out_dir.parent
    set_dir = parent / f".{out_dir.name}.set-{uuid.uuid4().hex}"
    link_tmp = parent / f".{out_dir.name}.linktmp-{uuid.uuid4().hex}"
    _write_marker_durably(
        out_dir,
        {
            "mode": "symlink-flip",
            "out": out_dir.name,
            "staging": staging.name,
            "set": set_dir.name,
            "link_tmp": link_tmp.name,
            "old_set": old_target,
        },
    )
    try:
        staging.rename(set_dir)
        # The set name must be durable BEFORE the public link can land on
        # it: two renames in one directory carry no ordering guarantee
        # through power loss, and a surviving link over a vanished set
        # name is a dangling publication.
        _fsync_dir(parent)
        os.symlink(set_dir.name, link_tmp)
        os.rename(link_tmp, out_dir)
    except BaseException:
        # Derive what happened from the filesystem, never from control
        # flow: an asynchronous exception (KeyboardInterrupt) can arrive
        # AFTER the final rename commits, and undoing then would move the
        # live set out from under the public link.
        if (
            out_dir.is_symlink()
            and os.readlink(out_dir) == set_dir.name
            and set_dir.is_dir()
        ):
            # Committed: finish forward exactly as the success path
            # would. The marker stays until the last step, so a second
            # interruption inside this handler leaves recovery a
            # complete record.
            _fsync_dir(parent)
            _dispose_set(out_dir, old_target)
            _remove_marker(out_dir)
            raise
        # Not committed: the previous publication is intact under the
        # public name; put the staged set back for the caller's cleanup
        # and withdraw the marker. The rollback renames must be durable
        # before the marker goes — losing the set->staging rename to a
        # power loss after a durable marker removal would strand a
        # .set-* orphan nothing records.
        if link_tmp.is_symlink():
            link_tmp.unlink()
        if set_dir.exists() and not staging.exists():
            set_dir.rename(staging)
        _fsync_dir(parent)
        _remove_marker(out_dir)
        raise
    _fsync_dir(parent)
    _dispose_set(out_dir, old_target)
    _remove_marker(out_dir)


def _dispose_set(out_dir: Path, name: object) -> None:
    """Remove a superseded set directory (or stray link) by marker name.

    The name is re-validated as a publisher-owned sibling basename here,
    at the last hand before ``rmtree`` — whatever path it arrived by.
    """
    if not name:
        return
    target = out_dir.parent / _owned_sibling_name(out_dir, name, kind="set")
    if target.is_symlink():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def _migrate_real_dir_to_symlink_layout(staging: Path, out_dir: Path) -> None:
    """One-time windowed migration of a legacy real directory, guarded.

    Plain POSIX renames cannot atomically replace a populated directory,
    so converting a legacy real-directory publication on an exchange-less
    filesystem necessarily passes through an instant with no public name:
    rename the old set away, rename the prepared link in. The marker is
    durable before anything moves and every interruption state rolls
    forward or back at the next build; the migration installs the symlink
    layout, so the window never recurs. An in-process failure of the
    final rename restores the previous publication immediately.
    """
    parent = out_dir.parent
    set_dir = parent / f".{out_dir.name}.set-{uuid.uuid4().hex}"
    link_tmp = parent / f".{out_dir.name}.linktmp-{uuid.uuid4().hex}"
    previous = parent / f".{out_dir.name}.previous-{uuid.uuid4().hex}"
    _write_marker_durably(
        out_dir,
        {
            "mode": "migrate",
            "out": out_dir.name,
            "staging": staging.name,
            "set": set_dir.name,
            "link_tmp": link_tmp.name,
            "previous": previous.name,
        },
    )
    try:
        staging.rename(set_dir)
        # Same ordering as the retarget: the set name must be durable
        # before any rename touches the public name.
        _fsync_dir(parent)
        os.symlink(set_dir.name, link_tmp)
        out_dir.rename(previous)
        os.rename(link_tmp, out_dir)
    except BaseException:
        # Derive the interruption point from the filesystem, never from
        # a local flag: an asynchronous exception can arrive between the
        # vacating rename and any flag assignment, and a flag read then
        # skips the restoration while the public name is missing.
        if (
            out_dir.is_symlink()
            and os.readlink(out_dir) == set_dir.name
            and set_dir.is_dir()
        ):
            # Committed: only disposal was left. Marker stays until the
            # last step so a second interruption leaves recovery a
            # complete record.
            _fsync_dir(parent)
            for leftover in (previous, staging):
                if leftover.exists():
                    shutil.rmtree(leftover)
            _remove_marker(out_dir)
            raise
        if not out_dir.exists() and not out_dir.is_symlink() and previous.exists():
            # Inside the window: the previous publication was vacated and
            # the new link never landed — restore it first, everything
            # else can wait. The restore must be durable before the
            # marker can be removed below, or a second power loss could
            # keep the marker deletion and lose the restore.
            previous.rename(out_dir)
            _fsync_dir(parent)
        if link_tmp.is_symlink():
            link_tmp.unlink()
        if set_dir.exists() and not staging.exists():
            set_dir.rename(staging)
        # Same discipline as the retarget rollback: every rollback
        # mutation durable before the marker recording it is removed.
        _fsync_dir(parent)
        _remove_marker(out_dir)
        raise
    _fsync_dir(parent)
    shutil.rmtree(previous)
    _remove_marker(out_dir)


def _recover_interrupted_publication(out_dir: Path) -> str | None:
    """Complete or roll back an interrupted publication, under the lock.

    Returns a description of the action taken, or ``None`` when there is
    no marker. The staged set was complete before any swap began (nothing
    is ever published earlier), so an interruption with the staged set
    still on disk rolls *forward*; only when the staged set is gone is
    the previous publication restored.
    """
    if not _publish_marker_path(out_dir).exists():
        return None
    with _publisher_lock(out_dir):
        return _recover_unlocked(out_dir)


def _recover_unlocked(out_dir: Path) -> str | None:
    marker_path = _publish_marker_path(out_dir)
    if not marker_path.exists():
        return None
    names = json.loads(marker_path.read_text())
    # Markers written before the symlink-layout fallback carry no mode
    # and follow the retired two-rename protocol.
    mode = str(names.get("mode", "legacy-two-rename"))
    if mode == "exchange":
        action = _recover_exchange(out_dir, names)
    elif mode == "symlink-flip":
        action = _recover_symlink_flip(out_dir, names)
    elif mode == "migrate":
        action = _recover_migration(out_dir, names)
    elif mode == "legacy-two-rename":
        action = _recover_legacy_two_rename(out_dir, names)
    else:
        raise RuntimeError(
            f"Publication marker {marker_path} carries unknown mode "
            f"{mode!r}; refusing to guess at recovery."
        )
    _remove_marker(out_dir)
    return action


def _recover_exchange(out_dir: Path, names: dict[str, object]) -> str:
    staging = out_dir.parent / _owned_sibling_name(
        out_dir, names["staging"], kind="staging"
    )
    if out_dir.is_symlink() or not out_dir.is_dir():
        # An exchange operates on two real directories and cannot vacate
        # the public name or turn it into a symlink; whatever produced
        # this state, it was not the publisher, and mutating anything
        # here (the staging name holds a complete displaced set) could
        # destroy data the marker never described.
        raise RuntimeError(
            f"Publication marker for {out_dir} records an exchange, but "
            "the public name is not a real directory; this state was not "
            "produced by the publisher and cannot be recovered "
            "automatically."
        )
    # Whether or not the exchange happened, exactly one complete set
    # answers to the public name and the other sits under the staging
    # name for disposal. Make the observed exchange durable before the
    # displaced set is deleted — a second power loss must not be able to
    # revert the exchange after its other side is already gone.
    _fsync_dir(out_dir.parent)
    if staging.exists():
        shutil.rmtree(staging)
    return "removed the displaced set left by an interrupted exchange"


def _recover_symlink_flip(out_dir: Path, names: dict[str, object]) -> str:
    parent = out_dir.parent
    staging = parent / _owned_sibling_name(out_dir, names["staging"], kind="staging")
    set_dir = parent / _owned_sibling_name(out_dir, names["set"], kind="set")
    link_tmp = parent / _owned_sibling_name(out_dir, names["link_tmp"], kind="linktmp")
    old_set = names.get("old_set")
    old_set_name = (
        _owned_sibling_name(out_dir, old_set, kind="set") if old_set else None
    )
    # The public name must be in a state the flip itself can produce:
    # vacant only on a first publication (no old set), otherwise a
    # symlink at the old or new set name. Anything else — a real
    # directory, a foreign target, an unexpectedly vacant name — was not
    # made by the publisher, and recovery must not overwrite or judge it.
    public_target = os.readlink(out_dir) if out_dir.is_symlink() else None
    expected_targets = {set_dir.name} | ({old_set_name} if old_set_name else set())
    valid_public = (
        public_target in expected_targets
        if public_target is not None
        else (old_set_name is None and not out_dir.exists())
    )
    if not valid_public:
        raise RuntimeError(
            f"Publication marker for {out_dir} records a symlink retarget, "
            "but the public name is not in any state the retarget can "
            "produce; refusing to recover over it. Resolve by hand and "
            "remove the marker."
        )
    if link_tmp.is_symlink() or link_tmp.exists():
        link_tmp.unlink()
    if (
        not staging.is_symlink()
        and staging.is_dir()
        and not set_dir.exists()
        and not set_dir.is_symlink()
    ):
        # Normalize first so the committed check below also heals the
        # torn power-loss state where the public rename persisted but the
        # set rename did not. Only a real staged directory is ever
        # renamed into the set name — the publisher stages nothing else.
        staging.rename(set_dir)
        _fsync_dir(parent)
    if set_dir.is_symlink() or (set_dir.exists() and not set_dir.is_dir()):
        # Only a real directory is a set. ``is_dir()`` follows a symlink
        # (an indirect route to judging a foreign target "committed"),
        # and a regular file here would be retargeted into publication
        # while the real old set is deleted behind it.
        raise RuntimeError(
            f"Publication marker for {out_dir} records a symlink retarget "
            "whose set name does not hold a real directory, which the "
            "retarget cannot have produced; refusing to recover over it. "
            "Resolve by hand and remove the marker."
        )
    if public_target == set_dir.name:
        if not set_dir.is_dir():
            # The public link outran a set that no longer exists anywhere
            # (not even under the staging name): nothing here can restore
            # a publication, and "intact" would be a lie.
            raise RuntimeError(
                f"Publication marker for {out_dir} records a symlink "
                "retarget whose public link points at a set that no "
                "longer exists; the publication cannot be recovered "
                "automatically."
            )
        # Crash after the retarget: only disposal was left. Make the
        # observed public rename durable BEFORE deleting the backup — a
        # second power loss must not be able to lose the rename after
        # the old set and marker are already gone.
        _fsync_dir(parent)
        _dispose_set(out_dir, old_set_name)
        return "finished a symlink retarget: disposed of the superseded set"
    if not set_dir.exists():
        # Nothing staged survives and the flip never happened. "Intact"
        # is only claimed after checking what the public link resolves
        # to — a target string alone can name a vanished, non-directory,
        # or indirect set.
        if old_set_name is None:
            return "cleared an aborted symlink retarget (nothing published yet)"
        old_set_path = parent / old_set_name
        if old_set_path.is_symlink() or not old_set_path.is_dir():
            raise RuntimeError(
                f"Publication marker for {out_dir} records a symlink "
                "retarget whose public link resolves to an old set that "
                "is not a real directory; the publication cannot be "
                "recovered automatically."
            )
        return "cleared an aborted symlink retarget (publication intact)"
    # Reuse the marker-recorded temp-link name: a crash between the
    # symlink and its rename then leaves a link the NEXT recovery pass
    # already knows to clear — a fresh name would orphan it unrecorded.
    _fsync_dir(parent)
    os.symlink(set_dir.name, link_tmp)
    os.rename(link_tmp, out_dir)
    _fsync_dir(parent)
    _dispose_set(out_dir, old_set_name)
    return "completed the interrupted symlink retarget from the staged set"


def _recover_migration(out_dir: Path, names: dict[str, object]) -> str:
    parent = out_dir.parent
    staging = parent / _owned_sibling_name(out_dir, names["staging"], kind="staging")
    set_dir = parent / _owned_sibling_name(out_dir, names["set"], kind="set")
    link_tmp = parent / _owned_sibling_name(out_dir, names["link_tmp"], kind="linktmp")
    previous = parent / _owned_sibling_name(out_dir, names["previous"], kind="previous")
    if out_dir.is_symlink() and os.readlink(out_dir) != set_dir.name:
        # The migration installs exactly one symlink — the public name
        # pointing at the marker-recorded set. Any other link was not
        # made by the publisher; judging it "committed" would discard the
        # real previous publication behind a foreign pointer.
        raise RuntimeError(
            f"Publication marker for {out_dir} records a layout "
            "migration, but the public name is a symlink the migration "
            "cannot have installed; refusing to recover over it. Resolve "
            "by hand and remove the marker."
        )
    if not out_dir.is_symlink() and out_dir.exists() and not out_dir.is_dir():
        # The migration moves real directories and installs one symlink;
        # a regular file (or anything else) at the public name is not a
        # state it can produce, and renaming it into ``previous`` would
        # eventually feed it to the disposal path.
        raise RuntimeError(
            f"Publication marker for {out_dir} records a layout "
            "migration, but the public name is not a directory or a "
            "publisher symlink; refusing to recover over it. Resolve by "
            "hand and remove the marker."
        )
    if link_tmp.is_symlink() or link_tmp.exists():
        link_tmp.unlink()
    if (
        not staging.is_symlink()
        and staging.is_dir()
        and not set_dir.exists()
        and not set_dir.is_symlink()
    ):
        # Normalize first so the committed check below also heals the
        # torn power-loss state where the final rename persisted but the
        # set rename did not. Only a real staged directory is ever
        # renamed into the set name.
        staging.rename(set_dir)
        _fsync_dir(parent)
    if set_dir.is_symlink() or (set_dir.exists() and not set_dir.is_dir()):
        # Only a real directory is a set: ``is_dir()`` follows a symlink
        # (an indirect route to a foreign "committed" judgment), and a
        # regular file would be installed as the publication while the
        # real previous set is deleted behind it.
        raise RuntimeError(
            f"Publication marker for {out_dir} records a layout "
            "migration whose set name does not hold a real directory, "
            "which the migration cannot have produced; refusing to "
            "recover over it. Resolve by hand and remove the marker."
        )
    if out_dir.is_symlink():
        if not set_dir.is_dir():
            raise RuntimeError(
                f"Publication marker for {out_dir} records a layout "
                "migration whose public link points at a set that no "
                "longer exists; the publication cannot be recovered "
                "automatically."
            )
        # Crash after the migration's final rename: only disposal was
        # left. Make the observed public rename durable BEFORE deleting
        # the backups — a second power loss must not be able to lose the
        # rename after the previous set and marker are already gone.
        _fsync_dir(parent)
        for leftover in (previous, staging):
            if leftover.exists():
                shutil.rmtree(leftover)
        return "finished the layout migration: disposed of the previous set"
    if set_dir.exists():
        # Roll forward: the staged set is complete, so finish installing
        # the symlink layout from wherever the crash left the migration.
        # The marker-recorded temp-link name is reused so a crash here
        # leaves nothing unrecorded.
        _fsync_dir(parent)
        if out_dir.exists():
            out_dir.rename(previous)
        os.symlink(set_dir.name, link_tmp)
        os.rename(link_tmp, out_dir)
        _fsync_dir(parent)
        if previous.exists():
            shutil.rmtree(previous)
        return "completed the interrupted layout migration from the staged set"
    if previous.exists():
        if previous.is_symlink() or not previous.is_dir():
            raise RuntimeError(
                f"Publication marker for {out_dir} records a layout "
                "migration whose previous-publication name is not a real "
                "directory; refusing to restore from it. Resolve by hand "
                "and remove the marker."
            )
        previous.rename(out_dir)
        # The restore must be durable before the marker can be removed:
        # losing this rename to a second power loss after the marker
        # deletion persisted would leave the name absent with no
        # recovery state at all.
        _fsync_dir(parent)
        return "restored the previous publication"
    if out_dir.exists():
        # This branch can be reached on the retry AFTER an interrupted
        # restore, with the restore rename observed but not yet durable;
        # the marker is about to be removed, so make the state durable
        # first.
        _fsync_dir(parent)
        return "cleared an aborted layout migration (publication intact)"
    raise RuntimeError(
        f"Publication marker for {out_dir} names directories that no "
        "longer exist; the publication cannot be recovered automatically."
    )


def _recover_legacy_two_rename(out_dir: Path, names: dict[str, object]) -> str:
    staging = out_dir.parent / _owned_sibling_name(
        out_dir, names["staging"], kind="staging"
    )
    previous = out_dir.parent / _owned_sibling_name(
        out_dir, names["previous"], kind="previous"
    )
    if out_dir.is_symlink() or (out_dir.exists() and not out_dir.is_dir()):
        # The retired two-rename protocol only ever moved real
        # directories; a symlink or regular file at the public name was
        # not its doing — judging a file "committed" here would delete
        # both the previous publication and the staged set behind it.
        raise RuntimeError(
            f"Publication marker for {out_dir} follows the retired "
            "two-rename protocol, but the public name is not a real "
            "directory that protocol can have produced; refusing to "
            "recover over it. Resolve by hand and remove the marker."
        )
    if out_dir.exists():
        # Interrupted after the swap completed but before cleanup. Make
        # the observed state durable before deleting the backups.
        _fsync_dir(out_dir.parent)
        if previous.exists():
            shutil.rmtree(previous)
        if staging.exists():
            shutil.rmtree(staging)
        return "removed leftover swap directories"
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise RuntimeError(
                f"Publication marker for {out_dir} names a staged set "
                "that is not a real directory; refusing to publish from "
                "it. Resolve by hand and remove the marker."
            )
        staging.rename(out_dir)
        _fsync_dir(out_dir.parent)
        if previous.exists():
            shutil.rmtree(previous)
        return "completed the interrupted publication from staging"
    if previous.exists():
        if previous.is_symlink() or not previous.is_dir():
            raise RuntimeError(
                f"Publication marker for {out_dir} names a previous "
                "publication that is not a real directory; refusing to "
                "restore from it. Resolve by hand and remove the marker."
            )
        previous.rename(out_dir)
        # Durable before the marker goes: see the migration restore.
        _fsync_dir(out_dir.parent)
        return "restored the previous publication"
    raise RuntimeError(
        f"Publication marker {_publish_marker_path(out_dir)} names "
        "directories that no longer exist; the publication cannot be "
        "recovered automatically."
    )


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
