"""Time the ACS source owners on actual archives with the memo off, cold or warm.

One phase per process, so a warm phase proves reuse across runs, not within one:

    python bench_acs_owners.py --acs SOURCES/acs --work WORK --phase off \
        --memo-root MEMO --memo-key KEY --out off.json
    ... --phase cold ...   # empty MEMO
    ... --phase warm ...   # the MEMO the cold phase wrote

Each phase issues the complete ACS source catalogue, then the native ACS
coverage for a deterministic ~1/1000 household selection drawn from that
catalogue (``sha256(seed || SERIALNO) mod 1000 == 0``). This is an owner-level
benchmark: it is not the graph run, selects no ASEC records and builds no
release. It records only digests, counts, timings and memo counters -- never a
source value. Captures stay under ``--work/<phase>`` for the operator to remove. The phase refuses to start below ``--min-available-gib`` of
available memory and stops itself if its own RSS passes ``--rss-limit-gib`` or
available memory falls below ``--floor-available-gib``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import psutil


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _watchdog(rss_limit, floor, done):
    process = psutil.Process()
    while not done.wait(1.0):
        if process.memory_info().rss > rss_limit:
            os.write(2, b"RSS_LIMIT\n")
            os._exit(137)
        if psutil.virtual_memory().available < floor:
            os.write(2, b"AVAILABLE_MEMORY_FLOOR\n")
            os._exit(138)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--acs", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--phase", choices=("off", "cold", "warm"), required=True)
    parser.add_argument("--memo-root", type=Path, required=True)
    parser.add_argument("--memo-key", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", default="native-source-auth-memo-20260923")
    parser.add_argument("--min-available-gib", type=float, default=25.0)
    parser.add_argument("--floor-available-gib", type=float, default=10.0)
    parser.add_argument("--rss-limit-gib", type=float, default=24.0)
    args = parser.parse_args(argv)

    available = psutil.virtual_memory().available
    if available < args.min_available_gib * 1024**3:
        raise SystemExit(f"INSUFFICIENT_AVAILABLE_MEMORY {available / 1024**3:.1f}")
    if args.phase == "cold" and args.memo_root.exists():
        raise SystemExit("COLD_PHASE_NEEDS_A_NEW_MEMO_ROOT")
    if args.phase == "warm" and not args.memo_root.exists():
        raise SystemExit("WARM_PHASE_NEEDS_THE_COLD_MEMO_ROOT")
    work = args.work / args.phase
    work.mkdir(parents=True, exist_ok=False)
    (work / "catalogue").mkdir()
    done = threading.Event()
    threading.Thread(
        target=_watchdog,
        args=(
            args.rss_limit_gib * 1024**3,
            args.floor_available_gib * 1024**3,
            done,
        ),
        daemon=True,
    ).start()

    from microcosm.build.us_runtime import acs_native_coverage_binding as native
    from microcosm.build.us_runtime import acs_population_catalogue as catalogue
    from microcosm.build.us_runtime import source_memo as memo

    report = {
        "schema": "microcosm.native-source-auth-memo.acs-owner-bench.v1",
        "phase": args.phase,
        "python": sys.version,
        "pid": os.getpid(),
        "available_gib_at_start": round(available / 1024**3, 1),
        "steps": {},
    }

    def step(name, function):
        cpu, wall = time.process_time(), time.perf_counter()
        value = function()
        report["steps"][name] = {
            "cpu_seconds": round(time.process_time() - cpu, 3),
            "wall_seconds": round(time.perf_counter() - wall, 3),
            "memo": memo.statistics(),
        }
        memo.reset_statistics()
        return value

    context = (
        memo.source_memo(None)
        if args.phase == "off"
        else memo.source_memo(args.memo_root, key_path=args.memo_key)
    )
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    with context:
        issued = step(
            "catalogue",
            lambda: catalogue.issue_acs_source_catalogue(
                args.acs, snapshot_root=work / "catalogue"
            ),
        )
        receipt = issued.to_bytes()
        owned = catalogue._lookup(issued)
        # An independent digest of the records the issuer actually holds (a
        # warm phase holds decoded memo values), one canonical line per record.
        records = hashlib.sha256()
        for part in (owned.records, owned.vacancies):
            for record in part:
                records.update(native.coverage._json(record, 1024**2) + b"\n")
            records.update(b"--\n")
        keys = tuple(
            sorted(
                row.key.native_id
                for row in issued.households
                if int(_sha(f"{args.seed}\0{row.key.native_id}".encode())[:15], 16)
                % 1000
                == 0
            )
        )
        report["catalogue"] = {
            "receipt_sha256": _sha(receipt),
            "records_sha256": records.hexdigest(),
            "counts": issued.receipt["counts"],
        }
        del records, owned, issued
        (work / "native").mkdir()
        coverage = step(
            "native_coverage",
            lambda: native.issue_acs_native_coverage(
                args.acs, snapshot_root=work / "native", serialnos=keys
            ),
        )
        report["native_coverage"] = {
            "selected_households": len(keys),
            "selected_keys_sha256": _sha(json.dumps(keys).encode()),
            "payload_sha256": _sha(coverage.payload),
            "frame_sha256": native._frame_sha256(coverage.frame),
            "households": int(coverage.frame.n("household")),
            "persons": int(coverage.frame.n("person")),
        }
        del coverage
    report["total"] = {
        "cpu_seconds": round(time.process_time() - started_cpu, 3),
        "wall_seconds": round(time.perf_counter() - started_wall, 3),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    done.set()
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
