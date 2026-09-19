"""Bounded invented membership timing; never reads Census records or archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(path) for path in sorted((ROOT / "packages").glob("*/src"))]

import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import pyarrow  # noqa: E402

from microcosm.build.us_runtime import acs_pums  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("original", "prepared"))
    args = parser.parse_args()
    assert Path(acs_pums.__file__).resolve().is_relative_to(ROOT)
    process = psutil.Process()
    # Key count matters: the old Arrow path rebuilds this set for every chunk.
    keys = frozenset(f"invented-{i:09d}" for i in range(200_000))
    values = pd.Series(
        [f"invented-{i:09d}" for i in range(15_000)] + ["absent", None] * 2_500,
        dtype=pd.StringDtype(storage="pyarrow"),
        name="SERIALNO",
    )
    expected = values.isin(keys)
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    cpu = time.process_time()
    lookup = acs_pums._serial_lookup(keys) if args.mode == "prepared" else None
    preparation = {
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
    }
    timings = []
    engine = None
    for i in range(20):
        started, cpu = time.perf_counter(), time.process_time()
        actual = (
            values.isin(keys)
            if args.mode == "original"
            else acs_pums._serial_isin(values, keys, lookup)
        )
        timings.append(
            {
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
            }
        )
        pd.testing.assert_series_equal(actual, expected)
        if lookup is not None:
            if i == 0:
                engine = lookup._engine
            assert lookup._engine is engine
    result = {
        "scope": "invented_arrays_only; not native preparation performance",
        "mode": args.mode,
        "keys": len(keys),
        "rows_per_chunk": len(values),
        "chunks": len(timings),
        "all_masks_match_pandas": True,
        "prepared_index_engine_reused": lookup is not None,
        "preparation": preparation,
        "first_lookup": timings[0],
        "warm_lookups": timings[1:],
        "total_cpu_seconds": preparation["cpu_seconds"]
        + sum(t["cpu_seconds"] for t in timings),
        "total_wall_seconds": preparation["wall_seconds"]
        + sum(t["wall_seconds"] for t in timings),
        "rss_before_lookup_bytes": rss_before,
        "rss_after_lookups_bytes": process.memory_info().rss,
        "whole_process_peak_rss_bytes_macos": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        # Includes shared Python strings: this is not net incremental RSS.
        "index_deep_bytes_including_shared_strings": None
        if lookup is None
        else lookup.memory_usage(deep=True),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "python": sys.version,
        "module_sha256": hashlib.sha256(
            Path(acs_pums.__file__).read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
