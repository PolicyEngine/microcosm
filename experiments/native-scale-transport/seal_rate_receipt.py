"""Per-dtype cost of the per-column frame seal against the cell-at-a-time walk.

Both sides are the shipped code: the walk is ``_frame_cell_encode`` plus a
newline per cell, exactly as ``_frame_identity`` fed the digest before this
change, and the candidate is ``_cells_blob``. Byte equality is asserted for
every column before either is timed, so a faster wrong answer cannot be
reported. Columns are invented; nothing is read and nothing but the named output
file is written. Not a build, not a certification, not release eligible.

    <venv>/bin/python seal_rate_receipt.py <out.json> [rows]
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path[:0] = [
    str(path)
    for path in sorted(
        (pathlib.Path(__file__).resolve().parents[2] / "packages").glob("*/src")
    )
]

from microcosm.build.us_runtime import (  # noqa: E402
    survey_population_preparation as owner,
)

# The frame's actual dtype census, read from the pilot's graph.json by the
# 2026-09-16 cost attribution: Int64 154 columns, float64 54, int64 30,
# string 28, bool 1, boolean 2.
CENSUS = {
    "Int64": 154,
    "float64": 54,
    "int64": 30,
    "string": 28,
    "bool": 1,
    "boolean": 2,
}


def _walk(series):
    out = bytearray()
    for value in series:
        out += owner._frame_cell_encode(value)
        out += b"\n"
    return bytes(out)


def _columns(rows, generator):
    money = np.where(
        generator.random(rows) < 0.85, 0.0, generator.gamma(2, 15000, rows)
    )
    integers = generator.integers(-(10**6), 10**6, size=rows, dtype=np.int64)
    missing = generator.random(rows) < 0.1
    return {
        "float64": pd.Series(np.round(money, 2)),
        "int64": pd.Series(integers),
        "Int64": pd.Series(pd.array(np.where(missing, None, integers), dtype="Int64")),
        "bool": pd.Series(generator.random(rows) < 0.5),
        "boolean": pd.Series(
            pd.array(
                np.where(missing, None, generator.random(rows) < 0.5), dtype="boolean"
            )
        ),
        "string": pd.Series(
            pd.array(
                generator.choice(["CA", "NY", "TX", "MT", "acs", "asec"], size=rows),
                dtype="string",
            )
        ),
    }


def _best(function, series, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        started = time.process_time()
        function(series)
        best = min(best, time.process_time() - started)
    return best


def main():
    out = pathlib.Path(sys.argv[1])
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    generator = np.random.default_rng(20260917)
    columns = _columns(rows, generator)
    result = {
        "rows_per_column": rows,
        "loadavg": os.getloadavg(),
        "dtype_census_from_graph_json": CENSUS,
        "columns": {},
    }
    for name, series in columns.items():
        assert owner._cells_blob(series) == _walk(series), name
        walk = _best(_walk, series)
        blob = _best(owner._cells_blob, series)
        result["columns"][name] = {
            "bytes_identical": True,
            "walk_microseconds_per_cell": round(walk / rows * 1e6, 4),
            "per_column_microseconds_per_cell": round(blob / rows * 1e6, 4),
            "ratio": round(walk / blob, 2),
        }
    weighted_walk = sum(
        CENSUS[name] * result["columns"][name]["walk_microseconds_per_cell"]
        for name in CENSUS
    ) / sum(CENSUS.values())
    weighted_blob = sum(
        CENSUS[name] * result["columns"][name]["per_column_microseconds_per_cell"]
        for name in CENSUS
    ) / sum(CENSUS.values())
    result["blended_on_the_frame_census"] = {
        "walk_microseconds_per_cell": round(weighted_walk, 4),
        "per_column_microseconds_per_cell": round(weighted_blob, 4),
        "ratio": round(weighted_walk / weighted_blob, 2),
    }
    result["release_eligible"] = False
    result["scope"] = (
        "Per-dtype rate on invented columns, both sides shipped code, byte "
        "equality asserted before timing. Not a build, not a certification, "
        "not release eligible."
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
