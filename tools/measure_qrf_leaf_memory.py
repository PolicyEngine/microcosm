#!/usr/bin/env python3
"""Measure one heavy QRF fit's peak resident memory, library versus in place.

quantile-forest 1.4.2 builds a forest's zero-padded leaf array as one array per
tree and then copies the list with ``np.array(list)``, which writes every
byte, padding included. ``microcosm.fit.qrf._InPlaceLeafQRF`` fills one
preallocated array instead. This script fits the same synthetic forest with
each class, each in a fresh Python process, and reports each process's peak
resident set size (``getrusage``'s ``ru_maxrss``) before and after the fit, so
the fit's own peak growth is the difference.

Two synthetic shapes stand in for the build's heavy fits. Neither reads survey
data.

- ``wage``: the CPS-ORG ``hourly_wage`` forest. 100 trees, a continuous
  positive target and no leaf cap. The full donor pool has 119,237 rows; the
  default here is 10,000. Nearly every row has its own predictor vector, so
  trees grow about one leaf per distinct bootstrap row, but
  ``--rows-in-widest-leaf`` rows share one vector that no split separates.
  That leaf is the widest, and its bootstrap draws set the array's padded
  width. The production width is not recorded anywhere; the build's 30-35 GB
  spike suggests about 150-350.
- ``near-discrete``: a base-stage near-discrete target. 32 trees, five
  distinct values dominated by zero, and the 4,096-entry cap that
  ``_fit_forest`` applies to targets with 2 to 32 distinct values. The
  production pool has up to 207,692 rows; the default here is 10,000.

The defaults finish in a few minutes on a laptop and stay within a few GB. A
larger ``--rows`` scales memory roughly in proportion to rows times width (the
cap, for ``near-discrete``) times trees, so raise it with care::

    uv run python tools/measure_qrf_leaf_memory.py
    uv run python tools/measure_qrf_leaf_memory.py --shape near-discrete
    uv run python tools/measure_qrf_leaf_memory.py --shape wage --rows 5000 --json

The script also hashes each leaf array after the fit and reports whether the
two are byte-identical. Hashing happens after the peak is read, so it never
raises the reported peak.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

IMPLEMENTATIONS = ("library", "in-place")
SHAPES = ("wage", "near-discrete")
DEFAULT_TREES = {"wage": 100, "near-discrete": 32}
DEFAULT_ROWS = {"wage": 10_000, "near-discrete": 10_000}


def _peak_resident_bytes() -> int:
    """Return this process's peak resident set size so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _wage_donors(
    rows: int, rows_in_widest_leaf: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic CPS-ORG-like donors: seven predictors and a positive wage.

    The predictors mirror ``ORG_PREDICTORS`` in kind: annual earnings, weekly
    hours, age, sex, Hispanic origin, a four-way race code and a state code.
    Earnings are continuous, so nearly every row has its own predictor vector
    and trees grow about one leaf per distinct bootstrap row, as the build's
    trees do. The first ``rows_in_widest_leaf`` rows share one predictor
    vector (a full-time worker at $15 an hour) with differing wages; no split
    separates them, so they set the widest leaf and the array's width.
    """
    generator = np.random.default_rng(seed)
    hours = generator.choice(
        [20.0, 25.0, 30.0, 35.0, 40.0, 40.0, 40.0, 45.0, 50.0, 60.0], rows
    )
    earnings = np.exp(generator.normal(10.6, 0.7, rows))
    predictors = np.column_stack(
        [
            earnings,
            hours,
            generator.integers(16, 86, rows),
            generator.integers(0, 2, rows),
            generator.random(rows) < 0.18,
            generator.integers(1, 5, rows),
            generator.integers(1, 57, rows),
        ]
    ).astype(np.float64)
    shared = min(rows, rows_in_widest_leaf)
    predictors[:shared] = [15.0 * 40.0 * 52.0, 40.0, 30.0, 1.0, 0.0, 1.0, 6.0]
    typical_wage = predictors[:, 0] / (52.0 * predictors[:, 1])
    wage = np.round(typical_wage * np.exp(generator.normal(0.0, 0.3, rows)), 2)
    return predictors, np.maximum(wage, 0.01)


def _near_discrete_donors(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic near-discrete donors: five values, zero for most rows.

    Rows whose first predictor is below 0.8 are always zero, so trees isolate
    them in a few large pure leaves that exceed the 4,096 cap. The other rows
    draw one of the five values at random and split into many small leaves.
    """
    generator = np.random.default_rng(seed)
    predictors = generator.random((rows, 6))
    predictors[:, 1:] = np.round(predictors[:, 1:] * 20.0)
    target = generator.choice([0.0, 1 / 6, 1 / 3, 1 / 2, 1.0], rows)
    target[predictors[:, 0] < 0.8] = 0.0
    return predictors, target


def _fit_in_this_process(arguments: argparse.Namespace) -> dict[str, object]:
    """Fit one forest with ``arguments.child`` and report its memory."""
    from quantile_forest import RandomForestQuantileRegressor

    from microcosm.fit import qrf

    if arguments.shape == "wage":
        predictors, target = _wage_donors(
            arguments.rows, arguments.rows_in_widest_leaf, arguments.seed
        )
        max_samples_leaf = None
    else:
        predictors, target = _near_discrete_donors(arguments.rows, arguments.seed)
        distinct = len(np.unique(target))
        if not 2 <= distinct <= qrf._DISCRETE_Y_UNIQUE_MAX:
            raise RuntimeError(f"near-discrete donors have {distinct} values")
        max_samples_leaf = qrf._DISCRETE_Y_LEAF_BOUND

    model_class = (
        RandomForestQuantileRegressor
        if arguments.child == "library"
        else qrf._InPlaceLeafQRF
    )
    model = model_class(
        n_estimators=arguments.trees,
        max_samples_leaf=max_samples_leaf,
        random_state=arguments.seed,
        n_jobs=arguments.n_jobs,
    )
    gc.collect()
    peak_before = _peak_resident_bytes()
    started = time.perf_counter()
    model.fit(predictors, target)
    fit_seconds = time.perf_counter() - started
    peak_after = _peak_resident_bytes()

    leaves = np.asarray(model.forest_.y_train_leaves)
    digest = None
    if not arguments.no_digest:
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(leaves))).hexdigest()
    return {
        "implementation": arguments.child,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "shape": arguments.shape,
        "rows": arguments.rows,
        "trees": arguments.trees,
        "max_samples_leaf": max_samples_leaf,
        "leaf_array_shape": list(leaves.shape),
        "leaf_array_bytes": int(leaves.nbytes),
        "leaf_array_sha256": digest,
        "fit_seconds": round(fit_seconds, 3),
        "peak_resident_bytes_before_fit": peak_before,
        "peak_resident_bytes_after_fit": peak_after,
        "fit_peak_growth_bytes": peak_after - peak_before,
    }


def _child_command(arguments: argparse.Namespace, implementation: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        implementation,
        "--shape",
        arguments.shape,
        "--rows",
        str(arguments.rows),
        "--trees",
        str(arguments.trees),
        "--rows-in-widest-leaf",
        str(arguments.rows_in_widest_leaf),
        "--seed",
        str(arguments.seed),
        "--n-jobs",
        str(arguments.n_jobs),
    ]
    if arguments.no_digest:
        command.append("--no-digest")
    return command


def _gibibytes(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def _report(results: list[dict[str, object]], *, digests: bool) -> str:
    first = results[0]
    lines = [
        f"shape {first['shape']}: {first['rows']:,} donor rows, "
        f"{first['trees']} trees, max_samples_leaf={first['max_samples_leaf']}",
        "leaf array: shape "
        + " x ".join(f"{size:,}" for size in first["leaf_array_shape"])
        + f" (trees x nodes x outputs x width), "
        f"{_gibibytes(int(first['leaf_array_bytes']))} if every page were resident",
    ]
    for result in results:
        lines.append(
            f"{result['implementation']}: peak RSS "
            f"{_gibibytes(int(result['peak_resident_bytes_after_fit']))}, "
            f"of which the fit added "
            f"{_gibibytes(int(result['fit_peak_growth_bytes']))}; "
            f"fit took {result['fit_seconds']:.1f} s"
        )
    if digests and len(results) == 2:
        identical = results[0]["leaf_array_sha256"] == results[1]["leaf_array_sha256"]
        lines.append("leaf arrays byte-identical: " + ("yes" if identical else "NO"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shape", choices=SHAPES, default="wage")
    parser.add_argument(
        "--rows", type=int, help="donor rows (default 10,000 for either shape)"
    )
    parser.add_argument(
        "--trees", type=int, help="trees (default: 100 for wage, 32 near-discrete)"
    )
    parser.add_argument(
        "--rows-in-widest-leaf",
        type=int,
        default=250,
        help="wage shape only: donor rows that share one predictor vector "
        "(default 250), which sets the array's width",
    )
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="quantile-forest n_jobs (default -1, as microcosm's fits use)",
    )
    parser.add_argument(
        "--implementation",
        choices=(*IMPLEMENTATIONS, "both"),
        default="both",
        help="which class to measure (default both, library first)",
    )
    parser.add_argument(
        "--no-digest", action="store_true", help="skip hashing the leaf arrays"
    )
    parser.add_argument("--json", action="store_true", help="print JSON results")
    parser.add_argument("--child", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.rows is None:
        arguments.rows = DEFAULT_ROWS[arguments.shape]
    if arguments.trees is None:
        arguments.trees = DEFAULT_TREES[arguments.shape]
    if arguments.rows < 2 or arguments.trees < 1:
        parser.error("--rows must be at least 2 and --trees at least 1")

    if arguments.child is not None:
        print(json.dumps(_fit_in_this_process(arguments)))
        return 0

    implementations = (
        IMPLEMENTATIONS
        if arguments.implementation == "both"
        else (arguments.implementation,)
    )
    results = []
    for implementation in implementations:
        completed = subprocess.run(
            _child_command(arguments, implementation),
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    if arguments.json:
        print(json.dumps(results, indent=2))
    else:
        print(_report(results, digests=not arguments.no_digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
