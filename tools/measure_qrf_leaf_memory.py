#!/usr/bin/env python3
"""Measure one heavy QRF fit's peak memory, quantile-forest's leaf array versus in place.

quantile-forest 1.4.2 builds a forest's zero-padded leaf array as one array per
tree and then copies the list with ``np.array(list)``, which writes every
byte, padding included. ``microcosm.fit.qrf._InPlaceLeafQRF`` fills one
preallocated array instead. This script fits the same synthetic forest through
microcosm's ``_fit_forest`` (the weighted resample, the near-discrete cap and
the forest), once with each class and each in a fresh Python process. For the
library run the child process replaces the class ``_fit_forest`` builds with
quantile-forest's own. Each child reports its peak memory just before and just
after the fit, so the fit's own peak growth is the difference.

Peak memory is read as follows:

- macOS: the process's lifetime peak physical footprint
  (``proc_pid_rusage``'s ``ri_lifetime_max_phys_footprint``), which counts
  pages macOS has compressed under memory pressure. The resident set size
  (``ru_maxrss``) is reported too, but it drops when pages are compressed, so
  on a busy machine it can read lower than the memory the fit used.
- Linux: the peak resident set size (``ru_maxrss``). Pages swapped out under
  memory pressure are not counted, so measure on a machine with free memory.

The result depends on the platform. On macOS, reading a never-written page of
an ``np.zeros`` array makes it resident; on Linux it maps the shared zero page,
which is not counted. Where Linux transparent huge pages are enabled for large
arrays, a first write makes a whole 2 MiB region resident, which shrinks the
in-place saving. The report records the platform, the page size and the huge
page mode.

Two synthetic shapes stand in for the build's heavy fits. Neither reads survey
data. Donor weights are lognormal, so ``_fit_forest``'s weighted resample
duplicates rows as it does for weighted survey donors.

- ``wage``: the CPS-ORG ``hourly_wage`` forest. 100 trees, a continuous
  positive target and no leaf cap. The full donor pool has 119,237 rows; the
  default here is 10,000. Earnings are continuous, so most rows have their
  own predictor vector, but ``--rows-in-widest-leaf`` rows share one vector
  that no split separates. That leaf is the widest, and its draws set the
  array's padded width. The production width is not recorded anywhere, so
  this value is a parameter, not a measurement.
- ``near-discrete``: a base-stage near-discrete target. 32 trees and five
  distinct values dominated by zero, so ``_fit_forest`` applies its
  4,096-entry cap. The production pool has up to 207,692 rows; the default
  here is 10,000. About 80% of rows are zero in a region the trees isolate in
  one leaf; from about 5,100 rows that leaf holds more than 4,096 draws, so
  the library's shuffle and truncation run. The width is 4,096 at any size.

The defaults finish in under a minute on a laptop and need a few GB. Memory
grows roughly in proportion to rows times width (the cap, for
``near-discrete``) times trees, so raise ``--rows`` with care::

    uv run python tools/measure_qrf_leaf_memory.py
    uv run python tools/measure_qrf_leaf_memory.py --shape near-discrete
    uv run python tools/measure_qrf_leaf_memory.py --shape wage --rows 5000 --json

Set ``POPULACE_FIT_N_JOBS`` to fix quantile-forest's thread count, as for a
build; the report records the value each fit used.

The script also hashes each leaf array after the fit and reports whether the
two arrays have the same shape, dtype and bytes. Hashing happens after the
peak is read, so it never raises the reported peak, but on macOS it reads
every page and so raises the process's real memory to the array's full size.
Pass ``--no-digest`` for large runs.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
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
#: scikit-learn accepts integer seeds below 2**32.
SEED_LIMIT = 2**32
_RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    """macOS ``struct rusage_info_v4`` from ``<sys/resource.h>``."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
            "ri_child_user_time",
            "ri_child_system_time",
            "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups",
            "ri_child_pageins",
            "ri_child_elapsed_abstime",
            "ri_diskio_bytesread",
            "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default",
            "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background",
            "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time",
            "ri_serviced_system_time",
            "ri_logical_writes",
            "ri_lifetime_max_phys_footprint",
            "ri_instructions",
            "ri_cycles",
            "ri_billed_energy",
            "ri_serviced_energy",
            "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


def _peak_resident_bytes() -> int:
    """Return this process's peak resident set size so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kibibytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _peak_footprint_bytes() -> int | None:
    """Return macOS's lifetime peak physical footprint, or None elsewhere."""
    if sys.platform != "darwin":
        return None
    library = ctypes.CDLL("/usr/lib/libproc.dylib")
    information = _RusageInfoV4()
    if library.proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(information)):
        raise OSError("proc_pid_rusage failed")
    return int(information.ri_lifetime_max_phys_footprint)


def _peak_memory() -> dict[str, int | None]:
    return {
        "resident_set": _peak_resident_bytes(),
        "physical_footprint": _peak_footprint_bytes(),
    }


def _huge_page_mode() -> str | None:
    """Return Linux's transparent huge page setting, or None elsewhere."""
    path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _environment() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "page_size_bytes": resource.getpagesize(),
        "transparent_huge_pages": _huge_page_mode(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "quantile-forest")
        },
    }


def _wage_donors(
    rows: int, rows_in_widest_leaf: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic CPS-ORG-like donors: seven predictors and a positive wage.

    The predictors mirror ``ORG_PREDICTORS`` in kind: annual earnings, weekly
    hours, age, sex, Hispanic origin, a four-way race code and a state code.
    Earnings are continuous, so most rows have their own predictor vector. The
    first ``rows_in_widest_leaf`` rows share one predictor vector (a full-time
    worker at $15 an hour) with differing wages; no split separates them, so
    they make the widest leaf and set the array's width.
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
    predictors[:rows_in_widest_leaf] = [15.0 * 40.0 * 52.0, 40.0, 30.0, 1, 0, 1, 6]
    typical_wage = predictors[:, 0] / (52.0 * predictors[:, 1])
    wage = np.round(typical_wage * np.exp(generator.normal(0.0, 0.3, rows)), 2)
    return predictors, np.maximum(wage, 0.01)


def _near_discrete_donors(rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic near-discrete donors: five values, zero for most rows.

    Rows whose first predictor is below 0.8 are always zero, and trees isolate
    them in one large leaf. The other rows draw one of the five values at
    random and split into many small leaves.
    """
    generator = np.random.default_rng(seed)
    predictors = generator.random((rows, 6))
    predictors[:, 1:] = np.round(predictors[:, 1:] * 20.0)
    target = generator.choice([0.0, 1 / 6, 1 / 3, 1 / 2, 1.0], rows)
    target[predictors[:, 0] < 0.8] = 0.0
    return predictors, target


def _donors(arguments: argparse.Namespace) -> tuple[np.ndarray, ...]:
    """Return predictors, target and lognormal donor weights."""
    if arguments.shape == "wage":
        predictors, target = _wage_donors(
            arguments.rows, arguments.rows_in_widest_leaf, arguments.seed
        )
    else:
        predictors, target = _near_discrete_donors(arguments.rows, arguments.seed)
    weights = np.random.default_rng(arguments.seed + 1).lognormal(
        0.0, 0.5, arguments.rows
    )
    return predictors, target, weights


def _fit_in_this_process(arguments: argparse.Namespace) -> dict[str, object]:
    """Fit one forest with ``arguments.child`` and report its memory."""
    from quantile_forest import RandomForestQuantileRegressor

    from microcosm.fit import qrf

    if arguments.child == "library":
        # This process exists only for this measurement; nothing else uses qrf.
        qrf._InPlaceLeafQRF = RandomForestQuantileRegressor
    predictors, target, weights = _donors(arguments)
    columns = tuple(f"predictor_{index}" for index in range(predictors.shape[1]))

    gc.collect()
    before = _peak_memory()
    started = time.perf_counter()
    # _fit_forest prints a note for near-discrete targets; stdout carries the
    # result, so send the note to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        forest = qrf._fit_forest(
            predictors,
            target,
            columns,
            weights,
            seed=arguments.seed,
            n_estimators=arguments.trees,
            max_samples_leaf=None,
            rng=np.random.default_rng(arguments.seed),
        )
    fit_seconds = time.perf_counter() - started
    after = _peak_memory()

    model = forest.model
    leaves = np.asarray(model.forest_.y_train_leaves)
    digest = None
    if not arguments.no_digest:
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(leaves))).hexdigest()
    measure = "physical_footprint" if after["physical_footprint"] else "resident_set"
    return {
        "implementation": arguments.child,
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "shape": arguments.shape,
        "rows": arguments.rows,
        "trees": arguments.trees,
        "rows_in_widest_leaf": (
            arguments.rows_in_widest_leaf if arguments.shape == "wage" else None
        ),
        "seed": arguments.seed,
        "n_jobs": model.n_jobs,
        "max_samples_leaf": model.max_samples_leaf,
        "leaf_array_shape": list(leaves.shape),
        "leaf_array_dtype": str(leaves.dtype),
        "leaf_array_bytes": int(leaves.nbytes),
        "leaf_array_sha256": digest,
        "fit_seconds": round(fit_seconds, 3),
        "memory_measure": measure,
        "peak_bytes_before_fit": before[measure],
        "peak_bytes_after_fit": after[measure],
        "fit_peak_growth_bytes": after[measure] - before[measure],
        "peak_resident_set_bytes_before_fit": before["resident_set"],
        "peak_resident_set_bytes_after_fit": after["resident_set"],
        "environment": _environment(),
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
    ]
    if arguments.no_digest:
        command.append("--no-digest")
    return command


def _size(value: int) -> str:
    if abs(value) >= 2**30:
        return f"{value / 2**30:.2f} GiB"
    return f"{value / 2**20:.1f} MiB"


def _array_description(result: dict[str, object]) -> str:
    shape = " x ".join(f"{size:,}" for size in result["leaf_array_shape"])
    return (
        f"{shape} (trees x nodes x outputs x width), {result['leaf_array_dtype']}, "
        f"{_size(int(result['leaf_array_bytes']))} if every page were resident"
    )


def _report(results: list[dict[str, object]], *, digests: bool) -> str:
    first = results[0]
    environment = first["environment"]
    trees = int(first["trees"])
    lines = [
        f"{environment['platform']}, page size {environment['page_size_bytes']:,} "
        f"bytes, transparent huge pages: {environment['transparent_huge_pages']}",
        f"shape {first['shape']}: {first['rows']:,} donor rows, {trees} "
        f"{'tree' if trees == 1 else 'trees'}, max_samples_leaf="
        f"{first['max_samples_leaf']}, n_jobs={first['n_jobs']}, seed {first['seed']}"
        + (
            f", {first['rows_in_widest_leaf']:,} rows in the widest leaf"
            if first["rows_in_widest_leaf"] is not None
            else ""
        ),
    ]
    if len({_array_description(result) for result in results}) == 1:
        lines.append("leaf array: " + _array_description(first))
    measure = (
        "peak physical footprint"
        if first["memory_measure"] == "physical_footprint"
        else "peak resident set"
    )
    for result in results:
        if len({_array_description(result) for result in results}) > 1:
            lines.append(
                f"{result['implementation']} leaf array: {_array_description(result)}"
            )
        lines.append(
            f"{result['implementation']}: {measure} "
            f"{_size(int(result['peak_bytes_after_fit']))}, of which the fit "
            f"added {_size(int(result['fit_peak_growth_bytes']))}; fit took "
            f"{result['fit_seconds']:.1f} s"
        )
        if result["implementation"] == "library" and int(
            result["fit_peak_growth_bytes"]
        ) < int(result["leaf_array_bytes"]):
            lines.append(
                "warning: the library fit's growth is below the leaf array's size, "
                "although its copy writes every byte; memory was compressed or "
                "swapped out during the fit, so these readings are too low"
            )
    if digests and len(results) == 2:
        identical = all(
            results[0][key] == results[1][key]
            for key in ("leaf_array_shape", "leaf_array_dtype", "leaf_array_sha256")
        )
        lines.append("leaf arrays byte-identical: " + ("yes" if identical else "NO"))
    return "\n".join(lines)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shape", choices=SHAPES, default="wage")
    parser.add_argument(
        "--rows", type=int, help="donor rows (default 10,000 for either shape)"
    )
    parser.add_argument(
        "--trees", type=int, help="trees (default 100 for wage, 32 near-discrete)"
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
    if arguments.rows < 2:
        parser.error("--rows must be at least 2")
    if arguments.trees < 1:
        parser.error("--trees must be at least 1")
    if not 0 <= arguments.seed < SEED_LIMIT:
        parser.error(f"--seed must be at least 0 and below {SEED_LIMIT:,}")
    if arguments.shape == "wage" and not (
        1 <= arguments.rows_in_widest_leaf <= arguments.rows
    ):
        parser.error("--rows-in-widest-leaf must be between 1 and --rows")
    if arguments.shape == "near-discrete" and arguments.child is None:
        distinct = len(np.unique(_donors(arguments)[1]))
        if distinct < 2:
            parser.error(
                f"with --rows {arguments.rows} the near-discrete target has "
                f"{distinct} distinct value; raise --rows"
            )
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
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
            stdout=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(
                f"the {implementation} measurement failed with exit code "
                f"{completed.returncode}; its error is printed above",
                file=sys.stderr,
            )
            return 1
        results.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    if arguments.json:
        print(json.dumps(results, indent=2))
    else:
        print(_report(results, digests=not arguments.no_digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
