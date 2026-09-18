"""Before/after measurement for the verify-once lane: the nine-node prefix.

A parameterised copy of the v5 pilot probe
(``~/PolicyEngine/_recovered/pilot-runs/native45-v5/probe_repeated_verification.py``).
Everything that decides *what is measured* is verbatim: the same nine-node
population prefix, the same 1/1000 sample, the same seed, the same stack
sampler at the same interval, the same ceilings, the same output shape. The
only changes are the ones a before/after comparison needs:

* the source tree, the staged run inputs and the output directory come from
  the environment, so the identical file measures both trees;
* every imported ``microcosm`` module is asserted to live under that tree,
  so a stray editable install cannot silently measure the wrong source;
* a resident-memory ceiling is enforced alongside the CPU ceiling;
* the load average and the interpreter's third-party versions are recorded
  with each run, because this machine carries other lanes' work;
* the run's verification-epoch record is copied out of the manifest the runner
  returns, so a measured run says how often it re-authenticated instead of
  leaving that to a unit test. This line was added after the three measurement
  files ``out.md`` quotes were written, so none of them carries the record; the
  before tree has no such record to carry. It is a read, and it measures
  nothing: the counted work is identical on either side of it.

Sources are read by path and never written. Nothing here is a build, a
certification or a release artifact.

    MEASURE_BASE=<source tree> MEASURE_RUN=<staged run dir> \
    MEASURE_PROBE=<output dir> MEASURE_LABEL=<label> MEASURE_HEAD=<sha> \
    <venv>/bin/python -I -B -S probe_verify_once.py
"""

import json
import os
import pathlib
import resource
import signal
import sys
import threading
import time
from fractions import Fraction

BASE = pathlib.Path(os.environ["MEASURE_BASE"]).absolute()
RUN = pathlib.Path(os.environ["MEASURE_RUN"]).absolute()
PROBE = pathlib.Path(os.environ["MEASURE_PROBE"]).absolute()
LABEL = os.environ["MEASURE_LABEL"]
HEAD = os.environ["MEASURE_HEAD"]
VENV = pathlib.Path(
    "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-v5/.venv"
)
SITE = VENV / "lib/python3.14/site-packages"
SUPPORT_SHA256 = "5edc0e77471ba31d550a1eed416d5b46ada0a35425718eb87cfabe4d66fe4960"
SOURCE_IDS = (
    ("district", "census-2025-cd119-NationalCD119.txt"),
    ("population", "census-2020-dec-pl-api-P1_001N"),
    ("puma", "census-2020-Census-Tract-to-2020-PUMA"),
)
CPU_CEILING = 1800
WALL_CEILING = 2400
RSS_CEILING = 16 * 1024**3
FLUSH_SECONDS = 10
OUT = PROBE / "repeated-verification-measurement.json"

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "POPULACE_FIT_N_JOBS",
    "POPULACE_FIT_PREDICT_WORKERS",
):
    os.environ[name] = "1"
os.environ["TZ"] = "UTC"
PROBE.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(PROBE)
resource.setrlimit(resource.RLIMIT_CPU, (CPU_CEILING + 60, CPU_CEILING + 60))
signal.alarm(WALL_CEILING)

sys.path[:0] = [str(p) for p in sorted((BASE / "packages").glob("*/src"))] + [str(SITE)]
START = time.monotonic()
LOAD_AT_START = os.getloadavg()


# The same refusals the pilot guard enforces: no country engine, no network, no
# child processes. This probe reads the staged sources and writes only in PROBE.
import importlib.abc  # noqa: E402


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "policyengine"
            or fullname.startswith("policyengine.")
            or fullname.startswith("policyengine_")
            or fullname.startswith("microcosm.build.uk_runtime")
        ):
            raise RuntimeError("Country engine import forbidden: " + fullname)


sys.meta_path.insert(0, NoEngine())


def audit(event, args):
    if (event.startswith("socket.") and event != "socket.gethostname") or event in {
        "subprocess.Popen",
        "os.system",
        "os.posix_spawn",
        "os.exec",
        "os.fork",
        "pty.spawn",
    }:
        raise PermissionError("External operation blocked: " + event)


sys.addaudithook(audit)

import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from microcosm.build.us_runtime import (  # noqa: E402
    acs_housing_universe_source,
    acs_person_coverage_authentication,
    acs_person_coverage_columns,
    asec_2024_native_population,
    asec_coverage_authentication,
    asec_current_money_source,
    graph_atomic_survey_population,
    survey_atomic_geography,
    survey_population_preparation,
)

# A before/after comparison is worthless if the two runs import the same files.
# ``-S`` already keeps site-packages' editable ``.pth`` entries out of the path;
# this proves it for every module that is actually loaded.
STRAY = sorted(
    name
    for name, module in sys.modules.items()
    if name.split(".")[0] == "microcosm"
    and getattr(module, "__file__", None)
    and not str(pathlib.Path(module.__file__).absolute()).startswith(str(BASE) + os.sep)
)
if STRAY:
    raise RuntimeError(
        "microcosm modules resolved outside the measured tree: " + repr(STRAY)
    )

STATS = {}


def measure(module, name, label):
    """Count calls and charge self+descendant CPU, excluding re-entrant nesting."""
    original = getattr(module, name, None)
    if original is None or not callable(original):
        STATS[label] = {"calls": 0, "cpu_seconds": 0.0, "status": "not-wrapped"}
        return
    row = STATS.setdefault(label, {"calls": 0, "cpu_seconds": 0.0, "status": "ok"})
    is_generator = name in {"_records", "_literal_csv_records"}

    def generator_wrapper(*args, **kwargs):
        # Charge only the producer's own time, not the consumer's body.
        row["calls"] += 1
        started = time.process_time()
        for item in original(*args, **kwargs):
            row["cpu_seconds"] += time.process_time() - started
            yield item
            started = time.process_time()
        row["cpu_seconds"] += time.process_time() - started

    def plain_wrapper(*args, **kwargs):
        row["calls"] += 1
        started = time.process_time()
        try:
            return original(*args, **kwargs)
        finally:
            row["cpu_seconds"] += time.process_time() - started

    setattr(module, name, generator_wrapper if is_generator else plain_wrapper)


TARGETS = [
    (
        acs_person_coverage_authentication,
        "_records",
        "acs_person_coverage_authentication._records",
    ),
    (
        acs_person_coverage_authentication,
        "load_authenticated_acs_person_coverage",
        "acs_person_coverage_authentication.load_authenticated_acs_person_coverage [outer]",
    ),
    (
        acs_person_coverage_authentication,
        "_inventory",
        "acs_person_coverage_authentication._inventory",
    ),
    (
        acs_person_coverage_columns,
        "_literal_csv_records",
        "acs_person_coverage_columns._literal_csv_records",
    ),
    (
        acs_person_coverage_columns,
        "_scan_acs_person_coverage",
        "acs_person_coverage_columns._scan_acs_person_coverage",
    ),
    (
        acs_person_coverage_columns,
        "read_acs_person_coverage_columns",
        "acs_person_coverage_columns.read_acs_person_coverage_columns [outer]",
    ),
    (
        asec_current_money_source,
        "_series_digest",
        "asec_current_money_source._series_digest",
    ),
    (
        asec_current_money_source,
        "_frame_signature",
        "asec_current_money_source._frame_signature",
    ),
    (
        asec_2024_native_population,
        "_file_identity",
        "asec_2024_native_population._file_identity",
    ),
    (
        asec_2024_native_population,
        "_frame_identity",
        "asec_2024_native_population._frame_identity",
    ),
    (asec_coverage_authentication, "_capture", "asec_coverage_authentication._capture"),
    (
        asec_coverage_authentication,
        "authenticate_asec_coverage",
        "asec_coverage_authentication.authenticate_asec_coverage [outer]",
    ),
    (
        acs_housing_universe_source,
        "_persisted_sha",
        "acs_housing_universe_source._persisted_sha",
    ),
    (
        survey_population_preparation,
        "_frame_identity",
        "survey_population_preparation._frame_identity",
    ),
    (
        survey_atomic_geography,
        "_population_stamp",
        "survey_atomic_geography._population_stamp",
    ),
    (
        survey_atomic_geography,
        "reconstruct_atomic_survey_geography",
        "survey_atomic_geography.reconstruct_atomic_survey_geography [outer]",
    ),
]
# The runtime authenticates its own producers: wrapping them raises
# SurveyPopulationPreparationError PRODUCER_CHANGED within 4 seconds (observed
# 2026-09-15). That refusal is correct, so the measurement is taken by stack
# sampling instead -- the same technique the pilot guard's ledger uses, and the
# only one that does not alter the code being measured.
WRAP_PRODUCERS = False
if WRAP_PRODUCERS:
    for module, name, label in TARGETS:
        measure(module, name, label)
else:
    for _module, _name, label in TARGETS:
        STATS[label] = {
            "calls": 0,
            "cpu_seconds": 0.0,
            "status": "not-wrapped: runtime refuses instrumented producers "
            "(PRODUCER_CHANGED); see sampled_cpu_ledger",
        }

DONE = threading.Event()
RESULT = {"status": "RUNNING", "verification_epoch": None}
# Sampling ledger, identical in shape to the pilot guard's: it never depends on
# the wrappers, so a function-identity refusal cannot cost us the measurement.
LEDGER = {}
SAMPLE_INTERVAL = 0.25


def sample_stacks():
    previous_key, previous_cpu = (), 0.0
    main = threading.main_thread().ident
    while not DONE.wait(SAMPLE_INTERVAL):
        cpu = time.process_time()
        frame = sys._current_frames().get(main)
        stack = []
        while frame is not None and len(stack) < 24:
            stack.append(
                (pathlib.Path(frame.f_code.co_filename).name, frame.f_code.co_name)
            )
            frame = frame.f_back
        del frame
        if previous_key:
            row = LEDGER.setdefault(
                "␟".join(f"{a}:{b}" for a, b in previous_key[:4]),
                {"cpu_seconds": 0.0, "samples": 0},
            )
            row["cpu_seconds"] += cpu - previous_cpu
            row["samples"] += 1
        previous_key, previous_cpu = tuple(stack), cpu


threading.Thread(target=sample_stacks, daemon=True, name="probe-sampler").start()


def _versions():
    import numpy
    import pandas

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
    }


def flush(status=None):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu = usage.ru_utime + usage.ru_stime
    rows = sorted(
        ({"check": label, **row} for label, row in STATS.items()),
        key=lambda r: -r["cpu_seconds"],
    )
    for row in rows:
        row["share_of_process_cpu"] = round(100 * row["cpu_seconds"] / (cpu or 1.0), 2)
        row["cpu_seconds_per_call"] = (
            round(row["cpu_seconds"] / row["calls"], 6) if row["calls"] else None
        )
    payload = {
        "protocol": "owned.us-native-repeated-verification-probe.v1",
        "measurement": LABEL,
        "status": status or RESULT["status"],
        "source_snapshot_label_head": HEAD,
        "source_tree": str(BASE),
        "staged_run_inputs": str(RUN),
        "sample": {"fraction": [1, 1000], "seed": 20260908},
        "scope": (
            "nine-node population prefix only; the source-admission path every "
            "larger graph pays first. Descriptive measurement, not a build, "
            "not a certification, not release eligible."
        ),
        "ceilings": {
            "cpu_seconds": CPU_CEILING,
            "wall_seconds": WALL_CEILING,
            "peak_rss_bytes": RSS_CEILING,
        },
        "process": {
            "cpu_seconds": cpu,
            "wall_seconds": time.monotonic() - START,
            "peak_rss_bytes": usage.ru_maxrss,
        },
        "machine": {
            "loadavg_at_start": LOAD_AT_START,
            "loadavg_now": os.getloadavg(),
            "versions": _versions(),
        },
        # The run's own verification-epoch record, copied off the manifest the
        # runner returns: capsules memoised, borrows answered from the memo,
        # borrows that re-ran the whole validation, and the unconditional final
        # re-validations that closed the epoch. `null` until the runner returns,
        # so every mid-run flush and every ceiling exit carries `null`, and the
        # before tree has no such record at all. It is read, never written, and
        # it is outside the manifest key, its JSON and every receipt, so
        # recording it moves nothing the run is identified by.
        "verification_epoch": RESULT["verification_epoch"],
        "nesting_note": (
            "Rows marked [outer] enclose the unmarked rows beneath them, so "
            "their seconds overlap; share_of_process_cpu is per-row and these "
            "shares deliberately do not sum to 100."
        ),
        "inner_check_cpu_seconds": sum(
            r["cpu_seconds"] for r in rows if "[outer]" not in r["check"]
        ),
        "checks": rows,
        "sampled_cpu_ledger": [
            {"code_chain": k.split("␟"), **v}
            for k, v in sorted(LEDGER.items(), key=lambda kv: -kv[1]["cpu_seconds"])[
                :60
            ]
        ],
        "sampled_ledger_total_seconds": sum(r["cpu_seconds"] for r in LEDGER.values()),
        "sampled_ledger_interval_seconds": SAMPLE_INTERVAL,
        "release_eligible": False,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def watch():
    while not DONE.wait(2.0):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu = usage.ru_utime + usage.ru_stime
        if usage.ru_maxrss > RSS_CEILING:
            RESULT["status"] = "RSS_CEILING_REACHED_PARTIAL_MEASUREMENT"
            flush(RESULT["status"])
            os._exit(137)
        if cpu > CPU_CEILING or time.monotonic() - START > WALL_CEILING:
            RESULT["status"] = "CEILING_REACHED_PARTIAL_MEASUREMENT"
            flush("CEILING_REACHED_PARTIAL_MEASUREMENT")
            os._exit(124)
        if int(time.monotonic() - START) % FLUSH_SECONDS < 2:
            flush()


threading.Thread(target=watch, daemon=True, name="probe-monitor").start()

geography = survey_atomic_geography.AtomicSurveyReconstruction(
    support_path=str(RUN / "geography/national-atomic-support.npz"),
    support_sha256=SUPPORT_SHA256,
    source_ids=SOURCE_IDS,
    seed=20260908,
)
status = "COMPLETED_PREFIX"
try:
    run_values = graph_atomic_survey_population.run_atomic_survey_population(
        RUN / "sources",
        snapshot_root=PROBE / "snapshots",
        store_root=PROBE / "graph-store",
        fraction=Fraction(1, 1000),
        seed=20260908,
        geography_config=geography,
        resume="auto",
        return_values=True,
    )
    # Take the epoch's counts and drop the run values again immediately, so
    # nothing this probe added stays resident while `flush` runs and peak RSS
    # stays comparable with the runs that predate this line.
    # Read with a default: a before-tree manifest has no such field, and this
    # probe is run against both trees. `None` is the "this tree does not carry
    # one" value the result starts at, so an absent field stays legible rather
    # than looking like an epoch that counted nothing.
    _epoch_record = getattr(run_values.manifest, "verification_epoch", None)
    RESULT["verification_epoch"] = (
        None if _epoch_record is None else dict(_epoch_record)
    )
    del run_values
except BaseException as error:  # noqa: BLE001 - the measurement records the class
    status = "STOPPED_" + type(error).__name__ + ": " + str(error)[:200]
finally:
    DONE.set()
    RESULT["status"] = status
    payload = flush(status)
    print(
        json.dumps(
            {
                "measurement": LABEL,
                "status": payload["status"],
                "cpu_seconds": payload["process"]["cpu_seconds"],
                "wall_seconds": payload["process"]["wall_seconds"],
                "peak_rss_bytes": payload["process"]["peak_rss_bytes"],
                "loadavg_at_start": payload["machine"]["loadavg_at_start"],
                "loadavg_at_end": payload["machine"]["loadavg_now"],
                "verification_epoch": payload["verification_epoch"],
                "top": [
                    {k: r[k] for k in ("code_chain", "cpu_seconds", "samples")}
                    for r in payload["sampled_cpu_ledger"][:8]
                ],
            },
            indent=2,
        )
    )
