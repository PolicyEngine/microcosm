"""The retention-seal lane's 1/15 run: above the transport ceiling that was.

A copy of the native-scale lane's committed ``harness19_tenth.py`` with exactly
three changes, each named here and nowhere else in the file:

* the sample is 1/15 -- 105,825 of the 1,587,376 source households, where the
  single bounded encode refused at 96,860 -- rather than 1/10;
* ``RSS_CEILING`` is the retention brief's 48 GiB rather than 64 GiB;
* the label and the docstring say which lane and which fraction.

Everything else, including the CPU and wall ceilings, the sampler, the flush
cadence, the receipt capture and every refusal, is byte-for-byte the tenth
harness's. Max chose 1/15 now and 1/10 when the machine is quiet (2026-09-17),
so this file exists beside the tenth harness rather than replacing it.

Original docstring follows.

The native-scale lane's 1/10 run: above the transport ceiling that was.

Identical to the lane's baseline harness except for the sample, the ceilings and
one addition. The sample is 1/10 -- 158,737 of the 1,587,376 source households,
where the single bounded encode refused at 96,860 -- so it runs against its own
source tree, whose ``selection-request.json`` carries ``fraction: [1,10]``
because the staged pilot request pins 1/1000 and ``_request`` requires the
argument to match it. Every other source file in that tree is an APFS clone of
the staged original and hashes identically; the recovered originals are not
written, linked or moved.

The addition: every flush captures the roster transport's own on-disk receipt
(``snapshots/preparation/header.json``) and the allocation receipt from the
manifest if one exists yet. The ceiling question is answered by the preparation
and allocation nodes, which run in the first minutes, so a run the CPU ceiling
truncates still carries the evidence for it.

Original docstring follows.

Baseline for the native-scale lane: the 19-node graph, uncapped by node.

This is ``harness19_verify_once.py`` from the verify-once lane with four
additions and no change to what is measured -- same call, same arguments, same
sample, same seed, same sampler, same interval, same refusals:

* the ceilings are the native-scale brief's (9,000 CPU-s, 48 GiB, 12,000 wall-s)
  rather than the verify-once lane's, so no node is cut short;
* a current-resident-size series is sampled every second through ``libproc``'s
  ``proc_pidinfo``, because ``ru_maxrss`` is a high-water mark and a retention
  slope needs the trace, not the peak;
* the ``run_atomic_survey_financial`` call is timed on its own, so the runner
  time outside the node loop is a subtraction between measured quantities
  rather than an inference;
* the returned manifest's per-node ``wall_time_s`` is recorded, which is what
  the subtraction subtracts.

Nothing is wrapped: the runtime still refuses instrumented producers, so
attribution is still the stack sampler's.

Original docstring follows.

Nineteen-node measurement for the verify-once lane, on this branch only.

The instrumentation is ``probe_verify_once.py`` unchanged -- the same stack
sampler at the same interval, the same ceilings, the same refusals, the same
output shape. The only difference is the call: this runs the base financial
graph with the exact arguments the accepted v4 cold pilot ran
(``~/PolicyEngine/_recovered/pilot-runs/native19-v2/run/test_native_nineteen_node_financial.py``:
fraction 1/1000, seed 20260908, geography seed 20260908,
``demographic_conditioning=True``, ``n_estimators=2``, ``resume="auto"``), so
its wall, CPU and peak RSS are comparable with that receipt's
5,362.05 s / 5,278.61 s / 8.67 GiB.

Two differences from that receipt are known and are not this change: it ran on
source snapshot ``2ca11c85a``, and its ``person-income-attachment.h5`` was the
pre-restoration ``5996dcdd...`` rather than the staged ``9ebc0ef2...``. Both
files are exactly 666,333,922 B, so the bytes the source tree hashes are the
same volume either way.

Original docstring follows.

Before/after measurement for the verify-once lane: the nine-node prefix.

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
  with each run, because this machine carries other lanes' work.

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
# Byte-transport lane: this tree's lock moved, so the venv is an argument.
VENV = pathlib.Path(os.environ["MEASURE_VENV"])
SITE = VENV / "lib/python3.14/site-packages"
SUPPORT_SHA256 = "5edc0e77471ba31d550a1eed416d5b46ada0a35425718eb87cfabe4d66fe4960"
SOURCE_IDS = (
    ("district", "census-2025-cd119-NationalCD119.txt"),
    ("population", "census-2020-dec-pl-api-P1_001N"),
    ("puma", "census-2020-Census-Tract-to-2020-PUMA"),
)
CPU_CEILING = 21600
WALL_CEILING = 43200
RSS_CEILING = 48 * 1024**3  # the retention brief's ceiling, not the tenth's
RSS_SAMPLE_SECONDS = 1.0
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


# ``resource.getrusage`` reports only the high-water mark, so a retention slope
# cannot be read from it. ``proc_pidinfo(PROC_PIDTASKINFO)`` reports the current
# resident size of this task; ctypes reaches it without a subprocess, which the
# audit hook below forbids.
import ctypes  # noqa: E402


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
_PID = os.getpid()
_PROC_PIDTASKINFO = 4


def current_rss():
    info = _ProcTaskInfo()
    size = ctypes.sizeof(info)
    got = _LIBPROC.proc_pidinfo(
        _PID, _PROC_PIDTASKINFO, ctypes.c_uint64(0), ctypes.byref(info), size
    )
    return int(info.pti_resident_size) if got == size else None


RSS_SERIES = []
RUNNER = {}
NODE_WALL = []


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
    graph_atomic_survey_financial,
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
RESULT = {"status": "RUNNING"}
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


def sample_rss():
    while not DONE.wait(RSS_SAMPLE_SECONDS):
        rss = current_rss()
        if rss is not None:
            RSS_SERIES.append(
                [round(time.monotonic() - START, 3), rss, round(time.process_time(), 3)]
            )


threading.Thread(target=sample_rss, daemon=True, name="probe-rss").start()


def _versions():
    import numpy
    import pandas

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
    }


def _receipts():
    """The roster transport's own on-disk receipt, read fresh on every flush.

    The ceiling question is decided by the preparation and allocation receipts,
    which exist minutes into the run, so a ceiling-truncated measurement still
    carries the evidence. Nothing here can fail the run.
    """
    result = {}
    try:
        header = PROBE / "snapshots/preparation/header.json"
        if header.exists():
            document = json.loads(header.read_bytes())
            result["preparation_roster_header"] = document
            result["preparation_receipt_bytes"] = document.get("size")
            result["preparation_segments"] = len(document.get("segments", ()))
            result["preparation_exceeds_old_ceiling"] = (
                document.get("size", 0) > 64 * 1024**2
            )
        spill = PROBE / "snapshots/preparation"
        if spill.is_dir():
            result["segment_files_on_disk"] = len(
                [path for path in spill.iterdir() if path.suffix == ".segment"]
            )
        # How far the graph got, for a measurement the CPU ceiling truncates:
        # the store gains one directory per written object.
        objects = PROBE / "graph-store/objects"
        if objects.is_dir():
            prefixes = [path for path in objects.iterdir() if path.is_dir()]
            result["store_object_directories"] = sum(
                len([child for child in prefix.iterdir() if child.is_dir()])
                for prefix in prefixes
            )
    except BaseException as error:  # noqa: BLE001 - never fatal
        result["error"] = type(error).__name__ + ": " + str(error)[:200]
    return result


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
        "protocol": "owned.us-native-scale-baseline-probe.v1",
        "measurement": LABEL,
        "status": status or RESULT["status"],
        "source_snapshot_label_head": HEAD,
        "source_tree": str(BASE),
        "staged_run_inputs": str(RUN),
        "sample": {"fraction": [1, 15], "seed": 20260908},
        "scope": (
            "nineteen-node base financial graph at 1/15 -- 105,825 source "
            "households, above the 96,860 the single bounded encode admitted. "
            "Descriptive measurement, not a build, not a certification, not "
            "release eligible."
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
        "receipts": _receipts(),
        "rss_series_note": (
            "[seconds_since_start, current_resident_bytes, process_cpu_seconds] "
            "sampled every rss_sample_seconds through proc_pidinfo. "
            "process.peak_rss_bytes remains ru_maxrss."
        ),
        "rss_sample_seconds": RSS_SAMPLE_SECONDS,
        "rss_series": list(RSS_SERIES),
        "runner": dict(RUNNER),
        "node_wall_times": list(NODE_WALL),
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
status = "COMPLETED_NINETEEN_NODE"
RUNNER["call_started_wall_s"] = round(time.monotonic() - START, 3)
RUNNER["call_started_cpu_s"] = round(time.process_time(), 3)
RUNNER["call_started_rss_bytes"] = current_rss()
try:
    result = graph_atomic_survey_financial.run_atomic_survey_financial(
        RUN / "sources",
        snapshot_root=PROBE / "snapshots",
        store_root=PROBE / "graph-store",
        fraction=Fraction(1, 15),
        seed=20260908,
        geography_config=geography,
        demographic_conditioning=True,
        n_estimators=2,
        resume="auto",
        return_values=True,
    )
    RUNNER["call_returned_wall_s"] = round(time.monotonic() - START, 3)
    RUNNER["call_returned_cpu_s"] = round(time.process_time(), 3)
    RUNNER["call_returned_rss_bytes"] = current_rss()
    RUNNER["call_wall_seconds"] = round(
        RUNNER["call_returned_wall_s"] - RUNNER["call_started_wall_s"], 3
    )
    RUNNER["call_cpu_seconds"] = round(
        RUNNER["call_returned_cpu_s"] - RUNNER["call_started_cpu_s"], 3
    )
    # The node loop's own span, from the receipts the executor writes at
    # executor.py:2821 (wall_time = perf_counter() - node_started, the whole
    # node-loop body). run_graph is called once per graph; the financial run
    # also runs a nested population prefix with its own manifest, so both loops
    # are recorded and the report says which subtraction is which.
    # Reading the receipts must not be able to lose a completed run: any
    # failure here is recorded and the timing above still flushes.
    try:
        for node_id, receipt in result.manifest.receipts.items():
            NODE_WALL.append([node_id, round(receipt.wall_time_s, 6)])
        RUNNER["node_loop_wall_seconds"] = round(sum(w for _, w in NODE_WALL), 3)
        RUNNER["node_count"] = len(NODE_WALL)
        RUNNER["outside_node_loop_wall_seconds"] = round(
            RUNNER["call_wall_seconds"] - RUNNER["node_loop_wall_seconds"], 3
        )
        prefix_receipts = result.prefix.manifest.receipts
        RUNNER["prefix_node_wall_times"] = [
            [node_id, round(r.wall_time_s, 6)] for node_id, r in prefix_receipts.items()
        ]
        RUNNER["prefix_node_wall_seconds"] = round(
            sum(r.wall_time_s for r in prefix_receipts.values()), 3
        )
        RUNNER["prefix_node_count"] = len(prefix_receipts)
        RUNNER["outside_both_node_loops_wall_seconds"] = round(
            RUNNER["call_wall_seconds"]
            - RUNNER["node_loop_wall_seconds"]
            - RUNNER["prefix_node_wall_seconds"],
            3,
        )
        allocation = result.manifest.receipts.get("survey_population.allocate")
        if allocation is not None:
            RUNNER["allocation_receipt"] = json.loads(
                json.dumps(allocation.receipt, sort_keys=True, default=str)
            )
        create = result.manifest.receipts.get("survey_population.create")
        if create is not None:
            RUNNER["create_receipt"] = json.loads(
                json.dumps(create.receipt, sort_keys=True, default=str)
            )
        RUNNER["manifest_key"] = result.manifest.key
        RUNNER["households"] = {
            entity: int(result.financial_population.frame.n(entity))
            for entity in result.financial_population.frame.entities
        }
    except BaseException as error:  # noqa: BLE001 - recorded, never fatal
        RUNNER["receipt_read_error"] = type(error).__name__ + ": " + str(error)[:200]
except BaseException as error:  # noqa: BLE001 - the measurement records the class
    status = "STOPPED_" + type(error).__name__ + ": " + str(error)[:200]
    RUNNER.setdefault("call_returned_wall_s", round(time.monotonic() - START, 3))
    RUNNER.setdefault("call_returned_cpu_s", round(time.process_time(), 3))
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
                "top": [
                    {k: r[k] for k in ("code_chain", "cpu_seconds", "samples")}
                    for r in payload["sampled_cpu_ledger"][:8]
                ],
            },
            indent=2,
        )
    )
