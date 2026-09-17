"""Baseline for the native-scale lane: the 19-node graph, uncapped by node.

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

This copy adds a fifth thing the baseline did not need: after the cold run it
replays the same graph against the same store with ``resume="require"``, which
is what a required replay is in this tree (``_preflight_require`` refuses with
``StoreMiss`` if any node is absent), and compares the two runs' manifest key,
node keys, artifact identities, receipts and the receipt digests the transport
carries. The cold run's own CPU and wall are recorded before the replay starts,
so the before/after comparison is unaffected by it.

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
CPU_CEILING = 9000
WALL_CEILING = 12000
RSS_CEILING = 48 * 1024**3
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
REPLAY = {}


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
        "sample": {"fraction": [1, 1000], "seed": 20260908},
        "scope": (
            "nineteen-node base financial graph at the v4 cold pilot's own "
            "arguments, uncapped by node, with a current-RSS trace and the "
            "runner/node-loop split. Descriptive measurement, not a build, not "
            "a certification, not release eligible."
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
        "rss_series_note": (
            "[seconds_since_start, current_resident_bytes, process_cpu_seconds] "
            "sampled every rss_sample_seconds through proc_pidinfo. "
            "process.peak_rss_bytes remains ru_maxrss."
        ),
        "rss_sample_seconds": RSS_SAMPLE_SECONDS,
        "rss_series": list(RSS_SERIES),
        "runner": dict(RUNNER),
        "required_replay": dict(REPLAY),
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
        fraction=Fraction(1, 1000),
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
    except BaseException as error:  # noqa: BLE001 - recorded, never fatal
        RUNNER["receipt_read_error"] = type(error).__name__ + ": " + str(error)[:200]

    # A required replay of the same graph against the store the cold run wrote.
    # Everything compared here is content-addressed or a receipt, so any
    # difference is a real difference and not a timestamp.
    def _projection(manifest):
        return {
            "key": manifest.key,
            "nodes": {
                node_id: {
                    "node_key": receipt.node_key,
                    "implementation_hash": receipt.implementation_hash,
                    "seed": receipt.seed,
                    "frame_key": receipt.frame_key,
                    "weight_key": receipt.weight_key,
                    "artifacts": {
                        entity + "." + column: identity
                        for (entity, column), identity in sorted(
                            receipt.artifact_keys.items()
                        )
                    },
                    "opaque_artifacts": dict(sorted(receipt.opaque_artifacts.items())),
                    "receipt": json.loads(
                        json.dumps(receipt.receipt, sort_keys=True, default=str)
                    ),
                }
                for node_id, receipt in manifest.receipts.items()
            },
            "content_addressed": json.loads(
                json.dumps(manifest.content_addressed, sort_keys=True, default=str)
            ),
        }

    def _store_tree(root):
        tree = {}
        for path in sorted(pathlib.Path(root).rglob("*")):
            if path.is_file():
                tree[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        return tree

    try:
        import hashlib

        cold = _projection(result.manifest)
        cold_store = _store_tree(PROBE / "graph-store")
        REPLAY["cold_manifest_key"] = cold["key"]
        REPLAY["cold_store_objects"] = len(cold_store)
        REPLAY["started_wall_s"] = round(time.monotonic() - START, 3)
        REPLAY["started_cpu_s"] = round(time.process_time(), 3)
        replayed = graph_atomic_survey_financial.run_atomic_survey_financial(
            RUN / "sources",
            snapshot_root=PROBE / "snapshots-replay",
            store_root=PROBE / "graph-store",
            fraction=Fraction(1, 1000),
            seed=20260908,
            geography_config=survey_atomic_geography.AtomicSurveyReconstruction(
                support_path=str(RUN / "geography/national-atomic-support.npz"),
                support_sha256=SUPPORT_SHA256,
                source_ids=SOURCE_IDS,
                seed=20260908,
            ),
            demographic_conditioning=True,
            n_estimators=2,
            resume="require",
            return_values=True,
        )
        REPLAY["returned_wall_s"] = round(time.monotonic() - START, 3)
        REPLAY["returned_cpu_s"] = round(time.process_time(), 3)
        REPLAY["wall_seconds"] = round(
            REPLAY["returned_wall_s"] - REPLAY["started_wall_s"], 3
        )
        REPLAY["cpu_seconds"] = round(
            REPLAY["returned_cpu_s"] - REPLAY["started_cpu_s"], 3
        )
        warm = _projection(replayed.manifest)
        warm_store = _store_tree(PROBE / "graph-store")
        REPLAY["manifest_key_identical"] = cold["key"] == warm["key"]
        REPLAY["node_roster_identical"] = list(cold["nodes"]) == list(warm["nodes"])
        REPLAY["content_addressed_identical"] = (
            cold["content_addressed"] == warm["content_addressed"]
        )
        differing = sorted(
            node_id
            for node_id in cold["nodes"]
            if cold["nodes"][node_id] != warm["nodes"].get(node_id)
        )
        REPLAY["nodes_differing"] = differing
        REPLAY["first_difference"] = (
            None
            if not differing
            else {
                "node": differing[0],
                "cold": cold["nodes"][differing[0]],
                "warm": warm["nodes"].get(differing[0]),
            }
        )
        REPLAY["every_node_was_a_store_hit"] = all(
            receipt.store_hit for receipt in replayed.manifest.receipts.values()
        )
        REPLAY["store_bytes_identical"] = cold_store == warm_store
        REPLAY["store_paths_added"] = sorted(set(warm_store) - set(cold_store))
        REPLAY["store_paths_changed"] = sorted(
            path
            for path in set(cold_store) & set(warm_store)
            if cold_store[path] != warm_store[path]
        )
        REPLAY["projection_sha256"] = {
            "cold": hashlib.sha256(
                json.dumps(cold, sort_keys=True).encode()
            ).hexdigest(),
            "warm": hashlib.sha256(
                json.dumps(warm, sort_keys=True).encode()
            ).hexdigest(),
        }
        # The roster transport's own on-disk receipt, for the pin table.
        header = PROBE / "snapshots/preparation/header.json"
        if header.exists():
            REPLAY["preparation_roster_header"] = json.loads(header.read_bytes())
        (PROBE / "cold-projection.json").write_text(
            json.dumps(cold, sort_keys=True, indent=2) + "\n"
        )
        (PROBE / "replay-projection.json").write_text(
            json.dumps(warm, sort_keys=True, indent=2) + "\n"
        )
        status = "COMPLETED_NINETEEN_NODE_AND_REQUIRED_REPLAY"
    except BaseException as error:  # noqa: BLE001 - recorded, never fatal
        REPLAY["error"] = type(error).__name__ + ": " + str(error)[:400]
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
