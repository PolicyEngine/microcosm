"""Emit machine-readable pytest timings for tests, files, and the session."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

TIMING_PREFIX = "MICROCOSM_TIMING "

_session_started = 0.0
_collection_finished = 0.0
_test_started: dict[str, float] = {}
_phase_seconds: dict[str, float] = defaultdict(float)
_outcomes: dict[str, set[str]] = defaultdict(set)
_module_collection_started: dict[str, float] = {}
_file_collection_seconds: dict[str, float] = defaultdict(float)
_file_started: dict[str, float] = {}
_file_finished: dict[str, float] = {}
_file_phase_seconds: dict[str, float] = defaultdict(float)
_file_test_count: dict[str, int] = defaultdict(int)


def _emit(payload: dict[str, Any]) -> None:
    print(
        TIMING_PREFIX
        + json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True),
        flush=True,
    )


def _path(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def pytest_sessionstart(session) -> None:
    """Reset timing state at the beginning of each pytest process."""

    global _session_started, _collection_finished
    _session_started = time.monotonic()
    _collection_finished = 0.0
    _test_started.clear()
    _phase_seconds.clear()
    _outcomes.clear()
    _module_collection_started.clear()
    _file_collection_seconds.clear()
    _file_started.clear()
    _file_finished.clear()
    _file_phase_seconds.clear()
    _file_test_count.clear()


def pytest_collection_finish(session) -> None:
    """Record when imports and test collection have completed."""

    global _collection_finished
    _collection_finished = time.monotonic()


def pytest_collectstart(collector) -> None:
    """Start timing collection for a test module."""

    nodeid = str(collector.nodeid)
    if nodeid.endswith(".py"):
        _module_collection_started[nodeid] = time.monotonic()


def pytest_collectreport(report) -> None:
    """Attribute test-module import and collection time to its file."""

    nodeid = str(report.nodeid)
    started = _module_collection_started.pop(nodeid, None)
    if started is not None:
        _file_collection_seconds[nodeid] += max(0.0, time.monotonic() - started)


def pytest_runtest_logstart(nodeid: str, location) -> None:
    """Start the wall clock for one collected test."""

    started = time.monotonic()
    path = _path(nodeid)
    _test_started[nodeid] = started
    _file_started.setdefault(path, started)


def pytest_runtest_logreport(report) -> None:
    """Accumulate setup, call, and teardown durations and outcomes."""

    nodeid = report.nodeid
    duration = float(getattr(report, "duration", 0.0))
    _phase_seconds[nodeid] += duration
    _outcomes[nodeid].add(str(report.outcome))


def pytest_runtest_logfinish(nodeid: str, location) -> None:
    """Emit one record after every test, including parametrized cases."""

    finished = time.monotonic()
    path = _path(nodeid)
    started = _test_started.pop(nodeid, finished)
    wall_seconds = max(0.0, finished - started)
    phase_seconds = _phase_seconds.pop(nodeid, 0.0)
    outcomes = _outcomes.pop(nodeid, set())
    if "failed" in outcomes:
        outcome = "failed"
    elif "skipped" in outcomes:
        outcome = "skipped"
    else:
        outcome = "passed"

    _file_finished[path] = finished
    _file_phase_seconds[path] += phase_seconds
    _file_test_count[path] += 1
    _emit(
        {
            "kind": "test",
            "nodeid": nodeid,
            "outcome": outcome,
            "path": path,
            "phase_seconds": phase_seconds,
            "wall_seconds": wall_seconds,
        }
    )


def pytest_sessionfinish(session, exitstatus: int) -> None:
    """Emit file aggregates and process-session overhead information."""

    finished = time.monotonic()
    for path in sorted(_file_test_count.keys() | _file_collection_seconds.keys()):
        test_wall_seconds = max(
            0.0,
            _file_finished.get(path, finished) - _file_started.get(path, finished),
        )
        collection_seconds = _file_collection_seconds[path]
        _emit(
            {
                "collection_seconds": collection_seconds,
                "kind": "file",
                "path": path,
                "phase_seconds": _file_phase_seconds[path],
                "test_count": _file_test_count[path],
                "test_wall_seconds": test_wall_seconds,
                "wall_seconds": collection_seconds + test_wall_seconds,
            }
        )
    _emit(
        {
            "kind": "session",
            "collection_seconds": max(
                0.0,
                (_collection_finished or finished) - _session_started,
            ),
            "exit_status": int(exitstatus),
            "wall_seconds": max(0.0, finished - _session_started),
        }
    )
