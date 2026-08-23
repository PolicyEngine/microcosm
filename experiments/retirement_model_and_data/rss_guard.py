#!/usr/bin/env python3
"""Fail-closed Darwin RSS guard for the retirement model-and-data audits and host gates."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LIMIT_BYTES = 14 * 1024**3
POLL_INTERVAL_SECONDS = 0.005
SELF_TEST_LIMIT_BYTES = 96 * 1024**2
PROC_PGRP_ONLY = 2
PROC_PPID_ONLY = 6
PROC_PIDTASKINFO = 4


class ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_uint32),
        ("messages_received", ctypes.c_uint32),
        ("syscalls_mach", ctypes.c_uint32),
        ("syscalls_unix", ctypes.c_uint32),
        ("csw", ctypes.c_uint32),
        ("threadnum", ctypes.c_uint32),
        ("numrunning", ctypes.c_uint32),
        ("priority", ctypes.c_int32),
    ]


LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
LIBPROC.proc_listpids.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_int,
]
LIBPROC.proc_listpids.restype = ctypes.c_int
LIBPROC.proc_pidinfo.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.c_int,
]
LIBPROC.proc_pidinfo.restype = ctypes.c_int


class GuardInterruptedError(RuntimeError):
    """Raised by a handled signal so the child process group is killed."""


def _list_pids(kind: int, value: int) -> set[int]:
    needed = LIBPROC.proc_listpids(kind, value, None, 0)
    if needed < 0:
        raise OSError("proc_listpids size query failed")
    capacity = max(32, needed // ctypes.sizeof(ctypes.c_int) + 32)
    buffer = (ctypes.c_int * capacity)()
    written = LIBPROC.proc_listpids(kind, value, buffer, ctypes.sizeof(buffer))
    if written < 0:
        raise OSError("proc_listpids enumeration failed")
    return {pid for pid in buffer[: written // ctypes.sizeof(ctypes.c_int)] if pid > 0}


def _resident_size(pid: int) -> int | None:
    info = ProcTaskInfo()
    written = LIBPROC.proc_pidinfo(
        pid,
        PROC_PIDTASKINFO,
        0,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if written == 0:
        return None
    if written != ctypes.sizeof(info):
        raise OSError(f"proc_pidinfo returned {written} bytes for pid {pid}")
    return int(info.resident_size)


def _descendants(root: int) -> set[int]:
    selected = {root}
    frontier = {root}
    while frontier:
        found = set().union(*(_list_pids(PROC_PPID_ONLY, pid) for pid in frontier))
        frontier = found - selected
        selected.update(frontier)
    return selected


def _terminate_group(pgid: int, pids: set[int]) -> None:
    try:
        os.killpg(pgid, signal.SIGSTOP)
    except (PermissionError, ProcessLookupError):
        pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGSTOP)
        except (PermissionError, ProcessLookupError):
            pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass


def _signal_handler(signum: int, _frame: object) -> None:
    raise GuardInterruptedError(f"received signal {signum}")


def _write_event(trace, event: dict[str, object]) -> None:
    trace.write(json.dumps(event, sort_keys=True) + "\n")
    trace.flush()


def _supervise(
    command: list[str],
    *,
    trace_path: Path,
    log_path: Path,
    limit_bytes: int,
    allow_descendant_session_escape: bool = False,
) -> dict[str, object]:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    peak_individual = 0
    peak_aggregate = 0
    maximum_process_count = 0
    maximum_sample_gap = 0.0
    previous_sample: float | None = None
    disposition = "completed"
    known_pids: set[int] = set()
    escaped_pids_seen: set[int] = set()

    with (
        trace_path.open("x", encoding="utf-8") as trace,
        log_path.open("x", encoding="utf-8") as log,
    ):
        try:
            child = subprocess.Popen(
                command,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except BaseException as error:
            final = {
                "disposition": (f"launch_error:{type(error).__name__}:{error}"),
                "limit_bytes": limit_bytes,
                "process_group_empty_postcondition": True,
            }
            _write_event(trace, {"final": final})
            return final

        pgid = child.pid
        try:
            _write_event(
                trace,
                {
                    "launch": {
                        "command": command,
                        "allow_descendant_session_escape": (
                            allow_descendant_session_escape
                        ),
                        "limit_bytes": limit_bytes,
                        "pgid": pgid,
                        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
                        "root_pid": child.pid,
                    }
                },
            )
        except BaseException:
            _terminate_group(pgid, {child.pid})
            child.wait()
            raise
        old_handlers = {
            signum: signal.signal(signum, _signal_handler)
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        }
        try:
            while True:
                sampled_at = time.monotonic()
                if previous_sample is not None:
                    maximum_sample_gap = max(
                        maximum_sample_gap, sampled_at - previous_sample
                    )
                previous_sample = sampled_at
                try:
                    group = _list_pids(PROC_PGRP_ONLY, pgid)
                    root_running = child.poll() is None
                    lineage = _descendants(child.pid) if root_running else set()
                    if root_running and child.pid not in group:
                        if child.poll() is None:
                            raise RuntimeError(
                                f"root pid {child.pid} escaped process group {pgid}"
                            )
                        root_running = False
                        lineage = set()
                    escaped = lineage - group
                    escaped_pids_seen.update(escaped)
                    if escaped and not allow_descendant_session_escape:
                        raise RuntimeError(
                            f"descendants escaped process group: {sorted(escaped)}"
                        )
                    selected = group | lineage
                    known_pids.update(selected)
                    rss = {
                        pid: value
                        for pid in selected
                        if (value := _resident_size(pid)) is not None
                    }
                    individual = max(rss.values(), default=0)
                    aggregate = sum(rss.values())
                    peak_individual = max(peak_individual, individual)
                    peak_aggregate = max(peak_aggregate, aggregate)
                    maximum_process_count = max(maximum_process_count, len(rss))
                    _write_event(
                        trace,
                        {
                            "aggregate_rss_bytes": aggregate,
                            "individual_rss_bytes": individual,
                            "monotonic": sampled_at,
                            "pids": sorted(rss),
                            "escaped_descendant_pids": sorted(escaped),
                            "root_running": root_running,
                        },
                    )
                    if individual >= limit_bytes or aggregate >= limit_bytes:
                        disposition = "rss_limit"
                        _terminate_group(pgid, selected)
                        break
                    if not root_running and not group:
                        break
                    time.sleep(POLL_INTERVAL_SECONDS)
                except BaseException as error:
                    disposition = f"monitor_error:{type(error).__name__}:{error}"
                    _write_event(trace, {"monitor_error": disposition})
                    _terminate_group(pgid, known_pids | {child.pid})
                    break
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)

        returncode = child.wait()
        group_survivors = _list_pids(PROC_PGRP_ONLY, pgid)
        if group_survivors:
            _terminate_group(pgid, group_survivors)
            deadline = time.monotonic() + 1.0
            while group_survivors and time.monotonic() < deadline:
                time.sleep(POLL_INTERVAL_SECONDS)
                group_survivors = _list_pids(PROC_PGRP_ONLY, pgid)
            if group_survivors:
                disposition = (
                    "postcondition_error:surviving_process_group_pids:"
                    f"{sorted(group_survivors)}"
                )
        group_empty = not group_survivors
        final = {
            "allow_descendant_session_escape": allow_descendant_session_escape,
            "disposition": disposition,
            "escaped_descendant_pids_observed": sorted(escaped_pids_seen),
            "limit_bytes": limit_bytes,
            "max_inter_sample_gap_seconds": maximum_sample_gap,
            "maximum_process_count": maximum_process_count,
            "peak_aggregate_rss_bytes": peak_aggregate,
            "peak_individual_rss_bytes": peak_individual,
            "process_group_empty_postcondition": group_empty,
            "returncode": returncode,
        }
        _write_event(trace, {"final": final})
        return final


def _self_test_command() -> list[str]:
    grandchild = "import time; hold = bytearray(48 * 1024**2); time.sleep(30)"
    root = (
        "import subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "hold = bytearray(48 * 1024**2); child.wait()"
    )
    return [sys.executable, "-c", root]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--allow-descendant-session-escape", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if args.self_test:
        if command or args.allow_descendant_session_escape:
            parser.error(
                "--self-test accepts neither a command nor session-escape opt-in"
            )
        command = _self_test_command()
        limit_bytes = SELF_TEST_LIMIT_BYTES
    else:
        if not command:
            parser.error("a command is required after --")
        limit_bytes = LIMIT_BYTES

    final = _supervise(
        command,
        trace_path=args.trace,
        log_path=args.log,
        limit_bytes=limit_bytes,
        allow_descendant_session_escape=args.allow_descendant_session_escape,
    )
    if args.self_test:
        passed = (
            final.get("disposition") == "rss_limit"
            and int(final.get("maximum_process_count", 0)) >= 2
            and final.get("process_group_empty_postcondition") is True
        )
        final = {**final, "self_test_passed": passed}
        print(json.dumps(final, sort_keys=True))
        raise SystemExit(0 if passed else 97)

    print(json.dumps(final, sort_keys=True))
    if final.get("disposition") != "completed":
        raise SystemExit(97)
    raise SystemExit(int(final["returncode"]))


if __name__ == "__main__":
    main()
