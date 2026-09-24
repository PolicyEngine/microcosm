#!/usr/bin/env python3
"""Run a command while logging its process-tree and runner memory usage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessMemory:
    """Memory data read from one process directory."""

    pid: int
    ppid: int
    rss_bytes: int
    pss_bytes: int | None
    command: str


def read_kib_fields(text: str) -> dict[str, int]:
    """Parse Linux procfs fields, converting values measured in KiB to bytes."""
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        fields[key] = value * 1024 if parts[1:] == ["kB"] else value
    return fields


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, RuntimeError):
        return None


def _read_processes(proc_root: Path) -> dict[int, ProcessMemory]:
    processes: dict[int, ProcessMemory] = {}
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return processes

    for entry in entries:
        if not entry.name.isdigit():
            continue
        status_text = _read_text(entry / "status")
        if status_text is None:
            continue
        status = read_kib_fields(status_text)
        try:
            pid = int(entry.name)
            ppid = status["PPid"]
        except (KeyError, ValueError):
            continue
        rss_bytes = status.get("VmRSS", 0)
        rollup_text = _read_text(entry / "smaps_rollup")
        pss_bytes = None
        if rollup_text is not None:
            pss_bytes = read_kib_fields(rollup_text).get("Pss")
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            arguments = []
        command = " ".join(
            argument.decode("utf-8", errors="replace")
            for argument in arguments[:3]
            if argument
        )
        processes[pid] = ProcessMemory(
            pid=pid,
            ppid=ppid,
            rss_bytes=rss_bytes,
            pss_bytes=pss_bytes,
            command=command[:160],
        )
    return processes


def process_tree(
    processes: dict[int, ProcessMemory], root_pid: int
) -> list[ProcessMemory]:
    """Return the root process and all recursively discovered descendants."""
    selected = {root_pid}
    while True:
        descendants = {
            process.pid for process in processes.values() if process.ppid in selected
        }
        expanded = selected | descendants
        if expanded == selected:
            break
        selected = expanded
    return [processes[pid] for pid in sorted(selected) if pid in processes]


def _read_integer(path: Path) -> int | None:
    value = _read_text(path)
    if value is None or value.strip() == "max":
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _read_events(path: Path) -> dict[str, int]:
    text = _read_text(path)
    if text is None:
        return {}
    events: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            events[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return events


def _cgroup_memory(cgroup_root: Path) -> dict[str, Any]:
    if (cgroup_root / "memory.current").exists():
        events = _read_events(cgroup_root / "memory.events")
        return {
            "cgroup_path": str(cgroup_root),
            "cgroup_current_bytes": _read_integer(cgroup_root / "memory.current"),
            "cgroup_peak_bytes": _read_integer(cgroup_root / "memory.peak"),
            "cgroup_limit_bytes": _read_integer(cgroup_root / "memory.max"),
            "cgroup_oom_events": events.get("oom"),
            "cgroup_oom_kills": events.get("oom_kill"),
        }

    memory_root = cgroup_root
    if not (memory_root / "memory.usage_in_bytes").exists():
        memory_root /= "memory"
    return {
        "cgroup_path": str(memory_root),
        "cgroup_current_bytes": _read_integer(memory_root / "memory.usage_in_bytes"),
        "cgroup_peak_bytes": _read_integer(memory_root / "memory.max_usage_in_bytes"),
        "cgroup_limit_bytes": _read_integer(memory_root / "memory.limit_in_bytes"),
        "cgroup_oom_events": None,
        "cgroup_oom_kills": _read_integer(memory_root / "memory.failcnt"),
    }


def _current_cgroup_memory(proc_root: Path, cgroup_root: Path) -> dict[str, Any]:
    membership = _read_text(proc_root / "self" / "cgroup") or ""
    for line in membership.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, relative_path = parts
        if hierarchy == "0" and not controllers:
            return _cgroup_memory(cgroup_root / relative_path.lstrip("/"))
        if "memory" in controllers.split(","):
            return _cgroup_memory(cgroup_root / "memory" / relative_path.lstrip("/"))
    return _cgroup_memory(cgroup_root)


def _memory_sample(
    *,
    root_pid: int,
    started_at: float,
    phase: str,
    proc_root: Path,
    cgroup_root: Path,
    exit_code: int | None = None,
) -> dict[str, Any]:
    meminfo_text = _read_text(proc_root / "meminfo") or ""
    meminfo = read_kib_fields(meminfo_text)
    total_bytes = meminfo.get("MemTotal")
    available_bytes = meminfo.get("MemAvailable")
    used_percent = None
    if total_bytes and available_bytes is not None:
        used_percent = round(100 * (total_bytes - available_bytes) / total_bytes, 2)

    tree = process_tree(_read_processes(proc_root), root_pid)
    pss_values = [process.pss_bytes for process in tree]
    tree_pss_bytes = None
    if tree and all(value is not None for value in pss_values):
        tree_pss_bytes = sum(value for value in pss_values if value is not None)
    top_processes = sorted(tree, key=lambda process: process.rss_bytes, reverse=True)[
        :5
    ]

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "phase": phase,
        "root_pid": root_pid,
        "exit_code": exit_code,
        "host_total_bytes": total_bytes,
        "host_available_bytes": available_bytes,
        "host_used_percent": used_percent,
        "swap_total_bytes": meminfo.get("SwapTotal"),
        "swap_free_bytes": meminfo.get("SwapFree"),
        "process_count": len(tree),
        "tree_rss_bytes": sum(process.rss_bytes for process in tree),
        "tree_pss_bytes": tree_pss_bytes,
        "top_processes": [asdict(process) for process in top_processes],
    }
    record.update(_current_cgroup_memory(proc_root, cgroup_root))
    return record


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024**2:.0f}MiB"


def _format_sample(record: dict[str, Any]) -> str:
    top = ";".join(
        f"{process['pid']}:{_format_bytes(process['rss_bytes'])}:{process['command']}"
        for process in record["top_processes"][:3]
    )
    return (
        f"[memory] phase={record['phase']} "
        f"elapsed={record['elapsed_seconds']:.1f}s "
        f"host_used={record['host_used_percent']}% "
        f"host_available={_format_bytes(record['host_available_bytes'])} "
        f"tree_rss={_format_bytes(record['tree_rss_bytes'])} "
        f"tree_pss={_format_bytes(record['tree_pss_bytes'])} "
        f"processes={record['process_count']} "
        f"cgroup_current={_format_bytes(record['cgroup_current_bytes'])} "
        f"cgroup_peak={_format_bytes(record['cgroup_peak_bytes'])} "
        f"oom={record['cgroup_oom_events']} "
        f"oom_kill={record['cgroup_oom_kills']} top={top or 'n/a'}"
    )


def run_monitored(
    command: Sequence[str],
    *,
    output: Path,
    interval_seconds: float = 5.0,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int:
    """Run ``command``, stream memory samples, and return its exit status."""
    if not command:
        raise ValueError("A command is required")
    if interval_seconds <= 0:
        raise ValueError("The sample interval must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    previous_handlers: dict[signal.Signals, Any] = {}

    def forward_signal(signum: int, _frame: Any) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, forward_signal)

    def emit(trace, phase: str, exit_code: int | None = None) -> None:
        record = _memory_sample(
            root_pid=process.pid,
            started_at=started_at,
            phase=phase,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
            exit_code=exit_code,
        )
        trace.write(json.dumps(record, sort_keys=True) + "\n")
        trace.flush()
        print(_format_sample(record), flush=True)

    try:
        with output.open("w", encoding="utf-8", buffering=1) as trace:
            emit(trace, "start")
            while True:
                try:
                    return_code = process.wait(timeout=interval_seconds)
                    break
                except subprocess.TimeoutExpired:
                    emit(trace, "sample")
            emit(trace, "final", return_code)
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)

    return return_code if return_code >= 0 else 128 - return_code


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_monitored(
        args.command,
        output=args.output,
        interval_seconds=args.interval_seconds,
    )


if __name__ == "__main__":
    sys.exit(main())
