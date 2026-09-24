"""Tests for the CI process-tree memory monitor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.ci_memory_monitor import (
    ProcessMemory,
    _current_cgroup_memory,
    process_tree,
    read_kib_fields,
    run_monitored,
)


def test_read_kib_fields_converts_proc_values_to_bytes() -> None:
    fields = read_kib_fields("MemTotal:       16384 kB\nMemAvailable:    4096 kB\n")

    assert fields == {
        "MemTotal": 16_777_216,
        "MemAvailable": 4_194_304,
    }


def test_process_tree_includes_only_root_and_descendants() -> None:
    processes = {
        10: ProcessMemory(10, 1, 1_000, 800, "pytest"),
        11: ProcessMemory(11, 10, 2_000, 1_500, "worker-0"),
        12: ProcessMemory(12, 11, 3_000, None, "test"),
        20: ProcessMemory(20, 1, 9_000, 8_000, "unrelated"),
    }

    assert [process.pid for process in process_tree(processes, 10)] == [10, 11, 12]


def test_current_cgroup_memory_uses_process_v2_membership(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    process_cgroup = cgroup_root / "actions-job"
    (proc_root / "self").mkdir(parents=True)
    process_cgroup.mkdir(parents=True)
    (proc_root / "self" / "cgroup").write_text("0::/actions-job\n")
    (process_cgroup / "memory.current").write_text("1048576\n")
    (process_cgroup / "memory.peak").write_text("2097152\n")
    (process_cgroup / "memory.max").write_text("4194304\n")
    (process_cgroup / "memory.events").write_text("oom 2\noom_kill 1\n")

    memory = _current_cgroup_memory(proc_root, cgroup_root)

    assert memory["cgroup_path"] == str(process_cgroup)
    assert memory["cgroup_current_bytes"] == 1_048_576
    assert memory["cgroup_peak_bytes"] == 2_097_152
    assert memory["cgroup_limit_bytes"] == 4_194_304
    assert memory["cgroup_oom_events"] == 2
    assert memory["cgroup_oom_kills"] == 1


def test_run_monitored_records_samples_and_preserves_exit_status(
    tmp_path: Path,
) -> None:
    output = tmp_path / "memory.jsonl"

    status = run_monitored(
        [sys.executable, "-c", "raise SystemExit(7)"],
        output=output,
        interval_seconds=0.01,
        proc_root=tmp_path / "missing-proc",
        cgroup_root=tmp_path / "missing-cgroup",
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert status == 7
    assert records[0]["phase"] == "start"
    assert records[-1]["phase"] == "final"
    assert records[-1]["exit_code"] == 7
    assert all(record["root_pid"] > 0 for record in records)
