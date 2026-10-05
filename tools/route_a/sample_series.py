"""Record a resource time series for one supervise.py stage (route A).

    python sample_series.py <supervise-dir> <series.csv> <supervisor-pid> [interval-seconds]

supervise.py keeps only the latest HEARTBEAT.json (peak RSS, overwritten every
15 s), so it cannot answer "what did RSS do over the run". This sampler appends
one row per interval with the stage's current process-tree RSS (the child named
in PID.json plus its descendants, the same tree supervise.py sums), machine
available RAM, swap in use and free disk. It stops when RESULT.json appears or
the supervisor process is gone. It records no row values of any dataset.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil

FIELDS = (
    "utc",
    "stage_dir",
    "child_pid",
    "tree_rss_bytes",
    "tree_cpu_seconds",
    "available_ram_bytes",
    "swap_used_bytes",
    "free_disk_bytes",
)


def _tree(pid: int) -> tuple[int, float]:
    try:
        root = psutil.Process(pid)
        members = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0, 0.0
    rss = 0
    cpu = 0.0
    for member in members:
        try:
            rss += member.memory_info().rss
            times = member.cpu_times()
            cpu += times.user + times.system
        except psutil.Error:
            pass
    return rss, cpu


def main() -> int:
    stage_dir = Path(sys.argv[1])
    series = Path(sys.argv[2])
    supervisor_pid = int(sys.argv[3])
    interval = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    new_file = not series.exists()
    with series.open("a", newline="") as stream:
        writer = csv.writer(stream)
        if new_file:
            writer.writerow(FIELDS)
        while True:
            if (stage_dir / "RESULT.json").exists():
                return 0
            if not psutil.pid_exists(supervisor_pid):
                return 0
            child_pid = 0
            try:
                child_pid = int(json.loads((stage_dir / "PID.json").read_text())["pid"])
            except (OSError, ValueError, KeyError):
                pass
            rss, cpu = _tree(child_pid) if child_pid else (0, 0.0)
            try:
                free = shutil.disk_usage(stage_dir).free
            except OSError:
                free = -1
            writer.writerow(
                (
                    datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    stage_dir.name,
                    child_pid,
                    rss,
                    round(cpu, 1),
                    psutil.virtual_memory().available,
                    psutil.swap_memory().used,
                    free,
                )
            )
            stream.flush()
            time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
