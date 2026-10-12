"""Process-tree resource sampling for telemetry events."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  # Base installs use the standard-library fallback.
    psutil = None


def _empty_sample(peak_rss: int) -> dict[str, Any]:
    return {
        "cpu_user_seconds": 0.0,
        "cpu_system_seconds": 0.0,
        "rss_bytes": 0,
        "peak_rss_bytes": peak_rss,
    }


class ProcessTreeSampler:
    """Collect cumulative CPU and resident memory for a build process tree."""

    def __init__(self, parent_pid: int) -> None:
        self.parent_pid = parent_pid
        self._peak_rss = 0
        self._parent_create_time = self._create_time()

    def _create_time(self) -> float | str | None:
        if psutil is not None:
            try:
                return psutil.Process(self.parent_pid).create_time()
            except (psutil.Error, OSError):
                return None
        stat_path = Path(f"/proc/{self.parent_pid}/stat")
        try:
            return stat_path.read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return None

    def parent_alive(self) -> bool:
        """Return whether the sampled process still has its original identity."""

        if psutil is not None:
            try:
                process = psutil.Process(self.parent_pid)
                return (
                    self._parent_create_time is not None
                    and process.create_time() == self._parent_create_time
                    and process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                )
            except (psutil.Error, OSError):
                return False
        try:
            os.kill(self.parent_pid, 0)
        except OSError:
            return False
        current = self._create_time()
        # /proc supplies a process-start tick that protects against PID reuse.
        # On platforms without it, existence is the best standard-library test.
        return self._parent_create_time is None or current == self._parent_create_time

    def sample(self) -> dict[str, Any]:
        """Sample the parent and every currently visible child process."""

        if psutil is None:
            return self._fallback_sample()
        processes = []
        try:
            parent = psutil.Process(self.parent_pid)
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.Error, OSError):
            pass
        user = 0.0
        system = 0.0
        rss = 0
        for process in processes:
            try:
                cpu = process.cpu_times()
                user += float(cpu.user) + float(getattr(cpu, "children_user", 0.0))
                system += float(cpu.system) + float(
                    getattr(cpu, "children_system", 0.0)
                )
                rss += int(process.memory_info().rss)
            except (psutil.Error, OSError):
                continue
        self._peak_rss = max(self._peak_rss, rss)
        return {
            "cpu_user_seconds": user,
            "cpu_system_seconds": system,
            "rss_bytes": rss,
            "peak_rss_bytes": self._peak_rss,
        }

    def _fallback_sample(self) -> dict[str, Any]:
        """Sample Linux process trees when the country engine is not installed."""

        process_rows: dict[int, tuple[int, float, float, int]] = {}
        try:
            clock_ticks = float(os.sysconf("SC_CLK_TCK"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return _empty_sample(self._peak_rss)
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                pid = int(stat_path.parent.name)
                fields = stat_path.read_text().rsplit(")", 1)[1].split()
                process_rows[pid] = (
                    int(fields[1]),
                    (int(fields[11]) + int(fields[13])) / clock_ticks,
                    (int(fields[12]) + int(fields[14])) / clock_ticks,
                    int(fields[21]) * page_size,
                )
            except (OSError, ValueError, IndexError):
                continue
        selected = {self.parent_pid}
        changed = True
        while changed:
            changed = False
            for pid, (parent, *_rest) in process_rows.items():
                if parent in selected and pid not in selected:
                    selected.add(pid)
                    changed = True
        user = sum(process_rows[pid][1] for pid in selected if pid in process_rows)
        system = sum(process_rows[pid][2] for pid in selected if pid in process_rows)
        rss = sum(process_rows[pid][3] for pid in selected if pid in process_rows)
        self._peak_rss = max(self._peak_rss, rss)
        return {
            "cpu_user_seconds": user,
            "cpu_system_seconds": system,
            "rss_bytes": rss,
            "peak_rss_bytes": self._peak_rss,
        }
