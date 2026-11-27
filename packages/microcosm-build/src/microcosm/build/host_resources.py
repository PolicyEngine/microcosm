"""How much memory and how many CPU cores this process may use.

Builds size their worker pools from the machine they run on (decision 1.1 · 2:
a budget of 75% of memory, and only the cores the process may use). These two
readings are what that sizing starts from:

- :func:`usable_cores`: the CPUs this process may run on
  (``os.process_cpu_count()``, which follows the CPU affinity mask, so
  ``taskset`` and container cpusets, and ``PYTHON_CPU_COUNT``), limited by a
  cgroup CPU quota rounded up (cgroup v2 ``cpu.max``, v1 ``cpu.cfs_quota_us``
  over ``cpu.cfs_period_us``). These are ``joblib.cpu_count()``'s rules
  without ``LOKY_MAX_CPU_COUNT``, which the release tool fills with its
  thread count.
- :func:`total_memory_bytes`: the machine's physical memory, limited by a
  cgroup memory limit (v2 ``memory.max``, v1 ``memory.limit_in_bytes``), as a
  container such as Modal's is limited.

The cgroup files are read from ``/sys/fs/cgroup`` on Linux only; a file that is
absent or cannot be read as a limit sets none. Pooled post-export scoring
(tracker task 1.4 · 7) uses these; task 1.8 · 1 adds the budget options on top.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

__all__ = ["total_memory_bytes", "usable_cores"]

_DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_root(cgroup_root: Path | None) -> Path | None:
    if cgroup_root is not None:
        return Path(cgroup_root)
    if sys.platform.startswith("linux"):
        return _DEFAULT_CGROUP_ROOT
    return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cpu_quota(root: Path) -> float | None:
    """The cgroup CPU quota in CPUs, or ``None`` for none."""
    text = _read(root / "cpu.max")
    if text is not None:
        fields = text.split()
        if len(fields) == 2 and fields[0] != "max":
            try:
                quota, period = int(fields[0]), int(fields[1])
            except ValueError:
                return None
            if quota > 0 and period > 0:
                return quota / period
        return None
    quota_text = _read(root / "cpu" / "cpu.cfs_quota_us")
    period_text = _read(root / "cpu" / "cpu.cfs_period_us")
    if quota_text is None or period_text is None:
        return None
    try:
        quota, period = int(quota_text), int(period_text)
    except ValueError:
        return None
    if quota > 0 and period > 0:
        return quota / period
    return None


def usable_cores(
    *, cpu_count: int | None = None, cgroup_root: Path | None = None
) -> int:
    """The CPU cores this process may use, at least one.

    ``cpu_count`` replaces ``os.process_cpu_count()``, and ``cgroup_root`` the
    cgroup directory, for tests.
    """
    if cpu_count is None:
        cpu_count = os.process_cpu_count() or 1
    cores = max(1, int(cpu_count))
    root = _cgroup_root(cgroup_root)
    if root is not None:
        quota = _cpu_quota(root)
        if quota is not None:
            cores = min(cores, max(1, math.ceil(quota)))
    return cores


def _memory_limit(root: Path) -> int | None:
    """The cgroup memory limit in bytes, or ``None`` for none."""
    for relative in ("memory.max", "memory/memory.limit_in_bytes"):
        text = _read(root / relative)
        if text is None:
            continue
        try:
            limit = int(text)
        except ValueError:
            # cgroup v2 writes "max" for no limit.
            return None
        return limit if limit > 0 else None
    return None


def total_memory_bytes(
    *, physical_memory_bytes: int | None = None, cgroup_root: Path | None = None
) -> int:
    """The memory this process's machine or container has, in bytes.

    ``physical_memory_bytes`` replaces the machine's physical memory
    (``SC_PHYS_PAGES`` pages of ``SC_PAGE_SIZE`` bytes, the same as
    ``hw.memsize`` on macOS), and ``cgroup_root`` the cgroup directory, for
    tests. A cgroup v1 limit of "no limit" is a number near ``2**63``, which
    the physical memory then bounds.
    """
    if physical_memory_bytes is None:
        physical_memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    total = int(physical_memory_bytes)
    root = _cgroup_root(cgroup_root)
    if root is not None:
        limit = _memory_limit(root)
        if limit is not None:
            total = min(total, limit)
    return total
