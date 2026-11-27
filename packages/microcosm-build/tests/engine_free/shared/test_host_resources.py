"""The memory and CPU cores a process may use (``microcosm.build.host_resources``).

Pool sizes are planned from these (decision 1.1 · 2): the cores the process
may run on, limited by a cgroup CPU quota, and the machine's memory, limited
by a cgroup memory limit. The cgroup files are read from a directory the test
writes, in both cgroup versions' layouts.
"""

from __future__ import annotations

import pytest

from microcosm.build import host_resources

GIB = 2**30


def _write(root, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def test_usable_cores_are_the_cpus_the_process_may_use(tmp_path) -> None:
    assert host_resources.usable_cores(cpu_count=10, cgroup_root=tmp_path) == 10
    assert host_resources.usable_cores() >= 1


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # cgroup v2: no quota, a fractional quota rounded up, a quota above
        # the CPUs the process may use.
        ({"cpu.max": "max 100000"}, 10),
        ({"cpu.max": "150000 100000"}, 2),
        ({"cpu.max": "1600000 100000"}, 10),
        # cgroup v1: no quota, a quota of 2.5 CPUs.
        ({"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, 10),
        ({"cpu/cpu.cfs_quota_us": "250000", "cpu/cpu.cfs_period_us": "100000"}, 3),
        # A file that cannot be read as a quota sets none.
        ({"cpu.max": "unexpected"}, 10),
        ({"cpu/cpu.cfs_quota_us": "100000", "cpu/cpu.cfs_period_us": "0"}, 10),
    ],
)
def test_a_cgroup_cpu_quota_limits_the_usable_cores(tmp_path, files, expected) -> None:
    for relative, text in files.items():
        _write(tmp_path, relative, text)
    assert host_resources.usable_cores(cpu_count=10, cgroup_root=tmp_path) == expected


def test_total_memory_is_the_machines_memory(tmp_path) -> None:
    assert (
        host_resources.total_memory_bytes(
            physical_memory_bytes=32 * GIB, cgroup_root=tmp_path
        )
        == 32 * GIB
    )
    assert host_resources.total_memory_bytes() > 0


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"memory.max": "max"}, 32 * GIB),
        ({"memory.max": str(8 * GIB)}, 8 * GIB),
        ({"memory.max": str(64 * GIB)}, 32 * GIB),
        # cgroup v1 reports "no limit" as a number near 2**63.
        ({"memory/memory.limit_in_bytes": str(2**63 - 4096)}, 32 * GIB),
        ({"memory/memory.limit_in_bytes": str(12 * GIB)}, 12 * GIB),
        ({"memory.max": "unexpected"}, 32 * GIB),
    ],
)
def test_a_cgroup_memory_limit_limits_the_total_memory(
    tmp_path, files, expected
) -> None:
    for relative, text in files.items():
        _write(tmp_path, relative, text)
    assert (
        host_resources.total_memory_bytes(
            physical_memory_bytes=32 * GIB, cgroup_root=tmp_path
        )
        == expected
    )
