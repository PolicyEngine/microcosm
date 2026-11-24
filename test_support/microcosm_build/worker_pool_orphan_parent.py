"""A parent process whose pool workers are busy when the test kills it.

Run as ``python -m test_support.microcosm_build.worker_pool_orphan_parent DIR``:
each worker writes ``DIR/worker-<pid>`` and then sleeps for a minute.
"""

from __future__ import annotations

import sys

from microcosm.build.worker_pool import WorkerPool, WorkerPoolSettings
from test_support.microcosm_build import worker_pool_units as units


def main(directory: str) -> None:
    with WorkerPool(
        WorkerPoolSettings(workers=2), unit_function=units.write_pid_then_sleep
    ) as pool:
        list(pool.map([(directory, 60.0), (directory, 60.0)]))


if __name__ == "__main__":
    main(sys.argv[1])
