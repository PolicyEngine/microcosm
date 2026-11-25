"""A parent process whose pool workers are busy when the test kills it.

Run as ``python -m test_support.microcosm_build.worker_pool_orphan_parent DIR``:
each worker writes ``DIR/worker-<pid>`` and then sleeps for a minute. With
``--fork-holder``, once both workers are running the parent forks a child that
keeps copies of the parent's pipe ends for a minute and writes its process ID
to ``DIR/holder``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from microcosm.build.worker_pool import WorkerPool, WorkerPoolSettings
from test_support.microcosm_build import worker_pool_units as units


def _fork_a_holder(directory: Path) -> None:
    while len(list(directory.glob("worker-*"))) < 2:
        time.sleep(0.1)
    holder = os.fork()
    if holder == 0:
        time.sleep(60)
        os._exit(0)
    (directory / "holder").write_text(str(holder))


def main(directory: str, fork_holder: bool) -> None:
    if fork_holder:
        threading.Thread(
            target=_fork_a_holder, args=(Path(directory),), daemon=True
        ).start()
    with WorkerPool(
        WorkerPoolSettings(workers=2), unit_function=units.write_pid_then_sleep
    ) as pool:
        list(pool.map([(directory, 60.0), (directory, 60.0)]))


if __name__ == "__main__":
    main(sys.argv[1], "--fork-holder" in sys.argv[2:])
