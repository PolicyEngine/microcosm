"""Wait until this host may take the next step, on Fleet ops' terms: no booking refuses this caller, and available memory
(psutil's virtual_memory().available) and free disk (statvfs, decimal GB) clear their floors.

Usage: host_wait.py <min available GiB> <min free GB> <log>. Returns as soon as all three hold. Writes a line to the log every 30 minutes while it waits.
"""

import os
import subprocess
import sys
import time
from datetime import UTC, datetime

import psutil

need_mem, need_disk, log = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
CALLER = "claude-state-child-spm"
RES = os.path.expanduser("~/reviews/fleet-ops/tools/reservation.py")
t = 0
while True:
    mem = psutil.virtual_memory().available / 2**30
    s = os.statvfs("/System/Volumes/Data")
    disk = s.f_bavail * s.f_frsize / 1e9
    booked = subprocess.run(
        [sys.executable, RES, "allowed", CALLER], capture_output=True
    ).returncode
    if mem >= need_mem and disk >= need_disk and booked == 0:
        sys.exit(0)
    if t % 30 == 0:
        with open(log, "a") as f:
            f.write(
                f"{datetime.now(UTC):%H:%M:%SZ} hold: available {mem:.1f} GiB of {need_mem:.0f}, free {disk:.1f} GB of {need_disk:.0f}, booking exit {booked}\n"
            )
    t += 1
    time.sleep(60)
