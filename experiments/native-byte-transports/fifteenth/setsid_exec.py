"""macOS has no setsid(1): start a new session, cap CPU, then exec the probe.

    python setsid_exec.py <cpu_seconds> <program> [args...]

The pid printed by the launching shell is the session leader and stays the
probe's own pid across the exec, so it is the pid to record and the only pid
that would ever be signalled.
"""

import os
import resource
import sys

os.setsid()
limit = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
os.execv(sys.argv[2], sys.argv[2:])
