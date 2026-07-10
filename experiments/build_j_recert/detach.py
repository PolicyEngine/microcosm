#!/usr/bin/env python3
"""Detach a Build J chunk so it survives session/terminal death.

Usage: detach.py <pidfile> <wrapper_log> <script.sh> [args...]

Launches `caffeinate -i bash <script.sh> [args...]` in its own session
(start_new_session=True -> setsid semantics; reparented to init if the
launching session dies), appends combined output to <wrapper_log>, records
the child PID in <pidfile>, and exits immediately. No killing watchdog:
the chunk runs to natural completion; monitoring is via the script's own
rc/append logs + a memory-pressure sampler. STAGING/LOCAL ONLY.
"""
import os
import subprocess
import sys
import time

pidfile, wrapper_log, script, *rest = sys.argv[1:]
logf = open(wrapper_log, "a", buffering=1)
logf.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] detach launching: "
           f"caffeinate -i bash {script} {' '.join(rest)}\n")
p = subprocess.Popen(
    ["caffeinate", "-i", "bash", script, *rest],
    stdout=logf,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
with open(pidfile, "w") as f:
    f.write(str(p.pid) + "\n")
logf.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] detached child pid={p.pid} "
           f"(caffeinate wrapper; bash+python are its children)\n")
print(f"detached pid={p.pid} pidfile={pidfile} log={wrapper_log}")
