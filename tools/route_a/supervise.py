"""Bounded supervisor for local hours-rebuild runs. Metadata-only progress.

    python supervise.py <new-output-dir> <config.json>

The config names ``argv``, ``cwd``, ``env`` and ``limits`` (wall_seconds,
cpu_seconds, rss_bytes, output_bytes, log_bytes, disk_floor_bytes,
disk_admission_bytes, available_ram_admission_bytes). The output directory must
not exist. The child runs in its own process group; on a breached limit the
supervisor kills exactly that group. It writes ADMISSION.json, COMMAND.json, a
HEARTBEAT.json every 15 seconds and RESULT.json; it never prints or records
row values, and the child's log is capped, not parsed.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil


def _tree_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def main() -> int:
    out = Path(sys.argv[1])
    config = json.loads(Path(sys.argv[2]).read_text())
    limits = config["limits"]
    out.mkdir(parents=True, exist_ok=False)

    available = psutil.virtual_memory().available
    free = shutil.disk_usage(out).free
    admitted = (
        available >= limits["available_ram_admission_bytes"]
        and free >= limits["disk_admission_bytes"]
    )
    (out / "ADMISSION.json").write_text(
        json.dumps(
            {
                "admitted": admitted,
                "observed_available_ram_bytes": available,
                "observed_free_disk_bytes": free,
                "limits": limits,
                "authorization": config.get("authorization"),
                "automatic_retry": False,
            },
            indent=1,
        )
        + "\n"
    )
    if not admitted:
        (out / "RESULT.json").write_text(
            json.dumps({"status": "REFUSED_ADMISSION"}, indent=1) + "\n"
        )
        return 3

    (out / "COMMAND.json").write_text(
        json.dumps(
            {
                "argv": config["argv"],
                "cwd": config["cwd"],
                "env": config.get("env", {}),
                "source_identity": config.get("source_identity"),
                "started_at_unix": time.time(),
            },
            indent=1,
        )
        + "\n"
    )
    env = {**os.environ, **config.get("env", {})}
    env.pop("UV_FROZEN", None)
    start = time.monotonic()
    peak_rss = 0
    cpu = 0.0
    reason = None
    last_beat = 0.0
    # A disk-floor breach must persist before it kills the child: local
    # snapshot churn on this Mac can swing free space by tens of GB within
    # seconds, and the staging child writes nothing until its export.
    floor_breach_seconds = float(limits.get("disk_floor_persist_seconds", 0))
    floor_breached_since = None
    with (out / "run.log").open("wb") as log:
        child = subprocess.Popen(
            config["argv"],
            cwd=config["cwd"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (out / "PID.json").write_text(json.dumps({"pid": child.pid}) + "\n")
        try:
            while child.poll() is None:
                try:
                    root = psutil.Process(child.pid)
                    members = [root, *root.children(recursive=True)]
                    rss = 0
                    cpu_now = 0.0
                    for member in members:
                        try:
                            rss += member.memory_info().rss
                            times = member.cpu_times()
                            cpu_now += times.user + times.system
                        except psutil.Error:
                            pass
                    peak_rss = max(peak_rss, rss)
                    cpu = max(cpu, cpu_now)
                except psutil.Error:
                    pass
                wall = time.monotonic() - start
                output_bytes = _tree_size(out)
                log_bytes = os.fstat(log.fileno()).st_size
                free_now = shutil.disk_usage(out).free
                if wall > limits["wall_seconds"]:
                    reason = "WALL"
                elif cpu > limits["cpu_seconds"]:
                    reason = "CPU"
                elif peak_rss > limits["rss_bytes"]:
                    reason = "RSS"
                elif output_bytes > limits["output_bytes"]:
                    reason = "OUTPUT"
                elif log_bytes > limits["log_bytes"]:
                    reason = "LOG"
                elif free_now < limits["disk_floor_bytes"]:
                    if floor_breached_since is None:
                        floor_breached_since = wall
                    if wall - floor_breached_since >= floor_breach_seconds:
                        reason = "DISK_FLOOR"
                else:
                    floor_breached_since = None
                if reason:
                    break
                if wall - last_beat >= 15:
                    last_beat = wall
                    (out / "HEARTBEAT.json").write_text(
                        json.dumps(
                            {
                                "pid": child.pid,
                                "wall_seconds": wall,
                                "cpu_seconds": cpu,
                                "peak_rss_bytes": peak_rss,
                                "output_bytes": output_bytes,
                                "log_bytes": log_bytes,
                                "free_disk_bytes": free_now,
                            }
                        )
                        + "\n"
                    )
                time.sleep(1.0)
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            returncode = child.wait()
    (out / "RESULT.json").write_text(
        json.dumps(
            {
                "status": "COMPLETED" if reason is None and returncode == 0 else "FAIL",
                "returncode": returncode,
                "refusal": reason,
                "wall_seconds": time.monotonic() - start,
                "cpu_seconds": cpu,
                "peak_rss_bytes": peak_rss,
                "output_bytes": _tree_size(out),
                "release_acceptance": False,
            },
            indent=1,
        )
        + "\n"
    )
    return 0 if reason is None and returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
