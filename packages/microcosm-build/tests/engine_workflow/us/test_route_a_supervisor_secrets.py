"""Route A's release credential through the real supervisor.

The supervisor imports psutil, which the engine-free job does not install, so
this runs in the US engine job. The engine-free twin in
``engine_free/us/test_route_a_driver.py`` checks the wrapper on its own.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil  # noqa: F401 - the supervisor needs it; fail here, not in the child

from test_support.paths import paths_for

ROUTE_A_TOOLS = paths_for("microcosm-build").repository / "tools" / "route_a"
_SECRET_ALIASES = (
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_TOKEN_MAX",
)


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_supervised_release_gets_the_credential_only_in_its_environment(
    tmp_path: Path,
) -> None:
    # This deliberately does not resemble a Hub credential and never invokes
    # the actual agent-secret executable.
    marker = "route-a-dummy-environment-value"
    secret = _write_executable(
        tmp_path / "secret-stub",
        """#!/bin/bash
set -eu
[ "$#" = 2 ] && [ "$1" = get ] && [ "$2" = HUGGING_FACE_TOKEN_MAX ]
printf '%s\\n' "$ROUTE_A_TEST_SECRET"
printf '%s\\n' "$ROUTE_A_TEST_SECRET" >&2
""",
    )
    report = tmp_path / "child-report.json"
    child = tmp_path / "child.py"
    child.write_text(
        """import json, os, sys
import psutil
marker = os.environ["ROUTE_A_TEST_SECRET"]
payload = {
    "token_present": os.environ.get("HF_TOKEN") == marker,
    "argv_contains_token": any(marker in a for a in psutil.Process().cmdline()),
    "aliases_present": [key for key in (
        "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_TOKEN_MAX"
    ) if key in os.environ],
    "pid": os.getpid(),
}
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream)
print("dummy release child finished")
""",
        encoding="utf-8",
    )
    config = tmp_path / "release-config.json"
    config.write_text(
        json.dumps(
            {
                "argv": [
                    "bash",
                    "-x",
                    str(ROUTE_A_TOOLS / "with_hf_token.sh"),
                    str(secret),
                    sys.executable,
                    str(child),
                    str(report),
                ],
                "cwd": str(tmp_path),
                "env": {"PYTHONUNBUFFERED": "1"},
                "limits": {
                    "wall_seconds": 60,
                    "cpu_seconds": 10,
                    "rss_bytes": 512 * 1024**2,
                    "output_bytes": 1024**2,
                    "log_bytes": 1024**2,
                    "disk_floor_bytes": 0,
                    "disk_admission_bytes": 0,
                    "available_ram_admission_bytes": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "ROUTE_A_TEST_SECRET": marker, "HF_TOKEN": "stale-value"}
    env.update(dict.fromkeys(_SECRET_ALIASES, "stale-value"))
    out = tmp_path / "release-sup"
    result = subprocess.run(
        [sys.executable, str(ROUTE_A_TOOLS / "supervise.py"), str(out), str(config)],
        env=env,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr.decode()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["token_present"] is True
    assert payload["argv_contains_token"] is False
    assert payload["aliases_present"] == []
    # exec, not a fork: the supervisor's recorded pid is the release itself,
    # so its process-group kill reaches the release.
    assert payload["pid"] == json.loads((out / "PID.json").read_text())["pid"]
    assert json.loads((out / "RESULT.json").read_text())["status"] == "COMPLETED"
    assert marker.encode() not in result.stdout + result.stderr
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert marker.encode() not in path.read_bytes(), path.name
