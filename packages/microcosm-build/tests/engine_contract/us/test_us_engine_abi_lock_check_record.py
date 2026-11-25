"""With their parent's record, US base processes skip the fresh lock derivation.

The full check of ``engine_abi.lock.json`` imports policyengine-us and builds
two tax-benefit systems. A step process or QRF worker that receives a record of
its parent's passed check compares the lock file's SHA-256, the installed
policyengine-us version and the take-up and sources domains instead, so it
starts without importing policyengine-us at all.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.us_runtime import take_up_contract
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
RECORD_VARIABLE = "MICROCOSM_ENGINE_ABI_LOCK_CHECK_RECORD"
US_LOCK_PATH = (
    _TEST_PATHS.package / "src" / "microcosm" / "build" / "us" / "engine_abi.lock.json"
)
BASE_TOOL = _TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"

_PROCESS_COMMANDS = {
    "step process": [str(BASE_TOOL), "--help"],
    "QRF worker": ["-m", "microcosm.build.us_runtime.puf_qrf_worker", "--help"],
}


def _imported_modules(importtime_stderr: str) -> set[str]:
    return {
        line.rsplit("|", 1)[1].strip()
        for line in importtime_stderr.splitlines()
        if line.startswith("import time:") and "|" in line
    }


def test_this_process_issues_a_record_of_its_us_lock_check() -> None:
    environment = take_up_contract.us_engine_abi_lock_check_environment()

    record = json.loads(environment[RECORD_VARIABLE])
    assert record["issuer_pid"] == os.getpid()
    assert record["lock_sha256"] == (
        hashlib.sha256(US_LOCK_PATH.read_bytes()).hexdigest()
    )
    assert record["engine"] == {
        "package": "policyengine-us",
        "version": importlib.metadata.version("policyengine-us"),
    }


@pytest.mark.parametrize("process", sorted(_PROCESS_COMMANDS))
def test_a_base_process_with_its_parents_record_does_not_import_policyengine_us(
    process: str,
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        **take_up_contract.us_engine_abi_lock_check_environment(),
    }

    completed = subprocess.run(
        [sys.executable, "-X", "importtime", *_PROCESS_COMMANDS[process]],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    imported = _imported_modules(completed.stderr)
    assert "microcosm.build.us_runtime" in imported
    assert "policyengine_us" not in imported


def test_the_full_us_bundle_loader_accepts_the_parents_record(
    tmp_path: Path,
) -> None:
    """The full loader compares the same take-up and sources domains.

    Had its digest of those domains differed, the child would refuse; had it
    ignored the record, it would import policyengine-us to derive the lock.
    """

    environment = {
        **os.environ,
        **take_up_contract.us_engine_abi_lock_check_environment(),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "importtime",
            "-c",
            "from microcosm.build.spec_engine import load_bundle\nload_bundle('us')\n",
        ],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    imported = _imported_modules(completed.stderr)
    assert "microcosm.build.spec_engine.engine_abi" in imported
    assert "policyengine_us" not in imported
