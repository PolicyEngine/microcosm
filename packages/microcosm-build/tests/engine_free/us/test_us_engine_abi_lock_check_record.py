"""US base processes reuse their parent's check of ``engine_abi.lock.json``.

``--stage all`` checks the policyengine-us lock file once and passes each step
process a record of that check. A step process passes the record on to the QRF
workers that ``primary_qrf_chain`` starts and to the clean import trace behind
the QRF worker identity. A step process or QRF worker refuses to start when the
lock file or the installed policyengine-us differs from its parent's record.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from microcosm.build.us_runtime import puf_qrf_chain, take_up_contract, worker_identity
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
RECORD_VARIABLE = "MICROCOSM_ENGINE_ABI_LOCK_CHECK_RECORD"
US_LOCK_PATH = (
    _TEST_PATHS.package / "src" / "microcosm" / "build" / "us" / "engine_abi.lock.json"
)
BASE_TOOL = _TEST_PATHS.repository / "tools" / "build_us_puf_support_base.py"


def _load_support_builder_module():
    spec = importlib.util.spec_from_file_location(
        "build_us_puf_support_base", BASE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stub_record_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    environment = {RECORD_VARIABLE: '{"stub": "record of this process"}'}
    monkeypatch.setattr(
        take_up_contract,
        "us_engine_abi_lock_check_environment",
        lambda: dict(environment),
    )
    return environment


def test_staged_step_processes_receive_this_processes_lock_check_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _load_support_builder_module()
    monkeypatch.delenv(RECORD_VARIABLE, raising=False)
    expected = _stub_record_environment(monkeypatch)

    environment = builder._staged_subprocess_environment()

    assert environment[RECORD_VARIABLE] == expected[RECORD_VARIABLE]


def test_qrf_workers_and_the_import_trace_receive_the_lock_check_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One launch environment serves the QRF workers and the identity trace.

    Both therefore check the lock file the same way, so the worker identity
    does not depend on whether the process computing it received a record.
    """

    expected = _stub_record_environment(monkeypatch)

    with worker_identity.primary_qrf_worker_launch_environment() as environment:
        assert environment[RECORD_VARIABLE] == expected[RECORD_VARIABLE]


@pytest.mark.parametrize(
    "caller_environment",
    [None, {"POPULACE_FIT_N_JOBS": "1"}],
    ids=["base tool passes no environment", "caller passes an environment"],
)
def test_qrf_workers_started_by_the_chain_receive_this_processes_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caller_environment: dict[str, str] | None,
) -> None:
    """The step process issues the record each of its QRF workers receives.

    A record that the worker merely inherits was issued by the step process's
    own parent, so the worker would ignore it and run the full check.
    """

    monkeypatch.setenv(RECORD_VARIABLE, '{"stub": "record of the parent"}')
    expected = _stub_record_environment(monkeypatch)
    monkeypatch.setattr(
        puf_qrf_chain, "_load_manifest", lambda root: {"initial_state": {}}
    )
    monkeypatch.setattr(
        puf_qrf_chain, "_manifest_strings", lambda manifest, key: ["first_target"]
    )
    monkeypatch.setattr(
        puf_qrf_chain.QRFChainState, "from_dict", staticmethod(lambda value: None)
    )
    monkeypatch.setattr(
        puf_qrf_chain,
        "_target_path",
        lambda root, manifest, index: tmp_path / "not-yet-written",
    )
    launched: list[dict[str, str]] = []

    def stopped_worker(
        command: list[str], *, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        launched.append(env)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(puf_qrf_chain.subprocess, "run", stopped_worker)

    with pytest.raises(RuntimeError, match="failed with exit code 1"):
        puf_qrf_chain.run_primary_puf_qrf_chain(
            tmp_path, environment=caller_environment
        )

    assert len(launched) == 1
    assert launched[0][RECORD_VARIABLE] == expected[RECORD_VARIABLE]
    for name, value in (caller_environment or {}).items():
        assert launched[0][name] == value


def _installed_policyengine_us_version() -> str:
    try:
        return importlib.metadata.version("policyengine-us")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _record_from_this_process(*, lock_sha256: str, engine_version: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "issuer_pid": os.getpid(),
            "lock_sha256": lock_sha256,
            "engine": {"package": "policyengine-us", "version": engine_version},
            "domains_sha256": "0" * 64,
        }
    )


_PROCESS_COMMANDS = {
    "step process": [sys.executable, str(BASE_TOOL), "--help"],
    "QRF worker": [
        sys.executable,
        "-m",
        "microcosm.build.us_runtime.puf_qrf_worker",
        "--help",
    ],
}


@pytest.mark.parametrize("process", sorted(_PROCESS_COMMANDS))
@pytest.mark.parametrize(
    ("difference", "refusal"),
    [
        ("lock file", "lock file differs"),
        ("installed policyengine-us", "installed policyengine-us"),
    ],
)
def test_a_base_process_refuses_to_start_when_its_parents_check_no_longer_holds(
    process: str,
    difference: str,
    refusal: str,
    tmp_path: Path,
) -> None:
    lock_sha256 = hashlib.sha256(US_LOCK_PATH.read_bytes()).hexdigest()
    engine_version = _installed_policyengine_us_version()
    if difference == "lock file":
        lock_sha256 = hashlib.sha256(b"a lock file the parent checked").hexdigest()
    else:
        engine_version = "0.0.0+checked-by-the-parent"
    environment = {
        **os.environ,
        RECORD_VARIABLE: _record_from_this_process(
            lock_sha256=lock_sha256,
            engine_version=engine_version,
        ),
    }

    completed = subprocess.run(
        _PROCESS_COMMANDS[process],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert completed.returncode != 0, completed.stdout
    assert refusal in completed.stderr, completed.stderr[-4000:]
