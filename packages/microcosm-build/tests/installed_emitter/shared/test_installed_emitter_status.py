"""The Modal stage runner records what the installed emitter left undelivered.

The runner's other tests stand in for the image emitter's ``status``. These run
a stage through the runner's own ``_run_stage`` against local directories, with
the separately installed ``microcosm-emitter`` at the image's emitter path: the
stage's tool starts it through the interface with its telemetry module, as a
build does, and the runner then runs its ``status`` and records the answer in
the receipt.

The tool's plan opts out of telemetry (``MICROCOSM_TELEMETRY_REQUIRED=0``), so
the emitter is given no collector token, and the test environment holds no
Hugging Face credential. The emitter therefore keeps the run's events in its
event queue file, as ``local_only``, and sends nothing. Its collector origin is
a closed loopback port all the same, so that nothing could reach the collector.

``tools/run_installed_emitter_tests.sh`` installs the minimum release builds
accept and the newest pinned release, and runs this file once for each, with
``MICROCOSM_EMITTER_COMMAND`` naming the installed command.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import socket
import subprocess
import sysconfig
from pathlib import Path

import pytest

from test_support.microcosm_build.us_modal_stage_plan_tool import (
    fake_plan,
    fake_tool,
    install_fake_runner,
    load_app,
    plan_lib,
)

#: The telemetry run a stage of the fake plan reports under.
RUN_ID = "fake-run.run"
PRODUCER_ID = "installed-emitter-status-test"
#: The fake runner's venv Python is a link to this interpreter, which then
#: does not find this environment's packages; a tool that imports the
#: interface puts them on its path, the first argument after its state.
_SITE_PACKAGES = sysconfig.get_paths()["purelib"]
#: A tool that starts the emitter with its telemetry module, records two
#: stages and the build's completion, and closes it, as a build program does.
#: The emitter, in its own session, holds the tool's output until it has
#: finished, so the runner asks for its status only then.
_EMITTING_TOOL = """\
import pathlib, sys

state, site_packages, collector, run_id, producer_id = sys.argv[1:6]
sys.path.insert(0, site_packages)
from microcosm_emitter_interface import EmitterClient, TelemetryRun, TelemetrySettings

run = TelemetryRun(
    run_id=run_id,
    country_code="US",
    pipeline="installed_emitter_status",
    producer_id=producer_id,
)
client = EmitterClient.start(
    telemetry=TelemetrySettings(
        run=run, development_collector_url=collector, heartbeat_seconds=3600
    )
)
if not client.available:
    sys.exit("the installed emitter did not start")
client.telemetry.stage("first_stage")
client.telemetry.stage("second_stage")
client.telemetry.complete()
client.close()
(pathlib.Path(state) / "out").mkdir(parents=True, exist_ok=True)
(pathlib.Path(state) / "out" / "result.txt").write_text("done\\n")
"""
#: A tool that starts the emitter the same way, writes the emitter host's
#: process ID to a file, records two stages and exits at once, without
#: finishing its run or closing the emitter, as a build that is killed does.
#: The emitter notices its parent's exit only afterwards, records the run's
#: failure and stops.
_UNFINISHED_TOOL = """\
import os, pathlib, sys

state, site_packages, collector, run_id, producer_id, pid_file = sys.argv[1:7]
sys.path.insert(0, site_packages)
from microcosm_emitter_interface import EmitterClient, TelemetryRun, TelemetrySettings

run = TelemetryRun(
    run_id=run_id,
    country_code="US",
    pipeline="installed_emitter_status",
    producer_id=producer_id,
)
client = EmitterClient.start(
    telemetry=TelemetrySettings(
        run=run, development_collector_url=collector, heartbeat_seconds=3600
    )
)
if not client.available:
    sys.exit("the installed emitter did not start")
# The interface (0.7.0) has no public way to name the host's process, so
# this reads the client's private handle on it.
pathlib.Path(pid_file).write_text(str(client._process.process.pid))
client.telemetry.stage("first_stage")
client.telemetry.stage("second_stage")
(pathlib.Path(state) / "out").mkdir(parents=True, exist_ok=True)
(pathlib.Path(state) / "out" / "result.txt").write_text("done\\n")
sys.stdout.flush()
os._exit(0)
"""
#: A tool that leaves a file that is not an event queue file where the
#: emitter would keep one, and starts no emitter.
_UNUSABLE_FILE_TOOL = """\
import pathlib, sys

sys.path.insert(0, sys.argv[2])
from microcosm_emitter_interface import default_spool_path

path = default_spool_path()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(b"not an event queue file")
(pathlib.Path(sys.argv[1]) / "out").mkdir(parents=True, exist_ok=True)
(pathlib.Path(sys.argv[1]) / "out" / "result.txt").write_text("done\\n")
"""


@pytest.fixture
def installed_command() -> str:
    command = os.environ.get("MICROCOSM_EMITTER_COMMAND", "").strip()
    if not command or shutil.which(command) is None:
        pytest.fail(
            "Run these tests with tools/run_installed_emitter_tests.sh, which sets "
            "MICROCOSM_EMITTER_COMMAND."
        )
    return command


@pytest.fixture
def runner(installed_command: str, tmp_path: Path, monkeypatch):
    """The runner's app, with the installed emitter at the image's emitter path.

    Only ``status`` runs at that path; ``version`` keeps its recorded answer.
    """

    app = load_app(monkeypatch, real_emitter_status=True)
    monkeypatch.setattr(plan_lib, "IMAGE_EMITTER_COMMAND", installed_command)
    return install_fake_runner(app, tmp_path, monkeypatch)


def _closed_loopback_origin() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _tool(script: str, *arguments: str) -> plan_lib.ToolSpec:
    return dataclasses.replace(
        fake_tool(),
        argv_builder=lambda plan, paths, state: ["-c", script, state, *arguments],
    )


def _opted_out_plan() -> dict:
    return fake_plan(max_wall_seconds=600, env={"MICROCOSM_TELEMETRY_REQUIRED": "0"})


def test_the_runner_records_the_events_the_installed_emitter_still_holds(
    runner,
) -> None:
    app = runner(
        _tool(
            _EMITTING_TOOL,
            _SITE_PACKAGES,
            _closed_loopback_origin(),
            RUN_ID,
            PRODUCER_ID,
        )
    )

    receipt = app._run_stage(_opted_out_plan())

    assert receipt["status"] == "COMPLETED"
    # The run's start, its two stages, the end of each stage when the run
    # completes (microcosm-emitter 0.14.0 and later) and the completion.
    undelivered = receipt["runner"]["telemetry"]["undelivered_events"]
    assert undelivered == {
        "runs": [
            {
                "run_id": RUN_ID,
                "producer_id": PRODUCER_ID,
                "upload_state": "local_only",
                "reason": "missing_huggingface_credential",
                "undelivered_events": 6,
            }
        ],
        "problem": None,
    }


def _process_running(pid: int) -> bool:
    """Whether ``pid`` is a process that has not exited.

    A process that has exited but that its parent, here the system's process
    reaper since the tool exited first, has not yet collected still answers
    a signal 0, so ``ps`` reads its state instead: none for a process that
    is gone, ``Z`` for one that has exited and awaits collection.
    """

    answer = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = answer.stdout.strip()
    return answer.returncode == 0 and bool(state) and not state.startswith("Z")


def test_the_runner_asks_for_status_only_once_the_build_s_emitter_has_exited(
    runner, monkeypatch, tmp_path: Path
) -> None:
    """The emitter host holds the tool's output, which the runner reads to its end.

    The interface starts the host in its own session with its standard
    output sent nowhere, so only the standard error it inherits holds the
    runner's pipe; this test fails if a change there let the runner ask for
    ``status`` while the host was still running.
    """

    pid_file = tmp_path / "emitter-host.pid"
    app = runner(
        _tool(
            _UNFINISHED_TOOL,
            _SITE_PACKAGES,
            _closed_loopback_origin(),
            RUN_ID,
            PRODUCER_ID,
            str(pid_file),
        )
    )
    runner_s_own_status = app._emitter_status
    host_running_when_asked = []

    def status(env: dict[str, str]) -> dict:
        host_running_when_asked.append(_process_running(int(pid_file.read_text())))
        return runner_s_own_status(env)

    monkeypatch.setattr(app, "_emitter_status", status)

    receipt = app._run_stage(_opted_out_plan())

    assert receipt["status"] == "COMPLETED"
    assert host_running_when_asked == [False]
    # The run's start, its two stages, the failure the emitter records once it
    # notices the build has gone, and before it the end of each open stage
    # (microcosm-emitter 0.14.0 and later).
    assert receipt["runner"]["telemetry"]["undelivered_events"] == {
        "runs": [
            {
                "run_id": RUN_ID,
                "producer_id": PRODUCER_ID,
                "upload_state": "local_only",
                "reason": "missing_huggingface_credential",
                "undelivered_events": 6,
            }
        ],
        "problem": None,
    }


def test_the_runner_records_no_runs_when_no_emitter_kept_an_event_queue_file(
    runner,
) -> None:
    app = runner(fake_tool())

    receipt = app._run_stage(_opted_out_plan())

    assert receipt["status"] == "COMPLETED"
    assert receipt["runner"]["telemetry"]["undelivered_events"] == {
        "runs": [],
        "problem": None,
    }


def test_the_runner_records_the_installed_emitter_s_refusal_of_an_unusable_file(
    runner,
) -> None:
    app = runner(_tool(_UNUSABLE_FILE_TOOL, _SITE_PACKAGES))

    receipt = app._run_stage(_opted_out_plan())

    assert receipt["status"] == "COMPLETED"
    path = Path(os.environ["XDG_CACHE_HOME"]) / "microcosm/telemetry/events.sqlite3"
    assert receipt["runner"]["telemetry"]["undelivered_events"] == {
        "runs": None,
        "problem": f"`microcosm-emitter status` exited 9: The file at {path} is "
        "not an event queue file.",
    }
