"""The d844 watcher, driven against a fake ``modal`` command.

The fake records every call and keeps its state in files: whether the app
is stopped, how many times the app list was read, and the log it streams.
No test runs the real Modal CLI, and none stops a real app.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from test_support.microcosm_build.us_modal_stage_plan_tool import ROOT

WATCH = ROOT / "tools" / "us_head_to_head_watch.sh"
PLAN = ROOT / "docs" / "us-modal-stage-route-a-h2h-plan.json"
APP = "ap-test"

FAKE_MODAL = r"""#!/bin/bash
D=$FAKE_DIR
echo "$*" >> "$D/calls"
case "$1 $2" in
  "app logs")
    if [ "$3" = "--help" ]; then
      [ -n "${FAKE_NEW_CLI:-}" ] && echo "  -f, --follow  Stream logs until the app stops"
      echo "Usage: modal app logs [OPTIONS] [APP_IDENTIFIER]"
      exit 0
    fi
    cat "$D/stream.log"
    exit 0 ;;
  "app stop")
    if [ "$3" = "--help" ]; then
      [ -n "${FAKE_NEW_CLI:-}" ] && echo "  -y, --yes  Skip the confirmation prompt"
      echo "Usage: modal app stop [OPTIONS] [APP_IDENTIFIER]"
      exit 0
    fi
    [ -n "${FAKE_STOP_FAILS:-}" ] && exit 1
    touch "$D/stopped"
    exit 0 ;;
  "app list")
    n=$(( $(cat "$D/list-count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$D/list-count"
    if [ -e "$D/stopped" ] || [ "$n" -gt "${FAKE_LIVE_LISTS:-1000}" ]; then
      state=stopped; at='"2026-11-16 10:00:00+00:00"'
    else
      state=deployed; at=null
    fi
    printf '[{"App ID": "ap-other", "State": "deployed", "Stopped at": null},'
    printf ' {"App ID": "%s", "Description": "d844", "State": "%s",' "${FAKE_APP:-ap-test}" "$state"
    printf ' "Tasks": "1", "Created at": "2026-11-16 08:00:00+00:00", "Stopped at": %s}]\n' "$at"
    exit 0 ;;
esac
exit 64
"""


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(int(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _line(epoch: float, text: str, rss: float = 11.0) -> str:
    return f"[h2h {_stamp(epoch)}] {text}: peak_rss={rss:.2f} GiB"


def _watch(
    tmp_path: Path,
    stream: list[str],
    *,
    started_ago: int = 3 * 3600,
    kept: str | None = None,
    **env: str,
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    fake = tmp_path / "fake"
    run = tmp_path / "run"
    fake.mkdir()
    run.mkdir()
    if kept is not None:
        # An earlier watcher's stream in the same run directory.
        (run / "app-1.log").write_text(kept)
    modal = fake / "modal"
    modal.write_text(FAKE_MODAL)
    modal.chmod(modal.stat().st_mode | stat.S_IXUSR)
    (fake / "stream.log").write_text("".join(f"{line}\n" for line in stream))
    (run / "started-at").write_text(_stamp(time.time() - started_ago) + "\n")
    environment = {
        **os.environ,
        "FAKE_DIR": str(fake),
        "MODAL": str(modal),
        "PYTHON": sys.executable,
        "WATCH_INTERVAL": "0",
        "WATCH_STOP_WAIT": "0",
        **env,
    }
    result = subprocess.run(
        ["bash", str(WATCH), APP, str(run), str(PLAN)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result, fake, run


def _calls(fake: Path) -> list[str]:
    return (fake / "calls").read_text().splitlines()


def _stalled() -> list[str]:
    """A candidate worker's slice two hours ago and nothing since: a stall."""

    then = time.time() - 2 * 3600
    return [
        _line(then - 60, "after initializing candidate slice worker"),
        _line(then, "after scoring candidate chunk 1/5 slice 1/177"),
    ]


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the watcher is a bash script"
)


def test_a_no_go_stops_the_app_and_confirms_it(tmp_path) -> None:
    result, fake, run = _watch(tmp_path, _stalled())
    assert result.returncode == 3, result.stderr
    calls = _calls(fake)
    assert f"app stop {APP}" in calls
    assert (fake / "stopped").exists()
    verdicts = (run / "go-no-go.log").read_text()
    assert "verdict=NO-GO" in verdicts and f"stopped {APP} (try 1)" in verdicts
    assert not (run / "ALARM").exists()
    # The streamed log is kept.
    assert "chunk 1/5 slice 1/177" in (run / "app-1.log").read_text()


def test_newer_modal_flags_are_used_when_the_cli_offers_them(tmp_path) -> None:
    result, fake, _ = _watch(tmp_path, _stalled(), FAKE_NEW_CLI="1")
    assert result.returncode == 3, result.stderr
    calls = _calls(fake)
    assert f"app logs --follow {APP}" in calls
    assert f"app stop --yes {APP}" in calls


def test_a_projection_that_cannot_answer_alarms_then_stops_the_app(tmp_path) -> None:
    now = time.time()
    unreadable = [
        _line(now - 60, "after scoring candidate chunk 1/5 slice 1/177"),
        _line(now - 30, "after scoring candidate chunk 1/4 slice 2/177"),
    ]
    result, fake, run = _watch(tmp_path, unreadable, WATCH_MAX_FAILURES="2")
    assert result.returncode == 4, result.stderr
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 3
    assert "1 poll(s) in a row" in alarms[0] and "2 poll(s) in a row" in alarms[1]
    assert "could not be judged" in alarms[2]
    assert f"app stop {APP}" in _calls(fake)
    assert "disagree on the chunk shape" in (run / "projection.err").read_text()


def test_a_stream_that_ends_while_the_app_runs_is_restarted(tmp_path) -> None:
    # Live for the start check and one poll, then finished.
    result, fake, run = _watch(tmp_path, [], FAKE_LIVE_LISTS="2")
    assert result.returncode == 0, result.stderr
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 2
    assert not any(call.startswith("app stop ap-") for call in calls)
    verdicts = (run / "go-no-go.log").read_text()
    assert "restarting it" in verdicts and "has finished" in verdicts
    assert "verdict=WAIT" in verdicts
    assert not (run / "ALARM").exists()


def _live() -> list[str]:
    """An incumbent pool that opened ten minutes ago: too early to judge."""

    now = time.time()
    return [
        _line(now - 600, "before starting incumbent slice workers", 3.7),
        _line(now - 560, "after initializing incumbent slice worker", 1.9),
    ]


def test_each_stream_has_its_own_file_and_replays_count_once(tmp_path) -> None:
    # Each restarted stream replays the whole log, as a reconnect can.
    result, fake, run = _watch(tmp_path, _live(), FAKE_LIVE_LISTS="3")
    assert result.returncode == 0, result.stderr
    assert _calls(fake).count(f"app logs {APP}") == 3
    streams = sorted(run.glob("app-*.log"))
    assert [path.name for path in streams] == ["app-1.log", "app-2.log", "app-3.log"]
    assert all("initializing incumbent" in path.read_text() for path in streams)
    verdicts = (run / "go-no-go.log").read_text()
    assert "were counted once" in verdicts
    assert "verdict=WAIT" in verdicts and "verdict=NO-GO" not in verdicts


def test_a_restarted_watcher_keeps_the_streams_it_finds(tmp_path) -> None:
    kept = "".join(f"{line}\n" for line in _live())
    # Live only for the start check, so one stream and one poll.
    result, fake, run = _watch(tmp_path, _live(), kept=kept, FAKE_LIVE_LISTS="1")
    assert result.returncode == 0, result.stderr
    assert (run / "app-1.log").read_text() == kept
    assert "initializing incumbent" in (run / "app-2.log").read_text()
    assert _calls(fake).count(f"app logs {APP}") == 1


def test_an_unconfirmed_stop_is_retried_then_raised(tmp_path) -> None:
    result, fake, run = _watch(
        tmp_path, _stalled(), FAKE_STOP_FAILS="1", WATCH_STOP_TRIES="2"
    )
    assert result.returncode == 5, result.stderr
    assert _calls(fake).count(f"app stop {APP}") == 2
    assert "stop it by hand" in (run / "ALARM").read_text()


def test_an_app_that_is_not_running_is_refused_at_the_start(tmp_path) -> None:
    result, fake, _ = _watch(tmp_path, _stalled(), FAKE_APP="ap-someone-else")
    assert result.returncode == 2
    assert "is not a running app" in result.stderr
    assert not any(call.startswith("app logs ap-") for call in _calls(fake))


def test_bad_arguments_and_a_missing_start_time_are_refused(tmp_path) -> None:
    result = subprocess.run(
        ["bash", str(WATCH), APP], capture_output=True, text=True, check=False
    )
    assert result.returncode == 2 and "usage:" in result.stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        ["bash", str(WATCH), APP, str(empty), str(PLAN)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "started-at" in result.stderr
