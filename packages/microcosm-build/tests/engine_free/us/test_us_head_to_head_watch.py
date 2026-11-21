"""The d844 watcher, driven against a fake ``modal`` command.

The fake records every call and keeps its state in files: whether the app
is stopped, how many times the app list was read, and the log it streams.
No test runs the real Modal CLI, and none stops a real app.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_support.microcosm_build import h2h_pool_sim as sim
from test_support.microcosm_build.us_modal_stage_plan_tool import ROOT

WATCH = ROOT / "tools" / "us_head_to_head_watch.sh"
PLAN = ROOT / "docs" / "us-modal-stage-route-a-h2h-plan.json"
WEIGHTS = ROOT / "docs" / "evidence" / "us-h2h-speed" / "chunk-weights.json"
APP = "ap-test"

FAKE_MODAL = r"""#!/bin/bash
D=$FAKE_DIR
echo "$*" >> "$D/calls"
help_tail() {
  # A long help, as typer prints: a reader that stops at the first match
  # leaves the writer with a broken pipe.
  if [ -n "${FAKE_LONG_HELP:-}" ]; then
    for i in $(seq 1 5000); do echo "  --option-$i  padding"; done
  fi
}
case "$1 $2" in
  "app logs")
    if [ "$3" = "--help" ]; then
      echo "Usage: modal app logs [OPTIONS] [APP_IDENTIFIER]"
      [ -n "${FAKE_NEW_CLI:-}" ] && echo "  -f, --follow  Stream logs until the app stops"
      help_tail
      exit 0
    fi
    # FAKE stream-N.log, when present, is what the N-th stream sees: a
    # restart that does not replay what the app printed before it.
    n=$(( $(cat "$D/logs-count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$D/logs-count"
    # Each stream's pid, so a test can see which streams still run.
    echo $$ > "$D/pid-$n"
    listed() { case ",$1," in *",$n,"*) return 0 ;; esac; return 1; }
    # FAKE_LOCK_RUN: once stream 1 starts, the run folder takes appends to
    # its files but no new file (review r9).
    if [ -n "${FAKE_LOCK_RUN:-}" ] && [ "$n" -eq 1 ]; then chmod a-w "$FAKE_RUN"; fi
    # FAKE_IGNORE_TERM lists streams (numbers, comma-separated) that ignore
    # TERM and live until the app is stopped, FAKE_STUBBORN_FOR seconds at
    # most (150): a client a plain kill does not end (review r9).
    stubborn=
    if listed "${FAKE_IGNORE_TERM:-}"; then stubborn=1; trap '' TERM; fi
    # A refused stream prints an error, no record, and ends at once: the
    # streams FAKE_REFUSE lists, every stream from FAKE_REFUSE_FROM on, or,
    # with FAKE_ONE_READER, a stream started while another still runs
    # (review r9). FAKE_ONE_READER=silent leaves that stream running, and
    # printing nothing, instead.
    refused=
    silent=
    if listed "${FAKE_REFUSE:-}"; then refused=1; fi
    if [ -n "${FAKE_REFUSE_FROM:-}" ] && [ "$n" -ge "$FAKE_REFUSE_FROM" ]; then
      refused=1
    fi
    if [ -n "${FAKE_ONE_READER:-}" ]; then
      for f in "$D"/pid-*; do
        [ "$f" = "$D/pid-$n" ] && continue
        if kill -0 "$(cat "$f")" 2>/dev/null; then
          if [ "$FAKE_ONE_READER" = silent ]; then silent=1; else refused=1; fi
        fi
      done
    fi
    if [ -n "$refused" ]; then
      echo "Error: the fake refuses log stream $n" >&2
      touch "$D/printed-$n"
      exit 1
    fi
    # Streams from FAKE_SILENT_FROM on run but print nothing.
    if [ -n "${FAKE_SILENT_FROM:-}" ] && [ "$n" -ge "$FAKE_SILENT_FROM" ]; then
      silent=1
    fi
    if [ -n "$silent" ]; then
      :
    elif [ -e "$D/stream-$n.log" ]; then
      cat "$D/stream-$n.log"
    else
      cat "$D/stream.log"
    fi
    touch "$D/printed-$n"
    # FAKE_ENDS_AT_ONCE lists streams that end once they have printed.
    if listed "${FAKE_ENDS_AT_ONCE:-}"; then exit 0; fi
    # FAKE stream-N.more, when present, is what the N-th stream prints later
    # in the poll interval: once the test's projection wrapper asks for it
    # (grow-N), which it acknowledges (grown-N). A stream that grows and
    # then hangs (review r7, should-fix 1).
    if [ -e "$D/stream-$n.more" ]; then
      for _ in $(seq 1 600); do [ -e "$D/grow-$n" ] && break; sleep 0.1; done
      cat "$D/stream-$n.more"
      touch "$D/grown-$n"
    fi
    if [ -n "$stubborn" ]; then
      for _ in $(seq 1 $(( ${FAKE_STUBBORN_FOR:-150} * 10 ))); do
        [ -e "$D/stopped" ] && exit 0
        sleep 0.1
      done
      exit 0
    fi
    # FAKE_END_AFTER_POLLS: the stream lives, printing nothing more, until
    # the watcher has logged that many polls or the app is stopped, then
    # ends (60 s at most, so nothing outlives a test).
    if [ -n "${FAKE_END_AFTER_POLLS:-}" ]; then
      for _ in $(seq 1 600); do
        polls=$(grep -c ' rc=' "$FAKE_RUN/go-no-go.log" 2>/dev/null)
        [ "${polls:-0}" -ge "$FAKE_END_AFTER_POLLS" ] && exit 0
        [ -e "$D/stopped" ] && exit 0
        sleep 0.1
      done
      exit 0
    fi
    # FAKE_HANG: the stream prints its log, then hangs without ending, as a
    # half-open connection can (review r6, SF-7).
    if [ -n "${FAKE_HANG:-}" ]; then exec sleep "$FAKE_HANG"; fi
    exit 0 ;;
  "app stop")
    if [ "$3" = "--help" ]; then
      echo "Usage: modal app stop [OPTIONS] [APP_IDENTIFIER]"
      [ -n "${FAKE_NEW_CLI:-}" ] && echo "  -y, --yes  Skip the confirmation prompt"
      help_tail
      exit 0
    fi
    # The streams still running when the stop came, one number a line.
    for f in "$D"/pid-*; do
      [ -e "$f" ] || continue
      kill -0 "$(cat "$f")" 2>/dev/null && echo "${f##*-}" >> "$D/running-at-stop"
    done
    [ -n "${FAKE_STOP_FAILS:-}" ] && exit 1
    touch "$D/stopped"
    exit 0 ;;
  "app list")
    n=$(( $(cat "$D/list-count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$D/list-count"
    # FAKE_LIST_SEQ names each read's answer in turn (the last repeats):
    # live, stopped, gone, fail, warning, dict, empty, nostate, or a "Stopped
    # at" that is neither null nor a timestamp: falsestop, emptystop,
    # wordstop ("garbage"), nullstop ("null") or zerostop ("0").
    mode=
    if [ -n "${FAKE_LIST_SEQ:-}" ]; then
      mode=$(printf '%s' "$FAKE_LIST_SEQ" | tr ',' '\n' | sed -n "${n}p")
      [ -n "$mode" ] || mode=${FAKE_LIST_SEQ##*,}
    fi
    if [ -z "$mode" ]; then
      if [ -e "$D/stopped" ] || [ "$n" -gt "${FAKE_LIVE_LISTS:-1000}" ]; then
        mode=stopped
      else
        mode=live
      fi
    fi
    case $mode in
      fail) echo "Error: could not connect" >&2; exit 1 ;;
      warning) echo "Warning: a newer modal client is available" ;;
      dict) echo '{"apps": []}'; exit 0 ;;
      empty) exit 0 ;;
      gone)
        echo '[{"App ID": "ap-other", "State": "deployed", "Stopped at": null}]'
        exit 0 ;;
      nostate)
        printf '[{"App ID": "%s", "Description": "d844"}]\n' "${FAKE_APP:-ap-test}"
        exit 0 ;;
      falsestop|emptystop|wordstop|nullstop|zerostop)
        case $mode in
          falsestop) at=false ;;
          emptystop) at='""' ;;
          wordstop) at='"garbage"' ;;
          nullstop) at='"null"' ;;
          zerostop) at='"0"' ;;
        esac
        printf '[{"App ID": "%s", "State": "ephemeral (detached)", "Stopped at": %s}]\n' \
          "${FAKE_APP:-ap-test}" "$at"
        exit 0 ;;
    esac
    if [ "$mode" = stopped ]; then
      state=stopped; at='"2026-11-16 10:00:00+00:00"'
    else
      state="ephemeral (detached)"; at=null
    fi
    printf '[{"App ID": "ap-other", "State": "deployed", "Stopped at": null},'
    printf ' {"App ID": "%s", "Description": "d844", "State": "%s",' "${FAKE_APP:-ap-test}" "$state"
    printf ' "Tasks": "1", "Created at": %s, "Stopped at": %s}]\n' "${FAKE_CREATED:-null}" "$at"
    exit 0 ;;
esac
exit 64
"""


RUNNER_BUDGET = "BUDGET: the tool may run 14505s (limited by max_wall_seconds)"


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(int(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _created(epoch: float) -> str:
    """``Created at`` as modal 1.3's ``app list --json`` prints it: local time
    with a colon offset (here UTC-5), as a JSON string."""

    when = datetime.fromtimestamp(int(epoch), timezone(timedelta(hours=-5)))
    return f'"{when.isoformat(sep=" ", timespec="seconds")}"'


def _line(epoch: float, text: str, rss: float = 11.0) -> str:
    return f"[h2h {_stamp(epoch)}] {text}: peak_rss={rss:.2f} GiB"


def _watch(
    tmp_path: Path,
    stream: list[str],
    *,
    started_ago: int = 3 * 3600,
    kept: dict[str, str] | None = None,
    weights: Path | None = None,
    budget_line: bool = True,
    streams: dict[int, list[str]] | None = None,
    more: dict[int, list[str]] | None = None,
    started: float | None = None,
    **env: str,
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    # The runner prints its BUDGET line before the tool starts, so a stream
    # begun before the tool shows it first; ``budget_line=False`` is a stream
    # begun late. ``more[N]`` is what stream N prints later, on request (see
    # FAKE_MODAL); ``started`` pins started-at, for a log made in advance.
    fake = tmp_path / "fake"
    run = tmp_path / "run"
    fake.mkdir()
    run.mkdir()
    for name, text in (kept or {}).items():
        # An earlier watcher's stream in the same run directory.
        (run / name).write_text(text)
    modal = fake / "modal"
    modal.write_text(FAKE_MODAL)
    modal.chmod(modal.stat().st_mode | stat.S_IXUSR)
    lines = [RUNNER_BUDGET, *stream] if budget_line else stream
    (fake / "stream.log").write_text("".join(f"{line}\n" for line in lines))
    for number, text in (streams or {}).items():
        (fake / f"stream-{number}.log").write_text("".join(f"{x}\n" for x in text))
    for number, text in (more or {}).items():
        (fake / f"stream-{number}.more").write_text("".join(f"{x}\n" for x in text))
    if started is None:
        started = time.time() - started_ago
    (run / "started-at").write_text(_stamp(started) + "\n")
    environment = {
        **os.environ,
        "FAKE_DIR": str(fake),
        "FAKE_RUN": str(run),
        "MODAL": str(modal),
        "PYTHON": sys.executable,
        # The app is created just after started-at is written.
        "FAKE_CREATED": _created(started + 5),
        "WATCH_INTERVAL": "0",
        "WATCH_STOP_WAIT": "0",
        "WATCH_START_WAIT": "0",
        # The fake's streams end at once, so each poll restarts one; the
        # settle wait has its own test.
        "WATCH_SETTLE": "0",
        # A new stream that prints nothing is taken as working after 2 s,
        # and a stream gets 1 s after TERM before KILL.
        "WATCH_READY": "2",
        "WATCH_KILL_GRACE": "1",
        # The watcher refuses to run without a pinned workspace.
        "MODAL_PROFILE": "test-profile",
        "MODAL_ENVIRONMENT": "test-env",
        **env,
    }
    argv = ["bash", str(WATCH), APP, str(run), str(PLAN)]
    if weights is not None:
        argv.append(str(weights))
    result = subprocess.run(
        argv,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result, fake, run


def _calls(fake: Path) -> list[str]:
    return (fake / "calls").read_text().splitlines()


def _pid(fake: Path, stream: int) -> int:
    return int((fake / f"pid-{stream}").read_text())


def _ended(pid: int, within: float = 3.0) -> bool:
    """Whether the process has ended (and been reaped) within ``within``
    seconds: a stream the watcher ended, or one it left behind."""

    deadline = time.monotonic() + within
    while True:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            # PermissionError: the pid now names another user's process.
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)


def _running_at_stop(fake: Path) -> list[int]:
    path = fake / "running-at-stop"
    return [int(x) for x in path.read_text().split()] if path.exists() else []


def _stalled() -> list[str]:
    """A candidate worker's slice two hours ago and nothing since: a stall."""

    then = time.time() - 2 * 3600
    return [
        _line(then - 70, "before starting candidate slice workers"),
        _line(then - 60, "after initializing candidate slice worker"),
        _line(then, "after scoring candidate chunk 1/5 slice 1/177"),
    ]


pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the watcher is a bash script"
)


def test_a_no_go_stops_the_app_and_confirms_it(tmp_path) -> None:
    result, fake, run = _overrun(tmp_path)
    assert result.returncode == 3, result.stderr
    calls = _calls(fake)
    assert f"app stop {APP}" in calls
    assert (fake / "stopped").exists()
    verdicts = (run / "go-no-go.log").read_text()
    assert "verdict=NO-GO" in verdicts and f"stopped {APP} (try 1)" in verdicts
    assert not (run / "ALARM").exists()
    # The streamed log is kept.
    assert "before starting incumbent slice workers" in (run / "app-1.log").read_text()


def test_newer_modal_flags_are_used_when_the_cli_offers_them(tmp_path) -> None:
    result, fake, _ = _overrun(tmp_path, FAKE_NEW_CLI="1")
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
    # Live for the start check and one poll, then finished. Launched ten
    # minutes ago, so an empty log is still inside the allowance.
    result, fake, run = _watch(tmp_path, [], started_ago=600, FAKE_LIVE_LISTS="2")
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
    result, fake, run = _watch(
        tmp_path, _live(), kept={"app-1.log": kept}, FAKE_LIVE_LISTS="1"
    )
    assert result.returncode == 0, result.stderr
    assert (run / "app-1.log").read_text() == kept
    assert "initializing incumbent" in (run / "app-2.log").read_text()
    assert _calls(fake).count(f"app logs {APP}") == 1


def test_a_gap_in_the_stream_numbers_overwrites_nothing(tmp_path) -> None:
    kept = "".join(f"{line}\n" for line in _live())
    # app-2.log without app-1.log: the new streams take 1 and then 3.
    result, _, run = _watch(
        tmp_path, _live(), kept={"app-2.log": kept}, FAKE_LIVE_LISTS="2"
    )
    assert result.returncode == 0, result.stderr
    assert (run / "app-2.log").read_text() == kept
    assert sorted(path.name for path in run.glob("app-*.log")) == [
        "app-1.log",
        "app-2.log",
        "app-3.log",
    ]


def test_an_unconfirmed_stop_is_retried_then_raised(tmp_path) -> None:
    # The list keeps showing the app running, as after a stop that failed.
    result, fake, run = _overrun(
        tmp_path, FAKE_STOP_FAILS="1", WATCH_STOP_TRIES="2", FAKE_LIVE_LISTS="1000"
    )
    assert result.returncode == 5, result.stderr
    assert _calls(fake).count(f"app stop {APP}") == 2
    assert "stop it by hand" in (run / "ALARM").read_text()


@pytest.mark.parametrize(
    ("created", "says"),
    [
        # Another session's stage, launched an hour before this attempt.
        ("before", "is not this attempt's app"),
        ("null", 'no "Created at"'),
        ('"garbage"', 'no "Created at"'),
        # No offset: it cannot be set against started-at.
        ("naive", 'no "Created at"'),
    ],
)
def test_an_app_not_created_by_this_attempt_is_refused_at_the_start(
    tmp_path, created, says
) -> None:
    # Review r6, SF-6: the watcher stopped whatever running app id it was
    # given; every stage is the app microcosm-us-stage, so an id copied from
    # `modal app list` can be another session's run.
    started = time.time() - 3 * 3600
    value = {
        "before": _created(started - 3600),
        "naive": f'"{datetime.fromtimestamp(int(started), UTC):%Y-%m-%d %H:%M:%S}"',
    }.get(created, created)
    result, fake, run = _watch(tmp_path, _stalled(), FAKE_CREATED=value)
    assert result.returncode == 2
    assert says in result.stderr
    assert "did not start" in (run / "ALARM").read_text()
    calls = _calls(fake)
    assert not any(call.startswith(("app logs ap-", "app stop ap-")) for call in calls)


def test_an_app_created_just_before_started_at_is_watched(tmp_path) -> None:
    # Clocks differ by a few seconds; up to WATCH_CREATED_SLACK is allowed.
    started = time.time() - 3 * 3600
    result, fake, run = _watch(
        tmp_path,
        _live(),
        started=started,
        FAKE_CREATED=_created(started - 30),
        FAKE_LIVE_LISTS="2",
    )
    assert result.returncode == 0, result.stderr
    assert f"watching {APP}" in (run / "go-no-go.log").read_text()
    assert "verdict=WAIT" in (run / "go-no-go.log").read_text()
    assert not (run / "ALARM").exists()


def test_an_app_that_is_not_running_is_refused_at_the_start(tmp_path) -> None:
    result, fake, _ = _watch(tmp_path, _stalled(), FAKE_APP="ap-someone-else")
    assert result.returncode == 2
    assert f"does not list {APP}" in result.stderr
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


@pytest.mark.parametrize(
    "answer",
    [
        "warning",
        "dict",
        "empty",
        "nostate",
        "fail",
        "falsestop",
        "emptystop",
        "wordstop",
        "nullstop",
        "zerostop",
        "gone",
    ],
)
def test_an_unreadable_list_after_a_stop_is_never_taken_as_stopped(
    tmp_path, answer
) -> None:
    # The list reads at the start, then never again: the stop is retried and
    # raised, not logged as confirmed. "gone" is a valid list without the app,
    # as from another workspace (review r4, WATCHER-1).
    result, fake, run = _overrun(
        tmp_path, FAKE_LIST_SEQ=f"live,{answer}", WATCH_STOP_TRIES="2"
    )
    assert result.returncode == 5, result.stderr
    assert _calls(fake).count(f"app stop {APP}") == 2
    verdicts = (run / "go-no-go.log").read_text()
    assert f"stopped {APP} (try" not in verdicts
    assert "stop it by hand" in (run / "ALARM").read_text()


def test_an_app_missing_from_the_list_when_the_stream_ends_is_not_finished(
    tmp_path,
) -> None:
    # Review r4, WATCHER-1: a list from another workspace once ended the
    # watcher with "has finished". Now it alarms and keeps the stream going
    # until the list shows the app stopped.
    result, _, run = _watch(tmp_path, _live(), FAKE_LIST_SEQ="live,gone,gone,stopped")
    assert result.returncode == 0, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert "could not be read" in (run / "ALARM").read_text()
    assert verdicts.count("has finished") == 1
    assert len(list(run.glob("app-*.log"))) == 3


@pytest.mark.parametrize("unset", ["MODAL_PROFILE", "MODAL_ENVIRONMENT"])
def test_a_start_without_a_pinned_modal_workspace_is_refused(tmp_path, unset) -> None:
    result, fake, run = _watch(tmp_path, _live(), **{unset: ""})
    assert result.returncode == 2
    assert "MODAL_PROFILE and MODAL_ENVIRONMENT" in result.stderr
    assert "did not start" in (run / "ALARM").read_text()
    assert not (fake / "calls").exists() or not any(
        call.startswith("app list") for call in _calls(fake)
    )


def test_a_run_folder_holding_an_earlier_attempt_is_refused(tmp_path) -> None:
    # Review r4, WATCHER-4: a relaunch into the same folder overwrites
    # started-at; the earlier attempt's worker starts held every verdict at
    # WAIT, so a run that cannot fit was never stopped.
    then = time.time() - 4 * 3600
    earlier = [
        RUNNER_BUDGET,
        _line(then, "before starting incumbent slice workers", 3.7),
        _line(then + 40, "after initializing incumbent slice worker", 1.9),
    ]
    kept = "".join(f"{line}\n" for line in earlier)
    result, fake, run = _watch(tmp_path, _live(), kept={"app-1.log": kept})
    assert result.returncode == 2
    assert "earlier attempt" in result.stderr
    assert "own run folder" in (run / "ALARM").read_text()
    assert not any(call.startswith("app logs ap-") for call in _calls(fake))
    assert (run / "app-1.log").read_text() == kept


def test_a_restarted_stream_that_does_not_replay_alarms_once(tmp_path) -> None:
    # A stream that ends and restarts without replaying the app's earlier
    # lines leaves a stretch no stream covers; a lost opening line can hide
    # there (review r4, ESTIMATE-3 = WATCHER-2).
    now = time.time()
    first = [
        RUNNER_BUDGET,
        _line(now - 1200, "before starting incumbent slice workers", 3.7),
        _line(now - 1160, "after initializing incumbent slice worker", 1.9),
    ]
    later = [_line(now - 300, "after initializing incumbent slice worker", 1.9)]
    result, _, run = _watch(
        tmp_path,
        later,
        budget_line=False,
        streams={1: first, 2: later},
        FAKE_LIVE_LISTS="4",
    )
    assert result.returncode == 0, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert "stream_holes" in verdicts and "verdict=NO-GO" not in verdicts
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "did not replay" in alarms[0]


def test_a_lost_line_after_a_hole_still_raises_the_gap_alarm(tmp_path) -> None:
    # The hole alarm comes first; a later lost opening line must still raise
    # its own alarm, not be taken as already reported.
    now = time.time()
    first = [
        RUNNER_BUDGET,
        _line(now - 1200, "before starting incumbent slice workers", 3.7),
        _line(now - 1160, "after initializing incumbent slice worker", 1.9),
    ]
    second = [_line(now - 300, "after initializing incumbent slice worker", 1.9)]
    third = [
        *second,
        # A candidate worker with no opening line: a lost line, so log_gap.
        _line(now - 290, "after initializing candidate slice worker", 1.9),
    ]
    result, _, run = _watch(
        tmp_path,
        third,
        budget_line=False,
        streams={1: first, 2: second, 3: third},
        FAKE_LIVE_LISTS="3",
    )
    assert result.returncode == 0, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert '"log_gap"' in verdicts and "stream_holes" in verdicts
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 2, alarms
    assert "did not replay" in alarms[0] and "lost lines" in alarms[1]


def test_an_unreadable_list_when_the_stream_ends_keeps_watching(tmp_path) -> None:
    # The stream ends after each poll; the list fails twice, then shows the
    # app stopped. The watcher alarms once, restarts the stream, and ends
    # only on a list it could read.
    result, fake, run = _watch(
        tmp_path, _live(), FAKE_LIST_SEQ="live,warning,fail,stopped"
    )
    assert result.returncode == 0, result.stderr
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 3
    assert not any(call.startswith("app stop ap-") for call in calls)
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "could not be read" in alarms[0]
    verdicts = (run / "go-no-go.log").read_text()
    assert "still could not be read" in verdicts and "has finished" in verdicts


def test_a_list_that_fails_at_the_start_is_retried(tmp_path) -> None:
    result, fake, run = _watch(
        tmp_path, _live(), FAKE_LIST_SEQ="fail,warning,live,stopped"
    )
    assert result.returncode == 0, result.stderr
    assert f"watching {APP}" in (run / "go-no-go.log").read_text()
    assert not (run / "ALARM").exists()


def test_a_list_that_never_reads_refuses_the_start_in_the_run_folder(
    tmp_path,
) -> None:
    result, fake, run = _watch(
        tmp_path, _live(), FAKE_LIST_SEQ="fail", WATCH_START_TRIES="2"
    )
    assert result.returncode == 2
    assert "could not be read" in result.stderr and "in 2 tries" in result.stderr
    assert (fake / "list-count").read_text().strip() == "2"
    assert "did not start" in (run / "ALARM").read_text()
    assert "did not start" in (run / "go-no-go.log").read_text()
    assert not any(call.startswith("app logs ap-") for call in _calls(fake))


def test_a_python_that_cannot_run_the_projection_is_refused_at_the_start(
    tmp_path,
) -> None:
    # An interpreter that reads the app list but cannot import the projection
    # (as macOS's /usr/bin/python3 3.9 cannot: no datetime.UTC).
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\n"
        'case "$1" in *us_head_to_head_projection.py)\n'
        '  echo "ImportError: cannot import name UTC" >&2; exit 1 ;;\n'
        "esac\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    result, fake, run = _watch(tmp_path, _live(), PYTHON=str(python))
    assert result.returncode == 2
    assert "did not answer a trial run" in result.stderr
    assert "did not start" in (run / "ALARM").read_text()
    assert "cannot import name UTC" in (run / "projection.err").read_text()
    calls = _calls(fake)
    assert not any(call.startswith(("app logs ap-", "app stop ap-")) for call in calls)


def test_one_failed_poll_between_good_ones_does_not_stop_the_app(tmp_path) -> None:
    # Projection calls 2 and 4 fail (call 1 is the start's trial run). With
    # WATCH_MAX_FAILURES=2 two failures that are not in a row stop nothing.
    count = tmp_path / "projections"
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\n"
        'case "$1" in *us_head_to_head_projection.py)\n'
        f'  n=$(( $(cat "{count}" 2>/dev/null || echo 0) + 1 )); echo $n > "{count}"\n'
        '  if [ "$n" -eq 2 ] || [ "$n" -eq 4 ]; then exit 1; fi ;;\n'
        "esac\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    result, fake, run = _watch(
        tmp_path,
        _live(),
        PYTHON=str(python),
        WATCH_MAX_FAILURES="2",
        FAKE_LIVE_LISTS="4",
    )
    assert result.returncode == 0, result.stderr
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 2
    assert all("1 poll(s) in a row" in alarm for alarm in alarms)
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_wait_far_past_the_deadline_counts_as_no_verdict(tmp_path) -> None:
    # Seventeen worker starts for sixteen workers keep the estimate at WAIT;
    # the lines are fresh, so no stall. The launch was 15,000 s ago, past the
    # plan's 14,505 s budget.
    now = time.time()
    log = [_line(now - 200, "before starting incumbent slice workers", 3.7)] + [
        _line(now - 100 + i, "after initializing incumbent slice worker", 1.9)
        for i in range(17)
    ]
    result, fake, run = _watch(
        tmp_path,
        log,
        started_ago=15_000,
        WATCH_WAIT_GRACE="0",
        WATCH_MAX_FAILURES="2",
    )
    assert result.returncode == 4, result.stderr
    alarms = (run / "ALARM").read_text()
    assert "still WAIT more than 0s past the budget's deadline, 1 poll(s)" in alarms
    assert f"app stop {APP}" in _calls(fake)


def test_a_wait_inside_the_grace_keeps_watching(tmp_path) -> None:
    now = time.time()
    log = [_line(now - 200, "before starting incumbent slice workers", 3.7)] + [
        _line(now - 100 + i, "after initializing incumbent slice worker", 1.9)
        for i in range(17)
    ]
    result, fake, run = _watch(tmp_path, log, started_ago=15_000, FAKE_LIVE_LISTS="3")
    assert result.returncode == 0, result.stderr
    assert not (run / "ALARM").exists()
    assert "verdict=WAIT" in (run / "go-no-go.log").read_text()


@pytest.mark.parametrize("stream", [[], ["the container is still building"]])
def test_a_hang_before_any_scorer_record_alarms_and_stops_nothing(
    tmp_path, stream
) -> None:
    # Launched two hours ago and no scorer record since: past the one-hour
    # allowance from the launch, so NO-GO. No restarted stream can show a
    # record printed after its restart, and streams that deliver nothing
    # read the same, so the watcher cannot show the log is current: it
    # alarms and leaves the run to the runner's budget and the class
    # timeout (review r9). It used to stop the app.
    result, fake, run = _watch(
        tmp_path, stream, started_ago=2 * 3600, FAKE_LIVE_LISTS="3"
    )
    assert result.returncode == 0, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert "no scorer record at all since the launch" in verdicts
    assert "NO-GO not confirmed" in (run / "ALARM").read_text()
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_log_gap_alarms_once_and_waits(tmp_path) -> None:
    # The parent took result 2, but slice 2's line is missing from the stream.
    t0 = time.time() - 600
    log = [
        _line(t0, "before starting candidate slice workers", 4.0),
        _line(t0 + 5, "after initializing candidate slice worker", 4.0),
        _line(t0 + 15, "after scoring candidate chunk 1/1 slice 1/40", 4.0),
        _line(t0 + 15, "after receiving candidate slice result", 4.0),
        _line(t0 + 25, "after receiving candidate slice result", 4.0),
        _line(t0 + 35, "after scoring candidate chunk 1/1 slice 3/40", 4.0),
        _line(t0 + 35, "after receiving candidate slice result", 4.0),
    ]
    # Each poll restarts the fake's stream (it ends at once) before it
    # projects, unless the stream is still printing; the list shows the app
    # finished after four reads, so the polls number three or four.
    result, fake, run = _watch(tmp_path, log, started_ago=900, FAKE_LIVE_LISTS="4")
    assert result.returncode == 0, result.stderr
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "lost lines" in alarms[0]
    verdicts = (run / "go-no-go.log").read_text()
    assert verdicts.count("verdict=WAIT") >= 3 and '"log_gap"' in verdicts
    assert verdicts.count("verdict=") == verdicts.count("verdict=WAIT")
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_stream_begun_after_the_runners_budget_line_alarms_once(tmp_path) -> None:
    # Scorer lines but no BUDGET line: the stream began late and may have
    # missed a shorter budget, so the projection never says GO and the
    # watcher raises the gap alarm once, without stopping the app.
    result, fake, run = _watch(
        tmp_path, _live(), budget_line=False, FAKE_LIVE_LISTS="3"
    )
    assert result.returncode == 0, result.stderr
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "began late" in alarms[0]
    verdicts = (run / "go-no-go.log").read_text()
    assert '"runner": "no BUDGET line"' in verdicts
    assert "verdict=GO" not in verdicts
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_second_stream_end_in_one_interval_is_restarted_before_the_poll(
    tmp_path,
) -> None:
    # Polls every 2 s, stream checks every 1 s. The fake's stream ends at
    # once, so the first restart comes at the first check; the second end in
    # the same interval waits for the poll, which restarts it before it
    # projects (review r6, SF-1: it used to project the dead stream first,
    # reading its stale file as a quiet scorer).
    result, fake, run = _watch(
        tmp_path,
        _live(),
        started_ago=900,
        WATCH_INTERVAL="2",
        WATCH_STEP="1",
        FAKE_LIVE_LISTS="3",
    )
    assert result.returncode == 0, result.stderr
    lines = (run / "go-no-go.log").read_text().splitlines()
    first_poll = next(i for i, line in enumerate(lines) if " rc=" in line)
    assert sum("restarting it" in line for line in lines[:first_poll]) == 2
    assert _calls(fake).count(f"app logs {APP}") == 3


def test_a_no_go_from_a_stream_that_just_started_waits_for_the_next_poll(
    tmp_path,
) -> None:
    # Every poll restarts the fake's stream (it ends at once), so the stream
    # the projection reads is always seconds old and may still be replaying:
    # its NO-GO is never acted on (review r6, SF-1's limit).
    result, fake, run = _watch(
        tmp_path, _stalled(), WATCH_SETTLE="3600", FAKE_LIVE_LISTS="4"
    )
    assert result.returncode == 0, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert verdicts.count("verdict=NO-GO") >= 3
    assert verdicts.count("may still be replaying") == verdicts.count("verdict=NO-GO")
    assert not (run / "ALARM").exists()
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_hung_stream_is_restarted_and_a_stall_it_still_shows_stops_nothing(
    tmp_path,
) -> None:
    # The stream prints its log and hangs, alive: `kill -0` sees nothing
    # wrong. The first NO-GO restarts it instead of stopping the app (review
    # r6, SF-7). The restarted stream shows the same stall and nothing
    # printed after the restart, so its NO-GO is not confirmed: a scorer
    # that went quiet and a stream stuck on an old stretch read the same
    # (review r9). The watcher alarms once, restarts once more
    # (WATCH_RECONNECTS=1), then leaves the run to the runner's budget. Each
    # poll waits for the newest stream to print: read before then, an empty
    # new stream gives WAIT (newest_stream_empty).
    started = int(time.time()) - 3 * 3600
    now = 3 * 3600
    calls = [(now, ""), (now, "printed-1"), (now, "printed-2"), (now, "printed-3")]
    result, fake, run = _watch(
        tmp_path,
        _stalled(),
        started=started,
        PYTHON=str(_judged_at(tmp_path, started, calls)),
        WATCH_INTERVAL="1",
        WATCH_STEP="1",
        WATCH_RECONNECTS="1",
        FAKE_END_AFTER_POLLS=str(len(calls) - 1),
        FAKE_LIVE_LISTS="1",
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run) == [
        "NO-GO",
        "restart",
        "NO-GO",
        "unconfirmed, restart",
        "NO-GO",
        "unconfirmed",
    ]
    verdicts = (run / "go-no-go.log").read_text()
    assert verdicts.count("has not grown since the last poll") == 1
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "NO-GO not confirmed" in alarms[0]
    calls_made = _calls(fake)
    assert calls_made.count(f"app logs {APP}") == 3
    assert not any(call.startswith("app stop ap-") for call in calls_made)


def test_a_restart_that_shows_a_fresh_slice_keeps_the_run_going(tmp_path) -> None:
    # Stream 1 hangs on a stall; the restart (stream 2) delivers a fresh
    # slice line, so the estimate moves on to WAIT and the app runs on.
    # test_a_go_after_the_restart_rearms_it_for_the_next_no_go pins the
    # re-arm a GO gives; this WAIT, which does not only wait for an empty
    # newest stream, takes the same branch.
    then = time.time() - 2 * 3600
    stalled = _stalled()
    fresh = [
        *stalled,
        _line(time.time() - 5, "after scoring candidate chunk 1/5 slice 2/177"),
    ]
    result, fake, run = _watch(
        tmp_path,
        stalled,
        streams={1: stalled, 2: fresh},
        FAKE_HANG="6",
        FAKE_LIVE_LISTS="3",
    )
    assert then and result.returncode == 0, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert verdicts.count("has not grown since the last poll") == 1
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


#: The simulated d844 run at f = 1, T = 300 s ends, by the estimate's count
#: (its last slice plus the 600 s allowance), 14,240 s after started-at:
#: 265 s inside the plan's 14,505 s. Its log to +11,400 s gives GO there
#: (14,238.5 s). A stream that then grows by the next 32 records and hangs
#: at +11,580 s gives NO-GO read at +12,000 s (14,925 s), with no damage
#: flag, where the whole log gives GO (14,240 s): review r7, should-fix 1.
HUNG_AT = 11_580


def _d844(started: int) -> sim.Run:
    weights = json.loads(WEIGHTS.read_text())
    return sim.run(
        started_at=started,
        workers=16,
        incumbent_seconds=[round(x) for x in weights["incumbent_mean_slice_seconds"]],
        candidate_seconds=[round(x) for x in weights["candidate_mean_slice_seconds"]],
        incumbent_slices=29,
        candidate_slices=177,
        start_cost=300,
        first_slice_extra=5,
        load_seconds=64,
        candidate_load_seconds=11,
        scorecard_seconds=600,
    )


def _between(run: sim.Run, after: float, upto: float) -> list[str]:
    """The run's records from ``after`` (exclusive) to ``upto`` seconds in,
    in the order ``sim.Run.log`` gives them."""

    rows = sorted([*run.incumbent.lines, *run.candidate.lines], key=lambda row: row[0])
    start = run.started_at
    return [text for at, text in rows if start + after < at <= start + upto]


def _judged_at(tmp_path: Path, started: int, calls: list[tuple[int, str]]) -> Path:
    """A ``PYTHON`` whose n-th projection call is judged ``--now`` at the
    n-th time, in seconds after ``started`` (the last repeats), once the
    fake has each file named for that call (comma-separated): ``printed-N``
    (stream N has printed its log), or ``grow-N``, which it creates to make
    stream N print its later lines, then waits for ``grown-N``. Growth asked
    for by the poll's own projection lands after the last poll took the
    file's size and before this poll reads it, whatever the host's load.
    ``fail`` makes that call exit 1 with no report, as a projection that
    cannot run does. Call 1 is the watcher's trial run. Each wait gives up
    after 60 s. Each call that projects first writes, to ``running``, the
    streams still running then (``_running``)."""

    fake = tmp_path / "fake"

    def wait(name: str) -> str:
        return (
            f'for _ in $(seq 1 600); do [ -e "{fake}/{name}" ] && break; '
            "sleep 0.1; done"
        )

    arms = []
    for number, (seconds, before) in enumerate(calls, start=1):
        names = [name for name in before.split(",") if name]
        steps = []
        for name in names:
            if name.startswith("grow-"):
                grown = name.replace("grow", "grown")
                steps.append(f'touch "{fake}/{name}"; ' + wait(grown))
            elif name != "fail":
                steps.append(wait(name))
        ready = "; ".join(steps) or ":"
        label = "*" if number == len(calls) else str(number)
        if "fail" in names:
            arms.append(f"    {label}) {ready}; exit 1 ;;")
        else:
            arms.append(f'    {label}) {ready}; now="{_stamp(started + seconds)}" ;;')
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\n"
        'case "$1" in *us_head_to_head_projection.py)\n'
        f'  n=$(( $(cat "{fake}/projections" 2>/dev/null || echo 0) + 1 ))\n'
        f'  echo "$n" > "{fake}/projections"\n'
        "  case $n in\n" + "\n".join(arms) + "\n  esac\n"
        f'  for f in "{fake}"/pid-*; do\n'
        '    [ -e "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null &&\n'
        f'      echo "$n ${{f##*-}}" >> "{fake}/running"\n'
        "  done\n"
        f'  exec "{sys.executable}" "$@" --now "$now" ;;\n'
        "esac\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    return python


def _running(fake: Path) -> dict[int, set[int]]:
    """The streams running at each projection call (``_judged_at``), by
    call number."""

    out: dict[int, set[int]] = {}
    path = fake / "running"
    for line in path.read_text().splitlines() if path.exists() else []:
        call, stream = line.split()
        out.setdefault(int(call), set()).add(int(stream))
    return out


def _watch_d844(
    tmp_path: Path,
    streams: dict[int, int | tuple[int, int]],
    more: dict[int, tuple[int, int]],
    calls: list[tuple[int, str]],
    *,
    budget: str = RUNNER_BUDGET,
    **env: str,
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """The watcher on the d844 run: stream N replays the log, from the
    runner's ``budget`` line, to ``streams[N]`` seconds in, or, given
    ``(a, b)``, does not replay and shows only the records from ``a``
    (exclusive) to ``b`` seconds in; then it prints ``more[N]`` when asked.
    A stream not in ``streams`` shows only the budget line. Each lives
    until the watcher has logged ``len(calls) - 1`` polls. ``env`` adds to
    or overrides the watcher's and the fake's settings."""

    started = int(time.time()) - 14_000
    run = _d844(started)

    def shown(span: int | tuple[int, int]) -> list[str]:
        if isinstance(span, tuple):
            return _between(run, *span)
        return [budget, *_between(run, 0, span)]

    settings = {
        "PYTHON": str(_judged_at(tmp_path, started, calls)),
        "WATCH_INTERVAL": "1",
        "WATCH_STEP": "1",
        "FAKE_END_AFTER_POLLS": str(len(calls) - 1),
        "FAKE_LIVE_LISTS": "1",
        **env,
    }
    return _watch(
        tmp_path,
        [],
        started=started,
        weights=WEIGHTS,
        streams={n: shown(s) for n, s in streams.items()},
        more={n: _between(run, a, b) for n, (a, b) in more.items()},
        **settings,
    )


def _polls(run_dir: Path) -> list[str]:
    """Each poll's verdict, each restart before acting on a NO-GO, each NO-GO
    not confirmed (and whether it restarted), each failed overlap and
    reconnect, each replacement of an empty stream, and the stop, in the
    order go-no-go.log gives them. Alarms are left out."""

    out = []
    for line in (run_dir / "go-no-go.log").read_text().splitlines():
        if " ALARM " in line:
            continue
        if " rc=" in line:
            out.append(line.split("verdict=")[1].split()[0])
        elif "NO-GO not acted on" in line:
            out.append("restart, grown" if " grew since" in line else "restart")
        elif "NO-GO not confirmed" in line:
            restarted = "restarting the stream" in line
            out.append("unconfirmed, restart" if restarted else "unconfirmed")
        elif "ended without a scorer record while the" in line:
            out.append("refused")
        elif "reconnected the log stream on its own" in line:
            out.append("reconnected")
        elif "has delivered no scorer record at" in line:
            out.append("empty, replaced")
        elif f"stopped {APP} (try" in line:
            out.append("stopped")
    return out


def test_a_stream_that_hangs_after_growing_is_restarted_before_a_no_go(
    tmp_path,
) -> None:
    # Review r7, should-fix 1: stream 1 holds the log to +11,400 s and grows
    # to +11,580 s early in the next interval, so its file has grown since
    # the last poll; then it hangs with its process alive, and read at
    # +12,000 s it gives a stale NO-GO. The watcher used to stop the app on
    # it. Now the first NO-GO restarts the stream whatever its growth, the
    # next poll reads the restarted stream's replay, and its GO keeps the
    # run going.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 11_400, 2: 12_600},
        {1: (11_400, HUNG_AT)},
        [(60, ""), (11_400, "printed-1"), (12_000, "grow-1"), (12_600, "printed-2")],
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == ["GO", "NO-GO", "restart, grown", "GO"]
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 2
    assert not any(call.startswith("app stop ap-") for call in calls)
    assert not (run_dir / "ALARM").exists()


def test_a_restart_that_shows_only_the_stale_log_does_not_stop_the_app(
    tmp_path,
) -> None:
    # Review r8, should-fix 1: the restarted stream replays only the stale
    # prefix, to +11,580 s, and hangs, so the +12,600 s poll reads the frozen
    # log again and gives NO-GO (16,125 s), where the whole log gives GO. The
    # watcher used to stop the app there. No stream started at the restart
    # shows a record printed after the +12,000 s poll, so it alarms and
    # restarts again; stream 3 sticks on the same prefix, and with
    # WATCH_RECONNECTS=1 the watcher leaves the run to the runner's budget.
    # A scorer that really went quiet at +11,580 s reads the same way, so the
    # watcher no longer stops that either.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 11_400, 2: HUNG_AT, 3: HUNG_AT},
        {1: (11_400, HUNG_AT)},
        [
            (60, ""),
            (11_400, "printed-1"),
            (12_000, "grow-1"),
            (12_600, "printed-2"),
            (13_200, "printed-3"),
        ],
        WATCH_RECONNECTS="1",
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == [
        "GO",
        "NO-GO",
        "restart, grown",
        "NO-GO",
        "unconfirmed, restart",
        "NO-GO",
        "unconfirmed",
    ]
    alarms = (run_dir / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "NO-GO not confirmed" in alarms[0]
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 3
    assert not any(call.startswith("app stop ap-") for call in calls)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root creates files in a read-only folder",
)
def test_a_restart_that_cannot_create_its_file_does_not_stop_the_app(
    tmp_path,
) -> None:
    # Review r8, should-fix 1: the run folder takes appends to its files but
    # no new file once stream 1 starts. The restart at the +12,000 s NO-GO
    # starts nothing, so the +12,600 s poll reads the same frozen stream and
    # gives NO-GO again (16,125 s), where the whole log gives GO. The watcher
    # used to stop the app there. No stream has started since the restart,
    # so none shows a record printed after it, and with WATCH_RECONNECTS=0
    # the watcher notes that no restart is left and stops nothing.
    try:
        result, fake, run_dir = _watch_d844(
            tmp_path,
            {1: 11_400},
            {1: (11_400, HUNG_AT)},
            [(60, ""), (11_400, "printed-1"), (12_000, "grow-1"), (12_600, "")],
            FAKE_LOCK_RUN="1",
            WATCH_RECONNECTS="0",
        )
    finally:
        (tmp_path / "run").chmod(0o755)
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == [
        "GO",
        "NO-GO",
        "restart, grown",
        "NO-GO",
        "unconfirmed",
    ]
    verdicts = (run_dir / "go-no-go.log").read_text()
    assert "ALARM could not create" in verdicts
    assert "ALARM NO-GO not confirmed" in verdicts
    assert [path.name for path in run_dir.glob("app-*.log")] == ["app-1.log"]
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_go_after_the_restart_rearms_it_for_the_next_no_go(tmp_path) -> None:
    # One restart per run of NO-GOs, re-armed by a GO: stream 2 replays the
    # log to +12,600 s (GO), grows to +12,780 s at the fourth poll and hangs;
    # read at +13,200 s it gives NO-GO (14,925 s). The GO between the two
    # NO-GOs re-armed the restart, so the watcher restarts again instead of
    # stopping, and stream 3's replay to +13,200 s gives GO.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 11_400, 2: 12_600, 3: 13_200},
        {1: (11_400, HUNG_AT), 2: (12_600, 12_780)},
        [
            (60, ""),
            (11_400, "printed-1"),
            (12_000, "grow-1"),
            (12_600, "printed-2"),
            (13_200, "grow-2"),
            (13_200, "printed-3"),
        ],
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == [
        "GO",
        "NO-GO",
        "restart, grown",
        "GO",
        "NO-GO",
        "restart, grown",
        "GO",
    ]
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 3
    assert not any(call.startswith("app stop ap-") for call in calls)
    # Each restart's old stream runs beside the new one through the next
    # poll's projection (call 4, then call 6), and has ended by the poll
    # after: no reader piles up across runs of NO-GOs (review r9).
    running = _running(fake)
    assert {1, 2} <= running[4] and 1 not in running.get(5, set())
    assert {2, 3} <= running[6]


#: The same run against a runner's budget of 13,640 s overruns: by the
#: estimate's count it ends at 14,240 s, 600 s past the budget. Its whole log
#: gives NO-GO from the +1,200 s poll on (14,263.7 s), while the incumbent's
#: workers are still starting.
SHORT_BUDGET = "BUDGET: the tool may run 13640s (limited by max_wall_seconds)"


def _overrun(
    tmp_path: Path, **env: str
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """The watcher on that overrunning run: NO-GO at +1,200 s, which
    restarts the stream; stream 2 replays the log to +1,800 s, so the NO-GO
    there comes with records printed after the restart, and the watcher
    stops the app."""

    return _watch_d844(
        tmp_path,
        {1: 1_200, 2: 1_800},
        {},
        [(60, ""), (1_200, "printed-1"), (1_800, "printed-2")],
        budget=SHORT_BUDGET,
        **env,
    )


def test_a_restart_that_does_not_replay_leaves_no_hole_and_stops_a_poll_later(
    tmp_path,
) -> None:
    # Review r8: stream 2 does not replay, so it shows only what the app
    # printed after the restart at +1,200 s. Ended there, stream 1 would
    # leave a hole until stream 2's first record (+1,264 s), and the starts
    # hold would keep the NO-GO as WAIT until every worker had started. The
    # old stream runs on beside the new one until the next poll has
    # projected both, so the +1,800 s poll reads the whole log, gives NO-GO
    # again, and the watcher stops the app.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: (1_200, 1_800)},
        {1: (1_200, 1_800)},
        [(60, ""), (1_200, "printed-1"), (1_800, "printed-2,grow-1")],
        budget=SHORT_BUDGET,
    )
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == ["NO-GO", "restart", "NO-GO", "stopped"]
    verdicts = (run_dir / "go-no-go.log").read_text()
    assert "stream_holes" not in verdicts and "starts_unconfirmed" not in verdicts
    assert verdicts.count("ended the log stream the last restart replaced") == 1
    assert not (run_dir / "ALARM").exists()
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 2
    assert calls.count(f"app stop {APP}") == 1
    # Both streams ran when the +1,800 s poll projected them.
    assert {1, 2} <= _running(fake)[3]


def test_a_wait_on_an_empty_new_stream_does_not_rearm_the_restart(tmp_path) -> None:
    # Stream 2 delivers nothing before the +1,800 s poll, so the estimate and
    # the stall check wait for it (WAIT, newest_stream_empty). That WAIT used
    # to re-arm the restart, so the +2,400 s NO-GO restarted again (review
    # r8, should-fix 3). It says nothing new about the run, so it leaves the
    # restart used: stream 2 then carries the log to +2,400 s, past the
    # restart at +1,200 s, and its NO-GO stops the app.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: (0, 0)},
        {2: (0, 2_400)},
        [(60, ""), (1_200, "printed-1"), (1_800, "printed-2"), (2_400, "grow-2")],
        budget=SHORT_BUDGET,
    )
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == ["NO-GO", "restart", "WAIT", "NO-GO", "stopped"]
    assert "newest_stream_empty" in (run_dir / "go-no-go.log").read_text()
    assert _calls(fake).count(f"app logs {APP}") == 2


def test_a_silent_second_reader_is_replaced_alone_and_the_next_no_go_stops(
    tmp_path,
) -> None:
    # Review r8, should-fix 3: a service that serves one reader and leaves a
    # second connected but silent. The +1,200 s NO-GO restarts the stream;
    # stream 2 runs beside stream 1 and prints nothing, so the +1,800 s poll
    # gives WAIT (newest_stream_empty) and stream 1 ends. That WAIT used to
    # re-arm the restart, so every NO-GO started the same cycle again and the
    # run went to the runner's budget. Now the second empty poll replaces
    # stream 2 with stream 3, started alone, which carries the log to
    # +3,000 s, past both its own start and the +2,400 s poll, so its NO-GO
    # stops the app.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 3: 3_000},
        {},
        [
            (60, ""),
            (1_200, "printed-1"),
            (1_800, "printed-2"),
            (2_400, ""),
            (3_000, "printed-3"),
        ],
        budget=SHORT_BUDGET,
        FAKE_ONE_READER="silent",
    )
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == [
        "NO-GO",
        "restart",
        "WAIT",
        "WAIT",
        "empty, replaced",
        "NO-GO",
        "stopped",
    ]
    assert _calls(fake).count(f"app logs {APP}") == 3
    # Both readers ran when the +1,800 s poll projected them.
    assert {1, 2} <= _running(fake)[3]


#: The polls after a restart at +1,200 s whose new stream prints nothing:
#: two empty polls, the second of which replaces it, then a NO-GO read from
#: the replacement's stale replay, not confirmed, which restarts again, and
#: a poll that waits for that empty new stream.
UNCONFIRMED_AFTER_EMPTY = [
    "NO-GO",
    "restart",
    "WAIT",
    "WAIT",
    "empty, replaced",
    "NO-GO",
    "unconfirmed, restart",
    "WAIT",
]


def test_a_replacement_that_only_replays_does_not_stop_the_app(tmp_path) -> None:
    # Review r9 check, N1: stream 2, started at the +1,200 s restart, prints
    # nothing, so the +1,800 s and +2,400 s polls give WAIT
    # (newest_stream_empty), and the second replaces it with stream 3. Stream
    # 3 replays the log only to +2,100 s, between the restart's poll and its
    # own start, then hangs, so the +3,000 s poll reads a log 900 s stale and
    # gives NO-GO. Those records are stamped after the restart, and the
    # watcher used to stop the app on them; none is stamped after stream 3's
    # own start or the +2,400 s poll, so now it alarms and restarts instead.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: (0, 0), 3: 2_100, 4: (0, 0)},
        {},
        [
            (60, ""),
            (1_200, "printed-1"),
            (1_800, "printed-2"),
            (2_400, ""),
            (3_000, "printed-3"),
            (3_600, "printed-4"),
        ],
        budget=SHORT_BUDGET,
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == UNCONFIRMED_AFTER_EMPTY
    alarms = (run_dir / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "NO-GO not confirmed" in alarms[0]
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_stream_that_delivered_then_ended_cannot_vouch_for_a_later_poll(
    tmp_path,
) -> None:
    # Review r9 check, N1: stream 2, started at the +1,200 s restart,
    # delivers records to +1,500 s and ends. Its replacement, stream 3, runs
    # and prints nothing, so the +1,800 s and +2,400 s polls give WAIT
    # (newest_stream_empty), and the second replaces stream 3 with stream 4,
    # which replays to +2,100 s and hangs. The +3,000 s NO-GO reads a log
    # 900 s stale. Stream 2's records are stamped after the restart and after
    # its own start, and the watcher used to stop the app on them; none of
    # the streams shows a record printed after the +2,400 s poll, so now it
    # alarms and restarts instead.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: 1_500, 3: (0, 0), 4: 2_100, 5: (0, 0)},
        {},
        [
            (60, ""),
            (1_200, "printed-1"),
            (1_800, "printed-3"),
            (2_400, ""),
            (3_000, "printed-4"),
            (3_600, "printed-5"),
        ],
        budget=SHORT_BUDGET,
        FAKE_ENDS_AT_ONCE="2",
        FAKE_LIVE_LISTS="2",
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == UNCONFIRMED_AFTER_EMPTY
    verdicts = (run_dir / "go-no-go.log").read_text()
    assert verdicts.count("the log stream ended while") == 1
    alarms = (run_dir / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "NO-GO not confirmed" in alarms[0]
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_records_from_before_a_new_stream_could_connect_do_not_confirm(
    tmp_path,
) -> None:
    # Review r9 check, N1: a stream's records count only from WATCH_READY
    # seconds after its launch, since what it replays covers the log up to
    # its connection. Here a client may take 500 s to connect: stream 2,
    # started at the +1,200 s restart, shows records to +1,650 s and hangs,
    # all of which it may have replayed. The +1,800 s NO-GO, which those
    # records alone would have confirmed, is not acted on, and with
    # WATCH_RECONNECTS=0 the run is left to the runner's budget.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: 1_650},
        {},
        [(60, ""), (1_200, "printed-1"), (1_800, "printed-2")],
        budget=SHORT_BUDGET,
        WATCH_READY="500",
        WATCH_RECONNECTS="0",
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == ["NO-GO", "restart", "NO-GO", "unconfirmed"]
    assert "NO-GO not confirmed" in (run_dir / "ALARM").read_text()
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def _functions(*names: str) -> str:
    """The watcher's shell functions ``names``, cut from the script, so a
    test can run one on its own."""

    text = WATCH.read_text()
    cut = []
    for name in names:
        start = text.index(f"\n{name}() {{\n") + 1
        cut.append(text[start : text.index("\n}\n", start) + 3])
    return "".join(cut)


def test_ending_a_stream_signals_only_the_watchers_own_child(tmp_path) -> None:
    # Review r9 check, N4: a stream that ended was reaped by bash, so by the
    # time the watcher ends it its pid can name another process. The watcher
    # used to send that pid TERM, then KILL after WATCH_KILL_GRACE. Here the
    # pid names a live process that ignores TERM and is not the shell's
    # child: it gets no signal. The shell's own child still gets TERM.
    other = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(60)\n",
        ]
    )
    try:
        script = (
            "set -uo pipefail\nGRACE=1\n"
            'note() { printf "%s\\n" "$1"; }\n'
            + _functions("alive", "end_stream")
            + f"alive {other.pid} || echo other-not-ours\n"
            + f"end_stream {other.pid}\n"
            + "sleep 30 & child=$!\n"
            + 'alive "$child" && echo child-alive\n'
            + 'end_stream "$child"\n'
            + 'alive "$child" || echo child-ended\n'
        )
        result = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert other.poll() is None, "the watcher signalled a process it did not start"
        assert result.stdout.split() == ["other-not-ours", "child-alive", "child-ended"]
    finally:
        other.kill()
        other.wait()


def test_a_failed_poll_after_the_restart_does_not_rearm_it(tmp_path) -> None:
    # Only a GO or WAIT re-arms the restart: a poll with no verdict between
    # two NO-GOs leaves it used, so the second NO-GO, which stream 2 carries
    # past the restart, stops the app.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: 2_400},
        {},
        [(60, ""), (1_200, "printed-1"), (1_800, "fail"), (2_400, "printed-2")],
        budget=SHORT_BUDGET,
    )
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == ["NO-GO", "restart", "none", "NO-GO", "stopped"]
    alarms = (run_dir / "ALARM").read_text()
    assert "no verdict from the projection (rc=1), 1 poll(s) in a row" in alarms
    assert _calls(fake).count(f"app logs {APP}") == 2


def test_a_stream_that_ignores_term_is_killed_and_reaped_before_the_stop(
    tmp_path,
) -> None:
    # Review r8, should-fix 2: stream 1 ignores TERM. Once the +1,800 s poll
    # had projected it beside stream 2, the watcher used to wait for it
    # without bound and never reached the stop. Now it sends KILL after
    # WATCH_KILL_GRACE, reaps the stream, and stops the app: stream 1 has
    # ended by the time the stop goes out.
    result, fake, run_dir = _overrun(tmp_path, FAKE_IGNORE_TERM="1")
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == ["NO-GO", "restart", "NO-GO", "stopped"]
    assert 1 not in _running_at_stop(fake)
    assert _ended(_pid(fake, 1), within=0)
    assert "sent KILL" in (run_dir / "go-no-go.log").read_text()


def test_a_watcher_that_ends_during_the_overlap_ends_the_replaced_stream(
    tmp_path,
) -> None:
    # Review r8, should-fix 2: the app finishes while the old stream still
    # runs beside its replacement, so the watcher exits with it set. Stream 1
    # ignores TERM; the exit still ends it (KILL after WATCH_KILL_GRACE) and
    # reaps it, so no reader outlives the watcher.
    result, fake, run_dir = _watch(
        tmp_path,
        _stalled(),
        FAKE_IGNORE_TERM="1",
        FAKE_STUBBORN_FOR="20",
        FAKE_LIVE_LISTS="1",
    )
    assert result.returncode == 0, result.stderr
    verdicts = (run_dir / "go-no-go.log").read_text()
    assert "NO-GO not acted on" in verdicts and "has finished" in verdicts
    # KILL, not the stream's own end FAKE_STUBBORN_FOR seconds later: the
    # watcher used to wait for it without bound.
    assert "sent KILL" in verdicts
    assert _ended(_pid(fake, 1))


def test_a_refused_second_stream_falls_back_to_one_stream_and_stops_a_poll_later(
    tmp_path,
) -> None:
    # Review r8, should-fix 3: Modal serves one reader of an app and refuses
    # a second. At the +1,200 s NO-GO the restart's stream 2 is refused while
    # stream 1 runs, so the watcher ends stream 1 and reconnects. Stream 3
    # replays the log, and the +1,800 s NO-GO, with its records printed after
    # the restart, stops the app. The watcher used to keep stream 1, whose
    # refused replacements left the newest file empty, so that poll gave WAIT
    # (newest_stream_empty), which re-armed the restart.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 3: 1_800},
        {},
        [(60, ""), (1_200, "printed-1"), (1_800, "printed-3")],
        budget=SHORT_BUDGET,
        FAKE_ONE_READER="1",
        FAKE_LIVE_LISTS="3",
    )
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == [
        "NO-GO",
        "restart",
        "refused",
        "reconnected",
        "NO-GO",
        "stopped",
    ]
    assert "the fake refuses log stream 2" in (run_dir / "app-2.log").read_text()
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 3
    assert calls.count(f"app stop {APP}") == 1


def test_a_refused_restart_beside_the_replaced_stream_falls_back_to_one_stream(
    tmp_path,
) -> None:
    # Review r8, should-fix 3: the restart's stream 2 replays to +1,200 s and
    # ends; its restarts while stream 1 still runs (3, between polls, and 4)
    # are refused. The watcher ends stream 1 and reconnects one stream at a
    # time: stream 5 replays the log, and the +1,800 s NO-GO stops the app.
    # The watcher used to restart beside stream 1 until the poll, whose empty
    # newest file gave WAIT, which re-armed the restart.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200, 2: 1_200, 5: 1_800},
        {},
        [(60, ""), (1_200, "printed-1"), (1_800, "printed-5")],
        budget=SHORT_BUDGET,
        FAKE_ENDS_AT_ONCE="2",
        FAKE_REFUSE="3,4",
        FAKE_LIVE_LISTS="3",
    )
    assert result.returncode == 3, result.stderr
    assert _polls(run_dir) == [
        "NO-GO",
        "restart",
        "refused",
        "reconnected",
        "NO-GO",
        "stopped",
    ]
    calls = _calls(fake)
    assert calls.count(f"app logs {APP}") == 5
    assert calls.count(f"app stop {APP}") == 1


def test_streams_that_are_always_refused_raise_an_alarm_and_stop_nothing(
    tmp_path,
) -> None:
    # Review r8, should-fix 3: after stream 1, Modal refuses every stream.
    # The restart's stream and both reconnects are refused, so the watcher
    # alarms that it could not regain a working stream. The newest file
    # stays empty, so the next poll gives WAIT (newest_stream_empty), and
    # nothing is stopped: the runner's budget bounds the run. How many polls
    # come before the fake's list shows the app finished depends on whether
    # a poll's own stream check meets a refused stream before it has exited,
    # so only the polls after the first WAIT are left open.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200},
        {},
        [(60, ""), (1_200, "printed-1"), (1_800, "")],
        budget=SHORT_BUDGET,
        FAKE_REFUSE_FROM="2",
        FAKE_LIVE_LISTS="3",
    )
    assert result.returncode == 0, result.stderr
    polls = _polls(run_dir)
    assert polls[:4] == ["NO-GO", "restart", "refused", "WAIT"]
    assert set(polls[4:]) <= {"WAIT", "empty, replaced"}
    assert "newest_stream_empty" in (run_dir / "go-no-go.log").read_text()
    alarms = (run_dir / "ALARM").read_text()
    assert "could not regain a working log stream" in alarms
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_new_stream_that_runs_but_prints_nothing_is_replaced_then_alarmed(
    tmp_path,
) -> None:
    # Review r8, should-fix 3: every stream after the first runs but
    # delivers nothing. The restart's stream passes the readiness check (it
    # runs), and the +1,800 s poll gives WAIT (newest_stream_empty). At the
    # second empty poll in a row the watcher replaces the newest stream
    # (WATCH_RECONNECTS=1); at the third it alarms. Nothing is stopped. The
    # watcher used to WAIT in silence.
    result, fake, run_dir = _watch_d844(
        tmp_path,
        {1: 1_200},
        {},
        [
            (60, ""),
            (1_200, "printed-1"),
            (1_800, "printed-2"),
            (2_400, "printed-2"),
            (3_000, "printed-3"),
        ],
        budget=SHORT_BUDGET,
        FAKE_SILENT_FROM="2",
        WATCH_RECONNECTS="1",
    )
    assert result.returncode == 0, result.stderr
    assert _polls(run_dir) == [
        "NO-GO",
        "restart",
        "WAIT",
        "WAIT",
        "empty, replaced",
        "WAIT",
    ]
    alarms = (run_dir / "ALARM").read_text().splitlines()
    assert len(alarms) == 1
    assert "no log stream has delivered a scorer record at 3 polls" in alarms[0]
    assert not any(call.startswith("app stop ap-") for call in _calls(fake))


def test_a_python_that_cannot_read_records_is_refused_at_the_start(
    tmp_path,
) -> None:
    # Every stop rests on reading the streams' records; an interpreter that
    # runs the projection but cannot load its record parser would leave
    # every NO-GO unconfirmed, so the watcher refuses to start.
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\n"
        'case "$2" in *"import _records"*)\n'
        '  echo "ImportError: no _records" >&2; exit 1 ;;\n'
        "esac\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    result, fake, run = _watch(tmp_path, _live(), PYTHON=str(python))
    assert result.returncode == 2
    assert "could not read scorer records" in result.stderr
    assert "did not start" in (run / "ALARM").read_text()
    calls = _calls(fake)
    assert not any(call.startswith(("app logs ap-", "app stop ap-")) for call in calls)


def test_help_is_read_whole_so_a_long_help_keeps_its_flags(tmp_path) -> None:
    result, fake, _ = _overrun(tmp_path, FAKE_NEW_CLI="1", FAKE_LONG_HELP="1")
    assert result.returncode == 3, result.stderr
    calls = _calls(fake)
    assert f"app logs --follow {APP}" in calls
    assert f"app stop --yes {APP}" in calls


def test_chunk_weights_reach_every_projection(tmp_path) -> None:
    weights = ROOT / "docs" / "evidence" / "us-h2h-speed" / "chunk-weights.json"
    argv = tmp_path / "argv"
    python = tmp_path / "python3"
    python.write_text(
        "#!/bin/bash\n"
        f'case "$1" in *us_head_to_head_projection.py) echo "$*" >> "{argv}" ;; esac\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    python.chmod(0o755)
    result, _, _ = _watch(
        tmp_path, _live(), weights=weights, PYTHON=str(python), FAKE_LIVE_LISTS="2"
    )
    assert result.returncode == 0, result.stderr
    calls = argv.read_text().splitlines()
    assert len(calls) >= 2
    assert all(f"--chunk-weights {weights}" in call for call in calls)
