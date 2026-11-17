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
    if [ -e "$D/stream-$n.log" ]; then cat "$D/stream-$n.log"; else cat "$D/stream.log"; fi
    exit 0 ;;
  "app stop")
    if [ "$3" = "--help" ]; then
      echo "Usage: modal app stop [OPTIONS] [APP_IDENTIFIER]"
      [ -n "${FAKE_NEW_CLI:-}" ] && echo "  -y, --yes  Skip the confirmation prompt"
      help_tail
      exit 0
    fi
    [ -n "${FAKE_STOP_FAILS:-}" ] && exit 1
    touch "$D/stopped"
    exit 0 ;;
  "app list")
    n=$(( $(cat "$D/list-count" 2>/dev/null || echo 0) + 1 ))
    echo "$n" > "$D/list-count"
    # FAKE_LIST_SEQ names each read's answer in turn (the last repeats):
    # live, stopped, gone, fail, warning, dict, empty, nostate, falsestop or
    # emptystop (a "Stopped at" that is neither null nor a timestamp).
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
      falsestop|emptystop)
        if [ "$mode" = falsestop ]; then at=false; else at='""'; fi
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
    printf ' "Tasks": "1", "Created at": "2026-11-16 08:00:00+00:00", "Stopped at": %s}]\n' "$at"
    exit 0 ;;
esac
exit 64
"""


RUNNER_BUDGET = "BUDGET: the tool may run 14505s (limited by max_wall_seconds)"


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(int(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    **env: str,
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    # The runner prints its BUDGET line before the tool starts, so a stream
    # begun before the tool shows it first; ``budget_line=False`` is a stream
    # begun late.
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
    (run / "started-at").write_text(_stamp(time.time() - started_ago) + "\n")
    environment = {
        **os.environ,
        "FAKE_DIR": str(fake),
        "MODAL": str(modal),
        "PYTHON": sys.executable,
        "WATCH_INTERVAL": "0",
        "WATCH_STOP_WAIT": "0",
        "WATCH_START_WAIT": "0",
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
    result, fake, run = _watch(
        tmp_path, _stalled(), FAKE_STOP_FAILS="1", WATCH_STOP_TRIES="2"
    )
    assert result.returncode == 5, result.stderr
    assert _calls(fake).count(f"app stop {APP}") == 2
    assert "stop it by hand" in (run / "ALARM").read_text()


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
    ["warning", "dict", "empty", "nostate", "fail", "falsestop", "emptystop", "gone"],
)
def test_an_unreadable_list_after_a_stop_is_never_taken_as_stopped(
    tmp_path, answer
) -> None:
    # The list reads at the start, then never again: the stop is retried and
    # raised, not logged as confirmed. "gone" is a valid list without the app,
    # as from another workspace (review r4, WATCHER-1).
    result, fake, run = _watch(
        tmp_path, _stalled(), FAKE_LIST_SEQ=f"live,{answer}", WATCH_STOP_TRIES="2"
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
    alarms = (run / "ALARM").read_text()
    assert alarms.count("did not replay") == 1


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
def test_a_hang_before_any_scorer_record_is_stopped(tmp_path, stream) -> None:
    # Launched two hours ago and no scorer record since: past the one-hour
    # allowance from the launch.
    result, fake, run = _watch(tmp_path, stream, started_ago=2 * 3600)
    assert result.returncode == 3, result.stderr
    verdicts = (run / "go-no-go.log").read_text()
    assert "no scorer record at all since the launch" in verdicts
    assert f"app stop {APP}" in _calls(fake)


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
    result, fake, run = _watch(tmp_path, log, started_ago=900, FAKE_LIVE_LISTS="3")
    assert result.returncode == 0, result.stderr
    alarms = (run / "ALARM").read_text().splitlines()
    assert len(alarms) == 1 and "lost lines" in alarms[0]
    verdicts = (run / "go-no-go.log").read_text()
    assert verdicts.count("verdict=WAIT") == 3 and '"log_gap"' in verdicts
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


def test_a_dead_stream_is_restarted_before_the_next_poll(tmp_path) -> None:
    # Polls every 2 s, stream checks every 1 s. The fake's stream ends at
    # once, so the first restart comes before the first poll; a second death
    # in the same interval waits for the poll.
    result, fake, run = _watch(
        tmp_path,
        _live(),
        started_ago=900,
        WATCH_INTERVAL="2",
        WATCH_STEP="1",
        FAKE_LIVE_LISTS="2",
    )
    assert result.returncode == 0, result.stderr
    lines = (run / "go-no-go.log").read_text().splitlines()
    first_poll = next(i for i, line in enumerate(lines) if " rc=" in line)
    assert any("restarting it" in line for line in lines[:first_poll])
    assert _calls(fake).count(f"app logs {APP}") == 2


def test_help_is_read_whole_so_a_long_help_keeps_its_flags(tmp_path) -> None:
    result, fake, _ = _watch(tmp_path, _stalled(), FAKE_NEW_CLI="1", FAKE_LONG_HELP="1")
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
