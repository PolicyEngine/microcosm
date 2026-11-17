#!/bin/bash
# Watch a running US head-to-head (d844) and stop it early when its own log
# says it will not finish inside its budget, has stalled, or cannot be read.
#
# usage: tools/us_head_to_head_watch.sh APP_ID RUN_DIR PLAN [CHUNK_WEIGHTS]
#
# RUN_DIR must hold `started-at` (UTC, written just before `modal run`), and
# must be this attempt's own folder: the watcher refuses a RUN_DIR whose
# streams hold scorer records from before started-at. MODAL_PROFILE and
# MODAL_ENVIRONMENT must be set to the workspace and environment the app runs
# in, so a profile or environment switched by another session mid-run cannot
# point the app list and the stop at another workspace.
# Writes RUN_DIR/app-N.log (the N-th log stream; a restarted stream may
# replay what earlier ones saw, so each is its own file and the projection
# counts a record once across them), RUN_DIR/go-no-go.log (one line per poll
# and per action) and RUN_DIR/ALARM (only when something needs a person).
#
# Exit:
#   0 the app is no longer running and the watcher did not stop it: it
#     finished, or the runner's budget, the class timeout, a crash or a stop
#     by hand ended it;
#   3 stopped on NO-GO;
#   4 stopped because the projection gave no usable verdict for
#     WATCH_MAX_FAILURES polls in a row (a WAIT more than WATCH_WAIT_GRACE
#     seconds past the budget's deadline counts as no verdict);
#   5 a stop was asked for but not confirmed;
#   2 bad arguments, or a refused start: MODAL_PROFILE or MODAL_ENVIRONMENT
#     is unset, RUN_DIR holds an earlier attempt's streams, the app is not
#     running, the app was created more than WATCH_CREATED_SLACK seconds
#     before started-at (another session's app, not this attempt's) or its
#     creation time cannot be read, the app list could not be read in
#     WATCH_START_TRIES tries, or the projection cannot run. A refused start is also written to
#     go-no-go.log and ALARM whenever RUN_DIR exists.
#
# The cap and the runner's budget bound the spend; this only saves part of it.
# The WAIT-OVERDUE escalation acts only after the class timeout has passed,
# so it cleans up a run the timeout should already have ended; it saves no
# spend.
# Env: WATCH_INTERVAL (seconds between polls, default 600), WATCH_STEP
# (seconds between checks that the log stream is alive, 10),
# WATCH_MAX_FAILURES (3), WATCH_WAIT_GRACE (1800), WATCH_STOP_TRIES (5),
# WATCH_STOP_WAIT (seconds after each stop request, 30), WATCH_START_TRIES
# (3), WATCH_START_WAIT (seconds between start checks, 10),
# WATCH_CREATED_SLACK (60), WATCH_SETTLE (seconds a log stream must have run
# before a NO-GO read from it is acted on, 120), MODAL (modal), PYTHON
# (python3; it runs the projection, so it must be the build's Python).
#
# Before acting on a NO-GO the watcher checks that the log it read is
# current. A stream started less than WATCH_SETTLE seconds earlier may still
# be replaying, so the files hold an old stretch of the log; and a stream
# whose file has not grown since the last poll may have hung while its
# process lives on, which `kill -0` cannot see (review r6). Either way it
# notes why and lets the next poll decide; for a file that has not grown it
# first restarts the stream, once until the file grows again.
set -uo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "usage: $0 APP_ID RUN_DIR PLAN [CHUNK_WEIGHTS]" >&2
  exit 2
fi
APP=$1
DIR=$2
PLAN=$3
WEIGHTS=${4:-}
INTERVAL=${WATCH_INTERVAL:-600}
STEP=${WATCH_STEP:-10}
MAX_FAILURES=${WATCH_MAX_FAILURES:-3}
WAIT_GRACE=${WATCH_WAIT_GRACE:-1800}
STOP_TRIES=${WATCH_STOP_TRIES:-5}
STOP_WAIT=${WATCH_STOP_WAIT:-30}
START_TRIES=${WATCH_START_TRIES:-3}
START_WAIT=${WATCH_START_WAIT:-10}
CREATED_SLACK=${WATCH_CREATED_SLACK:-60}
SETTLE=${WATCH_SETTLE:-120}
MODAL=${MODAL:-modal}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
VERDICTS=$DIR/go-no-go.log

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { printf '%s %s\n' "$(stamp)" "$1" >> "$VERDICTS"; }
alarm() {
  printf '%s ALARM %s\n' "$(stamp)" "$1" >> "$DIR/ALARM"
  note "ALARM $1"
}
refuse() {
  echo "$0: $1" >&2
  if [ -d "$DIR" ]; then alarm "the watcher did not start: $1"; fi
  exit 2
}

if ! STARTED=$(cat "$DIR/started-at" 2>/dev/null) || [ -z "$STARTED" ]; then
  refuse "$DIR/started-at is missing or empty"
fi
if [ -z "${MODAL_PROFILE:-}" ] || [ -z "${MODAL_ENVIRONMENT:-}" ]; then
  refuse "MODAL_PROFILE and MODAL_ENVIRONMENT must both be set to the app's workspace and environment"
fi

# Modal 1.4 and later need --follow to keep streaming, and 1.4.2 and later
# ask before stopping unless given --yes; 1.3 has neither flag. The help is
# read whole before matching: `grep -q` in a pipe under pipefail can fail the
# pipe when it exits before the producer finishes writing.
FOLLOW=()
HELP=$("$MODAL" app logs --help 2>&1)
case $HELP in *--follow*) FOLLOW=(--follow) ;; esac
YES=()
HELP=$("$MODAL" app stop --help 2>&1)
case $HELP in *--yes*) YES=(--yes) ;; esac

# 0 when the app is listed and not stopped; 1 when the list was read and
# lists the app as stopped; 2 when the list could not be read, is not the
# shape this expects, or does not list the app. With `start`, also 3 when the
# app was created more than WATCH_CREATED_SLACK seconds before started-at
# (every stage is the app microcosm-us-stage, so another session's running
# stage is listed beside this one and an id copied from the list can be
# theirs), and 4 when its "Created at" is missing or not a timestamp with an
# offset. `modal app list` lists
# running and recently stopped apps, so an app missing from it is more
# likely a list from another workspace than an app that ended unseen. Only
# the two deliberate codes from the check count; any other exit, a crash
# included, reads as 2.
app_state() {
  local listing rc
  listing=$("$MODAL" app list --json 2>/dev/null) || return 2
  printf '%s' "$listing" | "$PYTHON" -c '
import json, re, sys
from datetime import datetime
apps = json.load(sys.stdin)
if not isinstance(apps, list) or not apps or not all(
    isinstance(app, dict) and isinstance(app.get("App ID"), str) for app in apps
):
    sys.exit(2)
app = next((app for app in apps if app["App ID"] == sys.argv[1]), None)
if app is None:
    sys.exit(2)
if "State" not in app or "Stopped at" not in app:
    sys.exit(2)
state, at = app["State"], app["Stopped at"]
def stamp(value):
    # modal --json prints it with isoformat(sep=" ", timespec="seconds")
    # (modal/_utils/time_utils.py), as in 2026-11-16 05:00:00-05:00.
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?", value
    ):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
# The CLI gives null or a timestamp; any other value is a shape not known.
if not isinstance(state, str) or not (at is None or stamp(at)):
    sys.exit(2)
stopped = at is not None or state == "stopped"
if sys.argv[2] == "start" and not stopped:
    # Both need an offset: a naive time cannot be set against started-at.
    def moment(value):
        if not isinstance(value, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)", value
        ):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    created, started = moment(app.get("Created at")), moment(sys.argv[3])
    if created is None or started is None:
        sys.exit(13)
    if created < started - float(sys.argv[4]):
        sys.exit(12)
sys.exit(11 if stopped else 10)
' "$APP" "${1:-poll}" "$STARTED" "$CREATED_SLACK" 2>/dev/null
  rc=$?
  case $rc in
    10) return 0 ;;
    11) return 1 ;;
    12) return 3 ;;
    13) return 4 ;;
    *) return 2 ;;
  esac
}

stop_app() {
  local try
  for try in $(seq 1 "$STOP_TRIES"); do
    "$MODAL" app stop ${YES[@]+"${YES[@]}"} "$APP" >> "$VERDICTS" 2>&1
    sleep "$STOP_WAIT"
    app_state
    if [ $? -eq 1 ]; then
      note "stopped $APP (try $try)"
      return 0
    fi
    note "stop of $APP not confirmed (try $try)"
  done
  alarm "could not confirm that $APP stopped after $STOP_TRIES tries: stop it by hand"
  return 1
}

# Each stream claims the next unused number before it starts: the file is
# created here under noclobber, so a second watcher on the same RUN_DIR
# cannot overwrite a stream it did not write, and the next poll always finds
# the file even before the stream prints anything.
STREAMS=0
STREAM=
STREAM_SINCE=0
stream() {
  local next=$((STREAMS + 1))
  until ( set -C; : > "$DIR/app-$next.log" ) 2>/dev/null; do
    if [ ! -e "$DIR/app-$next.log" ]; then
      alarm "could not create $DIR/app-$next.log; the log is not being streamed"
      return 1
    fi
    next=$((next + 1))
  done
  STREAMS=$next
  "$MODAL" app logs ${FOLLOW[@]+"${FOLLOW[@]}"} "$APP" >> "$DIR/app-$next.log" 2>&1 &
  STREAM=$!
  STREAM_SINCE=$(date +%s)
}

size_of() { local n; n=$(wc -c < "$1" 2>/dev/null | tr -d ' '); printf '%s' "${n:-0}"; }

# A stream that ended while the app runs is restarted at once the first
# time in a poll interval; a second end in the same interval waits for the
# poll, up to WATCH_INTERVAL, so a stream that keeps failing does not spin.
# The poll restarts it before it projects, so a dead stream is never read
# as a quiet scorer (review r6): the new file starts empty, which holds the
# estimate until the stream delivers. Ends the watcher when the app has
# finished.
RESTARTED=0
UNREADABLE=0
check_stream() {
  kill -0 "$STREAM" 2>/dev/null && return 0
  if [ "$1" = between-polls ] && [ "$RESTARTED" -eq 1 ]; then return 0; fi
  app_state
  case $? in
    0)
      UNREADABLE=0
      note "the log stream ended while $APP runs; restarting it"
      ;;
    1)
      note "$APP has finished; the watcher ends"
      exit 0
      ;;
    *)
      if [ "$UNREADABLE" -eq 0 ]; then
        alarm "the log stream ended and the app list could not be read; restarting the stream"
      else
        note "the log stream ended and the app list still could not be read; restarting the stream"
      fi
      UNREADABLE=1
      ;;
  esac
  RESTARTED=1
  stream
}

# One projection over the given streams. Sets OUT, RC, VERDICT (GO, WAIT,
# NO-GO, WAIT-OVERDUE or none), GAP (gap when the streams lost lines or
# began late), HOLE (hole when no stream covers a stretch between the first
# record and the last) and EARLY (scorer records before started-at). GAP and
# HOLE alarm apart, so a hole raised first cannot hide a later lost line.
project() {
  local args
  args=("$HERE/us_head_to_head_projection.py" "$@" --plan "$PLAN" --started-at "$STARTED")
  if [ -n "$WEIGHTS" ]; then args+=(--chunk-weights "$WEIGHTS"); fi
  OUT=$("$PYTHON" "${args[@]}" 2>> "$DIR/projection.err")
  RC=$?
  local parsed
  parsed=$(printf '%s' "$OUT" | "$PYTHON" -c '
import calendar, json, sys, time
report = json.load(sys.stdin)
verdict = report.get("verdict") or "none"
def at(key):
    return calendar.timegm(time.strptime(report[key], "%Y-%m-%dT%H:%M:%SZ"))
if verdict == "WAIT" and at("now_utc") > at("deadline_utc") + float(sys.argv[1]):
    verdict = "WAIT-OVERDUE"
gap = "gap" if report.get("log_gap") else "-"
hole = "hole" if report.get("stream_holes") else "-"
print(verdict, gap, hole, int(report.get("records_before_start") or 0))
' "$WAIT_GRACE" 2>/dev/null)
  if [ "$(printf '%s' "$parsed" | wc -w | tr -d ' ')" = 4 ]; then
    read -r VERDICT GAP HOLE EARLY <<< "$parsed"
  else
    VERDICT=none; GAP=-; HOLE=-; EARLY=0
  fi
}

# ${ARRAY[@]+...} keeps an empty array from tripping set -u on bash 3.2
# (macOS's /bin/bash).
state=2
for try in $(seq 1 "$START_TRIES"); do
  app_state start
  state=$?
  [ "$state" -ne 2 ] && break
  [ "$try" -lt "$START_TRIES" ] && sleep "$START_WAIT"
done
case $state in
  0) ;;
  1) refuse "$APP is not a running app in \`$MODAL app list\`" ;;
  3) refuse "$APP was created more than ${CREATED_SLACK}s before started-at $STARTED, so it is not this attempt's app (every stage is microcosm-us-stage); take the id from this attempt's launch log" ;;
  4) refuse "\`$MODAL app list --json\` gives no \"Created at\" with an offset for $APP, so it cannot be tied to this attempt" ;;
  *) refuse "\`$MODAL app list --json\` could not be read, or does not list $APP, in $START_TRIES tries" ;;
esac
# The projection must answer before the run is left to it: an interpreter
# without the build's Python would fail every poll and stop a healthy run.
project /dev/null
if { [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; } || [ "$VERDICT" = none ]; then
  refuse "the projection did not answer a trial run with $PYTHON (rc=$RC; see $DIR/projection.err)"
fi
# Streams already in RUN_DIR are this attempt's only when a watcher restarts;
# records from before started-at mean a relaunch reused the folder, and the
# earlier attempt's worker starts would hold every verdict at WAIT.
EXISTING=()
for log in "$DIR"/app-*.log; do
  if [ -e "$log" ]; then EXISTING+=("$log"); fi
done
if [ ${#EXISTING[@]} -gt 0 ]; then
  project "${EXISTING[@]}"
  if [ "$VERDICT" = none ]; then
    refuse "the projection could not read the streams already in $DIR (rc=$RC; see $DIR/projection.err)"
  fi
  if [ "$EARLY" -gt 0 ]; then
    refuse "$DIR holds $EARLY scorer records from before started-at $STARTED: an earlier attempt's streams; give each attempt its own run folder"
  fi
fi
note "watching $APP (plan $PLAN, started-at $STARTED, every ${INTERVAL}s, MODAL_PROFILE=$MODAL_PROFILE, MODAL_ENVIRONMENT=$MODAL_ENVIRONMENT)"
stream
failures=0
gap_alarmed=0
hole_alarmed=0
LAST_FILE=
LAST_SIZE=0
STALE_RESTARTED=0
while true; do
  RESTARTED=0
  waited=0
  while [ "$waited" -lt "$INTERVAL" ]; do
    step=$STEP
    if [ $((INTERVAL - waited)) -lt "$step" ]; then step=$((INTERVAL - waited)); fi
    sleep "$step"
    waited=$((waited + step))
    check_stream between-polls
  done
  check_stream at-poll
  project "$DIR"/app-*.log
  NEWEST=$DIR/app-$STREAMS.log
  GREW=0
  if [ "$NEWEST" = "$LAST_FILE" ] && [ "$(size_of "$NEWEST")" -gt "$LAST_SIZE" ]; then
    GREW=1
    STALE_RESTARTED=0
  fi
  note "rc=$RC verdict=$VERDICT $(printf '%s' "$OUT" | tr -d '\n')"
  if [ "$GAP" = gap ] && [ "$gap_alarmed" -eq 0 ]; then
    gap_alarmed=1
    alarm "the log streams lost lines or began late (log_gap in go-no-go.log): after a late start the estimate gives no GO, after lost slice, opening or worker-start lines no estimate at all; with restart_overlaps in the report the gap may instead come from a restart that repeated a second's records, which are counted once; the stall check and the runner's budget still bound the run"
  fi
  if [ "$HOLE" = hole ] && [ "$hole_alarmed" -eq 0 ]; then
    hole_alarmed=1
    alarm "a restarted log stream did not replay what the app printed before it (stream_holes in go-no-go.log): the estimate waits while a slice or worker start that began before the hole ended has no line (hole_suspects, hole_start_suspects), takes the start cost without a start whose span holds the hole when that is less (starts_across_holes), and holds a NO-GO until the estimate is past the line by more than hole_margin_seconds; a lost line the estimate can prove raises the gap alarm on its own and keeps it waiting for good; the stall check and the runner's budget still bound the run"
  fi
  if [ "$RC" -eq 3 ] && [ "$VERDICT" = "NO-GO" ]; then
    AGE=$(( $(date +%s) - STREAM_SINCE ))
    if [ "$AGE" -lt "$SETTLE" ]; then
      note "NO-GO not acted on: the log stream started ${AGE}s ago and may still be replaying, so the files can hold an old stretch of the log; the next poll decides"
    elif [ "$GREW" -eq 0 ] && [ "$STALE_RESTARTED" -eq 0 ]; then
      note "NO-GO not acted on: $NEWEST has not grown since the last poll, so its stream may have hung; restarting it, and the next poll decides"
      kill "$STREAM" 2>/dev/null
      wait "$STREAM" 2>/dev/null
      STALE_RESTARTED=1
      stream
    else
      if stop_app; then exit 3; fi
      exit 5
    fi
    # A NO-GO left to the next poll is a verdict, not a failed poll.
    failures=0
  elif [ "$RC" -ne 0 ] || { [ "$VERDICT" != "GO" ] && [ "$VERDICT" != "WAIT" ]; }; then
    failures=$((failures + 1))
    if [ "$VERDICT" = "WAIT-OVERDUE" ]; then
      alarm "still WAIT more than ${WAIT_GRACE}s past the budget's deadline, $failures poll(s) in a row"
    else
      alarm "no verdict from the projection (rc=$RC), $failures poll(s) in a row"
    fi
    if [ "$failures" -ge "$MAX_FAILURES" ]; then
      alarm "stopping $APP: the log could not be judged for $failures polls"
      if stop_app; then exit 4; fi
      exit 5
    fi
  else
    failures=0
  fi
  LAST_FILE=$DIR/app-$STREAMS.log
  LAST_SIZE=$(size_of "$LAST_FILE")
done
