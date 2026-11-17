#!/bin/bash
# Watch a running US head-to-head (d844) and stop it early when its own log
# says it will not finish inside its budget, has stalled, or cannot be read.
#
# usage: tools/us_head_to_head_watch.sh APP_ID RUN_DIR PLAN [CHUNK_WEIGHTS]
#
# RUN_DIR must hold `started-at` (UTC, written just before `modal run`).
# Writes RUN_DIR/app-N.log (the N-th log stream; a restarted stream may
# replay what earlier ones saw, so each is its own file and the projection
# counts a record once across them), RUN_DIR/go-no-go.log (one line per poll
# and per action) and RUN_DIR/ALARM (only when something needs a person).
#
# Exit:
#   0 the app finished on its own;
#   3 stopped on NO-GO;
#   4 stopped because the projection gave no usable verdict for
#     WATCH_MAX_FAILURES polls in a row (a WAIT more than WATCH_WAIT_GRACE
#     seconds past the budget's deadline counts as no verdict);
#   5 a stop was asked for but not confirmed;
#   2 bad arguments, or a refused start: the app is not running, the app
#     list could not be read in WATCH_START_TRIES tries, or the projection
#     cannot run. A refused start is also written to go-no-go.log and ALARM
#     whenever RUN_DIR exists.
#
# The cap and the runner's budget bound the spend; this only saves part of it.
# Env: WATCH_INTERVAL (seconds between polls, default 600), WATCH_STEP
# (seconds between checks that the log stream is alive, 10),
# WATCH_MAX_FAILURES (3), WATCH_WAIT_GRACE (1800), WATCH_STOP_TRIES (5),
# WATCH_STOP_WAIT (seconds after each stop request, 30), WATCH_START_TRIES
# (3), WATCH_START_WAIT (seconds between start checks, 10), MODAL (modal),
# PYTHON (python3; it runs the projection, so it must be the build's Python).
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

# 0 when the app is listed and not stopped; 1 when the list was read and the
# app is stopped or no longer listed (`modal app list` lists running and
# recently stopped apps); 2 when the list could not be read or is not the
# shape this expects. Only the two deliberate codes from the check count;
# any other exit, a crash included, reads as 2.
app_state() {
  local listing rc
  listing=$("$MODAL" app list --json 2>/dev/null) || return 2
  printf '%s' "$listing" | "$PYTHON" -c '
import json, sys
apps = json.load(sys.stdin)
if not isinstance(apps, list) or not apps or not all(
    isinstance(app, dict) and isinstance(app.get("App ID"), str) for app in apps
):
    sys.exit(2)
app = next((app for app in apps if app["App ID"] == sys.argv[1]), None)
if app is None:
    sys.exit(11)
if "State" not in app or "Stopped at" not in app:
    sys.exit(2)
state, at = app["State"], app["Stopped at"]
# The CLI gives null or a timestamp; any other value is a shape not known.
if not isinstance(state, str) or not (
    at is None or (isinstance(at, str) and at.strip())
):
    sys.exit(2)
stopped = at is not None or state == "stopped"
sys.exit(11 if stopped else 10)
' "$APP" 2>/dev/null
  rc=$?
  case $rc in
    10) return 0 ;;
    11) return 1 ;;
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
}

# A stream that ended while the app runs is restarted at once the first
# time in a poll interval, and at the poll after that, so a stream that keeps
# failing does not spin. Ends the watcher when the app has finished.
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
# NO-GO, WAIT-OVERDUE or none) and GAP (gap when the streams lost lines).
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
print(verdict, "gap" if report.get("log_gap") else "-")
' "$WAIT_GRACE" 2>/dev/null)
  case $parsed in
    *" "*) VERDICT=${parsed%% *}; GAP=${parsed#* } ;;
    *) VERDICT=none; GAP=- ;;
  esac
}

# ${ARRAY[@]+...} keeps an empty array from tripping set -u on bash 3.2
# (macOS's /bin/bash).
state=2
for try in $(seq 1 "$START_TRIES"); do
  app_state
  state=$?
  [ "$state" -ne 2 ] && break
  [ "$try" -lt "$START_TRIES" ] && sleep "$START_WAIT"
done
case $state in
  0) ;;
  1) refuse "$APP is not a running app in \`$MODAL app list\`" ;;
  *) refuse "\`$MODAL app list --json\` could not be read in $START_TRIES tries" ;;
esac
# The projection must answer before the run is left to it: an interpreter
# without the build's Python would fail every poll and stop a healthy run.
project /dev/null
if { [ "$RC" -ne 0 ] && [ "$RC" -ne 3 ]; } || [ "$VERDICT" = none ]; then
  refuse "the projection did not answer a trial run with $PYTHON (rc=$RC; see $DIR/projection.err)"
fi
note "watching $APP (plan $PLAN, started-at $STARTED, every ${INTERVAL}s)"
stream
failures=0
gap_alarmed=0
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
  project "$DIR"/app-*.log
  note "rc=$RC verdict=$VERDICT $(printf '%s' "$OUT" | tr -d '\n')"
  if [ "$GAP" = gap ] && [ "$gap_alarmed" -eq 0 ]; then
    gap_alarmed=1
    alarm "the log streams lost lines or began late (log_gap in go-no-go.log): from now on the estimate gives no GO, and after lost slice lines no estimate at all; the stall check and the runner's budget still bound the run"
  fi
  if [ "$RC" -eq 3 ] && [ "$VERDICT" = "NO-GO" ]; then
    if stop_app; then exit 3; fi
    exit 5
  fi
  if [ "$RC" -ne 0 ] || { [ "$VERDICT" != "GO" ] && [ "$VERDICT" != "WAIT" ]; }; then
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
  check_stream at-poll
done
