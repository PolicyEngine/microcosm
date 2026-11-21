#!/bin/bash
# Watch a running US head-to-head (d844) and stop it early when its own log
# says it will not finish inside its budget or cannot be read; a stall, or a
# log it cannot show is current, raises an alarm instead.
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
# before a NO-GO read from it is acted on, 120), WATCH_READY (seconds a new
# stream gets to show it works, 30), WATCH_RECONNECTS (further restarts or
# reconnects before the watcher alarms and leaves the run to the runner's
# budget, 2), WATCH_KILL_GRACE (seconds a stream gets to end after TERM
# before KILL, 5), MODAL (modal), PYTHON (python3; it runs the projection,
# so it must be the build's Python).
#
# Before acting on a NO-GO the watcher checks that the log it read is
# current. A stream started less than WATCH_SETTLE seconds earlier may still
# be replaying, so the files hold an old stretch of the log: it notes that
# and lets the next poll decide. A stream can also hang while its process
# lives on, which `kill -0` cannot see (review r6), and it can do so after
# its file grew earlier in the same poll interval, so growth since the last
# poll does not show the stream is current (review r7). So the first NO-GO
# after a GO or WAIT, or after the watcher starts, is never acted on: the
# watcher starts a new stream and lets the next poll decide. The old stream
# runs beside the new one until the next poll has projected both, then
# ends. Ended at the restart, it would leave a hole whenever the new stream
# does not replay what the app printed before it, and the hole's holds would
# give WAITs that re-arm the restart, so each NO-GO would restart again and
# a run that overruns might never be stopped (review r8, simulated).
#
# A later NO-GO stops the app only when a stream started at or after the
# last restart holds a scorer record stamped after both the poll before
# (the restart's own poll, or a later one) and the stream's own start
# (counted from WATCH_READY seconds after its launch, on the projection's
# clock). What a stream replays covers the log only up to its connection,
# so a replay alone never passes, whichever restart, reconnect or
# replacement started the stream, and a stream that delivered after the
# restart but has since ended or hung cannot vouch for a later poll (review
# r9). A restart that could not create its file, or a new stream that shows
# only what the old one already had, gives no such record, and neither does
# a scorer that has gone quiet: the log cannot tell those apart. Without one
# the watcher alarms, once per run of NO-GOs, restarts again up to
# WATCH_RECONNECTS more times, and otherwise leaves the run to the runner's
# budget. The next GO or WAIT re-arms the restarts, except a WAIT that only
# waits for an empty newest stream (newest_stream_empty): a second reader
# that Modal leaves connected but silent would otherwise re-arm them at
# every NO-GO, so a run that overruns would never stop (review r9).
#
# A new stream that ends without a scorer record within WATCH_READY seconds
# while the old one runs (Modal may refuse a second reader of one app) makes
# the watcher end the old one and reconnect one stream at a time, up to
# WATCH_RECONNECTS tries. When the projection finds the newest stream empty
# (newest_stream_empty) at two polls in a row, the watcher replaces that
# stream, once a poll, up to WATCH_RECONNECTS times. Either way it alarms
# when no stream that delivers comes back. A serial reconnect gives up the
# overlap: if Modal refuses a second reader and does not replay, every
# restart becomes a reconnect that leaves a hole, the hole's holds give a
# WAIT that re-arms the restart, and each run of NO-GOs can end in another
# such WAIT, so only the runner's budget may end a run that overruns
# (inferred). Each stream it ends gets TERM, then KILL after
# WATCH_KILL_GRACE seconds, and is reaped, so a client that ignores TERM
# cannot hold the loop; it signals only its own children, so a pid reused
# after a stream ended gets nothing (review r9).
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
READY=${WATCH_READY:-30}
RECONNECTS=${WATCH_RECONNECTS:-2}
GRACE=${WATCH_KILL_GRACE:-5}
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
# The files of the streams started since the last restart before a NO-GO,
# and beside each the second from which its records show it delivers: its
# launch plus WATCH_READY, on the projection's clock. What a stream replays
# covers the log only up to its connection, which a client makes within
# seconds of its launch (inferred), so its replay alone never passes
# (review r9).
SINCE=()
SINCE_AT=()
# The projection's clock less this machine's, from the last projection that
# gave a time: 0 to a few seconds when the projection reads the wall clock,
# which it does unless given --now (as the tests' fake clock does).
CLOCK_OFFSET=0
PROJECTED_FROM=0
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
  SINCE+=("$DIR/app-$next.log")
  SINCE_AT+=("$((STREAM_SINCE + CLOCK_OFFSET + READY))")
}

size_of() { local n; n=$(wc -c < "$1" 2>/dev/null | tr -d ' '); printf '%s' "${n:-0}"; }

# Whether a process this watcher started still runs. A child that has
# exited but not yet been reaped (state Z) counts as ended, and so does a
# pid whose parent is not this watcher: the child ended, bash reaped it, and
# the pid may now name another process (review r9). When ps cannot say,
# `kill -0` decides, so a live client is never taken as ended.
alive() {
  kill -0 "$1" 2>/dev/null || return 1
  local parent state
  read -r parent state <<< "$(ps -o ppid= -o stat= -p "$1" 2>/dev/null)"
  [ -n "$parent" ] || return 0
  [ "$parent" = "$$" ] || return 1
  case $state in Z*) return 1 ;; esac
  return 0
}

# Ends a log stream this watcher started and reaps it: TERM, then KILL if it
# still runs WATCH_KILL_GRACE seconds later, so a client that ignores TERM
# cannot hold the loop (review r9). A pid that is no longer this watcher's
# running child gets no signal; `wait` returns at once for it.
end_stream() {
  local pid=$1 ticks=0
  [ -n "$pid" ] || return 0
  if alive "$pid"; then
    kill "$pid" 2>/dev/null
    while alive "$pid" && [ "$ticks" -lt $((GRACE * 10)) ]; do
      sleep 0.1
      ticks=$((ticks + 1))
    done
    if alive "$pid"; then
      kill -KILL "$pid" 2>/dev/null
      note "the log stream with pid $pid still ran ${GRACE}s after TERM; sent KILL"
    fi
  fi
  wait "$pid" 2>/dev/null
  return 0
}

# Whether a stream's file holds a line with a scorer record's stamp.
has_record() {
  grep -q -E '\[h2h [0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z\] ' "$1" 2>/dev/null
}

# Whether the newest stream works: it has delivered a scorer record, or it
# still runs WATCH_READY seconds after it started. One that ends with no
# record in that time does not (Modal may refuse a second reader of one app,
# review r9). A stream that only waits for the app to print passes.
ready() {
  local file=$DIR/app-$STREAMS.log ticks=0
  while true; do
    if has_record "$file"; then return 0; fi
    if ! alive "$STREAM"; then
      has_record "$file"
      return
    fi
    if [ "$ticks" -ge $((READY * 5)) ]; then return 0; fi
    sleep 0.2
    ticks=$((ticks + 1))
  done
}

# Ends the current stream and starts another, one at a time, until one works
# or WATCH_RECONNECTS tries have failed; then alarms.
reconnect() {
  local try=0
  while [ "$try" -lt "$RECONNECTS" ]; do
    try=$((try + 1))
    end_stream "$STREAM"
    stream || return 1
    if ready; then
      note "reconnected the log stream on its own (try $try)"
      return 0
    fi
  done
  alarm "could not regain a working log stream: $RECONNECTS reconnect(s), one stream at a time, each ended without a scorer record within ${READY}s; the estimate and the stall check wait for a stream that delivers, and the runner's budget bounds the run"
  return 1
}

# records_after SECOND FILE [SECOND FILE ...]: 0 when a file holds a scorer
# record stamped after the epoch second given before it, 1 when none does
# (or no file is given), 2 when they could not be read: only an answer the
# reader printed counts. It reads records with the projection's own parser.
records_after() {
  local answer
  if [ $# -eq 0 ]; then return 1; fi
  answer=$("$PYTHON" -c '
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from us_head_to_head_projection import _records
pairs = sys.argv[2:]
if len(pairs) % 2:
    sys.exit("records_after takes SECOND FILE pairs")
found = any(
    at > float(after)
    for after, path in zip(pairs[::2], pairs[1::2])
    for at in _records(Path(path).read_text(errors="replace").splitlines())[3]
)
print("found" if found else "none")
' "$HERE" "$@" 2>> "$DIR/projection.err")
  case $answer in
    found) return 0 ;;
    none) return 1 ;;
    *) return 2 ;;
  esac
}

# Starts a new stream beside the current one, which runs on until the next
# poll has projected both, and notes this poll's time: a later NO-GO is
# acted on only if a stream started from here on shows a scorer record
# printed after both the poll before that NO-GO and its own start. If the
# new stream ends with no record while the old one runs, the watcher ends
# the old one and reconnects one stream at a time.
restart_streams() {
  local old=$STREAM
  RESTARTS=$((RESTARTS + 1))
  EPOCH=$NOW_AT
  EPOCH_UTC=$NOW_UTC
  SINCE=()
  SINCE_AT=()
  stream || return 1
  if ready; then
    REPLACED=$old
    return 0
  fi
  if alive "$old"; then
    note "the new log stream ended without a scorer record while the old one ran (Modal may refuse a second reader of one app); ending the old one and reconnecting one stream at a time"
    end_stream "$old"
  else
    note "the new log stream ended without a scorer record within ${READY}s; reconnecting"
  fi
  reconnect
}

# A stream that ended while the app runs is restarted at once the first
# time in a poll interval; a second end in the same interval waits for the
# poll, up to WATCH_INTERVAL, so a stream that keeps failing does not spin.
# The poll restarts it before it projects, so a dead stream is never read
# as a quiet scorer (review r6): the new file starts empty, which holds the
# estimate until the stream delivers. A restart while a replaced stream
# still runs must work (ready), or the watcher ends the replaced one and
# reconnects one stream at a time. Ends the watcher when the app has
# finished.
RESTARTED=0
UNREADABLE=0
check_stream() {
  alive "$STREAM" && return 0
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
  stream || return 1
  if [ -n "$REPLACED" ] && ! ready; then
    note "the restarted log stream ended without a scorer record while the replaced one ran (Modal may refuse a second reader of one app); ending the replaced one and reconnecting one stream at a time"
    end_stream "$REPLACED"
    REPLACED=
    reconnect
  fi
  return 0
}

# One projection over the given streams. Sets OUT, RC, VERDICT (GO, WAIT,
# NO-GO, WAIT-OVERDUE or none), GAP (gap when the streams lost lines or
# began late), HOLE (hole when no stream covers a stretch between the first
# record and the last), EARLY (scorer records before started-at), NOW_AT and
# NOW_UTC (the time it judged at, as epoch seconds and as text) and EMPTY
# (empty when the newest stream has delivered nothing yet beside earlier
# ones that have: newest_stream_empty). GAP and HOLE alarm apart, so a hole
# raised first cannot hide a later lost line. PROJECTED_FROM is this
# machine's second just before the call; a report with a time also sets
# CLOCK_OFFSET against it, so the offset, and each stream start taken with
# it, errs late.
project() {
  local args before
  args=("$HERE/us_head_to_head_projection.py" "$@" --plan "$PLAN" --started-at "$STARTED")
  if [ -n "$WEIGHTS" ]; then args+=(--chunk-weights "$WEIGHTS"); fi
  before=$(date +%s)
  PROJECTED_FROM=$before
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
empty = "empty" if report.get("newest_stream_empty") else "-"
early = int(report.get("records_before_start") or 0)
print(verdict, gap, hole, early, at("now_utc"), report["now_utc"], empty)
' "$WAIT_GRACE" 2>/dev/null)
  if [ "$(printf '%s' "$parsed" | wc -w | tr -d ' ')" = 7 ]; then
    read -r VERDICT GAP HOLE EARLY NOW_AT NOW_UTC EMPTY <<< "$parsed"
    CLOCK_OFFSET=$((NOW_AT - before))
  else
    VERDICT=none; GAP=-; HOLE=-; EARLY=0; NOW_AT=0; NOW_UTC=-; EMPTY=-
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
# The check behind every stop must run too: one that cannot read records
# would leave every NO-GO unconfirmed.
records_after 0 /dev/null
if [ $? -ne 1 ]; then
  refuse "the watcher could not read scorer records with $PYTHON (see $DIR/projection.err)"
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
# The restarts this run of NO-GOs has had; a GO or WAIT clears it, except
# a WAIT on an empty newest stream.
RESTARTS=0
# The last restart's poll time (epoch seconds and text).
EPOCH=0
EPOCH_UTC=-
# The time the last poll that gave a time judged at (epoch seconds and
# text). A NO-GO is acted on only with a record printed after it, so a
# stream that delivered after the restart and has since ended or hung
# cannot vouch for a later poll (review r9).
LAST_POLL_AT=0
LAST_POLL_UTC=-
unconfirmed_alarmed=0
# Polls in a row that found the newest stream empty, the reconnects made
# for them, and whether their alarm went out.
EMPTY_POLLS=0
EMPTY_RECONNECTS=0
empty_alarmed=0
# The stream a restart replaced, until the next poll has projected it.
REPLACED=
# If the watcher ends first (the app finished), the replaced stream ends too.
trap 'end_stream "${REPLACED:-}"' EXIT
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
  # Growth only words the note below: a stream can hang after it grew.
  GREW=0
  if [ "$NEWEST" = "$LAST_FILE" ] && [ "$(size_of "$NEWEST")" -gt "$LAST_SIZE" ]; then
    GREW=1
  fi
  note "rc=$RC verdict=$VERDICT $(printf '%s' "$OUT" | tr -d '\n')"
  if [ -n "$REPLACED" ]; then
    end_stream "$REPLACED"
    REPLACED=
    note "ended the log stream the last restart replaced, once this poll had projected it beside the new one"
  fi
  if [ "$GAP" = gap ] && [ "$gap_alarmed" -eq 0 ]; then
    gap_alarmed=1
    alarm "the log streams lost lines or began late (log_gap in go-no-go.log): after a late start the estimate gives no GO, after lost slice, opening or worker-start lines no estimate at all; with restart_overlaps in the report the gap may instead come from a restart that repeated a second's records, which are counted once; the runner's budget still bounds the run, and a stall raises an alarm"
  fi
  if [ "$HOLE" = hole ] && [ "$hole_alarmed" -eq 0 ]; then
    hole_alarmed=1
    alarm "a restarted log stream did not replay what the app printed before it (stream_holes in go-no-go.log): the estimate waits while a slice that began before the hole ended has no line (hole_suspects), and a time NO-GO waits while a pool the hole could reach has logged fewer worker starts than workers (starts_unconfirmed), since a lost start line makes the estimate read late; a lost line the estimate can prove raises the gap alarm on its own and keeps it waiting for good; the runner's budget still bounds the run, and a stall raises an alarm"
  fi
  # A stream that runs but delivers nothing (newest_stream_empty at two
  # polls in a row) holds the estimate and the stall check for good, so the
  # watcher replaces it, once a poll, and alarms when that does not help.
  if [ "$EMPTY" = empty ]; then
    EMPTY_POLLS=$((EMPTY_POLLS + 1))
  else
    EMPTY_POLLS=0
    EMPTY_RECONNECTS=0
    empty_alarmed=0
  fi
  if [ "$EMPTY_POLLS" -ge 2 ]; then
    if [ "$EMPTY_RECONNECTS" -lt "$RECONNECTS" ]; then
      EMPTY_RECONNECTS=$((EMPTY_RECONNECTS + 1))
      note "the newest log stream has delivered no scorer record at $EMPTY_POLLS polls in a row; ending it and starting another (reconnect $EMPTY_RECONNECTS of $RECONNECTS)"
      end_stream "$STREAM"
      if stream && ! ready; then
        note "the new log stream ended without a scorer record"
      fi
    elif [ "$empty_alarmed" -eq 0 ]; then
      empty_alarmed=1
      alarm "no log stream has delivered a scorer record at $EMPTY_POLLS polls in a row, after $RECONNECTS reconnect(s): the streams may be failing or the app may have gone quiet; the estimate and the stall check wait (newest_stream_empty), and the runner's budget bounds the run; check the app by hand"
    fi
  fi
  if [ "$RC" -eq 3 ] && [ "$VERDICT" = "NO-GO" ]; then
    AGE=$(( $(date +%s) - STREAM_SINCE ))
    if [ "$AGE" -lt "$SETTLE" ]; then
      note "NO-GO not acted on: the log stream started ${AGE}s ago and may still be replaying, so the files can hold an old stretch of the log; the next poll decides"
    elif [ "$RESTARTS" -eq 0 ]; then
      if [ "$GREW" -eq 0 ]; then
        why="$NEWEST has not grown since the last poll, so its stream may have hung"
      else
        why="$NEWEST grew since the last poll, but its stream may have hung after it grew"
      fi
      note "NO-GO not acted on: $why; restarting it, with the old stream left running until the next poll, which acts on a NO-GO only if a stream started from here on shows a scorer record printed after both $NOW_UTC and its own start"
      restart_streams
    else
      # Act only on a log a stream started since the last restart has
      # carried past both the last poll (at or after that restart's) and
      # its own start (review r9).
      floor=$EPOCH
      if [ "$LAST_POLL_AT" -gt "$floor" ]; then floor=$LAST_POLL_AT; fi
      pairs=()
      i=0
      while [ "$i" -lt ${#SINCE[@]} ]; do
        after=${SINCE_AT[$i]}
        if [ "$after" -lt "$floor" ]; then after=$floor; fi
        pairs+=("$after" "${SINCE[$i]}")
        i=$((i + 1))
      done
      records_after ${pairs[@]+"${pairs[@]}"}
      fresh=$?
      if [ "$fresh" -eq 0 ]; then
        if stop_app; then exit 3; fi
        exit 5
      fi
      if [ "$fresh" -eq 1 ]; then
        why="no log stream started since the last restart (at $EPOCH_UTC) holds a scorer record printed after both the last poll ($LAST_POLL_UTC) and its own start"
      else
        why="the log streams started since the last restart could not be read (see $DIR/projection.err)"
      fi
      if [ "$unconfirmed_alarmed" -eq 0 ]; then
        unconfirmed_alarmed=1
        alarm "NO-GO not confirmed: $why, so the watcher cannot show that the log is current (a scorer that has gone quiet reads the same way); it restarts the stream up to $RECONNECTS more time(s) for this run of NO-GOs and otherwise leaves the run to the runner's budget; check the app by hand"
      fi
      if [ "$RESTARTS" -le "$RECONNECTS" ]; then
        note "NO-GO not confirmed: $why; restarting the stream (restart $((RESTARTS + 1)) of $((RECONNECTS + 1)) for this run of NO-GOs), with the old stream left running until the next poll, which decides"
        restart_streams
      else
        note "NO-GO not confirmed: $why; no restarts left for this run of NO-GOs, so the runner's budget bounds the run"
      fi
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
    # A WAIT that only waits for an empty newest stream says nothing new
    # about the run, so it leaves the restarts used (review r9).
    if [ "$EMPTY" != empty ]; then
      RESTARTS=0
      unconfirmed_alarmed=0
    fi
  fi
  LAST_FILE=$DIR/app-$STREAMS.log
  LAST_SIZE=$(size_of "$LAST_FILE")
  if [ "$NOW_AT" -gt 0 ]; then
    LAST_POLL_AT=$NOW_AT
    LAST_POLL_UTC=$NOW_UTC
  else
    # A poll with no report: its time on the projection's clock, from this
    # machine's (BSD date, then GNU date).
    LAST_POLL_AT=$((PROJECTED_FROM + CLOCK_OFFSET))
    LAST_POLL_UTC=$(date -u -r "$LAST_POLL_AT" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
      date -u -d "@$LAST_POLL_AT" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null ||
      printf '%s s' "$LAST_POLL_AT")
  fi
done
