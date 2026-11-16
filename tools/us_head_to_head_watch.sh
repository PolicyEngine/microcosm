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
# and per action) and RUN_DIR/ALARM (only when something needs a person). Exit: 0 the app finished on its own; 3 stopped on NO-GO;
# 4 stopped because the projection failed WATCH_MAX_FAILURES polls in a row;
# 5 a stop was asked for but not confirmed; 2 bad arguments, or an app that
# is not running at the start.
#
# The cap and the runner's budget bound the spend; this only saves part of it.
# Env: WATCH_INTERVAL (seconds, default 600), WATCH_MAX_FAILURES (3),
# WATCH_STOP_TRIES (5), MODAL (modal), PYTHON (python3).
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
MAX_FAILURES=${WATCH_MAX_FAILURES:-3}
STOP_TRIES=${WATCH_STOP_TRIES:-5}
MODAL=${MODAL:-modal}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
VERDICTS=$DIR/go-no-go.log
# A watcher restarted on the same RUN_DIR goes on from the files it finds.
STREAMS=0
while [ -e "$DIR/app-$((STREAMS + 1)).log" ]; do STREAMS=$((STREAMS + 1)); done

if ! STARTED=$(cat "$DIR/started-at" 2>/dev/null) || [ -z "$STARTED" ]; then
  echo "$0: $DIR/started-at is missing or empty" >&2
  exit 2
fi

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
note() { printf '%s %s\n' "$(stamp)" "$1" >> "$VERDICTS"; }
alarm() {
  printf '%s ALARM %s\n' "$(stamp)" "$1" >> "$DIR/ALARM"
  note "ALARM $1"
}

# Modal 1.4 and later need --follow to keep streaming, and 1.4.2 and later
# ask before stopping unless given --yes; 1.3 has neither flag.
FOLLOW=()
if "$MODAL" app logs --help 2>&1 | grep -q -- '--follow'; then FOLLOW=(--follow); fi
YES=()
if "$MODAL" app stop --help 2>&1 | grep -q -- '--yes'; then YES=(--yes); fi

# 0 when the app is listed and not stopped; 1 when it is stopped or gone;
# 2 when the list could not be read.
app_state() {
  local listing
  listing=$("$MODAL" app list --json 2>/dev/null) || return 2
  printf '%s' "$listing" | "$PYTHON" -c '
import json, sys
apps = {a.get("App ID"): a for a in json.load(sys.stdin)}
app = apps.get(sys.argv[1])
sys.exit(0 if app is not None and app.get("Stopped at") is None
         and app.get("State") != "stopped" else 1)
' "$APP" 2>/dev/null
  local rc=$?
  [ $rc -le 1 ] && return $rc
  return 2
}

stop_app() {
  local try
  for try in $(seq 1 "$STOP_TRIES"); do
    "$MODAL" app stop ${YES[@]+"${YES[@]}"} "$APP" >> "$VERDICTS" 2>&1
    sleep "${WATCH_STOP_WAIT:-30}"
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

STREAM=
stream() {
  STREAMS=$((STREAMS + 1))
  "$MODAL" app logs ${FOLLOW[@]+"${FOLLOW[@]}"} "$APP" > "$DIR/app-$STREAMS.log" 2>&1 &
  STREAM=$!
}

# ${ARRAY[@]+...} keeps an empty array from tripping set -u on bash 3.2
# (macOS's /bin/bash).
app_state
if [ $? -ne 0 ]; then
  echo "$0: $APP is not a running app in \`$MODAL app list\`" >&2
  exit 2
fi
note "watching $APP (plan $PLAN, started-at $STARTED, every ${INTERVAL}s)"
stream
failures=0
while true; do
  sleep "$INTERVAL"
  ARGS=("$HERE/us_head_to_head_projection.py" "$DIR"/app-*.log --plan "$PLAN" --started-at "$STARTED")
  if [ -n "$WEIGHTS" ]; then ARGS+=(--chunk-weights "$WEIGHTS"); fi
  OUT=$("$PYTHON" "${ARGS[@]}" 2>> "$DIR/projection.err")
  RC=$?
  VERDICT=$(printf '%s' "$OUT" | "$PYTHON" -c '
import json, sys
print(json.load(sys.stdin).get("verdict", ""))' 2>/dev/null)
  note "rc=$RC verdict=${VERDICT:-none} $(printf '%s' "$OUT" | tr -d '\n')"
  if [ "$RC" -eq 3 ] && [ "$VERDICT" = "NO-GO" ]; then
    if stop_app; then exit 3; fi
    exit 5
  fi
  if [ "$RC" -ne 0 ] || { [ "$VERDICT" != "GO" ] && [ "$VERDICT" != "WAIT" ]; }; then
    failures=$((failures + 1))
    alarm "no verdict from the projection (rc=$RC), $failures poll(s) in a row"
    if [ "$failures" -ge "$MAX_FAILURES" ]; then
      alarm "stopping $APP: the log could not be judged for $failures polls"
      if stop_app; then exit 4; fi
      exit 5
    fi
  else
    failures=0
  fi
  if ! kill -0 "$STREAM" 2>/dev/null; then
    app_state
    case $? in
      0) note "the log stream ended while $APP runs; restarting it"; stream ;;
      1) note "$APP has finished; the watcher ends"; exit 0 ;;
      *) alarm "the log stream ended and the app list could not be read; retrying"; stream ;;
    esac
  fi
done
