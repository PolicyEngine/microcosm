#!/bin/zsh
# The 1/10 run: 158,737 source households, above the 96,860 the single bounded
# encode admitted. Waits for the brief's 70 GB of headroom rather than competing
# with another lane, then starts its own session (macOS has no setsid(1)) and
# caps CPU at 21,600 s plus the harness's own 60 s of flush margin.
set -eu
MEASURE_DIR=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-scale-tenth/.measure
VENV=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-v5/.venv/bin/python
NEED_GB=${NEED_GB:-70}
DEADLINE=$((SECONDS + 21600))
# Do not compete with this lane's own 1/1000 after-run: its wall clock and its
# peak RSS are a measurement too.
AFTER_PID=${AFTER_PID:-}
if [ -n "$AFTER_PID" ]; then
  while ps -p "$AFTER_PID" > /dev/null 2>&1; do
    echo "$(date -u +%H:%M:%SZ) waiting for the after-run (pid $AFTER_PID) to finish"
    sleep 60
  done
  echo "$(date -u +%H:%M:%SZ) after-run (pid $AFTER_PID) has exited"
fi
available() {
  vm_stat | awk -F'[:.]' '/Pages free/{f=$2} /Pages inactive/{i=$2} /Pages speculative/{s=$2} /Pages purgeable/{p=$2} END{printf "%.1f", (f+i+s+p)*16384/1e9}'
}
while :; do
  GB=$(available)
  if awk -v a="$GB" -v n="$NEED_GB" 'BEGIN{exit !(a>n)}'; then
    echo "$(date -u +%H:%M:%SZ) gate open: available ${GB} GB > ${NEED_GB} GB"
    break
  fi
  if [ $SECONDS -ge $DEADLINE ]; then
    echo "$(date -u +%H:%M:%SZ) REFUSED: available ${GB} GB never exceeded ${NEED_GB} GB within 6 h"
    exit 3
  fi
  echo "$(date -u +%H:%M:%SZ) waiting: available ${GB} GB <= ${NEED_GB} GB"
  sleep 60
done
exec "$VENV" -I -B -S "$MEASURE_DIR/setsid_exec.py" 21660 "$VENV" -I -B -S "$MEASURE_DIR/harness19_tenth.py"
