#!/bin/zsh
# The 1/15 run: 105,825 source households, above the 96,860 the single bounded
# encode admitted. Max chose 1/15 now and 1/10 when the machine is quiet
# (2026-09-17); the native-scale lane's own 1/10 waiter stays queued at its own
# 70 GB gate and this run does not lower it.
#
# Gates, in order: do not compete with this lane's own 1/1000 after-run, then
# wait for more than 45 GB of available memory. Ceilings: RLIMIT_CPU 21,600 s
# (plus the harness's own 60 s flush margin), 43,200 wall-s, 48 GiB RSS.
# macOS has no setsid(1); setsid_exec.py starts the session and caps CPU before
# exec'ing the harness, and the harness keeps its own pid across the exec.
set -eu
TREE=${TREE:?set TREE to the detached measurement worktree}
MEASURE_DIR="$TREE/.measure"
VENV=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-v5/.venv/bin/python
NEED_GB=${NEED_GB:-45}
DEADLINE=$((SECONDS + 21600))
AFTER_PID=${AFTER_PID:-}
export MEASURE_BASE="$TREE"
export MEASURE_RUN=${MEASURE_RUN:-/Users/maxghenis/PolicyEngine/_worktrees/native-scale-runs/run-fifteenth}
export MEASURE_PROBE="$MEASURE_DIR/fifteenth"
export MEASURE_LABEL=${MEASURE_LABEL:-native-retention-seal-1-15}
export MEASURE_HEAD=${MEASURE_HEAD:?set MEASURE_HEAD to the sha the tree is detached at}
if [ -n "$AFTER_PID" ]; then
  while ps -p "$AFTER_PID" > /dev/null 2>&1; do
    echo "$(date -u +%H:%M:%SZ) waiting for the 1/1000 after-run (pid $AFTER_PID) to finish"
    sleep 60
  done
  echo "$(date -u +%H:%M:%SZ) the 1/1000 after-run (pid $AFTER_PID) has exited"
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
exec "$VENV" -I -B -S "$MEASURE_DIR/setsid_exec.py" 21660 \
     "$VENV" -I -B -S "$MEASURE_DIR/harness19_fifteenth.py"
