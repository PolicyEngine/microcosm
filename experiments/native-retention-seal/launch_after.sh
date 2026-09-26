#!/bin/zsh
# The 1/1000 after-run for the retention-seal lane: the transport lane's own
# committed harness, unedited, at the same arguments, on this branch's head,
# followed by the required replay of the same graph against its own store.
#
# The harness is a byte-identical copy of
# experiments/native-scale-transport/harness19_after_with_required_replay.py.
# The interpreter and site-packages are the same v5 venv both transport runs
# used, so the third-party versions the implementation manifest records are the
# ones those runs recorded.
#
# macOS has no setsid(1); setsid_exec.py starts the session and caps CPU before
# exec'ing the harness. The pid the launching shell prints is the session leader
# and stays the harness's own pid across the exec.
set -eu
TREE=${TREE:?set TREE to the detached measurement worktree}
MEASURE_DIR="$TREE/.measure"
VENV=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-v5/.venv/bin/python
NEED_GB=${NEED_GB:-40}
DEADLINE=$((SECONDS + 21600))
export MEASURE_BASE="$TREE"
export MEASURE_RUN=${MEASURE_RUN:-/Users/maxghenis/PolicyEngine/_recovered/pilot-runs/native45-v5/run}
export MEASURE_PROBE="$MEASURE_DIR/after"
export MEASURE_LABEL=${MEASURE_LABEL:-native-retention-seal-after-1-1000}
export MEASURE_HEAD=${MEASURE_HEAD:?set MEASURE_HEAD to the sha the tree is detached at}
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
exec "$VENV" -I -B -S "$MEASURE_DIR/setsid_exec.py" 9060 \
     "$VENV" -I -B -S "$MEASURE_DIR/harness19_after_with_required_replay.py"
