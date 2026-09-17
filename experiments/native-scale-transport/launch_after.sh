#!/bin/zsh
# After-run launcher for the native-scale lane: the same 19-node harness at the
# same arguments, on this branch's head. Waits for the brief's memory headroom
# rather than competing with another lane's live run, then starts its own
# session (macOS has no setsid(1)) and caps CPU before exec'ing the harness.
set -eu
MEASURE_DIR=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-scale-after/.measure
VENV=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-v5/.venv/bin/python
NEED_GB=${NEED_GB:-60}
DEADLINE=$((SECONDS + 14400))
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
    echo "$(date -u +%H:%M:%SZ) REFUSED: available ${GB} GB never exceeded ${NEED_GB} GB within 4 h"
    exit 3
  fi
  echo "$(date -u +%H:%M:%SZ) waiting: available ${GB} GB <= ${NEED_GB} GB"
  sleep 60
done
exec "$VENV" -I -B -S "$MEASURE_DIR/setsid_exec.py" 9060 "$VENV" -I -B -S "$MEASURE_DIR/harness19_base.py"
