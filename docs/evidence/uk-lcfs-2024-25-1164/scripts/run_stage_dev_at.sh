#!/bin/bash
# Stage-only development run of lcfs_consumption for a given tree (microcosm#1164).
# Usage: run_stage_dev_at.sh <tree> <label> [script]
# Runs the tree's own stage code (stage_dev_run.py, or nts_lcfs_chain_run.py for
# the NTS-then-LCFS chain), on the tabs its own spec pins, on one
# finished spine with this worktree's venv (PYTHONPATH puts the tree's packages
# first), behind a quiet-machine wait of at most 15 minutes, at low priority.
# Writes <script>.<label>.json (aggregates) and .households.parquet (unit
# records) to the acceptance directory outside the repository.
set -uo pipefail
PE="${PE:-$HOME/Desktop/PolicyEngine}"
TREE=$1
LABEL=$2
SCRIPT=${3:-stage_dev_run.py}
HERE=$(cd "$(dirname "$0")" && pwd)
D=$PE/data/ukds/acceptance/1164-lcfs-2024-25
VENV=${VENV:-$(cd "$HERE/../../../.." && pwd)/.venv}
SPINE=${SPINE:-$PE/data/build/uk-inputs/spine/uk_frs_spine_2024_25.h5}
LOG=$D/${SCRIPT%.py}.$LABEL.log
mkdir -p "$D"
echo "== stage dev run start $(date -u +%FT%TZ) tree $TREE $(git -C "$TREE" rev-parse --short HEAD) dirty=$(git -C "$TREE" status --short | wc -l | tr -d ' ') spine $SPINE" > "$LOG"
deadline=$(( $(date +%s) + 900 ))
while true; do
  busy=$(ps -eo pcpu,command | awk '$1 > 50.0' | grep -v -E "stage_dev_run|chain_run|awk|ps -eo" | wc -l | tr -d ' ')
  avail=$(vm_stat | awk '/Pages free|Pages inactive/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1e9}')
  if [[ $busy -le 2 && $avail -ge 6 ]]; then echo "== quiet enough at $(date -u +%FT%TZ): busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  if [[ $(date +%s) -ge $deadline ]]; then echo "== waited 15 min; proceeding at low priority: busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  sleep 30
done
cd "$TREE"
PP=$(ls -d "$TREE"/packages/*/src | tr '\n' ':')
PYTHONPATH="$PP" PE="$PE" HDF5_USE_FILE_LOCKING=FALSE /usr/bin/time -l nice -n 15 "$VENV/bin/python" "$HERE/$SCRIPT" "$SPINE" "$D/${SCRIPT%.py}.$LABEL.json" >> "$LOG" 2>&1
echo "== exit=$? $(date -u +%FT%TZ)" >> "$LOG"
