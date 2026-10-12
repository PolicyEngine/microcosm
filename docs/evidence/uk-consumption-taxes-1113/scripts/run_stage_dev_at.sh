#!/bin/bash
# Stage-only development run of lcfs_consumption for a given tree (microcosm#1113).
# Usage: run_stage_dev_at.sh <tree> <label>
# Runs the tree's own stage code on the head-r3 spine with the PR worktree's venv
# (PYTHONPATH puts the tree's packages first), behind a quiet-machine wait of at
# most 15 minutes, at low priority. Writes stage_dev_run.<label>.json beside this file.
set -uo pipefail
TREE=$1
LABEL=$2
D=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/vat-fuel-energy-measure-2026-10-08
VENV=/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-consumption-taxes/.venv
SPINE=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1095-uk-data-ports-2/arm-head-r3/spine/spine-head-r3.h5
LOG=$D/stage_dev_run.$LABEL.log
echo "== stage dev run start $(date -u +%FT%TZ) tree $TREE $(git -C "$TREE" rev-parse --short HEAD) dirty=$(git -C "$TREE" status --short | wc -l | tr -d ' ')" > "$LOG"
deadline=$(( $(date +%s) + 900 ))
while true; do
  busy=$(ps -eo pcpu,command | awk '$1 > 50.0' | grep -v -E "stage_dev_run|awk|ps -eo" | wc -l | tr -d ' ')
  avail=$(vm_stat | awk '/Pages free|Pages inactive/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1e9}')
  if [[ $busy -le 2 && $avail -ge 6 ]]; then echo "== quiet enough at $(date -u +%FT%TZ): busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  if [[ $(date +%s) -ge $deadline ]]; then echo "== waited 15 min; proceeding at low priority: busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  sleep 30
done
cd "$TREE"
PP=$(ls -d "$TREE"/packages/*/src | tr '\n' ':')
PYTHONPATH="$PP" HDF5_USE_FILE_LOCKING=FALSE /usr/bin/time -l nice -n 15 "$VENV/bin/python" "$D/stage_dev_run.py" "$SPINE" "$D/stage_dev_run.$LABEL.json" >> "$LOG" 2>&1
echo "== exit=$? $(date -u +%FT%TZ)" >> "$LOG"
