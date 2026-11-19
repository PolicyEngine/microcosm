#!/bin/bash
# Stage-only dev runs of lcfs_consumption for microcosm#1359: the base tree,
# then the head tree, on the same spine, with the head worktree's venv
# (policyengine-uk 2.145.0) and each tree's packages first on PYTHONPATH.
# Waits up to 15 minutes for a quiet machine, then runs at low priority.
set -uo pipefail
D=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance/1359-excise
VENV=/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1359/.venv
SPINE=/Users/mariajuaristi/Desktop/microcosm-1333-arm-spine/spine-b145-treat.h5
LOG=$D/stage_dev_run${LOG_SUFFIX:-}.log
echo "== start $(date -u +%FT%TZ)" > "$LOG"
deadline=$(( $(date +%s) + 900 ))
while true; do
  busy=$(ps -eo pcpu,command | awk '$1 > 50.0' | grep -v -E "stage_dev_run|awk|ps -eo" | wc -l | tr -d ' ')
  avail=$(vm_stat | awk '/Pages free|Pages inactive/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1e9}')
  if [[ $busy -le 2 && $avail -ge 4 ]]; then echo "== quiet enough at $(date -u +%FT%TZ): busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  if [[ $(date +%s) -ge $deadline ]]; then echo "== waited 15 min; proceeding at low priority: busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  sleep 30
done
BASE_LABEL=${BASE_LABEL:-base}; HEAD_LABEL=${HEAD_LABEL:-head}
for pair in "$BASE_LABEL:/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1359-base" "$HEAD_LABEL:/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1359"; do
  LABEL=${pair%%:*}
  TREE=${pair#*:}
  echo "== $LABEL $TREE $(git -C "$TREE" rev-parse --short HEAD) dirty=$(git -C "$TREE" status --short | wc -l | tr -d ' ') $(date -u +%FT%TZ)" >> "$LOG"
  cd "$TREE"
  PP=$(ls -d "$TREE"/packages/*/src | tr '\n' ':')
  PYTHONPATH="$PP" HDF5_USE_FILE_LOCKING=FALSE /usr/bin/time -l nice -n 15 "$VENV/bin/python" "$D/stage_dev_run_1359.py" "$TREE" "$SPINE" "$D/stage_dev_run.$LABEL.json" >> "$LOG" 2>&1
  echo "== $LABEL exit=$? $(date -u +%FT%TZ)" >> "$LOG"
done
