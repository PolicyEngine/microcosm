#!/bin/zsh
# Baseline launcher for the native-scale lane. Refuses if the machine does not
# have the brief's headroom; starts its own session (macOS has no setsid(1))
# and caps CPU at 9,000 s before exec'ing the harness.
set -eu
MEASURE_DIR=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-scale-baseline/.measure
VENV=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-native-v5/.venv/bin/python
AVAIL_GB=$(vm_stat | awk -F'[:.]' '/Pages free/{f=$2} /Pages inactive/{i=$2} /Pages speculative/{s=$2} /Pages purgeable/{p=$2} END{printf "%.1f", (f+i+s+p)*16384/1e9}')
echo "available_gb=$AVAIL_GB"
awk -v a="$AVAIL_GB" 'BEGIN{exit !(a>60)}' || { echo "REFUSED: available memory ${AVAIL_GB} GB <= 60 GB"; exit 3; }
exec "$VENV" -I -B -S "$MEASURE_DIR/setsid_exec.py" 9060 "$VENV" -I -B -S "$MEASURE_DIR/harness19_base.py"
