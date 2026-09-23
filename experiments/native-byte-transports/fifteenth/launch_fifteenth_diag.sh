#!/bin/zsh
# The 1/15 refusal diagnostic. The 1/15 measurement (pid 45669) STOPPED at
# 605.65 CPU-s with SurveyPopulationPreparationError: PREPARATION_ISSUANCE_REFUSED,
# a catch-all that discards its cause (`raise ... from None`,
# survey_population_preparation.py:2084). This re-runs the same fraction in the
# same measured tree with harness19_diag_fifteenth.py -- the 1/15 harness plus
# the committed 1/10 diagnostic's sys.monitoring RAISE block, which changes no
# byte of the measured tree and so moves no source pin.
#
# Gate: more than 45 GB available, the same gate the measurement used; never
# lowered. Ceilings: RLIMIT_CPU 21,660 s, harness soft 21,600 CPU-s,
# 43,200 wall-s, 48 GiB RSS -- the measurement's, unchanged.
set -eu
TREE=${TREE:?set TREE to the detached measurement worktree}
MEASURE_DIR="$TREE/.measure"
# Byte-transport lane: this tree's lock moved, so the venv is an argument.
VENV=${MEASURE_VENV:?set MEASURE_VENV to the measured venv}/bin/python
export MEASURE_VENV
NEED_GB=${NEED_GB:-45}
DEADLINE=$((SECONDS + 21600))
export MEASURE_BASE="$TREE"
export MEASURE_RUN=${MEASURE_RUN:-/Users/maxghenis/PolicyEngine/_worktrees/native-scale-runs/run-fifteenth}
export MEASURE_PROBE="$MEASURE_DIR/fifteenth-diag"
export MEASURE_LABEL=${MEASURE_LABEL:-native-retention-seal-1-15-diagnostic}
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
exec "$VENV" -I -B -S "$MEASURE_DIR/setsid_exec.py" 21660 \
     "$VENV" -I -B -S "$MEASURE_DIR/harness19_diag_fifteenth.py"
