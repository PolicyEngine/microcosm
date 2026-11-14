#!/bin/zsh
# Run one ACS-local state in household parts, each part in acs_rerun.py's two steps under the 8 GiB stop, then merge the parts
# into the state's output: for states whose baseline alone passes 8 GiB (Texas and California).
# Each step waits for 48 GiB available and 58 GB free first, on Fleet ops' terms. A finished part skips on a rerun.
# Usage: acs_parts.sh <fips> <tag> <parts> [<output folder, default run/<tag>>]
SP=${0:a:h}; cd $SP
F=$1; TAG=$2; K=$3; DEST=${4:-run/$TAG}
FF=${(l:2::0:)F}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
LOG=${PARTS_LOG:-run/acs_parts.log}; PY=env/.venv/bin/python
DIR=run/parts/${DEST//\//_}/$FF-k$K
note() { print -r -- "$(date -u +%TZ) $*" >> $LOG; }
[[ -e $DEST/$FF.parquet ]] && { note "$FF $TAG already in $DEST"; exit 0; }
gate() { python3 -I host_wait.py 48 58 $LOG; }
gate
nice -n 5 $PY parts_ids.py $F $K $DIR >> $LOG 2>&1 || { note "failed $FF $TAG parts_ids exit $?"; exit 1; }
for (( k = 0; k < K; k++ )); do
  gate
  env TWO_PART=1 OUT=$DIR/p$k HOUSEHOLDS=$DIR/p$k.npy nice -n 5 $PY acs_rerun.py $F $TAG >> $LOG 2>&1 || { note "failed $FF $TAG part $k exit $?"; exit 1; }
done
nice -n 5 $PY parts_merge.py $F $TAG $K $DIR $DEST >> $LOG 2>&1 || { note "failed $FF $TAG merge exit $?"; exit 1; }
