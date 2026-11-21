#!/bin/zsh
# The post-run scripts that need only the published-rent pass, one at a time: the 51-state check against the deck, then the
# ACS-local evaluation and the scripts that read it. Each script's output goes to run/logs/<script>.txt.
SP=${0:a:h}; cd $SP
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
PY=env/.venv/bin/python
# dash_swing.csv in git holds the -1 fill's version; keep it beside the rerun's for the before-and-after.
[[ -e run/dash_swing_fill_minus1.csv ]] || cp run/dash_swing.csv run/dash_swing_fill_minus1.csv
for s in acs_rerun_check acs_eval persist noise_ci dash_swing; do
  print -r -- "$(date -u +%TZ) $s"
  args=()
  [[ $s == acs_eval ]] && args=(acs)
  nice -n 5 $PY $s.py $args > run/logs/$s.txt 2>&1 || { print -r -- "failed $s exit $?"; exit 1; }
done
print -r -- "$(date -u +%TZ) post-acs scripts finished"
