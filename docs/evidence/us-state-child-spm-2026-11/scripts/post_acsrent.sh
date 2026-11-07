#!/bin/zsh
# After the corrected-rent parts: the corrected-rent evaluation, the reform comparison, the fill comparison and the Florida
# trace, then South Dakota's reform under the -1 fill and the unit traces. One process at a time, each after the host gate.
# Each script's output goes to run/logs/<name>.txt.
# Usage: post_acsrent.sh <acsrent_parts.sh pid>
SP=${0:a:h}; cd $SP
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
PY=env/.venv/bin/python; LOG=run/post_acsrent.log
note() { print -r -- "$(date -u +%TZ) $*" >> $LOG; }
P=$1
args="$(ps -p $P -o args= 2>/dev/null)"
[[ -z $args || $args == "zsh acsrent_parts.sh" ]] || { note "stopped: pid $P is not acsrent_parts.sh"; exit 1; }
while kill -0 $P 2>/dev/null; do sleep 30; done
n=$(ls run/acsrent/*.parquet | wc -l | tr -d ' ')
[[ $n == 51 ]] || { note "stopped: run/acsrent holds $n states, not 51"; exit 1; }
gate() { python3 -I host_wait.py 48 58 $LOG; }
step() { local out=$1; shift; gate; note "$out"; nice -n 5 "$@" > run/logs/$out.txt 2>&1 || { note "failed $out exit $?"; exit 1; }; }
step acsrent_eval $PY acs_eval.py acsrent
step acs_reform_cmp $PY acs_reform_cmp.py
step fill_cmp $PY fill_cmp.py
step fl_reform $PY fl_reform.py
gate; note "South Dakota under -1 with the reform"
FILL=-1 WITH_REFORM=1 OUT=run/check_fill-1_reform nice -n 5 $PY acs_rerun.py 46 acs >> $LOG 2>&1 || { note "failed South Dakota run exit $?"; exit 1; }
step fill_trace_SD $PY fill_trace.py SD check_fill-1_reform
step fill_trace_CA $PY fill_trace.py CA check_fill-1
step fill_trace_DE $PY fill_trace.py DE check_fill-1
note "post-acsrent scripts finished"
