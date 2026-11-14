#!/bin/zsh
# Split the ACS-local file through range requests, with no local copy of it, then prepare the keys and start the rerun.
# Fleet ops' terms: one process, one thread, nice 5, the 8 GiB stop, no more than 3 MB/s, and 48 GiB available and 58 GB free before each chunk and each state.
# A stop loses little: run this script again and the split starts after its last finished chunk.
W=${0:a:h}; cd $W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
URL=https://huggingface.co/datasets/policyengine/populace-us/resolve/725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5
SHA=3ce781c440a7f933b1d952ff4e9d9f65f4698c64b94d2fad165af0c662838d6c
SIZE=9819603807
MEM_GIB=48; FREE_GB=58
LOG=run/fetch_split.log; PY=env/.venv/bin/python
note() { print "$(date -u +%TZ) $*" >> $LOG; }
run_step() {
  python3 -I host_wait.py $MEM_GIB $FREE_GB $LOG
  nice -n 5 $PY "$@"
}
mkdir -p run/logs
note "range split launched"
SPLIT_GATE="$MEM_GIB $FREE_GB $LOG" SPLIT_SIZE=$SIZE SPLIT_SHA256=$SHA SPLIT_CHUNK=200000 nice -n 5 $PY -I acs_split.py $URL run/acs_split >> run/logs/acs_split.txt 2>&1
rc=$?
if [ $rc != 0 ]; then note "stopped: split exit $rc, see run/logs/acs_split.txt"; exit $rc; fi
note "split done: MB $(du -sm run/acs_split | cut -f1), free $(python3 -I ~/reviews/us-hub/scripts/free_gb.py) GB"
run_step rentfix.py > run/logs/rentfix.txt 2>&1 || { note "stopped: rentfix"; exit 1; }
run_step rent_map.py > run/logs/rent_map.txt 2>&1 || { note "stopped: rent_map"; exit 1; }
run_step acs_hh_split.py > run/logs/acs_hh_split.txt 2>&1 || { note "stopped: acs_hh_split"; exit 1; }
run_step -I blank_holders.py run/acs_split run/blank_holders_state.csv > run/logs/blank_holders.txt 2>&1 || note "blank_holders failed, rerun not held"
run_step -I parent_blanks.py > run/logs/parent_blanks.txt 2>&1 || note "parent_blanks failed, rerun not held"
note "rerun launch"
RERUN_FREE_GB=$FREE_GB zsh acs_rerun.sh
note "rerun finished: exit $?"
