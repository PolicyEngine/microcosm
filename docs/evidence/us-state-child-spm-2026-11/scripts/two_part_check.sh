#!/bin/zsh
# Hold the ACS-local rerun between states, check that acs_rerun.py's two steps give exactly what one process gives, run the
# published-rent states the 8 GiB stop refused, and let the rerun go on. Every exit path releases the hold.
# Each run waits for 48 GiB available and 58 GB free, on Fleet ops' terms, and keeps the 8 GiB stop.
# Usage: two_part_check.sh <acs_rerun.sh pid>
SP=${0:a:h}; cd $SP
DRIVER=$1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export IGNORE_HOLD=1
LOG=run/two_part_check.log; PY=env/.venv/bin/python; OUT=run/two_part_check
note() { print -r -- "$(date -u +%TZ) $*" >> $LOG; }
[[ "$(ps -p $DRIVER -o args=)" == "zsh acs_rerun.sh" ]] || { note "stopped: pid $DRIVER is not the rerun driver"; exit 1; }
release() { rm -f run/hold && note "hold released"; }
trap release EXIT
trap 'exit 130' INT TERM HUP
: > run/hold
note "hold set"
# The driver's next run waits at its start. Wait for a run that started before the hold, which has no held.<pid> mark.
busy() { local pid; for pid in $(pgrep -P $DRIVER -f acs_rerun.py); do [[ -e run/held.$pid ]] || return 0; done; return 1; }
while busy; do sleep 20; done
note "rerun held between states"
run() {  # run VAR=value [VAR=value ...] <fips> <tag>
  python3 -I host_wait.py 48 58 $LOG
  env "${@[1,-3]}" nice -n 5 $PY acs_rerun.py "${@[-2]}" "${@[-1]}" >> $LOG 2>&1
  local rc=$?
  (( rc == 0 )) || note "failed ${@[-2]} ${@[-1]} exit $rc"
  return $rc
}
fail=0
for f in 56 10 05; do run TWO_PART=1 OUT=$OUT/acs $f acs || fail=1; done
run TWO_PART=0 OUT=$OUT/acsrent_one 56 acsrent || fail=1
run TWO_PART=1 OUT=$OUT/acsrent 56 acsrent || fail=1
$PY -I two_part_cmp.py >> $LOG 2>&1 || fail=1
if (( fail )); then note "two steps did not reproduce one process, so the refused states wait"; exit 1; fi
: > run/two_part_ok
note "two steps reproduce one process"
for f in 48 06; do run TWO_PART=1 $f acs; done
note "refused states finished"
