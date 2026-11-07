#!/bin/zsh
# Hold the ACS-local rerun between states, check that states run in household parts give exactly what whole-state runs give,
# run the published-rent states whose baseline alone passes 8 GiB (Texas in 3 parts, California in 4), and let the rerun go
# on. Every exit path releases the hold.
# Usage: parts_check.sh <acs_rerun.sh pid>
SP=${0:a:h}; cd $SP
DRIVER=$1
export IGNORE_HOLD=1 PARTS_LOG=run/parts_check.log
LOG=run/parts_check.log; PY=env/.venv/bin/python
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
fail=0
zsh acs_parts.sh 56 acs 2 run/parts_check/acs || fail=1
zsh acs_parts.sh 10 acs 3 run/parts_check/acs || fail=1
zsh acs_parts.sh 5 acs 2 run/parts_check/acs || fail=1
zsh acs_parts.sh 56 acsrent 2 run/parts_check/acsrent || fail=1
zsh acs_parts.sh 50 acsrent 3 run/parts_check/acsrent || fail=1
$PY -I acs_parts_cmp.py >> $LOG 2>&1 || fail=1
if (( fail )); then note "parts did not reproduce whole-state runs, so Texas and California wait"; exit 1; fi
: > run/parts_ok
note "parts reproduce whole-state runs"
zsh acs_parts.sh 48 acs 3 || note "Texas did not finish"
zsh acs_parts.sh 6 acs 4 || note "California did not finish"
note "Texas and California runs ended"
