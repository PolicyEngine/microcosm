#!/bin/zsh
# Run a command while the ACS-local rerun waits between states, so one process runs at a time. Every exit path releases the
# hold. The command runs after the host gate (48 GiB available, 58 GB free).
# Usage: with_hold.sh <acs_rerun.sh pid> <log> <command...>
SP=${0:a:h}; cd $SP
DRIVER=$1; LOG=$2; shift 2
note() { print -r -- "$(date -u +%TZ) $*" >> $LOG; }
[[ "$(ps -p $DRIVER -o args=)" == "zsh acs_rerun.sh" ]] || { note "stopped: pid $DRIVER is not the rerun driver"; exit 1; }
[[ -e run/hold ]] && { note "stopped: another hold is set"; exit 1; }
release() { rm -f run/hold && note "hold released"; }
trap release EXIT
trap 'exit 130' INT TERM HUP
: > run/hold
note "hold set"
# The driver's next run waits at its start. Wait for a run that started before the hold, which has no held.<pid> mark.
busy() { local pid; for pid in $(pgrep -P $DRIVER -f acs_rerun.py); do [[ -e run/held.$pid ]] || return 0; done; return 1; }
while busy; do sleep 20; done
note "rerun held between states"
python3 -I host_wait.py 48 58 $LOG
IGNORE_HOLD=1 "$@" >> $LOG 2>&1
rc=$?
note "command exit $rc"
exit $rc
