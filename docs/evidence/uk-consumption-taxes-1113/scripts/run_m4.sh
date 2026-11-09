#!/bin/bash
# M4 runner: wait for a quieter machine, then two solves (base, drop4) and the report, all at low priority.
set -uo pipefail
PE="${PE:-$HOME/Desktop/PolicyEngine}"
D=$PE/data/ukds/acceptance/vat-fuel-energy-measure-2026-10-08
TREE=$PE/repos/populace-national-main
LOG=$D/run_m4.log
echo "== M4 start $(date -u +%FT%TZ)" > "$LOG"
deadline=$(( $(date +%s) + 3600 ))
while true; do
  busy=$(ps -eo pcpu,command | awk '$1 > 50.0' | grep -v -E "m4_resolve|awk|ps -eo" | wc -l | tr -d ' ')
  avail=$(vm_stat | awk '/Pages free|Pages inactive/ {gsub("\\.","",$NF); s+=$NF} END {printf "%d", s*16384/1e9}')
  if [[ $busy -le 2 && $avail -ge 4 ]]; then echo "== quiet enough at $(date -u +%FT%TZ): busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  if [[ $(date +%s) -ge $deadline ]]; then echo "== waited 60 min; proceeding at low priority: busy=$busy avail=${avail}GB" >> "$LOG"; break; fi
  sleep 30
done
cd "$TREE"
for arm in base drop4; do
  /usr/bin/time -l nice -n 15 .venv/bin/python "$D/m4_resolve.py" solve --arm $arm >> "$LOG" 2>&1
  echo "== $arm exit=$? $(date -u +%FT%TZ)" >> "$LOG"
done
nice -n 15 .venv/bin/python "$D/m4_resolve.py" report > "$D/m4_report.stdout" 2>&1
echo "== report exit=$? $(date -u +%FT%TZ)" >> "$LOG"
