#!/bin/zsh
# Every ACS-local state, one process at a time, smallest first: rent as published, then rent on the right households.
# Fleet ops' terms for this run: one process, one thread, nice 5, an 8 GiB stop and under 300 MB of new disk.
# Before each state it waits for 48 GiB available and RERUN_FREE_GB (default 65) GB free, and while a host booking refuses this caller.
SP=${0:a:h}; cd $SP
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
LOG=run/acs_rerun.log; PY=env/.venv/bin/python
gate() { python3 -I host_wait.py 48 ${RERUN_FREE_GB:-65} $LOG; }
ORDER=(${(f)"$(ls -Sr run/acs_split/*-person.pkl.gz | sed -E 's#.*/([0-9]{2})-person.pkl.gz#\1#')"})
print "start $(date -u +%FT%TZ) states ${#ORDER}" >> $LOG
# What the model does with a blank, on the smallest state, and two states that must repeat the earlier -1 runs.
gate; print "blank check on ${ORDER[1]}" >> $LOG; FILL=none nice -n 5 $PY acs_rerun.py ${ORDER[1]} acs 2>&1 | tail -3 | cut -c1-300 >> $LOG
for f in 10 6; do gate; FILL=-1 nice -n 5 $PY acs_rerun.py $f acs >> $LOG 2>&1; done
for TAG in acs acsrent; do
  for f in $ORDER; do
    gate
    nice -n 5 $PY acs_rerun.py $f $TAG >> $LOG 2>&1
    rc=$?; [ $rc = 0 ] || print "failed $f $TAG exit $rc $(date -u +%TZ)" >> $LOG
  done
  print "pass $TAG done $(date -u +%TZ) MB $(du -sm run/$TAG | cut -f1)" >> $LOG
done
print "done $(date -u +%FT%TZ)" >> $LOG
