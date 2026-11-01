#!/bin/zsh
# Cut the remaining 36 ACS-local states, then simulate them two at a time, smallest first, and drop each cut when its state finishes.
SP=${0:a:h}; cd $SP
S=~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots
ALL=(1 2 4 5 6 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56)
REST=()
for f in $ALL; do n=$(printf %02d $f); [ -f run/acs/$n.parquet ] || REST+=$f; done
ONLY_FIPS=${(j:,:)REST} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/.venv/bin/python - <<PY 2>&1 | tail -2 | cut -c1-200
import sys; sys.path.insert(0,"$SP"); sys.argv=["x"]
import cutrun, pathlib
d=pathlib.Path("$SP/run/acs_cuts"); d.mkdir(exist_ok=True,parents=True)
print(cutrun.cut("$S/725d89993f6b2dc4821fd5c3620f0f3d4d630102/populace_us_2024_acs_local.h5", d))
PY
ls -Sr run/acs_cuts | sed 's/.pkl//' | OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 xargs -P 2 -I{} sh -c 'env/.venv/bin/python acs_one.py {} >> run/acs.log 2>&1 && [ -f run/acs/{}.parquet ] && rm -f run/acs_cuts/{}.pkl'
echo done $(ls run/acs | wc -l)
