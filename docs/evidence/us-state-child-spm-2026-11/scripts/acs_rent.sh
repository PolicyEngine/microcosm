#!/bin/zsh
SP=${0:a:h}; cd $SP
while [ $(ls run/acs | wc -l) -lt 51 ]; do /bin/sleep 30; done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 env/.venv/bin/python acs_rent_cut.py 2>&1 | tail -3 | cut -c1-200
ls -Sr run/acsrent_cuts | sed 's/.pkl//' | ACS_TAG=acsrent OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 xargs -P 3 -I{} sh -c 'env/.venv/bin/python acs_one.py {} >> run/acsrent.log 2>&1 && [ -f run/acsrent/{}.parquet ] && rm -f run/acsrent_cuts/{}.pkl'
echo done $(ls run/acsrent | wc -l)
