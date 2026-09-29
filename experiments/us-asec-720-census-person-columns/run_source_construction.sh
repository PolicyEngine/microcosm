#!/bin/zsh
# Source-construction stage only (issue #720 fix verification), route A's inputs.
set -euo pipefail
WT=/Users/maxghenis/PolicyEngine/_worktrees/us-asec-720-coverage-recodes
OUT=/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/asec-720-fix
STO=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage
ARC=/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education
cd "$WT"
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
/usr/bin/time -l "$WT/.venv/bin/python" -B tools/build_us_puf_support_base.py \
  --stage source_construction \
  --checkpoint-dir "$OUT/checkpoints" \
  --asec-h5 2024=$STO/census_cps_2024.h5 \
  --asec-h5 2023=$STO/census_cps_2023.h5 \
  --asec-h5 2022=$STO/census_cps_2022.h5 \
  --asec-h5-sha256 2024=ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d \
  --asec-h5-sha256 2023=cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88 \
  --asec-h5-sha256 2022=7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e \
  --puf-h5 $STO/puf_2024.h5 \
  --puf-source-year-csv $STO/puf_2015.csv \
  --acs-h5 $STO/acs_2022.h5 \
  --asec-education-source 2022=$ARC/asecpub23csv.zip \
  --asec-education-source 2023=$ARC/asecpub24csv.zip \
  --asec-education-source 2024=$ARC/asecpub25csv.zip \
  --target-year 2024 --seed 0 --n-estimators 32 \
  --ledger-facts /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl \
  --assign-congressional-districts \
  --congressional-district-vintage-crosswalk "$WT/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv" \
  --congressional-district-seed 0 \
  --block-ladder-artifact /Users/maxghenis/PolicyEngine/_buildf-runtime/inputs/us_block_ladder_2020.npz \
  --out "$OUT/base-out"
