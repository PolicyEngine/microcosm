#!/bin/bash
# Build J STEP 1 — rebuild the current source-staged base pool.
# The branch now restores CPS/ACS housing inputs in the reusable base builder,
# so Build F's pre-housing base and any pre-prior-year-income base are stale.
# Cache reuse is allowed only when every newly restored persisted leaf carries
# signal.
#
# Compute discipline: ~12 min (Build F was 12m14s), one chunk (<30 min). Detached via
# setsid+caffeinate; pidfile + real rc + append log + memory-pressure sampler.
# NO killing watchdog (six jetsam kills + healthy-run kills taught this — sample only).
# STAGING/LOCAL ONLY.
set -u
RT=/Users/maxghenis/PolicyEngine/_buildj-runtime
WT=/Users/maxghenis/PolicyEngine/_worktrees/populace-build-j-recert
USD=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage
BIN=/Users/maxghenis/PolicyEngine/_buildf-runtime/inputs   # canonical base inputs (Build F set)
AGING_FACTS="$BIN/consumer_facts_builde_aging_v5.jsonl"     # a5d34d4a
CDX="$BIN/congressional_district_vintage_crosswalk.csv"     # 383a6666
BLADDER="$BIN/us_block_ladder_2020.npz"                     # 7ba39b95
OUTDIR="$RT/out/base-j"
BASE="$OUTDIR/base_populace_us_2024_puf_support.h5"
LOGDIR="$RT/logs/buildj-run"
LOG="$LOGDIR/chain_base.log"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PLOG="$LOGDIR/pressure_base_$TS.log"
mkdir -p "$LOGDIR" "$OUTDIR"
say() { echo "[$(date -u +%FT%TZ)] $1" | tee -a "$LOG"; }
base_has_restored_signal() {
  .venv/bin/python -c '
import sys
import pandas as pd

required = {
    "person": (
        "pre_subsidy_rent",
        "self_employment_income_last_year",
        "previous_year_income_available",
    ),
    "spm_unit": ("receives_housing_assistance", "spm_unit_tenure_type"),
    "household": ("tenure_type",),
}
try:
    with pd.HDFStore(sys.argv[1], mode="r") as store:
        tables = {
            entity: store.select(entity, columns=list(names))
            for entity, names in required.items()
        }
except (KeyError, TypeError, ValueError):
    raise SystemExit(1)
ok = all(
    name in tables[entity] and tables[entity][name].nunique(dropna=False) > 1
    for entity, names in required.items()
    for name in names
)
raise SystemExit(0 if ok else 1)
' "$1"
}
rm -f "$LOGDIR/base.rc"

cd "$WT" || { say "FATAL cannot cd $WT"; echo 2 > "$LOGDIR/base.rc"; exit 2; }
source .venv/bin/activate 2>/dev/null

# ---- integrity preflight on inputs (reproduce Build F's exact base) ----
chk() { local got exp; got=$(shasum -a 256 "$1" | cut -c1-16); [ "$got" = "$2" ] || { say "FATAL sha mismatch $1: $got != $2"; echo 2 > "$LOGDIR/base.rc"; exit 2; }; }
for f in "$USD/census_cps_2024.h5" "$USD/census_cps_2023.h5" "$USD/census_cps_2022.h5" "$USD/puf_2024.h5" "$USD/acs_2022.h5" "$AGING_FACTS" "$CDX" "$BLADDER"; do
  [ -f "$f" ] || { say "FATAL missing input $f"; echo 2 > "$LOGDIR/base.rc"; exit 2; }
done
chk "$AGING_FACTS" a5d34d4aad325d8c
chk "$CDX" 383a666631aafd4f
chk "$BLADDER" 7ba39b959068181b
chk "$USD/acs_2022.h5" 0b319b496f19a691

SHORT=$(git rev-parse --short HEAD)
PEUS=$(.venv/bin/python -c "from importlib.metadata import version; print(version('policyengine-us'))" 2>/dev/null)
say "BUILD J BASE START commit=$SHORT pe-us=$PEUS pid=$$ pressure_log=$PLOG"
say "  inputs: asec 2024/2023/2022 + puf_2024 + acs_2022; aging-facts a5d34d4a; cdx 383a6666; bladder 7ba39b95; seed 0 n-est 32"

if [ -f "$BASE" ] && base_has_restored_signal "$BASE"; then
  say "BASE: already present with nondefault restored input surface at $BASE — reusing"
else
  if [ -f "$BASE" ]; then
    say "BASE: cached artifact is stale or restored-input-default-only — rebuilding in place"
  fi
  # ---- memory-pressure sampler (self-terminates on python exit; SAMPLE ONLY, no kill) ----
  ( echo "ts_utc,free_pct,swap_used_mb,py_rss_mb,py_pid"
    while :; do
      fp=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2);print $2}')
      sw=$(sysctl -n vm.swapusage 2>/dev/null | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
      pid=$(pgrep -f 'build_us_puf_support_base.py' | head -1)
      rss=""; if [ -n "$pid" ]; then rss=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print int($1/1024)}'); fi
      echo "$(date -u +%FT%TZ),${fp:-},${sw:-},${rss:-},${pid:-}"
      [ -z "$pid" ] && sleep 4 && pid2=$(pgrep -f 'build_us_puf_support_base.py' | head -1) && [ -z "$pid2" ] && break
      sleep 15
    done ) >> "$PLOG" 2>&1 &
  SAMPLER=$!

  say "BASE: rebuilding (3-year ASEC pool 2024+2023+2022, pinned ACS 2022 rent donor, 2x PUF clone, spec-less equal thirds)"
  .venv/bin/python tools/build_us_puf_support_base.py \
    --asec-h5 2024="$USD/census_cps_2024.h5" \
    --asec-h5 2023="$USD/census_cps_2023.h5" \
    --asec-h5 2022="$USD/census_cps_2022.h5" \
    --puf-h5 "$USD/puf_2024.h5" \
    --acs-h5 "$USD/acs_2022.h5" \
    --target-year 2024 \
    --seed 0 --n-estimators 32 \
    --ledger-facts "$AGING_FACTS" \
    --assign-congressional-districts \
    --congressional-district-vintage-crosswalk "$CDX" \
    --congressional-district-seed 0 \
    --block-ladder-artifact "$BLADDER" \
    --out "$OUTDIR" \
    >> "$LOGDIR/base_j.log" 2>&1
  rc=$?
  kill "$SAMPLER" 2>/dev/null
  say "BASE: python exited rc=$rc"
  if [ $rc -ne 0 ]; then echo "$rc" > "$LOGDIR/base.rc"; say "BASE FAILED rc=$rc — diagnose in base_j.log"; exit "$rc"; fi
fi

if ! base_has_restored_signal "$BASE"; then
  echo 2 > "$LOGDIR/base.rc"
  say "BASE FAILED: output does not persist all restored nondefault inputs"
  exit 2
fi

BASE_SHA=$(shasum -a 256 "$BASE" | cut -d' ' -f1)
echo "$BASE_SHA" > "$LOGDIR/base.sha"
say "BASE sha: $BASE_SHA"
if [ "$BASE_SHA" = "18833fb68e60ee74461608d81a5c5ab7d52435e17026d9e3b062d9de18d6871f" ]; then
  echo 2 > "$LOGDIR/base.rc"
  say "BASE FAILED: Build F's pre-housing SHA was reused"
  exit 2
else
  say "BASE sha differs from pre-housing Build F as required"
fi
echo "0" > "$LOGDIR/base.rc"
say "BUILD J BASE DONE rc=0 sha=$BASE_SHA"
