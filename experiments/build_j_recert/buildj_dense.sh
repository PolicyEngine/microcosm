#!/bin/bash
# Build J DENSE arm — STEP D1: materialize the target-frame checkpoint (WITH the
# scf_wealth asset stage + SNAP source stages) and attempt the dense solve/gates/H5.
# Mirrors buildh_dense.sh EXACTLY; the ONLY additions vs Build H: --scf-summary-extract
# (the pre-fetched, sha-pinned rscfp2022.dta so assets flow, #368/#373) and a FRESH
# --checkpoint-root buildj-dense (the scf_wealth source stage busts the Build H frame
# checkpoint). base 18833fb6 / v8 facts 94b7155f / ref c2065b64 / registry d71c59514e3a
# all unchanged. --dense-default-dataset -> no L0, warm-start-compatible. Detached,
# pressure-sampled, NO killing watchdog. If jetsam/session kills the SOLVE, the atomic
# frame checkpoint survives -> Step D2 direct-solve loads it (bounded). STAGING/LOCAL ONLY.
set -u
RT=/Users/maxghenis/PolicyEngine/_buildj-runtime
WT=/Users/maxghenis/PolicyEngine/_worktrees/populace-build-j-recert
BASE="$RT/out/base-j/base_populace_us_2024_puf_support.h5"
REF=/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5   # c2065b64
FACTS=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildh_v8.jsonl
SCF="$RT/inputs/scf_cache/rscfp2022.dta"
LOGDIR="$RT/logs/buildj-run"
OUT="$RT/out/buildj-run/dense"
LOG="$LOGDIR/chain_dense.log"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PLOG="$LOGDIR/pressure_dense_$TS.log"
mkdir -p "$LOGDIR" "$OUT"
say() { echo "[$(date -u +%FT%TZ)] $1" | tee -a "$LOG"; }
rm -f "$LOGDIR/dense.rc"

cd "$WT" || { say "FATAL cannot cd $WT"; echo 2 > "$LOGDIR/dense.rc"; exit 2; }
source .venv/bin/activate 2>/dev/null

# ---- integrity preflight ----
[ -f "$BASE" ] || { say "FATAL base missing $BASE"; echo 2 > "$LOGDIR/dense.rc"; exit 2; }
BASE_SHA=$(shasum -a 256 "$BASE" | cut -d' ' -f1)
REF_SHA12=$(shasum -a 256 "$REF" | cut -c1-12)
FACTS_SHA=$(shasum -a 256 "$FACTS" | cut -d' ' -f1)
SCF_SHA12=$(shasum -a 256 "$SCF" | cut -c1-12)
SHORT=$(git rev-parse --short HEAD)
PEUS=$(.venv/bin/python -c "from importlib.metadata import version; print(version('policyengine-us'))" 2>/dev/null)
say "BUILD J DENSE(D1) START commit=$SHORT pe-us=$PEUS pid=$$ pressure_log=$PLOG"
say "  base sha:  $BASE_SHA"
say "  ref  sha:  ${REF_SHA12}…"
say "  facts sha: $FACTS_SHA (v8)"
say "  scf  sha:  ${SCF_SHA12}… (rscfp2022.dta member)"
if [ "$BASE_SHA" != "18833fb68e60ee74461608d81a5c5ab7d52435e17026d9e3b062d9de18d6871f" ]; then
  say "WARN base sha != 18833fb6 (got $BASE_SHA) — Build J base differs; proceeding (finding, not fatal) but note in verdict"
fi
if [ "$REF_SHA12" != "c2065b642ab0" ]; then say "FATAL ref sha prefix mismatch: $REF_SHA12"; echo 2 > "$LOGDIR/dense.rc"; exit 2; fi
if [ "$FACTS_SHA" != "94b7155f7ca9e2de32ddb3a0add2fff2d8c66e73147fe5bd112cff3ba69b1669" ]; then
  say "FATAL v8 facts sha mismatch"; echo 2 > "$LOGDIR/dense.rc"; exit 2; fi
if [ "$SCF_SHA12" != "6b8dd2d935a7" ]; then say "FATAL scf member sha mismatch: $SCF_SHA12"; echo 2 > "$LOGDIR/dense.rc"; exit 2; fi

RID="populace-us-2024-buildj-dense-$SHORT-$TS"
say "RELEASE(dense): starting id=$RID (full base; scf_wealth ON; fresh buildj-dense checkpoint)"

# ---- memory-pressure sampler (SAMPLE ONLY; self-terminates on python exit; NO kill) ----
( echo "ts_utc,free_pct,swap_used_mb,py_rss_mb,py_pid"
  while :; do
    fp=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2);print $2}')
    sw=$(sysctl -n vm.swapusage 2>/dev/null | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
    pid=$(pgrep -f 'build_us_fiscal_refresh_release.py' | head -1)
    rss=""; if [ -n "$pid" ]; then rss=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{print int($1/1024)}'); fi
    echo "$(date -u +%FT%TZ),${fp:-},${sw:-},${rss:-},${pid:-}"
    [ -z "$pid" ] && sleep 4 && pid2=$(pgrep -f 'build_us_fiscal_refresh_release.py' | head -1) && [ -z "$pid2" ] && break
    sleep 15
  done ) >> "$PLOG" 2>&1 &
SAMPLER=$!

.venv/bin/python tools/build_us_fiscal_refresh_release.py \
  --base-h5 "$BASE" \
  --ledger-facts "$FACTS" \
  --ledger-facts-sha256 "$FACTS_SHA" \
  --export-input-mass-reference-h5 "$REF" \
  --scf-summary-extract "$SCF" \
  --out "$OUT" \
  --release-id "$RID" \
  --checkpoint-root "$RT/checkpoints/buildj-dense" \
  --seed 0 \
  --dense-default-dataset \
  --skip-reform-validation \
  --no-staging \
  >> "$LOGDIR/release_dense.log" 2>&1
rc=$?
kill "$SAMPLER" 2>/dev/null
echo "$rc" > "$LOGDIR/dense.rc"
echo "$RID" > "$LOGDIR/dense.release_id"
say "RELEASE(dense): exited rc=$rc id=$RID"
CKPT="$RT/checkpoints/buildj-dense/target_frame_checkpoint.h5"
if [ -f "$CKPT" ]; then say "frame checkpoint PRESENT ($(shasum -a256 "$CKPT"|cut -c1-12)…, $(ls -la "$CKPT"|awk '{print $5}') bytes) — D2 direct-solve can load it"; else say "frame checkpoint ABSENT — materialization did not reach checkpoint"; fi
say "BUILD J DENSE(D1) DONE rc=$rc"
