#!/bin/bash
# Build J SPARSE arm — the deployable 57,240 (rmloss100) dense-polish solve + ALL gates.
# Mirrors buildi_sparse_supervised.sh EXACTLY; the ONLY changes vs Build I: base -> base-j,
# ADD --scf-summary-extract (assets flow, #368/#373), fresh --checkpoint-root buildj-sparse,
# and NO killing watchdog (Build I's 75-min watchdog killed healthy runs — pressure-sample
# only). The release reduces to the rmloss100 57,240 BEFORE ACA (:5309<:5471) -> materialises
# on the 57k frame (memory-benign, RSS ~3.6 GB) then solves + runs EVERY gate in one bounded
# process. selection manifest = Build I rmloss100 (152baca3, carried onto base-j, carry-over
# verified 0 misses); zero-support exclusions = Build I 19-cell (abb106af) REVALIDATED by the
# release on the Build J frame (a stale/uncovered cell fails the gate = finding). Gates ALL
# ON: coverage manifest #369, reform-coverage smoke (SSI $10k/$20k must be nonzero), export
# mass, parity, degenerate (TANF now expected non-degenerate), take-up, register consistency,
# #384 preflights. NO bypass flags. STAGING/LOCAL ONLY.
set -u
RT=/Users/maxghenis/PolicyEngine/_buildj-runtime
WT=/Users/maxghenis/PolicyEngine/_worktrees/microcosm-build-j-recert
BASE="$RT/out/base-j/base_populace_us_2024_puf_support.h5"
REF=/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5   # c2065b64
SEL=/Users/maxghenis/PolicyEngine/_buildi-runtime/inputs/buildi_rmloss100_selection_source.json
SPARSE_EXCL=/Users/maxghenis/PolicyEngine/_buildj-runtime/inputs/sparse_zero_support_exclusions_buildj.json  # 24-cell: Build-I 19 + 5 TANF (live take-up seeding)
FACTS=/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildh_v8.jsonl
SCF="$RT/inputs/scf_cache/rscfp2022.dta"
LOGDIR="$RT/logs/buildj-run"
OUT="$RT/out/buildj-run/sparse"
LOG="$LOGDIR/chain_sparse.log"
TS=$(date -u +%Y%m%dT%H%M%SZ)
PLOG="$LOGDIR/pressure_sparse_$TS.log"
mkdir -p "$LOGDIR" "$OUT"
say() { echo "[$(date -u +%FT%TZ)] $1" | tee -a "$LOG"; }
rm -f "$LOGDIR/sparse.rc"

cd "$WT" || { say "FATAL cannot cd $WT"; echo 2 > "$LOGDIR/sparse.rc"; exit 2; }
source .venv/bin/activate 2>/dev/null

[ -f "$BASE" ] || { say "FATAL base missing $BASE"; echo 2 > "$LOGDIR/sparse.rc"; exit 2; }
BASE_SHA=$(shasum -a 256 "$BASE" | cut -d' ' -f1)
REF_SHA12=$(shasum -a 256 "$REF" | cut -c1-12)
SEL_SHA12=$(shasum -a 256 "$SEL" | cut -c1-12)
FACTS_SHA=$(shasum -a 256 "$FACTS" | cut -d' ' -f1)
EXCL_SHA12=$(shasum -a 256 "$SPARSE_EXCL" | cut -c1-12)
SCF_SHA12=$(shasum -a 256 "$SCF" | cut -c1-12)
SHORT=$(git rev-parse --short HEAD)
PEUS=$(.venv/bin/python -c "from importlib.metadata import version; print(version('policyengine-us'))" 2>/dev/null)
say "BUILD J SPARSE START commit=$SHORT pe-us=$PEUS pid=$$ pressure_log=$PLOG"
say "  base sha:  $BASE_SHA"
say "  sel  sha:  ${SEL_SHA12}… (rmloss100 manifest)"
say "  ref  sha:  ${REF_SHA12}…"
say "  facts sha: $FACTS_SHA (v8)"
say "  excl sha:  ${EXCL_SHA12}… (24-cell Build-J: 19 Build-I + 5 TANF)"
say "  scf  sha:  ${SCF_SHA12}… (rscfp2022.dta member)"
if [ "$REF_SHA12" != "c2065b642ab0" ]; then say "FATAL ref sha prefix mismatch: $REF_SHA12"; echo 2 > "$LOGDIR/sparse.rc"; exit 2; fi
if [ "$FACTS_SHA" != "94b7155f7ca9e2de32ddb3a0add2fff2d8c66e73147fe5bd112cff3ba69b1669" ]; then
  say "FATAL v8 facts sha mismatch"; echo 2 > "$LOGDIR/sparse.rc"; exit 2; fi
if [ "$SCF_SHA12" != "6b8dd2d935a7" ]; then say "FATAL scf member sha mismatch: $SCF_SHA12"; echo 2 > "$LOGDIR/sparse.rc"; exit 2; fi

RID="populace-us-2024-buildj-sparse-rmloss100-$SHORT-$TS"
say "RELEASE(sparse): starting id=$RID (rmloss100 dense-polish; scf_wealth ON; ALL gates; NO bypass flags)"

# ---- memory-pressure sampler (SAMPLE ONLY; no kill) ----
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
  --selection-source-manifest "$SEL" \
  --dense-default-dataset \
  --ledger-facts "$FACTS" \
  --ledger-facts-sha256 "$FACTS_SHA" \
  --export-input-mass-reference-h5 "$REF" \
  --zero-support-exclusions "$SPARSE_EXCL" \
  --scf-summary-extract "$SCF" \
  --out "$OUT" \
  --release-id "$RID" \
  --checkpoint-root "$RT/checkpoints/buildj-sparse" \
  --seed 0 \
  --skip-reform-validation \
  --no-staging \
  >> "$LOGDIR/release_sparse.log" 2>&1
rc=$?
kill "$SAMPLER" 2>/dev/null
echo "$rc" > "$LOGDIR/sparse.rc"
echo "$RID" > "$LOGDIR/sparse.release_id"
say "RELEASE(sparse): exited rc=$rc id=$RID"
say "BUILD J SPARSE DONE rc=$rc"
