#!/bin/bash
# Route A (microcosm epic #956, acceleration A): a from-scratch US base, then a
# --base-h5 release candidate on the committed labelled Ledger feed, then the
# release-gate preflight and the publisher's offline --preflight-only check.
#
# Builds a release candidate, runs offline preflights, and leaves publication
# to Max. Release staging telemetry is enabled unless ROUTE_A_STAGING=0.
#
# Usage: ./route_a.sh [--resolve]
# --resolve fetches refs, hashes inputs, and checks commit pins; no stages run.
# Settings come from ROUTE_A_ENV, defaulting to route_a.env beside this script.
# Accepted stages are skipped on restart. Live supervisors are waited for;
# failed attempts are moved aside. Releases use fresh IDs and shared caches.
set +x
set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROUTE_A_ENV=${ROUTE_A_ENV:-$SCRIPT_DIR/route_a.env}
SUP=$SCRIPT_DIR/supervise.py
SAMPLER=$SCRIPT_DIR/sample_series.py
TOKEN_WRAPPER=$SCRIPT_DIR/with_hf_token.sh
GIB=1073741824

MODE=run
case "${1:-}" in
  --resolve|--dry-run) MODE=resolve ;;
  "") ;;
  -h|--help) sed -n '2,/^set +x$/p' "$0"; exit 0 ;;
  *) echo "usage: $0 [--resolve]" >&2; exit 64 ;;
esac

[ -f "$ROUTE_A_ENV" ] || { echo "missing settings: $ROUTE_A_ENV (copy route_a.env.example)" >&2; exit 64; }
# shellcheck source=/dev/null
. "$ROUTE_A_ENV"
: "${PE:?set PE in route_a.env}" "${MAIN:?set MAIN}" "${WT_ROOT:?set WT_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}" "${CHAIN_ROOT:?set CHAIN_ROOT}" "${CHAIN_LOG:?set CHAIN_LOG}"
: "${DISK_PATH:?set DISK_PATH}" "${UV:?set UV}" "${PYTHON:?set PYTHON}"
: "${STORAGE:?set STORAGE}" "${EDU:?set EDU}" "${FEED:?set FEED}"
: "${LADDER:?set LADDER}" "${SSI:?set SSI}" "${SCF:?set SCF}"
: "${AGENT_SECRET:?set AGENT_SECRET}" "${ROUTE_A_STAGING=1}" "${RELEASE_RAM_GIB:=50}"
case "$ROUTE_A_STAGING" in 0|1) ;; *) echo "ROUTE_A_STAGING must be 0 or 1" >&2; exit 64 ;; esac
mkdir -p "$RUN_ROOT"
LOG=$RUN_ROOT/route_a.log
: "${COMMIT:=}" "${REQUIRE_ON_MAIN:=1}" "${REQUIRE_SPM_ROLE:=1}" "${PREFLIGHT_COMMIT:=}"
: "${PREFLIGHT_NEW_LINEAGE_ARGS:=}" "${EARLY_START_AFTER_HEAVY:=0}" "${PRUNE_BASE_CHECKPOINTS:=0}"
: "${RELEASE_DESPITE_PREFLIGHT_FAIL:=0}" "${RELEASE_EXTRA_ARGS:=}" "${EXPORT_MASS_REF_H5:=}"
: "${RESOURCE_WAIT_HOURS:=12}"

# ---------------------------------------------------------------------------
# Inputs, resolved and hashed on this machine on 2026-09-23 (see status.md).
# Format: role|path|sha256|bytes|where main pins it
# ---------------------------------------------------------------------------
FEED_SHA=a81cbcc504caa71e4eb23b4d70a11528c8e6db6523e41283f8482569a23df04c
# The base stage reads the feed only for the SOI congressional-district return
# counts. On 2026-09-23 the pinned feed and the 2026-09-16 base feed
# (consumer_facts_builde_aging_v5.jsonl, a5d34d4a...) gave the same 436-row
# distribution through main's congressional_district_distribution_from_ledger_facts;
# after the vintage crosswalk, 73 weights differ by at most 2.2e-16 relative.
# The pinned feed is used for both stages so the lineage has one feed identity.
# The feed pinned since the W-2 relabel (Chronicle f98acf4) differs from that
# one only in its Form W-2 item rows, so every district row is the same bytes.
BASE_LEDGER_FACTS=${BASE_LEDGER_FACTS:-$FEED}
BASE_LEDGER_FACTS_SHA=${BASE_LEDGER_FACTS_SHA:-$FEED_SHA}
# d713 (Max 2026-10-03): the next certified build uses the Connecticut-fixed ladder from microcosm#1072
# (CT blocks get planning-region CBSAs; only cbsa_code differs from 7ba39b95). The published 4b57d15a2 release used 7ba39b95.
# QRF tail-concentration exclusion register. Empty by default since 2026-09-23:
# the Build P register was measured on another lineage and cannot match a
# whole-base export (route A pre-mortem); a route-A register must be measured
# on the export that ships. Set QRF_TAIL_EXCLUSIONS to pass one.
TAIL=${QRF_TAIL_EXCLUSIONS-}
SSI_SHA=25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a
ASEC_2024_SHA=ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d
ASEC_2023_SHA=cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88
ASEC_2022_SHA=7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e
CROSSWALK_REL=packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv
CROSSWALK_SHA=c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec
FEED_PIN_REL=packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json
ASEC_PIN_REL=packages/microcosm-build/src/microcosm/build/us_runtime/asec_sources.py
SPM_ROLE_REL=packages/microcosm-build/src/microcosm/build/us_runtime/spm_independence_role.py
BASE_FLAGS="--stage --checkpoint-dir --asec-h5 --asec-h5-sha256 --puf-h5 --puf-source-year-csv --acs-h5 --asec-education-source --target-year --seed --n-estimators --ledger-facts --assign-congressional-districts --congressional-district-vintage-crosswalk --congressional-district-seed --block-ladder-artifact --out"
RELEASE_FLAGS="--base-h5 --ledger-facts --ledger-facts-sha256 --qrf-tail-concentration-exclusions --ssi-take-up-prior-weight-basis --ssi-take-up-prior-weight-basis-sha256 --scf-summary-extract --out --release-id --checkpoint-root --seed"
[ "$ROUTE_A_STAGING" = 0 ] && RELEASE_FLAGS="$RELEASE_FLAGS --no-staging"
for extra in $RELEASE_EXTRA_ARGS; do
  case "$extra" in --*) RELEASE_FLAGS="$RELEASE_FLAGS ${extra%%=*}" ;; esac
done
PREFLIGHT_FLAGS="--base-h5 --ledger-facts --ledger-facts-sha256 --release-manifest --json-out"

input_rows() {
  cat <<EOF
asec_2024_h5|$STORAGE/census_cps_2024.h5|$ASEC_2024_SHA|323994739|us_runtime/asec_sources.py ASEC_SOURCE_ARTIFACTS
asec_2023_h5|$STORAGE/census_cps_2023.h5|$ASEC_2023_SHA|299036610|us_runtime/asec_sources.py ASEC_SOURCE_ARTIFACTS
asec_2022_h5|$STORAGE/census_cps_2022.h5|$ASEC_2022_SHA|301129278|us_runtime/asec_sources.py ASEC_SOURCE_ARTIFACTS
puf_2024_h5|$STORAGE/puf_2024.h5|7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df|316939164|us/spec/sources.yaml
puf_2015_csv|$STORAGE/puf_2015.csv|0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df|126034649|us/spec/sources.yaml
acs_2022_h5|$STORAGE/acs_2022.h5|0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4|472220686|us/source_stages.json, us_runtime/housing_inputs.py
asec_education_2022_zip|$EDU/asecpub23csv.zip|d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11|150165063|us_runtime/education_assistance_source.py
asec_education_2023_zip|$EDU/asecpub24csv.zip|cdb39cdac34bef99dd0940ab28e306f692404c2eea44d85dfd634214872a0a09|148664101|us_runtime/education_assistance_source.py
asec_education_2024_zip|$EDU/asecpub25csv.zip|318845a2b5e0034eb2973898de1738f4df0025727de38499e7669cb9c0deef0b|147271429|us_runtime/education_assistance_source.py
base_ledger_facts|$BASE_LEDGER_FACTS|$BASE_LEDGER_FACTS_SHA|-|us/chronicle_feed.json when it is the pinned feed
block_ladder_npz|$LADDER|6840b990acdfa2003d7723da5595e3cfff7205a4fa3d8d6206ed835c85c3f233|18991218|d713: Connecticut-fixed ladder rebuilt at microcosm#1072 (af98853e); _build_artifacts/us-ct-cbsa/verify_vs_route_a.json
release_ledger_facts|$FEED|$FEED_SHA|164591409|us/chronicle_feed.json facts_sha256 (Chronicle f98acf4 = c5e5bf8 plus the W-2 items' TY2020 labels, bare feed, 39,155 rows)
${TAIL:+qrf_tail_exclusions|$TAIL|${QRF_TAIL_EXCLUSIONS_SHA:-9bd497dc0d03dd793b53a228979e99408e62fdd2430bdd0d04df5fc7576e6229}|-|operator-supplied register; the release records its sha in diagnostics}
ssi_take_up_prior_basis|$SSI|$SSI_SHA|4782|--ssi-take-up-prior-weight-basis-sha256
scf_summary_extract|$SCF|6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1|24904185|us/source_stages.json, us_runtime/scf_wealth.py
EOF
}

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
fail() {
  log "FAIL: $*"
  echo "$(date '+%F %T') $*" > "$RUN_ROOT/ROUTE_A_FAILED"
  exit 1
}
sha_of() { openssl dgst -sha256 -r "$1" 2>/dev/null | cut -c1-64; }
free_bytes() { df -k "$DISK_PATH" | awk 'NR==2 {print $4*1024}'; }

INPUT_PROBLEMS=()
verify_inputs() {  # fills INPUT_PROBLEMS; writes $RUN_ROOT/inputs.resolved.json
  INPUT_PROBLEMS=()
  local json=$RUN_ROOT/inputs.resolved.json.tmp first=1 role path sha size pin actual asize status
  printf '{\n "resolved_at": "%s",\n "host": "%s",\n "inputs": [\n' "$(date '+%F %T %Z')" "$(hostname -s)" > "$json"
  while IFS='|' read -r role path sha size pin; do
    [ -n "$role" ] || continue
    status=ok; actual=; asize=
    if [ ! -f "$path" ]; then
      status=missing; INPUT_PROBLEMS+=("$role missing: $path")
    else
      asize=$(stat -f %z "$path")
      if [ "$size" != "-" ] && [ "$asize" != "$size" ]; then
        status=size_mismatch; INPUT_PROBLEMS+=("$role size $asize != $size: $path")
      fi
      actual=$(sha_of "$path")
      if [ "$actual" != "$sha" ]; then
        status=sha_mismatch; INPUT_PROBLEMS+=("$role sha256 $actual != $sha: $path")
      fi
    fi
    [ $first = 1 ] || printf ',\n' >> "$json"
    first=0
    printf '  {"role": "%s", "path": "%s", "expected_sha256": "%s", "observed_sha256": "%s", "bytes": "%s", "status": "%s", "pinned_by": "%s"}' \
      "$role" "$path" "$sha" "$actual" "$asize" "$status" "$pin" >> "$json"
  done < <(input_rows)
  printf '\n ]\n}\n' >> "$json"
  mv "$json" "$RUN_ROOT/inputs.resolved.json"
}

COMMIT_BLOCKERS=()
COMMIT_NOTES=()
check_commit() {  # $1 = build ref, $2 = preflight ref; fills COMMIT_BLOCKERS/COMMIT_NOTES
  local ref=$1 pref=$2 full branches tok missing=
  COMMIT_BLOCKERS=(); COMMIT_NOTES=()
  full=$(git -C "$MAIN" rev-parse --verify --quiet "$ref^{commit}") || {
    COMMIT_BLOCKERS+=("build commit $ref does not exist locally after git fetch origin"); return; }
  branches=$(git -C "$MAIN" branch -r --contains "$full" 2>/dev/null | grep -v -- '->' | tr -d ' ' | tr '\n' ' ')
  if [ -z "$branches" ]; then
    COMMIT_BLOCKERS+=("build commit $full is on no origin branch (not pushed)")
  else
    COMMIT_NOTES+=("build commit $full is on: $branches")
  fi
  if ! git -C "$MAIN" merge-base --is-ancestor "$full" origin/main; then
    if [ "$REQUIRE_ON_MAIN" = 1 ]; then
      COMMIT_BLOCKERS+=("build commit $full is not reachable from origin/main (set REQUIRE_ON_MAIN=0 only for a PR-tree receipt run)")
    else
      COMMIT_NOTES+=("build commit $full is not on origin/main (REQUIRE_ON_MAIN=0: PR-tree receipt run)")
    fi
  fi
  if git -C "$MAIN" show "$full:$FEED_PIN_REL" 2>/dev/null | grep -q "\"facts_sha256\": \"$FEED_SHA\""; then
    COMMIT_NOTES+=("us/chronicle_feed.json at $full pins facts_sha256 $FEED_SHA")
  else
    COMMIT_BLOCKERS+=("us/chronicle_feed.json at $full does not pin facts_sha256 $FEED_SHA (the release tool would refuse the feed)")
  fi
  local s
  for s in $ASEC_2024_SHA $ASEC_2023_SHA $ASEC_2022_SHA; do
    git -C "$MAIN" show "$full:$ASEC_PIN_REL" 2>/dev/null | grep -q "$s" \
      || COMMIT_BLOCKERS+=("asec_sources.py at $full lacks canonical ASEC pin $s (the base would refuse the CLI pin)")
  done
  if [ "$(git -C "$MAIN" show "$full:$CROSSWALK_REL" 2>/dev/null | openssl dgst -sha256 -r | cut -c1-64)" = "$CROSSWALK_SHA" ]; then
    COMMIT_NOTES+=("packaged CD vintage crosswalk at $full = $CROSSWALK_SHA")
  else
    COMMIT_BLOCKERS+=("packaged CD vintage crosswalk at $full is not $CROSSWALK_SHA")
  fi
  if git -C "$MAIN" show "$full:tools/build_us_fiscal_refresh_release.py" 2>/dev/null | grep -q '_check_committed_us_ledger_feed_pin'; then
    COMMIT_NOTES+=("release tool at $full holds the feed to the committed pin")
  else
    COMMIT_BLOCKERS+=("release tool at $full has no committed-feed-pin check (#955 missing)")
  fi
  # Every flag the driver passes must be declared by the tool at that commit.
  local tool flag missing
  tool="tools/build_us_puf_support_base.py:$BASE_FLAGS"
  missing=
  for flag in ${tool#*:}; do
    git -C "$MAIN" show "$full:${tool%%:*}" 2>/dev/null | grep -q -- "\"$flag\"" || missing="$missing $flag"
  done
  if [ -n "$missing" ]; then COMMIT_BLOCKERS+=("${tool%%:*} at $full does not declare:$missing")
  else COMMIT_NOTES+=("${tool%%:*} at $full declares every flag the driver passes"); fi
  # shellcheck disable=SC2086 # flags are a whitespace-separated option list
  if git -C "$MAIN" show "$full:tools/build_us_fiscal_refresh_release.py" | \
      "$PYTHON" "$SCRIPT_DIR/check_flags.py" $RELEASE_FLAGS; then
    COMMIT_NOTES+=("release parser at $full declares every flag the driver passes")
  else
    COMMIT_BLOCKERS+=("release parser at $full does not declare every driver flag")
  fi
  if git -C "$MAIN" cat-file -e "$full:$SPM_ROLE_REL" 2>/dev/null; then
    COMMIT_NOTES+=("SPM independence role stage (#959) present at $full")
  elif [ "$REQUIRE_SPM_ROLE" = 1 ]; then
    COMMIT_BLOCKERS+=("#959 SPM independence role stage absent at $full ($SPM_ROLE_REL); without it the release's reform-coverage smoke refuses (docs/us-release-build-rule.md section 2)")
  else
    COMMIT_NOTES+=("#959 absent at $full; REQUIRE_SPM_ROLE=0")
  fi
  if [ -z "$PREFLIGHT_NEW_LINEAGE_ARGS" ]; then
    COMMIT_NOTES+=("PREFLIGHT_NEW_LINEAGE_ARGS empty: both release-gate preflights will be BLOCKED (tools/preflight_us_release_gates.py requires --selection-source-manifest)")
  else
    local pfull
    pfull=$(git -C "$MAIN" rev-parse --verify --quiet "$pref^{commit}") || {
      COMMIT_BLOCKERS+=("preflight commit $pref does not exist locally"); return; }
    missing=
    for flag in $PREFLIGHT_FLAGS; do
      git -C "$MAIN" show "$pfull:tools/preflight_us_release_gates.py" 2>/dev/null | grep -q -- "\"$flag\"" || missing="$missing $flag"
    done
    [ -z "$missing" ] || COMMIT_BLOCKERS+=("tools/preflight_us_release_gates.py at $pfull does not declare:$missing")
    tok=${PREFLIGHT_NEW_LINEAGE_ARGS%% *}
    if git -C "$MAIN" show "$pfull:tools/preflight_us_release_gates.py" 2>/dev/null | grep -q -- "\"$tok\""; then
      COMMIT_NOTES+=("preflight tool at $pfull declares $tok")
    else
      COMMIT_BLOCKERS+=("preflight tool at $pfull does not declare \"$tok\" (new-lineage PR not in that commit?)")
    fi
    git -C "$MAIN" branch -r --contains "$pfull" 2>/dev/null | grep -qv -- '->' \
      || COMMIT_BLOCKERS+=("preflight commit $pfull is on no origin branch")
  fi
}

refuse_bad_release_args() {
  local a
  for a in $RELEASE_EXTRA_ARGS; do
    case "${a%%=*}" in
      --dense-default-dataset)
        # Max's ruling d122 (2026-09-23): route A's first certified release is
        # calibrated whole-base dense. Allowed only with this explicit opt-in.
        [ "${DENSE_RELEASE_D122:-0}" = 1 ] || fail "RELEASE_EXTRA_ARGS may carry --dense-default-dataset only with DENSE_RELEASE_D122=1 (Max's d122 ruling)" ;;
      --skip-reform-validation|--allow-unpinned-feed|--selection-source-*|\
      --selection-mass-protection|--evidence-release|--evidence-failure-owners|--allow-*|\
      --skip-reform-coverage-smoke|--skip-out-of-sample-reforms|--no-target-materialization-cache|\
      --exact-k*|--pool-manifest*|--staging-*|--no-staging|--release-id|--out|--checkpoint-root|--base-h5|--ledger-facts*)
        fail "RELEASE_EXTRA_ARGS may not carry $a (it would make the run not a release, or it is set by the driver)" ;;
    esac
  done
}

report_resolution() {  # $1 = label of the ref checked
  local x
  log "inputs: ${#INPUT_PROBLEMS[@]} problem(s); see $RUN_ROOT/inputs.resolved.json"
  for x in ${INPUT_PROBLEMS[@]+"${INPUT_PROBLEMS[@]}"}; do log "  INPUT PROBLEM: $x"; done
  for x in ${COMMIT_NOTES[@]+"${COMMIT_NOTES[@]}"}; do log "  commit ($1): $x"; done
  for x in ${COMMIT_BLOCKERS[@]+"${COMMIT_BLOCKERS[@]}"}; do log "  BLOCKER ($1): $x"; done
}

release_command() {
  local -a extra=() staging=() launcher=()
  # shellcheck disable=SC2206 # retain the driver's word-split extra-args contract
  extra=($RELEASE_EXTRA_ARGS)
  case "${ROUTE_A_STAGING-1}" in
  0)
    staging=(--no-staging)
    launcher=(/usr/bin/env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_TOKEN_MAX)
    ;;
  1)
    launcher=("$TOKEN_WRAPPER" "$AGENT_SECRET")
    ;;
  *) fail "ROUTE_A_STAGING must be 0 or 1" ;;
  esac
  RELEASE_ARGV=("${launcher[@]}" "$PY" -B tools/build_us_fiscal_refresh_release.py
    --base-h5 "$BASE_H5"
    --ledger-facts "$FEED" --ledger-facts-sha256 "$FEED_SHA"
    ${TAIL:+--qrf-tail-concentration-exclusions "$TAIL"}
    --ssi-take-up-prior-weight-basis "$SSI" --ssi-take-up-prior-weight-basis-sha256 "$SSI_SHA"
    --scf-summary-extract "$SCF"
    --out "$REL_OUT" --release-id "$RID" --checkpoint-root "$REL_CKPT" --seed 0
    ${staging[@]+"${staging[@]}"} ${extra[@]+"${extra[@]}"})
}

# ---------------------------------------------------------------------------
# Dry run.
# ---------------------------------------------------------------------------
FETCH_OK=1
git -C "$MAIN" fetch origin --quiet || FETCH_OK=0
if [ "$MODE" = resolve ]; then
  LOG=$RUN_ROOT/resolve.log
  REF=${COMMIT:-origin/main}
  PREF=${PREFLIGHT_COMMIT:-$REF}
  [ "$FETCH_OK" = 1 ] || { echo "git fetch origin failed" >&2; exit 1; }
  log "resolve: build ref $REF ($(git -C "$MAIN" rev-parse --short=12 "$REF" 2>/dev/null)), preflight ref $PREF"
  verify_inputs
  check_commit "$REF" "$PREF"
  report_resolution "$REF"
  log "free disk $(( $(free_bytes) / GIB )) GiB; chain: $(grep -E '^[0-9-]{10} [0-9:]{8} ' "$CHAIN_LOG" | tail -1)"
  [ ${#INPUT_PROBLEMS[@]} = 0 ] || exit 2
  exit 0
fi

# ---------------------------------------------------------------------------
# Real run.
# ---------------------------------------------------------------------------
LOCK=$RUN_ROOT/.route_a.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  other=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    echo "another route_a.sh (pid $other) holds $LOCK" >&2; exit 75
  fi
  rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null
  mkdir "$LOCK" || { echo "cannot take $LOCK" >&2; exit 75; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null' EXIT
rm -f "$RUN_ROOT/ROUTE_A_FAILED"

log "route A driver start (pid $$)"
[ "$FETCH_OK" = 1 ] || log "warning: git fetch origin failed; checking pins against local refs"
[ -n "$COMMIT" ] || fail "COMMIT is unset in route_a.env: choose a pushed commit carrying #959 (and the new-lineage preflight PR)"
C=$(git -C "$MAIN" rev-parse --verify --quiet "$COMMIT^{commit}") || fail "COMMIT $COMMIT not found after git fetch origin"
PC=$(git -C "$MAIN" rev-parse --verify --quiet "${PREFLIGHT_COMMIT:-$C}^{commit}") || fail "PREFLIGHT_COMMIT $PREFLIGHT_COMMIT not found"
refuse_bad_release_args
verify_inputs
check_commit "$C" "$PC"
report_resolution "$C"
[ ${#INPUT_PROBLEMS[@]} = 0 ] || fail "input problems (see above)"
[ ${#COMMIT_BLOCKERS[@]} = 0 ] || fail "commit blockers (see above)"

RUN=$RUN_ROOT/run-${C:0:12}
mkdir -p "$RUN"

# --- 1. Clean worktree(s) at pushed commits, with a locked environment. Light
# (git worktree add + uv sync), so it runs before the wait and fails early. ---
ensure_worktree() {  # $1 = full sha; sets WT
  local c=$1
  WT=$WT_ROOT/microcosm-route-a-${c:0:12}
  if [ -d "$WT" ]; then
    [ "$(git -C "$WT" rev-parse HEAD)" = "$c" ] || fail "$WT exists but is not at $c"
  else
    git -C "$MAIN" worktree add --detach "$WT" "$c" >>"$LOG" 2>&1 || fail "git worktree add $WT $c"
    log "created worktree $WT at $c"
  fi
  [ -z "$(git -C "$WT" status --porcelain)" ] || fail "worktree $WT is dirty"
  if [ ! -f "$WT/.venv/.route_a_synced" ]; then
    log "uv sync --all-packages --locked --extra us in $WT"
    (cd "$WT" && "$UV" sync --all-packages --locked --extra us) >>"$LOG" 2>&1 || fail "uv sync in $WT"
    "$WT/.venv/bin/python" -c 'import psutil, policyengine_us' >>"$LOG" 2>&1 || fail "venv in $WT lacks psutil/policyengine_us"
    date '+%F %T' > "$WT/.venv/.route_a_synced"
  fi
  [ -z "$(git -C "$WT" status --porcelain)" ] || fail "worktree $WT dirty after uv sync"
}
ensure_worktree "$C"; W=$WT
ensure_worktree "$PC"; PW=$WT
PY=$W/.venv/bin/python
[ "$(sha_of "$W/$CROSSWALK_REL")" = "$CROSSWALK_SHA" ] || fail "packaged crosswalk digest in $W"
PEUS=$("$PY" -c 'import importlib.metadata as m; print(m.version("policyengine-us"))')
[ -x "$W/.venv/bin/microcosm-publish-release" ] || fail "no microcosm-publish-release in $W/.venv"
log "build worktree $W (policyengine-us $PEUS); preflight worktree $PW"

# --- 2. Wait for the overnight chain to leave the machine. ---
chain_alive() {
  pgrep -f 'overnight_chain\.sh' >/dev/null 2>&1 && return 0
  pgrep -f "supervise\.py $CHAIN_ROOT/run/" >/dev/null 2>&1 && return 0
  return 1
}
chain_terminal_line() {
  grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} (CHAIN DONE|FAIL: )' "$CHAIN_LOG" 2>/dev/null | tail -1
}
waited=0
while chain_alive; do
  if [ "$EARLY_START_AFTER_HEAVY" = 1 ] && grep -q 'local release preflight passed' "$CHAIN_LOG" 2>/dev/null; then
    log "overnight chain still running its certification/publication tail; heavy stages are over (EARLY_START_AFTER_HEAVY=1)"
    break
  fi
  [ $(( waited % 1800 )) = 0 ] && log "waiting for the overnight chain: $(grep -E '^[0-9-]{10} [0-9:]{8} ' "$CHAIN_LOG" | tail -1 | cut -c1-160)"
  sleep 300; waited=$(( waited + 300 ))
done
end_line=$(chain_terminal_line)
if chain_alive; then
  log "overnight chain still alive in its tail; starting early (last terminal line: ${end_line:-none})"
else
  case "$end_line" in
    *"CHAIN DONE"*) log "overnight chain finished: $end_line" ;;
    *"FAIL: "*) log "overnight chain ended in failure; the machine is free, proceeding: $end_line" ;;
    *) log "overnight chain process gone without CHAIN DONE or FAIL; proceeding" ;;
  esac
fi
verify_inputs
[ ${#INPUT_PROBLEMS[@]} = 0 ] || { report_resolution "$C"; fail "inputs changed while waiting"; }

# --- helpers for supervised stages ---
avail_ram_bytes() { "$PY" -c 'import psutil; print(psutil.virtual_memory().available)'; }
wait_resources() {  # $1 RAM GiB, $2 disk GiB, $3 label, $4 disk hint
  local ram=$(( $1 * GIB )) disk=$(( $2 * GIB )) deadline=$(( $(date +%s) + RESOURCE_WAIT_HOURS * 3600 )) n=0
  while [ "$(avail_ram_bytes)" -lt "$ram" ] || [ "$(free_bytes)" -lt "$disk" ]; do
    [ "$(date +%s)" -lt "$deadline" ] || fail "$3: needs $1 GiB available RAM and $2 GiB free disk; have $(( $(avail_ram_bytes) / GIB )) / $(( $(free_bytes) / GIB )) GiB after ${RESOURCE_WAIT_HOURS} h. ${4:-}"
    [ $(( n % 6 )) = 0 ] && log "$3 waiting: RAM $(( $(avail_ram_bytes) / GIB ))/$1 GiB, disk $(( $(free_bytes) / GIB ))/$2 GiB. ${4:-}"
    n=$(( n + 1 )); sleep 300
  done
}

AUTH="Route A for microcosm epic #956 acceleration A, staged 2026-09-23 by an Opus 5.5 subagent; builds a local release candidate only; publication is Max's call. Build commit $C."

# write_config <name> <rss GiB> <ram adm GiB> <disk adm GiB> <wall s> <cpu s> <cwd> <env json> -- argv...
write_config() {
  local name=$1 rss=$2 ram=$3 adm=$4 wall=$5 cpu=$6 cwd=$7 envj=$8; shift 9
  "$PY" - "$RUN/$name-config.json" "$cwd" "$C" "$rss" "$ram" "$adm" "$wall" "$cpu" "$envj" "$AUTH" "$@" <<'EOF'
import json, subprocess, sys
path, cwd, commit, rss, ram, adm, wall, cpu, envj, auth, *argv = sys.argv[1:]
G = 1024 ** 3
head = subprocess.check_output(["git", "-C", cwd, "rev-parse", "HEAD"], text=True).strip()
json.dump({
    "argv": argv,
    "cwd": cwd,
    "env": json.loads(envj),
    "authorization": auth,
    "source_identity": {"worktree": cwd, "head": head, "build_commit": commit},
    "limits": {
        "wall_seconds": int(wall), "cpu_seconds": int(cpu),
        "rss_bytes": int(float(rss) * G),
        # supervise.py counts only its own directory (child log + JSON) here.
        "output_bytes": 2 * G, "log_bytes": 512 * 1024 * 1024,
        "disk_floor_bytes": 20 * G, "disk_floor_persist_seconds": 180,
        "disk_admission_bytes": int(float(adm) * G),
        "available_ram_admission_bytes": int(float(ram) * G),
    },
}, open(path, "w"), indent=1)
EOF
  [ -s "$RUN/$name-config.json" ] || fail "could not write $name config"
}

stage_done() { [ -f "$RUN/$1-sup/ACCEPTED" ]; }
result_accepted() {  # $1 dir, $2 accepted return codes ("0" or "0 2")
  local rc
  [ -f "$1/RESULT.json" ] || return 1
  grep -q '"refusal": null' "$1/RESULT.json" || return 1
  rc=$(sed -n 's/^ *"returncode": \(-\{0,1\}[0-9]*\),*$/\1/p' "$1/RESULT.json")
  case " $2 " in *" $rc "*) echo "$rc"; return 0 ;; esac
  return 1
}
wait_prior_supervisor() {  # $1 name: a killed driver can leave its supervisor running
  while pgrep -f "supervise\.py $RUN/$1-sup " >/dev/null 2>&1; do
    log "$1: a supervisor from an earlier driver is still running it; waiting"; sleep 300
  done
}
run_stage() {  # $1 name, $2 accepted return codes, $3 RAM GiB, $4 disk GiB, $5 disk hint
  local name=$1 accept=$2 dir=$RUN/$1-sup sp smp rc
  local deadline=$(( $(date +%s) + RESOURCE_WAIT_HOURS * 3600 ))
  wait_prior_supervisor "$name"
  if rc=$(result_accepted "$dir" "$accept"); then
    echo "returncode=$rc $(date '+%F %T')" > "$dir/ACCEPTED"; log "$name finished earlier (rc $rc); accepted"; return 0
  fi
  [ -d "$dir" ] && mv "$dir" "$dir.failed-$(date +%s)"
  while :; do
    wait_resources "$3" "$4" "$name" "${5:-}"
    log "$name start"
    "$PY" "$SUP" "$dir" "$RUN/$name-config.json" >>"$LOG" 2>&1 &
    sp=$!
    "$PY" "$SAMPLER" "$dir" "$RUN/$name.series.csv" "$sp" 30 >/dev/null 2>&1 &
    smp=$!
    wait "$sp"
    kill "$smp" 2>/dev/null; wait "$smp" 2>/dev/null
    if grep -q '"status": "REFUSED_ADMISSION"' "$dir/RESULT.json" 2>/dev/null; then
      [ "$(date +%s)" -lt "$deadline" ] || fail "$name refused admission until the deadline: $(tr -d '\n ' < "$dir/ADMISSION.json" | cut -c1-200)"
      log "$name refused admission (a race with another process); retrying in 5 min"
      mv "$dir" "$dir.refused-$(date +%s)"; sleep 300; continue
    fi
    break
  done
  if rc=$(result_accepted "$dir" "$accept"); then
    echo "returncode=$rc $(date '+%F %T')" > "$dir/ACCEPTED"
    log "$name ACCEPTED (rc $rc) $(tr -d '\n ' < "$dir/RESULT.json" | cut -c1-220)"
    return 0
  fi
  fail "$name $(tr -d '\n' < "$dir/RESULT.json" 2>/dev/null | cut -c1-300); log $dir/run.log"
}

PF_ENV='{"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}'
BUILD_ENV='{"PYTHONUNBUFFERED": "1"}'
NOSECRETS=(/usr/bin/env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN -u HUGGINGFACE_HUB_TOKEN -u HUGGING_FACE_TOKEN_MAX
           -u SLACK_WEBHOOK_POPULACE_US -u SLACK_WEBHOOK_POPULACE_UK POPULACE_RELEASE_ENV=/dev/null HF_HUB_OFFLINE=1)

# --- 3. Prefetch the donors the release tool fetches itself when no flag names
# them (build_us_fiscal_refresh_release.py :9444, :10393, :10401, :10595,
# :10708, :10749 on origin/main), so a network failure surfaces in minutes,
# not hours into the release. Each fetch helper verifies its pinned digest per
# its module docstring; results land in ~/.cache/microcosm. ---
if ! stage_done prefetch; then
  write_config prefetch 16 12 26 10800 20000 "$W" "$BUILD_ENV" -- "$PY" -B -c '
import json
from microcosm.build.us_runtime import (
    fetch_asec_2023_weeks_unemployed_source, fetch_org_2024_donor,
    fetch_scf_2022_full_extract, fetch_scf_2022_summary_extract,
    fetch_sipp_2023_financial_asset_donor, fetch_sipp_2023_tip_donor)
paths = {}
for fn in (fetch_asec_2023_weeks_unemployed_source, fetch_scf_2022_summary_extract,
           fetch_sipp_2023_financial_asset_donor, fetch_scf_2022_full_extract,
           fetch_sipp_2023_tip_donor, fetch_org_2024_donor):
    paths[fn.__name__] = str(fn())
    print(fn.__name__, paths[fn.__name__], flush=True)
print(json.dumps(paths, indent=1))
'
  # Non-fatal here: the base needs none of these, so a fetch problem should not
  # hold it back. The release stage below requires the prefetch to pass.
  if ! ( run_stage prefetch 0 12 26 ); then
    rm -f "$RUN_ROOT/ROUTE_A_FAILED"
    log "prefetch failed (see $RUN/prefetch-sup/run.log); the base proceeds and the release retries the prefetch"
  fi
fi

# --- 4. Base stage from raw sources (main + #959). Measured 2026-09-16 under
# policyengine-us 1.819.0: 2,787.90 s wall, 6,238 CPU-s, 72.47 GB (67.5 GiB)
# peak, 44 GB of frame checkpoints + a 2.35 GB H5; under contention 5,058 s,
# 8,560 CPU-s, 65.10 GB. Not re-measured on policyengine-us 2.2.1. ---
BASE_CKPT=$RUN/base-checkpoints
BASE_OUT=$RUN/base-out
BASE_H5=$BASE_OUT/base_populace_us_2024_puf_support.h5
if ! stage_done base; then
  mkdir -p "$BASE_CKPT"
  have_ckpt=$(( $(du -sk "$BASE_CKPT" | cut -f1) / 1048576 ))
  base_disk=$(( 20 + 50 - have_ckpt )); [ $base_disk -lt 26 ] && base_disk=26
  write_config base 100 80 "$base_disk" 21600 64800 "$W" "$BUILD_ENV" -- "$PY" -B tools/build_us_puf_support_base.py \
    --stage all --checkpoint-dir "$BASE_CKPT" \
    --asec-h5 "2024=$STORAGE/census_cps_2024.h5" --asec-h5 "2023=$STORAGE/census_cps_2023.h5" --asec-h5 "2022=$STORAGE/census_cps_2022.h5" \
    --asec-h5-sha256 "2024=$ASEC_2024_SHA" --asec-h5-sha256 "2023=$ASEC_2023_SHA" --asec-h5-sha256 "2022=$ASEC_2022_SHA" \
    --puf-h5 "$STORAGE/puf_2024.h5" --puf-source-year-csv "$STORAGE/puf_2015.csv" --acs-h5 "$STORAGE/acs_2022.h5" \
    --asec-education-source "2022=$EDU/asecpub23csv.zip" --asec-education-source "2023=$EDU/asecpub24csv.zip" \
    --asec-education-source "2024=$EDU/asecpub25csv.zip" \
    --target-year 2024 --seed 0 --n-estimators 32 \
    --ledger-facts "$BASE_LEDGER_FACTS" \
    --assign-congressional-districts --congressional-district-vintage-crosswalk "$W/$CROSSWALK_REL" \
    --congressional-district-seed 0 --block-ladder-artifact "$LADDER" \
    --out "$BASE_OUT"
  run_stage base 0 80 "$base_disk"
fi
[ -f "$BASE_H5" ] || fail "base stage accepted but $BASE_H5 is missing"
if [ ! -f "$RUN/base.sha256" ] || [ "$BASE_H5" -nt "$RUN/base.sha256" ]; then
  echo "$(sha_of "$BASE_H5")  $BASE_H5" > "$RUN/base.sha256"
fi
BASE_SHA=$(cut -c1-64 "$RUN/base.sha256")
log "base H5 $BASE_H5 sha256 $BASE_SHA ($(stat -f %z "$BASE_H5") bytes)"
if [ "$PRUNE_BASE_CHECKPOINTS" = 1 ] && ls "$BASE_CKPT"/*.frame.h5 >/dev/null 2>&1; then
  log "PRUNE_BASE_CHECKPOINTS=1: removing base frame checkpoints ($(du -sh "$BASE_CKPT" | cut -f1)) after the base H5 was hashed"
  rm -f "$BASE_CKPT"/*.frame.h5
fi

# --- 5. Release-gate preflight on the fresh base (32.58 s / 9.76 GB on 09-16).
# Needs the new-lineage mode: on origin/main --selection-source-manifest is
# required (tools/preflight_us_release_gates.py:273-277). ---
PF_BLOCKED=0
PF_COMMON=(--base-h5 "$BASE_H5" --ledger-facts "$FEED" --ledger-facts-sha256 "$FEED_SHA")
[ -n "$EXPORT_MASS_REF_H5" ] && PF_COMMON+=(--export-input-mass-reference-h5 "$EXPORT_MASS_REF_H5")
if [ -z "$PREFLIGHT_NEW_LINEAGE_ARGS" ]; then
  PF_BLOCKED=1
  log "preflight-base BLOCKED: PREFLIGHT_NEW_LINEAGE_ARGS is empty (new-lineage preflight PR not landed); the release still runs"
  echo "BLOCKED $(date '+%F %T'): no new-lineage preflight mode" > "$RUN/preflight-base.BLOCKED"
else
  # shellcheck disable=SC2206
  PF_MODE=($PREFLIGHT_NEW_LINEAGE_ARGS)
  if ! stage_done preflight-base; then
    write_config preflight-base 24 16 22 3600 7200 "$PW" "$PF_ENV" -- "${NOSECRETS[@]}" "$PW/.venv/bin/python" -B \
      tools/preflight_us_release_gates.py "${PF_COMMON[@]}" "${PF_MODE[@]}" --json-out "$RUN/preflight-base.json"
    if [ "$RELEASE_DESPITE_PREFLIGHT_FAIL" = 1 ]; then run_stage preflight-base "0 1 2" 16 22
    else run_stage preflight-base "0 2" 16 22; fi
  fi
  log "preflight-base: $(cat "$RUN/preflight-base-sup/ACCEPTED") (0 clean, 2 AT-RISK only, 1 FAIL); report $RUN/preflight-base.json"
fi

# --- 6. Release tool, --base-h5 arm, committed feed, no
# selection source. Unmeasured at this size: without a selection the tool
# materializes PolicyEngine over the whole ~353k-household base
# (build_us_fiscal_refresh_release.py:9500-9506 on origin/main); July's 57k
# household run took 2 h 42 min at 85 GB. Limits are sized to the machine:
# 110 GiB RSS, configurable wall and available-RAM admission. ---
refuse_bad_release_args
stage_done prefetch || run_stage prefetch 0 12 26
REL_CKPT=$RUN/release-checkpoints
wait_prior_supervisor release
if ! stage_done release && ! result_accepted "$RUN/release-sup" 0 >/dev/null; then
  RID=populace-us-2024-${BASE_SHA:0:7}-${C:0:12}-$(date -u +%Y%m%dT%H%M%SZ)
  REL_OUT=$RUN/release-out/$RID
  release_command
  write_config release 110 "$RELEASE_RAM_GIB" 45 "${RELEASE_WALL_SECONDS:-345600}" 2000000 "$W" "$BUILD_ENV" -- "${RELEASE_ARGV[@]}"
fi
if ! stage_done release; then
  run_stage release 0 "$RELEASE_RAM_GIB" 45 "Set PRUNE_BASE_CHECKPOINTS=1 in route_a.env to reclaim the base's frame checkpoints."
fi
RID=$("$PY" -c 'import json,sys; a=json.load(open(sys.argv[1]))["argv"]; print(a[a.index("--release-id")+1])' "$RUN/release-sup/COMMAND.json") \
  || fail "cannot read the release id from release-sup/COMMAND.json"
REL_OUT=$RUN/release-out/$RID
REL_DIR=$REL_OUT/releases/$RID
[ -f "$REL_DIR/release_manifest.json" ] || fail "release accepted but $REL_DIR/release_manifest.json is missing"
log "release candidate $RID at $REL_DIR"

# --- 7. Release-gate preflight against the built release manifest. ---
if [ "$PF_BLOCKED" = 1 ]; then
  log "preflight-release BLOCKED: no new-lineage preflight mode"
  echo "BLOCKED $(date '+%F %T'): no new-lineage preflight mode" > "$RUN/preflight-release.BLOCKED"
else
  if ! stage_done preflight-release; then
    write_config preflight-release 24 16 22 3600 7200 "$PW" "$PF_ENV" -- "${NOSECRETS[@]}" "$PW/.venv/bin/python" -B \
      tools/preflight_us_release_gates.py "${PF_COMMON[@]}" "${PF_MODE[@]}" \
      --release-manifest "$REL_DIR/release_manifest.json" --json-out "$RUN/preflight-release.json"
    run_stage preflight-release "0 2" 16 22
  fi
  log "preflight-release: $(cat "$RUN/preflight-release-sup/ACCEPTED"); report $RUN/preflight-release.json"
fi

# --- 8. Publisher preflight, offline: publish_cli.py returns after
# prepare_release when --preflight-only is set (:325-331 on origin/main), and
# builds no Hub client. Secrets are stripped as overnight_chain.sh does. ---
if ! stage_done publisher-preflight; then
  write_config publisher-preflight 24 16 22 7200 14400 "$W" "$PF_ENV" -- "${NOSECRETS[@]}" \
    "PATH=$W/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" "$W/.venv/bin/microcosm-publish-release" \
    "$REL_DIR" --repo-id policyengine/populace-us --artifact-root "$REL_OUT/artifacts" --preflight-only
  run_stage publisher-preflight 0 16 22
fi
PUB_VERDICT=$(grep -E '^\{"valid"' "$RUN/publisher-preflight-sup/run.log" | tail -1)
log "publisher --preflight-only: $PUB_VERDICT"

# --- 9. Hand-off. Nothing was published. ---
STATE="done"; [ "$PF_BLOCKED" = 1 ] && STATE=blocked
OUTF=$RUN_ROOT/ROUTE_A_DONE.json; [ "$STATE" = blocked ] && OUTF=$RUN_ROOT/ROUTE_A_BLOCKED.json
PFB=BLOCKED; PFR=BLOCKED
if [ "$PF_BLOCKED" = 0 ]; then
  PFB=$(cat "$RUN/preflight-base-sup/ACCEPTED"); PFR=$(cat "$RUN/preflight-release-sup/ACCEPTED")
fi
"$PY" - "$OUTF" "$STATE" "$C" "$PC" "$PEUS" "$BASE_H5" "$BASE_SHA" "$RID" "$REL_DIR" "$REL_OUT/artifacts" \
  "$PFB" "$PFR" "$PUB_VERDICT" <<'EOF'
import json, sys
(out, state, commit, pcommit, peus, base_h5, base_sha, rid, rel_dir, art, pfb, pfr, pub) = sys.argv[1:]
json.dump({
    "state": state,
    "published": False,
    "build_commit": commit,
    "preflight_commit": pcommit,
    "policyengine_us": peus,
    "base_h5": base_h5,
    "base_sha256": base_sha,
    "release_id": rid,
    "release_dir": rel_dir,
    "artifact_root": art,
    "preflight_base": pfb,
    "preflight_release": pfr,
    "publisher_preflight_only": pub,
    "next": (
        "Publication is Max's call: tools/publish_release.sh "
        f"{rel_dir} --repo-id policyengine/populace-us --artifact-root {art}"
    ),
}, open(out, "w"), indent=1)
EOF
if [ "$STATE" = blocked ]; then
  log "ROUTE A BLOCKED: base and release candidate built, release-gate preflight needs the new-lineage mode; nothing published"
  exit 4
fi
log "ROUTE A DONE: release candidate $RID passed the release tool's gates and both preflights; nothing published (Max's call)"
