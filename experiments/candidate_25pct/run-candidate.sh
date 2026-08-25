#!/bin/bash
# Serial host launcher for the 25% one-surface + pkg3 legacy candidate.
# This is a legacy release arm and is not exact-k certified.
# OWNER RULING (Max, 2026-08-24): option A. The sparse candidate uses the
# current legacy fixed-penalty L0 selection at its literal default 0.8 and
# accepts the non-exact realized record count. It derives its own support from
# the new 25% pool: no frozen selection-source manifest, no exact-count rule,
# and no pi_hi. Keogh mass protection is omitted because current main documents
# that flag for protect-swapped carriers in a frozen selection, not cold L0.

set -u
set -o pipefail

export PATH="/Users/maxghenis/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="/Users/maxghenis"
export PYTHONUNBUFFERED=1
unset POPULACE_LOGBOOK_PREV_ROW_DIGEST

WT="/Users/maxghenis/PolicyEngine/_worktrees/microcosm-candidate-runbook"
ROOT="/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25"
LABEL="com.microcosm.candidate25"
GO_MARKER="/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/.max-go"
PYTHON="$WT/.venv/bin/python"
POOL_TOOL="$WT/tools/build_us_multispine_pool.py"
RELEASE_TOOL="$WT/tools/build_us_fiscal_refresh_release.py"
SCORER_TOOL="$WT/tools/score_us_release_head_to_head.py"
CANONICAL_LAUNCHER="$WT/experiments/candidate_25pct/run-candidate.sh"

POOL_ROOT="$ROOT/pool"
POOL_H5="$POOL_ROOT/pool.h5"
POOL_MANIFEST="$POOL_ROOT/pool.manifest.json"
POOL_GATES="$POOL_ROOT/pool.gates.json"
POOL_CHECKPOINTS="$POOL_ROOT/checkpoints"
POOL_MANIFEST_PIN="$POOL_ROOT/pool.manifest.sha256"
POOL_DONE="$POOL_ROOT/.stage-complete"
POOL_LOG="$POOL_ROOT/build.log"
POOL_RSS="$POOL_ROOT/rss.csv"

DENSE_ROOT="$ROOT/release-dense"
DENSE_CHECKPOINTS="$DENSE_ROOT/checkpoints"
DENSE_RELEASE_ID_FILE="$DENSE_ROOT/release_id.txt"
DENSE_ARTIFACT="$DENSE_ROOT/artifacts/populace_us_2024.h5"
DENSE_DONE="$DENSE_ROOT/.stage-complete"
DENSE_LOG="$DENSE_ROOT/release.log"
DENSE_RSS="$DENSE_ROOT/rss.csv"

SPARSE_ROOT="$ROOT/release-sparse"
SPARSE_CHECKPOINTS="$SPARSE_ROOT/checkpoints"
SPARSE_RELEASE_ID_FILE="$SPARSE_ROOT/release_id.txt"
SPARSE_ARTIFACT="$SPARSE_ROOT/artifacts/populace_us_2024.h5"
SPARSE_DONE="$SPARSE_ROOT/.stage-complete"
SPARSE_LOG="$SPARSE_ROOT/release.log"
SPARSE_RSS="$SPARSE_ROOT/rss.csv"

MAIN_LOG="$ROOT/run-candidate.log"
CODE_PIN="$ROOT/code.commit"

ASEC_RAW="/Users/maxghenis/PolicyEngine/_buildo-runtime/out/591-pawtyp-pool/asec-producer-checkpoints/asec_raw_stage.checkpoint.h5"
ASEC_RAW_SHA="51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe"
ACS_HOUSEHOLD="/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0/csv_hus.zip"
ACS_HOUSEHOLD_SHA="8281008e53de98f0ef81e7a2ee5a8725991dda1ecfd2713ead73246425e515d0"
ACS_PERSON="/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/inputs/acs_2024_1yr/afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894/csv_pus.zip"
ACS_PERSON_SHA="afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894"
ACS_RENT="/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/acs_2022.h5"
ACS_RENT_SHA="0b319b496f19a6913066f9c5ea572edfda3d78a187be6f375846617d0b441bd4"
PUF_H5="/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5"
PUF_H5_SHA="7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df"
PUF_SOURCE="/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv"
PUF_SOURCE_SHA="0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df"

LEDGER="/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl"
LEDGER_SHA="b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080"
EXPORT_REFERENCE="/Users/maxghenis/PolicyEngine/_buildg-runtime/forensics/populace_us_2024.h5"
EXPORT_REFERENCE_SHA="c2065b642ab00da74746afdfd9f06890e5f32f9b10bd6610ff236452d40f39c5"
SCF_SUMMARY="/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/scf_cache/rscfp2022.dta"
SCF_SUMMARY_SHA="6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1"
SCF_FULL="/Users/maxghenis/.cache/microcosm/scf/p22i6.dta"
SCF_FULL_SHA="61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a"
SSI_BASIS="/Users/maxghenis/PolicyEngine/_buildo-runtime/out/buildo-run/dense-p2/releases/populace-us-2024-buildp-dense-cae8640-20260728T050443Z/us_ssi_take_up.json"
SSI_BASIS_SHA="56118bde095b8ef2559a26a3478ff5f8b61939eca402dffcec61189e7de631e3"
SPARSE_SSI_BASIS="/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/attempt6_basis_schema3_seed.json"
SPARSE_SSI_BASIS_SHA="25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a"
ASEC_WEEKS="/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education/asecpub23csv.zip"
ASEC_WEEKS_SHA="d2e000250782adfbdd7f29c82b66d866591a30f0d330496698ec19f9c784ce11"
SIPP_FULL="/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023.csv"
SIPP_FULL_SHA="5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2"
SIPP_TIPS="/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/pu2023_slim.csv"
SIPP_TIPS_SHA="1f0bcb8e045ef1118e8eba4b4a2997bdaaf947bd0dd09d41fa7c7d5657a3d7d5"
ORG_WAGES="/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_org_2024_wages.csv.gz"
ORG_WAGES_SHA="66fa5b6aa4087413b691038767b51f603281ff55411b58259922f78e67460372"
CD_CROSSWALK="$WT/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv"
CD_CROSSWALK_SHA="c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec"
PUMA_LADDER="/Users/maxghenis/PolicyEngine/_worktrees/populace-acs-clone/build/us/us_puma_ladder_2020.npz"
PUMA_LADDER_SHA="39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8"

INCUMBENT_EVIDENCE="$WT/experiments/replacement_scorecard/incumbent_48b9d479.json"
INCUMBENT_EVIDENCE_SHA="b2ad1a07f9668bc5d796cc9de99ef12da781b1ee8163ea65781871a20da441c8"
INCUMBENT_H5="/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/26dcad66867687f15735dc4926523e3741920836/populace_us_2024.h5"
INCUMBENT_H5_SHA="48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e"

DRY_RUN=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *)
    printf 'usage: %s [--dry-run]\n' "$0" >&2
    exit 2
    ;;
esac
if [ "$#" -gt 1 ]; then
  printf 'usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

RUN_TS="$(/bin/date -u +%Y%m%dT%H%M%SZ)"
CODE_SHA=""
CODE_SHA8=""
DENSE_RELEASE_ID=""
DENSE_RELEASE_DIR=""
DENSE_RELEASE_MANIFEST=""
DENSE_BUILD_MANIFEST=""
SPARSE_RELEASE_ID=""
SPARSE_RELEASE_DIR=""
SPARSE_RELEASE_MANIFEST=""
SPARSE_BUILD_MANIFEST=""

utc_now() {
  /bin/date -u +%FT%TZ
}

emit() {
  local line
  line="[$(utc_now)] $*"
  printf '%s\n' "$line"
  if [ "$DRY_RUN" -eq 0 ] && [ -d "$ROOT" ]; then
    printf '%s\n' "$line" >> "$MAIN_LOG"
  fi
}

die() {
  emit "FATAL $*"
  exit 1
}

cleanup_label() {
  local rc="$1"
  trap - EXIT
  if [ "$DRY_RUN" -eq 0 ]; then
    emit "launchd self-removal label=$LABEL rc=$rc"
    /bin/launchctl remove "$LABEL" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

sha256_file() {
  /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

check_sha256() {
  local role="$1"
  local path="$2"
  local expected="$3"
  local actual
  local size

  [ -f "$path" ] || die "missing input role=$role path=$path"
  actual="$(sha256_file "$path")" || die "cannot hash role=$role path=$path"
  [ "$actual" = "$expected" ] || die "sha mismatch role=$role path=$path expected=$expected actual=$actual"
  size="$(/usr/bin/stat -L -f %z "$path")" || die "cannot stat role=$role path=$path"
  emit "INPUT OK role=$role size=$size sha256=$actual path=$path"
}

check_scf_header() {
  local expected_prefix
  local actual_prefix
  expected_prefix="3c73746174615f6474613e3c6865616465723e3c72656c656173653e3131383c2f72656c656173653e3c627974656f726465723e4c53463c2f627974656f7264"
  actual_prefix="$(/usr/bin/xxd -l 64 -p "$SCF_FULL" | /usr/bin/tr -d '\n')" || die "cannot inspect SCF full-extract header"
  [ "$actual_prefix" = "$expected_prefix" ] || die "SCF full extract is not the expected Stata release-118 little-endian header"
  emit "INPUT HEADER OK role=scf-full-extract format=stata-dta release=118 byteorder=LSF path=$SCF_FULL"
}

require_parser_flags() {
  local surface="$1"
  local source="$2"
  shift 2
  local flag

  for flag in "$@"; do
    /usr/bin/grep -Fq -- "$flag" "$source" || die "current HEAD lacks parser flag surface=$surface flag=$flag source=$source"
    emit "PARSER FLAG OK surface=$surface flag=$flag source=$source"
  done
}

check_code_authority() {
  local dirty
  local launcher_sha
  local canonical_sha

  [ -x "$PYTHON" ] || die "missing executable Python path=$PYTHON"
  [ -f "$POOL_TOOL" ] || die "missing pool builder path=$POOL_TOOL"
  [ -f "$RELEASE_TOOL" ] || die "missing release builder path=$RELEASE_TOOL"
  [ -f "$SCORER_TOOL" ] || die "missing scorer path=$SCORER_TOOL"
  [ -f "$CANONICAL_LAUNCHER" ] || die "missing canonical launcher path=$CANONICAL_LAUNCHER"
  [ -f "$0" ] || die "cannot resolve invoked launcher path=$0"
  launcher_sha="$(sha256_file "$0")" || die "cannot hash invoked launcher path=$0"
  canonical_sha="$(sha256_file "$CANONICAL_LAUNCHER")" || die "cannot hash canonical launcher path=$CANONICAL_LAUNCHER"
  [ "$launcher_sha" = "$canonical_sha" ] || die "invoked launcher differs from committed canonical copy invoked_sha=$launcher_sha canonical_sha=$canonical_sha"
  emit "LAUNCHER OK sha256=$launcher_sha invoked_path=$0 canonical_path=$CANONICAL_LAUNCHER"
  CODE_SHA="$(/usr/bin/git -C "$WT" rev-parse HEAD)" || die "cannot resolve worktree HEAD"
  CODE_SHA8="${CODE_SHA:0:8}"
  dirty="$(/usr/bin/git -C "$WT" status --porcelain)" || die "cannot inspect worktree status"
  [ -z "$dirty" ] || die "tracked worktree is dirty; commit or remove changes before launch"
  emit "CODE OK commit=$CODE_SHA worktree=$WT"

  require_parser_flags pool "$POOL_TOOL" \
    --asec-raw-stage-h5 --asec-raw-stage-h5-sha256 \
    --acs-household-zip --acs-household-zip-sha256 \
    --acs-person-zip --acs-person-zip-sha256 \
    --acs-rent-h5 --acs-rent-h5-sha256 \
    --puf-h5 --puf-h5-sha256 \
    --puf-source-year-csv --puf-source-year-csv-sha256 \
    --puma-ladder --puma-ladder-sha256 \
    --congressional-district-vintage-crosswalk \
    --congressional-district-vintage-crosswalk-sha256 \
    --sample-fraction --sample-seed \
    --clone-attachment-fraction --clone-attachment-seed \
    --checkpoint-root --out
  require_parser_flags dense "$RELEASE_TOOL" \
    --base-h5 --dense-default-dataset \
    --ledger-facts --ledger-facts-sha256 \
    --export-input-mass-reference-h5 \
    --asec-2023-weeks-unemployed-source \
    --scf-summary-extract --scf-full-extract \
    --sipp-tip-donor --sipp-vehicle-donor --org-wages-donor \
    --ssi-take-up-prior-weight-basis \
    --ssi-take-up-prior-weight-basis-sha256 \
    --checkpoint-root --release-id --seed --epochs \
    --skip-reform-validation --no-staging --out
  require_parser_flags sparse "$RELEASE_TOOL" --l0-refit-lambda-share
  require_parser_flags scorer "$SCORER_TOOL" \
    --incumbent --candidate --ledger-facts --out-prefix
}

check_pool_inputs() {
  check_sha256 asec-raw-stage-h5 "$ASEC_RAW" "$ASEC_RAW_SHA"
  check_sha256 puma-ladder "$PUMA_LADDER" "$PUMA_LADDER_SHA"
  check_sha256 cd-crosswalk "$CD_CROSSWALK" "$CD_CROSSWALK_SHA"
  check_sha256 acs-household-zip "$ACS_HOUSEHOLD" "$ACS_HOUSEHOLD_SHA"
  check_sha256 acs-person-zip "$ACS_PERSON" "$ACS_PERSON_SHA"
  check_sha256 acs-rent-h5 "$ACS_RENT" "$ACS_RENT_SHA"
  check_sha256 puf-h5 "$PUF_H5" "$PUF_H5_SHA"
  check_sha256 puf-source-year-csv "$PUF_SOURCE" "$PUF_SOURCE_SHA"
}

check_release_inputs() {
  check_sha256 ledger-v9.4 "$LEDGER" "$LEDGER_SHA"
  check_sha256 export-input-mass-reference "$EXPORT_REFERENCE" "$EXPORT_REFERENCE_SHA"
  check_sha256 scf-summary-extract "$SCF_SUMMARY" "$SCF_SUMMARY_SHA"
  check_sha256 scf-full-extract "$SCF_FULL" "$SCF_FULL_SHA"
  check_scf_header
  check_sha256 asec-2023-weeks-source "$ASEC_WEEKS" "$ASEC_WEEKS_SHA"
  check_sha256 sipp-full-donor "$SIPP_FULL" "$SIPP_FULL_SHA"
  check_sha256 sipp-tips-donor "$SIPP_TIPS" "$SIPP_TIPS_SHA"
  check_sha256 cps-org-wages-compressed "$ORG_WAGES" "$ORG_WAGES_SHA"
  check_sha256 packaged-cd-crosswalk "$CD_CROSSWALK" "$CD_CROSSWALK_SHA"
}

check_dense_inputs() {
  check_release_inputs
  check_sha256 dense-ssi-prior-weight-basis "$SSI_BASIS" "$SSI_BASIS_SHA"
}

check_sparse_inputs() {
  check_release_inputs
  check_sha256 sparse-ssi-prior-weight-basis "$SPARSE_SSI_BASIS" "$SPARSE_SSI_BASIS_SHA"
}

check_reporting_inputs() {
  check_sha256 incumbent-evidence "$INCUMBENT_EVIDENCE" "$INCUMBENT_EVIDENCE_SHA"
  check_sha256 incumbent-h5 "$INCUMBENT_H5" "$INCUMBENT_H5_SHA"
}

check_immutable_inputs() {
  check_pool_inputs
  check_release_inputs
  check_sha256 dense-ssi-prior-weight-basis "$SSI_BASIS" "$SSI_BASIS_SHA"
  check_sha256 sparse-ssi-prior-weight-basis "$SPARSE_SSI_BASIS" "$SPARSE_SSI_BASIS_SHA"
  check_reporting_inputs
}

print_command() {
  local label="$1"
  shift
  local command_arg
  local quoted
  local rendered="$label"

  for command_arg in "$@"; do
    quoted="$(printf '%q' "$command_arg")"
    rendered="$rendered $quoted"
  done
  printf '%s\n' "$rendered"
  if [ "$DRY_RUN" -eq 0 ]; then
    printf '%s\n' "$rendered" >> "$MAIN_LOG"
  fi
}

validate_sparse_command_contract() {
  local validation

  validation="$("$PYTHON" - "$RELEASE_TOOL" "${SPARSE_COMMAND[@]:2}" <<'PY'
import ast
import sys
from pathlib import Path

source = Path(sys.argv[1])
tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

constant_values = []
l0_flag_defaults = []
for node in ast.walk(tree):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name)
            and target.id == "DEFAULT_L0_REFIT_LAMBDA_SHARE"
            for target in targets
        ):
            constant_values.append(ast.literal_eval(node.value))
    if not isinstance(node, ast.Call) or not node.args:
        continue
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--l0-refit-lambda-share"
    ):
        continue
    for keyword in node.keywords:
        if keyword.arg == "default":
            l0_flag_defaults.append(keyword.value)

argv = sys.argv[2:]
present = set(argv)
declared_options = {
    argument.value
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "add_argument"
    for argument in node.args
    if isinstance(argument, ast.Constant)
    and isinstance(argument.value, str)
    and argument.value.startswith("--")
}

failures = []
if constant_values != [0.8]:
    failures.append(
        "current builder does not define the owner-ruled L0 default exactly once "
        f"as 0.8: {constant_values!r}"
    )
if len(l0_flag_defaults) != 1 or not (
    isinstance(l0_flag_defaults[0], ast.Name)
    and l0_flag_defaults[0].id == "DEFAULT_L0_REFIT_LAMBDA_SHARE"
):
    failures.append("--l0-refit-lambda-share is not wired to the current default constant")

prohibited = {
    "--allow-ecps-parity-gaps",
    "--allow-input-coverage-gaps",
    "--allow-input-mass-drift",
    "--allow-qrf-tail-concentration",
    "--allow-reform-coverage-smoke-failures",
    "--allow-unaged-dollar-targets",
    "--dense-default-dataset",
    "--evidence-release",
    "--evidence-failure-owners",
    "--exact-k",
    "--exact-k-pi-hi",
    "--input-mass-minimum-reference-total",
    "--input-mass-relative-tolerance",
    "--l0-refit-lambda-share",
    "--l2-lambda",
    "--learning-rate",
    "--max-weight-ratio",
    "--pool-manifest",
    "--pool-manifest-sha256",
    "--pool-release-id",
    "--qrf-tail-concentration-exclusions",
    "--refit-l2-lambda",
    "--selection-mass-protection",
    "--selection-mode",
    "--selection-join-key",
    "--selection-source-h5",
    "--selection-source-manifest",
    "--skip-reform-coverage-smoke",
    "--target-family-loss-multiplier",
    "--warm-start-calibration-npz",
    "--zero-support-exclusions",
    "--no-age-targets",
}
unexpected = sorted(present & prohibited)
if unexpected:
    failures.append(f"sparse command enters a prohibited/tuned path: {unexpected}")
unknown = sorted(
    token for token in present if token.startswith("--") and token not in declared_options
)
if unknown:
    failures.append(f"sparse command uses options absent from the current parser: {unknown}")

for flag, value in (("--seed", "0"), ("--epochs", "6000")):
    if argv.count(flag) != 1:
        failures.append(f"sparse command must contain {flag} exactly once")
        continue
    index = argv.index(flag)
    if index + 1 >= len(argv) or argv[index + 1] != value:
        failures.append(f"sparse command must set {flag} to {value}")
for flag in ("--skip-reform-validation", "--no-staging"):
    if argv.count(flag) != 1:
        failures.append(f"sparse command must contain {flag} exactly once")
if failures:
    raise SystemExit("; ".join(failures))
print(
    "SPARSE LEGACY CONTRACT OK "
    "l0_lambda_share=0.8 source=none exact_k=none pi_hi=none "
    "selection_mass_protection=none operator_waivers=none "
    "seed=0 epochs=6000 no_staging=true"
)
PY
)" || die "sparse command violates owner ruling A or current parser defaults"
  emit "$validation"
}

reclaimable_gib() {
  /usr/bin/vm_stat | /usr/bin/awk '
    NR == 1 {
      page_size = $8
      gsub(/[^0-9]/, "", page_size)
    }
    /^Pages free:|^Pages inactive:|^Pages speculative:|^Pages purgeable:/ {
      pages = $3
      gsub(/\./, "", pages)
      total += pages
    }
    END {
      if (!page_size) exit 2
      printf "%d\n", int(total * page_size / 1024 / 1024 / 1024)
    }
  '
}

builders_busy() {
  if /usr/bin/pgrep -f '[.]venv/bin/python.*[/]tools/build_us_multispine_pool[.]py' >/dev/null 2>&1; then
    return 0
  fi
  if /usr/bin/pgrep -f '[.]venv/bin/python.*[/]tools/build_us_fiscal_refresh_release[.]py' >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

on_ac_power() {
  /usr/bin/pmset -g batt 2>/dev/null | /usr/bin/grep -Fq "Now drawing from 'AC Power'"
}

wait_ready() {
  local stage="$1"
  local needed_gib="$2"
  local reclaimable
  local busy
  local ac
  local go

  while :; do
    reclaimable="$(reclaimable_gib 2>/dev/null || printf '0')"
    busy=0
    builders_busy && busy=1
    ac=0
    on_ac_power && ac=1
    go=0
    [ -f "$GO_MARKER" ] && go=1
    if [ "$reclaimable" -ge "$needed_gib" ] && [ "$busy" -eq 0 ] && [ "$ac" -eq 1 ] && [ "$go" -eq 1 ]; then
      emit "PRECONDITION READY stage=$stage need_reclaimable_gib=$needed_gib reclaimable_gib=$reclaimable busy=$busy ac=$ac go=$go"
      return 0
    fi
    emit "PRECONDITION WAIT stage=$stage need_reclaimable_gib=$needed_gib reclaimable_gib=$reclaimable busy=$busy ac=$ac go=$go poll_seconds=300"
    /bin/sleep 300
  done
}

ready_now() {
  local stage="$1"
  local needed_gib="$2"
  local reclaimable
  local busy
  local ac
  local go

  reclaimable="$(reclaimable_gib 2>/dev/null || printf '0')"
  busy=0
  builders_busy && busy=1
  ac=0
  on_ac_power && ac=1
  go=0
  [ -f "$GO_MARKER" ] && go=1
  if [ "$reclaimable" -ge "$needed_gib" ] && [ "$busy" -eq 0 ] && [ "$ac" -eq 1 ] && [ "$go" -eq 1 ]; then
    emit "PRECONDITION RECHECK READY stage=$stage need_reclaimable_gib=$needed_gib reclaimable_gib=$reclaimable busy=$busy ac=$ac go=$go"
    return 0
  fi
  emit "PRECONDITION CHANGED stage=$stage need_reclaimable_gib=$needed_gib reclaimable_gib=$reclaimable busy=$busy ac=$ac go=$go; repeating wait-and-authenticate cycle"
  return 1
}

sample_tree_rss() {
  local stage="$1"
  local root_pid="$2"
  local csv="$3"
  local sample
  local process_count
  local rss_kib
  local rss_gib

  if [ ! -s "$csv" ]; then
    printf 'ts_utc,stage,root_pid,process_count,tree_rss_kib,tree_rss_gib\n' >> "$csv"
  fi
  while /bin/kill -0 "$root_pid" >/dev/null 2>&1; do
    sample="$(/bin/ps -axo pid=,ppid=,rss= | /usr/bin/awk -v root="$root_pid" '
      {
        pid[$1] = 1
        parent[$1] = $2
        rss[$1] = $3
      }
      END {
        count = 0
        total = 0
        for (candidate in pid) {
          for (seen_pid in seen) delete seen[seen_pid]
          cursor = candidate
          while (cursor != "" && cursor != "0" && !seen[cursor]) {
            if (cursor == root) {
              count += 1
              total += rss[candidate]
              break
            }
            seen[cursor] = 1
            cursor = parent[cursor]
          }
        }
        printf "%d %d\n", count, total
      }
    ')"
    process_count="${sample%% *}"
    rss_kib="${sample#* }"
    rss_gib="$(/usr/bin/awk -v kib="$rss_kib" 'BEGIN {printf "%.3f", kib / 1024 / 1024}')"
    printf '%s,%s,%s,%s,%s,%s\n' "$(utc_now)" "$stage" "$root_pid" "$process_count" "$rss_kib" "$rss_gib" >> "$csv"
    /bin/sleep 30
  done
}

run_monitored() {
  local stage="$1"
  local stage_log="$2"
  local rss_csv="$3"
  shift 3
  local root_pid
  local sampler_pid
  local rc

  print_command "COMMAND stage=$stage" /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "$@"
  /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "$@" >> "$stage_log" 2>&1 &
  root_pid=$!
  sample_tree_rss "$stage" "$root_pid" "$rss_csv" &
  sampler_pid=$!
  if wait "$root_pid"; then
    rc=0
  else
    rc=$?
  fi
  /bin/kill "$sampler_pid" >/dev/null 2>&1 || true
  wait "$sampler_pid" >/dev/null 2>&1 || true
  emit "STAGE EXIT stage=$stage rc=$rc log=$stage_log rss_csv=$rss_csv"
  return "$rc"
}

write_new_file() {
  local path="$1"
  local content="$2"
  local temporary="${path}.tmp.$$"

  [ ! -e "$path" ] || die "refusing to replace existing state file path=$path"
  (umask 022; printf '%s\n' "$content" > "$temporary") || die "cannot write temporary state path=$temporary"
  /bin/mv "$temporary" "$path" || die "cannot install state path=$path"
}

pin_code_authority() {
  local recorded

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ -f "$CODE_PIN" ]; then
      recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$CODE_PIN")"
      if [ "$recorded" = "$CODE_SHA" ]; then
        emit "CODE PIN OBSERVED commit=$recorded path=$CODE_PIN execute_mode_conflict=false"
      else
        emit "CODE PIN OBSERVED recorded_commit=$recorded planned_commit=$CODE_SHA path=$CODE_PIN execute_mode_conflict=true dry_run_mutation=false"
      fi
    else
      emit "CODE PIN PLANNED commit=$CODE_SHA path=$CODE_PIN"
    fi
  elif [ -f "$CODE_PIN" ]; then
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$CODE_PIN")"
    [ "$recorded" = "$CODE_SHA" ] || die "candidate root is pinned to another commit recorded=$recorded current=$CODE_SHA"
    emit "CODE PIN CONSUMED commit=$recorded path=$CODE_PIN"
  else
    write_new_file "$CODE_PIN" "$CODE_SHA  $WT"
    emit "CODE PIN RECORDED commit=$CODE_SHA path=$CODE_PIN"
  fi
}

recheck_code_authority() {
  local canonical_sha
  local current
  local dirty
  local launcher_sha
  local recorded

  current="$(/usr/bin/git -C "$WT" rev-parse HEAD)" || die "cannot re-resolve worktree HEAD"
  [ "$current" = "$CODE_SHA" ] || die "worktree HEAD changed while launcher waited pinned=$CODE_SHA current=$current"
  dirty="$(/usr/bin/git -C "$WT" status --porcelain)" || die "cannot reinspect worktree status"
  [ -z "$dirty" ] || die "worktree changed while launcher waited; refusing stage action"
  launcher_sha="$(sha256_file "$0")" || die "cannot rehash invoked launcher path=$0"
  canonical_sha="$(sha256_file "$CANONICAL_LAUNCHER")" || die "cannot rehash canonical launcher path=$CANONICAL_LAUNCHER"
  [ "$launcher_sha" = "$canonical_sha" ] || die "launcher changed while waiting invoked_sha=$launcher_sha canonical_sha=$canonical_sha"
  [ -f "$CODE_PIN" ] || die "missing candidate-root code pin path=$CODE_PIN"
  recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$CODE_PIN")"
  [ "$recorded" = "$CODE_SHA" ] || die "candidate-root code pin changed recorded=$recorded expected=$CODE_SHA"
  emit "CODE RECHECK OK commit=$CODE_SHA launcher_sha256=$launcher_sha"
}

validate_pool_outputs() {
  local validation

  validation="$("$PYTHON" - \
    "$POOL_MANIFEST" "$POOL_H5" "$POOL_GATES" \
    asec_raw_stage "$ASEC_RAW" "$ASEC_RAW_SHA" \
    acs_household "$ACS_HOUSEHOLD" "$ACS_HOUSEHOLD_SHA" \
    acs_person "$ACS_PERSON" "$ACS_PERSON_SHA" \
    acs_rent_donor "$ACS_RENT" "$ACS_RENT_SHA" \
    processed_puf "$PUF_H5" "$PUF_H5_SHA" \
    puf_source_year "$PUF_SOURCE" "$PUF_SOURCE_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, h5_path, gates_path = map(Path, sys.argv[1:4])
raw_provenance = sys.argv[4:]
if len(raw_provenance) % 3:
    raise SystemExit("internal launcher error: pool provenance arguments are not triples")
expected_provenance = {
    raw_provenance[index]: (Path(raw_provenance[index + 1]), raw_provenance[index + 2])
    for index in range(0, len(raw_provenance), 3)
}
for path in (manifest_path, h5_path, gates_path):
    if not path.is_file():
        raise SystemExit(f"missing pool output: {path}")

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()

payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("status") != "simulation_ready" or payload.get("simulation_ready") is not True:
    raise SystemExit("pool manifest is not simulation_ready")
sampling = payload.get("sampling", {})
if sampling.get("sample_fraction") != 0.25 or sampling.get("sample_seed") != 578:
    raise SystemExit(f"pool sampling receipt is wrong: {sampling}")
attachment = payload.get("clone_attachment", {})
if attachment.get("fraction") != 1.0 or attachment.get("seed") != 578:
    raise SystemExit(f"pool clone-attachment receipt is wrong: {attachment}")
provenance_pins = payload.get("provenance_pins")
if not isinstance(provenance_pins, dict):
    raise SystemExit("pool manifest lacks provenance_pins")
if set(provenance_pins) != set(expected_provenance):
    raise SystemExit(
        "pool provenance roles mismatch: "
        f"recorded={sorted(provenance_pins)} expected={sorted(expected_provenance)}"
    )
for role, (expected_path, expected_sha) in expected_provenance.items():
    block = provenance_pins.get(role)
    if not isinstance(block, dict):
        raise SystemExit(f"pool provenance pin is not an object: {role}")
    if Path(str(block.get("path"))).resolve() != expected_path.resolve():
        raise SystemExit(f"pool provenance path mismatch for {role}")
    if block.get("expected_sha256") != expected_sha:
        raise SystemExit(f"pool expected provenance sha mismatch for {role}")
    if block.get("actual_sha256") != expected_sha:
        raise SystemExit(f"pool actual provenance sha mismatch for {role}")
    if block.get("size_bytes") != expected_path.stat().st_size:
        raise SystemExit(f"pool provenance size mismatch for {role}")
digests = {}
for block_name, expected_path in (("pool_h5", h5_path), ("agreement_diagnostics", gates_path)):
    block = payload.get(block_name)
    if not isinstance(block, dict):
        raise SystemExit(f"pool manifest lacks {block_name}")
    if Path(str(block.get("path"))).resolve() != expected_path.resolve():
        raise SystemExit(f"pool manifest {block_name} path does not match {expected_path}")
    actual = digest(expected_path)
    if block.get("sha256") != actual:
        raise SystemExit(f"pool manifest {block_name} sha mismatch: {block.get('sha256')} != {actual}")
    if block.get("size_bytes") != expected_path.stat().st_size:
        raise SystemExit(f"pool manifest {block_name} size mismatch")
    digests[block_name] = actual
print(f"POOL OUTPUTS OK h5_sha256={digests['pool_h5']} gates_sha256={digests['agreement_diagnostics']}")
PY
)" || die "pool output authentication failed"
  emit "$validation"
}

record_pool_manifest_pin() {
  local actual
  local recorded

  actual="$(sha256_file "$POOL_MANIFEST")" || die "cannot hash pool manifest"
  if [ -f "$POOL_MANIFEST_PIN" ]; then
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$POOL_MANIFEST_PIN")"
    [ "$recorded" = "$actual" ] || die "pool manifest changed after pin recorded=$recorded actual=$actual"
  else
    write_new_file "$POOL_MANIFEST_PIN" "$actual  $POOL_MANIFEST"
  fi
  if [ -f "$POOL_DONE" ]; then
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$POOL_DONE")"
    [ "$recorded" = "$actual" ] || die "pool completion marker disagrees with manifest recorded=$recorded actual=$actual"
  else
    write_new_file "$POOL_DONE" "$actual  $POOL_MANIFEST"
  fi
  emit "POOL MANIFEST PIN CONSUMABLE sha256=$actual path=$POOL_MANIFEST pin_file=$POOL_MANIFEST_PIN"
}

consume_pool_manifest_pin() {
  local stage="$1"
  local recorded
  local actual

  [ -f "$POOL_MANIFEST_PIN" ] || die "missing recorded pool manifest pin path=$POOL_MANIFEST_PIN"
  recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$POOL_MANIFEST_PIN")"
  actual="$(sha256_file "$POOL_MANIFEST")" || die "cannot rehash pool manifest before release"
  [ "$recorded" = "$actual" ] || die "release refused changed pool manifest recorded=$recorded actual=$actual"
  validate_pool_outputs
  emit "POOL MANIFEST PIN CONSUMED stage=$stage sha256=$actual path=$POOL_MANIFEST"
}

pool_is_complete() {
  local required
  local state
  for required in "$POOL_H5" "$POOL_MANIFEST" "$POOL_GATES"; do
    if [ ! -f "$required" ]; then
      if [ -f "$POOL_DONE" ]; then
        die "pool completion marker exists but output is missing path=$required"
      fi
      return 1
    fi
  done
  if [ ! -f "$POOL_DONE" ]; then
    state="$("$PYTHON" - "$POOL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(f"{payload.get('status')}:{str(payload.get('simulation_ready')).lower()}")
PY
)" || die "cannot read existing pool manifest state path=$POOL_MANIFEST"
    if [ "$state" = "gate_failed:false" ]; then
      [ ! -f "$POOL_MANIFEST_PIN" ] || die "gate-failed pool conflicts with an existing manifest pin path=$POOL_MANIFEST_PIN"
      emit "STAGE RESUME stage=pool reason=prior-gate-failed-output checkpoints=$POOL_CHECKPOINTS"
      return 1
    fi
    [ "$state" = "simulation_ready:true" ] || die "unrecognized pool manifest state without completion marker state=$state"
  fi
  validate_pool_outputs
  record_pool_manifest_pin
  return 0
}

set_dense_release_id() {
  local candidate

  candidate="populace-us-2024-onesurface-pkg3-legacy-dense-${CODE_SHA8}-${RUN_TS}"
  if [ "$DRY_RUN" -eq 1 ]; then
    DENSE_RELEASE_ID="$candidate"
  elif [ -f "$DENSE_RELEASE_ID_FILE" ]; then
    DENSE_RELEASE_ID="$(/usr/bin/awk 'NF {print $1; exit}' "$DENSE_RELEASE_ID_FILE")"
  else
    [ ! -f "$DENSE_ARTIFACT" ] || die "dense artifact exists without persisted release id path=$DENSE_ARTIFACT"
    write_new_file "$DENSE_RELEASE_ID_FILE" "$candidate"
    DENSE_RELEASE_ID="$candidate"
  fi
  printf '%s\n' "$DENSE_RELEASE_ID" | /usr/bin/grep -Eq '^populace-us-2024-onesurface-pkg3-legacy-dense-[0-9a-f]{8}-[0-9]{8}T[0-9]{6}Z$' || die "invalid persisted dense release id value=$DENSE_RELEASE_ID"
  printf '%s\n' "$DENSE_RELEASE_ID" | /usr/bin/grep -Eq "^populace-us-2024-onesurface-pkg3-legacy-dense-${CODE_SHA8}-[0-9]{8}T[0-9]{6}Z$" || die "persisted dense release id does not match pinned commit commit=$CODE_SHA value=$DENSE_RELEASE_ID"
  DENSE_RELEASE_DIR="$DENSE_ROOT/releases/$DENSE_RELEASE_ID"
  DENSE_RELEASE_MANIFEST="$DENSE_RELEASE_DIR/release_manifest.json"
  DENSE_BUILD_MANIFEST="$DENSE_RELEASE_DIR/build_manifest.json"
  emit "DENSE RELEASE ID value=$DENSE_RELEASE_ID state_file=$DENSE_RELEASE_ID_FILE"
}

validate_dense_release() {
  local validation

  validation="$("$PYTHON" - \
    "$DENSE_ARTIFACT" \
    "$DENSE_RELEASE_MANIFEST" \
    "$DENSE_BUILD_MANIFEST" \
    "$DENSE_RELEASE_DIR/qrf_tail_concentration.json" \
    "$DENSE_RELEASE_DIR/calibration_diagnostics.json" \
    "$POOL_MANIFEST" \
    "$DENSE_RELEASE_ID" "$CODE_SHA" "$LEDGER_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

artifact, release_path, build_path, qrf_path, diagnostics_path, pool_manifest_path = map(
    Path, sys.argv[1:7]
)
release_id, code_sha, ledger_sha = sys.argv[7:10]
for path in (
    artifact,
    release_path,
    build_path,
    qrf_path,
    diagnostics_path,
    pool_manifest_path,
):
    if not path.is_file():
        raise SystemExit(f"missing dense release output: {path}")

digest = hashlib.sha256()
with artifact.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
artifact_sha = digest.hexdigest()
release = json.loads(release_path.read_text(encoding="utf-8"))
build = json.loads(build_path.read_text(encoding="utf-8"))
qrf = json.loads(qrf_path.read_text(encoding="utf-8"))
diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
if release.get("build", {}).get("build_id") != release_id:
    raise SystemExit("dense release manifest build id mismatch")
entry = release.get("artifacts", {}).get("populace_us_2024", {})
if entry.get("sha256") != artifact_sha:
    raise SystemExit("dense artifact hash does not match release manifest")
default = release.get("build", {}).get("default_dataset", {})
if default.get("method") != "dense_no_l0" or default.get("sparse") is not False:
    raise SystemExit(f"dense default-dataset identity is wrong: {default}")
if default.get("epochs") != 3000:
    raise SystemExit(f"dense epoch receipt is wrong: {default.get('epochs')}")
if release.get("build", {}).get("selection_source") != {"enabled": False}:
    raise SystemExit("dense release unexpectedly used a selection source")
if build.get("code", {}).get("git_commit") != code_sha:
    raise SystemExit("dense build commit does not match launcher commit")
if build.get("ledger_artifact", {}).get("facts_sha256") != ledger_sha:
    raise SystemExit("dense build Ledger facts pin mismatch")
staging = build.get("staging", {})
if staging.get("enabled") is not False:
    raise SystemExit(f"dense build unexpectedly enabled staging: {staging}")
qrf_surface = qrf.get("surface", {})
if qrf.get("enforced") is not True:
    raise SystemExit("dense release did not enforce QRF tail concentration")
if qrf.get("tail_concentration", {}).get("passed") is not True:
    raise SystemExit("dense release did not pass QRF tail concentration")
if qrf_surface.get("reviewed_exclusions_file") is not None:
    raise SystemExit("dense release used a reviewed-exclusions file")
if qrf_surface.get("reviewed_exclusions_sha256") is not None:
    raise SystemExit("dense release recorded a reviewed-exclusions sha")
if qrf_surface.get("reviewed_exclusions") != {}:
    raise SystemExit("dense release used nonempty reviewed exclusions")
pool_h5 = pool_manifest.get("pool_h5", {})
pool_sha = pool_h5.get("sha256")
if not isinstance(pool_sha, str) or len(pool_sha) != 64:
    raise SystemExit("authenticated pool manifest lacks a pool H5 sha")
base_sha = diagnostics.get("build", {}).get("base_dataset_sha256")
if base_sha != pool_sha:
    raise SystemExit(
        f"dense release base dataset does not match pool: {base_sha} != {pool_sha}"
    )
print(f"DENSE RELEASE OK artifact_sha256={artifact_sha} release_id={release_id}")
PY
)" || die "dense release authentication failed"
  emit "$validation"
}

dense_is_complete() {
  local actual
  local recorded

  if [ -f "$DENSE_DONE" ]; then
    validate_dense_release
    actual="$(sha256_file "$DENSE_ARTIFACT")" || die "cannot hash completed dense artifact"
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$DENSE_DONE")"
    [ "$recorded" = "$actual" ] || die "dense completion marker disagrees with artifact recorded=$recorded actual=$actual"
    return 0
  fi
  if [ -f "$DENSE_RELEASE_MANIFEST" ] || [ -f "$DENSE_BUILD_MANIFEST" ]; then
    [ -f "$DENSE_RELEASE_MANIFEST" ] && [ -f "$DENSE_BUILD_MANIFEST" ] && [ -f "$DENSE_ARTIFACT" ] || die "partial certified dense manifest/artifact set requires owner inspection"
    validate_dense_release
    actual="$(sha256_file "$DENSE_ARTIFACT")" || die "cannot hash dense artifact while reconstructing marker"
    write_new_file "$DENSE_DONE" "$actual  $DENSE_ARTIFACT"
    emit "DENSE COMPLETION MARKER RECONSTRUCTED sha256=$actual"
    return 0
  fi
  return 1
}

record_dense_done() {
  local actual
  local recorded

  validate_dense_release
  actual="$(sha256_file "$DENSE_ARTIFACT")" || die "cannot hash dense artifact"
  if [ -f "$DENSE_DONE" ]; then
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$DENSE_DONE")"
    [ "$recorded" = "$actual" ] || die "dense completion marker changed recorded=$recorded actual=$actual"
  else
    write_new_file "$DENSE_DONE" "$actual  $DENSE_ARTIFACT"
  fi
  emit "DENSE ARTIFACT RECORDED sha256=$actual path=$DENSE_ARTIFACT"
}

set_sparse_release_id() {
  local candidate

  candidate="populace-us-2024-onesurface-pkg3-legacy-sparse-${CODE_SHA8}-${RUN_TS}"
  if [ "$DRY_RUN" -eq 1 ]; then
    SPARSE_RELEASE_ID="$candidate"
  elif [ -f "$SPARSE_RELEASE_ID_FILE" ]; then
    SPARSE_RELEASE_ID="$(/usr/bin/awk 'NF {print $1; exit}' "$SPARSE_RELEASE_ID_FILE")"
  else
    [ ! -f "$SPARSE_ARTIFACT" ] || die "sparse artifact exists without persisted release id path=$SPARSE_ARTIFACT"
    write_new_file "$SPARSE_RELEASE_ID_FILE" "$candidate"
    SPARSE_RELEASE_ID="$candidate"
  fi
  printf '%s\n' "$SPARSE_RELEASE_ID" | /usr/bin/grep -Eq '^populace-us-2024-onesurface-pkg3-legacy-sparse-[0-9a-f]{8}-[0-9]{8}T[0-9]{6}Z$' || die "invalid persisted sparse release id value=$SPARSE_RELEASE_ID"
  printf '%s\n' "$SPARSE_RELEASE_ID" | /usr/bin/grep -Eq "^populace-us-2024-onesurface-pkg3-legacy-sparse-${CODE_SHA8}-[0-9]{8}T[0-9]{6}Z$" || die "persisted sparse release id does not match pinned commit commit=$CODE_SHA value=$SPARSE_RELEASE_ID"
  SPARSE_RELEASE_DIR="$SPARSE_ROOT/releases/$SPARSE_RELEASE_ID"
  SPARSE_RELEASE_MANIFEST="$SPARSE_RELEASE_DIR/release_manifest.json"
  SPARSE_BUILD_MANIFEST="$SPARSE_RELEASE_DIR/build_manifest.json"
  emit "SPARSE RELEASE ID value=$SPARSE_RELEASE_ID state_file=$SPARSE_RELEASE_ID_FILE"
}

validate_sparse_release() {
  local validation

  validation="$("$PYTHON" - \
    "$SPARSE_ARTIFACT" \
    "$SPARSE_RELEASE_MANIFEST" \
    "$SPARSE_BUILD_MANIFEST" \
    "$SPARSE_RELEASE_DIR/qrf_tail_concentration.json" \
    "$SPARSE_RELEASE_DIR/calibration_diagnostics.json" \
    "$POOL_MANIFEST" \
    "$SPARSE_RELEASE_DIR/us_ssi_take_up.json" \
    "$SPARSE_RELEASE_ID" "$CODE_SHA" "$LEDGER_SHA" "$SPARSE_SSI_BASIS_SHA" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

(
    artifact,
    release_path,
    build_path,
    qrf_path,
    diagnostics_path,
    pool_manifest_path,
    ssi_path,
) = map(Path, sys.argv[1:8])
release_id, code_sha, ledger_sha, sparse_ssi_basis_sha = sys.argv[8:12]
for path in (
    artifact,
    release_path,
    build_path,
    qrf_path,
    diagnostics_path,
    pool_manifest_path,
    ssi_path,
):
    if not path.is_file():
        raise SystemExit(f"missing sparse release output: {path}")

digest = hashlib.sha256()
with artifact.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
artifact_sha = digest.hexdigest()
release = json.loads(release_path.read_text(encoding="utf-8"))
build = json.loads(build_path.read_text(encoding="utf-8"))
qrf = json.loads(qrf_path.read_text(encoding="utf-8"))
diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
ssi = json.loads(ssi_path.read_text(encoding="utf-8"))
if release.get("build", {}).get("build_id") != release_id:
    raise SystemExit("sparse release manifest build id mismatch")
entry = release.get("artifacts", {}).get("populace_us_2024", {})
if entry.get("sha256") != artifact_sha:
    raise SystemExit("sparse artifact hash does not match release manifest")
default = release.get("build", {}).get("default_dataset", {})
if default.get("method") != "l0_refit" or default.get("sparse") is not True:
    raise SystemExit(f"sparse default-dataset identity is wrong: {default}")
if default.get("l0_lambda_share") != 0.8:
    raise SystemExit(
        f"sparse L0 penalty share is not the owner-ruled default: {default.get('l0_lambda_share')}"
    )
if default.get("selection_epochs") != 6000 or default.get("refit_epochs") != 6000:
    raise SystemExit(
        "sparse epoch receipt is wrong: "
        f"selection={default.get('selection_epochs')} refit={default.get('refit_epochs')}"
    )
candidate_count = default.get("n_candidate_households")
selected_count = default.get("n_selected_households")
exported_count = default.get("n_exported_households")
for label, value in (
    ("candidate", candidate_count),
    ("selected", selected_count),
    ("exported", exported_count),
):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"sparse {label} household count is invalid: {value!r}")
if selected_count > candidate_count:
    raise SystemExit(
        f"legacy L0 selected more than its candidate pool: "
        f"selected={selected_count} candidate={candidate_count}"
    )
if exported_count != selected_count:
    raise SystemExit(
        f"sparse export count differs from realized L0 support: {exported_count} != {selected_count}"
    )
expected_lambda = 0.8 / candidate_count
if not math.isclose(
    float(default.get("l0_lambda", math.nan)),
    expected_lambda,
    rel_tol=1e-12,
    abs_tol=0.0,
):
    raise SystemExit(
        "sparse fixed L0 penalty does not equal default share / candidate count: "
        f"{default.get('l0_lambda')} != {expected_lambda}"
    )
if default.get("selection_l2_lambda") != 0.0 or default.get("refit_l2_lambda") != 0.0:
    raise SystemExit("sparse release used a tuned L2 penalty")
if release.get("build", {}).get("selection_source") != {"enabled": False}:
    raise SystemExit("sparse release unexpectedly used a selection source")
if build.get("calibration", {}).get("selection_source") != {"enabled": False}:
    raise SystemExit("sparse build manifest unexpectedly used a selection source")
if release.get("build", {}).get("warm_start_calibration") != {"enabled": False}:
    raise SystemExit("sparse release unexpectedly used warm-start calibration weights")
if build.get("calibration", {}).get("warm_start") != {"enabled": False}:
    raise SystemExit("sparse build unexpectedly used warm-start calibration weights")
if "exact_k_ladder" in release.get("build", {}) or "exact_k_ladder" in build:
    raise SystemExit("sparse release unexpectedly entered the exact-k ladder")
if build.get("dataset", {}).get("default") != default:
    raise SystemExit("sparse build and release default-dataset receipts differ")
if diagnostics.get("build", {}).get("default_dataset") != default:
    raise SystemExit("sparse diagnostics and release default-dataset receipts differ")
if any(
    str(row.get("name", "")).startswith("selection_mass_protection.")
    for row in diagnostics.get("targets", ())
    if isinstance(row, dict)
):
    raise SystemExit("sparse release injected frozen-selection mass protection")
if build.get("code", {}).get("git_commit") != code_sha:
    raise SystemExit("sparse build commit does not match launcher commit")
if build.get("ledger_artifact", {}).get("facts_sha256") != ledger_sha:
    raise SystemExit("sparse build Ledger facts pin mismatch")
expected_ssi_basis = {
    "kind": "release_artifact",
    "source_schema_version": 3,
    "source_sha256": sparse_ssi_basis_sha,
}
if ssi.get("prior_weight_basis") != expected_ssi_basis:
    raise SystemExit(
        "sparse SSI prior-weight basis receipt mismatch: "
        f"{ssi.get('prior_weight_basis')} != {expected_ssi_basis}"
    )
if ssi.get("schema_version") != 4 or ssi.get("measurement_phase") != "release_final":
    raise SystemExit("sparse SSI delivery artifact is not current-schema release-final")
if build.get("staging", {}).get("enabled") is not False:
    raise SystemExit(f"sparse build unexpectedly enabled staging: {build.get('staging')}")
if build.get("gates", {}).get("calibration", {}).get("passed") is not True:
    raise SystemExit("sparse release calibration gates did not pass")
qrf_surface = qrf.get("surface", {})
if qrf.get("enforced") is not True:
    raise SystemExit("sparse release did not enforce QRF tail concentration")
if qrf.get("tail_concentration", {}).get("passed") is not True:
    raise SystemExit("sparse release did not pass QRF tail concentration")
if qrf_surface.get("reviewed_exclusions_file") is not None:
    raise SystemExit("sparse release used a reviewed-exclusions file")
if qrf_surface.get("reviewed_exclusions_sha256") is not None:
    raise SystemExit("sparse release recorded a reviewed-exclusions sha")
if qrf_surface.get("reviewed_exclusions") != {}:
    raise SystemExit("sparse release used nonempty reviewed exclusions")
pool_h5 = pool_manifest.get("pool_h5", {})
pool_sha = pool_h5.get("sha256")
if not isinstance(pool_sha, str) or len(pool_sha) != 64:
    raise SystemExit("authenticated pool manifest lacks a pool H5 sha")
base_sha = diagnostics.get("build", {}).get("base_dataset_sha256")
if base_sha != pool_sha:
    raise SystemExit(
        f"sparse release base dataset does not match pool: {base_sha} != {pool_sha}"
    )
print(
    f"SPARSE RELEASE OK artifact_sha256={artifact_sha} release_id={release_id} "
    f"candidate_households={candidate_count} realized_households={selected_count}"
)
PY
)" || die "sparse release authentication failed"
  emit "$validation"
}

sparse_is_complete() {
  local actual
  local recorded

  if [ -f "$SPARSE_DONE" ]; then
    validate_sparse_release
    actual="$(sha256_file "$SPARSE_ARTIFACT")" || die "cannot hash completed sparse artifact"
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$SPARSE_DONE")"
    [ "$recorded" = "$actual" ] || die "sparse completion marker disagrees with artifact recorded=$recorded actual=$actual"
    return 0
  fi
  if [ -f "$SPARSE_RELEASE_MANIFEST" ] || [ -f "$SPARSE_BUILD_MANIFEST" ]; then
    [ -f "$SPARSE_RELEASE_MANIFEST" ] && [ -f "$SPARSE_BUILD_MANIFEST" ] && [ -f "$SPARSE_ARTIFACT" ] || die "partial certified sparse manifest/artifact set requires owner inspection"
    validate_sparse_release
    actual="$(sha256_file "$SPARSE_ARTIFACT")" || die "cannot hash sparse artifact while reconstructing marker"
    write_new_file "$SPARSE_DONE" "$actual  $SPARSE_ARTIFACT"
    emit "SPARSE COMPLETION MARKER RECONSTRUCTED sha256=$actual"
    return 0
  fi
  return 1
}

record_sparse_done() {
  local actual
  local recorded

  validate_sparse_release
  actual="$(sha256_file "$SPARSE_ARTIFACT")" || die "cannot hash sparse artifact"
  if [ -f "$SPARSE_DONE" ]; then
    recorded="$(/usr/bin/awk 'NF {print $1; exit}' "$SPARSE_DONE")"
    [ "$recorded" = "$actual" ] || die "sparse completion marker changed recorded=$recorded actual=$actual"
  else
    write_new_file "$SPARSE_DONE" "$actual  $SPARSE_ARTIFACT"
  fi
  emit "SPARSE ARTIFACT RECORDED sha256=$actual path=$SPARSE_ARTIFACT"
}

POOL_COMMAND=(
  "$PYTHON" "$POOL_TOOL"
  --sample-fraction 0.25
  --sample-seed 578
  --clone-attachment-fraction 1.0
  --clone-attachment-seed 578
  --asec-raw-stage-h5 "$ASEC_RAW"
  --asec-raw-stage-h5-sha256 "$ASEC_RAW_SHA"
  --acs-household-zip "$ACS_HOUSEHOLD"
  --acs-household-zip-sha256 "$ACS_HOUSEHOLD_SHA"
  --acs-person-zip "$ACS_PERSON"
  --acs-person-zip-sha256 "$ACS_PERSON_SHA"
  --acs-rent-h5 "$ACS_RENT"
  --acs-rent-h5-sha256 "$ACS_RENT_SHA"
  --puf-h5 "$PUF_H5"
  --puf-h5-sha256 "$PUF_H5_SHA"
  --puf-source-year-csv "$PUF_SOURCE"
  --puf-source-year-csv-sha256 "$PUF_SOURCE_SHA"
  --puma-ladder "$PUMA_LADDER"
  --puma-ladder-sha256 "$PUMA_LADDER_SHA"
  --congressional-district-vintage-crosswalk "$CD_CROSSWALK"
  --congressional-district-vintage-crosswalk-sha256 "$CD_CROSSWALK_SHA"
  --checkpoint-root "$POOL_CHECKPOINTS"
  --out "$POOL_H5"
)

trap 'cleanup_label $?' EXIT
if [ "$DRY_RUN" -eq 0 ]; then
  /bin/mkdir -p "$ROOT" "$POOL_ROOT" "$DENSE_ROOT" "$SPARSE_ROOT" || die "cannot create candidate output directories root=$ROOT"
fi

cd "$WT" || die "cannot cd to worktree path=$WT"
emit "RUN START mode=$([ "$DRY_RUN" -eq 1 ] && printf dry-run || printf execute) label=one-surface+pkg3,legacy-release-arm,not-exact-k-certified"
check_code_authority
pin_code_authority
check_immutable_inputs
set_dense_release_id
set_sparse_release_id
emit "OWNER RULING A ACTIVE sparse_path=legacy-cold-l0 default_lambda_share=0.8 realized_count=non-exact selection_source=none exact_k=none pi_hi=none keogh_mass_protection=omitted zero_operator_waivers=true"

DENSE_COMMAND=(
  "$PYTHON" "$RELEASE_TOOL"
  --base-h5 "$POOL_H5"
  --dense-default-dataset
  --ledger-facts "$LEDGER"
  --ledger-facts-sha256 "$LEDGER_SHA"
  --export-input-mass-reference-h5 "$EXPORT_REFERENCE"
  --asec-2023-weeks-unemployed-source "$ASEC_WEEKS"
  --scf-summary-extract "$SCF_SUMMARY"
  --scf-full-extract "$SCF_FULL"
  --sipp-tip-donor "$SIPP_TIPS"
  --sipp-vehicle-donor "$SIPP_FULL"
  --org-wages-donor "$ORG_WAGES"
  --ssi-take-up-prior-weight-basis "$SSI_BASIS"
  --ssi-take-up-prior-weight-basis-sha256 "$SSI_BASIS_SHA"
  --seed 0
  --epochs 3000
  --checkpoint-root "$DENSE_CHECKPOINTS"
  --release-id "$DENSE_RELEASE_ID"
  --out "$DENSE_ROOT"
  --skip-reform-validation
  --no-staging
)

SPARSE_COMMAND=(
  "$PYTHON" "$RELEASE_TOOL"
  --base-h5 "$POOL_H5"
  --ledger-facts "$LEDGER"
  --ledger-facts-sha256 "$LEDGER_SHA"
  --export-input-mass-reference-h5 "$EXPORT_REFERENCE"
  --asec-2023-weeks-unemployed-source "$ASEC_WEEKS"
  --scf-summary-extract "$SCF_SUMMARY"
  --scf-full-extract "$SCF_FULL"
  --sipp-tip-donor "$SIPP_TIPS"
  --sipp-vehicle-donor "$SIPP_FULL"
  --org-wages-donor "$ORG_WAGES"
  --ssi-take-up-prior-weight-basis "$SPARSE_SSI_BASIS"
  --ssi-take-up-prior-weight-basis-sha256 "$SPARSE_SSI_BASIS_SHA"
  --seed 0
  --epochs 6000
  --checkpoint-root "$SPARSE_CHECKPOINTS"
  --release-id "$SPARSE_RELEASE_ID"
  --out "$SPARSE_ROOT"
  --skip-reform-validation
  --no-staging
)
validate_sparse_command_contract

if [ "$DRY_RUN" -eq 1 ]; then
  emit "PRECONDITION PLAN stage=pool poll_seconds=300 need_reclaimable_gib=85 checks=no-pool-or-release-builder,AC-power,go-marker:$GO_MARKER"
  print_command "COMMAND stage=pool" /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "${POOL_COMMAND[@]}"
  emit "POOL MANIFEST PIN DEFERRED dynamic_output=$POOL_MANIFEST release-wrapper-will-record-and-consume-full-sha256"
  emit "PRECONDITION PLAN stage=release-dense poll_seconds=300 need_reclaimable_gib=85 checks=no-pool-or-release-builder,AC-power,go-marker:$GO_MARKER"
  print_command "COMMAND stage=release-dense" /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "${DENSE_COMMAND[@]}"
  emit "PRECONDITION PLAN stage=release-sparse poll_seconds=300 need_reclaimable_gib=85 checks=no-pool-or-release-builder,AC-power,go-marker:$GO_MARKER"
  print_command "COMMAND stage=release-sparse" /usr/bin/time -l /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "${SPARSE_COMMAND[@]}"
else
  recheck_code_authority
  if pool_is_complete; then
    emit "STAGE SKIP stage=pool reason=validated-output-and-gates path=$POOL_H5"
  else
    while :; do
      wait_ready pool 85
      recheck_code_authority
      check_pool_inputs
      recheck_code_authority
      ready_now pool 85 && break
    done
    if pool_is_complete; then
      emit "STAGE SKIP stage=pool reason=validated-output-completed-while-waiting path=$POOL_H5"
    else
      emit "STAGE START stage=pool checkpoint_root=$POOL_CHECKPOINTS"
      run_monitored pool "$POOL_LOG" "$POOL_RSS" "${POOL_COMMAND[@]}" || die "pool stage failed; checkpoints retained for idempotent retry"
      validate_pool_outputs
      record_pool_manifest_pin
    fi
  fi

  consume_pool_manifest_pin release-dense
  recheck_code_authority
  if dense_is_complete; then
    emit "STAGE SKIP stage=release-dense reason=validated-artifact-and-manifests path=$DENSE_ARTIFACT"
  else
    while :; do
      wait_ready release-dense 85
      recheck_code_authority
      check_dense_inputs
      consume_pool_manifest_pin release-dense
      recheck_code_authority
      ready_now release-dense 85 && break
    done
    if dense_is_complete; then
      emit "STAGE SKIP stage=release-dense reason=validated-output-completed-while-waiting path=$DENSE_ARTIFACT"
    else
      emit "STAGE START stage=release-dense release_id=$DENSE_RELEASE_ID checkpoint_root=$DENSE_CHECKPOINTS"
      run_monitored release-dense "$DENSE_LOG" "$DENSE_RSS" "${DENSE_COMMAND[@]}" || die "dense stage failed; checkpoints and release id retained for idempotent retry"
      record_dense_done
    fi
  fi

  consume_pool_manifest_pin release-sparse
  recheck_code_authority
  if sparse_is_complete; then
    emit "STAGE SKIP stage=release-sparse reason=validated-artifact-and-manifests path=$SPARSE_ARTIFACT"
  else
    while :; do
      wait_ready release-sparse 85
      recheck_code_authority
      check_sparse_inputs
      consume_pool_manifest_pin release-sparse
      recheck_code_authority
      ready_now release-sparse 85 && break
    done
    if sparse_is_complete; then
      emit "STAGE SKIP stage=release-sparse reason=validated-output-completed-while-waiting path=$SPARSE_ARTIFACT"
    else
      emit "STAGE START stage=release-sparse release_id=$SPARSE_RELEASE_ID checkpoint_root=$SPARSE_CHECKPOINTS"
      run_monitored release-sparse "$SPARSE_LOG" "$SPARSE_RSS" "${SPARSE_COMMAND[@]}" || die "sparse stage failed; checkpoints and release id retained for idempotent retry"
      record_sparse_done
    fi
  fi
fi

emit "INCUMBENT EVIDENCE path=$INCUMBENT_EVIDENCE sha256=$INCUMBENT_EVIDENCE_SHA (evidence only; scorer consumes the H5)"
if [ "$DRY_RUN" -eq 1 ]; then
  emit "DRY-RUN COMPLETE no pool/release builder, scorer, publication, promotion, staging, or launchd mutation ran"
  emit "DENSE ARTIFACT planned_path=$DENSE_ARTIFACT sha256=pending-stage-2a"
  emit "SPARSE ARTIFACT planned_path=$SPARSE_ARTIFACT sha256=pending-stage-2b"
else
  emit "RUN COMPLETE dense and owner-ruling-A sparse legacy candidates built or authenticated"
  emit "DENSE ARTIFACT path=$DENSE_ARTIFACT sha256=$(sha256_file "$DENSE_ARTIFACT")"
  emit "SPARSE ARTIFACT path=$SPARSE_ARTIFACT sha256=$(sha256_file "$SPARSE_ARTIFACT")"
fi
print_command "SCORER COMMAND dense" /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "$PYTHON" "$SCORER_TOOL" \
  --incumbent "$INCUMBENT_H5" \
  --candidate "$DENSE_ARTIFACT" \
  --ledger-facts "$LEDGER" \
  --out-prefix "$ROOT/scores/dense-head-to-head"
print_command "SCORER COMMAND sparse" /usr/bin/env -u POPULACE_LOGBOOK_PREV_ROW_DIGEST "$PYTHON" "$SCORER_TOOL" \
  --incumbent "$INCUMBENT_H5" \
  --candidate "$SPARSE_ARTIFACT" \
  --ledger-facts "$LEDGER" \
  --out-prefix "$ROOT/scores/sparse-head-to-head"
