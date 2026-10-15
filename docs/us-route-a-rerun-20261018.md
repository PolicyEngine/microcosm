# Route A rerun on 18 October 2026 (d1386)

Prepared on 15 October against cached `origin/main`
`412f916536b66577e4e0b42c707285085df7c302`. Fetching was denied by this
workspace's sandbox; these findings must be checked against the commit selected
on 18 October. Nothing in this preparation launched a build or called Modal.

Max's d1386 selects whatever has merged to `main` on 18 October. It must contain
`9802825200929c445856d969f4231287c185d784` (#1186, the #1178 Social Security
fix). The base runs on Modal; release, preflights and certification run locally
through `tools/route_a/`. Staging is enabled. Publication remains separate.

The assigned checkout is an older detached commit and its shared Git metadata
is read-only. Preparation therefore used an ignored, extracted cached-main tree
inside the assigned workspace, with independent local Git metadata. The portable
patch/bundle is against that main snapshot; apply it to a current-main prep
worktree before running this sequence. No caller metadata was written.
The requested external settings directory was also outside the writable
sandbox. The prepared, uncommitted settings fallback is
`build/route-a-rerun-20261018/route_a.env`; the reproducible template is
`docs/us-route-a-rerun-20261018.env.example`. Install it at the requested
machine-local path in step 1. Never commit `route_a.env`.

## Inputs and registration

The new plan is `docs/us-modal-stage-route-a-rerun-base-plan.json`.
`PICK_MAIN_COMMIT_ON_2026_10_18` is deliberately not a SHA: existing `validate`
returns 2 while it remains. Step 1 writes a concrete machine-local plan without
editing the template. All eleven inputs use the runbook's
`volume://cas/sha256/<digest>/<basename>` layout on
`microcosm-us-stage-inputs`; the volume name is not part of the URI.

| Input or setting | Change since `4b57d15a2` |
| --- | --- |
| Block ladder | d713 CT-fixed `6840b990acdfa2003d7723da5595e3cfff7205a4fa3d8d6206ed835c85c3f233`, 18,991,218 bytes; replaces `7ba39b95…` |
| ASEC H5s, income years 2024/2023/2022 | Same pins as the historical plan, in that order |
| ASEC education, income years 2022/2023/2024 | Same three archive pins; the first also seeds the unemployment-weeks cache |
| ASEC 2025 | Main added it to general defaults; Route A explicitly supplies 2024/2023/2022, so no fourth source is added |
| Processed PUF, TY2015 raw PUF, ACS 2022 | Same pins as the historical plan |
| Base feed | Retains `consumer_facts_us_c5e5bf8.jsonl`, SHA `b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801` |
| Release feed | Main repinned under #940 to `consumer_facts_us_51aa40f.jsonl`, SHA `965a29ac9458edb58a05184917ee4a19772fd3dc7b4588db6eb6da5803641aac`, 169,038,757 bytes, 40,178 rows; the machine file was verified |
| In-tree district crosswalk | NC correction changed `c7cb040b…` to `a347303fff3fea145f758488c43cb94355df5a8cb5e512552b2ee3791d2aa233`, 78,202 bytes |

The release feed retains all 39,158 old facts and adds 1,020 TY2023 state AGI
facts; see `us-chronicle-feed-repin.md`. Base district counts are unchanged,
so retaining its old feed is intentional. `BASE_LEDGER_FACTS` and its SHA are
explicitly independent from release `FEED`, `FEED_SHA` and `FEED_BYTES`.
The base builder itself is unchanged since `4b57d15a2`; runtime code has changed,
including the raw-PUF partnership self-employment correction. Identical input
pins do not imply identical outputs.

`us-puf-support-base-rerun` preserves the historical base registration's argv,
inputs, cache seed, hash-seed handling, mirroring and resources, and pins the
current crosswalk. The historical `us-puf-support-base` registration and its
plan/reference remain unchanged. Both refuse an unrecognized crosswalk. The
driver's feed and crosswalk expectations are configurable, with historical
defaults; this rerun's settings explicitly pin the new values. If main changes
any pin again, review and update the affected settings/registration before
launching; do not suppress a check or substitute unreviewed bytes.

The base retains non-preemptible placement, four requested/limited CPUs,
112 GiB memory, 14,400 seconds of tool budget and a 16,500-second class timeout.
The planner estimates $14.63 including reserve and $14.90 at the timeout and
request, below the $15 base cap. As the existing runbook explains, this is not
an absolute billed ceiling: memory above the request, checks, startup, image
builds and storage add cost. A second paid attempt needs Max's go. No resource
limit or cap interpretation is changed here.

## Lineage decision

Recommend a fresh local reference from **only named stages
`source_construction` and `pre_clone_enrichment`**, on the selected clean commit
and CT ladder, followed by strict `compare-lineage`. These stages add no Modal
cost. The retained local interrupted attempt's
`overnight-20260923/route-a/run-4b57d15a287c/base-sup.failed-1790651711/RESULT.json`
reports 2,054.6823 seconds elapsed and 12,516,982,784 bytes peak RSS (11.66 GiB),
after 000/001 completed. These are whole-attempt measurements, not isolated
000/001 timings; no retained local stage profile permits finer attribution.
Its two compared checkpoint files total 1,485,203,161 bytes. The current source
stage also writes an auxiliary raw-ASEC checkpoint; that total is not the full
local disk requirement. Allow RAM/disk headroom and
use supervision; shared free-disk decline cannot be attributed to those stages.

The old Modal receipt reports 6,234.9 seconds tool wall, 6,281.6 seconds container
wall, 70.69 GiB peak RSS and a $5.67 list-price estimate for the completed old
base. These observations do not bound this rerun's time or cost.

A new hand-off requires a COMPLETED, non-budget-stopped receipt, strict
`base-out` verification, the actual Modal context matching its receipt digest,
and a reference with the chosen commit, all eleven input digests, normalized
all-stage argv/environment, pipeline, run config, source-code digest,
dependency versions, hash seed 0 and actual 000/001 checkpoint digests. Every
comparison must pass. Interpreter and other thread-variable differences are
reported; only hash seed is compared among thread variables. Equality covers
two stages, not the later 22, district/ladder assignment or final export.
Release gates and certification still run.

Do not relabel the old reference with a new commit/ladder or derive a local
reference from Modal outputs. `make_local_reference.py` reads the actual local
context/checkpoints and validates their inputs and source identity. Keep its
JSON and the supporting local evidence with this run.

The old `base-sup/PROVENANCE.md` records a strict mismatch at stage 001:
313/432,523 `person.pre_subsidy_rent` values differed, maximum absolute
`8.7e-11`, relative `6.08e-15`. A prior session used a narrow exception. That
does not authorize this run. Any mismatch, skipping the new local reference,
or relaxing the hand-off needs **Max's explicit decision** with the new
evidence. There is no automatic waiver path in this preparation. Stop before
writing `ACCEPTED` while that decision is outstanding.

## Settings to review on 18 October

- `COMMIT`: empty until selected from freshly fetched main; must include #1186.
- `FEED`, `FEED_SHA`, `FEED_BYTES`, `BASE_LEDGER_FACTS`, its SHA and
  `CROSSWALK_SHA`: recheck against the selected tree. The two feeds differ on
  purpose. ASEC/PUF/ACS/education pins must also remain compatible.
- `PREFLIGHT_COMMIT` and `PREFLIGHT_NEW_LINEAGE_ARGS`: empty commit defaults to
  the build commit; verify its parser and gates still accept `--new-lineage`.
- `EXPORT_MASS_REF_H5`: inherited
  `$PE/_buildg-runtime/forensics/populace_us_2024.h5` (354,374,956 bytes).
  Review its suitability and digest as the comparison reference for this run.
  The driver does not include this file in `inputs.resolved.json`; record its
  digest separately. The release must retain its export-mass gate.
- `QRF_TAIL_EXCLUSIONS` and `_SHA`: the d490 historical register is provisional
  for this lineage. Re-measure and update the file and SHA together. Its four
  columns are `bond_assets`, `estate_income`,
  `long_term_capital_gains_on_collectibles`, and
  `w2_wages_from_qualified_business`. Under d490 a new measurement may retain
  threshold **0.75** and the same or a subset of these columns without another
  ruling. Drop stale entries. Adding a column or lowering the threshold needs
  Max's ruling. Its old reasons and expiry do not attest a new export. The
  pre-solve dry run uses base weights; the final weighted-export gate still
  determines whether the register is current.
- `DENSE_RELEASE_D122` and `RELEASE_EXTRA_ARGS`: retain the historical dense
  whole-base, national/state target surface and batch size 2000. The original
  d122 text described the first release; confirm this remains the intended
  configuration for the rerun.
- `RELEASE_RAM_GIB=50`, `RELEASE_WALL_SECONDS=345600`,
  `RESOURCE_WAIT_HOURS=12` and batch size 2000: inherited admission/budget
  settings, unmeasured on the new commit and current load. The 110 GiB RSS
  cap remains enforced.
- Staging: `ROUTE_A_STAGING=1`; destination explicitly
  `policyengine/populace-us-staging`. Check runtime helper access and dashboard
  delivery on launch day. Do not fetch/store credentials during preparation.
- Small-H5 guard fixture: choose a real small written export with enough
  households for at least two batches. Do not substitute a full export for it.

## Exact launch-day sequence

Run the blocks in **Bash**, with `set -euo pipefail`, from a prep checkout
containing these changes. `PREP` is that checkout; `W` below is a separate,
clean checkout of the chosen build commit. Nothing writes to the old run or
`microcosm-route-a-4b57d15a287c` worktree. There is no automatic date scheduler.

### 1. Select main, install settings and fill the plan

```bash
set -euo pipefail
PE=/Users/maxghenis/PolicyEngine
MAIN=$PE/microcosm
PREP=$PE/_worktrees/route-a-rerun-prep
LOCAL=$PE/_recovered/scratch-backup/893/route-a-rerun-20261018
SETTINGS=$LOCAL/route_a.env
PLAN=$LOCAL/base-plan.json
LSUBMIT=$PE/_recovered/scratch-backup/893/bin/lsubmit.sh
cd "$PREP"
git -C "$MAIN" fetch origin
C=$(git -C "$MAIN" rev-parse 'origin/main^{commit}')
git -C "$MAIN" merge-base --is-ancestor 9802825200929c445856d969f4231287c185d784 "$C"
mkdir -p "$LOCAL"
# Once only: preserve any existing operator edits.
test -e "$SETTINGS" || cp docs/us-route-a-rerun-20261018.env.example "$SETTINGS"
python3 - "$MAIN" "$C" "$PLAN" "$SETTINGS" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

main, commit, output, settings = sys.argv[1:]
def shown(path):
    return subprocess.check_output(["git", "-C", main, "show", f"{commit}:{path}"])

feed = json.loads(shown("packages/microcosm-build/src/microcosm/build/us/chronicle_feed.json"))
if feed["facts_sha256"] != "965a29ac9458edb58a05184917ee4a19772fd3dc7b4588db6eb6da5803641aac":
    raise SystemExit("REFUSED: main changed the release feed; review its pins/settings")
crosswalk = "packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv"
if hashlib.sha256(shown(crosswalk)).hexdigest() != "a347303fff3fea145f758488c43cb94355df5a8cb5e512552b2ee3791d2aa233":
    raise SystemExit("REFUSED: main changed the crosswalk; review the rerun registration")
plan = json.loads(Path("docs/us-modal-stage-route-a-rerun-base-plan.json").read_text())
plan["source"]["commit"] = commit
plan["run_id"] = f"route-a-rerun-base-20261018-{commit[:12]}"
text = Path(settings).read_text()
if ': "${COMMIT:=}"' not in text:
    raise SystemExit("REFUSED: settings already pinned; review before replacing COMMIT")
if Path(output).exists():
    raise SystemExit("REFUSED: concrete plan already exists; preserve the attempt identity")
Path(output).write_text(json.dumps(plan, indent=2) + "\n")
Path(settings).write_text(text.replace(': "${COMMIT:=}"', ': "${COMMIT:=' + commit + '}"'))
print(commit)
PY
python3 tools/modal_us_stage_plan.py validate "$PLAN" > "$LOCAL/plan-validation.json"
# Review SETTINGS and the uncertainties above before proceeding.
. "$SETTINGS"
test "$COMMIT" = "$C"
test "$ROUTE_A_STAGING" = 1
shasum -a 256 "$EXPORT_MASS_REF_H5" > "$LOCAL/export-mass-reference.sha256"
```

Use a fresh shell or clear stale Route A environment overrides: defaults in
the settings file intentionally let nonempty environment values win. `lsubmit`
does not inherit them, so persist every intended change in the settings file.

### 2. Engine-free checks, resolve and missing input uploads

```bash
uv run pytest -n 4 -rs \
  packages/microcosm-build/tests/engine_free/us/test_us_modal_stage_plan_tool.py \
  packages/microcosm-build/tests/engine_free/us/test_us_modal_stage_puf_support_base.py \
  packages/microcosm-build/tests/engine_free/us/test_route_a_driver.py \
  packages/microcosm-build/tests/engine_free/us/test_us_route_a_local_reference.py
uv run ruff check .
uv run ruff format --check tools/modal_us_stage_plan.py tools/route_a/make_local_reference.py \
  packages/microcosm-build/tests/engine_free/us/test_route_a_driver.py \
  packages/microcosm-build/tests/engine_free/us/test_us_modal_stage_puf_support_base.py \
  packages/microcosm-build/tests/engine_free/us/test_us_route_a_local_reference.py
python3 tools/ci_test_plan.py verify
ROUTE_A_ENV="$SETTINGS" bash tools/route_a/route_a.sh --resolve | tee "$LOCAL/resolve-initial.log"
# Resolve exit 0 alone is insufficient: it can report commit blockers.
if rg 'BLOCKER' "$LOCAL/resolve-initial.log"; then exit 1; fi
# Inspect inputs.resolved.json; every registered file must be verified.
python3 tools/modal_us_stage_plan.py upload-commands "$PLAN" --shell \
  "asec_2024_h5=$STORAGE/census_cps_2024.h5" \
  "asec_2023_h5=$STORAGE/census_cps_2023.h5" \
  "asec_2022_h5=$STORAGE/census_cps_2022.h5" \
  "puf_2024_h5=$STORAGE/puf_2024.h5" "puf_2015_csv=$STORAGE/puf_2015.csv" \
  "acs_2022_h5=$STORAGE/acs_2022.h5" \
  "asec_education_2022_zip=$EDU/asecpub23csv.zip" \
  "asec_education_2023_zip=$EDU/asecpub24csv.zip" \
  "asec_education_2024_zip=$EDU/asecpub25csv.zip" \
  "base_ledger_facts=$BASE_LEDGER_FACTS" "block_ladder_npz=$LADDER" \
  > "$LOCAL/upload-base-inputs.sh"
# Hashes are checked before this script is emitted; it skips existing CAS files.
# The CT upload may already have completed; do not overwrite unrelated paths.
bash "$LOCAL/upload-base-inputs.sh"
MODAL=$(command -v modal)
MICROCOSM_MODAL_PLAN="$PLAN" "$MODAL" run tools/modal_us_stage.py \
  > "$LOCAL/modal-check.log" 2>&1
```

Preparation also ran repository-wide `ruff format --check .` with locked Ruff
0.15.16: 45 unchanged cached-main files fail that check. All five changed Python
files pass; the scoped launch-day command above checks them without rewriting
unrelated sources or their identities. Reassess this baseline after fetching.

Read the check report: no problems, branch/source/crosswalk/eleven inputs
verified, disk admission, write probe and hash seed accepted. A failed check
blocks the paid launch. Check all five pinned-commit tests passed
instead of skipped in the full-history checkout. No Modal command is run as
part of preparing this document.

### 3. Launch the Modal base under launchd

```bash
"$LSUBMIT" route-a-rerun-20261018-base "$LOCAL/base.launchd.log" "$PREP" \
  /usr/bin/env MICROCOSM_MODAL_PLAN="$PLAN" \
  "$MODAL" run --detach tools/modal_us_stage.py --run
```

Keep launchd's default `ProcessType=Standard`; do not set `KEEPALIVE=1`.
Use the same reviewed plan for this one paid attempt. A failure or budget stop
does not authorize an automatic relaunch.

### 4. Produce the local reference (scheduled, supervised local work)

This step runs engines and must wait for machine admission. It was not run
during preparation. Create the clean chosen-commit checkout and locked US env.
The script below supervises each named stage with the existing local base
limits: 100 GiB RSS cap, 80 GiB RAM/70 GiB disk admission, six-hour wall limit
and persistent 20 GiB disk floor. Admission refusal needs a later operator
retry; it does not silently run on the overloaded machine. It never invokes
`--stage all` locally.

```bash
W=$WT_ROOT/microcosm-route-a-${C:0:12}
if [ ! -d "$W" ]; then git -C "$MAIN" worktree add --detach "$W" "$C"; fi
test "$(git -C "$W" rev-parse HEAD)" = "$C"
test -z "$(git -C "$W" status --porcelain)"
(cd "$W" && "$UV" sync --all-packages --locked --extra us)
REFROOT=$LOCAL/local-reference
REF=$LOCAL/base-local-reference.json
mkdir -p "$REFROOT/base-checkpoints"
BASE_ARGS=(--checkpoint-dir "$REFROOT/base-checkpoints"
  --asec-h5 "2024=$STORAGE/census_cps_2024.h5"
  --asec-h5 "2023=$STORAGE/census_cps_2023.h5"
  --asec-h5 "2022=$STORAGE/census_cps_2022.h5"
  --asec-h5-sha256 2024=ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d
  --asec-h5-sha256 2023=cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88
  --asec-h5-sha256 2022=7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e
  --puf-h5 "$STORAGE/puf_2024.h5" --puf-source-year-csv "$STORAGE/puf_2015.csv"
  --acs-h5 "$STORAGE/acs_2022.h5"
  --asec-education-source "2022=$EDU/asecpub23csv.zip"
  --asec-education-source "2023=$EDU/asecpub24csv.zip"
  --asec-education-source "2024=$EDU/asecpub25csv.zip"
  --target-year 2024 --seed 0 --n-estimators 32 --ledger-facts "$BASE_LEDGER_FACTS"
  --assign-congressional-districts
  --congressional-district-vintage-crosswalk "$W/packages/microcosm-build/src/microcosm/build/us_runtime/data/congressional_district_vintage_crosswalk.csv"
  --congressional-district-seed 0 --block-ladder-artifact "$LADDER"
  --out "$REFROOT/base-out")
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  declare -p W PREP PLAN REFROOT REF BASE_ARGS
  cat <<'BASH'
for stage in source_construction pre_clone_enrichment; do
  python3 - "$REFROOT/$stage-config.json" "$W" "$stage" "${BASE_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path
output, worktree, stage, *args = sys.argv[1:]
gib = 1024 ** 3
config = {
    "cwd": worktree,
    "argv": [worktree + "/.venv/bin/python", "-B", "tools/build_us_puf_support_base.py", *args, "--stage", stage],
    "env": {"PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"},
    "authorization": "d1386: local 000/001 comparison evidence only",
    "limits": {"wall_seconds": 21600, "cpu_seconds": 64800,
               "rss_bytes": 100*gib, "available_ram_admission_bytes": 80*gib,
               "disk_admission_bytes": 70*gib, "disk_floor_bytes": 20*gib,
               "disk_floor_persist_seconds": 180,
               "output_bytes": 2*gib, "log_bytes": 512*1024**2},
}
with Path(output).open("x") as stream:
    json.dump(config, stream, indent=2)
PY
  "$W/.venv/bin/python" "$PREP/tools/route_a/supervise.py" \
    "$REFROOT/$stage-sup" "$REFROOT/$stage-config.json"
done
python3 "$PREP/tools/route_a/make_local_reference.py" --plan "$PLAN" \
  --checkpoint-root "$REFROOT/base-checkpoints" --repo-root "$W" --out "$REF"
BASH
} > "$LOCAL/local-reference.sh"
"$LSUBMIT" route-a-rerun-20261018-reference "$LOCAL/reference.launchd.log" "$PREP" \
  /bin/bash "$LOCAL/local-reference.sh"
```

Wait for this job to finish successfully and `$REF` to exist before step 5.
Retain the per-stage supervisor records. On refusal/failure, inspect them and
move only this attempt's failed supervisor/config files aside before retrying;
the two stages' completed-prefix evidence remains in the fresh reference root.

### 5. Fetch, compare and hand off

After the Modal base completes, fetch only the receipt, output and context.
Check the CLI's downloaded directory layout before selecting `RECEIPT`.

```bash
R=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$PLAN")
FETCH=$LOCAL/modal-runs/$R
mkdir -p "$FETCH/state"
"$MODAL" volume get microcosm-us-stage-runs "runs/$R/receipts" "$FETCH/"
"$MODAL" volume get microcosm-us-stage-runs "runs/$R/state/base-out" "$FETCH/state/"
"$MODAL" volume get microcosm-us-stage-runs \
  "runs/$R/state/base-checkpoints/stage_run_context.json" "$FETCH/"
RECEIPT=$(python3 - "$FETCH" <<'PY'
import sys
from pathlib import Path
paths = list((Path(sys.argv[1]) / "receipts").glob("all-*.json"))
if len(paths) != 1:
    raise SystemExit("REFUSED: expected one attempt receipt; inspect all receipts manually")
print(paths[0])
PY
)
CONTEXT=$FETCH/stage_run_context.json
LINEAGE=$FETCH/compare-lineage.json
RUN=$RUN_ROOT/run-${C:0:12}
(
  set -euo pipefail
  test ! -e "$RUN/base-out" && test ! -e "$RUN/base-sup"
  python3 tools/modal_us_stage_plan.py verify-receipt "$RECEIPT" \
    --state-root "$FETCH/state" --prefix base-out --strict
  python3 tools/modal_us_stage_plan.py compare-lineage "$RECEIPT" \
    --reference "$REF" --run-context "$CONTEXT" > "$LINEAGE"
  mkdir -p "$RUN/base-out"
  cp -p "$FETCH/state/base-out/"* "$RUN/base-out/"
  python3 tools/modal_us_stage_plan.py verify-receipt "$RECEIPT" \
    --state-root "$RUN" --prefix base-out --strict
  H5=$RUN/base-out/base_populace_us_2024_puf_support.h5
  shasum -a 256 "$H5" > "$RUN/base.sha256"
  if [ -e "$RUN/base-checkpoints" ]; then
    mv "$RUN/base-checkpoints" "$RUN/base-checkpoints.local-$(date +%s)"
  fi
  mkdir -p "$RUN/base-sup"
  cp "$RECEIPT" "$RUN/base-sup/modal-receipt.json"
  cp "$CONTEXT" "$RUN/base-sup/modal-stage_run_context.json"
  cp "$LINEAGE" "$RUN/base-sup/compare-lineage.json"
  cp "$REF" "$RUN/base-sup/local-reference.json"
  RECEIPT_SHA=$(shasum -a 256 "$RECEIPT" | cut -c1-64)
  printf '{\n "status": "MODAL_RECEIPT",\n "returncode": 0,\n "refusal": null,\n "receipt_sha256": "%s",\n "compare_lineage": "compare-lineage.json"\n}\n' \
    "$RECEIPT_SHA" > "$RUN/base-sup/RESULT.json"
  echo "returncode=0 $(date '+%F %T') modal receipt sha256 $RECEIPT_SHA" \
    > "$RUN/base-sup/ACCEPTED"
)
```

`ACCEPTED` is last. If any comparison fails, stop and retain the evidence for
Max; do not write the marker. A partly executed copy block needs inspection
before any retry. Do not remove old run evidence.

### 6. Release guards, register review and release under launchd

Regenerate the maintained eight-household, five-state fixture into this run's
scratch, then use it as `SMALL_H5`. The old guard fixture no longer exists. Its
retained PR4 round-2 report measured the sweep at 6,351.77 seconds (106 minutes),
4.75 GiB peak RSS and 28 serial workers, so schedule that check too. From `W`, run
the complete guard sweep and the release's pre-solve register dry run before
starting the driver. These engine checks are deliberately deferred until launch
day. The dry run's inherited d490 register is a candidate to grade, not proof
that it applies to the new weighted export. Resolve any certain failure and
review AT-RISK results before release.

```bash
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  declare -p W LOCAL RUN SETTINGS
  cat <<'BASH'
. "$SETTINGS"
(cd "$W" && .venv/bin/python - "$LOCAL" <<'PY'
import sys
from pathlib import Path
from tools.sweep_us_post_export_scoring import _load_builder
from test_support.microcosm_build.us_post_export_scoring import (
    _GUARD_HOUSEHOLDS, _write_engine_h5,
)
_write_engine_h5(_load_builder(), Path(sys.argv[1]) / "guard-fixture", _GUARD_HOUSEHOLDS)
PY
)
SMALL_H5=$LOCAL/guard-fixture/populace_us_2024.h5
(cd "$W" && .venv/bin/python tools/sweep_us_post_export_scoring.py \
  --dataset-path "$SMALL_H5" --batch-size 3 --reforms-per-worker 4 \
  --output "$LOCAL/guard-sweep.json")
EXTRA=($RELEASE_EXTRA_ARGS)
gate_rc=0
(cd "$W" && .venv/bin/python tools/dry_run_us_release_gates.py \
  --json-out "$LOCAL/dry-run-gates.json" -- \
  --base-h5 "$RUN/base-out/base_populace_us_2024_puf_support.h5" \
  --ledger-facts "$FEED" --ledger-facts-sha256 "$FEED_SHA" \
  --qrf-tail-concentration-exclusions "$QRF_TAIL_EXCLUSIONS" \
  --ssi-take-up-prior-weight-basis "$SSI" \
  --ssi-take-up-prior-weight-basis-sha256 25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a \
  --scf-summary-extract "$SCF" --out "$LOCAL/gate-dry-run" \
  --release-id route-a-rerun-20261018-gate-check \
  --checkpoint-root "$LOCAL/gate-checkpoints" --seed 0 --no-staging "${EXTRA[@]}") || gate_rc=$?
case "$gate_rc" in
  0) echo 'Release guards completed cleanly' ;;
  2) echo 'AT-RISK: review dry-run-gates.json before starting release'; exit 2 ;;
  *) echo 'Release guard dry run failed'; exit "$gate_rc" ;;
esac
BASH
} > "$LOCAL/release-checks.sh"
"$LSUBMIT" route-a-rerun-20261018-checks "$LOCAL/checks.launchd.log" "$PREP" \
  /bin/bash "$LOCAL/release-checks.sh"
```

Wait for the checks job to finish. Inspect its exit status and both reports;
an exit 2 requires operator review of AT-RISK findings before continuing.
A failed sweep or a certain gate failure blocks release. Then start the driver:

```bash
# If settings/register changed during review, source again and resolve again.
. "$SETTINGS"
ROUTE_A_ENV="$SETTINGS" bash tools/route_a/route_a.sh --resolve | tee "$LOCAL/resolve-release.log"
if rg 'BLOCKER' "$LOCAL/resolve-release.log"; then exit 1; fi
test -f "$RUN/base-sup/ACCEPTED"
"$LSUBMIT" route-a-rerun-20261018-release "$LOCAL/release.launchd.log" "$PREP" \
  /usr/bin/env ROUTE_A_ENV="$SETTINGS" /bin/bash "$PREP/tools/route_a/route_a.sh"
```

`lsubmit.sh` passes HOME/PATH and explicit job variables only. The `/usr/bin/env`
arguments above are necessary for plan/settings delivery. The driver skips the
accepted Modal base and runs prefetch, base preflight, release with staging,
release preflight and offline publisher preflight. There is no release-only
flag. Confirm staging telemetry arrives and retain the local hand-off and final
gate evidence. A `ROUTE_A_DONE.json` is a candidate hand-off with
`published: false`; publication requires its separate decision.
