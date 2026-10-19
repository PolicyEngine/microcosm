# Route A rerun on 18 October 2026 (d1386)

Prepared on 15 October against `origin/main`
`412f916536b66577e4e0b42c707285085df7c302`. Nothing in this preparation
launched a build or called Modal. The 18 October start did not happen; Max
confirmed on 19 October that the build launches that day on current `main`
without #1275 (d1386, d1556).

Max's d1386 selects whatever has merged to `main` at launch. It must contain
`9802825200929c445856d969f4231287c185d784` (#1186, the #1178 Social Security
fix). The base runs on Modal; release, preflights and certification run locally
through `tools/route_a/`. Staging is enabled. Publication remains separate.

The machine-local settings file is
`$PE/_recovered/scratch-backup/893/route-a-rerun-20261018/route_a.env`, from
`docs/us-route-a-rerun-20261018.env.example`. Never commit `route_a.env`.

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
and a reference with the chosen commit, all eleven input digests, pipeline, run
config, source-code digest, dependency versions, hash seed 0 and actual 000/001
checkpoint digests. Every comparison must pass. The reference's argv and
environment are written from the plan's registry (`planned_argv(plan)`), not
read from the local run, so their comparison only confirms the registry; the
strict masked `run_config`, source and dependency identity, hash seed,
pipeline and checkpoint digests carry the evidence. Interpreter and other thread-variable differences are
reported; only hash seed is compared among thread variables. Equality covers
two stages, not the later 22, district/ladder assignment or final export.
Release gates and certification still run.

Do not relabel the old reference with a new commit/ladder or derive a local
reference from Modal outputs. `make_local_reference.py` reads the actual local
context/checkpoints and validates their inputs and source identity. Keep its
JSON and the supporting local evidence with this run.

The old `base-sup/PROVENANCE.md` records a strict mismatch at stage 001:
313/432,523 `person.pre_subsidy_rent` values differed, maximum absolute
`8.7e-11`, relative `6.08e-15`. A prior session accepted it with a scripted check.
Max ruled on 19 October (d1712) that a repeat of exactly that class may be
accepted for this run: every entity table has the same shape and index, totals
are equal, and every differing value is within 1e-12 relative. Step 5b runs
`tools/route_a/lineage_class_check.py`, which is narrower: any difference
outside `person.pre_subsidy_rent` stops for Max, and so does a total that
differs by more than 1e-12 relative. Its JSON verdict goes into the run's
provenance. Any other mismatch,
skipping the new local reference, or relaxing the hand-off still needs **Max's
explicit decision** with the new evidence. Stop before writing `ACCEPTED` while
that decision is outstanding.

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
  `w2_wages_from_qualified_business`. The d177/d450 waivers behind them expire
  at the next certified build, which is this one. Max ruled on 19 October
  (d1711) that d490 covers this build: start with this register, and if the
  measured columns over 0.75 are the same or fewer, apply them on the same
  terms, renewed to expire 2026-12-31 or at the next certified build. Drop
  stale entries. Adding a column or lowering the threshold still needs Max's
  ruling. The old reasons and expiry do not attest a new export. The
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
PREP=$PE/_worktrees/microcosm-route-a-rerun-prep
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
if grep -q 'BLOCKER' "$LOCAL/resolve-initial.log"; then exit 1; fi
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
# One paid attempt at a time: base.fc records the spawned call.
if [ -e "$LOCAL/base.fc" ]; then
  echo "REFUSED: $LOCAL/base.fc records spawned call $(cat "$LOCAL/base.fc"); check it before another paid launch"
  exit 1
fi
# launchd appends to its log, so each attempt gets its own.
ATTEMPT_LOG=$LOCAL/base.launchd.$(date -u +%Y%m%dT%H%M%SZ).log
"$LSUBMIT" route-a-rerun-20261018-base "$ATTEMPT_LOG" "$PREP" \
  /usr/bin/env MICROCOSM_MODAL_PLAN="$PLAN" \
  "$MODAL" run --detach tools/modal_us_stage.py --run --spawn
# Wait at most 30 minutes for this attempt's call id. Stop early if the launcher
# exits without one (a refused check or an auth error).
for _ in $(seq 360); do
  grep -q '"function_call_id"' "$ATTEMPT_LOG" 2>/dev/null && break
  state=$(launchctl print "gui/$(id -u)/com.maxghenis.claude893.route-a-rerun-20261018-base" 2>/dev/null || true)
  case "$state" in *"state = running"*|*"state = spawn scheduled"*) ;; *) break ;; esac
  sleep 5
done
FC=$(sed -n 's/.*"function_call_id": "\(fc-[A-Za-z0-9]*\)".*/\1/p' "$ATTEMPT_LOG" | sed -n 1p)
test -n "$FC" || { echo "No call id in $ATTEMPT_LOG; read it before anything else"; exit 1; }
echo "$FC" > "$LOCAL/base.fc"
```

`--spawn` returns once the stage is spawned and prints its `function_call_id`.
Run it under `--detach`. Without `--spawn` the client waits on the call, and on
19 October the first attempt (`all-20261019T163840131111Z`) was cancelled
after 822 s, 99 s after the Mac went to sleep on battery. The cancellation is
attributed to the sleep from its timing. A probe that day showed a spawned call
finishing after its client exited; client sleep itself was not tested.

The launcher's exit code no longer says how the stage ended; the runs volume
does (see step 5). To block on the call instead, use the modal tool's own
interpreter (system `python3` has no `modal`):
`~/.local/share/uv/tools/modal/bin/python -c 'import modal; print(modal.FunctionCall.from_id("<id>").get())'`.
It returns the receipt, or re-raises the error of a rejected, refused or failed
attempt. Remove `base.fc` only when its call has ended and a relaunch has
Max's go. A launchd job's own exit status is in
`launchctl print gui/$(id -u)/com.maxghenis.claude893.<label>`.

Keep launchd's default `ProcessType=Standard`; do not set `KEEPALIVE=1`.
Use the same reviewed plan for each paid attempt. A failure, cancellation or
budget stop does not authorize an automatic relaunch: each relaunch is another
paid run and needs Max's go (runbook, "The budget stop").

### 4. Produce the local reference (scheduled, supervised local work)

This step runs engines and must wait for machine admission. It was not run
during preparation. Create the clean chosen-commit checkout and locked US env.
The script below supervises each named stage with limits sized to these two
stages, not the whole base: a 32 GiB RSS cap (the interrupted whole attempt
peaked at 11.66 GiB), 24 GiB RAM and 30 GiB disk admission, a six-hour wall
limit and the persistent 20 GiB disk floor. Admission refusal needs a later operator
retry; it does not silently run on the overloaded machine. It never invokes
`--stage all` locally.

```bash
W=$WT_ROOT/microcosm-route-a-${C:0:12}
RUN=$RUN_ROOT/run-${C:0:12}
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
               "rss_bytes": 32*gib, "available_ram_admission_bytes": 24*gib,
               "disk_admission_bytes": 30*gib, "disk_floor_bytes": 20*gib,
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
Every paid attempt shares the plan's run id, `R=$(python3 -c 'import
json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$PLAN")`, so
`runs/$R/attempts/` can hold earlier attempts' records. Read the record whose
`modal.function_call_id` equals `$(cat "$LOCAL/base.fc")`, and the receipt
whose `attempt_id` equals that record's `attempt_id`. This call ends in one
of these states on `microcosm-us-stage-runs`:

- **A receipt** under `runs/$R/receipts/` with this attempt's `attempt_id`.
  The stage COMPLETED or FAILED; step 5 below accepts only COMPLETED.
- **This attempt's record** with `outcome` `refused` or `error`, and no
  receipt: the stage refused or raised.
- **This attempt's record with a `last_seen_epoch` more than about 300 s old**,
  and no receipt: the container was killed.
- **No record for this call**, and `FunctionCall.get()` raises: the plan was
  rejected before an attempt was recorded (`_run_stage`).

Only the first leads to a hand-off. The others are failures; any relaunch
needs Max's go.

```bash
R=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$PLAN")
FETCH=$LOCAL/modal-runs/$R
mkdir -p "$FETCH/state"
"$MODAL" volume get microcosm-us-stage-runs "runs/$R/receipts" "$FETCH/"
"$MODAL" volume get microcosm-us-stage-runs "runs/$R/attempts" "$FETCH/"
"$MODAL" volume get microcosm-us-stage-runs "runs/$R/state/base-out" "$FETCH/state/"
"$MODAL" volume get microcosm-us-stage-runs \
  "runs/$R/state/base-checkpoints/stage_run_context.json" "$FETCH/"
# This call's receipt: its attempt record names the call, the receipt the attempt.
RECEIPT=$(python3 - "$FETCH" "$(cat "$LOCAL/base.fc")" <<'PY'
import json
import sys
from pathlib import Path
fetch, call = Path(sys.argv[1]), sys.argv[2]
records = [
    json.loads(p.read_text()) for p in (fetch / "attempts").glob("all-*.json")
]
mine = [r for r in records if (r.get("modal") or {}).get("function_call_id") == call]
if len(mine) != 1:
    raise SystemExit(f"REFUSED: {len(mine)} attempt records for call {call}")
attempt = mine[0]["attempt_id"]
receipts = [
    p for p in (fetch / "receipts").glob("all-*.json")
    if json.loads(p.read_text()).get("attempt_id") == attempt
]
if len(receipts) != 1:
    raise SystemExit(f"REFUSED: {len(receipts)} receipts for attempt {attempt}")
print(receipts[0])
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
  # APFS clone: the run copy shares the fetched blocks, so it costs no extra
  # disk. That holds only when both paths are on the same APFS volume, as here;
  # across volumes cp -c silently makes a full copy.
  cp -pc "$FETCH/state/base-out/"* "$RUN/base-out/"
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

### 5b. A stage-001 lineage difference (d1712)

When `compare-lineage` reports exactly one problem, the stage-001 checkpoint
digest, the block above stops before copying anything. Run the d1712 check on
the two stage-001 checkpoints. Modal mirrors its copy to the runs volume
(about 0.84 GB); fetch it to scratch and delete it afterwards.

```bash
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
MODAL_001=$SCRATCH/001_pre_clone_enrichment.frame.h5
"$MODAL" volume get microcosm-us-stage-runs \
  "runs/$R/state/base-checkpoints/001_pre_clone_enrichment.frame.h5" "$SCRATCH/"
# The receipt's digest for this file must match what was fetched.
EXPECTED=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(next(o["sha256"] for o in r["outputs"] if o["path"].endswith("001_pre_clone_enrichment.frame.h5")))' "$RECEIPT")
test "$(shasum -a 256 "$MODAL_001" | cut -c1-64)" = "$EXPECTED"
CLASS=$LOCAL/lineage-class-check-stage001.json
"$W/.venv/bin/python" tools/route_a/lineage_class_check.py \
  "$REFROOT/base-checkpoints/001_pre_clone_enrichment.frame.h5" "$MODAL_001" "$CLASS"
```

The fetched file must hash to the receipt's digest for it, the same digest
`compare-lineage` printed for Modal. The scratch copy is removed on any exit.
Exit 0 means the difference is inside the class. Then run the rest of the step 5 block by hand from `mkdir -p
"$RUN/base-out"`, and copy `$CLASS` and the check script into `$RUN/base-sup/`.
Add a `lineage_exception` entry (ruling, check file, verdict) to `RESULT.json`
and name the exception in `ACCEPTED`. Exit 1 stops for Max.

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

Each check runs under `tools/route_a/supervise.py`, like step 4, so the RSS
cap, RAM and disk admission and the disk floor apply. The dry run's limits
follow the driver's preflight stage scaled up to a full base load; the sweep's
follow its measured 4.75 GiB peak. A supervisor refusal prints its reason and
returns 99. The fixture and sweep don't need the base, so they can run while it
builds: generate and submit the script with `GUARDS_ONLY=1` (job
`route-a-rerun-20261018-guards`), then again with `GUARDS_ONLY=0` (job
`route-a-rerun-20261018-checks`) once `ACCEPTED` exists. The second run reuses the recorded guard results.

```bash
GUARDS_ONLY=${GUARDS_ONLY:-0}
# Separate files: bash reads a running script as it goes, so never rewrite one in use.
if [ "$GUARDS_ONLY" = 1 ]; then CHECK_NAME=guards; else CHECK_NAME=checks; fi
{
  echo '#!/bin/bash'
  echo 'set -euo pipefail'
  declare -p W PREP LOCAL RUN SETTINGS GUARDS_ONLY
  cat <<'BASH'
. "$SETTINGS"
CHECKS=$LOCAL/release-checks
mkdir -p "$CHECKS"
# supervised NAME RSS_GIB RAM_GIB DISK_GIB WALL_S CPU_S INPUT... -- ARGV...
# Prints the child's return code, or 99 when the supervisor refused or killed it.
# The config records W's HEAD and the sha256 of each named INPUT (inputs only:
# an output file would change between passes). A check whose recorded config
# is identical is not rerun: its result is reused, and the reuse is logged. A
# different config (a new register, say) refuses with 99. Move that check's
# -config.json and -sup aside to rerun it.
supervised() {
  local name=$1 rss=$2 ram=$3 disk=$4 wall=$5 cpu=$6 rendered
  local inputs=()
  shift 6
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do inputs+=("$1"); shift; done
  [ $# -gt 0 ] || { echo "$name: no -- before ARGV" >&2; echo 99; return; }
  shift
  rendered=$(mktemp)
  python3 -c '
import hashlib, json, os, subprocess, sys
output, worktree, rss, ram, disk, wall, cpu, count, *rest = sys.argv[1:]
inputs, argv = rest[: int(count)], rest[int(count):]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

gib = 1024 ** 3
config = {
    "cwd": worktree,
    "argv": argv,
    "env": {"PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"},
    "authorization": "d1386: release guard checks before the release",
    "limits": {"wall_seconds": int(wall), "cpu_seconds": int(cpu),
               "rss_bytes": int(rss) * gib, "available_ram_admission_bytes": int(ram) * gib,
               "disk_admission_bytes": int(disk) * gib, "disk_floor_bytes": 20 * gib,
               "disk_floor_persist_seconds": 180,
               "output_bytes": 2 * gib, "log_bytes": 512 * 1024 ** 2},
}
config["source_identity"] = {
    "worktree_head": subprocess.check_output(
        ["git", "-C", worktree, "rev-parse", "HEAD"], text=True
    ).strip(),
    # Relative inputs resolve in the worktree, where the check runs.
    "input_sha256": {path: sha256(os.path.join(worktree, path)) for path in inputs},
}
with open(output, "w") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
' "$rendered" "$W" "$rss" "$ram" "$disk" "$wall" "$cpu" "${#inputs[@]}" \
    ${inputs[@]+"${inputs[@]}"} "$@" || {
    echo "$name: could not render the config" >&2; rm -f "$rendered"; echo 99; return
  }
  if [ -e "$CHECKS/$name-sup" ]; then
    if ! cmp -s "$rendered" "$CHECKS/$name-config.json"; then
      echo "$name: recorded config differs from this run's; move $name-config.json and $name-sup aside to rerun" >&2
      rm -f "$rendered"; echo 99; return
    fi
    if [ ! -f "$CHECKS/$name-sup/RESULT.json" ]; then
      echo "$name: supervisor directory without RESULT.json; still running or crashed" >&2
      rm -f "$rendered"; echo 99; return
    fi
    echo "$name: reusing the recorded result (identical config)" >&2
    rm -f "$rendered"
  else
    mv "$rendered" "$CHECKS/$name-config.json"
    "$W/.venv/bin/python" "$PREP/tools/route_a/supervise.py" \
      "$CHECKS/$name-sup" "$CHECKS/$name-config.json" >&2 || true
  fi
  python3 -c '
import json, sys
result = json.load(open(sys.argv[1]))
if result.get("status") == "REFUSED_ADMISSION" or result.get("refusal"):
    print("supervisor:", result.get("status"), result.get("refusal"), file=sys.stderr)
    print(99)
else:
    print(result["returncode"])
' "$CHECKS/$name-sup/RESULT.json"
}
fixture_rc=$(supervised guard-fixture 8 6 30 3600 3600 \
  tools/sweep_us_post_export_scoring.py test_support/microcosm_build/us_post_export_scoring.py -- \
  "$W/.venv/bin/python" -c '
import sys
from pathlib import Path
from tools.sweep_us_post_export_scoring import _load_builder
from test_support.microcosm_build.us_post_export_scoring import _GUARD_HOUSEHOLDS, _write_engine_h5
_write_engine_h5(_load_builder(), Path(sys.argv[1]) / "guard-fixture", _GUARD_HOUSEHOLDS)
' "$LOCAL")
[ "$fixture_rc" = 0 ] || { echo "Guard fixture failed ($fixture_rc)"; exit 1; }
SMALL_H5=$LOCAL/guard-fixture/populace_us_2024.h5
sweep_rc=$(supervised guard-sweep 16 12 30 21600 43200 \
  "$SMALL_H5" tools/sweep_us_post_export_scoring.py -- \
  "$W/.venv/bin/python" tools/sweep_us_post_export_scoring.py \
  --dataset-path "$SMALL_H5" --batch-size 3 --reforms-per-worker 4 \
  --output "$LOCAL/guard-sweep.json")
[ "$sweep_rc" = 0 ] || { echo "Guard sweep failed ($sweep_rc)"; exit 1; }
if [ "$GUARDS_ONLY" = 1 ]; then echo 'Guard checks passed; gate dry run deferred'; exit 0; fi
test -f "$RUN/base-sup/ACCEPTED" || { echo "The base is not ACCEPTED (step 5)"; exit 1; }
EXTRA=($RELEASE_EXTRA_ARGS)
gate_rc=$(supervised gate-dry-run 64 40 30 21600 43200 \
  "$RUN/base-out/base_populace_us_2024_puf_support.h5" "$FEED" "$QRF_TAIL_EXCLUSIONS" \
  "$SSI" "$SCF" tools/dry_run_us_release_gates.py tools/build_us_fiscal_refresh_release.py -- \
  "$W/.venv/bin/python" tools/dry_run_us_release_gates.py \
  --json-out "$LOCAL/dry-run-gates.json" -- \
  --base-h5 "$RUN/base-out/base_populace_us_2024_puf_support.h5" \
  --ledger-facts "$FEED" --ledger-facts-sha256 "$FEED_SHA" \
  --qrf-tail-concentration-exclusions "$QRF_TAIL_EXCLUSIONS" \
  --ssi-take-up-prior-weight-basis "$SSI" \
  --ssi-take-up-prior-weight-basis-sha256 25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a \
  --scf-summary-extract "$SCF" --out "$LOCAL/gate-dry-run" \
  --release-id "populace-us-2024-gate-check-${COMMIT:0:12}" \
  --checkpoint-root "$LOCAL/gate-checkpoints" --seed 0 --no-staging "${EXTRA[@]}")
case "$gate_rc" in
  0) echo 'Release guards completed cleanly' ;;
  2) echo 'AT-RISK: review dry-run-gates.json before starting release'
     # The report must be this run's: a stale one would not say exit 2.
     python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["exit_code"] == 2, d["exit_code"]; print("report status:", d["status"])' "$LOCAL/dry-run-gates.json"
     exit 2 ;;
  *) echo "Release guard dry run failed ($gate_rc)"; exit 1 ;;
esac
BASH
} > "$LOCAL/release-$CHECK_NAME.sh.new"
# Never rewrite a script a running job is reading. lsubmit refuses a running
# label and replaces a finished one; check the same state first, without a
# pipe (grep -q under pipefail can SIGPIPE launchctl and read as no match).
state=$(launchctl print "gui/$(id -u)/com.maxghenis.claude893.route-a-rerun-20261018-$CHECK_NAME" 2>/dev/null || true)
case "$state" in
  *"state = running"*|*"state = spawn scheduled"*)
    echo "route-a-rerun-20261018-$CHECK_NAME is running; wait for it"; exit 1 ;;
esac
mv "$LOCAL/release-$CHECK_NAME.sh.new" "$LOCAL/release-$CHECK_NAME.sh"
"$LSUBMIT" "route-a-rerun-20261018-$CHECK_NAME" "$LOCAL/$CHECK_NAME.launchd.log" "$PREP" \
  /bin/bash "$LOCAL/release-$CHECK_NAME.sh"
```

Wait for the checks job to finish. Inspect its exit status and both reports;
an exit 2 requires operator review of AT-RISK findings before continuing.
A failed sweep or a certain gate failure blocks release.

Records written by an earlier form of this helper have no `source_identity`,
so they are refused as a different config. To keep one, check that it ran on
the same worktree HEAD and inputs, then move it aside and rerun, or rewrite its
`-config.json` in the new form. Don't edit its result.

### 6b. Renew the QRF tail register (d1711)

Every reason in the d490 register says its waiver expires at the next certified
build, which is this one. The release checks only that each reason is
non-empty (`_load_qrf_tail_concentration_exclusions`), so expired text would
be bound into the certified manifest unless it is renewed first. Write
`$LOCAL/qrf_tail_exclusions_d1711.json` with:

- one entry for each register column the dry run still grades above 0.75;
- no entry for a stale column (under 0.75), and no new column (a new one stops
  for Max);
- each reason starting with the d1711 renewal: Max's ruling of 2026-10-19
  that d490 covers this build, on the d177/d450 terms, expiring 2026-12-31 or
  at the next certified build after this one. It names the build commit, the
  base sha256 and the dry run's share, then keeps the earlier history.

Set `QRF_TAIL_EXCLUSIONS` and `QRF_TAIL_EXCLUSIONS_SHA` in the settings file
to the new file, then resolve again.

### 7. Start the release from a frozen driver

bash reads a running script as it goes, so a 33-hour release must not run the
prep checkout's `tools/route_a/route_a.sh` while that branch is still being
edited. Export the reviewed driver into the run folder, read-only, and start
it from there. Book the host first (Fleet ops' `reservation.py ready`).

```bash
# If settings/register changed during review, source again and resolve again.
. "$SETTINGS"
DRIVER_REV=$(git -C "$PREP" rev-parse --short HEAD)
DRIVER=$LOCAL/driver-$DRIVER_REV
if [ ! -e "$DRIVER" ]; then
  mkdir -p "$DRIVER"
  git -C "$PREP" archive "$DRIVER_REV" tools/route_a | tar -x -C "$DRIVER" --strip-components=2
  chmod -R a-w "$DRIVER"
fi
ROUTE_A_ENV="$SETTINGS" bash "$DRIVER/route_a.sh" --resolve | tee "$LOCAL/resolve-release.log"
if grep -q 'BLOCKER' "$LOCAL/resolve-release.log"; then exit 1; fi
test -f "$RUN/base-sup/ACCEPTED" || { echo "The base is not ACCEPTED (step 5)"; exit 1; }
pmset -g batt | grep -q "AC Power" || { echo "Not on AC power; the release needs ~33 h"; exit 1; }
"$LSUBMIT" route-a-rerun-20261018-release "$LOCAL/release.launchd.log" "$LOCAL" \
  /usr/bin/env ROUTE_A_ENV="$SETTINGS" /bin/bash "$DRIVER/route_a.sh"
```

`lsubmit.sh` passes HOME/PATH and explicit job variables only. The `/usr/bin/env`
arguments above are necessary for plan/settings delivery. The driver skips the
accepted Modal base and runs prefetch, base preflight, release with staging,
release preflight and offline publisher preflight. There is no release-only
flag. Confirm staging telemetry arrives and retain the local hand-off and final
gate evidence. A `ROUTE_A_DONE.json` is a candidate hand-off with
`published: false`; publication requires its separate decision.
