# Running a US build stage on Modal

Epic #956 acceleration item E: heavy US stages should not have to queue on
the one 128 GiB build machine. This runbook covers the smallest working path:
one registered tool (`tools/build_us_acs_local_release.py`, stages
`materialize`, `calibrate`, `qa`, `finalize`, `package`, or `all`) run on
Modal from a pinned commit, with inputs fetched by digest and outputs listed
with sha256 receipts.

Two files do the work:

- `tools/modal_us_stage.py` is the Modal app. It builds the image, stages the
  inputs, runs the tool and writes the receipt.
- `tools/modal_us_stage_plan.py` is its pure half, standard library only.
  It validates plans, builds the argv, sizes resources, hashes and mirrors
  files, and writes and verifies receipts. Unit tests:
  `packages/microcosm-build/tests/test_us_modal_stage_plan_tool.py`.

Nothing here uploads to the Hugging Face Hub, touches `latest.json` or
notifies anyone. The furthest a stage goes is `package`, which writes a
release directory onto the runs volume. Publication stays the human step in
`tools/publish_release.sh`.

## How a run works

1. **A plan pins everything.** A plan is a JSON file
   (`microcosm-modal-us-stage-plan/1`; example:
   `docs/us-modal-stage-example-plan.json`). It names the tool, the stage, a
   `run_id`, a full 40-hex commit and the branch it was pushed on, and every
   input as `{uri, sha256}`. Options (`soi_mode`, `hh_chunk`, `epochs` and
   so on) and environment overrides (`MICROCOSM_*`, `POPULACE_*`,
   `*_NUM_THREADS`) come from allowlists. The runner sets the flags for
   input paths, checkpoints and outputs, and a plan cannot pass them. It
   also cannot pass `--allow-dirty`: every run builds from a clean clone.
2. **The image is the commit.** It starts from `debian_slim` with Python
   3.14, the minor version the local US builds record (`runtime.python`
   3.14.4 in the #974 build manifest; Modal served 3.14.2). It adds git and uv 0.11.7, makes a shallow
   clone of the plan's commit from GitHub, checks out the plan's branch name,
   and asserts `HEAD` equals the commit. It then runs
   `uv sync --all-packages --extra us --frozen` against that tree's own
   `uv.lock` into `/opt/venv` and asserts `git status --porcelain` is empty.
   A commit that is not on GitHub fails the build. `checkout -B` would put
   any branch name on any commit, so each container also proves the name
   before it runs anything: it fetches the plan's branch from the remote
   (commits only, into a scratch repository) and requires
   `git merge-base --is-ancestor <commit> <branch tip>`. The check and the
   run both refuse a plan whose branch is missing or does not contain the
   commit, and the receipt records the tip it was checked against. The
   release tool's `_repo_code_identity` therefore records the real sha and a
   branch that contained it when the stage ran. The runner
   code comes from your checkout, not from the pinned commit, so any pushed
   commit whose tree has the tool can run, including commits older than the
   runner. The receipt records the sha256 of both runner files.
3. **Inputs are verified before the tool starts.** A `volume://` input is a
   path on the `microcosm-us-stage-inputs` volume. Local files go there
   content-addressed, at `cas/sha256/<digest>/<name>`, and the plan refuses a
   CAS path whose digest is not the input's own. An `hf://` input is a Hub
   file at an explicit revision:
   `hf://datasets/<org>/<repo>@<revision>/<path>`. Public repos need no
   token. Each input is copied or downloaded to a stable local path,
   `/work/inputs/<name>/<file>`, hashed in the same pass, and refused on
   mismatch.
4. **State lives on the runs volume.** Each `run_id` keeps
   `runs/<run_id>/state/` on `microcosm-us-stage-runs`. That directory holds
   the checkpoints, the calibrated H5, the release root for `package` and
   the stage logs. A stage pulls the state to local disk and runs the tool
   there, since materialize writes a memory-mapped matrix. It then mirrors
   the state back: changed files are copied and files the tool deleted are
   removed. Later stages of the same run pick up from that state. Before
   paying for inputs, a stage after materialize checks that the run's
   `checkpoints/run_identity.json` exists and pins the same staging and
   ladder digests. The tool re-verifies the staging digest itself
   (`_verify_run_identity`).
5. **Every output has a receipt.** `runs/<run_id>/receipts/<stage>-<utc>.json`
   (`microcosm-modal-us-stage-receipt/1`) records:
   - the plan and its sha256;
   - the commit, the clone's `HEAD`, clean state and branch;
   - the sha256 of the runner files and of `uv.lock`;
   - the resource class, the exact argv, the return code and wall seconds;
   - peak child RSS and the list-price cost estimate;
   - every verified input;
   - the sha256 of the earlier receipts in the run;
   - every file in the state tree with its bytes and sha256.

   A failed stage still mirrors its state, so `calibrate` can resume from
   `weights_latest.npz`, and its receipt says `FAILED`.

## Resources and cost

The heavy and light classes are sized from the #974 measured peaks
(`experiments/us-acs-local-hours-rebuild-20260922/run-resources-and-staging-excerpt.json`,
totals SOI surface). On the state SOI surface, the overnight build of 23
September measured materialize locally at a 77.9 GB peak, 5,569 s of tool
wall and 5,472 CPU-s (4,459 targets; see the acceptance attempt below). On
Modal the same stage ran two to five times slower per chunk and held 15 to
24 GB more RSS at the same point, so the local wall times in this table
understate Modal's wall and cost.

| Stage | Measured locally | Class | Request | List price at measured wall |
| --- | --- | --- | --- | --- |
| materialize | 74.8 GB, 5,067 s wall, 4,990 CPU-s | heavy | 4 cores, 128 GiB, 8 h timeout | about $1.71 |
| calibrate | 67.6 GB, 415 s | heavy | 4 cores, 128 GiB | about $0.14 |
| qa | 21.9 GB, 172 s | light | 2 cores, 48 GiB, 4 h timeout | about $0.02 |
| finalize | 23.3 GB, 79 s | light | 2 cores, 48 GiB | about $0.01 |
| package | 21.8 GB, 79 s | light | 2 cores, 48 GiB | about $0.01 |
| check | n/a | check | 2 cores, 8 GiB, 30 min timeout | cents |

The prices are Modal's list prices for standard compute, read from
modal.com/pricing on 22 September 2026: $0.0000131 per core-second and
$0.00000222 per GiB-second. Modal bills the higher of the request and actual
use. The heavy class costs about $1.21 an hour, so an 8-hour timeout costs at
most about $9.70. The engine pass in materialize is single-threaded (CPU
seconds roughly equal wall seconds), so extra cores would not speed it up.
The table leaves out volume storage and image builds.

## Commands

Run everything from a checkout of this branch. The Modal CLI must be on
`PATH` and logged in to the `policyengine` workspace (`modal profile
current`). The plan-module commands run on plain `python3` and need neither
Modal nor the workspace environment.

```bash
# 0. Once per workspace. The app also creates these volumes on first use.
modal volume create microcosm-us-stage-inputs
modal volume create microcosm-us-stage-runs

# 1. Put local inputs on the inputs volume, content-addressed. `digest`
#    prints each file's sha256, the plan input, and the exact upload command.
python3 tools/modal_us_stage_plan.py digest \
  <run>/acs_multispine_staging.h5 <run>/acs_multispine_staging.summary.json \
  <feed>/consumer_facts.jsonl <ladder>/us_puma_ladder_2020.npz
modal volume put microcosm-us-stage-inputs <file> cas/sha256/<digest>/<name>

# 2. Write the plan (copy docs/us-modal-stage-example-plan.json) and
#    validate it locally. This prints the argv, the resource class and the
#    estimate, and exits 2 with REFUSED on a loose plan.
python3 tools/modal_us_stage_plan.py validate plan.json

# 3. Check on Modal (default mode; 2 cores / 8 GiB). It builds the image,
#    verifies the clone, imports the environment, runs the pinned tool's own
#    _parse_args on the built argv, checks every input digest (Hub inputs
#    by their LFS sha256, without downloading) and the run's prior state.
MICROCOSM_MODAL_PLAN=plan.json modal run tools/modal_us_stage.py

# 4. Run the stage. --detach keeps it running if this terminal goes away;
#    the receipt lands on the runs volume either way. Set "max_wall_seconds"
#    in the plan to cap the cost below the class's hard timeout: the runner
#    stops the tool then, and the receipt says FAILED, stopped_at_budget.
#    The budget covers every attempt: when Modal restarts a preempted
#    container, the time the cut-short attempts ran comes off it (see
#    "Preemption" below).
MICROCOSM_MODAL_PLAN=plan.json modal run --detach tools/modal_us_stage.py --run

# 5. Fetch the state and verify it against the receipt.
mkdir -p modal-runs
modal volume get microcosm-us-stage-runs runs/<run_id> ./modal-runs/
python3 tools/modal_us_stage_plan.py verify-receipt \
  ./modal-runs/<run_id>/receipts/<stage>-<utc>.json \
  --state-root ./modal-runs/<run_id>/state
```

To prove the run path on a new commit or workspace for well under a cent,
run `docs/us-modal-stage-smoke-plan.json` (tool `runner-smoke`). It is an
inline script, so it needs no file in the pinned tree. It imports the synced
environment, reads every staged input (two volume files and one Hub file)
and writes one state file, which goes through the same staging, mirroring
and receipt code as a real stage.

Next stage: copy the plan, change `stage`, keep the `run_id`, and drop `feed`
if you like, since only materialize reads it. Run steps 3 and 4 again. Run
one stage of a run at a time, because two concurrent stages would race on the
same state directory.

For a gated or private Hub input, set `MICROCOSM_MODAL_HF_SECRET` to the name
of a Modal secret that holds `HF_TOKEN`, for example `huggingface-token` in
the `policyengine` workspace. Otherwise no secret is attached. Only the
runner uses the token, to download inputs. The tool always runs with
`HF_HUB_OFFLINE=1` (huggingface_hub then refuses every request) and without
any variable whose name looks like a credential (`KEY`, `TOKEN`, `SECRET`,
`PASSW`, `SIGNING`, `CREDENTIAL`); the receipt lists the names removed,
never their values. A plan cannot set such a variable either.

## First acceptance: replay #974 materialize

`docs/us-modal-stage-example-plan.json` replays the materialize stage of the
full-scale local build of 22 September (#974). The plan pins the build
commit `cadaf418` (branch `local-acs-hours-rebuild-20260921`), that run's
staging H5 and staging summary, the `chronicle_us_b571381` feed and the PUMA
ladder, all by the digests the #974 receipt records. A Modal replay should
reproduce these values from that receipt's `run_identity`:

- `targets_sha256` `0f447ce92b4279881382fdd4006be47cfbdd6972be438f967a7cca97c29a4f81`
- `n_targets` 1,247, from 760 admin specs, all compiled
- 1,588,854 households
- no dropped population cells

Compare `runs/replay-974-materialize/state/checkpoints/run_identity.json`
with
`experiments/us-acs-local-hours-rebuild-20260922/build_manifest.json`. A
match shows that the Modal image and platform reproduce the local target
compile.

## Verified on Modal, 22 September 2026

These runs were in the `policyengine` workspace. The main checkout, the
overnight build and the Hub were not touched.

- **Volumes.** `microcosm-us-stage-inputs` and `microcosm-us-stage-runs`
  were created. Three small inputs of the #974 build went to `cas/sha256/`:
  the PUMA ladder (446,791 bytes), the staging summary (498,113 bytes) and
  the `chronicle_us_b571381` feed (164,624,488 bytes). The 10.7 GB staging
  H5 was not uploaded.
- **Check of the #974 replay plan, which reported one problem, as
  intended.** The image built from `cadaf418`. `HEAD` matched, the tree was
  clean, and the branch was checked out. The environment synced from that
  tree's lock: policyengine-us 2.2.1 and policyengine-core 3.32.5, the
  versions #974 records. The pinned tool's `_parse_args` accepted the built
  argv and returned `["materialize"]`. The three uploaded inputs were
  verified by sha256 on the volume. The only problem reported was
  `staging_h5` not being on the volume, which is the correct refusal.
- **Smoke run (`--run`).** `runner-smoke-cadaf418` completed in 7 seconds.
  It staged and verified the two volume inputs and the public Hub file
  `policyengine/populace-us@85a1ccb0…/latest.json`, and wrote
  `smoke/inputs.json`. The receipt is
  `runs/runner-smoke-cadaf418/receipts/smoke-2026-09-23T032440Z.json`. After
  `modal volume get`, `verify-receipt --strict` passed locally with 2
  outputs and 0 problems. Modal served Python 3.14.2. The local builds
  recorded 3.14.4.

The heavy materialize replay has not been run. It needs the staging H5 on
the inputs volume, which is about 10.7 GB to upload, plus one heavy run at
about $1.71 list price for the measured wall time. That run is the next
step:

```bash
# On the build machine, the #974 staging H5 is under
# _recovered/scratch-backup/893/local-hours-rebuild-20260921/full-staging-a3/.
python3 tools/modal_us_stage_plan.py digest <that dir>/acs_multispine_staging.h5
# expect f335f573…; then run the printed `modal volume put …` line
MICROCOSM_MODAL_PLAN=docs/us-modal-stage-example-plan.json modal run tools/modal_us_stage.py
MICROCOSM_MODAL_PLAN=docs/us-modal-stage-example-plan.json \
  modal run --detach tools/modal_us_stage.py --run
```

## Acceptance attempt: 23 September materialize on the state surface

The plan `docs/us-modal-stage-acceptance-20260923-plan.json` ran the
materialize stage of the overnight build of 23 September on Modal
(`run_id` `overnight-20260923-materialize-state`). It pinned build commit
`767312d6` on branch `overnight-acs-local-20260923`, `soi_mode` `state`,
`hh_chunk` 20,000, the local run's peak-limit environment and a
26,400-second wall budget. The same stage ran locally at the same time from
the same inputs. The Modal run did not finish, so its outputs could not be
compared.

- **Inputs.** The staging H5 (`ed2e6308…`, 10,685,765,051 bytes) went to
  the inputs volume in 255 seconds, about 42 MB/s, and the staging summary
  (`3aab5e5e…`) went up too. The feed and ladder were already there.
- **Check.** It passed with no problems. The image built from `767312d6`,
  the clone was clean on its branch, and the environment synced to
  policyengine-us 2.2.1 and policyengine-core 3.32.5, the same versions as
  the local environment. The tool's parser accepted the argv, and all four
  inputs matched their digests on the volume.
- **Run.** The container started at 06:03:06 UTC and the tool's first
  log line came at 06:06:22, so staging and verifying the inputs took under
  3.5 minutes. The tool's own timings, Modal against local:

  | Step | Modal | Local |
  | --- | --- | --- |
  | Start to staging frame loaded (hashing, specs, load) | 255 to 281 s | 79 s |
  | One chunk of 20,000 households | 149 to 355 s | 66 to 80 s |
  | peak RSS after chunk 1 | 66.3 GB | 51.0 GB |
  | peak RSS after chunk 29 | 74.5 GB | 51.0 GB |

  The container ran gVisor (`Linux-4.19.0-gvisor-x86_64`) with Python
  3.14.2. In the check-class container Modal set `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `BLIS_NUM_THREADS` to that
  class's CPU request, 2. In the heavy container `nproc` reported 4. The
  local run used macOS arm64 with Python 3.14.4, and its config set none of
  those variables.
  A sample inside the container, taken with `modal container exec`, showed
  the tool process using about 97% of one core (59.4 CPU-s in 61 s). The
  stage was CPU-bound on one core, not throttled.
- **Preempted twice.** Modal preempted the first container after 54
  minutes (after chunk 12 of 80) and the second after 2 hours 4 minutes
  (after chunk 29). Each time it restarted the function from zero. The run
  was launched before the attempt ledger existed, so the budget restarted
  too. It was stopped by hand (`modal app stop`) 34 seconds into a third
  attempt, because at the observed pace a third attempt needed 6.5 to 7
  more hours and would have taken the total past $10.
- **Cost.** Modal's workspace billing report
  (`modal.billing.workspace_billing_report`, hourly) shows $3.57 for the
  run through 09:00 UTC. The last 2.5 minutes add about $0.05, for about
  $3.62 in total. The check cost $0.02. Billing was at the request, about
  $1.21 an hour.
- **Local result for the next comparison.** The local run finished in
  5,569 s with a 77.9 GB peak. `run_identity`: staging `ed2e6308…`, ladder
  `39a2ab2a…`, 1,588,854 households, 4,459 targets (3,972 admin specs
  declared and compiled, plus 487 state and CD population cells), no
  dropped cells, `targets_sha256`
  `d843209746bdcf96a831fa90d6fa2fe7aa2969fa32290db35afa9889f7c5d377`.
  Checkpoint sha256: `held_back_columns.json` `026c455b…`,
  `reviewed_null_fills.json` `727f39c1…`, `targets.json` `d8432097…`
  (the same as `targets_sha256`), and `target_frame_lean.h5`
  (28,749,924,512 bytes) `ae7d0ae7…`. HDF5 files can differ byte for byte
  when their stored values match, so the lean H5 should also be compared
  dataset by dataset. The build machine keeps per-dataset digests of the
  local file next to the overnight run.

At the pace observed, one uninterrupted Modal materialize on the state
surface takes about 3.5 to 7 hours. That is about $4 to $9 on preemptible
placement, or $13 to $26 with `"nonpreemptible": true`, against 1.5 hours
locally. The Modal path works, but it does not yet make this stage cheaper
or faster. The run did show that the stage runs as one long single-core
loop over 80 chunks of households. If the chunks are independent, which
this runbook has not checked in `materialize_chunked`, fanning them out
across containers would let preemption lose one chunk instead of the run.

## Data placement

Inputs go to the `policyengine` Modal workspace. Upload only files that are
allowed there. No registered stage takes the restricted IRS PUF files
(`--puf-source-year-csv` in `tools/build_us_puf_support_base.py`), and the
base stage is not registered. Logs hold only what the tool prints. Receipts
hold digests, sizes and paths, never file contents.

## What this does not cover yet

- **The graph-native line (#893).** The DAG executor lives on #893's branch,
  not on main. Registering it means one more `ToolSpec` with its CLI, inputs
  and state layout. The executor runs nodes one after another, and the 1/15
  run peaked at 46.7 GB (#956). A full-source run needs the byte-transport
  and parallel-executor work first, plus a resource class sized from a
  measured full-scale peak.
- **The base stage (`tools/build_us_puf_support_base.py`)** is not
  registered. Its peak was reported at 72 GB; this runner has not measured
  it. It requires the processed PUF (`--puf-h5`), and
  `--puf-source-year-csv` (the restricted TY2015 IRS PUF CSV) whenever the
  processed PUF is nonzero. Nobody has decided whether those files may sit
  on Modal volumes.
- **The staging stage (`tools/build_us_acs_multispine_base.py`)** is not
  registered either. It takes a directory input (`--inputs-dir`, the ACS
  PUMS archive cache), which the plan format does not support. Supporting it
  would take an archive digest plus extraction.
- **Preemption and out-of-memory kills.** Functions run on Modal's default
  (preemptible) placement. When Modal preempts a container it restarts the
  function on the same input, from scratch, whatever `retries` says; this
  happened to the 23 September acceptance run after 54 minutes. State is
  only mirrored and the receipt only written when the tool exits, so the
  cut-short attempt leaves no receipt and its work is lost, but it is
  billed. Each attempt therefore writes
  `runs/<run_id>/attempts/<stage>-<utc>.json` when it starts and rewrites
  it every 120 seconds. A restart charges the time of every earlier attempt
  of the same plan that never wrote a receipt to `max_wall_seconds`, and
  refuses to start when less than a minute is left. The check reports those
  attempts and the time left. To launch again past a spent budget, raise
  `max_wall_seconds` (a new plan digest) or use a new `run_id`. A plan can
  set `"nonpreemptible": true` to run on Modal's non-preemptible placement,
  which avoids restarts at three times the list price for CPU and memory
  (modal.com/docs/guide/preemption, read 23 September 2026); `validate` and
  the receipt price it that way. The heavy class sets a memory request but
  no hard limit. What Modal does with a container killed for memory has not
  been observed here.
- **Certification.** A receipt proves which bytes a stage produced. It does
  not certify a release. Preflight (`tools/preflight_us_release_gates.py`)
  and certification still run on the output as before.
