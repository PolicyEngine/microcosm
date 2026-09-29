# Running a US build stage on Modal

Epic #956 acceleration item E: heavy US stages should not have to queue on
the one 128 GiB build machine. This runbook covers the smallest working path:
a registered tool run on Modal from a pinned commit, with inputs fetched by
digest and outputs listed with sha256 receipts. Three tools are registered
(`TOOLS` in the plan module): `tools/build_us_acs_local_release.py` (tool
`us-acs-local-release`, stages `materialize`, `calibrate`, `qa`,
`finalize`, `package`, or `all`), Route A's PUF-support base,
`tools/build_us_puf_support_base.py` (tool `us-puf-support-base`, stage
`all`; see "Route A's base stage"), and `runner-smoke` (stage `smoke`), an
inline script that proves the run path on any pushed commit (see
"Commands").

Two files do the work:

- `tools/modal_us_stage.py` is the Modal app. It builds the image, stages the
  inputs, runs the tool and writes the receipt.
- `tools/modal_us_stage_plan.py` is its pure half, standard library only.
  It validates plans, builds the argv, sizes resources, hashes and mirrors
  files, and writes and verifies receipts. Unit tests (engine-free):
  `packages/microcosm-build/tests/engine_free/us/test_us_modal_stage_plan_tool.py`
  and, for the base,
  `packages/microcosm-build/tests/engine_free/us/test_us_modal_stage_puf_support_base.py`.

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
   so on) and environment overrides (`MICROCOSM_*`, `POPULACE_*`, the
   thread counts `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
   `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `BLIS_NUM_THREADS`, and
   `PYTHONUNBUFFERED`, so a plan can match a local run that set it) come
   from allowlists; an environment variable whose name
   looks like a credential (`KEY`, `TOKEN`, `SECRET`, `PASSW`, `SIGNING`,
   `CREDENTIAL`) is refused even under an allowed prefix. `PYTHONHASHSEED`
   is outside the allowlist, so no plan can set it (see "Route A's base
   stage", Environment). The runner sets the flags for input paths,
   checkpoints and outputs, and a plan cannot pass them. It also cannot pass
   `--allow-dirty`: every run builds from a clean clone. A registered
   option's flag must be a plain long flag with no `=` in it that is neither
   a runner-owned flag nor a prefix of one: the tools parse with argparse,
   whose `allow_abbrev` (on by default) reads a unique prefix such as
   `--allow-geography` as the flag it begins, and `--flag=value` as the flag
   and a value. The plan and the argv refuse any other registration.
   `"nonpreemptible": true` asks for Modal's non-preemptible placement at
   three times the list price (see "Preemption and restarts").
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
   `/work/inputs/<name>/<file>`, and refused on a digest mismatch before the
   tool starts: a volume input is hashed while it is copied, a Hub input
   after it downloads. A Hub revision may be a branch name; the sha256 still
   pins the bytes, so a branch that has moved fails verification.
4. **State lives on the runs volume.** Each `run_id` keeps
   `runs/<run_id>/state/` on `microcosm-us-stage-runs`. That directory holds
   the checkpoints, the calibrated H5, the release root for `package` and
   the stage logs. A stage pulls the state to local disk and runs the tool
   there, since materialize writes a memory-mapped matrix. Each file is
   hashed while it is pulled, and before anything uses the pulled state it
   is verified against the run's latest receipt (by `finished_at`): every
   file the receipt lists, with its bytes and sha256, and nothing else.
   With no receipt the state must be empty. A mismatch means a stage was
   cut short after changing the state, and the stage refuses; start a new
   `run_id`. When the tool exits the state is mirrored back: each changed
   file is hashed while it is copied under a temporary name in its own
   directory and renamed over the old one, and files the tool deleted are
   removed, so a mirror cut short never leaves a half-written file. A file
   the tool did not touch is neither copied nor read again; its digest is
   the one verified when it was pulled. Each file of the state is read at
   most once on the way in and at most once on the way out. A stage may
   name paths the push copies first (the base's output and evidence), and
   may declare that the rest of its state only serves a resume; after a
   clean exit such a stage copies only those paths and hashes the rest
   into the receipt (see "What the push copies" under Route A's base).
   Later stages of the same run pick up from that state. Before
   paying for inputs, a stage after materialize checks that the run's
   `checkpoints/run_identity.json` exists and pins the same staging and
   ladder digests. The tool re-verifies the staging digest itself
   (`_verify_run_identity`).
5. **Every output has a receipt.** `runs/<run_id>/receipts/<stage>-<utc>.json`
   (`microcosm-modal-us-stage-receipt/1`) records:
   - the plan and its sha256;
   - the commit, the clone's `HEAD`, clean state, and the branch with the
     remote tip it was verified against;
   - the sha256 of the runner files and of `uv.lock`;
   - the resource class (with its CPU limit, if any) and placement, the
     exact argv, the return code and wall seconds;
   - peak child RSS and the list-price cost estimate, for this container and
     for every earlier unfinished attempt of the plan;
   - the attempt id and this attempt's share of `max_wall_seconds`;
   - every verified input;
   - the names, never the values, of environment variables withheld from
     the tool (credentials, and the container's `PYTHONHASHSEED` if it had
     one);
   - `runner.python_hash_seed`: the container's `PYTHONHASHSEED`, what the
     tool's environment carried (always none or `0`; the runner refuses
     anything else before staging), and the value the tool's stage
     interpreters ran with (`0` for the base, whose pinned tool gives its
     stages `0` when it has none; none for a tool that sets no default);
   - the sha256 of the earlier receipts in the run, and the receipt the
     pulled state matched;
   - every file in the state tree with its bytes and sha256, and which of
     them were hashed but not copied to the volume
     (`outputs_not_mirrored`, empty unless the stage's rest is resume
     state).

   A failed stage still mirrors its state, so `calibrate` can resume from
   `weights_latest.npz`, and its receipt says `FAILED`.

## Resources and cost

The heavy and light classes are sized from the #974 measured peaks
(`experiments/us-acs-local-hours-rebuild-20260922/run-resources-and-staging-excerpt.json`,
totals SOI surface). On the state SOI surface, the overnight build of 23
September measured materialize locally at a 77.9 GB peak, 5,569 s of tool
wall and 5,472 CPU-s (4,459 targets; see the acceptance attempt below). On
Modal the same stage ran 3.3 to 7.6 times slower per chunk (3.93 times over
chunks 1 to 29) and held 15 to 24 GB more RSS at the same chunk, so the
local wall times in this table understate Modal's wall and cost.

| Stage | Measured locally | Class | Request | List price at measured wall |
| --- | --- | --- | --- | --- |
| materialize | 74.8 GB, 5,067 s wall, 4,990 CPU-s | heavy | 4 cores, 128 GiB, 8 h timeout | about $1.71 |
| calibrate | 67.6 GB, 415 s | heavy | 4 cores, 128 GiB | about $0.14 |
| qa | 21.9 GB, 172 s | light | 2 cores, 48 GiB, 4 h timeout | about $0.02 |
| finalize | 23.3 GB, 79 s | light | 2 cores, 48 GiB | about $0.01 |
| package | 21.8 GB, 79 s | light | 2 cores, 48 GiB | about $0.01 |
| check | n/a | check | 2 cores, 8 GiB, 30 min timeout | cents |
| PUF-support base (`all`) | 72.5 GB, 2,788 s wall, 6,238 CPU-s | base | 4 cores (limit 4), 112 GiB, 4 h 35 min timeout | about $0.84 ($2.52 non-preemptible) |

The prices are Modal's list prices for standard compute, read from
modal.com/pricing on 22 September 2026 and unchanged on 29 September:
$0.0000131 per core-second and $0.00000222 per GiB-second. Modal bills the
higher of the request and actual use (modal.com/docs/guide/resources). The
heavy class costs about $1.21 an hour at its request, so an 8-hour timeout
costs about $9.70 at the request; CPU or memory used above the request
bills above that. With `"nonpreemptible": true` every figure is three times
higher: about $3.63 an hour for the heavy class. Only the base class sets
a CPU limit (equal to its request); the others are request-only for CPU,
and every class is request-only for memory, so a stage that used more cores
or more memory than its request would be billed for them. The engine pass in
materialize is single-threaded (CPU seconds roughly equal wall seconds), so
extra cores would not speed it up. The table leaves out volume storage and
image builds. The base class is sized in "Route A's base stage" below.

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
#    Once a plan exists, `upload-commands PLAN ROLE=PATH ...` hashes each file
#    against the plan's digest for that role and refuses the whole set on any
#    mismatch; with --shell it writes a script that uploads only the files
#    the volume does not list yet (see "Route A's base stage", step 1).

# 2. Write the plan (copy docs/us-modal-stage-example-plan.json) and
#    validate it locally. This prints the argv, the resource class and the
#    estimate, and exits 2 with REFUSED on a loose plan.
python3 tools/modal_us_stage_plan.py validate plan.json

# 3. Check on Modal (default mode; 2 cores / 8 GiB). It builds the image,
#    verifies the clone and that the plan's branch contains its commit,
#    imports the environment, runs the pinned tool's own _parse_args on the
#    built argv, checks every input digest (Hub inputs by their LFS sha256,
#    without downloading), hashes the run's state against its latest
#    receipt, and reports the attempts charged to the budget and any
#    attempt of the run that may still be running.
MICROCOSM_MODAL_PLAN=plan.json modal run tools/modal_us_stage.py

# 4. Run the stage. --detach keeps it running if this terminal goes away;
#    the receipt lands on the runs volume either way. Set "max_wall_seconds"
#    in the plan to cap the cost below the class's hard timeout: the runner
#    stops the tool then (SIGTERM to the tool's process group, which the
#    tool runs in alone, so a stage child it spawned stops too; SIGKILL to
#    whatever is left after 60 s), and the receipt says FAILED,
#    stopped_at_budget. Whatever the plan says, the runner also stops the
#    tool by the class timeout less the stage's runner reserve, counted
#    from the start of the function, so pulling state and staging inputs
#    come out of the tool's time and the state is always mirrored with a
#    receipt before the timeout (the receipt's runner.tool_budget says
#    which limit applied).
#    The budget covers every attempt that never finished: when Modal
#    restarts a preempted container, the time the cut-short attempts ran
#    comes off it (see "Preemption and restarts" below). A stop at the
#    budget writes a receipt, so launching the same plan again gets the
#    whole budget again.
MICROCOSM_MODAL_PLAN=plan.json modal run --detach tools/modal_us_stage.py --run

# 5. Fetch the state and verify it against the receipt. A receipt whose
#    status is not COMPLETED, or that says stopped_at_budget, does not
#    verify: its state can hold a partly written output. Add --allow-failed
#    to check the bytes of such a state anyway (for a resume, say).
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
same state directory. The attempt ledger enforces this, best effort (see
"Preemption and restarts").

For a gated or private Hub input, set `MICROCOSM_MODAL_HF_SECRET` to the name
of a Modal secret that holds `HF_TOKEN`, for example `huggingface-token` in
the `policyengine` workspace. Otherwise no secret is attached. Only the
runner uses the token, to download inputs. The tool always runs with
`HF_HUB_OFFLINE=1` (huggingface_hub then refuses every request) and without
any variable whose name looks like a credential (`KEY`, `TOKEN`, `SECRET`,
`PASSW`, `SIGNING`, `CREDENTIAL`); the receipt lists the names removed,
never their values. A plan cannot set such a variable either. The tool
also never gets the container's `PYTHONHASHSEED`: the runner removes it
(and lists its name with the others), and refuses before pulling or
staging anything if the tool's environment would still carry a value
other than `0` (see "Route A's base stage", Environment).

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
the inputs volume, which is about 10.7 GB to upload, plus one heavy run.
That is about $1.71 at the local wall time, but at the Modal pace measured
on 23 September (3.93 times local, below) it is closer to 5.5 hours: about
$6.70 if nothing preempts it, or about $20 non-preemptible, which a run that
long needs (see "Preemption and restarts"). Add `"nonpreemptible": true` and
a `max_wall_seconds` to a copy of the plan first. That run is the next
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
`hh_chunk` 20,000 and a 26,400-second wall budget. The same stage ran
locally at the same time from the same inputs. The Modal run did not finish,
so its outputs could not be compared.

The plan as launched (`b2e34fe4f`, plan sha256 `5ec595c7…`) also set four
peak-limit variables copied from the local run's environment:
`MICROCOSM_ACS_POOL_PEAK_LIMIT_BYTES`,
`MICROCOSM_STAGING_EXPORT_PEAK_LIMIT_BYTES` and their `POPULACE_` twins. At
`767312d6` none of them reaches materialize. The two `MICROCOSM_` names are
read nowhere. The two `POPULACE_` names only set defaults for
`with_optional_acs_spine` and `_preflight_staging_export`, which only the
staging builder calls. They were removed from the committed plan, whose
sha256 is now `d1341ec0…`, so a relaunch of this file is a new plan to the
attempt ledger.

- **Inputs.** The staging H5 (`ed2e6308…`, 10,685,765,051 bytes) went to
  the inputs volume in 255 seconds, about 42 MB/s, and the staging summary
  (`3aab5e5e…`) went up too. The feed and ladder were already there.
- **Check.** It passed with no problems. The image built from `767312d6`,
  the clone was clean on its branch, and the environment synced to
  policyengine-us 2.2.1 and policyengine-core 3.32.5, the same versions as
  the local environment. The tool's parser accepted the argv, and all four
  inputs matched their digests on the volume.
- **Run.** It was launched detached at 06:02:58 UTC (02:02:58 EDT) as app
  `ap-mzp14wEyVLIFlYX5qQQxFZ`, on runner commit `1d80287af`, before the
  attempt ledger existed. The container started at 06:03:06 UTC and the
  tool's first log line came at 06:06:22, so staging and verifying the
  inputs took under 3.5 minutes. The tool's own timings, Modal against
  local:

  | Step | Modal | Local |
  | --- | --- | --- |
  | Start to staging frame loaded (hashing, specs, load) | 280.6 and 254.6 s | 79.3 s |
  | One chunk of 20,000 households | 144.0 to 354.5 s | 36.8 to 80.1 s (mean 66.3 s over 80) |
  | Chunks 1 to 29 in all (second attempt) | 6,832 s | 1,737 s |
  | peak RSS after chunk 1 | 66.3 GB | 51.0 GB |
  | peak RSS after chunk 29 | 74.5 GB | 51.0 GB |

  Per chunk Modal was 3.3 to 7.6 times slower, and 3.93 times over chunks 1
  to 29.

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
  too. It was stopped by hand (`modal app stop`, 09:02:22 UTC) 34 seconds
  into a third attempt, because at the observed pace a third attempt needed
  6.5 to 7 more hours and would have taken the total past $10. It left no
  receipt and no state: that runner mirrors state and writes the receipt
  only when the tool exits, and the runs volume has no
  `runs/overnight-20260923-materialize-state`.
- **Cost.** Modal's workspace billing report
  (`modal.billing.workspace_billing_report`, hourly) shows $3.57 for the
  run through 09:00 UTC. The last 2.5 minutes add about $0.05, for about
  $3.62 in total. The check cost $0.02. Billing was at the request, about
  $1.21 an hour.
- **Local result for the next comparison.** The local run finished in
  5,604.5 s of wall (5,569 s in the tool) with a 77.9 GB peak (the tool's
  log; 76.7 GB by the supervisor). `run_identity`: staging `ed2e6308…`, ladder
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
surface takes about 5.3 to 6.9 hours: 80 chunks at the second attempt's
mean of 235.6 s, or at its last five chunks' 308.2 s, plus about 4.5
minutes of loading. That is about $6.40 to $8.40 at the preemptible list
price if nothing preempts it, and $19 to $25 with `"nonpreemptible": true`,
against 1.6 hours locally. At the preemption rate this run saw, a 5- to
7-hour preemptible run finishes uninterrupted 1 to 3% of the time, so
non-preemptible is the practical placement for it. The Modal path works,
but it does not yet make this stage cheaper or faster. The run did show that the stage runs as one long single-core
loop over 80 chunks of households. If the chunks are independent, which
this runbook has not checked in `materialize_chunked`, fanning them out
across containers would let preemption lose one chunk instead of the run.

## Route A's base stage

On 29 September 2026 Max ruled to run only Route A's base stage on Modal:
non-preemptible, capped at 4 hours of wall, about $15. The release, its
preflight and certification stay on the build machine.

**How this change reads the cap (not confirmed by Max).** The ruling caps
the run at 4 hours of wall and about $15; it does not say whose wall, the
tool's or the container's. This registration reads the 4 hours as the
tool's wall: the plan's `max_wall_seconds` is 14,400. Each container may
then execute for up to the class timeout, 16,500 seconds (4 hours 35
minutes): the tool's 4 hours, 5 minutes for staging and 30 minutes for
stopping the tool, mirroring and the receipt. It holds the $15 at $14.90,
the class's request for the whole timeout at the non-preemptible list
price. If Max means the container's execution time, the other reading is a
14,400-second class timeout with `max_wall_seconds` cut to 12,300 (the same
35 minutes of staging and reserve), which lists at $13.00; that is a
registry and plan change, not a relaunch flag.

**How the cap holds.** Modal's timeout bounds a function's execution time
(modal.com/docs/guide/timeouts, read 29 September 2026), so no attempt
executes longer, whatever hangs. A test holds the $14.90 at or below $15
(`BASE_COST_CAP_USD`). That figure is the ceiling at the request, and
Modal bills CPU and memory at the higher of the request and actual use
(modal.com/docs/guide/resources, read 29 September 2026):

- *CPU use above the request is throttled.* The base's functions pass
  `cpu=(4.0, 4.0)`, a CPU limit equal to the request. Modal's docs say
  that above a container's CPU limit the host begins to throttle its CPU
  use, that a function can set the limit explicitly with that tuple, and
  that without one the default soft limit is 16 physical cores above the
  request (same page). So Modal throttles the base above its 4-core
  request, and the $14.90 counts CPU at the request on that basis. Without
  the limit, a stage using 20 cores for the whole timeout would list at
  $25.28. See "Resources" for why the pinned tool could use more than 4
  cores.
- *Memory is not capped.* Use above 112 GiB is billed at use, about $0.024
  per GiB-hour non-preemptible. This is the one way an attempt can list
  above $14.90. A memory limit at the request would turn that into an OOM
  kill of the whole container (Modal's docs: a hard memory limit "will ...
  kill containers which attempt to exceed the limit"), and the runner
  mirrors state and writes the receipt from inside that container, so the
  attempt would end with no receipt and no mirrored state. The class keeps
  memory request-only; setting a limit is Max's call.
- The timeout is not exact to the second: Modal says a function "may run
  a handful of seconds longer" (modal.com/docs/guide/timeouts). The $0.10
  between $14.90 and $15 is about 110 seconds of this class.
- Scheduling is not execution time, and container startup is timed
  separately (`startup_timeout`, which defaults to the function's
  `timeout`; same page), so neither is inside the execution bound. The
  $14.90 counts neither of them, nor building the image.
- *Each check is outside the $14.90 too.* A check (step 3 below) runs in
  the check class, 2 cores and 8 GiB with a 30-minute timeout, and lists
  at up to about $0.08 at that request if it runs its whole timeout; more
  if it uses more than its 2 cores or 8 GiB, since that class sets no
  limit.

So the $0.10 of headroom is shared: the seconds past the timeout, every
check, and whatever container startup and image builds add all come out of
it. One check that ran its whole timeout would leave about $0.02 of it.

The billed figure is in Modal's workspace billing report.

**What runs.** Tool `us-puf-support-base` runs
`tools/build_us_puf_support_base.py --stage all` with the command Route A's
driver built for commit `4b57d15a` (`route_a.sh` section 4, recorded in the
run directory's `base-config.json`), flag for flag and value for value. Only
the paths differ: the inputs are staged at `/work/inputs/<role>/<file>`, the
checkpoints go to `/work/state/base-checkpoints` and the output to
`/work/state/base-out`. The scalar settings (target year 2024, seed 0, 32
estimators, district seed 0, `--assign-congressional-districts`) are fixed
in the registration, not plan options: a different setting is a different
base, and changing it is a reviewed registry change. The tool's other flags
(`--base-h5`, the smoke limit, the equivalence harness, the ladder escape
hatches) are runner-owned, so no plan can pass them. A test refuses a
registry option for each of the 26 owned flags, and another for every
abbreviation of each withheld flag (argparse would read
`--allow-geography` as `--allow-geography-ladder-gate-failures`) and for a
flag with `=` in it.

Two kinds of test hold the command:

- In every checkout, CI included: the built argv equals a copy of the local
  command with its paths tokenized
  (`packages/microcosm-build/tests/fixtures/modal_us_stage/route_a_base_command_4b57d15a287c.json`),
  and the registration's owned flags equal the flags of this tree's tool.
- Only in a clone that has commit `4b57d15a2`, four tests in
  `test_us_modal_stage_puf_support_base.py` that read that commit with
  `git show` and skip without it:
  - `test_the_tool_s_own_parser_reads_the_same_build_from_both_commands`:
    the tool is loaded from that commit's own source, not from this tree's
    copy, and its `_parse_args` and `_stage_cli_args` read both commands
    into the same settings and the same child command for every outer
    stage, paths aside;
  - `test_the_owned_flags_are_the_pinned_tool_s_flags`: its flags equal
    the registration's owned flags;
  - `test_the_registered_crosswalk_digest_is_4b57d15a2_s`: the district
    crosswalk at that commit has the registered sha256;
  - `test_the_pinned_tool_locks_the_reference_s_run_config_from_the_modal_argv`:
    its `_stage_run_config` locks, from the Modal argv, the run config the
    local run locked (paths by file name), including `PYTHONHASHSEED` `0`
    when the runner's tool environment removed a container's value, and a
    forced `12345` is what compare-lineage refuses.

CI checks out only the commit under test (`actions/checkout` fetches one
commit by default), so there those four tests report a skip rather than
pass. Run them in a full clone before a paid run (step 2 below).

The plan is `docs/us-modal-stage-route-a-base-plan.json` (run
`route-a-base-4b57d15a287c`, branch `main`).

- **Inputs.** Eleven files, content-addressed on `microcosm-us-stage-inputs`
  with the digests `route_a.sh` verified on 23 September: the 2024, 2023
  and 2022 ASEC H5s, `puf_2024.h5`, `puf_2015.csv`, `acs_2022.h5`, the
  three ASEC Census person archives (`asecpub23csv.zip` to
  `asecpub25csv.zip`), the Ledger feed `consumer_facts_us_c5e5bf8.jsonl`
  and the block ladder `us_block_ladder_2020.npz`.
- **The district crosswalk is not an input.** The tool reads it from the
  pinned clone. The registration pins its sha256 (`c7cb040b…`, the
  `4b57d15a` version; main has since changed the file), and the check and
  the run refuse a clone whose copy differs.
- **The 2023 ASEC archive is seeded, not downloaded.** Route A passes no
  `--asec-2023-weeks-unemployed-source`, so the tool looks in
  `~/.cache/microcosm/cps/asec_2023/` and downloads the archive from
  www2.census.gov when it is missing. The build machine had it cached. The
  runner copies the staged `asec_education_2022_zip` input there first (the
  same file: the tool pins both to `d2e00025…`, and a plan with another
  digest for that input is refused), so the stage makes no Census request.
  The tool still verifies the archive itself. The receipt lists the copy
  under `runner.home_seeds`.
- **Environment.** The plan sets only `PYTHONUNBUFFERED=1`, as the local
  run did. It sets no thread counts, because the local run set none;
  Modal's container defaults apply, and the receipt records them
  (`runner.thread_env`, `os_cpu_count`, `cpu_affinity`), as does the tool's
  own `stage_run_context.json` (`thread_environment`, which step 6 reports
  next to the local run's, refusing only a different `PYTHONHASHSEED`). As
  in every stage, the tool runs with `HF_HUB_OFFLINE=1`, which the local
  run's recorded environment did not set. The base needs no Hub file: every
  source is passed as a path, and the only fetch the tool itself calls is
  the Census download above (the library fetches an ASEC person archive
  only for an unmapped year, and all three are mapped).
- **The hash seed is held to the local run's.** At `4b57d15a2` the tool's
  `--stage all` parent runs each outer stage in a child interpreter with
  `PYTHONHASHSEED` defaulted to `0` (`_staged_subprocess_environment`, a
  `setdefault`, so a value already in its environment would win), records
  the value in the run config it locks (`thread_environment`, `0` when
  unset) and refuses a named stage without one. The local run recorded
  `0`, and its command set only `PYTHONUNBUFFERED`. So the runner removes
  any `PYTHONHASHSEED` the container has from the tool's environment, and
  the tool's default gives its stages `0`. A plan cannot set it. If the
  tool's environment would still carry another value, the run refuses
  before pulling or staging anything (a refusal, not charged to the
  budget) and the check reports a problem. The receipt records the
  container's value, the tool's and the stages' (`runner.python_hash_seed`),
  and step 6 refuses a run context whose
  `thread_environment.PYTHONHASHSEED` is not the local run's `0`.
- **The interpreter and platform differ; the package versions do not.**
  The local base for `4b57d15a2` ran on free-threaded CPython 3.14.7 on
  macOS arm64. The build worktree's `.venv/pyvenv.cfg` names
  `cpython-3.14+freethreaded-macos-aarch64-none`, and the run's own
  `stage_run_context.json` records `3.14.7 free-threading build … [Clang
  22.1.3]`. That venv's `numpy/__config__.py` gives its BLAS as
  `accelerate`. The image is `debian_slim` with the standard (GIL) build of
  CPython 3.14 (`IMAGE_PYTHON_VERSION`; Modal served 3.14.2 in September)
  on Linux x86_64 under gVisor. There `uv.lock` at `4b57d15a2` resolves the
  manylinux x86_64 wheel of numpy 2.4.6. The lock pins one version of each
  numeric distribution the tool fingerprints, the same six versions the
  local run recorded: numpy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0,
  quantile-forest 1.4.2, policyengine-us 2.2.1 and h5py 3.16.0. No run has
  shown that this platform reproduces the Mac's numbers. Step 6 below
  compares the only outputs the local run left, the checkpoints of its first
  two outer stages, byte for byte before the base is used. The 22 stages
  after them, the primary PUF QRF chain included, have no local output to
  compare with.

**Resources.** Class `base`: 4 cores, 112 GiB, a 16,500-second timeout.

- *Memory.* The local peak was 72.47 GB (67.5 GiB, `/usr/bin/time -l` of
  the 16 September A1d run under policyengine-us 1.819.0). That run is not
  this build. It ran commit `51c31438` (`_buildq-runtime/RECEIPT-phase2.md`
  on the build machine), and seven later commits that `4b57d15a2` includes
  changed the builder itself (`git log 51c3143829..4b57d15a2 --
  tools/build_us_puf_support_base.py`), among them the SPM independence
  role stage (`9d26595bc`) and the restored Census person columns
  (`39b8e7b63`), besides the move to policyengine-us 2.2.1. Route A's own
  run of `4b57d15a2` stopped after two outer stages, so no peak of this
  build has been measured: the first Modal receipt's `peak_rss_bytes` is
  the first. On 23 September Modal held 15 to 24 GB more than the build
  machine at the same point of materialize, so the worst case seen is about
  90 GiB; 112 GiB leaves about 22 GiB over that. The heavy class's 128 GiB would
  put the capped run over $15 (below). Memory is a request, not a limit:
  use above 112 GiB is billed at use, above the ceiling (see "How the cap
  holds").
- *CPU.* The A1d base used 6,238 CPU-s in 2,788 s of wall locally, 2.2
  cores on average. That average does not bound what the tool asks for. At
  `4b57d15a2`, with `POPULACE_FIT_PREDICT_WORKERS` and
  `POPULACE_FIT_N_JOBS` unset, the primary-QRF draw sizes its thread pool
  from `os.cpu_count()` and the forest fit passes `n_jobs=-1` (microcosm-fit
  `qrf.py`, `_predict_workers` and `_fit_n_jobs`; the worker identity
  records both, `worker_identity.py`). The local run left both unset (its
  `stage_run_context.json` records them as unset). On the build machine
  `os.cpu_count()` is 18 (`sysctl -n hw.ncpu`), so an unpinned pool there
  has 18 threads; the 2.2-core average is over the whole 16 September run
  and bounds no phase of it. Neither value has been observed in a Modal
  container. On 23 September Modal set
  `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
  `BLIS_NUM_THREADS` to the class's CPU request in the check container, and
  `nproc` reported 4 in the heavy one. That does not show what
  `os.cpu_count()` returns: GNU `nproc` takes `OMP_NUM_THREADS` as its
  minimum
  (gnu.org/software/coreutils/manual/html_node/nproc-invocation.html, read
  29 September 2026). The check now reports its own container's
  `os.cpu_count()`, affinity and thread variables (`check_container_cpu`,
  a hint from the check class), and the first measurement in the base's
  container is the receipt's `runner.os_cpu_count` and `cpu_affinity`.
  Because of that, the base's functions set a CPU limit equal to the
  request (`cpu=(4.0, 4.0)`), and Modal throttles CPU use above it (see
  "How the cap holds"). The limit may slow phases that ran on more than 4
  cores locally. The whole local run's 6,238 CPU-s is 1,560 s of 4
  cores.
- *Pinning the pools.* Setting `POPULACE_FIT_PREDICT_WORKERS` or
  `POPULACE_FIT_N_JOBS` in the plan would also bound the pools. The cost of
  that is not parity with the local run's locked config, which the Modal
  run never has: at `4b57d15a2` the tool locks resolved paths (`/work/...`
  on Modal, `/Users/...` locally) and the thread variables it reads from its
  environment (`thread_environment`: the BLAS and OpenMP counts, both
  `POPULACE_FIT_*` variables and `PYTHONHASHSEED`; `_stage_run_config`), and
  Modal set four of those thread counts in the check container where the
  local run set none. Pinning would change only `thread_environment` (and
  the controls the primary-QRF worker identity records,
  `worker_identity.py`). `compare-lineage` requires the rest of the run
  config to equal the local run's, paths by file name, and reports
  `thread_environment` without refusing on it but for `PYTHONHASHSEED`,
  which the runner holds at the local run's `0` (step 6). What is not known
  is whether the worker count changes the numbers. The pinned code says it
  does not: the draw's chunking "only changes *when* each row is computed,
  never *what* it is", and the forests are "deterministic per random_state
  regardless of worker count" (microcosm-fit `qrf.py`). No test at
  `4b57d15a2` compares two worker counts (its QRF tests pin both variables
  to 1), and no run has. The local run left both unset, so its predict pool
  was `os.cpu_count()` wide; an unpinned Modal run's pool is whatever its
  container reports. If the CPU limit changes what `os.cpu_count()` returns
  (not observed either way), the same question applies to it. Pinning is a
  call for the builder or Max, not a plan edit made here.
- *Disk.* The base writes about 50 GB: 44 GB of frame checkpoints, the
  2.35 GB H5, about 2.4 GB of staged inputs and the seeded archive with its
  extracted member. Modal gives each container a disk quota of 512 GiB by
  default (modal.com/docs/guide/resources, read 29 September 2026: "a
  per-container disk quota that defaults to 512 GiB"), so the function sets
  no `ephemeral_disk`. The same page says the worker's own SSD also limits
  writes, so the stage checks first: it refuses before staging anything
  when `/work` has less than 70 GiB free (the build machine's admission
  floor for the same command, 69 GiB). That refusal is not charged to the
  budget, and the receipt records the free space it saw
  (`runner.work_disk`).
- *Timeout and the runner's reserve.* 16,500 seconds is the 4-hour budget,
  the 30-minute runner reserve and 5 minutes for staging. The runner stops
  the tool at whichever comes first: the plan's budget, or 14,700 seconds
  after the function started (the timeout less the reserve). A slow pull
  or staging therefore shortens the tool's time, never the reserve. The
  reserve covers stopping the tool (up to 60 s), then hashing and mirroring
  the state, then the runner's identity, the receipt and the attempt's
  final commit (`RECEIPT_RESERVE_SECONDS`, 120 s, an allowance). After a
  stop or a failure the push copies the whole state, about 50 GiB
  (`BASE_MIRRORED_STATE_GIB`): the local run's 44 GB of checkpoints and
  2.35 GB H5, plus the final checkpoint's hard-linked alias
  (`stage_all.frame.h5`), which the mirror copies as a second file. After a
  clean exit it copies about 4 GB (see "What the push copies").
- *The write probe.* No container-to-volume write rate has been measured,
  so the check measures one. It copies 1 GiB to the runs volume, commits
  it, deletes it, and reports `runs_volume_write_probe`. The check fails
  when 50 GiB at that rate would not fit in the 1,620 s the reserve leaves
  for the mirror (1,800 s less the stop's 60 s and the receipt's 120 s),
  which means below about 33 MB/s. The only measured volume rate here is
  reads of at least 58 MB/s (the acceptance attempt above). Modal describes
  volume bandwidth as up to 2.5 GB/s and not guaranteed
  (modal.com/docs/guide/volumes, read 29 September 2026). One small probe
  from the check container is a hint, not a guarantee. If a mirror did run
  into the timeout, Modal would end the function before the receipt was
  written. The run's volume state would then be a partial mirror with no
  receipt, which the next attempt refuses (new `run_id`). The push copies
  the output and its evidence first (below), so even then they are the
  files most likely to be on the volume.

**What the push copies.** Every push copies the stage's `mirror_first`
paths before any other file, in this order: `base-out/`, `logs/`, then
`base-checkpoints/stage_run_context.json`, `stage_profile.json`,
`000_source_construction.frame.h5` and `001_pre_clone_enrichment.frame.h5`
(what step 6 compares). What follows depends on how the tool ended:

- *Clean exit* (return code 0, not stopped at the budget). Nothing else is
  copied. A finished `--stage all` has nothing to resume, so the other
  checkpoints (about 42 GB) are hashed into the receipt and left in the
  container. The receipt lists them under `outputs_not_mirrored`, and an
  older copy of one on the volume is removed, since it would not match the
  receipt. The push is about 4 GB instead of about 50, which also takes
  most of the mirror's time off the bill. The run's volume state then
  differs from its receipt by exactly those files: `verify-receipt` without
  `--prefix` names each one as not mirrored, and a relaunch of the finished
  run is refused, as it should be.
- *Stop at the budget, or failure.* The whole state is copied, output and
  evidence first, so the next attempt can resume (below).

**After a stop the checkpoints are mirrored.** They live in the state
directory, so the runner hashes them into the receipt and copies them to
the runs volume when the tool stops. The run is non-preemptible, so
preemption is not the risk; the budget is. At the per-chunk slowdowns
measured for materialize on Modal (3.3 to 7.6 times the build machine), the
2,788-second local base would take 2.6 to 5.9 hours, and a stop at 4 hours
without checkpoints would lose the whole run. With them, `--stage all`
resumes from the last completed outer stage (`_run_staged_all`). The tool
locks its whole run config in `stage_run_context.json` (input paths and
digests, settings, its code identity and the thread variables it sees) and
refuses a resume whose config differs. Every path in it is the same in
every attempt of the run, and so are the image, the plan's environment and
the class, so a relaunch of the same plan should resume. If Modal gave the
new container different thread variables, the tool would refuse, and the
run would need a new `run_id`. The cost has three parts:

- one copy of about 46 GB when the tool stops, hashed as it is copied;
- the same pulled back by a resuming attempt, hashed and verified in that
  one pass, with the pull's time taken from that attempt's tool budget;
- about 46 GB on the runs volume until it is deleted.

The `base-out` directory (the H5, its summary JSON and the capital-gains
tail manifest, which the tool also copies to `base-checkpoints/artifacts/`)
is mirrored either way, first.

**The budget stop.** The tool runs in its own process group. At the budget
the runner sends SIGTERM to the group and SIGKILL to whatever is left 60
seconds later. `--stage all` runs each outer stage as a child interpreter.
Stopping only the parent (as the runner did before) would leave that child
running with the log pipe open. The runner reads the pipe to its end, so it
would wait out the child's whole stage past the budget. The receipt then
says FAILED with `stopped_at_budget`, and relaunching the same plan gets
the whole budget again (see "Preemption and restarts").

A relaunch is another paid run and needs Max's go. Its ceiling at the
request is the same $14.90, because the class timeout bounds every attempt,
so two attempts can list at up to about $29.80 (plus any memory used above
112 GiB). The relaunch also gets less tool time than the first attempt. It
first pulls and verifies about 46 GB of state: up to about 13 minutes at
the volume read rate measured on 23 September (at least 58 MB/s, measured
for staging inputs), and that comes out of its tool budget. Before the
relaunch, its check hashes the same state in place on the volume, inside
the check class's 30-minute timeout.

**Cost.** At the 14,400-second budget plus 30 minutes of runner time, the
plan's list-price estimate is $14.63 non-preemptible. At the local wall it
would be $2.52. The ceiling of one attempt at the request, the class
timeout, lists at $14.90 (`estimated_usd_at_timeout` in `validate`).
Modal throttles CPU use above the CPU limit, which equals the request;
memory above 112 GiB bills above the ceiling (see "How the cap holds").
Each check lists at up to about $0.08 on top. An earlier draft of this
class had a 6-hour timeout, which listed at $19.51, over the cap. The
receipt records the container's wall and its list-price cost at the
request (`estimated_usd_container_at_list_price`) and `runner.tool_budget`
(the tool's budget and which limit set it).

**Run it.** From a checkout that has this registration and the full
history of `main` (so the pinned-commit tests run):

```bash
# 1. Inputs, on the build machine (paths from route_a.sh input_rows()).
#    upload-commands re-hashes each file and refuses the whole set if any
#    file's sha256 is not the plan's; the script it writes uploads each
#    file to cas/sha256/<digest>/<name> unless the volume already has it.
S=${STORAGE:?set STORAGE to route_a.sh STORAGE, the processed ASEC, PUF and ACS H5 directory}
E=$HOME/PolicyEngine/_buildm-runtime/inputs/asec_education
python3 tools/modal_us_stage_plan.py upload-commands --shell \
  docs/us-modal-stage-route-a-base-plan.json \
  asec_2024_h5=$S/census_cps_2024.h5 asec_2023_h5=$S/census_cps_2023.h5 \
  asec_2022_h5=$S/census_cps_2022.h5 puf_2024_h5=$S/puf_2024.h5 \
  puf_2015_csv=$S/puf_2015.csv acs_2022_h5=$S/acs_2022.h5 \
  asec_education_2022_zip=$E/asecpub23csv.zip \
  asec_education_2023_zip=$E/asecpub24csv.zip \
  asec_education_2024_zip=$E/asecpub25csv.zip \
  base_ledger_facts=$HOME/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl \
  block_ladder_npz=$HOME/PolicyEngine/_buildf-runtime/inputs/us_block_ladder_2020.npz \
  > upload-route-a-base.sh && bash upload-route-a-base.sh

# 2. Validate locally: the argv, class base (cpu_limit 4.0), the $14.63
#    estimate and the $14.90 ceiling at the request
#    (estimated_usd_at_timeout). Then run the base's tests, and confirm the
#    four pinned-commit tests passed rather than skipped (-rs lists skips).
python3 tools/modal_us_stage_plan.py validate docs/us-modal-stage-route-a-base-plan.json
uv run pytest -rs packages/microcosm-build/tests/engine_free/us/test_us_modal_stage_puf_support_base.py

# 3. Check on Modal (check class, up to about $0.08). It checks the clone
#    and branch, the crosswalk digest, the pinned tool's _parse_args on the
#    argv, and all eleven inputs hashed on the volume. It also probes the
#    runs volume's write rate (runs_volume_write_probe; the check fails if
#    50 GiB would not mirror inside the runner's reserve). Its work_disk
#    field shows what the check container's /work reports against the
#    stage's 70 GiB, and the check fails when that is less (would_refuse;
#    the paid container checks its own disk again before staging). It fails
#    too if the tool's environment would carry a PYTHONHASHSEED other than
#    none or 0 (python_hash_seed shows the container's and the tool's).
#    check_container_cpu shows its os.cpu_count(), affinity and thread
#    variables (a hint: the base's own container is another class).
MICROCOSM_MODAL_PLAN=docs/us-modal-stage-route-a-base-plan.json \
  modal run tools/modal_us_stage.py

# 4. Run it: paid and non-preemptible. The estimate is $14.63. The class
#    timeout bounds this attempt at $14.90 list at its request; Modal
#    throttles CPU use above the limit (the request), and memory above
#    112 GiB would bill above that.
MICROCOSM_MODAL_PLAN=docs/us-modal-stage-route-a-base-plan.json \
  modal run --detach tools/modal_us_stage.py --run

# 5. Fetch the receipts and base-out only, not the 44 GB of checkpoints,
#    and verify base-out against the receipt. verify-receipt refuses a
#    receipt that is not COMPLETED or says stopped_at_budget: after a stop
#    during final_export, base-out can hold a truncated H5 that the FAILED
#    receipt lists byte for byte. Check where the files landed (see Commands
#    step 5) and point --state-root at the directory that holds base-out/.
R=route-a-base-4b57d15a287c
mkdir -p modal-runs/$R/state
modal volume get microcosm-us-stage-runs runs/$R/receipts ./modal-runs/$R/
modal volume get microcosm-us-stage-runs runs/$R/state/base-out ./modal-runs/$R/state/
RECEIPT=./modal-runs/$R/receipts/all-<utc>.json
python3 tools/modal_us_stage_plan.py verify-receipt "$RECEIPT" \
  --state-root ./modal-runs/$R/state --prefix base-out --strict

# 6. Required before the base is used: compare the Modal run with the local
#    run of the same commit and inputs: its command, its inputs, the run
#    config its tool locked, and its outputs as far as the local run went.
#    It stopped after its first two outer stages, whose frame checkpoints
#    are deterministic (written without HDF5 timestamps, with no paths or
#    platform strings in them). The comparison needs the COMPLETED receipt
#    of step 5 and the Modal run's own stage_run_context.json, whose sha256
#    must be the one the receipt lists. The reference is
#    docs/us-modal-stage-route-a-base-local-reference.json.
modal volume get microcosm-us-stage-runs \
  runs/$R/state/base-checkpoints/stage_run_context.json ./modal-runs/$R/
python3 tools/modal_us_stage_plan.py compare-lineage "$RECEIPT" \
  --reference docs/us-modal-stage-route-a-base-local-reference.json \
  --run-context ./modal-runs/$R/stage_run_context.json
```

`compare-lineage` exits 0 only when all of these hold:

- The receipt says `COMPLETED` and not `stopped_at_budget` (the same test
  as `verify-receipt`).
- Its argv, with the container's paths turned into the fixture's tokens
  (`<python>`, `<input:ROLE>/<file>`, `<state>`, `<repo>`), is the local
  run's command token for token (the reference's `argv`, a copy of the
  fixture). The plan's environment is the local run's (`PYTHONUNBUFFERED=1`),
  apart from the variables the tool records in `thread_environment`.
- Each of the eleven inputs the receipt verified has the sha256 the local
  run used.
- The `000_source_construction` and `001_pre_clone_enrichment` checkpoints
  have the local run's sha256 (`be8c2689…` and `5a52f337…`).
- The Modal run locked the local run's outer-stage pipeline
  (`pipeline_sha256` `906c9c02…`, 24 stages) and the local run's run
  config. The reference keeps all 27 keys of that config
  (`_stage_run_config` at `4b57d15a2`) with each path cut to its file name
  and `YEAR=` prefixes kept, and the Modal config is compared the same way,
  so another directory is the same file and another file name is not. That
  covers every setting: seed, estimators, target year, district and ladder
  seeds, district assignment, the ladder escape hatches, the ASEC order and
  pins, and the PUF digests. It covers the code identity too:
  `source_sha256` `c02fd057…` (the digest of the committed tree of
  `4b57d15a2`, so an untracked file in either tree would change it) and the
  same six dependency versions.
- Of `thread_environment`, `PYTHONHASHSEED` is the local run's `0`. The
  tool's stage interpreters run with it (see "The hash seed is held to the
  local run's" above), so a run that recorded another value is not the
  local run's build, and the report shows both values under
  `thread_environment_compared`.

Two things are reported and never refused: the interpreter string
(`python`) and every other `thread_environment` variable whose value
differs (`thread_environment_differences`). The tool reads those variables
from its environment, Modal sets some of them itself (see "Resources"), and
whether they change the numbers has not been shown. The report says
`compared_outputs_match`, with `stages_compared` 2 of `stages_total` 24 and
the 22 stages it did not compare (`stages_not_compared`).

Read a match for what it is. It shows that the Modal platform reproduced
source construction and pre-clone enrichment byte for byte. That is real
evidence about the numerics: pre-clone enrichment already fits and draws
from a QRF (the ACS rent imputation, `impute_us_pre_subsidy_rent`, which
`with_us_housing_inputs` calls, in `housing_inputs.py` at `4b57d15a2`).
It is not evidence about the stages after it. Stages 002 onward
(clone feature extraction, the primary PUF QRF chain, QRF finalization,
the capital-gains tail, district and block-ladder assignment and the final
export) have no Mac reference, so nothing here shows that they match what
the Mac would produce. The base is then a Linux x86_64 build of this
lineage, verified against the Mac only through stage 001; say so when it
is handed on.

A frame checkpoint that differs means the Modal platform did not reproduce
the Mac's bytes. That can be the numbers or the HDF5 layout; a
dataset-by-dataset comparison tells which (the push keeps both compared
checkpoints on the volume even after a clean exit). Either way, stop:
record it in the run's notes, do not hand the base to Route A, and get
Max's call before the base is used. The same holds for any other problem
it reports: a different command, input, setting or code identity means the
Modal run is not the local run's build.

**Hand the base to Route A.** Route A's driver skips its base stage only
when `$RUN/base-sup/ACCEPTED` exists (`stage_done base` in `route_a.sh`).
Without that file, rerunning `route_a.sh` builds the base again locally. It
would resume from the failed local run's checkpoints in
`$RUN/base-checkpoints` and write its own H5 into `$RUN/base-out`. The
release would then certify that local base, not the bytes the Modal receipt
proves, because `base.sha256` is recomputed from whatever file is there.
After steps 5 and 6 pass, and before rerunning `route_a.sh`, run step 7. It
runs step 6's `compare-lineage` again itself, so no base reaches Route A
without it, whatever was run before:

```bash
# 7. $RUN is Route A's run directory for the build commit
#    (.../overnight-20260923/route-a/run-4b57d15a287c on the build machine).
#    R and RECEIPT are step 5's; the run context is step 6's download. The
#    block runs in a subshell with `set -euo pipefail`, so the first command
#    that fails (the guard, compare-lineage, the copy, verify-receipt, any
#    write) ends it and nothing after it runs. compare-lineage runs before
#    anything under $RUN is written: a Modal run that is not the local run's
#    build stops the block with $RUN untouched. Its JSON report goes to
#    ./modal-runs/$R/ first and into base-sup/ as the hand-off's evidence.
#    ACCEPTED is written last, so a stopped block never hands Route A a
#    base. What the earlier commands wrote stays (a partial $RUN/base-out,
#    perhaps base.sha256 or base-sup): look at it, remove it by hand, and
#    run the block again.
RUN=<Route A run directory>
CONTEXT=./modal-runs/$R/stage_run_context.json
LINEAGE=./modal-runs/$R/compare-lineage.json
(
  set -euo pipefail
  if [ -e "$RUN/base-sup" ] || [ -e "$RUN/base-out" ]; then
    echo "refusing: $RUN already has base-sup or base-out; never overwrite a base" >&2
    exit 1
  fi
  # Step 6, required: exits 1 (and ends the block) on any problem, which it
  # prints to stderr; its JSON report goes to $LINEAGE.
  python3 tools/modal_us_stage_plan.py compare-lineage "$RECEIPT" \
    --reference docs/us-modal-stage-route-a-base-local-reference.json \
    --run-context "$CONTEXT" > "$LINEAGE"
  mkdir -p "$RUN/base-out"
  cp -p ./modal-runs/$R/state/base-out/* "$RUN/base-out/"
  # The copies, in place, against the receipt (COMPLETED only, nothing extra).
  python3 tools/modal_us_stage_plan.py verify-receipt "$RECEIPT" \
    --state-root "$RUN" --prefix base-out --strict
  H5=$RUN/base-out/base_populace_us_2024_puf_support.h5
  echo "$(shasum -a 256 "$H5" | cut -c1-64)  $H5" > "$RUN/base.sha256"
  # The failed local run's partial checkpoints: keep them, out of the way.
  if [ -e "$RUN/base-checkpoints" ]; then
    mv "$RUN/base-checkpoints" "$RUN/base-checkpoints.local-$(date +%s)"
  fi
  mkdir -p "$RUN/base-sup"
  cp "$RECEIPT" "$RUN/base-sup/modal-receipt.json"
  # The evidence the hand-off rests on: compare-lineage's report and the
  # run context it compared.
  cp "$LINEAGE" "$RUN/base-sup/compare-lineage.json"
  cp "$CONTEXT" "$RUN/base-sup/modal-stage_run_context.json"
  RECEIPT_SHA=$(shasum -a 256 "$RECEIPT" | cut -c1-64)
  printf '{\n "status": "MODAL_RECEIPT",\n "returncode": 0,\n "refusal": null,\n "receipt": "%s",\n "receipt_sha256": "%s",\n "compare_lineage": "compare-lineage.json"\n}\n' \
    "$(basename "$RECEIPT")" "$RECEIPT_SHA" > "$RUN/base-sup/RESULT.json"
  echo "returncode=0 $(date '+%F %T') modal receipt $(basename "$RECEIPT") sha256 $RECEIPT_SHA" \
    > "$RUN/base-sup/ACCEPTED"
)
```

`RESULT.json` has the one-key-per-line shape `route_a.sh`'s
`result_accepted` reads (`"refusal": null`, `"returncode": 0`); the extra
`compare_lineage` line names the report next to it. `ACCEPTED` is what
`stage_done` checks. `base-sup/` then holds the receipt, compare-lineage's
report (`compared_outputs_match` true, 0 problems, the compared hash seed)
and the run context it compared. Then rerun `route_a.sh`. It skips the
base, keeps `base.sha256` (the H5, copied with its mtime, is not newer than
it), and logs that digest for the H5 it passes to the preflight and the
release.

The release and certification still run locally. Route A's driver takes
the base H5 from its run directory and runs the preflight, the release and
the publisher's `--preflight-only` check on the build machine. The receipt
proves which bytes Modal produced; it does not certify the base. After the
release is accepted, delete the checkpoints from the runs volume
(`modal volume rm -r microcosm-us-stage-runs runs/$R/state/base-checkpoints`).
After a clean exit that directory holds only the run context, the profile
and the two compared checkpoints (about 1.5 GB); after resumed attempts it
can also hold older checkpoints. The run's state already differs from its
receipt by the checkpoints that were never copied, so a later stage of the
same `run_id` would refuse either way; none is planned.

## Preemption and restarts

Functions run on Modal's preemptible placement unless the plan sets
`"nonpreemptible": true`. When Modal preempts a container it restarts the
function on the same input, from zero, whatever `retries` says
(modal.com/docs/guide/preemption). The 23 September acceptance run was
preempted twice. The state is mirrored and the receipt written only when
the tool exits, so a preempted attempt's tool work is lost, but it is
billed.

**The attempt ledger.** Each attempt writes
`runs/<run_id>/attempts/<stage>-<attempt id>.json` when it starts and
rewrites it every 120 seconds until it ends: through the lock wait, input
staging, the tool, hashing and mirroring. The record ends with one outcome.

| Outcome | When | Charged to the budget |
| --- | --- | --- |
| `receipt` | the tool ran and a receipt was written, COMPLETED or FAILED | no |
| `refused` | the lock or the budget stopped it before it staged anything | no |
| `error` | any exception after its first record: a digest mismatch, a failed pull or mirror, a pulled-state mismatch, a runner bug | yes |
| none | preempted (or still running) | yes |

**The budget.** A new attempt charges every unfinished attempt of the same
plan (same `run_id`, stage and plan sha256) to `max_wall_seconds`. Its tool
gets what is left, and it refuses to start with less than 60 seconds. A
preempted attempt is charged from the start of `_run_stage` to its last
record, so the charge leaves out the container's cold start and image load,
up to 120 seconds after the last record (more if a heartbeat write failed),
and the preemption grace period. A stop at the budget is not a preemption:
the tool is stopped, the state mirrored and a FAILED receipt with
`stopped_at_budget` written, so that attempt is finished and launching the
same plan again gets the whole budget again. To launch past a spent budget,
raise `max_wall_seconds` or use a new `run_id`. Any change to the plan,
`nonpreemptible` included, is a new plan sha256 and a fresh budget.

**The lock.** Two attempts of one run would race on its state directory,
so the ledger is also the run's lock. An attempt refuses to start while an
earlier attempt of the same run, of any stage or plan, is still writing its
record. A record less than 300 seconds old is ambiguous, because Modal
restarts a preempted input within moments. The new attempt waits 270
seconds (two heartbeats) and reads again. A record that moved belongs to a
running attempt, and the new one refuses; a record that did not move
belongs to a dead one, and the new attempt starts. A restart after
preemption can therefore spend up to 4.5 more minutes. There is no
override: a dead attempt stops blocking after one wait, and a running one
must not be raced. Modal volumes have no atomic lock, so two attempts that
start within one volume commit of each other can both pass; run one stage
of a run at a time. The check reports recent records without waiting.

**The state after a cut.** The mirror writes each file under a temporary
name and renames it, so it never leaves a half-written file. A preemption
during mirroring can still leave some files new and some old. The next
attempt verifies the pulled state against the run's latest receipt and
refuses that mix; start a new `run_id`.

**When to set `"nonpreemptible": true`.** Non-preemptible placement costs
three times the list price for CPU and memory (modal.com/docs/guide/preemption
and modal.com/pricing, read 23 September 2026): about $3.63 an hour for the
heavy class instead of $1.21. Set it for heavy stages expected to run
longer than about an hour. The acceptance run was preempted twice in about
2.97 hours of running (after 54 minutes and after 2 hours 4 minutes), a
rate λ of about 0.67 an hour. Each preemption restarts the stage from zero,
so at that rate a stage of T hours on preemptible placement takes
(e^(λT) − 1)/λ hours in expectation:

| Stage length | Uninterrupted on preemptible | Expected preemptible hours and cost | Non-preemptible cost |
| --- | --- | --- | --- |
| 1 h | 51% | 1.4 h, $1.73 | $3.63 |
| 2 h | 26% | 4.2 h, $5.12 | $7.27 |
| 3 h | 13% | 9.7 h, $11.78 | $10.90 |
| 6 h | 2% | 83 h, $101 | $21.81 |

Past about an hour a preemptible attempt is more likely than not to be cut
short. Non-preemptible placement pays for itself in expected cost from
about 2.8 hours. Before that it costs up to three times as much for a
result that arrives on time. A `max_wall_seconds` budget also usually stops
a long preemptible run before it finishes. Two preemptions in one run is a
small sample: a 95% interval for the rate runs from about 0.08 to 2.4 an
hour, so read the table as an order of magnitude. Light stages and short
heavy stages (calibrate took 7 minutes locally) are cheaper preemptible.

## Data placement

Inputs go to the `policyengine` Modal workspace. Upload only files that are
allowed there. On 29 September 2026 Max ruled that the IRS PUF files may sit
on the `policyengine` Modal volumes for Route A's base ("puf there is
fine"): the processed PUF (`puf_2024.h5`, `--puf-h5`) and the restricted
TY2015 IRS PUF CSV (`puf_2015.csv`, `--puf-source-year-csv`), both on
`microcosm-us-stage-inputs` under `cas/sha256/`. The ruling covers that
workspace's volumes and the base stage; it does not put the PUF anywhere
else (the Hub, logs, receipts). The base's checkpoints and output on
`microcosm-us-stage-runs` are built from the PUF. This change reads the
ruling as covering them too; that is this change's reading, not Max's
words, which named the PUF. Logs hold only what the tool prints. Receipts
hold digests, sizes and paths, never file contents.

## What this does not cover yet

- **The graph-native line (#893).** The DAG executor lives on #893's branch,
  not on main. Registering it means one more `ToolSpec` with its CLI, inputs
  and state layout. The executor runs nodes one after another, and the 1/15
  run peaked at 46.7 GB (#956). A full-source run needs the byte-transport
  and parallel-executor work first, plus a resource class sized from a
  measured full-scale peak.
- **The base stage on Modal is unmeasured.** It is registered (see
  "Route A's base stage") and sized from an older local run (A1d, seven
  builder commits before `4b57d15a2`); its Modal wall, peak, CPU visibility
  (`os.cpu_count()` in its container) and the runner's hashing and
  mirroring time are estimates until the first receipt, whose
  `peak_rss_bytes` is the first peak measured for this build. The
  check's write probe is the only measurement of the runs volume's write
  rate before then. Step 6 shows whether the platform reproduces the local
  bytes of the first two outer stages only. The local run left no output
  for the other 22, so whether Modal reproduces the Mac there is not known
  and cannot be checked against this reference.
- **The staging stage (`tools/build_us_acs_multispine_base.py`)** is not
  registered either. It takes a directory input (`--inputs-dir`, the ACS
  PUMS archive cache), which the plan format does not support. Supporting it
  would take an archive digest plus extraction.
- **Work lost to preemption, and memory kills.** A preempted attempt's
  tool work is lost: the state is mirrored only when the tool exits, and
  materialize has no resume point inside its chunk loop. (The base resumes
  from its last completed outer stage, but only from state an earlier
  attempt mirrored, that is, after the tool exited.) Non-preemptible
  placement avoids preemption (see "Preemption and restarts"). The heavy
  class sets a memory request but no hard limit. What Modal does with a
  container killed for memory has not been observed here.
- **Certification.** A receipt proves which bytes a stage produced. It does
  not certify a release. Preflight (`tools/preflight_us_release_gates.py`)
  and certification still run on the output as before.
