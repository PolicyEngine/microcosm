# Route A release driver

`route_a.sh` builds a US PUF-support base, builds a fiscal-refresh release
candidate from that base, runs two release-gate preflights and runs the
publisher with `--preflight-only`. It leaves a hand-off with `published: false`.
Publication remains a separate manual step that Max authorizes.

The driver, `supervise.py`, `sample_series.py`, `check_flags.py` and
`with_hf_token.sh` live together here. The driver, supervisor and sampler
originated in the off-repository Route A tooling. The release stage now enables
staging telemetry by default; `ROUTE_A_STAGING=0` restores `--no-staging`.

## Configure and run

Copy the example and edit its paths and commit before running:

```bash
cp tools/route_a/route_a.env.example tools/route_a/route_a.env
# Edit tools/route_a/route_a.env, then resolve without launching a stage:
bash tools/route_a/route_a.sh --resolve
# After reviewing inputs.resolved.json and every BLOCKER in resolve.log:
bash tools/route_a/route_a.sh
```

`ROUTE_A_ENV=<settings-file>` selects another settings file. The local
`route_a.env` is ignored by git. The example uses shell defaults, so a
nonempty environment override wins. `ROUTE_A_STAGING` accepts only `0` or `1`;
an explicitly empty value is refused. Do not put credentials in either file.
`RELEASE_EXTRA_ARGS` and `PREFLIGHT_NEW_LINEAGE_ARGS` retain shell word splitting;
embedded quotes do not quote paths, and paths used there must have no spaces.

All machine locations come from the settings file:

| Setting | Purpose |
| --- | --- |
| `PE`, `MAIN`, `WT_ROOT` | PolicyEngine root, git repository used for fetching refs and creating worktrees, and detached build/preflight worktree parent |
| `RUN_ROOT` | Driver lock, logs, input resolution, terminal hand-offs and `run-<12-character-commit>` directories |
| `CHAIN_ROOT`, `CHAIN_LOG` | Previous overnight chain's supervisor run parent and chain log |
| `DISK_PATH` | Filesystem checked by the driver's free-space wait |
| `UV`, `PYTHON` | uv executable and engine-free interpreter for the static parser check |
| `STORAGE`, `EDU` | Processed ASEC/PUF/ACS inputs and the three ASEC education person archives |
| `FEED`, `LADDER`, `SSI`, `SCF` | Pinned Ledger feed, block ladder, SSI prior basis and SCF summary extract |
| `AGENT_SECRET` | Credential helper executable used by the release wrapper |
| `EXPORT_MASS_REF_H5`, `QRF_TAIL_EXCLUSIONS` | Export-mass reference and optional tail exclusion register |

Choose a pushed `COMMIT`; the example deliberately leaves it empty. By default
the build commit must be reachable from `origin/main`, carry the SPM role stage,
pin the expected feed and ASEC inputs, carry the expected district crosswalk,
and declare the driver's base and release flags. `PREFLIGHT_COMMIT` defaults to
the build commit; with gate preflights enabled it must be on a remote branch.
The static release flag check
reads `add_argument` declarations without importing the release tool, including
flags in `RELEASE_EXTRA_ARGS`.

`--resolve` (also `--dry-run`) fetches refs, hashes registered inputs, writes
`inputs.resolved.json`, reports commit blockers and launches no stages. It exits
nonzero for failed fetches or input problems; commit blockers are reported in
`resolve.log` and do not alone change that mode's exit status. A real run refuses
both input problems and commit blockers.

The example retains the historical d122 dense whole-base national/state
settings, batch size 2000, export-mass reference and d490 tail register from the
published run. Those are operator settings, not a new waiver. The driver permits
`--dense-default-dataset` only with `DENSE_RELEASE_D122=1`, and refuses the
release-defeating, selection, evidence, exact-k, staging and driver-owned
overrides listed in `refuse_bad_release_args`, including `--flag=value` forms.
Use `ROUTE_A_STAGING=0` to opt out; `--no-staging` in extra arguments is refused.

The next base uses the d713 CT-fixed ladder at
`$PE/_build_artifacts/us-ct-cbsa/us_block_ladder_2020.npz`, verified against
sha256 `6840b990acdfa2003d7723da5595e3cfff7205a4fa3d8d6206ed835c85c3f233`
and 18,991,218 bytes. Use a fresh `RUN_ROOT` for that attempt: an existing
`base-sup/ACCEPTED` skips the base stage. Merely pointing `LADDER` at new bytes
does not rebuild a previously accepted base.

The driver uses macOS `stat -f` and requires bash, git, openssl, uv and the
configured paths. It creates clean detached worktrees at the chosen commits,
runs `uv sync --all-packages --locked --extra us`, and checks that their Python
can import psutil and policyengine-us. The supervised stages use that worktree's
interpreter.

## Stages and gates

Before launching, the driver verifies every registered input digest and recorded byte
size, checks commit pins, and takes a driver lock. It waits while the old
overnight chain is alive, unless `EARLY_START_AFTER_HEAVY=1` and its log records
`local release preflight passed`. It verifies inputs again after that wait.

| Stage | Command and acceptance |
| --- | --- |
| `prefetch` | Fetches the ASEC unemployment archive, SCF summary/full extracts, SIPP financial-assets/tips donors and ORG donor through the release's fetch helpers. Exit 0 is required before release. An initial failure lets the base proceed; release retries it. |
| `base` | Runs `build_us_puf_support_base.py --stage all` on the pinned raw inputs, with seed 0, 32 estimators, target year 2024, district assignment/crosswalk and block ladder. Requires exit 0 and the expected H5, then hashes it. |
| `preflight-base` | Runs `preflight_us_release_gates.py` on that H5 and the pinned feed, with the configured new-lineage mode and optional export-mass reference. Accepts 0 (clean) or 2 (AT-RISK); `RELEASE_DESPITE_PREFLIGHT_FAIL=1` also accepts 1 here. |
| `release` | Runs `build_us_fiscal_refresh_release.py --base-h5` with the feed, SSI basis, SCF extract and configured release arguments. Requires exit 0 and `release_manifest.json`. Staging telemetry is enabled for this stage. |
| `preflight-release` | Repeats the release-gate preflight with the candidate's release manifest. Accepts 0 or 2. |
| `publisher-preflight` | Runs `microcosm-publish-release --preflight-only` against the local release directory and artifacts. Requires exit 0. The publisher calls `prepare_release` and returns before its publishing call. |

Both gate preflights and the publisher preflight remove the HF credential
variables and Slack webhook variables, set `POPULACE_RELEASE_ENV=/dev/null` and
`HF_HUB_OFFLINE=1`. The base and prefetch retain their original command and
`PYTHONUNBUFFERED=1` configuration. An empty `PREFLIGHT_NEW_LINEAGE_ARGS` records
both gate preflights as `BLOCKED`, permits the build, and ends with
`ROUTE_A_BLOCKED.json` and exit 4 instead of `ROUTE_A_DONE.json`.

Every stage runs through the supervisor's admission check and limits:

| Stage | RSS cap (GiB) | RAM admission (GiB) | Disk admission (GiB) | Wall limit (seconds) | CPU limit (seconds) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefetch | 16 | 12 | 26 | 10800 | 20000 |
| base | 100 | 80 | max(26, 70 − checkpoint GiB) | 21600 | 64800 |
| gate preflights | 24 | 16 | 22 | 3600 | 7200 |
| release | 110 | `RELEASE_RAM_GIB` (example: 50) | 45 | `RELEASE_WALL_SECONDS` (example: 345600) | 2000000 |
| publisher preflight | 24 | 16 | 22 | 7200 | 14400 |

The driver waits up to `RESOURCE_WAIT_HOURS` (example: 12) for admission and
retries a supervisor admission race every five minutes. Every config includes
a 20 GiB disk floor that must persist for 180 seconds before `DISK_FLOOR` kills
the process group. The supervisor also enforces RSS, wall, CPU, a 2 GiB size
limit for its own output directory and a 512 MiB child-log limit. The output
directory limit does not count the build's separate artifacts/checkpoints.

For each local attempt, the driver writes `<stage>-config.json` and the
supervisor records admission. Admitted launches also record `COMMAND.json`,
`PID.json`, heartbeat, result and child-log files under `<stage>-sup/`;
refused admission writes a result without launching the command.
The sampler appends process-tree RSS/CPU,
available RAM, swap and disk to `<stage>.series.csv` every 30 seconds until a
result appears or the supervisor disappears. These resource records stay local.

On restart, accepted stages are skipped; a result with `refusal: null` and an
allowed return code can be accepted without rerunning. A live prior supervisor
is waited for, and failed/refused directories are moved aside. Base stages
resume by completed checkpoint prefix. Failed release attempts get a new
release ID while retaining the shared checkpoint root. With
`PRUNE_BASE_CHECKPOINTS=1`, base `*.frame.h5` files are removed after hashing the
finished H5. Terminal hand-offs record the candidate paths and preflight
verdicts, plus a suggested manual publication command; they never execute it.

## Published run's release argv

The following preserves the argv order and values in the historical
`run-4b57d15a287c/release-config.json`, replacing only machine paths with
variables. `W` is that run's build worktree, `RUN` its run directory,
`QRF_TAIL_EXCLUSIONS` its `qrf_tail_exclusions_routea_d490_20260930.json`, and
`EXPORT_MASS_REF_H5` its `populace_us_2024.h5` reference. Its config environment
was `PYTHONUNBUFFERED=1`.

```bash
RID=populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z
"$W/.venv/bin/python" -B tools/build_us_fiscal_refresh_release.py \
  --base-h5 "$RUN/base-out/base_populace_us_2024_puf_support.h5" \
  --ledger-facts "$FEED" \
  --ledger-facts-sha256 b85437390021777e746f507c5890305496baf5fc7f2c78ba08ddb090f4839801 \
  --qrf-tail-concentration-exclusions "$QRF_TAIL_EXCLUSIONS" \
  --ssi-take-up-prior-weight-basis "$SSI" \
  --ssi-take-up-prior-weight-basis-sha256 25fe8af50a99d717f3408b2de7f0849d2307d4f05b1a7d55d2703999002fff0a \
  --scf-summary-extract "$SCF" \
  --out "$RUN/release-out/$RID" \
  --release-id "$RID" \
  --checkpoint-root "$RUN/release-checkpoints" \
  --seed 0 --no-staging \
  --maximum-microsim-batch-size 2000 \
  --target-surface national_state \
  --dense-default-dataset \
  --export-input-mass-reference-h5 "$EXPORT_MASS_REF_H5"
```

That config capped RSS at 110 GiB, admitted at 50 GiB available RAM/45 GiB free
disk, and used wall/CPU limits of 345600/2000000 seconds. This is a historical
receipt of the published `4b57d15a2` run, not the command to launch the next
attempt. This migration does not modify its files or its publication.

## Modal base hand-off

Follow [the Modal runbook](../../docs/us-modal-stage-runbook.md#route-as-base-stage)
to verify a completed base receipt and compare its lineage before copying any
base into the Route A run directory. Its hand-off block refuses existing
`base-out` or `base-sup`, runs `compare-lineage`, copies `base-out`, verifies the
copies with `verify-receipt --prefix base-out --strict`, writes `base.sha256`,
moves old local checkpoints aside, stores the receipt/comparison/run-context
evidence in `base-sup`, and writes `ACCEPTED` last. Set that block's `RUN` to
`$RUN_ROOT/run-<12-character-build-commit>` before rerunning this driver.

The committed Modal example plan and local reference target the historical
`4b57d15a2` base and its `7ba39b95…` ladder. They do not qualify a new CT-fixed
base: its plan and comparison evidence must match the chosen commit and CT
ladder. The existing comparison covers only the first two of 24 outer stages;
it does not prove Mac/Linux equality for the remaining stages. The receipt
identifies produced bytes; release gates and certification still run locally.

## Staging and the Build progress tab

For a new release with staging enabled, the config contains the wrapper path
and credential-helper path followed by the release argv. The wrapper disables
shell tracing, removes inherited HF credential aliases, calls the configured
`agent-secret get HUGGING_FACE_TOKEN_MAX` at run time, refuses a failed/empty
lookup, exports `HF_TOKEN` and immediately `exec`s the release. It does not put
the returned value into argv, a config or a log. Do not fetch a token before
starting the driver. With `ROUTE_A_STAGING=0`, the release gets `--no-staging`,
inherited HF credential variables are removed and the wrapper is not called.

The release parser defaults to `policyengine/populace-us-staging` and reads
`POPULACE_STAGING_REPO_ID` from its environment. Export that variable in the
settings file to override the destination; a blank value uses the default.
The upload storage constructs `HfApi()` with ambient credentials, so the
wrapper's `HF_TOKEN` supplies authentication. Uploads are best-effort and stop
after three consecutive failures; local telemetry continues to be written.

Telemetry starts inside the release after compiling its target registry;
starting the driver or waiting for admission creates no staging run. Once
uploads reach the dashboard's staging repository, the records for its Build
progress tab include the release's run ID and candidate release ID, current status/stage,
timestamps, messages and stage details from `progress.json`, the event history
from `events.ndjson`, and calibration epoch/loss/budget-search information from
`calibration_progress.json`. The builder emits stages for loading the base,
source enrichments, target compilation, calibration, export/gates, diagnostics
and manifests, and uploads attached diagnostics as they are written. Completion
records status `passed`; reported failures record status `failed`.

Telemetry is written locally under
`<release-out>/staging/runs/<release-id>/` and uploaded under
`runs/<release-id>/`, with `run_manifest.json`, `latest_staging.json` and
`runs.json` allowing discovery. The base, donor prefetch, offline preflights
and supervisor resource series do not emit these staging records. If uploads
fail, dashboard visibility is not guaranteed. A staging upload does not publish
the release or update production `latest.json`.
