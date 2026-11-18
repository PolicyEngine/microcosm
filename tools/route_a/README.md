# Route A release driver

`route_a.sh` builds a US PUF-support base, then a fiscal-refresh release
build of the national dataset from that base, and stops. The build is written
and checked by its own gates, but it is not publishable: the run ends with
`ROUTE_A_BUILT.json`, so the build can be evaluated first (microcosm#1585 item
11, PR 4a). A build has no tier: its own files are in `<release-out>/build`,
and certification places them in the release of the tier it chooses.

`route_a.sh --certify` (or `ROUTE_A_CERTIFY=1`) on the same `RUN_ROOT` skips
the accepted stages and certifies that build with
`tools/certify_us_national_build.py`, which runs the release's post-export
checks on the written H5 and writes the manifests publication requires.
`ROUTE_A_CERTIFY_TIER` chooses the tier: `certified` (the default) certifies
the build into `microcosm-us-2024-national`; `evidence` certifies it into
`microcosm-us-2024-evidence`, never the US default, once every failure has an
owner, from `ROUTE_A_EVIDENCE_FAILURE_OWNERS` (an absolute path to a JSON
object of failure pattern and owner issue) or the standing register,
`packages/microcosm-build/src/microcosm/build/us/evidence_failure_owners.json`.
Once the build is accepted, the driver does not hash the raw inputs again,
since certification reads none of them. It then runs
`preflight_us_release_gates.py` against the release manifest and
`microcosm-publish-release-build` with `--preflight-only`, and leaves a
hand-off with `published: false`. Publication remains a separate manual step
that Max authorizes. Only a run with `--certify` is a from-scratch build and
certification receipt: a run that stops at `ROUTE_A_BUILT.json` has not been
through the post-export checks or either preflight.

A build whose release gates fail still writes its H5 and its record and exits
3. The release stage accepts that exit, so the build is kept: a run without
`--certify` ends with `ROUTE_A_BUILT.json` (`release_gates_failed: true`) and
exit 3, and a run with `--certify` certifies it only at the evidence tier,
refusing at the certified tier before the certify stage starts. Certifying at
a later commit, which `tools/certify_us_national_build.py` allows when that
commit changes only tests, docs, changelog fragments and the standing
register, is done by hand from a checkout of that commit; the driver certifies
in the build's own worktree.

The driver, `supervise.py`, `sample_series.py`, `check_flags.py` and
`with_hf_token.sh` live together here. The driver, supervisor and sampler
originated in the off-repository Route A tooling. The release and certify
stages enable staging telemetry by default; `ROUTE_A_STAGING=0` restores
`--no-staging` for both; when the build staged and its certification does not,
the build manifest records the opt-out.

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
| `STORAGE`, `EDU` | Processed ASEC/PUF/ACS inputs and the pooled years' three ASEC Census person archives (see "ASEC pool") |
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

## ASEC pool

The base pools main's default ASEC years,
`asec_sources.ASEC_DEFAULT_POOL_INCOME_YEARS`: income years 2023-2025 (ASEC
2024-2026), passed newest first. Max moved Route A to this pool from the first
certified build after November 2026 (d2603, 7 November 2026). `STORAGE` holds
`census_cps_2025.h5`, `census_cps_2024.h5` and `census_cps_2023.h5`, and `EDU`
holds `asecpub26csv.zip`, `asecpub25csv.zip` and `asecpub24csv.zip`.
`tools/fetch_us_asec_sources.py` resolves each H5 against its pin and prints
its path; copy or link the file into `STORAGE` under its own name.
`fetch_asec_education_assistance_source(year, cache_dir=EDU)` downloads an
archive into `EDU`, verifies it and extracts its person file there.
`--resolve` hashes them all. The base also reads the LKWEEKS sidecar, the
income-2022 archive `asecpub23csv.zip`, in every mode. The driver passes no
path for it: the prefetch stage fills the fetcher's cache,
`~/.cache/microcosm/cps/asec_2023/`, and the base checks the pin there.

The driver writes each year and pin out. Its build-commit check refuses a
commit whose `asec_sources.py` or `education_assistance_source.py` lacks one of
them. `test_route_a_asec_pool.py` fails when the pooled years leave the
default pool or a pin leaves its registry, so the PR that pins a newer year
moves Route A too. Builds through the November 2026 certified build pooled
2022-2024. Each keeps the driver of its own commit, and main keeps their Modal
registrations, so their plans still validate.

## Resolving

`--resolve` (also `--dry-run`) fetches refs, hashes registered inputs, writes
`inputs.resolved.json`, reports commit blockers and launches no stages. It exits
nonzero for failed fetches or input problems; commit blockers are reported in
`resolve.log` and do not alone change that mode's exit status. A real run refuses
both input problems and commit blockers.

### A build from main

The feed and crosswalk defaults are the published run's, so a build from a
newer main commit usually needs settings beyond `COMMIT`. Run `--resolve`
first and set what its blockers name:

- **Feed.** When `us/chronicle_feed.json` at the commit pins another
  `facts_sha256`, the blocker names it. Set `FEED` to that feed's file,
  `FEED_SHA` to the digest and `FEED_BYTES` to the file's size.
- **Crosswalk.** When the packaged congressional-district vintage crosswalk
  changed, the blocker gives the commit's digest. Set `CROSSWALK_SHA` to it
  once the change is reviewed.
- **Interpreter.** A fresh worktree syncs with
  `uv sync --python "$UV_SYNC_PYTHON"` (default `3.14`). Without it uv takes
  the newest installed CPython, and the lock's torch wheels stop at 3.14. The
  sync marker records the setting, and a resumed chain whose setting changed
  stops rather than keep the old venv silently. A worktree synced before the
  marker recorded it keeps its venv whatever the setting, with a log line
  saying so, so an in-flight chain is not re-synced under it; remove the
  marker to re-sync. The sync drops `UV_FROZEN`: uv refuses `--locked`
  beside `UV_FROZEN=1`, which a profile shell may export.
- **Emitter.** `--resolve` reports the `microcosm-emitter` the release would
  start against the commit's `minimum_emitter_version.json`.
  `check_emitter.py` repeats the release's checks without importing
  microcosm:
  - the command is `MICROCOSM_EMITTER_COMMAND`, else `microcosm-emitter` on
    `PATH`;
  - `version` must exit 0 within 15 s and print a JSON object;
  - the version must be a release at or above the floor;
  - with telemetry required, the protocol list must hold version 1.

  With `MICROCOSM_TELEMETRY_REQUIRED=1` a failed check is a blocker, because
  the release would stop before any work. Otherwise it is a note: the release
  would run without collector telemetry or staging hand-overs. A relative
  command path is refused, because the stages run in the build worktree. A
  `MICROCOSM_TELEMETRY_REQUIRED` other than `1`, `0` or unset is a blocker, and
  so is a floor file without an `X.Y.Z` release, because the release refuses
  both. The driver exports both settings when the settings file sets them.
  Publisher preflight starts no emitter.
- **Location rule.** `LOCATION_RULE` sets how the base locates households
  (microcosm#1047, #696).
  - `legacy`, the default, keeps the published run's command: a congressional
    district drawn from SOI return counts, then a block within it.
  - `block_v1` draws one 2020 census block per household within its CPS ASEC
    source geography, and takes every other geography from that block. The
    base then drops `--ledger-facts`, `--assign-congressional-districts` and
    `--congressional-district-seed`, and passes
    `--location-rule block_v1 --location-seed 0`. `--resolve` also checks that
    the base tool at the commit declares those two flags.
  - The driver refuses any other value. `block_v1` needs a block ladder that
    carries each block's PUMA (see **Block ladder**).
  - `block_v1` is for a base that the driver builds itself. The Modal base
    registration keeps the legacy location flags, so a Modal base is always a
    `legacy` base (see "Modal base hand-off").
- **Block ladder.** `LADDER_SHA` and `LADDER_BYTES` default to the published
  run's ladder (d713, rebuilt at microcosm#1072). A build with another ladder
  sets `LADDER`, `LADDER_SHA` and `LADDER_BYTES` together.
  - Under `block_v1`, `verify_inputs` also lists the ladder's arrays with
    `PYTHON`. It refuses a ladder without the per-block `puma` array that
    `load_us_location_ladder` requires, and names a file it cannot read.
  - `tools/build_us_block_ladder_artifact.py` on main writes that array.
    Rebuild the ladder with it and set the three settings to the new file.

The example retains the historical d122 dense whole-base national/state
settings, batch size 2000, export-mass reference and d490 tail register from the
published run. Those are operator settings, not a new waiver. The driver permits
`--dense-default-dataset` only with `DENSE_RELEASE_D122=1`, and refuses the
release-defeating, selection, evidence, exact-k, staging and driver-owned
overrides listed in `refuse_bad_release_args`, including `--flag=value` forms.
Use `ROUTE_A_STAGING=0` to opt out; `--no-staging` in extra arguments is refused.

By default the base uses the d713 CT-fixed ladder at
`$PE/_build_artifacts/us-ct-cbsa/us_block_ladder_2020.npz`, verified against
`LADDER_SHA` (default
`6840b990acdfa2003d7723da5595e3cfff7205a4fa3d8d6206ed835c85c3f233`) and
`LADDER_BYTES` (default 18,991,218). Use a fresh `RUN_ROOT` for that attempt: an existing
`base-sup/ACCEPTED` skips the base stage. Merely pointing `LADDER` at new bytes
does not rebuild a previously accepted base.

The driver uses macOS `stat -f` and requires bash, git, openssl, uv and the
configured paths. A file openssl cannot read is an input problem or a failed
run, never an empty digest. Under launchd, macOS can refuse the system's own
tools access to an external drive: on 2026-11-17, in a launchd job,
`/usr/bin/openssl`, `/usr/bin/shasum` and `/usr/bin/python3` got "Operation
not permitted" there, while a Homebrew openssl and the worktree's Python
read the same files. Put Homebrew's `bin` before `/usr/bin` on such a job's
`PATH`, or copy the inputs to the internal disk. It creates clean detached worktrees at the chosen commits,
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
| `release` | Runs `build_us_fiscal_refresh_release.py --base-h5` with the feed, SSI basis, SCF extract and configured release arguments; its `--release-id` is the build ID. Accepts exit 0, or 3 for a build that failed its release gates once its H5 and record were written, and requires `<release-out>/build_record.json`. Staging telemetry is enabled for this stage. Without `--certify` the run ends here with `ROUTE_A_BUILT.json`. |
| `certify` | With `--certify` only: runs `tools/certify_us_national_build.py <release-out>`, with `--evidence-release` (and `--evidence-failure-owners`) when `ROUTE_A_CERTIFY_TIER=evidence`; with `ROUTE_A_CERTIFY=1` the commit check, before any stage, requires the certification parser at the build commit to declare them. Certification stages where its build staged, in a run of its own, so with `ROUTE_A_STAGING=1` it gets the release stage's staging credential; with `ROUTE_A_STAGING=0` it runs as the release stage does, online without the HF credential variables, and with `--no-staging`. Requires exit 0 and `<release-out>/releases/<release>/release_manifest.json`, `<release>` being `microcosm-us-2024-national` or, at the evidence tier, `microcosm-us-2024-evidence`. |
| `preflight-release` | Repeats the release-gate preflight with the candidate's release manifest. Accepts 0 or 2. |
| `publisher-preflight` | Runs `microcosm-publish-release-build --preflight-only` against the release directory certification placed the build in, `<release-out>/releases/<release>`, and the artifacts. Requires exit 0. The command checks the build with its release contract (the evidence contract for the evidence release), prints what it would publish, and starts no emitter. It takes no `--extra-file`: the command publishes every artifact the release manifest lists, including the two SCF education-debt fit tables of a release run with `--scf-education-debt` in `RELEASE_EXTRA_ARGS`. |

Both gate preflights and the publisher preflight remove the HF credential
variables and Slack webhook variables, set `POPULACE_RELEASE_ENV=/dev/null` and
`HF_HUB_OFFLINE=1`. The base and prefetch retain their original command and
`PYTHONUNBUFFERED=1` configuration. An empty `PREFLIGHT_NEW_LINEAGE_ARGS` records
both gate preflights as `BLOCKED`, permits the build, and ends with exit 4:
after a run without `--certify`, with `ROUTE_A_BUILT.json`; after
certification, with `ROUTE_A_BLOCKED.json` instead of `ROUTE_A_DONE.json`.

Every stage runs through the supervisor's admission check and limits:

| Stage | RSS cap (GiB) | RAM admission (GiB) | Disk admission (GiB) | Wall limit (seconds) | CPU limit (seconds) |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefetch | 16 | 12 | 26 | 10800 | 20000 |
| base | 100 | 80 | max(26, 70 − checkpoint GiB) | 21600 | 64800 |
| gate preflights | 24 | 16 | 22 | 3600 | 7200 |
| release | 110 | `RELEASE_RAM_GIB` (example: 50) | 45 | `RELEASE_WALL_SECONDS` (example: 345600) | 2000000 |
| certify | 110 | `RELEASE_RAM_GIB` (example: 50) | 45 | `CERTIFY_WALL_SECONDS` (default: 345600) | 2000000 |
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
finished H5. A run without `--certify` ends with `ROUTE_A_BUILT.json`, which
records the build, its directory and record, whether its release gates
failed and the base preflight's verdict, and says how to certify it. A failed
certification is its own stage, so a rerun with `--certify` repeats only it,
never the build. After certification the hand-off file written at the end
(`ROUTE_A_DONE.json` or `ROUTE_A_BLOCKED.json`), which replaces
`ROUTE_A_BUILT.json`, records the tier, the release
and the published build ID, the candidate paths and preflight verdicts, plus
two manual commands the driver never runs: `publish_command` publishes the
build through the emitter into its folder of the release certification chose,
and `promote_command` then makes it that release's current build (the root
README's "Releasing & alerts").

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

A Modal base is a `legacy` base. Its registration passes the legacy location
flags, and no plan can pass `--location-rule`. Do not hand a Modal base to a
run with `LOCATION_RULE=block_v1`. The driver skips a base stage that is
already accepted, so the release would use the `legacy` base.

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
starting the driver. The certify stage starts the same way. With
`ROUTE_A_STAGING=0`, the release and certification get `--no-staging`,
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
`calibration_progress.json`. That file and the hosted telemetry carry each
solve's first and last epoch and every tenth epoch between them, the rule the
UK build follows too (`microcosm.build.calibration_epoch_telemetry`); the
stride grows past ten only when the run's solves total more than 24,000
epochs. Because each solve's rows start at epoch 1, the epoch falls wherever
the refit begins, even when `--epochs` is ten or fewer. The builder emits
stages for loading the base, source enrichments, target compilation,
calibration, export/gates and diagnostics, and uploads attached diagnostics
as they are written. Completion records status `passed`; reported failures
record status `failed`. Certification opens a staging run of its own,
`<release-id>-certify`, where its build staged: its stages are the post-export
checks and the manifests, it attaches their files, and a failed certification
records `failed` there. The release's build manifest names that run as its
staging run and the build's inside it, so the publisher's staging check reads
the run that scored what it publishes.

Telemetry is written locally under
`<release-out>/staging/runs/<run-id>/` and uploaded under `runs/<run-id>/`,
with `run_manifest.json`, `latest_staging.json` and `runs.json` allowing
discovery. The base, donor prefetch, offline preflights
and supervisor resource series do not emit these staging records. If uploads
fail, dashboard visibility is not guaranteed. A staging upload does not publish
the build or change `releases.json` in `policyengine/populace-us`.

## 18 October rerun (d1386)

Use [the prepared rerun sequence](../../docs/us-route-a-rerun-20261018.md)
and [its settings template](../../docs/us-route-a-rerun-20261018.env.example).
For a newer committed feed/crosswalk, set `FEED_SHA`, `FEED_BYTES` and
`CROSSWALK_SHA` together with the corresponding paths after checking the chosen
commit. Historical defaults remain unchanged. `BASE_LEDGER_FACTS` and its SHA
can intentionally retain the old base-only feed while the release uses main's
new pin. An actual machine-local `route_a.env` must stay outside Git.
