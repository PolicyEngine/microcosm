# US national build runbook: Route A on policyengine-us 2.145.3

This runbook is for the operator who runs the next US national build. That is the
first build on policyengine-us 2.145.3, policyengine-core 3.36.1 and
spm-calculator 1.1.0, with `block_v1` household location. The build runs through
`tools/route_a/route_a.sh` (see [tools/route_a/README.md](../tools/route_a/README.md)).
Since #1672 the driver builds the release and stops with `ROUTE_A_BUILT.json`;
`route_a.sh --certify` then certifies that build and ends with a hand-off. It
publishes nothing.

Certification takes about two days, so for this build it waits. The build is
probed on a household sample first, in about 10 hours (step 5). María then
assesses the build and the probe and decides whether to certify it (step 6).
The operator certifies it only when she asks (step 7).

The settings below come from María's rulings of 17 and 18 November 2026. On the
hosts, staging and publication she ruled on 18 November:

- **M1, base host:** the base runs locally for now. A `block_v1` base cannot run
  on Modal today, since the Modal plan keeps the legacy location flags.
- **M2, build and certification host:** a laptop with enough memory for the
  limits in section 1, with the PUF inputs on it.
- **M3, staging upload:** yes. With `ROUTE_A_STAGING=1` the build and its
  certification upload progress and diagnostics to
  `policyengine/populace-us-staging`.
- **M4:** #1365 is closed as superseded by #1676.
- **M5, publication and promotion:** María decides, and coordinates with Max.

## 1. Host

| Need | Value | Where it comes from |
| --- | --- | --- |
| OS | Linux or macOS | The driver uses GNU or BSD `stat` and `df -P` |
| Tools | bash, git, openssl, pgrep, uv, CPython 3.14 | `UV_SYNC_PYTHON=3.14`: the lock's torch wheels stop at cp314 |
| Base stage | 80 GiB free RAM to start; RSS cap 100 GiB; 6 h wall cap | `route_a.sh` base config |
| Release stage | 50 GiB free RAM to start; RSS cap 110 GiB; 45 GiB free disk to start; 96 h wall cap | `route_a.sh` release config |
| Probe (step 5) | About 21 GiB peak; about 10 hours | The 2 October probe of the published export ([evidence](evidence/us-export-subsample-design/route-a-p005/README.md)) |
| Certify stage | The release stage's limits (`CERTIFY_WALL_SECONDS` sets its wall cap) | `route_a.sh` certify config |
| Disk | About 50 GiB for the base (44 GB of frame checkpoints and a 2.35 GB H5 on 16 September), plus the release's checkpoints | `route_a.sh` base notes |
| Telemetry | `microcosm-emitter` 0.16.0 or newer on `PATH` (or `MICROCOSM_EMITTER_COMMAND`) | `minimum_emitter_version.json` |
| Secret | `AGENT_SECRET`: an executable that prints the Hugging Face token for `get HUGGING_FACE_TOKEN_MAX` | `with_hf_token.sh` (only with staging on) |

Keep the run alive after logout: launchd on macOS, or tmux or a systemd user
unit on Linux (README, "Running on Linux").

## 2. Inputs

| Role | Setting | File | Pin |
| --- | --- | --- | --- |
| ASEC pool (d2603, #1559) | `STORAGE` | `census_cps_2023.h5`, `census_cps_2024.h5`, `census_cps_2025.h5` | The driver's `ASEC_*_SHA` |
| ASEC person archives | `EDU` | `asecpub24csv.zip`, `asecpub25csv.zip`, `asecpub26csv.zip` | The driver's `ASEC_EDU_*_SHA` |
| PUF and ACS donors | `STORAGE` | As the published run | Their stages check them. PUF-derived: they stay on the build laptop (M2) |
| Feed | `FEED`, `FEED_SHA`, `FEED_BYTES` | Build it, see below | `facts_sha256` `1d45a89c28177a95371b4cc4068d3657c7bf12bd8edad7c82fe25bf453f9ea07` |
| Block ladder (F18) | `LADDER`, `LADDER_SHA`, `LADDER_BYTES` | `us_block_ladder_2020.npz`, staged at `policyengine/populace-us-staging` `inputs/us-block-ladder/528085de/` | `528085de8b4a9ba3ba3f25e841158c8f2807e790f00bf084ba5dc4378a7d1217`, 19,098,783 bytes |
| District crosswalk | `CROSSWALK_SHA` | Packaged in the build commit | `a347303fff3fea145f758488c43cb94355df5a8cb5e512552b2ee3791d2aa233` |
| SCF 2022 summary extract | `SCF` | `rscfp2022.dta` from the Fed's current `scfp2022s.zip` (republished on 9 October 2026 in 2025 dollars; see below) | `6ff73d5206ed9344a955ba66f8edee2d9f74297188f03e823346413cfacfc4e0`, 34,487,960 bytes |
| SSI prior basis | `SSI` | As the published run | The driver's default |
| Export-mass reference | `EXPORT_MASS_REF_H5` | The published run's `populace_us_2024.h5` | As the published run |

Get the SCF extract once. The Federal Reserve replaced `scfp2022s.zip` on
9 October 2026 with a file in 2025 dollars, and the original is no longer served
(#1368). The build reads the new file and converts its dollar columns back to
2022 dollars, so the older `rscfp2022.dta` (`6b8dd2d9…`) on some build machines
fails the driver's pin. From the build worktree, this downloads the new zip,
checks both pinned digests and prints the cached path to set as `SCF`:

```bash
.venv/bin/python -c "from microcosm.build.us_runtime.scf_wealth import fetch_scf_2022_summary_extract as f; print(f())"
```

Build the feed once, at Chronicle `41b2fb2b864a28c0eda03322b77a97ef147dfa3c`
(`us/chronicle_feed.json` gives the full recipe), with
`tools/build_us_chronicle_feed.py --skip-artifact`. Then set `FEED_SHA` to the
pin above and `FEED_BYTES` to the file's size in bytes. `--resolve` refuses
a feed whose digest is not the build commit's pin.

## 3. Settings

Copy `tools/route_a/route_a.env.example` to `tools/route_a/route_a.env`, set the
machine paths, then set these:

```bash
COMMIT=<the main commit that carries this runbook>
LOCATION_RULE=block_v1
LADDER=<path>/us_block_ladder_2020.npz
LADDER_SHA=528085de8b4a9ba3ba3f25e841158c8f2807e790f00bf084ba5dc4378a7d1217
LADDER_BYTES=19098783
FEED=<path to the feed built above>
FEED_SHA=1d45a89c28177a95371b4cc4068d3657c7bf12bd8edad7c82fe25bf453f9ea07
FEED_BYTES=<its size in bytes>
CROSSWALK_SHA=a347303fff3fea145f758488c43cb94355df5a8cb5e512552b2ee3791d2aa233
MICROCOSM_TELEMETRY_REQUIRED=1
ROUTE_A_STAGING=1            # staging upload ruled yes (M3, 18 November)
DENSE_RELEASE_D122=1
PRUNE_BASE_CHECKPOINTS=1
UV_SYNC_PYTHON=3.14
RELEASE_EXTRA_ARGS="--maximum-microsim-batch-size 2000 --target-surface national_state --dense-default-dataset --export-input-mass-reference-h5 $EXPORT_MASS_REF_H5 --scf-education-debt --target-family-loss-multiplier ssa=4"
# No QRF tail register: leave QRF_TAIL_EXCLUSIONS unset.
```

Why each one differs from the example:

- `block_v1` and the F18 ladder: ruled for this build (#1047, #1559). The release
  copies the base's `populace_location_rule` onto the national H5, so the county
  line can use it as a `block_v1` donor.
- `--scf-education-debt`: ruled on. Its four columns are engine inputs on
  2.145.3.
- `--target-family-loss-multiplier ssa=4`: ruled for the Social Security total,
  which the 19 October run missed by 6.4%. The published 24 October build used it.
- `MICROCOSM_TELEMETRY_REQUIRED=1`: with it, `--resolve` blocks on a missing or
  old emitter, before any stage. Without it, the publisher preflight refuses at
  the very end.
- No QRF tail register. Since 17 November 2026 the release decides the QRF tail
  gate at base (pre-calibration) weights and only reports the share at final
  weights. A register measured at final weights, such as the published run's
  d490 file, reads as stale and fails the run.

## 4. Steps

Run each command from the `MAIN` checkout. Send back what each step lists.

### Step 1: resolve (minutes)

```bash
bash tools/route_a/route_a.sh --resolve
```

- Success: `resolve.log` has no `BLOCKER` line, and `inputs.resolved.json` shows
  every input `ok`.
- Send back: the last 40 lines of `resolve.log`.

### Step 2: base and base preflight, then stop (about 1 to 2 hours)

```bash
STOP_BEFORE_RELEASE=1 bash tools/route_a/route_a.sh
```

- The base took 2,788 s alone and 5,058 s under contention on 16 September, with
  a 72 GB peak. The base preflight took 33 s and 10 GB.
- Success: `run-<12-character commit>/base-sup/ACCEPTED`, `run-<12-character commit>/base.sha256`, a
  preflight verdict of 0 or 2, and `run-<12-character commit>/release.STOPPED`.
- Send back: the base sha256, the `preflight-base` line in `route_a.log`, and
  `preflight-base.json` if the verdict is 2.

### Step 3: dry-run the release's gates (about 35 minutes)

```bash
cd "$WT_ROOT/microcosm-route-a-<first 12 characters of COMMIT>"
.venv/bin/python -B tools/dry_run_us_release_gates.py \
  --release-config "$RUN_ROOT/run-<first 12 characters of COMMIT>/release-config.json" \
  --json-out "$RUN_ROOT/run-<first 12 characters of COMMIT>/release-dry-run.json"
```

`route_a.log` prints this command, with the paths filled in, when it stops.

- It replays the release's base load, input stages and pre-solve gates (at most
  2,049 s on the September run), then grades every data-dependent register at
  base weights. Expect RAM close to the release's base load.
- Exit 0 is clean and 2 is AT-RISK only: go on to step 4. Exit 1 is a certain
  failure: stop and ask (S1).
- The QRF tail item is now exact at base weights. If
  `domestic_production_ald` or `qualified_bdc_income` is over 0.75 there, stop and
  ask: the planned fix skips QRF outputs no release-year engine variable reads.
- Send back: the table the tool prints and `release-dry-run.json`.

### Step 4: build the release (about 7 to 9 hours)

```bash
STOP_BEFORE_RELEASE=0 bash tools/route_a/route_a.sh
```

- The driver keeps the accepted base and preflight, builds the release (through
  calibration, the export and the release gates) and stops. The 19 October and
  3 November runs reached their terminal gates 8.3 and 6.5 hours in.
- The build takes no tier flag: it refuses `--evidence-release`, and
  certification chooses the tier (section 7).
- Success: `release-sup/ACCEPTED` is 0, `<release-out>/build_record.json`
  exists, and the run ends with `ROUTE_A_BUILT.json`. `<release-out>` is
  `build_dir` in `ROUTE_A_BUILT.json`.
- A build that fails a release gate exits 3, with `release_gates_failed: true`
  in `ROUTE_A_BUILT.json`, and keeps its H5 and record: stop and ask (S2).
- Send back: `ROUTE_A_BUILT.json`, `route_a.log`,
  `<release-out>/build_record.json`, the build's `calibration_diagnostics.json`,
  `qrf_tail_concentration.json` and `input_mass_parity.json` (in
  `<release-out>/build/`), and the `<release-out>/gate_battery_shadow`
  directory.
- Do not run step 7 yet.

### Step 5: probe the build on a household sample (about 10 hours)

From the build worktree, with `REL_OUT` set to `build_dir` in
`ROUTE_A_BUILT.json`:

```bash
cd "$WT_ROOT/microcosm-route-a-<first 12 characters of COMMIT>"
.venv/bin/python -B tools/sample_us_export_households.py \
  --export "$REL_OUT/artifacts/microcosm_us_2024.h5" \
  --fraction 0.05 --seed 0 --out "$RUN_ROOT/probe/subsample"
.venv/bin/python -B tools/probe_us_post_export.py \
  --export "$RUN_ROOT/probe/subsample/microcosm_us_2024.h5" \
  --out "$RUN_ROOT/probe/report" \
  --reference-release-dir "$REL_OUT/build" \
  --stages take_up_participation,reform_coverage_smoke,reform_validation,demographics,source_coverage
```

- The sample is about 7% of the export's households, drawn by support channel
  and source year. It keeps with certainty every household a thin smoke probe
  rests on, and each probe's dominant households. The probe runs
  certification's own post-export checks on it, through the release tool's
  functions, and certifies nothing (the release build rule,
  [Probing an export on a household subsample](us-release-build-rule.md#probing-an-export-on-a-household-subsample)).
  On the published export on 2 October, on policyengine-us 2.2.1, the sample
  took 13 minutes and 3.1 GiB, and the probe 9 hours 40 minutes (the smoke
  5.1 hours, reform validation 2.7 hours) with a 21 GiB peak.
- `--stages` leaves out the probe's QRF tail stage. It grades the export at
  its final weights, while since 17 November the build decides that gate at
  base weights (step 4's `qrf_tail_concentration.json` holds the verdict), and
  certification does not run it.
- After a build that failed its gates (exit 3), run this step only if María
  asks. The probe takes no failed build as its reference, so replace
  `--reference-release-dir "$REL_OUT/build"` with
  `--target-surface national_state --calibration-diagnostics "$REL_OUT/build/calibration_diagnostics.json"`.
- Success: `$RUN_ROOT/probe/report/probe_report.json`, with every stage
  `completed` in `summary.stage_status`.
- Send back: `probe_report.json` and `passes.jsonl` from
  `$RUN_ROOT/probe/report`, and `sample_receipt.json` from
  `$RUN_ROOT/probe/subsample`.

### Step 6: wait for María's assessment

Leave the run as it is: certification reads the build in place. María reads
what steps 4 and 5 sent back:

- the build's own verdicts: `release_gates_failed` in `ROUTE_A_BUILT.json`,
  and the release-gate and artifact-integrity failures in `build_record.json`;
- its quality: the calibration losses and target fit in
  `calibration_diagnostics.json` beside the published 24 October build's,
  input-mass parity, the QRF tail at base weights and its final-weight report,
  the gate shadow's comparison (S3), and the items in section 6;
- what certification would find: the probe's `summary.authoritative_failures`
  and `summary.probe_failures`, which should be empty. An informational
  failure, such as a smoke probe close to its floor or resting on few
  households, is a risk to weigh, not a verdict.

Her answer is one of:

- **Certify** (step 7) at the certified tier: the build passed its gates and
  the probe found nothing certification would refuse.
- **Build again** at a commit that fixes what failed. Certification at a later
  commit accepts only changes to tests, docs, changelog fragments and the
  standing evidence owners register (section 7), so a fix to code needs a new
  build (steps 1 to 4 at the new commit).
- **Certify as evidence** (`ROUTE_A_CERTIFY_TIER=evidence`), never the US
  default, when every failure has an owner.

### Step 7: certify, then the two preflights (about 48 hours), when María asks

```bash
bash tools/route_a/route_a.sh --certify
```

- At the tier she names: `ROUTE_A_CERTIFY_TIER=certified` (the default), or
  `evidence` with `ROUTE_A_EVIDENCE_FAILURE_OWNERS` for any failure the
  standing register does not own.
- On the same `RUN_ROOT`, the driver skips the accepted stages, does not hash
  the raw inputs again, and runs `tools/certify_us_national_build.py` on the
  build. Certification runs the post-export checks (the reform-coverage smoke,
  reform validation, demographics) on the written H5 and writes the manifests.
  On 19 October these checks took about 48 of the run's 56.5 hours.
- Then it runs the release-gate preflight on the release manifest and the
  publisher's `--preflight-only` check, and writes the hand-off
  (`ROUTE_A_DONE.json` or `ROUTE_A_BLOCKED.json`) with `published: false`.
- A failed certification reruns alone with `--certify`; the build is never
  rerun.
- Success: the release manifest exists, the release-gate preflight verdict is 0
  or 2, and `publisher --preflight-only` prints `{"valid": true ...}`.
- Send back: the hand-off file, `route_a.log` and `release_manifest.json`. Do
  not upload or publish anything yourself.

## 5. Stop and ask

Stop, keep every file, and report when:

- **S1** a preflight or the dry run exits 1.
- **S2** a terminal release gate fails. Watch `partnership_income` and
  `social_security_survivors` in input-mass parity.
- **S3** the gate shadow's comparison
  (`us_national.gate_battery_shadow_comparison.json` in
  `<release-out>/gate_battery_shadow`) records a verdict difference or a
  shadow error, or the file is missing. Since #1635's second round the shadow
  only warns, so the release still finishes: read the file after the release
  and report it with step 4's files.
- **S4** the supervisor stops a stage at a cap (RSS, wall, CPU or disk).
- **S5** certification's reform-coverage smoke fails on a probe definition.
- **S6** the probe reports an authoritative failure or a probe failure, or a
  stage that did not complete.

Max's failure log (#1654) is the place for new observed stops.

## 6. What only this build settles

- Social Security total against SSA with `ssa=4` on the new engine.
- `partnership_income` input-mass parity after #1532 and #1619.
- The QRF tail at base weights: collectibles after its new q0.999 cap, and
  `domestic_production_ald` and `qualified_bdc_income`.
- Target fit after the engine's concept moves: federal AGI now includes state
  tax refunds, estate income and taxable disability benefits.
- The effect of the four take-up inputs left at the engine default (Medicare
  Part D, CSFP, school meals, Alaska PFD) and of the new defaults
  (`tenant_pays_utilities`, `living_arrangements_allow_for_food_preparation`
  and `estate_income_would_be_qualified`).
- `block_v1` with the rent fix (#1444) on a fresh base: county and district fit.
- The voluntary-filing PTC probe in the reform-coverage smoke (floor $100M).
  policyengine-us 2.145.3 lets a tax unit that takes up ACA coverage claim the
  credit without another reason to file, so the probe now measures only the
  pre-take-up credit (`aca_ptc`) of voluntary filers who do not take up
  coverage. No household's received credit (`assigned_aca_ptc`) moves. Step
  5's probe gives the first reading.
- Time and memory on the new engine, which projects one more year (to 2036) in
  post-export scoring.

## 7. Build, then certify: the behavior to expect

#1672 split the release into a build and a certification, and #1681 brought
María's two rulings on it (18 November 2026):

- **A build has no tier.** Certification chooses it: certified, the default,
  into `microcosm-us-2024-national`, or evidence into
  `microcosm-us-2024-evidence`, never the US default, once every failure has
  an owner. The build refuses `--evidence-release`.
- **Certification may run at a later commit than the build's,** as a UK cut
  is certified at its sign-off's head, when the commits since change only
  tests, docs, changelog fragments and the standing evidence owners register.
  Any other change needs a new build. The driver certifies in the build's own
  worktree; certifying at a later commit is done by hand from a checkout of
  that commit ([tools/route_a/README.md](../tools/route_a/README.md)).

Route A still stops after the build by default (`ROUTE_A_CERTIFY=0`); step 7
asks for certification with `--certify`.
