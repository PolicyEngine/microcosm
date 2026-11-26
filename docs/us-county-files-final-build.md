# County Files final build runbook (microcosm#1661)

This runbook is for the operator who runs the final County Files build: the
ACS staging (R1), the 2024 county release (R2), then the projection to 2026
and 2027. It runs on the next US national build as its donor, on
policyengine-us 2.145.3 (microcosm#1676). The proof of concept (#1660,
#1647) ran with a temporary setup. None of it is used here: no patched donor,
no `--allow-default-donor-column`, no `--county-drop-empty-soi-rows`, and a
published donor manifest.

The final run starts once the national build is certified and published.
Before that, the same commands run files-only as a rehearsal (section 5).

## 1. Host and checkout

- Memory: 128 GiB. Peaks on the 17 November proof of concept: R1 48.0 GiB,
  the AGI check 40.7 GiB, R2 materialize 59.7 GiB, calibrate 47.7 GiB. Earlier
  full materialize runs peaked at 75.6 to 93.9 GB ([the rung doc](us-acs-local-county-rung.md)).
  Run nothing heavy beside it.
- Disk: keep 60 GB free. The staging H5 is about 11.5 GB and the calibrated
  file about 10.6 GB.
- Checkout: a clean worktree at the commit María names, outside which `$RUN`
  and every log live. R2 records the commit, and package refuses a dirty tree
  (an untracked file counts). Run `uv sync --python 3.14 --all-packages
  --locked --extra us` there (policyengine-us 2.145.3), then
  `uv run --no-sync` for every command.

## 2. Inputs

```bash
export RUN=/abs/path/county-files-final                    # outside the checkout
export DONOR_H5=/abs/path/microcosm_us_2024.h5              # the national build's H5
export DONOR_MANIFEST=/abs/path/release_manifest.json       # its certified release manifest
export LADDER=/abs/path/us_puma_ladder_2020.npz             # SHA-256 39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8
export FEED=/abs/path/consumer_facts_us_41b2fb2.jsonl
export FEED_SHA256=1d45a89c28177a95371b4cc4068d3657c7bf12bd8edad7c82fe25bf453f9ea07
export COUNTY_FACTS=/abs/path/consumer_facts_us_county_3e9d3a0.jsonl
export COUNTY_FACTS_SHA256=67b0998aa2ee1aeed6637dab6a846f405246706508074b1c1b5bab7bd98c8bea
export PYTHONUNBUFFERED=1
```

- The donor is the national build's dataset (`microcosm_us_2024.h5`) and its
  release manifest from `releases/microcosm-us-2024-national/<build>/`, from
  the Hub once published or from the certified build's folder. R1 checks the
  H5's SHA-256 against the manifest's one microdata artifact.
- The ladder is the one the proof of concept used. A copy is in the private
  `policyengine/populace-us-staging` dataset at
  `inputs/us-puma-ladder/39a2ab2a/`.
- The feed is the national build's feed: the same pin.
- Build the county facts at Chronicle
  `3e9d3a0694e8ca7e31ed5319c3ceb1bde92b12e0` (about 8 minutes), or copy them
  from the staging dataset's `inputs/`. Either way the loader checks the
  digest.
- R1 fetches the pinned ACS PUMS archives into `--inputs-dir` and the pinned
  SIPP 2023 file (3.73 GB) into its cache on first use.

## 3. Before you start

Two data entries, both María's rulings, must be committed before R2's
calibrate starts. Calibrate fixes the commit that package accepts, so commit
them before calibrate, never while it runs:

- the reviewed-fill register re-reviewed on 2.145.3 (microcosm#1675),
  with `reviewed_against` at the installed engine. Finalize refuses without
  it.
- the national donor's entry in `REVIEWED_DONOR_LINEAGES`
  (`tools/build_us_acs_local_release.py`), from the review below. Calibrate
  refuses the staging without it, and finalize does again. R1 and
  materialize do not need it, so they can run while the donor's evidence is
  reviewed.

### The donor's lineage review

R2's inherited limitations describe the Build O donor. For a new national
build, each one is kept or dropped from that build's own evidence, one ruling
per item. The entry lists the ones that hold; the release drops the others:

- `ssi_aged_band_collapse_inherited` (microcosm#507): the aged SSI
  baseline. Compare the build's weighted SSI recipients aged 65 and over with
  SSA's count. It holds if the aged band still falls well short of it.
- `miscellaneous_income_loss_side_donor_defect` (microcosm#393): the
  prevalence of a negative `miscellaneous_income` against SOI's loss-return
  prevalence. It holds while that distortion remains.
- `tips_return_count_carrier_deficit_inherited`: the tip-return count
  against its ledger target, in the build's calibration diagnostics. It holds
  if the build still misses it by a material margin.
- `donor_sparse_selection_training_set`: the donor manifest's microdata
  artifact. It holds only for a sparse selection, and drops for a dense
  full-row donor.
- `social_security_cps_reporters_erased_inherited` (microcosm#1178): whether
  the build's PUF-support records keep their CPS-reported Social Security,
  which microcosm#1186 fixed where a national base is built. It drops when
  they do.

## 4. Steps

### Step 1: R1, the staging (about 4 hours)

```bash
POPULACE_ACS_POOL_PEAK_LIMIT_BYTES=100000000000 \
POPULACE_STAGING_EXPORT_PEAK_LIMIT_BYTES=100000000000 \
uv run --no-sync python tools/build_us_acs_multispine_base.py \
  --base-h5 "$DONOR_H5" --donor-release-manifest "$DONOR_MANIFEST" \
  --out-h5 "$RUN/acs_multispine_staging.h5" \
  --summary "$RUN/acs_multispine_staging.summary.json" \
  --inputs-dir <ACS archive cache> --puma-ladder "$LADDER" \
  --period 2024 --seed 0 --geography-seed 0 --acs-share 0.5 \
  --n-estimators 32 --max-targets-per-fit 8 --chunksize 100000 \
  --donor-channel auto --location-rule legacy \
  --hours-under15-policy us_hours_under15_zero_completion_v1
```

- `--location-rule legacy` keeps each donor's own `block_v1` tract, district,
  county and 2023 county. The ACS rows get the PUMA ladder's counties, as in
  the proof of concept. The summary records the donor's rule
  (`base.location`) and the engine (`engine_versions`).
- R1 checks the donor in its first minutes: the work columns,
  `strike_benefits`, the partnership and S-corp split, the SPM role, the
  take-up inputs, complete WIC take-up decisions, and the ESI employer
  premium with the raw ASEC coverage fields and CPS record id on every donor
  row (microcosm#454). A refusal there means the donor lacks one. Stop and
  send the log.
- Never pass `--allow-esi-premium-gaps`, to R1 or R2: it records red ESI
  gates instead of failing, for diagnostic runs only, and a release built
  with it cannot be published as the County Files.
- R1 transfers the ESI premium onto ACS rows and draws their WIC take-up with
  the national generator before pooling (microcosm#1154); R2 draws the ACS
  tax units' EITC take-up in its take-up stage. None of them needs a flag.

### Step 2: the two checks (minutes)

```bash
uv run --no-sync python tools/check_us_acs_local_reviewed_fills.py \
  --staging-summary "$RUN/acs_multispine_staging.summary.json" \
  --out "$RUN/reviewed_fills_precheck.json"
uv run --no-sync python tools/check_us_acs_county_agi_support.py \
  --h5 "$RUN/acs_multispine_staging.h5" \
  --staging-summary "$RUN/acs_multispine_staging.summary.json" \
  --county-column county_fips_2023 --puma-ladder "$LADDER" --county-min-support 3 \
  --out "$RUN/county_agi_support.json"
```

- Both must exit 0. A fill without a register entry (exit 2) needs a ruling
  and a commit before step 3. The AGI check applies R2's minimum-support
  moves, its SOI support included. Exit 2 or 3 names the counties R2 would
  refuse. Stop and send both reports.

### Step 3: R2, the 2024 county release (about 4.5 hours)

```bash
R2="uv run --no-sync python tools/build_us_acs_local_release.py"
C="--staging-h5 $RUN/acs_multispine_staging.h5 --staging-summary $RUN/acs_multispine_staging.summary.json \
   --checkpoint-dir $RUN/checkpoints --ladder $LADDER --out-h5 $RUN/populace_us_2024_acs_local.h5 \
   --families snap,medicaid,soi --geographies state,cd,county --soi-mode state_county \
   --county-column county_fips_2023 --county-holdout-fraction 0.1 --county-holdout-salt acs-local-county-v1 \
   --county-min-support 3 --county-target-loss-cap 10 --county-max-weight-ratio 5 --hh-chunk 20000 --seed 0"
clean() { test -z "$(git status --porcelain)" || { git status --porcelain; echo "dirty tree"; return 1; }; }
clean && $R2 --stage materialize $C --feed "$FEED" --feed-sha256 "$FEED_SHA256" \
  --county-facts "$COUNTY_FACTS" --county-facts-sha256 "$COUNTY_FACTS_SHA256" --sample-fraction 1.0
clean && $R2 --stage calibrate $C
$R2 --stage qa $C
$R2 --stage finalize $C
clean && $R2 --stage package $C --out "$RUN/release"
```

- Materialize refuses a staging made on another engine in its first
  minutes. It then computes each household's AGI (minutes) and runs minimum
  support, SOI support included (`county_support.json`). The engine pass
  takes about 3 hours.
- Before the AGI, after the SSI and Medicaid take-up, materialize draws the
  ACS tax units' ACA take-up and plan ratio after one more engine pre-pass
  (time not measured yet) and writes `acs_local_aca_take_up.npz` to the
  checkpoint (microcosm#1720). Reported subsidized coverage (HIMRKS)
  anchors take-up, thinned by a keyed draw where a state's anchors exceed
  its remaining SOI return count. Its grade in `run_identity.json` is
  report-only: it never stops the run.
- If the register or the donor's lineage entry is not in the checkout yet,
  commit it after materialize and before calibrate (section 3). The `clean`
  check before calibrate then holds, and calibrate refuses a donor without
  its entry before it solves.
- One stage per process, so a stop resumes at its stage.

### Step 4: the publisher's preflight (seconds; publishes nothing)

```bash
uv run --no-sync microcosm-publish-release-build "$RUN/release/releases/microcosm-us-2024-local" \
  --repo-id policyengine/populace-us --artifact-root "$RUN/release" --preflight-only
```

- It must print a valid result. This runs the local-area release contract,
  which R2 does not run itself. Do not publish: publication is María's call
  with Max.

### Step 5: the projection to 2026 and 2027

Follow [the projection guide](us-local-annual-projection.md#running-it-on-the-2024-county-build)
with `$PKG=$RUN/release`, on policyengine-us 2.145.3 (the build's
`built_with_model_package`). Two of its steps are not automated (#1627):
step 3, each projected year's household measures, and step 4, the acceptance
report. The report needs six passing checks per year (schema, source
identity, year, demographics, input aggregates and runtime), and their
criteria are not set yet, a tolerance for county and state share drift among
them. Until they are, the 2024 county release can be published without the
projected years.

## 5. The rehearsal, before certification ends (files only)

Run steps 1 to 4 on the built but not yet certified national H5, without
`--donor-release-manifest`. Use production settings: R1 is deterministic,
so its staging and both checks' results are the final run's. Expected
differences from the final run:

- The staging records `donor_release: null`. So neither calibrate nor
  finalize asks for the donor's `REVIEWED_DONOR_LINEAGES` entry, and
  finalize writes the Build O limitations.
- The preflight refuses `donor_release` null. Any other refusal is a finding.
- Until the re-reviewed register is in the checkout, finalize refuses the
  engine mismatch. It writes `gate_summary.json` first, so read the other
  gates there. Package and the preflight then need a run whose calibrate ran
  at a commit with the register, since package refuses another commit.

The PUF-derived donor stays on the build laptop until it is published, so
the rehearsal runs there.

## 6. What to send, and when to stop

Send the logs and JSON records after step 2, and after step 4 or any stop,
never the H5s:

```bash
( cd "$RUN" && out="$HOME/county-files-evidence-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" && find . -type f \( -name '*.json' -o -name '*.log' -o -name '*.csv' \) -not -name '*.h5' -print0 | tar -czf "$out" --null -T - && echo "$out" )
```

Stop, keep every file, and send it when:

- R1 refuses the donor;
- a check exits other than 0;
- a stage refuses;
- the preflight is invalid (other than the rehearsal's `donor_release`
  refusal).
