# US release: main builds the certified default from raw sources

## The rule

Main must demonstrably build the certified US default from raw sources and pass
its preflight and certification gates (decided 15 September 2026).

The last from-scratch build is Build P (28 July 2026, policyengine-us 1.764.6).
Every later default is a supplied-parent enrichment of that file. A pull request
large enough to change how the dataset is built carries a from-scratch build and
certification receipt from its own tree before it merges. Main carries the same
build as a standing check afterwards.

A dataset earns default status separately, by beating the incumbent. The rule
main enforces is microcosm#578's: the candidate beats the incumbent on the
frozen comparison register, both rescored on it, with strict improvement
(`_exact_k_frozen_register_fit_gate`,
`tools/build_us_fiscal_refresh_release.py:6897`). That gate runs on the
`--exact-k` arm only (`:11149`), and that arm takes a pool manifest, which is
mutually exclusive with `--base-h5`. A `--base-h5` release runs no improvement
gate. Main has no US held-out test: the rotated-fold holdout in
`microcosm.build.holdout` is imported by the UK runtime and UK tools only. So a
US release says it beats the incumbent on the frozen register, and says it beats
it on held-out cells only once a US holdout exists and has run.

Publication stays a human step: `tools/publish_release.sh` wraps
`microcosm-publish-release`, which uploads `latest.json` last
(`microcosm.data.release.publish_release`), and only that pointer makes a
release the certified default.

## What stands between main and that build

A from-scratch attempt ran on main at `51c314382` on 15 and 16 September 2026
and reached no release. The causes below were re-read on main at `d1196af10`.
The engine lock that also blocked it is fixed: main resolves policyengine-us
2.2.1, policyengine-core 3.32.5 and spm-calculator 1.0.0.

### 1. No Chronicle feed on the build machine carries dimension labels

`_validate_chronicle_hierarchy_labels`
(`packages/microcosm-build/src/microcosm/build/ledger_targets.py:1025`) requires
exactly one Chronicle-owned label for every dimension a target selects on, and
refuses to substitute the identifier. The pinned US feed
(`consumer_facts_buildn_v9_4.jsonl`, `b3c08356…`, named in
`us/target_parity_manifest.json` and `us/target_parity_feed_families.json`)
predates those labels, so target compilation refuses. The refusal was measured
on the `--base-h5` arm, 49.59 seconds in. The pool arm never reached target
compilation in that attempt, and it cannot take the bare pinned feed at all,
because `--exact-k` needs an artifact directory with a manifest; the compile
call is shared, so the same validator stands in front of it. Every other stage
waits behind this one.

Chronicle main writes the labels (`dimension_labels`, `dimension_value_labels`,
`layout.groupby_dimension_label`), and the UK feed on main is already pinned to
such an export (`uk/chronicle_feed.json`). The re-pin was approved on 18
September 2026. The US needs a fresh export, the two
parity resources regenerated together with
`tools/build_us_target_parity_manifest.py`, and a `us/chronicle_feed.json` that
records the export the way the UK file does. `chronicle build-bundle` takes one
`--year`, and the pinned US feed carries many periods: tax years 2020 to 2026,
fiscal years 2023, 2024 and 2026 to 2029, calendar years 2018, 2023 and 2024,
and the months 2024-12 and 2025-12 (counted from the feed's own rows). One
`--year 2023` export at Chronicle `c5e5bf8` reproduces 555 of the feed's 586
record sets, every row labelled; the rest, including the JCT tax expenditures
and the CBO revenue projections that age dollar targets, sit in other years'
bundles. So the export's scope is settled, and its coverage compared with the
pinned feed record set by record set, before anything is re-pinned.

**22 September 2026:** the labelled pin landed in PR #955:
`us/chronicle_feed.json` (Chronicle `c5e5bf8`, a bare feed) with the two
parity resources regenerated, and `tools/build_us_fiscal_refresh_release.py`
now holds the loaded feed to that pin (`--allow-unpinned-feed` waives it for a
reviewed diagnostic run only). See `docs/us-chronicle-feed-repin.md`. The
`--base-h5` arm can take the pin; the `--exact-k` arm needs a consumer artifact
with a manifest, which Chronicle refuses at `c5e5bf8` (chronicle#277), so it
waits on that fix.

### 2. Nothing in the build emits the SPM independence role

In policyengine-us 2.2.1 one SPM unit with no classified adult (age 18 or over,
or age 15 or over with `is_spm_independent_minor_role`) refuses the SPM
measurement for every unit in the simulation that holds it. The first
simulation of the written release file is the reform-coverage smoke gate
(`us_reform_coverage_smoke_gate` in `tools/build_us_fiscal_refresh_release.py`),
which runs after the export write and before the calibration NPZ; its LIHEAP
probe reaches SPM composition through `spm_unit_benefits` (the probe list in
`tools/build_us_release_input_coverage_manifest.py`), so a population with such
a unit refuses there. The reform-validation diagnostics that follow the NPZ
(`_write_reform_validation`; 104 `in_poverty` rows from
`us/state_spm_poverty_levels.json`) would refuse the same way; they record, they
do not gate. These stages now score the written file in household batches, not
one whole-dataset simulation each (see
[Post-export scoring](#post-export-scoring)); the batch that holds such a unit
refuses, and the refusal still aborts the stage.

The certified default passes because `tools/build_us_spm_role_enrichment.py`
added the role afterwards, and that tool accepts only Build P's exact bytes
(`:117-118`). The pins record what the role does there: 28 SPM units have no
member aged 18 or over, none lacks a classified adult once the role is present
(`packages/microcosm-data/src/microcosm/data/source_enrichment.py`,
`EXPECTED_COUNTS`). A dataset built from raw sources today carries no role.

Decided 17 September 2026:

- A preflight check counts SPM units with no classified adult before the engine
  runs, and names the remedy.
- The role is delivered for a base that is not Build P.
- SPM units are never re-grouped to make the count zero. That changes the
  poverty measurement the 104 rows exist to check.
- No adult is invented where the source delivers none.

Decided 18 September 2026: the role reaches a dataset built from raw sources as
a build-stage input leaf that carries the source role. Build P's enrichment
lane stays pinned to Build P.

### 3. The July selection does not map onto a rebuilt base

`check_selection_carryover` (`us_runtime/release_gate_preflight.py:313`) failed
on the rebuilt base: 15,228 capital-gains own-tail donors sit in households the
frozen July identity list does not name (the list names households, and the
selection mask is household-level, so those donors' tax units are dropped)
(`assert_puf_capital_gains_tail_survives_selection`,
`us_runtime/puf_capital_gains_tail.py:465`). The release tool does not require a
selection source (`--selection-source-manifest` is optional, and the reduction
at `tools/build_us_fiscal_refresh_release.py:9019` runs only when one is given).
Without one, calibration runs on the full base of about 353,000 households. The
only measured comparator is July's run on about 57,000 households: 2 hours 42
minutes at 85 GB. Nothing has been measured at full-base size.

A dataset built this way is a new lineage, not a replay of Build P: no frozen
support, a different capital-gains tail stratum, a different congressional
district assignment and a different feed identity.

Decided 18 September 2026: the next certified default may be a new lineage. The
`--exact-k` arm refuses a frozen selection source outright, so the July
selection is out of scope there, not merely inconvenient.

**23 September 2026:** the preflight gained an explicit new-lineage mode, so a
`--base-h5` release built without a selection source can be preflighted as this
rule requires. Until then `tools/preflight_us_release_gates.py` required
`--selection-source-manifest` and always ran `check_selection_carryover`.
`--new-lineage` is mutually exclusive with the manifest, which stays required
without it. It records `selection_carryover` as `SKIPPED` with reason
`new_lineage`. It keeps the one refusal in that check that belongs to the base,
not to a selection, as `capital_gains_tail_presence`: the base must carry the
materialized PUF capital-gains own-tail, the same call the release tool makes
on every arm after loading the base. It runs the other checks unchanged on the
whole base. With `--release-manifest`, it also requires that release to record
`build.selection_source` as `{"enabled": false}`.

**23 September 2026:** without a selection source the release runs its stages
on every support copy the base carries, including the capital-gains own-tail
copy (clone index 2) of each tail household. That copy keeps its source IDs and
the `puf_tax_detail` channel, so the base from `build_us_puf_support_base.py`
holds two PUF-role rows for each tail source person (42,336 on base
`cb1bd1e6…`). SIPP Head Start and voluntary filing identified a copy by source
ID and role on bases without a raw spine ID, so they refused that base. Release
stages now identify a copy by source ID and clone index
(`support_copy_rank_series`), whether or not the base carries a raw spine ID.
A repeated pair is still refused. Source-level decisions fan out to the tail
copy from the lowest surviving clone index. Row-level imputations treat the
tail copy as a PUF-role row, as the base's own post-transfer stages do.
Only a base without a raw spine ID may lack clone indices. An assembled table
(one with a raw spine ID) must carry its support channel, its clone index and
a complete, non-null `<entity>_source_id`. `require_assembled_support_provenance`
refuses one that lacks any of them.
The metadata-presence check includes the raw spine ID, so losing both columns
cannot make an assembled table look historical. `support_role_series` calls
the validator before resolving roles; this covers the prior-year and SSI
summaries before occurrence pairing while preserving their attested source
bytes and the bundle digest. Head Start, voluntary filing and copy ranking
also validate explicitly before their fallbacks. Missing provenance refuses
imputation and fails or raises from the signal gates. With clone indices the
gates compare every copy of a source unit, including repeated
`(source ID, clone index)` copies; imputations refuse repeated pairs.

Temporary source-kernel projections validate assembled provenance before
removing the channel, clone index and raw spine ID together. They retain the
stable source IDs, and their outputs are compared with or merged into the
original assembled table. This lets historical source kernels consume the
projection without treating intentionally removed provenance as corruption.

Until 23 September 2026, `--dense-default-dataset` was diagnostic only and a
release build left it unset, so the default was the sparse dataset that runs on
standard machines. Decided 23 September 2026 (microcosm#956): route A's first
certified release, which is not the default, calibrates the whole base dense
(`--dense-default-dataset`) on the national and state target surface
(`--target-surface national_state`), and sparse L0 tuning is pursued after it.
The standing principle for the graph era is the same target surface for every
dataset size, with L0 the only difference across sizes.

### 4. Three raw inputs exist in one untracked directory

The base stage reads six processed inputs. The licensed IRS public use file pair
already lives in a token-required Hugging Face repository, and the 2022 ACS rent
donor is already on the Hub. The three processed CPS ASEC files
(`census_cps_2022.h5`, `census_cps_2023.h5`, `census_cps_2024.h5`) exist only in
an untracked directory on the build machine. They derive from public Census
files. The base stage records their digests and compares them to nothing.

Decided 18 September 2026: the three files are mirrored, unchanged, to a public
PolicyEngine dataset repository on Hugging Face; microcosm pins revision, digest
and size and fetches them as it fetches the SIPP donor; the base stage verifies
the digest it is given. `us/spec/sources.yaml` is generated from a sealed
six-role receipt and does not change until a real build re-cuts it.

## Post-export scoring

The reform-coverage smoke, reform validation and demographics score the written
release H5 through one household-batched scorer
(`_HouseholdBatchedPostExportScorer` in
`tools/build_us_fiscal_refresh_release.py`, microcosm#956). Each used to build
one Microsimulation over the whole file.

- The scorer loads the written H5 once, binds its SHA-256, and refuses to let
  the release manifest pin different bytes.
- It refuses a file whose group units do not each nest in one household, or
  whose batches do not partition every entity, before any engine exists.
- Batches hold at most `--maximum-microsim-batch-size` households (default
  5,000; route A runs 2,000). Every engine declares the release SPM selection
  (`US_RELEASE_SPM_SELECTION`, county geography) and is released before the
  next batch builds.
- Each stage's baseline requests are learned before export by an engine-free
  dry run and scored in one pass over the batches, in ascending period order.
  One engine refuses a period earlier than one it has already computed.
  The real-engine test probes the MD CCS request-order premise by requesting
  2025 state taxes after 2027 keys. It checks for the
  `md.msde.ccs.payment.informal.rates` error and warns if that error is absent;
  it does not require the upstream bug to persist.
- Each reform builds one tax-benefit system and scores its measure batch by
  batch. Each batch engine gets that system alone, without `reform=`: given
  both, policyengine-us creates a baseline branch, which clones the system.
- Every statistic reads full-length values with their engine weights.

The totals equal a whole-file simulation's up to floating-point summation order
only if every scored measure is additive across households and never reads the
`baseline` branch that a batch reform engine lacks. The scorer rejects these
known violations at run time:

- In a multi-batch pass, an engine refuses once it has computed a formula that
  aggregates over its whole simulation. In policyengine-us 2.2.1 these are the
  Medicaid SLCSP state sums behind `medicaid_cost` and the weighted household
  and SPM-unit income deciles (`US_POPULATION_AGGREGATE_VARIABLES`, the same
  list batched target materialization guards). Each batch would compute them
  over itself alone.
- A reform engine refuses once it has computed a formula that reads the
  baseline branch (`POST_EXPORT_BASELINE_BRANCH_READERS`). These are the
  Medicaid denominator a reform holds at baseline, and the behavioral-response
  measurements.
- A reform pass refuses a reform that sets the labor-supply or capital-gains
  response parameters off their baseline values. Without a baseline branch,
  those responses score 0. A whole-file engine also scores 0 at the engine's
  zero default elasticities.

Tests pin the watched lists against the installed engine's formula sources,
exercise each guard on a real engine, and compare the shipped baseline plans
(18 smoke keys, 75 validation keys and 1 demographics key) against whole-file
values on a small written H5 in three batches.

On 2026-09-24, the sweep below completed on a written fixture containing eight
households across five states, with Medicaid enrollees, under policyengine-us
2.2.1. All 107 reform passes cleared the guards in three batches: 41 smoke
passes and 66 validation passes, covering 52 out-of-sample specs, 12 in-sample
fallbacks and two OBBBA pre-baselines. The 18 OBBBA stacked provision states
were included. All three baseline plans and four selected reform passes
matched whole-file values at `rtol=1e-12, atol=1e-6`; their weights matched
exactly. This is evidence for the fixture's calculation paths.

The complete reform sweep is a separate pre-rerun step:

```bash
uv run --no-sync python tools/sweep_us_post_export_scoring.py \
  --dataset-path /path/to/small-written-export.h5 --batch-size 3 \
  --output /path/to/guard-sweep.json
```

The input must contain enough households for at least two batches. The tool
records the shipped consumers' requests, including every out-of-sample
validation reform and the OBBBA pre-baseline and stacked states. It also
exercises the fallback simulations used when an in-sample estimate is absent.
It scores at most four reform passes per worker, serially, and compares all
baseline values and four reform passes with whole-file engines. Workers stop
at a sampled 7 GiB RSS threshold and report their peak RSS, elapsed time,
counts and the scored H5 digest.
Run this fixture sweep before R4's full-export timing steps, and repeat it
after a probe, spec or engine-lock change before starting a release.
R4's existing full-export rehearsal times the smoke and validation
baseline plans plus three selected reform passes; it does not cover every
reform and does not replace this sweep.

`calibration_diagnostics.json` records the plan under `build.post_export_scoring`
(method, batch size, batch count, each stage's baseline keys and periods), and
`reform_coverage_smoke.json` records what the smoke scored under
`post_export_scoring`. Poverty rows stay record-only. Before the export write
the build frees the target frame and the calibration result's target tables:
`Frame.with_weights` copies every table, target columns included. Dense and
L0 results drop those frames; exact-k substitutes the clean export frame for
its receipt's later household-count check. The fixture sweep does not measure
memory or wall time of the batched stages at full size.

## Measured stage times

From the September attempt, on one 128 GiB machine, under policyengine-us
1.819.0:

| Stage | Wall time | Peak memory |
|---|---|---|
| Base from raw sources (`tools/build_us_puf_support_base.py --stage all`), machine mostly free | 46 min 28 s | 72.47 GB |
| The same under contention | 1 h 24 min 18 s | 65.10 GB |
| Preflight on the fresh base | 32.58 s | 9.76 GB |
| Target compilation, to its refusal | 49.59 s | 3.40 GB |
