# microcosm

The population stack: one kernel datatype — the **`Frame`**, a weighted
sampling frame of entity tables — and packages as operators on it. One PEP 420
`microcosm` namespace, shipped as shard distributions; a `microcosm` metapackage
will pin the constellation.

| package | import | role | succeeds |
|---|---|---|---|
| `microcosm-frame` | `microcosm.frame` | the kernel: Frame, typed weights, strata, links, weighted accounting, unit structure, rules-engine protocol | microdf, microunit |
| `microcosm-fit` | `microcosm.fit` | conditional models (weight-aware by construction) | ad hoc imputation scripts |
| `microcosm-calibrate` | `microcosm.calibrate` | representation: targets → calibrated weights (APG / L0) | microcalibrate |
| `microcosm-build` | `microcosm.build` | population build plans, donor graphs, release gates, and country build stages | one-off build drivers |
| `microcosm-data` | `microcosm.data` | published population registry and lazy engine loaders | country-specific data packages |

Firm support is **experimental**. The frame kernel can declare firm entity
tables and validate person-firm `jobs` link tables, but link-aware operators,
firm calibration targets, and firm release pipelines are not production
surfaces yet.

See [DESIGN.md](DESIGN.md) for the charter: why the rebuild, the kernel
semantics, the RulesEngine protocol (policyengine-us today, Axiom rulespec-us
next), longitudinal design (one weight per trajectory), and the process rules
(behavioral contract tests, constellation versioning, environment-carrying
artifacts).

Incumbent comparisons and historical replacement benchmarks live outside this
repo. The live Microcosm repo owns the library, build contracts, published
population registry, and acceptance gates.

## Development

```bash
uv sync --all-packages   # workspace install (all members + dev groups)
uv run pytest            # all packages, incl. behavioral contract tests
uv run ruff check .
```

Packaged country specs load once per process: `load_country_spec("uk")`
returns the same object on every call. In a notebook or other long-lived
session, edits to a country package under
`packages/microcosm-build/src/microcosm/build/<country>/` are not picked up
until you restart the kernel, load it by path with
`load_country_spec(Path(...))` (always re-read), or call
`microcosm.build.country_spec._load_packaged_country_spec.cache_clear()`.

## Build progress and staging run files

Supported US and UK build commands always start the local telemetry emitter
service. It reports live progress to the hosted collector when the operator's
existing Hugging Face login is accepted; otherwise it retains the events
locally and the build continues. Every build on a machine shares one local
queue. A build's login decides only that build's own run. A run whose service
has exited is delivered later by another build's service whose login the
collector accepts for it. This live event path is independent of the staging
run files described below.

US fiscal refresh builds also write pre-release staging run files **by
default**. Progress JSON is uploaded to `policyengine/populace-us-staging`
while the build runs (best-effort — a missing token or failed upload never
fails the build), so every candidate shows up on the staging dashboard before
it is published. Disable these files with `--no-staging`, or point them
elsewhere with `--staging-repo-id` / `POPULACE_STAGING_REPO_ID`. An *empty*
`POPULACE_STAGING_REPO_ID` is ignored rather than read as off, and staging with
no destination at all is an argparse error. `--no-staging` does not disable
the local telemetry emitter service.

The build manifest records what staging did: the run id, the destination, and
how many files actually reached it, or an explicit `enabled: false` for a
declared opt-out. `microcosm-publish-release` **refuses** a release whose build
meant to stage and delivered nothing (`--allow-missing-staging` overrides); a
declared `--no-staging` build publishes without the flag:

```bash
python tools/build_us_fiscal_refresh_release.py \
  --ledger-facts consumer_facts.jsonl \
  --out /tmp/microcosm-build
```

This writes `progress.json`, `events.ndjson`, `calibration_progress.json`, and
final candidate diagnostics under `runs/<run_id>/` without updating production
`latest.json`.

The UK commands (`tools/build_uk_frs_spine.py`, a shim over the package's
`uk_runtime.spine_build`, and `microcosm-build-uk` / `tools/build_uk_full.py`,
whose `--release-role` builds either the national or the dense line;
`tools/build_uk_rowwise_candidate.py` is a stub over the same driver) stage
version 2 staging run files to `policyengine/populace-uk-staging` under the
same switch. The build command also **stages the finished dataset bundle** it built,
national, dense or exact-count, under `staged/<run_id>/` in the
private `policyengine/populace-uk-private` repository so the team can inspect
it without publishing it: `releases/` and `latest.json` are untouched, the
release contract is not consulted, and a `releasable: false` size run stages
like any other. Fetch a bundle with `tools/fetch_uk_staged_dataset.py`;
re-stage a finished run directory with `tools/stage_uk_rowwise_candidate.py`.
See [docs/uk-staging-operations.md](docs/uk-staging-operations.md).

See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) for the measured memory,
disk, and CPU footprint of developing and building locally (and what to budget
on a build machine — RAM is the binding constraint).

## Release-gate preflight

Several release gates fail on facts that are already determined by the base
pool, the frozen selection, and the target/coverage registry — no calibration
solve needed to see them. `tools/preflight_us_release_gates.py` recovers those
signals in **minutes** so a two-hour release launch is not the first place a
knowable defect surfaces:

```bash
uv run python tools/preflight_us_release_gates.py \
  --base-h5 out/base-m/base_populace_us_2024_puf_support.h5 \
  --selection-source-manifest inputs/buildm_keogh_swap_selection_source.json \
  --export-input-mass-reference-h5 forensics/populace_us_2024.h5
```

It is read-only against the H5 artifacts and reports, per check, `PASS` /
`FAIL` / `AT-RISK` with the measured numbers (exit `1` on any FAIL, `2` on
AT-RISK only, `0` clean):

1. **Selection carryover** — the frozen selection-source manifest maps cleanly
   onto the base pool (the frozen-support recovery contract, run pre-solve).
2. **Zero-support preview** — compiled positive fiscal targets whose
   materialized support is ~0 under the selection at base weights stay a
   structural zero after the solve. Direct-column targets are checked;
   engine-derived measures are marked *not statically checkable* (pass
   `--ledger-facts` to compile the target surface).
3. **Export-mass parity risk** — each export-mass column's pool mass at *base*
   weights against its reference band, honoring the release tool's
   `US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS` register (reused, never
   re-declared). A column out of band pre-solve is flagged for review.
4. **Smoke-probe support audit** — every reform-coverage probe leaf's pool vs
   *selected* nonzero support and pool sign-leg decomposition. A leaf with pool
   support but zero selected support **fails** (the input the frozen selection
   cannot express); a thin selection or a signed leaf whose net sign
   contradicts the probe's `expected_sign` is AT-RISK.
5. **SPM measurement composition** — every SPM unit has a classified adult. A
   single unit without one makes spm-calculator refuse the whole population's
   SPM measurement, which the release otherwise hits hours in.

**A new lineage** — a release built on a fresh base with no selection source
([docs/us-release-build-rule.md](docs/us-release-build-rule.md) §3) — has no
frozen selection to carry over. Say so explicitly with `--new-lineage` in place
of `--selection-source-manifest` (the two are refused together; with neither,
the manifest is required as before):

```bash
uv run python tools/preflight_us_release_gates.py \
  --base-h5 out/base/base_populace_us_2024_puf_support.h5 \
  --new-lineage \
  --ledger-facts inputs/consumer_facts.jsonl
```

The report records `selection_carryover` as `SKIPPED` with reason
`new_lineage`. The one refusal inside that check that belongs to the base
rather than to a selection — the base must carry the materialized PUF
capital-gains own-tail, which the release tool also requires on every arm —
still runs, as `capital_gains_tail_presence`. Every other check runs unchanged
on the whole base, which is what a release without a selection calibrates. Given
`--release-manifest`, `--new-lineage` also requires that release to record no
selection source.

**Run it** at base-build exit, before any release launch, and after any change
to the selection-source manifest or the target/coverage registry. The
synthetic-fixture unit tests
(`packages/microcosm-build/tests/engine_free/us/test_us_release_gate_preflight.py`) run in the
normal `uv run pytest` suite; the real-H5 mode above is a local/runbook step.

### Dry-running the release's registers

The checks above read the raw base. The release's waiver registers are graded on
something else: the *staged* frame, after the input stages the release runs
itself. Five QRF stages run inside the release (`scf_wealth`, `org_wages`,
`ssi_disability_criteria`, `sipp_head_start`, `voluntary_filing_input`), and
the release grades those registers only at the end. On 2026-09-26, route A run
`310842b986d7` ran 13,707 s and then failed on its QRF tail-concentration
waiver register alone. After microcosm#1033 changed the QRF draws:

- four columns crossed the 0.75 top-100 share unwaived (`bond_assets`, which
  exists only after the release's SCF stage, `domestic_production_ald`,
  `estate_income` and `w2_wages_from_qualified_business`);
- two entries went stale (`alimony_expense`, `qualified_bdc_income`);
- one went thin (`farm_income`).

That run's own `build.timing` splits it: at most 2,049 s for the base load,
input stages and pre-solve gates, then 9,951 s of target compilation and
1,668 s of calibration. The dry run replaces everything after the first part.

The dry run replays the release itself and stops before the expensive part:

```bash
# The release config a supervisor saved ("argv" holds the full command):
uv run python tools/dry_run_us_release_gates.py \
  --release-config run/release-config.json \
  --json-out run/dry-run-gates.json

# A new base, or a candidate register, against that same config:
uv run python tools/dry_run_us_release_gates.py \
  --release-config run/release-config.json \
  --base-h5 base-out/base_populace_us_2024_puf_support.h5 \
  --qrf-tail-concentration-exclusions candidate_register.json \
  --json-out dry-run-gates.json

# Or the release command itself, with one flag added:
uv run python tools/build_us_fiscal_refresh_release.py <release args> \
  --dry-run-gates-report dry-run-gates.json
```

`--dry-run-gates-report` runs the release's own code path: the same argv, base
load, input stages and pre-solve gates. It stops where the staged frame would
go to target materialization. There it grades the staged frame at its base
weights, calling the release tool's own gate functions (none is
re-implemented), and exits `1` on any certain failure, `2` on AT-RISK only,
and `0` when clean. An argparse error also exits `2` but writes no report, so
the wrapper returns `64` whenever no report was written. It writes only its
report: nothing under `--out`, no staging run files, no receipts. The local
telemetry emitter service still reports dry-run progress. A base or
donor that the config does not name locally is still downloaded, into the same
caches the release uses. A refusal before the stop point becomes the report's
certain failure. A crash while grading is reported as the dry run's own error,
never as a release refusal. The checks:

- **`qrf_tail_register`** covers every QRF output and every register entry. It
  reports each one's release class at base weights (`over`, `at_or_under`,
  `thin`, `dense`, `absent`, `non_numeric` or `not_qrf_output`), the verdict
  that class gives (`used`, `stale`, `unused`, `unwaived`, `waived` or, under
  `--evidence-release`, `owned`), and every verdict the solve could still
  reach. A register file that does not load is a certain failure here: the
  release reads it only at its terminal gates. Under `--evidence-release` it
  is AT-RISK when an owner matches the release's degraded-mode line, because
  with other terminal failures on record the release ships it.
- **`export_input_mass`** runs the export input-mass gate with the staged
  frame standing in for the export. It classifies the
  `US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS` register as the gate does: `used`,
  `below_reference_floor` or `unused`.
- **`degenerate_input_register`**, **`ecps_parity_register`**,
  **`input_coverage_register`**, **`stored_inputs`**, **`spm_composition`**,
  **`zero_support_preview`** and **`pre_solve_battery`** grade the remaining
  base-computable pre-export gates. `pre_solve_battery` holds the batched early
  failures and every signal gate's lines.
- **`export_signal_regrades`** covers the health-input and
  reported-coverage-vintage gates, which the release grades before the solve
  and again on the export. Neither reads weights, so on the full-pool path the
  re-grade repeats the staged verdict. On the L0 path it can flip, and the
  check flags each signal a selection could empty.
- Under `--evidence-release`, a certain failure whose every release line
  matches an owner pattern (the release's own matching rule) becomes AT-RISK,
  marked OWNED, because the evidence tier ships it as a known failure. Any
  unowned line still refuses. The exception is the input-mass-reference,
  degenerate-input and eCPS gates when no earlier terminal failure is on
  record: the release raises on them before the solve, outside the evidence
  batch, so they stay certain failures.
- **`not_previewable`** lists every gate that depends on the solve, on target
  materialization or on the written H5, so the report never implies coverage
  it lacks.

The report also records what the run is bound to:

- the base sha256;
- the build commit;
- the staged frame's sha256 and the target-frame materializer identity, which
  the release's own checkpoint will carry;
- whether an existing target-frame checkpoint matches it, in which case the
  release will skip materialization;
- each register's path, sha256 and entry count.

**What is certain.** The solve can change two things: record weights, and on
the L0 path which records ship. The full-pool solve parametrizes each weight as
`exp(log w)`, so on that path carrier counts, nonzero shares and every value
are exact at base weights. A column's top-k share is not exact, and nor is any
weighted mass. For each graded item the dry run lists the release classes that
the stated margins allow. If they all give one verdict, the item is certain (a
FAIL when that verdict fails). If they give several and some fail, the item is
AT-RISK. With every margin at zero, the verdict equals the release gate's
verdict at base weights, which the tests check against the release tool's own
functions.

On the L0 path the solve also picks which households ship and refits their
weights. Nobody has measured how that moves a top-k share, so by default no
share verdict there is certain. A value-only verdict the release re-grades on
the selected export (input coverage, the two signal re-grades, zero support)
stays certain only while every signal it rests on keeps at least one record
under the carrier-retention margin. The default margins:

| Margin | Default | Basis |
| --- | --- | --- |
| `--dry-run-tail-share-rise-margin` | 0.35 | largest measured rise +0.299 (run 310842b986d7, 32 columns); d177 register pairs up to +0.29 |
| `--dry-run-tail-share-fall-margin` | 0.05 | largest measured fall −0.033 |
| `--dry-run-mass-drift-margin` | 0.10 | flags in-band columns near the ±50% edge; measured drift moves reach +0.84, so no nonzero-mass verdict is certain |
| `--dry-run-l0-tail-share-rise-margin`, `--dry-run-l0-tail-share-fall-margin` | 1.0 | L0 path only; unmeasured, so the default admits any share |
| `--dry-run-support-nonzero-share-margin` | 0.02 | L0 path only; a conservative default, not a measurement |
| `--dry-run-support-carrier-retention` | 0.25 | L0 path only; the smallest kept fraction of the records carrying one signal; a conservative default, not a measurement |

The measurements behind them are in
[experiments/us-release-dry-run-margin-evidence.md](experiments/us-release-dry-run-margin-evidence.md).

**Checked on the failed run.** The dry run was replayed on run 310842b986d7's
own release config with its d177 register. The replay ran at commit
`21c1f9ba3`, the first revision of this tool. Later revisions added the
`export_signal_regrades` check, the L0 bounds, evidence-tier ownership and
the unloadable-register handling, none of which changes a full-pool,
non-evidence verdict on a register that loads:

- It exits `1` on `farm_income`, a thin column (469 carriers) whose unused
  entry is certain.
- Every other column that release refused is AT-RISK, and none is PASS. That
  includes `bond_assets`, at 0.520 at base weights against the release's 0.766.
- At the stop point, the staged frame with the release's saved final weights
  attached reproduces the release's recorded tail surface exactly: every
  share, carrier count, refusal and register-mismatch entry.
- Its staged-frame digest matches the identity of the release's own
  target-frame checkpoint.

On a saturated host the replay reached its stop point in at most 17,455 s.
Its report's 17,691 s was taken after grading (236 s) and the replay's own
differential. The whole process got 0.79 CPU-seconds per second, was switched
out 115 million times, and used 14,046 CPU-seconds, grading and differential
included. The original run had used about 11,000 CPU-seconds by the same
point, which it reached in at most 2,049 s.

Exit `0` does not certify export input mass: calibration moves a column's drift
further than the band, so only a structural refusal there is certain (the
report's `not_previewable` check says so).

**Run it** after the base build exits and before launching the release, with
the release config you will launch. Run it again after any change to a waiver
register. The certainty model lives in
`microcosm.build.us_runtime.release_gate_dry_run`; its tests
(`packages/microcosm-build/tests/engine_free/us/test_us_release_gate_dry_run.py`,
including Hypothesis properties) run in the normal `uv run pytest` suite.

## Releasing & alerts

The [native SPM role source-enrichment lane](docs/us-native-spm-role-source-enrichment.md)
creates a new US H5 from the exact reviewed BuildP parent, preserves its original
variables and schema-5 calibration evidence, and requires fresh country/wrapper
compatibility checks. It has a local candidate builder and uses the regular
publisher's contract with `--parent-h5` and `--preflight-only`. The same release
type publishes the [reported-receipt child of the national default](docs/us-reported-receipt-source-enrichment.md)
as a tag-only donor for the ACS local chain. It is never the `latest.json`
default.

The non-default ACS local-area chain (`tools/build_us_acs_local_release.py`)
calibrates to the SOI `state` surface by default, the 4,459-target contract of
Build O and Build P; `--soi-mode totals` and `--soi-mode full` are explicit
opt-ins. See
[the ACS local-area SOI target surface](docs/us-acs-local-soi-target-surface.md)
for what each mode contains and where the build records it. Its
`--l2-basis chi_square` and `--mass-parametrization softmax` options (defaults:
the historical `record` and `projection`) penalize distance from the design
weights; see [penalized calibration toward the design weights](docs/calibration-l2-basis.md)
for the algebra, the evidence and the measured frontier.

National and ACS local-area builds now use the same typed schema-8 calibration
diagnostics writer. The local builder adds its Census population marginals to a
versioned `TargetRegistry`, including provider, category, geography, and target
hierarchy, before calibration. Both builders always attempt diagnostics after
the calibrated dataset exists. If construction, validation, serialization, or
writing fails, the release manifest records the failure and publication emits a
warning without discarding the dataset release.

Current UK national and rowwise builders use that same schema and writer. The
UK extension is fully typed: weight summaries, zero-weight strata,
geography-level pass rates, local fit summaries, and rotated holdout evidence
are validated at construction, including their cross-field reconciliations.
UK release workflows stop when diagnostics are unavailable because their later
release checks require that evidence; historical schema-6 and schema-7 UK
artifacts remain readable through isolated compatibility validation.

Standard publication uploads the locally built `releases/<id>/` artifacts to
the Hugging Face dataset, tags the release, and updates `latest.json`. It runs
on the build machine (it needs the freshly built H5), so it isn't a CI step:

```bash
tools/publish_release.sh releases/<id> --repo-id policyengine/populace-us
```

`tools/publish_release.sh` is a thin wrapper around `microcosm-publish-release`
(all arguments pass straight through). The moment `latest.json` goes live, the
publish CLI posts a release alert to Slack — `#populace-us` or `#populace-uk`,
chosen from the repo id.

Promotable UK release lines use pointers named `latest-<line>.json`. Publish a
cut for inspection with `--no-latest --tag-name <cut-tag>`, then promote the
reviewed cut with `--promote-line <line> --tag-name <cut-tag>`. Promotion moves
only that line pointer; the UK repository-global `latest.json` remains frozen
on the June 2023 release. Promotion reuses the immutable cut tag the inspect publication created (it checks the tagged manifest is byte-identical) and writes only the pointer commit, so the two-step sequence and a retry after a failed pointer commit both work. A line's registry entry is registered off the default variant until its first promotion; the default flips in a follow-up after the pointer exists.

The publisher uploads only the contract files, the release manifest's
artifacts and any `--extra-file`. The US fiscal-refresh tool therefore binds
its terminal gate verdicts as manifest artifacts: `input_coverage.json`,
`input_mass_parity.json`, `qrf_tail_concentration.json` and
`reform_coverage_smoke.json`. Both manifests also record the per-run QRF tail
register (`qrf_tail_register`) and the export-mass reference, so a waiver
ships with the release it waives, and a `gate_evidence` block that says of
each verdict whether it is bound, skipped by flag or never evaluated.

A US release may not store a column that looks like a policyengine-us variable
(lowercase snake_case) unless the engine it is certified against defines that
variable or `microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS`
registers the column with a reviewed reason (microcosm#1026: the engine
ignores such a column, which is how a renamed WIC take-up input shipped
unread). Three release seams refuse one, each against the engine the release
records as built-with:

- the fiscal-refresh tool, in its batched pre-export gates, and it grades the
  written H5, which must earn the same verdict (the exact-k ladder lane runs
  this tool);
- the source-enrichment probe, at certification, validation and publication;
- the ACS local-area chain's package stage, before it assembles the release
  directory.

A refusal names each column: rename it to its live input, or add a reviewed
register entry. The published default, its reported-receipt child and the
2026-09-23 ACS local-area release built on that child all store two retired
engine inputs, `would_claim_wic` and `medicare_part_b_premiums` (replaced by
`takes_up_wic_if_eligible` and `medicare_part_b_premiums_reported`), so each
would now be refused.
Check local files from HDF metadata alone with
`uv run python -m microcosm.data.stored_inputs path/to/populace_us_2024.h5`.

US exact-k ladder candidates use a tag-only lane. Run
`tools/build_us_exact_k_ladder_release.py`, then execute the `publish_command`
recorded in `package_result.json`. That command includes `--create-tag`,
`--no-latest`, and `--tag-only`: it uploads the immutable release and creates its
tag without committing candidate artifacts or release copies to the production
main branch. The launcher also forces `--no-staging`, so the build writes neither
a production nor a staging pointer. The candidate is therefore available only
by its explicit release id or tag until a separate promotion updates
`latest.json`. Because Slack alerts are coupled to that production pointer
update, tag-only publication sends no release alert. The promotion is the
standard publish of the same release directory, without `--no-latest` and
`--tag-only`: it reuses the existing release-id tag once the tagged
`release_manifest.json` is byte-identical to the local one, and writes only the
main commit that carries `latest.json` (microcosm#450). A tag that describes
another cut refuses before any commit.

Evidence-tier releases (microcosm#506) are the third lane: the best available
artifact when terminal gates failed, built with
`tools/build_us_fiscal_refresh_release.py --evidence-release` (which records
every gate failure with an owner issue in the release manifest's
`known_failures` block, or refuses) and published with

```bash
tools/publish_release.sh releases/<id> --repo-id policyengine/populace-us --evidence
```

The `--evidence` flag validates against the evidence release contract — a
certified-shape release is refused under it and vice versa — tags the
immutable release as usual, and moves only `latest-evidence.json`; the
certified `latest.json` pointer and the pe.py certification path never see
evidence artifacts. Each evidence publish supersedes the last, so
`latest-evidence.json` always names the best current evidence artifact
(consumers: `microcosm.data.latest_evidence_release`). Its Slack alert is
labeled as an evidence-tier publish.

The alert is a **no-op unless the channel's incoming-webhook URL is set**, so
configure it once on the build machine:

```bash
cp tools/release.env.example tools/release.env   # then paste the webhook URLs
```

`tools/release.env` is gitignored; the wrapper loads it (or you can just export
`SLACK_WEBHOOK_POPULACE_US` / `SLACK_WEBHOOK_POPULACE_UK` in your shell) and
warns if neither is set. After that, every release publishes with an automatic
Slack alert.

Canonical UK exact-k builds also require a stable, base64-encoded 32-byte
release key. Source `tools/release.env` before the national build as well as
publication. Two variables carry it during the report-format migration —
export both from the same key material:

- `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` — what the national build signs
  with (the gate-battery executor) and what schema-4 report verification
  reads.
- `POPULACE_UK_TERMINAL_GATE_SIGNING_KEY` — what schema-3 (legacy-format)
  report verification reads; retires with the legacy format.

The gate battery authenticates the complete report, canonical release id, and
exact calibration-diagnostics digest with HMAC-SHA256; the persistence seam
cannot sign caller-composed gate results, and publication independently
verifies the report from the same out-of-band key. If the key is missing or
malformed, a full-scale build persists the unsigned report and then refuses
to stage, and publication rejects unsigned reports.
