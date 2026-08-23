# Rare signed-tail battery lane notes

## 2026-08-21 — initialization

- Branch: `battery-rare-signed-tails`, starting at `2c7a7218` (`origin/main`).
- Binding scope: diagnose the 48 QED reds and the one-sided Keogh leg from the
  frozen arm-split adjudication; repair generating mechanisms only.
- Forbidden changes: gates, bands, ceilings, floors, folds, seeds, and owner-only
  exclusions/register entries.
- Build discipline: at most a 1% sample; check for a live
  `build_us_multispine_pool` process before every build; remain below 15 GiB RSS;
  run off-chain without `--logbook-prev-row-digest`; never touch
  `logbook-pending-chain.txt`.
- Environment receipt: `uv sync --all-packages --extra us` failed first on the
  sandboxed default UV cache, then on network DNS while fetching the locked
  pandas wheel from a writable cache. The primary-checkout venv lacks the build
  shard's `jsonschema`; the compatible `microcosm-f1` venv is used read-only for
  the full suite with lane-local sources and `UV_NO_SYNC=1`.
- GitNexus receipt: a fresh local index exists, but the CLI cannot query it
  because this worktree is absent from the read-only global registry. Direct
  source/call-site tracing and regression tests will substantiate every claim.
- No build has run and no chain-bearing file has been read or written as an
  input to a build.

## Evidence ledger

Mechanism classifications, realized-regime recomputations, code citations,
test receipts, build commands, RSS/queue receipts, and failure-line diffs will
be appended here as they are established.

### Frozen regime recomputation

- All 48 baseline-red QED checks use four receipted availability patterns. Each
  pattern at a given entity uses the same complete donor identity (person
  108,073; tax unit 57,630; SPM unit 43,961).
- Recomputing signs at `zero_atol=1e-6` gives 35
  `zero_inflated_positive` checks and 13 `three_sign` checks. There are no
  degenerate or single-sign QED targets, so a regime substitution cannot explain
  or honestly repair any of the 48 QED failures.
- The route partition is 17 early ASEC-to-ACS gap-fill QED checks and 31 late
  producer-complement checks (20 PUF-only, 10 ASEC-source-only, one overlapping
  producer target).
- Rare-tail actual-donor sign support is: collectibles +18 (late PUF; native
  ASEC +82), alimony +61 (early), casualty +27 (late PUF; native ASEC +79), farm
  operations -89 (early), and prior-year self-employment -48 (early). All five
  remain gated and sign-capable; the adjudication therefore still requires
  dense/additional evidence before a target-specific refit.

### Keogh one-sided leg

- Native ASEC clone 0 contains exactly two positive Keogh values: `2,040` and
  `30,000`. The late transfer is hard-wired to ASEC-origin clone 1, whose Keogh
  support is entirely zero, so its recomputed regime is `degenerate_zero`.
- The frozen bank contains 1,736,840 finite Keogh recipient draws and every draw
  is exactly zero. This proves loss before transfer fitting, not loss during the
  recipient merge.
- The retirement producer uniformly caps its 108,073-row ASEC training donor at
  5,000 rows and explicitly permits a rare output to remain zero when that draw
  contains no carrier. The smallest honest repair must preserve sign support in
  that capped training donor (with sampling-weight correction), rather than
  declare ACS absence or alter a terminal gate.

### Process visibility receipt

- Both `ps` and `pgrep -af build_us_multispine_pool` are denied by the managed
  macOS sandbox (`operation not permitted` / no process list). No build has been
  started: the binding pre-build queue check cannot yet be satisfied.

## 2026-08-22 — salvage disposition

- Inspected `git show --stat 6597ca87` and the complete diff against HEAD before
  altering the salvaged working tree. The salvage combined a sound narrow
  retirement fix with a much broader, unverified receipt-closure rewrite and
  stale generated hashes; cherry-picking it whole was not suite-safe.
- Recovered and then minimized:
  - the carrier-union retirement cap, common-zero weight calibration, and its
    mechanism regressions;
  - the seed-protocol description of the changed draw consumption;
  - exact QRF-regime capture for ordinary and banked ACS transfers;
  - bank schema 2, late receipt schema 4, legacy/pool/stacked materializer
    invalidation, and the focused receipt regressions.
- Discarded:
  - `.gitnexus/` binary/index material (moved recoverably to
    `/private/tmp/microcosm-rare-signed-tails-gitnexus-salvage`; the salvage ref
    also retains it);
  - the descriptor-plan and active-target indexing rewrite because it changed
    target-bank execution beyond the proven signed-tail defect;
  - the broad target-row/group, routed-wrapper, and sidecar validators and their
    tests because they were unrelated to regime persistence and introduced
    fixture failures;
  - generated bundle/coverage pins and final-report claims, which described the
    broad salvage tree rather than this selected implementation.
- Regenerated every retained source-attested hash from the selected tree. No
  salvage pin was copied.

## 2026-08-22 — corrected frozen classification

- All 48 red QED checks have four availability patterns, not one. The 17 early
  person checks use `pattern_00_677f6490`, `pattern_01_5874881e`,
  `pattern_02_7c3bceda`, and `pattern_03_04f75638`; the 28 late person checks
  use `pattern_00_36785ebf`, `pattern_01_c6777728`, `pattern_02_5e7dd311`, and
  `pattern_03_76a0101a`; the three late group-entity checks use
  `pattern_00_741356db`, `pattern_01_7c62ad55`, `pattern_02_d9f964c9`, and
  `pattern_03_2e5c739d`.
- Every pattern for a target binds the same complete donor identity: person
  108,073 rows/index SHA
  `8ecfa8bcb4a52bdb34654a00a910a698463337d6d10cc2da8750bb29bd47a8d2`,
  tax unit 57,630 rows/SHA
  `da2a8ee2216d1b7ffc77f19d0bf8e74978a8a5d9e46c6ccae7d13e15cb9a2fb0`,
  or SPM unit 43,961 rows/SHA
  `371654a8deb82ed2dc168c65729647e565b0044e06a6a03ce0d0976d17466865`.
- The correct counting identity is 48 checks, 42 unique targets, 168 unique
  target-pattern records, and 192 check-pattern links. At check level the
  regime tally is 35 zero-inflated-positive/13 three-sign; at target level it
  is 35/seven. No target is degenerate or single-sign.
- Five rows remain additional-evidence starvation blockers: ordinals 16, 28,
  33, 46, and 75. QED ordinals 78, 80, and 82 are separately co-classified as
  upstream support starvation caused by the retirement training cap: the old
  sample retained 102/2,057, 4/161, and 2/61 native positive carriers. Keogh
  retained zero of two.
- `experiments/battery_rare_signed_tails/realized_regimes.json` records every
  row, its four authenticated patterns, fit-boundary support, route, realized
  regime, terminal carrier/QED values, mechanism class, and smallest honest
  owner-scoped remedy. Its generator refuses changed authority hashes or count
  identities.
- The retrospective schema-1 bank proves target/pattern identity, predictors,
  donor index, weights, and `zero_atol`, but did not persist regimes. Counts are
  explicitly labeled fit-boundary reconstructions. Final transferred values are
  not a valid substitute for early fits because later producers can mutate
  those fields; for example, rental-income fit support is 216/104,057/3,800,
  while the final transferred ASEC clone-1 values are
  2,259/101,987/3,827.

## 2026-08-22 — selected mechanism implementation

- The retirement training cap now retains every row nonzero for any of its four
  targets, samples only the shared all-target-zero stratum under the unchanged
  5,000-row cap and seed, preserves carrier weights, and ratio-calibrates only
  the sampled-zero weights. It fails closed when the carrier union leaves no
  zero-stratum slot.
- Ordinary and banked ACS transfers now bind every fitted target to a known QRF
  regime. Target-bank schema 2 persists that label in each transition and
  projects it into operational receipts; schema-1 artifacts rebuild.
- Early direction, late group, and aggregate late receipts expose the same
  fitted-pattern regime rows. The canonical aggregate is validated as the exact
  concatenation of the 19 group receipts.
- Receipt/materializer identities advance to late receipt 4, legacy checkpoint
  4, pool checkpoint 8, and stacked checkpoint 12. The canonical H5 loader was
  updated in lockstep to accept newly emitted legacy materializer 4.
- Source-attested outputs were regenerated from this selected tree: US bundle
  SHA `d30642501b9b21cd4510492ebbc931e25f4bfd3576db0f96de4edded1aac8c72`;
  F0 field coverage 41,381/41,381; inventory coverage 40/40.
- Focused regressions cover production-call retention of both Keogh carriers,
  exact carrier/common-zero weight semantics, invalid/over-cap inputs, regime
  label closure, cold/warm bank persistence, schema-1 rebuild, authentic late
  aggregation, legacy H5 compatibility, and tamper rejection. Selected Ruff,
  H5 loader, field-usage, and coverage checks are green.

## 2026-08-22 — third-salvage disposition and independent artifact verification

- Salvage refs two (`8131e7af`) and three (`930a3de1`) are byte-identical
  trees; the pre-session working tree (including the untracked
  `experiments/battery_rare_signed_tails/` files) matched salvage three
  exactly, so recovery required no cherry-picks. Nothing was discarded.
- This environment restored `ps`/`pgrep` and network. The binding
  `uv sync --all-packages --extra us` completed in-worktree, replacing the
  earlier read-only borrowed-venv arrangement.
- Independent frozen-artifact verification performed this session (not
  inherited from the killed attempts):
  - All 42 target checkpoints' banked draws are sign-consistent with the
    recorded realized regimes at `zero_atol=1e-6`; zero inconsistencies. All
    seven `three_sign` targets show both negative and positive decoded draws,
    proving both signed classes existed in their fit donors.
  - Keogh bank draws: 1,970,973 stored, 1,736,840 finite, zero nonzero,
    `max_abs = 0.0` — the fitted transfer regime was degenerate.
  - Transferred-frame census: ASEC clone 0 keogh has exactly two nonzero
    values `{2,040, 30,000}` over 108,073 rows; ASEC clone 1 and every ACS
    clone are all-zero. The ACS absence is manufactured upstream, not
    structural.
  - The five starvation blockers' donor sign-carrier counts verify exactly:
    collectibles +18 (ASEC clone 1) / +82 native, casualty +27 / +79 native,
    alimony +61 native, farm operations −89 native, prior-year
    self-employment −48 native.
- Citation correction: the pool/stacked/legacy materializer version constants
  live in `tools/build_us_multispine_pool.py:231,282,319` (not
  `stacked_spine.py`); `h5_io.py:103` mirrors legacy 4 and
  `spine.yaml`/`spine.schema.json` pin stacked 12.
- The retirement cap parameter itself is untouched: `sources.yaml:2138`
  still declares `max_train_samples: 5000` and `sources.yaml` is not
  modified on this branch. Only the sampling design inside the cap changed.

## 2026-08-22 — final verification session (headless resume)

- Resumed on the intact working tree (matches salvage three; nothing reset).
  `ps` shows no `build_us_multispine_pool` process; the host queue log only
  ever waits on disk for its own `pkg1` step. No build was started from this
  lane at any point.
- Formatting: `ruff format` was applied to exactly the five files whose
  format drift this branch introduced (`acs_transfer.py`,
  `acs_transfer_bank.py`, `test_us_acs_transfer.py`,
  `test_us_multispine_pool_tool.py`, `test_us_retirement_distributions.py`);
  merge-base copies of every other touched file were already unformatted on
  `origin/main`, and the CI gate is `uv run ruff check .` (green), so those
  were left untouched.
- **Stale first-batch attestation pins found and corrected.** The previous
  sessions regenerated source-attested identities in two batches and the
  first batch described a tree state that no longer exists. On the final
  selected tree the following pins were recomputed through the same code
  paths the tests use and corrected:
  - BE bundle `spec_sha256` → `91d3a58e669950c7fab0aefeb74687278efca82d413a900e9ab1878b09a7a5a3`,
    UK → `54c0b70167b9d36e857a116ca4c80c74af142b6f09d5894ee979de62b9634738`
    (`test_spec_engine_country_bundles.py`); the shared `seeds.py`
    retirement-site declaration feeds every country bundle.
  - Loader golden semantic-hash vector →
    `43989f610f9691daf69e9ee74c1aa17f2a8bc3f07b9992489c65fa8b1ad747de`
    (`test_spec_engine_loader.py`).
  - Inventory `seed_protocol` →
    `d317a1cb0dc24ee49e528ac3a0d3337952b30c3e6610fd49026770a42b88d6be`,
    `seed_map` →
    `f9e71d22a4c8608d1e2ca99912f60dd06272e785e0babcca46a452c1a6df9562`
    (`inventory_coverage.py` EXPECTED_HASHES).
  - US bundle `spec_sha256` is
    `5b0014c3eb6cb121f0a9f2138ab860be30ef2251dd6ff4ec38cbf5f778899554` on the
    final tree. The earlier journal claim of `d30642501b9b…` described the
    intermediate first-batch tree and is historicized here; the committed
    report and `test_us_multispine_pool_tool.py` pins were corrected.
  - `docs/evidence/spec-engine/us-f0-coverage.json` was regenerated with
    `uv run python tools/spec_engine_coverage.py` — 41,381/41,381
    configuration fields, 40/40 inventory checks, exit 0.
  - `field_usage.py` pins and the identity-contract tests already matched the
    final tree (second-batch regeneration) and were not touched.
  - Two further cross-pins of the moved stacked-authority digest surfaced in
    the full-shard run and were corrected the same way:
    `test_spec_engine_legacy_adapter.py:200` (aggregate `34e86d1e…`) and
    `test_spec_engine_stacked_authority_semantics.py:77,93-95` (aggregate
    plus its `late_producer_schedule` component `bf95c78e…` →
    `870c38d740cf92d6ce7a96cac4c44864b03d87634d59177d701fecf3b2d94436`; all
    other components verified unchanged against the live projection). A
    repo-wide sweep for every pre-move value then returned zero hits.
  - A third same-class find: the exact-k ladder e2e downstream-consumer
    fixture fabricated a legacy publication manifest with
    `materializer_version: 3` (`test_us_exact_k_ladder_e2e.py:140`), which
    the branch's loader now correctly rejects (that rejection is itself a
    branch regression test). The fixture now models the current v4
    publication; the file re-ran green (3/3). A repo-wide sweep found no
    other stale version literals in test fixtures.
  - Suite-receipt composition note: the full build-shard run was launched
    before these three expectation fixes; its F marks (collection indices
    847, 945, and 2993-2995) are exactly those tests. All three files were
    re-run green in isolation on the final tree (6/6, 4/4, 3/3), and no
    product source changed between the runs, so the shard receipt composes:
    full run green everywhere else + isolated green for the three fixed
    expectation files.
- **Keogh leg identity correction:** the terminal leg is
  `person/source_operator_retirement_distributions/keogh_distributions[clone_0]/positive`
  (not `taxable_keogh_…`). Full-scale f025 emits
  `weighted positive-leg incidence ratio 0 is outside [0.8, 1.25]
  (asec=1.43573e-05, acs=0)`, confirming the adjudication: incidence fails
  before the five-carrier floor skips only the sign's QED
  (branch `stacked_spine.py:12068-12087`, floor at `3027-3031`).
- **Frozen-artifact re-verification (this session's own receipts):** the
  Keogh target bank (`late_producer_dag/person/source_operator_retirement_distributions/targets/000__keogh_distributions.h5`)
  decodes to 1,970,973 stored / 1,736,840 finite draws with zero nonzero and
  `max_abs = 0.0`; the transferred checkpoint's person
  `keogh_distributions` column holds exactly two nonzero values
  `{2,040, 30,000}` across all 1,970,973 pooled person rows. Both match the
  prior sessions' censuses exactly.
- **Baseline 1% projection** (`baseline_1pct_projection.json`, generated by
  `project_baseline_1pct.py` against the host-built
  `battery-verify/baseline1pct/pool.gates.json`): 127 failure lines over 93
  legs; 47 of the 48 frozen reds are red at 1%; ordinal 16 (collectibles
  positive) is the only frozen red without a 1% line; the Keogh leg emits no
  1% line in either direction because a 1% ASEC sample almost surely carries
  neither of the two native carriers. Gate-level Keogh proof therefore
  requires a full-scale build; the 1% after-build proves the 401k/403b/sep
  starvation repairs and overall non-regression instead.

## 2026-08-22 — newest salvage, completion audit, and final receipts

- Inspected newest snapshot `bfe794ad` with `git show --stat` and compared it
  with HEAD and the three older snapshots. It supersedes `930a3de1`,
  `8131e7af`, and `6597ca87` wherever they overlap. The pre-existing tracked
  working state matched the newest selected implementation, and its experiment
  and changelog files were present untracked, so no cherry-pick or reset was
  needed. Retained: the narrow retirement carrier-union fix, exact regime
  receipts/materializer invalidation, regressions, and experiment evidence.
  Discarded from older overlapping snapshots: `.gitnexus` binary state,
  descriptor/active-target rewrites, and unrelated broad validator work.
- The binding sync completed with a writable cache:
  `UV_CACHE_DIR=/private/tmp/microcosm-rare-signed-tails-uv-cache uv sync
  --all-packages --extra us` → `Checked 100 packages`.
- Re-ran `build_realized_regime_evidence.py --check`; all authority hashes,
  48/42/168/192 identities, pattern IDs, donor lengths, and donor hashes are
  current. An independent raw-draw audit of all 42 target checkpoints found 35
  zero-inflated-positive and seven three-sign targets, zero regime/sign
  inconsistencies, and both signed classes in every three-sign target.
- Re-censused the frozen retirement mechanism from the selective H5 columns:
  native ASEC clone 0 has exactly two Keogh carriers `{2,040, 30,000}`; ASEC
  clone 1 and every ACS clone are zero. Replaying the old seeded uniform cap
  retained Keogh 0/2, 401k 102/2,057, 403b 4/161, and SEP 2/61. The five intact
  sparse route donors remain +18/+61/+27/-89/-48 carriers.
- Found one resume-validation defect in the newest salvage: canonical JSON is
  written with sorted mapping keys, but the regime validator compared mapping
  iteration order with model-target order. A valid durable checkpoint therefore
  rebuilt unnecessarily. `stacked_spine.py` now checks exact key-set equality;
  the separately persisted `model_targets` list remains the ordering authority.
  `test_realized_regime_receipt_survives_canonical_json_key_sorting` proves a
  sorted authentic map resumes and a missing target fails closed. The exact
  durable-resume boundary test also passes.
- The first complete build-shard run found one remaining stale source-attested
  *test assertion*: `late_producer_schedule_receipt.payload_sha256`. The
  structural `schedule_sha256` stayed unchanged. The authoritative US bundle
  generator passed byte-for-byte at bundle SHA `5b0014c3…9554`; the derived
  payload assertion was updated from `7766f2e…` to live `6af0cc77…`, its
  focused node passed, and the entire build shard was rerun green.
- Final mandatory verification on the same source tree:
  - calibrate: 201 passed, 2 warnings, 36.25s;
  - data: 275 passed, 1 skipped, 37.50s;
  - fit: 93 passed, 1 warning, 249.72s;
  - frame: 294 passed, 36 skipped, 1 warning, 888.52s;
  - build: 5,973 passed, 39 skipped, 1,941 warnings, 7,858.53s;
  - repo-wide Ruff: `All checks passed!`;
  - US bundle generator `--check`: passed;
  - realized-regime evidence generator `--check`: passed.
- Regenerated the 1% baseline projection into `/private/tmp`; it is
  byte-identical to the committed projection (127 failure lines, 93 legs,
  47/48 frozen QED reds visible). Baseline-vs-baseline use of the diff tool
  reports 127 shared and zero added, removed, or changed lines.
- No branch-specific after artifact exists. The host path named `pkg3` belongs
  to the unrelated `microcosm-pkg3-two-part` worktree, and the queue contains no
  rare-signed-tail step; using it would fabricate evidence. Current managed
  `ps`/`pgrep` calls are sandbox-denied, superseding the earlier historical
  process-visibility note. No pool build was started, no chain input was used,
  and `logbook-pending-chain.txt` was never touched. The actual 1% after diff
  and the full-scale Keogh gate proof remain explicit host-owner actions.
