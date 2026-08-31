# Belgium calibration, validation, and behavior-boundary contract

## State

Continued on 2026-08-31 on `be-benefit-participation-targets-resume` from the
completed local handoff at `490caa69`. Live GitHub access is restored:
authoritative Microcosm main remains `d1e3e397`, and existing #824 remains a
draft at its reviewed head `ed6dc3d8`. Chronicle #213 now supplies the stricter
source-faithfulness ruling that Opgroeien, Iriscare, and Ostbelgien require
native publisher artifacts. The Belgium contract therefore carries a separate
typed publisher-source readiness gate in addition to population construction,
input, period, and completeness gates.

Reviewed Microcosm #824 head `ed6dc3d8` remains an ancestor, and authoritative
main `d1e3e397` is merged at `03e4d494`, bringing in reviewed #825 head
`510e3e6c`. Exact Chronicle #212 base/head `10597ae6`/`0f75a2bb` and the live
#213 head `fe3fd816` are recorded in the contract. Preserved `.lane-inputs/`
remains intact and untracked.

## Done

- Finished the 15-row Belgium target surface, ten typed nullable person-input
  contracts, ten one-to-one declared Chronicle record-to-input links, and a separate
  three-row monetary inventory. GRAPA has an exact unit/scope mapping but is
  still missing its input, period alignment, and completeness receipt; all
  nine child mappings additionally keep scheme construction `required_missing`.
- Corrected every ownership inversion: Microcosm owns and populates measured or
  latent flags, PolicyEngine consumes them and owns non-legal mechanics, and
  Axiom receives no synthetic behavior concept. Pension and ONEM now carry the
  same explicit consumer/support direction.
- Enforced unknown-as-null, complete-imputation readiness, exact selector/basis
  periods, typed declared scheme source/scope/entity contracts, one-to-one mapping identity,
  explicit missing geography bridges, recursive value-free monetary profiles,
  and refusal of behavioral names in both input IDs and columns.
- Enforced validation-only exclusion from calibration compilation. PIT remains
  a blocked calibration candidate; ONSS and NBB are validation-only; EUROMOD,
  FPB, constructed comparisons, monthly monetary snapshots, and unreviewed
  HFCS wealth remain outside the calibration surface.
- Regenerated the Belgium country golden only after semantic review. Its country
  fingerprint is now `75448d958ef56404fd343ae0ba19bc047f7d4f1d739b06c0970fb4281d618772`;
  the separately tested bundle spec identity remains
  `935f8dedcd2d1b99abe57c1b5d990bc345f4cffbc7ab0cc8791733bde6721d14`.
- All 548 focused country, gate, Ledger, monetary, profile, population-input,
  and bundle cases pass. Focused Ruff and `git diff --check` pass.
- After reconciling Chronicle #212 with #213, all 730 focused country, gate,
  Ledger, monetary, profile, population-input, and bundle cases pass again;
  repository Ruff, the unchanged 123-package lock, the 319-file CI grouping,
  JSON parsing, and `git diff --check` also remain green.
- Committed the complete semantic and reviewed-golden step as `d77ccf78`
  (`Complete Belgium target input boundary`) without staging lane bookkeeping.
- Verified the unchanged lock offline (123 packages), CI grouping (319 tracked
  test files, `verification=ok`), repository Ruff, and both worktree and
  main-to-head diff whitespace checks.
- Passed all 1,688 runnable shared-spec CI cases with 43 expected skips and two
  explicit deselections, plus the full calibration shard (212 passed). The two
  deselected US parity regeneration cases fail identically on
  `main@d1e3e397`: the local feed exposes a CBO `source_projection` while the
  committed US reference is `observed_only`; this branch does not alter that
  reference or assertion path.
- A broader build-shard attempt reached 1,218 passes and 18 skips before being
  stopped after those same two baseline failures were captured. The complete
  shared-spec group then passed as recorded above.
- Reconfirmed that `main@d1e3e397` and reviewed #824 head `ed6dc3d8` are both
  ancestors; this branch is 13 commits ahead and zero behind authoritative
  main. Live fetch, `gh pr view 824`, and the explicit
  `git push origin HEAD:be-benefit-participation-targets` all failed only on
  DNS resolution. No duplicate PR, force-push, readiness change, or merge was
  attempted.
- Wrote the complete base/head, dependency, ownership/readiness, test, blocker,
  and frozen-review report to `.lane-inputs/OUT.md` without staging lane
  bookkeeping.

- Re-read the complete Fable v3 review and official benefit-participation law
  audit. Confirmed that every receipt/status flag is Microcosm-populated,
  PolicyEngine only consumes it and owns non-legal mechanics, and Axiom may see
  only documented legal events/statuses/claims.
- Attempted `git fetch --prune origin`; it failed because the managed shell
  cannot resolve `github.com`. The canonical local clone has the same
  `origin/main` (`d1e3e397`) and exact #824 branch/head, so no newer local
  authority is available and the existing #825 merge remains the safe base.
- Verified salvage `b89fa566` is a child of `779cf9e2` and every one of its ten
  blobs matches the worktree. The five product files are preserved; the five
  lane-input files remain untracked, so the salvage must not be cherry-picked
  wholesale.
- Made nullable boolean input storage explicit and added a separate
  `completeness_readiness` gate. Unknown values remain null; activation requires
  a complete-imputation receipt and proves `n_unknown=0`. Focused population
  and country-profile tests pass, and both new test files are explicitly named
  for the shared-spec CI group.
- Read `AGENTS.md`, `CLAUDE.md`, `README.md`, `DESIGN.md`, and the GitNexus
  impact-analysis workflow.
- Inspected the stopped lane's branch, status, worktrees, remotes, refs,
  reflog, committed history, staged and unstaged diffs, salvage refs, preserved
  inputs, journals, and the exact reviewed diffs for PR #824 and PR #825. No
  tracked work was discarded or overwritten.
- Attempted the required direct `git fetch origin main`; the managed shell
  cannot resolve `github.com`. Imported the canonical clone's already-fetched
  `origin/main` instead, verified it is exactly `d1e3e397`, and verified merge
  parent `510e3e6c` contains the generic monetary-target primitives.
- Located reviewed PR #824 at exact head `ed6dc3d8`. Its two commits add a
  GRAPA validation declaration and a generic execution blocker, but incorrectly
  say PolicyEngine supplies receipt/behavior flags. This continuation corrects
  the contract to Microcosm-owned measured or latent data consumed by
  PolicyEngine.
- Located reviewed Chronicle PR #212 at exact head `0f75a2bb`. It declares the
  GRAPA and regional child-benefit record identities and scopes, but its
  dashboard captures and manual transcriptions are not activation-grade where
  the newer Chronicle #213 source audit requires native publisher artifacts.
- Confirmed the immutable boundary: Chronicle owns publisher facts and
  provenance; Microcosm owns population construction, calibration/validation
  selection, measured or latent flag inputs, and their population; PolicyEngine
  consumes those inputs and owns take-up assignment, labor response, and Axiom
  orchestration. Axiom may receive only exact public-document concepts, never
  synthetic take-up concepts.
- Merged authoritative main without conflict or duplicated commits. The merge
  preserves both exact reviewed ancestries and makes #825's value-free
  `MonetaryTargetProfile`, exact accounting basis, prepared-measure receipt,
  and binding refusal rules available to the Belgium package.
- Audited every reviewed Chronicle #212 GRAPA and regional child-benefit row.
  The exact person-unit mappings that can be declared are scheme/statistical-
  scope mappings, never NUTS proxies: Opgroeien basic-amount children;
  Iriscare entitled children and payment recipients; Ostbelgien paid children
  and payment recipients; and the four Walloon child partitions. Publisher
  family/household units remain unmapped because equivalence to a SILC
  household is not established. Every declared person mapping must remain
  blocked until Microcosm populates the exact measured or latent status flag.
- Audited the broader Belgian fact catalog. PIT 2023 is exact but mismatched to
  SILC-2023's 2022 income reference; ONEM 2024 is a monthly-average recipient
  stock without a matching population flag or period; the merged SFPD pension
  package labels a January snapshot as calendar-year 2025; and ONSS explicitly
  calls its current Axiom Article-17 mapping approximate. None may silently
  activate. NBB national accounts, Eurostat, EUROMOD, FPB/BFP, and constructed
  comparisons are validation-only.
- Confirmed that no reviewed Chronicle branch contains an official, pinned
  NBB/ECB HFCS fact. The preserved object is only an offline-fetch handoff, so
  wealth stays blocked pending a checksummed workbook ingest and exact
  interview-period/support mapping.

## Next

- Verify the source-readiness changes and regenerated Belgium golden, commit,
  normal-fast-forward the existing `be-benefit-participation-targets` branch,
  update only draft #824, and freeze its exact head for Fable review. Never
  open a duplicate, mark ready, merge, publish, or run a restricted build.

# ACS predictor release join

> **Historical note (2026-08-28).** This journal describes the
> `acs-predictor-release-join` lane as of 2026-08-27. The branch has since
> been merged into the `stacked-release-fix-train` integration branch
> together with the #794 gate-alignment and #798/#799 pregnancy/prior-year
> fixes, with spec envelope digests and coverage evidence regenerated over
> the union tree. Treat the "State"/"Next" sections below as history;
> check git/GitHub for current truth.

## State

Complete on 2026-08-27. The owner-approved release-time join from stacked-pool
ACS source lineage to the SHA-pinned 2024 one-year ACS person/household zips is
implemented, receipted, real-pool exercised, and fully verified. It populates
the six archived donor models' CPS-named predictors through reviewed
native-ACS crosswalks with strict hash, lineage, collision, totality, universe,
and clone-fan-out contracts. Model selection logic and every gate threshold
remain unchanged. The completed evidence and handoff are in `out.md`. No
network access, pool build, release build, publication, push, retraining,
threshold change, or launcher-contract edit occurred.

## Done

- Read `CLAUDE.md` and the prior weeksgate report's six owner-ruling items with
  their release-call and model-consumer evidence.
- Confirmed the requested branch `acs-predictor-release-join` is clean at
  `606cbd69`, based on `stacked-release-gate-alignment`.
- Read the GitNexus exploration and impact-analysis workflows. This workspace
  exposes neither GitNexus repository resources nor query tools, so the same
  call/dependency analysis will be performed directly from source and tests.
- Recorded the required source zips and SHA-256 pins, strict exact/total join
  contract, explicit crosswalk and receipt requirements, and verification
  boundary.
- Proved that `person_source_id` is not a reversible ACS key: ACS people are
  sorted by `(SERIALNO, SPORDER)`, receive a zero-based raw spine ID, and then
  receive a collision-dependent assembly offset. The pool retains the raw
  spine ID, `source_row_id`, `source_person_id`, household `SERIALNO`, and clone
  metadata, so the release join will use the retained semantic
  `(SERIALNO, integral SPORDER)` key and treat `person_source_id` only as the
  one-to-many clone fan-out identity.
- Audited the supplied candidate pool read-only: 856,626 distinct ACS source
  people expand to 1,736,840 rows (856,626 clone 0, 856,626 clone 1, and
  23,588 clone 2), with no duplicate `(person_source_id, clone_index)` pair.
  Every ACS row agrees with its raw spine/source lineage, and all selected
  people match the pinned raw person archive exactly.
- Verified both local archives against the charter pins. The person archive
  has 3,422,888 unique `(SERIALNO, SPORDER)` rows and no household orphans;
  the household archive has 1,631,969 unique serials, including 1,531,614
  occupied records. Both contain every requested native predictor.
- Established the disability universes from the pinned archive and the
  archived repository mapping: DEAR/DEYE are complete at every age;
  DREM/DPHY/DDRS are asked from age 5; DOUT from age 15; native code 1 is the
  consumer's difficulty bin and code 2 (plus an age-valid universe blank) is
  its non-difficulty bin.
- Established the consumed race/Hispanic bins: both SCF models distinguish
  White, Black, Asian, Hispanic, and Other; ORG distinguishes Hispanic,
  non-Hispanic White, non-Hispanic Black, and Other. `RAC1P`/`HISP` can map
  exactly to those bins without inventing detailed CPS combinations.
- Recovered the complete 2024 Census detailed-occupation-to-`POCCU2` consumed
  grouping from the native ASEC relationship and confirmed that ACS `OCCP`
  uses the same detailed codes. `PEIOOCC` is therefore a direct carry, while
  `POCCU2` will use an explicit reviewed 53-bin table; blank out-of-universe
  occupation maps to code 0, military to 52, and code 9920 to 53.
- Confirmed ACS `TEN` maps to the SPM vehicle model's three consumed tenure
  bins (mortgaged owner, outright owner, non-owner); no-cash-rent and verified
  group-quarters blanks belong to the non-owner bin. Confirmed the SSI model's
  `SSI_VAL` use is only the `> 0` reporter anchor and that native ACS `SSIP` is
  already carried as harmonized `ssi_reported`, observed exactly from age 15.
- Added the dedicated `acs_release_predictors` release boundary. It verifies
  the two canonical archive pins before opening either zip, streams only
  selected households, validates exact archive members and headers, rejects
  raw/person/clone collisions, binds retained pool lineage to
  `(SERIALNO, SPORDER)`, requires total one-to-one source-person matching, and
  fans mapped values to clones only through `person_source_id`.
- Added explicit disability, race/Hispanic, 530-code occupation, and tenure
  tables. A canonical crosswalk payload is pinned at SHA-256
  `1d4906242e9c73e31b3283659e5cad8242b8cbc42914ab6fa59547a10c8770e9`
  and rides the JSON-ready join receipt with per-model/per-predictor
  ASEC-native, ACS-joined, and still-null counts.
- Preserved CPS disability universe semantics (`-1` below the question age)
  and the ACS occupation universe. Blank `PEIOOCC` uses the CPS NIU sentinel
  `-1`; blank `POCCU2` remains 0 through age 15 and maps to the consumed
  no-occupation code 53 only from age 16. This explicitly preserves the
  one-year ACS/CPS source-universe gap instead of assigning every ACS
  15-year-old a never-worked status without source evidence. The explicit
  occupation table covers every one of the 530 codes in the pinned ACS person
  archive and every consumed POCCU2 bin.
- Changed the SSI-disability reporter read, without source routing, to
  row-wise coalesce measured ASEC `SSI_VAL` with harmonized native ACS
  `ssi_reported`. Adult blanks and conflicting dual reporters fail; genuine
  below-age-15 ACS blanks remain null in the frame and become false only for
  the consumer's `> 0` predicate.
- Hardened the join after independent crosswalk review: raw ACS `SSIP` and
  `ADJINC` now travel through the pinned join and must agree exactly with every
  native clone-0 `ssi_reported` value under the established adjusted-dollar
  formula. Raw `ESR`/`OCCP` must obey their exact age-16 universes, and all
  ASEC predictor receipt cells must be numeric and finite with complete,
  nonnegative `SSI_VAL`.
- Updated the SSI signal diagnostic to use the same row-wise reporter coalesce
  as the model consumer, while retaining the archived native-role anchor
  scope. A lost positive ACS-native reporter can therefore no longer evade the
  release gate merely because `SSI_VAL` is null on physical ACS rows.
- Added focused tests for crosswalk identity/all consumed bins, exact join and
  clone invariance, ASEC byte preservation, receipt contents, missing joins,
  raw and source-identity collisions, hash refusal, no-ACS identity, and SSI
  coalescing/universe refusal. Coverage now also fixes the age-15 occupation
  gap, malformed ESR refusal, malformed ASEC SSI refusal, raw SSI attestation,
  and gate-side ACS reporter preservation. The complete join, SSI, and
  source-blindness test files pass together, and focused Ruff is green.

- The release CLI now accepts the person/household zip and lowercase 64-hex
  SHA-256 options as an all-or-none set. It invokes the authenticated join
  after the last unrelated native-input gate and before SCF wealth, therefore
  before all six archived donor-model stages, then carries the complete join
  receipt into both `build_manifest.json` and `release_manifest.json`.
- Added parser refusal tests, a source-order contract over all six model calls,
  an end-to-end mocked main corridor that verifies the exact four join
  arguments and runtime ordering, an AST contract that binds the saved receipt
  to the sole manifest call, and JSON round-trip assertions for both manifests.
  Focused Ruff, five parser/order/manifest cases, and all six parametrized main
  corridor cases pass.
- Exercised the hardened join read-only on the complete supplied candidate.
  The 3,239,263,147-byte H5 matches its frozen manifest SHA-256
  `871b7e6467675a1e9475b54fd1baf64c53c0f75a3258b8357303a8df0d53642d`.
  The current official loader refuses that older candidate before H5 loading
  because its archived primary-QRF worker binding predates this branch's
  execution identity; this is an existing candidate/code-version mismatch.
  Loading those independently manifest-hash-verified bytes with their frozen
  assembly receipt allowed the join boundary itself to be tested without
  writing an artifact.
- The real join passed every source, raw-key, universe, SSI-attestation,
  totality, collision, and clone-fan-out check: 856,626 unique ACS source
  people matched 856,626 raw people in 382,903 households and populated
  1,736,840 support rows (856,626 each at clone indices 0 and 1, plus 23,588
  at clone index 2). Every CPS-named predictor consumed by the six models has
  234,133 valid ASEC-native and 1,736,840 ACS-joined cells with zero nulls.
  The logical SSI reporter anchor has 234,133 ASEC cells, 1,475,235 observed
  ACS cells, and exactly 261,605 preserved child-universe null support rows.
- The first real-data attempt exposed fixed-format HDF's expected object dtype
  for mixed-source columns. Tightened the ASEC validator to inspect each cell,
  accepting object-wrapped real numbers while still refusing strings,
  nonfinite values, nulls, and negative SSI. A focused H5-shape regression and
  all 15 join tests pass before the successful full-pool rerun.
- Added `changelog.d/acs-release-predictor-join.fixed.md`, describing the
  pinned release join, reviewed mappings, fail-closed lineage, dual-manifest
  receipts, and unchanged model/gate behavior.
- Repository Ruff passes, and the CI inventory verifier reports 310 tracked
  tests with `verification=ok`. Four complete pytest shards pass in separate
  processes: frame 295 passed/36 skipped, fit 93 passed, calibrate 203 passed,
  and data 318 passed/2 skipped.
- The first complete build-shard process reached 100% with 6,575 passed and 45
  skipped, plus five failures and six fixture errors. All eleven were the same
  expected source-attestation drift: `ssi_disability_criteria.py` belongs to
  both the direct and QRF seed-kernel inventories, so this task's runtime edit
  moved the seed protocol, compiled US seed map, every country spec identity,
  the minimal loader golden, and the generated coverage evidence. No ACS join,
  release CLI, manifest, archived-model behavior, or gate test failed.
- Applied the repository's established five-file source-identity repin only:
  seed protocol `59a098f9...31d8b`, US seed map `ce3850d8...e42ab`, US spec
  `16b7d5e6...dca38`, UK spec `2f921e4c...33a62`, BE spec
  `c87a0012...34ba`, and minimal-loader golden `b4946105...f2af`; regenerated
  `docs/evidence/spec-engine/us-f0-coverage.json`. All 25 affected cases and
  focused Ruff pass. The US bundle generator `--check` passes at the new spec
  identity, and coverage `--check` passes at 42,122/42,122 fields and 41/41
  inventory checks.
- Re-ran the complete build shard after the reviewed repin: 6,586 passed and
  45 skipped, with exit code 0. Re-ran final repository Ruff, the 310-file CI
  inventory verifier, both retained spec `--check` commands, and
  `git diff --check`; all pass. Wrote the required final report to `out.md`.

## Next

- No work remains in this lane. The dispatcher owns rebasing and the launcher
  contract update. A future authorized build must produce a pool whose current
  source-attested worker identity passes the official release loader; the
  supplied older candidate is useful join evidence but cannot be promoted.

# Weeksgate: stacked release gates and integer-week provenance

## State

Complete on 2026-08-27. Real-pool provenance has refuted the proposed
post-transfer amount-mapping mechanism: every fractional week is an ACS-origin
non-native clone prediction outside the calibration's clone-0 recipient scope.
The source codec, weeks-gate architecture, source-scope, clone-layout, and
stable-identity repairs are implemented and focused-tested. The complete
release-call roster is classified; six archived-model input assumptions require
owner rulings and are deliberately reported instead of guessed. Repository-wide
Ruff, the CI inventory verifier, and all five full pytest shards pass in their
required independent processes. The completed provenance, audit, verification,
judgment calls, and host-owned checkpoint-rerun consequence are in `out.md`.
No network access, artifact build, publication, push, pool build, or release
build is in scope.

## Done

- Read `CLAUDE.md` and the GitNexus debugging workflow.
- Confirmed branch `stacked-release-gate-alignment` is clean at `4f453746`.
- Confirmed the local GitNexus CLI is installed but the repository is not yet
  indexed. Its offline analyzer parsed the repository but could not register
  the index because the sandbox forbids writes to `~/.gitnexus`; the generated
  local index was moved out of the worktree to `/private/tmp`.
- Recorded the four requested workstreams: fractional-week provenance and PUF
  misclassification; integer-support calibration repair; stacked/legacy weeks
  gate alignment; and the full release-side gate archaeology sweep.
- Recorded the required verification boundary: repository Ruff plus one pytest
  process per shard, with no pool/release builds.
- Read the fixed-format HDF5 blocks directly and classified all 369 noninteger
  `weeks_unemployed` rows: 360 are ACS clone 1 (355 UC=0, 5 UC>0) and 9 are ACS
  clone 2 (all UC=0); all are positive, all 369 values are distinct, and the
  exact range is 1.0003521955067698--37.796501228614694.
- Confirmed zero nonintegers on ASEC rows and ACS clone 0. The receipted
  calibration covers exactly the 856,626 ACS clone-0 rows, maps 8,419 carrier
  amounts onto observed ASEC support with zero donor-support violations, and
  records QED 0.5882352941176471 to 0.0.
- Reproduced the 5,218-row false "PUF" classification: the legacy role helper
  calls every clone index above zero `puf_tax_detail`, regardless of raw source
  channel. The rows are all ACS-origin clones: 4,733 integer clone-1 rows, 355
  fractional clone-1 rows, 121 integer clone-2 rows, and 9 fractional clone-2
  rows with nonzero weeks while UC is nonpositive.
- Traced the actual fractional mechanism to the ACS transfer target codec:
  PolicyEngine-US declares `weeks_unemployed` as physical `float`, so the
  generic QRF path treats it as continuous even though its reviewed source
  contract is integer-supported. The later calibration repairs clone 0 only.
- Bound every final fractional value bit-for-bit to the late-transfer target
  bank. Its raw QRF draw contains 711 nonintegers: 342 ACS clone 0, 360 clone 1,
  and 9 clone 2. Post-transfer calibration changes 13,417 clone-0 rows and
  eliminates all 342 clone-0 fractions; the 369 out-of-scope clone fractions
  pass through unchanged.
- Added `weeks_unemployed` to the ACS transfer's authority-bound discrete
  numeric target registry. The existing codec now snaps every prediction to
  actual observed ASEC donor support, and the execution-contract identity
  receipts the policy alongside the two mortgage-year targets.
- Added a focused ACS-transfer regression proving weeks predictions retain
  observed integer support and the execution contract declares the target.
- Regenerated the checked-in US imputation authority so
  `transfer_execution.discrete_numeric_targets` includes `weeks_unemployed`.
  The generator's compile and byte-staleness checks pass at bundle spec SHA
  `821d5838da3ac368170e61e017f1a72648f93e8a011aa40e33b8c2b4b14511f3`
  at that source-fix step; the later runtime/seed identity repin below
  supersedes this intermediate digest. The spec-bundle and imputation-
  semantics suites pass.
- Passed the complete ACS-transfer test file (65 tests), the complete
  post-transfer calibration receipt-contract file (47 tests), focused Ruff,
  and `git diff --check` using the prebuilt `.venv` directly. A task-local
  `UV_CACHE_DIR` later allowed the required `uv run --no-sync` commands to run
  against that same prebuilt environment without accessing `~/.cache/uv`.
- Modernized the weeks summary/gate to derive its roster from actual assembled
  source channels, while retaining the legacy ASEC/PUF role path. The ASEC
  source-validity scope, direct native-clone reconciliation scope, and reviewed
  UC-constraint scope are now distinct and explicitly receipted in details.
- Kept all four plausibility-band tuples and every numeric threshold unchanged;
  non-ASEC assembled channels use the unchanged legacy recipient band.
- Added stacked ASEC+ACS and legacy ASEC+PUF fixtures covering roster detection,
  raw-source scoping, native reconciliation, and UC constraint ownership. The
  complete weeks file passes (26 passed, 1 skipped) with focused Ruff.
- Replayed the updated gate over the supplied pool's exact live arrays and
  weights. It now reports 234,133 valid ASEC source rows, 108,073 exact native
  reconciliation rows, 982,686 UC-constrained rows, zero source/UC failures,
  both unchanged channel bands passing, and only the genuine 369 nonintegers.
- Added a centralized provenance-owner API that distinguishes validated
  physical source channels from legacy clone-operator roles. The weeks gate
  now consumes that API rather than reading provenance columns directly, and
  both repository source-blindness tripwires pass alongside the stacked and
  legacy provenance/weekly-signal suites.
- Made WIC's deterministic draw key prefer the assembly-unique
  `person_source_id` on multispine frames, before the source-local raw identity
  triple. Distinct ASEC/ACS records can no longer collide when their vintages
  align, while all support clones of one assembled person retain the same draw
  and the legacy key order remains unchanged.
- Completed the release-tool call-roster audit. It found unambiguous physical
  ASEC scoping repairs in SSI take-up, workers' compensation, alimony,
  retirement contributions/distributions, and Medicare; clone-2 layout fixes
  in Head Start, voluntary filing, and prior-year income; and a stable-key fix
  in WIC. Archived ASEC-only predictor assumptions in SSI disability, SCF
  wealth, SCF auto loans, and SIPP vehicles require explicit model-owner
  rulings and will be reported rather than guessed.
- Reworked the assembled Head Start and voluntary-filing receiver layouts to
  key by assembly-unique source ID plus explicit clone index, accept clone 2
  and later, reject duplicate source/clone rows, choose clone 0 (or the lowest
  surviving clone) deterministically, and fan one source-level decision to all
  clones. Their legacy role-only duplicate contract remains unchanged.
- Replaced occurrence-pair clone diagnostics in voluntary filing and
  prior-year income with all-clone grouping on assembled source IDs, so a
  clone-2-only divergence is now detected. Focused tests and Ruff passed for
  all three module/test pairs.
- Scoped Medicare and retirement release diagnostics to physical ASEC source
  rows, separating raw-source validity from native direct-carry reconciliation
  where transferred clones intentionally differ. Kept the producer kernels
  origin-blind: the authenticated-pool release path skips those producers, and
  indirect physical-source routing would violate the repository's population-
  operator boundary. Their 60 focused module/source-blindness tests and Ruff
  pass after that review correction.
- Scoped alimony and workers' compensation raw validity to every physical ASEC
  clone and exact source-carry checks to physical ASEC native rows, leaving all
  clone-operator plausibility bands unchanged. Stacked ASEC+ACS and legacy
  ASEC+PUF fixtures pass (28 alimony and 21 workers' compensation tests).
- Changed SSI reporter-lineage capture to validate `SSI_VAL` only on physical
  ASEC rows and accept null ACS raw-source cells. Assignment remains source-
  blind: it consumes the source-ID set captured before L0, or obtains that set
  through the reporter helper when no explicit set is supplied. The complete
  SSI take-up file passes (71 tests).
- Fixed SSI-disability's non-fatal clone-divergence diagnostic to group every
  assembled clone by source person, so clone-2-only divergence is reported.
  The existing decision not to make divergence gate-fatal remains unchanged
  for an owner ruling; the complete focused file and source-blindness checks
  pass.
- Narrowed physical-channel resolution to a gate/reporter-only provenance API,
  removed its general runtime/PUF-support re-exports, and added a static exact-
  caller contract (including internal mask-helper callers). A future derive,
  impute, or wrapper use now fails the source-blindness suite instead of
  passing through indirection.
- Replayed the repaired release gates read-only against the supplied pool.
  Alimony, Medicare, retirement contributions/distributions, workers'
  compensation, and SSI reporter capture pass; the weeks gate now fails only
  on the genuine 369 fractional values. Prior-year income remains outside its
  unchanged availability band and WIC finds pregnant nonfemale rows, both
  genuine data/spec outcomes rather than stacked-layout archaeology.
- Completed an adversarial review of the repaired code and focused tests with
  no additional implementation defect found. It confirmed six owner-ruling
  items: SSI disability criteria, SCF wealth, SCF auto loans, SIPP vehicles,
  SIPP tips, and ORG wages/FLSA all consume ASEC-only archived predictors on a
  frame whose 1,736,840 physical ACS rows carry null source cells. ORG is
  guaranteed to fail its unchanged race/occupation bands; SIPP tips' unchanged
  tipped-occupation band passes while the ACS channel is dead.
- Passed repository-wide Ruff and the CI test-group inventory verifier. The
  calibrate, data, fit, and frame shards pass in four independent pytest
  processes.
- Corrected the reviewed WIC seed protocol to match the implemented assembled-
  multispine key precedence: assembly-unique `person_source_id` first, then the
  unchanged legacy raw/support/person fallbacks. An exact seed-grammar test now
  binds that order.
- Re-pinned the fail-closed spec-engine proof after adding one authored
  transfer-execution field and one resolved seed-protocol field: 42,122 total
  fields (32,352 authored and 9,770 resolved), complete exact-pointer claims,
  and all 41 inventory checks. Regenerated the committed coverage report and
  validated the final US spec SHA
  `5f44d96d45e9aabcea2d565ef063d68bfc0652df1b38b08aa31ce6896d15f371`.
- Verified in a detached `origin/main` worktree, using the same prebuilt venv,
  that the old BE, UK, and minimal-spec golden vectors still pass there. Their
  current repins therefore reflect this branch's attested runtime and seed-
  protocol changes rather than environment drift. All 102 tests in the eight
  directly affected spec-engine files pass; generated-bundle and coverage-
  report byte checks, focused Ruff, and `git diff --check` also pass.
- Ran the full build shard after that coherent spec repin. It reached 100% with
  exactly one failure and no errors: the multispine constants-adapter fixture
  still expected the former live US spec SHA. Updated only that live-binding
  expectation to the regenerated final SHA; the separate arbitrary checkpoint
  identity fixture remains deliberately unchanged.
- Passed the complete multispine-pool-tool file after that correction and
  committed the coherent fixture repin as `12a918ed`.
- Reran the entire build shard from zero in one process: 6,608 tests collected,
  100% reached, and pytest exited 0 with expected skips only. All five full
  package shards, repository-wide Ruff, generated-artifact checks, the spec
  coverage proof, the CI test inventory, and `git diff --check` are green.
- Wrote the final provenance tables, mechanism verdict, per-file rationale,
  exhaustive release-gate audit, owner-ruling list, verification evidence, and
  judgment calls to `out.md`.

## Next

- Host session: rerun `late_transfer -> simulated -> terminal-gates` from the
  candidate checkpoints because the discrete weeks codec changes pool content.
- Review the six archived-model owner rulings in `out.md`; do not reinterpret
  their missing ACS predictors through a gate-only threshold/scope change.

# Historical: gate-failed base-pool release lane

## State

Complete on 2026-08-26. Containment, opt-in carriage, and preflight surfacing
are implemented and fully verified. The implementation rejects both redundant
green-pool waivers and any release-manifest receipt that does not exactly match
the pool authenticated by preflight; the preflight's historically required
base/selection inputs and exit semantics are unchanged. No pool or release was
built, no artifact was published, and nothing was pushed.

## Done

- Confirmed the assigned branch and worktree.
- Recorded the v2 charter: close the legacy bare-H5 multispine bypass, add an
  explicit release-build opt-in, carry the authenticated red verdict, and
  surface it in publication preflight without making it an automatic
  publication failure.
- Confirmed that no network, artifact builds, publishing, or pushes are in
  scope.
- Traced the strict manifest loader, current stacked-only terminal-failure
  exception, H5 identity stamp, pool sidecar naming, legacy release arm,
  release manifests, and both preflight output modes.
- Reviewed the salvage branch's final source and test diff line by line. Its
  shared classifier/path-binding/receipt approach closes the bypass without
  changing either loader's contract and was retained with the subsequent
  coherence corrections recorded below.
- Completed the `simulation_ready` / `gate_failed` / loader consumer audit.
  Exact-k remains deliberately strict and head-to-head scoring remains the
  existing authenticated evidence exception.
- Identified report-only downstream caveats: stacked producer metadata still
  names only the k-ladder readiness consumer; red pool publication returns
  status 1 and stops shell chains; ACS-local derivatives keep a donor revision
  but do not project the nested red verdict; generic release consumers tolerate
  and ignore the additive receipt.
- Added a release/preflight-specific authenticated pool loader over the shared
  `require_simulation_ready` seam. The strict simulation-ready and existing
  scoring-only loader contracts remain unchanged.
- Closed the bare-H5 path by detecting either the canonical sibling manifest
  or the H5's stamped pool identity, requiring the sidecar, authenticating the
  publication triple, and binding it to the exact requested H5 path.
- Added `--allow-gate-failed-base-pool` only to the legacy `--base-h5` arm.
  It admits only a current authenticated stacked `gate_failed` pool, rejects a
  green or non-pool use, and never affects the exact-k arm.
- Added the self-contained `base_pool` receipt to both manifests, including
  status/readiness, immutable pool identities, flag use, gates JSON SHA-256,
  failure count/list, and the complete terminal verdict.
- Kept the static preflight inputs mandatory, authenticated its base identically,
  displayed red evidence prominently without changing its exit calculation,
  and required optional release-manifest carriage to match the authenticated
  receipt exactly.
- Hardened carried verdict normalization so nested pass/failure pairs and the
  aggregate verdict must be coherent and a red battery cannot report zero
  failures.
- Passed focused Ruff and the complete builder/H5/preflight test files after
  the containment changes.
- Passed the broader exact-k, launcher, release-contract, and publish-guard
  regression suites.
- Passed repository-wide Ruff and the CI test-group inventory verifier.
- Passed every pytest shard in its own process: build 6,545 passed / 45
  skipped; calibrate 203 passed; data 318 passed / 2 skipped; fit 93 passed;
  frame 295 passed / 36 skipped. Aggregate: 7,454 passed, 83 skipped.
- Confirmed `git diff --check` is clean and that no battery bounds,
  tolerances, plans, or terminal gate logic changed.
- Wrote the complete handoff, consumer audit, manifest schema, verification
  receipts, and commit inventory to `out.md`.

## Next

- Human review and merge of `release-from-gate-failed-pool`.
- Any later artifact operation remains separate: an operator must deliberately
  choose the red-pool flag, then run publication preflight and make the human
  publication decision. This lane performed none of those operations.

## Historical prior lane

The stacked-pool-to-release CD-vintage provenance lane previously maintained
this journal and completed before this work began. It authenticated and
applied household geography after source assembly, carried that authority
through checkpoint and publication identities, published verified CD-vintage
H5 attributes, and reached the unchanged release guard through the shared
fixed/table-aware reader. Its final verification was 7,241 passed, 77 skipped,
with repository-wide Ruff and anti-rot checks green. Full details remain at
commit `2263df36` (the parent of this lane's first journal commit).

The still-earlier PolicyEngine-US 1.819.0 lock-bump lane merged into
`origin/main` at `7b90bb18` on 2026-08-24; its final state remains at commit
`05d254aa` and its detailed receipts remain in the historical section of
`_LANE-NOTES.md`.
