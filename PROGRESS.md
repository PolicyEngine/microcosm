# Pregnancy / prior-year defect lane (issues #798 and #799)

## State

Resumed on 2026-08-27 on branch `pregnancy-prioryear-defects`, based on
`stacked-release-gate-alignment` at `606cbd69`. The root-cause findings are
adopted and implementation is in progress for the nonfemale-pregnancy producer
defect and the owner-approved rung-aware prior-year availability release floor.
Both fixes are implemented and their focused runtime suites are green; the
source-attested spec/coverage repin and full repository verification are in
progress. The authored 0.05 floor, upper bound, all other bands, thresholds,
seeds, and batteries remain unchanged. This lane will not build, publish, or
push pool or release artifacts.

## Done

- Read `CLAUDE.md` and the GitNexus debugging workflow.
- Confirmed the assigned branch starts clean at `606cbd69`.
- Recorded the required source fix, structural refusal guard and receipt,
  checkpoint-identity review, focused regressions, real-pool decompositions,
  changelog, repository Ruff, and independent full-shard pytest boundary.
- Recorded Task 2 as diagnosis-only unless the evidence proves an unambiguous
  transfer defect.
- Rebuilt a transient local GitNexus graph offline and queried pregnancy/WIC
  execution paths. Registration alone failed because the sandbox forbids the
  CLI's global `~/.gitnexus` registry; the usable worktree-local index remains
  untracked and will be removed before handoff.
- Inspected all 1,970,973 person rows in the supplied 25% pool. There are 108
  `is_pregnant=true` nonfemale rows, all on physical ACS records: 45 clone 0,
  61 clone 1, and 2 clone 2. ASEC has zero; sex/channel/clone assembly is
  internally consistent.
- Localized the pregnancy defect to the ACS QRF path. The ASEC producer hard-
  conditions its stable draw on female ages 15--44, while ACS transfer treats
  sex and age only as soft predictors, models physical clone rows separately,
  and has no pregnancy-domain postcondition. The pool has 11,287 ACS source
  people whose clones disagree on pregnancy and zero ASEC disagreements.
- Proved the prior-year shortfall is not ACS dilution or an ACS transfer hole.
  Physical ASEC and ACS are both about 4.3% available because assembly samples
  each raw ASEC year independently before the adjacent-year `PERIDNUM` join.
  Of 18,518 sampled current rows that match the intact full predecessor files,
  only 4,724 retain a predecessor after sampled-to-sampled joining: weighted
  match survival is 25.4117%, the expected 25% rung effect. Full pooled ASEC
  availability is 16.9147%, and selected current rows joined to intact prior
  files are 16.9541%.
- Received the owner ruling that assembly-before-join sampling is the accepted
  mechanism verdict. The prior-year release gate may scale only its availability
  floor by the assembly's sampled-to-sampled match-survival factor, recorded in
  or derived from the pool manifest. Rung 1.0 must remain byte-identical to the
  existing gate; the authored 0.05 constant and upper bound do not change.
- Implemented the owner-approved prior-year gate policy in `f5284a07`. An
  authenticated production stacked manifest now restores its version-4 sample
  receipt to the loaded frame; the availability gate scales only the authored
  lower floor by that rung, conditionally receipts the factor/applied floor,
  and leaves full-rung output byte-identical. Legacy/no-rung frames retain the
  original gate. The prior-year and H5 focused suites pass (23 and 60 tests),
  focused Ruff and `git diff --check` pass, and the real candidate-25 manifest
  validates at factor 0.25.
- Implemented pregnancy's hard female-age-15--44 policy before any requested
  pregnancy QRF. The transfer validates donor and recipient structure up
  front, draws only one eligible clone-0 representative per assembled source
  person, fans that value to missing clones, assigns structural false to
  ineligible missing rows, and refuses preexisting/final domain or clone
  disagreement with explicit counts.
- Added a sealed structural receipt with disjoint QRF, clone-fanout,
  preexisting-value-fanout, and ineligible-false accounting. Production receipt
  validation authenticates the policy digest, zero-violation postconditions,
  source-person topology, and exact equality to the transferred row count.
- Bound the policy into the transfer execution contract used by late-stage
  checkpoint/target-bank identity, isolated pregnancy from unrelated bounded
  QRF families, declared its structural source-person input, and regenerated
  the authored US imputation spec/schema projection.
- Closed two adversarial restart cases: partial ineligible clone surfaces no
  longer double-count receipt categories, and a complete pregnancy surface is
  still preflighted and carries a zero-imputation structural proof even when a
  different requested family remains active.
- Added source, transfer, gate, receipt, identity, all-ineligible, clone-fanout,
  mixed-active, and stacked-execution regressions. The complete pregnancy and
  ACS-transfer files, the complete stacked-spine file before the final
  preflight refactor, and focused post-refactor tests pass; focused Ruff and
  `git diff --check` pass.
- Extended the real-pool pregnancy decomposition: the 108 pregnant nonfemale
  rows are joined by 58 pregnant female rows outside ages 15--44, for 166 hard-
  domain violations, all on ACS and all isolated to one clone. ASEC has zero.

## Next

- Complete the source-attested spec/coverage repin and rerun all directly
  affected spec, transfer, and stacked suites.
- Commit the coherent pregnancy implementation and journal state.
- Run repository Ruff, generated/coverage/inventory checks, and all five pytest
  shards in independent processes.
- Write and commit the complete handoff to `out.md`.

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
