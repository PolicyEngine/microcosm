# F1 residual brokered-QRF repair — 2026-08-21

## State

- Work is local on `spec-engine-f1` at opening HEAD `4e3b6e35ee0a`; nothing
  has been pushed, no build has run, and `logbook-pending-chain.txt` remains
  untouched.
- This continuation is limited to the inherited deliverable-A brokered-QRF
  suite failure recorded by the r6 close. Landed executor, broker, 72-site
  ledger, frame-restoration, certification-runner, comparator, and host-handoff
  work will not be redone.
- The tracked worktree opened clean. The required `_F1-CHARTER-R6.md` snapshot
  is no longer present: the preceding r6 close records that it was an untracked
  owner instruction copy removed after its operative orders were journaled.
- The exact failing test ids and assertions still need a fresh reproduction at
  this HEAD. The inherited record names the primary-QRF Joblib ambient clock
  refusal and the single-worker cgroup-probe refusal; neither is yet treated as
  the diagnosed root cause in this continuation.

## Done

- Read `CLAUDE.md`, the approved `_F1-CHARTER.md`, all of
  `_F1-LANE-NOTES.md`, `PROGRESS.md`, and `FINAL_REPORT.md`, then inspected
  `git log --oneline -30` as ordered.
- Read the GitNexus debugging workflow. The required live R6 snapshot was
  searched for under the available worktrees and confirmed absent.
- Ran the required dependency sync. The direct command was blocked only by
  the managed user-cache permission; the established writable cache overlay
  completed `uv sync --all-packages --extra us` with 100 packages checked.
- Confirmed the branch and tracked worktree are clean before this journal edit.

## Next

1. Reproduce the brokered-QRF residual exactly and record every failing test id,
   exception, and assertion without changing worker counts or broker policy.
2. Trace the physical QRF draw path against the 72-site seed ledger and broker
   contracts, then decide whether the fix is unambiguous or needs an owner
   ruling.
3. Add a regression, implement only the honest contract-preserving repair, and
   run focused plus full memory-safe suite and Ruff gates before the next code
   commit.

---

# F1 r6 B/C final audit continuation — 2026-08-21

## State

- Work is local on `spec-engine-f1`; nothing has been pushed, and no pool or
  sample build has run in this continuation.
- Deliverables B and C remain complete. Three independent audits found no
  concrete runner, typed-receipt, comparator, synthetic-test, or host-handoff
  gap, so the already-landed implementation was not rewritten.
- Five untracked owner instruction snapshots were the only opening dirt. They
  were read, classified as non-product inputs rather than runner/wiring work,
  and removed file-by-file without stash. Their disposition is journaled in
  `_F1-LANE-NOTES.md`.

## Done

- Read `CLAUDE.md`, `_F1-CHARTER-R5.md`, `_F1-CHARTER-R6.md`, the approved
  `_F1-CHARTER.md` including the D4 two-tier ruling, and the current r6 state
  and handoff in `_F1-LANE-NOTES.md`.
- Inspected status and recent history. The B/C implementation and earlier
  verification are committed; no landed executor, broker, restoration, or
  value-kind work will be redone.
- Completed three independent read-only audits of the runner/typed receipt,
  synthetic comparator coverage, focused tests, and exact host commands; all
  agree the r5 B/C contract is satisfied.
- Replayed the exact B/C contract batch: 72/72 runner, comparator,
  artifact-comparison, and pool-artifact-coverage tests pass. An independent
  81-test receipt collection/comparison batch also passes.
- Repository-wide Ruff, relevant byte compilation, every runner CLI help path,
  generated US bundle, generated coverage, and whitespace checks pass. The US
  spec remains `05edd87390d841c5b444267cd674d8bb15ed518b12577268d2e2c2de82976079`;
  coverage remains 41,911/41,911 fields and 40/40 inventory checks.
- Preserved the already-documented broader-suite boundary without changing
  deliverable-A wiring: the B/C suite is green, while no unqualified
  repository-wide green claim is made for the separate brokered-QRF failure.
- Recorded the final audit in `_F1-LANE-NOTES.md` and appended the requested
  closeout report to `FINAL_REPORT.md`.

## Next

1. Stop without running a pool/sample build, host comparator, kill/resume
   exercise, publication, or push.
2. The high-memory host follows the exact sequential handoff and the owner
   adjudicates the expected fail-closed verdict.

---

# F1 continuation r6 certification-runner progress — 2026-08-20

## State

- Work is local on `spec-engine-f1`; nothing has been pushed, and no pool or
  sample build has run in this continuation.
- The requested scope is deliverables B and C only: revalidate or complete the
  single-build certification runner and four-receipt comparator, then append
  the exact high-memory-host handoff.
- The B/C implementation and handoff are committed through `424c4998`. The
  audited, incomplete post-HEAD wiring WIP was reverted file-by-file without
  stash; the deterministic identities moved by the already-landed wiring and
  its structural operator-map audit are now refreshed at `7265a88a`.
- An independent verification reprise at `b26c79d6` found no B/C source change
  to make: the committed runner, comparator, typed receipt, and host handoff
  satisfy the narrow r6 contract. The verification journal is committed at
  `8a5878c2`, and the required final report is recorded in `FINAL_REPORT.md`.
- `_F1-CHARTER-R2.md` through `_F1-CHARTER-R6.md` are untracked owner-provided
  instruction copies. They are being read as authority but are not product
  deliverables and will not be staged without explicit reason.

## Done

- Read `CLAUDE.md`, `_F1-CHARTER-R5.md`, `_F1-CHARTER-R6.md`, the approved
  `_F1-CHARTER.md` including the D4 two-tier ruling, and the complete current
  `_F1-LANE-NOTES.md` journal.
- Inspected branch status and history. The executor dispatch, ledger RNG,
  narrow/full frame restoration, boolean value-kind preservation, and the
  prior fail-closed runner are committed; no landed work will be redone.
- Started independent read-only audits of the dirty wiring, runner/receipt
  contract, comparator tests, and host-command handoff.
- Audited all 22 dirty tracked files. The interdependent WIP advanced physical
  wiring in intent but was not verifiably committable: the affected batch had
  five failures, including a new executor scope regression, a stale catalog
  count, stale spec pins, and incomplete bundle callback routing; its new
  full-graph fixture also raised a malformed `any()` call. Sixteen touched
  Python files were not Ruff-formatted.
- Reverted those 22 exact tracked paths with `git checkout --`. The five
  untracked owner charter copies were preserved unmodified and uncommitted.
- Revalidated the committed certification runner without changing it. Its 35
  dedicated tests pass; the runner/comparator plus artifact-comparison and
  artifact-coverage batch passes all 72 tests. Focused Ruff, byte-compilation,
  CLI-help, generated-bundle, and generated-coverage checks also pass.
- Confirmed that the committed receipt remains fail-closed and honest at this
  production state: node-reuse evidence is empty/incomplete, final-H5 selector
  inventory is unsupported, and calibration inventory is incomplete. A host
  comparator may therefore emit a well-formed FAIL; B does not imply that the
  separate deliverable-A evidence is complete.
- Refreshed the generated identity attestations that the landed wiring had
  moved without re-pinning. The US bundle is now
  `05edd87390d841c5b444267cd674d8bb15ed518b12577268d2e2c2de82976079`;
  coverage reports 41,911/41,911 fields and 40/40 inventory checks. The
  affected 111-test identity/structural batch is decomposed green (108 passing
  unchanged tests, then the three corrected failures passing in isolation).
- Appended the exact high-memory-host sequence at `424c4998`: documentation-only
  resume gate, constants A/B, bundle A/B, and the four-receipt comparator,
  including six authenticated source pins, strict fresh roots, pipeline exit
  handling, recovery rules, and the 78.91--96.95 GiB historical RSS envelope.
- Final B/C verification is green: 72/72 runner/comparator/artifact tests pass;
  repository-wide Ruff, runner/library byte compilation, CLI help, US bundle
  generation, coverage generation, and whitespace checks pass.
- Repeated that complete B/C gate independently at `b26c79d6`: the 35-test
  synthetic runner/comparator module and the combined 72-test contract batch
  pass unchanged. Repository-wide Ruff, byte compilation, CLI help, generated
  US bundle, generated coverage, and `git diff --check` also pass. The normal
  `uv run` entry point could not initialize its external user cache under the
  managed sandbox, so tests ran through the already-synced `.venv` without
  dependency or source changes.
- Reproduced the already-journaled broader-suite boundary unchanged: the exact
  brokered primary-QRF test fails after Joblib calls `time.sleep(0.01)` inside
  an ambient-clock-denied producer session. No deliverable-A wiring or masking
  test edit was made under this B/C-only order.
- Committed the independent audit/verification journal at `8a5878c2` and
  appended the final closeout report to `FINAL_REPORT.md`.
- Attempted the complete 301-module repository inventory in fresh serial
  processes. The first 26 modules passed 609 tests with one expected skip and
  a 2.016 GiB maximum RSS. Module 27 exposed a committed deliverable-A broker
  failure in `test_brokered_in_process_chain_is_checkpoint_byte_exact`:
  Joblib's parallel wait calls `time.sleep(0.01)`, which the physical broker
  correctly rejects as ambient clock access. Constraining Loky to one worker
  avoids that call but exposes 16 refused `os.stat` probes of
  `/sys/fs/cgroup/cpu.max` and
  `/sys/fs/cgroup/cpu/cpu.cfs_quota_us`, both outside the declared sink. This
  continuation did not change that out-of-scope landed wiring and makes no
  repository-wide green claim.

## Next

1. Stop after this B/C closeout as ordered; do not run the four host builds or
   push.
2. The high-memory host runs the four commands strictly sequentially and the
   owner adjudicates the expected well-formed FAIL receipt.
3. Any repair of the separate brokered-QRF deliverable-A regression requires a
   continuation that authorizes wiring changes; it is outside this B/C-only
   charter.

---

# F1 continuation r4 verification reprise — 2026-08-20

## State

- Work remains local on `spec-engine-f1`; nothing has been pushed, no build has
  run in this resumed continuation, and the logbook pending chain is untouched.
- The committed honest-stop state at `cddb6f18` was re-audited independently.
  The required fetch still fails on sandbox DNS; the subsequent local merge of
  cached `origin/main` reports already up to date. Final #698 remains present
  through `da45dfcd`, together with the superseding 72-site correction.
- The mandated `git fetch origin && git merge origin/main --no-edit` was the
  first command attempted. DNS is unavailable in the sandbox, so fetch failed
  before the chained merge. Cached `origin/main` is stale at `164027e2`.
- Final #698 commit `c4e1eb7f` is already an ancestor of this branch through
  merge `da45dfcd`; the subsequent 72-site correction and generated identity
  pins are present. The authorized US-extra sync completed for 100 packages.
- Deliverable 7 remains under the completed parallel-lane custody documented
  below and in `_F1-LANE-NOTES.md`; this continuation will not modify its
  closure, segment, or dashboard retargeting surfaces.
- Deliverables 5 and 6 remain NOT RUN. Fresh source and plan audits confirm
  that the cold dual-mode D4 prerequisite is still absent and that the known
  1% primary-QRF path violates the per-process 20 GiB RSS bound.
- Current-head validation is complete without a pool build. The serial full
  suite reported 7,189 passed, 74 skipped, and one fixed-60-second subprocess
  timeout under severe host contention. The exact failed test then passed
  unchanged, and its complete 78-test module passed unchanged; repository-wide
  Ruff and whitespace checks also pass. This is decomposed green evidence, not
  a claim that the one-shot invocation exited zero.
- Fresh generated-file checks pass unchanged: the US bundle resolves to
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`,
  and coverage remains 41,911/41,911 fields with 40/40 inventory checks.

## Done

- Attempted the required network sync first and recorded the sandbox failure.
- Verified locally that final #698 is already merged and preserved the existing
  72-site seed ledger over its superseded 53-site version.
- Ran `uv sync --all-packages --extra us` successfully from the isolated cache.
- Read the repository operating rules and the appended D7/main-lane handoff;
  launched independent read-only audits of history and D5/D6 readiness.
- Recomputed the bundle envelope and compiler identities from source. The
  generated US bundle and coverage `--check` gates pass unchanged at 72 sites,
  57 owners, 131 bindings, 41,911/41,911 fields, and 40/40 inventory checks.
- Confirmed the current pins: compiler ABI
  `72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a`,
  seed protocol
  `fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e`,
  seed map
  `f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30`,
  and US spec
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`.
- Re-audited D5/D6 readiness independently. The production path still has no
  `execute_node`, `USPoolKernelAuthorities`, collector, or comparator call;
  no dual-mode build fixture or certification runner exists; exact H5/bank and
  calibration inventories remain unsealed; and historic f001 primary-QRF
  peaks remain 78.91--96.95 GiB.
- Ran the current-head serial suite to completion: 7,189 passed and 74 skipped;
  the sole failure was
  `test_exchange_interrupted_before_cleanup_reclaims_displaced_set`, whose
  child import/fsync path exceeded its fixed 60-second timeout while the host
  was contended. Source tracing found no blocking code path, the exact retry
  passed, and all 78 tests in `test_us_trade_imdb_bulk.py` then passed in
  331.51 seconds without any test or production edit.
- Ran repository-wide Ruff and `git diff --check`; both pass.
- Repeated three independent read-only audits of merge/identity state, D4
  physical wiring, and D5/D6 runner/resource readiness. They agree that no
  re-pin is needed and no charter-valid D5/D6 run may start: physical bundle
  execution and production comparison remain absent, while measured cold 1%
  QRF peaks remain 78.91--96.95 GiB against the 20 GiB process ceiling.

## Next

1. Keep the honest stop at deliverable 5 and commit this refreshed evidence and
   final report; do not fabricate the absent certification evidence files.
2. A future continuation must first complete D4 and a
   behavior-preserving sub-20-GiB memory redesign before any 1% launch.

---

# Historical F1 continuation r4 stop at `e14355c6`

## State

- Work is local on `spec-engine-f1`; nothing has been pushed, no sample build
  has run in this continuation, and the logbook pending chain is untouched.
- Cached `origin/main` and the final #698 branch are merged. The 72-site seed
  ledger supersedes the merged 53-site ledger, and the corrected compiler,
  seed, bundle, loader, and graph identities are pinned.
- Deliverable 7 is complete at `5228cf5c`, with its latest verification receipt
  at `c0a75253`. Its exact-cell code correction was swept into concurrent
  shared-index commit `5875be22`; history was preserved and the boundary is
  documented in the D7 handoff.
- Deliverable 4 is not complete: bundle mode selects typed plan/provenance
  authorities but physical stages still execute through constants, and the
  production artifact collector/comparator is not wired. Deliverables 5 and 6
  therefore cannot be certified honestly.
- The first post-merge full-suite run exposed only three fail-closed audit
  pins for the new authority modules. After the reviewed classification and
  runtime-graph correction, the clean committed-tree rerun collected 7,262
  tests, reached 100%, and exited 0. Repository-wide Ruff is also clean.
- A 1% run is prohibited under the unchanged lane rule: four recorded cold
  primary-QRF peaks are 78.91--96.95 GiB RSS, above the per-process 20 GiB
  ceiling. Additional host RAM alone does not authorize those processes.

## Done

- Attempted the required fetch first; DNS was unavailable, so merged cached
  `origin/main` at `35fb3ed0` and cached final #698 at `da45dfcd` while
  preserving both main's schema/guard/archive work and the 72-site correction.
- Corrected the finalizer structural delta, late-transfer effects, three
  target-balanced cap protocols, two source seed materials, and five
  deterministic hash classifications; regenerated the US bundle and F0
  coverage evidence in `5875be22`.
- Re-pinned the 72-site/57-owner/131-binding seed protocol and all affected
  country, loader, graph, and compiled-map identities. Coverage now reports
  41,911/41,911 fields and 40/40 inventory checks.
- Verified the correction suite (101 passed), merged/D7 suite (229 passed), D7
  exact-cell suite (70 passed), generated-bundle check, coverage check, focused
  spine audit (3 passed), complete spine-blindness module (495 passed), Ruff,
  and whitespace checks at their respective checkpoints.
- Verified a clean full-suite rerun after the audit correction: 7,262 tests
  collected, 100% reached, exit 0, with expected skips only. After the
  mechanical import-order correction, full Ruff and the complete pool-tool
  test module both exit 0.
- Confirmed read-only that no valid owner 1%/25% command exists at this state:
  the physical executor and sealed two-tier production comparison remain
  preconditions, and the 1% path must first be made to stay below 20 GiB RSS
  per process unless the owner explicitly changes that constraint.
- Recorded the exact old/interim/final identity vectors, clean validation
  receipts, D4 blockers, D5/D6 NOT RUN status, and owner preconditions in the
  lane journal, rollout status, and final report. No certification evidence
  files were fabricated.

## Next

1. Stop this continuation at the honest D4 boundary with the tracked tree
   green and the final handoff committed.
2. A future authorized continuation must wire the 38 compiled producer nodes
   through the executor/brokers, seal exact artifact member inventories, add
   calibration ownership, and pass the cold D4 fixture gate.
3. Only after D4 passes and every 1% process is bounded below 20 GiB RSS (or
   the owner explicitly changes that constraint) may D5/D6 run and an exact
   25% command and expected comparison vector be issued.

## F1 D7 split-out progress — 2026-08-20 audit at `c0a75253`

This historical section records the split-out state at `c0a75253`. It
supersedes only earlier D7 status text and does not update the later main-lane
D4--D6/D8 state above.

### State

- Deliverable 7 is complete at `5228cf5c`: every active production-shaped
  derived-closure, segment, and lineage-dashboard consumer is compiler-fed,
  and exact-cell authority preserves disjoint graph/family atoms.
- Concurrent main-lane commit `12df8c45` advanced the shared branch after the
  D7 handoff. It changed no D7 source or test file.
- The D7 working tree is clean apart from three pre-existing untracked charter
  drafts. Nothing was pushed and no pool build or sample rung ran.

### Done

- Re-audited tracked tests/tools and found no active reference to the held
  lineage YAML, 392-column inventory, old closure helper, or authored-class
  dashboard fields. Two independent read-only reviews approved the compiler
  boundary and exact-cell closure.
- Re-ran all four D7 modules: 70 passed in 420.42 seconds. Two separate hash
  seeds emitted byte-identical dashboard JSON with SHA-256
  `03c08eb0e59b4ce2fa0d2ffe2bcf62e503005569b1171520a46068dec99ef7df`.
- Ran every test shard serially. Calibrate, data, fit, and frame completed with
  865 passed and 37 skipped. The build shard completed with 6,322 passed, 37
  skipped, and three spine-blindness failures collected before concurrent
  commit `12df8c45` classified the new main-lane runtime modules. The exact
  three tests pass at current HEAD (3 passed in 4.42 seconds); the main lane
  separately records all 495 tests in that module passing.
- Scoped Ruff check/format and the D7 diff whitespace check pass. Repository-
  wide Ruff still reports one main-lane import-order finding at
  `tools/build_us_multispine_pool.py:1157`; D7 does not alter that D5/D6/D8
  driver.
- Validation was serialized with no build or sample workload. The sandbox
  blocks both `ps` and `/usr/bin/time -l` resource reporting, so an exact RSS
  peak is unavailable and is not asserted.

### Next

- No D7 implementation work remains. Preserve the compiler-derived seam and
  exact-cell regression when the main lane changes compiler outputs.
- The main lane owns its clean one-shot full-suite rerun, the unrelated Ruff
  import-order finding, physical executor work, and certification gates.

## F1 D7 split-out verification addendum — 2026-08-20

This is the latest D7 `state`/`done`/`next` receipt. It does not modify the
resumed main-lane state above.

### State

- D7 remains complete. No D7 implementation or test file changed between
  `5228cf5c` and parent HEAD `4deb8fb8`; later main-lane commits updated only
  their driver import order and journals/docs.
- The scoped output and coordination notes now record both the clean test
  receipts and the resource-limit violation discovered by measured rerun.

### Done

- After D7 report commit `c0a75253`, all four D7 modules passed again: 70
  passed in 350.29 seconds. Darwin `ru_maxrss` measured 4,231,233,536 bytes
  (3.94 GiB), below the split-out's 15 GiB limit.
- A fresh build-shard run completed with 6,325 passed and 37 expected skips in
  6,820.57 seconds. Together with the unchanged four other serial shard
  receipts, observed suite results are 7,190 passed and 74 expected skips.
  Main-lane journals independently record their clean committed-tree full-
  suite result.
- The build-shard wrapper measured 30,950,326,272 bytes (28.82 GiB) peak RSS.
  That exceeded the user's 15 GiB ceiling even though it was a test run, not a
  pool/sample build. Heavy testing stopped immediately after the measurement;
  this is recorded as an operating-order violation, not represented as
  compliant.
- Current-HEAD repository-wide Ruff, scoped D7 format, held-fixture searches,
  and whitespace checks pass. Commit `030c0613` fixed the prior unrelated
  main-lane import-order finding.

### Next

- No D7 code work remains. Do not repeat the monolithic build shard under a
  15 GiB ceiling; any future verification must use measured, smaller process
  groups that remain below the limit.
- No push, pool build, sample rung, restricted-data access, or publication is
  authorized or pending from D7.
