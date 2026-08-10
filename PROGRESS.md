# Progress: microcosm #653

## State

The failure mechanism, complete late-producer/source-input inventory, and
36-node executable DAG are implemented and documented on
`tail-stratum-support-652`, based on the three preserved #652 commits. The
checkout was clean at the start and was three commits ahead of the locally
available `origin/main` (`e9a352ca`). No fetch was performed because this task
forbids network access. A shared-ref update outside this worktree has since made
Git report the branch behind by one; the task remains on its required checkout
without rebasing, resetting, or shelving. Focused verification is green; the
exact #583 shard and every foreground workspace chunk were green, but final
independent review found additional doctrine gaps. Implementation is reopened:
optional absence must not excuse invalid numerics, every transfer's cross-grain
validation inputs must be declared, and persisted readiness/source/transfer
proofs need content binding. Final report assembly is paused until those gaps
and the complete proof rerun are closed.

## Done

- Read the repository agent guide and the applicable debugging, data-pipeline,
  and development-standard instructions.
- Confirmed the active branch and preserved #652 commit chain:
  `c2bc06fe`, `9f184a07`, and `54d2dee6`.
- Located the late-transfer, post-clone source-completion, adult-care, SSTB,
  operator-boundary, checkpoint, and ordering-document surfaces to audit.
- Confirmed the executable order is PUF pass, then all post-clone source
  operators, then the 70-target late transfer. Adult care strictly consumes
  `sstb_self_employment_income_before_lsr` before that transfer can fill it.
- Reconstructed the failing checkpoint population: 342,732 ACS-origin rows and
  43,260 ASEC-origin rows per clone role. All 43,260 failing cells are on
  ASEC-origin clone-0 recipients; the issue's ACS-origin parenthetical is not
  supported by the saved checkpoint.
- Audited all 16 post-clone source operators, the primary PUF/tail producer,
  and all 19 canonical late-transfer groups. The direct scheduling edges are
  PUF/SSTB transfer to adult care, PUF/tuition transfer to education,
  pregnancy to WIC, and childcare to adult care, plus the declared PUF-role
  predictor and producer-to-transfer edges.
- Confirmed education currently hides its tuition dependency with
  `fillna(0.0)` and incorrectly claims the tuition passthrough as a source
  output. The DAG must make tuition PUF-only, transfer it before education,
  and make nonfinite tuition fail closed.
- Added red regressions for unfilled-input refusal before callback invocation,
  a deterministic named synthetic cycle, and byte-stable topology under
  reversed registry iteration. The focused test initially failed at collection
  because the deliberately specified DAG module did not yet exist.
- Implemented the pure producer-DAG core. It canonicalizes contracts and
  edges, derives lexically stable Kahn waves, reports a deterministic DFS cycle
  path, hashes canonical JSON bytes, and fences callbacks on exact filled-input
  or declared-absence evidence. Its three doctrine regressions now pass.
- Made qualified tuition a strict PUF-owned education input: nonnumeric,
  nonfinite, or negative tuition/assistance now fails; tuition is preserved
  byte-for-byte; education owns only assistance plus five AOTC facts. The
  source-producer surface is now 29 targets with two PUF overlaps, and 30
  education/partition regressions pass.
- Declared and import-validated the production late graph: one primary-PUF
  producer, all 16 post-clone source producers with full structured kernel
  input inventories, and the exact 19 bounded late-transfer groups covering
  70 targets. Its derived edges include pregnancy to WIC, childcare and
  SSTB batch 5 to adult care, and tuition batch 2 to education. Seven graph
  and registry doctrine regressions pass, including reconstruction under
  reversed registry iteration.
- Tightened every descriptive inventory into an executable contract gate:
  primary PUF declares 15 effective requirements and all 65 outputs; all 16
  source and 19 transfer nodes declare required alternatives and named
  tolerated-absence receipts for optional availability predictors. The full
  graph now derives 48 real edges, including primary-PUF dependencies into ten
  source operators and all 19 transfer groups. Nine DAG regressions pass.
- Split the post-clone source chain into a guarded single-producer entrypoint
  and an exact 16-receipt finalizer. The compatibility entrypoint now uses the
  same narrow API; deferred source inputs materialize only once after complete
  execution. All 54 multispine-pool tests pass.
- Integrated primary PUF/tail, all 16 source operators, and all 19 bounded
  transfers into one executable schedule. Clone attachment is now an explicit
  primary-PUF output and prerequisite of every post-clone source, so the first
  wave contains only `primary_puf_qrf`; the graph has 36 nodes, 54 edges, and
  wave sizes `(1, 17, 14, 3, 1)`.
- Put the primary PUF callback behind the same readiness fence as every other
  producer. Its donor and checkpoint are carried as explicit available-input
  receipts; optional sidecars remain counted declared absences, never zero
  fills. Finite-numeric input kind is now contract data, so object-backed
  `inf` or nonnumeric late inputs fail at the DAG boundary.
- Bound the complete DAG receipt to stacked authority v8, pool and stacked
  checkpoint materializers v4/v8, late-registry schema v2, and companion pool
  manifest schema v5. Cold execution, checkpoint emission/resume, manifest
  construction, publication, and schema-5 consumer loading all validate the
  exact 36-row execution, 16-source finalization, 19 transfer groups, and
  source/transfer aliases.
- Added executor-level and publication regressions for the derived order,
  single source finalization, batch-5-before-adult-care, nonfinite object
  numerics, forged execution rows, forged derived order, and missing schema-5
  DAG proof. The combined DAG, stacked, tool, and H5 suites pass after an
  independent review exposed and the implementation closed the hidden
  clone-attachment and unauthenticated-receipt gaps.
- Published the full primary-PUF, 16-source, and common/per-group transfer input
  inventories in the operator-ordering doctrine, together with all 54 edges,
  the five derived waves, schedule/payload hashes, readiness rule, cycle rule,
  new schema versions, and corrected 43-PUF/29-source/two-overlap accounting.
  Extended the #652 changelog fragment so the stacked tail and late-DAG fixes
  ship as one local PR train.
- A documentation-to-registry audit found that executable alternative columns
  dropped their declared `finite_numeric` kind during contract construction.
  Preserved the kind in the production registry and added a registry-level
  regression covering primary PUF, adult-care SSTB, education tuition, and an
  optional transfer predictor; the focused DAG file now passes all 10 tests.
- Replaced a Python/Pandas-version-specific pickle golden in the #652 tail
  preservation regression with an exact same-runtime comparison against the
  pre-#652 allocation path. The assignment SHA remains pinned, and the live
  pre-fix path and new all-adequate path have identical tables, dtypes, weights,
  strata, mass log, and frame digest.
- Ran the focused late-DAG, stacked, tool, H5, tail, adult-care, education, and
  transfer suites in the foreground: exactly 518 tests passed. The only golden
  changed by the finite-kind fix was the expected authority-bound legacy
  manifest digest; pool H5 and agreement bytes remained unchanged. Targeted
  Ruff check, format check, and diff check pass.
- Ran the #583 source-spine-blindness shard in the foreground. Its first pass
  fail-closed on the two new modules, so classified the pure scheduler as a
  reviewed non-operator module, classified the data-only registry as a narrow
  provenance owner, required both in the pool import graph, and moved the
  pinned graph size from 61 to 63. The complete shard then passed exactly 495
  tests.
- Ran the full workspace in eight non-overlapping foreground chunks. Exact
  results were: 795 passed/36 skipped; 1,446/26; 1,161/1; 460/0; 653/2;
  480/0; 495/0; and 324/1. Total: 5,814 passed, 66 skipped, 5,880 collected,
  with zero failures or errors. JUnit receipts independently carry those
  counts and prove the partition covers all 190 build test files plus every
  frame, fit, calibrate, and data test.
- Ran repository-wide `ruff check .`: pass. Repository-wide
  `ruff format --check .` reports 30 pre-existing files outside this branch's
  diff; none was rewritten. The format check over all 19 Python files changed
  since the task base passes, as do both the branch and worktree
  `git diff --check` gates. The worktree is clean.
- Final independent review found that adult-care support role should require
  clone index plus channel; several callback-numeric inputs were declared only
  nonnull; optional absence receipts currently conflate missing cells with
  invalid nonfinite cells; transfer wrappers consume undeclared all-entity
  cross-grain provenance; and schema-5 execution rows trust persisted counts
  rather than a hash-linked live transition proof. Reopened the implementation
  rather than issuing a premature ready verdict.
- Added production-registry regressions that require adult care's clone index
  plus channel as one all-of support-role input and require every value passed
  to a strict numeric callback path to carry `finite_numeric` contract
  semantics. The focused DAG suite now fails only on those deliberately red
  assertions (the first failure masks two additional source-kind assertions).
- Corrected those registry contracts and bumped the late-registry schema to
  v3: adult care now requires clone index and channel together; `PEDISDRS`,
  full-time-college status, and raw `ED_VAL` are finite-numeric; and every
  component or ACS aggregate in the transfer social-security, retirement, and
  investment alternatives is finite-numeric. All 12 focused DAG tests pass;
  the contract-only change preserves 54 edges and wave sizes `(1, 17, 14, 3,
  1)` while changing the schedule/payload identity as intended.
- Added red readiness regressions proving that a declared-absence receipt may
  authorize missing optional cells but never nonnumeric or nonfinite values.
  One exercises the generic fence and one poisons a canonical adult-care
  transfer predictor; both fail because the implementation does not yet expose
  separate missing and invalid counts.
- Split readiness into independent missing-row and invalid-value maps. Missing
  optional cells alone can mint the named absence receipt; present nonnumeric,
  infinite, or nonfinite values always refuse the callback and name both the
  logical input and declared producing stage. The generic and canonical
  transfer regressions pass, as does the real 36-node executor regression.
- Made both readiness maps exact contract surfaces: omitting a declared input
  can no longer default to a false zero, and extra inputs also fail with a
  canonical diagnostic. The focused DAG file now passes all 14 regressions.

## Next

- Add red production-level regressions for the final-review findings, implement
  the strengthened input/readiness/transition contracts, and commit each
  coherent step.
- Rerun focused, exact #583, all foreground chunks, and Ruff/diff gates after
  the final fixes.
- Write the final gradeable mechanism/edge/fix/proof report to the requested
  output file and stdout, commit it, and leave the worktree clean.
