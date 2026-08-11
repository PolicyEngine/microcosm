# Round 9 progress: post-PUF dtype-family integrity

## State

The smoke-r6 mechanism is fixed at both affected materialization seams.
Primary-PUF finalization now preserves all eight canonical QBI outputs as
nullable booleans, and the shared source-output merge preserves twelve
physical-boolean callback outputs as nullable booleans instead of widening
CPS-only alignments to `object`. A registry-driven guard now rejects every
late callback output whose physical dtype disagrees with its declared metric
family before the DAG records the producer. Local proof is complete on the
final implementation: focused 45/45, exact PR #583 495/495, and all 225
workspace test files at 5,914 passed plus 68 skipped with zero failures or
errors. Repository-wide ruff, scoped format, and range-diff checks pass. No
build was run; the only next step is the external 1% smoke-r7 certification.

## Done

- Read `CLAUDE.md` and the applicable PolicyEngine data and development
  standards guidance.
- Confirmed the requested branch and exact starting commit with a clean
  worktree.
- Compared the checkout with the locally cached `origin/main`: the branch is
  81 commits ahead and 15 behind. The requested train checkout is being
  preserved because this round explicitly targets PR #660 and forbids network
  access.
- Established the required workflow: mechanism before fix, registry-declared
  metric families as the dtype authority, regression coverage across every
  late-stage write, no build execution, and a commit after each coherent step.
- Read the full launcher traceback and chained error/logbook receipts. The
  exact path is `run_stacked_late_producer_dag` ->
  `transfer_stacked_post_puf_group` -> `transfer_acs_inputs` ->
  `_fill_recipient_nulls`, whose positional assignment rejects booleans into a
  `float64` Series.
- Identified the failing group and first target from the banked execution
  order. All eight batch-4 prediction banks are written before merge begins;
  the declared first merge is `farm_rent_income_would_be_qualified`.
- Proved from the r6 artifacts that the assembled 38,604-person checkpoint has
  none of the eight batch-4 QBI columns. The dtype is therefore not restored
  from that checkpoint.
- Proved the late target bank binds the exact target and has 80,395 recipient
  rows, 38,604 finite predictions, and support exactly `{0, 1}`. The prediction
  decoder correctly returns boolean values.
- Traced the `float64` materialization to primary-PUF finalization, which calls
  `_ensure_float_output_column` for every person output and writes boolean
  placements as numeric 0.0/1.0. This is not a scope mask misapplied as values
  and not a boolean producer targeting a monetary column.
- Confirmed the canonical metric registry declares the target
  `boolean_incidence` and retains the exact authority split: 79 monetary, 48
  boolean, and 4 categorical targets.
- Added a registry-derived primary-PUF materialization regression covering all
  eight canonical QBI boolean outputs under the stacked preserve-nulls
  doctrine. It proves non-owned cells remain null and requires a boolean
  physical dtype on every output.
- Ran the new test against the unfixed implementation and captured the
  expected RED result: the first output,
  `estate_income_would_be_qualified`, is `float64` rather than boolean.
- Audited all 163 registered late-write occurrences, representing 90 unique
  targets: 120 monetary, 37 boolean, and 6 categorical occurrences; 67
  monetary, 20 boolean, and 3 categorical unique targets.
- Found the same mismatch class in the shared source-output merge. Nine of
  twelve source-produced booleans had no incumbent column and therefore
  widened to `object` when aligned across non-source rows; three existing
  gap-fill booleans happened to retain boolean storage.
- Fixed primary-PUF boolean materialization at its owning layer. Existing
  observed non-booleans, including numeric 0/1 values, are rejected rather
  than silently coerced. The retiring legacy zero-fill path and its protocol-5
  byte pin remain unchanged.
- Fixed the shared source merge to preserve physical booleans as pandas
  nullable booleans across unowned rows. Numeric incumbents and non-boolean
  callback values targeting boolean-materialized columns fail closed.
- Added a registry-authoritative late callback guard. It checks all registered
  callback outputs before receipts are recorded and permits the physical
  representations required by the 79/48/4 metric families.
- Extended tail-transfer coverage to prove nullable QBI booleans survive the
  per-stratum tail clone copy without rewriting clone-0 absence.
- Passed the post-format focused mechanism suite: 11 passed, with only two
  pre-existing DataFrame-fragmentation warnings. No build was run.
- Passed the complete focused proof: 40 passed, 0 skipped, 0 failed, and 0
  errors. This includes all 29 late-producer DAG tests plus the 11 owning-seam,
  registry, legacy, executor, QBI, source-merge, and tail-clone regressions.
- Passed the exact PR #583 spine-blindness proof: 495 passed, 0 skipped, 0
  failed, and 0 errors.
- Partitioned all 225 workspace test files exactly once into chunks of 32, 32,
  32, 32, 32, 29, 1, and 35 files.
- The first chunk-3 run found one stale integration expectation in adult-care
  coverage: it asserted that a CPS-only physical boolean widened to `object`.
  Updated it to require `BooleanDtype`, matching the canonical family and the
  shared-merge regression, and passed the corrected test in isolation.
- Passed corrected chunk 3: 779 passed and 4 skipped, with no failures or
  errors.
- Passed full-workspace chunks 1, 2, 5, 6, 7, and 8 respectively at 711/2,
  562/21, 773/2, 768/1, 495/0, and 795/36 passed/skipped, with no failures or
  errors. Chunk 6 includes three new stacked-spine cases relative to the prior
  baseline; chunk 7 independently repeats the exact 495-test #583 proof.
- Chunk 4 exposed one downstream consequence across four tests. The first
  trigger is monetary `qualified_tuition_expenses`, whose allocation basis is
  canonical boolean `is_full_time_college_student`; three later QBI amounts
  similarly use `business_is_sstb`. The generic numeric path attempted to fill
  nullable boolean incidence with floating `0.0`.
- Declared the exact four monetary-output/boolean-basis pairs and verified both
  sides against the canonical metric registry. The shared allocator now maps
  boolean incidence to a transient 0/1 vector without changing stored dtype;
  it rejects numeric 0/1, mixed, textual, and nonfinite family drift. Only the
  explicitly selected retiring legacy policy accepts numeric incidence, and
  then only exact finite `{0, 1}` support.
- Added direct tuition and QBI placement regressions proving the flagged person
  receives each monetary total and the basis remains `BooleanDtype`. Applied
  the same strict normalizer to both allocation directions; its focused five
  tests pass.
- Updated the existing multispine producer test to require that object-backed
  assembled `is_female` values become nullable booleans before production
  transfer. All four formerly failing chunk-4 tests now pass in a focused run.
- Re-ran the expanded focused proof on the final implementation: 45 passed, 0
  skipped, 0 failed, and 0 errors.
- Re-ran PR #583 on the final implementation: exactly 495 passed, 0 skipped, 0
  failed, and 0 errors.
- Re-ran every one of the 225 workspace test files on the final implementation
  in exact chunks of 32, 32, 32, 32, 32, 29, 1, and 35 files. Exact
  passed/skipped counts were 711/2, 562/21, 779/4, 1,031/2, 773/2, 768/1,
  495/0, and 795/36: aggregate 5,914 passed and 68 skipped, with zero failures
  or errors among 5,982 collected cases.
- Passed repository-wide `ruff check .`, `ruff format --check` over all nine
  Python files changed since `14de14ce`, and `git diff --check
  14de14ce..HEAD`. The worktree was clean before this progress update.
- Gradeable smoke-r7 prediction: the first batch-4 merge for
  `farm_rent_income_would_be_qualified` will assign boolean predictions into a
  nullable-boolean incumbent instead of `float64`. All eight primary QBI flags
  and all twelve source-produced flags will remain physical booleans; the nine
  formerly object-widened source flags will use `BooleanDtype`. The registry
  guard should accept all 20 unique late boolean targets and all 163
  registered late-write occurrences. Smoke r7 should complete all 38
  late-producer DAG nodes and reach at least phase `transferred`; absent an
  unrelated later defect, it should continue through `derived`, `seeded`,
  `simulated`, `terminal_gates`, `terminal_receipt_written`, and
  `publication_completed`.

## Next

- Run the external 1% smoke-r7 certification using the committed train tip.
- Confirm the logbook row includes `transferred` (minimum prediction) and does
  not contain a post-PUF dtype-family assignment error.
