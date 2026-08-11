# Round 9 progress: post-PUF dtype-family integrity

## State

The smoke-r6 mechanism is fixed at both affected materialization seams.
Primary-PUF finalization now preserves all eight canonical QBI outputs as
nullable booleans, and the shared source-output merge preserves twelve
physical-boolean callback outputs as nullable booleans instead of widening
CPS-only alignments to `object`. A registry-driven guard now rejects every
late callback output whose physical dtype disagrees with its declared metric
family before the DAG records the producer. The 40-test focused proof and the
exact 495-test PR #583 spine-blindness proof both pass with zero skips,
failures, or errors. The full-workspace chunk matrix is in progress. Its first
chunk-3 pass exposed one stale adult-care assertion that required the former
`object` widening; the test now requires nullable boolean storage and passes
in isolation. Chunk 3 will be rerun in full.

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

## Next

- Run all 225 workspace test files in eight exact-count chunks, followed by
  rerunning corrected chunk 3, followed by ruff, format, and diff checks.
- Commit the full proof receipts and clean-worktree handoff state.
- Record a gradeable smoke-r7 prediction and final verdict.
