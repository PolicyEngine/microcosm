# Round 9 progress: post-PUF dtype-family integrity

## State

The smoke-r6 mechanism is adjudicated. The first target in late-transfer group
`person/puf_tax_itemization__batch_4`,
`farm_rent_income_would_be_qualified`, is canonically `boolean_incidence` and
its transfer prediction is correctly boolean. Primary-PUF finalization wrongly
materializes its target Series as `float64`; pandas then refuses the boolean
merge. No implementation change has been made yet.

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

## Next

- Complete the all-late-write audit and add a failing registry-driven
  regression that covers the full 131-target dtype-family authority.
- Fix primary-PUF boolean materialization at its owning layer, with explicit
  validation rather than a permissive or silent coercion.
- Run the focused and complete requested proof matrix.
- Record a gradeable smoke-r7 prediction and final verdict.
