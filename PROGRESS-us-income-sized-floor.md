# PROGRESS — sized export: reconcile the published operands with the published rule

Lane: `us-income-sized-floor-20260906`, a child of the frozen reviewed head
`cbb701214387eada9be786b66bc74a42745e5b8c` (branch
`us-income-sized-review-fixes-20260906`, preserved untouched).

Closing the one new should-address finding from the incremental Fable review of
the sized export, `sized-fixes-fable51-capture.md` **code-1**: the profile
publishes `sampling_fraction`, each cohort's `eligible_households` and each
cohort's realised count, and pins the rule string, but never checks that the
counts are what the rule produces from the operands. A receipt whose operands
contradict its own published rule is accepted today, and the surface draws that
rule as the method behind the counts.

Correction, per stratum, after the existing forced-inclusion and
`realized = requested + added` checks:
`requested == math.floor(fraction * eligible)`.

The consumer mirror lives in `calibration-diagnostics` on
`calibration-income-sized-floor-20260906`.

## State
- [x] Producer: replay the rule per stratum in `_cohort_selection_receipt`
- [x] Producer: regressions for a false eligible operand and a false fraction
- [x] Producer: rename the test that is named for an acceptance and asserts a refusal
- [x] Producer: retire the stale "independent recording" vocabulary in three docstrings
- [x] Doc: say that the rule is replayed, since it no longer only carries operands
- [x] Changelog fragment
- [x] Focused sized suite green

## Done
- `_cohort_selection_receipt` now requires
  `requested == math.floor(fraction * eligible[year])` per stratum, placed after
  the forced-inclusion and `realized = requested + added` checks so every case
  aimed at those still reaches them. New refusal reason: *Recorded cohort
  selection contradicts the recorded selection rule.*
- Three regressions at both invented shapes: a false eligible operand restated
  everywhere it is recorded, a false fraction, and a positive case whose exact
  share is `x.9` so a replay that rounded would refuse a correct run.
- `test_the_producers_own_per_stratum_arithmetic_is_accepted` renamed to
  `test_a_stratum_that_added_units_beyond_its_draw_refuses_by_name`.
- The three "independent recording(s)" docstrings now say *location*, which is
  the published vocabulary. Left alone: `_accounting_columns`' "independent
  witness of the sample size", which is a different and defensible claim -- the
  byte extents are derived from the person count, not a restatement of it.
- Focused sized suite: **135 passed, 0 failed, 0 errors, 0 skipped**, 1.80 s
  (129 before). `ruff check` clean. The 226 unchanged older-exporter neighbours
  were not rerun, as instructed.
- Digests after the change, for the consumer's fixture pins:
  `graph_evidence_income_sized.py` `b781734a…`, its test helper `57604c11…`.
  `graph_evidence_income_v3.py` `0259cbc6…`, `graph_evidence.py` `970d9554…`
  and `us_runtime/graph_asec_income.py` `4d2a9534…` are unchanged, so the
  generator's accepted-head gate still holds.

## Next
- Nothing here. The consumer lane regenerates the fixtures against this head.

## Not in this lane
- No genuine 505 export. Root runs it after review, against its own aggregate
  binding `us-income-sized-export-505hh-root-binding.json`.
- No source, model, dataset or engine execution; no environment change; no push.
- The 226 unchanged older-exporter neighbours are not rerun.
