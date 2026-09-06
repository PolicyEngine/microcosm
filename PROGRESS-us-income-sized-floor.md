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
- [ ] Producer: replay the rule per stratum in `_cohort_selection_receipt`
- [ ] Producer: regressions for a false eligible operand and a false fraction
- [ ] Producer: rename the test that is named for an acceptance and asserts a refusal
- [ ] Producer: retire the stale "independent recording" vocabulary in three docstrings
- [ ] Doc: say that the rule is replayed, since it no longer only carries operands
- [ ] Changelog fragment
- [ ] Focused sized suite green

## Done
- Opened the lane. Nothing else yet.

## Next
- The producer edit, then the tests, then the doc.

## Not in this lane
- No genuine 505 export. Root runs it after review, against its own aggregate
  binding `us-income-sized-export-505hh-root-binding.json`.
- No source, model, dataset or engine execution; no environment change; no push.
- The 226 unchanged older-exporter neighbours are not rerun.
