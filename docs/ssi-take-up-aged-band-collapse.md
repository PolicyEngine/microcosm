# SSI take-up aged-band collapse

## Root cause

Build N's SSI goals are already age-banded. PR #476 compiles the three SSA
recipient rows as ordinary age-sliced calibration targets; PR #477 added the
builder lookup that also passes them to `ssi_take_up.py` as priors. The generic
`take_up.py` is unrelated and only seeds TANF and EITC. The regression comes
from PR #477's one-shot assignment lifecycle, not from a national-only goal or
a binding swap cap.

The stage computes each Bernoulli threshold on pre-calibration weights as
`band target / weighted(uncapped_ssi > 0)`, freezes the flags, and lets the
5,672-target calibration/refit change the weights. Build N fixed the 65+ threshold
at 5.9057 percent on 40.336 million weighted candidates. On release weights the
same band had 3.995 million candidate capacity, so its target-implied threshold
was 59.6213 percent; the frozen threshold nevertheless remained in force. With
reporter anchors and the weighted hash realization, 8.4364 percent of all
people age 65+ were flag-true and only 0.984 million candidate-weight recipients
remained in the builder's 2024-12 release-weight diagnostic, versus the 2.382
million target. Issue #507's separate 0.94 million result is the TY2026
PolicyEngine baseline calculated from the exported dataset.

PR #431 did not truncate this assignment: its cap only rejected an excessive
post-refit stale-to-fresh national delta. PR #448 only recorded that delta's
per-pass trajectory, and PR #446 added optional selection-mass targets (Build N
used one for Keogh distributions, not SSI). PR #477 deleted the reconcile loop,
cap, and trajectory before Build N and explicitly made every age-band count
miss scorecard-only.

## Target contract

The pinned source is SSA SSI Monthly Statistics, December 2024, Table 1,
"Number of recipients," row "Total with—Federal payment" (Supplemental
Security Record, 100-percent data):

- Under 18: 1,001,922
- Ages 18–64: 3,905,779
- Ages 65+: 2,382,142
- National sum: 7,289,843

Source: https://www.ssa.gov/policy/docs/statcomps/ssi_monthly/2024-12/table01.html

These are the exact cells in the current source contract, not exact versions of
issue #507's requested approximately 0.98 million / 3.99 million / 2.42 million
benchmarks. Those rounded values do not reproduce one row of the pinned table,
so their exact cells and source row must be adjudicated before they become named
constants. Neither set may be mixed with the separate national
`ssi_recipients` registry target of 7,404,820: PR #430 rescaled the current age
shares to that total because the two official measures differ by 114,977. A
future implementation must choose one coherent target system and define its
national goal as the sum of the three enforced band goals.

## Why the fix does not belong only in seeding

A pre-calibration greedy match can drift when household weights change. A
post-calibration flag rewrite would then stale SSI-dependent ACA, Medicaid,
SNAP, other-health inputs, the materialized target matrix, and calibration
diagnostics. The minimal correct production lifecycle is therefore a bounded
post-selection loop on fixed support: assign against current weights, gate each
band with an explicit tolerance, replay every SSI-dependent stage, rematerialize
targets, refit, and remeasure. The deleted loop replayed ACA, Medicaid, and
other-health; a replacement must additionally verify and replay state SNAP
assignment because it computes live SSI-sensitive eligibility. Swap caps and
pass history must be per band so an aged shortfall cannot cancel a working-age
excess in a national delta.

The all-band gate is not currently feasible: Build N's under-18 candidate
capacity is only 177,582 against the 1,001,922 target. That modeled eligibility
or candidate-domain defect belongs to the child SSI disability work tracked in
#453; it does not by itself prove the records are absent from support. Until it
is fixed, the pipeline cannot truthfully reconcile all three bands; silently
treating saturation as success would violate the requested caseload gate.
