# UK local cross-grain reconciliation (#802)

The UK local build applies one standing rule before constructing a calibration
matrix: when the same measurement concept is bound at more than one grain, the
highest bound grain wins in the order country, constituency, then local
authority. The shared operator in `microcosm.build.cross_grain` changes only
the lower-grain target values. It never changes loss weights, scales, caps, or
the solve signature.

## Detection and bridges

Exact matches use the measurement fields `concept`, `entity`, `map_to`, and
`filters`. Two explicit UK bridges cover relationships that those fields
cannot express on their own:

- The 10-cell `ons.household_composition.*` partition sums to the national
  household-count control and bridges to the ladder-derived
  `census_households/households` metric.
- `dwp.uc.households` bridges to `dwp.uc.households_by_area`. The four
  `dwp.uc.payment_distribution_*` rows also match the by-area target exactly
  and form a separate exhaustive national partition.

These declarations live beside the UK Ledger target orchestration in
`uk_runtime/ledger_targets.py`. Tests pin the precedence and complete bridge
membership so a rule change requires an explicit doctrine-constants review.

## Geography legs and refusals

The bound control's geography defines the factor legs. A UK control produces
one factor over England, Wales, Scotland, and Northern Ireland. A GB control
covers England, Wales, and Scotland; a surface that also contains Northern
Ireland then fails as unparented. Country controls produce one factor for each
declared country. Area codes map to countries using the same ONS-prefix logic
as the local target runtime.

The pass also refuses a partially bound declared partition, a target covered
by two bridges, incompatible controls at the same winning grain, an empty or
unparented leg, a zero lower total with a nonzero control, a sign flip, and any
non-finite input or result. Every successful factor records the parent,
constituent target ids, area count, old and new totals, relative shift, and
declared factor. Dry-run plans and build manifests carry the pass receipt even
when no inconsistency is in force.

## Census evidence and current effect

Published constituency household counts are disclosure-controlled. At review,
the measured constituency-sum differences from the corresponding national
census totals were E&W +105, Scotland -554, and Northern Ireland +3 — a
relative magnitude around 2e-5. The tension the reconciliation actually
addresses is a separate, larger one: the roughly 1–2% vintage gap between
census-day 2021/2022 counts and any bound 2023 national household count. The
published local values still bind as published when no same-concept national
control is bound. If one is bound in the same solve, country wins and the
standing rule rescales the local values before calibration.

Today the rowwise candidate binds only `census_households/constituency` and
declares no bound national targets. The pass therefore records an absence
receipt and leaves all target values numerically identical. Increment #762
must extend the mixed-grain surface through
`apply_uk_cross_grain_reconciliation`; the existing rowwise area-type fence
remains the structural backstop.

## Rescope from the issue text

Issue #802 originally called for a committed enumeration artifact and a
per-family ruling register. This implementation deliberately replaces both
with a live pass and a standing, pinned rule. The receipt enumerates the
inconsistencies actually present in each assembled surface, avoiding a second
artifact that could become stale. The precedence and bridges are the ruling,
while the refreshed `census_disclosure_control_noise` adjudication records the
specific disclosure-control acceptance. The issue text should be updated to
reflect this rescope during review.
