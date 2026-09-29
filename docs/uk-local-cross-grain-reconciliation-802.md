# UK local cross-grain reconciliation (#802)

The UK local build applies one standing rule before constructing a calibration
matrix: when the same measurement concept is bound at more than one grain, the
highest bound grain wins in the order country, constituency, then local
authority. The shared operator in `microcosm.build.cross_grain` changes only
the lower-grain target values. It never changes loss weights, scales, caps, or
the solve signature.

## Detection and bridges

Exact matches use the measurement fields `concept`, `entity`, `map_to`, and
`filters`. Explicit UK bridges cover relationships that those fields
cannot express on their own:

- The 10-cell `ons.household_composition.*` partition sums to the national
  household-count control and bridges to the Chronicle-compiled
  `ons.census.households` contract target. Since microcosm#791 the ten cells
  bind on the `frs_relationships` stage's `household.ons_household_type`
  column, so the bridge is fully bound (it was reviewed-unbound while three
  cells carried measure exclusions).
- `dwp.uc.households` bridges to `dwp.uc.households_by_area`. The four
  `dwp.uc.payment_distribution_*` rows also match the by-area target exactly
  and form a separate exhaustive national partition.

Eight age bridges pair the single-cell UK controls
`ons.population.age_0_9_by_region` through
`ons.population.age_70_79_by_region` with the corresponding local
`ons.age.0_10` through `ons.age.70_80` bands. They represent the same
integer-age populations, but the national inclusive and local half-open filter
encodings do not signature-match. Each bridge therefore rescales both the
constituency and local-authority band to the `K02000001` UK total over the
declared England, Wales, Scotland, and Northern Ireland legs. The national
80--89 band has no local counterpart and is deliberately not bridged.

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

Published constituency and local-authority household counts compile from the
pinned Chronicle feed and remain disclosure-controlled. Against the retired
OA-ladder diagnostic sums, constituency mean/max absolute differences are
7.4/29 households in England, 6.4/15 in Wales, 49.6/157 in Scotland (net
−557), and 7/16 in Northern Ireland (net +4). The NI result uses NISRA's
published DZ2021→PARLCON24 lookup; the retired postcode inference misplaced
10 Data Zones and reached a maximum difference of 694.

The Chronicle cells total 28,061,271 households at constituency grain and
28,061,277 at local-authority grain. A15 uprates each grain separately to the
2025 Ledger control of 29,003,000; A17 applies the corresponding grain factor
to eligible census-tenure holds. If a same-concept national control is bound
in the solve, country wins and the standing rule rescales both local grains.

## Rescope from the issue text

Issue #802 originally called for a committed enumeration artifact and a
per-family ruling register. This implementation deliberately replaces both
with a live pass and a standing, pinned rule. The receipt enumerates the
inconsistencies actually present in each assembled surface, avoiding a second
artifact that could become stale. The precedence and bridges are the ruling,
while the refreshed `census_disclosure_control_noise` adjudication records the
specific disclosure-control acceptance. The issue text should be updated to
reflect this rescope during review.
