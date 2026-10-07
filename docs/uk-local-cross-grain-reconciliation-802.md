# UK cross-grain reconciliation (#802, #905, #1123)

When the UK target surface binds one quantity at more than one geography
grain, the higher grain is the control and the lower grain is rescaled to it.
The shared operator in `microcosm.build.cross_grain` changes only target
values. It never changes loss weights, scales, caps or the solve signature.

## Grains and legs

The precedence is country, nation, region, constituency, local authority
(microcosm#1123 added the nation grain):

- **country**: the UK (`K02000001`), Great Britain (`K03000001`) and England
  and Wales (`K04000001`);
- **nation**: England (`E92000001`), Wales, Scotland and Northern Ireland,
  when a family binds them as nations;
- **region**: the nine English regions, plus Wales, Scotland and Northern
  Ireland when a region-tier fan-out binds them beside the regions
  (`cross_grain_grain: region`, microcosm#905);
- **constituency** and **local authority**: the two local grains.

Country, nation and region rows control the grains below them. A **leg** is a
region-tier code (nine English regions, Wales, Scotland, Northern Ireland).
Every control covers the legs its geography declares, and a lower row covers
the legs of its own geography. England spans its nine regional legs, so a UK
row controls an England row and the three other nations jointly, and the
England row then controls its regions. A control rescales the unclaimed rows
on its legs by one factor. The nearest control claims first. A row that would
straddle two controls is refused.

A control whose legs are only partly covered at a middle grain is refused,
unless the uncovered legs are licensed empty for every leaf target (no data at
any grain there). Without this guard, the rows present would take the whole
control.

With no control-grain row in a group, the top grain present controls the rest
(a constituency-only family reconciles its authorities to its constituencies
per leg).

## Declarations and enforcement

Every declaration lives in `uk/cross_grain_declarations.json`, which the
country-spec fingerprint covers:

- **bridges**: higher-grain targets whose measurement signature cannot match
  the lower side's. The household-composition partition controls the census
  household cells, and the GB UC caseload controls UC households by area.
  Eight age bridges join the region-tier ONS age bands (inclusive integer
  ages) to the local bands (half-open). The 80-89 band has no local cell.
- **partitions**: member targets that divide a parent at one geography,
  either `exhaustive` (members sum to the parent) or `share_of_parent`
  (members move by the parent's factor and keep their shares).
- **local_routes**: local targets with no published higher control, each with
  its reason.
- **relations**: overlapping targets at different grains that are not one
  quantity (`independent`) or are bounded parts of a whole (`subset`).
- **signature_incomplete**: targets whose measurement block does not express
  their population (the binding's filters or groupby do), each with its
  reason.

`uk_runtime.cross_grain_declarations.uk_cross_grain_coverage_violations` turns
the declarations into an enforced rule (microcosm#1123). It refuses:

- a local target with no route to a higher control;
- two targets at different grains whose measurements overlap but which are
  neither reconciled nor declared;
- a declaration that names an unknown target or no longer applies.

The surface build runs the check once per process
(`uk_cross_grain_coverage_receipt`), and its receipt travels with the
reconciliation receipt. `uk/target_doctrine_exceptions.json` tolerates the
gaps that still exist, each naming the change that closes it. A tolerated
entry that no longer matches a gap is refused as stale.

## Refusals and receipts

The operator also refuses:

- a partially bound declared bridge (unless the missing members are reviewed
  exclusions);
- a target covered by two bridges;
- incompatible controls over the same legs (beyond summation-order rounding);
- an empty or unparented leg;
- a vanishing lower total under a nonzero control;
- a sign flip;
- any non-finite value.

Every factor records its parent, legs, area count, old and new totals,
relative shift and declared factor. Partition receipts record each cell's
parent value and factor.

## Census households

Published constituency and local-authority household counts compile from the
pinned Chronicle feed and remain disclosure-controlled. The cells total
28,061,271 households at constituency grain and 28,061,277 at local-authority
grain. A15 uprates each grain to the 2025 Ledger control of 29,003,000, and
A17 applies the grain factor to eligible census-tenure holds (#887). The
household-composition bridge then rescales both local grains to the
composition total.
