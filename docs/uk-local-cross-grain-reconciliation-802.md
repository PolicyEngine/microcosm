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

## The national register is reconciled first

Before #1123, only the joint local surface was reconciled, and only its local
cells were written back. A region row rescaled to its country control fed the
constituencies below it, but the national register the solve binds kept the
raw region value.

`uk_runtime.national_reconciliation.reconcile_uk_national_registry` now
reconciles the national register itself, and writes the reconciled values
back into it. Both target loaders run it before the measure exclusions and
before the frozen scoring register is compared:
`load_uk_full_target_inputs` and `load_uk_national_target_inputs`. The
national-only graph therefore binds the same values as the full graph.

It reconciles two relations:

- **exact measurement signatures** across the country, nation and region
  grains, for example the CGT regional cells under the UK total;
- **band bridges** (`band_bridges` in the declarations). A projected UK
  income-tax-liabilities band from HMRC ITL Table 2.5, or a run of such bands,
  controls the SPI regional cells from Table 3.11 whose band nests in it. The
  regional `£200,000 and over` cells take the four ITL bands from £200,000 up.

Every reconciled row keeps `cross_grain_value_before`, `cross_grain_factor`
and `cross_grain_control`. Fan-out targets, meaning several cells at one
geography, are distributions, not controls.

The validation-period registers stay as compiled, because compile parity
measures the facts themselves.

The joint local pass then refuses any factor away from one on a group whose
lower grain is a control grain
(`national_reconciliation.assert_uk_national_rows_unmoved`): the two passes
must agree.

## Household, income and caseload controls (#1123)

Five relations reconcile the families the K=25 audit found out of line.

**Census households.** Each nation's official household estimate controls its
census cells, through a per-geography bridge (`per_geography`: each control
covers the legs of the geography it sits at):

- the LFS 2025 rows for England's nine regions;
- NRS 2025 for Scotland;
- Welsh Government mid-2024, rolled to 2025 by population growth;
- Northern Ireland's Census 2021 districts, rolled by LPS dwelling stock.

The UK household-composition partition takes the nations' sum in the
national pass (`sum_bridges`). A15 uprates the census cells to that sum, and
the LFS UK total is a diagnostic with its gap recorded.

**Tenure.** The census tenure cells partition their authority's households
(`share_of_parent`). England's shares are drifted by SPREE's 2022–2024 share
change before the partition.

**HMRC income by area.** The SPI band partition, summed over every band of
the full compiled register, controls the Tables 3.14 and 3.15 area cells per
grain (`fanout_sum_bridges`). A factor more than 2% from one is refused. A
fan-out is never an exact-signature control: its cells are a distribution.

**UC child bands.** The bands partition each constituency's UC caseload
(`share_of_parent`).

**Small cells.** A small-cell deferral still counts its value in its leg's
control, through a reconciliation row that never reaches the solve.

## Declarations and enforcement

Every declaration lives in `uk/cross_grain_declarations.json`, which the
country-spec fingerprint covers:

- **band_bridges**: projected UK ITL bands that control the SPI regional
  band cells nesting in them (see above).
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
