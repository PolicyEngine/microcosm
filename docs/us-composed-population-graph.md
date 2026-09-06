# Composed prepared-ASEC / native-ACS population graph

`us_runtime.graph_composed_population` declares the smallest real graph that
composes the authenticated prepared ASEC population with the native ACS
population: whole-source CREATE, stacked-spine assembly, weight harmonization,
geography, and a checkpoint boundary. It is an engineering intermediate toward a
national and congressional-district release. It is not a certified population, a
calibration, a transfer, a score, or a release; every receipt says
`release_eligible: false`.

The reviewed raw-stage seam is unchanged. `graph_sources.us_assembly_graph`
still reads `asec_raw_stage` through `us-asec-raw-stage-v4`, and
`us.production.spine_prepare@2` keeps its kernel ref, contract, params, sources
and node ids. This module is a separately versioned seam over the prepared
source; the two coexist. Node *keys* do move — see "Implementation scope" —
because declaring any new stage changes the implementation identity every stage
binds.

| Node | Kernel | Operation |
| --- | --- | --- |
| `composed_population.prepare` | `us.composed_population.prepare@1` (new) | Read both authenticated sources whole, bridge string storage, `prepare_stacked_spine`, canonical dtype promotion |
| `composed_population` | `us.production.spine_harmonize@2` (**reused**) | DESIGN → IMPORTANCE allocation, mass receipt, completed stacked manifest |
| `assign_us_puma_ladder.*` | the four geography kernels (**reused**) | Lookup import, boundary, PUMA/county/CD assignment, gate |

## Why one CREATE over two sources, not two parent populations

`Node.base` is a single population version. The real declarations refuse every
other shape: a tuple base is "not a structural node", a `CREATE` node "has no
base or population", and a structural node "declares base, not population". An
`EXPAND` over one arm can only invent entrant rows; it cannot bind the other
arm's rows or lineage. So the executor cannot structurally combine two parent
populations, and composition is one `CREATE` over two declared sources — the
shape `USAssemblyCreateKernel` already uses. No executor was mocked and no new
graph API was added to work around this.

## The one bridged schema difference

The prepared ASEC Frame already satisfies the stacked-spine source contract
unchanged: US schema, household-only DESIGN weights, no link tables, positive
mass, integral nonnegative clone-safe IDs, and no preassembled support
provenance. The only difference from the native ACS arm is physical string
storage. The prepared source emits graph column storage
(`string[python, pd.NA]`) because its own CREATE writes graph cells directly,
while every pre-assembly build boundary — and therefore the ACS reader — uses
`CANONICAL_STRING_DTYPE` (`string[python, nan]`). `_stack_spine_tables` requires
shared columns to have identical dtypes.

`bridge_prepared_string_storage` moves the prepared arm down to
`CANONICAL_STRING_DTYPE` and verifies that every affected column's missingness
mask is unchanged. Moving that arm, rather than the ACS arm, leaves the ACS side
byte-identical to today's production assembly, and `_canonical_assembly`
promotes the assembled frame back to graph storage afterwards under its own
null- and value-equality checks. The bridge changes no value, adds no column and
fills nothing; it is recorded in the CREATE receipt as `storage_bridge`.

## What the composed population carries

Source-specific columns stay absent on the opposite arm as declared nulls, never
filled with zero: raw ACS `SERIALNO` strings keep their leading zeros and are
missing on ASEC rows; native `asec_H_SEQ` and 22-digit `PERIDNUM` are missing on
ACS rows; `TYPEHUGQ` is ACS-only and `H_TENURE` ASEC-only. The ASEC arm's
unknown/validity/allocation axes, housing universe and provenance columns, and
its three restored income-year cohorts all survive, as do full person/group
memberships and the receipt-validated support-provenance columns that record the
exact source-to-output mapping.

The CREATE node emits, besides the frame context and the stack preparation, a
typed `source_origin` artifact, the prepared arm's own `asec_frame_context`, and
pass-through of the prepared arm's four typed evidence artifacts
(`current_money`, `prepared_receipt`, `housing_universe`,
`income_observations`), so the corrected-leaves, reported-income, transfer and
calibration nodes have everything they need without re-reading the source.

`source_origin` (`microcosm.us.composed_population_source_origin@1`) digests each
arm's native identity columns on both entities, its cohort counts, mass and
weight kind; the sampling controls and per-arm draw receipts; and — read back
off the assembled frame's own provenance columns — the ordered
channel / raw-pre-remap-id / pre-clone-id / output-id mapping per entity. The
raw `*_spine_source_id` is what binds an output row back to its source row; the
`*_source_id` equals the entity id at CREATE and binds nothing on its own, so
both are digested.

### The carried evidence is positional over the whole prepared arm

`current_money`, `prepared_receipt`, `housing_universe` and
`income_observations` all describe the **whole** prepared ASEC population and
are positional in its order; they are not re-sliced to the composed rows. The
CREATE therefore also emits `asec_frame_context`, the prepared arm's own typed
context encoded before stacking, and refuses if the prepared receipt's
`entity_rows` disagree with the arm it just read. The origin document records
`asec_evidence_alignment` with the evidence rows, the arm rows, the composed
ASEC rows and a `composed_is_whole_arm` flag, so a consumer never has to assume
alignment. At `sample_fraction < 1.0` the flag is false and the counts differ.

A **binding node is still required** before the prepared-slice consumers
(corrected leaves, reported income) can attach: their kernels require an empty
context metadata and prepared rows equal to context rows, neither of which holds
for a harmonized composed population. What this graph provides is the evidence
bytes and the identity needed to write that binding, not the binding itself.

## Known asymmetries, not repaired here

The native ACS arm supplies a mapped `age`; the prepared ASEC arm supplies only
raw `A_AGE`. The composed person table therefore has `age` observed on ACS rows
and missing on ASEC rows. Geography tolerates this (its boundary node keeps every
row and never reads age values), but a national or CD release needs a cross-arm
age. Producing one is an income/leaf-mapping decision owned by the corrected-leaf
and transfer nodes, not by this composition, so nothing here fills it.

The prepared arm is also drawn here with the stacked spine's unstratified
household floor, while the prepared slice's own selection node draws by
`source_year` stratum. At `sample_fraction = 1.0` the two coincide; below it,
this graph's ASEC draw is not stratified by income year.

## Genuine-small preparation and runner proposal (not executed)

Nothing genuine has been run. This section is the proposal.

**Inputs, from existing frozen inventories only.**

- ACS 2024 1-year national PUMS, exactly the two archives in
  `us_runtime/acs_2024_1yr_sources.json`: `csv_hus.zip`
  (`8281008e…e515d0`, 251,500,587 bytes) and `csv_pus.zip`
  (`afdc6d90…c469894`, 602,847,146 bytes).
- The prepared ASEC directory: exactly the ten names in
  `asec_prepared_source.PREPARED_SOURCE_FILES` — `asec_raw_stage.checkpoint.h5`,
  `asec_household_observations.checkpoint.h5`,
  `asec_person_income_source.checkpoint.h5`, `restoration.receipt.json`, the
  three `asec_household_cohort_{2022,2023,2024}.h5` files, and the three
  official `pppub{23,24,25}.csv` members — authenticated by its own restoration
  receipt as documented in `us-asec-raw-stage-v4.md`.
- Geography lookups: the packaged PUMA ladder and 117th-to-119th crosswalk
  already declared by `LOOKUP_SOURCES`.

No new source, no new pin and no download is proposed.

**Source preparation is not sampling.** Both arms are read whole before any
draw. `sample_fraction` reduces the assembled population only. Neither the ACS
archives nor the prepared HDF/CSV members support a fractional read, so a small
fraction does **not** buy a small read: full-source I/O and the peak memory of
the whole loaded arms are incurred at `sample_fraction = 1.0` and at
`sample_fraction = 0.01` alike. Any runner that reports otherwise is wrong.

**Runner.** A single deterministic invocation, `resume="forbid"`, one store
directory, one written manifest:

```
composed_population_graph(
    frame_column_declarations(<composed frame>),
    sample_fraction=<declared>, sample_seed=578,
)
```

with `sources = {asec_prepared: <directory>, acs_native: <directory>,
us_puma_ladder_2020: …, us_cd_crosswalk_117_119: …}`. It must refuse before any
read if either source directory's roster is wrong, if the prepared receipt's
`source_kind`/`file_roster` disagree, or if the declared column inventory
differs from what the sources actually produce — all three refusals already
exist and are covered by tests.

**Memory measurement plan.** Record, not estimate:

1. `resource.getrusage(RUSAGE_SELF).ru_maxrss` at process start, and after each
   of: prepared-ASEC arm loaded; native-ACS arm loaded; both arms live
   simultaneously; preparation returned; harmonization returned; geography
   assigned. Where the peak falls is a question for the measurement, not an
   assumption: both arms stay referenced through `_canonical_assembly` while
   `_prepared_tables` copies each table and `pd.concat` builds the stack, so the
   peak is sources plus stacking copies rather than sources alone.
2. `tracemalloc` peak between the same checkpoints, so allocator high-water
   marks are separated from RSS retained by the two loaded arms.
3. Per-arm `Frame` footprint by entity: `DataFrame.memory_usage(deep=True)`.
4. Wall time and the store's on-disk size per node.

The first genuine run is a **measurement**, not a result: it establishes whether
both arms fit simultaneously on the target machine and what the true peak is. No
fraction is preapproved on memory grounds, and no scaling law may be inferred
from a single point. If the two arms do not fit simultaneously, that is a real
finding to report, not a reason to sample the source.

**What a genuine run would and would not establish.** It would establish
execution, peak memory, and cache-key stability on real bytes. It would not
establish accuracy, representativeness, calibration, congressional-district
fitness, or release eligibility, and it would not license any of the
downstream nodes this graph merely leaves room for.

## Implementation scope

The reviewed stage is `composed_population_v1`: the module inventory is exactly
the union of `assembly_prepare`, `asec_prepared_v3` and `geography` plus the
composing module, with their (identical or subset) dependency sets. Because that
union is a superset of each parent, no existing module's covered-import
intersection changes. The stage binds the two codecs it actually reads —
`us-asec-prepared-current-money-v3` and `us-acs-native-2024-v1` — and both
runtime-option families (pandas string storage for the ACS reader, and the csv
field size limit for the prepared CSV members).

Adding any stage edits `graph_implementation.py`, which is a bound module of
every stage. Every existing US stage implementation identity therefore moves
with this change and every existing US graph cache is invalidated. That is a
property of the reviewed-fence design, not of this composition.
