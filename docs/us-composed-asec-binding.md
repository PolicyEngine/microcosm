# Binding the carried prepared-ASEC evidence to the composed population

`us_runtime.graph_composed_asec_binding` and
`us_runtime.graph_composed_asec_measures` declare the three nodes that attach
the prepared ASEC arm's carried evidence to the ASEC rows of the composed
population declared in [us-composed-population-graph.md](us-composed-population-graph.md),
and derive the already-reviewed corrected measurements over exactly those rows.
The stage is `composed_asec_binding_v1`. It is an engineering intermediate: it
is not a certified population, a calibration, a transfer, a score or a release,
and every receipt says `release_eligible: false`.

| Node | Kernel | Operation |
| --- | --- | --- |
| `composed_population.asec_bind` | `us.composed_population.asec_bind@1` (new) | Resolve the ASEC arm's composed rows to their source rows; emit the typed binding, the resolved source rows and the reviewed money slice; own one boolean arm cell per bound entity |
| `composed_population.asec_cps_carried_current` | `us.composed_population.asec_cps_carried_current@1` (new) | `derive_cps_carried_current_leaves` (**reused**) under the arm mask |
| `composed_population.asec_reported_income` | `us.composed_population.asec_reported_income@1` (new) | `derive_reported_income` (**reused**) under the arm mask |

## Why the accepted prepared-slice kernels could not simply be appended

They refuse, and the refusals are real rather than cosmetic. Declaring
`us.asec_prepared.select_households@3` verbatim over the composed population,
with the composed CREATE's carried artifacts on its edges, fails immediately at
`PREPARED_CONTEXT_METADATA`: a composed context carries the assembly and
stacked-spine manifests, where that kernel requires an empty one. Behind that
first refusal sit three more, each exercised in the test module:

- its `_document` also requires an empty mass log and a **design** household
  weight, and harmonization has already moved the population to importance;
- `PREPARED_RECEIPT_ROWS` and `BODY_ROW_ALIGNMENT` compare the prepared
  receipt's `entity_rows` and the money body's rows with the population's rows,
  and the composed population has both arms in it;
- `verify_graph_housing_rows` and `derive_reported_income` are given positions
  and identifiers that must resolve inside the prepared arm's own buffers, and
  composed row positions and composed row identifiers do not.

None of those contracts is weakened here. The composed population is bound with
new nodes that satisfy them on the arm's own coordinates.

## How a composed row is resolved to a source row

By source arm plus the arm's own pre-remap identifier and income-year cohort —
never by composed row position, and never by the post-offset `*_source_id`
(which equals the entity id at CREATE and binds nothing on its own).

1. `*_support_channel` gives the arm. An unexpected third channel, a missing
   channel or an empty arm refuses.
2. `*_spine_source_id` gives the arm's original identifier for each of its rows.
3. The source's own ordered identity comes from the carried evidence, not from
   an assumption: `income_observations`' `person_id` for persons and
   `housing_universe`' `household_id` for households. Both are admitted against
   the prepared receipt before they are read, so the ordering is
   receipt-authenticated.
4. The mapping refuses if it is empty, if the source identity repeats an
   identifier, if the composed rows repeat one, if any identifier resolves to
   nothing, or if the resolved positions are not strictly increasing. The last
   is a checked property of the actual mapping, never imposed by sorting.
5. `source_year` on the composed rows must equal `income_year` at the resolved
   positions, and the income artifact's own three declared crosschecks —
   `source_household_id`, `A_LINENO`, `A_AGE` — must agree cell by cell.
6. The household evidence is verified with the reviewed
   `verify_graph_housing_rows` against a view restated under the arm's own
   pre-remap identity, with one income year per household derived from the
   arm's persons; a mixed-year household refuses.

Nothing assumes where the arm's rows sit. They need not be a prefix and need not
be contiguous; sorted or interleaved group tables are supported and covered by a
test whose invented ACS arm interleaves with the ASEC one. Subsampling is
supported: at `sample_fraction < 1.0` fewer rows bind, the resolved positions
are a genuine non-contiguous subset of the whole arm, and the binding document
records `composed_is_whole_arm: false`.

## What binds each artifact to its producer

The bind node takes seven typed edges. The population's `frame_context` comes
from the population's own context producer. The other six — the origin document
and the five positional evidence artifacts — must all come from the composed
CREATE (`NODE_EVIDENCE_PRODUCER`) **and** must share one producing node
execution (`ARTIFACT_PRODUCER_SPLIT`), because all six are positional over one
prepared population and one composition of it.

The origin is held to that producer identity as well as to its content. The
composed context's row identities are checked against the actual tables, and
`source_origin`'s per-entity provenance digests are **recomputed** against those
tables: `bind_composed_source_origin` deliberately verifies no digest against
any population, so a document describing a different frame is caught here
(`ORIGIN_DOES_NOT_DESCRIBE_THIS_FRAME`). That recomputation is what catches a
wrong frame; it cannot tell two CREATE executions that describe identical tables
apart, which is why the producer key the binding document records is pinned by
the equality rather than carried unverified.

The bind node emits three artifacts:

- `asec_binding` (`microcosm.us.composed_asec_binding@1`), the typed document:
  the composed and arm identities, the resolved source positions' digests, the
  cohort counts, the demographic evidence, and every consumed payload digest and
  producer key.
- `arm_rows` (`microcosm.us.composed_asec_arm_rows@1`), a framed binary artifact
  carrying the resolved source positions and original identifiers. The
  measurement nodes need those to reconstruct the reviewed accounting, and this
  is how they get them **without** reading a provenance column. It is admitted
  only against the binding document that already digests every buffer.
- `selected_current_money`, cut by the reviewed `select_current_money` at the
  resolved coordinates, exactly as the prepared slice's own selection node cuts
  it.

## Writing only the arm's rows

The bind node owns one boolean cell per bound entity —
`person_composed_asec_row`, `household_composed_asec_row`,
`spm_unit_composed_asec_row` (entity-prefixed because `Frame` requires globally
unique column names). Both measurement nodes declare every input slice and every
owned cell under that mask. That is not a convention: `Slice.rows` masks the
table the kernel receives, so a measurement kernel cannot see an ACS row, and
`population._patch_columns` refuses any change to storage outside an `Owned`
mask, missingness included. A test compares the bound population with the
unbound composed population column by column with `storage_equal` on every ACS
row of every entity, group quarters included.

Three corrected leaves — `age`, `employment_income_before_lsr` and
`self_employment_income_before_lsr` — are columns the native ACS mapping already
produces. On those the masked write fills the ASEC rows and leaves the ACS cells
exactly as they were. Every other leaf and all six `asec_reported_*` columns are
new, so the ACS rows keep a declared null. Nothing is inferred as zero.

## Demographics: what is bound, and the gap that is not

The composed parent reports `age` on the ACS arm only. It is bound here through
the **already reviewed** mapping rather than a new one:
`derive_cps_carried_current_leaves` declares `age` among its corrected leaves and
derives it from the routing column `A_AGE`, which the composed person table
carries on both arms (the ACS reader maps `AGEP` into `A_AGE` in
`acs_pums._with_structural_columns`). The value is verified against the income
artifact's own `A_AGE` at the resolved positions before it is written.

The reference period is recorded, not harmonized. The ASEC observation is the
interview household one year after the income year — the reference period the
income artifact declares and this stage verifies — so the three restored cohorts
are observed in ASEC survey years 2023, 2024 and 2025 while the native ACS arm's
`age` is observed in its own vintage. The binding document records
`observed_period_by_cohort` per cohort with `aged_from_a_date: false` and
`cross_arm_period_harmonized: false`. **No person is aged from any interview or
survey date**, and reconciling the two periods is a calibration decision this
stage deliberately leaves open.

Two demographic fields a calibration will also want are **not** bound by this
stage, and the binding records exactly what it observed rather than guessing:

| Column | Reviewed ASEC mapping | Required source column |
| --- | --- | --- |
| `is_female` | `cps_carried` (`is_female = A_SEX == 2`) | `A_SEX` |
| `is_household_head` | `relationship_inputs` (`is_household_head = P_SEQ == 1`) | `P_SEQ` |

Binding either column needs a reviewed cross-arm mapping decision — a decision
about the prepared arm and its ACS counterpart, not a choice this stage may
make. Whether the prepared arm *carries* the source column is a separate
question, and the two must not be confused:

- **Observed on genuine source.** The frozen raw-stage parent
  (`asec_raw_stage.checkpoint.h5`, the prepared arm's parent) declares a
  176-column person table containing both `A_SEX` and `P_SEQ`, and the CREATE
  `frame_context` of the genuine 505-household run — the very payload this node
  reads as `asec_frame_context` — declares a 179-column person roster
  containing both. Read as metadata only, from the existing prepared-source
  directory and run artifacts.
- **Observed on the invented fixture.** The invented parent frame the test
  module builds carries neither column.

The binding document therefore records what it saw rather than generalising
either observation into a source contract: each unbound entry carries
`present_in_prepared_arm` and `absent_from_prepared_arm` for the population in
hand, plus `bound_by_this_stage: false`.

The gap fails closed on the thing that would actually make it stale — a write.
`refuse_unbound_demographic_writes` screens the corrected-leaf and
reported-income owned-column rosters and the document's own `bound` entries, so
a reviewed mapping that started producing `is_female` or `is_household_head`
refuses at declaration time (`ASEC_DEMOGRAPHIC_COLUMN_BOUND`). Presence of
`A_SEX` in the source is not a refusal and never was a licence.

## Source-blindness

Resolving which arm a row came from is the binding module's whole operation, so
`graph_composed_asec_binding.py` is a reviewed source-spine provenance owner. It
writes no measurement. `graph_composed_asec_measures.py` is an ordinary
classified module under the source-blind guard: it reads no channel and no
spine-source-ID column, receiving the arm as a declared boolean mask and its
source coordinates as a typed artifact.

## Implementation scope

`composed_asec_binding_v1`'s module inventory is exactly
`composed_population_v1`'s plus the two binding modules, with the identical
dependency set. Because that inventory is a superset of the composing scope and
therefore of each of its parents, no existing module's covered-import
intersection moves. The stage binds the same two codecs the composing CREATE
binds — `us-asec-prepared-current-money-v3` and `us-acs-native-2024-v1` — rather
than claiming an identity narrower than its actual inputs, and both
runtime-option families for the same reason. Declaring the stage edits
`graph_implementation.py`, so every existing US stage identity moves and every
existing US graph cache is invalidated; that is the reviewed-fence design, not a
property of this binding.

## Genuine-small runner proposal (not executed)

Nothing genuine has been run. The proposal is exactly the composed graph's
runner with the bound graph substituted, on the same frozen inventories:

```
composed_asec_population_graph(
    frame_column_declarations(<composed frame>),
    sample_fraction=<declared>, sample_seed=578,
)
```

with the same `sources` mapping. Source preparation is still not sampling: both
arms are read whole regardless of the fraction, so a small fraction does not buy
a small read. The three added nodes are pure — no source, no engine, no network —
so their marginal cost over the composed run is the arm resolution, one money
slice and two derivations over the ASEC arm. A first genuine run would establish
execution and cache-key stability on real bytes and nothing else: not accuracy,
not representativeness, not calibration, not congressional-district fitness, and
not release eligibility.
