# Sample-size-aware income graph evidence export

`packages/microcosm-build/src/microcosm/build/graph_evidence_income_sized.py`
exports the same five-node ASEC reported-income engineering rung as
[the income-v3 profile](us-income-v3-graph-evidence-export.md), at whatever
sample size the run actually holds.

It is a **separately named versioned successor**, not a widening. The reviewed
profile describes exactly one accepted run and pins its size — fifteen
households, fifty-two persons, twenty-five tax units, five households per source
cohort — as literal constants compared for equality. That is the right shape for
one frozen run and the wrong shape for a second one: a run whose cohorts are not
equal to each other is refused before its size is even looked at. Relaxing those
constants in place would weaken a profile that already passed review and would
change the `exporter_sha256` its published identities depend on, so this module
imports the reviewed one and edits nothing in it. Neither profile accepts the
other's root binding.

```sh
python packages/microcosm-build/src/microcosm/build/graph_evidence_income_sized.py \
  --run-root PATH_TO_RECORDED_RUN \
  --operator-declaration PATH_TO_PINNED_DECLARATION \
  --root-export-binding PATH_TO_ACCEPTED_ROOT_EXPORT_BINDING \
  --output-root NEW_EXCLUSIVE_QUALITY_ROOT
```

## What is closed, and what varies

The semantic profile is fixed and does not depend on the cohort size. Every one
of these is inherited from the reviewed module by reference rather than restated,
so the two profiles cannot drift apart on anything that is not the size:

- the five node identities, their labels, their typed artifact rosters and their
  `@3` kernel references;
- the producer edges, including the enforced rule that the accounting node
  consumes `create` and `select_households` and **never** the engine;
- the twenty-eight column, ninety-one byte accounting row, its three permitted
  dtypes and its contiguous layout;
- the thirty-three money field axes and the seven housing-universe evidence
  fields;
- the six reported quantities and their status, validity, zero-origin and
  allocation axes, and the hand arithmetic order they are accounted in;
- the price basis `census_ccpiu_annual_2024`, published only when the accounting
  node's own recorded implementation manifest pins
  `us_runtime/asec_current_money_price_basis_v1.json` by the digest
  `88cec207…`, so the name is bound to the bytes the conversion read;
- the five withheld claims, including `prior_wages`, which is outside a
  cross-sectional scope rather than something this run failed to measure;
- the nine unmeasured dimensions, unchanged and still nine;
- the three source cohort **years**. The roster `{2022, 2023, 2024}` is part of
  the closed semantics. Only the per-cohort **counts** vary.

What varies is the sample size: households, persons, tax units, the other three
entity row counts, and the per-cohort household counts.

## The size is measured, not declared

No count in this profile is taken from a single field. Each is read from the
run's own recorded outputs, required to agree everywhere the run writes it down,
and required to equal what an accepted root output review independently
declares. A recording that disagrees with another recording of the same fact
refuses immediately and names both sites.

| fact | where the run records it |
|---|---|
| households | `direct-receipt` `entity_rows.household`; `direct-sampler` `realized_household_count` and `sampling_unit.realized_unit_count`; the entity-structure output's `entity_rows` and `expected_households`, when the producer writes one |
| persons | `direct-receipt` `entity_rows.person`; the same receipt's `reported_income.rows`; each execution manifest's `cps_carried_current` and `reported_income` node receipts' `person_rows`, in both the cold and the warm phase |
| tax units | `direct-receipt` `entity_rows.tax_unit`; the entity-structure output's `entity_rows` and `nesting.tax_unit.rows`, when the producer writes one |
| spm units | `direct-receipt` `entity_rows.spm_unit`; each manifest's `cps_carried_current` node receipt `spm_unit_rows`, both phases |
| each cohort | `direct-receipt` `realized_households_by_cohort`; `direct-sampler` `strata.<year>.realized_units` |

The node row counters are checkpoint header counts the producer already wrote
down as integers; reading them costs no array decode. The cohort selection
receipt is read as bounded JSON for counts and the selection rule only — the
selected household id digests stay digests, and no identifier is ever decoded.

On top of agreement, the size has to be arithmetically coherent with the rest of
the evidence:

- every count is a positive safe integer, and persons are never fewer than
  households;
- the per-cohort households sum to the household count, and no cohort realises
  more households than it recorded as eligible;
- each accounting column's byte extent is its dtype width times the person
  count, the offsets are contiguous, and the body totals ninety-one bytes times
  the person count;
- each result column's three signed classes cover the person rows exactly;
- the zero-origin classes partition the person rows exactly once across the
  three cohort years. The reviewed profile only shapes these counts; this one
  adds them up. Both genuine runs satisfy it exactly, and the producer builds
  them as a partition of the accounted rows by income year and class.

Finally, households and persons — the two counts everything else is sized by —
must each have at least two independent recordings inside the run before the
export proceeds. The other entity counts deliberately do not carry that
requirement: the first producer records the tax-unit count exactly once, and
demanding a second would refuse a correct run for recording less. They are
reconciled against root's declared scope instead.

Each fact, its count, how many places recorded it and which places those were
are published under `scope.measurement`, so a reader can assess the size rather
than take it.

## Producer schemas are explicit adapters

Two producers have recorded runs of these five kernels, and their receipts are
structurally different — not the same receipt renamed. The second one adds three
run-receipt fields, two fields to each phase receipt, five fields to the direct
result, two to each graph result, and five output files.

Each supported producer is one named `ProducerSchema` naming the run and worker
schema strings it accepts and the exact extra fields and outputs it records:

| adapter | run schema | adds |
|---|---|---|
| `income_v3_first_genuine` | `microcosm.income_v3_first_genuine_graph_run.v1` | nothing |
| `income_v3_measured_505hh` | `microcosm.income_v3_505hh_graph_run.v1` | `draft_roster_sha256`, `namespace_paths`, `purged_sys_path_entries`; `available_memory_abort` per stage; `namespace_paths` and `purged_sys_path_entries` per phase receipt; `boundary_continuity`, `entity_structure`, `full_parent_money_field_axes_sha256`, `position_continuity` and `prepared_receipt_continuity` on the direct result; `boundary_continuity` and `prepared_receipt_continuity` on each graph result; `cold-`, `direct-` and `warm-continuity.json`, `direct-entity-structure.json` and `direct-full-money-field-axes.json` |

The root binding names which adapter it approved, and the run's own recorded
schema string must be that adapter's. A run recorded by one producer and
declared as the other is refused on the closed field roster, which is the honest
failure. An adapter widens a **field roster**, never a check.

Three of those additional fields are admitted as keys and never read, on
purpose. `position_continuity.accepted_positions` in the larger genuine run
reads `{household: 15, person: 52}` — those are the accepted fifteen-household
*parent* positions carried forward for continuity, disclaimed in the run's own
sibling scope string, and they are not that run's sample. The same trap sits in
`full_parent_money_field_axes_sha256` and in the `create` node's recorded
`prepared.entity_rows`, both of which count the full parent frame. This module
reads none of them.

`namespace_paths` and `purged_sys_path_entries` hold private filesystem paths.
They are admitted so a correct run is not refused for recording more, nothing
reads them, and no published string may contain a path: `bounded_tree` applies
the publisher-text check to every string the export writes, and a test puts real
path-shaped values in those fields and asserts none reaches a written byte.

## The root export binding

There is no default pin constant. `freeze_sized_pins` builds `SizedPins` from an
accepted `microcosm.income_v3_sized_root_export_binding.v1` file. A missing file,
a different schema or profile, a non-APPROVE verdict, an unsupported producer
schema, a missing identity, or a scope that is not a positive internally
consistent set of counts all refuse before anything is read from the run. The run
receipt is never authority for itself.

Beyond the identities the reviewed profile's pins carry, the binding declares
`producer_schema`, `selected_entity_rows` and `realized_households_by_cohort` —
the scope root observed while reviewing the outputs. That is what the measured
size is reconciled against. A run that is internally consistent about a size root
did not approve is refused.

## What this profile still does not do

It calculates nothing, calibrates nothing, fits nothing and infers nothing. It
reads recorded aggregate evidence, checks it against itself, and republishes it.
Cache equality and recorded parity are engineering evidence; they authorize no
release, national claim or district analysis, and the published quality block
carries no score. The nine unmeasured dimensions remain unmeasured at every size
— a larger sample measures no more of them than a smaller one.

The six recorded calls in a cold phase are **five graph kernels and one separate
source-preparation helper**. The graph has five nodes. Source preparation is
counted apart from the kernels everywhere this profile reports it.

## Tests

`packages/microcosm-build/tests/test_graph_evidence_income_sized.py` exercises
the profile at two invented shapes — nine households with equal cohorts on the
first producer, forty-one households with deliberately unequal cohorts (14, 14,
13) on the second — because a profile only ever read at one size still has its
size baked in somewhere. Both shapes are invented and say so; no test reads a run
directory, a checkpoint or any source data.

The suite also asserts that `graph_evidence_income_v3.py` and `graph_evidence.py`
are byte-identical to the accepted heads this successor was built on, which is
the most direct back-compat proof available: nothing here can have changed what
the fifteen-household profile accepts, refuses or publishes.
