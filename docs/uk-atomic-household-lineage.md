# UK household identity after cloning

The shared atomic-area operators need a stable household key after initial cloning. `uk_runtime.atomic_household_lineage` projects supplied original-source identities through explicit selection and expansion records, using the existing `household_draw_key` encoding. It adds no assignment algorithm, data reader, issuance registry or full-build host.

A key includes the declared source, source vintage, original household ID and ordered clone path. It excludes final offset IDs, row order, sample size, calibration weights and financial values. Exact int64 IDs above `2**53` remain distinct. Original households keep their previous path; a new child appends its explicitly supplied branch ordinal. An original branch does not gain an inferred zero ordinal.

## Contract

`project_atomic_household_keys(roots, *, steps, final_ids)` returns a detached two-column table: int64 `household_id` and nullable-string `geography_household_key`, in the requested final order. Roots have exactly `household_id`, `source`, `source_vintage` and `source_household_id`. An empty selection is a valid description, not evidence that an empty dataset is useful or releasable.

Steps are immutable descriptions:

- `HouseholdExpansion(branch, before_ids, after_ids, parent_pairs, child_ordinals)` carries exact household axes, the household part of the shared executor's `receipt["expand"]`, and explicit child ordinals. The existing ordered branches are SPI support, CGT incidence, CGT band donors and geographic support. Each arriving household needs one known preceding parent and one ordinal. Two children of the same parent cannot reuse an ordinal in the same branch.
- `HouseholdSelection(before_ids, after_ids)` explicitly removes households while preserving their keys. This supports the source-family sample between the completed source spine and geographic expansion. It does not check that the sample respected the host's family-sampling rule.

Each before-axis must have exactly the previous step's household IDs. Expansion cannot remove incumbents; selection cannot add households. Unknown or null parents, repeated or reordered branches, incomplete parent/ordinal coverage, duplicate keys and a mismatched final axis refuse. Axis ordering may change, because ordering is not identity. The function never guesses ancestry from ID arithmetic, support flags or model values.

## Host responsibilities

These checks establish internal consistency of supplied descriptions. A dataclass, DataFrame or serialized receipt does not prove its source. The receiving host must retain and check the actual original source owner, complete before/after populations and source/EXPAND/FILTER evidence. It must authenticate the complete structural stage roster, the explicit branch-ordinal convention, and the exact final receiving household axis before and after relevant I/O. A coherent invented history can pass this pure helper; it is not an admission gate.

The host can extract the household pairs from its verified shared EXPAND receipt without decoding household ID offsets. The helper returns the key column for the existing `uk_atomic_assignment_definition` and `atomic_geography_nodes` APIs. Support-byte provenance, observed-region constraints and versioned mapping relations remain separate responsibilities. Larger geographies continue to derive from the assigned atomic area.

The full-build graph now adopts this key: `uk.full.identity` (`UKGeographyIdentityKernel`) reads the spine's explicit lineage columns and the pool clone index and writes `geography_household_key` with exactly this encoding, and `test_uk_atomic_identity_graph.py` holds the kernel to this projector on the executor's actual EXPAND receipt. The existing source stage order still interleaves enrichment before later SPI/CGT cloning; moving atomic assignment before every enrichment stage remains a separate staged migration under the geography-first umbrella issue (PolicyEngine/microcosm#931). This helper does not claim that migration, native FRS qualification or a new UK release.

## Shared lineage readers

The UK's copy lineage is declared once, as data, in `uk_runtime.household_lineage.UK_COPY_LINEAGE`: every copy's true source is `source_household_id`, and each copying step (the SPI support channel, the CGT support split, high-value support, the CGT incidence clone, the CGT residential split and the geographic clone) names its marker and ordinal columns in path order. The country-neutral readers in `microcosm.build.source_lineage` take that declaration as a parameter, so another country can declare its own lineage and reuse them (microcosm#931):

- `record_sources` gives every record of any entity its true source through household membership. Every UK floor counts distinct sources through it (`receipt_counts.record_source_households`), so a household's copies and clones count once.
- `copy_paths` reads every household's ordered clone path; `uk_household_clone_paths` is the UK's strict entry for the identity kernel.
- `replica_keys` and `collapse_replicas` keep one training row per true source and path without the geographic branch, with the clones' weights summed. The FRS-trained fits fold their training rows through `household_lineage.collapse_geographic_clones`: the SPI stage-2 and reference forests, the housing shell, the UC reporter redraw, the UC capital donor cells, the salary-sacrifice forest and the SPI employment shell. Once geographic clones precede enrichment, each fit still sees each survey record once. A frame without geographic clones keeps its rows and weights value for value.
- Each fit passes what it reads (its predictors and targets, or the columns they derive from), and clones fold only where they agree on it. Clones that differ, through a draw keyed on their own ids or a stage run per area, stay rows of their own. Each variant carries the summed weight of the clones that share it, so the fit sees every clone's value at its share and nothing is dropped.

## Verification scope

Invented tests cover the full four-branch path, explicit selection, exact large IDs, source and ordinal changes, stable keys under row reordering and pool growth, unknown/colliding ancestry refusal and detached output. A real shared CREATE/EXPAND/FILTER graph supplies the household lineage receipt and replays from cache. Its projected keys then feed the existing seven-node atomic graph, including three system imports, assignment, derivation and its gate. That graph also completes required replay and preserves the supplied entity tables, weights, strata and metadata. No native source, private archive, country engine or external resource is used.
