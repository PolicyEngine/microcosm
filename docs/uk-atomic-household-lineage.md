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

No production UK host is changed here. María's full-build graph currently uses its existing ladder draw after geographic expansion. Adopting this key and shared assignment must be coordinated in that host. The existing source stage order also interleaves enrichment before later SPI/CGT cloning; moving atomic assignment before every enrichment stage requires a separate staged migration. This helper does not claim that migration, native FRS qualification or a new UK release.

## Verification scope

Invented tests cover the full four-branch path, explicit selection, exact large IDs, source and ordinal changes, stable keys under row reordering and pool growth, unknown/colliding ancestry refusal and detached output. A real shared CREATE/EXPAND/FILTER graph supplies the household lineage receipt and replays from cache. Its projected keys then feed the existing seven-node atomic graph, including three system imports, assignment, derivation and its gate. That graph also completes required replay and preserves the supplied entity tables, weights, strata and metadata. No native source, private archive, country engine or external resource is used.
