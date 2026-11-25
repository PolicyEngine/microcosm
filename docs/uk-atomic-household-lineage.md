# UK household identity after cloning

The UK spine copies survey households. Once its source stages have read the FRS, it clones every FRS household into K geographic clones, before the area draw and every later stage. Later stages add the SPI support channel's copies and income band donors, high-value home copies, CGT support copies, CGT incidence twins and residential arms. Every household still records the survey household it came from and the copy steps that made it. This page describes those records, the ids and the key each area draw uses, and how fits and receipts count copies back to survey records (microcosm#931).

## True source and copy path

Every household carries its true source in `source_household_id`: the id of the FRS household it was made from. The spine's `geographic_expand` stage writes it on every household before any other copy step, with `source_year` and `source_household_key` (`"{source_year}:{source_household_id}"`), and every later copy keeps it. Persons and benefit units read their true source through household membership.

Each copy step marks its copies with a marker column, an ordinal column or both. A household's copy path lists `(branch, ordinal)` for each step it is a copy in, in path order. A household that is not a copy in a step adds nothing for it, so no path holds a zero ordinal. The UK declares its copy lineage once, as data, in `uk_runtime.household_lineage.UK_COPY_LINEAGE`. It has six branches, in path order:

| Branch | Made by | Lineage columns | Geography key and area |
| --- | --- | --- | --- |
| `spi_support_channel` | `spi_support_channel` (an SPI copy, ordinal 1) and `spi_income_band_donors` (an income band donor, ordinal 2) | `household_support_channel` (`spi` on a copy, `frs` on an original) and `household_support_clone_index` | Its own, from `spi_copy_areas` |
| `cgt_support_split` | `cgt_support_split` | `household_is_cgt_support_copy` and `cgt_support_copy_index` | Its parent's |
| `high_value_support` | `high_value_home_support` | `household_is_high_value_support_copy` and `high_value_support_copy_index` | Its parent's |
| `cgt_incidence_clone` | `cgt_incidence_clone` (one twin, ordinal 1) | `household_is_capital_gains_clone` | Its parent's |
| `cgt_residential_split` | `cgt_residential_split` | `household_is_cgt_residential_clone` and `cgt_residential_clone_index` | Its parent's |
| `geographic_support` | `geographic_expand` | `household_clone_index` | Its own, from `geographic_identity` and `atomic_area_assign` |

Path order is the order a key lists the branches. It is not the order the spine runs the steps. The spine clones first, then runs the support channel, high-value home support, the CGT support split, the incidence clone and the residential split. A path is read from the columns, so the order the steps ran does not change it. A clone's geographic branch comes last in its path, although the spine makes it first. A CGT support copy of a high-value copy lists `cgt_support_split` before `high_value_support`.

## Geographic clones

`geographic_expand` gives every FRS household K clones through the shared core, `microcosm.build.clone_expansion`, under the UK's declaration. K is a parameter of the spine build (`--n-clones`, one by default). The national line builds on a spine of one clone, and the dense line takes the K of the spine it binds. Clone 0 is the household's own rows and ids. Clone k repeats its household, benefit units and people with every id moved up by k times the geographic field's radix (see [Ids](#ids)). Each clone takes `w / K` of the household's design weight, so the K clones stand for the survey household once. Each table carries each row's clone index, from 0 to K - 1 (`household_clone_index`, `benunit_clone_index` and `person_clone_index`), and every later copy keeps its parent's. `geographic_clones.geographic_clone_count` reads K from the household's index. A spine below full scale samples its FRS root household by household, before any copy step.

Clone 0 keeps its household's ids and key at every K. So wherever a draw is keyed on a row's own id or key, clone 0 of a K-clone spine draws what a one-clone spine draws. The support channel copies each clone on a uniform keyed on the clone's own id and seats the income band donors within each clone, so clone 0 copies and seats the households a one-clone spine copies and seats. The CGT support split sizes a clone's copies on its weight times K, its survey household's weight, so a clone it divides takes about the copies its household takes at one clone. Its walk can stop inside a household's clones and its snap can count one copy fewer, so it nests only approximately. `frs_take_up`'s anchored residuals count unweighted rows within each clone, so they nest at every K. `frs_person_draws`' apprentice walk sums weights within each clone. Dividing each weight by K scales those sums exactly only when K is a power of two, so at another K a near-tie can move clone 0. Steps that pool over the frame, such as ranks, rakes and scales over every row, nest only approximately.

## Ids

Copy ids come from fixed decimal fields. `uk_runtime.copy_id_table` declares them in the order the spine applies them, least significant first, each with its ordinal bound and radix:

| Field | Ordinal bound | Radix |
| --- | --- | --- |
| `vintage` (reserved for pooled FRS vintages; nothing mints it) | 10 | `10**8` |
| `geographic_support` | 100 | `10**9` |
| `spi_support_channel` | 3 | `10**11` |
| `high_value_support` | 6 | `10**12` |
| `cgt_support_split` | 10,000 | `10**13` |
| `cgt_incidence_clone` | 2 | `10**17` |
| `cgt_residential_split` | 8 | `10**18` |

Ordinals run below the bound. Copy k in a field takes `id + k * radix` on all five id columns (`household_id`, `benunit_id`, `person_id`, `person_household_id` and `person_benunit_id`), and ordinal 0 means a row is not a copy in that field. So a copy's id depends only on its source id and its copy path, never on K. Its immediate parent is its id with that field's digits cleared (`copy_id_table.parent`). The executor's EXPAND receipt pairs copies with their parents that way, except for the support channel's copies, which name their parent in `{entity}_source_id`. On the clone-first spine that column names the geographic clone the copy was made from. It is lineage the executor reads, never the key of a count. The lineage readers read paths from the declared columns, never from ids.

## The geography key

`geography_household_key` is `household_draw_key` over the source (`frs`), the FRS vintage `frs_release.json` pins for the household's `source_year`, its true source and its copy path. It leaves out a row's own (copy) ids, row order, the clone count, sample size, weights and financial values. Exact int64 ids above `2**53` stay distinct.

Two stages write it:

- `geographic_identity`, right after `geographic_expand`, keys every clone. Clone 0's path is empty and clone k's is `(("geographic_support", k),)`. So clone 0's key is the key a one-clone spine gives its household, and a larger K never changes an existing key. `atomic_area_assign` draws each clone's atomic area on its key, and the larger geographies derive from that area.
- `spi_copy_areas`, right after `spi_housing_shell`, which draws the copies' council-tax band, gives every SPI copy and income band donor a key over its own path on the support and geographic branches. It then draws the copy's area within its region by the households of that band, or by all households where the region holds none of it. Until then a copy repeats its parent's key.

Copies made later, by `high_value_home_support`, `cgt_support_split`, `cgt_incidence_clone` and `cgt_residential_split`, keep their parent's key. They also keep its area, ladder, local authority, BRMA, rural fuel duty relief flag, council tax and rebate. The transferred-phase gate `uk_copy_area_agreement` holds each of them to its parent on those columns, its region and its clone index, and each support-channel copy to its parent on region and clone index. So the key names an area draw, not a row. A household that has later copies shares its key with them, and a release's keys are not unique per row. A household's own identity is its true source and copy path.

## Shared lineage readers

The country-neutral readers in `microcosm.build.source_lineage` take the lineage declaration as a parameter, so another country can declare its own lineage and reuse them:

- `record_sources` gives every record of any entity its true source through household membership. Every UK floor counts distinct sources through it (`receipt_counts.record_source_households`), so a household's clones and copies count once. A stage that may run before the first copy step says so (`before_copies`). On the clone-first spine that step is `geographic_expand`, which writes the source column. Where a receipt counts a set that could grow with K, such as the sources a stage flags in any clone, it counts among clone 0's rows, so it reads the same at every clone count.
- `copy_paths` reads every household's ordered copy path from the declared columns, never from ids, weights or values. `geographic_identity` and `spi_copy_areas` read it with the branches they require. `household_lineage.uk_household_clone_paths` is the UK's strict entry, which the Carer's Allowance and childcare copy take-up stages read.
- `replica_keys` and `fold_replicas` keep one training row per true source and path without the merged branches, with the merged rows' weights summed. The UK's folds merge the geographic branch alone. Through the UK entries each variant trains on its lowest-id clone, whatever the row order. `ReplicaFold.key_group` codes each selected record's key. The FRS-trained fits fold their training rows through the UK entry (`household_lineage.fold_geographic_clones`; `collapse_geographic_clones` returns its rows and weights alone): the SPI stage-2 and reference forests, the housing shell, the UC reporter redraw, the UC capital donor cells, the salary-sacrifice forest and the SPI employment shell. UC capital counts its cells' donors in survey records, the distinct keys among a cell's rows: a record's geographic clones count once, and every other copy counts as a record of its own. On a frame of more than one geographic clone, each of these stages records the fold in its evidence as `geographic_fold` (`household_lineage.geographic_fold_receipt`). It gives the clone count, the true source households the fit selected and those whose clones differ on what it reads, rounded as every receipt count is. It gives the share that differ, floored to the hundredth, only where ten or more of the selected sources differ and none or ten or more do not. On the clone-first spine geographic clones precede enrichment, so each fit sees each survey record once wherever its clones agree on what it reads. A frame of one clone keeps its rows and weights value for value.
- Each fit passes what it reads (its predictors and targets, or the columns they derive from), and clones fold only where they agree on it. Clones that differ, through a draw keyed on their own ids or a stage run per area, stay rows of their own. Each variant carries the summed weight of the clones that share it, so the fit sees every clone's value at its share and nothing is dropped. From the support channel on, each clone's people carry their own `person_source_id`, the key the SPI income stage's transport trains on, so clones that differ train under keys of their own. The stage refuses a frame whose differing clones share a key after the fold, as clones made after the support channel would.

## Releases

The release boundary drops no lineage column on either line, except that the national line drops the three clone indices, which are constant at one clone. The dense release keeps each entity's clone index as the lineage of its clones. A release renumbers ids into the engine's range, one plus the rank of each build id, and keeps each build id beside it (`person_build_id`, `benunit_build_id` and `household_build_id`). A release id depends on which rows the release holds. A build id depends only on its row's survey id and its household's copy path. `{entity}_source_id` names build ids, not release ids.

## The pure projector

`uk_runtime.atomic_household_lineage.project_atomic_household_keys(roots, *, steps, final_ids)` projects supplied original-source identities through explicit expansion and selection records to keys in the same encoding (`household_draw_key`). It adds no assignment algorithm, data reader, issuance registry or build host. No build calls it. The spine's stages key households themselves. The dense line's identity node, which its tests held to this projector, retired when the dense line began to build on the spine's clones.

It returns a detached two-column table: int64 `household_id` and nullable-string `geography_household_key`, in the requested final order. Roots have exactly `household_id`, `source`, `source_vintage` and `source_household_id`. An empty selection is a valid description, not evidence that an empty dataset is useful or releasable.

Steps are immutable descriptions:

- `HouseholdExpansion(branch, before_ids, after_ids, parent_pairs, child_ordinals)` carries exact household axes, the household part of the shared executor's `receipt["expand"]`, and explicit child ordinals. Its branch is one of the six, and the steps come in path order. Each arriving household needs one known preceding parent and one ordinal. Two children of the same parent cannot reuse an ordinal in the same branch.
- `HouseholdSelection(before_ids, after_ids)` explicitly removes households while preserving their keys. It does not check any sampling rule.

Each before-axis must have exactly the previous step's household IDs. Expansion cannot remove incumbents; selection cannot add households. Unknown or null parents, repeated or reordered branches, incomplete parent/ordinal coverage, duplicate keys and a mismatched final axis refuse. Axis ordering may change, because ordering is not identity. The function never guesses ancestry from ID arithmetic, support flags or model values.

Because it takes steps in path order, the projector cannot describe a spine's history. The spine clones first and makes high-value copies before the CGT support split, and the projector refuses either order as a reordered branch.

These checks establish only the internal consistency of supplied descriptions. A dataclass, DataFrame or serialized receipt does not prove its source, and a coherent invented history can pass. The helper is not an admission gate. Its output is the key column the UK's `uk_atomic_assignment_definition` and the shared operator's `assign_atomic` read.

## Verification scope

Tests on invented data hold each piece. None reads survey data or a licensed file.

- `test_uk_geographic_expand.py` and `test_uk_geographic_identity.py` hold the clones, their ids, indices and source columns, and their keys, which equal `household_draw_key` itself, are unique and stay put as K grows.
- `test_uk_household_lineage.py` runs the strict reader on the support channel's, CGT and geographic branches at one clone and at two, with its refusals. `test_uk_high_value_home_support.py` reads the high-value branch.
- `test_uk_spine_geography.py` holds the support channel's copies to keys and areas of their own. `test_uk_stage_health.py` and the shared `test_copy_agreement.py` hold later copies to their parents.
- `test_source_lineage.py`, `test_uk_geographic_clone_collapse.py` and `test_uk_receipt_counts.py` hold the readers, the folds and the counts, and `test_uk_copy_id_table.py` the id fields.
- `test_uk_clone_first_nesting.py` runs the real expansion, copy and draw stages at several clone counts up to 15. It holds clone 0 to the one-clone frame, exactly or within stated tolerances, and the CGT support split only approximately. `test_h2_clone_first.py`, which runs after merge, runs the H2 spine at one clone and at four. Its keys are unique through `area_brma`, the support channel's copies repeat their parent's key until `spi_copy_areas` gives each its own, and later copies carry their parents' keys.
- The projector's tests (`test_uk_atomic_household_lineage.py`) cover paths through the support channel, the CGT steps and the geographic branch, explicit selection, exact large IDs, source and ordinal changes, stable keys as rows reorder and households are added, refusal of unknown or colliding ancestry, and detached output. A real shared CREATE/EXPAND/FILTER graph supplies the household lineage receipt; it runs twice, and the second run must take every node from the store. Its projected keys feed the shared operator, as the spine's area stages run it: each household's area is the same in any row order, and the derived ladder passes the shared validation.
