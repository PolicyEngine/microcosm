# US target nesting across geography levels

A target for a small area must agree with the target of the area that
contains it. This page says how the US build makes the rows of a lower
geography level from a control at a higher level, and where each choice is
declared. The tracking issue is microcosm#1760. The work so far changes no
target value.

## What is decided, and what the code applies

- **Ruled, 27 Nov 2026.** The Historic Table 2 state row is the state parent
  of a district row on every surface, and a parent that it does not contain
  is not removed. The nine decisions of
  [US target nesting rule: design and rulings](us-target-nesting-rule-design.md)
  are accepted. The declaration is one general model in the shared operator,
  with no construct for one country only.
- **In the code.** The US nests its rows on that model: one operator, one
  declaration in the UK's key names, one set of metadata keys, one receipt
  format. The values are the values the tools made before.
- **Not in the code yet.** Each ruled change of a target. Those need the
  Chronicle feed after PolicyEngine/chronicle#451, and a re-pin of the feed.

## One operator, one declaration

The US uses the cross-grain operator that the UK uses,
`microcosm.build.cross_grain`. The operator groups the rows of one quantity,
finds the control of each lower leg, divides the control among the leg's rows,
checks that the rows sum to the control, and writes a receipt. It changes
target values only, never loss weights.

| Part | Where | What it holds |
|---|---|---|
| The operator | `microcosm.build.cross_grain` | Grouping, the control of each leg, bridges and bridge selectors, the bound of a bridge, the members of a leg, the division, the sum check, the receipt, and the metadata keys of a reconciled row. Shared with the UK |
| The row-level pass | `microcosm.build.cross_grain_controls` | `reconcile_rows` runs the operator on target rows and says what it did with each row. `sum_of_lower_rows` is the arithmetic of a derived control. The three allocators stay until the feed re-pin. No country in it |
| The US declaration | `us/cross_grain_declarations.json` | The US choices, under the key names of `uk/cross_grain_declarations.json` |
| The reader | `us_runtime/cross_grain_declarations.py` | Typed views of the declaration, and the coverage check |
| The US adapter | `us_runtime/target_nesting.py` | Which specs or facts are the rows of each pass, the checks that are US rules, and the specs made from the operator's answer |

The US levels are the feed's own geography levels: `country`, `state`,
`congressional_district`, `county`. Districts and counties both divide a
state. They do not nest in each other, and only `country` and `state` hold
controls.

## The model

The model has six things. Each one is general: a country declares it, and the
shared code runs it.

| Thing | What it is | In the US declaration |
|---|---|---|
| Measurement | Two targets with the same measurement at two levels are one quantity. Nothing else is declared | `same_measurement`: the ladder population of a state and of its counties |
| Bridge | Two targets with different measurements that measure one quantity. A bridge names its sides by id, or by a selector that makes one bridge for each pair of targets | `bridge_selectors`: Historic Table 2 state rows over district-file rows; state SOI rows over county-file rows |
| Derived control | A control that no source publishes as one row | `derived_controls`: a state's households are the sum of its county estimates |
| Bound | The limits on the factor of a bridge, around one or around the median factor across the legs, and the action out of bound: `refuse` the pass or `unbind` the leg | Both selectors: +0.25 and -0.20 around the median, `unbind` |
| Members | The members that each leg must hold. A member can stand for two geographies | `members`: the districts of the current plan; the counties of the ladder, with one combined cell |
| Not reconciled | A pair of levels that is not scaled, with a reason | `relations` (six `subset` pairs) and `local_routes` (three families with no higher control) |

`bridges` (by id) and `partitions` are empty in the US file.

### What the bound of a bridge does

The district bridge and the county bridge declare the same bound.

The factor of a leg is its control over the sum of its rows. A factor mixes a
part that every state shares (coverage, aging) with a part of one state. The
median across states takes the shared part out. A state whose factor is more
than 1.25 times the median, or less than 0.8 times it, is one where the two
publications disagree about that state. The operator unbinds that leg: its
district or county rows leave the surface, the receipt names the leg, and its
state row stays. On the pinned inputs that is four district legs and no
county leg.

## The four passes

| Row of the issue | Declared pass | Tool function | The control | The rows |
|---|---|---|---|---|
| 5 | `historic_table_2_state_vs_district_file` | `us_acs_local_cd_surface.state_cd_soi_surface` | The Historic Table 2 state row | District-file rows of the state |
| 6 | `state_soi_vs_county_file` | `us_acs_local_county_surface.state_county_soi_surface` | The broad state SOI row that exists | County facts of the state. One declared combined cell covers a county with no IRS row |
| 7 | `acs_households_state_is_the_county_sum` | `us_acs_local_county_surface.acs_household_surface` | The sum of the county estimates. It is a new state target | The county estimates |
| 8 | `ladder_population_state_vs_county` | `us_acs_local_county_surface.with_county_populations` | The state total of the 2020 PUMA ladder | The ladder's county overlaps, summed by county |

Each tool function keeps its name and signature. Each one calls the adapter,
and the adapter calls one function, `run_pass`, which builds the rule from
the declaration and runs the row-level pass.

The four passes do not yet run on one joint surface. Until the feed re-pin
each pass has its own allocator, and the county rows are made at a later
stage than the district rows.

### What the operator checks

- **Members.** A leg must hold exactly its declared members: the districts of
  the current plan, or the counties of the ladder (less a county that a
  combined cell covers). A declared leg with no row is refused in the same
  way.
- **The bound** of the district bridge, as above.
- **Arithmetic.** A total that is not finite, a lower sum below `1e-9` of the
  control, a control and rows of opposite sign, and rows that do not sum to
  the control within `1e-9` are refused.

### What the adapter checks

These are US rules, with US texts.

- **District SOI.**
  - The registers of defective, unparented and excluded district measures
    (`registers` in the declaration).
  - A state with one district on the source plan has no district rows.
  - The rows of a block and their parent must materialize in the same way.
  - A concept takes its state parent from one source in every state.
  - The surface is checked again after the pass: each district block must add
    up to a bound state parent (`closure_rtol`).
- **County population.** The operator refuses a county with no state and a
  state with no positive county population.
- **County SOI.** Each fact is an all-return fact of a tax year, finite and
  not negative, with the unit of its parent. Return count and adjusted gross
  income need a state parent in every state. A block needs a positive sum.
  A combined cell joins two counties of one state. A county that no row
  covers is refused, unless its leg is unbound and the receipt names it.
- **County households.** Every county of the ladder must carry an estimate,
  each state needs a Census name and a positive sum.

## What stays until the feed re-pin

`until_the_feed_repin` in the declaration holds what goes when Microcosm
pins the feed after PolicyEngine/chronicle#451.

- **The allocators.** The operator's own rule is
  `row x control / numpy sum of the rows`. The US rows were made with
  `math.fsum`, and the county rows with a residual rule. So a rule can name
  an allocator (`CrossGrainRule.allocator`) that divides one control among
  its rows. With no allocator the operator runs its own rule, with no float
  operation added or removed; the UK declares none. The US declares one for
  each pass, to keep the values bit for bit. At the re-pin they go, and
  district and county values move in the last bits.

  | Allocator | Rule | Pass |
  |---|---|---|
  | `proportional_fsum` | `row x control / exactly rounded sum` | District SOI |
  | `share_residual_to_largest` | `control x share`; the rounding residual of the shares, then of the products, goes to the largest row | County SOI, county households |
  | `scale_residual_to_largest` | `control x row / total`; the rounding residual goes to the largest row | County population |

- **The level bridges.** Six concepts are only in the district file. Their
  state parent is the district file's own state row, multiplied by a sibling
  concept's ratio of the two files. One function of the adapter makes these
  parents, as before. The rows under them are tied by their own measurement
  (`cross_grain_control` reads `exact_signature`), and a bridged state row
  reads `level_bridge:<sibling>`. After the re-pin Historic Table 2 holds the
  six parents, and this function goes.
- **The compile rules of rows 1 to 4**, which stay in the compile:
  - Rows 1 and 2 (capital gains, taxable interest): each row is multiplied by
    one factor, `control / published national total`.
  - Row 3 (state AGI bands): a partition inside one area.
  - Row 4 (the district file's own state row): shared code with its rule as
    data (`reconcile_ledger_target_hierarchy`). Its factor is 1 on the pinned
    feed.
- **The known gaps.** The SOI families with two national rows
  (microcosm#1243), and state age rows against state totals (microcosm#1528,
  #1526, #1531).

## Metadata keys and receipts

A row that the operator reconciles keeps the operator's keys. A UK row and a
US row have the same ones.

| Key | What it holds | The key before |
|---|---|---|
| `cross_grain_parent` | The name of the control row | `state_cd_parent_target_name`, `state_county_parent_target_name` |
| `cross_grain_value_before` | The row's value before the pass | `state_cd_cd_file_value`, `state_county_source_value` |
| `cross_grain_factor` | The factor of the row's leg | `state_cd_rebase_factor`, `state_cd_level_bridge_factor` |
| `cross_grain_control` | What reconciled the row: a bridge id, `exact_signature`, or `level_bridge:<sibling>` | `state_cd_parent_basis`, `state_cd_level_bridge_sibling` |
| `cross_grain_members` | The counties that a combined cell stands for | `state_county_cell_members` |

These old keys have no new one, because another place holds the value: the
parent's value and the sum of the rows (the parent row, and the operator's
receipt), the share of a county row (its value over its parent's), the
vintage rule and the basis of a combined cell (the declaration and the
receipt).

The readers of the keys read the new ones: the loss weights
(`us_runtime/target_loss_weights.py`), the hold-outs and their pro-rata
baselines, and the county reports. The dated runs under `experiments/` read
registries of their own date and are not changed.

Receipts:

- **The surface receipts** (`soi_surface` in `materialize_rss.json`, and
  `county_surface.json`) keep their counts, drops and summaries. Each one
  names its bridge under `bridge`, with the bound and the legs that the
  bound unbound. The district receipt also lists the level bridges it
  distrusts under `level_bridge_out_of_band`, and the factors under
  `factor_by_measure`.
- **The operator's receipt** goes to `target_nesting.json`, which the
  materialize stage writes beside its other files. It holds one entry for
  each pass: the pass id, the allocator, and the operator's receipt (each
  group and each leg with its control, its old and new totals and its
  factor). No stage reads the file, and it is not in the run identity.

## What differs from the code before

On valid data the rows and their values are the same. These things differ.

1. **Keys and receipt fields**, as above. So the bytes of the `state_cd`
   registry, roles and receipts moved, and their pins were measured again.
   `test_us_target_nesting_district.py` holds each row of that surface on the
   pinned feed to the bits of its old value.
2. **Refusal texts.** A refusal that the operator makes has the operator's
   text: a leg that does not hold its members, a sum of zero under a control,
   opposite signs, a county with no state.
3. **One check is gone:** that the district file's own state row is the sum
   of its district rows within `1e-6`. Row 4 of the compile makes that sum
   hold before this pass runs, and the district file's state row is not a
   control of a concept that Historic Table 2 holds.
4. **One refusal is new:** a concept that takes its state parent from
   Historic Table 2 in some states and from the district file in others. The
   old code accepted that. It does not occur on the pinned feed.
5. **With the bound turned off** (only tests do that), a concept whose
   Historic Table 2 level is zero in every state keeps district rows of
   zero. The old code dropped them, because such a concept has no median
   factor.
6. **One rule is new for the county SOI rows:** the bound of the county
   bridge. The code before had none. It unbinds no leg on the pinned inputs.
7. **Three cases that the oldest code computed are refused**, as since the
   first move to the operator: a block whose rows sum to a value that is not
   zero but is below `1e-9` of its control, a county SOI block under a
   negative state row, and a control of `-0.0`. None occurs on the pinned
   feed.

## The county rows on the pinned inputs

`test_us_target_nesting_county.py` reads the real inputs when three
environment variables name them: `MICROCOSM_US_CHRONICLE_FACTS` (the pinned
feed), `MICROCOSM_US_PUMA_LADDER_ARTIFACT` (the pinned PUMA ladder) and
`MICROCOSM_US_CHRONICLE_COUNTY_FACTS` (the pinned county facts file of
`us/chronicle_county_facts.json`, 922 MB). Without them the two tests skip.

On those inputs the three county passes give the rows of the code before,
with the same value bits:

| Pass | Rows |
|---|---|
| County population, 2020 counties | 3,143 |
| County population, 2023 county-equivalents | 3,144 |
| County SOI (return count, adjusted gross income) | 6,286: two for each of 3,143 cells. One cell stands for Kalawao and Maui |
| County households | 51 state rows and 3,144 county rows |

The same test file pins the bytes of those rows.

The county SOI bridge declares the bound of the district bridge (ruled 27 Nov
2026). Its factors (the state row over the sum of the county facts) were
measured on those inputs:

| Measure | Legs | Median factor | Lowest and highest, as a ratio to the median | Legs outside 0.8 to 1.25 |
|---|---:|---:|---|---:|
| `adjusted_gross_income` | 51 | 1.1207 | 0.998 to 1.052 | 0 |
| `return_count` | 51 | 1.0000 | 0.9997 to 1.0005 | 0 |

The median of 1.12 for adjusted gross income is a gap that every state has.
It is consistent with the aging of an amount from tax year 2022 to the target
year; a count is not aged. The states farthest from the median are Montana
(1.052) and Wyoming (1.038). So the bound unbinds no county leg on these
inputs, and it changes no county value.

## The coverage test

`test_us_cross_grain_coverage.py` reads the declaration. With the pinned feed
it compiles the registry and checks that each family at a level below
`country` is covered by a selector, a derived control, a same-measurement
entry, a relation, a local route, a compile rule or a known gap; that no
declaration names rows that are gone; that each compile rule's marker is on
rows of its family and levels; and that no `subset` part is larger than its
whole beyond the declared `max_excess`.

It is a test of the declaration. It refuses nothing in a build and it is not
a release gate.

## The UK

The UK graph hashes `cross_grain.py`. The edits to that file make the UK
target nodes, and every node after them, compute again on the next UK run.
The outputs are the same: `test_cross_grain_allocator.py` compares the
operator with a copy of the file before any edit, bit for bit, on 400 seeded
surfaces, and it pins the list of definitions that the branch changed or
added. The UK declares none of the new options. The UK graph does not hash
`cross_grain_controls`.

## Not done yet

After PolicyEngine/chronicle#451 and the feed re-pin:

- the ruled target changes: Historic Table 2 parents for the six concepts,
  district capital gains under the Historic Table 2 state row, the `full`
  surface nests its district rows, row 4 goes;
- the allocators go, and the four passes run on one joint surface;
- the derived control of the kind `scaled`, for the national controls of the
  state rows (decision 2);
- the ACS district rows (decision 7) and the ladder under the state estimate
  (decision 4), each in its own PR.
