# US target nesting across geography levels

A target for a small area must agree with the target of the area that
contains it. This page says how the US build makes the rows of a lower
geography level from a control at a higher level, and where each choice is
declared. The tracking issue is microcosm#1760. Stage A, described here,
changes no target value.

## What Stage A decides: nothing

Stage A moves the mechanisms that exist to one operator and writes their
choices down as data. Each mechanism is declared as the code ran it before
the move. These decisions of microcosm#1760 are open, and Stage B applies
them:

1. Which level holds the level value, for each family.
2. Two sources at two levels: whether district rows from a survey are shares
   of a state row from estimates, or stay as levels.
3. Row 7: whether the state households target stays the sum of the county
   values, or becomes a state level that the counties divide.
4. The parent row in the loss, when all its children are in the loss too.
5. The factor bound: one declared bound for each bridge, as in the UK, or
   the median band of row 5.
6. Children that are not complete: district plans, a county with no IRS row,
   a state with one district.

After Stage A, each decision is an edit to the declaration, and for some an
allocator. It is not a new mechanism.

## One operator, one declaration

The US uses the cross-grain operator that the UK uses,
`microcosm.build.cross_grain`. The operator groups the rows of one quantity,
finds the control of each lower leg, divides the control among the leg's rows,
checks that the rows sum to the control, and writes a receipt. It changes
target values only, never loss weights.

| Part | Where | What it holds |
|---|---|---|
| The operator | `microcosm.build.cross_grain` | Grouping, the control of each leg, the division, the sum check, the receipt. Shared with the UK. |
| The helpers | `microcosm.build.cross_grain_controls` | Allocators, derived controls, the factor band, the completeness check, the self-consistency check. No country in it. |
| The US declaration | `us/cross_grain_declarations.json` | The levels, the levels that hold controls, each mechanism with its registers and reasons, and the inventory. |
| The reader | `us_runtime/cross_grain_declarations.py` | Typed views of the declaration. |
| The US adapter | `us_runtime/target_nesting.py` | Which specs or facts are the control and the rows, the refusals with their US texts, the old metadata keys and receipts. |

The declaration uses the key names of `uk/cross_grain_declarations.json` where
the meaning is the same: `grain_precedence`, `control_grains`,
`signature_fields`, `bridges` and `partitions`. The US levels are the feed's
own geography levels: `country`, `state`, `congressional_district`, `county`.
Districts and counties both divide a state. They do not nest in each other,
and only `country` and `state` hold controls.

### The allocator

The operator's standing rule is `row x control / numpy sum of the rows`. The
US rows were made with `math.fsum`, and the county rows with a residual rule.
On the pinned feed the numpy sum changes the last bit of 5,533 of the 21,743
district values. So a rule can declare an allocator
(`CrossGrainRule.allocator`), a function that divides one control among its
rows. With no allocator the operator runs the standing rule, with no float
operation added or removed; the UK declares none.

| Allocator | Rule | Used by |
|---|---|---|
| `proportional_fsum` | `row x control / exactly rounded sum` | District SOI shares |
| `share_residual_to_largest` | `control x share`; the rounding residual of the shares, then of the products, goes to the largest row | County SOI shares, county households |
| `scale_residual_to_largest` | `control x row / total`; the rounding residual goes to the largest row | County population |

The two residual allocators only reproduce the old county arithmetic. Stage B
replaces them with one allocator.

## The four mechanisms on the operator

Each one is declared under `mechanisms`. The tool function keeps its name and
signature and calls the adapter.

| Row of the issue | Mechanism | Tool function | The control | The rows |
|---|---|---|---|---|
| 5 | `state_congressional_district_soi` | `us_acs_local_cd_surface.state_cd_soi_surface` | The Historic Table 2 state row. For a concept that only the district file has: the district file's state row, multiplied by a sibling concept's ratio of the two files (`scaled_by_sibling_ratio`) | District file rows of the state |
| 8 | `state_county_population` | `us_acs_local_county_surface.with_county_populations` | The state total of the 2020 PUMA ladder | The ladder's county overlaps, summed by county |
| 6 | `state_county_soi` | `us_acs_local_county_surface.state_county_soi_surface` | The broad state SOI row that exists | County facts of the state. One declared combined cell covers a county with no IRS row |
| 7 | `county_state_acs_households` | `us_acs_local_county_surface.acs_household_surface` | The sum of the county estimates (`sum_of_lower_rows`). It is a new state target | The county estimates |

### What the adapter does before the operator runs

These checks are US rules. They run first, with the texts they always had, so
a refusal reads the same as before.

- **District SOI.**
  - The district rows of a block must be exactly the current plan's districts
    of the state (from the district crosswalk). A state with one district on
    the source plan has no district rows.
  - The district file's own state row must equal the sum of its district rows
    within `1e-6` (`self_consistency`).
  - A block whose ratio (control over the sum of its rows) is more than 1.25
    from its concept's median across states is dropped and recorded
    (`factor_band`). The Historic Table 2 state row stays.
  - The registers of defective, unparented and excluded district measures and
    the six level bridges are in the declaration, with their reasons.
- **County population.** Each state needs a positive total and at least one
  county with a positive population. Each county needs a state.
- **County SOI.** Each state block must hold every county of the ladder.
  A county with no fact is refused unless it is a declared combined cell.
  Return count and adjusted gross income must have a state parent in every
  state.
- **County households.** Every county of the ladder must carry an estimate,
  and each state needs a positive sum.

### What the operator refuses

The operator applies its own refusals to every block: a total that is not
finite, a lower sum below `1e-9` of the control, a negative factor, and rows
that do not sum to the control within `1e-9`. On valid data the result is the
old result. Three cases that the old code computed are now refused:

- a block whose rows sum to a value that is not zero but is below `1e-9` of
  its control;
- a county SOI block under a negative state row;
- a control of `-0.0`.

None occurs on the pinned feed. The pinned county facts hold return counts
and adjusted gross income only, and both are positive in every state.

## Receipts

- **The old receipts and metadata keys stay**, with the same bytes: the
  `state_cd_*` and `state_county_*` keys, the `soi_surface` receipt in
  `materialize_rss.json`, and `county_surface.json`. Other code reads these
  keys (the loss weights, the pro-rata baselines, the reviewed limitations).
- **The operator's own receipt goes to a new file**, `target_nesting.json`,
  which the materialize stage writes beside its other files. It holds one
  entry for each pass: the mechanism, the allocator, the number of blocks and
  the operator's receipt (each group and each leg with its control, its old
  and new totals and its factor). No stage reads the file, and it is not in
  the run identity.

One receipt format for both countries is a Stage B step, because it changes
the bytes of the registry.

## The inventory

`inventory` in the declaration lists each US family that two levels measure,
with one route:

| Route | Meaning |
|---|---|
| `mechanism` | The shared operator makes the rows (rows 5 to 8). |
| `compile_rule` | A pass of the compile makes them and stays where it is in Stage A (rows 1 to 4). |
| `same_source` | Both levels are sums of one source's cells. |
| `no_higher_control` | No row at a higher level measures the quantity. |
| `not_reconciled` | Two levels measure it and no rule ties them yet. The entry gives the reason and the issue. |
| `not_in_the_fit` | The rows are not compiled. |

### Why rows 1 to 4 stay in the compile

- **Rows 1 and 2 (capital gains, taxable interest).** Each row is multiplied
  by one factor, `control / published national total`. The rows are not scaled
  to sum to the control, and the control and the total are read from the
  facts, not from rows of the registry.
- **Row 3 (state AGI bands).** It is a partition inside one area. The
  partition is read from the facts, and only some of its bands bind.
- **Row 4 (the district file's own state row).** It is already shared code
  with its rule as data (`reconcile_ledger_target_hierarchy`, declared in
  `us/fiscal_target_references.json`). It sums with the builtin `sum`, formats
  its factor with 15 significant digits and writes `hierarchy_*` keys. The
  bytes of the `full` and `state_cd` surfaces depend on all three.

### Families that do not nest today

- The SOI families with two national rows (microcosm#1243).
- State age rows and state totals (microcosm#1528, #1526, #1531).
- The state rows of SNAP, Medicaid and CHIP, SSI and TANF against the national
  row of each family. This is not checked yet.
- District age and household rows from ACS, which are not in the fit
  (microcosm#1731).

## The coverage test

`test_us_cross_grain_coverage.py` reads the inventory. It checks that each
entry is well formed, that rows 1 to 8 each have one entry, and that each
mechanism has one. With the pinned feed it also compiles the registry and
checks that each family at a level below `country` has an entry, that no
entry names rows that are gone, and that each compile rule's marker is on
rows of the declared family and levels.

It is a test of the declaration. It refuses nothing in a build and it is not
a release gate.

## The UK

The UK graph hashes `cross_grain.py`. The edit that added the allocator makes
the UK target nodes, and every node after them, compute again on the next UK
run. The outputs are the same: `test_cross_grain_allocator.py` compares the
operator with a copy of the file before the edit, bit for bit, on 400 seeded
surfaces. The UK graph does not hash `cross_grain_controls`.

## Not in Stage A

- Every target change (Stage B of microcosm#1760): one allocator for the
  county rows, one receipt format, the ACS district rows, district and county
  rows in one run, the families of microcosm#1243, row 4 on the operator.
- One function for the two matrix splits (`split_carriers_into`,
  `split_state_rows`). They do no arithmetic on target values.
- A pin of the county rows on the pinned county facts file. The equivalence
  tests of the county rows use synthetic blocks.
