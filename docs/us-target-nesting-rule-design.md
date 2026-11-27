# US target nesting rule: design and rulings

**Status: the rule is ruled, and no code applies it yet.** The nine decisions
of section 4 were accepted on 27 Nov 2026. Section 5 says how the rulings
are declared: its principle is ruled, and its model is the design that
follows from it. This page is the design step of Stage B of
microcosm#1760. Stage A, which is on this branch, changes no target
([US target nesting](us-target-nesting.md)). The target changes wait for the
feed re-pin after PolicyEngine/chronicle#451.

Each number is one of two kinds:

- **measured**: run on the pinned feed (Chronicle `41b2fb2b`), compiled at
  2024 as the release does;
- **read**: read in the code or in an issue.

## 1. What is decided

| Date | Ruling | Where |
|---|---|---|
| 26 Nov 2026 | Stage A uses the cross-grain operator that the UK uses. No target changes for now, so that PR #1719 keeps its behavior. | microcosm#1760 |
| 27 Nov 2026 | The Historic Table 2 state row is the state parent of a district row on every surface, for every parent that it contains. A parent that it does not contain is not removed. | [comment on #1760](https://github.com/PolicyEngine/microcosm/issues/1760#issuecomment-7217969778) |
| 27 Nov 2026 | The nine recommendations of section 4 are accepted. | This page |
| 27 Nov 2026 | The way the rulings are declared must follow how the UK declares its own (section 5). | This page |
| 27 Nov 2026 | Clean country-agnostic code comes first: the simpler and the more general, the better, with no exceptions for one country. A UK re-key is accepted for it. | This page, section 5 |

## 2. The rule

Nine statements. Each decision in section 4 is one of them.

1. **One level holds the level value of a quantity.** It is the highest
   geography level at which a current source publishes a complete total of
   the same concept.
2. **Every level below it holds shares.** The operator divides the parent
   among the lower rows, so they sum to it. The lower source supplies the
   shares only.
3. **Levels stack as a chain.** Country, then state, then district and
   county. Districts and counties both divide a state. They do not nest in
   each other. A lower leg takes its nearest control.
4. **The same parent on every surface.** A surface that binds a lower row
   uses the parent that the rule names. *(Ruled for state to district.)*
5. **A parent that the higher source lacks is kept** from the source that
   has it. Its control is then the lower source's own total, or the sum of
   the lower rows. *(Ruled.)*
6. **Lower rows that do not cover the parent keep their shares of their own
   file's total.** When the lower rows are not all the parts of the parent
   (the 50 states and DC against a US row that also counts other areas),
   each lower row takes one factor: the control over its own file's total.
   The lower rows are not scaled to sum to the control.
7. **Two sources of one concept nest. Two concepts do not.** Lower rows from
   another source than the parent are shares of the parent when both measure
   the same concept. When they measure different concepts, both stay as
   levels, and the pair is declared with a reason.
8. **Each mechanism declares its bound and its members.** A block whose
   factor is out of its bound is dropped and recorded, and the parent stays.
   A leg must hold exactly its declared members, or the compile refuses. A
   leg with one member is a copy of its parent and does not bind.
9. **Nesting changes values, never weights.** Whether a parent and its
   children both stay in the loss is a rule of the loss (microcosm#1729).

## 3. What each family does today, and what the rule gives

"Sum" is the sum of the lower rows over the higher row, in the compiled
registry of the pinned feed. **(measured)**

| Family | Levels bound | Today | Sum (measured) | Under the rule |
|---|---|---|---|---|
| IRS SOI, capital gains and taxable interest | country, state, district | Rows 1 and 2: each row times one factor onto the newer national control | by construction | Statement 6. No change in values. The arithmetic moves to the operator |
| IRS SOI, the 14 families of microcosm#1243 | country (two rows), state | Two national rows for one quantity. No rule | Historic Table 2 states over the Historic Table 2 US row, 61 pairs: median 0.995. Lowest all-returns pair: 0.981 | Statement 6: the newer Table 1.x row is the control, the state rows keep their shares, and the Historic Table 2 US row stops binding |
| IRS SOI, district file | state, district | Row 4: the district file's own state row (factor 1). Row 5 (`state_cd`): the Historic Table 2 state row | district file states over its own US row: 1.000 (40 concepts) | Statements 2, 4 and 5. Ruled |
| IRS SOI, county file | state, county | Row 6: shares of the Historic Table 2 state row | not measured (the county facts are not on this machine) | Statement 2. No change |
| Census population by age | country, state | 18 age bands at each level, from one vintage. No rule | 1.0000 in each of the 18 bands | Statement 2 holds already. No rule needed |
| Ladder population (local build) | state, district, county | Totals from the 2020 ladder. Row 8 scales counties to the state | 1 by construction | Decision 4 |
| ACS households (county rung) | state, county | Row 7: the state target is the sum of the county values | 1 by construction | Decision 5 |
| SNAP | country, state | No rule | households 0.9990, benefits 0.9980 (51 states) | Decision 3 |
| Medicaid and CHIP | country, state | No rule | Medicaid 1.0038 (51 states); Medicaid with CHIP 1.0000 (50); CHIP 0.5875 (31) | Decision 3 |
| TANF | country, state | No rule | 0.9096 (29 states) | Decision 3 |
| ACA, SSI by state, state income tax | state | No national row of the same role | none | Nothing to nest |
| ACS district age and households | district | Not in the fit (microcosm#1731) | none | Decision 7 |

CHIP has 31 state rows because a reviewed rule drops the 20 M-CHIP states.
TANF has 29 state rows in the compiled registry.

## 4. The nine decisions

**Ruled, 27 Nov 2026: the recommended option of each decision is accepted.**
The other options stay on the page as the record of what was considered.

### Decision 1. The level holder (statements 1 to 3)

The level value of a quantity is held by the highest level that has a
current source with a complete total of the same concept. Lower levels hold
shares.

**Recommended: yes.** It is the proposal of the issue, and the 27 Nov ruling
is one case of it.

### Decision 2. National to state for the SOI families (statement 6)

This covers rows 1 and 2 and the 14 families of microcosm#1243.

- **(a) One factor.** Each state row is multiplied by the national control
  over the US total of its own file. The states keep their shares. Returns
  from other areas stay out of the states. Rows 1 and 2 do this today.
- **(b) States scaled to sum to the control.** The states then take the
  returns of other areas too. At the median that is 0.5% of a quantity. The
  largest gap of an all-returns concept is 2% (dividends). **(measured)**

**Recommended: (a).** On the operator, the control of the states is a
derived control: the national control times the states' share of their
file's US total. Where a newer Table 1.x row is the control, the Historic
Table 2 US row stops binding, as it does for capital gains and taxable
interest. Each of the 14 concept matches must be verified first, as
microcosm#1243 says.

### Decision 3. Program families: SNAP, Medicaid, CHIP, TANF

- **(a) No scaling.** The national row and the state rows are cells of one
  publication of one vintage. Declare them `same_source`, and report the sum
  of the states over the national row in the nesting receipt.
- **(b) Scale the states to the national row** where all states are present
  (SNAP, Medicaid).

**Recommended: (a).** Where all states are present, the gaps are 0.1% to
0.4%. Their causes are not established here: SNAP's national row may count
areas outside the 50 states and DC, and one state's Medicaid row is from
another month. CHIP and TANF do not have all states, so scaling them would
put the missing states' mass on the present ones.

### Decision 4. State population on one vintage (microcosm#1526, PR #1531)

Today the national build fits the Census estimates by age for each state.
The local build fits state, district and county totals from the 2020 ladder
and no age row.

- **(a) The current Census estimate holds the state level.** The ladder
  supplies the shares of districts and counties in that state total. The
  local build can then bind the state age rows too.
- **(b) The ladder keeps its own levels.**

**Recommended: (a).** It changes the local state totals. Which vintage and
which age groups is the decision of PR #1531, not of this rule.

### Decision 5. County households, row 7

- **(a) Keep.** The state target is the sum of the county values. It is the
  case of statement 5: no higher source of the same concept is bound.
- **(b) Bind a state level from a current source**, and let the counties
  divide it. It needs a state households fact of the same concept. Whether
  the feed holds one is not checked.

**Recommended: (a) now, and (b) when the state fact exists.**

### Decision 6. Bounds and members (statement 8)

- **The bound.** A mechanism between two sources declares the band around
  the median factor (1.25 for district SOI today). A mechanism inside one
  source declares none (ladder population, county households).
  **Recommended: yes.** For county SOI, which has no bound today: add the
  band as a report first, and decide to enforce it after the factors are
  measured on the county facts.
- **Members.** A leg holds exactly its declared members: the district plan
  of the crosswalk, the counties of the ladder. A declared combined cell
  covers a county with no IRS row. An area with one district does not bind
  at district level. **Recommended: keep all three, as today.**

### Decision 7. ACS district age and household rows (microcosm#1731)

The district rows come from a survey. The state rows come from the Census
estimates.

- **(a) Shares of the state row**, with the band of decision 6.
- **(b) Levels.**

**Recommended: (a) for the age rows.** The household rows follow decision 5.

### Decision 8. The parent in the loss (statement 9)

When a parent and all its children bind, the loss counts the parent two
times.

- **(a) Both stay.** The weight rule by level of microcosm#1729 sets their
  weights.
- **(b) The parent leaves the loss** where all its children bind.

**Recommended: (a).** It keeps this rule about values only.

### Decision 9. One allocator, one key set, one receipt

At the re-pin the bytes of the registry move anyway.

- **One allocator**, `proportional_fsum`, for every mechanism. The two
  residual allocators of the county rows go. County values move in the last
  bits.
- **One set of metadata keys** in place of `state_cd_*`, `state_county_*`
  and `hierarchy_*`. Three readers change with it: the loss weights, the
  pro-rata baselines and the reviewed limitations.
- **One receipt**: `target_nesting.json`.

**Recommended: yes to all three, in the re-pin PR.**

**Ruled later on 27 Nov 2026: no allocator.** After the re-pin the US divides
a control with the operator's own rule, as the UK does (section 5.2).

### Rulings of 27 Nov 2026 on the steps

Ten calls came up when steps 1 and 2 of section 5.5 were written. Each is
ruled, with the recommended option.

1. **No allocator after the re-pin.** Decision 9 narrows from one allocator
   to none. Before the allocators go, check that no reader needs rows that
   sum exactly to their control.
2. **A concept takes its state parent from one source in every state.** A
   concept with a Historic Table 2 parent in some states and a district-file
   parent in others is refused. A concept that Historic Table 2 lacks in
   every state keeps its district-file parent.
3. **The check that the district file agrees with itself stays out.** The
   compile holds that sum before the pass, and after Chronicle PR #451 the
   file has no state row to compare.
4. **The county rows are compared and pinned on the pinned county facts
   file now**, not at step 3.
5. **Merge timing.** This branch does not merge while a UK release is between
   its first build and its sign-off, because the UK graph re-keys.
6. **PR #1755 merges first.** This branch then takes its changes.
7. **The rulings are posted on microcosm#1760**, after the text is approved.
8. **Medicaid stays a `subset`** with an allowed excess of 1%, and Medicaid
   with CHIP 0.1%.
9. **Two readings of the first ruling, for step 3.** The `full` surface nests
   its district rows in the Historic Table 2 state row. District capital
   gains nest in the Historic Table 2 state row.
10. **Decisions 4 and 7 go in their own PRs.**

## 5. How the rulings are declared: one general model

**The principle (ruled 27 Nov 2026).** Clean country-agnostic code comes
first. The simpler and the more general, the better. No construct exists for
one country only. The UK graph may re-key for it: its nodes compute again,
with the same outputs.

So this section does not add US parts beside the operator. It states one
model for every country, in the shared operator, and it lists what goes.

### 5.1 The model

Six things. The first, the second and the last are the UK's constructs as
they are. The other three make general what each country does in its own
code today.

| # | Thing | What it is | Today |
|---|---|---|---|
| 1 | **Measurement** | Each target has a measurement. Two targets with the same measurement at two levels are one quantity. Nothing else is declared | UK: yes. US: Stage A makes one for each block |
| 2 | **Bridge** | The bridge between concepts: two targets with different measurements that measure one quantity, with a `reason`. A bridge names its sides by id, or by a selector that makes one bridge for each measure | UK: `bridges`, by id. Its 24 band bridges are written one by one |
| 3 | **Derived control** | A control that no source publishes as one row. Two kinds: `sum` (of declared rows) and `scaled` (a value times a numerator over a denominator) | UK: three kinds in UK code (`fanout_sum_bridges`, `sum_bridges`, `band_bridges`), all sums. US: three helpers |
| 4 | **Bound** | Limits on the factor of a bridge (`max_factor_shift`, `min_factor_shift`), around a reference: one, or the median across the legs of the bridge. And the action out of bound: `refuse` or `unbind` the lower rows of that leg | UK: around one, refuse, in the operator. US: around the median, unbind, in US code |
| 5 | **Members** | The members that each leg must hold. A member can stand for two geographies. A leg can be licensed empty | UK: a membership file and licensed empty legs. US: three checks in US code |
| 6 | **Not reconciled** | `relations` (`subset`, `independent`) and `local_routes`, each with a reason | UK: yes. US: the `inventory` routes of Stage A |

Partitions at one geography stay as they are. The operator runs one time on
one joint surface of all the bound rows, and it writes one receipt and the
`cross_grain_*` metadata keys.

### 5.2 What goes

| What | Why it goes |
|---|---|
| The allocator hook and the three allocators | Every country uses the operator's one rule: the row times the control over the sum of the rows. Stage A needs the allocators only to prove the move with the old bytes. They go at the re-pin. District and county values then move in the last bits. This narrows decision 9 from one allocator to none (ruled 27 Nov 2026) |
| The US keys `mechanisms` and `inventory` | The US file has the UK's keys only |
| The US checks before the operator (factor band, completeness, self-consistency) | They are things 4 and 5 of the model |
| The sibling-ratio bridge | After the re-pin, Historic Table 2 holds the six parents |
| The US metadata keys (`state_cd_*`, `state_county_*`) and the two US receipts | The `cross_grain_*` keys and the pass receipt |
| Four passes in four tool functions | One pass |
| Later, in the UK: `fanout_sum_bridges`, `sum_bridges`, `band_bridges` as three kinds | Each is a derived control of the kind `sum`. `max_index_shift` is a bound around a declared reference. This is a consolidation of its own, not part of this branch |

One option is not taken: an exactly rounded sum as the operator's one rule.
It does not depend on the order of the rows, but it would change UK values
in the last bits. A re-key is accepted; a change of UK values is not asked
for here.

### 5.3 Each ruling in the model

| Ruling | In the model |
|---|---|
| The level holder and the chain (decision 1) | `grain_precedence`, `control_grains` |
| One source at two levels: the ladder, the Census age bands | Thing 1. Nothing to declare |
| State to district SOI (ruled 27 Nov) and state to county SOI | Thing 2: one bridge selector for each, from the Historic Table 2 state rows to the district rows or the county rows. The measure ids that differ between the two files are named in it |
| National to state SOI: rows 1 and 2 and the 14 families (decision 2) | Things 2 and 3: a bridge whose control is `scaled`: the national value, times the sum of the state rows, over the US total of the states' own file. The `reason` holds the verified concept match |
| County households (decision 5) | Things 2 and 3: a bridge whose control is the `sum` of the county rows |
| The band of 1.25 (decision 6) | Thing 4: `max_factor_shift` 0.25 and `min_factor_shift` -0.20, around the median across the legs, with `unbind` |
| District plans, the combined cell, at-large areas (decision 6) | Thing 5 |
| SNAP, Medicaid, CHIP, TANF (decision 3) | Thing 6: `relations`, `subset` |
| ACA, SSI by state, state income tax | Thing 6: `local_routes` |
| State AGI bands (row 3) | A partition, `share_of_parent`, when row 3 moves |
| One key set, one receipt (decision 9) | The operator's own |

### 5.4 Sketches

The target ids are examples. A US target id is the record set and the
measure of a compiled row, as a UK contract target is one target with many
cells. The US has no written contract, so the measurement of a row comes
from declared metadata fields.

A bridge selector, with its bound:

```json
{
  "bridge_id": "historic_table_2_state_vs_district_file",
  "higher": {"grain": "state", "record_set_spec_prefix": "irs_soi.historic_table_2."},
  "lower": {"grain": "congressional_district", "record_set_spec_prefix": "irs_soi.congressional_district_"},
  "pair_on": ["source_measure_id", "agi_lower_bound", "agi_upper_bound", "filing_status"],
  "paired_ids": {"qbi_amount": "qualified_business_income_deduction_amount"},
  "bound": {"max_factor_shift": 0.25, "min_factor_shift": -0.2, "around": "median_across_legs", "out_of_bound": "unbind"},
  "reason": "The district file gives each district's share of its state. Historic Table 2 gives the complete state total (IRS district data guide)."
}
```

A bridge with a derived control of the kind `scaled`:

```json
{
  "bridge_id": "table_1_4_vs_historic_table_2_states:taxable_pension_income_amount",
  "control": {
    "derive": "scaled",
    "value": "irs_soi.table_1_4:taxable_pension_income_amount",
    "numerator": {"sum_of": "irs_soi.historic_table_2.state_broad_totals:taxable_pension_income_amount"},
    "denominator": "irs_soi.historic_table_2.us:taxable_pension_income_amount"
  },
  "lower_side": "irs_soi.historic_table_2.state_broad_totals:taxable_pension_income_amount",
  "reason": "The same Form 1040 line in both tables (microcosm#1243). The states keep their shares of the Historic Table 2 US total."
}
```

A pair that is not reconciled, in the UK's own shape:

```json
{
  "targets": ["usda_snap.state:total_benefits", "usda_snap.national:total_benefits"],
  "relation": "subset",
  "part": "usda_snap.state:total_benefits",
  "whole": "usda_snap.national:total_benefits",
  "reason": "Cells of one publication. The national row may count areas outside the 50 states and DC."
}
```

### 5.5 Steps

Each step is one or more ordered commits on this branch.

1. **The model in the shared operator.** Derived controls, the reference and
   the action of a bound, members, and bridge selectors. The UK declares
   none of the new options. A test holds the UK outputs to the operator
   before the change, bit for bit. The UK graph re-keys.
2. **The US on the model.** One US file in the UK's key names, and one pass
   on one joint surface. **The target values stay as Stage A left them. The
   metadata keys and the receipts change**, so the pinned bytes of the US
   registry move. The readers of the old keys move with them: the loss
   weights, the hold-outs and their pro-rata baselines, the county reports.
3. **After Chronicle PR #451: the re-pin and the ruled target changes.** The
   allocators go here.

Step 2 meets PR #1755 in two places: its pin test holds the registry bytes,
and its hold-out declaration reads the old key
(`state_cd_parent_target_name`, now `cross_grain_parent`). The PR that merges
second takes the other's change.

**Status, 27 Nov 2026.** Steps 1 and 2 are on the branch. Two parts of the
model are not built, because no rule in force uses them yet:

- the derived control of the kind `scaled`, which decision 2 needs;
- one pass on one joint surface. In step 2 the four passes call one function,
  each on its own surface: each pass keeps its allocator until the re-pin,
  and the county rows are made at a later stage than the district rows.

Step 2 also settled the two open points below. Medicaid is a `subset` with a
declared `max_excess`. No US surface needs rows that bind nothing yet.

Two points to settle when the code is written:

- **Medicaid.** The state rows sum to 1.0038 of the national row, so a
  `subset` relation needs a declared allowance, or the pair is `independent`.
- **Rows that bind nothing.** The UK puts cells that do not bind on the
  surface, so that they take their share of a control. The US has the same
  need for a state row that a surface leaves out of its loss.

## 6. What each decision changes

| Decision | Data | Code | Targets move |
|---|---|---|---|
| Ruled, 27 Nov | `us/cross_grain_declarations.json`, `us/fiscal_target_references.json` | Row 1 keeps district capital-gains rows as shares. The `full` surface nests its district rows | Yes, with the re-pin |
| 2 | A mechanism for national to state | A derived control. Rows 1 and 2 call the operator | The 14 families: yes. Rows 1 and 2: no |
| 3 | Inventory routes | A line in the receipt | No |
| 4 | A mechanism for the ladder under the state estimate | The local population rows | Yes |
| 5 | None now | None now | No |
| 6 | The bound of each mechanism | A report for county SOI | No |
| 7 | A mechanism for the ACS district rows | The rows of microcosm#1731 | New targets |
| 8 | None | None | No |
| 9 | Allocator names, key names | Three readers | Last bits, and key names |

## 7. Order

1. Steps 1 and 2 of section 5.5. They change no target value.
2. Chronicle PR #451 merges. Microcosm re-pins its feed.
3. One set of commits on this branch for the ruled decisions and the re-pin,
   with one build. Decisions 4 and 7 can come later in their own PRs.

## 8. Not measured

- The county factors. The pinned county facts file is not on this machine.
- Anything on a feed after Chronicle PR #451.
- The concept match of each of the 14 families of microcosm#1243.
