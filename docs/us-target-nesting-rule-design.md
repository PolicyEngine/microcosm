# US target nesting rule: design for rulings

**Status: a proposal. It is not in force, and no code reads this page.** It
is the design step of Stage B of microcosm#1760. Stage A, which is on this
branch, changes no target ([US target nesting](us-target-nesting.md)).

Each number is one of two kinds:

- **measured**: run on the pinned feed (Chronicle `41b2fb2b`), compiled at
  2024 as the release does;
- **read**: read in the code or in an issue.

## 1. What is decided

| Date | Ruling | Where |
|---|---|---|
| 26 Nov 2026 | Stage A uses the cross-grain operator that the UK uses. No target changes for now, so that PR #1719 keeps its behavior. | microcosm#1760 |
| 27 Nov 2026 | The Historic Table 2 state row is the state parent of a district row on every surface, for every parent that it contains. A parent that it does not contain is not removed. | [comment on #1760](https://github.com/PolicyEngine/microcosm/issues/1760#issuecomment-7217969778) |

Everything below is open.

## 2. The proposed rule

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

## 4. Decisions, each with a recommendation

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

## 5. What each decision changes

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

## 6. Order

1. Rulings on section 4.
2. Chronicle PR #451 merges. Microcosm re-pins its feed.
3. One set of commits on this branch for the ruled decisions and the re-pin,
   with one build. Decisions 4 and 7 can come later in their own PRs.

## 7. Not measured

- The county factors. The pinned county facts file is not on this machine.
- Anything on a feed after Chronicle PR #451.
- The concept match of each of the 14 families of microcosm#1243.
