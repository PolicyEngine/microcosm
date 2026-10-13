# Congressional-district interest paid: what the column counts and what it binds

The congressional-district SOI file has no column for the itemized interest
deduction (Schedule A line 10). Chronicle's `congressional_district_2022`
package published one of its parts under the line 10 measure ids, and the
compiler bound it as the whole. This page records what each source counts,
the three remedies we measured, and the one `fiscal_targets.py` applies
(microcosm#1038).

## What each source counts

| Source | Columns or measure ids | What it counts |
|---|---|---|
| District file (`22incd.csv`, `22incddocguide.docx`) | N19300 / A19300 | "Mortgage interest paid", Schedule A line 8a |
| District file | N19500 / A19500 | "Home mortgage from personal seller", line 8b |
| District file | N19530 / A19530 | "Deductible points", line 8c |
| District file | N19570 / A19570 | "Investment interest paid", line 9 |
| Table 2.1 (feed, TY2023) | `interest_paid_deduction_*` | The line 10 total: $208.18B on 11,944,817 returns |
| Table 2.1 | `home_mortgage_interest_*` | Total home mortgage interest: $171.36B on 11,644,348 returns, exactly line 8a ($167.68B) plus 8b ($3.69B) |
| Table 2.1 | `mortgage_interest_paid_*`, `home_mortgage_personal_seller_*`, `deductible_points_*`, `investment_interest_paid_*` | Lines 8a, 8b, 8c and 9 |

Chronicle's district-file package reads N19300/A19300, line 8a, under
`interest_paid_deduction_{returns,amount}`. Its concept label,
`irs_soi.interest_paid_deduction`, is Table 2.1's line 10 concept. The file
carries the other three parts too, but the package does not publish them.

In the file's US row, line 8a is $140.14B of the $163.71B that lines 8a + 8b
+ 8c + 9 sum to: 85.6%. It is 98.2% of 8a + 8b.

## What the model measures

- `interest_paid_deduction_*` maps to PE-US `interest_deduction`. That
  variable adds `deductible_mortgage_interest` and `non_mortgage_interest`
  (policyengine-us 2.2.1), so it is the line 10 concept.
- `home_mortgage_interest_*` maps to `deductible_mortgage_interest`, read as
  `deductible_mortgage_interest_tax_unit`, the home mortgage interest left
  after the acquisition-debt caps (microcosm#511).
- Every row of either measure is masked to itemizers (`itemized_only`, from
  `_SOI_ITEMIZED_ONLY_VARIABLES`), whatever universe its record set names.
  The district file's rows and Table 2.1's therefore materialize over the
  same tax units.

## What main did

The district file's state, district and US rows compiled onto
`interest_deduction` as line 10 levels. On the pinned feed (`51aa40f`,
compiled at 2024 with aging and the packaged crosswalk), on the `full`
surface:

| | District-file states (51), and districts (436), each sum | National Table 2.1 line 10 target | Ratio |
|---|---:|---:|---:|
| Amount | $152.36B | $226.33B | 0.673 |
| Returns | 11,352,500 | 11,944,817 | 0.950 |

The amount gap is mostly concept (8a against line 10) and vintage: the file
is TY2022 data stamped 2023, set against Table 2.1 TY2023.

## Three remedies, measured

Every number below comes from the sha-matched `22incd.csv` (`137522878a...`)
and Historic Table 2 TY2022 (`22in55cmcsv.csv`). Scripts and outputs are in
`~/PolicyEngine/_reviews/cd-interest-paid-concept-20261012/`.

**(a) Line 10 shares.** Each row would bind `interest_deduction` as its share
of the file's US row times the Table 2.1 line 10 control. This assumes a
district's share of line 8a equals its share of line 10. Investment interest
is concentrated where the assumption fails. Line 8a is 38.8% of line 10 in
FL-21, and 44.9% and 45.0% in NY-10 and NY-12. Across the 436 district units
of the 117th plan, a district's target would be off from its line 10 share
by:

- a median of 7.3%, a 95th percentile of 17.4%, and at most 54.6% (FL-21);
- 4.66% of the national total misallocated between districts (half the sum
  of absolute share differences), about $10.5B of the aged control.

At state level it is off by a median of 5.3% and at most 29.8% (Wyoming),
with 3.07% of the total misallocated. Against Historic Table 2's own state
line 10 totals, an independent publication, it is off by at most 56.1%
(Wyoming).

**(b) Drop the rows.** Interest would keep only Table 2.1's two national
rows. That discards 974 district-file targets whose information is almost
all mortgage interest, which the file measures well. Mortgage interest is
unevenly spread: a district's share of line 8a differs from its share of
returns by 30.3% of the total (half the sum of absolute share differences),
from its share of AGI by 22.2%, and from its share of itemized deductions by
14.0%. Option (c)'s proxy error is 0.21% on the same scale.

**(c) Mortgage interest shares (applied).** Each row binds
`deductible_mortgage_interest` as its share of the file's US row times Table
2.1's home mortgage interest control (8a + 8b). This assumes a district's
share of line 8a equals its share of 8a + 8b. Line 8b is 1.8% of 8a + 8b
nationally, so the share moves little:

- districts: a median of 0.45%, a 95th percentile of 1.4%, at most 2.4%
  (CA-35), and 0.21% of the total misallocated;
- states: a median of 0.38%, at most 1.3% (West Virginia), and 0.17%
  misallocated;
- against Historic Table 2's state 8a + 8b totals: a median of 0.58% and at
  most 3.0% (West Virginia), with 0.29% misallocated. That includes the
  district file's coverage gap, which is 1.7% for 8a (CD/HT2 0.983, WV
  0.954).

The counts proxy is as close. At state level, a state's share of the file's
N19300 differs from its share of Historic Table 2's N19300 + N19500 by a
median of 0.43%, at most 2.8% (North Dakota).

So (c) keeps the geography (a) and (b) would lose or distort, at about a
twentieth of (a)'s error. The concepts it leaves without district targets,
points (8c) and investment interest (line 9), are 17.7% of line 10
nationally. Table 2.1's line 10 row still binds their national total.

## How the rule works

`_SOI_DISTRICT_COMPONENT_SHARE_COLUMNS` in `fiscal_targets.py` registers
N19300 and A19300, keyed by the IRS column rather than the measure id:

- **Reference.** `_soi_reference_from_fact` sends any district-file fact
  that reads a registered column to `deductible_mortgage_interest`, as a
  count for N19300 and a sum for A19300, with role
  `soi_fiscal_distribution`. It flags the fact for the rebase. A Chronicle
  package that relabels the column, for example to Table 2.1's
  `mortgage_interest_paid_*` ids, compiles to the same targets.
- **Refusal.** A district-file `interest_paid_deduction_*` fact that reads
  any other column, or names none, raises `UnreviewedSoiDistrictColumnError`.
  The file has no line 10 column, so such a fact is unreviewed by
  construction.
- **Rebase** (`_rebase_soi_district_component_shares`). Each flagged row
  becomes `control x row / US row`:
  - The US row is the same record set's national all-AGI row of the same
    column, filing status and period.
  - The control is the latest national all-AGI `home_mortgage_interest_*`
    fact in the itemized-returns universe not after the build period, from
    outside the district files.
  - Two different controls, or two different US rows, at the deciding period
    raise. Feed order never decides (microcosm#1035).
  - A row with no control at or after its own period, or no nonzero US row,
    is dropped, never shipped as a level.
  - The file's US row retires, because Table 2.1's row owns the national
    concept.
- **Declarations.** Every kept row carries:
  - `soi_source_concept = schedule_a_line_8a_home_mortgage_interest_paid_to_financial_institutions`;
  - `soi_control_concept = schedule_a_lines_8a_8b_total_home_mortgage_interest`;
  - `soi_share_total_source_record_id`, the US row;
  - `soi_source_column_id`;
  - `soi_return_universe_bridge = itemized_returns`;
  - the `uprating_*` fields every rebase carries.
- **Periods.** The value lands at the control's period, and target aging
  gives it the control's own remaining link, the CBO AGI series for amounts.
  The file's ty2023 stamp cancels in the share: the same package stamped
  2022 gives the same targets.

### Invariants

These hold for every fact set the compile accepts, and
`test_us_soi_district_interest_shares.py` checks them, including with
Hypothesis:

- No district-file row binds `interest_deduction`.
- One factor scales every row of a record set, column and filing status, so
  districts keep their within-state shares.
- Where the file's states sum to its US row, as they do on the pinned feed,
  its states sum to the control, and so do its districts.
- No row carries the rebase flag after the compile.
- Rows are nonnegative.
- The output does not depend on feed order, or on a common scale of the
  file's values.

## Effect on the pinned feed

`full` surface, compiled at 2024 with aging and the packaged crosswalk:

| | Before | After |
|---|---:|---:|
| District-file amount rows: 51 states, and 436 districts, each sum | $152.36B on `interest_deduction` (0.673x line 10) | $186.31B on `deductible_mortgage_interest` (1.000x the national home mortgage target) |
| District-file return rows, each sum | 11,352,500 (0.950x) | 11,644,348 (1.000x) |
| California amount | $38.13B | $46.63B |
| Rebase factor, amount / returns | — | 1.222854 / 1.025708 |
| Compiled specs | 33,250 | 33,248 (the file's two US rows retire) |
| `national_state` | 6,102 at `e1f9906bf0e0` | unchanged |

No other spec changes in name, value or metadata. The amount factor includes
TY2022-to-TY2023 growth in mortgage interest: Table 2.1 TY2023 line 8a is
$167.68B, against the file's TY2022 $140.14B.

### The ACS local `state_cd` surface

`state_cd` used to lift the district file's interest-paid state rows onto
the Historic Table 2 basis with the itemized-deductions bridge (amounts) and
the return-count bridge (counts). The bridges corrected the file's coverage
gap and its stamp. A share of a national control has neither: its states
already sum to the control. Bridging it as well would lift its amounts about
6%, and its counts about 2%, a second time (the bridges' medians). So
`us_acs_local_cd_surface.py`
binds a district-file state row that names its share's denominator
(`is_national_share_spec`) as compiled, as its districts' parent, with basis
`cd_file_state_total_national_share`. `STATE_CD_LEVEL_BRIDGES` no longer
lists interest paid, so a row the compiler failed to rebase would refuse the
surface rather than bind unbridged.

The bridge was also the wrong instrument here. Wyoming and South Dakota fell
out of its factor band because their charitable deductions differ between
the two files: HT2/CD A19700 is 2.10 and 1.98. Their mortgage interest
agrees, at 1.007 and 1.028. Both states' interest-paid rows now bind. The
`state_cd` contract counts are in docs/us-acs-local-soi-target-surface.md.
