# SOI business, partnership and rental "net income (less loss)" concepts

Historic Table 2 (HT2) and the congressional-district file publish business,
partnership/S-corp and rental/royalty income as one signed column each. Until
microcosm#1228 the target compiler bound those columns to one-sided model
concepts. This page records what each IRS column counts, what the model
measures, and how `fiscal_targets.py` binds them now.

## What each column counts

| Measure ids | Column (HT2, district file) | Definition (2022 and 2023 doc guides) | Form line |
|---|---|---|---|
| `schedule_c_income_{returns,amount}` | N/A00900 | "business or professional net income (less loss)" | Schedule 1 line 3 |
| `partnership_scorp_income_{returns,amount}` | N/A26270 | "partnership/S-corp net income (less loss)" | Schedule E line 32 |
| `rental_royalty_income_{returns,amount}` | N/A25870 | "total rental and royalty net income (less loss)" | Schedule E line 26 |

The amount nets losses against income. The count is every return with an
entry on the line, net income or net loss. Table 1.4 splits each line into a
"net income" column and a "net loss" column, and the HT2 columns are their
union.

**Business income.** Table 1.4 "Business or profession" net income and net
loss:

| Tax year | HT2 N00900 / T1.4 (NI + NL returns) | HT2 A00900 / T1.4 (NI − NL) | T1.4 NI returns / HT2 N00900 | T1.4 NI amount / HT2 A00900 |
|---|---:|---:|---:|---:|
| 2020 | 1.0059 | 1.0225 | 0.716 | 1.266 |
| 2021 | 1.0064 | 1.0195 | 0.732 | 1.233 |
| 2022 | 0.9999 | 1.0272 | 0.724 | 1.289 |
| 2023 | 1.0070 | 1.0334 | 0.698 | 1.362 |

**Partnership and S corporation income.** TY2020 is the last year Table 1.4
published combined partnership-and-S-corp columns. That year HT2 N26270
(9,011,930) equals the combined NI + NL returns (9,001,513) within 0.12%,
and only 65.5% of them are net-income returns. In TY2021–TY2023 Table 1.4
splits partnership from S corporation, so no combined net-income count
exists. HT2 A26270 equals the summed partnership and S-corp nets within
0.2–0.35% in every year.

**Rental and royalty income.** N25870/A25870 first appear in TY2022. Table
1.4's "Total rental and royalty" includes Form 4835 farm rental. Schedule E
line 26 does not, because farm rental goes on line 40. In TY2022 HT2 A25870
is 0.919 of the Table 1.4 total net, and 0.983 once farm rental is removed.
In TY2023 the ratios are 0.919 and 0.992. HT2 N25870 is 0.97 and 0.96 of the
Table 1.4 NI + NL returns, about the farm-rental-only returns.

The workbooks are the feed-pinned `20in14ar.xls` … `23in14ar.xls` and
`20in55cmcsv.csv` … `23in55cmcsv.csv`. The script and its output are in
`~/PolicyEngine/_reviews/soi-net-income-less-loss-20261012/`
(`scripts/identities.py`, `out/identities.out`).

## What the model measured before #1228

`tools/build_us_fiscal_refresh_release.py` `_signed_component` /
`_soi_component_row` sum each SOI concept per tax unit:

- `business_net_profits`, mapped to `self_employment_income`, took the
  **positive part** and counted units with a **positive** total.
- `partnership_and_s_corp_income`, mapped to
  `tax_unit_partnership_s_corp_income`, did the same.
- `rent_and_royalty_net_income`, mapped to `rental_income +
  farm_rent_income`, kept the sign but counted a **positive** total.

Two further gaps sat on the same rows:

- **SSTB businesses.** The frame moves a specified-service business's whole
  Schedule C amount from `self_employment_income_before_lsr` to
  `sstb_self_employment_income_before_lsr` (`us_runtime/qbi_inputs.py`).
  Schedule 1 line 3 is every Schedule C, which is PE-US
  `total_self_employment_income`. The business target never saw SSTB income.
- **Farm rental.** The rental concept added Form 4835 farm rental, which
  line 26 leaves out.

## How the targets bind now

`SOI_NET_INCOME_LESS_LOSS_VARIABLES` registers three signed concepts:

| Concept | PE-US column (tax-unit total) | Amount | Count | `soi_source_concept` |
|---|---|---|---|---|
| `business_net_income_less_loss` | `total_self_employment_income` | signed sum | total ≠ 0 | `form_1040_schedule_1_line_3_net_income_less_loss` |
| `partnership_and_s_corp_net_income_less_loss` | `tax_unit_partnership_s_corp_income` | signed sum | total ≠ 0 | `schedule_e_line_32_net_income_less_loss` |
| `rent_and_royalty_net_income_less_loss` | `rental_income` | signed sum | total ≠ 0 | `schedule_e_line_26_net_income_less_loss` |

- The six measure ids bind these concepts.
- Only HT2 and the district file are reviewed to publish them
  (`_SOI_NET_INCOME_LESS_LOSS_FAMILIES`). The same id from any other family
  raises `UnregisteredSoiNetIncomeLessLossFamilyError`, because Table 1.4
  publishes these lines as one-sided columns.
- The one-sided concepts (`business_net_profits` / `business_net_losses`,
  `partnership_and_s_corp_income` / `_losses`, `rent_and_royalty_net_income` /
  `_losses`) are reserved for Table 1.4's split columns. No pinned fact uses
  them. Two of them changed so a future leg binds correctly:
  - the business legs now read `total_self_employment_income`;
  - `rent_and_royalty_net_income` is now a positive part, like every other
    income leg. It keeps farm rental, as Table 1.4's total does.

  PolicyEngine/chronicle#346 asks Chronicle for those legs.

On the pinned feed (`consumer_facts_us_51aa40f`, compile at 2024 with aging,
the packaged district crosswalk and Medicaid substitution) the change:

- rebinds 3,294 specs: 312 on `national_state` (6 ids × 51 states + the US
  row) and the district file's 2,982 on `full`;
- changes no target value, and no spec enters or leaves either surface;
- moves `national_state` `e1f9906bf0e0` → `38c1db8880da` and `full`
  `f5bd921d6b21` → `9f6c65fe8734`.

## Invariants

- **Leg conservation.** For every signed tax-unit vector and nonnegative
  weights:
  - a less-loss count equals the net-income leg's count plus the net-loss
    leg's count;
  - a less-loss amount equals the net-income leg's amount minus the net-loss
    leg's amount.

  This is the identity the IRS columns satisfy. Hypothesis property:
  `test_net_income_less_loss_rows_are_the_sum_of_their_table_1_4_legs`.
- **Bounds and sign symmetry.** The less-loss count is at least the
  net-income count, and the amount at most the net-income amount. Negating
  every value keeps the count and negates the amount. Same property.
- **Family-only binding.** For any id, tax year and record-set shape, a fact
  binds its line's signed concept exactly when its family is HT2 or the
  district file, and refuses otherwise. Property
  `test_net_income_less_loss_binding_depends_only_on_the_family`.
- **Pinned feed.** `test_pinned_feed_net_income_less_loss_rows_bind_signed_concepts`
  checks:
  - every business, partnership and rental row binds its line's concept;
  - only those rows bind a signed concept;
  - no row binds a one-sided leg;
  - `national_state` carries 312 of them;
  - each family's 51 state rows sum to at most the US row.

## The method choice

These rows are on `national_state`, the surface the certified Route A
release calibrates, so changing what they bind is a ruling. Three options
were measured on the release's own calibration checkpoint (see "Effect on
the Route A calibration" below):

1. **Concept-exact (this PR).** Bind each column to the signed tax-unit
   total of its line. This needs no new facts and no share assumption.
2. **Rebase onto a one-sided Table 1.4 control**, as #1036 does for capital
   gains: HT2 state rows as shares of a national net-income leg. This needs
   PolicyEngine/chronicle#346. There is no combined partnership/S-corp
   net-income column after TY2020, and state shares of a net amount stand in
   for state shares of profits.
3. **Drop** the rows whose concept differs. That is all but the rental
   amount: 260 on `national_state`.

**The frame limits option 1.** The `puf_tax_detail` stage lists
`self_employment_income_before_lsr` under `nonnegative_outputs`
(`us/spec/sources.yaml`). As a result:

- the Route A frame carries 0.39M weighted Schedule C loss units before
  calibration (0.65M after), against 8.39M loss returns in Table 1.4
  TY2022;
- a nonzero Schedule C count can still be met only by adding profit
  carriers until the frame carries losses;
- partnership and rental (line 26) do carry losses: 1.68M and 3.66M units before
  calibration.
