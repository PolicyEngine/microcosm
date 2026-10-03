# SOI capital-gains concepts and the rebase control

`capital_gains_gross` targets come from three IRS SOI products that count
different populations under the same measure ids (`net_capital_gains_returns`,
`net_capital_gains_amount`). This page records what each one counts, which
one the model measures, and how `fiscal_targets.py` combines them
(microcosm#1035).

## What each table counts

| Source | Columns | What it counts |
|---|---|---|
| Table 1.4 (`23in14ar.xls`) | col 37 / 38, "Sales of capital assets reported on Form 1040, Schedule D: Taxable net gain" | Schedule D returns that net to a gain, and that gain |
| Table 1.4 | col 39 / 40, "Taxable net loss" | Schedule D returns that net to a loss, and the loss after the $3,000 limit |
| Table 1.4 | col 35 / 36, "Capital gain distributions reported on Form 1040" | Returns that report only distributions, without Schedule D |
| Historic Table 2 (`22in55cmcsv.csv`), congressional-district file (`22incd.csv`) | N01000 / A01000 | "Number of returns with net capital gain (less loss)" and its amount, at Form 1040 line 7 (both documentation guides) |

Line 7 carries a Schedule D gain, a loss-limited Schedule D loss, or
distributions reported without Schedule D (2022 Form 1040 instructions, line
7, Exception 1). Columns 35, 37 and 39 are disjoint: Table 1.3 row 18, "Sales
of capital assets net gain", equals col 35 + col 37 exactly in TY2022 and
TY2023.

N01000 is the union of the three Table 1.4 populations:

| Tax year | HT2 US N01000 | Table 1.4 col 35 + 37 + 39 | Difference | HT2 A01000 vs col 36 + 38 − 40 |
|---|---:|---:|---:|---:|
| 2020 | 29,008,620 | 29,003,885 | +0.016% | −0.63% |
| 2021 | 32,996,180 | 33,076,998 | −0.244% | −0.29% |
| 2022 | 30,465,850 | 30,461,045 | +0.016% | −0.17% |
| 2023 | 29,481,840 | 29,434,188 | +0.162% | −0.28% |

In TY2022, col 37 (12,915,122) is 42.4% of N01000. The amounts are close
(A01000 is 1.4% below col 38), because distributions and capped losses are
small in dollars. The counts differ by 2.36x.

The congressional-district file covers a subset of HT2: it leaves out
territories, APO/FPO and foreign addresses and returns without a matched ZIP
code. Its US N01000 is 98.0% of HT2's.

## What the model measures

`capital_gains_gross` maps to PE-US `capital_gains`, which is short-term plus
long-term gains, i.e. Schedule D before the loss limit. Distributions reported
without Schedule D are the separate `non_sch_d_capital_gains`, with its own
Table 1.4 col 35 target. The SOI materializer counts a tax unit when the
unit's summed `capital_gains` is positive and sums its positive part, so the
count is col 37 and the sum is col 38. The Build P release met col 37 at
12,391,446 against 12,392,020.

No released record has negative capital gains, so the model has no Schedule D
loss returns. An N01000 count therefore has no model counterpart at any
level.

## How the targets combine

- **Controls.** Only a family registered with the model's concept
  (`_SOI_CAPITAL_GAINS_FAMILY_CONCEPTS`, today Table 1.4 alone) can supply the
  national level, and the latest national all-AGI fact not after the build
  period wins. HT2 and congressional-district rows never do, whatever their
  period stamp. Two different records at the winning period raise
  `AmbiguousSoiCapitalGainsControlError`, so feed order never decides.
- **Historic Table 2 rows** enter only as shares. Each state row is its share
  of the HT2 US total, scaled to the control, landed at the control's period,
  and tagged `soi_source_concept = form_1040_line_7_net_gain_or_loss` and
  `soi_control_concept = schedule_d_taxable_net_gain`. A row with no control
  at or after its own period is dropped, because a line-7 level would be a
  2.36x mismatch. The HT2 national all-AGI rows retire because the control's
  own row owns the national concept.
- **Congressional-district rows** are not controls either, and their own
  targets (on the `full` surface only) enter the same way: each state and
  district row is its share of the congressional-district file's US row,
  scaled to the control and tagged with the same two concept keys, and the
  file's US row retires (microcosm#1038, below). Every share row names its
  denominator, the HT2 or congressional-district US row, in
  `soi_share_total_source_record_id`.

The share step assumes that each state's share of line-7 returns equals its
share of Schedule D gain returns. No IRS state product publishes a gain-only
count, so this cannot be checked directly. AGI composition is one measurable
source of difference: weighting each state's HT2 N01000 by AGI class with
Table 1.4's gain share per class would move TY2022 state targets by −2.9% (WV)
to +4.3% (DC), with a median absolute change of 1.6% and no state beyond 5%.

## How the returns control drifted

The rebase used to keep the first fact among equal-period candidates. The
July pin listed Table 1.4 ty2023 first, so Build P rebased the 51 state
return counts onto 12,392,020 (factor 0.406751165649407; states summing to
12,289,836). The September re-pin (`consumer_facts_us_c5e5bf8`) sorts rows by
`aggregate_fact_key`, a content hash. There, the congressional-district US row
sorts first; it carries TY2022 data stamped ty2023 (PolicyEngine/chronicle#117).
It became the control (factor 0.97964), and the states summed to 29,599,604,
2.39x the national target of the same model quantity on the same surface. The
amount control happened to stay on Table 1.4, but reversing the feed would
have moved it to the congressional-district row (factor 0.92455 instead of
0.77190).

## Effect on the pinned feed

Compiled at 2024 with target aging and the packaged congressional-district
crosswalk, then narrowed to `national_state`:

| Targets | Before | After |
|---|---:|---:|
| 51 HT2 state return counts, sum | 29,599,604 | 12,289,836 |
| California | 3,774,052 | 1,566,997 |
| Texas | 2,108,705 | 875,540 |
| Florida | 2,093,099 | 869,060 |
| New York | 1,974,435 | 819,791 |
| National Table 1.4 return count (unchanged) | 12,392,020 | 12,392,020 |
| `national_state` registry | `d315c75804ef` | `65e4dde11c83` |

- All 60 HT2 return-count specs move by the same factor, 0.415202720927. That
  is the 51 state rows plus 9 congressional-district proxies, which are on
  `full` only.
- The 60 matching amount specs keep their values and gain only the two
  concept keys.
- No spec enters or leaves either surface, which keeps 32,842 compiled and
  5,694 on `national_state`.
- The state return counts are back at the Build P values; California's
  1,566,996.66 equals the July scorecard's.

`test_pinned_feed_no_soi_state_family_sums_past_its_national_target` holds
the invariant across all 63 SOI state families on the surface.

## Congressional-district rows on the full surface (microcosm#1038)

The release tool's default `--target-surface` is `full`, which also carries
the congressional-district file's rows: a US row, 51 state rows and 436
current districts per measure. Before microcosm#1038 they compiled as line-7
levels onto `capital_gains_gross`. On the pinned feed, their states and their
districts each summed to 29,845,710 returns, 2.41x the national Table 1.4
target of 12,392,020 on the same surface. Their amounts summed to $1,522.2B
against $1,270.9B, 1.198x. The amount gap is mostly the file's TY2022 data
stamped ty2023 and aged from 2023. The count gap is the concept.

There were two ways to fix this. Option (a) rebases the rows as shares, like
the HT2 rows. Option (b) drops them with a reviewed exclusion. Microcosm#1038
takes (a):

- **The share assumption is already the HT2 one.** Each district's share of
  line-7 returns stands in for its share of Schedule D gain returns. AGI
  composition is the measurable source of difference. Reweighting each of the
  428 numbered districts' N01000 by AGI class with Table 1.4's per-class gain
  share moves its share by a median 2.1% (95th percentile 4.7%, largest
  CA-18 +9.7%). Amounts move by a median 0.5% (95th percentile 2.4%).
- **Dropping the rows would discard a much larger signal.** District
  capital-gains incidence varies far more than state incidence. N01000 per
  return runs from 8.3% to 31.5% between the 5th and 95th percentile
  districts, against 13.7% to 23.4% across states. Within a state, the
  district share left over after the district's AGI distribution has a log
  standard deviation of 0.227 for counts and 0.223 for amounts. The
  AGI-composition error of the share proxy is 0.024 and 0.009.
- **The rebased rows agree with the HT2 state rows about as well as other
  pairs on `full` already do.** The congressional-district and HT2 state
  return counts for capital gains differ by a median 0.6% (95th percentile
  2.1%). The `return_count` pair differs by 1.6% (2.4%).
  - Amounts differ more: a median 2.8% (95th percentile 15.4%, Montana
    −27.0%). The cause is item suppression in the congressional-district
    file's A01000. Montana's is 32.5% below HT2's at the state level. That is
    still inside the qualified-dividends amount pair, 11.6% (21.7%).

The congressional-district file has no rows for other areas, so its US row
equals the sum of its 51 states. The rebased states and districts therefore
each sum to the control itself. HT2's US row includes other areas, so its
rebased states sum to 99.2% of the control for returns and 98.0% for
amounts.

**Vintage.** A share divides a row by the US row of the same package, so the
package's period stamp cancels. The value lands at the control's period
(2023) and ages from there. It is the same whether 22incd.csv is stamped
ty2023, as on the pinned feed, or truthfully at 2022
(`test_congressional_district_capital_gains_ignore_the_package_stamp`).

microcosm#1030 corrects restamped facts to age from their data year, but it
refuses any restamped fact that was rebased. Before it merges with this change,
that refusal must admit a share whose `soi_share_total_source_record_id` is a
restamp of the same package, and leave `uprating_to_period` at the control's
period. Starting the aging at 2022 instead would apply the Table 1.4
2022-to-2023 fall twice (x0.761).

### Effect on the pinned feed

Release compile at 2024, as above:

| `full` surface | Before | After |
|---|---:|---:|
| Congressional-district capital-gains returns, states and districts (each sum) | 29,845,710 | 12,392,020 |
| Congressional-district capital-gains amounts, states and districts (each sum) | $1,522.19B | $1,270.86B |
| California returns / amount | 3,808,930 / $214.08B | 1,581,478 / $178.73B |
| New York returns / amount | 2,015,540 / $127.96B | 836,858 / $106.83B |
| Compiled targets, all on `full` | 32,842 (`e02123644d42`) | 31,376 (`059dc56d78db`) |
| `national_state` targets | 5,694 (`65e4dde11c83`) | 5,694 (`47dce807b412`) |

- The 974 congressional-district capital-gains rows move by one factor per
  measure: 0.415202720927 for returns and 0.834893818418 for amounts.
- The two US rows retire.
- The Historic Table 2 rows keep their values. The 120 of them (102 on
  `national_state`) gain only `soi_share_total_source_record_id`.

### Two mislabeled columns in the same package

On the same surface, Chronicle's `congressional_district_2022` package reads
four IRS columns under measure ids whose concept they do not carry. They are
the only differences between its column mapping and the Historic Table 2
package's (all 56 measures compared):

| Measure id | Congressional-district column | Doc-guide definition | Column the measure means |
|---|---|---|---|
| `limited_state_local_taxes_returns` / `_amount` | N18425 / A18425 | State and local income taxes (Schedule A line 5a) | N18460 / A18460, limited state and local taxes (line 5e) |
| `premium_tax_credit_returns` / `_amount` | N85530 / A85530 | Additional Medicare tax (Form 8959 line 24) | N85770 / A85770, total premium tax credit |

The SALT amount rows summed to 1.98x their national target on `full`. The
premium-tax-credit amount was already dropped by a hard-coded check, but the
returns rows compiled as `assigned_aca_ptc` recipients.

`US_FISCAL_TARGET_SOURCE_COLUMN_EXCLUSIONS` in `fiscal_targets.py` now drops
all four, keyed by (measure id, `layout.source_column_id`). It lists them in
the release's exclusion receipt. A corrected package that reads the right
columns passes untouched.

`test_pinned_feed_no_congressional_district_family_sums_past_its_national_target`
holds the invariant for every congressional-district family on `full`.

