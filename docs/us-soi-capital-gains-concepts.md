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
- **Congressional-district rows** are not controls. Their own targets, which
  are on the `full` surface only, are not reconciled here.

The share step assumes that each state's share of line-7 returns equals its
share of Schedule D gain returns. No IRS state product publishes a gain-only
count, so this cannot be checked directly. AGI composition is one measurable
source of difference: weighting each state's HT2 N01000 by AGI class with
Table 1.4's gain share per class would move TY2022 state targets by −2.9% (WV)
to +4.3% (DC), with a median of 1.6% and no state beyond 5%.

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
