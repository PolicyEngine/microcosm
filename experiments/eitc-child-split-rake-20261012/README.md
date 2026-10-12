# EITC child-split total rows against their AGI bins (2026-10-12)

## Question

`_uprate_cross_period_eitc_decompositions` (`us_runtime/fiscal_targets.py`)
scales the SOI Table 2.5 EITC-by-AGI-and-children rows (TY2023) to the TY2024
filing-season EITC by AGI (mid-November, `25inweek47.xlsx`). After
microcosm#1227 each row takes:

- a bin: (active EITC in its AGI group) / (TY2023 EITC in that group, all
  splits), so each AGI group keeps its TY2023 child mix;
- a split's all-AGI total row: active / source over all AGI (×1.0418 for
  amount, ×0.9753 for returns), so the totals keep the TY2023 all-AGI child
  mix.

The AGI groups grew at different rates (amount ×0.85 at $5,000–$10,000 to
×1.73 at $50,000 and over; returns ×0.81 to ×1.38). So per split, the total
row and the sum of its 28 bins disagree. Both are on the national_state and
full surfaces, so calibration is asked to hit both.

| Split | Amount: total / bins | Returns: total / bins |
|---|---|---|
| Childless | 1.157 | 1.062 |
| One child | 1.018 | 1.001 |
| Two children | 0.975 | 0.961 |
| Three or more | 0.993 | 0.951 |

## Recommendation: (c), the two-margin rake

Rake each decomposition's child × AGI-group matrix of TY2023 bins (iterative
proportional fitting) to the splits' total rows and the active AGI groups.
The total rows keep their #1227 values, and each split's bins then sum to its
total row. Each AGI group's bins sum to its active EITC, and the bins of one
split in one group share one factor. The PR implements this.

No source forces the choice: no TY2024 child mix is published (§1). But the
evidence points one way:

- Over 11 year pairs, keeping the all-AGI child mix predicted next year's
  split totals better than keeping each AGI group's mix (§2).
- It is far closer to the other EITC-by-children targets already on the
  surface (§3).
- The TY2023 mix it keeps is confirmed by SOI's full-population TY2023 data
  (§4).

## The four options on the pinned feed

Release-equivalent compiles of the pinned feed (`consumer_facts_us_51aa40f.jsonl`,
sha256 965a29ac…; target period 2024, packaged CD crosswalk, aged, Medicaid
substitution, then `_select_target_surface`). Source:
`results/surface_differential.json` and `results/t25_rows.csv`.

| | (a) status quo | (b) totals := Σ bins | (c) rake | (d) drop totals |
|---|---|---|---|---|
| What moves | nothing | 8 total rows, ×0.864–1.052 | 224 bins, ×0.857–1.226 (total rows keep their values) | 8 total rows removed |
| Total row = Σ its bins | no (0.951–1.157) | yes | yes | n/a |
| Implied childless share of EITC (amount / returns) | 3.71% / 27.73% (totals), 3.21% / 26.11% (bins) | 3.21% / 26.11% | 3.71% / 27.73% | 3.21% / 26.11% |
| national_state | 6,102 at 3fad14f85468 | 6,102 at 937b3df341a7 | 6,102 at 31bd18096d33 | 6,094 at 41a1fa2b013f |
| full | 33,250 at c3aaea9c1e8c | 33,250 at 0f294d4f9312 | 33,250 at ac32014f3325 | 33,242 at 469b8ed2c5db |
| Table 2.5 share of loss weight, national_state / full | 1.555% / 0.605% | 1.555% / 0.605% | 1.548% / 0.603% | 1.272% / 0.494% |
| Table 2.5 rows' weights vs (a) | — | ×0.930–1.026 | ×0.926–1.101 | ×1.0003–1.0029 |
| Largest weight move of any other row | — | 8×10⁻⁶ | 1.5×10⁻⁴ | 2.9×10⁻³ |

All four conserve the active EITC ($69.04B, 23.84M returns). Every option
changes only Table 2.5 rows: no other target's value or metadata moves. The
loss weights move because they are sqrt-value-weighted and renormalized
(`target_loss_weights.py`).

What each says about each split's national total:

| Split | (a) total row | (a) bins, = (b) = (d) | (c) |
|---|---|---|---|
| Childless amount | $2.562B | $2.215B | $2.562B |
| One child amount | $24.41B | $23.98B | $24.41B |
| Two children amount | $25.94B | $26.61B | $25.94B |
| Three or more amount | $16.13B | $16.24B | $16.13B |
| Childless returns | 6.610M | 6.225M | 6.610M |
| One child returns | 8.470M | 8.465M | 8.470M |
| Two children returns | 5.643M | 5.871M | 5.643M |
| Three or more returns | 3.114M | 3.276M | 3.114M |

(b) and (d) carry the same information (the bins); (b) adds weight on the
bins' sums, (d) removes the totals' weight. (c) is the only consistent option
that keeps the all-AGI child mix.

## Evidence

### 1. No TY2024 EITC child mix is published (checked live 2026-10-12)

- SOI Table 2.5 ([AGI tables page](https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income),
  updated 27-Mar-2026) lists `00in25ic.xls`–`23in25ic.xls`; `24in25ic.xls`
  returns 404.
- SOI preliminary data ([individual returns page](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns),
  updated 21-Jul-2026): the latest article is TY2022.
- SOI geographic data with EITC by number of children ([what's new](https://www.irs.gov/statistics/soi-tax-stats-whats-new),
  updated 29-Sep-2026): the latest is TY2023 state, county and metro data
  (August 2026), and TY2022 congressional-district data.
- SOI filing-season Table 1 for TY2024 ([filing-season page](https://www.irs.gov/statistics/filing-season-statistics),
  `25inweek47.xlsx`, byte-identical to the feed's source, sha256 5b8f0573…):
  EITC returns and amount by AGI only.
- [IRS EITC Central statistics](https://www.irs.gov/tax-professionals/eitc-central/statistics-for-tax-returns-with-the-earned-income-tax-credit-eitc)
  (updated 28-Jul-2026): TY2024 national 24M claims and $69.6B (through
  cycle 202553), by state only.

### 2. Backtest on published Table 2.5, TY2013–2023

`backtest.py` uprates Table 2.5 TY *s* to TY *s*+1 the way the compiler does,
under each option, and scores the result against the published Table 2.5 TY
*s*+1. All options conserve the active total, which the mid-November file
puts 3–6% below complete-year SOI (§5), so predictions are scored against
the TY *s*+1 table rescaled to the same total: the options differ only in how
they split it.

- *fs*: the active AGI groups are SOI's mid-November filing-season file for
  TY *s*+1, as production does with TY2024. These files exist for TY2017–2019
  and TY2021–2024.
- *oracle*: the active groups are TY *s*+1 Table 2.5 itself, coarsened to the
  same ten groups, to isolate the child-mix assumption.
- Clean pairs only: TY2020–2022 changed the credit (ARPA's childless
  expansion and its reversal, the prior-year earned-income election). On
  the ARPA pairs (TY2020→2021, TY2021→2022), both approaches miss childless
  EITC by 35–332% and the bins miss by less, as expected when the law moves
  the child mix. TY2023→2024 has no such break: every maximum credit and
  AGI limit rose 5.3–5.5% ([IRS EITC tables](https://www.irs.gov/credits-deductions/individuals/earned-income-tax-credit/earned-income-and-earned-income-tax-credit-eitc-tables)).
  See `results/backtest.log`.

Mean absolute error of the four splits' national totals (scored as above):

| Margins, kind | Bins: (a) bins, (b), (d) | All-AGI mix: (a) totals, (c) | Split×group cells TVD, bins / (c) |
|---|---|---|---|
| fs, amount (4 pairs) | 5.39% (worst 19.7%) | 2.87% (worst 18.0%) | 0.0368 / 0.0366 |
| fs, returns (4 pairs) | 5.01% (worst 10.8%) | 2.07% (worst 7.5%) | 0.0361 / 0.0321 |
| oracle, amount (7 pairs) | 3.03% (worst 14.5%) | 2.09% (worst 18.0%) | 0.0257 / 0.0244 |
| oracle, returns (7 pairs) | 2.68% (worst 9.6%) | 1.60% (worst 7.5%) | 0.0250 / 0.0228 |

- **Childless is the systematic miss.** The bins under-predicted childless
  EITC in all 11 clean runs, by 1.1–19.7%. The all-AGI mix missed it both
  ways.
- **Pair by pair.** The all-AGI mix had the lower mean error in 8 of 11 amount
  runs and 10 of 11 returns runs.
- **The exceptions.** In TY2015→2016 both did well, and the bins did better.
  In TY2022→2023 childless EITC fell from 4.38% to 3.71% of the amount, so
  keeping the TY2022 mix missed it by +18%; the bins missed by −4.4% to −7.0%.
- **Cells.** On the split × AGI-group cells, (c) was as good or better on
  average, and its within-cell shape is the bins' own.

### 3. Consistency with the other EITC-by-children targets

national_state already carries SOI Historic Table 2 TY2022 state rows by
number of children (51 states × 10 measures, aged ×1.1203 for amounts).
Summed to the nation, they give the child mix below. The full surface's
congressional-district US rows give the same mix: they are SOI TY2022 data,
which Chronicle stamps TY2023 (PolicyEngine/chronicle#117).

Total variation distance between child mixes, in percentage points
(`results/consistency.json`):

| Against | Amount, bins / (c) | Returns, bins / (c) |
|---|---|---|
| HT2 TY2022 states on national_state | 1.62 / 0.55 | 2.24 / 0.63 |
| CD US rows on full | 1.61 / 0.54 | 2.21 / 0.60 |
| SOI full-population state data, TY2023 | 1.17 / 0.24 | 1.68 / 0.19 |

(b) and (d) would leave the two national EITC-by-children families three to
four times further apart than (c) does.

### 4. Full-population TY2023 data confirm Table 2.5's TY2023 mix

SOI's TY2023 state data (`23in55cmcsv.csv`, August 2026; all returns, not a
sample; not yet in the pinned feed) give the US EITC child mix as:

- amount: 3.68%, 35.42%, 37.37%, 23.53%, against Table 2.5's 3.71%, 35.35%,
  37.58%, 23.36%;
- returns: 27.80%, 35.41%, 23.61%, 13.19%, against 27.73%, 35.53%, 23.67%,
  13.06%.

So the TY2022→2023 fall in childless EITC (4.14% to 3.68% of the amount in the
full-population data) is real, not Table 2.5 sampling noise. (c) carries this
latest published mix to TY2024. The bins project a further fall, to 3.21%.
No published source supports that fall, and the bins' record in §2 is to
under-predict childless EITC.

### 5. Separate finding: the mid-November totals run 3–6% below complete-year SOI

| TY | Returns: mid-November / Table 2.5 | Amount: mid-November / Table 2.5 |
|---|---|---|
| 2017 | 0.968 | 0.970 |
| 2018 | 0.948 | 1.012 |
| 2019 | 0.949 | 0.968 |
| 2021 | 0.971 | 0.973 |
| 2022 | 0.939 | 0.957 |
| 2023 | 0.946 | 0.957 |

The TY2024 EITC targets (`irs_soi.ty2024.filing_season_week47.*`, also release
critical-fit gates in `us_critical_targets.py`) are mid-November levels. All
four options conserve that total, so this is outside the choice here. It is
proposed as a separate follow-up.

## Reproduce

```bash
D=/tmp/eitc-soi
uv run --no-project --with xlrd==2.0.1 --with openpyxl==3.1.5 \
  python -I fetch_sources.py $D          # downloads and verifies sources.json
python3 -I backtest.py $D/tsv results/backtest.json <pinned feed> > results/backtest.log
cd <repo> && .venv/bin/python experiments/eitc-child-split-rake-20261012/surface_differential.py \
  $PWD experiments/eitc-child-split-rake-20261012/results
python3 -I consistency.py results/surface_differential.json $D results/consistency.json
```

`backtest.py` first checks its parser against the pinned feed: it reproduces
all 498 Table 2.5 and filing-season facts exactly. `surface_differential.py`
needs the checkout that carries `_rake_eitc_child_splits`. Option (a) patches
the rake to the identity, and (b) and (d) post-process (a)'s rows.
