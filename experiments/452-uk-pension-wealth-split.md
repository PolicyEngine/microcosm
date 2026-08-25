# #750 / uk-data#452 (M2): private pension wealth split out of `corporate_wealth` — measurement receipt

Committed receipt for the WAS-stage change that emits `private_pension_wealth`
as its own household column and keeps `corporate_wealth` to the share-like
holdings. Companion JSON: `452-uk-pension-wealth-split-receipt.json`
(aggregates only — UKDS EUL; no unit records). Scripts and the full
diagnosis live outside the repo (`~/PolicyEngine/_uk-defects-452-448/`,
`measure/pension_split/`).

## Why

On the incumbent enhanced FRS (`enhanced_frs_2024_25.h5`, uk-data 1.56.14 /
1.56.16) at 2025, 46% of benefit-unit records that report Universal Credit
fail the £16,000 capital test, 96% of them on `corporate_wealth` alone. The
WAS stage folds WAS private pension wealth less the current-employment DB
component (`totalpenr8_aggr − dvvaldbt_scaper8_aggr`) into
`corporate_wealth`, and the model counts that variable as UC capital. The
value of a right to receive a pension under an occupational or personal
pension scheme is disregarded capital (UC Regs 2013 Sch 10 para 10; HB Regs
2006 Sch 6 paras 31–32 and HB(SPC) Regs 2006 Sch 6 para 24; JSA Regs 1996
Sch 8 paras 28–29; ESA Regs 2008 Sch 9 paras 28–29; IS Regs 1987 Sch 10
paras 23–23A; SPC Regs 2002 Sch V paras 22–23, Part I).

## What the remainder is (donor, weighted £bn)

WAS round 8 household tab (sha `18b3eb98…`, 15,128 households, 26.99m
weighted): total private pension wealth 4,807.7; current-employment DB at
the SCAPE rate 1,148.7; remainder 3,659.0 = current occupational DC 313.9 +
AVCs 12.2 + retained DC 550.0 + current personal pensions 203.7 + retained
DB 863.6 + pensions in payment 1,692.4 + expected from a former spouse 23.3
(the ten-component identity is exact on 100% of rows; the remainder is never
negative). Share-like holdings (employee shares, UK shares, unit and
investment trusts, stocks-and-shares ISA) 659.4, so the pension component is
84.7% of the old `corporate_wealth` mass. Hence the column is **not**
DB-free (retained DB and DB pensions in payment stay inside it): it is named
`private_pension_wealth` with the current-DB exclusion stated first in its
documentation, not `non_db_pension_wealth`.

## The fixed-draw legal effect (the causal number)

On the final code's own draws (per-segment seeds, seed 0, 100 trees), the
drawn `private_pension_wealth` column was folded back into `corporate_wealth`
on the same rows and UC recomputed — every draw identical, only the
capital-test classification differs. Split minus folded, 2025, weights
unchanged:

| | folded back (pension counted) | split (pension disregarded) | fixed-draw effect |
|---|---|---|---|
| UC benefit units, UK | 4.869 | 6.268 | **+1.399** |
| UC benefit units, GB | 4.761 | 6.090 | +1.329 |
| UC spend, £bn | 59.7 | 71.4 | +11.7 |
| eligible benefit units | 8.67 | 15.57 | +6.90 |
| reporter records over £16k | 3,195 | 1,742 | −1,453 |
| GB housing / LCWRA / carer / childcare / child elements | 3.77 / 1.70 / 0.57 / 0.24 / 2.21 | 4.59 / 2.06 / 0.62 / 0.40 / 2.82 | +0.82 / +0.35 / +0.06 / +0.17 / +0.61 |
| GB single / lone parent / couple no children / couple with children | 2.36 / 1.52 / 0.19 / 0.68 | 3.01 / 1.90 / 0.27 / 0.91 | +0.65 / +0.37 / +0.08 / +0.23 |

With reporters anchored: 4.889 → 6.345 (+1.456).

## Whole-implementation comparison (redraw pair, shared-seed variant)

The WAS wealth block was re-imputed on every household of the 1.56.16
artifact (seed 0, 100 trees), once with the pre-split arithmetic (OLD) and
once with the split (NEW), both under the pre-fix shared-seed pattern, the
household columns swapped into the single-year dataset before the simulation
is built, and UC measured at 2025 with policyengine-uk 2.91.0. OLD and NEW
are independently fitted and drawn, so this pair compares whole
implementations rather than isolating the classification (the fixed-draw
table above does that); no matched OLD′ exists for the final per-segment-seed
branch. Benefit units in millions; GB = excluding Northern Ireland.

| | incumbent artifact | OLD re-draw | NEW re-draw | NEW − OLD |
|---|---|---|---|---|
| UC benefit units, UK | 6.346 | 5.193 | 6.266 | **+1.073** |
| UC benefit units, GB | 6.199 | 5.036 | 6.064 | +1.028 |
| UC spend, £bn | 75.3 | 63.1 | 71.7 | +8.6 |
| eligible benefit units (WA adult, capital ≤ £16k) | 10.52 | 9.77 | 16.26 | +6.49 |
| reporter records over £16k (weighted m) | 2,958 (1.38) | 3,165 (2.04) | 1,757 (1.28) | −1,408 (−0.76) |
| GB housing element | 4.868 | 4.082 | 4.592 | +0.510 |
| GB LCWRA element | 2.102 | 1.696 | 2.055 | +0.359 |
| GB carer element | 0.641 | 0.561 | 0.645 | +0.084 |
| GB childcare element | 0.471 | 0.260 | 0.403 | +0.143 |
| GB child element | 3.018 | 2.345 | 2.869 | +0.524 |
| GB single | 2.969 | 2.471 | 2.880 | +0.408 |
| GB lone parent | 2.135 | 1.657 | 1.922 | +0.266 |
| GB couple, no children | 0.225 | 0.222 | 0.326 | +0.104 |
| GB couple with children | 0.871 | 0.686 | 0.935 | +0.250 |

With FRS reporters anchored (`would_claim_uc |= universal_credit_reported > 0`):
OLD 5.217 → NEW 6.340 (UK), NEW′ 6.345. The incumbent's own upper bound (counterfactual
C, `corporate_wealth` → 0 on the incumbent draws) is 6.881 UK.

Allocation-key check on the NEW draw (why the model keeps
`corporate_sector_wealth = corporate_wealth + private_pension_wealth` as the
key for shareholding, corporate land value and the employer-NI capital
response): the narrowed key puts 86.7% of its mass in the top 10% of
households and 28.7% in the top 1% (preserved key: 54.4% / 14.6%).

Regression receipt on the same run: chain segment 1 is untouched by the
split, so the `owned_land` and `property_wealth` draws are identical OLD vs
NEW at seed 0 (weighted totals 119.939 and 11,411.434 £bn in both); only the
segment-2/3 columns move. On this re-draw the recipient-side `corporate_wealth`
mass is 2,499 vs the incumbent's 9,149 £bn at 2024 (drift −0.73), inside the
`uk_input_mass_parity` tolerance.

Engine coupling: `private_pension_wealth` is engine-unknown until
policyengine-uk publishes the companion variable; bump the `uk` extra lock to
that release before any national build or calibration (the UK targets
`ons.land.corporate_land_value` / `land_value` are materialised through the
engine's `corporate_sector_wealth` key), and re-pin the `efrs-post-calibration`
input-mass reference only after the uk-data mirror ships (re-pinning first
would breach the gate on `corporate_wealth`).

## Per-segment child seeds (review finding, second commit)

`impute_was_wealth` reused one `RegimeGatedQRF` across its three chain
segments, and `start_chain` respawns the fit and draw streams from the model
seed on every call — so the k-th target of every segment consumed the same
quantile and sign-gate uniforms per recipient, coupling `owned_land` with
the first segment-2 target and `property_wealth` with the second (now the
share-like holdings, the countable quantity). The adversarial review measured
it on train/hold-out halves of the donor (30 trees; companion JSON
`452-uk-pension-wealth-split-coupling-receipt.json`):

| hold-out probability | observed | stage as-is (same seed) | production child seeds |
|---|---|---|---|
| P(shares excl. ISA > 0 \| property_wealth = 0) | 0.055 | 0.011 | 0.039 |
| P(stocks-and-shares ISA > 0 \| property_wealth = 0) | 0.031 | 0.017 | 0.025 |
| P(private pension wealth > 0 \| property_wealth = 0) | 0.560 | 0.540 | 0.528 |

(The receipt was re-run with `was_wealth_segment_seeds` — the exact production
derivation — after review caught that the first variant derived its child
seeds with a modulo-2^31 truncation and measured 0.045; the unit test now
golden-pins the production tuple.)

The stage now derives one child seed per segment from the declared seed
(`SeedSequence(0).spawn(3)` → 3757552657, 673228719, 3241444873); the
declared `fit_weighted_qrf_chain.seed: 0` stays the root and the spec digest
is unchanged (the note is documentation). Re-measured with the fix (NEW′):
UC benefit units 6.268m UK (vs 6.266m without the fix), eligible 15.57m (vs
16.26m), reporter records over £16k 1,742 (vs 1,757); GB elements within
±0.05m except singles +0.13m and couples without children −0.06m. The
caseload effect of the split is unchanged; the fix corrects who holds the
countable assets. Seed-to-seed swings in the tail-dominated totals
(`corporate_wealth` 2,499 → 1,794 £bn, `savings` 1,489 → 899 £bn at 2024 for
the two draws) are the stage's known realisation variance (cf. the
`owned_land` exclusion receipt), not an effect of either change.

## Caveats (travel with the numbers)

- Weights are not recalibrated; the UK surface now compiles a UC caseload
  target (microcosm#735), which absorbs part of any move in a real build.
- OLD is not the incumbent's own draw: this package draws a continuous
  quantile over the full leaf conditional, the incumbent's microimpute QRF a
  random point on a 10-quantile grid, and the predictor surface here is the
  released h5 (no `num_bedrooms`, post-SPI incomes), not the FRS spine. On
  the same households the OLD re-draw sits 1.15m benefit units below the
  incumbent artifact — a draw-machinery difference outside this change and
  worth its own look (the re-draw also carries lower savings / other-property
  mass than the incumbent and higher property wealth).
- `private_pension_wealth` is engine-unknown at 2.91.0, so the NEW numbers
  are the model-side-unchanged outcome; the policyengine-uk companion PR
  makes the disregard explicit and keeps `total_wealth` and the exposure
  keys whole.
- `uk_input_mass_parity` will report `corporate_wealth` at roughly −0.85
  against the incumbent reference (pension mass moved to a column the
  reference lacks); inside the armed tolerance (4.52), so no exclusion
  receipt is added — the drift is explained here.
