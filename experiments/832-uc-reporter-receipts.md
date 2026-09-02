# 832 — benefit-unit UC reporter draw: I1 before-receipts

Committed, disclosure-safe record of the #832 I1 measurements (2026-09-01) — the evidence
anchors for this PR's `spine_swap_signed_differences.json` entries and the decision inputs
for the plan's D2/D3 adjudications. Raw measurement scripts and JSON receipts live
licensed-side in `data/ukds/acceptance/832-uc-reporter/` (not in this repository); every
aggregate below is weighted or count-based with minimum cell count 3, and each receipt pins
its inputs by digest: spine-l H5 `5176e6ec…` (the `uk-publication-stack-829` build,
14/14 spine gates), policyengine-uk 2.92.1 at year 2024.

## Part A — screen sizing

The pre-take-up-award screen (`max(0, uc_maximum_amount − uc_income_reduction) > 0`,
capital from the dataset's coherent `uc_reported_capital`) measured on spine-l:

| Channel | Benunits (weighted) | Screened-eligible | Share |
|---|---:|---:|---:|
| base FRS | 18.549m | 4.655m | 25.1% |
| SPI | 17.034m | 2.416m | 14.2% |

The screened-eligible SPI domain is **not** tiny — the demotion-only fallback is refuted
and the draw arm proceeds. Of the current 2.001m weighted SPI post-fill reporters, only
**0.758m pass the screen**: the other ~1.24m are the dead-reporter mass the issue measured
(1.249m), independently reproduced here. Screened SPI mass by family type: lone parent
0.351m, couple with children 0.298m, single (no children) 1.453m, couple without children
0.314m.

## Part B — within-cell earnings gradients on the base channel

Base-FRS benefit-unit reporter rates by benefit-unit earnings quintile, within
family-type × UC-child-band cells (records ≥ 50; weighted rates):

| Cell | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| couple with children, band 2 | 50.9% | 13.0% | 1.1% | 0.7% | 0.4% |
| couple with children, band 3+ | 81.5% | 67.8% | 29.7% | 5.8% | 2.0% |
| lone parent, band 1 | 78.9% | 85.2% | 77.4% | 52.8% | 12.3% |
| lone parent, band 3+ | 95.2% | 87.6% | 88.4% | 93.6% | 22.5% |
| single, band 0 | 16.2% | 17.2% | 17.4% | 7.4% | 1.2% |

Within-cell gradients run 10–100× from bottom to top quintile in every with-children
cell. A composition-cell hot-deck cannot represent this — the income axis is precisely
the defect under repair — so the D2 recommendation (benefit-unit RegimeGatedQRF) stands
and the hot-deck fallback stays dormant.

## Part C — offline benefit-unit fit dry-run

A `RegimeGatedQRF` (seed 42, default estimator settings) fitted on base-FRS clone-0
screened benefit units (3,271 train / 778 held-out, deterministic identity-keyed 80/20
split), target = benefit-unit summed reported UC, predictors = is_married, UC child band,
benefit-unit employment / self-employment / investment income, claimant and partner
earnings split, claimant age, region one-hot:

- **Held-out reporter rate**: 46.0% actual vs 46.8% drawn (weighted; single draw).
  Cell agreement within binomial noise, e.g. lone parent band 1: 80.3% vs 80.7%;
  single band 0: 35.1% vs 35.1%; the one marginal cell (couple with children band 3+,
  48 records: 75.8% vs 58.4%) sits at ~2.6σ on a small cell.
- **Positive-amount quantiles (GBP)**: p25 7,039 vs 8,097; p50 11,316 vs 11,773;
  p75 15,918 vs 16,209; p90 21,394 vs 22,424.

Applied to the screened SPI domain (current state measured on spine-l for comparison):

| Family type | Screened | Model draws | Current reporters | Current converters |
|---|---:|---:|---:|---:|
| lone parent | 0.351m | **0.233m** | 0.171m | 0.102m |
| couple with children | 0.298m | 0.106m | 0.706m | 0.152m |
| single (no children) | 1.453m | 0.490m | 0.549m | 0.343m |
| couple without children | 0.314m | 0.107m | 0.575m | 0.160m |

Every drawn reporter converts by construction (the screen), so the lone-parent cell —
the largest deferred deficit — gains ≈ +0.131m weighted initial support from the SPI
channel. **Honest warning, recorded for the I5/I6 re-measure**: faithful base-channel
rates draw fewer couple-with-children reporters (0.106m) than currently convert
(0.152m) — ≈ −0.046m initial pressure on a currently-fitting cell — because the SPI
channel's income profile suppresses reporting probability even inside the screened
domain. That residual composition question (who gets drawn eligible on the SPI channel)
is the #145 draw-quality boundary the plan's R3 declares; this fix does not stretch to
cover it.

## Part D — conversion before

Reporter→positive-award conversion on spine-l (weighted; `reported > 0` at benefit-unit
grain → modeled `universal_credit > 0`), the acceptance-criterion-2 "before" side:

| Family type | Base FRS | SPI channel |
|---|---:|---:|
| lone parent | 95.0% (0.632/0.665m) | 59.8% (0.102/0.171m) |
| couple with children | 86.3% (0.460/0.533m) | 21.5% (0.152/0.706m) |
| single (no children) | 88.0% (0.868/0.987m) | 62.6% (0.343/0.549m) |
| couple without children | 81.5% (0.153/0.188m) | 27.9% (0.160/0.575m) |

Matches the issue's evidence table (base 95%/86%, SPI 58%/20%) within re-measurement
tolerance on the newer spine.

## Part F — spine-m actuals for the signed differences

The I5 twin rebuild (spine-m, `data/ukds/acceptance/spine-m-832/`, 14/14 spine gates,
engine 2.92.1; measured by the licensed session on 2026-09-01) re-derives the two
I1-bounded signatures from the built artifact:

- **`universal_credit_reported` (person)**: nonzero reporter records 7,298 → **5,730** of
  113,649 persons, a candidate nonzero share of **0.050418** against the incumbent's
  0.057359 — the candidate sits **below** the incumbent by 0.006941. The I1 structural
  range [0.038962, 0.067990] was a ceiling-and-floor bound; the measured draw lands in
  the lower half, as the ~46% base reporter rate predicted, so the above-incumbent half
  of that range is not reached in practice.
- **`would_claim_uc` (benunit)**: the reporter set feeding the coherence OR-refresh shrank,
  so the flag count fell by 804 records to a candidate share of **0.548741** — 0.001951
  **below** the incumbent's 0.550692, inside the whole-spine ±0.02 acceptance band. The
  #828-era lift signature (candidate above, bounded at 0.0553) no longer described the
  measured direction and was **retired** on 2026-09-02 (María's ruling on PR #835): an
  in-band difference carries no signature, so a future out-of-band move in either
  direction is flagged as unsigned.
- Whole-spine twin compare vs spine-l: exactly four stage-owned columns differ
  (`universal_credit_reported`, `frs_benunit_capital`, `uc_reported_capital`,
  `would_claim_uc`); every other column is byte-identical, and the base-FRS channel shows
  zero reporter transitions.

## Part E — the arm comparison, discharged mechanically

Issue #832 asked for a focused comparison of (1) adding benefit-unit predictors to the
existing stage-2 joint draw vs (2) a dedicated benefit-unit-grain draw. Measured against
the code, arm 1's blast radius is not a matter of judgment: the stage-2 fill is a
chained-equations draw whose 29 outputs each condition on the drawn values of every
earlier output, with `universal_credit_reported` at chain position 6 — changing its
predictors re-draws all 23 downstream reported-benefit columns on every SPI row and moves
the pinned stage-2 output surface (`_assert_income_stage_parameters` plus three test
pins). Arm 2 (this PR) leaves stage 18 byte-identical and confines the diff to one column
on SPI rows plus the coherence stage's dependents, which is what makes the whole-spine
twin comparison attributable. Parts B and C above are arm 2's positive evidence. A full
two-build A/B remains available separately if adjudicated worth its licensed cost
(plan A1: run by María post-PR if still wanted).
