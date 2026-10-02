# Sampling design of the US export subsample: coverage of the probe's standard errors

`tools/probe_us_post_export.py` labels a reform-coverage smoke verdict
*authoritative* when its effect clears or misses its floor by several
design-based standard errors. That label is a statistical claim, so the
standard errors have to cover. This folder measures how often they do, on the
published Route A release export
(`populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z`, 352,932 households,
907,863 persons), and records why the tools' defaults are what they are.

## Method

`scripts/coverage_experiment.py` is engine-free. For each of the 41 shipped
smoke probes it uses a proxy with a known full-export total: `y_h`, the
household total of the absolute values of the probe's binding inputs
(booleans as 1), and `T = sum w_h y_h`. This is the support and the tails of
the probe's effect, not the effect itself, which needs the engine. For each
design and each of 200 seeds it

1. builds the certainty set with the sampler's own `probe_carriers` and
   `size_certainty_households`;
2. draws at fraction 0.05 with the sampler's own `draw_households`;
3. estimates `T` with the drawn weights, and its variance with the probe's own
   `stratified_variance_terms`;
4. records `z = |T_hat - T| / SE` and the effective number of households the
   variance estimate rests on, `(sum q)^2 / sum q^2`.

A draw is *eligible* for authority when the effective count is at least 30
(or the probe is take-all and no drawn household carries it). A *miss* is
`z > k`: the probe would be confidently wrong.

```bash
python docs/evidence/us-export-subsample-design/scripts/coverage_experiment.py \
  --export <release>/artifacts/populace_us_2024.h5 \
  --smoke <release>/releases/<id>/reform_coverage_smoke.json \
  --out coverage.json --seeds 200 \
  --design 5:0 --design 30:0 --design 30:1 --design 30:2 --design 30:3
```

A design is `<thin-probe certainty threshold>:<size certainty multiplier>`.
`coverage.json` holds every probe's row for every design.

## Results (2026-10-02, 200 seeds, fraction 0.05)

| Design | Sample | Certainty households | Probes missing in over 1% of all draws at 3 SE (no guard) | Probes eligible in most draws | Misses among eligible draws at 4 SE | Same, without the count-like proxy |
|---|---|---|---|---|---|---|
| 5:0 (the salvaged design) | 17,656 (5.0%) | 13 | 26 of 41 | 12 | 0.23% | 1 of 2,426 |
| 30:0 | 18,122 (5.1%) | 505 | 23 | 14 | 0.43% | 5 of 2,809 |
| 30:3 | 23,579 (6.7%) | 6,247 | 17 | 18 | 0.44% | 0 of 3,421 |
| **30:2 (default)** | 25,827 (7.3%) | 8,614 | 13 | 20 | 0.27% | 1 of 3,885 |
| 30:1 | 35,945 (10.2%) | 19,264 | 14 | 26 | 0.68% | 0 of 4,941 |

What the numbers say:

- **Unguarded, the standard errors do not cover.** Under the salvaged design
  26 of 41 probes miss by more than 3 SE in over 1% of draws (nominal 0.3%);
  the worst, `form_4952_election_neutralization` (394 carrier households),
  misses in 58%. The weighted totals are dominated by a few records
  (household weights run from 0.02 to 9,740 and the inputs are heavy-tailed):
  a sample that misses them underestimates the total, and its standard error
  is too small for the same reason. Stratifying by weight class as well does
  not help (an earlier run: 24 of 41 still missed).
- **The effective-households guard removes almost every miss.** Requiring the
  variance estimate to rest on at least 30 effective households, at 4 SE,
  leaves 0 to 5 misses in 2,400 to 4,900 eligible draws, outside one proxy.
- **Size certainty buys eligibility.** Keeping the households that dominate a
  probe's input mass raises the probes eligible in most draws from 12 to 20
  at multiplier 2, for 7.3% of the pool instead of 5.0%. Multiplier 1 reaches
  26 at 10.2%.
- **Threshold 30 makes the two worst rare probes exact.** At threshold 5,
  `obbba_casualty_loss_limit` (101 carriers) and
  `form_4952_election_neutralization` (394) were drawn, about 5 and 20
  households each. At 30 their carriers are taken whole (505 households).

## The limit of the guard

One proxy keeps missing under every design: `household_head_childcare_cap_
neutralization`, whose only binding input is the `is_household_head` flag, so
`y_h` is a count of heads that is constant across almost every household. Its
misses among eligible draws at 4 SE run from 2.8% (5:0) to 18.7% (30:1), and
5.3% at the default. The total's error then comes from the few households
that deviate, which size certainty cannot see (every household has about the
same mass) and whose absence from a draw leaves a small variance estimate
spread over many households. The probe's real effect, a childcare cap among
heads with childcare expenses, is not shaped like its flag. The case still
shows what the guard cannot rule out: an effect that is nearly constant with
rare large deviations. "Authoritative" is a calibrated label, not a proof.
