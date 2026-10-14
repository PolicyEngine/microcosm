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

Regenerated after commit `8e8f525a3`, which made a value that is constant
over a stratum give exactly zero variance (its residuals had been left at
rounding level).

| Design | Sample | Certainty households | Probes missing in over 1% of all draws at 3 SE (no guard) | Probes eligible in most draws | Misses among eligible draws at 3 SE | At 4 SE |
|---|---|---|---|---|---|---|
| 5:0 (the salvaged design) | 17,656 (5.0%) | 13 | 25 of 41 | 11 | 11 of 2,426 | 1 of 2,426 |
| 30:0 | 18,122 (5.1%) | 505 | 22 | 13 | 9 of 2,809 | 5 of 2,809 |
| 30:3 | 23,579 (6.7%) | 6,247 | 16 | 17 | 14 of 3,421 | 0 of 3,421 |
| **30:2 (default)** | 25,827 (7.3%) | 8,614 | 12 | 19 | 14 of 3,885 | 1 of 3,885 |
| 30:1 | 35,945 (10.2%) | 19,264 | 13 | 25 | 19 of 4,941 | 0 of 4,941 |

What the numbers say:

- **Unguarded, the standard errors do not cover.** Under the salvaged design
  25 of 41 probes miss by more than 3 SE in over 1% of draws (nominal 0.3%).
  The worst are `form_4952_election_neutralization` (394 carrier households,
  58%), `collectibles_gain_neutralization` (1,138, 54%) and
  `obbba_casualty_loss_limit` (101, 44%). The weighted totals are dominated
  by a few records: household weights run from 0.02 to 9,740, and the inputs
  are heavy-tailed. A sample that misses those records underestimates the
  total, and its standard error is too small for the same reason.
  Stratifying by weight class as well did not help. An earlier comparison
  (200 seeds, counting draws with at least 5 drawn carriers) found 26 of 41
  probes missing in over 1% of draws without weight classes and 24 of 41
  with five weight classes per stratum.
- **The effective-households guard removes almost every miss.** Requiring the
  variance estimate to rest on at least 30 effective households leaves, at
  3 SE, 0.3% to 0.45% of eligible draws missing (about the nominal 0.27%).
  At 4 SE, the probe's default, it leaves 0 to 1 per design, except 30:0's 5.
- **Size certainty buys eligibility.** Keeping the households that dominate a
  probe's input mass raises the number of probes eligible in most draws from
  11 to 19 at multiplier 2, for 7.3% of the pool instead of 5.0%. Multiplier
  1 reaches 25, at 10.2%.
- **Threshold 30 makes the two worst rare probes exact.** At threshold 5,
  `obbba_casualty_loss_limit` and `form_4952_election_neutralization` were
  drawn, with about 5 and 20 sampled households each. At 30 their carriers
  are taken whole: 505 certainty households instead of 13.

## What the guard does not rule out

- **A probe that is only rarely eligible.** Under 30:0,
  `salt_refund_income_neutralization` was eligible in 3 of 200 draws, and all
  3 missed. Those draws had just reached 30 effective households because a
  few large households were drawn. The default design takes such households
  with certainty; it still has one miss in 3,885 eligible draws
  (`wic_claim_neutralization`).
- **The proxy is not the effect.** Each probe's real effect is a function of
  its inputs, computed by the engine, and its tails can differ from the
  inputs'. The calibration is on the inputs' mass. A probe run with
  `--reference-release-dir` reports each effect's z-score against the
  full-size build's, which is one draw on the effects themselves.

"Authoritative" is a calibrated label, not a proof.
