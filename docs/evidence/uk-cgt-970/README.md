# Evidence for microcosm#970 (UK CGT sub-exempt gainers)

Aggregate-only extracts of the licensed receipts behind `experiments/970-cgt-sub-exempt-remainder.md`. The
licensed artifacts themselves (spine and calibrated H5 files, full sidecars, the FRS/SPI/WAS/LCFS/ETB inputs) live
outside the tree under `data/ukds/acceptance/970-cgt-sub-exempt/` and `runs/uk-623-first-calibrated/`.

- `v20-fence-receipt.json`: the projection fence on the calibrated v20 artifact, directly and as the signed seam
  receipt of calibrating the v20 spine through the branch (blocked at terminal, no H5).
- `spine-970-receipts.json`: the rebuilt 31-stage spine's anchor and remainder receipts and spine gate verdicts.
- `v22-calibration-receipts.json`: the v22 seam verdict, fence details, calibration summary, CGT target errors, the
  pass-2 score against the incumbent, and the clone/donor split of the remaining entrants.

Redaction: per-record values are withheld (the remainder's per-band and overall minimum and maximum amounts);
quantiles, counts, masses and target errors are kept.
