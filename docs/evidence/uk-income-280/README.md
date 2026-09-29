# Evidence for the income-anchor lane (PolicyEngine/chronicle#280, microcosm#1006)

Aggregate-only extracts of the licensed v21c national run behind microcosm#1006: the first
national calibration on the income-anchor target surface (HMRC Income Tax liabilities and the
OBR rows at the calendar-2025 window, the SPI 2023-24 bands uprated by declared indices, the
Table 3.11 rows on the region tier) on spine-u, the spine with the reserved SPI income band
donors. The licensed artifacts (the spine and calibrated H5 files, the full sidecars, the
FRS/SPI/WAS/LCFS/ETB inputs) live outside the tree under `data/ukds/acceptance/` and
`runs/uk-623-first-calibrated/spine-assessment-v21c/`; the head-to-head page with the
downstream legs is the private evaluation site's `national-v21c-2026-09-23` report.

- `v21c-receipts.json`, written by `scripts/extract_v21c_receipts.py` from the run tree:
  the run identity (code f6d33a26 on the pre-rebase head, spine-u 897a0f3e, release
  candidate false), the calibration summary (53,806 records all nonzero, loss 0.3072 to
  0.0091, 97.6 percent of the 1,062 targets within 10 percent, Kish ESS 9,193), the six
  terminal gate verdicts with the two signed target-fit deferrals as applied, the compiled
  target, estimate and relative error of the 119 income-anchor rows (OBR income tax, Universal
  Credit and fuel duty, the HMRC liabilities rows, the SPI amount bands, the region tier's
  200,000-and-over rows and the deferred South East row), the pass-2 score against the
  incumbent snapshot in the run tree (910 wins to 32 on 942 common targets, 120 rows pruned as
  incumbent-unresolvable, losses 0.0105 against 0.2923), and the band-donor receipts (120
  donors per band at the published band weights, carriers by region, the propensity table's
  cell count, and the resample's per-band pool sizes, composite counts, weighted taxpayers,
  regional matches and realised band means).

Redaction: per-record values are withheld (each band's realised minimum and maximum total
income, the reserved households themselves); counts, masses, means and target errors are kept.

The measurements are on the pre-rebase head (f6d33a26), and they do not hold at the current
head. A licensed spine build from the rebased branch is blocked by the student-loan PLAN_5
realisation gate (deviation 1.1027 against the limit of 1.0; raised to 2.0 for assessment builds
only, locally), and the national calibration at 8d36ecc4 blocks at the terminal gate
`uk_cgt_projection_entrants`, which #979 introduced after v21c's tree: 146,920 weighted
sub-exempt gainers cross the frozen exempt amount by 2030 against the bound of 73,000, most of
the mass on about fifty of #970's CGT band-donor rows whose weight floats with the target set.
So this head cannot produce a certified cut, the six green gate verdicts in `v21c-receipts.json`
are v21c's, and the fence is settled on the #970/#979 side (microcosm#1006, comments 5812328880
and 5814299594; the doc's limitations section). The realised-income pool cut from the review
round is likewise unmeasured until that rebuild. The receipt's text fields are the run's own
compiled metadata and still spell the lane as the run did (`microcosm#280`); the tree now cites
PolicyEngine/chronicle#280.
