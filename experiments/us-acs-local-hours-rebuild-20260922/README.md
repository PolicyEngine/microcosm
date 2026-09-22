# Local ACS local-area rebuild with native usual hours, 2026-09-22

A full-scale, uncapped run of the supported ACS local-area chain on a tree
that integrates #923 (native ACS usual-hours recovery) and #941 (the
`hours_worked_signal` finalize gate) on top of `18c6d39a0`, with the
Medicaid substitution-hierarchy fix from #955 and the package-stage refusals
from #973. It answers #765 at full scale and records the receipts a later
release decision needs. It is build evidence only: the artifact was not
uploaded, published or certified, and `tools/publish_release.sh` was not run.

## Inputs, by digest

| Input | Identity |
| --- | --- |
| Donor | `populace_us_2024_receipt_qualified.h5`, sha256 `009469727409e58c0c9c1252581292497b688b76f1d59b51947483fb0f89221c`: the published national default (`6496cc43…`; the Hub signs these bytes under two release names, `populace-us-2024-spm-20260909`, which `latest.json` names at the acquisition revision `8ab57ffc`, and `populace-us-2024-spm-20260915`, the name `donor-receipt-qualification.json` carries) qualified by #972 with `receives_wic`, `receives_snap`, `receives_tanf` (receipt: `donor-receipt-qualification.json`) |
| ACS 2024 1-year archives | household `8281008e…`, person `afdc6d90…` (the pinned source manifest) |
| PUMA ladder | `us_puma_ladder_2020.npz`, sha256 `39a2ab2a…` |
| Calibration feed | the labelled Chronicle consumer artifact `chronicle_us_b571381`, facts sha256 `4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f`, passed to the materialize stage as `--feed-sha256`; `feed-receipt.json` carries the artifact's digests and its value comparison against the feed #955 pins (39,158 facts each; every value and cell identical; metadata differs on 994 source rows and 166 compiled specs from the chronicle#277 authority fix) |
| Engine | policyengine-us 2.2.1, policyengine-core 3.32.5, spm-calculator 1.0.0 (the workspace lock) |

Staging settings (recorded in `build_manifest.json` → `staging_orchestration`):
uncapped, 32 trees, 8 targets per fit, ACS share 0.5, seed 0, under-15 hours
policy `us_hours_under15_zero_completion_v1` (a labelled modeling assumption:
510,098 modeled zeros at ages 0–14, counted in the staging summary excerpt).

## Stages and resources

`run-resources-and-staging-excerpt.json` records each stage's wall, CPU and
peak RSS under a bounded supervisor: staging 7,218 s / 103,105 CPU-s / 42.4 GB;
materialize 5,067 s / 74.8 GB as the supervisor sampled it (`build_manifest.json`
→ `materialize.peak_rss_gb` records 75.6 GB from the tool's own sampling; 80
chunks of 20,000 households, SOI totals mode, 760 administrative + 487
population targets); calibrate 415 s;
qa 172 s; finalize 79 s; package 79 s. Two earlier staging attempts were lost
to an unset pool-memory limit (the historical script exports
`*_PEAK_LIMIT_BYTES`) and to the supervisor's own disk floor during backup
snapshot churn; neither was a gate refusal.

## Results

- `gate_summary.json`: all eight finalize gates passed at full geography (0
  ladder population cells dropped): `acs_local_hours_signal`,
  `hours_worked_signal` (worked share 0.542 within [0.35, 0.62]; mean weekly
  hours of workers 37.35 within [30, 45]; 106 distinct usual-hours values),
  `us_puma_ladder_gate`, `calibration`, `input_coverage`,
  `spine_composition`, `spine_ssi_qa`, `consumer_ready`.
- `hours-comparison-vs-buildo.json` (persons aged 16+, per spine, aggregates
  only; both files identified by build id and digest in their `release`
  blocks): in the published ACS-local Build O
  (`populace-us-2024-buildo-acs-local-77e2061-20260724T110908Z`),
  `weekly_hours_worked_before_lsr` on the ACS spine has one distinct value;
  all 2,872,990 rows are exactly 40. In this build it has 100 distinct values:
  38.5% zero, 28.7% exactly 40, 61.5% positive, mean among positive 37.5. The
  ASEC spine is identical in both columns. The native `hours_worked_last_week`
  on the ACS spine shifts slightly because the donor transfer was re-run:
  81,723 distinct values against 82,213, zero share 45.2% against 45.3%, mean
  among positive 37.5 against 37.4.
- `calibration_diagnostics.json`: 1,247 targets, 800 epochs, loss 0.157 →
  0.153, 82.9% of targets within 10%, Kish ESS 24,010 (1.51%), weight-ratio
  cap 5 reached, mass conserved. These are mechanics on the totals-mode
  contract, not fit-quality evidence; the published Build O calibrated on the
  full state SOI surface (4,461 targets in its `calibration_diagnostics.json`,
  recorded in `hours-comparison-vs-buildo.json`), a contract this run did not
  use.
- `spm-composition-counts.json`: 5,274 of 1,591,514 SPM units in the
  consumer file have no classified adult under the locked engine's rule
  (5,273 single-member; oldest member under 15 in 1,946, aged 15–17 in
  3,328), equal on every field to the published Build O file (the
  `incumbent` block). The `decomposition` block reconciles the count from two
  independent measurements of the inputs: the ACS spine before transfer has
  5,246 unresolved units, all group-quarters minors (1,946 under 15, 3,300
  aged 15–17; 0 in housing units), and the national donor has 28 under the
  head/spouse fallback; 5,246 + 28 = 5,274 and 3,300 + 28 = 3,328. The
  consumer file carries no independence-role column (the export holds it
  back on this tree), so the engine's head/spouse fallback applies on both
  spines. The engine-side measurement universe
  (policyengine-us#9462) and the role stage (#959) are the fixes.
- `spine_qa.json`: SSI incidence 0.0187 (ASEC-PUF), 0.0236 (ACS), plain
  consumption true.

## Files

`sha256sums.txt` is the packaged release directory's checksum list, copied
verbatim. Six of its twelve entries are carried here and match it
(`build_manifest.json`, `calibration_diagnostics.json`, `gate_summary.json`,
`release_manifest.json`, `spine_qa.json`, `us_source_coverage.json`); the
other six (the 9.8 GB artifact, `consumer_export.json`,
`consumer_reviewed_null_fills.json`, `held_back_columns.json`,
`reviewed_null_fills.json`, `run_identity.json`) are not, so `sha256sum -c`
reports them missing. `release_manifest.json` names `policyengine/populace-us`
and this build's id in its `repo_id` and `revision` fields; those are the
coordinates a publish would use, and since the publish step was not run no
such revision exists on the Hub. `feed-receipt.json`,
`hours-comparison-vs-buildo.json` and `spm-composition-counts.json` were
written by this record's counts-only measurement scripts, not by the release
tool.

## What this is not

Not a release candidate: the target contract (SOI totals mode) and the SPM
consumer scope need rulings, and the package guard in #973 refuses capped
smokes but does not judge contracts. Poverty is a comparison diagnostic only
and played no part in any gate, target or selection.
