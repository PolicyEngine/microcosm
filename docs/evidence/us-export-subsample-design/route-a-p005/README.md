# p = 0.05 probe of the published Route A export

The first real run of `tools/sample_us_export_households.py` and
`tools/probe_us_post_export.py`. It ran on the published Route A release
`populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` and compared the
probe's verdicts with that release's own full-size outputs.

`probe_report.json` is the probe's report as written. The only change is that
local paths are replaced by `<release>` (the release's output folder) and
`<run>` (this run's folder). `passes.jsonl` is the probe's per-pass timing
log, unchanged. Run
`scripts/summarize_probe_report.py route-a-p005/probe_report.json` to print
every table below from it.

These are probe diagnostics, not release evidence. The release's verdicts are
its own full-size ones.

## What ran

| | |
|---|---|
| Source export | `<release>/artifacts/populace_us_2024.h5`: 352,932 households, sha256 `417d23ae…1559` |
| Subsample | 25,827 households (7.3%), 8,614 of them certainty; seed 0, fraction 0.05, thresholds at the tools' defaults (30 and 2); sha256 `92a8373f…d391` |
| Sampler | drew at commit `5f72575a9`, 2026-10-02 17:38 to 17:51 EDT, peak 3.1 GiB; all six strata conserve their weight totals to about 1e-16 |
| Probe | commit `cf23a8edc` with `--reference-release-dir <release>/releases/<id>`, 2026-10-02 18:13 EDT to 2026-10-03 03:53 EDT |
| Engine | policyengine-us 2.2.1, policyengine-core 3.32.5, CPython 3.14.7 (free-threaded), macOS arm64 |

```bash
python tools/sample_us_export_households.py \
  --export <release>/artifacts/populace_us_2024.h5 \
  --fraction 0.05 --seed 0 --out <run>/subsample
python tools/probe_us_post_export.py \
  --export <run>/subsample/populace_us_2024.h5 --out <run>/probe \
  --reference-release-dir <release>/releases/<id>
```

**The commits in the receipt and the report are wrong.** The receipt names
`e39b14b66` and the report names `0a80bc884`. Each tool read git HEAD when it
wrote its record, minutes after it started, and the worktree had moved on by
then. The commits above are the ones the run script logged at launch.

- Between `5f72575a9` and the merged tools, only a comment in the sampler
  changed, so the subsample is what the merged sampler draws.
- Between `cf23a8edc` and the merged probe, three things changed: the probe's
  fault isolation, its stage names (`load` became `identify`, `load` and
  `design`), and the take-all rule, which now counts non-carrier effect
  households instead of summing their effects. No fault path fired in this
  run. The two authoritative take-all probes equal the full-size effects bit
  for bit.

Both tools now read the commit when they load and record their own sha256
beside it.

## Result

The run found 0 authoritative release failures, 0 probe failures and 2
informational failures.

| Stage | Verdict |
|---|---|
| Design rebuilt from the subsample against its receipt | pass |
| Stored-input gate | pass. The written-H5 premise cannot be reproduced standalone (the probe has only the written bytes) |
| QRF tail | source export: pass, and it equals the release's `qrf_tail_concentration.json` (verdict, columns, top-k shares). Subsample: informational fail, because the top-k and minimum-record rules count records, so the subsample checks different columns and tails |
| Take-up participation and stale count-calibrated columns | all 7 checks pass, on the source and on the subsample |
| Reform-coverage smoke, 41 probes | 40 pass. 8 are authoritative; all 8 pass and agree with the full-size build. One informational fail (below) |
| Reform validation, 55 reform passes | every reform scores. Budget effects are weighted estimates, so they are informational. Across 241 rows the median relative difference from full size is 15%; 11 of those rows are read from the build's calibration fit, as the release does. The largest is 107% (`spm_child_poverty_az`); state SPM poverty rows differ most |
| Demographics | pass. 218 of 436 congressional districts have under 50 sampled household records (informational; this scales with p). The source export has none under 50 |
| Post-export scoring plans | identical to the reference build's for demographics, smoke and validation |
| Source coverage | pass (the gate does not read the dataset) |

## Smoke effects against full size

`z` is the subsample effect minus the full-size effect, divided by the
subsample's design-based standard error. The relative difference is that
difference over the full-size effect's magnitude.

- 37 of 41 probes have a z-score. The other four have a standard error of 0:
  - the two take-all probes;
  - `cdcc_adult_care_expense_neutralization`, whose 397 effect households in
    the subsample are all certainty households;
  - the failing probe.
- Median |z| is 0.56. The largest is 3.40 (`head_start_take_up_neutralization`,
  informational, 2.0 effective households). None exceeds 4.
- The 8 authoritative verdicts rest on 34.5 to 366 effective households, and
  their largest |z| is 0.68.
  - The two authoritative take-all probes, `form_4952_election_neutralization`
    and `keogh_distribution_neutralization`, equal the full-size effects
    exactly.
  - `obbba_casualty_loss_limit` is also take-all, but drawn households carry
    part of its effect. It is 4.9% off and labelled informational.

**One verdict disagrees with the full-size build:**
`tx_snap_additional_vehicle_exemption_abolition`.

- The subsample effect is 0; at full size it is $19.7M.
- The probe's carriers are 309,580 households in the pool, and none of the
  22,889 carriers in the subsample changed SNAP. So the effect reaches few of
  the probe's carriers.
- The verdict rests on 0 effective households, so the guard labels it
  informational.

This is the limit the [design README](../README.md#what-the-guard-does-not-rule-out)
names: the coverage proxy is a probe's inputs, not its effect.

| Probe | Passes | Authority | Effect ($M) | Full size ($M) | Relative difference | z | Effective households |
|---|---|---|---|---|---|---|---|
| `alimony_expense_ald_abolition` | yes | informational | -3,410.7 | -2,708.6 | -25.9% | -0.67 | 1.5 |
| `aotc_abolition` | yes | informational | 1,540.0 | 1,743.7 | -11.7% | -0.76 | 9.1 |
| `cdcc_adult_care_expense_neutralization` | yes | informational | -128.5 | -128.5 | +0.0% |  | 0.0 |
| `child_support_expense_snap_deduction_abolition` | yes | informational | 147.0 | 102.1 | +43.9% | 0.52 | 2.1 |
| `child_support_received_snap_exclusion` | yes | informational | 1,612.9 | 1,584.8 | +1.8% | 0.08 | 3.8 |
| `collectibles_gain_neutralization` | yes | informational | 2,541.4 | 2,649.2 | -4.1% | -0.54 | 2.1 |
| `disability_benefits_snap_exclusion` | yes | informational | 874.9 | 721.0 | +21.3% | 0.36 | 2.4 |
| `domestic_production_ald_reactivation` | yes | informational | 7,152.7 | 7,649.1 | -6.5% | -0.89 | 3.0 |
| `educator_expense_ald_abolition` | yes | informational | 143.9 | 162.5 | -11.5% | -1.33 | 18.7 |
| `form_4952_election_neutralization` | yes | authoritative (take-all) | 1,158.0 | 1,158.0 | +0.0% |  | 0.0 |
| `fsla_overtime_premium_neutralization` | yes | authoritative | -21,614.6 | -22,081.0 | +2.1% | 0.44 | 34.5 |
| `head_start_take_up_neutralization` | yes | informational | 2,263.6 | 4,744.1 | -52.3% | -3.40 | 2.0 |
| `household_head_childcare_cap_neutralization` | yes | informational | -10,040.6 | -8,761.1 | -14.6% | -1.00 | 19.1 |
| `housing_assistance_take_up_neutralization` | yes | informational | 11,864.9 | 9,557.1 | +24.1% | 0.89 | 5.1 |
| `keogh_distribution_neutralization` | yes | authoritative (take-all) | 9.0 | 9.0 | +0.0% |  | 0.0 |
| `medicare_take_up_neutralization` | yes | authoritative | 513,492.5 | 516,893.9 | -0.7% | -0.27 | 366.3 |
| `obbba_auto_loan_interest` | yes | informational | -943.2 | -995.4 | +5.2% | 0.56 | 3.6 |
| `obbba_casualty_loss_limit` | yes | informational (take-all) | 1,396.4 | 1,331.4 | +4.9% | 0.80 | 2.2 |
| `obbba_cdcc` | yes | informational | -1,052.1 | -1,229.2 | +14.4% | 1.30 | 11.6 |
| `obbba_misc_itemized_deductions` | yes | informational | 19,895.9 | 20,552.9 | -3.2% | -0.63 | 14.3 |
| `obbba_no_tax_on_overtime` | yes | authoritative | -21,614.6 | -22,081.0 | +2.1% | 0.44 | 34.5 |
| `obbba_no_tax_on_tips` | yes | informational | -1,009.2 | -1,076.2 | +6.2% | 0.75 | 3.1 |
| `pre_subsidy_rent_neutralization` | yes | authoritative | 10,942.3 | 10,483.8 | +4.4% | 0.48 | 39.1 |
| `prior_year_self_employment_neutralization` | yes | informational | 364,226.0 | 374,557.7 | -2.8% | -0.50 | 22.5 |
| `qbi_farm_operations_income_exclusion` | yes | informational | 4,990.2 | 4,612.9 | +8.2% | 0.47 | 8.7 |
| `qbi_farm_rent_income_exclusion` | yes | informational | 1,182.6 | 1,171.3 | +1.0% | 0.04 | 6.3 |
| `qbi_reit_ptp_rate_abolition` | yes | informational | 1,743.9 | 1,744.6 | -0.0% | -0.04 | 25.0 |
| `qbi_wage_property_guardrails_zeroed` | yes | informational | 80,256.2 | 64,141.3 | +25.1% | 0.99 | 1.2 |
| `salt_refund_income_neutralization` | yes | informational | -44.6 | -48.5 | +7.9% | 0.44 | 2.9 |
| `savers_credit_abolition` | yes | informational | 2,097.4 | 1,854.7 | +13.1% | 1.42 | 13.0 |
| `self_employed_health_premium_neutralization` | yes | informational | -1,001.2 | -1,001.3 | +0.0% | 0.92 | 7.6 |
| `spm_unit_energy_subsidy_neutralization` | yes | informational | 3,681.4 | 3,738.0 | -1.5% | -0.16 | 14.5 |
| `ssi_asset_limit_10k_20k` | yes | informational | 1,628.4 | 2,692.8 | -39.5% | -2.09 | 2.7 |
| `ssi_disability_criteria_neutralization` | yes | authoritative | 35,713.4 | 32,954.2 | +8.4% | 0.68 | 35.1 |
| `ssi_take_up_neutralization` | yes | authoritative | 60,701.3 | 62,558.6 | -3.0% | -0.38 | 59.6 |
| `tip_income_neutralization` | yes | informational | -1,009.2 | -1,076.2 | +6.2% | 0.75 | 3.1 |
| `tx_snap_additional_vehicle_exemption_abolition` | **no** | informational | 0.0 | 19.7 | -100.0% |  | 0.0 |
| `unrecaptured_section_1250_gain_neutralization` | yes | informational | 3,505.1 | 3,463.2 | +1.2% | 0.13 | 4.9 |
| `voluntary_filing_aca_ptc_neutralization` | yes | informational | 4,382.2 | 6,389.4 | -31.4% | -1.35 | 3.7 |
| `wic_claim_neutralization` | yes | informational | 6,518.5 | 6,617.2 | -1.5% | -0.16 | 18.3 |
| `workers_compensation_snap_exclusion` | yes | informational | 369.1 | 288.5 | +28.0% | 0.41 | 1.3 |

## Cost

Other sessions saturated the host for most of the run (load 100 to 140, swap
nearly full), so wall times overstate the work. CPU seconds are the better
measure.

| Stage | Wall (s) | CPU (s) | Peak RSS (GiB) |
|---|---|---|---|
| Reform-coverage smoke | 18,452 | 6,264 | 12.2 |
| Reform validation | 9,627 | 7,611 | 20.8 |
| Demographics | 36 | 33 | 20.9 |
| QRF tail (source and subsample) | 92 | 15 | 1.3 |
| Take-up participation | 31 | 12 | 0.9 |

The process peaked at 20.9 GiB (psutil, sampled every 0.5 s). `passes.jsonl`
records RSS before and at peak for each of the batched scorer's 99 passes:

- 0.8 GiB before the smoke's baseline pass, and 7.4 GiB after it;
- 12.2 GiB by the end of the smoke, and 20.8 GiB by the end of validation;
- two dips mid-smoke, to 3.8 and 0.8 GiB, while the host was swapping.

Under memory pressure macOS compresses and swaps pages, so these RSS figures
understate the footprint. This run does not show what holds the memory.
