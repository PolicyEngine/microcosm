# Pensioner taxpayer targets: baseline and candidate builds (microcosm#1069)

Two local UK national builds, run on 2026-10-28 with `--no-staging --no-publish-orrery`. Neither is a release (`release_authorized: false`). The files here are aggregates only.

| | Baseline | Candidate |
|---|---|---|
| Source | main 7d45c0c33 | this branch at 8925039d |
| Build id | `uk-frs-calibration-attempt-20261028T035853Z-d46150c3` | `uk-frs-calibration-attempt-20261028T052241Z-132af9e6` |
| H5 sha256 | `de754f0b…96cc` | `2834c9e8…d0df` |
| Targets | 1,273 | 1,297 (+24 `hmrc_itl_age`) |

What both builds share, from `build_provenance.json`:

- **Spine:** the same `spine_build_state` (`fb8b4a99…c650`) and `spine_provenance` digests, and the same input H5.
- **Engine:** policyengine-uk 2.162.8.
- **Feed:** Chronicle 272c987, with the same facts digest.

The candidate was built at 8925039d. The commits after it change no compiled target value: the national compile-parity receipt, regenerated from the pinned feed at 1526b4fd, is byte-identical to the committed one. They change:

- comments, contract notes and the vendored file's purpose text, which no compiled value reads;
- the spine fixture's digest for that file;
- the State Pension age applier, which now refuses a year whose twelve areas exceed the United Kingdom count. The vendored areas are 142,000 to 186,000 below it in every year, so the refusal never fires;
- the appliers, which lose an amount index that no target declares;
- the parity receipt's routing, which matches exact prefixes and routes the same rows;
- tests and these evidence scripts.

## Files

- **`pensioner_taxpayers_{baseline,candidate}.json`**, from `scripts/measure_pensioner_taxpayers.py` on each H5 at 2025, weighted. It holds:
  - taxpayers of State Pension age and aged 65 and over in the twelve region-tier areas;
  - State Pension recipients, split into taxpaying and non-taxpaying (the two sum to the recipients).
- **`allowance_cost_{baseline,candidate}.json`**, from `scripts/allowance_cost.py`. It runs uk-autumn-budget-dashboard-2026 PR #182's State Pension allowance at dashboard commit fa9c42b1, both designs, through policyengine.py 6.11.1 (policyengine-uk 2.151.0, the dashboard's engine), over 2027-28 to 2031-32.
- **`build_comparison.json`**, from `scripts/compare_builds.py`. It gives each build's target, solved estimate and relative error for:
  - the SPI State Pension and private pension band rows;
  - the DWP and DfC recipient rows;
  - the age rows. The baseline doesn't bind these, so its values are the measured ones against the candidate's targets.
- **`hmrc_implied_recipients.json`**, from `scripts/hmrc_implied_recipients.py`. It holds the taxpaying and non-taxpaying recipients HMRC implies, the inputs they come from, and each build's split against them.
- **`build_provenance.json`**, from `scripts/build_provenance.py` on each build's `build.json`, `build_record.json` and `rowwise_candidate_manifest.json`. It holds:
  - the microcosm commit each build ran from, both clean trees;
  - the build ids and `release_authorized`;
  - the H5, spine and feed digests;
  - the engine version the measure provider ran.
- **`state_pension_band_totals.json`**, from `scripts/state_pension_band_totals.py` on main's and this branch's compile-parity receipts. Per band it gives:
  - the compiled 2025 value before and after the re-index;
  - each band's share of the total;
  - the raked alternative.

## Results

### Fit

| Rows | Baseline | Candidate |
|---|---|---|
| Taxpayers of State Pension age, 12 areas (target 8,841,750) | 8,932,401 (+1.03%), not bound | 8,833,654 (−0.09%) |
| Taxpayers aged 65 and over, 12 areas (target 9,317,750) | 9,437,320 (+1.28%), not bound | 9,310,812 (−0.07%) |
| SPI State Pension count, 12 bound bands, mean absolute error | 1.96% | 0.50% |
| SPI State Pension count, £20,000–30,000 band | +17.9% | +4.1% |
| SPI State Pension amount, 12 bound bands, mean absolute error | 2.39% | 0.87% |
| SPI State Pension amount, £20,000–30,000 band | +20.0% | +6.1% |
| SPI private pension count, mean absolute error | 0.62% | 0.41% |
| SPI private pension amount, mean absolute error | 0.83% | 1.10% |
| SPI private pension amount, £15,000–20,000 band | −7.8% | −11.1% |
| Solve final loss | 0.00553 | 0.00536 |
| Rows within 10% | 98.98% | 98.84% |

The count and amount totals differ between builds because the candidate re-indexes the targets.

### State Pension recipients

The recipient count is the same in both builds: 12.371m (12,370,841 in the baseline and 12,371,640 in the candidate), against the DWP and DfC target of 12,383,022.

| | Baseline | Candidate | HMRC-implied |
|---|---|---|---|
| Taxpaying recipients | 8,839,850 | 8,743,572 | 8.65m–8.81m |
| Non-taxpaying recipients | 3,530,991 | 3,628,068 | 3.58m–3.74m |

The HMRC-implied range is computed in `hmrc_implied_recipients.json`:

- SPI 2023-24 counts 7.98m taxpayers with a State Pension (Table 3.6, all ranges). Over ITL Table 2.1's 8.16m taxpayers of State Pension age in 2023-24, that is a share of 97.79%.
- Holding that share, it is applied to the calendar-2025 window of two levels:
  - the twelve region-tier areas, 8,841,750, giving 8,646,711;
  - the United Kingdom row, 9,005,000, giving 8,806,360.
- Non-taxpaying recipients are the bound DWP and DfC recipients, 12,383,022, minus each end.

Both of the candidate's counts fall inside the range, and both of the baseline's fall outside it.

### Sole-income State Pension allowance

This is PR #182's design: State Pension the only income, savings interest disregarded. The table gives the people directly affected and the change in the government balance.

| Year | Baseline | Candidate |
|---|---|---|
| 2027-28 | 1,234,867, −£84m | 1,326,038, −£93m |
| 2028-29 | 1,366,972, −£149m | 1,463,651, −£164m |
| 2029-30 | 1,534,480, −£223m | 1,641,495, −£244m |
| 2030-31 | 1,724,990, −£310m | 1,834,565, −£338m |
| 2031-32 | 1,731,415, −£330m | 1,841,789, −£361m |

The `total_income` design is the module default on the dashboard's main since then. It changes as follows:

- 2027-28: from 574,240 people and −£28m to 593,358 and −£30m;
- 2031-32: from 1,214,067 and −£148m to 1,252,129 and −£154m.
