# Why Populace US state child SPM poverty rates miss Census

State child SPM poverty on the default Populace US file misses Census P60-290 Table 18 by 4.8 points on average, and 38 of 51 jurisdictions fall outside Census's margin. This note splits that miss into sampling noise, weight concentration and systematic error, decomposes the largest gaps, and lists what would close them.

Model figures come from real PolicyEngine US simulations at 2.90.2 (`469ad12cb957`), one state cut at a time as `slideshows/appam-2026/analysis/state_baseline.py` on PolicyEngine/policyengine-slides#95 does. Lines marked arithmetic are not re-simulations. No poverty rate entered a calibration target: these comparisons are holdouts.

## What we found

1. **Noise drives most of the state-to-state scatter on the default file.** A state rate carries a replicate standard error of about 3.6 points, so a 90% interval runs about ±6 points. Child-weighted effective samples run from 26 units in Maine to 310 in California. Noise accounts for about 60% of the cross-state variance in the gaps.
2. **A common upward shift of about 3 points sits underneath it.** On the dense file, which halves the noise, the mean gap stays at +3.6.
3. **State-specific error with a standard deviation of about 3 points survives more data.** Delaware, Florida, Maryland, Massachusetts, Louisiana, Mississippi and Hawaii keep their gaps on the dense file. Tennessee, Rhode Island, Vermont, Missouri, DC and Montana lose most or all of theirs.
4. **The rent imputation puts rent on the wrong households.** About 91% of base households carry another household's rent, in every Populace US file we checked, including the local-area file. PolicyEngine/microcosm#1444 fixes it. {{RENT_SUMMARY}}
5. **The default file is not the state path.** `policyengine.py` routes state regions to the ACS-local file. {{ACS_SUMMARY}}

## What the comparison measures

Table 18 averages income years 2023 to 2025 and carries a margin of 1.0 to 3.7 points. The model column is PolicyEngine's 2025 rate.

| Measure | Default (Build P) | Dense (`b5db384`) | ACS-local (Build O 10/17) |
| --- | --- | --- | --- |
| Household records | 57,240 | 352,932 | 1,586,908 |
| Mean gap | +3.24 | +3.62 | {{ACS_MEAN}} |
| Standard deviation of gaps | 5.11 | 4.02 | {{ACS_SD}} |
| Mean absolute gap | 4.80 | 4.33 | {{ACS_MAD}} |
| Correlation with Census | .52 | .65 | {{ACS_CORR}} |
| Outside Census's margin, of 51 | 38 | 38 | {{ACS_BEYOND}} |
| Median state standard error | 3.57 | 1.86 | {{ACS_SE}} |
| Median child Kish sample, source households | 59 | 305 | 100 |
| State-specific standard deviation τ | 3.25 | 3.07 | not fitted |

The default file's standard error comes from 80 replicate calibrations: a Poisson bootstrap of source households, then a full re-solve. The linearized standard error, 4.04, runs a little higher. The dense and ACS-local standard errors linearize the ratio and cluster on source household. Census's own root-mean-square standard error is 1.35.

## Noise and weight concentration

Calibration nearly doubles the default file's noise. Initial weights give a median state standard error of 2.20 and a Kish sample of 42,355; published weights give 4.04 and 9,174. In Massachusetts, 1,279 of 1,684 SPM units carry weights below 1, so a few records set the state's rate.

We tested whether noise alone could produce the pattern. A null with a common shift plus replicate noise gives a mean absolute gap of 4.25 (95% range 3.56 to 4.98) and a gap standard deviation of 4.12 (3.40 to 4.95). The observed 5.11 falls outside that range. Adding a state-specific τ of 3.25 reproduces the observed 4.94 and 5.23. A DerSimonian-Laird fit on the replicate standard errors gives Q = 88.4 on 50 degrees of freedom. So noise is necessary to explain the table, and it is not sufficient.

The dense file confirms the split. It halves the standard error and leaves τ at 3.07, and its gaps correlate .65 with the default file's.

## Decomposition of the largest gaps

Each row adds up to the gap on the dense file, at its own weights and records:

| Step | DC | DE | TN | MD | FL | LA | MT | State mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASEC period and sample | -2.9 | 1.6 | -2.0 | 1.4 | -0.8 | -2.1 | -1.7 | -0.34 |
| Calibration | 8.8 | 4.8 | 6.1 | 3.7 | 6.5 | 2.3 | 3.7 | 2.74 |
| Thresholds | 2.7 | 3.2 | 2.4 | 2.2 | 5.9 | 2.4 | 1.9 | 2.89 |
| Cash income | -1.8 | 1.0 | -0.6 | 0.6 | 0.2 | -4.7 | -0.3 | -0.79 |
| SNAP, housing and other in-kind | -1.1 | 0.3 | -1.6 | -2.1 | -3.8 | -5.0 | -2.3 | -2.16 |
| Taxes and expenses | -1.2 | -0.9 | -1.1 | 1.2 | 0.3 | 0.0 | -1.5 | -0.01 |
| 2024 to 2025 | 1.2 | 3.0 | 2.0 | 1.5 | 3.3 | 0.1 | 1.3 | 1.28 |
| **Gap** | **5.8** | **13.0** | **5.3** | **8.5** | **11.6** | **-7.0** | **1.0** | **3.62** |

The steps run in order. The first compares Table 18 with the ASEC 2022 to 2024 pool at survey weights. Calibration compares Census's own poverty flag at initial and final weights. An exact Shapley over nine swapped components then moves from Census's resources and thresholds to PolicyEngine's for 2024, and the last step moves the model to 2025. `outputs/dense_steps.csv` and `outputs/default_steps.csv` hold all 51 states and the nine components.

Three readings follow from the state table:

| State | Census | Default gap (SE) | Dense gap (SE) | ACS-local gap (SE) | Reading |
| --- | --- | --- | --- | --- | --- |
| DC | 20.3 | +14.1 (6.1) | +5.8 (3.4) | {{DC}} | Mostly noise |
| DE | 8.7 | +13.7 (4.4) | +13.0 (3.2) | {{DE}} | Persists; donors hold 25.4, ACS rows 15.0 |
| TN | 10.0 | +13.2 (3.9) | +5.3 (1.8) | {{TN}} | Half noise, half systematic |
| MD | 10.7 | +12.5 (4.7) | +8.5 (2.8) | {{MD}} | Persists at a lower level |
| FL | 18.4 | +9.8 (2.8) | +11.6 (1.7) | {{FL}} | Persists on the donors; ACS rows sit at 13.3 |
| LA | 20.7 | -8.6 (3.3) | -7.0 (1.7) | {{LA}} | Persists; cash and SNAP run high |
| MT | 9.3 | -6.8 (3.4) | +1.0 (1.5) | {{MT}} | Noise on the default file |

## The common shift

**Calibration moves weight toward poorer households.** Census's flag reads 13.03 at dense initial weights and 16.31 at final weights. On the default file, re-solves that drop one target family at a time show the balance:

| Targets kept | Census-flag rate | PE 2025 rate |
| --- | --- | --- |
| Population only | 15.01 | 16.43 |
| Population plus SNAP, SSI and ACA | 19.06 | 20.97 |
| Population plus Medicaid | 18.65 | 20.65 |
| Population plus all SOI tax targets | 6.87 | 6.69 |
| Population plus CTC only | 11.88 | 12.57 |
| All, as published | 16.50 | 17.17 |

Benefit administrative targets pull poverty up, tax targets pull it down, and no target anchors a state, child and income cell. On the dense file, income targets start within 10% (AGI -4.2%) while benefit receipt starts far below target (SNAP -37.8%, Medicaid -26.2%, EITC -50.8%, ACA -43.9%). Those misses line up with the +3.3 point rise, though we ran the ablations on the default file only. A chi-square anchor to design weights lowers noise and leaves the shift alone: at λ = .03 the median standard error falls to 2.78, and the mean gap rises to 4.77.

**Pooled records keep nominal income from their source year.** PolicyEngine holds every record to a 2024 threshold, while a 2022 record's market income keeps its nominal 2022 value. That accounts for 2.3 to 2.9 points of the thresholds step. Aging 2022 and 2023 ASEC records by Census's growth in 20th-percentile child resources (a real national simulation on the default file) lowers the national 2025 rate from 17.17 to 16.29, the mean gap from 3.24 to 2.45, and the mean absolute gap from 4.80 to 4.05, with 33 states outside the margin rather than 38. `outputs/cf_state.csv` holds each state.

**The model overshoots from 2024 to 2025.** PolicyEngine rises 1.48 points while Census holds at 13.4. Reference thresholds match Census exactly for 2025, so the difference sits in resources. Census's child resources grew 4.2% to 6.1% across the 10th to 50th percentiles, and PolicyEngine's employment income grew 3.97%. The adult SSN requirement in `gov.irs.credits.ctc.adult_ssn_requirement_applies` starts in 2025 and removes refundable CTC from units holding 2.48% of child weight; 83% of that weight has no adult with an eligible SSN card. It explains only 11% of the movement into poverty.

**Partnership and S-corp income stays out of SPM resources** (PolicyEngine/policyengine-us#9919). Adding it before tax lowers the 2024 rate from 16.62 to 15.29 on the dense file, all of it on PUF records. This line is arithmetic on simulation output.

**Offsets hide part of the shift.** PolicyEngine's SNAP matches USDA totals after calibration and runs about twice Census's recorded value on low-income child units, which pulls poverty down 2.7 to 3.4 points.

## The rent bug

`impute_us_pre_subsidy_rent` draws rent for households in table order, and `person_rent[head_mask] = predicted` writes those draws to heads in person order. The two orders agree for 9.15% of households.

| Pairing, 168,852 base households | Rented | Owned with mortgage | None |
| --- | --- | --- | --- |
| Rent on the household's own head, as published | 40.8% | 27.1% | 25.1% |
| The n-th head's rent with the n-th household's tenure | 100% | 0% | 0% |

The second row is the test: it restores the link exactly, so the draws are right and only their destination is wrong. The default, dense and ACS-local files all carry the same draws. ACS rows inherit the problem through the gap fill: 39.6% of ACS renters carry rent, and so do 30% of ACS owners, even though the file stores native `RNTP` for every renter. The build's `positive_rent_nonrenter` receipt recorded 30,838 such heads, and `test_signal_gate_preserves_archived_qrf_rent_outside_renter_household` allowed them.

The harm reaches poverty through `hud_gross_rent`. `hud_hap` takes the lesser of the maximum subsidy and gross rent less tenant payment, so a Census recipient with no rent gets nothing. On the default file, 74% of Census housing recipients among child units on ASEC rows get no model subsidy. {{RENT_TABLE}}

A second cause sits on the PUF half. `impute_us_housing_assistance_to_puf_support` thresholds a person-level QRF at 0.5, which leaves 2.05% of PUF-half units as recipients against 3.73% on the ASEC half (2.82% against 7.75% on low-income child units), and only 9.6% of ASEC recipients keep the flag on their PUF clone.

## What this means for the Child Poverty Impact Dashboard

The dashboard reads ACS-local slices, not the default file. Master serves Build P's August 19 slices under policyengine-us 1.825.2. PolicyEngine/child-poverty-impact-dashboard#127, still a draft, moves it to Build O 10/17 through `policyengine.py` 6.11.1.

{{DASHBOARD}}

Its stored audits give a sense of the instability. The 2021 CTC reform's 2024 state effect swings between builds: Arizona from -10.3 to -3.8 points, DC from -1.3 to -8.1, North Carolina from -2.9 to -6.3.

So state baselines and state reform effects on the dashboard both need a stated uncertainty, and neither should read as a point estimate to the tenth. The legacy `backend/` and `cpid_calc/` paths read `enhanced_cps_2024`, and PR #127 says the frontend no longer calls them.

## What would fix it

{{FIXES}}

## Limits

- Corrected-rent runs keep the published weights. Calibration fit those weights to outcomes that used misplaced rent, so a rebuild can move results further.
- The #1078 solves come from the 2026-09-23 Build O checkpoint. We applied them to outcomes from the 10/17 file, which shares its household ids.
- The aged run scales 42 income columns on 2022 and 2023 ASEC records by a single child-resource factor. PUF records stay as they are, and a real fix would age each income source separately.
- ACS-local cuts add `spm_unit_spm_universe_status` as `INCLUDED` everywhere, because the release predates that input. The dashboard's audit found masking moves child rates by a negligible amount.
- The ablations ran on Build P, not the dense file.
- We did not verify why Census records less SNAP than administrative totals imply.

## Reproduction

Scripts sit in `scripts/` and write beside themselves under `run/` and `asec/`. `outputs/` holds the tables behind this note, and `receipts.json` pins the inputs.

1. `extract_state.py`, `assemble.py`, `cutrun.py`, `assemble_dense.py`: state cuts of each file.
2. `noise.py`, `vardec.py`, `rep_boot.py`, `rep_eval.py`, `dl.py`, `dense_fit.py`: noise.
3. `decomp.py`, `decomp2.py`, `steps.py`, `year.py`, `paired.py`, `thr.py`: decomposition.
4. `ablate_eval.py`, `ablate_noise.py`, `fam.py`, `chi_solve.py`, `chi_eval.py`, `dmiss.py`: calibration.
5. `national_cf.py`, `cf_eval.py`, `ctc_ssn.py`, `part.py`: aged incomes, the CTC reform and partnership income.
6. `rent2.py` to `rent5.py`, `rentfix.py`, `dense_rent.py`, `sparse_rent.py`, `runcuts.py`, `rent_eval.py`, `house.py`, `house_sim.py`, `puf_house.py`: rent and housing.
7. `acs_hh.py`, `acs_more.sh`, `acs_one.py`, `acs_eval.py`, `acs_rent_cut.py`, `acs_rent.sh`, `acs_stored.py`: the ACS-local path.
