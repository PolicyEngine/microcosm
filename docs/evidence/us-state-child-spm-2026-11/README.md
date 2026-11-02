# Why Populace US state child SPM poverty rates miss Census

State child SPM poverty on the default Populace US file misses Census P60-290 Table 18 by 4.8 points on average, and 38 of 51 jurisdictions fall outside Census's margin. This note splits that miss into sampling noise, weight concentration and systematic error, decomposes the largest gaps, and lists what would close them.

Model figures come from real PolicyEngine US simulations at 2.90.2 (`469ad12cb957`), one state cut at a time as `slideshows/appam-2026/analysis/state_baseline.py` on PolicyEngine/policyengine-slides#95 does. Differences, standard errors and fits are arithmetic on those runs' output, and the two figures that change an input without re-simulating say so. No poverty rate entered a calibration target: the dense file's 6,102 target names include none. That makes these comparisons holdouts of the target, not of the sample, because the ASEC pool and Table 18 share households.

## What we found

1. **Noise accounts for most of the default file's state scatter, and the data pin that share down poorly.** A state rate carries a replicate standard error of about 3.6 points, or about ±6 at 90%, and that error runs low because it ignores CPS design. Child-weighted effective samples run from 19 in DC to 277 in California. Three fits put noise at 60%, 71% and 80% of the cross-state variance, and the first carries a 95% interval of about 10% to 89%.
2. **On the dense pool, calibration creates most of the systematic error, and most of the added state error lands on ASEC rows.** At initial weights, the mean gap reads 0.00, the mean absolute gap 1.78, and 16 jurisdictions fall outside Census's margin. At published weights those read +3.62, 4.33 and 38. ASEC rows go from a mean absolute gap of 2.44 to 7.30, and PUF rows from 2.95 to 3.16. State-specific error rises about 1.7 times, from 1.88 to 3.20 points, once we allow for households the pool shares with Table 18, and a test with no shared households keeps the direction. This result covers the dense pool only. On the default file, support selection puts the shift in place before calibration starts (+3.27 at initial weights) and state-specific error stays flat, and the ACS-local file stores no initial weights. The dense zero nets +2.20 on ASEC rows against -2.20 on PUF rows, and initial weights miss the administrative targets, so this result locates the problem and does not recommend skipping calibration.
3. **Some state error predates calibration, and we cannot yet pin it on particular states.** At dense initial weights, state gaps still exceed their sampling noise (Q = 170.5 on 50 degrees of freedom after allowing for shared households). Louisiana has the largest joint miss, about 2.8 standard errors the same way at dense initial weights and on ACS rows, which draw on another survey. A miss that large on both turns up by chance about one time in five (p = .19), so Louisiana gives a lead and not a finding. Florida, Massachusetts and Maryland sit within 3 points at dense initial weights and mostly miss after calibration.
4. **The rent imputation puts rent on the wrong households.** Of records that should carry rent, 89% do not hold their own, in every Populace US file we checked, including the local-area file. PolicyEngine/microcosm#1444 carries the fix. With published weights held fixed, the fix nearly doubles modelled housing subsidies on the dense file, from $7.3bn to $13.9bn, and lowers the 2025 child rate 0.7 points. It barely moves state fit: the mean absolute gap goes from 4.33 to 4.13 on dense, 4.80 to 4.71 on default and 3.18 to 3.21 on ACS-local, where contract rent stands in on ACS rows. It lifts SNAP by less than 1%. Across 15 states, it moves a state's 2021 CTC effect by a median of 0.29, and by more than a point in Delaware and Florida, where a few donor households drive the swing.
5. **The default file is not the state path.** `policyengine.py` routes state regions to the ACS-local file. There the mean gap falls to +0.88 and the mean absolute gap to 3.18, 29 jurisdictions fall outside Census's margin, and 23,407 donor households hold 68% of child weight. ACS rows alone reach a mean absolute gap of 2.46 and a correlation of .69 with a median standard error of 1.06, and still leave 22 outside. Raising the ACS share of the mass does the most of any change we ran on this path: 2.79 to 2.16, or 2.88 to 2.23 with corrected rent, and the median error on a state's reform effect falls from 1.30 to 0.56.

## What the comparison measures

Table 18 averages income years 2023 to 2025 and carries a margin of 1.0 to 3.7 points. The model column is PolicyEngine's 2025 rate. Standard errors below linearize the ratio and cluster on source household, counting an ASEC record and its PUF clone as one cluster, so the columns compare like with like. Q and τ first treat model and Census as independent. The two rows after them subtract the covariance from households they share, as the section on where the error enters explains.

| Measure | Default, initial | Default, published | Dense, initial | Dense, published | ACS-local (Build O 10/17) |
| --- | --- | --- | --- | --- | --- |
| Household records | 57,240 | 57,240 | 352,932 | 352,932 | 1,586,908 |
| Mean gap | +3.27 | +3.24 | 0.00 | +3.62 | +0.88 |
| Standard deviation of gaps | 4.01 | 5.11 | 2.40 | 4.02 | 4.12 |
| Mean absolute gap | 4.21 | 4.80 | 1.78 | 4.33 | 3.18 |
| Correlation with Census | .65 | .52 | .78 | .65 | .54 |
| Outside Census's margin, of 51 | 36 | 38 | 16 | 38 | 29 |
| Median state standard error | 2.20 | 4.04 | 1.07 | 1.86 | 2.98 |
| Median child-weighted Kish sample | 211 | 51 | 635 | 305 | 88 |
| Q on 50 degrees of freedom | 109.5 | 71.5 | 94.4 | 147.8 | 73.3 |
| State-specific standard deviation τ | 2.74 | 2.73 | 1.50 | 3.07 | 2.02 |
| Q, allowing for shared households | 139.5 | 76.0 | 170.5 | 188.4 | not run |
| τ, allowing for shared households | 2.95 | 2.90 | 1.88 | 3.20 | not run |

Default is Build P; dense is `b5db384`. The published snapshot does not store ACS-local initial weights, so that column has no initial counterpart. The default file also has a replicate standard error from 80 replicate calibrations: a Poisson bootstrap of source households, then a full re-solve. Its median reads 3.57, and its ratio to the linearized error runs from .53 in West Virginia to 2.17 in Montana. Census's own root-mean-square standard error is 1.35.

## Noise and weight concentration

Calibration nearly doubles the default file's noise. Initial weights give a median state standard error of 2.20 and a national Kish sample of 42,355; published weights give 4.04 and 9,174. In Massachusetts, 1,279 of 1,684 SPM units carry weights below 1, so a few records set the state's rate.

We tested whether noise alone could produce the pattern, using a common shift plus replicate noise:

| Statistic | Observed | Null median | Null central 95% | One-sided p |
| --- | --- | --- | --- | --- |
| Mean absolute gap | 4.80 | 4.25 | 3.44 to 5.13 | .10 |
| Standard deviation of gaps | 5.11 | 4.12 | 3.27 to 5.12 | .03 |
| Outside Census's margin | 38 | 36 | 30 to 42 | .31 |

Only the standard deviation rejects, and adding a state-specific τ of 3.25 lifts the null medians to 4.94 and 5.23, just past the observed values. A DerSimonian-Laird fit gives Q = 88.4 and τ = 3.25, with a Q-profile 95% interval of 1.71 to 4.84. That interval puts noise's share of the variance anywhere from about 10% to 89%, around a point estimate of 60%. Replicates also centre away from the published values, by 1.83 points on average and by -6.93 in Tennessee, -5.76 in Rhode Island and +4.92 in Montana. Measuring error around the published values gives Q = 64.8 (p = .08), τ = 2.28 and a noise share of 80%, and linearized errors give 71%, so the default file alone barely separates noise from state-specific error.

These standard errors run low. Replicates hold support, the target matrix and imputations fixed, and source-household clusters ignore CPS strata, primary sampling units and rotation overlap. τ absorbs that, along with the mismatch between one model year and a three-year Census average.

Firmer evidence comes from the dense file (Q = 147.8, τ = 3.07, 95% interval 2.33 to 4.34) and ACS rows (Q = 131.6, τ = 2.08, 95% interval 1.62 to 3.22). The dense file adds records, not surveys: every channel maps to the same ASEC households, so its .65 gap correlation with the default file partly reflects shared sample noise. ACS rows give the only independent sample. Their gaps correlate .30 with the dense file's at initial weights and .42 at published weights, but both gaps subtract the same Census estimate, and that alone would give about .16 and .09. ACS rows also take tax and rent inputs from donors by QRF, so they can share model error, and they carry published weights.

## Where the error enters

Calibration moves both halves of the dense pool up:

| Dense file, 2025 | Initial weights | Published weights |
| --- | --- | --- |
| All rows, child rate | 13.86 | 18.19 |
| ASEC rows, child rate | 16.38 | 22.13 |
| ASEC rows, mean gap | +2.20 | +6.98 |
| ASEC rows, correlation with Census | .88 | .68 |
| ASEC rows, τ | 1.28 | 4.55 |
| PUF rows, child rate | 11.34 | 15.16 |
| PUF rows, mean gap | -2.20 | +0.98 |
| PUF rows, correlation with Census | .55 | .52 |
| PUF rows, τ | 2.43 | 2.82 |

ASEC rows at initial weights hold the tightest state pattern in this note, with a +2.20 shift, partly because they compare the survey with itself. PUF rows start 5 points below them, so the dense pool's zero comes from two offsetting gaps.

The initial-weight columns flatter the pool. The pool holds income years 2022 to 2024 and Table 18 averages 2023 to 2025, so they share households, and ASEC rows at initial weights reproduce Census's own state rates on Census's flag (correlation .98 or more in each year). In raw ASEC, the 2022 to 2024 average correlates .96 with Table 18, while 2022 and 2025 correlate .68. Subtracting the covariance from shared clusters moves τ from 1.50 to 1.88 (95% interval 1.41 to 2.59) at initial weights and from 3.07 to 3.20 (2.57 to 4.48) at published weights. So state-specific error rises about 1.7 times, not double, and the intervals just overlap. That adjustment ignores rotation overlap and design effects, so it still understates the covariance.

A stricter test uses records that share no household with their benchmark: dense 2022 records against Census's 2024 to 2025 average in raw ASEC.

| 2022 records against Census 2024-25 | Initial weights | Published weights |
| --- | --- | --- |
| All rows, mean absolute gap | 2.47 | 6.08 |
| All rows, mean absolute gap around the column's mean | 2.33 | 4.36 |
| All rows, correlation | .67 | .58 |
| All rows, median state standard error | 1.78 | 3.13 |
| ASEC rows, mean absolute gap | 3.44 | 9.44 |
| ASEC rows, correlation | .80 | .60 |
| PUF rows, mean absolute gap | 3.13 | 4.72 |
| PUF rows, correlation | .41 | .42 |

The direction survives. This test uses a third of the records, so both columns carry more noise, and calibration targets the pool rather than a single source year, so the published column may overstate the damage. Still, the 1.78 and .78 at initial weights overstate how well the pool fits, and calibration still adds state error.

The calibration step differs by flag and by file:

| Calibration step, state mean | Census's flag | PE 2024 | PE 2025 | Largest state moves, PE 2025 |
| --- | --- | --- | --- | --- |
| Dense | +2.74 | +3.33 | +3.62 | -2.06 to +10.75 |
| Default | +1.58 | -0.05 | -0.03 | -10.31 to +8.36 |

On the default file, calibration nets to nothing on PolicyEngine's flag and still moves a typical state 3 points. The shift arrives earlier: selecting 57,240 records from the pool moves Census's flag from 13.03 to 15.06 at initial weights.

## Decomposition of the largest gaps

Each column adds up to the gap on the dense file, at its own weights and records:

| Step | DC | DE | TN | MD | FL | LA | MT | State mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASEC period, sample and selection | -2.9 | 1.6 | -2.0 | 1.4 | -0.8 | -2.1 | -1.7 | -0.34 |
| Calibration | 8.8 | 4.8 | 6.1 | 3.7 | 6.5 | 2.3 | 3.7 | 2.74 |
| Thresholds | 2.7 | 3.2 | 2.4 | 2.2 | 5.9 | 2.4 | 1.9 | 2.89 |
| Cash income | -1.8 | 1.0 | -0.6 | 0.6 | 0.2 | -4.7 | -0.3 | -0.79 |
| SNAP, housing and other in-kind | -1.1 | 0.3 | -1.6 | -2.1 | -3.8 | -5.0 | -2.3 | -2.16 |
| Taxes and expenses | -1.2 | -0.9 | -1.1 | 1.2 | 0.3 | 0.0 | -1.5 | -0.01 |
| 2024 to 2025 | 1.2 | 3.0 | 2.0 | 1.5 | 3.3 | 0.1 | 1.3 | 1.28 |
| **Gap** | **5.8** | **13.0** | **5.3** | **8.5** | **11.6** | **-7.0** | **1.0** | **3.62** |

The steps run in order. The first compares Table 18 with the ASEC 2022 to 2024 pool at survey weights, and includes record selection. Calibration compares Census's own poverty flag at initial and final weights. An exact Shapley over nine swapped components then moves from Census's resources and thresholds to PolicyEngine's for 2024, and the last step moves the model to 2025. `outputs/dense_steps.csv` and `outputs/default_steps.csv` hold all 51 states and the nine components.

The order matters. The Shapley runs at final weights only, so it says nothing about how weights interact with the swapped components, and the table above shows the calibration step changing with the flag.

Ten states read this way, with standard errors in brackets:

| State | Census | Default gap | Dense, initial | Dense, published | ACS-local | ACS rows | ACS-local donors | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DC | 20.3 | +14.1 (10.6) | -4.9 (1.7) | +5.8 (3.4) | +0.9 (5.3) | +1.8 (5.5) | +0.3 | Too noisy to read on any file |
| DE | 8.7 | +13.7 (6.9) | +5.3 (1.7) | +13.0 (3.2) | +13.5 (6.9) | +6.3 (2.9) | +16.7 | High everywhere, and donors widen it |
| TN | 10.0 | +13.2 (5.1) | -0.1 (0.9) | +5.3 (1.8) | +6.6 (3.1) | +8.3 (1.4) | +5.7 | High on ACS rows, fine on the initial pool |
| MD | 10.7 | +12.5 (6.0) | +2.7 (1.5) | +8.5 (2.8) | +6.9 (4.2) | +0.7 (1.2) | +9.8 | Calibration and donors |
| FL | 18.4 | +9.8 (4.0) | +0.9 (0.9) | +11.6 (1.7) | +4.3 (3.5) | -5.1 (0.6) | +8.7 | Calibration and donors; ACS rows read low |
| MA | 12.8 | +7.4 (5.1) | +0.5 (1.0) | +7.9 (1.9) | +8.6 (4.6) | +2.5 (1.2) | +11.7 | Calibration and donors |
| LA | 20.7 | -8.6 (3.5) | -7.0 (0.9) | -7.0 (1.7) | -9.0 (2.3) | -6.4 (1.4) | -10.1 | Low everywhere; cash and SNAP run high |
| MS | 19.1 | -6.2 (4.0) | -5.2 (1.1) | -4.0 (1.7) | -5.3 (2.9) | -3.2 (1.7) | -6.4 | Low everywhere |
| HI | 13.0 | -6.0 (2.8) | -2.2 (1.2) | -4.2 (1.3) | -7.1 (2.0) | -7.9 (0.8) | -6.7 | Low, and lowest on ACS rows |
| MT | 9.3 | -6.8 (1.6) | -1.1 (0.9) | +1.0 (1.5) | -4.5 (1.3) | +1.2 (2.7) | -7.4 | Low on donors only |

## The common shift

**Calibration moves weight toward poorer households on the dense file.** Census's flag reads 13.03 at dense initial weights and 16.31 at final weights. On the default file, re-solves that keep one target family at a time show the families pulling apart:

| Targets kept, default pipeline | Census's flag | PE 2025 | State correlation |
| --- | --- | --- | --- |
| Initial weights, no solve | 15.06 | 16.70 | .65 |
| Population only | 15.01 | 16.43 | .60 |
| Population plus SNAP, SSI and ACA | 19.06 | 20.97 | .64 |
| Population plus Medicaid | 18.65 | 20.65 | .52 |
| Population plus EITC | 17.15 | 18.55 | .60 |
| Population plus all SOI | 13.69 | 14.41 | .53 |
| Population plus CTC only | 11.88 | 12.57 | .65 |
| All targets, same pipeline | 15.92 | 16.23 | .42 |
| All targets, as published | 16.50 | 17.17 | .52 |

Benefit targets and the EITC pull the rate up, and CTC and joint SOI targets pull it down. With every family in, the national rate lands near population-only, at 16.23 against 16.43, while the state correlation falls from .60 to .42. The last two rows land about a point apart, and they differ in the 11 JCT rows the re-solves drop, the rebuilt matrix and the solver path. `outputs/ablate_eval.csv` holds the drop-one runs too.

On the dense file, benefit receipt starts far below target. Summing each family's cells, which mixes dollars and counts where a family holds both, SNAP starts 37.8% short, Medicaid 26.2%, EITC 50.8% and ACA 43.9%, while AGI starts 4.2% short. Those misses line up with the +3.3 point rise, though we ran the ablations on the default file only.

A chi-square anchor to the default file's initial weights lowers noise and overshoots on level. The grid starts from the same-pipeline re-solve, not from the published weights, so that row anchors it:

| Weights, default file | Mean gap | Correlation | Median state SE | Census's flag |
| --- | --- | --- | --- | --- |
| Published, for reference | 3.24 | .52 | 4.04 | 16.50 |
| Same pipeline, no anchor | 2.93 | .42 | 4.27 | 15.92 |
| λ = .003 | 3.07 | .45 | 3.72 | 16.09 |
| λ = .01 | 3.42 | .52 | 3.28 | 16.40 |
| λ = .03 | 4.77 | .57 | 2.78 | 17.34 |
| λ = .1 | 5.18 | .68 | 2.50 | 17.74 |
| λ = .3 | 4.55 | .69 | 2.38 | 16.90 |
| Initial | 3.27 | .65 | 2.20 | 15.06 |

The mean gap rises above both endpoints, so a pull toward initial weights cannot explain it, and we do not know why it overshoots. Correlation climbs from .42 to .69 across the grid and ends at .65. We did not test an anchor on the dense pool.

**Pooled records keep nominal income from their source year.** PolicyEngine holds every record to a 2024 threshold, while a 2022 record's market income keeps its nominal 2022 value. That accounts for 2.3 to 2.5 points of the thresholds step. Aging 2022 and 2023 ASEC records by Census's growth in 20th-percentile child resources (a real national simulation on the default file) lowers the national 2025 rate from 17.17 to 16.29, the mean gap from 3.24 to 2.45, and the mean absolute gap from 4.80 to 4.05, with 33 jurisdictions outside the margin rather than 38. `outputs/cf_state.csv` holds each state.

**The model overshoots from 2024 to 2025.** PolicyEngine rises 1.48 points while Census holds at 13.4. Reference thresholds match Census for 2025, and swapping Census's 2024 thresholds into the test moves PolicyEngine's 2024 rate only from 15.69 to 15.72, so the difference sits in resources. That swap is arithmetic on simulated resources (`outputs/logs/thr_census.txt`). Between the 10th and 30th percentiles, child SPM resources grew 3.6% to 5.7% in Census's data, against 1.6% to 2.8% on the default file and 2.5% to 3.2% on dense (`outputs/resource_growth.csv`).

The CTC's adult SSN requirement explains little of this. `gov.irs.credits.ctc.adult_ssn_requirement_applies` starts in 2025. Units whose refundable CTC falls to zero that year hold 2.48% of child weight, and 83% of that weight has no adult in the SPM unit with an eligible SSN card. Those units hold 11% of the child weight moving into poverty, so the rule can account for at most that share through this channel.

**Partnership and S-corp income stays out of SPM resources** (PolicyEngine/policyengine-us#9919). Adding it before tax lowers the 2024 rate from 16.62 to 15.29 on the dense file, all of it on PUF records. This line is arithmetic on simulation output.

**Offsets hide part of the shift.** PolicyEngine's SNAP matches USDA totals after calibration and runs about twice Census's recorded value on low-income child units, which pulls poverty down 2.7 to 3.4 points.

## The rent bug

`impute_us_pre_subsidy_rent` draws rent for households in table order, and `person_rent[head_mask] = predicted` writes those draws to heads in person order. The two orders agree for 9.15% of households. On the dense file, 46.4% of records hold the wrong rent, and 89% of records that should carry rent do not hold their own.

| Pairing, 168,852 base households | Rented | Owned with mortgage | None |
| --- | --- | --- | --- |
| Rent on the household's own head, as published | 40.8% | 27.1% | 25.1% |
| The n-th head's rent with the n-th household's tenure | 100% | 0% | 0% |

The second row is the test: it restores the link exactly, so the draws are right and only their destination is wrong. The default and dense files carry the same draws, and so do ACS-local donors.

ACS rows inherit the problem. `build_optional_acs_multispine` passes the base to `transfer_acs_inputs` as its donor, and `pre_subsidy_rent` sits among the transfer targets, so ACS rows learned from misplaced rent: 39.6% of ACS renters carry rent, and so do 30% of ACS owners. The file stores native `RNTP` for all 336,110 renter households, and `acs_transfer.py` leaves it out of this path because contract rent differs from pre-subsidy rent. A 2023 to 2025 pipeline smoke run recorded 30,838 such heads in its `positive_rent_nonrenter` receipt (`experiments/us-asec-2025-source/receipts/pipeline_smoke_2023_2025.json`), and `test_signal_gate_preserves_archived_qrf_rent_outside_renter_household` allowed them.

Rent reaches poverty by several paths. SNAP's shelter deduction, some state TANF programs and state credits read it, and the largest path we measured runs through `hud_gross_rent`. `hud_hap` takes the lesser of the maximum subsidy and gross rent less tenant payment, so a recipient whose rent and utility allowance fall short of the tenant payment gets nothing. On the default file, 74% of Census housing recipients among child units on ASEC rows get no model subsidy. Moving rent back onto the right households, with published weights held fixed, gives:

| All 51 states | Dense, as published | Dense, corrected | Default, as published | Default, corrected |
| --- | --- | --- | --- | --- |
| Modelled housing subsidy, 2024 | $7.26bn | $13.88bn | $4.50bn | $7.51bn |
| SPM units with a subsidy | 1.23M | 2.57M | 0.68M | 1.34M |
| SNAP, 2024 | $89.43bn | $90.14bn | $99.63bn | $100.46bn |
| SNAP at initial weights, 2024 | $57.58bn | $58.86bn | $68.00bn | $70.45bn |
| Child SPM rate, 2024 | 16.62 | 15.88 | 15.69 | 15.20 |
| Child SPM rate, 2025 | 18.19 | 17.49 | 17.17 | 16.81 |
| Mean gap | +3.62 | +3.06 | +3.24 | +3.03 |
| Mean absolute gap | 4.33 | 4.13 | 4.80 | 4.71 |
| Outside Census's margin, of 51 | 38 | 36 | 38 | 34 |

The fix closes 0.56 of the dense file's 3.62 mean gap and 0.21 of the default file's 3.24. State steps run from -2.93 to +0.40 on dense and from -5.38 to +3.02 on default. It leaves the SNAP miss at initial weights in place.

On the ACS-local file, a similar exercise, with contract rent standing in on ACS rows, moves the mean gap from +0.88 to +0.49, the mean absolute gap from 3.18 to 3.21, and the correlation from .54 to .51, with 29 jurisdictions outside Census's margin both ways. ACS rows go from a mean absolute gap of 2.46 to 2.53. So rent matters here for getting housing right, not for state fit at published weights.

The fix moves reform effects about as far as it moves baselines, and those moves show how few households can set a state's effect. A reform pass covered 15 states drawn from the default file's large gaps, not its top 15: it includes Arkansas, Hawaii and Maine and leaves out California, Oklahoma and West Virginia, which has the largest effect. On those states, with published weights held fixed, a state's 2021 CTC effect moves 0.59 on average (median 0.29) and its baseline rate moves 0.47 (median 0.24). The effect moves nearly cancel in the mean (-0.04), though the median of the 15 goes from -3.39 to -2.59 (`outputs/acsrent_reform.csv`).

Only two states move more than a point. Florida goes from -3.53 to -6.27 and Delaware from -2.97 to -1.76, and almost all of each swing sits on donors. In Florida, one donor with 1.21% of the state's child weight moves the effect 1.21 points, and ten source households hold 90% of the gross change. In Delaware, five source households hold all of it (`outputs/logs/fl_reform.txt`). These runs also use stand-in rent on ACS rows, so they show sensitivity and not the rebuilt file's answer.

A separate gap sits on the PUF half. `impute_us_housing_assistance_to_puf_support` replaces each PUF-half person's receipt with a QRF draw, and the unit takes its first person's draw. Unweighted, 2.05% of PUF-half units receive assistance against 3.73% on the ASEC half (2.82% against 7.75% among low-income child units), and 9.6% of ASEC recipients keep the flag on their PUF clone.

## What this means for the Child Poverty Impact Dashboard

The dashboard reads ACS-local slices, not the default file. Master serves Build P's August 19 slices under policyengine-us 1.825.2. PolicyEngine/child-poverty-impact-dashboard#127, still a draft, moves it to Build O 10/17 through `policyengine.py` 6.11.1.

**Baselines.** At 2.90.2, the 10/17 file gives a mean gap of +0.88, a mean absolute gap of 3.18, and 29 of 51 jurisdictions outside Census's margin. Its median state standard error of 2.98 means about ±4.9 points at 90%. A common shift plus model and Census noise gives a mean absolute gap of 2.79 (central 95% 2.22 to 3.43), so noise can carry most of the 3.18. The dashboard's stored audits show Build P fitting better than Build O at the same model: at 1.825.2 the mean absolute gap reads 2.78 against 3.13, with 23 jurisdictions outside the margin against 31. So PR #127's move costs some baseline fit.

**Reform effects.** The 2021 CTC reform lowers the median state's 2025 child rate 2.97 points, with a median standard error of 1.29, or about ±2.1 at 90%. That error runs smaller in points than a baseline's, and larger for its size: the median ratio of standard error to effect reads .45, against .24 for baselines. The largest effects carry the largest errors: West Virginia at -11.85 (4.61), Missouri at -8.96 (3.70) and DC at -8.65 (4.34).

The stored audits give a second reading. At 1.825.2, Build P and Build O differ on a state's 2025 effect by 0.55 on average (standard deviation 0.83, or 0.59 per build), about half the clustered error. Those builds share source records, so that comparison understates sampling noise. Model version moves effects further: the same 10/17 file at 1.825.2 and 2.30.1 differs by 1.11 on average (standard deviation 1.72), and Arizona goes from -11.92 to -6.37.

So state baselines and state reform effects on the dashboard both need a stated uncertainty, and neither should read as a point estimate to the tenth. The rent code predates the August 19 build, so those slices should carry the bug, though we did not test them, and the fix moves some state effects by more than a point. The legacy `backend/` and `cpid_calc/` paths read `enhanced_cps_2024`, and PR #127 says the frontend no longer calls them.

## What would fix it

| Change | Measured on | Mean absolute gap | Mean gap |
| --- | --- | --- | --- |
| No calibration (diagnostic only) | Dense | 4.33 to 1.78 | +3.62 to 0.00 |
| Rent fix, weights fixed | Dense | 4.33 to 4.13 | +3.62 to +3.06 |
| Rent fix, weights fixed | Default | 4.80 to 4.71 | +3.24 to +3.03 |
| Rent fix, weights fixed | ACS-local | 3.18 to 3.21 | +0.88 to +0.49 |
| ACS share 0.5 to 0.9, λ = .03 | ACS-local solves | 2.79 to 2.16 | +0.54 to -0.45 |
| The same solves, corrected rent | ACS-local solves | 2.88 to 2.23 | +0.16 to -0.78 |
| Single-factor income aging | Default | 4.80 to 4.05 | +3.24 to +2.45 |

This table cannot rank the changes, because its rows use different files, and benefit receipt, a dense anchor and the ASEC-PUF gap have no row. So the order below rests on judgement: prerequisites first, then the change we measured on the state path, then changes measured elsewhere or not at all. Aging reaches as far on the default file (0.75) as the ACS share does on ACS-local (0.63), so it could rank higher.

One warning covers steps 3 and 4. Aging and better benefit receipt should lower rates, and we have not tested either on the ACS-local path. That path already reads +0.88, and -0.45 at share 0.9, or -0.78 with corrected rent, so changes sized on the default file could overshoot there.

1. **Put rent back and rebuild.** PolicyEngine/microcosm#1444 carries the assignment fix. It comes first because it corrects a bug and every later step should run on the rebuilt base, not because it closes the largest gap. After a rebuild, tighten `us_housing_inputs_signal_gate` so positive rent outside renter households fails the build, confirm ACS rows pick up the corrected link through the donor transfer, and re-run this note. Calibration then fits to corrected housing, which the fixed-weight runs here cannot show.
2. **Limit what calibration does to states.** Two levers apply:

   - Give ACS records more of the state mass. `--acs-share` raises the weight on real state samples. On all 51 states:

     | Solve | Mean absolute gap | Correlation | Outside margin | Median state SE | Median reform SE |
     | --- | --- | --- | --- | --- | --- |
     | Share 0.5, release-like | 2.79 | .56 | 27 | 2.82 | 1.30 |
     | Share 0.7, λ = .03 | 2.40 | .64 | 23 | 1.86 | 0.88 |
     | Share 0.9, λ = .03 | 2.16 | .75 | 22 | 1.09 | 0.56 |
     | Share 0.964, λ = .03 | 2.14 | .77 | 22 | 0.87 | 0.47 |

     With corrected rent, the share 0.9 row reads 2.23, with 23 outside the margin (`outputs/acsrent_solves.csv`). These standard errors fall, and bias remains: at share 0.9, DC reads 24.47 against Census's 20.3. PolicyEngine/microcosm#1078 puts the cost at 36% and 43% more held-out target error on its two folds at share 0.9 with no penalty, mostly on SOI cells, so the share needs a methodology decision.
   - Test an anchor on the dense pool. Initial weights there carry a mean absolute gap of 1.78, against 4.21 on the default file, so an anchor could keep more of that pattern. This needs a solve; we have not run one.
3. **Close benefit receipt before calibration.** Better take-up and receipt inputs should reduce the tilt, because calibration then has less to do, though we have not tested that. Rent does not close the SNAP gap.
4. **Age pooled incomes** (microcosm#944). A crude single-factor run removes 0.88 points nationally. A real fix ages each income source and draws the pool from 2023 to 2025 to match Table 18.
5. **Close the gap between ASEC and PUF rows.** They start 5 points apart at initial weights, and the dense pool's zero depends on that offset. Start with cash income: at published weights, moving to PolicyEngine's cash income adds 0.75 points on ASEC rows and removes 2.06 on PUF rows, while housing adds 1.65 and 1.16 (`outputs/dense_shapley9_asec.csv`, `outputs/dense_shapley9_puf_tax_detail.csv`). Two known fixes should widen the gap. Carrying the ASEC housing flag to its PUF clone adds subsidy to PUF rows only, and partnership and S-corp income (PolicyEngine/policyengine-us#9919; PolicyEngine/policyengine-us#10573 partly addresses it) takes PUF rows from 13.63 to 11.27 in 2024, at published weights, and leaves ASEC rows at 20.52. Make those for their own reasons, not to close this gap.
6. **Reconcile 2025 resources near the line.** Compare PolicyEngine's and Census's child resource growth by income source, starting with the uprating series for low earners.
7. **Check the state misses that persist.** Rank states by the smaller of their two z-scores at dense initial weights and on ACS rows, counting only misses in the same direction (`outputs/state_persistence.csv`): Louisiana 2.80, Delaware 1.91, Ohio 1.40, Illinois 1.39, Mississippi 1.38 and Washington 1.31. These stay leads. Chance alone gives 0.7 states past 1.96 (we see one, p = .53), 1.4 past 1.645 (two, p = .41) and 3.0 past 1.28 (six, p = .05), and shared Census noise feeds those counts. Start with Louisiana and Delaware, looking at state tax and benefit rules and at Census's state estimates.
8. **Report state child rates as a standing holdout** with standard errors, in every release's receipts. Poverty stays a comparison and never a target.

## Limits

- Corrected-rent runs keep the published weights. Calibration fit those weights to outcomes that used misplaced rent, so a rebuild can move results further.
- The ACS-local corrected-rent run gives donors their base record's rent and gives ACS renter heads twelve times `RNTP`. Contract rent stands in for pre-subsidy rent there, and a rebuild would transfer rent from corrected donors instead.
- The #1078 solves come from the 2026-09-23 Build O checkpoint. We applied them to outcomes from the 10/17 file, which shares its household ids, and that mismatch alone moves the mean absolute gap: 2.79 for the release-like solve against 3.18 for the 10/17 weights. The cost figures come from share 0.9 with no penalty, not from the λ = .03 rows.
- The aged run scales 42 income columns on 2022 and 2023 ASEC records by a single child-resource factor. PUF records stay as they are, the weights stay fit to unaged income, and it tests 0.88 of the 2.3 to 2.5 points the Shapley assigns.
- policyengine-us 2.90.2 has no `spm_unit_spm_universe_status`, so it ignores that column and these runs measure every SPM unit, group quarters included. That choice matters at state level: PR #127 puts the August 19 release's 2024 median relative gap against ACS 2023 at -15.9% with all SPM units and -18.2% with group quarters excluded.
- The ablations ran on Build P, not the dense file.
- Initial weights miss the administrative targets, so the initial-weight columns diagnose where error enters. They do not describe a usable file.
- We did not verify why Census records less SNAP than administrative totals imply.

## Reproduction

Scripts sit in `scripts/` and write beside themselves under `run/` and `asec/`. `outputs/` holds the tables behind this note, `outputs/logs/` holds the printed output of scripts that write no table, and `receipts.json` pins the inputs, including paths outside this repository.

1. `dl.py`, `asec_extract.py`, `extract_state.py`, `assemble.py`, `cutrun.py`, `assemble_dense.py`, `cols.py`, `idcheck.py`: downloads, state cuts of each file and id checks.
2. `noise.py`, `vardec.py`, `rep_boot.py`, `rep_eval.py`, `noise_ci.py`, `dense_fit.py`, `kish.py`: noise.
3. `initial_fit.py`, `overlap.py`: initial and published weights by channel, sample overlap and persistent state misses.
4. `decomp.py`, `decomp2.py`, `steps.py`, `year.py`, `paired.py`, `thr.py`, `thr_census.py`, `flip.py`, `resource_growth.py`: decomposition.
5. `ablate_eval.py`, `ablate_noise.py`, `fam.py`, `chi_solve.py`, `chi_eval.py`, `dmiss.py`: calibration.
6. `national_cf.py`, `cf_eval.py`, `ctc_ssn.py`, `part.py`: aged incomes, the CTC reform and partnership income.
7. `rent2.py` to `rent5.py`, `rentfix.py`, `dense_rent.py`, `sparse_rent.py`, `runcuts.py`, `rent_eval.py`, `house.py`, `house_sim.py`, `puf_house.py`: rent and housing.
8. `acs_hh.py`, `acs_w.py`, `acs_more.sh`, `acs_one.py`, `acs_eval.py`, `acsrent.py`, `acs_rent_run.py`, `acs_reform_cmp.py`, `fl_reform.py`, `acs_stored.py`: the ACS-local path.
9. `dash_swing.py`: the dashboard's stored audits.
10. `prep_evidence.py`, `receipts.py`: package this folder and write `receipts.json`.
