# microcosm#968: BHC poverty on the calibrated UK national line against HBAI

Evidence for [PolicyEngine/microcosm#968](https://github.com/PolicyEngine/microcosm/issues/968), filed with PR #967. The memo below is the diagnosis as posted on the issue on 2026-09-21; the two sections after it record what the review added. Every number comes from the scripts in `scripts/` run on the files listed in `receipts.json` (digests, engine policyengine-uk 2.99.1, feed c5e5bf8); their JSON outputs are in `outputs/`.


Diagnosis memo, 2026-09-21. One engine throughout (policyengine-uk 2.99.1, the uk-data venv), the #731 reproduction conventions (person-weighted share of people in flagged households). Receipts: `scripts/` and `outputs/` in this directory, digests and sources in `receipts.json`. DWP sources: HBAI FYE 2024 summary tables (`HBAI_summary_results.ods`, tables 1.2b, 1.3a–1.6a), the HBAI table pack (chart 2.4 income bands, tables 2.1db, 3.5db, 3.6db, 4.5db, 4.6db), the HBAI quality and methodology report, and the FRS 2023/24 methodology tables M.6a, M.6b and M.8.

## The answer in four lines

- The scorecard's poverty row compares two different concepts. The engine's `in_poverty_bhc` is the **absolute** BHC flag (60 % of the 2010/11 median uprated by CPI: £381/week in 2024, £403/week in 2026). HBAI's "~17 %" is the **relative** BHC rate. HBAI's absolute BHC rate for FYE 2024 is 15 %, its relative rate 17 % (17.3 % from the published bands), its median £650/week.
- Under HBAI's own concept (relative, 60 % of the person-weighted median, computed within the file), the v20 national line measures 15.3 % at 2024 and 14.6 % at 2026. The gap to HBAI is about 2 points, not 8.
- Under the absolute concept the 15 → 9.3 gap is mostly an income-level effect, not a poverty effect: every engine file's median sits 15–20 % above HBAI's (£745 at spine-s design weights, £751 v20, £780 eFRS, all at 2024), so a fixed £381 line bites at 51 % of the engine's median where DWP's bites at 57 %. At 57 % of its own median v20 gives 13.3 % against HBAI's 15 %, the same ~2-point gap.
- The eFRS is not a clean reference: on the relative concept it measures 19.7 % (2.4 points above HBAI) with an effective sample of 1,166 households and 0.58 % of people in non-positive-income households.

## 1. Definition audit against HBAI

What the engine does versus DWP, checked line by line (`hbai_household_net_income.py`, `household_equivalisation_bhc.py`, `in_poverty_bhc.py`, `in_relative_poverty_bhc.py`, `poverty_threshold_bhc.py`):

- Income concept: the engine's HBAI adds and subtracts match DWP's list (earnings, self-employment, investments, pensions, maintenance, all benefits and tax credits, benefits in kind; minus income tax, NI, council tax, pension contributions, maintenance paid, student loan repayments). No divergence found. Small engine-only additions (tax-free childcare, healthy start vouchers) are under £1bn.
- Equivalisation: the modified OECD scale with identical weights (0.67 / 0.33 / 0.20 / 0.33). The engine's adult threshold is 18 where HBAI's dependent child runs to 19 in full-time education, but both sides give that person 0.33, so no divergence.
- Unit: both count individuals in households. No divergence.
- Absolute line: the engine's CPI-uprated 2010/11 line (£372/week at 2023, £381 at 2024) is within 1–3 % of DWP's (60 % of the 2010/11 median restated in 2023/24 prices is £370/week). No material divergence.
- **Relative line: the engine's `in_relative_poverty_bhc` uses a household-weighted median.** HBAI's median is over individuals. On v20 at 2024 the household-weighted flag gives 13.4 % against 15.3 % person-weighted, a 2-point understatement. Its label also says "AHC". Neither the probe nor the scorecard uses this flag today, but anything that does is biased low. (There is also no `poverty_threshold_ahc` variable, only `poverty_line_ahc`; cosmetic.)
- **Concept: the probe reads the absolute flag and the issue text calls it relative.** This is the bulk of the "8.7 vs 17" gap.
- Year basis: HBAI FYE 2024 is April 2023 to March 2024; the files are on a 2024 basis and the engine's medians grow about 3.6 % a year (v20: £751 → £778 → £800). At most about 3 points of the median gap is timing.

Where the engine's median level comes from is not a poverty question and is left open here, with two markers: the ONS household-finances basis puts the FYE 2024 equivalised median at £706/week (ONS "Average household income, UK: FYE 2024"), 9 % above HBAI on definitions alone, and the reported-benefit swap in section 3 barely moves the median (£745 → £741 at design weights), so the level difference lives in market incomes, taxes and uprating, not in simulated benefits.

## 2. The right numbers, one engine

Relative BHC, 60 % of the person-weighted median, share of individuals:

- HBAI FYE 2024: 17 % published, 17.3 % from the £10 income bands.
- spine-s at design weights, 2024: 16.2 %.
- v20 national, calibrated, 2024: 15.3 %. 2025: 15.3 %. 2026: 14.6 %.
- eFRS 1.57.3, 2024: 19.7 %. 2026: 19.3 %.
- X50 55k local draw, 2026: 13.8 %.
- June populace_uk_2023, 2026: 20.0 %.

Absolute BHC (engine line), share of individuals:

- HBAI FYE 2024: 15 %.
- spine-s design 2024: 10.4 %. v20 2024: 9.3 %. v20 2026: 8.7 %. eFRS 2024: 11.2 %. June 2026: 15.1 %.

Medians (person-weighted, £/week): HBAI £650; spine-s design £745; v20 £751 (2024), £800 (2026); eFRS £780 (2024).

AHC, for completeness (HBAI relative 21 %, absolute 18 %): v20 2024 relative 19.5 %, absolute 13.1 %; eFRS 23.7 % and 14.6 %.

The June file's 16.7 % "match" was two errors cancelling: an absolute reading against a relative benchmark, on a file whose relative rate is 20 %.

## 3. Decomposing the microcosm layer (same records, same engine)

spine-s and the v20 H5 are the same 52,846 households in the same order; only `household_weight` differs. At 2024:

- Design weights, design line: 16.16 %. Calibrated weights, design line: 14.96 % (reweighting moves 1.20 points of mass out of the below-line region). Calibrated weights, calibrated line: 15.30 % (the median moves £745 → £751, adding 0.34 back). **Net calibration effect: −0.86 points.** At 2026 the same split gives 15.59 → 14.41 → 14.63, net −0.96.
- Absolute flag: 10.36 → 9.31 (−1.05).

Basis swap (replace the engine's simulated means-tested benefits with the FRS-reported amounts on the same records; `measure_968_basis_swap.py`):

- v20 calibrated weights: 15.30 → 15.94 % (+0.64). Universal Credit alone +0.29, child benefit +0.31, the rest under 0.1 each. Absolute flag 9.31 → 10.31.
- spine-s design weights: 16.16 → 16.21 %. At design weights the simulated-versus-reported basis is not the story.
- eFRS: 19.68 → 20.20 %.

So the FRS under-reporting that the DWP itself documents (FRS 2023/24 captures 67 % of UC families, 74 % of Housing Benefit, 71 % of Pension Credit, 61 % of ESA, table M.6a) costs the calibrated line about 0.6 points of measured poverty and nothing at design weights.

Take-up allocation (`measure_968_takeup_probe.py`, v20 at 2024, calibrated weights). **Caveat (adversarial review on PR #967, 2026-09-22): the probe holds the number of sampled benefit-unit records fixed, not the weighted caseload.** With unequal calibrated weights the arms represent different numbers of UC families (`outputs/takeup_probe.json`): baseline 6.37m, the three re-draws 6.57–6.66m, poorest-first 6.28m, richest-first 6.95m, so the poorest-to-richest arms differ by 0.67m families (10.7 %). The spans below are therefore a joint allocation-and-caseload sensitivity, not an allocation-only effect at a fixed caseload; a weighted-caseload-preserving rerun is the controlled version and has not been run.

- Baseline draw: 15.30 % (6.37m UC families, £76.8bn).
- Three alternative random draws of the same residual count: 15.40–15.59 %.
- Residual given to the poorest entitled units: 14.05 % (6.28m families). To the richest: 16.66 % (6.95m families). **The two arms span 2.6 points, with the represented caseload moving by 10.7 % between them: a joint allocation-and-caseload sensitivity, not an allocation-only bound.**
- Reported claimants only, no residual: 16.12 % (4.67m families, £63.6bn). Everyone entitled claims: 14.85 % (8.61m).
- Child relative rate across the same runs: 17.6 % to 21.0 %.

Reading: the residual gap to HBAI (about 2 points at 2024) is entirely inside the span of "who gets the drawn Universal Credit". The anchored-residual draw assigns the non-reporting claimants uniformly over entitled units; real non-take-up is not uniform (it concentrates in small entitlements), and HBAI's non-reporters are not uniform either.

## 4. Bottom-decile composition and shape

- Bottom of the distribution against HBAI's published bands (share of individuals below x % of the median, BHC): HBAI 6.2 / 11.0 / 17.3 / 24.5 / 33.1 at 40 / 50 / 60 / 70 / 80 %. v20 2024: 4.9 / 8.9 / 15.3 / 23.8 / 32.8. spine-s design: 5.1 / 9.8 / 16.2 / 24.0 / 32.7. eFRS: 7.1 / 11.8 / 19.7 / 27.1 / 34.8. v20's deficit is concentrated below half the median (deep poverty), where it is 1.3–2.1 points short; from 70 % of the median up it matches HBAI.
- The zero-income spike: HBAI has 0.66m individuals (1.0 %) at £0–10/week. spine-s design 0.34 %, v20 0.16 %, eFRS 0.62 %. Half of v20's deep-tail deficit is this spike; the engine's imputation fills incomes that the FRS leaves at zero.
- Quantile ratios to the median: HBAI p10 0.48, p30 0.765, p70 1.31, p90 1.96, mean/median 1.22. v20 calibrated: 0.52, 0.765, 1.27, 1.75, 1.13. spine-s design: 0.51, 0.77, 1.29, 1.85, 1.14. eFRS: 0.46, 0.73, 1.32, 1.87, 1.13. The engine files are compressed at both ends; calibration compresses the top further (p90 ratio 1.85 → 1.75). That is the scorecard's open Gini item, not a poverty item, but it shares a cause candidate (weights on the SPI-synthetic and top households).
- Income sources by quintile (share of gross, state support): HBAI 47 / 36 / 22 / 10 / 3. v20 2024: 41 / 31 / 20 / 13 / 4.5. The bottom two quintiles carry less state support and more market income than HBAI's; the top two carry more. Not a benefits-too-high-at-the-bottom signature.
- Groups, relative BHC, v20 2024 versus HBAI FYE 2024: children 18.8 vs 23 (the largest group gap, but not on one definition: the probes count `age < 18`, while HBAI's dependent child is every under-16 plus 16–19-year-olds in full-time non-advanced education or training who are unmarried and living with their parents, so dependent 18–19-year-olds are missing from the file's child population and independent 16–17-year-olds are in it; the difference is not read as model error here, and a rerun on HBAI's definition is the fix, adversarial review on PR #967); pensioners 17.2 vs 19; working-age 13.6 vs 15. Children by family size: 1 child 9.9 vs 15, 2 children 15.3 vs 18, 3+ 32.3 vs 35. Tenure (household basis): social renters 20–22 vs 29, private renters 15.8 vs 22, owned outright 15.0 vs 18, mortgagors 12.5 vs 8. v20 has too few poor renters and too many poor mortgagors; the eFRS has the opposite renter error (social 31–44 %). Family typing here is household-level while HBAI's is benefit-unit-level, so single-adult categories are not comparable and are omitted.

## 5. Attribution

Against HBAI FYE 2024, relative BHC 17.3 %, v20 at 2024 measures 15.3 %:

- Concept and level (the family-wide layer as the issue framed it): the "8.7 vs 17" gap is an absolute reading against a relative benchmark on files whose income level is 15–20 % above HBAI's. Not a poverty defect; a scorecard-basis error plus the open income-level question.
- Family-wide residual at design weights: about 1.1 points (16.2 vs 17.3), of which roughly half is the missing zero-income spike and the rest the year basis and the compressed bottom tail.
- Calibration: −0.9 points (reweighting at a fixed line −1.2, line shift +0.3).
- Simulated-versus-reported benefit basis: −0.6 points on the calibrated weights, ~0 at design weights.
- Take-up allocation: ±1.3 points around the draw across the poorest-first and richest-first arms, which also move the represented caseload by 10.7 % between them (see the caveat in §3); the span brackets the whole residual but is not a controlled allocation-only bound.

## 6. Fix path as offered (rulings recorded below)

- A. Scorecard and probe (definition fix, do first): make `scorecard_731.py` report the pair HBAI publishes, absolute BHC (benchmark 15 %) and relative BHC on the person-weighted median (17 %), plus the AHC pair (18 % / 21 %) and the median (£650). Re-state the #731 poverty row on that basis; the row then reads v20 14.6 % relative and 8.7 % absolute at 2026, against 17 % and 15 % at FYE 2024 with the caveat that HBAI's latest actual is FYE 2024 and the Resolution Foundation's outlook has typical incomes flat and child poverty rising to 2029/30.
- B. Engine (policyengine-uk#1865): `in_relative_poverty_bhc` should take the median over individuals (person-weighted), and its label should say BHC; consider a `poverty_threshold_ahc` twin of the BHC variable. Not a microcosm blocker.
- C. Microcosm, the substantive ~2-point signal: (i) the UC residual draw's allocation is the biggest single lever; a non-uniform residual (entitlement-size or income-ranked) is a modelling choice worth a spec, with the caseload bands unchanged; (ii) children at 18.8 vs 23 and the tenure pattern (too few poor renters) are the two rows to chase, starting from what the region × tenure and family-composition targets do to renter weights in the bottom two deciles; (iii) the zero-income spike is an imputation-policy question (whether households the FRS records at zero should be simulated into benefit receipt).
- D. Doctrine: do not put an HBAI poverty target on the surface. HBAI is a survey-reported-income statistic on a source that misses a third of UC families, and the DWP's own methodology report warns the bottom decile should not be read as living standards. Keep poverty a watched free outcome, but watch the right pair, with a T4-style diagnostic anchor (relative and absolute BHC/AHC, the median, the p10/p50 and p90/p50 ratios) cited to HBAI FYE 2024 tables 1.2b and 1.3a.
- Verdict on "the benchmark basis differs": partly. The concept and level differences are basis; the residual 2 points and the child and tenure pattern are real signals, bounded by the take-up allocation's span, and are the follow-up.

## 7. After the review (2026-09-21)

- **Direction of a realistic take-up allocation** (review point 3; `outputs/takeup_direction.json`). Ranking the drawn Universal Credit residual by entitlement size, largest first, which is DWP's finding that non-take-up concentrates in small entitlements, gives 14.9 % relative BHC at 2024 against the file's 15.3 % (children 18.7 % against 18.8 %; UC spend £84.3bn against £76.8bn at the same number of drawn records). Smallest first gives 15.9 %. So the correction the evidence favours moves measured poverty away from HBAI, not towards it, and the residual gap should be read as at least what the memo states. Same caveat: this arm represents 6.57m UC families against the baseline draw's 6.37m (+3.2 %, `outputs/takeup_direction.json`), so the 0.4-point move is allocation and caseload together, not allocation at a fixed claimant count.
- **Year basis** (review point 4). The scorecard probe now carries an `hbai_basis` block measured at 2024, the closest engine year to HBAI FYE 2024, with the same pair plus the p10 and p90 ratios to the median and the under-£10 share; the evaluation reports show that block in its own table beside HBAI FYE 2024 and keep the 2026 rows file against file only.
- **Position taken (2026-09-21).** The scorecard is made accurate on concept and year, and the evidence is filed here. Poverty is not chased as a benchmark: HBAI is a survey statistic from another producer, not an administrative total, and a poverty rate is not a calibration target. The income-level and tenure work the decomposition points at is already under way on its own merits and is not tied to moving a poverty rate; the residual 2-point relative gap and the children and tenure rows stay on record here rather than in a follow-up issue.

## Receipts

- `receipts.json`: the five files with sha256 digests and byte sizes, the engine, the feed pin, the DWP and ONS sources, and what each output holds.
- `scripts/`: `measure_968_poverty_audit.py`, `measure_968_basis_swap.py`, `measure_968_tail_profile.py`, `measure_968_takeup_probe.py`, `measure_968_takeup_direction.py`, and `scorecard_731.py` (the #731 scorecard probe with the HBAI pair and the 2024 block).
- `outputs/`: one JSON per script, plus the scorecard outputs for the national candidate, the enhanced FRS and the three K=20 local files.
