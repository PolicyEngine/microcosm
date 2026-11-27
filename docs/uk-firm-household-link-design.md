# UK firm-household link: design note

Design only, with no code dependency. Firm columns come from the
`labour_link`, `sic_division` and `consumption` export groups
(`uk_runtime/firm_table.py`). None of the firm-side names below is on `main`
yet: every `firm_*` column and module named here, `ons_employment`, the
public-sector rescale, `--bpe-turnover`, the household-sales checks and
`uk_household_purchases_supply_decomposition` arrive with the later PRs of
the UK firm stack on #1706. The aim is comparison, not record matching;
each side calibrates to its own external totals.

## Employer size in the household surveys

| Source | Variable | Meaning | Codes | In Microcosm today |
|---|---|---|---|---|
| LFS/APS | `MPNR02` | Employees at the workplace, current job | 1: 1-10, 2: 11-19, 3: 20-24, 4: don't know but under 25, 5: 25-49, 6: 50-249, 7: 250-499, 8: don't know but 50-499, 9: 500+; -8/-9 missing | In the pinned APS tab `apsp_a24m25_eul_pwta22.tab`, but not in `UK_APS_OCCUPATION_CLEANING["columns"]` |
| LFS/APS | `SOLOR`, `ONETEN` | Self-employed with or without employees; exact count 1-10 | 1/2; 1-10 | In the file, not read |
| LFS/APS | `PUBLICR` | Private or public sector | 1/2 | In the file, not read |
| FRS | `job.tab` `NUMEMP` | Workplace size, per job | 1-9, blank | `job.tab` is pinned; this column is not read |
| FRS | `job.tab` `OREMP` | Possibly organisation size | 1-9, blank | Not read; meaning unverified |

Source for the `MPNR02` labels: LFS user guide volume 3 (2021), p. 124
(<https://doc.ukdataservice.ac.uk/doc/8928/mrdoc/pdf/lfs_user_guide_vol3_variabledetails_2021od.pdf>).

In the 2024-25 APS:

- nearly every employee (`INECAC05` 1) answers `MPNR02`;
- self-employed people answer it only if they have employees (`SOLOR` 2).

FRS `NUMEMP` uses the same nine codes as `MPNR02`, in a pattern that fits the
LFS scheme. But no labels were found for `NUMEMP`, `OREMP` or `ETYPE`. **Step
zero** is to verify them against the UKDS FRS 2024-25 data dictionary (SN 9563).
If `OREMP` is organisation size, it is closer than any APS variable to the
firm table's enterprise concept.

## Household-side imputation

The FRS observes the band, so this is mostly a read.

1. **Read.** Add a person-level derivation alongside `derive_frs_employment`
   (`frs_employment.py`). It reads `NUMEMP` (and `OREMP`, if verified) from the
   main job: `JOBTYPE` 1, `frs_spine.FRS_MAIN_JOB_JOBTYPE`. That is the job that
   `adult.tab` SIC and `MJOBSECT` describe. People with no employee job get
   `NONE`.
2. **Fill.** Some rows have no usable code: blanks, the don't-know codes 4 and 8,
   and rows the SPI employment shell redrew. Draw these from an APS donor model,
   following the `aps_occupation.py` pattern:
   - read `MPNR02`, `PUBLICR` and `SOLOR` with `read_pinned_columns`;
   - fit `microcosm.fit.CategoricalClassifier`, set up as in `sub_major_model`,
     on age, sex, region, status group, arcsinh pay, SIC section and
     public/private;
   - draw with `inverse_cdf_draw` at `stable_identity_uniforms` keyed on
     `person_source_id`, with the SPI recipients' own key, as
     `impute_soc2020_sub_major_group` does.

   Codes 4 and 8 draw within their range. `RegimeGatedQRF`
   (`etb_vat.impute_etb_vat`) models continuous outcomes, so it does not fit
   a band.
3. **Bridge to enterprise bands.** Workplace size is at most enterprise size, so
   the mismatch runs one way: chains, banks and public bodies put workers in
   small workplaces of large enterprises. The household and BPE band schemes
   share only the 50, 250 and 500 edges. The comparison grid is therefore
   under 50, 50-249, 250-499 and 500+. Finer enterprise bands need a labelled
   assumption: a draw of P(enterprise band | workplace band, section, public),
   restricted to enterprise ≥ workplace, from the ONS UK Business tables of
   local units and enterprises by size band.
4. **Self-employed.** The self-employed are not firm employees.
   - Sole traders (`SOLOR` 1) correspond to the firm table's assumed working
     proprietors (`ons_employment` less `firm_employees`).
   - Employers are compared by legal status, using `ONETEN` and `MPNR02`.
5. **Public sector.** FRS `employment_sector` and `PUBLICR` cover every public
   employer. `firm_is_public_sector` covers only ONS statuses 5 and 6, not
   public corporations. It stays small unless the
   `scale_uk_firm_public_sector_employment` rescale is used, and that rescale
   distorts size bands. Compare the private sector first; treat the public
   sector as one block with no size split.

## Marginals to align

Neither side is calibrated on these today:

- The committed firm contract (`uk_firms_targets.json`) holds firm counts, VAT
  registration and VAT liability.
- The person side holds national PAYE RTI payrolled employees and pay.
- The firm-side sources in the table are value-free entries in
  `firm_target_candidates.py`.

| Marginal | Person side (FRS) | Firm side | Notes |
|---|---|---|---|
| Employee jobs, SIC section × public/private | APS employees by industry; RTI by sector (proposed) | RTI payrolled employees by SIC (candidate); BPE; ONS public sector employment | Main job vs all jobs |
| Jobs by region | RTI NUTS1 (residence) | BPE (enterprise location); BRES (workplace) | Diagnostic only |
| Jobs by size band | `NUMEMP`/`MPNR02` via the bridge | BPE 2025 Tables 1 and 2 (already read by `firm_headcount`) | Coarse grid |
| Paybill by section | FRS employment income, calibrated to RTI pay | ASHE 4.7a mean × employees; RTI paybill candidate | ASHE leans high |
| Share at or below the secondary threshold | FRS pay distribution | ASHE percentile interpolation (`firm_pay_distribution`) | Firm-side tail below p10 is guessed |

The comparison runs after each side's own `calibrate` solve and reports weighted
gaps banded by the disclosure rules. Feeding one side's totals into the other's
`solve_constraint_system` would make it circular.

## Consumption link

The bridge works on totals by SIC division. No data links a household to the
firm it buys from.

1. LCFS COICOP spending, levelled to Consumer Trends, is at purchasers' prices
   on the domestic concept.
2. Spread each COICOP division over products using SUT Table 3, normalised by
   column.
3. Split each product with `uk_household_purchases_supply_decomposition`:
   - domestic basic output goes to the maker's division;
   - distributors' margins go to divisions 45-47;
   - imports and taxes less subsidies leave the firm sector.
4. The result is household-implied sales at basic prices by division, per
   household. The firm side's `firm_household_sales` is not yet comparable: it
   applies purchasers'-price shares to turnover, its trade turnover is gross
   (resold goods included), and its turnover includes excise. A comparable
   firm-side figure would multiply trade turnover by a margin rate (ABS gross
   margin, a new source) and net excise out of divisions 11-12 and 19.
5. Left out on both sides: imputed rent (04.2, `CPA_L68A`) and narcotics
   (02.3). Expected shortfalls on the firm side:
   - the national vs domestic concept gap;
   - the default £50m turnover cap (lifted by the opt-in `--bpe-turnover`);
   - non-market and imputed output.
6. The new check is a per-division ratio. It is reported, not gated, as
   `uk_firm_household_sales_by_coicop` is.

Vintages differ (SUT 2024, LCFS 2024-25, BPE 2025); state, do not reconcile.

## Where it belongs

Use separate PRs, not the #1706 firm stack.

- **Scope.** The #1706 stack is the opt-in, experimental firm table. Most of this work
  changes the household pipeline: a new `uk/source_stages.json` stage, new FRS
  and APS columns, receipts, disclosure banding, and person outputs that
  household release gates see.
- **Review size.** The firm stack is already large. The household stage needs reviewers
  who know the FRS/APS imputation, and step zero blocks it anyway.
- **Coupling.** A new person column changes household content identity and
  forces rebuilds. The firm table can ship without it, and the firm-support plan
  keeps firm releases separate until joint calibration exists.

Order:

1. A household PR: read `NUMEMP`, fill from APS `MPNR02`, build the size bridge.
2. A diagnostics PR: the labour marginals and the consumption bridge.
3. The firm-side basic-price column, with either PR.
