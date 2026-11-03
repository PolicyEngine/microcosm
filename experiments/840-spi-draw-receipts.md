# #840 receipts: the twin-conditional SPI draw

Receipts for the microcosm#840 PR (bridge predictors, the no-record rule and the
Rosenblatt transport of the twin's standing into the SPI stage-1 draw). Every
figure is a weighted aggregate over 10 or more records or a published number;
cells under 10 records are written "<10" (CD137 cl. 8 / CD171 §5.2.1); record
counts stay with the raw receipts beside the builds, and this note carries
shares and weighted totals. Raw
receipts stay licensed-side under `data/build/840-twins/` on the author's
machine.

## R0. Tape receipts (SPI 2022-23 PUT, FACT-weighted)

The bridge predictors are validated against the published Annex A codes (pp.
14-17 of the UKDS documentation). The figures are the stage receipt
(`bridge_validation` on the stage evidence), measured on the full tape with
the stage's preparation (employment income PAY + EPB + TAXTERM, pension
PENSION + SRP, trade max(0, PROFITS − CAPALL − LOSSBF)); an earlier
exploration with the raw columns gave 0.961 overall and a partnership
agreement near zero, which that preparation does not reproduce.

| Check | Weighted share |
|---|---:|
| Leaf-defined `main_source` agrees with MAINSRCE (1 pay, 2 pension, 3/4 trade, 5/6 other) over classified records | 0.969 |
| ... by code: pay / occupational pension / sole trader / partnership / other | 0.983 / 0.987 / 0.851 / 0.744 / 0.916 |
| `PAY > 0` among MAINSRCE 1 | 0.995 |
| `SEINC_NUM = 1` among `PROFITS > 0`; `PROFITS > 0` among `SEINC_NUM = 1` | 1.000; 0.898 |
| `SRP > 0` among `SPA = 1`; `SPA = 1` among `SRP > 0` | 0.942; 0.999 |
| Composite records (-1, no codes) | 0.002 |
| MAIND claimants (dropped as a predictor) | 0.047 |
| INDUSTRY07 blank among `PAY > 0` (industry deferred to increment 2) | 0.078 |

`has_trade` and `at_or_over_spa` take the codes on the tape (the leaves serve
only a composite record), `main_source` and `has_pay` the leaves; the receipt
reports each code against its leaf both ways.

## R1. Spine twins at design (prior) weights

Control: origin/main `d1fda720b`. Treatment: the branch at the same base.
Same inputs (`spine-request.json`), engine policyengine-uk 2.122.2, seed 42.

### R1a. Treatment stage evidence (`hmrc_spi_income_spine`, 13 min, peak RSS 13.2 GiB)

All 36 spine gates pass; the HMRC 208-fact replay stays `reviewed_exclusions_only`
(every fact fenced, as before); the effective-mass columns sit at 0.0119
(`gift_aid`) and 0.0008 (`charitable_investment_gifts`) against the 1e-6 floor;
the TEI + TII identity is exact on every SPI-channel row checked.

| Recipient domain | Share | Weight |
|---|---:|---:|
| Drawn recipients (claimants and partners aged 16+) | 0.86 of the domain's rows | |
| ... sharing a twin with another drawn copy (band-donor stacking) | 0.11 of the drawn | |
| Undrawn, no taxable income (the no-record rule) | 0.14 of the domain's rows | 3.80m |
| Reserved band carriers drawn without taxable income | 0.014 of the drawn | |
| Dependants aged 16 to 19 kept on twin values | 0.03 of the channel's 16+ rows | 0.84m |
| Reference forest training rows (FRS claimants and partners with taxable income) | 0.50 of the FRS base adults | |

Tape validation (the stage receipt, R0): main-source agreement 0.969 overall
(pay 0.983, occupational pension 0.987, sole trader 0.851, partnership 0.744,
other 0.916), `PAY > 0` among MAINSRCE 1 0.995, `SEINC_NUM = 1` among
`PROFITS > 0` 1.000, `SRP > 0` among `SPA = 1` 0.942, `SPA = 1` among `SRP > 0`
0.999, composites 0.002.

Transport receipt (Kolmogorov distance from U[0,1); the quantile over the
twin's positive rows; the sign over all drawn recipients):

| Stage-1 target | Source | FRS regime | KS sign | KS quantile | Positive share of drawn |
|---|---|---|---:|---:|---:|
| `hmrc_spi_pay` | twin | zero-inflated positive | 0.007 | 0.044 | 0.56 |
| `savings_interest_income` | twin | zero-inflated positive | 0.007 | 0.049 | 0.44 |
| `hmrc_spi_state_pension_income` | twin | zero-inflated positive | 0.007 | 0.025 | 0.32 |
| `private_pension_income` | twin | zero-inflated positive | 0.010 | 0.035 | 0.27 |
| `self_employment_income` | twin | zero-inflated positive | 0.008 | 0.076 | 0.09 |
| `dividend_income` | twin | zero-inflated positive | 0.005 | 0.029 | 0.07 |
| `property_income` | twin | zero-inflated positive | 0.010 | 0.043 | 0.05 |
| `hmrc_spi_other_income` | twin | zero-inflated positive | 0.014 | 0.051 | 0.01 |
| `hmrc_spi_other_social_security_income` | twin | zero-inflated positive | 0.006 | 0.113 | 0.01 |
| `hmrc_spi_unemployment_benefit_income` | twin | zero-inflated positive | 0.011 | n/a | <0.001 |
| `hmrc_spi_incapacity_benefit_income` | identity (degenerate analogue) | all zero | 0.006 | n/a | 0 |
| 7 leaves without an FRS analogue | identity | | 0.005-0.010 | 0.004-0.008 | |

By main source the sign uniforms are uniform within the pay (0.54 of the drawn),
pension (0.35) and trade (0.08) groups (KS 0.01-0.03) and skew high in the
"other" group (0.03; KS 0.11-0.25 on dividends, interest and property),
which is the group defined by holding those leaves: a conditional position
restricted to rows selected on the outcome is not uniform, and the overall
uniformity is the receipt. The in-sample atom the D3 ruling anticipated shows
as the 0.03-0.08 quantile distances on the large targets.

### R1b. Control against treatment (design weights, GB and NI, calendar 2025)

Control spine: origin/main `123927679` (with #1167); treatment: the branch on
`d1fda720b` plus the stage change. The arms differ in size only through the
CGT clone arms downstream of the changed incomes.

**Confinement.** On the pre-CGT rows (the same households on both sides) the stage itself leaves every FRS-channel cash and reported column
byte-identical: the FRS person columns that move are the frame-level stages
downstream (`capital_gains` and its CGT companions on a fifth of the FRS
rows, the salary-sacrifice conversion on 1 % of them, where `employment_income` moves with
`pension_contributions_via_salary_sacrifice` and `employee_pension_contributions`,
`student_loan_plan` on <10). FRS household columns move only on the QRF-imputed
consumption and energy columns (the ripple #834 documented) and
`household_weight`, which the CGT mass moves shift within −2.9 % to +1.1 %
(1st to 99th percentile); FRS mass 16.743m to 16.732m, SPI 11.906m to 11.917m.
FRS benefit units move on the residual take-up draws solved over the whole frame
(`would_claim_uc`, `would_claim_pc`, Child Benefit). Everything else that moves
is on SPI-channel rows.

**The award screen** (`max(0, uc_maximum_amount − uc_income_reduction) > 0`, 2025,
benefit units with an adult under 66; FRS rows identical on both arms):

| Family type | FRS | SPI control | SPI treatment |
|---|---:|---:|---:|
| Lone parents | 0.644 | 0.539 | 0.646 |
| Couples with children | 0.287 | 0.252 | 0.297 |
| Singles without children | 0.355 | 0.233 | 0.407 |
| Couples without children | 0.100 | 0.074 | 0.125 |

Over all benefit units the lone-parent shares are 0.639 / 0.538 / 0.645.

**Universal Credit reporters after `uc_reporter_redraw`** (weighted; in
brackets the SPI channel's distinct FRS source households carrying them as a
share of the FRS channel's):

| Family type | FRS | SPI control | SPI treatment |
|---|---:|---:|---:|
| Lone parents | 0.827m | 0.554m (0.43) | 0.750m (0.58) |
| Couples with children | 0.376m | 0.237m (0.51) | 0.380m (0.78) |
| Singles without children | 1.094m | 0.450m (0.33) | 0.965m (0.69) |
| Couples without children | 0.115m | 0.032m (0.24) | 0.119m (0.76) |

**Pay and trade by employment status** (SPI claimants and partners aged 16+,
share with pay > 0; the uk-data#529 defect):

| Status | FRS pay | SPI control pay | SPI treatment pay | FRS trade | SPI control trade | SPI treatment trade |
|---|---:|---:|---:|---:|---:|---:|
| Full-time employed | 0.983 | 0.831 | 0.980 | 0.014 | 0.114 | 0.016 |
| Part-time employed | 0.970 | 0.794 | 0.969 | 0.029 | 0.091 | 0.027 |
| Full-time self-employed | 0.026 | 0.769 | 0.051 | 0.908 | 0.126 | 0.912 |
| Part-time self-employed | 0.049 | 0.743 | 0.056 | 0.885 | 0.074 | 0.866 |
| Retired | 0.000 | 0.281 | 0.017 | 0.000 | 0.061 | 0.006 |
| Unemployed | 0.000 | 0.848 | 0.017 | 0.000 | 0.095 | 0.005 |
| Student | 0.000 | 0.935 | 0.003 | 0.000 | 0.051 | 0.000 |
| Carer | 0.000 | 0.835 | 0.011 | 0.000 | 0.086 | 0.002 |
| Long-term disabled | 0.000 | 0.790 | 0.015 | 0.000 | 0.100 | 0.003 |

The residual 1-2 % of pay among SPI non-workers is the reserved band carriers,
whose leaves come from the band pools.

**Lone-parent adults by FRS adult income quintile** (FRS claimants and
partners, weighted): FRS 0.30 / 0.13 / 0.17 / 0.20 / 0.19; SPI control 0.13 /
0.19 / 0.19 / 0.21 / 0.28; SPI treatment 0.35 / 0.14 / 0.12 / 0.16 / 0.22.

**Composition after the no-record rule** (working-age units, weighted share
with no taxable income): singles FRS 0.243, SPI control 0.038, SPI treatment
0.239; lone parents FRS 0.220, SPI control 0.027, SPI treatment 0.235. The SPI
singles' screen share above the FRS one comes with a larger share of low
earners (pay above zero and under £12,000: 0.166 against 0.085), see R1c.

### R1c. Rank fidelity, the in-sample atom and the cross-fit

The first treatment (reference forest fitted on every FRS adult and scored
in-sample) kept the twin's standing only partly: among employed twins, the
Spearman correlation of copy pay with twin pay was 0.49 overall and 0.40
within gender x region x age-band cells (control: 0.13 and −0.02). The cause
is the in-sample atom the D3 ruling anticipated, measured here on the pre-CGT
FRS base adults with the stage's forest settings: a twin's own
value occupied a median 0.36 (self-employment), 0.15 (interest), 0.74
(dividends), 0.46 (private pension), 0.73 (property) and 0.23 (pay) of its
conditional's mass (90th percentile 0.62-0.99), because the weighted bootstrap
fills a row's own leaves with its copies. The located quantile was then
mostly the random placement inside that atom. Marginal uniformity (R1a) does
not show this, rank fidelity does.

Spearman correlation of the located quantile with the twin's value, over
positive rows, three ways of fitting the reference (in-sample; five folds
keyed on the row; five folds keyed on the predictor cell, so a cell is located
by one forest it never saw):

| Target | Positive share of FRS base adults | Exact cells: in-sample / row folds / cell folds | Overall: in-sample / row folds / cell folds |
|---|---:|---|---|
| `self_employment_income` | 0.09 | 0.97 / 0.88 / 0.98 | 0.48 / 0.58 / 0.70 |
| `savings_interest_income` | 0.46 | 0.98 / 0.91 / 0.99 | 0.69 / 0.72 / 0.85 |
| `dividend_income` | 0.07 | 0.40 / 0.60 / 0.72 | 0.37 / 0.63 / 0.78 |
| `private_pension_income` | 0.30 | 0.55 / 0.68 / 0.80 | 0.46 / 0.63 / 0.78 |
| `property_income` | 0.04 | 0.35 / 0.60 / 0.84 | 0.43 / 0.64 / 0.83 |
| `hmrc_spi_pay` | 0.54 | 0.66 / 0.73 / 0.83 | 0.51 / 0.61 / 0.73 |

The stage therefore cross-fits the reference in five folds keyed on the
predictor cell (`reference_model.folds`, `fold_key`); a copy shares its twin's
cell, so the forest that locates it never saw the twin. The residual
within-cell shuffle is the chain: a leaf is located given the twin's observed
earlier leaves, which differ between rows of a cell. The rebuilt treatment's
receipts follow in R1d.

**The imported pay marginal.** With the twin's rank carried, the level is the
tape's: among earners the tape holds 30 % under £12,570 and 56 % under
£25,000 against 12 % and 35 % on the FRS adults, the annual tax-record
concept against the FRS's annualised current pay. On the first treatment the
copies of employed twins sat at 21 % under £12,570 and 24 % at £12,570-25,000
(twins 10 % and 21 %), which is why the SPI singles' screen share sits above
the FRS one.

### R1d. The rebuilt treatment (cross-fitted reference, 16 min, peak RSS 9.3 GiB)

Same base, inputs, engine and seed as R1a; the reference cross-fitted in five
folds keyed on the predictor cell (each fold holds a fifth of the training
rows and scores a fifth of the recipients, no in-sample fold). All 36 spine gates pass,
the replay stays `reviewed_exclusions_only`, the effective-mass columns and
the TEI + TII identity hold as in R1a. The recipient domain is unchanged
(0.86 of the domain's rows drawn, 0.14 undrawn, 0.014 of the drawn being
carriers without taxable income).

Held-out location moves the quantile uniformity receipt from the in-sample
values (R1a) to its honest level: KS quantile 0.085 (pay), 0.070 (interest),
0.077 (state pension), 0.080 (private pension), 0.082 (property), 0.108
(dividends), 0.123 (self-employment); the sign uniforms stay at 0.006-0.012.

**Rank fidelity of the draw** (Spearman of the copy's leaf with the twin's,
both positive; overall / within gender x region x age-band cells / within
gender x region x whole-year-age cells):

| Leaf | Control | First treatment (in-sample) | Rebuilt (cross-fitted) |
|---|---|---|---|
| Pay | 0.13 / −0.02 / −0.02 | 0.49 / 0.40 / 0.44 | 0.59 / 0.52 / 0.50 |
| Self-employment | 0.16 / −0.25 / −0.11 | 0.51 / 0.47 / 0.78 | 0.72 / 0.64 / 0.78 |
| Private pension | 0.07 / 0.03 / 0.03 | 0.49 / 0.47 / 0.51 | 0.66 / 0.63 / 0.62 |
| Interest | 0.09 / 0.00 / −0.01 | 0.45 / 0.39 / 0.41 | 0.55 / 0.46 / 0.43 |
| Dividends | 0.16 / 0.03 / −0.17 | 0.32 / 0.33 / 0.47 | 0.53 / 0.51 / 0.56 |
| Property | −0.08 / −0.52 / −0.93 | 0.39 / 0.27 / 0.05 | 0.73 / 0.68 / 0.27 |

The draw's fidelity sits below the located quantile's (R1c) because the tape
forest's own conditional varies between rows of a cell and the chain locates
each leaf given the twin's observed earlier leaves.

**The award screen** (benefit units with an adult under 66; all units in
brackets): lone parents FRS 0.644 (0.639), SPI control 0.539 (0.538), SPI
rebuilt 0.653 (0.653); couples with children 0.287 / 0.252 / 0.292; singles
without children 0.355 / 0.233 / 0.324 over all units; couples without
children 0.100 / 0.074 / 0.131 over all units.

**Universal Credit reporters** (weighted; in brackets the SPI channel's
distinct FRS source households as a share of the FRS channel's): lone parents
FRS 0.827m, SPI control 0.554m (0.43), SPI rebuilt 0.773m (0.59); couples
with children 0.376m / 0.237m (0.51) / 0.403m (0.83); singles without
children 1.094m / 0.450m (0.33) / 0.938m (0.68); couples without children
0.115m / 0.032m (0.24) / 0.115m (0.81).

**Pay and trade by employment status** on the rebuilt SPI rows match the FRS
rows as in R1b (full-time employed pay 0.979 against 0.983; full-time
self-employed trade 0.908 against 0.908; retired, unemployed, students,
carers and the long-term disabled at 0.3-1.7 % pay, the band carriers).
Lone-parent adults by FRS income quintile: SPI rebuilt 0.36 / 0.15 / 0.13 /
0.16 / 0.20 against FRS 0.30 / 0.13 / 0.17 / 0.20 / 0.19.

**Confinement and the signed-difference draft.** On the pre-CGT households
(same ids on both arms) the FRS-channel rows move only on:
`household_weight` through the CGT mass moves (−3.8 % to +1.5 %, 1st to 99th
percentile; FRS mass 16.743m to 16.747m, SPI 11.906m to 11.902m); the
QRF-imputed consumption and energy columns (`housing_water_and_electricity_consumption`,
`domestic_energy_consumption`, `electricity_consumption`, `gas_consumption`,
`transport_consumption`, `petrol_spending`, `diesel_spending`,
`full_rate_vat_expenditure_rate`); the CGT lane (`capital_gains`,
`cgt_residential_probability`, `capital_gains_asset_type`, `capital_gains_badr`,
`cgt_support_copies`); the salary-sacrifice conversion (1 % of FRS persons, where
`employment_income`, `employee_pension_contributions`,
`pension_contributions_via_salary_sacrifice` and
`salary_sacrifice_pre_conversion_pay` move together); the residual take-up
draws (`would_claim_child_benefit`, `child_benefit_opts_out` and
`would_claim_pc`, under 0.2 % of FRS units each); `student_loan_plan` (<10). No FRS cash or
reported column moves otherwise. The SPI-channel rows move on every stage-1
output and its HMRC aggregates, the stage-2 reports and pension contributions,
the disability flags, `receives_benefits_in_own_right`, `would_claim_uc`, the
recorded capitals, the housing shell, the WAS, LCFS, ETB and NTS draws and
the CGT lane: 67 person, 8 benefit-unit and 58 household columns, the
inventory for the signed-difference entries (kept outside the repository
until signed).

The plan's stage receipts and where they sit: payload confinement (R1b,
R1d, R1e); recipient domain (R1a, R1d); transport receipt and Kolmogorov
distances by main source (R1a, R1d); screened share by channel and family
type (R1b, R1d, R1e); pay and trade by employment status (R1b, R1d); lone
parents' standing in the FRS income quintiles (R1b, R1d); lone-parent UC
reporters, weighted and by distinct source household (R1b, R1d); the HMRC
band replay, the distributional-mass gate and the spine gates (R1a, R1d).

### R1e. Downstream moves on the rebuilt treatment (control against R1d)

_Rows matched by id, source id, channel and CGT-lane structure (every
household, person and benefit unit whose keys agree on both arms); every
statistic a weighted aggregate at the control
household weights over a channel's rows; the paired comparisons weight
copy and twin alike with the copy's weight. Scripts
`downstream_columns_840_v2.py`, `engine_aggregates_840.py`,
`paired_reports_840.py` and `paired_screen_840.py` (session scratchpad), outputs
beside the twins._

**FRS-channel rows.** The FRS rows move only through frame-level stages after
stage 17, by five mechanisms, none on a cash or reported column except the
salary-sacrifice conversion:

| Mechanism | Columns (FRS rows moved) | Net change of the FRS channel's total |
|---|---|---|
| The CGT lane re-ranks gains over the frame as the SPI rows' investment incomes and wealth draws rise | `capital_gains` (a fifth of FRS persons), `capital_gains_badr` (0.5 %), `capital_gains_residential_property` (1 %), `cgt_residential_probability` (2 %), `cgt_support_copies` (under 0.1 % of households) | gains −21.9 % (£33.5bn to £26.3bn at own weights over the full tables; SPI rows £86.2bn to £92.6bn; all rows −0.7 %); engine capital gains tax −21.8 % on FRS rows, +7.2 % on SPI rows, −0.5 % in all |
| The energy and fuel rakes re-target the frame | `electricity_consumption` (0.89 of FRS households), `gas_consumption` (0.86), `domestic_energy_consumption` (0.97), `housing_water_and_electricity_consumption` (0.97), `petrol_spending` (0.54), `diesel_spending` (0.24), `transport_consumption` (0.73), `full_rate_vat_expenditure_rate` (0.12) | electricity −1.5 %, gas −1.2 %, domestic energy −1.4 %, housing-water-electricity −0.4 % (row moves of about ±3.5 % in both directions); petrol and diesel +1.9 % and transport +0.4 % by one factor on every row; the VAT rate 0.0 % net; engine fuel duty +1.9 %, VAT −0.05 % |
| The salary-sacrifice conversion rate (the stage's headcount target counts the SPI rows' predicted sacrifice) | `pension_contributions_via_salary_sacrifice` (2.4 % of FRS persons); `employment_income`, `employee_pension_contributions`, `salary_sacrifice_pre_conversion_pay` (0.9 % each) | converted persons 2.80m to 3.05m; pay −0.12 % (£558.4bn to £557.8bn), employee contributions −3.7 %, sacrificed contributions +3.0 %; engine income tax −0.03 %, NI −0.11 % |
| The CGT mass moves between the arms | `household_weight` (0.98 of FRS households) | FRS mass 16.860m to 16.867m; 97.3 % of the mass within ±1 %, 98.5 % within ±4 %, 1.5 % beyond |
| The take-up residual draws | `would_claim_child_benefit`, `child_benefit_opts_out`, `would_claim_pc` (under 0.2 % of FRS units each); `student_loan_plan` (<10 persons) | −0.13 %, −2.2 %, +0.09 %; engine child benefit −0.5 %, pension credit +0.2 %, Universal Credit +0.0006 % |

The FRS channel's engine net income moves −0.6 %, the gains; income tax, NI,
VAT and the benefits move under 0.2 % except fuel duty and capital gains tax
above.

**SPI-channel rows, the income leaves** (the adults in the stage domain
against their twins, copy weights; the twin is the composition
yardstick, the tape the magnitude yardstick):

| Leaf | Share positive: twin / copy control / copy rebuilt | Mean per adult: twin / copy control / copy rebuilt |
|---|---|---|
| Pay | 0.583 / 0.742 / 0.587 | £24.3k / £27.6k / £25.3k |
| Self-employment | 0.090 / 0.099 / 0.089 | £2.75k / £2.72k / £3.12k |
| Private pension | 0.133 / 0.174 / 0.170 | £1.84k / £2.28k / £2.56k |
| Savings interest | 0.452 / 0.373 / 0.375 | £351 / £384 / £519 |
| Dividends | 0.054 / 0.095 / 0.090 | £128 / £1,949 / £2,153 |
| Property | 0.036 / 0.049 / 0.046 | £291 / £572 / £628 |
| State pension (reported) | 0.115 / 0.112 / 0.114 | £1,274 / £1,291 / £1,304 |

Who earns now follows the twin (0.587 against 0.583; control 0.742). The
positive shares of the other leaves are the tape's class probabilities given
the bridge (the transport preserves the rank; the gate's zero mass is the
tape's), so savings interest stays under the twins' share and dividends and
property above it; the magnitudes are the tape's conditional on the bridge,
which lifts the investment leaves 10-50 % over the control (pensioner and
investment-main-source cells now draw from their own tape conditionals
instead of the cell marginal). Among earners the pay bands are the tape's:
22.5 % under £12,570, 22.7 % at £12,570-25,000, 32.3 % at £25,000-50,000,
15.4 % at £50,000-100,000 and 7.1 % above, against the twins' 10.4 / 21.0 /
43.6 / 19.9 / 5.1 % (control copies 22.9 / 25.4 / 34.1 / 12.9 / 4.7 %): the
low tail is the annual tax-record concept (R1c), unchanged from the control;
the top tail is heavier than the control's. Over all SPI rows pay −6.9 %,
self-employment +12 %, private pension +15 %, interest +37 %, dividends
+19 %, property +25 %; the HMRC aggregates follow
(`hmrc_spi_total_investment_income` +20 %, `hmrc_spi_total_earned_income`
−3.5 %, `hmrc_spi_assessable_income` −1.3 %).

**SPI-channel rows, reports and flags** (stage 2 and the SPI-row stages after
it; copies against twins, share reporting or flagged):

| Column | Twin | Copy control | Copy rebuilt | Lone parents twin / rebuilt | Singles twin / rebuilt |
|---|---|---|---|---|---|
| `universal_credit_reported` | 0.107 | 0.061 | 0.108 | 0.404 / 0.372 | 0.100 / 0.099 |
| `pip_dl_reported` | 0.066 | 0.039 | 0.075 | 0.071 / 0.074 | 0.097 / 0.099 |
| `pip_m_reported` | 0.052 | 0.031 | 0.058 | 0.054 / 0.054 | 0.076 / 0.075 |
| `carers_allowance_reported` | 0.021 | 0.015 | 0.026 | 0.063 / 0.072 | 0.012 / 0.018 |
| `housing_benefit_reported` | 0.023 | 0.010 | 0.019 | 0.008 / 0.009 | 0.046 / 0.035 |
| `esa_contrib_reported` | 0.009 | 0.004 | 0.010 | 0.007 / 0.007 | 0.008 / 0.009 |
| `pension_credit_reported` | 0.011 | 0.006 | 0.006 | 0.000 / 0.000 | 0.024 / 0.014 |
| `council_tax_benefit_reported` | 0.065 | 0.053 | 0.056 | 0.133 / 0.105 | 0.104 / 0.086 |
| `child_benefit_reported` | 0.138 | 0.138 | 0.136 | 0.637 / 0.396 | 0.000 / 0.057 |
| `is_disabled_for_benefits` | 0.096 | 0.063 | 0.100 | 0.095 / 0.093 | 0.148 / 0.141 |
| `receives_benefits_in_own_right` | 0.130 | 0.064 | 0.115 | 0.409 / 0.376 | 0.137 / 0.105 |
| `would_claim_uc` (benefit units) | 0.480 | 0.489 | 0.496 | 0.755 / 0.737 | 0.415 / 0.431 |

The reports land on the twins' shares where the control sat 40-60 % under
them. Two residuals are stage 2's, not the draw's: on the drawn copies
(0.86 of the domain's rows, 82 % of its weight) it imputes disability and carer
reports above the twins' (PIP 0.035 against 0.025, carers 0.012 against
0.006, as on the control's copies), and it gives child benefit to 5.7 % of
childless singles' copies (8.0 % on the control) and to 0.40 of lone
parents' copies against 0.64 of their twins. The undrawn copies (0.14 of the
rows, 18 % of the weight; almost all by the rule, the rest drawn copies that
happen to equal their twins) carry their twins' reports except where
`spi_benefit_coherence` and the take-up stage rewrite them (UC 0.322 against
0.351, `receives_benefits_in_own_right` 0.348 against 0.459, council tax
benefit 0.082 against 0.123), the exemption the plan's step 6 holds for
review.

**SPI-channel households, the draws after stage 17** (per SPI household,
control to rebuilt; FRS households in brackets). The WAS wealth draws follow
the investment incomes up: net financial wealth £125k to £154k (£101k), gross
financial wealth £131k to £159k (£106k), savings £26.1k to £32.4k (£22.3k),
corporate wealth £39.4k to £53.5k (£33.2k), stocks-and-shares ISA £17.6k to
£21.9k (£14.9k), other residential property £40.3k to £48.7k (£27.0k);
property wealth £253k to £258k (£222k) and private pension wealth £178k to
£184k (£172k) barely; consumer debt £4.82k to £4.35k (£3.72k). The housing
shell: renters 0.318 to 0.362 (0.331) with rent £3.04k to £3.36k (£2.96k),
mortgage interest £2.02k to £1.94k (£1.35k), council tax −0.5 %. The LCFS
consumption draws move 0-5 % toward the FRS level (clothing £1.19k to £1.13k,
communication £882 to £847, transport £7.25k to £7.04k, food £3.85k to
£3.76k; FRS £1.04k, £766, £6.09k, £3.71k); energy +2 % (electricity £1.03k
to £1.05k; £994); vehicles 1.18 to 1.12 (1.09); the NTS bus draws +4-7 %.
Recorded capital (`uc_reported_capital`) £78.0k to £83.3k per unit (£44.1k).
The SPI rows' CGT-lane weights move more than the FRS rows' (80 % of the mass
within ±1 %, 10 % beyond ±4 %).

**Engine aggregates by channel** (calendar 2025, each arm's own weights, the
full tables):

| Variable | FRS control → rebuilt | SPI control → rebuilt |
|---|---|---|
| Employment income | £594.8bn → £594.2bn | £657.0bn → £610.7bn |
| Self-employment income | £66.2bn | £72.7bn → £82.2bn |
| Savings interest | £12.5bn | £11.7bn → £15.4bn |
| Dividends | £5.3bn | £60.2bn → £65.7bn |
| Property income | £10.0bn | £15.7bn → £18.5bn |
| Income tax | £148.9bn → £148.8bn | £205.0bn → £222.5bn |
| National insurance | £27.5bn → £27.4bn | £26.8bn → £24.8bn |
| Capital gains tax | £7.4bn → £5.8bn | £20.7bn → £22.2bn |
| Universal Credit | £36.18bn → £36.18bn | £18.4bn → £36.5bn |
| PIP | £14.2bn | £6.7bn → £11.9bn |
| Carer's allowance | £2.18bn | £1.28bn → £2.18bn |
| Housing benefit | £5.96bn | £0.74bn → £1.11bn |
| Pension credit | £4.16bn → £4.17bn | £0.42bn → £0.51bn |
| Child benefit | £7.07bn → £7.04bn | £6.35bn → £6.32bn |
| Household net income | £946.3bn → £940.7bn | £840.9bn → £835.8bn |

SPI income tax rises 8.6 % while SPI pay falls 7 %: the pay mass now sits on
the 0.44 of adults who earn (control 0.55) at the tape's main-source-pay
magnitudes, and the investment leaves rise; income tax per unit on the copies
is 1.49 times the twins' (control 1.38). The design-weight levels are not the
calibrated ones; the direction is the tape's.

**Universal Credit, copies against twin units** (engine; copy weights;
working-age units in brackets):

| Family | Screened: twin / copy control / copy rebuilt | Recipients: twin / control / rebuilt | Mean award per unit: twin / control / rebuilt |
|---|---|---|---|
| Lone parents | 0.649 / 0.547 / 0.666 | 0.582 / 0.449 / 0.584 | £9,135 / £5,458 / £9,357 |
| Couples with children | 0.254 / 0.262 / 0.302 | 0.181 / 0.175 / 0.223 | £2,422 / £1,600 / £2,891 |
| Singles without children (working-age in brackets) | 0.283 / 0.188 / 0.331 (0.356 / 0.236 / 0.415) | 0.186 / 0.124 / 0.214 (0.233 / 0.156 / 0.269) | £1,591 / £673 / £1,690 (£1,996 / £846 / £2,120) |
| Couples without children | 0.098 / 0.078 / 0.135 | 0.062 / 0.045 / 0.091 | £604 / £308 / £736 |
| All | 0.289 / 0.223 / 0.331 | 0.208 / 0.156 / 0.237 | £2,403 / £1,301 / £2,585 |

Lone parents land on their twins. The childless units and the couples with
children pass the screen 17-37 % more often than their twins and receive
15-48 % more often: the imported low-pay tail (22.5 % of earning copies under
£12,570 against 10.4 % of their twins), where a copy of a low or mid earner,
or of the second earner of a couple, falls below the taper more often than
its twin. Over all units the copies receive 14 % more often and 8 % more per
unit than the twins (the control: 25 % less often, 46 % less). By channel at
design weights the SPI channel's UC recipients per unit are now 1.00 (lone
parents), 1.07 (couples with children), 1.58 (singles) and 1.35 (couples
without children) times the FRS channel's (control 0.77, 0.83, 0.92, 0.67);
the childless cells of the national UC surface and the award bands are the
sibling watch for the full build (R2).

**Reading.** No column moves against its mechanism's direction. Three
magnitudes to hold in view: the childless units' screen above their twins
(the tape's annual pay concept, exposed now that who earns follows the twin);
the investment leaves and the wealth draws 10-50 % above the control on SPI
rows, with the CGT lane moving a fifth of the FRS channel's gains to SPI rows
at design weights; and the stage-2 disability and carer over-imputation on
drawn copies, a stage-2 residual the no-record rule narrows at the domain
level.

### R1f. Twins on the #1200 base (13 October)

Control: #1200's branch `uk-data-ports-1095-3` at `a50ffba5e` (it carries
#1145 and #1198). Treatment: this branch rebased on that commit, with the
three integration items the rebase exposed (the recipient flag the landlord
stage reads, the finance-costs leaf on identity uniforms, the gainful
self-employment refresh on the root's whole rule). Same inputs, engine
(policyengine-uk 2.123.0) and seed; 13 and 27 minutes (the second beside a
fixture regeneration). All 37 spine gates pass on both arms, including
`uk_stage_property_components` and `uk_stage_spi_benefit_coherence`. Rows
matched as in R1e. The receipts of R1b to R1e
reproduce within a point or two; the differences below are the base's.

**The award screen** (all units; FRS / SPI control / SPI rebuilt): lone
parents 0.634 / 0.532 / 0.636; couples with children 0.274 / 0.238 / 0.281;
singles without children 0.200 / 0.184 / 0.322; couples without children
0.093 / 0.073 / 0.123. Lone-parent UC reporters 0.827m FRS, SPI 0.584m to
0.779m, the SPI channel's distinct source households rising from 0.45 to
0.60 of the FRS channel's; couples with children 0.376m, SPI 0.229m to 0.408m
(0.49 to 0.85); singles 1.094m, SPI 0.434m to 1.012m (0.33 to 0.73). Full-time
employees with pay 0.979 against 0.983 on FRS rows; full-time self-employed
with trade 0.908 against 0.908.

**Copies against their twins** (share, twin / copy control / copy rebuilt):
earners 0.583 / 0.743 / 0.587; UC reported 0.107 / 0.062 / 0.111 (lone
parents 0.404 / 0.277 / 0.375; singles 0.100 / 0.045 / 0.106); PIP daily
living 0.066 / 0.040 / 0.075; carer's allowance 0.021 / 0.016 / 0.026;
housing benefit 0.023 / 0.010 / 0.019. Undrawn copies 0.14 of the domain's rows,
18.3 % of its weight. Engine, per unit (twin / copy control / copy rebuilt):
lone parents screened 0.646 / 0.540 / 0.648, receiving 0.581 / 0.454 /
0.575, award £9,110 / £5,412 / £9,159; couples with children screened 0.241
/ 0.248 / 0.291, receiving 0.178 / 0.172 / 0.221; singles screened 0.280 /
0.187 / 0.329, receiving 0.184 / 0.116 / 0.220; couples without children
screened 0.090 / 0.076 / 0.128, receiving 0.057 / 0.047 / 0.085; all units
screened 0.283 / 0.219 / 0.325, receiving 0.206 / 0.152 / 0.238, award
£2,385 / £1,263 / £2,577; income tax per unit £8,746 / £11,990 / £12,960.
By channel the SPI channel's UC recipients per unit are 0.99 (lone
parents), 1.08 (couples with children), 1.64 (singles) and 1.34 (couples
without children) times the FRS channel's (control 0.79, 0.83, 0.87, 0.74).

**Engine aggregates by channel** (own weights; control to rebuilt): SPI
Universal Credit £17.9bn to £36.4bn (FRS £35.9bn), PIP £6.7bn to £11.9bn
(FRS £14.2bn), carer's allowance £1.28bn to £2.18bn (FRS £2.18bn), housing
benefit £0.77bn to £1.19bn (FRS £5.95bn), pension credit £0.39bn to
£0.47bn; SPI income tax £204.4bn to £221.8bn (+8.5 %), national insurance
−7.4 %, employment income −7.0 %, self-employment +13 %, savings interest
+32 %, dividends +9 %, property income +18 %; FRS income tax −0.03 %,
national insurance −0.10 %, VAT −0.04 %, fuel duty +1.8 %, child benefit
−0.5 %, pension credit +0.7 %; capital gains tax FRS £7.24bn to £6.00bn
(−17 %), SPI £20.75bn to £21.61bn (+4 %); household net income FRS −0.5 %,
SPI −0.8 %.

**FRS-channel rows** move as in R1e, plus one mechanism the merged base
adds: `property_rental_income` (4 % of FRS persons, −6.2 % of the FRS
channel's total), because `property_components` ranks both channels'
landlords by profit against the PRIS bands, so the SPI rows' redrawn
property incomes move the FRS landlords' places. The rest: the CGT lane
(`capital_gains` on a fifth of FRS persons, −15.9 %; `capital_gains_badr` −25 %;
`capital_gains_residential_property` +18 %; `cgt_residential_probability`
−5 %), the energy and fuel rakes (electricity −1.5 %, gas −1.3 %, domestic
energy −1.4 %, petrol +1.8 %, diesel +1.9 %, transport +0.4 %), the
salary-sacrifice conversion (0.9 % of FRS persons; converted 2.78m to 3.03m; pay
−0.10 %, employee contributions −3.2 %, sacrificed +4.5 %), the CGT mass
moves on `household_weight` (97.4 % of the FRS mass within ±1 %, 1.4 %
beyond ±4 %), the take-up residual draws (`would_claim_pc`,
`would_claim_child_benefit`, `child_benefit_opts_out`, under 0.3 % of FRS
units each) and, new on this base, the LCFS consumption columns on under
0.2 % of FRS households at under 0.1 % of their totals. The SPI rows' leaves and the draws after
stage 17 move as in R1e (pay −6.9 %, self-employment +12 %, private pension
+16 %, interest +37 %, dividends +19 %, property +26 %; net financial
wealth £125k to £154k per SPI household against £101k on FRS households).

## R2. Full local build

_One build at the latest run's configuration on the merged head (plan of
2026-10-12). R1f was measured on #1200's head (`a50ffba5e`); the branch now
sits on `main` after #1260._
