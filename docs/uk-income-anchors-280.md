# UK income targets at the calibration year (PolicyEngine/chronicle#280 lane)

How the UK national calibration binds the HMRC income facts at its period, why
the 2023-24 SPI rows move before they bind, and what the calendar-year window
means. Rulings by María on 2026-09-22; evidence in the 2026-09-21 assessment
behind PolicyEngine/chronicle#280.

## The finding

The v20 national candidate (September 2026) fitted every SPI component band
and still measured income tax 13 percent under the OBR line, the state-pension
bands 8 to 18 percent over and the self-employment £20-40k bands over. All 129
`hmrc_spi` references were the published 2023-24 values bound at period 2025
as identity holds: a value moves only when its reference declares an
`uprating_index` with a UK applier, and the family declared none. The engine
fixes every pensioner's amount at the period's rate, so the state-pension
family showed the seam exactly: summed over the taxpayer bands the calibrated
spine sat at 1.005 of the 2023-24 target times the new State Pension rate ratio.
Employment income sat at 0.906 of its uprated target, private pensions at 0.918,
dividends at 0.924: roughly £120bn of 2025 taxable income the weights could not
supply.

## The period convention

The calibration binds at calendar 2025. In the engine the OBR indices are
calendar-year series keyed at 1 January, so period 2025 is calendar 2025 for
incomes; tax-year policy amounts (rates, allowances, the State Pension) carry
the 2025-26 value for the whole period (verified on the v20 file: state pension
at 2025 over the 2024 report is 1.041 for every recipient). Facts published by
April-to-March years therefore bind as the months to the end of the calendar
year: three twelfths of the year opening in Y-1 and nine twelfths of the year
opening in Y. That is the `calendar_year_window` value operation
(`ledger_targets.CALENDAR_YEAR_WINDOW_WEIGHTS`): it resolves under
`latest_not_after`, takes both years from one series identity, refuses a window
with either year absent, and writes each member's year, weight, value and
assertion on the spec.

Some publishers give each year its own measure: DWP's benefit expenditure and
caseload tables carry `expenditure_2024` and `expenditure_2025`, which the
series identity reads as two series of one year each. A window over such a
table declares `source_measure_id_by_opening_year` in its selector, mapping each
of the two years to its measure. A fact matches only the measure its own year
declares, and the two declared measures form the window's one series. The map
names exactly the two years of the window, and a selector that declares it
cannot also declare `source_measure_id` (microcosm#1095: the Great Britain
pension-age Housing Benefit spending row).

Facts already at a calendar-year period (the DWP monthly cubes) keep their
calendar-2025 windows; the ESA rows join them below.

## Uprating the SPI 2023-24 rows

Every `hmrc_spi` reference declares an `uprating_index`; the appliers live in
`uk_runtime/hmrc_uprating.py` and are registered in `UK_UPRATING_APPLIERS`, so
the generator and the runtime apply the same factor and the committed
membership carries the uprated value with its receipt.

- Amount rows: employment, self-employment, private pension and property
  income follow `hmrc.itl_2026.total_income_growth_by_total_income_band`,
  HMRC's projected total income in the band, the index their count rows'
  growth belongs with (microcosm#1063; property since microcosm#1106). The
  others take `policyengine_uk_parameter:<path>`, the ratio of the pinned
  engine's parameter between 1 January of the fact's opening year and 1
  January of the calibration year, the instants the SPI donor rebasing already
  uses: dividends `obr.per_capita.gdp`. Savings interest takes two legs
  (microcosm#1095): `ons.household_interest_income`, the index of its
  history, to the dataset's year, then `obr.per_capita.gdp`, the index the
  engine's dataset projection applies, to the calibration year (1.1420 x
  1.0474 = 1.1961 on the pinned engine values; 1.0438 and 1.1920 before
  policyengine-uk 2.155.0 took population from the OBR's March 2026
  determinants). The state pension follows
  `gov.dwp.state_pension.new_state_pension.amount`, the rate the engine pays
  (203.85 to 230.25 a week, 1.1295). On policyengine-uk 2.98.0 and 2.100.0 the
  2023 to 2025 factors are 1.1067, 1.0344, 1.0826, 1.0877 and, for household
  interest alone, 1.1819. The compile
  reads the values from the vendored `hmrc_uprating_engine_pins.json` (the six
  parameters at every 1 January 2019 to 2026, written from the installed engine
  by `tools/pin_uk_uprating_engine_values.py`, whose `--check` refuses drift), so
  it runs and reproduces without policyengine-uk; a `requires_uk` test holds the
  pins in lockstep with the installed engine, and an engine bump re-runs the tool.

  The SPI donors reach the frame's year apart from these indices for interest
  and other income (microcosm#1418 savings follow-up). The income stage first
  scales each donor's two columns to HMRC's published 2023-24 Table 3.7 amounts
  in its band of total income (`spi_income.rebase_spi_donor_savings`, the rows
  vendored as `uk/hmrc_spi_table_3_7_savings_rows.json`), then grows them on the
  OBR's savings income path the Table 3.7 targets take
  (`hmrc_uprating.uk_obr_savings_income_level`). Each band's factor is the
  published amount over the tape's FACT-weighted one; a band the tape carries
  none of is left as it is. From the frame's year on, the engine projects both
  columns at GDP per head.
- Count rows: `hmrc.itl_2026.taxpayer_count_growth_by_total_income_band`, HMRC's
  own projected growth in Income Tax payers for the Table 2.5 band that contains
  the SPI band (Income Tax liabilities statistics, July 2026), as the
  calendar-2025 window of the 2024-25 and 2025-26 projections over the 2023-24
  outturn, read from the vendored `hmrc_itl_taxpayer_counts.json` so a resource
  that lags the feed pin refuses. Two SPI bands share each of the 30-50k,
  50-100k and 200-500k Table 2.5 bands; 1m+ takes 1m-2m and 2m+.

The spec metadata records the index, the parameter or resource, the instants
or years, both values, the factor and the pre-alignment value; the membership
lists the same under `uprating_holds`.

## The calibration-year anchors

HMRC's Table 2.5 binds three targets (the `hmrc_itl` family) at the
calendar-2025 window: Income Tax payers (`income_tax > 0`), total income and
Income Tax liabilities by eleven total-income bands, all sliced on the engine's
`total_income`. They
are the anchors the component bands lack: every component can fit while tax
is short, and a level shift of the whole family is invisible to the family
itself. The window of the two projections sums to £322.7bn of liabilities and
39.5m taxpayers; OBR's FY2025-26 receipts line (£331.4bn) stays bound beside
it, 2.7 percent apart, the difference being the fiscal-year basis of the OBR
row.

The SPI savings-interest rows (Table 3.7, taxpayers' interest from banks and
building societies, ISA interest excluded) bind uprated by the household
interest index. The ONS UKEA HAXV row is households' D.41 interest resources in
the national accounts, a concept no cash-interest variable reaches; it stays on
the measure-exclusion register as a measured diagnostic with that reason
(microcosm#866).

## ESA on the payment-type cube

The three ESA rows bind DWP's Stat-Xplore ESA caseload by payment type
(chronicle#282) as the mean of the four quarterly points inside calendar 2025
(February, May, August, November), summed over payment types:

- `dwp.esa_claimants`: contributions based, both, income based; the published
  total's credits-only claimants (78k to 84k through 2025) have their National
  Insurance record credited without payment and are left out rather than
  described as unpublished;
- `dwp.esa_contrib_claimants`: contributions based plus both, matching the
  model's `esa_contrib > 0`, which includes dual claimants;
- `dwp.esa_income_claimants`: income based plus both, matching `esa_income > 0`.

2025 is the year of the ESA income-related managed migration to Universal
Credit (income based 499k in February to 30k in November), so a single point
would misstate the year; the calendar mean is what the FRS 2024-25 base can be
asked to represent.

## The OBR rows at the window

The 27 OBR receipts and expenditure rows that resolved by latest-not-after to
FY2025-26 now bind at the calendar-2025 window too (three twelfths of the
FY2024-25 outturn, nine twelfths of the FY2025-26 forecast, from the March
2026 EFO series), the same rule as the HMRC liabilities anchors. Income tax
moves from £331.4bn to £325.1bn, the window of the £305.9bn outturn and the
£331.4bn forecast, and sits 0.7 percent above the HMRC liabilities window
(£322.7bn): the two anchors now share a basis. The Universal Credit total
takes the window of the sum of its two welfare-cap cells (`calendar_year_window`
over a two-concept selector: one series per declared concept, both years of
each), so the total and its two components bind on one basis. One OBR row keeps
a fiscal-year declaration: the cars share of fuel duty is pinned to the
FY2024-25 outturn, the only year the vehicle split is published for, and the
contract row says so in a `period_basis` field the generator writes onto the
compiled spec's metadata rather than in prose alone.

## The region tier

The SPI Table 3.11 rows (income and tax by region and country, all taxpayers)
bind on the twelve-area region tier (microcosm#905) as the `hmrc_spi_region`
family: Income Tax payers, total income and Income Tax liabilities by the ten
regional total-income bands, one reference per area scoped by the household-
region predicate the generator stamps, sliced on the engine's `total_income`
by the row's own band edges. The publisher's regional bands stop at 200,000
and over. Each row moves to 2025 by HMRC's projected growth for the same
measure in the Table 2.5 band(s) the regional band spans (Income Tax
liabilities statistics, July 2026): payers by `taxpayer_count_growth`, total
income by `total_income_growth`, liabilities by `total_tax_growth`, all read
from the vendored `hmrc_itl_taxpayer_counts.json`, which now carries the
three Table 2.5 measures. Two SPI bands share the 30-50k and 50-100k Table
2.5 bands; the 200k-and-over row takes the window over 200k-500k, 500k-1m,
1m-2m and 2m-and-over together. Ruled by María on 2026-09-22: the regions bind
uprated, not as diagnostics. Table 2.2 publishes taxpayer counts by region only
and is not bound.

## Reserved income rows on the spine

The spine stage `spi_income_band_donors` (between `spi_support_channel` and
`hmrc_spi_income_spine`) reserves 120 SPI-channel households for each HMRC
Table 2.5 total-income band from £200,000: 200k-500k, 500k-1m, 1m-2m and 2m
and over. Each is a whole FRS household copied at clone index 2, flagged
`household_is_spi_income_band_donor` with its band in
`spi_income_band_donor_lower_bound`, with one carrier adult
(`person_is_spi_income_band_carrier`) drawn without replacement by the SPI
2022-23 tape's FACT-weighted propensity for the band given region, sex and
age band. The copy starts at the build tax year's published band taxpayers
over 120 (359k/120 = 2,992 for 200k-500k on the 2024-25 projection, 61k/120
= 508, 20k/120 = 167 and 10k/120 = 83), mass added and receipted as the CGT
band donors' is. The income stage's `resample_band_donor_leaves` operation
then gives every carrier a band-conditional draw: one tape record whose
total income *as realised on the frame* (its stage-1 leaves at the stage's
uprating factors plus the published remainder of TEI + TII held nominal) lies
in the band, FACT-weighted with replacement from the carrier's region where
that regional pool holds at least 20 records, otherwise nationally, all
eighteen stage-1 leaves copied together and uprated as the forest draws are.
The bands and the reserved weights are the calibration year's (Table 2.5
counts for the build tax year), so membership is asserted on the uprated
amount: the operation refuses any carrier whose realised total lands outside
its band and receipts `carriers_outside_band` per band (zero by construction).
The propensity that seats carriers by region, sex and age reads the tape's
published bands; the pools that give them their amounts read the uprated ones.
Composite records stay in the pools as published. Non-carrier adults in donor
households keep the ordinary forest draw; stage 2 refills their FRS-only
inputs as for every synthetic row. The stage-health gate
`uk_stage_spi_income_band_donors_support` checks that every band carries its
donors at positive band-exact weight with one carrier each. The methodology
memo and the upper-tail investigation are summarised in microcosm#1006 and
the aggregate receipts of the v21c run are under `docs/evidence/uk-income-280/`:
before this stage the spine, like the enhanced FRS, carried no record above
£1.51m of total income.

Two uprating conventions sit side by side by design. The national SPI amount
rows move by the engine's own component indices (earnings, mixed income, GDP
per head, the private-pension index, the new State Pension rate, and for
savings interest household interest to the dataset's year and GDP per head
after it), because each row is one income component and the engine uprates
that component by that index: the target and the model then agree on what
"2025 employment income" means. The regional Table 3.11 rows are totals and
tax by band with no component split, so no engine index applies; they move by
HMRC's own projected per-band growth in taxpayers, income and tax (Table 2.5),
the publisher's forecast of the very quantity the row measures. Counts move by
the HMRC growth on both tiers. Mixing an engine index into the regional rows,
or a publisher growth rate into a single component, would bind a quantity
neither source defines.

## Limitations from the licensed runs

Three findings from local licensed builds of this branch, none of which the
PR's CI can see because it never builds the licensed spine (recorded on
microcosm#1006, comments 5812328880 and 5814299594):

- **The spine build is blocked by the student-loan realisation gate.** A
  licensed spine build from this branch stops at the `transferred` phase with
  `uk_stage_student_loans_realization: PLAN_5 realization_deviation 1.1027
  exceeds 1.0`. The PLAN_5 top-up is a few rows (817 weighted persons expected
  from 136,538 eligible on spine-u, realised as two rows, deviation 0.30), so
  the check swings with which rows the draw lands on; main's spine passed at
  0.96. The stage is unchanged here, but its eligible pool and weights depend
  on everything upstream, the band donors included. The limit stays at 1.0 in
  this tree; the fix belongs in the gate (a tolerance for a top-up of a couple
  of rows) and needs a ruling. Assessment builds raised it to 2.0 locally only.
- **The SPI copies, band donors included, inherited their FRS parent's wealth,
  spending and VAT** while this lane was open: on spine-u all 45 WAS, LCFS and
  ETB columns on the SPI rows equalled the parent's, and the band donors' median
  gross financial wealth was flat across the bands (£48k, £67k, £50k and £44k).
  The next section runs the SPI block before those stages (microcosm#1012).
- **The national calibration at this head blocks on the CGT projection
  fence.** With the spine rebuilt from the rebased head, the terminal gate
  `uk_cgt_projection_entrants` (introduced by #979) refuses: 146,920 weighted
  sub-exempt gainers cross the frozen annual exempt amount by 2030 under the
  engine's uprating, against the bound of 73,000. The measure and the spine are
  the same as #979's own build, which passed at 68,534; at prior weights both
  spines fail (108k), and about 101k of it sits on roughly fifty of #970's CGT
  band-donor rows whose gains sit just under £3,000. Nothing binds the
  sub-exempt gainer count, so their weight floats with the objective: #979's
  target set pulled it down (0.62×), this lane's 1,062-target set pulls it up.
  The v21c measurements in this doc and on the evaluation page are the
  pre-rebase build (f6d33a26), before the gate existed; this head cannot produce
  a certified cut until the fence is settled on the #970/#979 side (bind the
  count, spread the band-donor mass over lighter rows, or cap their weight
  ratio).
  **2026-09-29 (microcosm#1045):** settled by retiring the band-donor stage.
  The Table 3 redraw owns every gain amount and places rows by wealth rank,
  so the demoted donors were an artefact of drawing support by income and
  placing it by wealth; #1045 replaces the stack with `cgt_support_split`, a
  deterministic mass-conserving split of the wealthiest households of each
  Table 3 income column into light copies before the incidence clone, and
  the sub-exempt population becomes the anchor-trimmed Advani–Summers clone
  composition (its 2030 crossing is 6,000 to 7,500 on the existing arms).

## SPI rows before the donor imputations

The SPI block (`frs_hmrc_spine_leaves`, `spi_support_channel`,
`spi_income_band_donors`, `hmrc_spi_income_spine`) runs right after
`frs_brma`, ahead of `was_wealth`, `regional_property_uprating`,
`nts_bus_travel`, `lcfs_consumption`, `etb_vat` and `etb_services`. Those six
stages condition on income, and the support stages copy whole FRS households.
In the enhanced FRS order, which the spine used to follow, each SPI copy kept
the wealth, spending, VAT rate, public-service use and bus travel drawn for its
FRS parent's income, while its adults' incomes were replaced by SPI draws. A
£2m band donor then carried the financial wealth of the median FRS household it
was copied from.

With the SPI block first, every row, the FRS base rows included, is imputed
from its final incomes. The base rows' incomes are their own reports: the SPI
stage-1 forest no longer redraws their dividends (uk-data#498,
microcosm#1095).

Three details follow from the order:

- `uc_reporter_redraw` stays after the six stages. Its award screen runs the
  engine, and for benefit units without an FRS capital answer, UC capital falls
  back to the household's WAS `savings`, property and `corporate_wealth`. The
  graph opens a reader-isolation version before it, as it does for
  `uc_capital_coherence`.
- `regional_property_uprating` takes each region's factor over FRS-base owners
  only, the population the factor was defined on, and applies it to every
  owner, the SPI rows included.
- The six stages now see the SPI support mass allocation, so their
  weight-dependent steps run at the frame's prior (importance) weights: the
  gas-connection walk, the NEED rake and DESNZ level, the ETB NHS
  normalisation and the NTS receipts.

## The SPI households' housing

The support copies and band donors are whole FRS households, so after the
reorder their wealth and spending followed their SPI incomes while their
housing was still the parent's. On the reordered spine the SPI half's detached
share sat at 27-29 % in every income band (the FRS half rises from 19 % to 47 %
by £200k-1m), social renting stayed at 11-16 % above £50k and 16 % above £1m,
and 402k of the SPI channel's 506k prior-weighted Housing Benefit households
were owner-occupiers.

`spi_housing_shell` runs right after the SPI income chain and imputes the
housing of every SPI-channel household, the household counterpart of the SPI
stage-2 FRS-only fill. It trains on the FRS base households and conditions on
region, household type, adults, children, single-adult status, the reference
person's age and the household's income components; no engine run, so housing
benefits do not become circular. Tenure, dwelling type, bedrooms and council
tax band are drawn in that order by weighted multiclass classifiers with an
inverse-CDF draw; council tax, rent, both mortgage repayments, insurance,
service charges, water, Northern Ireland rates, subletting, Housing Benefit and
council tax benefit by one chained regime-gated QRF conditioned on those
categories. The two benefit targets are trained on the head benefit unit's
Housing Benefit and the reference person's council tax benefit. Every draw
reads uniforms keyed on household id. Structural zeros are learned from the
training households (an amount is zero wherever the FRS has no positive value
for that tenure or region); mortgage payments need a mortgaged tenure, Housing
Benefit a rented one, and council tax benefit is capped at council tax. Both
drawn benefits land on the household reference person, with every other
person's zeroed, and the stage checks it. Region, single-adult status, BRMA and
composition stay the recipient's; FRS rows are unchanged.

Imputation was chosen over whole-record donor matching on a holdout of FRS
households (`tools/validate_uk_housing_imputation.py`). Within FRS support the
two, and a forest-weighted whole-record draw, sit near the sampling noise; on
the top 5 % of incomes held out and filled from the rest, which is the
extrapolation the SPI copies need, imputation came closest on ownership (0.818
against 0.855 observed; matching 0.787), detached share (0.396 against 0.446;
matching 0.328) and four or more bedrooms (0.456 against 0.557; matching
0.355), with the most distinct bundles. All three under-draw council tax band
F and above at the top (about 0.19 against 0.35).

## The SPI draw follows the twin (microcosm#840)

Until microcosm#840 the stage-1 tape forest conditioned on age, gender and
region alone and drew each recipient's quantile from the model's random
stream, so a copied lone parent on a low wage drew the pay of a random
taxpayer of her age, gender and region, and a copied adult out of work drew
pay at all. The SPI copies then passed the Universal Credit award screen at
about half the rate of the FRS rows they were copied from, and their
employment status disagreed with their drawn incomes (uk-data#529).

The stage now does two things with the twin's own leaves. First, four bridge
predictors join age, gender and region: `main_source` (the largest of pay,
pension and trade) and `has_pay`, which the tape and the twin derive by the
same formula from their own leaves, and `has_trade` and `at_or_over_spa`,
which the tape takes from its published indicator codes (SEINC_NUM, the
self-employment pages filed, and SPA; the leaf formulas serve only a
composite record without codes) and the twin from its trade leaf or
self-employed status and its age against State Pension age. The published
MAINSRCE code validates the leaf-defined main source (0.969 of the classified
weight on the full 2022-23 tape with the stage's preparation: pay 0.983,
occupational pension 0.987, sole trader 0.851, partnership 0.744, other 0.916;
the floor is 0.95) and the receipt reports each indicator code against its
leaf both ways. An employee therefore draws from earners and a pensioner from
pensioners.

Second, the quantile is the twin's. Reference forests fitted on the FRS base
claimants and partners, with the same predictors and chain order and the FRS
analogue of every tape leaf as their targets, locate each twin's observed
leaves on the FRS conditional (`conditional_uniforms`, a Rosenblatt
transform: the sign uniform is the twin's position in the gate's cumulative
class probabilities, the quantile its position on the sign's magnitude forest,
each leaf chained on the twin's observed earlier leaves), and the tape forest
draws at those uniforms. The copy receives the SPI income of someone in the
same position among SPI people like it as the twin holds among FRS people
like it: the FRS relation between composition and income rank is kept, the
SPI levels and tail are imported. Leaves without an FRS analogue, and the
placements inside a class or a flat stretch of a conditional, take uniforms
keyed on the recipient's identity, so no model random stream is consumed and
a draw depends on the twin and the identity alone. The reference forests are
cross-fitted in five folds keyed on the predictor cell (gender, region,
whole-year age and the four bridge predictors): a forest's conditional at a
row it was trained on holds that row's own value as an atom, and under the
weighted bootstrap a twin's own value took a median 0.23 (pay) to 0.74
(dividends) of its conditional's mass on the 2024-25 spine, so an in-sample
location was mostly the random placement inside that atom; with a copy
sharing its twin's cell, the forest that locates it never saw the twin, and
every row of a cell is located by one forest, so the located quantiles stay
monotone in the twin's value within the cell. A full build refuses a
reference the cells cannot cross-fit (only a sampled rung or a synthetic
fixture may locate in-sample, and its receipt says so), a recipient whose
fold differs from its twin's, and a reference whose sign classes differ from
the tape forest's on any located target. One difference between the two
forests remains: the reference locates a leaf given the twin's observed
earlier analogue leaves, while the tape forest draws it given every earlier
output, the three identity-uniform leaves (`other_investment_income`,
`gift_aid`, `charitable_investment_gifts`) included, so a located quantile is
read off a conditional those identity draws also shape; pay, which follows
them in the chain, has the weakest rank fidelity, and moving the analogue
targets ahead of the identity leaves is the follow-up to measure. Because the
located quantile is deterministic given the twin, copies of one twin that the
band-donor stage stacks now share every stage-1 draw, not only the
identity-keyed ones. The manifest declares the bridge, the analogue map and
the reference model (seed 45, the stage seed plus three; five folds), and the
stage evidence carries the per-target transport receipt: the analogue, the
FRS regime and sign classes, the uniform source and the Kolmogorov distance
of the uniforms from U[0,1), overall and by main source. The counts in the
stage evidence stay with the build; the published receipts carry shares.

A twin with no taxable income (no pay, trade or pension, and at most £1,000 of
taxable interest, dividends, property and miscellaneous income together) has
no income tax record, so the tape holds no population to condition on; its
copy keeps the twin's values at stage 1 and stage 2, and the recipient-domain
receipt counts the rows and weight so kept. Reserved band carriers are always
drawn, since their leaves come from the band pools. A twin whose only record
is the State Pension counts as having one and is drawn against a tape that
mostly holds taxpayers, which can lift its pension leaves. FRS rows are
byte-unchanged.

**After the rebase onto #1145 and #1200.** The restricted finance costs
(`property_finance_costs`, the last stage-1 target) reach the FRS rows only
at `property_components`, after this stage, so the twin has no analogue and
the leaf takes identity-keyed uniforms. The stage now flags the rows it drew
(`person_is_spi_income_recipient`: the claimants and partners aged 16 and over
with taxable income, and the reserved band carriers), and `property_components`
reads the flag to tell tape-concept rows from FRS-concept ones: an undrawn copy
keeps its twin's property income on the FRS concept, with the sublet split and
the imputed finance costs its twin gets, instead of being read as a tape draw.
The UC gainful self-employment refresh on drawn rows takes the root's whole
rule, the self-employed main job as well as the earnings route, since
`has_trade` now conditions the draw; undrawn copies keep their twin's flag.
The stage evidence counts the rows each route flags.

## The gas connection

`lcfs_consumption` imposes each region's published gas-connected share (DESNZ
gas meters over electricity meters) on the drawn gas before the NEED rake.
It used to disconnect the households with the smallest drawn gas first. Once
the SPI support households drew gas from their own, higher incomes, that walk
piled the disconnections onto low-draw FRS flats: Scottish flats' connected
share fell from 76 % to 71 % while terraced and semi-detached houses rose. NEED
publishes Scotland's gas means by property type and for all dwellings; on the
tilted mix the property-type means implied a Scottish level 2.6 % above the
all-dwellings mean, and the rake could not close it. At 300 sweeps the income
margins converge to zero and Scotland's property-type cells all settle 2.54 %
(reorder alone) or 2.56 % (with the housing stage) below target, against the
gate's 2.5 %. The housing stage is not the cause.

The walk now takes gas-positive households in an identity-keyed uniform order
(seed 0, salt `lcfs_consumption:gas_disconnection`) with the same
weight-fitting skip, so every group of households loses the same expected
share of its drawn connected mass and only the level of the drawn connection
moves to the published share. Its converged residual is 1.05 % (Scotland
gas by property type), with the income margins at zero.

The recipient rake also runs 200 sweeps instead of 50. With the SPI
households' energy drawn from their own incomes, the income margins start
further from NEED's gradient, and at 50 sweeps they were still falling
(electricity 3.14 %, gas 2.76 %). The per-sweep receipt shows both under
2.5 % from sweep 59 and flat by 200 (1.16 % and 1.36 % at 100, 0.91 % and
1.12 % at 200, 0.82 % and 1.05 % at 300), so the gate reads a converged
residual rather than a truncated one, as its governance note asks. The 2.5 %
tolerance is unchanged. The gate now checks the convergence instead of
asserting it: each fuel's residual may move by at most 0.1 point over the last
10 sweeps (the final head moves 0.014 points; the 50-sweep receipts on main and
after the reorder were still falling 0.11 to 0.75 points).

## Not done here

- The property-income amount rows are bound since microcosm#1106: the 13 SPI
  Table 3.7 net amounts bind unscaled on `property_income`, landlords' profit
  after allowable expenses and before residential finance costs, uprated by
  the band index the count rows' growth belongs with
  (`hmrc.itl_2026.total_income_growth_by_total_income_band`, ruled
  2026-10-11), and the
  incumbent's x1.9 scaling is retired (María's ruling of 2026-10-05). On the
  FRS side the `property_components` stage draws each FRS landlord's
  restricted finance costs from the SPI tape and adds them back to the rent
  from other property they report (ROYYR1, after the expenses on show card
  K6, which include mortgage payments and loan interest), so the FRS concept
  reaches the SPI's; the ruling with Max on that add-back is pending. The
  mortgage capital the card also nets stays netted, and sub-letting rent
  stays as reported, which SUBALLOW records as before or after expenses.
- The ESA rows' migration residual and the SPI support channel's benefit fill
  (microcosm#840, #867, #869) are spine items.
