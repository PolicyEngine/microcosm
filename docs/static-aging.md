# Static aging

`microcosm.calibrate.static_aging` projects a base-year frame to later years.
It is the cross-sectional special case of the charter's longitudinal rule,
built now because the country models extend the single-year file themselves
and get it wrong (policyengine-us #9526 and #9527 record the double count of
population growth that led here, and microcosm#333 carries the decision this
step needs before it can merge).

## The split

Two things change between years, and the step keeps them apart.

**Weights carry who exists.** For each projection year the weight entity's
weights are recalibrated to that year's projected population by demographic
cell: age and sex from SSA's Trustees Report projection for the US, ONS for
the UK. Nothing else is targeted. A program count or an income total is an
output of rules, take-up and demographics; a weight vector forced to match
one can no longer be wrong about it, so it can no longer be checked. Filers
by income bin has the same problem in income form.

**Factors carry how much.** A column that follows a national total (an IRS
SOI series, a CBO projection) gets the factor that makes its weighted total
under the year's weights grow, from the base year, by exactly the total's
projected growth, so the factor absorbs only what demographics did not
explain. The series' level is never imposed: a column may follow a series
for a broader concept than its own, as every dollar input on the default AGI
series does. A column that follows a per-person rate or a price index (CMS
per-capita spending, CPI-U) gets the index ratio. Unmapped columns carry
over unchanged.

**Program counts are predictions.** `score_predictions` compares the
projected frame's counts and totals with an external projection and reports
the ratio. Nothing feeds back.

## Anchoring

By default each cell's target is the frame's own base-year weighted count
times the projection's growth for that cell (`anchor="frame"`), so the
base-year calibration is kept and only projected change is applied. The
population concepts differ (SSA's area population runs above the frame's
Census-calibrated total), and forcing the level would move the base year.
`anchor="projection"` targets absolute counts for a frame that should adopt
the projection's level.

## Bounds

`max_weight_ratio` (default 5) bounds how far any record's weight may move
from its base-year value across the whole horizon, since each year is fitted
from the base-year weights, not from the previous year's. Cells no record
supports are skipped rather than targeted; they appear in the year's
`demographic_fit` with a zero base.

## What it is not

The base cross-section is reweighted once per year with no person identity
across years. No transition happens: employment status stays at its
base-year distribution by age, so a projected downturn appears only as slower
per-capita income growth spread over everyone. That is the Dynamics
operator's job (sequencing step 6), and this step is replaced, not extended,
when it lands.

## Country adapters

`microcosm.frame.adapters.policyengine_us.uprating_series` reads the series
each PolicyEngine-US variable uprates by and evaluates them at the projection
years: national totals under the calibration tree (directly or through the
`_per_capita` series PolicyEngine-US derives) come back as totals, everything
else as indices. `multi_year_dataset` exports a base-year bundle plus its
projected years as a `USMultiYearDataset`, which the engine treats as already
extended.
