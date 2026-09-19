# Static aging

`microcosm.calibrate.static_aging` projects a base-year frame to later years.
It reweights a fixed cross-section independently for each year. The proposed
operator remains subject to the cross-sectional-versus-longitudinal design
decision in [#333](https://github.com/PolicyEngine/microcosm/issues/333).

## The split

Two things change between years, and the step keeps them apart.

**Weights carry who exists.** For each projection year the operator recalibrates
the weight entity's weights to that year's projected population by demographic
cell. The US reader supplies age and sex from SSA's Trustees Report; the package
does not yet include a UK projection reader. It targets demographics only. A program count
or an income total is an output of rules, take-up and demographics; a weight vector forced to match
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

The frame must store weights only for `weight_entity`. Other entities derive
their weights from it, so explicit stale person or tax-unit weights cannot
override the projected weights. The operator supports person-level weights
as well as group weights, and preserves the frame's link tables.

`max_weight_ratio` (default 5) caps each record's weight at that multiple of
its base-year weight. It imposes no reciprocal lower bound. The operator fits
each year from the base-year weights. It skips cells that no record supports;
they appear in the year's
`demographic_fit` with a zero base.

## Income and losses

For a national-total column with both positive and negative values, the operator
applies the series' growth to each gross component separately. If `g` is the
series growth ratio, `B+` and `B-` are the base weighted positive and negative
totals, and `A+` and `A-` are those totals under the projected weights before
scaling, the factors are `g * B+ / A+` and `g * B- / A-`.

Both factors remain positive. Each record keeps its sign, each gross component
grows by `g`, and their sum preserves the projected net growth, including when
the base net is zero or reweighting changes its sign. This assumes that positive
amounts and losses share the same growth rate; a net projection alone does not
identify their separate growth rates. `YearProjection.factors` represents these
pairs with `microcosm.frame.SignedScale`. Index columns retain one common ratio.

An all-zero column stays zero. A component with zero weighted support in both
years uses an identity factor. If only one year has weighted support, the
operator raises an error: positive factors cannot create or remove support.
Both `frame_for` and the US dataset exporter apply factors in float64.

## What it is not

The base cross-section is reweighted once per year with no person identity
across years. No transition happens: employment status stays at its
base-year distribution by age, so a projected downturn appears only as slower
per-capita income growth spread over everyone. That is the Dynamics
operator's job (sequencing step 6), and this step is replaced, not extended,
when it lands.

## Country adapters

`microcosm.frame.adapters.policyengine_us.uprating_series` reads the engine's
dataset-extension overrides before each variable's declared uprating series.
It evaluates those series at the projection years. CBO and IRS SOI calibration
series, including the source totals behind derived `_per_capita` parameters,
come back as totals; other series come back as indices.
`multi_year_dataset` exports a base-year bundle plus its projected years as a
`USMultiYearDataset`, which the engine treats as already extended.
