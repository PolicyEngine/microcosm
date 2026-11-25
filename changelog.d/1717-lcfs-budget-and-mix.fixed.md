The LCFS consumption stage's imputed spending ran a quarter above its donor in the bottom income decile (microcosm#1717). Two fixes.

The donor's income predictors carry the frame's concepts. `employment_income` is `p008p`, normal gross wage or salary on the 13-week rule, where it was `b303p`, the pay last time paid. `self_employment_income` sums `p047p` and `p037p`, income from main and from subsidiary self-employment, where it was `b3262p`, the first subsidiary job's profit alone (LCF 2024-25 User Guide Volume F). The frame's predictors are gross pay and self-employment profit.

The spending is no longer drawn target by target, each target conditioned on the earlier random draws, which compounded the forests' pull to the middle at the bottom of the income range. The new `draw_budget_and_mix` operation draws each household's budget, total spending over the twelve COICOP divisions, from its neighbourhood in a forest fitted to the donors' log total spending, read between neighbouring donors' totals. Its mix comes from one donor drawn from a second forest's neighbourhood, fitted to the division shares with the predictors and log total spending as features. The household takes that donor's spending scaled to its budget: the divisions sum to the budget, each fuel, fare and energy part keeps the donor's share of its division, and the excise shares are the donor's own. Both draws read uniforms keyed on the household id. The general tools are `forest_neighbourhoods`, `keyed_neighbour_donors` and `keyed_neighbour_quantiles` in `microcosm.fit.donors`; the `microcosm.fit` package itself is unchanged.

On the LCFS 2024-25 diaries, fitted on half the households to impute the other half over 12 splits, total spending in the bottom decile comes out at 0.97 of actual, and every other decile within 5%. The spread of spending among similar households is 1.05 of actual, and the correlations between divisions sit at their sampling floor. On the stage's own input, 0.5% of frame households share an exact basket with another: mixes repeat, at 34,953 frame households on 4,995 donors, and budgets do not.

A `decile_fit` receipt sets the frame's spending beside the donor's by income decile on every build. Each survey is ranked on its own equivalised predictor income, and the receipt gives aggregates with support bands, withholding any decile on fewer than three source households. The report-only gate `uk_stage_lcfs_consumption_decile_fit` warns when a decile 1 to 9 leaves the donor's shape, its ratio over the pooled ratio of deciles 4 to 7, by more than 10%.

Surfaces that move in lockstep:

- the sources manifest and its projection;
- a schema branch and the operation-kind allow-list;
- the stage's hash bindings, which now include `microcosm.fit.donors`, and the spine's seed key;
- the H2 fixture and the coverage manifest;
- the gate battery's registries and digests.
