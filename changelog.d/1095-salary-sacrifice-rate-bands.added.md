The three HMRC salary-sacrifice income tax relief bands now bind: basic, higher and additional rate, from Table 6.1 for 2024-25 (uk-data#533, microcosm#1095). They had been held out of the fit since microcosm#1069, because their band test compared adjusted net income with after-allowance thresholds and ignored the Scottish bands.

The measure resolver gains a declared rate-class allocation. One counterfactual simulation per substitution pair returns the sacrifice to pay, as uk-data does, and is cached. Each person's relief is the rise in tax on `earned_taxable_income`, bracket by bracket, under their rest-of-UK or Scottish schedule (`pays_scottish_income_tax`). The schedule is read from the simulation's own fiscal-year parameters, so the Scottish bands are the ones in force from 6 April. The brackets group as follows:

- a bracket taxed below 30% is basic;
- the schedule's top rate is additional;
- the rest are higher, so the Scottish advanced rate counts as higher.

A contribution that straddles a boundary is relieved at each rate, and the personal allowance taper lands in the brackets the extra pay reaches. The three bands sum to the change in tax on earned income, and the resolver's receipt records the gap to the change in `income_tax` (for example, dividends pushed into the higher dividend band). A binding that declares anything other than the exact allocation is refused. uk-data's worked examples at 2025 reproduce exactly: £454 basic and £692 higher, Scotland's £349.02 and £981.96, and £6,000 all higher where the taper applies.

policyengine-uk puts the Scottish top-rate threshold at £112,570 rather than the statutory £125,140 (pe-uk#2130), so it files some advanced-rate relief as additional. The Table 6.2 all-rates total (£8.8bn) becomes a diagnostic on the higher-rate band, since the bands carry the same relief. The three measure exclusions that held the bands out, due to expire on 25 November, are retired. The contract now holds 390 targets and 1,231 active references.
