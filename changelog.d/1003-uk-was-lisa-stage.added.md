A person-grain `was_lisa` stage joins the UK spine right after `was_wealth` (microcosm#1003; María's rulings of 2026-09-23 and 2026-09-29). It gives every adult a Lifetime ISA holding, imputed from the Wealth and Assets Survey round-8 person tab (UKDS SN 7215, End User Licence), and exports three new cells: `person.has_lifetime_isa`, `person.lifetime_isa_balance` and `household.household_lifetime_isa_balance`.

The donor. The person tab is pinned by size and sha256 beside the `was_wealth` household tab and read selectively (`read_pinned_tab(..., columns=)`). `clean_was_lisa_donor` joins each person to its household on `CASER8` and refuses the tab unless the released values `DVFLISAvR8` sum to the household's `DVFLISAVR8_aggr`. A missing or negative value is refused, never read as zero. The response class is kept for the receipt and never used as a predictor: observed, banded, ONS-imputed or rule-impossible. A declared credibility rule recodes holders in age bands from 45 up to non-holders (LISAs open only to adults under 40 since April 2017). Holders above £40,000 stay holders but leave the balance fit.

The model. Ownership is a weighted ridge logistic with an unpenalised intercept (`impute_lifetime_isa_ownership`). Its predictors are the age group, log earnings, household net income, gross financial wealth, cash ISA, stocks-and-shares ISA and savings, private renting, sex and the number of children. An adult holds when a person-keyed uniform falls below their probability. The balance is the regime-gated QRF fitted on the credible holders and drawn at a second person-keyed quantile (`impute_lifetime_isa_balance`), so it is positive exactly when the person holds.

Coherence. A household total above its `gross_financial_wealth` draw is scaled down pro rata (`cap_lifetime_isa_to_financial_wealth`, receipted), because WAS counts LISAs inside gross financial wealth. The household cell is the sum of its persons (`aggregate_person_to_household`). No `was_wealth` column is rewritten. Balances are 2020-22 pounds from a Great Britain donor, not uprated. No contribution or bonus column is created.

Gates. `uk_stage_was_lisa_support` (`stage_health`, support clip, release-blocking) joins the spine gate scope. `uk_support` gains `was_lisa_support_bounds.json`, generated with `--check` by `tools/build_uk_was_lisa_support_bounds.py`.

Surfaces that move in lockstep:

- the sources manifest and its projection;
- five schema branches and the operation-kind allow-list;
- the graph roster, kernel registry and spine tool (`--was-person-tab`);
- the H2 fixture (35 stages, a synthetic `was_person.csv`) and the coverage manifest;
- the export allow-lists, the country package and the gate-policy digests;
- two net-new-column signed differences.

Receipts are in `experiments/1003-uk-lisa-receipts.md`, aggregates in `docs/evidence/uk-lisa-1003/`, and the topic doc is `docs/uk-lisa-1003.md`.
