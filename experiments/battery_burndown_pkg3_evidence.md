# pkg3 1% verify build: battery failure-line diff vs baseline

Same code path, sample fraction 0.01 seed 578, checkpoint-resumed;
physical (mirror-deduplicated) failure lines from pool.gates.json.

- baseline1pct: **127** failures
- pkg3:         **114** failures
- greened: **15**, introduced: **2**

## Greened by pkg3
- person/adult_care/pre_subsidy_care_expenses[clone_0]/positive: weighted positive-leg incidence ratio 0.561425 is outside [0.8, 1.25] (asec=0.000768207, acs=0.000431291).
- person/model_required_numeric/unemployment_compensation[clone_0]/positive: conditional-quantile envelope distance 0.352941 exceeds 0.25.
- person/simulated_output/ssi[clone_0]/positive: weighted positive-leg incidence ratio 1.34148 is outside [0.8, 1.25] (asec=0.0203726, acs=0.0273295).
- person/source_operator_child_support/child_support_expense[clone_0]/positive: conditional-quantile envelope distance 0.953846 exceeds 0.25.
- person/source_operator_child_support/child_support_expense[clone_0]/positive: weighted positive-leg incidence ratio 0.171281 is outside [0.8, 1.25] (asec=0.00523716, acs=0.000897025).
- person/source_operator_child_support/child_support_received[clone_0]/positive: conditional-quantile envelope distance 1 exceeds 0.25.
- person/source_operator_child_support/child_support_received[clone_0]/positive: weighted positive-leg incidence ratio 0.242882 is outside [0.8, 1.25] (asec=0.00928306, acs=0.00225469).
- person/source_operator_disability_benefits/disability_benefits[clone_0]/positive: conditional-quantile envelope distance 1.37353 exceeds 0.25.
- person/source_operator_prior_year_income/self_employment_income_last_year[clone_0]/positive: conditional-quantile envelope distance 0.834721 exceeds 0.25.
- person/source_operator_prior_year_income/self_employment_income_last_year[clone_0]/positive: weighted positive-leg incidence ratio 1.37675 is outside [0.8, 1.25] (asec=0.0249193, acs=0.0343077).
- person/source_operator_weeks_unemployed/weeks_unemployed[clone_0]/positive: conditional-quantile envelope distance 0.736842 exceeds 0.25.
- person/source_operator_weeks_unemployed/weeks_unemployed[clone_0]/positive: weighted positive-leg incidence ratio 0.0253844 is outside [0.8, 1.25] (asec=0.0341695, acs=0.000867374).
- person/source_operator_workers_compensation/workers_compensation[clone_0]/positive: weighted positive-leg incidence ratio 0.0476147 is outside [0.8, 1.25] (asec=0.00326463, acs=0.000155444).
- spm_unit/source_operator_energy_subsidy/spm_unit_energy_subsidy[clone_0]/positive: conditional-quantile envelope distance 0.666667 exceeds 0.25.
- spm_unit/source_operator_energy_subsidy/spm_unit_energy_subsidy[clone_0]/positive: weighted positive-leg incidence ratio 0.240477 is outside [0.8, 1.25] (asec=0.033554, acs=0.00806897).

## Present only under pkg3 (same checks, ratios moved toward band, still red)
- person/simulated_output/ssi[clone_0]/positive: weighted positive-leg incidence ratio 1.33548 is outside [0.8, 1.25] (asec=0.0203726, acs=0.0272073).
- person/source_operator_weeks_unemployed/weeks_unemployed[clone_0]/positive: weighted positive-leg incidence ratio 0.0313711 is outside [0.8, 1.25] (asec=0.0341695, acs=0.00107194).
