The Pension Credit stage stores the SI 2019/37 mixed-age couple saving (`has_mixed_age_couple_pension_credit_saving`, pe-uk#1940, uk-data#519, microcosm#1095). The value is the engine's own default, evaluated once at the survey-year period on the post-SPI receipts.

The default reads a birth year computed from age and the model year. microcosm holds ages fixed, so the default would drop one more protected cohort each later model year.

The flag holds for a mixed-age couple that meets all of these:

- the older member was born by 5 February 1954;
- the couple reports Pension Credit, or pension-age Housing Benefit without an income-related legacy benefit;
- the couple reports no Universal Credit.

The entitlement the stage redraws take-up on reads the same value.
