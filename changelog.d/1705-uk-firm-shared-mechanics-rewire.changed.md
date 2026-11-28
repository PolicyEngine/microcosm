The UK firm port now runs on the country-neutral `microcosm.build.firms` layer (#1705):
- `firm_universe` uses its thinning and the two-universe functions, with the HMRC VAT bands declared as `UK_VAT_REGISTRATION_SPEC`;
- `firm_rules.UKFirmRuleOutputs` is a `FirmRuleOutputs` carrying `vat_liability_if_registered`;
- band turnover is drawn with `draw_in_band`;
- the policyengine-uk snapshot tool reads with `read_entity_variables`;
- the receipts tool uses `firms.receipts`.

The same-seed thinned 2024-25 population is unchanged, compared exactly against the code before the rewire on the same `main`.

The build receipt now records the rule period apart from the years the source data describe (`data_years`). For 2024-25 the ONS frame's turnover is mostly from 2023-24 VAT returns (ONS UK Business 2025, note 4). The receipt also notes that a change dated later in the tax year, such as 1 January, is not apportioned unless the parameter is day-weighted.
