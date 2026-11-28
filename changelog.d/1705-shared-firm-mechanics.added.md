`microcosm.build.firms`: country-neutral mechanics for synthetic firm populations, taken out of the UK firm port so another country can reuse them (#1705). A country passes its band edges, band roles, registered counts, engine package and receipts as data:
- `draw_in_band` maps caller-drawn uniforms into a band, log-uniform for an open band, and keeps the input's type and dtype;
- `stratified_thin` thins a candidate draw by stratum, with exact base weights;
- a `RegistrationSpec` declares a register's bands as excluded, registrable or selected, which `registration_propensity` and `assign_scope_and_registration` use to reconcile a business register with a tax-registration count;
- `FirmRules`, `FirmRuleOutputs` and `align_rule_outputs` form the seam to a rule engine, with named outputs keyed by firm id, plus `weighted_totals`;
- `read_entity_variables` reads a policyengine-core package's entity variables from source with `ast`, and `input_status_problems` and `export_schema_problems` check a build's input-status record and its export against them;
- `firms.receipts` sets model totals beside sourced official receipts, without inventing a gap for an unset input or a missing figure.

The UK firm port (#1706) is to be rebuilt on these and will pin that its outputs are unchanged. The firm-only `Frame` is not included: it needs `EntitySchema` root populations.
