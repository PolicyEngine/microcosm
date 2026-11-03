The UK spine now sets `is_blind` from the FRS blind registration question (uk-data#523, microcosm#1095). Before, the column was never written, so it was False for everyone and took policyengine-uk's default.

A person is blind when `SPCREG1` is 1, meaning registered blind or severely sight impaired with the local authority or social services. Registration follows the consultant ophthalmologist's certificate that the engine's `is_blind` names. The adult and child tabs ask the same question, so children are read too.

Partial-sight registration (`SPCREG2`) does not meet the severely-sight-impaired test, so it is not read. A person who was not asked the question (blank `SPCREG1`) is not blind.

In FRS 2024-25, 29 adult records (57.3k at the `gross4` design weights) are registered blind. Fewer than 10 child records are, so that count is suppressed.

On the locked policyengine-uk 2.100.0, `is_blind` is read by:

- the TV licence concession;
- the Disabled Students' Allowance qualifying condition;
- the Council Tax Reduction non-dependant exemption;
- the Tax-Free Childcare age limit and higher amount for a disabled child.

`person.is_blind` is a net-new export column. It joins the `uk_export_surface` allow-list (`gates.json` and `UK_ALLOWED_EXTRA_EXPORT_COLUMNS`) and the spine graph's root person booleans. The following are regenerated:

- `source_stages.json`;
- the H2 spine fixture, whose synthetic adult and child tabs now carry `SPCREG1`;
- the coverage manifest;
- the mirrored gate digests.
