# US post-transfer parity string normalization

The pool-tool synthetic helper and unchanged stacked assembly retain the
following textual cells as pandas `object` under legacy string inference.
Interface amendment 10 requires graph strings to use
`StringDtype(storage="python")`. The generator casts exactly this audited
surface before serializing the common source used by both the oracle and
graph. No values, row order, column order, weights, mass records, receipts,
or metadata change.

- `person`: `person_support_channel`
- `household`: `household_support_channel`
- `tax_unit`: `tax_unit_support_channel`
- `spm_unit`: `spm_unit_support_channel`
- `family`: `family_support_channel`
- `marital_unit`: `marital_unit_support_channel`
- source strata: `person.__stratum__`
