# Housing derivation from declared tables

`derive_us_housing_tables(person, household, spm_unit)` returns copied household and SPM-unit tables, in that order. It reads the existing `H_TENURE`, `SPM_CAPHOUSESUB`, and `SPM_TENMORTSTATUS` fields and preserves the original mappings and refusals. Its four derived leaves are `household.tenure_type`, `spm_unit.receives_housing_assistance`, `spm_unit.takes_up_housing_assistance_if_eligible`, and `spm_unit.spm_unit_tenure_type`.

SPM values align through `person_spm_unit_id` to each supplied SPM row, regardless of person or group row order. Group indices, column order and unrelated columns survive; inputs remain unchanged. Existing positive measured subsidy sets both assistance flags, zero and negative zero leave them false, and negative subsidy or disagreement within an SPM unit retains the previous refusal. The existing zero household-tenure mapping remains `NONE`.

The `derive_us_housing_inputs(Frame)` wrapper retains its US schema requirement and delegates to this helper, preserving other entity tables, typed weights, strata, mass history and metadata. Callers provide actual linked table views; the pure helper does not construct missing entities or reinterpret source fields.

The flat behavioral tests compare against values, dtypes, indices and refusal messages captured from the unedited wrapper at `9aac31c67def0733bb5e6f3acc6e26c734c2781a`; the synthetic fixture records that source fingerprint. Graph declarations and source projection remain the outer build's responsibility.
