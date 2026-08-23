UK weighted-integrity evidence now comes from the shared Frame-typed
helpers: `uk_input_mass_totals` wraps the country-agnostic
`input_mass_totals` (removing the exported `household_weight` column —
plumbing, not mass) and `uk_qrf_tail_concentration_columns` reads the
Frame directly, replacing the UK-only table walkers
(`uk_dataset_input_mass_totals`, `_uk_entity_weights`). Totals keys are
flat frame column names instead of `entity.column`, so the reviewed
enhanced-FRS input-mass reference digest was re-frozen keys-only against
the #610 values (131 columns before and after, values byte-identical).
Benunit nesting ("a benunit's members share a household") is now a
stated UK frame invariant enforced at construction and re-proved at every
stage seam by `validate_uk_national_frame`; orphan benunits stay
unrepresentable at the kernel.
