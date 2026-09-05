# Prior-year person preparation

`prepare_us_prior_year_person(person, *, weights, seed, time_period, predictors=None)` runs the existing source manifest and handlers on a real person table. Its canonical `Weights` vector must follow that person row order. The helper does not construct entity tables or infer weights.

Before cloning, the table carries no support roles and requires no other entity or predictor table. It retains the original adjacent-income-year join, allocation-flag and unavailable-value handling, signed self-employment losses and fallback rules. It also preserves the original presence-based pass-through when all outputs exist and required raw fields are absent, including partially null incumbents; this pass-through is not a completeness certificate.

The legacy post-clone path requires the actual person/tax-unit predictor projection, with the expected columns and identical person index. It runs the original joint QRF and removes the original formula-owned wage output. `with_us_prior_year_income_inputs` derives that projection from the full Frame, calls the same table helper, and preserves other entity tables, weights, strata, metadata and history. Existing complete Frames still pass through unchanged.

This extraction does not change monetary periods or apply inflation factors. Source-year alignment and graph fit/apply composition are separate work. Supplying the wrong number of weights, an untyped weight vector, missing or misaligned support predictors, or unrequested predictors on a source-only table is refused.

Verification covers exact archived-wrapper parity on source rows, nondefault person indices, partial-null pass-through and a real tiny post-clone 100-tree QRF run; all four cases retain identical complete checkpoint bytes on the current platform. Existing prior-year tests and the source-blindness suite also pass. This is engineering evidence, not national or district calibration acceptance.
