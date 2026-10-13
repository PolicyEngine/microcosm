# microcosm-diagnostics

`microcosm-diagnostics` owns the versioned calibration-diagnostics models,
runtime validation, and the only atomic JSON writer for current schema-8
documents. Solver adapters may assemble a typed model, but they do not write
the artifact directly. Release assemblers validate and copy current documents
without rewriting them; compatibility code may still adapt historical schema-6
and schema-7 documents. This package deliberately does not depend on the
calibration solver or distribution packages.

Schema 8 includes typed UK extension models for the shipped-weight summary,
zero-weight strata, geography-level pass rates, observation bases, family and
area fit summaries, and measured or explicitly skipped rotated holdout runs.
The models reject undeclared fields and reconcile counts, shares, ratios, fold
summaries, and shared top-level values before serialization. Solver options,
target metadata, and build provenance remain explicitly extensible JSON
mappings because their keys are producer-defined; every stable diagnostics
record has a declared model. Historical schema-6 and schema-7 UK documents
retain a read-only compatibility validator in `microcosm-data`; it is not used
for schema-8 production or validation.

Every current model rejects `NaN`, positive infinity, and negative infinity at
any nesting depth. Producers may convert a calculated value to `null` when
`null` explicitly means that the measurement is unavailable, but values that
reach validation as non-finite numbers fail construction. The writer also uses
strict JSON serialization with non-finite output disabled, so a value introduced
after model construction returns a structured serialization failure instead of
silently becoming `null`.
