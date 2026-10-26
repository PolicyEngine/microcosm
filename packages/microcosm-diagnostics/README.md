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

`calibration_diagnostics_v8_json_schema()` generates a Draft 2020-12 JSON
Schema for schema 8 from `CalibrationDiagnosticsV8`, for consumers that check
the file without importing these models. The models remain the authority.
Their cross-field reconciliations, finite-number and non-blank-text checks,
and strict integers cannot be written in a schema generated this way. The
schema's description lists each of these rules
(`CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS`), and a test fails if a model
validator is added without being listed. Complete validation still goes
through `parse_calibration_diagnostics`.

The generated schema is committed, and shipped in the wheel, at

`packages/microcosm-diagnostics/src/microcosm/diagnostics/schemas/calibration_diagnostics.v8.schema.json`

`load_calibration_diagnostics_v8_json_schema()` reads it. After changing a
schema-8 model, regenerate it with
`uv run --no-sync python tools/generate_calibration_diagnostics_json_schema.py`;
`--check` reports a stale file without writing. A test fails while the
committed file differs from freshly generated output, and the US, UK and
transport build tests validate the diagnostics they write against it.
