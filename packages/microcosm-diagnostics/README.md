# microcosm-diagnostics

`microcosm-diagnostics` owns the versioned calibration-diagnostics models,
runtime validation, and the only atomic JSON writer for current documents.
Solver adapters assemble a typed schema-8 model, but they do not write the
artifact directly: the writer publishes it as schema 9. Release assemblers
validate and copy current documents without rewriting them; compatibility code
may still adapt historical schema-6 and schema-7 documents. This package
deliberately does not depend on the calibration solver or distribution
packages.

## Schemas 8 and 9

Producers assemble schema 8 (`CalibrationDiagnosticsV8`), in which every target
row carries its provider, category, geography and dimension labels. Schema 9
(`CalibrationDiagnosticsV9`) is what the writer publishes: the same document
with each of those labels declared once, in a top-level `hierarchy_catalogs`
object, and referred to by ID from `targets[].hierarchy` (`category_id`,
`geography_id`, `dimensions[].id` and `value_id`; the target's own label stays
on the row). One identifier therefore cannot carry two labels in a published
file. Every other field is schema 8's.

`catalog_calibration_diagnostics` builds the catalogs from the rows, listing
entries in the order the rows first use them. It refuses
(`HierarchyLabelConflictError`) a document whose rows give one identifier more
than one label, or one geography more than one level, unless a
`HierarchyLabelChoices` names the label: per geography ID, per dimension value,
or, for dimensions whose values are geography IDs (Chronicle's `geography`),
the label of the geography with that ID. A choice applies only where the rows
disagree. The writer reports a refusal as the failure code `label_conflict`
and writes nothing. `expand_calibration_diagnostics` rebuilds the schema-8
document from a schema-9 one. Releases published before schema 9 keep their
schema-8 files, and `parse_calibration_diagnostics` reads both.

Both schemas include typed UK extension models for the shipped-weight summary,
zero-weight strata, geography-level pass rates, observation bases, family and
area fit summaries, and measured or explicitly skipped rotated holdout runs.
The models reject undeclared fields and reconcile counts, shares, ratios, fold
summaries, and shared top-level values before serialization. Solver options,
target metadata, and build provenance remain explicitly extensible JSON
mappings because their keys are producer-defined; every stable diagnostics
record has a declared model. Historical schema-6 and schema-7 UK documents
retain a read-only compatibility validator in `microcosm-data`; it is not used
for current production or validation.

Every current model rejects `NaN`, positive infinity, and negative infinity at
any nesting depth. Producers may convert a calculated value to `null` when
`null` explicitly means that the measurement is unavailable, but values that
reach validation as non-finite numbers fail construction. The writer validates
the schema-9 document it builds, so a non-finite value introduced into a
schema-8 model after construction is refused there, and it serializes with
non-finite output disabled.

## JSON Schemas

`calibration_diagnostics_v9_json_schema()` and
`calibration_diagnostics_v8_json_schema()` generate Draft 2020-12 JSON Schemas
from `CalibrationDiagnosticsV9` and `CalibrationDiagnosticsV8`, for consumers
that check the file without importing these models. The models remain the
authority. Their cross-field reconciliations (including, for schema 9, that
every row refers to catalog entries that exist), finite-number and
non-blank-text checks, and strict integers cannot be written in a schema
generated this way. Each schema's description lists these rules
(`CALIBRATION_DIAGNOSTICS_V9_JSON_SCHEMA_GAPS`,
`CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS`), and a test fails if a model
validator is added without being listed. Complete validation still goes
through `parse_calibration_diagnostics`.

The generated schemas are committed, and shipped in the wheel, at

`packages/microcosm-diagnostics/src/microcosm/diagnostics/schemas/calibration_diagnostics.v9.schema.json`

for the documents current builds publish, and at

`packages/microcosm-diagnostics/src/microcosm/diagnostics/schemas/calibration_diagnostics.v8.schema.json`

for documents published before schema 9.
`load_calibration_diagnostics_v9_json_schema()` and
`load_calibration_diagnostics_v8_json_schema()` read them. After changing a
model, regenerate both with
`uv run --no-sync python tools/generate_calibration_diagnostics_json_schema.py`;
`--check` reports a stale file without writing, and `--schema-version` limits
the tool to one of them. A test fails while a committed file differs from
freshly generated output, and the US, UK and transport build tests validate the
diagnostics they write against the schema-9 file.
