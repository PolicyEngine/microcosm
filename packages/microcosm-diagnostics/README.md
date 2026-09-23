# microcosm-diagnostics

`microcosm-diagnostics` owns the versioned calibration-diagnostics models,
runtime validation, and the only atomic JSON writer for current schema-8
documents. Solver adapters may assemble a typed model, but they do not write
the artifact directly. Release assemblers validate and copy current documents
without rewriting them; compatibility code may still adapt historical schema-6
and schema-7 documents. This package deliberately does not depend on the
calibration solver or distribution packages.
