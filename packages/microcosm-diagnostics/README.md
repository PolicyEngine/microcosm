# microcosm-diagnostics

`microcosm-diagnostics` owns the versioned calibration-diagnostics models,
runtime validation, and only atomic JSON writer shared by Microcosm producers
and release validators. Solver adapters may assemble a typed model, but they do
not write the artifact directly. This package deliberately does not depend on
the calibration solver or distribution packages.
