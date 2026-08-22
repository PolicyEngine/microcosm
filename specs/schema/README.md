# Bundle schemas — moved into the package

The fifteen-schema catalog lives at
`packages/microcosm-build/src/microcosm/build/spec_engine/schema/` —
package data, one copy, identical bytes in editable installs and wheels.
This directory retired as a drafting location on 2026-08-18 (the wheel
force-include it fed broke sdist builds and created an editable/wheel
dual-source asymmetry).
