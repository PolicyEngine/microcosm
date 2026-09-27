Add optional benchmark publishers and executable reform definitions to schema-v1
reform validation reports, ported from PR629 at
`a0441178c9a74bf55dd5aeea62ff9039092c3432`. Preserve scores, years, signs, model
calls, and the `microcosm` payload key; the eight shipped configs gain only the
229 publisher fields from that source.

Record the actual baseline and reform worlds for simulated comparisons,
including each OBBBA stack step and its merged initial counterfactual when used.
Configured OBBBA repeal definitions remain distinct from those executed
comparisons. Copy definitions so report mutations cannot change input specs.
