Add separate `fit.qrf.train@1` and `fit.qrf.apply@1` graph kernels with typed,
reusable model artifacts and stable entity-coordinate draws. The fitted QRF's
new `predict_from_uniforms` API supports stateless chained predictions while
preserving the existing `predict` RNG stream. Training excludes zero-weight
rows from its effective support and records both source and resolved weight
provenance.
