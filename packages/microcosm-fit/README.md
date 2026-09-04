# microcosm-fit

The conditional-models operator of the [microcosm](../../DESIGN.md) stack —
imported as `microcosm.fit`. It fits conditional distributions `P(y | x)` over a
`microcosm.frame.Frame` and draws from them.

## Weight-aware by construction

Every fit reads the Frame's typed weights. There is no unweighted default: a
fit that ignores weights cannot be expressed except by passing `weights="none"`
explicitly, and the function says why that is the only escape hatch. `weights`
selects which typed weight vector to use — by default the **design** weights of
the entity that owns the predictors and targets.

This closes the 2026-06 weight-handling failure mode where a silently ignored
weight column reproduced a high-income regime's mass at the wrong scale. Here
the weights are materialized into the fit by **weighted bootstrap**: training
rows are importance-resampled by weight before each forest is grown, so leaf
distributions — and every value drawn from them — reflect the weighted
population, not the unweighted sample.

## The canonical model

`QRF` (alias `RegimeGatedQRF`) is a regime-gated, sequentially-chained
quantile-regression-forest imputer:

- **Regime gates.** Each numeric target's sign support (negative / zero /
  positive) is detected structurally (unweighted) from the training data. A
  zero-inflated target gets a zero-vs-nonzero gate so its zero mass is
  preserved exactly; a sign-mixed target gets a gate per sign so draws never
  interpolate across a zero crossing.
- **Chaining.** Targets are imputed sequentially; each conditions on the
  predictors plus the targets already drawn, so the joint structure across
  targets is preserved.
- **Draws.** A random quantile is sampled per row (seeded) and the forest is
  queried at it, so the draws sample the weighted conditional.

```python
from microcosm.fit import fit

fitted = fit(frame, predictors=["age", "is_male"], targets=["capital_gains"])
draws = fitted.predict(frame)  # one column per target

# Unweighted is opt-in and explicit:
fitted_unweighted = fit(frame, predictors, targets, weights="none")
```

## Reusable graph models

`microcosm.fit.graph_models` separates donor training from recipient draws.
Register `QRFTrainKernel()` and `QRFApplyKernel()` in the graph kernel registry.
The training node reads one donor population Slice containing `predictors`
then `targets`, owns no columns, and declares
`ArtifactOutput("model", QRF_MODEL_TYPE)`. Its required parameters are tuples
`predictors`, `targets`, and an integer `seed`; `n_estimators`, `zero_atol` and
`max_samples_leaf` retain the public fitter's defaults.

Each application node reads a recipient Slice containing exactly those
predictors and declares `ArtifactInput("model", train_node_id, "model",
QRF_MODEL_TYPE)`. It owns one all-row `float64` column per target in fitted
chain order. Parameters `random_stream=("sha256-u53-v1", experiment_id,
replicate, base_seed)` and integer `period` control its draws. Random
coordinates include the entity ID, target, period and draw kind. Give the same
person the same stream and period to couple counterfactual draws; change the
experiment or replicate for a distinct set of draws. Recipient edits preserve
the fitted model's cache identity; donor values and effective weights do not.

The new training kernel excludes zero-weight rows before regime detection and
records the source typed weight kind separately from the public DataFrame
fitter's resolved `explicit` weights. Both kernels declare platform-bitwise
numerics; neither promises cross-platform prediction equality. Model artifacts
contain validated versioned metadata and a pickle from trusted local training.
Content verification establishes integrity, not safe loading of untrusted
pickle. The existing combined `fit.qrf@1` kernel remains available.

Outside a graph, `FittedRegimeGatedQRF.predict_from_uniforms` accepts mappings
`quantiles={target: array}` and `sign_uniforms={target: array}`. Both must name
every fitted target and carry finite arrays in `[0, 1)` aligned to recipient
rows. The method preserves fitted forests and RNG state. Keep each row's
uniforms with its identity when reordering or batching. Ordinary `predict()`
retains its existing stateful RNG behavior.

The synthetic two-destination integration is runnable with
`python -m microcosm.build.transfer_example --output <directory>`; see the
microcosm-build documentation for its separate calibration and held-out checks.

## Dependencies

The heavy dependencies (`scikit-learn`, `quantile-forest`) live here, never in
`microcosm-frame`: an analyst doing imputation installs this shard; an analyst
doing only calibration never pulls them.

`scikit-learn` is unpinned on the upper side (`>=1.5`): the shard's test suite
passes on both 1.8 and 1.9, so it coexists in one environment with a
`scikit-learn>=1.9` consumer. This requires `quantile-forest>=1.4.2`, the first
release that tracks the sklearn-1.9 tree ABI
(zillow/quantile-forest#152, #153); older `quantile-forest` fails to import
under sklearn 1.9.
