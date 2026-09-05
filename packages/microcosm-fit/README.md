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

## Legacy draws from derived recipient matrices

The target-at-a-time legacy protocol in `microcosm.fit.graph_legacy_qrf`
retains the archived weighted fit, observed donor target prefix, raw recipient
target prefix, sequential RNG streams and prediction chunking. It differs from
the ID-keyed draw protocol above. Each donor fit owns a frozen model and a
small training checkpoint; recipient changes can reuse these fits when the
donor population has independent source ancestry.

For prepared features whose existing pandas index differs from the population
index, a build-owned preparation node can emit
`ArtifactOutput("matrix", RECIPIENT_MATRIX_TYPE)` using
`microcosm.fit.model_input.encode_recipient_matrix(features, entity=...,
entity_ids=...)`. The graph does not create a population or change its index.
V1 accepts only a nonempty ordinary int64 `pd.Index`, with a None/string name,
exactly equal to the ordered unique entity IDs. Ordered predictor names must be
unique, and every predictor must already be finite little-endian float64.
RangeIndex, nullable Int64, object and MultiIndex inputs are refused without
casting. The canonical header is limited to 64 KiB and numeric body lengths
must match exactly. Recipient weights are absent because application does not
consume them.

```python
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_qrf import legacy_qrf_apply_matrix_nodes
from microcosm.fit.model_input import RECIPIENT_MATRIX_TYPE

# fit_nodes is the complete tuple from legacy_qrf_train_nodes. The preparation
# node owns the real matrix artifact and its country-specific row association.
apply_nodes = legacy_qrf_apply_matrix_nodes(
    "rent.apply",
    population="recipient_stage",
    fit_nodes=fit_nodes,
    matrix_producer="rent.prepare_recipient",
    seed=31,
    phase="early_gap_fill",
)
```

Register `LegacyQRFApplyMatrixKernel` alongside the existing training kernel.
Every application reads the same declared matrix edge. It owns the unchanged
raw-target format and a distinct `MATRIX_APPLY_STATE_TYPE` checkpoint wrapping
the existing legacy packet, plus the matrix SHA-256 and actual graph producer
key. Both bindings must agree across the chain, including when identical bytes
come from a new producer. Model/sibling training provenance, platform scope,
ordered raw-prefix digests and legacy state/RNG validation run in the same
shared body used by the existing Slice adapter. Model envelopes validate
version, implementation and the mandatory sibling SHA before trusted-local
unpickling; they do not make foreign pickle safe. These artifacts are
platform-bitwise and cannot silently cross architecture/OS/Python boundaries.

Both application implementation hashes include that shared module; matrix
application also includes its codec. The refactor rekeys existing Slice apply
nodes once, while their state and raw-target bytes retain the legacy format.
The committed golden fixture freezes pre-refactor raw bytes and all application
state except platform-bound fitted-model identifiers. The portable test obtains
only those identifiers from the actual independent sibling training outputs.

A country finalizer must read `decode_matrix_apply_state`, the final raw draw
and the preparation's placement artifact. It must verify that the state's
matrix digest and actual producer key match the placement's matrix digest and
producer key, and that the last raw-history digest equals the consumed raw
artifact. It must reconstruct the original prepared index from placement IDs
and name, independently recompute the current household-to-head association,
and require exact agreement with the stored positions and person-row identity.
For US rent, the original index is an unnamed household-ID int64 Index; scatter
positions follow household draw order, which can differ from person order.
Neither the generic raw decoder nor the matrix state parser certifies placement
by itself. Finalization, decoding, calibration and acceptance remain build-owned.

Storage is `8 * rows * (predictors + 1)` body bytes plus the header. At 62
predictors this is 50.4 MB for 100,000 rows and 504 MB for one million rows
(decimal MB). Encoding materializes the returned bytes and temporary buffers;
decoding views the body but validates finiteness/uniqueness and constructs pandas
index metadata. The existing application also copies predictor/raw-prefix data,
and the executor loads and hashes artifact bytes for consumers. Peak memory is
not a fixed multiple of payload size: measure the real small rung and per-target
loads before scaling. This codec provides no streaming execution guarantee.
