# Legacy QRF target chains in the graph

`microcosm.fit.graph_legacy_qrf` declares one fitted-model node and one application
node per ordered target. It preserves the existing QRF estimator, resolved donor
weights, observed donor prior targets, raw recipient prior draws, and sequential
PCG64 streams. It does not select a donor universe or change target encoding,
null authority, finalization, calibration, or a country's scientific recipe.

## Declarations and artifacts

```python
from microcosm.fit.graph_legacy_qrf import (
    LegacyQRFTrainKernel,
    LegacyQRFApplyKernel,
    legacy_qrf_train_nodes,
    legacy_qrf_apply_nodes,
)

fits = legacy_qrf_train_nodes(
    "conditional.fit",
    population="prepared.donor",
    entity=entity,
    predictors=ordered_predictors,
    targets=ordered_encoded_model_targets,
    seed=registered_pattern_seed,
    n_estimators=registered_tree_count,
    phase=registered_outer_phase,
)
draws = legacy_qrf_apply_nodes(
    "recipient.apply",
    population="prepared.recipient",
    fit_nodes=fits,
    seed=registered_pattern_seed,
    phase=registered_outer_phase,
)
# Register both kernel classes and add these nodes to the existing graph.
```

The donor population contains the already encoded predictors and complete ordered
model-target columns with the exact typed entity weights. The recipient population
contains encoded predictors. Both retain their original pandas table indices.
The graph store preserves supported integer and string indices; it currently
rejects MultiIndex. These wrappers do not replace indices with graph entity IDs.

Each fit owns `model` and `training_state` artifacts. The small canonical JSON
checkpoint contains the donor/configuration/fit-RNG state and ordered model hash
history. Its successor consumes that checkpoint without loading the previous
forest. Recipient rows, weights, source pins, and application RNG are absent from
the fit inputs and training checkpoint.

Each application consumes the model and matching training checkpoint from the
same fit producer. Later applications also consume the previous application state
and every prior raw-target artifact in order. It owns one `raw_draw` artifact and
one small `apply_state` checkpoint. The checkpoint binds ordered model and raw
target hashes; changed prefix bytes, index/order, or chain seed fail validation.
`read_raw_target` validates the raw envelope against its target and pandas index
and returns the exact readonly float64 bits. Call it on bytes loaded through the
verified graph store. Production decoding and output installation are separate
build operations; never feed snapped or calibrated values back as raw priors.

The initial application stream is exactly
`numpy.random.SeedSequence(seed).spawn(2)[1]`, as in `QRF.start_chain`. Use the
existing pattern/base seed for legacy parity. Later targets advance the stored
stream. The wrappers retain zero-target no-draw transitions and the existing
10,000-row prediction chunks. A separately declared application seed can reuse
the same fit, but that creates a different draw experiment; it does not establish
equivalence to the original seed.

## US preparation boundaries

For ACS transfer, `acs_transfer._fit_family_patterns_banked` defines the existing
entity/family/availability-pattern boundary. It creates complete-case target
encodings, the ordered required/observed-optional predictor tuple, complete donor
support, `_model_frame`, and `_encoded_predictor_frame` for the selected recipient
positions. `_pattern_seed` binds base seed, entity, family, and observed-optional
names. Preserve those operations and their order. Pattern ordinal is bookkeeping,
not a seed or model identity. Model-target order can differ from exported leaf
order when a joint codec owns multiple leaves.

For primary PUF transfer, `prepare_us_puf_tax_detail_chain_inputs` supplies the
donor model Frame and recipient feature table. The stacked path requires complete
recipient predictors and preserves absent output cells. Keep the schema-6 source
universe, target-order and clone-role authority with the preparation and finalizer.
The graph model nodes consume the prepared tax-unit surface; they do not perform
person allocation or `finalize_us_puf_tax_detail_predictions`.

The current stacked gap-fill owner projects donors from the assembled population.
For reuse across recipient rungs, a production graph must declare the prepared
donor population independently of recipient sources and sampling fractions.
Filtering a combined CREATE still inherits its recipient dependencies. Reuse is
valid only while actual donor support, values, weights, target encoding and
pattern stay fixed. A changed donor rung requires new fits. These wrappers do
not themselves port the ACS/PUF preparation or install final production columns.

## Trust, resources and evidence

All model/state/raw outputs declare `Numeric.PLATFORM_BITWISE`. Consumers require
the current graph platform scope. Model bytes bind implementation, dependency
versions, platform, donor inputs and fit state; envelope checks and the expected
full-byte SHA precede unpickling. Load only trusted local producer/store artifacts:
a checksum does not make an untrusted pickle safe. Fitted forests can contain
source-private values and are not public dashboard attachments.

Implementation invalidation is conservative: changes to shared codecs or the
underlying target protocol can invalidate fits. One application loads one target
forest; successors fit from small state. Raw prefix memory still grows with the
number of chained targets, and graph population retention has its own cost.
Profile the actual encoded source populations, serialization and retention before
scaling; small fixtures do not establish a production memory bound.

The flat `test_graph_legacy_qrf.py` suite compares graph raw bits and checkpoint
state with direct legacy calls, checks fresh-store required reuse, spies on real
fits across recipient changes, verifies donor/pattern invalidation, and exercises
malformed raw/model envelopes. These are engineering tests, not evidence that a
national or congressional-district candidate passes its release criteria.
