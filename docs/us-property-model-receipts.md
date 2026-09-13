# Property model receipt verification

`verify_property_model_receipts(nodes, donor_population, recipient_population,
artifacts)` in `graph_property_income_receipts.py` reconstructs the existing
four fit and four apply receipt dictionaries. The caller passes the ordered
property fragment, the two original `Population` branches, and authenticated
artifact bytes keyed by `(node_id, output_name)`. Additional non-model fragment
nodes and their artifacts may be present; the country host verifies those.

The caller must first bind the store bytes, typed producer outputs and exact
population dependencies. This helper loads the existing trusted model format,
which contains pickle. A self-consistent digest or a returned receipt does not
authenticate arbitrary model bytes or establish source authority. The owning
host retains responsibility for source qualification, complete population
identity, physical seals, borrowed inputs and final I/O checks.

## Checks

The helper reconstructs the shared graph declarations for the four ordered
property targets and compares their complete normative bytes. This binds input
slices, population versions, predictors, target order, seed, phase, fit options,
model/training sibling edges, prior raw-draw edges and output artifact types.
Training stays donor-only. The existing model protocol initializes its exact
chain-start state against the original design-weight donor without fitting.
Each trusted model must match the complete ordered training history, actual
consumed float64 donor bytes, donor pandas index, resolved weights, configuration
and preceding/succeeding training states. The first three property components
remain nonnegative and the four-component sum must equal the observed donor
aggregate. Both branches require finite float64 model columns and unique int64
person IDs.

Ordinary population apply checkpoints have a pandas-index identity but no
independent recipient-feature digest. The verifier therefore performs four
additional `apply_target` operations using the retained fitted models, current
recipient features, raw prefix and exact initial draw RNG. It compares the
resulting raw bytes and complete application states with the published
artifacts. It validates every model/raw hash in the ordered application history.
It never calls `fit_target` or a model-fitting convenience method.

This replay establishes output consistency with the supplied feature values;
a model can produce the same output for different inputs. Exact feature and
person-ID provenance remains the caller's authenticated population binding.
The existing protocol uses pandas row labels for model state and raw draws;
the verifier does not substitute entity IDs or invent a new identity digest.
Graph owners must continue to bind the distinct int64 person axis themselves.

## Scope and cost

The return value uses the existing raw kernel receipt schema; the host adds and
checks the executor capability/input-writer envelope through its existing
`_states` path. It adds no status,
source issuer, receipt artifact type, model format or host framework. The
helper does not fit, reconcile, attach clones, assign tax treatment or modify
its input frames. Cost per call is one chain initialization, four trusted model
decodes and four deterministic apply replays over the recipient branch. The
owning host decides when to perform this validation; no native-scale runtime
claim follows from the small tests.

Tests reuse the actual twelve-node property graph fixture with two trees,
44 invented donor persons, four recipient persons, non-default pandas row
labels, int64 IDs above `2**53` and zero-weight records. Original cold and
required-cache kernel receipts match the verifier. Separate controls alter
donor/recipient axes and values, design weights, graph declarations, model and
training bytes, model/raw history and draw RNG. Country engines, native sources,
the full PUF host and publication are outside these checks.
