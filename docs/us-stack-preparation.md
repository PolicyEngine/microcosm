# US source preparation and harmonization

US source construction can emit a real DESIGN population before importance
allocation. `prepare_stacked_spine` performs the existing seeded household
sampling, source mass normalization, table stacking, structural ID remapping,
and provenance construction. `harmonize_stacked_spine_weights` then computes
the actual IMPORTANCE weights from declared household views and a preparation
receipt. These are separate operations; preparation does not calculate or
store the final weight vector.

```python
from microcosm.build.us_runtime.stacked_spine import (
    prepare_stacked_spine,
    harmonize_stacked_spine_weights,
)

prepared = prepare_stacked_spine(
    asec, acs, sample_fraction=1.0, sample_seed=578,
)
harmonized = harmonize_stacked_spine_weights(
    household=prepared.frame.table("household")[[
        "household_id", "household_support_channel",
    ]],
    weights=prepared.frame.weights_for("household"),
    preparation=prepared.receipt,
)
```

The result contains typed `weights`, completed `metadata`, the completed
stacked-spine `receipt`, and `legacy_mass_log`. For an ordinary direct call,
`finish_stacked_spine(prepared)` constructs and validates the final Frame.
`assemble_stacked_spine` composes the same operations and preserves historical
sampling controls, output tables, weights, strata, metadata, and ordered mass
logs. Its existing accepted source weight kinds remain supported. The new
preparation functions require actual DESIGN source weights; they do not
relabel calibrated or importance weights.

For more than two source channels, `prepare_spines` and
`harmonize_spine_weights` expose the same table/weight separation without
ASEC/ACS sampling. Channels retain the existing anchor-first, then lexical
order. Group table IDs retain the existing stable sort. A permutation is
stored only when the assembled household ordering differs from concatenated
source order. Harmonization restores that source order before computing sums
and roundoff corrections, then realigns the output. It does not require a new
sorting convention for valid source Frames.

## The normalized DESIGN reference

Production sampling retains the existing whole-household selection and
multiplies sampled source weights by `M_full / M_sample`. Here the masses are
sums of the source weights, so this ratio generally differs from the realized
inverse household inclusion count `N / n` for unequal weights. At a sample
fraction of one, normalization leaves the original weights unchanged. This
is the existing normalized source-design reference, not a new claim about a
survey estimand. Legacy pilot sampling retains its existing direct allocation
from sampled ACS mass.

Importance allocation retains the existing share rule and exact residual
correction: each source receives its declared share of the selected anchor
source's incoming mass. No calibration threshold or cap changes here. A
solver's start-relative cap and an executor's bound relative to the DESIGN
reference are separate requirements. A later adapter must detect any conflict
without silently loosening either bound.

## Preparation evidence and graph integration

`StackedSpinePreparation.receipt` has schema
`microcosm.us.stacked-spine-preparation.v1`; its nested assembly context has
schema `microcosm.us.spine-preparation.v1`. Both are recursively frozen,
JSON-compatible mappings. The graph's `canonical_json` encodes them directly;
decoded JSON can be passed to the harmonization helpers.

The preparation receipt contains sampling controls and realized sampling
evidence, full-source masses, shares, anchor, assembly provenance and native
ACS group-quarter lineage. Its nested assembly context binds actual normalized
household weight bytes, assembled ordered IDs and channels, original source
ID order, per-channel counts and incoming masses, and ordered source log
segments. Numeric digests use contiguous little-endian int64 IDs and float64
weights; the channel digest uses length-prefixed UTF-8 strings. No completed
stacked manifest is attached until allocation succeeds. The existing completed
manifest validator continues to reject preparation Frames.

The preparation producer validates the complete Frame and source lineage.
The view-only consumer verifies the household input bindings and computes
allocation; it relies on the graph's authenticated typed artifact edge for
the producer-owned sampling and non-household evidence. These digests do not
make a caller-forged receipt authoritative. A graph adapter must declare the
preparation artifact and its actual household columns and DESIGN weight view
as inputs.

Graph REWEIGHT outputs carry the executor's structural mass ledger. The exact
legacy per-source operator history returned as `legacy_mass_log` belongs in
the typed US context; an adapter must preserve it in order without replacing
the executor ledger. The helper does not invent a successful gate, execute a
calibration, or change clone ancestry. This change supplies the operations;
the production graph adapter owns their execution wiring.
