# CPS-carried inputs in the US graph

`us_runtime.graph_cps_carried.us_cps_carried_nodes` declares the existing
measured CPS operation before hours, prior-year inputs and rent preparation.
Pass the actual upstream column inventory, population version and producer
of the full US frame-context artifact. Register the implementations with
`register_us_cps_carried_kernels`. Both inner nodes declare the authored outer
phase `prepare_multispine_source_inputs_for_clone`.

1. `early_preparation.cps_carried.boundary` validates the existing pre-clone
   source authority and raw evidence in the registered `multispine_pool`
   helper, verifies full-context counts, and opens an all-row graph version.
2. `early_preparation.cps_carried` runs the actual person/SPM source projection,
   measured mapping and ID merge. It owns only the existing CPS-carried person
   leaves and three SPM leaves, then publishes the completed full context and
   the original one-operator receipt. Downstream uses this context producer
   and the boundary population version.

There is no existing mandatory CPS signal gate. This operation publishes no
invented certification or fitted-model claim. Source/category/structure errors
retain their original refusal and prevent completed context publication.

The shared `derive_us_cps_carried_tables` helper copies and transforms only the
real person and SPM tables. The legacy `derive_us_cps_carried_inputs` Frame
wrapper delegates to it and retains its explicit optional-sidecar arguments.
The graph passes no sidecars and has no live source fallback: PAW_TYP and
the restored NOW_* observations already belong to the authenticated v4 source.
Source-role projection stays in the registered source owner. The ordinary
graph module has no source-provenance exemption.

The executor supplies actual structural IDs and person memberships. Owning
SPM outputs supplies the actual SPM table even when it initially contains only
its ID; no dummy column, empty Slice or invented SPM row is needed. Actual
incumbent fields are declared as reads, including the automatic incumbent read
of an owned rewrite. The source helper selects and merges by IDs, preserves
full-pool row order, and uses canonical resolved Weights to retain selected
population validity. Full descriptors and legacy metadata/history remain
authenticated; a partial Frame never receives whole-pool metadata.

The original presence semantics apply. All-null family columns are absent
on the selected source view; partially null incumbents remain present. Raw
OI_VAL/OI_OFF intentionally recompute their three other-income leaves. Native
ACS values remain untouched, and original formula-owned-column and missing
source-pair refusals remain enforced. The annual receipt interpretation for
TANF/SNAP and the reporting-adult WIC carrier are unchanged.

Money retains its existing nominal source-year basis. This adapter performs
no price alignment and does not establish a pool of 2024-adjusted incomes.
Source-year-aware price alignment is a separate scientific operation.

Graph rewrites require stable dtypes. Missing boolean leaves become nullable
`boolean`; boolean incumbents must already use that representation. Plain
bool or numeric placeholder incumbents refuse explicitly. Supported numeric
incumbent dtypes remain unchanged; incompatible assignments retain the legacy
refusal and are never silently cast. The preserved real f001 geography schema
has `is_female=boolean`, age and existing income leaves=float64, with no SPM
incumbents, so the native ACS surface is supported. US context version 1
requires default unnamed group RangeIndex; arbitrary person indices are
supported. The graph accepts one explicit weight source. The direct Frame API
retains its wider representation and weight-vector support.

Implementation identity binds the explicit CPS, alimony, source-boundary,
context and provider module closure, shared Frame/graph runtime behavior,
the source_stages.json resource and NumPy/pandas/PyArrow versions. It also binds
the live output/formula projection consumed by the receipt and the declared
read/output rosters. Changes elsewhere inside a listed module conservatively
invalidate this slice. No source-import, geography, rent or QRF adapter is
used as a substitute implementation identity.

Tests execute the actual helper and graph on tiny synthetic sources. They
compare direct values, ordered receipts, typed context, native peer cells,
null/dtype behavior, weights and strata on cold, warm and required-cache runs.
Pre-refactor complete codec/context/receipt fingerprints remain pinned.
Frame requires sorted group IDs: shuffled SPM alignment is tested at the pure
table seam, and shuffled person membership order is tested in valid Frames.
No real source data or fitted model is needed for these checks.
