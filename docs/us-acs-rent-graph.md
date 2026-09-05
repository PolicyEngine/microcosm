# ACS rent graph adapter

The build-owned adapter exposes the registered ACS rent operation as independent source, fit, recipient preparation, draw, assignment-validation, and finalization nodes. It calls the actual archived donor loader and housing preparation/finalization helpers. It does not load a candidate population, select a source origin, infer missing predecessor inputs, calibrate weights, or certify a release.

`us_acs_rent_nodes` consumes a truthful, explicitly declared recipient projection after the measured hours, prior-year, relationship, and tenure preparation steps. The caller supplies its full `Owned` inventory, `base`, and the producing `frame_context` gate. The caller also declares `RENT_SOURCE` and passes an authenticated descriptor from `prepare_acs_rent_source_descriptor` for the exact registered ACS 2022 archive. The source CREATE independently reloads the pinned archive and recomputes that descriptor. Production registration retains the archive pin; tests use an explicit separate registry with the SHA of a synthetic archived-layout H5 file.

```python
from microcosm.build.us_runtime.graph_acs_rent import (
    RENT_OUTPUT_CONTEXT,
    RENT_OUTPUT_POPULATION,
    register_us_acs_rent_kernels,
    us_acs_rent_nodes,
)
from microcosm.build.us_runtime.graph_acs_rent_source import (
    RENT_SOURCE,
    prepare_acs_rent_source_descriptor,
    register_us_acs_rent_source_codec,
)

# Local, explicit preflight; acs_rent_path is the registered archived source.
descriptor = prepare_acs_rent_source_descriptor(acs_rent_path)
register_us_acs_rent_source_codec(source_codecs)
register_us_acs_rent_kernels(kernels, source_codecs=source_codecs)
nodes = us_acs_rent_nodes(
    prepared_columns,
    base=prepared_population,
    context_producer=prepared_context_gate,
    donor_descriptor=descriptor,
    seed=production_seed,
)
# Add RENT_SOURCE to Graph.sources and bind its name to acs_rent_path.
# Next population: RENT_OUTPUT_POPULATION.
# Next context: ArtifactInput(..., RENT_OUTPUT_CONTEXT, 'frame_context', ...).
```

The default forest size remains the actual housing helper's 100 estimators; the caller must provide the existing production seed. Donor sample selection, allocation masks, categorical levels, design weights, fit RNG, sequential draw RNG, raw draws, clipping, and the corrected household-to-head scatter retain the reviewed helper contracts. The donor staging IDs are its existing prepared sample-index labels, not Census household IDs. They retain the actual helper's reset/sample/filter index and never derive from recipient membership, weights, or input source pins.

The recipient operator consumes only declared person and household views. It constructs a two-entity view of those actual tables for the actual helper, preserving their original pandas indices and household weight authority; it does not fabricate missing US entity tables. The matrix contains the helper's exact finite float64 predictors on its ordinary int64 household-ID index. Both matrix and placement artifacts declare current-platform `PLATFORM_BITWISE` scope. Model pickle artifacts retain the fit package's trusted-local-producer and implementation/version/platform envelope checks; this is not an untrusted model interchange format.

Each rent kernel binds its explicit source closure and NumPy/pandas/PyArrow versions; the donor also binds h5py. The implementation identity records pandas `mode.string_storage` and `future.infer_string`, plus the resolved dtype, storage and array classes from constant-size construction, `astype(str)` and state-code padding probes. Changing either option or the reported PyArrow version changes rent node keys and invalidates required-cache reuse. Donor and recipient receipts record that string runtime, and those encoding calls refuse a detected option/backend change during their execution. Run a graph with fixed pandas options; the before/after fence does not make process-global pandas options safe to mutate concurrently. These guards do not assert numerical equivalence between Python and Arrow string backends.

The recipient preparation reads the input population, so an unavailable incumbent rent column cannot accidentally read the final rewrite. The assignment gate runs on the output population and consumes no rewritten rent input. The finalizer reads its own incumbent through the graph's declared rewrite contract. The caller's population/context dependencies enforce the actual predecessor order.

Placement binds the matrix SHA, original person index and ordered IDs, household IDs, head IDs/positions, and the exact incoming context/encoding producer keys and hashes. The gate and finalizer both recheck current household/head membership, the original indices, complete training/application-state agreement, model history, same-producer raw/state siblings, matrix producer/SHA, and the final raw SHA. A named failed assignment gate leaves inspectable evidence and blocks finalization. A passing assignment gate attests only this placement contract, not rent distribution quality or national/CD release eligibility.

All inner nodes declare `phase='prepare_multispine_source_inputs_for_clone'`. A phase names the owning construction phase; it is not a promise that every ready auxiliary node runs consecutively inside that population phase. The outer driver must derive the 11-stage evidence from the explicit main population/context dependency path, record auxiliary donor/model branches separately, and preserve every node's declared phase. It must not add recipient scheduling dependencies to the donor branch.

The current boundary uses an all-person FILTER. Valid Frame linkage already requires every group row to be referenced. Defensive incoming-context count checks reject stale context. Generic FILTER currently resets valid nondefault group indices. The adapter tests the production-style unnamed RangeIndex group representation and an independently indexed person table. The outer integration must include the companion US context-producer guard, which refuses nondefault or named group indices before emitting context; that producer can inspect all actual group tables. The person index remains arbitrary. This adapter does not include the companion guard or change generic FILTER semantics, and it does not claim arbitrary group-index preservation.

For `H` recipient households, `P` people, and `C` encoded predictors, the serialized matrix body is `8*H*(C+1)` bytes, placement vectors are `24*H` bytes, and the raw rent vector is `8*H` bytes. The placement and matrix canonical headers are each bounded at 64 KiB. The final person vector adds `8*P` bytes. These are payload sizes, not a peak-memory estimate: pandas preparation, view construction, encoding and decoding allocate additional arrays. The real archived loader materializes its source arrays before the existing joint sample cap of 10,000; preflight and first CREATE each load it. Forest memory and 100-tree timing require local profiling before a real build. Tiny two-tree tests establish correctness, not production capacity.

The outer build still owns source selection/merge, H5/runtime stage receipts, release validation and national/CD calibration. The in-repo integration tests use actual archived-layout H5 loading, actual predecessor tenure preparation, actual QRF fitting/drawing, and actual nullable Frame checkpoint roundtrips. They make no tax or policy-result claims.
