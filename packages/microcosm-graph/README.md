# microcosm-graph

One object replaces stages, families, batches, banks, and whole-run
authority receipts: a content-addressed DAG of cell-ownership nodes.

A `Node` declares the slices it reads, the cells it owns, its parameters,
and the kernel that computes it. Its key is the hash of that declaration,
the artifact keys of its inputs, and the kernel's implementation hash. The
executor projects immutable input views, runs the kernel, patches only the
owned positions with a storage-preserving assignment, and memoizes every
output in a content-addressed store keyed by node key. Seeds derive from
node keys. Provenance (the run manifest) is a list of node keys plus signed
human decisions and never feeds back into a key.

`docs/graph-acceptance.md` is the definition of done: every property there
is an executable test, committed red, and the shard is finished when none
is marked `xfail`.

Module map:

| Module | Owns |
|---|---|
| `decl.py` | Frozen declarations (`Graph`, `Node`, `Slice`, `Owned`, `SourceRef`) and `compile_graph` — **frozen interface** |
| `kernel.py` | `Kernel` protocol, `KernelContext`, `KernelResult`, `Capabilities`, `KernelRegistry`, `source_hash` — **frozen interface** |
| `canonical.py` | Canonical JSON of the normative projection; domain-separated SHA-256 |
| `keys.py` | Node key and artifact key derivation |
| `store.py` | `ContentStore`: atomic content-addressed artifacts, codecs, resume policy |
| `population.py` | Immutable population versions: `Frame` + owner map + weight lineage + mass ledger |
| `executor.py` | `run_graph`: projection, patching, ownership enforcement, receipts |
| `manifest.py` | `RunManifest`, `NodeReceipt`, human decision records |
| `view.py` | `describe(node)`: the one-screen view |

The shard depends on `microcosm-frame` only. Kernels that wrap fit,
calibrate, or a rules engine live in those shards and register here.

## Reusable typed artifacts

A fitted model can be a dependency across population versions without owning a
population column. `ArtifactOutput("model", ArtifactType("example.model", 1))`
declares a required byte output. A consumer names it with
`ArtifactInput("fitted", "train", "model", ArtifactType("example.model", 1))`
and reads `context.artifacts["fitted"].payload`. The executor exposes only the
declared aliases as immutable `ArtifactValue` objects, including producer
identity and numeric scope. A nominal type/version is an interface contract;
the consuming kernel must validate its decoded payload. Existing opaque
model/diagnostic bytes remain supported, but cannot satisfy an edge unless the
producer declares their type.

The compiler adds these edges to dependency ordering, cycle checks, and gate
ancestry. A change to recipient inputs can reuse the fitted producer. Typed
node cache records use schema 2; runs carrying typed edges or outputs use
manifest schema 3. Legacy nodes omit the new empty declarations from keys and
JSON, and legacy runs retain manifest schema 2. Cache reload validates the same
contracts as fresh execution.

If a gate raises while producing typed evidence, its ordinary failed verdict
and exception receipt remain available, but its declared byte artifacts do
not exist. The executor records their names in `receipt.execution` under
`schema="microcosm.graph.execution.v1"`; required
consumers and their causal descendants are `unreached` and execute no kernel.
They produce no replacement columns, weights, frames, or model bytes.
Independent branches and readers of the available verdict still run. An
unreached release remains evidence-tier. These outcomes are cached and replay
under `resume="require"` with no kernel calls. Exceptional cache records use
schema 3 and exceptional run manifests use schema 4, which older readers must
refuse; healthy typed records/manifests retain schemas 2/3. Execution metadata
with that namespace is reserved to the executor; historical free-form
`execution` diagnostics remain compatible. Cached and portable blocker
provenance is validated before use.

After correcting a transient gate failure, replay the full graph with
`resume="forbid"` to recompute the gate and refresh the affected cache records.
Subsequent `auto` and `require` runs can reuse the recovered results. Deleting
only the gate's cached receipt leaves stale descendant records, which the
executor refuses. Invalid receipts returned by a kernel reject the node;
they are not converted into gate-failure evidence.

Numeric contracts are deliberately restrictive: bitwise artifacts permit any
consumer class; platform-bitwise artifacts require platform-bitwise consumers;
tolerance-bound artifacts require tolerance-bound consumers with their own
output tolerance. Mixed platform/tolerance artifact inputs are refused. The
executor does not infer error propagation through an arbitrary computation.

## Population retention

`run_graph(..., population_retention="lazy")` keeps live populations through
their last scheduled use, then releases them. The default `"eager"` behavior
is unchanged. Every realized version remains available through
`manifest.population(version)` and `manifest.populations[version]` as a normal
`PopulationView` with the full Frame API. Listing versions, membership checks,
manifest keys and JSON do not load Frames. Repeated lookup returns the same
view while a caller holds it; weak memoization permits collection afterwards.
Unreached structural nodes still have no population.

Versions with ordinary column patches receive an auxiliary final-Frame
snapshot, so historical views include those patches. Snapshot identities bind
both producing receipts and actual codec payload hashes: a tolerance-bound
recomputation can produce different bytes under an unchanged node key. An
unchanged version reuses its structural artifact; a fresh computation colliding
with an older structural artifact gets a snapshot of its actual result. These
attachments do not change node keys, published node artifacts, portable
manifest schemas or manifest identity. Lazy reads verify payloads and producer
binding, and refuse artifacts changed or removed after the run.

Lazy execution requires a writable store. `resume="require"` still performs
full preflight and executes no kernels, but can reconstruct a missing auxiliary
attachment from verified cached results. `auto` verifies/reuses existing
attachments; `forbid` refreshes them. The existing atomic Frame codec publishes
snapshots after staging; staging files are renamed without a second payload
copy. Structural node column artifacts are written one copied Series at a time,
and source hashing uses bounded reads with the same byte identity and checks
for detected source changes.

The live manifest retains each Frame's frozen metadata descriptor alongside
its store reference. This preserves eager/lazy metadata on the same execution
path. The existing Frame codec omits metadata on warm source loads, and portable
manifests omit population attachments entirely; this option does not add a
portable metadata contract. Use declared typed context artifacts when metadata
must survive those boundaries.

Memory retention follows the live DAG frontier plus the current kernel and
write working set. Kernels, caller-held views, and metadata/provenance still
consume memory; this is not an absolute memory cap. Auxiliary snapshots add
disk space and I/O for patched versions. Profile a small end-to-end graph before
increasing the population size.

## Stable random coordinates

Opt-in kernels declare `SeedSource.KEYED`, put their stream tuple in normative
node parameters, and include `microcosm.graph.randomness` in their implementation
hash. The existing node-key RNG behavior remains available unchanged.

```python
from microcosm.graph import keyed_uniform

stream = ("sha256-u53-v1", "comparison", 0, 42)
values = keyed_uniform(
    stream=stream,
    keys=[(person_id, "mortality", 2027, 0) for person_id in person_ids],
)
```

The read-only float64 result is stable through reordering, chunk boundaries, and
unrelated inserted identities. Duplicate coordinates intentionally repeat a
draw. Integer and string identities differ. The versioned algorithm hashes a
canonical, type-tagged coordinate tuple with the stream and maps the leading
53 digest bits to `[0, 1)`. This is an explicit experiment stream independent
of cache identity; changing its normative specification still invalidates the
application node.
