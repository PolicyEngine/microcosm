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
