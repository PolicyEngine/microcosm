# US production graph development

The source and geography graph stages run the existing US operations on actual
declared inputs. They are development stages, not a terminal pool or a certified
dataset. Downstream source preparation, fitted transfers, structural support,
validation and calibration must still run before a release can be evaluated.

`us_runtime.graph_sources.us_assembly_graph` declares two nodes under the
`assemble_stacked_spine` phase. The preparation CREATE reads the authenticated
operator-free ASEC v4 checkpoint and the two national ACS2024 PUMS archives. It
performs the existing whole-household sampling and full/sample mass ratio
normalization, then stacks the actual DESIGN weights. Its separate preparation
artifact contains source lineage and allocation controls, with no precomputed
importance weights or completed stacked-spine manifest. This captures the
normalized sampled DESIGN anchor used by later graph weight constraints.

The REWEIGHT node reads declared source-channel and membership views plus the
same producer's typed preparation/context artifacts. The existing harmonization
helper computes IMPORTANCE weights with exact source-order floating operations.
Household share allocation and the graph's declared person mass by source
stratum are distinct checks. All six resolved entity weights are compared with
the direct operation in the tiny integration tests.

`us_runtime.graph_geography.us_geography_nodes` then declares lookup import,
an all-row version boundary, household assignment and its integrity gate.
The raw-byte source codec reads bounded regular files; existing decoders validate
the PUMA2020 NPZ and the 117th-to-119th crosswalk. Their district universe must
match exactly. Assignment preserves native PUMAs and the existing seed0 draw
order. It reads household cells only and owns PUMA, congressional district and
county outputs. Every inner node declares its outer phase. The integrity gate
checks the existing assignment rules; it does not establish CD fit or accuracy.

The versioned `microcosm.us.frame_context` artifact carries audited operator
metadata, ordered entity IDs/column inventories, explicit weight kinds and the
legacy operator mass history. Population cells, memberships, weights and strata
remain graph-owned. Context and preparation must have the declared typed producer;
reconstruction checks the available entity identities. Unknown operator metadata
and operational fields are refused. Geography updates only the household column
inventory while preserving untouched descriptors.

Two mass histories are retained deliberately. The executor records the actual
CREATE-to-REWEIGHT transition in its aggregate ledger. The context retains the
source-ordered legacy operation history. A direct-checkpoint comparison restores
the authenticated legacy history and metadata after checking the graph population
identity; it never overwrites the executor's manifest. Nullable integer storage
uses the existing Frame checkpoint schema4 extension and preserves exact values,
including observed zero and integers above2**53.

Tests cover actual small source construction, normalized anchors, source/codec
identity, tampered context, missing or corrupt required-cache inputs, source
mutation, direct output/weight/context parity, complete HDF5 byte parity, geography
failure evidence and a fresh-store replay with zero source calls. Model and source
implementation identities use the reviewed scopes described below. The source
importer binds its actual h5py dependency.

## Implementation identities

The `@2` US graph kernels use explicit whole-module manifests from
`us_runtime.graph_implementation`. `implementation_manifest(stage)` returns the
module/resource digests, installed dependency versions, codec references and
live source-boundary registry projections used by `implementation_hash(stage)`.
Kernel receipts retain that small manifest without source rows or local paths.
The kernel's `Capabilities.dependencies` and the manifest use the same tuple.

| Scope | Bound behavior | Dependencies |
|---|---|---|
| `asec_codec_v4` | Authenticated ASEC checkpoint, source pins, nullable H5 and source boundary | numpy, pandas, h5py |
| `acs_codec_2024` | ACS parser/native mapping, microunit Python and rule YAML, source boundary | numpy, pandas, microunit, PyYAML, pyarrow |
| `assembly_prepare` | Both codec identities, household sampling/normalization and DESIGN stack preparation | numpy, pandas, microunit, tables, h5py, PyYAML, pyarrow |
| `assembly_harmonize` | Actual importance allocation and typed preparation/context | numpy, pandas |
| `geography` | Lookup decoding, version boundary, assignment and integrity gate | numpy, pandas |

All scopes conservatively bind the existing Frame and graph Python rosters plus
the shared context, dtype boundary and manifest helper. The checked-in
`graph_implementation_inventory.json` lists each scope's modules and resources.
It records canonical import-module names with explicit classifications. Compact
fingerprints retain the lexical uses of unbound helpers and resource access
expressions; ordinary symbol edits in an already bound module need no inventory
refresh. Added dependencies/resources, missing modules and
new unclassified Frame/graph/microunit files fail before cache lookup. The fence
is a static review aid, not a Python sandbox or a proof of every possible dynamic
call. A scope change requires review of the actual entrypoint's transitive route;
do not automatically refresh the inventory merely to make a failure pass.
For an inspectable audit, `_dependency_details(module_bytes, canonical_name,
_covered_imports(canonical_name, inventory))` expands those two fingerprints.
The expansion is review output and is not duplicated in packaged production data.

The two live `operator_boundary` registries bind operator-output and formula-owned
columns. Editing a later rent/hours operator body or adding aggregate-quality code
does not invalidate assembly. Changing a provider's exported column roster does.
The current shared assembly modules still contain other functions, so edits
within those files may conservatively invalidate more than one scope. Updating
the shared inventory is also a conservative identity change for every scope.

Microunit's installed version alone cannot identify an editable rule helper or
YAML file. The ACS scopes therefore include both byte inventories and PyYAML's
version. Geography NPZ/crosswalk bytes remain declared source inputs, separate
from implementation resources. Default packaged geography resources belong to
other entrypoints and are not read by this graph route. File discovery does not
import optional Frame adapters just to obtain their paths.
The ACS parser can use Arrow before canonicalizing its output. Its scope also
binds PyArrow's version, pandas string-storage/inference options and the resolved
string backend. A tiny real CSV fixture matches values, weights, strata and
metadata across backends, while the column-name Index and a string subsequently
added by native mapping (`tenure_type`) retain their backend. These are distinct
implementation identities; the fixture does not establish complete backend
equivalence.

`us_source_codecs()` verifies each registered name's real loader module,
qualified name and source location. Preparation repeats that validation before
required-cache lookup. Generic graph source keys and cache behavior are unchanged;
this adds no separate decoded-source cache. The private `_decode_graph_*` reader
seams allow tiny fixtures while retaining the actual registered entrypoints.

This is a new identity baseline. Historical `@1`/c71 stores and successful profiles
remain valid evidence for their exact code and inputs; they are not aliased or
rewritten as `@2` cache hits. A runner must record separate preparation,
harmonization and geography identities, then establish its new cold/replay
baseline. The old private `_source_implementation_hash` spelling now means only
preparation, not a shared geography hash. No profiling runner is changed here.

Real-source runs must keep their source hashes, code identity, measured memory,
wall time, store size and failure receipts. A successful fraction run remains
development evidence. A failed run with no saved population has no population
quality metrics. Neither state changes a production, staging or latest pointer.
