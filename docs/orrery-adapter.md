# Orrery inputs

Microcosm writes the files from which the collector
(PolicyEngine/microcosm-collector) builds a graph's Orrery graph files: the
compiler schema (`graph.schema.json`), the execution overlay
(`graph.execution.json`) and the upstream files they link
(`upstream-<sha256>.json`). Microcosm owns the calculation and field
semantics those files record. It converts nothing into
`graph-explorer/v1`, the JSON format accepted by
[Orrery](https://github.com/TheAxiomFoundation/orrery): the collector does
that with its own copy of the projection, in
`telemetry_collector/orrery/projection/`, which is the reference for the
Orrery document's structure, relationships, evidence panels and limits.
Orrery owns graph navigation and presentation, and staff view a build's graph
through the run viewer (microcosm-runs), which reads it from the collector.
[UK graph inputs](orrery-publication.md) describes how a UK build registers
and hands over these files.

The files are made separately from graph execution:

1. `compile_graph(graph)` validates a `Graph` and derives operation order,
   population versions, field owners, and operation predecessors. The executor
   uses this `CompiledGraph`.
2. `graph_schema(compiled)` derives a portable static metadata artifact from
   that same `CompiledGraph`, and `save_graph_schema` writes it as
   `graph.schema.json` with copies of the upstream files it links.
3. `save_run_evidence` records each executed phase's native evidence.
   `collect_execution_evidence` checks that evidence against the schema and
   the content store and returns the execution overlay, which
   `execution_overlay_bytes` writes as `graph.execution.json`.
4. The collector projects the schema and the overlay into the Orrery graph
   files.

The schema is therefore a generated artifact, not an instruction that
Microcosm executes. A saved schema and overlay let another process read the
graph and its recorded execution without importing the original graph
construction code. That process needs none of the build runtime either:
validating a saved schema and its presentation contract
(`validate_graph_schema`, `validate_presentation`) and reading an overlay
file (`load_execution_overlay`) need only the standard library, and
`microcosm.graph` loads its runtime exports (the executor, content store,
kernels, codecs and evidence capture, which need numpy, pandas and
microcosm-frame) on first access. A test
(`packages/microcosm-graph/tests/engine_free/shared/test_graph_execution_overlay_file.py`)
runs those three calls in an interpreter that refuses to import numpy,
pandas, microcosm-frame or the store. The package's declared dependencies are
unchanged.

A kernel is the implementation of one graph operation, with a declared input,
output, numerical and execution contract.

## Compiler schema

Use the public API when the `Graph` or `CompiledGraph` is available:

```python
from microcosm.graph import compile_graph, graph_schema
from microcosm.graph.canonical import canonical_json

compiled = compile_graph(graph)
payload = canonical_json(graph_schema(compiled))
```

`save_graph_schema(graph, directory, presentation=...)` writes the same
schema as `graph.schema.json`, with a presentation contract and copies of
its linked upstream files (see below). The schema has exactly these root
fields:

```text
protocol, country, graph_sha256, graph, compiled,
fields, input_bindings, extensions
```

Only `extensions` is producer-defined.
`microcosm.graph.schema.validate_graph_schema` restores and recompiles the
embedded graph and requires every other field to equal the newly derived
result. A caller cannot change an owner, predecessor, population version,
field provider, or input binding while retaining a valid schema. The inputs
command below and the collector's projection both validate a saved schema
this way. Every serialized string in the schema, operation contracts
included, may be up to 131,072 characters, because operation parameters can
hold serialized validation-check contracts larger than 64 KiB; the schema's
other bounds are under [Evidence and limits](#evidence-and-limits).

## Fields and input bindings

Every field record identifies one visible value by:

- `population`, `entity`, and `column`: its coordinate;
- `provider`: the operation whose artifact supplies that value;
- `declared_in`: the operation containing the nearest applicable `Owned`
  declaration;
- `dtype`, `rows`, `ownership`, and `rewrite`: the declaration details.

A structural operation carries all fields from its base population. For such a
carried field, `provider` is the structural operation because its output
artifact is what downstream operations read; `declared_in` still points to the
earlier `Owned` declaration. An operation that writes or rewrites a field is
the provider after that operation completes.

`input_bindings` are also derived entirely from `Graph` and `CompiledGraph`.
They contain no Frame values and do not observe kernel behavior. Each record
connects one declared input role to the exact pre-operation field provider:

| `kind` | Declared role | `rows` |
| --- | --- | --- |
| `slice` | A column named by a `Slice` | The slice's row selector |
| `slice_mask` | The boolean column used to select a slice's rows | `all` |
| `output_mask` | The boolean column limiting an `Owned` output | `all` |
| `rewrite_incumbent` | The prior value replaced by `Owned(rewrite=True)` | The output's row selector |
| `materialized_expand_output` | A new column physically installed by an `EXPAND` operation and read by its following ownership-claim operation | The output's row selector |

For a rewrite, both the explicit slice and the implicit incumbent role resolve
to the value present before that operation. The final rewritten field is a
different identity. Repeated declared roles remain repeated schema records,
even when they refer to the same coordinate.

For `materialized_expand_output`, the input field's provider is the `EXPAND`
operation and its declaration source is the following claim operation. The
claim's final output receives a separate field identity, preserving both the
physical materialization and the ownership declaration in the schema.

These records describe values supplied to an operation. They do not claim that
a kernel read every supplied value at runtime, and they do not reproduce the
executor's complete causal-writer history, which may be refined by runtime
receipts.

## Country inputs and presentation

The schema keeps the interfaces introduced by #888. The graph schema remains
`microcosm.graph.schema.v1`; compiler facts still come from the same
compiler. Country is descriptive metadata. Shared graph code neither imports
a country builder nor dispatches on country names or operation prefixes.

Countries can supply `extensions["microcosm.presentation"]` with protocol
`microcosm.graph.presentation.v1`:

```python
from microcosm.graph import graph_schema

presentation = {
    "protocol": "microcosm.graph.presentation.v1",
    "scope": {"id": "national", "label": "National build"},
    "groups": [
        {"id": "sources", "label": "Sources", "sources": ["survey"]},
        {"id": "enrichment", "label": "Enrichment", "operations": ["impute"]},
    ],
    "operations": {"impute": {"description": "Fit and draw as one execution unit"}},
}
schema = graph_schema(
    compiled, extensions={"microcosm.presentation": presentation}
)
```

Group members name real operations or sources. Optional `parent` names another
group. `microcosm.graph.presentation.validate_presentation` rejects unknown
members, duplicate membership and containment cycles; the build's graph step
and the inputs command run it before they hand a schema over, since the
collector's projection refuses an invalid contract. Operation/source
records can provide `label`, `description`, `composite` metadata and
`references`, each with a `label` and optional HTTP(S)/relative `url` and
`sha256`. In the Orrery document the collector makes, groups contain their
members and every dependency edge is kept; a field is contained by the
operation whose artifact provides it, so folding an operation folds its
versioned fields and folding a group folds both. The collector's projection
uses the scope's `id` in the document id (`["microcosm", country, scope id]`)
and its label in the default title, so spine, dense and national snapshots
of one country stay distinguishable.

Scope boundaries name an `operation` and `kind`, with `upstream` references
requiring both URL and digest, or a `missing` reason. UK inputs identify raw
spine, full dense, checkpoint dense and national scopes. UK groups are an
explicit roster (`uk_runtime.orrery_contract.UK_GROUPS`): sources, enrichment
(spine stages with their ownership helpers), population pool, geography,
targets, calibration, checks and export. An operation that is not rostered
lands in an "other" group, and the maintained declarations are tested to have
none. Composite descriptions come from the maintained UK operation inventory; a
coupled fit/draw stays one operation and one cache boundary.

Checkpoint provenance is a small shared record. A producer writes
`graph_declaration`, `graph_manifest` and optionally `graph_schema` and
`graph_execution_evidence` entries, each with `path` and `sha256`, into its
checkpoint sidecar; `microcosm.graph.checkpoint_references` turns them into
upstream references. Only files whose bytes still match their digest are
linked; an absent file is named in the `missing` reason together with its
recorded digest, never linked by a local path. The UK spine driver writes all
four entries. `microcosm.graph.save_graph_schema` copies linked local bytes
into `upstream-<sha256>.json` beside `graph.schema.json` and relinks them
relatively, so the schema and the recorded bytes it links move together, and
a UK build hands them to the collector as `orrery_upstream` files. Neither
the schema nor the overlay reconstructs an earlier spine from current source
code or matching field names.

## Recorded execution

Capture evidence immediately after execution, while the executor's run-end
source identities are available:

```python
from microcosm.graph import save_run_evidence

index = save_run_evidence(
    compiled, manifest, store=store, directory=Path("attempt-evidence"),
    attempt_id="my-attempt", phase="numerical",
    artifact_summaries=summary_providers,
)
```

`summary_providers` is optional. It maps `(artifact_type_name, schema_version)`
to a callback accepting `ArtifactSummaryContext` and returning JSON aggregates.
`microcosm.calibrate.graph_evidence.CALIBRATION_SUMMARY_PROVIDERS` uses maintained
problem, solution and result codecs. The UK adapter
(`uk_runtime.orrery_contract.UK_SUMMARY_PROVIDERS`) extends that registry with
recorded target uncertainty and size-search/draw/refit summaries. Shared
evidence code does no country lookup. Native summaries are captured once per
artifact and reused across later phases.

Providers are trusted code: they must emit aggregates and omit record-level
values. During a build, evidence capture is diagnostic: a provider that raises
is recorded as `provider_failed` with its error beside the artifact binding
and the build continues. `collect_execution_evidence`, when it is given an
`artifact_summaries` registry, runs providers strictly and refuses on a
failure; the UK graph step and the inputs command pass none and use the
summaries recorded during the build.

The shared calibration summaries carry each target's initial and achieved
values, residual and relative error, and the weight distribution's counts,
mass, extremes, quantiles and effective sample size. The UK registry drops
every target row's model figures and every validation check's failure lines:
a UK calibration summary keeps each row's target, tolerance, source, family
and recorded uncertainty, and a UK validation-check summary keeps each
check's identifier, phase, criticality, status, reason and number of failure
lines, with enforcement or blocking verdicts in its overview. The UK graph
step refuses to hand over summaries that carry either
([UK graph inputs](orrery-publication.md)). No summary contains entity axes,
record values, per-record weights, draw masks or fitted model bytes. An
overlay from licensed data, or a document made from it, needs the same output
clearance as the calibration diagnostics it summarises before it is posted
anywhere public.

`save_run_evidence` appends an `execution.evidence.json` index with protocol
`microcosm.graph.execution-input.v1`, plus exact graph, manifest, binding and
summary files. `publish_run_evidence(index, out_dir)` flattens that durable
attempt bundle into one publisher directory as `evidence-<attempt>-<phase
identity>-<file>.json` and merges the index with attempts published there
earlier, after re-checking every published file's digest. A repeated or resumed
build into the same output directory therefore keeps the record of the attempt
that computed a result next to the replay that reused it; a published file
that no longer verifies refuses publication instead of being dropped.
`microcosm.graph.run-binding.v1` records the graph/manifest byte digests,
manifest key, attempt, phase, platform and identities of used sources. The
overlay is collected from these native files; they are not submitted to the
collector. Native evidence files are bounded by bytes (1 GiB) and nesting
depth; the compiler schema's per-item budget applies to schemas and overlays,
not to copied manifests, which hold over a million JSON items on a licensed
spine. The byte cap is set by the largest file a real build writes — the
K=25/60k dense build's numerical and preflight graphs are 399 MiB (41.4M
items, 2.2 GiB peak RSS to parse) — with about 2.5x headroom. Digest checks
read bytes without parsing, so hashing a graph that size never pays the parse
cost. A serialized historical manifest without this binding cannot supply
missing run-end source identities by reading today's source files.

### Execution overlay file

Collect the overlay from an existing bundle without executing kernels, and
save it as its own file:

```python
import json
from pathlib import Path

from microcosm.graph import ContentStore, collect_execution_evidence, load_run_evidence
from microcosm.graph.execution_overlay import (
    execution_overlay_bytes,
    load_execution_overlay,
)

schema = json.loads(Path("build/graph.schema.json").read_bytes())
store = ContentStore(Path("build/.graph-store"), create=False)
runs = load_run_evidence(
    Path("build/execution.evidence.json"), store=store,
    reference_base=Path("build"),
)
execution = collect_execution_evidence(schema, runs=runs, store=store)
Path("build/graph.execution.json").write_bytes(execution_overlay_bytes(execution))

# Later, in a process without the store, numpy, pandas or microcosm-frame:
overlay = load_execution_overlay(Path("build/graph.execution.json"))
```

The overlay is a `microcosm.graph.execution-evidence.v1` object. It holds the
schema's `graph_sha256`; one record per supplied phase, with its attempt,
graph and manifest digests, manifest key, timestamps, source identities,
operation states, artifacts with their per-file payload digests, the keys of
its summaries and its references; each summary once; and `verification` and
`history` fields saying what was checked and that only supplied phases are
recorded. The collector's projection embeds it at `metadata.execution`.

The file is the overlay's canonical JSON: UTF-8, sorted keys and no
whitespace. It has none of the transport tags for numbers JavaScript cannot
represent; the collector's projection adds those. Reading it back gives an
equal object, so the collector projects the same document from the file as
from the in-memory overlay. The reader takes only a regular file within the
schema's 32 MiB serialized limit, which evidence collection also applies to
the overlay. It refuses duplicate keys, non-finite numbers, values outside
the schema's JSON bounds and any other protocol. The overlay's graph binding
and records are checked by the collector's projection. Its native-evidence
links are relative to the directory it was collected for (`reference_base`).

`collect_execution_evidence` recompiles each recorded graph, compares
declarations and dependency sets to the schema's graph, recomputes
receipt/artifact identities using recorded inputs and platform, and checks
required store payload digests. Graph/manifest mismatches, missing required
bytes and corrupt files refuse collection. Outputs a validation check left
unavailable, omitted operations and unconfigured summary providers are
explicit. Earlier phases not supplied remain unknown.

### Country wiring

Other countries supply the same three inputs: a presentation dictionary (groups,
scope, optional labels and references), an artifact-summary registry keyed by
artifact type, and driver wiring that calls `save_graph_schema`,
`save_run_evidence`/`publish_run_evidence` and writes the checkpoint provenance
record. No shared module changes are needed.

UK full/national drivers save `graph.schema.json`, `execution.evidence.json`
and flat `evidence-*.json` files beside `graph.json` and `operations.json`.
Durable copies also remain in the content store's attempt directory if a
later phase interrupts the build. The UK graph step writes the overlay as
`graph.execution.json` beside the schema and registers the schema, the
overlay, the native evidence and the linked upstream files as build outputs
([UK graph inputs](orrery-publication.md)).

The final national schema, declaration and operation inventory
(`operations.json`) all include the appended `uk.full.national.readback`
operation. Earlier endpoint runs remain separate phases of the same attempt.
A final cache hit does not erase earlier computation.

## Writing inputs for a saved build

`python -m microcosm.build.uk_runtime.orrery_inputs` writes the collector's
inputs for one UK build into a new directory from the build's saved files,
without executing a kernel. It is for a build whose files exist only on one
machine, such as one staged only locally or interrupted:

```sh
uv run python -m microcosm.build.uk_runtime.orrery_inputs \
  --schema path/to/build/graph.schema.json \
  --evidence path/to/build/execution.evidence.json \
  --store path/to/build/.graph-store \
  --output path/to/new-inputs
```

- **`--schema`:** the build's `graph.schema.json`, written to the output
  unchanged. The command refuses a symbolic link, a device, a file over
  32 MiB, duplicate JSON keys, non-finite numbers and a schema that
  `validate_graph_schema` or `validate_presentation` rejects.
- **`--evidence` and `--store`:** given together or not at all. They are the
  build's `execution.evidence.json` and its content store (`<out>/.graph-store`
  unless the build was given `--graph-store`). With them, the command
  collects the overlay the build would have written and writes it as
  `graph.execution.json`. It refuses summaries that carry a target row's
  figures or a check's failure lines, as the build's graph step does.
- **Upstream files:** each `upstream-<sha256>.json` file that the schema's
  extensions or the overlay link is read from beside the schema and copied
  unchanged. The command refuses one that is missing or whose bytes no
  longer match the digest in its name.
- **`--output`:** a new directory. It must not exist, either as given or
  once resolved. The files are written into a hidden sibling directory, which
  is moved into place only once every file is written, and the command prints
  the resolved path and the names it wrote.

For an interrupted build, pass the attempt directory's `graph.schema.json`
and `graph-evidence/execution.evidence.json`, and the same store. The spine
sidecar identifies its native index under the checkpoint directory; use that
directory's `graph.schema.json` and `node-graph` store.

The command uploads nothing and starts no emitter. A later emitter command,
`microcosm-emitter backfill-orrery <dir>`, is planned
to run it and submit the result as an `orrery-inputs` submission. Without
`--evidence`, the directory has no execution overlay, which today's emitter
and collector both require for that submission. The Python function is
`write_orrery_inputs(schema_path, output, *, evidence=None, store=None)`,
which returns the names it wrote.

## Reconstructed declarations

A build from before #1144 recorded its graph declaration, but not the run
binding that evidence collection validates, so its execution cannot be shown.
`microcosm.graph.reconstruction` rebuilds that build's schema from the recorded
declaration alone and marks it as reconstructed:

```python
from microcosm.graph.reconstruction import (
    code_revision,
    reconstructed_schema,
    reconstruction_marker,
)

marker = reconstruction_marker(
    git_commit=recorded_commit,
    git_dirty=False,
    revision_origin="recorded",
    reconstruction_revision=code_revision(__file__),
    inputs=inputs,  # one {name, role, sha256, bytes, verification} per file read
)
schema = reconstructed_schema(declaration_bytes, marker)
```

Only the declaration comes from the build. The schema's `graph` section and
`graph_sha256` reproduce the recorded file byte for byte. Its `compiled`,
`fields` and `input_bindings` sections are computed by the code running the
reconstruction. A reconstruction made at another revision can therefore
differ in those parts, which is why the marker records that revision
separately from the build's.

The marker is the producer extension `microcosm.reconstruction`:

```json
{
  "protocol": "microcosm.graph.reconstruction.v1",
  "reconstructed": true,
  "basis": "declaration_only",
  "source_revision": {
    "git_commit": "<full lowercase commit>",
    "git_dirty": false,
    "origin": "recorded"
  },
  "reconstruction_revision": {
    "git_commit": "<full lowercase commit, or null>",
    "git_dirty": false
  },
  "inputs": [
    {"name": "graph.json", "role": "declaration", "sha256": "<hex>",
     "bytes": 4096, "verification": "no_recorded_digest"},
    {"name": "rowwise_candidate_manifest.json", "role": "code_pin",
     "sha256": "<hex>", "bytes": 2048, "verification": "recorded_digest_matched"}
  ],
  "scope": "Rebuilt after the build from its recorded graph declaration, the only part taken from the build: ..."
}
```

- **`source_revision`:** the code that declared the graph. `origin` is
  `recorded` when the build wrote the revision itself, and `caller_asserted`
  when the build recorded none and the caller supplied one. `git_dirty` is
  `null` when the build did not record it.
- **`reconstruction_revision`:** the code that performed the reconstruction,
  and so computed every section except the declaration. `code_revision(path)`
  reads it from the git checkout that tracks `path` (the reconstructing
  module's own file), not from the current directory, and runs git without
  the repository-selecting variables of `git rev-parse --local-env-vars`
  (`GIT_DIR`, `GIT_WORK_TREE` and the rest), so an inherited one cannot point
  it at another checkout. Both fields are `null`
  when no checkout tracks that file, as for an installed wheel. `git_dirty` is
  `null` alone when git could not check the working tree.
- **`inputs`:** every file the reconstruction read, in name order, by digest
  and size. Exactly one input is the declaration, and its digest equals the
  schema's `graph_sha256`. `verification` compares the file with the digests
  the build itself recorded for it:

  | Value | Meaning |
  | --- | --- |
  | `recorded_digest_matched` | Every recorded digest matched. |
  | `fallback_digest_matched` | The build marked one recorded digest as taken before it rewrote the file. That digest was skipped, and another recorded digest matched. |
  | `only_stale_digest_recorded` | Every recorded digest is marked as taken before the rewrite, so nothing the build recorded checks the file. |
  | `no_recorded_digest` | The build recorded no digest for the file. |

  In the last two cases the file is bound only by the digest in the marker.
- **Exact bytes:** `reconstructed_schema` refuses a declaration that the
  current serializer would not write byte for byte. The schema's `graph`
  section is therefore the recorded file, not a reinterpretation of it. A file
  that is not a graph declaration at all, such as a JSON array, is also refused
  with a `ValueError`.
- **Where it travels:** the marker is saved in `graph.schema.json`, so the
  collector's projection of the saved schema keeps it, and the Orrery
  document shows it at
  `metadata.microcosm.extensions["microcosm.reconstruction"]`.
  `validate_reconstruction(schema)` returns the marker, or `None` for a schema
  that was not reconstructed.
- **No execution claims:** a reconstructed schema has no execution overlay,
  so a document made from it has no activities, artifacts, statuses or
  `metadata.execution`. No operation appears as executed, cached or blocked by
  a validation check.

### UK builds before #1144

UK dense and national builds from before #1144 wrote `graph.json` beside their
outputs, and recorded their code revision in `rowwise_candidate_manifest.json`.
This command writes a reconstructed, marked schema for such a build's output
directory:

```sh
uv run python -m microcosm.build.uk_runtime.orrery_reconstruction \
  --build path/to/past-build \
  --output path/to/reconstructed-schema
```

The declaration is the build's own `graph.json`. It is not rebuilt by running
the UK graph builders at the current revision. The declaration format has not
changed since the UK graph driver, so a recorded file reproduces byte for byte;
one that does not is refused. The schema's compiled sections are computed by
the code running the command, and the marker records that code's revision as
`reconstruction_revision`.

The build directory is only read. The command:

- **Reads:** `graph.json`, which is required, and also `build.json`,
  `rowwise_candidate_manifest.json` and `sha256sums.txt` where present. Links,
  devices and inputs over 32 MiB are refused.
- **Checks recorded digests:** a file whose bytes disagree with a digest the
  build recorded for it is refused. `build.json` binds the candidate manifest,
  and `sha256sums.txt` binds the files it lists. One exception covers the dense
  driver merged with #901 (2026-09-29). It recorded the manifest's digest in
  `build.json` with the note `staging receipts are appended after
  publication`, then appended `staging_delivery` and `staged_dataset` to the
  manifest and refreshed only `sha256sums.txt`. `build.json` was reissued over
  the final bytes, without the note, only from #1057 (2026-09-30). A
  `build.json` digest with that exact note that no longer matches is skipped
  and reported, and the manifest must then match `sha256sums.txt`
  (`fallback_digest_matched`). A build run with `--no-staging` or
  `--no-staged-dataset` wrote no digest list, so its manifest is reported as
  `only_stale_digest_recorded`, with a warning.
- **Resolves the revision:** it takes the candidate manifest's `git_commit`
  and `git_dirty`, which must agree with the copy under `identity.code`.
  `--revision` must match that commit, or supplies one when the build recorded
  none; that revision is marked `caller_asserted`. With neither, it refuses.
- **Refuses later builds:** a build that records Orrery evidence
  (`graph.schema.json` or `execution.evidence.json`) is refused, and the
  refusal points to `python -m microcosm.build.uk_runtime.orrery_inputs`,
  which writes such a build's inputs (see
  [Writing inputs for a saved build](#writing-inputs-for-a-saved-build)).
- **Refuses missing declarations:** a directory without `graph.json` is refused,
  and the refusal names the build records that were found. Staged bundles never
  carried `graph.json`, and builds from before the UK graph driver never wrote
  it.

It writes the schema and its report into a new directory, and moves the
directory into place only after both files are written. The output must not
exist, either as given (a symbolic link counts, even one whose target does not
exist) or once resolved. It must not be the build directory or lie inside it
once `..` and links are resolved. The command writes to that resolved path, so
a path that passes through a missing folder inside the build creates nothing
there, and a read-only build can be reconstructed. It prints the resolved path,
because the path as given need not reach it:

| File | Contents |
| --- | --- |
| `graph.schema.json` | Compiler schema of the recorded declaration, with the reconstruction marker |
| `orrery.reconstruction.json` | Summary report, protocol `microcosm.uk.orrery-reconstruction-report.v2` |

The summary report, of which the command prints a short version, has:

- `status` (`reconstructed`), `basis` (`declaration_only`) and `files`, the
  files the command wrote besides the report (`["graph.schema.json"]`);
- the source revision and the reconstruction revision;
- the build's identity, and the declaration's name, country, `graph_sha256`
  and numbers of operations and sources;
- the inputs and how each was verified;
- the recorded digests skipped as stale, each with its note and the records the
  file was checked against instead;
- the expected files that are missing;
- evidence found but not bound to operations: run manifests,
  `evidence-index.json`, `operations.json` and the content store;
- warnings, for a dirty or unrecorded working tree in the build or the
  reconstruction, an unknown reconstruction revision, a caller-asserted
  revision, a skipped stale digest, or a declaration the build recorded no
  digest for. Before #1144, the last one applies to every build.

Version 2 of the report drops version 1's `publication_id` and
`published_files`. The Python function is
`reconstruct_uk_schema(build, output, *, revision=None)`, which returns the
report.

Output is deterministic: reconstructing the same build twice from the same
code revision gives identical bytes. The command uploads nothing and starts
no emitter. The collector can make the Orrery document from the schema as it
does from a build's, but the schema has no execution overlay, which today's
emitter and collector both require for an `orrery-inputs` submission.
Submitting it therefore waits for planned collector work, to accept a marked
schema without an overlay and record it as reconstructed, and for the
planned emitter command.

## Evidence and limits

A schema contains static declarations only. It does not contain entity IDs,
memberships, field values, source bytes, kernel code, observed runtime reads,
cache results, execution receipts, verifier assessments, or release
decisions. The execution overlay adds the recorded aggregates above. It
records which keys and byte digests were checked; that does not establish
signatures or release certification, and the overlay carries no Orrery
cryptographic `Receipt` verification verdicts.

A schema, and an overlay written or read back, may hold at most 2,000,000
JSON items nested at most 64 deep, with strings of at most 131,072
characters, integers of at most 1,024 bits, finite numbers only, string
object keys only and a canonical serialized size of at most 32 MiB. Anything
else is refused rather than silently shortened. Native evidence files have
their own bounds (see [Recorded execution](#recorded-execution)). The Orrery
document's own limits belong to the collector's projection.

Runtime evidence comes only from supplied records. Precise per-output value
lineage cannot be inferred from static reads or operation ancestry; it
remains #865.
