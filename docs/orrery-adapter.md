# Orrery adapter

`microcosm.graph.orrery` converts a Microcosm graph declaration and its
compiler-derived metadata, with optional recorded execution evidence, into
`graph-explorer/v1`, the JSON format accepted by
[Orrery](https://github.com/TheAxiomFoundation/orrery). Microcosm owns the
calculation and field semantics. Orrery owns graph navigation, presentation,
and self-contained HTML export.

This conversion is separate from graph execution:

1. `compile_graph(graph)` validates a `Graph` and derives operation order,
   population versions, field owners, and operation predecessors. The executor
   uses this `CompiledGraph`.
2. `graph_schema(compiled)` derives a portable static metadata artifact from
   that same `CompiledGraph`.
3. `orrery_json(compiled, execution=...)` converts the metadata and any supplied
   evidence into Orrery input. Omitting `execution` keeps declaration-only export.
4. The separately installed Orrery command can turn that JSON into a
   self-contained HTML file.

The schema is therefore a generated artifact, not an instruction that
Microcosm executes. The direct API performs steps 2 and 3 in memory. A saved
schema lets another process repeat step 3 without importing the original graph
construction code.

A kernel is the implementation of one graph operation, with a declared input,
output, numerical and execution contract.

## Export from Python

Use the public API when the `Graph` or `CompiledGraph` is already available:

```python
from pathlib import Path

from microcosm.graph import compile_graph, orrery_json

compiled = compile_graph(graph)
Path("orrery.json").write_text(
    orrery_json(compiled, title="Microcosm UK"),
    encoding="utf-8",
)
```

Passing the original `Graph` to `orrery_json` produces the same bytes because
the function compiles it first. `orrery_document` returns the same result as
detached Python dictionaries and lists.

To persist the intermediate compiler schema, serialize `graph_schema(compiled)`
with `microcosm.graph.canonical.canonical_json`. The schema has exactly these
root fields:

```text
protocol, country, graph_sha256, graph, compiled,
fields, input_bindings, extensions
```

Only `extensions` is producer-defined. Before presentation,
`validate_graph_schema` restores and recompiles the embedded graph and requires
every other field to equal the newly derived result. A caller cannot change an
owner, predecessor, population version, field provider, or input binding while
retaining a valid schema.

## Export from the command line

The command requires an explicit input type and creates a new output file:

```sh
uv run python -m microcosm.graph.orrery \
  --graph graph.json \
  --output orrery.json \
  --title "Microcosm UK"
```

For a previously saved compiler schema, use `--schema` instead of `--graph`:

```sh
uv run python -m microcosm.graph.orrery \
  --schema compiled-schema.json \
  --output orrery.json
```

The command does not infer the input type. It rejects duplicate JSON keys,
non-finite numbers, streams and devices, oversized input, an invalid graph or
schema, and an existing output path. It does not install Orrery, download
assets, or execute a population operation.

Install the viewer independently, then create the HTML file:

```sh
npm install --save-dev @axiom-foundation/orrery
npx orrery --input orrery.json --output orrery.html
```

Orrery validates the JSON before export. Its HTML contains the viewer assets
and does not require a network connection to render. Microcosm CI parses a
generated document with the compatible range declared by its contract-test
package and a locked set of transitive dependencies. The consumer contract is
`@axiom-foundation/orrery >=0.6.1 <0.7.0`, pinned by
`tools/orrery-contract/package-lock.json` for CI.

## Country inputs and presentation

The schema and exporter retain the interfaces introduced by #888. The graph
schema remains `microcosm.graph.schema.v1`; compiler facts still come from the
same compiler. Country is descriptive metadata. Shared export code neither
imports a country builder nor dispatches on country names or operation prefixes.

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
group. Groups use Orrery containment and preserve every dependency edge. A field
node is contained by the operation whose artifact provides it, so folding an
operation folds its versioned fields and folding a group folds both. The
validator rejects unknown members, duplicate membership and containment cycles.
Operation/source records can provide `label`, `description`, `composite`
metadata and `references`, each with a `label` and optional HTTP(S)/relative
`url` and `sha256`. When the scope has an `id`, the Orrery document id becomes
`["microcosm", country, scope id]` and the default title uses the scope label,
so spine, dense and national snapshots of one country stay distinguishable in a
host's revision picker and in baseline comparisons.

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
checkpoint sidecar; `microcosm.graph.checkpoint_references` projects them into
upstream references. Only files whose bytes still match their digest are
linked; an absent file is named in the `missing` reason together with its
recorded digest, never linked by a local path. The UK spine driver writes all
four entries. `microcosm.graph.save_graph_schema` copies linked local bytes
into `upstream-<sha256>.json` beside `graph.schema.json` and relinks them
relatively, so a published build and a later HTML export travel together. The
exporter never reconstructs an earlier spine from current source code or
matching field names.

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
problem, solution and result codecs. The UK adapter extends that registry with
recorded target uncertainty and size-search/draw/refit summaries. No country
lookup occurs in the shared exporter. Native summaries are captured once per
artifact and reused across later phases.

Providers are trusted projections: they must emit aggregates and omit
record-level values. During a build, evidence capture is diagnostic: a provider
that raises is recorded as `provider_failed` with its error beside the artifact
binding and the build continues; the explicit export command runs providers
strictly and refuses on a failure.

An execution export from a licensed run carries the achieved value of every
target, local-area targets included, and the weight distribution's extremes and
quantiles. Inside the private build bundle that is fine; before an execution
snapshot or its HTML is posted anywhere public it needs the same output
clearance as the calibration diagnostics it summarises.

`save_run_evidence` appends an `execution.evidence.json` index with protocol
`microcosm.graph.execution-input.v1`, plus exact graph, manifest, binding and
summary files. `publish_run_evidence(index, out_dir)` flattens that durable
attempt bundle into one publisher directory as `evidence-<attempt>-<phase
identity>-<file>.json` and merges the index with attempts published there
earlier, after re-checking every published file's digest. A repeated or resumed
build into the same output directory therefore keeps the record of the attempt
that computed a result next to the replay that reused it; a published file
that no longer verifies refuses publication instead of being dropped. `microcosm.graph.run-binding.v1` records the graph/manifest byte
digests, manifest key, attempt, phase, platform and identities of used sources.
These native files support later export; they do not create an Orrery snapshot. Native evidence files are bounded by bytes (1 GiB) and nesting depth; the compiler schema's per-item budget applies to schemas and Orrery documents, not to copied manifests, which hold over a million JSON items on a licensed spine. Digest checks read bytes without parsing.
A serialized historical manifest without this binding cannot supply missing
run-end source identities by reading today's source files.

Read and export an existing bundle without executing kernels:

```python
from microcosm.graph import ContentStore, collect_execution_evidence, load_run_evidence
from microcosm.graph.orrery import orrery_json_from_schema

store = ContentStore(Path("build/.graph-store"), create=False)
runs = load_run_evidence(
    Path("build/execution.evidence.json"), store=store,
    reference_base=Path("build"),
)
execution = collect_execution_evidence(schema, runs=runs, store=store)
Path("build/graph.orrery.json").write_text(
    orrery_json_from_schema(schema, execution=execution), encoding="utf-8"
)
```

The equivalent command keeps the existing input/output structure:

```sh
uv run python -m microcosm.graph.orrery \
  --schema build/graph.schema.json \
  --evidence build/execution.evidence.json \
  --store build/.graph-store \
  --output build/graph.orrery.json
npx orrery --input build/graph.orrery.json --output build/graph.orrery.html
```

`--evidence` and `--store` must appear together.
Native evidence links and saved-schema relative references are rebased to the
command's output directory; keep the referenced files available with the HTML.

Other countries supply the same three inputs: a presentation dictionary (groups,
scope, optional labels and references), an artifact-summary registry keyed by
artifact type, and driver wiring that calls `save_graph_schema`,
`save_run_evidence`/`publish_run_evidence` and writes the checkpoint provenance
record. No shared module changes are needed.
UK full/national drivers save
`graph.schema.json`, `execution.evidence.json` and flat `evidence-*.json` files
beside `graph.json` and `operations.json`. Durable copies also remain in the
content store's attempt directory if a later phase interrupts the build. For
that case, pass the attempt's `graph.schema.json` and
`graph-evidence/execution.evidence.json`, and the same store. The spine sidecar
identifies its native index under the checkpoint directory; use that directory's
`graph.schema.json` and `node-graph` store. Orrery JSON and HTML remain opt-in.

The final national schema, declaration and inventory all include the appended
`uk.full.national.readback` operation. Earlier endpoint runs remain separate
phases of the same attempt. A final cache hit does not erase earlier computation.

Validation recompiles each recorded graph, compares declarations and dependency
sets to the exported graph, recomputes receipt/artifact identities using recorded
inputs and platform, and checks required store payload digests. Graph/manifest
mismatches, missing required bytes and corrupt files refuse export. Unavailable
gate-blocked outputs, omitted operations and unconfigured summary providers are
explicit. Earlier phases not supplied to the exporter remain unknown.

## Orrery evidence panels

| Surface | Exported evidence |
| --- | --- |
| Record | Operation declarations, attempt/phase history with kernel role, duration, numerical keys, mass and weight transitions, artifact keys, calibration/size aggregates and per-gate outcomes |
| Sources | Presentation references and recorded checkpoint references |
| Activity | One record per operation and supplied phase; the kernel (`name@version`, implementation hash) as the agent; declared operation/source inputs; artifact references including the digest-bound native graph/manifest/binding/summary files; phase timestamps |
| Status badges | Gate verdict first on gate kernels only, then execution state, then an attempt-level cache summary |

Orrery cards show the first two badges. A gate kernel therefore leads with
`Gate: pass`/`fail`/`evidence_absent`/`unreached`; other operations carry no
gate badge. The cache badge summarises every supplied phase: `Computed:
numerical · reused 2×` names the phase that computed the result before counting
replays, `Reused: 3 cache hits; computation not in supplied phases` is explicit
when no supplied phase computed it, and `Cache: unknown` means no receipt.

Artifact references are one per content-store object (frame, column, weights,
typed or opaque artifact), labelled by producing operation, output name, kind
and file count. A single-file object carries its byte digest; the per-file
payload digests of every object are in `metadata.execution.phases[].artifacts`.
This keeps a structural operation's activity to tens of references rather than
one row per column file.

UK gate batteries (spine, full and national gate reports) are summarised per
gate: identifier, phase, criticality, status, failure lines and reason, with
enforcement or blocking verdicts in the overview. Gate detail payloads and
target or area tables stay in the artifact.

Per-target tables contain target, initial and achieved values, residuals,
relative error, tolerance and recorded uncertainty where available. Full tables
appear once in `metadata.execution.summaries`; selected-node Record data shows
an explicit 50-row preview with `total_rows` and `preview_limit`. Weight summaries
contain counts, mass, quantiles and effective sample size. They contain no entity
axes, record values, per-record weights, draw masks or fitted model bytes.

Orrery's existing panels can present these records. A dedicated calibration table,
phase selector, linked upstream-run expansion and group-focused layout controls
remain viewer follow-ups; the exporter does not add those UI features. Precise
per-output mathematical lineage remains #865.

Orrery 0.6.1's Activity panel includes downstream activities that received the
selected record as an input. This can produce a long list for structural nodes;
the artifact labels include their producing operation and output name so they
remain identifiable. Browsing a synthetic saved national run confirmed groups,
Record fields, separate status badges, Sources links and phase activities.

## Large snapshots and deep links

Field nodes are the bulk of a UK snapshot: the maintained dense declaration has
76 operations, 15 sources and about 5,300 field nodes with 20,000
relationships. Orrery lays out every visible node, so the whole-graph view of
such a snapshot takes a minute to appear and is unreadable until filtered. The
offline HTML honours Orrery's URL hash, so publish or open it with a location
that starts from the operations:

```text
graph.orrery.html#kinds=["operation","source","group"]
graph.orrery.html#kinds=["operation","source","group"]&focusId=["operation","uk.full.dense"]&depth=1
graph.orrery.html#collapsedIds=["[\"group\",\"enrichment\"]"]
```

Fields remain searchable and inspectable from the index, and expanding one
operation shows exactly its versioned fields because fields are contained by
their provider. Every serialized string in the compiler schema and the export,
operation contracts included, may be up to 131,072 characters; the overall
document, complexity, node, and edge limits remain unchanged. These are declaration and synthetic diagnostic measurements, not a
licensed population run.

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
physical materialization and the ownership declaration in the presentation.

These records describe values supplied to an operation. They do not claim that
a kernel read every supplied value at runtime, and they do not reproduce the
executor's complete causal-writer history, which may be refined by runtime
receipts.

## Orrery document structure

The adapter creates operation, source, and versioned-field nodes. Descriptions
are promoted to Orrery's visible `description` property while the complete
declaration remains in `data`. `metadata.microcosm` is a bounded summary of the
compiler schema (protocol, country, graph and canonical schema digests,
compiled order, record counts and the producer extensions); the complete schema
is the separately saved `graph.schema.json` whose bytes have
`metadata.microcosm.schema_sha256`. Orrery renders snapshot metadata as text,
so the viewer document does not repeat declarations that its records already
carry.

| Relationship | Category | Meaning |
| --- | --- | --- |
| `compiled_predecessor` | dependency | An operation dependency derived by `compile_graph` |
| `declared_read` | dependency | A slice, mask, or rewrite-incumbent input binding |
| `artifact_input` | dependency | A typed producer-to-consumer artifact declaration |
| `provided_field` | provenance | The operation that supplies a versioned field |
| `structural_input` | provenance | A base population field carried into a structural operation |
| `declared_source` | provenance | A named external input declaration |

Structural carriage does not assert that values are unchanged. Declared reads
do not infer an individual mathematical formula for each output. Artifact and
source declarations do not establish that the referenced bytes exist or were
validated.

Pre-rewrite values receive auxiliary field nodes when necessary. Stable IDs
include the population, coordinate, provider, and declaration source, so the
pre-rewrite value and final rewritten value cannot collide.

## Evidence and limits

Declaration-only export contains static declarations. It does not contain entity IDs,
memberships, field values, source bytes, kernel code, observed runtime reads,
cache results, execution receipts, verifier assessments, or release decisions.
Content revisions identify metadata bytes; they are not execution keys or
authenticity claims. Execution export adds the recorded aggregates above, but
does not emit Orrery cryptographic `Receipt` verification verdicts. Checking keys
and byte digests does not establish signatures or release certification.

Python integers outside JavaScript's safe range are represented as
`{"integer_literal": "..."}` before Orrery parses the document. Large integral
floats use `{"float_literal": "..."}`. Exports reject non-finite numbers,
excessive nesting, non-JSON objects, more than 20,000 presentation nodes or
100,000 relationships, input above 32 MiB, and output above 64 MiB. They fail
instead of silently omitting records.

Runtime evidence comes only from supplied records. Precise per-output value
lineage cannot be inferred from static reads or operation ancestry.
