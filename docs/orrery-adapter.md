# Orrery adapter

`microcosm.graph.orrery` converts a Microcosm graph declaration and its
compiler-derived metadata into `graph-explorer/v1`, the JSON format accepted by
[Orrery](https://github.com/TheAxiomFoundation/orrery). Microcosm owns the
calculation and field semantics. Orrery owns graph navigation, presentation,
and self-contained HTML export.

This conversion is separate from graph execution:

1. `compile_graph(graph)` validates a `Graph` and derives operation order,
   population versions, field owners, and operation predecessors. The executor
   uses this `CompiledGraph`.
2. `graph_schema(compiled)` derives a portable static metadata artifact from
   that same `CompiledGraph`.
3. `orrery_json(compiled)` converts the static metadata into Orrery input. It
   does not execute the graph.
4. The separately installed Orrery command can turn that JSON into a
   self-contained HTML file.

The schema is therefore a generated artifact, not an instruction that
Microcosm executes. The direct API performs steps 2 and 3 in memory. A saved
schema lets another process repeat step 3 without importing the original graph
construction code.

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
package and a locked set of transitive dependencies.

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
declaration remains in `data`. The entire compiler schema remains available at
`metadata.microcosm`.

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

The export contains static declarations only. It does not contain entity IDs,
memberships, field values, source bytes, kernel code, observed runtime reads,
cache results, execution receipts, verifier assessments, or release decisions.
Content revisions identify metadata bytes; they are not execution keys or
authenticity claims.

Python integers outside JavaScript's safe range are represented as
`{"integer_literal": "..."}` before Orrery parses the document. Large integral
floats use `{"float_literal": "..."}`. Exports reject non-finite numbers,
excessive nesting, non-JSON objects, more than 20,000 presentation nodes or
100,000 relationships, input above 32 MiB, and output above 64 MiB. They fail
instead of silently omitting records.

Runtime receipts and precise per-output value lineage could be added by a
separate runtime-aware adapter in the future. They cannot be inferred from the
static compiler schema and are intentionally absent here.
