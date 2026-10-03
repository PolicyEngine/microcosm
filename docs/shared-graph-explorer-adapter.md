# Shared graph explorer adapter

`microcosm.graph.explorer` converts a `microcosm.graph.schema.v1` metadata
export into `graph-explorer/v1`, the portable contract consumed by [Orrery](https://github.com/TheAxiomFoundation/orrery), the
shared graph viewer. Microcosm owns the graph's calculation
and schema meanings. The shared package owns navigation, rendering and bundled
offline HTML.

The adapter adds no JavaScript dependency to Microcosm. It reads no microdata,
executes no population operation, and leaves `explain_html` unchanged.

## Open a saved schema in Orrery

From a synced Microcosm checkout, export a saved schema with:

```sh
uv run python -m microcosm.graph.explorer \
  --input compiled-schema.json --output graph.json --title "Microcosm US"
graph-explorer --input graph.json --output graph.html
```

The second command uses the separately installed, accepted shared viewer
0.3.0 CLI, whose package name is still `@axiom-foundation/graph-explorer`.
The forthcoming Orrery package rename does not change the JSON contract.
Microcosm does not install or upgrade the viewer as part of export.

Open `graph.html` using a local static server or your existing artifact viewer.
The HTML contains its viewer assets and needs no CDN. Direct `file://` opening
is outside the current Microcosm browser acceptance.

The Python command accepts only saved schema JSON, rejects duplicate keys and
non-finite numbers, bounds input before decoding, and creates a new output file.
It refuses an existing output, including the input path. A saved manifest or
microdata file is not a schema export. Review schema metadata before sharing it:
the entire supplied metadata is retained, and canvas filtering does not redact it.

The same exporter is available as a Python API:

```python
import json
from pathlib import Path

from microcosm.graph.explorer import graph_explorer_json

schema = json.loads(Path("compiled-schema.json").read_text(encoding="utf-8"))
Path("graph.json").write_text(graph_explorer_json(schema), encoding="utf-8")
```

The schema producer is being integrated separately. In a checkout containing
`microcosm.graph.schema.graph_schema`, the input can be produced directly with
`graph_schema(compiled)`. This adapter also accepts previously saved exports;
it does not require that producer to be installed. `graph_explorer_document`
returns the same document as detached Python dictionaries and lists.

The shared package's built CLI accepts:

```sh
graph-explorer --input graph.json --output graph.html
```

The shared package must supply its built viewer assets. Microcosm does not
download them or launch a build automatically. HTML export and browser
verification are distinct checks; a successfully written HTML file alone does
not establish that its embedded viewer runs.

## What the graph contains

The document contains every operation, source declaration, visible field version
and input binding in the supplied schema. A field identity contains its
population, entity, column, producer and nearest declaration. A pre-rewrite
input therefore remains distinct from the final value in the same population.
Population coordinates are retained in `data`; they are not replaced by a
presentation revision or containment parent.

| Edge kind | Meaning |
| --- | --- |
| `compiled_predecessor` | An operation dependency listed in the supplied compiler metadata |
| `produced` | The provider of a versioned field value, including a structural carrier |
| `declared_read` | An operation's slice, slice mask, output mask or rewrite incumbent; role and row mask are retained |
| `structural_input` | Ancestry from a completed base population's fields into its structural successor |
| `source` | A named external input and its declared codec |
| `artifact` | A producer-to-consumer dependency with its exact alias, artifact name and nominal type/version |

Structural ancestry does not imply unchanged values. Declared reads describe
operation-level dependencies; they do not infer a separate mathematical formula
for each output. Typed artifact declarations do not establish the existence of
runtime artifact bytes. Source declarations remain domain nodes; citation
references cannot substitute for their codec contract.

The complete original metadata stays in `metadata.microcosm`, including compiled
owners/order/versions and declaration details such as `mass_partition`. Exact
Python integers outside JavaScript's safe range are transported as
`{"integer_literal": "..."}` before JavaScript parses the document. Large integral
floats use `{"float_literal": "..."}` so their type is not silently changed to
integer. The raw declaration digest is retained separately from presentation
revisions.

## Scope, identity and evidence

`complete_supplied_schema` means the complete supplied metadata snapshot. It
does not mean the complete US recipe. Entity IDs and memberships, scientific
units, period semantics, source preparation internals and implicit runtime reads
cannot be inferred when the source schema omits them. Declared ownership is
not evidence that a field contains valid materialized values.

The converter checks JSON bounds, declaration digest consistency, references,
field/declaration agreement and exact read-role coverage. It does not recompile
or authenticate an imported dictionary. The document revision hashes the whole
supplied schema, including its compiler tables; node and edge revisions hash
their transport records before the revision field is attached. These are
metadata content digests, not execution keys or source-byte attestations.

No execution, authorship, cache, gate or release status is inferred from schema
metadata. This first adapter supplies no runtime activities or Receipt verdicts.
The existing deterministic run viewer retains its independent cache/gate axes.
A later run adapter must bind actual run evidence and keep these statuses
separate. Receipt assessments must come from a configured host verifier and
bind to the exact exported document bytes, not merely the source declaration's
hash. Custody verification does not establish scientific correctness.

Exports fail explicitly at their development bounds rather than silently
truncating: 256 operations, 256 sources, 20,000 presentation nodes, 100,000 edges,
32 MiB input and 64 MiB output. Prospective transport bytes are charged as
records are added. These are metadata limits, not microdata size limits or a
claim about total Python process memory. The viewer can focus or collapse a
complete document without changing the underlying export.
