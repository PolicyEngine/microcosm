# Pre-calibration stages

A transport graph's calibration may read only what the calibration ancestry
guard admits (`validate_transport_calibration_ancestry` in
`microcosm.build.transport.compose`). Without stages, the guard refuses
benefit-unit engine semantics upstream of a `calibrate.*` node or the
calibration endpoint, hold-out sources, and `holdout_references` selections
and the graph resource's own digest in the `transport_graph` rows its
declaration pass reaches through population, base and artifact links. It does
not follow column reads, and its composed pass checks no selections.
Person-entity rules bindings are admitted.

The optional `precalibration_stages` resource declares the rules bindings whose
outputs feed calibration, and exactly which of their outputs flow on:

```json
{
  "schema_version": 1,
  "stages": [
    {
      "id": "nz.precal.main_benefit_rates",
      "rules_binding": "main_benefits_rates",
      "rulespec_path": "<the binding's rulespec_path>",
      "outputs": [
        {"entity": "person", "column": "<binding variable>", "purpose": "calibration_measure"}
      ],
      "derived_outputs": [
        {"entity": "person", "column": "<stage node output>", "purpose": "calibration_measure"}
      ],
      "citation": "<method card row or ruling>"
    }
  ]
}
```

## Validation

Key sets are closed at every level, and `schema_version` is the integer 1.
Stage ids are distinct, safe node ids. Each stage names an existing binding,
repeats its `rulespec_path` exactly, and no binding appears in two stages.
A stage declares at least one output or derived output, and they are all
distinct. Each output is on the binding's entity, names one of its
`variables`, and has a `purpose` of `calibration_measure` or
`classification` (a judgment another pre-calibration node reads). An output may
not be a hold-out `(entity, measure)`, and a stage's module may not produce a
hold-out measure. The resource is checked before either graph pass.

`derived_outputs` is optional. It lists columns that a stage node computes
from the binding but that are not binding variables, such as a gross-up
`simulate.solve_zero@1` node's own output. A stage whose binding variables stay
inside its nodes lists `"outputs": []` and only derived outputs. Each row has
the same closed shape and purposes as an `outputs` row. It names any entity,
may not repeat a binding variable on the binding's entity (that belongs in
`outputs`), and may not repeat any other output of the stage or a hold-out
measure. Each derived output must be produced by one of the stage's nodes
upstream of calibration, which produces the columns it owns. The composed
graph is checked when there is one, otherwise the declaration.

With stages listed, every field the checks read is validated before either
pass uses it, on every declared node and rules binding, except three values
that the checks read as composition does and two EXPAND parameters they read
conservatively, both described below:

- **Nodes.** `id` and `kernel` are text, and ids are distinct. `population`,
  `base` and `rules_binding` are text or null, and `structural`, when
  present, is one of `none`, `create`, `filter`, `expand` and `reweight`.
  `params` is an object. Each resource selection in it names a text
  resource, a list `path` and a known encoding, as activation requires. Each
  binding selection (an object with `binding` and `field`) names both as
  text. `inputs` is a list of objects with a text `entity` and a list of text
  `columns`; `outputs` is a list of objects with a text `entity`, a text
  `column` and, when present, a text `rows` mask (absent is every row);
  `artifact_inputs` is a list of objects with a text `producer`. `sources` is
  a list of text.
- **Rules bindings.** Each names a distinct text `id`, a text `entity` and
  `rulespec_path`, and a list of text `variables`.
- **The calibration endpoint**, when present, is text. An explicit null
  refuses too: it would drop a calibration node whose kernel is not
  `calibrate.*`.
- **Entitlement rows**, in the composed pass. Each row the entitlement
  extension instantiates has a text id that no skeleton row or other
  entitlement row uses, and its `params` follow the rules above.

Three values get no shape check of their own, because the checks read them
exactly as composition does. An output's `rewrite` flag is read by
truthiness, as the compiler reads it. A selection `path` is followed through
mappings only, as composition follows it: a path that leaves the resource's
mappings selects nothing for the checks, and composition refuses it. A
`prepared` value counts only when it is `"engine_refs"`, the one prepared
parameter composition admits.

Two EXPAND parameters are read as the executor reads them when the
declaration spells them out, and otherwise conservatively. An EXPAND's
`expand_cells` names its cells when it is a list of `[entity, column, dtype]`
texts; any other value, such as a resource selection, leaves the cells to
composition, so the declaration pass counts that EXPAND as a writer of every
column read through its version.
A node's `materialized_expand_outputs` claims the cells it lists when it is a
list of `entity.column` texts; any other value claims every output that is not
a rewrite. The composed pass reads both from the compiled parameters exactly
as the executor does, and a malformed one refuses there.

A malformed node field refuses, naming the node: for example "node 'mid' has
malformed inputs", "node 'mid' is declared twice" or "node id None is not
text". A malformed rules binding refuses naming the binding, as in "needs
rules binding 'A' to name its entity, rulespec_path and variables". Read
character by character, a string column, source or path list would otherwise
drop a read, a hold-out source or a selection, and a repeated id would hide a
node. The guard without stages is unchanged.

## Which nodes belong to a stage

In the declaration pass, a node belongs to a stage when its `rules_binding` is
the stage's binding or any `{"binding": id}` selector in its parameters names
it. In the composed pass, a node belongs to a stage when its parameters carry
the binding's prepared engine reference. A node that names a listed and an
unlisted benefit-unit binding refuses. A parameter carrying the whole prepared
`engine_refs` inventory is bookkeeping, not a stage invocation, and still
refuses upstream of calibration.

## Rules

Only nodes in the ancestry of a calibration node are checked. The calibration
nodes are every `calibrate.*` node and the calibration endpoint, in the
declaration and in the composed graph. A node outside the ancestry of every
calibration node, such as a stage binding's extension checkpoint after the
fit, is an output, and these rules leave it alone; the shape checks above
still cover every declared node. A node after the central fit that a variant
calibration reads is in that variant's ancestry, so it is checked.

- Absent, or `"stages": []`, is the guard without stages, outcomes and messages
  alike.
- A listed benefit-unit stage admits its own nodes, their engine reference and
  outputs, and group-encoder outputs read upstream only by listed benefit-unit
  stage nodes. Any other benefit-unit semantic upstream refuses as
  "not a declared pre-calibration stage".
- A listed person stage admits nothing new; it only adds the rules below.
- A stage changes columns, not the population. A structural node (CREATE,
  FILTER, EXPAND or REWEIGHT) that belongs to a listed stage refuses,
  naming the node, its kind and its stage binding. A stage declares only
  columns, but a structural node also writes what every later kernel of its
  version receives without a `Slice`: ids, group memberships, weights,
  strata, the frame metadata and the mass log. No other node writes them, so
  with this rule they never come from a stage's node. A structural node
  outside every stage is checked like any other: its inputs are reads, and
  its parameters, sources and artifact inputs face the same checks. So an
  EXPAND's cells are never a stage's output either; a stage claim that owns
  a cell produces it.
- A node outside a listed stage that reads a column whose values one of the
  stage's own nodes wrote upstream of calibration, other than its `outputs`
  and `derived_outputs`, refuses as "undeclared pre-calibration stage output".
  Reads among a stage's own nodes do not count.
- The rule follows each column read, explicit or through a rewrite or claim,
  to every writer whose values it receives, as the executor tracks them. In
  the reader's population version that is the owner of the column. An owner
  that keeps earlier values passes on the writers below it too: a rewrite
  receives the incumbent, a masked owner (an output with a `rows` mask)
  leaves the incumbent's values on the rows outside its mask, and an EXPAND
  claim receives the EXPAND's values. While the version owns the column
  nowhere, or its owner keeps earlier values, the version's EXPAND counts
  when it installs the column as a cell, and the walk goes on to the base
  version, down to CREATE. A full owner that is not a rewrite or claim stops
  it.
- A structural node reads its inputs from its base. A rewrite reads the
  column it rewrites, and a claim, a node that names an EXPAND cell in its
  `materialized_expand_outputs`, reads that cell, both without a `Slice`.
  Each counts as a read. Another binding that declares the same variable
  name, or another node that owns a column of the same name in another
  version, is therefore unaffected.
- The composed pass resolves writers exactly. Where a declaration leaves the
  version unknown, the declaration pass counts every declared writer of the
  column instead: a reader with no population, a version that is not declared
  and a cycle of bases all fall back this way. A declared owner without a
  population version (a node with no population, such as CREATE, or a
  structural node's own outputs) counts in every version.
- Outputs are columns, so a stage's artifacts are never declared. A node
  outside the stage that reads an artifact one of the stage's nodes produces
  upstream of calibration refuses the same way.
- A parameter that carries the prepared `engine_refs` inventory refuses as
  "prepared engine_refs map", on stage nodes too. A mapping carries it when it
  has an `engine_refs` key or any key or value carries it, and a list when
  any item does. A string carries it when it holds the quoted key
  `"engine_refs"` followed by a colon, as written or once JSON escapes are
  decoded, layer by layer, up to 32 layers; a string still escaped after 32
  layers counts as carrying it. So the inventory may be a JSON string in any
  spelling (braces or the key written as escapes included), nested in
  strings, JSON embedded in a longer string, or the data a resource selection
  names. An encoder that escapes a backslash as `\\` doubles them at each
  layer, but one that writes `\u005c` grows only linearly: 33 layers of the
  quoted key take 344 characters. So the bound is reachable, and text still
  escaped after it refuses whatever it decodes to. Escapes decode one at a
  time, so a surrogate pair stays two code points; the key is ASCII, so this
  does not change what is found. Text that only names engine_refs, without
  the quoted key and colon, is not the inventory; prose that quotes the key
  and a colon refuses like the map itself. Each layer is one linear
  pass over the string, so the check takes at most 33 passes. The inventory
  is bookkeeping, not a stage invocation.
- The hold-out, selection and digest checks still apply to every upstream node,
  stage nodes included, so each stage's ancestry is clean. They read each
  node's declared selections: its `transport_graph` row or, in the composed
  pass, the entitlement row the extension instantiated it from, whose
  parameters are resolved before the graph exists. A code extension's node has
  no declaration, so its compiled parameters get the engine-reference and
  inventory checks but not the selection checks.

Both passes apply these rules; the composed pass covers compiled predecessors
and extension nodes. No node parameter or hash changes.

## Invariants

These are tested, as properties and examples, in
`tests/engine_free/shared/test_transport_composed_graph.py`.

- With no stages, the guard accepts and refuses exactly as the frozen copy of
  the guard without stages does, malformed benefit-unit entities included.
- Person stages only restrict: a graph accepted with only person stages listed
  is accepted without them.
- Dropping a person stage never refuses a graph its listing admitted. While
  another stage stays listed, dropping a benefit-unit stage never admits a
  graph its listing refused, except when the refusal was that stage's own
  declaration error: its schema, a hold-out measure, or a derived output that
  no upstream stage node produces. Dropping the last stage gives exactly the
  guard without stages, which can admit more: its declaration pass follows
  structural and artifact edges, not column reads, and a resource selection
  leaves no marker in the composed graph.
- With stages listed, no hold-out source and no declared hold-out selection
  (in a `transport_graph` or entitlement row) is upstream of calibration, and
  no stage output or stage module carries a hold-out measure. The guard
  without stages is unchanged, so it still misses a hold-out selection that
  calibration reaches only through column reads, and any hold-out selection
  in an entitlement row.
- With stages listed, the outcome is deterministic. Shape checks run first,
  over every declared node and binding, so a malformed declaration refuses
  naming its first malformed node even when an earlier node would refuse on
  substance. Then, in each pass, the first structural stage node upstream of
  calibration refuses, then the first derived output no upstream stage node
  produces, and otherwise the first offending node in declaration order. The
  composed pass runs only after the declaration pass admits: it checks the
  entitlement rows' shapes and reads the compiled EXPAND cells and claims,
  then applies the same order to the graph's nodes. One message is not
  fixed: when several links name undeclared nodes, which of them a refusal
  names follows set iteration order, as in the guard without stages.
- With stages listed, the checks never read a malformed field so as to admit
  more than composition would. Every field they read is validated first,
  refusing naming its node, or read exactly as composition reads it. The two
  EXPAND parameters are read conservatively in the declaration and exactly in
  the composed pass, where a malformed one refuses.
- With stages listed, no node of a listed stage upstream of calibration is
  structural, in either pass, so the values kernels receive without a
  `Slice` (ids, memberships, weights, strata, frame metadata and the mass
  log) come upstream of calibration only from nodes outside every stage.
- The composed pass gives every column read the writers the executor's
  static writer lists give it (`microcosm.graph.executor._input_writers`),
  on every generated graph, the writer and EXPAND examples, and the composed
  NZ and entitlement graphs. Those lists cover column reads only, which is
  why the rule above covers the values kernels receive without one.
