# Pre-calibration stages

A transport graph's calibration may read only what the calibration ancestry
guard admits (`validate_transport_calibration_ancestry` in
`microcosm.build.transport.compose`). Without stages, the guard refuses every
benefit-unit engine semantic upstream of a `calibrate.*` node or the
calibration endpoint, and every hold-out source, hold-out selection and the
graph resource's own digest. Person-entity rules bindings are admitted.

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
      "citation": "<method card row or ruling>"
    }
  ]
}
```

## Validation

Key sets are closed at every level, and `schema_version` is the integer 1.
Stage ids are distinct, safe node ids. Each stage names an existing binding,
repeats its `rulespec_path` exactly, and no binding appears in two stages.
Outputs are nonempty and distinct; each is on the binding's entity, names one
of its `variables`, and has a `purpose` of `calibration_measure` or
`classification` (a judgment another pre-calibration node reads). An output may
not be a hold-out `(entity, measure)`, and a stage's module may not produce a
hold-out measure. The resource is checked before either graph pass.

## Which nodes belong to a stage

In the declaration pass, a node belongs to a stage when its `rules_binding` is
the stage's binding or any `{"binding": id}` selector in its parameters names
it. In the composed pass, a node belongs to a stage when its parameters carry
the binding's prepared engine reference. A node that names a listed and an
unlisted benefit-unit binding refuses. A parameter carrying the whole prepared
`engine_refs` inventory is bookkeeping, not a stage invocation, and still
refuses upstream of calibration.

## Rules

Only nodes upstream of calibration are checked. A stage binding's nodes after
calibration, such as extension checkpoints, are outputs and are unaffected.

- Absent, or `"stages": []`, is the guard without stages, outcomes and messages
  alike.
- A listed benefit-unit stage admits its own nodes, their engine reference and
  outputs, and group-encoder outputs read upstream only by listed benefit-unit
  stage nodes. Any other benefit-unit semantic upstream refuses as
  "not a declared pre-calibration stage".
- A listed person stage admits nothing new; it only adds the next rule.
- A node outside a listed stage that reads one of the stage binding's variables
  not in the stage's `outputs` refuses as "undeclared pre-calibration stage
  output". Variables a stage's own nodes read do not count.
- The hold-out, selection, digest and prepared-map checks still apply to every
  upstream node, stage nodes included, so each stage's ancestry is clean.

Both passes apply these rules; the composed pass covers compiled predecessors
and extension nodes. No node parameter or hash changes.

## Invariants

These are tested, as properties and examples, in
`tests/engine_free/shared/test_transport_composed_graph.py`.

- With no stages, the guard accepts and refuses exactly as the frozen copy of
  the guard without stages does.
- Person stages only restrict: a graph accepted with only person stages listed
  is accepted without them.
- Dropping a person stage never refuses a graph its listing admitted; dropping
  a benefit-unit stage never admits a graph its listing refused.
- No hold-out source or selection is upstream of calibration, with or without
  stages, and no stage output or stage module carries a hold-out measure.
- With stages listed, a refusal names the first offending node in declaration
  order.
