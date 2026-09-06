# PROGRESS — composed prepared-ASEC + native-ACS population graph

Lane branch `microcosm-us-composed-population-20260906`, based on
`us-launch-integration-20260906` at `ab10f77854b7b9e8057a07dbec40d15c10b96297`.
`origin/main` (`c1b83241`) is an ancestor of that base; the base is 124 commits
ahead and 0 behind. Review root for receipts:
`~/architecture-reviews/microcosm-20260904`, files prefixed
`us-composed-population-`.

## State

The composed seam and its reviewed implementation scope exist and hash. Tests
are not written yet, so nothing here is evidence of execution parity.

## Done

- Worktree/branch from the frozen base; base relation recorded.
- Probe 1 (`us-composed-population-probe1-prepared-frame.*`): the real prepared
  ASEC Frame satisfies `spine_assembly._validate_source_frame` unchanged —
  US schema, household-only DESIGN weights, no links, integral nonnegative IDs,
  no preassembled support provenance.
- Probe 2 (`us-composed-population-probe2-compose.*`): composing it with a real
  native ACS Frame built from invented PUMS archives has exactly **one** schema
  difference — `tax_unit_role_input` / `filing_status_input` / `PERIDNUM` are
  `string[python, pd.NA]` on the prepared arm and `string[python, nan]` on the
  ACS arm, which `_stack_spine_tables` refuses. With that one storage bridge,
  `prepare_stacked_spine` passes untouched.
- Probe 3 (`us-composed-population-probe3-two-parent.*`): the executor cannot
  structurally combine two parent populations. A tuple `base` is "not a
  structural node"; a `CREATE` node "has no base or population"; a structural
  node "declares base, not population". A single `CREATE` over two declared
  sources compiles — the shape production assembly already uses.
- `graph_composed_population.py`: two-source CREATE + the **reused**
  `USSpineHarmonizeKernel` + the **reused** `us_geography_nodes`. The narrow
  bridge, a typed source-origin artifact, and pass-through of the prepared
  arm's four typed evidence artifacts so later nodes can be appended.
- Reviewed implementation scope `composed_population_v1`: STAGE_DEPENDENCIES
  entry, inventory stage as the union of `assembly_prepare`, `asec_prepared_v3`
  and `geography` plus the new module, both bound codecs, both runtime-option
  families. All eight stages hash.

## Known consequence

Adding any stage edits `graph_implementation.py`, which is a bound module of
every stage, so **every** existing US stage implementation identity moves and
every existing US graph cache is invalidated. Baseline and post-edit digests
are preserved in `us-composed-population-stage-hashes-{base,after}.json`.

## Next

- The direct / cold-graph / fresh-process-warm test file.
- Refusal coverage: bad source, context, producer identity, missing evidence.
- Genuine-small preparation/runner proposal (authored, not executed).
