# PROGRESS — composed prepared-ASEC + native-ACS population graph

Lane branch `microcosm-us-composed-population-20260906`, based on
`us-launch-integration-20260906` at `ab10f77854b7b9e8057a07dbec40d15c10b96297`.
`origin/main` (`c1b83241`) is that base's merge base; the base is 124 commits
ahead and 0 behind. Review root for receipts:
`~/architecture-reviews/microcosm-20260904`, files prefixed
`us-composed-population-`. Nothing pushed, merged or published.

## State

Complete for this lane. The composed graph exists, runs, and is covered by 25
passing tests over invented sources through the real codecs and kernels. No
genuine source has been read and no genuine run has been performed; the
genuine-small run is proposed in `docs/us-composed-population-graph.md` only.

## Done

- Probes (preserved in the review root, not committed):
  - The real prepared ASEC Frame satisfies `spine_assembly._validate_source_frame`
    unchanged.
  - Composing it with a real native ACS Frame has exactly **one** schema
    difference — physical string storage on three columns — after which
    `prepare_stacked_spine` passes untouched.
  - The executor cannot structurally combine two parent populations: tuple
    `base`, `CREATE`-with-base and structural-node-with-population all refuse.
    A single `CREATE` over two sources compiles.
- `graph_composed_population.py`: one new CREATE kernel, the **reused**
  `USSpineHarmonizeKernel`, the **reused** geography nodes, the narrow storage
  bridge, a typed `source_origin` artifact and pass-through of the prepared
  arm's four typed evidence artifacts.
- Reviewed implementation scope `composed_population_v1`: the union of
  `assembly_prepare`, `asec_prepared_v3` and `geography` plus the new module,
  both bound codecs, both runtime-option families.
- `test_us_graph_composed_population.py`: 30 tests — direct, cold graph, and
  fresh-process warm replay with zero kernels; native identity, absence,
  membership, lineage, mass, importance anchor, evidence, geography, checkpoint;
  the refusal cases; a `sample_fraction = 0.5` case; and a structural case with a
  reordered multi-SPM household.
- The root's independent Fable 5.1 review of `76d1a9b8` raised 0 critical and 4
  should-address items. All four are fixed: the false "node keys unchanged"
  claim, an origin digest that bound nothing back to either arm, unrecorded
  alignment between the whole-arm prepared evidence and the composed rows, and an
  overclaiming binder docstring. See the report's review section.
- `docs/us-composed-population-graph.md`, including the genuine-small
  preparation/runner proposal and its memory measurement plan.

## Not done, deliberately

No engine, PUF cloning, transfer, calibration, dashboard, CD reference, housing
authority or dynamics work, and no new income-source mapping. The CREATE node
exposes the artifacts those nodes need so they can be appended.

## Known consequences and limits

See `us-composed-population-REPORT.md` for the full list. The load-bearing ones:
adding any stage moves **every** existing US stage implementation identity and
invalidates every existing US graph cache; `person.age` is observed on the ACS
arm only; the 50/50 household mass shares are the inherited #578 pilot default,
not a decision this lane makes.
