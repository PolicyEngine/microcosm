# PR #847 gate round 2 final report

Date: 2026-09-02

Branch: `fix-847-r2`

Base: `142cb0c0e7d2f982c9ed61a2fc93a653ebb1619f`
(`origin/graph-amendments-11-13` at dispatch)

Worktree: `/private/tmp/microcosm-fix-847-r2`

## Outcome

F2, F3, F4, F5, F7, and F10 are implemented. All regression tests and the
required final matrix pass. No frozen interface file or acceptance test was
edited, no network request succeeded after the required fetch attempt, and
nothing was pushed.

The first full run exposed one legitimate UK EXPAND rewrite that the initial F4
guard treated as an ordinary copied-cell mutation. That integration was fixed
without changing the frozen interface or an acceptance test: the executor now
recognizes only a full-row `Owned(rewrite=True)` claimant in the EXPAND's
compiled population version, intersected with `expand_cells`. A dedicated
ordinary cold/cache regression was added, and H2 is green.

## Isolation and environment

The required fetch was attempted before worktree creation:

```text
fatal: unable to access 'https://github.com/PolicyEngine/microcosm.git/':
Could not resolve host: github.com
```

The run required no network, and the existing local remote-tracking ref was
already exactly `142cb0c0e7d2f982c9ed61a2fc93a653ebb1619f`. The worktree and
branch were then created at that SHA.

The requested fresh cache path was empty, so the exact locked all-package
US+UK environment was synchronized offline from the populated temporary cache:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache   uv sync --all-packages --locked --extra us --extra uk --offline
```

Import isolation was verified immediately before the final suite:

```text
/private/tmp/microcosm-fix-847-r2/packages/microcosm-graph/src/microcosm/graph/__init__.py
```

## F2 — schema-v1 tolerance compatibility

Commit: `6def5b4915ff67b72dd5c2cc46752c3ab15f917f`
(`fix(graph): migrate legacy manifest capabilities`)

The portable schema is now v2. A schema-v1 node whose raw capabilities omit
`tolerance` loads with `legacy_capabilities: true`, preserves the immutable raw
mapping, and is forced to `hit: false`. Cache lookup names the legacy tolerance
gap as a miss, and certified loading names it as `unreached`. Serialization
retains the raw capabilities exactly and never fabricates `tolerance`; the
v2-emitted form round-trips stably. The committed fixture is
`packages/microcosm-graph/tests/fixtures/manifests/v1_tolerance_bound_without_tolerance.json`.

Regression nodes:

- `packages/microcosm-graph/tests/test_graph_manifest.py::test_v1_tolerance_bound_manifest_loads_as_legacy_cache_miss`
- `packages/microcosm-graph/tests/test_graph_executor.py::test_cache_load_names_legacy_capabilities_without_tolerance`

Fail-before command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_manifest.py::test_v1_tolerance_bound_manifest_loads_as_legacy_cache_miss   packages/microcosm-graph/tests/test_graph_executor.py::test_cache_load_names_legacy_capabilities_without_tolerance -q
```

Observed red excerpt (condensed):

```text
FF
ValueError: tolerance_bound numerics require a tolerance
legacy cache receipt did not produce the named legacy_capabilities/tolerance miss
2 failed
```

Pass-after command: the same command.

```text
..                                                                       [100%]
2 passed
```

## F3 — complete coordinate-writer tolerance provenance

Commit: `ce291b8345ecca7109c5e90444ad57dc6b91b516`
(`fix(graph): aggregate coordinate writer tolerances`)

Single-producer lookup was replaced by causal writer chains covering the
originating producer, every EXPAND that actually materialized entrant rows,
masked/full rewrites, and claimants. Reader tolerance is the componentwise
maximum of `rtol`, `atol`, and `ulps` across non-bitwise writers. Claimants see
their claimed coordinates. Mass-partition entrant rows attribute the EXPAND
writer. Dotted `tolerance_writers` maps are portable in capabilities and in
tolerance-reporting evidence. Cache records lacking or disagreeing with this
provenance are named misses; `require` fails preflight and `auto` recomputes.

Regression nodes:

- `...::test_entrant_expand_aggregates_all_coordinate_writer_tolerances[producer-bound]`
- `...::test_entrant_expand_aggregates_all_coordinate_writer_tolerances[expand-and-claim-bound]`
- `...::test_entrant_expand_aggregates_all_coordinate_writer_tolerances[componentwise-maximum]`
- `...::test_masked_writer_preserves_upstream_tolerance_provenance`
- `...::test_expand_lineage_receipt_and_materialized_cell_survive_cache`
- `...::test_partitioned_graph_accepts_structural_entrant_partition_values`
- `...::test_cache_misses_receipt_without_tolerance_writer_provenance`

Here `...` is
`packages/microcosm-graph/tests/test_graph_executor.py`.

Fail-before commands:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_entrant_expand_aggregates_all_coordinate_writer_tolerances   packages/microcosm-graph/tests/test_graph_executor.py::test_partitioned_graph_accepts_structural_entrant_partition_values -q

UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_masked_writer_preserves_upstream_tolerance_provenance -q

UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_cache_misses_receipt_without_tolerance_writer_provenance -q
```

Observed red excerpts (condensed):

```text
claimant/read tolerances omitted producer, EXPAND, or claimant writers
mass-partition writer attribution stopped at CREATE
masked ownership erased the upstream bound
pre-provenance cache receipt was accepted
```

Pass-after command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_entrant_expand_aggregates_all_coordinate_writer_tolerances   packages/microcosm-graph/tests/test_graph_executor.py::test_masked_writer_preserves_upstream_tolerance_provenance   packages/microcosm-graph/tests/test_graph_executor.py::test_expand_lineage_receipt_and_materialized_cell_survive_cache   packages/microcosm-graph/tests/test_graph_executor.py::test_partitioned_graph_accepts_structural_entrant_partition_values   packages/microcosm-graph/tests/test_graph_executor.py::test_cache_misses_receipt_without_tolerance_writer_provenance -q
```

```text
.......                                                                  [100%]
7 passed
```

The broader executor plus C-acceptance run was `46 passed`.

## F4 — copied-row storage invariance

Primary commit:
`fb927b4d84b9b998bfe288faf31c6eaa304f5e4f`
(`fix(graph): preserve copied expand rows`)

UK integration follow-up:
`a226f32f2ce6741ec44dc4a1e9197caae747326a`
(`fix(graph): honor declared expand rewrites`)

Cold materialization and cached restore now compare each copied addition with
its source for every carried data column and reject with the coordinate plus
target/source ids on any difference. Entity ids and person memberships retain
their structural remapping semantics. A copied carried coordinate may differ
only when an `expand_cells` overlay has a same-version compiled claimant with
`Owned(rewrite=True, rows="all")`. Entrants remain materializable, and a
non-rewrite claimant cannot authorize a copied-row mutation.

Regression nodes:

- `packages/microcosm-graph/tests/test_graph_executor.py::test_entrant_expand_rejects_mutated_copied_carried_values`
- `packages/microcosm-graph/tests/test_graph_population.py::test_expand_lineage_carries_rows_remaps_memberships_and_restores_cache`
- `packages/microcosm-graph/tests/test_graph_executor.py::test_expand_allows_copied_value_declared_as_same_version_rewrite`

Initial fail-before command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_entrant_expand_rejects_mutated_copied_carried_values   packages/microcosm-graph/tests/test_graph_population.py::test_expand_lineage_carries_rows_remaps_memberships_and_restores_cache -q
```

```text
FAILED ...test_entrant_expand_rejects_mutated_copied_carried_values
FAILED ...test_expand_lineage_carries_rows_remaps_memberships_and_restores_cache
Failed: DID NOT RAISE <class 'microcosm.graph.errors.NodeRejectedError'>
Failed: DID NOT RAISE <class 'microcosm.graph.population.PopulationError'>
```

The full suite then exposed the frozen-interface split declaration used by the
UK CGT stage:

```text
FAILED ...test_h2_uk_spine_parity
Node 'cgt_band_donors' ... changed carried storage in person.capital_gains
for copied target/source ids [(100003001, 3001), ...]
1 failed, 350 passed, 2 warnings in 122.01s (0:02:02)
```

The follow-up regression was captured red before the compiled rewrite bridge:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_expand_allows_copied_value_declared_as_same_version_rewrite -q
```

```text
Node 'copy_rewritten_income' ... changed carried storage in person.income
for copied target/source ids [(4, 1)]
1 failed
```

Pass-after commands:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_expand_allows_copied_value_declared_as_same_version_rewrite   packages/microcosm-graph/tests/test_graph_executor.py::test_entrant_expand_rejects_mutated_copied_carried_values   packages/microcosm-graph/tests/test_graph_population.py::test_expand_lineage_carries_rows_remaps_memberships_and_restores_cache -q

UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_acceptance_h_parity.py::test_h2_uk_spine_parity -q
```

```text
...                                                                      [100%]
3 passed
.                                                                        [100%]
1 passed, 1 warning
```

## F5 — portable D2 mass receipt for every structural change

Commit: `fb165711d459fecd8764b1954febb82ee5d91657`
(`fix(graph): receipt every structural mass change`)

Every FILTER, EXPAND, and REWEIGHT now receives the executor-authored portable
mass record after authoritative ledger reconstruction, independent of policy,
partitions, entrant strata, or a kernel-authored mass payload. The record
contains `policy`, `before`, `after`, `stratum_before`, and `stratum_after`,
plus `partition` when applicable.

Regression node:

`packages/microcosm-graph/tests/test_graph_executor.py::test_group_entrant_expand_manifest_json_carries_mass_record`

Fail-before command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_executor.py::test_group_entrant_expand_manifest_json_carries_mass_record -q
```

```text
KeyError: 'mass'
1 failed
```

Pass-after command: the same command.

```text
.                                                                        [100%]
1 passed
```

The broader executor plus B-ownership and D-weight run was `57 passed`.

## F7 — rendered per-partition mass ledger

Commit: `4da1e8fae2ec3c5163ed2333b87d22080320bc5a`
(`fix(graph): render partition mass ledgers`)

The self-contained mass-ledger view now renders partition value, stratum,
before, after, and signed delta in deterministic order whenever
`mass["partition"]` exists. Missing before/after cells render as zero, so losses
and gains remain visible even when aggregate stratum mappings are empty. The
ordinary explain assertion pins the complete table header and representative
positive/negative deltas. There was no checked-in generated HTML or digest pin
to regenerate.

Regression node:

`packages/microcosm-graph/tests/test_graph_explain.py::test_calibration_view_renders_partition_mass_with_deltas`

Fail-before command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_explain.py::test_calibration_view_renders_partition_mass_with_deltas -q
```

```text
AssertionError: assert 'Mass by household.period partition' in rendered
1 failed
```

Pass-after commands:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_explain.py::test_calibration_view_renders_partition_mass_with_deltas -q
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_explain.py -q
```

```text
1 passed
14 passed
```

## F10 — collision-safe PopulationView entity access

Commit: `d42afdeb83950c11142cad57adc6ef79e197dbc4`
(`fix(graph): disambiguate population entity access`)

`PopulationView.entity(name)` is the reliable accessor and calls the base table
implementation directly. Non-colliding attribute shorthand remains. When a
declared entity collides with a public `Frame` member, attribute lookup routes
to a guided `AttributeError` recommending `.entity(name)` instead of silently
returning the unrelated member. The exact default `view.person` alias remains
available. Lookup is guarded against missing-slot recursion during pickle
protocol probes.

Regression nodes:

- `packages/microcosm-graph/tests/test_graph_manifest.py::test_population_view_entity_accessor_handles_frame_attribute_collisions[metadata]`
- The same node with parameters `[schema]`, `[table]`, and `[entities]`.

Fail-before command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_manifest.py::test_population_view_entity_accessor_handles_frame_attribute_collisions -q
```

```text
FFFF                                                                     [100%]
AttributeError: PopulationView has no attribute 'entity'
4 failed
```

Pass-after commands:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_manifest.py::test_population_view_entity_accessor_handles_frame_attribute_collisions   packages/microcosm-graph/tests/test_graph_manifest.py::test_manifest_json_round_trip_and_population_view -q
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache uv run pytest   packages/microcosm-graph/tests/test_graph_manifest.py -q
```

```text
.....                                                                    [100%]
5 passed
29 passed
```

## Final verification

Exact required package suite:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache   uv run pytest packages/microcosm-graph packages/microcosm-fit/tests
```

```text
352 passed, 1 warning in 107.07s (0:01:47)
```

The warning is the existing PolicyEngine UK Pydantic class-config deprecation
from `policyengine_uk/utils/scenario.py`.

Ruff was run over all 10 touched Python files:

```text
ruff check: All checks passed!
ruff format --check: 10 files already formatted
```

CI grouping:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache   uv run python tools/ci_test_groups.py --verify
```

```text
tracked_test_files=354
verification=ok
```

Acceptance burndown:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-impl-uv-cache   uv run python tools/graph_acceptance_burndown.py --verify
```

```text
green     41  A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 B6 B7 C1 C2 C3 C4 C5 D1 D2 D3 D4 D5 D6 E1 E2 E3 E4 E5 F1 F2 F3 F4 F5 G1 G2 G3 H1 H2 H3
red        0  none
missing    0  none
verification=ok
```

Final integrity audit:

- `git diff --check`: passed.
- No diff in `decl.py`, `kernel.py`, or `docs/graph-interface.lock`.
- No `test_acceptance_*.py` file changed.
- No push, PR mutation, successful network fetch, artifact publication, or
  release action was performed.

## Unimplemented rulings and proposed acceptance-test diffs

None. Every ruling was implemented. No acceptance-test change is needed or
proposed.
