# UK target sets

The UK declares its target sets the way the US does
([US target sets](us-target-sets.md)). Since microcosm#1759 the UK choices are
data in
[`packages/microcosm-build/src/microcosm/build/uk/target_sets.json`](../packages/microcosm-build/src/microcosm/build/uk/target_sets.json),
read by the shared
[`microcosm.calibrate.target_sets`](../packages/microcosm-calibrate/src/microcosm/calibrate/target_sets.py).
The UK names (file, products, predicate) are in
[`microcosm.build.uk_runtime.target_sets`](../packages/microcosm-build/src/microcosm/build/uk_runtime/target_sets.py).

The move changed no target, selection or receipt. The proofs are listed under
"How the move was checked" below.

## The level rule

Every compiled UK target has one geography level. The file declares how to
read it:

```json
"level_rule": {
  "keys": ["geography_level", "ledger_geography_level"],
  "values": ["country", "region", "constituency", "la", "brma"],
  "conflict": "refuse"
}
```

- The level is the first of the two keys that has a value.
- When both keys have a value, they must agree.
- The level must be one of the five grains. A target with no level, or with
  another level, is refused.

`uk_runtime.graph_targets.target_geography` and the national problem
(`graph_national.national_target_geography_levels`) read levels with this
rule. The US file gives the short form `level_keys`, which is an open rule
that takes the first key.

## The products

| Product | Where it runs | Option | Modes |
|---|---|---|---|
| `full` | `uk.full.target_selection@1` (`uk_runtime/graph_targets.py`) | `--target-geographies` of the full build | `levels` |
| `calibration_line` | `uk_runtime/full_targets.py`, through `ledger_targets.uk_registry_on_calibration_line` | the line of the build | `national`, `local` |

### `full`: the solve surface by level

The `levels` mode has one clause for each of the five levels. The selection
node keeps the clauses of the levels that the build requests (all of them by
default), in compile order, with the shared `select_by_levels`. Its receipt
lists every target, kept or not (schema
`microcosm.calibrate.target-selection.v1`, read by `uk_runtime/full_gates.py`).

Two filters run before the selection and stay in the node:

- `--households-only` keeps only the `census_households/constituency` family;
- with no area BRMAs (`--brma-localisation none`), the node drops the `brma`
  rows, because they would bind on the spine's regional BRMA draw
  (microcosm#1305).

The receipt records the registry after these filters.

### `calibration_line`: what each line binds

A UK build runs on one of two lines: the national line
(`load_uk_national_target_inputs`, the release-cut certifier) or the local
line (the full dense build, which binds the national rows beside the local
surface). A contract target in `uk/uk_population_targets.json` binds on both
lines unless its `calibration_lines` field names one. Each line's mode has one
clause that keeps every compiled national reference except the ones that the
predicate `off_calibration_line` names. The UK code binds that predicate to
the contract targets that leave the line out.

The membership stays with each contract target, with its reason. The rows
leave right after the compile, before reconciliation, the measure exclusions
and the frozen-register check. `UK_CALIBRATION_LINES` is the declared modes.

## Hold-outs

The UK and US hold-outs keep different draws on purpose:

- The UK rowwise hold-out (`uk_runtime/local_rowwise.py`,
  `rotated_uk_local_holdout`) is report-only. It deals the local target rows
  into 5 rotated folds (seed 20260529) and solves once for each fold.
- The US local hold-out holds out hashed blocks of units
  (`us/local_target_construction.json`).

Both now share the last two steps in `microcosm.build.holdout`: the
`HoldoutSplit` type (`fold_splits` for the UK, `target_holdouts.role_split`
for the US) and `held_estimates`. `held_estimates` multiplies in float64. The
UK problem matrix is float64, so the UK estimates are the same bytes as
before.

The UK hold-out node (`uk.full.rotated-holdout@1`) hashes the shared hold-out
module. It also hashes the modules it reaches when it rebuilds the problem
inputs: the declared level rule and the review core.

The UK fold count and seed stay constants in `local_rowwise.py`
(`UK_LOCAL_HOLDOUT_FOLDS`, `UK_LOCAL_HOLDOUT_SEED`), as María ruled on
26 Nov 2026. A stricter UK hold-out, for example one that declares its draw as
data, would start there.

## Signed entries

The UK reviewed files sign each entry (approved by, approved on, expires on).
The UK code reads their dates and windows with the shared review core in
[`microcosm.build.declared_registers`](../packages/microcosm-build/src/microcosm/build/declared_registers.py):
`read_strict_json`, `canonical_date`, `ReviewWindow` and `stale_entries`. The
users are:

- the calibration measure exclusions and the support floor
  (`uk_runtime/measure_simulation.py`, `uk_runtime/support_floor.py`);
- the reviewed exclusions of the target-fit, local target-fit and area-support
  gates (`uk_runtime/weighted_integrity.py`).

Each UK function keeps its error texts and its error class, and decides what
a lapse does: it refuses a run, fails a gate, or reports. A gate that fails
closed records the class name of the error, so the reviewed-exclusion reader
still raises a plain `ValueError`.

Not moved:

- The small-cell deferrals (`ledger_targets.assert_uk_local_deferrals_in_force`)
  check only the expiry, and they also accept compact dates. Moving them would
  change what they accept.
- The other UK signed files (for example the uprating holds) can use the core
  later.

## The support floor stays UK code

The UK zero-support floor (`uk_runtime/support_floor.py`) drops the target rows
that no source household carries. The US county minimum support
(`tools/us_acs_local_county_surface.py`, `plan_minimum_support`) moves donor
households in the frame before targets exist. These are different policies at
different stages, so a shared module would have one caller.
`uk_runtime/off_surface_cells.py` builds cross-grain rows, so it belongs to
the nesting work of microcosm#1760.

## How the move was checked

- `test_uk_level_selection.py` compares the selection by level with verbatim
  copies of the removed `select_targets` and UK resolver, on 400 random
  registries and requests, refusals included.
- `test_uk_calibration_line_selection.py` compares the line filter with a
  verbatim copy of the old function.
- `test_holdout.py` compares the split and the held estimates with the
  replaced UK and US expressions, byte for byte.
- `test_uk_review_loaders.py` and `test_uk_reviewed_exclusions_core.py`
  compare the signed-entry loaders with verbatim copies of the old code, by
  value, warning, error class and error text.
- `test_uk_graph_terminal.py` pins the shared modules in the hash of the
  hold-out kernel and of each gate kernel.
- `test_uk_target_surface_pins.py` pins the bytes of the UK target nodes'
  artifacts on the pinned Chronicle feed: the full surface, each selection
  and the national registry.

## What a UK build computes again

`uk/target_sets.json` is a solve-scoped UK resource, so it is in the UK solve
fingerprint. The spine gates also hash that fingerprint. So the first UK build
after this change computes these again:

- the spine from `frs_brma` on;
- the target nodes and every node after them;
- the national graph.

The outputs are the same bytes.
