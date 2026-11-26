# Shared calibration machinery for local areas

The UK local build and the US local build each calibrate a pooled file to
targets at several geography levels. This page lists the pieces of that work
that are shared code, what each country declares, and how to turn each option
on. It is the reference for microcosm#1729.

Two rules hold for every piece:

- **A default changes nothing.** With the default declarations each build
  calls the solver with the arguments it used before and gets the same
  weights. An option changes an output only when a build asks for it.
- **Nothing here blocks a release.** The gate and the diagnostics report. A
  caller decides what a failure means.

## What is shared, and what each country declares

| Piece | Shared code | UK local | US local (ACS) |
|---|---|---|---|
| L2 penalty anchor | `calibrate(l2_anchor=, l2_anchor_weights=)` | `uniform` on the refit stage (`UK_LOCAL_SIZE_REFIT_L2_ANCHOR`) | `--l2-anchor` (default `initial`), `--l2-anchor-weights` |
| Target loss weights | `microcosm.calibrate.target_loss_rules` | `UK_LOCAL_TARGET_WEIGHT_RULE_DECLARATIONS` (`uk_runtime.target_weights`), `UK_NATIONAL_TARGET_WEIGHT_RULE_DECLARATIONS` (`uk_runtime.national_doctrine`) | `US_ACS_LOCAL_TARGET_WEIGHT_RULES`, `--target-weight-rule` |
| Loss cap for a level or a family | `target_loss_caps`, `single_cap_equivalent` | one number (`UK_LOCAL_TARGET_LOSS_CAP`) | one number; `--county-target-loss-cap-scope county_rows` |
| Area support | `gates.area_support_gate`; measurement in `microcosm.build.area_support` | three absolute floors in `uk/gates.json` | `district_ess_collapse_gate` declares a relative and an absolute ESS floor |
| Weight bound | `max_weight_ratio`, `max_weight_reference`; contract in `microcosm.calibrate.weight_bound` | `UK_LOCAL_MAX_WEIGHT_RATIO`, against each record's own weight | `--max-weight-ratio`, `--weight-bound-reference` (default `own`) |
| Which rows carry each target | `microcosm.calibrate.spine_target_shares`, `tools/spine_target_shares.py` | not run | run on a checkpoint |

## Target loss weights

A rule gives each target a loss weight from what the target is: its **level**
(nation, state, district, local authority), its **family**, its value
**basis** (`count` or `amount`) and its **value**. A country says what each row
is (`TargetLossRows`) and declares the rule by name with its parameters
(`TargetWeightRule`).

| Rule | What it does |
|---|---|
| `uniform` | Every target weighs one. |
| `level_equal` | Each level takes one equal share; rows in a level share it equally. |
| `level_family_equal` | Each level takes one equal share, each family in a level one equal share of it. With `count_power` (for example `0.5`), the count rows of a family share its count budget in proportion to `max(abs(value), 1) ** count_power`. |
| `sqrt_value_concept_budget` | The US formula: value weight over its basis mean, a budget for each concept group, an equal split between counts and amounts. It reads no level. |

Any rule may carry `family_multipliers`.

- **US.** The default, `concept_budget`, is the formula every build has used.
  It is the shared rule `sqrt_value_concept_budget`, and gives the same
  weights bit for bit. `--target-weight-rule level_family_equal_sqrt_count`
  gives each geography level one equal share of the loss. On the September
  surface the state SOI rows held about 90% of the loss weight under the
  default.
- **UK.** The UK computes every loss weight with the shared rules.
  `uk_runtime.target_weights` says what a UK row is and declares each local
  rule name as a shared rule: the doctrine's
  `nation_grain_family_equal_sqrt_count` is `level_family_equal` with
  `count_power` 0.5, over the UK's own mapping of rows to levels. The national
  build's `family_equal` (`uk_runtime.national_doctrine`) is `level_equal`
  with each family as one level. The vectors are the same as before, bit for
  bit: the tests hold them to verbatim copies of the UK arithmetic as it
  stood before the move.

  The shared module is in the hash of the UK nodes that bind loss weights
  (`uk.full.problem@1`, the two national kernels and the rotated holdout), so
  an edit to a shared rule re-keys them.

## A loss cap for a level or a family

The solver's loss is `sum(w * min(abs(miss) / s, cap)) / sum(w)` with one cap.
`target_loss_caps(default, by_level=..., by_family=...)` declares a cap for
each row: one number when nothing is declared, a vector otherwise. A family
cap wins over a level cap, and an override that matches no row is an error.

The solver needs no change to honor a vector. For row caps `c`, weights `w`
and scales `s`, `single_cap_equivalent` returns

- `cap0 = sum(w * c) / sum(w)`,
- `w' = w * c / cap0` and `s' = s * c / cap0`.

Then `w' * min(abs(miss) / s', cap0) == w * min(abs(miss) / s, c)` for every
row and `sum(w') == sum(w)`. The loss value, each row's contribution and the
gradient are those of the loss with row caps. When every row has the same cap
the weights and scales come back unchanged.

How to read a run with row caps:

- The solver records the transformed weights and scales, and `cap0` as its
  cap. So in `calibration_diagnostics.json` the top-level `past_cap_census`
  and each target's `target_loss_scale` and `target_loss_weight` read every
  row against `cap0`, not against the row's own cap.
- The declared caps and the census against them are in the `target_loss_caps`
  block, in the tool's summary and in the `build` block of
  `calibration_diagnostics.json`: the default cap, the cap of each level, the
  rows at each cap, each row that is not at the default, and
  `past_cap_census` against each row's own cap (`row_cap_census`). A run with
  one cap has no such block.

**US county builds.** By default the county profile's cap (10) applies to
every row of a county build, as before. With
`--county-target-loss-cap-scope county_rows` it applies to the county rows
only, and the state and district rows keep `--target-loss-cap`. The weight
ratio is a bound on records, not on targets, so it stays one number for the
run.

## Area support

There is one gate, `microcosm.build.gates.area_support_gate`. It checks every
area of a declared roster against the floors a country declares. Each floor
is optional: `minimum_rows`, `minimum_distinct_sources`,
`minimum_effective_sample_size`, and `minimum_effective_sample_size_ratio`, a
floor relative to the area's starting ESS (the rule microcosm#1078 adopted).

The UK local build declares the three absolute floors (50 rows, ESS 50 and 50
distinct sources). With those three and no relative floor the gate gives the
result it always gave: a test holds it to a verbatim copy of the gate as it
stood before the floors became optional, reviewed exclusions included.

`microcosm.build.area_support` measures what the gate reads:

- `effective_sample_size_by_group` is the Kish ESS of the records in each
  area.
- `area_support_table` builds the long-form table from a weight vector and
  each record's area: positive-weight rows, distinct positive-weight source
  households, ESS, and the ESS under the starting weights.
- `area_support_floor_table` gives the gate's verdict area by area, with one
  flag for each declared floor.

The US district gate (`us_local_district_ess_collapse` in `us/gates.json`)
declares a relative floor of 0.25 and an absolute floor of 15 to the shared
gate. Its entry in `gate_summary.json` is the same as before. It is
report-only unless a build passes the existing `--district-ess-gate-blocking`.

## The weight bound

`max_weight_ratio` is a hard bound on each record: no returned weight exceeds
`max_weight_ratio` times the record's **reference**, which is its weight on
the frame the solve was given.

1. The reference is the frame, never the warm start. An epoch-batched solve
   keeps the bound against the frame's weights on every batch.
2. A new frame is a new reference. A stage whose frame carries an earlier
   stage's calibrated weights bounds against those, so two stages at ratio `R`
   admit `R ** 2` against the first stage's starting weights.
3. The bound is relative, so a heavy record gets a larger absolute bound. In
   the US local file a donor row starts about 27 times heavier than an ACS
   row, so its absolute bound is about 27 times higher under the same ratio.

**A declared reference.** Point 3 is a choice. `calibrate` and
`solve_constraint_system` take `max_weight_reference`, one positive weight for
each record, and then bound each record at `max_weight_ratio` times that
reference. `weight_bound_reference` builds one: `mean`, or `group_mean` for
the mean starting weight of the record's area. The frame's weights still set
the starting point, the mass and the L2 anchor. A record that starts above its
bound starts from the bound, with the mass it gives up spread below the
bounds. Under mass conservation the bounds must admit the total, or the solve
refuses. Without `max_weight_reference` the solve is the historical one, byte
for byte.

In the US local tool, `--weight-bound-reference mean`, `state_mean` or
`district_mean` turns it on. The default, `own`, is each household's own
starting weight.

`weight_bound_report` measures any weight vector against a reference: the
largest ratio, the records above the bound, the records that start above it,
and whether the bounds admit the starting mass. Read it on a finished solve
before you turn a declared reference on: in the pooled US file the donor rows
start far above a shared bound at the default ratio, so the bound moves
starting mass off them.

## Which rows carry each target

`spine_target_shares` reports, for each target, the share of its weighted
total that each spine's rows carry, against that spine's share of the weight
among the records the target is about. A target that the donor rows carry
almost alone pulls weight onto them. `excess` (share minus mass share) and
`lift` (their ratio) rank the targets.

The tool reads saved inputs and needs no build:

```bash
uv run python tools/spine_target_shares.py \
    --us-acs-local-checkpoint <run>/checkpoints --out shares.json --spine <donor spine>
```

It reports at the design weights, and at the calibrated weights when
`weights_latest.npz` is in the checkpoint. For any other pooled build pass
`--matrix`, `--weights` and `--spines`. `--minimum-carriers N` withholds a
target that a spine carries with fewer than `N` distinct source households;
pass the disclosure floor for a build on licensed microdata.

## What only a full build can show

The tests prove that the default declarations send the solver the same
arguments and return the same arrays on small fixtures, and that the UK
vectors and the UK gate result are those of the code before the move. They do
not run a full ACS local build or a UK build. A full build is the only proof
for the full surface, the full pool and the stages around the solve.

## What re-keys in the UK graph

The UK graph hashes the bytes of the modules its kernels name. This work
edits some of them, with the same outputs, so these nodes compute again on
the next run instead of reading a cached result:

- every UK solve node, through `calibrate.adam@1` (`calibrate/solve.py`);
- the kernels of `uk_runtime/graph_targets.py`, which hash their own module:
  `uk.full.target_compilation@1`, `target_selection@1`, `measures@1` and
  `problem@1`. The problem also hashes the modules that decide its loss
  weights, which now include the shared rules. Those modules are named in
  `uk_runtime/target_compile_modules.py`
  (`UK_FULL_PROBLEM_LOSS_WEIGHT_MODULE_NAMES`) and no longer imported by
  `graph_targets`, so the other three kernels have no import edge to them;
- the kernels of `uk_runtime/graph_national.py`: `uk.full.national_targets@1`,
  `national_problem@1`, `national-gates@1` and `national-readback@1`;
- the kernels of `uk_runtime/graph_terminal.py`: the rotated holdout,
  `uk.full-gates@1` (which also hashes `build/gates.py`), its preflight, and
  the three export kernels;
- outside the UK, the transport gate kernel, which hashes `build/gates.py`.

The UK spine kernels are not among them.

No `uk/gates.json` digest, gate replay or receipt moves. The calibrate parity
pin is re-recorded.
