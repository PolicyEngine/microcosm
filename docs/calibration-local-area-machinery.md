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
| Target loss weights | `microcosm.calibrate.target_loss_rules` | `uk_runtime.target_weight_declarations` (declared; see the note below) | `US_ACS_LOCAL_TARGET_WEIGHT_RULES`, `--target-weight-rule` |
| Loss cap for a level or a family | `target_loss_caps`, `single_cap_equivalent` | one number (`UK_LOCAL_TARGET_LOSS_CAP`) | one number; `--county-target-loss-cap-scope county_rows` |
| Area support | `microcosm.build.area_support` | `gates.area_support_gate`, floors in `uk/gates.json` | `district_ess_collapse_gate` declares its floors to the shared gate |
| Weight bound | `max_weight_ratio`; contract in `microcosm.calibrate.weight_bound` | `UK_LOCAL_MAX_WEIGHT_RATIO` | `--max-weight-ratio` |
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
- **UK.** `uk_runtime.target_weight_declarations` states each UK rule name as
  a shared rule: the doctrine's `nation_grain_family_equal_sqrt_count` is
  `level_family_equal` with `count_power` 0.5, over the UK's own mapping of
  rows to levels. A test holds the shared rule equal to the UK code bit for
  bit under all six rules. The UK solve still computes its vector in
  `uk_runtime.target_weights`, because that module is part of the UK graph's
  code identity and an edit to it re-keys the UK target and solve nodes.

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

Two things to know when you read a run with row caps:

- The solver records the transformed weights and scales, and `cap0` as its
  cap. The declared row caps are in the tool's summary (`target_loss_caps`).
- The past-cap census of the calibration diagnostics measures every row on the
  default scales against `cap0`, not against the row's own cap.

**US county builds.** By default the county profile's cap (10) applies to
every row of a county build, as before. With
`--county-target-loss-cap-scope county_rows` it applies to the county rows
only, and the state and district rows keep `--target-loss-cap`. The weight
ratio is a bound on records, not on targets, so it stays one number for the
run.

## Area support

`microcosm.build.area_support` measures and checks the support of each area:

- `effective_sample_size_by_group` is the Kish ESS of the records in each
  area.
- `area_support_table` builds the long-form table from a weight vector and
  each record's area: positive-weight rows, distinct positive-weight source
  households, ESS, and the ESS under the starting weights.
- `area_support_gate` checks every area of a declared roster against the
  floors a country declares. Each floor is optional: `minimum_rows`,
  `minimum_distinct_sources`, `minimum_effective_sample_size`, and
  `minimum_effective_sample_size_ratio`, a floor relative to the area's
  starting ESS (the rule microcosm#1078 adopted).
- `area_support_floor_table` gives the same verdict area by area.

With the three absolute floors and no relative floor the gate calls
`microcosm.build.gates.area_support_gate`, which the UK local build uses with
50 rows, ESS 50 and 50 distinct sources. A test holds the general rule equal
to that function wherever both apply.

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

`weight_bound_reference` declares another reference (`mean`, or `group_mean`
for the mean starting weight of the record's area), and `weight_bound_report`
measures any weight vector against it: the largest ratio, the records above
the bound, the records that start above it, and whether the bounds admit the
starting mass. The report is how to read what a declared-reference bound
would do before a solve enforces one. The solver itself bounds against the
frame's weights only.

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
arguments and return the same arrays on small fixtures. They do not run a
full ACS local build or a UK local build. A full build is the only proof for
the full surface, the full pool and the stages around the solve.
