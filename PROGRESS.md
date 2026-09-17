# Lane journal — native retention seal (`native-retention-seal`)

Branch `native-retention-seal`, from `origin/native-scale-transport` at
`a64f7b733`. Worktree `~/PolicyEngine/_worktrees/microcosm-native-retention`.

Implements Max's 2026-09-17 decision on the transport lane's report §10
question 2: **option (b) — replace the object comparison with a content
seal**, plus 1/15 now and 1/10 when the machine is quiet, JSON segment bodies
kept, and the four row-count ceilings left to another lane.

This file is a session-handoff journal. Accurate when written, historical
afterwards — check git/GitHub for current truth.

## State

Reading the authorities and the code at this head. Nothing implemented.

## Done

- Verified worktree/branch/head; synced `uv sync --all-packages --locked --extra us` (exit 0, pandas 3.0.3).
- Read `docs/us-native-scale-transport.md` (481 lines) and
  `experiments/native-scale-transport/out.md` (839 lines) in full.
- Read at this head: `survey_population_replay.py` (208 lines, whole file),
  `survey_atomic_geography._population_stamp` / `_copy_population`,
  `survey_population_preparation._frame_identity` / `_cell` /
  `_frame_cell_encode`, `graph_context.encode_us_frame_context`,
  `population._storage_parts`, `executor._observer_snapshot` and the observer
  call site.

## Next

1. `docs/us-native-retention-seal.md` — the four required answers.
2. Discrimination battery (pre-change), then implement.
3. 1/1000 cold + required replay; then the 1/15 run.

## Open findings (provisional, to be proved)

- `_population_stamp` is documented as **not** comparable across
  reconstructions (`survey_atomic_geography.py:230-237`): it folds masked
  `_data` under nulls. `same_replayed_population` deliberately *permits* the
  store's zeroed null backing (`survey_population_replay.py:79-82`,
  `NONCANONICAL_NULL_BACKING`). So a stamp-equality seal would refuse runs the
  object comparison accepts. The seal this lane needs is a **replay seal**, not
  the stamp.
