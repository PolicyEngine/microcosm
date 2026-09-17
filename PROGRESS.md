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

Design established and proved by running code at this head. The transport
report's proposed mechanism for option (b) does not hold; the seal has to be
purpose-built. Nothing implemented yet.

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

## Findings, proved by running code at this head

**Neither existing seal is the seal.** The transport report's §10 question 2(b)
says "`_population_stamp` already folds everything `same_replayed_population`
compares except the *type* assertions". That sentence is wrong in **both**
directions, and both halves are proved by scripts committed under
`experiments/native-retention-seal/`:

1. **`_population_stamp` is too strict.**
   `probe_stamp_vs_comparison.py` builds a US_SCHEMA population, round-trips its
   frame through `ContentStore.put_frame`/`load_frame`, and gets:
   `same_replayed_population` **ACCEPTS** (the store zeroed `_data` under the
   null mask, which `NONCANONICAL_NULL_BACKING`,
   `survey_population_replay.py:79-82`, deliberately permits) while
   `_population_stamp(expected) != _population_stamp(actual)`. So a
   stamp-equality seal would **refuse a required replay**. Its own docstring
   says so (`survey_atomic_geography.py:231-237`).
2. **`_frame_identity` is too weak.** `probe_frame_identity_gaps.py` finds three
   discriminations `same_replayed_frame` makes that `_frame_identity` does not:
   float64 NaN **payload bits** (two quiet NaNs; `NATIVE_BITS`), quiet versus
   signalling NaN (`NATIVE_BITS`), and
   `DataFrame.flags.allows_duplicate_labels` (`TABLE_TYPE_OR_FLAGS`). `_cell`
   maps every NaN to `None`, so `_frame_identity` spells all of them `null`.
   It also cannot be applied to a non-US_SCHEMA frame at all (`FRAME_TYPE`).

**Therefore the seal is purpose-built and lives in `survey_population_replay.py`,
beside the comparison it replaces**, folding exactly the bytes each comparison
compares — no more, no less.

**A fourth gap, from pandas rather than from this repo.**
`pd.DatetimeIndex._comparables == ['name', 'freq']`, and two DatetimeIndexes with
equal values, dtype and name but different `freq` are **not** `identical()` while
their bytes are equal. Any seal that folds only (class, dtype, name, bytes)
misses it. The seal folds `type(index)._comparables` generically.

**Admitted dtypes are narrow**, which bounds the problem: `_series` refuses
`CategoricalDtype`, `Float64Dtype` and `DatetimeTZDtype` with
`UNSUPPORTED_EXTENSION_DTYPE`; only masked integer/boolean, `StringDtype`,
`object` and plain numpy dtypes get through.
