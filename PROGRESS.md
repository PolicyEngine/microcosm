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

Implemented and green on the replay battery (107 tests). Base financial,
property and property-tax graph tests green. Completion host and person-status
tests re-running after the retention flag. Neither measurement run has started:
both are armed and gated.

## Done

- Verified worktree/branch/head; synced `uv sync --all-packages --locked --extra us`.
- Read both authorities in full and the code at this head.
- `docs/us-native-retention-seal.md` — the four required answers.
- `survey_population_replay.py` — the purpose-built seal, beside the
  comparison it replaces.
- `test_us_survey_population_replay.py` — every mutation now runs through BOTH
  the object comparison and the seal, and must reach the same verdict with the
  same code; plus the discriminations the first battery missed.
- `executor.py` — `run_graph(_population_observer_detach=False)`, its own
  commit, four tests, no US import.
- `graph_atomic_survey_financial.py` — declared-consumer retention, both
  comparisons sealed, `_node_population_stamp` on both arms.
- Repaired a **stale implementation contract inherited from the transport
  branch** and added the guard that would have caught it.
- Armed both runs with their gates and their source trees.

## Next

1. Re-run the completion-host and person-status tests on the final head.
2. The 1/1000 cold run + required replay (gate: >40 GB available).
3. The 1/15 run behind it (gate: >45 GB available).
4. Draft PR against `native-scale-transport`.

## Inherited defect, repaired here (for the report and for Max)

`survey_population_preparation._spill_roster` gained a `Path.read_bytes` on
`native-scale-transport` at **`b6081efcb`** ("Hash a spill segment that was
already there"). That added a `resource_accesses` entry, leaving
`graph_implementation_inventory.json`'s declared `resource_accesses_sha256`
stale, so `implementation_manifest("authenticated_survey_population_v1")`
**refused** — and `SurveyPopulationCreateKernel.implementation_hash` calls it.
**No 19-node graph run was possible at `a64f7b733`.** Bisected across
`5ff889814` → `5307249b3` (both clean) → `baaf4270c` onward (all refusing).
The transport report's §4 "No committed pin moves" was recomputed before
`b6081efcb` and is stale at its own tip. Repaired in `43fb39270`, guarded by
`test_us_implementation_inventory_contracts.py` (135 assertions; reverting the
re-pin turns both arms red, verified).

## What moves, measured exactly

`experiments/native-retention-seal/implementation-identity-receipt.json`:

| comparison | roster module digests that move |
|---|---|
| `a64f7b733` → US-only commits | **none** |
| `a64f7b733` → head with the executor commit | `microcosm.graph/executor.py`, in **all ten** stages |

`survey_population_replay.py`, `graph_atomic_survey_financial.py` and
`survey_atomic_geography.py` are in no stage roster; `executor.py` is in every
one. The re-pin above moves `inventory_sha256`, which is also in every stage
manifest.

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
