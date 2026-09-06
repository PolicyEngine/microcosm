# Composed prepared-ASEC / native-ACS binding lane

Session journal (history, not state — see `CLAUDE.md` "Root journals are history").
Child of frozen `ab86b5d2`, branch `us-composed-asec-bindings-20260906`,
base `origin/main` c1b83241.

## Scope

Bind the carried prepared-ASEC evidence to the ASEC rows of the composed
ACS+ASEC population and expose the already-reviewed corrected monetary and
demographic measurements over those rows only, leaving the native ACS arm
byte-for-byte unchanged. No source read, no engine, no calibration, no release.

## State

Reading and reproduction phase. Baseline `test_us_graph_composed_population.py`
is green in this child (30 passed).

## Established by reading (not assumed)

- `Slice.rows` masks the table the kernel receives (`executor.py`), and
  `_patch_columns` refuses any change to non-owned storage
  (`population.py::storage_equal`). A masked `Owned` is therefore the
  framework's own byte-for-byte preservation guarantee for a shared column.
- The composed person table carries `A_AGE` on **both** arms
  (`acs_pums.py::_with_structural_columns` maps `AGEP` into `A_AGE`), while
  `age` is ACS-only. `derive_cps_carried_current_leaves` already declares
  `age` as a reviewed ASEC leaf derived from `A_AGE`.
- `A_SEX` is absent from the composed person table; `is_female` is ACS-only.
- ASEC-only integer columns arrive as nullable `Int64` on the composed frame
  (missing on ACS rows), while the reviewed verifiers require `int64`.
- `income_observations` carries the prepared arm's ordered original
  `person_id`/`income_year`; `housing_universe` carries its ordered original
  `household_id`. Both are receipt-bound to the prepared receipt.

## Done

- [x] worktree + ff to ab86b5d2, own `.venv` (imports resolve to this child)
- [x] read AGENTS/CLAUDE, docs/us-composed-population-graph.md, the composed
      graph, the two ASEC kernels and every helper they use
- [x] probed the real invented composed fixture for arm shapes and dtypes

## Next

- [ ] reproduce the accepted kernels' refusals on the composed fixture
- [ ] implement the typed binding node + the two measurement nodes
- [ ] classification, inventory, docs, tests
