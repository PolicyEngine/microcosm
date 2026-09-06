> Integration note, 2026-09-06: this lane was merged locally into the US launch integration checkout. The state and next steps below are historical. Root reviewed the later buffer-member refusal and retained the original author’s observation-axis assertions. Genuine composed-run validation and calibration remain separate work.

# Composed prepared-ASEC / native-ACS binding lane

Session journal (history, not state — see `CLAUDE.md` "Root journals are history").
Child of frozen `ab86b5d2`, branch `us-composed-asec-bindings-20260906`,
base `origin/main` c1b83241.

## Scope

Bind the carried prepared-ASEC evidence to the ASEC rows of the composed
ACS+ASEC population and expose the already-reviewed corrected monetary and
demographic measurements over those rows only, leaving the native ACS arm
byte-for-byte unchanged. No source read, no engine, no calibration, no release.

## State at the frozen commit `24f32366` (2026-09-06)

Implementation, tests and docs are **in the commit**, not still ahead of it:
`graph_composed_asec_binding.py` and `graph_composed_asec_measures.py` declare
the three nodes, `graph_implementation.py` and its inventory declare the
`composed_asec_binding_v1` stage, `test_us_spine_blindness.py` classifies both
modules, `test_us_graph_composed_asec_binding.py` covers the stage on invented
sources, and `docs/us-composed-asec-binding.md` documents it. The earlier
"reading and reproduction phase" wording below was true when written and was
left stale by the commits that followed it; it is corrected here rather than
deleted. Nothing genuine had been run at that commit.

## Established by reading (not assumed)

- `Slice.rows` masks the table the kernel receives (`executor.py`), and
  `_patch_columns` refuses any change to non-owned storage
  (`population.py::storage_equal`). A masked `Owned` is therefore the
  framework's own byte-for-byte preservation guarantee for a shared column.
- The composed person table carries `A_AGE` on **both** arms
  (`acs_pums.py::_with_structural_columns` maps `AGEP` into `A_AGE`), while
  `age` is ACS-only. `derive_cps_carried_current_leaves` already declares
  `age` as a reviewed ASEC leaf derived from `A_AGE`.
- `A_SEX` is absent from the **invented fixture's** parent frame, so
  `is_female` is ACS-only *on that fixture*. **Corrected 2026-09-06:** this was
  a property of the invented parent, not of source, and the original line here
  stated it without that qualifier. Genuine prepared ASEC bytes carry both
  `A_SEX` and `P_SEQ` — observed as metadata in the frozen raw-stage parent
  checkpoint (176 person columns) and in the CREATE `frame_context` of the
  genuine 505-household run (179 person columns). See
  `PROGRESS-us-composed-asec-review-fixes.md` and review finding code-1.
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
- [x] reproduced the accepted kernels' refusals on the composed fixture
- [x] implemented the typed binding node + the two measurement nodes
- [x] classification, inventory, docs, tests

## Next

Superseded. The Fable 5.1 review of `24f32366` and its four findings are
implemented on `us-composed-asec-review-fixes-20260906`; that branch's journal
is `PROGRESS-us-composed-asec-review-fixes.md`.
