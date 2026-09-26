# #901 re-based onto main: receipts

Date: 2026-09-25. Branch `uk-full-build-graph-registration`, new history on `origin/main`
8f628b1e7 (main after #1006). Old head kept locally as the tag `uk-901-pre-main-rebase`
(2da4f421, stacked on #893's 2300e56b). Plan: `repos/uk-901-main-rebase-plan.md` (approved
2026-09-25); audit: `repos/uk-901-rebase-to-main-audit.md`.

## R0. What was ported and what was not

- Nothing from #893's shared code was needed by #901: the graph-kernel amendments the UK graph
  binds (platform-bitwise scope, keyed seeds, artifact inputs, rewrite ownership, the raw-bytes
  codec, the typed-artifact scope rule) were on main before #893 branched (merge-base 15ebde806).
- Folded in: Max's #918 (17 commits, cherry-picked with `-x`, authorship kept), renumbered from
  graph amendments 25/26 to 26/27 because main recorded the observer opt-in (#950/#951) as 25;
  `docs/graph-interface.lock` re-recorded (decl.py `b25ae4a6…`, kernel.py `24045ab2…`).
- #901's own shared additions (country-agnostic): `artifact_files`, `stage_evidence`, the
  gate-phase payload codec and `record_phase`, `calibrate.artifacts`,
  `calibrate.target_selection`, the `TargetSpec` dict codec.
- Left with #893: table identity, survey allocation, the CD benchmark, the legacy-QRF fit family,
  grouped bounds and target snapshots, the frame-checkpoint v4 schema, two performance fast paths.

## R1. Phase-1 spine consolidation: licensed A/B

Inputs identical on both sides: FRS 2024-25 raw tables, SPI 2022-23 PUT, HMRC collated tables
2023-24, WAS round 8, LCFS 2023-24 household and person, ETB 1977-2024, the five NTS tabs; no
sampling; `--no-staging`; checkpoints on. Script and outputs:
`data/ukds/acceptance/901-rebase-ab/` (`build_ab_spine.sh`, `spine-a/`, `spine-b/`, logs).

- A = main 8f628b1e7, `tools/build_uk_frs_spine.py`; B = branch 80f706050, the six-line shim over
  `uk_runtime/spine_build.py`.
- Both runs block at phase `transferred` on `uk_stage_student_loans_realization` (PLAN_5
  realization_deviation 1.1021725784754537 exceeds 1.0). That gate is main's own limitation at
  this head, recorded by #1006; neither side reaches an H5. Wall: A 284 s, B 308 s.
- The two 26-gate reports (`spine-a.spine_gates.json`, `spine-b.spine_gates.json`) are identical
  after removing the timestamped `release_id` and the signature that covers it: same gates, same
  outcomes, same evidence digests, same `blocked_at_phase`.
- The two content stores hold 8,076 stored tables each (`values.npy`); the multiset of their
  bytes is identical, with no table unique to either side. Node keys differ, as declared: the
  artifact declarations on the spine nodes move every key.
- Owed: the full-H5 A/B once a spine can be built at main's head, and the dense and national
  parity builds through the graph driver (see R4).

## R2. Phase results and test counts (JUnit-counted)

- Commit 2 (renumber, re-lock): `packages/microcosm-graph/tests` 690 passed / 1 skipped, twice
  (before and after the renumbering); `tools/graph_acceptance_burndown.py --verify` ok.
- Commit 3 (shared additions): 66 passed (new suites, gate-register pins, registry codec).
- Phase 1 (spine consolidation, 80f706050): targeted 82 passed; `spine-uk` group 2,825 passed /
  21 skipped; H2 parity 1/1 after regeneration (oracle `edb1659b…`).
- Phase 2 (graph build beside the drivers, 3df839f48): targeted 367 passed; whole `uk` group
  2,599 passed / 20 skipped; lock-hash test 24 passed; spot-check 203 passed.
- Phase 3 (dense role, 9bc39f8aa): whole `uk` group 2,654 passed / 20 skipped; role files 214
  passed; both drivers' `--help` exit 0.
- Phase 4a (national dispatch, 142268fa2): whole `uk` group 2,571 passed / 20 skipped;
  90 + 140 + 145 targeted.
- Phase 4b (HMRC tail retirement, ffe6ed7d2): retirement files 458 passed / 2 skipped;
  whole `uk` group 2,549 passed / 20 skipped; `shared-spec` 2,100 passed / 48 skipped;
  microcosm-data contract 254; H2 3 passed / 1 skipped; spot-check 337 passed / 1 skipped.
- Registry re-point (7eaf6ff3d): `test_frame_serializer_registry` green.
- Derived surfaces regenerated with their tools, never pasted: H2 spine parity fixture
  (`tools/graph_uk_spine_fixture.py`), release-input coverage manifest (`--check` current, 145
  required inputs, `source_stages.json` 73c3a14f…); gate-register digests did not move.
  `uv.lock` relocked; `APPROVED_UV_LOCK_SHA256` = `3bd27a6c…`.

## R3. Behaviour changes recorded for review

- A blocked *assembled* spine gate now fails inside `run_graph`, so the gate report lives in
  the content store on that path (main wrote `spine_gates.json` before raising); a blocked
  *transferred* gate still writes the file (R1).
- `numerical_dependencies` pins installed versions into the H2 fixture.
- `_normalise_uk_local_bound_families` accepts an empty declaration (country-only scope).
- Per-epoch `calibration_progress` staging rows come from the dense solve only.
- The HMRC family names `spi_income_band_donors` (#1006) as a predecessor and admits its two
  operation kinds; the contract otherwise refused the drifted operation order.
- `tools/build_uk_rowwise_dataset.py` stays main's (its tests load it by path; it still serves
  `--candidate-clone-counts`).

## R4. Blocker for the dense and national licensed parity

The graph driver refuses every existing acceptance spine at its strict checkpoint gate
(`gates_manifest_sha256 differs from current declarations`), by #901's design, and a fresh spine
cannot be built at main's head (R1). Options put to María on 2026-09-25: wait for the #1012
lane to settle main's fence; a non-release measurement flag that accepts a stale gate manifest
and records the mismatch; or a measurement-only branch over #1012. The national role is parity
by identity regardless (same engine, moved code); the dense comparison needs a spine.
