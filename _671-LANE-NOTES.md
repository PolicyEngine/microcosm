# microcosm#671 lane notes

## State

- Branch: `adult-care-clone-alignment-671` from `d185c8a7`.
- Phase 1 code, tests, documentation, and local validation are complete in
  `3feab98c`, `1fe4f961`, and `69e2ded3`. No pipeline builds were run.
- Implementation arm: train the bounded `person/adult_care` late-transfer
  family from the ASEC-origin clone-0 owner projection used by its by-origin
  battery comparator. Other late-transfer families remain on clone 1.
- The implementation and compatibility-path scope have been independently
  re-reviewed with no remaining blockers. The owner retains phase 2, rebase,
  push, PR, and issue-receipt work.

## Done

- Read `CLAUDE.md` and confirmed the PR-CI/certification boundary and root
  journal rules.
- Verified the charter against
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/reviews/sol_r14_diag.log` and
  the read-only 10% checkpoint receipts under
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/dev-f010-r7/`:
  clone-1 donor 30 carriers / $15,000 median, clone-0 comparator 37 carriers /
  $3,000 median, raw ACS-to-clone-1 envelope 0.142851, final ACS-to-clone-0
  envelope 1.836066, and reconciliation 1,731 -> 421 via 818 false-flag,
  491 role-structure, and one ranking removal.
- Traced the late-transfer owner declaration, its hash-bound virtual-resource
  receipt, the clone-1 runtime projection, and the clone-0 battery scope in
  `stacked_spine.py`.
- Chose the charter's same-clone-owner arm because the adult-care group is
  already a bounded two-target producer, so a group-specific donor projection
  changes only that family and downstream CDCC values. A common
  post-reconciliation surface would require a broader comparator/gate contract
  change without evidence that it is necessary for this alignment fix.
- Added the canonical
  `US_ADULT_CARE_DONOR_COMPARATOR_CLONE_INDEX = 0` declaration. The hash-bound
  late-producer schedule now carries every group's donor owner, with adult care
  on clone 0 and all other groups pinned to the existing clone 1.
- Routed the late-transfer model-config binding and the actual runtime donor
  projection through that schedule-bound group declaration. Registry schema
  17 / execution receipt schema 4, the stacked authority, resource semantics,
  bank identity, and aggregate receipts all bind the selected clone leg.
- Added fail-closed validation for live-group/schedule drift and for either
  adult-care target disagreeing with its battery comparator. Positive tests
  bind both targets across the group, schedule, authority, model config,
  comparator registry, and observed donor frame; negative tests refuse both a
  forged clone-1 declaration and a runtime group rebound away from the signed
  schedule.
- Preserved the exported whole-surface compatibility API as two scoped legs:
  adult care alone uses clone 0, while the unchanged 68-target remainder stays
  in one clone-1 evaluation. Its 69-model-target bank indexes remain 0–1 and
  2–68, its receipt explicitly claims only those two legs, and no-op legs
  treat a `None` resolved channel as neutral. The production 19-group DAG is
  unchanged.
- Documented the coupled adult-care operator contract in
  `docs/us-multispine-operator-ordering.md`: raw flag draw, expense conditioned
  on that draw, then false-flag/role/ranking reconciliation, all against the
  shared clone-0 donor/comparator owner.
- Coherent live receipt values after the declaration change:
  schedule SHA-256
  `b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5`,
  schedule-payload SHA-256
  `d351b87c43ae2d6a8ece68285507e94cc4a6285a1e1edeebdf2a049f9049b37f`,
  and stacked-authority SHA-256
  `b4266a98a26ac6808b68cb68aff966540a434454ef615f745fc486e2acf6c378`.

## Validation receipts

- `UV_CACHE_DIR=/private/tmp/microcosm-671-uv-cache uv run --no-sync pytest packages/microcosm-build/tests`
  -> **5,364 passed, 32 skipped** in 1,441.67 seconds.
- `UV_CACHE_DIR=/private/tmp/microcosm-671-uv-cache uv run --no-sync pytest packages/microcosm-build/tests/test_us_plan.py packages/microcosm-build/tests/test_us_stacked_spine.py`
  -> **267 passed** in 170.53 seconds.
- Final focused compatibility/bank regression selection -> **4 passed**,
  including adult-only active, remainder-only active, and all-no-op receipts.
- `UV_CACHE_DIR=/private/tmp/microcosm-671-uv-cache uv run --no-sync ruff check .`
  -> **All checks passed**.
- The managed sandbox refused `uv`'s default cache under `~/.cache/uv`, so the
  commands used a writable temporary cache and `--no-sync` with the already
  synced workspace environment. No dependency installation or network access
  occurred.
- No pandas string-storage (`str` dtype) mismatch occurred, so there is no
  expected-fixed-by-#672 failure to record.

## Next

- No phase-1 code, test, documentation, or local-validation work remains.
- Owner: rebase onto the mainline carrying #672 at PR time without folding that
  sibling fix into this lane, then run the chartered 4%/10% battery pair after
  the occupied full-scale build row reconciles.
- Owner: review the phase-2 artifact evidence, push the branch, open the PR,
  and post issue receipts. This lane did not push or run a pipeline build.

## Owner note (2026-08-13 ~23:40 ET, supersedes the dispatch prompt on one point)

The dispatch prompt says the prior work is uncommitted. That is now stale:
the owner committed the accumulated two-lane diff as WIP checkpoint
`3feab98c` ("WIP: adult-care donor/comparator clone alignment (#671)") to
protect it from harness restarts. So: `git diff` is clean; the prior work is
IN HEAD. Continue from HEAD — do not re-derive, do not revert the WIP
commit. Inspect it with `git show 3feab98c` and finish the Next list above
(validation has still never run). Further commits go on top; small and
message-style-matched as instructed.

Historical-status note: those continuation instructions were fulfilled by
`1fe4f961`, `69e2ded3`, and the validation receipts above; the statement that
validation had never run is no longer current.
