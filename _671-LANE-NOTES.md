# microcosm#671 lane notes

## State

- Branch: `adult-care-clone-alignment-671` from `d185c8a7`.
- Phase 1 only: code, tests, and operator-ordering documentation. No pipeline
  builds have been or will be run in this lane.
- Implementation arm: train the bounded `person/adult_care` late-transfer
  family from the ASEC-origin clone-0 owner projection used by its by-origin
  battery comparator. Other late-transfer families remain on clone 1.

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

## Next

- Add one canonical adult-care donor/comparator clone declaration and route
  both the late model-config receipt and runtime donor projection through it.
- Add positive alignment and negative mismatch-refusal tests.
- Document the coupled flag -> expense -> reconciliation ordering contract.
- Run the requested focused suites, `test_us_plan.py`, touched-package pytest,
  and Ruff; then review impact, commit small `#671` changes, and leave clean.
