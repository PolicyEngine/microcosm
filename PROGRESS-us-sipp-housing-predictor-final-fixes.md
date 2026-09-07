# US SIPP housing predictor — final metadata/docs fixes lane

## Scope

Bounded local source metadata correction, delegated from the root verification
lane at parent commit `35dcd20c1` (branch `us-sipp-housing-predictor-root-20260906`,
matching `origin/main` `c1b83241`). Source of the work: the reviewer's report
`/Users/maxghenis/architecture-reviews/microcosm-20260904/sipp-housing-predictor-final-fable51-capture.md`,
required findings 1-3 and low docs findings 4-5. No genuine microdata reads,
fits, graph runs, publication, push, or remote merge. No numerical/donor
changes; the frozen sixteen-field producer (`sipp_december_source.py`,
`sipp_measurements.py`) and the existing test file
(`test_us_sipp_housing_projection.py`) are untouched.

This lane is preparation only. Root verifies and writes proposal 4 and runs
the first-32 gate separately; nothing here executes an extraction.

## State

Done. All five findings addressed in code metadata/docs; recipe hash re-pinned
in the same commit; focused housing tests green; ruff clean.

## Done

1. **Finding 1 (required) — THCYINCPOV mislabeled as an income alternative.**
   Renamed `sipp_housing_predictors.py`'s `PREDICTOR_FIELDS_DELIBERATELY_ABSENT`
   values for `THCYINCPOV`/`AHCYINCPOV` and `sipp_housing_projection.py`'s
   `income` block key from `..._alternative_not_carried` to
   `..._income_to_poverty_ratio_not_carried`. Added one clause to the
   `THTOTINC` `PREDICTOR_LIMITATIONS` tuple stating THCYINCPOV is a ratio, not
   a dollar figure, and could not substitute for THTOTINC even if carried.
   Fixed the matching prose at doc line ~440 (now describes THCYINCPOV as a
   ratio). Re-pinned `RECIPE_SHA256` in the same commit (old
   `66ed8871ed1a1dbb73b8407c08c3e66dc8f4162ca91170c163ae8654cda4505e` → new
   `e8c809b2a1788d3b2016d03f6ac4b9f37b955bd7a7b855cd94372d7ab4ea1bc4`,
   computed from `recipe_bytes()` after all metadata edits, including finding 4's).
2. **Finding 2 (required) — doc named the wrong current extraction proposal.**
   Verified `us-sipp-housing-projection-extract-v3.py`'s real SHA-256
   (`5ec13bb2ee33c84bc0e6d497a1ebff74a2c22a104c6bb4a1ebdfa5ddd4d27ac2`) matches
   the value supplied for this task — confirmed by direct `shasum -a 256`.
   Rewrote the doc's "Extraction proposal" section: names v3
   (`sipp-december-housing-extraction-proposal/3`) as current, superseding v2;
   states the already-executed extraction was run by proposal version 3's
   script under recipe version 2 (disambiguating proposal-script version from
   recipe version, the doc's original conflation); states that a proposal
   version 4 is required before any recipe-version-3 extraction and must pin
   the new recipe hash, confirm the 38-column header, and keep the first-32
   gate unchanged.
3. **Finding 3 (required) — continuity precondition conflated the two gates.**
   Rewrote the doc's "Recipe identity and refusals" continuity paragraph to
   state that a first-32 extraction under recipe version 3 compares against
   the accepted **first-32** recipe-version-2 receipt's measurement digest,
   membership digest, and its own counts (33 households, 15 admitted over 32
   selected SSUIDs — confirmed from
   `us-sipp-housing-first32-root-output-review.json` in the review directory),
   and that the 17,646 / 4,348 / 4,057 figures belong only to the separately
   approved all-SSUID run. Explicit: clearing the first-32 gate implies no
   all-SSUID approval.
4. **Finding 4 (low) — TAGE_EHC "one contract" overstatement.** Added one
   sentence, in both `sipp_housing_predictors.py`'s `PREDICTOR_LIMITATIONS`
   and the doc, noting the frozen producer labels the out-of-domain reading
   `missing_item` (`sipp_measurements._number`'s own lexical class) while this
   housing kernel relabels the identical reading `invalid_domain` — confirmed
   by reading `sipp_housing_measurements.py:34,314,352-357` and
   `sipp_measurements.py:281`. Value and domain still agree; only the lexical
   label differs.
5. **Finding 5 (low) — agreement counter precedence.** Documented, without
   changing behavior, that `_agreement()` checks `roster_coverage_incomplete`
   before a publisher `copy_conflict`, so a conflicted value under incomplete
   roster coverage reports only as `roster_coverage_incomplete` in that one
   counter — the per-field class counters still carry the conflict separately.
   Added to the function's docstring and to the doc's household-size-agreement
   paragraph. No reordering of the `if` branches; no behavior change.

## Environment and verification

- Interpreter reused unchanged:
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python`
  (Python 3.14.4).
- Every subprocess purged foreign editable `microcosm.*` paths from `sys.path`
  (`.../packages/microcosm-*/src` entries) and re-inserted only this
  worktree's six shards, then propagated the same six-shard list via the
  `PYTHONPATH` env var — the pattern in
  `/Users/maxghenis/architecture-reviews/microcosm-20260904/us-sipp-housing-predictor-frozen-tests-r2.py`.
- `ruff check` and `ruff format --check` on both changed source files: clean.
- Focused test run — `packages/microcosm-build/tests/test_us_sipp_housing_projection.py`
  only, unmodified — 172 passed, 0 failed, exit code 0.
- `verify_recipe()` (implicitly, via the repinned `RECIPE_SHA256` matching a
  fresh `recipe_bytes()` hash computed after all metadata edits) succeeds; the
  full test file — which calls `project()` and asserts `recipe_bytes()` against
  `RECIPE_SHA256` in several places — passing is the behavioral confirmation.
- No numeric or donor value changed anywhere; only dict-key/value strings,
  docstrings, prose, and the one recipe-identity constant that those strings
  feed.

## What this lane did **not** do (explicitly out of scope)

- No genuine SIPP byte read, no extraction run, no proposal 4 written.
- No edits to `sipp_december_source.py`, `sipp_measurements.py`, or
  `test_us_sipp_housing_projection.py` — confirmed byte-identical via
  `git diff --stat` returning nothing for each.
- No edits to any file under `/Users/maxghenis/architecture-reviews/` (external
  previous reports) — those were read only, for verification.
- No broad test suite run — only the one housing test file.
- No claim that any historical test failure was reproduced or demonstrated on
  the base; none was attempted here.
- My own verification above ("focused tests pass locally, in this worktree, on
  this interpreter") is a self-check by the agent that performed the edits.
  It is not root's independent acceptance of this lane's work, and it is not a
  claim that the first-32 or all-SSUID gates have been (re-)run or approved —
  root performs that separately.

## Next

- Root reviews this diff against the five findings above.
- Root (separately) authors proposal version 4 pinning the new
  `RECIPE_SHA256` (`e8c809b2a1788d3b2016d03f6ac4b9f37b955bd7a7b855cd94372d7ab4ea1bc4`),
  confirms the 38-column header against the live parent, and runs the first-32
  gate under recipe version 3 — none of that is performed by this lane.
