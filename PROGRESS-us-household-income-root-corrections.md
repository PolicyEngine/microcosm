# US household reported income — root corrections lane journal

Bounded successor correction for the completed household publisher-income
lane, closing the three findings root's canonical review of frozen
`0a2cb7dc` plus its follow-up note raised against the author's successor head
`da7fe1d4`. Source records: root's full report
`/tmp/policyengine-command-runs/d1d611a3749e/household-income-0a2cb7dc-review-full-report.md`
and the follow-up
`/Users/maxghenis/architecture-reviews/microcosm-20260904/us-household-reported-income-root-review-followup.md`.

Author worktree `microcosm-us-household-reported-income-20260906` stays
**clean at `da7fe1d42969a9c61c6952f31f3a52bc92b19adf`** throughout — an
earlier attempt in this session mistakenly edited it directly; the diff was
captured as a patch, the author worktree was reverted with `git checkout --`,
and the patch was replayed here instead. Branch
`microcosm-us-household-income-root-corrections-20260906`, cut from fetched
`origin/main` (`c1b8324183819cf91511c002bf724a2676524e87`), then fast-forwarded
to `da7fe1d4` (`origin/main` is an ancestor of it; nothing was rewritten).

## State

In progress. Investigation complete; the three corrections are about to be
implemented as separate commits.

## Done

- Read root's full review report and follow-up note, this repo's `CLAUDE.md`,
  and the source files each finding names:
  `household_reported_income.py`, `asec_housing_universe.py`,
  `asec_household_observations.py`, `asec_current_money_price_basis_v1.json`,
  and the classification allowlists in `test_us_spine_blindness.py`.
- Confirmed the workbook's own `sha256`/`size_bytes` (distinct from the
  packaged JSON resource's digest) already live in the immutable
  `asec_current_money_price_basis_v1.json` `workbook` object.
- Confirmed `asec_housing_universe.py` classifies `HRHTYPE` 9/10 as group
  quarters over the same prepared ASEC household table that carries
  `asec_HTOTVAL`, and that `with_asec_household_observations` retains the raw
  `asec_H_LIVQRT`/`asec_HRHTYPE` fields that classification keys on —
  confirming the successor's "only ACS carries GQ records" claim is false.
- Confirmed both modules this lane's parent branch added
  (`household_reported_income.py`, `acs_household_source_columns.py`) are
  already present in `_OTHER_US_RUNTIME_MODULES` in
  `test_us_spine_blindness.py`, with no entry in
  `_SPINE_BLIND_OPERATOR_MODULES` or `_SOURCE_SPINE_PROVENANCE_OWNERS`.
- Created the fresh child worktree from fetched `origin/main`, fast-forwarded
  to `da7fe1d4`, and verified the author worktree's independence.
- `uv sync --all-packages` in the fresh child worktree (no `.venv` existed
  yet); resolved from the shared uv cache in ~1.2s.

## Next

1. Correction 1: pair the ASEC price-basis authority entry's URL with the
   workbook's own digest, keeping the packaged resource's digest separately
   labelled; add a focused regression.
2. Correction 2: correct `two_arm_comparability`'s false GQ-coverage claim and
   the matching doc section; add a stable declaration of the required
   ASEC housing-universe-mask binding plus a regression.
3. Correction 3: re-run the module-classification test to confirm no fix is
   needed (already verified above), and record that explicitly.
4. Run only the affected household-income tests, the two narrow
   classification tests, ruff, and the CI test-group inventory verification —
   no whole-workspace or build suites.
5. Close this journal with final state/verification and write the two report
   files.

## Limits held throughout

- No push, merge, publication, or CI trigger.
- No genuine ACS/ASEC microdata opened; every test uses tiny invented pandas
  inputs, matching the existing file's own convention.
- No new mask API and no out-of-universe rule to be added to either
  measurement function.
- No provenance exemption to be added to the classification allowlists.
- Author worktree and every other listed worktree/environment left untouched.
