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

Complete and green. Three commits beyond the fast-forwarded `da7fe1d4` base:
the journal open, correction 1, and correction 2 (correction 3 needed no code
change — see below). Exactly four files differ from `da7fe1d4`: this journal,
`docs/us-household-reported-income.md`,
`packages/microcosm-build/src/microcosm/build/us_runtime/household_reported_income.py`,
and `packages/microcosm-build/tests/test_us_household_reported_income.py`.
No other lane's file touched. No push, merge, publication, genuine input, or
target activation. Author worktree
`microcosm-us-household-reported-income-20260906` reconfirmed clean at
`da7fe1d42969a9c61c6952f31f3a52bc92b19adf` after this lane finished.

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

- **Correction 1 (A3 / price-basis identity), commit `fd1acc85`.** Added
  `_ASEC_PRICE_BASIS_WORKBOOK_URL/_SHA256/_SIZE_BYTES` read from the existing
  immutable `workbook` object in `asec_current_money_price_basis_v1.json`.
  `ASEC_HTOTVAL_ORIGIN`'s final authority entry now pairs that URL with the
  workbook's own `sha256`/`size_bytes`, and keeps the packaged JSON's own
  identity under explicit, separate `resource`/`resource_sha256` keys — the
  same field names the module's evidence payload already uses. No numeric
  value or algorithm changed. Added
  `test_asec_price_basis_authority_pairs_url_with_workbook_digest`, which
  reads the real packaged resource, hashes it independently, and asserts the
  two digests are distinct and each correctly labelled.
- **Correction 2 (successor `two_arm_comparability` GQ claim), commit
  `28a15368`.** Rewrote the "household universe" difference's `consequence`
  to distinguish file record coverage (both files can carry GQ-flagged rows)
  from measurement universe (only the ACS arm's own dictionary declares an
  N/A pattern, so only it derives an out-of-universe code); preserved the
  "All Households" publisher wording and the true ACS GQ evidence. Added a
  stable, exported declaration — `ASEC_MEASUREMENT_REQUIRES_INDEPENDENT_HU_MASK`,
  also returned as `two_arm_comparability(...)["asec_universe_binding"]` —
  stating that the ASEC arm's raw total does not by itself establish
  occupied-HU membership and that a B19001-style consumer must bind the
  independently carried HU mask first. No out-of-universe rule was added to
  either pure measurement function and no mask API was introduced; this only
  documents the existing gap. Updated `docs/us-household-reported-income.md`
  with a new "File record coverage versus measurement universe" subsection
  and a cross-reference from the `AVAILABILITY`/`QUALITY` paragraph. Added
  `test_two_arm_comparability_declares_the_asec_hu_binding_requirement`,
  which asserts the stable constant (not prose matching) and regresses the
  retracted false claim string.
- **Correction 3 (module classification).** Verified, no fix needed: both
  modules this lane's parent branch added
  (`household_reported_income.py`, `acs_household_source_columns.py`) are
  already present in `_OTHER_US_RUNTIME_MODULES` in
  `test_us_spine_blindness.py`, neither is in `_SPINE_BLIND_OPERATOR_MODULES`
  or `_SOURCE_SPINE_PROVENANCE_OWNERS`, and
  `test_us_runtime_module_classification_is_complete` reports zero
  unclassified/stale/overlap entries both before and after corrections 1-2.
  No classification edit was made; no provenance exemption was granted to
  either module. Confirmed with a direct script read of the test module's own
  allowlists, then with the real pytest run recorded below.
- Final verification, all green (exact commands/counts/receipts in
  `/Users/maxghenis/architecture-reviews/microcosm-20260904/us-household-income-root-corrections-REPORT.md`):
  - `pytest packages/microcosm-build/tests/test_us_household_reported_income.py`
    → **47 passed** (45 pre-existing + 2 new), run after each correction and
    again at the final head.
  - `pytest test_us_spine_blindness.py::test_us_runtime_module_classification_is_complete
    test_us_spine_blindness.py::test_runtime_classification_rejects_a_new_unreviewed_module`
    → **2 passed**.
  - `ruff check` and `ruff format --check` on both changed Python files →
    clean at every commit.
  - `tools/ci_test_groups.py --verify` → `verification=ok`; both changed test
    files still appear in their expected groups, never `[defaulted]`.
- Did **not** run the whole workspace or build suites, per the explicit
  instruction that two prior author-wide checks already cost 15+ CPU-minutes
  against 45 focused + 729 already-green neighbours.

## Next

- Root and the requested Fable/independent reviews of this correction.
- No further implementation is expected in this lane; nothing here changed
  numeric values, the ASEC/ACS measurement algorithms, or the classification
  allowlists' membership.

## Limits held throughout

- No push, merge, publication, or CI trigger.
- No genuine ACS/ASEC microdata opened; every test uses tiny invented pandas
  inputs, matching the existing file's own convention.
- No new mask API and no out-of-universe rule added to either measurement
  function.
- No provenance exemption added to the classification allowlists.
- The author worktree was briefly edited in error. Those edits were captured,
  removed with `git checkout --`, and replayed in the correction worktree. The
  author worktree was verified clean at `da7fe1d4` before and after; no other
  listed worktree or environment was changed.
