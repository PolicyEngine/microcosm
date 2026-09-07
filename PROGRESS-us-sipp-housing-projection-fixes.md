# US SIPP December housing projection — review-fix lane journal

Branch `microcosm-us-sipp-housing-projection-fixes-20260906`. Worktree created
from fetched `origin/main` `c1b8324183819cf91511c002bf724a2676524e87` and
fast-forwarded to the reviewed head
`725d9c54d5436f64ec2c49e9ccab496c1bfede1b` (`origin/main` is an ancestor of that
head, so the fast-forward is exact). The author worktree
`_worktrees/microcosm-us-sipp-housing-projection-20260906` is untouched.

Scope: close S1–S6 from the independent Fable 5.1 review of the December SIPP
housing projection, with consequential tests and accurate docs. No genuine
source extraction, no fit, no transfer, no publication, no push.

## State

Complete and root-reviewable. S1–S6 closed, verified, documented. Nothing was
pushed and no PR exists; no genuine extraction was run and no SIPP byte was
opened. The report is `us-sipp-housing-projection-fixes-REPORT.md`/`.json` under
`/Users/maxghenis/architecture-reviews/microcosm-20260904/`.

## Done

- Read the review context, the Fable 5.1 capture and the root r2 reproductions;
  read both housing modules, the frozen internals they reuse, the tests, the doc
  and the v1 proposal and self-check. Confirmed hooks disabled
  (`core.hooksPath=/dev/null`) and a clean head at the reviewed commit.
- Baseline: 240 tests passed across the four SIPP files before any change.
- **S1** `_voucher_evidence()` grades EVOUCHER's preserved observations, because
  `measurement_class` reports the flag once a value is discarded. An explicit
  `ERENTSUB = 2` now admits only `expected_blank_not_in_universe`; copy
  conflict, suppressed, `answer_present` and `applicable_without_answer` are
  retained as their own classes, and anything unreadable is
  `assistance_voucher_unresolved`. Owners' conflicting or withheld evidence
  becomes `owner_assistance_evidence_unresolved` instead of being called
  out-of-universe. Imputed negatives still never become explicit.
- **S2** New `us-sipp-housing-projection-extract-v2.py` and
  `us-sipp-housing-projection-extract-v2-selfcheck.py` under the review root; v1
  files left byte-identical as history. Continuity refuses before any write,
  output collisions refuse before the parent is opened, `--no-snapshot` is
  removed, an available-memory preflight runs before a guard-bracketed copy, and
  `canonical_json` replaces `default=str`. The `all_ssuids` gate is unchanged.
- **S3** `selection_evidence()` counts distinct SSUIDs, digests the sorted set,
  refuses more than 32 under `first_32_ssuids`, and records
  `source_authentication: caller_asserted` with the note that a count is not
  proof of a relative selection. `source_identity()` records the checked pins
  separately, with no byte proof claimed from supplied rows.
- **S4** `implementation_binding()` binds both housing modules and both frozen
  modules at receipt level, outside `recipe_bytes()`; the frozen modules stay
  byte-identical, verified against the reviewed head.
- **S5** Zero, invalid-domain, blank and absent topcode-marker records counted
  per household and aggregated, with the $0 `TUTILS_*` versus $1 `TRNTMT_*`
  minimum documented and no zero scored as a cleared non-topcode.
- **S6** 32 new tests, including all four S1 triggers, November filtering, a
  two-residence SSUID, `owner_debt_absent`, no member receipt in an assisted
  household, `ETENURE = 3` with a reported `ERENTSUB = 2`, and a renter whose
  `ARENTMORT = 0`. The exclusion-ladder test now asserts every declared rung is
  reachable.
- Cheap review corrections taken: `target_year` threaded into the frozen roster
  call, `assistance_not_in_universe` given its own rung, owner debt reported as
  *indicated* with its evidence classes, the receipt doc corrected to "no
  individual dollar amount", and the straight-quote transcription note added.
- Recipe bumped to `sipp-december-housing-projection/2`, receipt format to
  `sipp_december_housing_projection_v2`, `RECIPE_SHA256` repinned to
  `e878b88f…1c80`. No version 1 receipt exists.
- Re-ran the root r2 reproductions against the fix: all four erroneous
  admissions now refuse, the expected blank stays admitted, and 40 SSUIDs
  labelled `first_32` refuse with `SELECTION_EXCEEDS_FIRST_32`.

## Verification

- 275 passed across the four SIPP files (240 before; the housing file went from
  70 to 105 tests). JUnit and logs under
  `us-sipp-housing-projection-fixes-work/`.
- `tools/ci_test_groups.py --verify` = `verification=ok`; `ruff check` clean over
  `packages/`, `docs/`, `tools/` and both new proposal scripts.
- The proposal version 2 self-check passes every assertion, including a
  continuity refusal that leaves the output directory empty and an
  `OUTPUT_EXISTS` refusal reached before a nonexistent parent is touched.
- Every module in the frozen producer's `_contract()` binding closure is
  byte-identical to the reviewed head, so already-published vehicle-equity
  receipts keep their producer binding.
- 55 loaded `microcosm.*` module files, all inside this worktree and all
  matching `HEAD`.

## Next (for root, not this lane)

- Review this lane, then run
  `us-sipp-housing-projection-extract-v2.py --execute` for `first_32_ssuids`.
- Read the topcode-marker state counts in that output **before** interpreting
  any marker total; the real incidence of zero utilities markers is unknown.
- Only after reading that receipt and diagnostics, consider the gated
  `all_ssuids` rung, whose approval flag is unchanged.
