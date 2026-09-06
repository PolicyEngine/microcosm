# US housing receipt consumer — review fix lane

Session-handoff journal. Accurate when written; check git for current truth.

Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-housing-consumer-fixes-20260906`
Branch: `us-housing-consumer-fixes-20260906`
Base: fetched `origin/main` = `c1b8324183819cf91511c002bf724a2676524e87`, fast-forwarded
to the reviewed head `ffbef26eaba498028b64d646e369651999c71900` (original lane worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-housing-consumer-20260906`,
merge base `ab10f77854b7b9e8057a07dbec40d15c10b96297`, left untouched).

Fixes the root+Fable review at
`/tmp/policyengine-command-runs/81a7c43a2f3b/housing-consumer-ffbef26e-review-full-report.md`
(peer capture
`/Users/maxghenis/architecture-reviews/microcosm-20260904/us-housing-consumer-fable51-capture.md`).
The full write-up lives in
`/Users/maxghenis/architecture-reviews/microcosm-20260904/us-housing-consumer-fixes-REPORT.md`
with its machine-readable twin `-REPORT.json` and the lane journal `-PROGRESS.md`.

## State

Complete at `2103cedd2208e33b48ba8af12421c409f3d54936`. One focused invocation,
exit 0: 99 passed, 0 failed, 0 skipped. Nothing pushed, merged or published.

## Done

- [x] Reproduced the reviewed baseline in this worktree: 69 passed, 3 failed,
      matching root's 72/3. Located the AST offences at consumer `197:68` and `474:83`.
- [x] **C1** Interpretation carries `household_id` and `income_year`;
      `resolve_spm_receipt_scope` refuses any caller axis that is not the artifact's own
      axis in order (`CALLER_HOUSEHOLD_IDENTITY`). Person and SPM-unit permutation stays
      valid. The test that codified the defect is replaced by refusal and permutation
      cases.
- [x] **C2** Both new `us_runtime` modules classified as ordinary non-registry modules;
      the two dynamic-selector subscripts cleared with an annotated module-level sort
      key. No guard weakened. All five named guards pass.
- [x] **C3** `receipt_columns_from_attached_household` takes a validated
      `HousingStatusAttachedAsec`, derives every provenance field from it, and refuses
      a plain DataFrame. The forged-label parameters are removed. The `uint8` columns
      are required, not cast. The synthetic path stays available and honest, and the
      positive attachment test reuses the existing invented-byte source fixture.
- [x] **A1** `resolve_spm_receipt_scope` requires bound occupied-HU evidence from the
      typed `asec_housing_universe` artifact or attachment, matched on households,
      cohort and authentication. Group quarters, unestablished units and household-kind
      conflicts take their own withheld classes; the raw receipt view is untouched;
      `distribute` moves only bound-HU negatives. No default, no waiver.
- [x] **A2** `docs/us-housing-counterfactual-charge-requirements.md` added and named by
      the runtime record as a repository-relative path plus a symbolic dependency,
      labelled a required future measurement with `measured_here: False`.
- [x] Practical items: read-only stored arrays, single artifact validation,
      codebook-bound diagnostics with unnamed-bit refusal, preserved routing/quarters/
      validity/zero-origin/scope context, clarified take-up and utility-reimbursement
      claims, portable doc links, renamed mislabelled probe test.
- [x] Replayed root's three reproductions against the fix: all fail closed.

## Not done, deliberately

No engine formula change. The HUD 2024 data correction, utility allowance, bedrooms,
shared-SPM and the LA payment-standard schedule stay with their own lanes. No original
microdata, no fit, no target activation, no push, merge, deploy or publication. The
counterfactual full contract charge remains unavailable; this consumer produces none.

## Next

Root arranges the independent Fable incremental review of this head.
