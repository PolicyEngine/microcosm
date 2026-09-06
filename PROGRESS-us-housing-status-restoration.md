# US ASEC housing-status restoration — lane progress

Branch `microcosm-us-housing-status-restoration-20260906`, dependent worktree
created from the frozen integration base `ab10f77854b7b9e8057a07dbec40d15c10b96297`
(`us-launch-integration-20260906`). Upstream `origin/main` was fetched at
`c1b8324183819cf91511c002bf724a2676524e87`; the base is 124 commits ahead of it
and zero behind, so this lane is a clean descendant of upstream main.

Assignment: build the bounded source-only ASEC housing-status restoration
specified in `us-housing-receipt-restoration-source-contract.md`. No genuine
value read, no fit, no engine run, no publication, no push, no merge.

## State

**The contract's implementation already exists in the frozen base.** The
September 5 lane `us-housing-observed-status-20260905` landed
`asec_housing_status.py`, `asec_housing_status_source.py`,
`docs/us-asec-housing-status.md` and two test files at commit `0d132480`;
the ownership correction `1ae0d347` followed. An independent review requested
changes (three critical ownership defects), and the follow-up owned-axes review
APPROVED the corrected head `4686e2ea`. Both are ancestors of this base.

This lane therefore does **not** re-author the module. It audits the integrated
implementation against the contract on a fresh dependent worktree, closes the
residual gaps this assignment names, and issues the new report. Prior attempt
evidence in the shared review directory is preserved, not overwritten.

## Done

- [x] Inspected base git status/branch/remotes; base worktree clean, untouched.
- [x] Fetched `origin/main`; recorded the ancestor relation.
- [x] Created the fresh dependent worktree and branch.
- [x] Read `AGENTS.md`, `CLAUDE.md` and the full source contract.
- [x] Read the prior author report and both independent reviews; confirmed the
      three critical findings are resolved in this base.
- [x] Established this progress journal.

## Next

- [ ] Gap audit: enumerated contract/assignment test list vs. integrated tests.
- [ ] Close real gaps only; no duplicate re-implementation.
- [ ] Graph seam: reuse-only decision, else document the concrete declaration.
- [ ] Run affected + new tests, ruff, CI group inventory; durable JUnit/logs.
- [ ] Write `us-housing-status-restoration-REPORT.md` / `.json` and the receipt.
