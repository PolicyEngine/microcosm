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

- [x] Gap audit against the contract and the assignment's enumerated test list.

## Audit result

The integrated slice already satisfies most of the contract. Thirty-one test
cases cover the classification table, the codec, artifact tamper, the
non-mutating attachment and exact parent recovery. Confirmed **open** gaps:

| # | Gap | Kind |
| --- | --- | --- |
| G1 | Questionnaire hashes and locators are not bound. Contract section 4 requires "dictionary/questionnaire hashes and locators"; only dictionary hashes and page indices are bound. | contract binding |
| G2 | Reordered-household exact join is untested. The `get_indexer` join is the core correctness claim. | test |
| G3 | Duplicate cohort native key (`NATIVE_KEY`) refusal is unexercised. | test |
| G4 | Missing scope key (`SOURCE_KEY_COVERAGE`) refusal is unexercised. | test |
| G5 | A cohort missing a required column is unexercised on the housing path. | test |
| G6 | External-link storage is refused only through the shared reader's own tests, never through the housing loader. | test |
| G7 | `unknown_reasons` — the contract's named required-answer-gap reason — is asserted nowhere. | test |
| G8 | Raw-code retention on conflict and unknown rows is unasserted, though the contract requires "retain the raw code in all cases". | test |
| G9 | No test proves the suite is blocked from genuine storage paths, or that loaded module paths and byte hashes belong to this worktree. | test |

## Graph seam decision: add nothing

The seam already exists downstream and is owned by other lanes.
`asec_prepared_source.py` calls `load_authenticated_housing_status` and
`attach_housing_status`, composes the housing universe on top, and issues the
`microcosm.us.asec_prepared_receipt.v3` receipt; `graph_asec_prepared.py`
declares the graph node and `graph_housing_universe.py` binds the transport,
keying on `parent_kind == "HousingStatusAttachedAsec"`. Adding another producer
or attachment seam here would duplicate that wiring and force the broad
stage/inventory rewrite the assignment forbids. This lane documents the existing
seam instead of re-declaring it, and touches none of those files.

All housing definition and implementation digests are live projections, never
frozen constants, so editing the housing modules re-derives every downstream
receipt rather than invalidating a pinned one. That is verified by running the
downstream prepared/universe/graph suites, not assumed.

## Next

- [ ] Close G1 with a hash-pinned packaged authority resource.
- [ ] Close G2-G9 with distinct-failure-mode tests, no permutation padding.
- [ ] Update the doc for the resource and the existing seam.
- [ ] Run affected + new tests, ruff, CI group inventory; durable JUnit/logs.
- [ ] Write `us-housing-status-restoration-REPORT.md` / `.json` and the receipt.
