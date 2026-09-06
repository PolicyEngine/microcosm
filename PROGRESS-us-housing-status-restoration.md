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

## Retry continuation (2026-09-06T07:20Z)

The first attempt of this assignment was cut off at 03:20 UTC when its Subfleet
lane hit a hard account limit (`us-housing-status-restoration-capture.lane.log`).
Its worktree, branch and five commits were preserved and are continued here, not
re-authored. Prior evidence in the shared review directory is left untouched.

Preserved state inherited from that attempt: the gap audit above, the packaged
authority resource closing G1, and the tests closing G2-G9. The attempt died
holding one uncommitted edit — the graph implementation inventory entry for the
new resource — which is why its affected-test capture recorded 42 setup errors
in `test_us_graph_asec_prepared`, all reading
`Unclassified US dependency/resource contract: .../asec_housing_status.py`.

That edit was verified against the actual module bytes before being committed:
`_dependency_contract` recomputed from the file now equals the declared contract,
and `implementation_manifest("asec_prepared_v3")` binds both housing modules and
the new resource. Committed as `Classify the housing status authority resource
access`.

## Done (this continuation)

- [x] Inspected the preserved worktree, branch, five commits and dirty file.
- [x] Re-derived and committed the inventory classification; 42 errors resolved.
- [x] Verified G1-G9 closure by reading every new test, not by trusting the journal.

## Adversarial audit (this continuation)

Eight independent dimension reviewers ran against the contract — classification
conformance, source authentication and the exact join, artifact immutability and
tamper checking, the non-mutating attachment, forbidden scope creep, test
meaningfulness, the packaged authority resource, and the graph seam. Every
finding was then handed to a refute-first verifier. Ten findings were raised;
four survived. Evidence:
`us-housing-status-restoration-continuation-audit.json`.

Clean with zero findings: source authentication/join, artifact immutability,
the non-mutating attachment, and the authority resource.

Three surviving findings were test-coverage gaps this assignment names, and are
now closed in `Cover relevant allocation, per-field zero origin and every
conflict bit`:

- A relevant `I_HPUBLI=1` never appeared in any fixture, so `public_quality`
  never took the `allocated` value anywhere in the repo.
- Three of the four zero-origin masks were unasserted, including the §3 row that
  requires preserving the routed-out `HLORENT` zero.
- Conflict bits 1, 2 and 16 were never checked against the published codebook.

Each new test was mutation-checked: ignoring the allocation flag, collapsing the
conflict bits to a bool, and sharing one zero-origin mask across all four fields
each make exactly the intended test fail. The pre-existing exact-key join,
unknown-reason and raw-retention tests were mutation-checked the same way.

The fourth surviving finding is a semantics question left for root, below.

## Open decision for root: group-quarters scope

`_derive` conditions `known`/`excluded` on `H_HHTYPE` and `H_TENURE` only.
HRHTYPE 9/10 feeds the informational `group_quarters` column, never a scope
mask, so an interviewed group-quarters renter classifies through the ordinary
table and can carry `receipt_valid=1`. Contract §3 asks the three context fields
to distinguish interviewed occupied housing units from "owners, group quarters,
noninterviews and unknown tenure", and §5 lists "known owner/GQ exclusions"; the
implementation honours the owner half and defers the GQ half.

This is disclosed rather than silent — the header declares
`non_housing_unit_scope: not_derived`, the doc states it, and
`test_group_quarters_is_scope_flag_not_added_receipt_exclusion` locks it. The
downstream `asec_housing_universe.py` does derive `occupied_hu` with GQ outside
the housing-unit universe, and `asec_prepared_source.py` composes it on the
attached frame, so the integrated path is covered; the standalone status
artifact is not. `asec_housing_observed_group_quarters` ships in the same
attached column set, so a consumer can apply it today.

This lane did not change it. The assignment routes owner/GQ/noninterview
semantics to the existing reviewed HU adapter, and the behaviour is locked by a
test the earlier independent review approved. Changing it is a semantics
decision for root, not a unilateral edit.

## Next

- [ ] Root review, then an independent Fable PR-style review.
- [ ] Root ruling on the group-quarters scope question above.
- [ ] Next rung: a tiny source-only genuine read, per the report's proposal.
