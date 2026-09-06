# US SIPP December housing predictor projection — lane journal

Worktree `_worktrees/microcosm-us-sipp-housing-predictor-projection-20260906`,
branch `microcosm-us-sipp-housing-predictor-projection-20260906`, based on
`5534a0fc155dfdfe3b399725ed44fbb9d42f859e` (the accepted housing projection head).

## Lineage

`origin/main` was fetched at lane open and stands at
`c1b8324183819cf91511c002bf724a2676524e87`. `git merge-base 5534a0fc origin/main`
is `c1b83241` itself and `git rev-list --count 5534a0fc..origin/main` is 0, so the
accepted head already contains the whole of maintained main and no merge is
needed. The local `main` checkout in the primary clone is 947 commits behind
`origin/main` and was not used as the lineage reference. No other worktree was
touched.

## Scope

Extend the December housing projection with the primary-source-authenticated
predictor observations the next unassisted contract-rent fit will need. Source
observations only. Explicitly out of scope in this increment: price bridge, fit,
draw, donor mask selection, model input binding, PolicyEngine calculation, and
any read of real microdata.

## State

- [x] Read root inputs (`housing-launch-method-decision/REPORT.md`,
      `us-sipp-housing-all-ssuids-root-output-review.json`,
      `us-survey-multispine-puf-clone-contract.md`), the existing source
      definition, projection, docs and tests.
- [ ] Re-check the nine field locators against the pinned dictionary PDF.
- [ ] Transcribe the nine contracts.
- [ ] Recipe/version/schema bump.
- [ ] Tests.
- [ ] Docs and changelog.
- [ ] Report.

## Standing corrections carried from root

The Fable memo is a proposal, not an adopted launch decision. Its section 2.3
comparison, its section 4 "carried" claim, its zero-rent characterisation, its
reading of numerical topcode markers as independent validation, and its claim
that new predictor exclusions preserve the old admission counts are not repeated
here.
