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
- [x] Re-checked all nine locators against the pinned dictionary PDF
      (sha256 `0f1c5fbd…6db894f`, 3,991,831 bytes, 3,371 pages, verified by
      `shasum` before reading). Printed page numbers were read off the foot of
      each page, and the existing v2 locators were re-verified the same way.
- [x] Transcribed the nine contracts into
      `packages/microcosm-build/src/microcosm/build/sipp_housing_predictors.py`.
- [x] Recipe bumped to `sipp-december-housing-projection/3`, receipt format to
      `sipp_december_housing_projection_v3`, `SUPERSEDED_RECIPE_VERSIONS` gains
      `/2`, `RECIPE_SHA256` moved in the same commit, and the new module is in
      `IMPLEMENTATION_MODULES`.
- [x] Tests: 33 new contracts; 333 pass across the four SIPP test files.
- [x] Docs, changelog and `CLAUDE.md`.
- [ ] Report at `R/us-sipp-housing-predictor-projection-REPORT.{md,json}`.

## What the dictionary re-read changed

| Field | Printed page | Flag | Correction against the proposal |
| --- | ---: | --- | --- |
| `TAGE_EHC` | 797 | none (`NA`) | max 90; also the one column shared with the frozen producer |
| `RHNUMPER` | 891 | `AHNUMPER` | Type-2 exclusion is read from `RHNUMPERWT2`, not stated on the field |
| `RHNUMU18` | 892 | `AHNUMU18` | — |
| `THTOTINC` | 2866 | `AHTOTINC` | signed monthly; the kernel's single sentinel sat inside its domain |
| `TEHC_ST` | 3311 | `AEHC_ST` | 53 published codes, not 51 and not the range 01–56 |

## Not done, by scope

No price bridge, fit, draw, donor mask selection, model input binding or
PolicyEngine calculation. No real microdata read and no version 3 extraction
proposal written. Metro fields skipped. The active composition integration and
the `sources/`/output directories were not edited.

## Standing corrections carried from root

The Fable memo is a proposal, not an adopted launch decision. Its section 2.3
comparison, its section 4 "carried" claim, its zero-rent characterisation, its
reading of numerical topcode markers as independent validation, and its claim
that new predictor exclusions preserve the old admission counts are not repeated
here.
