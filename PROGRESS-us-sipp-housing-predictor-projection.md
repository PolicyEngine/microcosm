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
- [x] Tests. Counted at this head, not carried from an earlier draft:
      `test_us_sipp_housing_projection.py` holds **128** test functions and
      collects **171** tests, against **79** functions at the accepted base, so
      **49** are new. Across all eight `test_us_sipp_*.py` files: **389 passed,
      1 skipped**. An earlier version of this file said 33 and 333, and the
      first report draft said 42 and 382; both predated the last commits and
      neither is the number at this head.
- [x] Docs, changelog and `CLAUDE.md`.
- [x] Report at `R/us-sipp-housing-predictor-projection-REPORT.{md,json}`.
- [x] Independent Fable 5.1 peer review of the increment, and the fixes it
      found (below).

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

## Peer review and the fixes it produced

One bounded read-only Fable 5.1 peer reviewed the increment at
`662c8169` against the accepted base. Its output is at
`R/sipp-housing-predictor-fable51-capture.md`. Nine findings; each was checked
against the code before acting on it, and one was not adopted.

| # | Finding | Verdict | Commit |
| --- | --- | --- | --- |
| 1 | Version 3 does **not** write a different measurement digest | confirmed by running base and head over the same rows | `7b5589f0` |
| 2 | Report §10 authorised a per-origin model, which root forbids | confirmed against the root adjudication | report only |
| 3 | Working tree differed from the reviewed head; three failures unexplained | tree now clean; the three are pre-existing | see below |
| 4 | Stale and contradictory documentation (four items) | three confirmed, one not adopted | `d81baa54` |
| 5 | The lower-bound age breaks "reasons imply no value" | confirmed: 1 case in ~400 measurements | `c78250d0` |
| 6 | `predictor_complete` admits wholly allocated predictors | confirmed | `5e752ed6` |
| 7 | Header presence of the eight new columns is unverified | confirmed | `d81baa54` |
| 8 | A copy conflict can mask a member outside the universe | confirmed | `5e752ed6` |
| 9 | `state_and_dc_code_count` was a hard-coded 51 | confirmed | `2624d199` |

**Finding 4's fourth item was not adopted.** The peer asked the report to say
"entries" or "pages" rather than "nineteen fields". The report already says
"nineteen **value** fields", already states that the nine columns are five value
fields plus four status flags, and already states that the flags have no pages
of their own. Nothing there is wrong, so nothing was changed.

**Finding 3's three failures are pre-existing and unrelated.** They are
`test_spec_engine_country_bundles.py`'s pinned spec digests for `am`, `be` and
`uk`. Established rather than assumed: this lane's diff touches no spec, bundle
or country file, and `git ls-tree` over `spec_engine/` and over
`microcosm/build/{am,be,uk}/` is byte-identical between the accepted base and
this head. The repository history shows these pins being re-attested when
upstream rule packages move (`06e03665`, `c9d41eb0`).

## What the digest check actually showed

The accepted base implementation was materialised from git into a scratch
package and run beside this head over the same invented rows built by the
repository's own test helpers. Both produced measurement digest
`66b03416…9dbe79a5` and membership digest `d6fe1555…9032a1c5`, and every
household's exclusion agreed. Those two values are now pinned in the test file
from the base, so this head cannot move them silently. This is the continuity
check a version 3 extraction must pass beside the 17,646 / 4,348 / 4,057 ladder
counts.
