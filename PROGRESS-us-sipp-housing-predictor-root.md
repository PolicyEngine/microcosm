# December SIPP housing predictor projection — root verification lane

## State

Open on 2026-09-06 on `us-sipp-housing-predictor-root-20260906`, branched from
`f20e413a130fc7bba2d6e91b3af88f3af48b26a7` on
`microcosm-us-sipp-housing-predictor-projection-20260906`, which is itself
seventeen commits on top of the accepted housing projection
`5534a0fc155dfdfe3b399725ed44fbb9d42f859e`.

That predictor branch was **not written by this session**. A separate local
session built it between 07:09 and 08:37 today and exited; its worktree
`_worktrees/microcosm-us-sipp-housing-predictor-projection-20260906` is clean at
`f20e413a` and is left untouched. This lane inherits that work rather than
rebuilding it, and its job is to verify every claim in it against the primary
documents and the code, correct what is wrong, and carry the increment to a
head whose report is true of that head.

Nothing is pushed, merged, deployed or published. No agent is spawned. No SIPP,
ASEC or ACS byte is read. No fit, draw, price bridge, donor mask, model input
binding or PolicyEngine calculation is performed.

## Lineage

`origin/main` fetched at lane open, `c1b8324183819cf91511c002bf724a2676524e87`.
`git rev-list --count 5534a0fc..origin/main` is 0 and
`git merge-base 5534a0fc origin/main` is `c1b83241`, so the accepted base
already contains the whole of maintained `origin/main` and no merge is needed.
Local `main` in the primary clone is far behind `origin/main` and is not the
lineage reference.

## Done

- Established that the target branch already existed and its author had exited,
  before making any edit. `lsof` on every live `claude` process showed none
  holding that worktree, and the two transcripts naming the branch belong to
  sessions absent from the process table.
- Opened this journal. The repository convention is one `PROGRESS-<lane>.md`
  per lane and the root `PROGRESS.md` is an unrelated F1 journal that a previous
  lane had already overwritten once; it is not touched here.

- **Verified the dictionary independently.** The cached PDF hashes to the pinned
  `0f1c5fbd…6db894f` and holds 3,371 pages. For each of the nine fields the page
  carrying its own `Name:` line was located from the extractor's page list and
  the printed number read off the foot of that same page, never inferred from a
  text-stream boundary. All five value fields agree with the transcription:
  `TAGE_EHC` 797 flag `NA`, `RHNUMPER` 891 flag `AHNUMPER`, `RHNUMU18` 892 flag
  `AHNUMU18`, `THTOTINC` 2866 flag `AHTOTINC`, `TEHC_ST` 3311 flag `AEHC_ST`.
  The four `A*` flags have no `Name:` entry of their own, as claimed.
  `TEHC_ST`'s published list is 53 codes across printed pages 3311-3312, and the
  transcribed `_STATE_CODES`/`_STATE_LABELS` pair equals it exactly, in order.
  51 are a state or DC; `03 07 14 43 52 57 58 59` are not published at all.
  The secondary locators are right too: `RHNUMPERWT2` 899, `RHNUMU18WT2` 900,
  `THTOTINCT2` 2867, `THCYINCPOV` 2880 flag `AHCYINCPOV`, `TEHC_METRO` 3301.
  Cross-checking all 24 recipe entries against the PDF found **zero** mismatches
  of page, printed page, status flag, universe, length or file class; the only
  entries without a page are the five identity columns the recipe declares as
  `kind: identity` with authority delegated to the frozen producer.
- **Stripped the formatter sweep.** `7b5589f0` had carried thirty-seven
  unrelated files. Parsing each at the base and at `f20e413a` gave an identical
  AST in all thirty-seven, so they were reflow only and were restored to the
  accepted base. The increment's diff is now the nine files it means to change.
- **Verified the frozen producer.** `sipp_december_source.py` and
  `sipp_measurements.py` hash identically at `5534a0fc` and at this head.
- **Verified the version 2 continuity pins by running the base.** The accepted
  base implementation was extracted and run over the same ladder fixture: it
  produces `66b03416…` and `d6fe1555…`, exactly the constants the test pins, and
  the same per-household exclusions as this head. With the nine columns absent
  from the rows entirely, both digests and every exclusion are unchanged again.
- **Fixed a real gap in the published contract.** `predictor_complete` was
  emitted but absent from the declared status order, and `DISAGREEMENT_STATES`
  and `INCOME_SIGN_STATES` were never published at all (the latter was
  referenced nowhere). All are now declared in the diagnostics and the recipe,
  with a test that drives every state and asserts each counter's keys are a
  subset of its declared vocabulary. `RECIPE_SHA256` moved to `66ed8871…`.
- **Probed the behaviour adversarially.** Out-of-domain and sentinel ages give
  coverage `none` and no maximum and no bound; partial coverage gives no maximum
  and a real bound with distinct reasons; two members at 90 give a maximum of 90
  recorded as the domain maximum; negative and published-minimum incomes are
  preserved signed; publisher-vs-roster disagreement is reported and corrected
  in neither direction; `TEHC_ST` 60 against 61 is a copy conflict.
- Suite at this head: **172 passed**. `tools/ci_test_groups.py --verify` is ok.
  `ruff check` passes on the changed files and on the restored ones.

- Wrote the lane report to
  `R/us-sipp-housing-predictor-projection-REPORT.md` and `.json`. The
  predecessor's report, written at the intermediate head `aa102314` and stale by
  eight commits, was preserved as
  `…-REPORT.superseded-aa102314.{md,json}` rather than overwritten, and nothing
  in it was copied.

## Next

- Nothing outstanding in this lane. The next increment is the donor-table
  preparation contract in section 8 of that report, which must run against the
  combined ASEC + ACS survey population and never an ASEC-only route. The
  undeclared period and price bridge between `THTOTINC`'s signed December-2022
  monthly amount and an annual recipient income is its largest open item.

## What this lane is not

This is a source and method acceptance at this head. It is not a launch
readiness declaration, not an adoption of the housing-launch-method-decision
memo - whose state domain, exclusion-rung and column-count proposals were each
declined on source grounds - and not evidence about any quantity a version 3
extraction has not yet produced. No extraction has been run at version 3.
