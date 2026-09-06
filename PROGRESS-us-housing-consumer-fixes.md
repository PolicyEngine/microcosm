# US housing receipt consumer — review fix lane

Worktree: `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-housing-consumer-fixes-20260906`
Branch: `us-housing-consumer-fixes-20260906`
Base: `origin/main` = `c1b8324183819cf91511c002bf724a2676524e87`, fast-forwarded to reviewed head
`ffbef26eaba498028b64d646e369651999c71900` (original lane worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-housing-consumer-20260906`, merge base `ab10f778`).

Fixing the root+Fable review at
`/tmp/policyengine-command-runs/81a7c43a2f3b/housing-consumer-ffbef26e-review-full-report.md`
(peer capture `/Users/maxghenis/architecture-reviews/microcosm-20260904/us-housing-consumer-fable51-capture.md`).

## State

Writing the consequential tests for C1/C2/C3/A1/A2 before touching the consumer.

## Done

- Created the fresh child worktree from fetched `origin/main`, fast-forwarded to `ffbef26e`.
- Read the full root report, the peer capture and the root reproduction receipts.
- Read the consumer, the probes, `asec_housing_status{,_source}`, `asec_housing_universe{,_source}`,
  the five named spine-blindness guards and the existing invented-byte source fixtures.
- Reproduced the reviewed baseline in this worktree: 69 passed, 3 failed
  (`test_us_runtime_module_classification_is_complete`,
  `test_runtime_classification_rejects_a_new_unreviewed_module`,
  `test_runtime_population_operators_are_source_spine_blind`).
- Located the AST offences exactly: `asec_housing_receipt_consumer.py:197:68` and `:474:83`,
  both `key=lambda ...: ...[1]`. The guard marks every **unannotated** parameter a possible
  column container, and a lambda parameter can never be annotated, so an annotated
  module-level sort key clears it without weakening the guard.

## Next

1. Land the tests, then C1 household identity binding, C2 classification + typed sort keys,
   C3 typed attached reader, A1 bound HU evidence, A2 portable requirements link,
   plus read-only arrays, codebook-bound diagnostics and the labelling clarifications.
