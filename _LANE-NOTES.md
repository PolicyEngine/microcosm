# UK diagnosis lane notes

## 2026-08-22 — lane start and environment

- Branch: `uk-caseload-counciltax-diagnosis`, starting clean at `2aa96795`
  (`origin/main`), which contains the post-#735 target-surface union and #733
  FRS 2024-25 retarget merge.
- Scope: diagnose the upstream UK data issues #452 and #448 using committed facts,
  receipts, fixtures, parity instruments, and code. This lane will not build a
  pool, alter a gate/band/ceiling/fold/seed, change an owner-only exclusion,
  push, publish, or post the draft comments.
- Environment: `uv sync --all-packages --extra us` could not complete because
  the sandbox denied the global uv cache and outbound DNS. The
  `microcosm-spec-engine` sibling has an identical `uv.lock` and all locked
  dependencies; validation uses its environment with `--no-sync` and this
  worktree's package sources first on `PYTHONPATH`.
- GitNexus: local analysis indexed 626 files / 12,541 nodes / 33,290 edges,
  then registration failed on the sandboxed global registry. Mechanism tracing
  therefore uses direct source and call-site inspection, with every final
  mechanism claim cited to a repository module and line.
- Initial validation: the build shard reached 4,661 passing tests before one
  root-tree provenance guard rejected a retired-package name in this journal.
  The wording now names the upstream issues without naming the archived
  package, and the isolated guard passes (1 passed in 60.68s). The interrupted
  red run is not a commit gate; a clean full rerun follows.
- Clean-suite validation then exposed two unrelated host-latency failures. On
  the first attempt, 6,032 build tests passed and 38 skipped before an atomic
  US-publication crash probe's child import exceeded its fixed 60-second
  timeout. A direct timing showed that CLI import alone took 55.00 seconds, and
  the focused test passed after warm-up. On the second attempt, the same 6,032
  build tests passed and 38 skipped, but a different US import-entry CLI child
  exceeded its 300-second timeout; that focused test then passed in 55 seconds.
  The calibration (203 passed), data (275 passed, 1 skipped), fit (93 passed),
  and frame (294 passed, 36 skipped) shards and Ruff all pass. These are
  environment/process-startup incidents, not UK diagnosis failures, but this
  lane still requires a single green exact full-suite exit before committing.
- Public-source verification: current Scottish, Welsh, and English 2025-26
  council-tax releases and the UC Regulations pension-capital disregard were
  checked on their official sites. GitHub issue/PR pages could not be fetched
  through either the web cache or the sandboxed CLI, so issue-state claims use
  the task brief and committed repository evidence.
- No pool build has run. A read-only, aggregate-only 2025 council-tax
  diagnostic used the locally cached SHA-pinned enhanced-FRS artifact; it did
  not write an artifact, calibrate weights, or expose record-level values.

## 2026-08-22 — diagnosis and handoff

- Exact initial-journal validation completed in one invocation: build 6,033
  passed / 38 skipped; calibrate 203 passed; data 275 passed / 1 skipped; fit 93
  passed; frame 294 passed / 36 skipped; Ruff passed. The journal was committed
  as `e1c36b58` only after that green result.
- Exact-lock aggregate diagnostic: aligned GB final-award caseload is 6.1795
  million against 6.7589 million in administration (-0.5794 million, -8.6%). A
  separate 2.91.0 run reproduced the displayed UC counts despite its changed
  final formula
  (`policyengine-uk@2.89.0:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-15`;
  `policyengine-uk@c93e1a05:policyengine_uk/variables/gov/dwp/universal_credit/universal_credit.py:4-36`).
  The aligned GB reporter bridge, four element gaps, family-type gaps, and
  child-count gaps are recorded in `experiments/uk_diagnosis/uc_caseload_452.md`.
- UC causal boundary: the report labels the code-confirmed take-up, reporting,
  eligibility, and weighting paths, measured associations, and required
  counterfactuals separately rather than assigning additive shares; the full
  module-and-line citations live beside each path in the UC report.
- Council-tax diagnostic: the Scottish/Welsh result changes materially with
  gross/net and official/fixture comparators. Correct use of the household net
  floor raises England from the original raw £40.927 billion to £41.440 billion;
  nominal CTR exceeds gross by £0.512 billion on 462 records
  (`policyengine-uk@2.89.0:policyengine_uk/variables/household/consumption/council_tax_less_benefit.py:16-21`).
  The active national reference and inactive country facts require signed
  definition reconciliation in #736.
- `owned_land` is downstream of gross council tax, but it predicts later wealth
  outputs including savings
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:333-339,398-423`),
  so the reports preserve a possible savings → CTR → net interaction rather
  than claiming full independence. The #733 71.4% national / 112.9%
  maximum-region values are explicitly marked task-supplied because the named
  receipt was absent; no seed experiment or imputation change was run here.
- Exact-lock diagnostics used writable temporary copies of the two cached,
  SHA-pinned licensed artifacts because the HDF loader requests write access.
  They emitted aggregate values only; all temporary copies were deleted and no
  licensed rows or artifacts were added to the worktree.
- Deliverables: `experiments/uk_diagnosis/uc_caseload_452.md`,
  `experiments/uk_diagnosis/council_tax_448.md`,
  `experiments/uk_diagnosis/comment_drafts.md`, and `FINAL_REPORT.md`. Draft
  comments remain unposted and owner-controlled.
- Closing validation policy: the deliverable commit is created only after the
  exact `packages/*/tests` shard loop and Ruff exit successfully on the complete
  tree. No file is edited between that gate and commit.
