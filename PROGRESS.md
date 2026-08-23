# Progress: UK UC caseload and council tax diagnosis

## State

Diagnosis and owner handoff are complete on
`uk-caseload-counciltax-diagnosis`, based on `origin/main` at `2aa96795`. This
lane remained diagnostic only: no pool build, calibration, gate tuning,
exclusion change, publication, push, or issue comment was run.

## Done

- Read the repository and PolicyEngine-wide `CLAUDE.md` instructions before
  acting.
- Confirmed the branch started clean at the post-#735/#733 `origin/main` merge.
- Attempted the required `uv sync --all-packages --extra us`. The managed
  sandbox first denied the global uv cache and then denied PyPI DNS access from
  a writable cache while fetching `quantile-forest==1.4.2`.
- Identified a complete sibling environment with the byte-identical
  `uv.lock` (`ea7af780...`) and verified that `uv run --no-sync`, directed to
  that environment with this worktree's five `src` trees first on
  `PYTHONPATH`, imports this branch's code.
- Applied the GitNexus debugging workflow. Analysis produced a local index of
  626 files, but registration failed because the sandbox forbids writing
  `~/.gitnexus/registry.json`; source, call-site, fixture, and committed-receipt
  audits are the fallback.
- Completed parallel read-only audits of the UC facts/mechanisms and the
  council-tax/`owned_land` paths. No sub-agent edited the worktree.
- Isolated and corrected one root-journal wording that tripped the repository's
  retired-package-name guard; the focused contract now passes.
- Exercised the entire test surface. Two clean build-shard attempts each passed
  6,032 tests and skipped 38 before a different unrelated US CLI subprocess
  exceeded its fixed timeout. Both sole failures pass in isolation after their
  imports are warm; the other package shards pass (203, 275, 93, and 294 tests,
  with expected skips) and Ruff passed. A later exact run passed all shards in
  one invocation: build 6,033 passed / 38 skipped; calibrate 203 passed; data
  275 passed / 1 skipped; fit 93 passed; frame 294 passed / 36 skipped; Ruff
  passed. That green run gated the initial journal commit.
- Quantified the exact-lock UC diagnostic overall, by four administrative
  elements, four family types, and six child-count bands. Recorded an aligned GB
  reporter path, separated measured associations from causal claims, and
  delimited exactly what #735's target contract changes. A 2.91.0 cross-check
  reproduced the displayed counts despite its final-formula drift.
- Corrected council-tax net to use the model's household zero floor, identifying
  a £0.512 billion English nominal-CTR/gross mismatch. Completed an exact-lock,
  fixed-row/fixed-wealth/fixed-weight sensitivity of the shared CTR switch.
- Separated gross council-tax definition/reconciliation from the `owned_land`
  sparse-tail instability while preserving the possible later
  savings → CTR → net path
  (`packages/microcosm-build/src/microcosm/build/uk_runtime/was_wealth.py:398-423`).
  Preserved the task-supplied #733 receipt values as unverified supplied
  evidence because the named receipt was absent.
- Wrote the two evidence reports, owner-ready unposted issue-comment drafts,
  and `FINAL_REPORT.md`, with mechanism claims tied to module-and-line citations.
- Created the closing diagnosis commit only after the exact package-shard loop
  and Ruff gate succeeded on the complete tree; no files were edited between
  that gate and commit.

## Next

- Owner reviews the saved drafts and replaces relative links with immutable
  commit permalinks before posting them upstream.
- microcosm#736 adjudicates geography, grain, period, and quantity definitions;
  any reconciliation or exclusion remains owner-signed.
- A licensed lane runs the remaining predeclared UC and `owned_land`
  counterfactuals, completes the CTR/gross decompositions, and performs a real
  post-#735 calibration.
