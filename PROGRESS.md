# Progress: UK UC caseload and council tax diagnosis

## State

Evidence collection is complete on `uk-caseload-counciltax-diagnosis` from
`origin/main` at `2aa96795`; report drafting is next. This lane is diagnostic
only: no pool build, calibration, gate tuning, exclusion change, publication,
push, or issue comment has been run.

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
  with expected skips) and Ruff passes. A final exact full-suite attempt on this
  journal state remains the commit gate.

## Next

- Obtain one green exact full-suite exit and commit this initial journal.
- Quantify UC gaps by element and family type, with explicit denominators and
  licensed-run limitations; delimit what #735's household target can fix.
- Separate council-tax level error from `owned_land` release instability and
  identify the smallest defensible remedy for each without changing gates or
  owner-only exclusions.
- Write the two evidence reports, unposted issue-comment drafts, final report,
  and complete per-shard pytest plus Ruff validation for every commit.
