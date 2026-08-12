# Progress

## State

Round 10 is in progress on `tail-stratum-support-652` at retrigger commit
`0e6cfae5`. The real 1% smoke-r7 build reached the stacked-tail preservation
guard after 1,890 seconds and found a second write to recipient-owned
`person.self_employed_pension_contributions_desired` on clone 2. The immediate
task is to derive explicit final-producer ownership for all three known
PUF/source-completion overlap targets from the certified two-spine pipeline,
then enforce the complete overlap surface without weakening the preservation
guard or running a build.

## Done

- Confirmed a clean worktree on the requested branch and exact retrigger HEAD.
- Confirmed the Round 9 commits are intact immediately below the retrigger.
- Read `CLAUDE.md` and the GitNexus debugging workflow. GitNexus MCP tools are
  unavailable in this session, so the producer/call-path trace will use local
  source, tests, commit history, and the supplied build checkpoints.
- Honored the no-network constraint; no fetch, push, GitHub call, or build has
  been attempted.

## Next

- Inspect smoke-r7 checkpoints and logs to prove the failing write sequence.
- Trace the certified two-spine treatment of all three overlap targets.
- Enumerate every DAG-permitted producer-output/transfer/tail-owned overlap and
  add registry-driven ownership receipts plus an enforcement test.
- Implement the smallest ownership-consistent write/snapshot correction.
- Run focused proof, exact 495-test #583 proof, full workspace chunked proof,
  ruff/format/diff checks, and record the changelog and smoke-r8 prediction.
