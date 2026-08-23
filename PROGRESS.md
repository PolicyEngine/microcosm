# Progress: 25% replacement-candidate runbook, round 2

Date: 2026-08-23

Branch: `candidate-25pct-runbook`

## State

**In progress at ordered recovery item 1: the Ledger v9.4 consumer artifact.**
Round 1 stopped correctly because only the pinned bare facts JSONL was present.
Round 2 is authorized to recover the missing inputs without fabricating a path
or digest. No pool build, release build, publication, promotion, push, or
logbook-chain write has occurred.

The candidate mode is owner-ruled as exact-k. The owner-stated pending default
for the release solver seed is `0`; it will be recorded as pending Max's
ratification. The `pi_hi` value remains subject to direct source and legacy-lane
evidence before it may enter a command.

## Done

- Read `CLAUDE.md`, the prior `FINAL_REPORT.md`,
  `experiments/candidate_25pct/input_audit.md`, and
  `experiments/candidate_25pct/dry_run.md` before taking round-2 action.
- Confirmed the worktree starts clean at `4f616cb2b0f9`, four commits ahead of
  `origin/main` (`d69131a3534a`).
- Read the GitNexus exploration workflow. No GitNexus repository resource or
  query tool is exposed in this session, so code-contract tracing will use the
  checked-out sources and local repository history directly.
- Established the serial stop rule: later inputs may be researched read-only in
  parallel, but they will not be produced until every prior ordered input is
  reproduced and verified.

## Next

1. Locate the Chronicle checkout and the exact commit/tag and export command
   that produced `consumer_facts_buildn_v9_4.jsonl`.
2. Regenerate the artifact under a temporary directory and require byte-for-byte
   SHA-256 equality with
   `b3c0835631a446eb96aa84d86f3ee962d15ca356174c7114db52974f1cacc080`.
3. Only after equality, install the complete artifact at
   `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/inputs/ledger-v9_4/`
   and record the facts and manifest digests.
4. Continue through SCF, incumbent diagnostics, the frozen surface, and then the
   guarded script in the owner's required order. Stop honestly at the first
   input that cannot be produced.
