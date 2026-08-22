# Progress: passive-pass-through-722 re-verdict closure

## State

Re-verdict findings B1 and B3 are being closed offline on
`passive-pass-through-722` from starting HEAD `bc0ddb0a`. No production code,
restricted artifacts, or supervisor-owned tests have been changed. No fetch,
push, pull-request mutation, rebase, reset, checkout, or stash is in scope.

## Done

- Read `CLAUDE.md` and the adjudicated B1/B3 sections of
  `sol-run-reverdict-0822.log` before changing the branch.
- Read the prior audit's stream-isolation note: its uncommitted historical
  composition produced byte-identical v1/v2/v3 outputs and preserved global
  NumPy RNG state.
- Confirmed the worktree starts clean at `bc0ddb0a` on
  `passive-pass-through-722`.
- Attempted the GitNexus exploration workflow. No GitNexus repository resource
  or query tool is exposed in this session, so direct source and call-path
  inspection is the fallback.
- Reserved
  `packages/microcosm-build/tests/test_spec_engine_engine_abi.py` and
  `packages/microcosm-build/tests/test_us_multispine_pool_tool.py` for their
  other owner; this task will not touch them.

## Next

- Inspect the runtime version gate and historical synthetic harness, then add
  the durable parametrized B1 stream-isolation regression.
- Add a test-local aggregate and bisection plus synthetic and env-gated B3
  cross-checks; run the restricted replay once with the supplied artifact.
- Run only targeted tests and touched-file Ruff checks, update the two branch
  handoff reports with exact CI/certification scope and commit mappings, and
  write the final certification record to `FINAL_REPORT.md`.
