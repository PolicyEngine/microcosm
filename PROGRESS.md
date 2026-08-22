# Progress: passive-pass-through-722 re-verdict closure

## State

Re-verdict finding B1 is closed locally and B3 is in progress offline on
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
- Froze the exact historical `qbi-v3-wiring` simulator, post-QRF reconciler,
  and v1/v2/v3 assumption resources from `cfbf6330` as SHA-pinned test-only
  fixtures. Current HEAD intentionally carries none of that retired runtime.
- Added the CI-executable synthetic B1 regression over the historical runtime's
  literal supported-version gate `(1, 2, 3)`. Every parameter realizes a
  positive passive row count and weighted aggregate, byte-preserves the full
  15-leaf simulation surface plus its routing-owned self-employment leaf, and
  preserves both global NumPy state and every version-family draw (including
  the v2/v3 host-classifier seed).
- The targeted B1 node completed with all three parameters passing (exit 0).
  Touched-file Ruff check and format check are clean. An earlier duplicate was
  manually interrupted during the installed-engine collection import (exit
  130); it produced no test result. No run exited 137.

## Next

- Add a test-local aggregate and bisection plus synthetic and env-gated B3
  cross-checks; run the restricted replay once with the supplied artifact.
- Run only targeted tests and touched-file Ruff checks, update the two branch
  handoff reports with exact CI/certification scope and commit mappings, and
  write the final certification record to `FINAL_REPORT.md`.
