# Progress: passive-pass-through-722 re-verdict closure

## State

Re-verdict findings B1 and B3 are closed locally offline on
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
- Replaced the circular B3 calibration check with test-local evidence parsing,
  inverse-CDF integration, odds shifting, row-wise weighted aggregation, and
  bisection. Those helpers consume only committed JSON and caller-supplied raw
  arrays; they call no production calibration or subordinate helper.
- The two-row synthetic CI node has the hand solution `shift = ln(3)` and
  aggregate `(100*1 + 200*2) * 0.5 * 0.75 = 187.5`; the test-local and
  production solvers agree within the artifact-derived binary64 floor. Its
  final run passed (exit 0).
- The required restricted replay passed (exit 0) with the supplied SHA-pinned
  `puf_2024.h5`: test-local solved shift `-1.157105426398319`, committed shift
  `-1.157105426398319`, effective shift tolerance
  `2.220446049250313e-16`, and independently seeded aggregate
  `$55,021,131,518.061035` (`0.7187449327011208%` above the provisional
  `$54,628,492,000` midpoint). No test run exited 137.
- Committed the B1 closure as `bd537c42` and the B3 closure as `92dd3568`.
  Both received independent static review with no defects found.
- Re-ran the committed B3 synthetic node (`1 passed`) and exercised the
  env-unset restricted node (`1 skipped`). The earlier artifact-backed replay
  passed once. Touched-file Ruff check and format check are clean.
- Updated `FINAL_REPORT.md` and `PROGRESS_PASSIVE_722.md` with a dated
  2026-08-22 record that separates CI coverage from the env-gated replay,
  maps each finding to its closing commit, and marks older self-certification
  language as superseded.

## Next

- No local implementation work remains. The supervisor can push the committed
  handoff and let clean-runner CI arbitrate the suite; no local push or PR
  mutation is in scope.
