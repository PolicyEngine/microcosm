# PROGRESS — deflake-l0-refit-ess-test (issue #307)

## Goal
TEST-ONLY deflake of
`packages/populace-calibrate/tests/test_solve.py::test_refit_l0_selection_threads_l2_lambda_to_refit`.
It flips sign at a ~1e-5 ESS margin (torch/BLAS nondeterminism) and falsely
reddens unrelated PRs' wheels jobs.

## Constraints
- Do NOT change solver code or any numerics — semantics owned elsewhere.
- Preference order:
  (a) make refit deterministic in this test (seed/deterministic mode)
  (b) enlarge fixture or L2 lambda so expected ESS gap clears noise floor by OOM
  (c) LAST resort: assert explicit margin from MEASURED run-to-run spread
      (run ~20x to measure; just-big-enough tolerance that hides signal is
      unacceptable — must state measured spread + chosen margin).

## Acceptance
- 20 consecutive passes locally (loop, pipefail).
- Assertion still fails if L2 threading is broken (prove by inverting behavior).

## Status
- [x] Worktrees created
- [ ] Read the test + solver API
- [ ] Reproduce flake / measure spread
- [ ] Choose approach
- [ ] Implement
- [ ] 20x consecutive pass
- [ ] Prove it still catches broken L2 threading
- [ ] Full pytest + ruff green
- [ ] PR
