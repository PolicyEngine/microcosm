# PROGRESS — populace #301 (formula-owned rejection set from engine metadata)

Branch: `formula-owned-metadata` | Worktree: populace-formula-guard
Issue: PolicyEngine/populace#301

## State (resuming predecessor, killed mid-task)
- Tip 1ede4ec = lead-salvaged WIP of 4 files (VERIFIED coherent).
- Salvaged: adapter `formula_owned_outputs()`, `resolve_formula_owned_outputs()`,
  `assert_formula_owned_blocklist_current()`, `_reject_formula_owned_outputs(engine=)`,
  4 adapter tests in test_policyengine_us_adapter.py.
- `test_us_puf_support.py`: imports added (PUF_TAX_DETAIL_FORMULA_OWNED_OUTPUTS,
  assert_formula_owned_blocklist_current, resolve_formula_owned_outputs) but NO new
  test bodies yet — this is the remaining work.

## Remaining
1. [IN PROGRESS] Add #301 tests to test_us_puf_support.py after existing
   test_puf_tax_detail_refuses_formula_owned_outputs block (ends ~L419):
   - Workspace-deterministic (fake engine injected): resolve_ union logic,
     assert_ drift check both directions, engine=None static fallback.
   - PE-US-gated (importorskip): real engine catches genuinely formula-owned
     output NOT on static list (income_tax); static list SUBSET of engine set.
2. [ ] Run full `uv run pytest` (workspace) — green.
3. [SKIP] Changelog fragment — repo has NO changelog system (see note). N/A.
4. [ ] Delete this file, push, open PR (--body-file, no @-mentions, no merge).

## Environment facts (verified)
- Repo has NO changelog.d/, NO towncrier, NO CHANGELOG.md. Merged PRs #295/#290/#289
  carried no fragment. Parent PolicyEngine towncrier rule does NOT apply. → no fragment.
- policyengine-us is an OPTIONAL extra ([us]/[policyengine]), not a workspace dep.
  CI `uv sync --all-packages` does NOT install it → PE-US absent in CI (gated tests
  skip there). LOCAL .venv HAS it installed → gated tests RUN locally; engine=None
  path resolves to the LIVE engine locally, so workspace-deterministic tests inject
  an explicit fake engine to stay environment-independent.

## Semantic questions deferred
(none yet)
