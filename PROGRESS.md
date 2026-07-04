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
1. [ ] Add #301 tests to test_us_puf_support.py after existing
       test_puf_tax_detail_refuses_formula_owned_outputs block:
       - PE-US-gated: real engine catches genuinely formula-owned output NOT on
         static list (e.g. income_tax); static list is SUBSET of engine-derived set.
2. [ ] Run full `uv run pytest` (workspace) — green.
3. [ ] Changelog fragment per repo convention.
4. [ ] Delete this file, push, open PR (--body-file, no @-mentions, no merge).

## Semantic questions deferred
(none yet)
