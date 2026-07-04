# PROGRESS — populace #300 (build-level weights audit)

Branch: `weights-audit` | Worktree: populace-weights-audit
Issue: PolicyEngine/populace#300

## State
- NOT STARTED yet. Formula-guard #301 lands first (separate worktree/PR).
- Branch checked out, empty (no diff vs origin/main).

## Plan (TDD — failing test first)
1. [ ] `gh issue view 300` — read spec; stop+report if mismatch with this plan.
2. [ ] Locate where fits resolve their weight kind (populace-fit / build stage plans).
3. [ ] Failing test first: a release/build audit that records the resolved weight
       kind per fit and FAILS when an unlisted unweighted fit appears.
4. [ ] Implement minimal focused diff to make it pass.
5. [ ] Full `uv run pytest` green + `uv run ruff check .`.
6. [ ] Delete this file, push, open PR (--body-file, no @-mentions, no merge).

## Semantic questions deferred
- Weight semantics are guardian-owned. Any ambiguity about what "unweighted"
  means, or which fits are legitimately allowed to be unweighted, goes in the
  final report — do NOT decide unilaterally.

## Notes
- Repo has NO changelog system (no changelog.d/, no towncrier, no CHANGELOG.md).
  Recent merged PRs (#295/#290/#289) carried no changelog fragment. So the
  parent PolicyEngine towncrier convention does NOT apply here — no fragment.
- CI: `uv sync --all-packages` does NOT install the `[us]` extra, so
  policyengine_us is absent in CI; PE-US-gated tests must importorskip.
