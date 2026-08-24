# Progress: stacked-pool to release CD-vintage provenance

## State

In progress. The required all-package US environment is synchronized from the
locked artifacts, the candidate-chain defect report has been reviewed, and no
source implementation has changed yet. The production stacked builder still
lacks a post-assembly household congressional-district authority, while the
release preflight still probes a table-only H5 layout
(`tools/build_us_multispine_pool.py:441-577,3719-3786`;
`tools/build_us_fiscal_refresh_release.py:2565-2661`).

No build, release, push, guard weakening, operator-boundary change, or
`logbook-pending-chain.txt` access has occurred.

## Done

- Read `CLAUDE.md` and the full candidate-chain defect report before changing
  the worktree.
- Ran the required `uv sync --all-packages --extra us` first. The managed
  sandbox blocked the user cache and network, so recovered with the completed
  current-lock Python 3.14 environment and an exact offline sync. All five
  editable workspace packages now resolve from this worktree.
- Confirmed the branch starts clean at `origin/main` commit `7b90bb18`.
- Established this committed state/done/next journal and opened the new lane
  section in `_LANE-NOTES.md`.
- Passed the complete baseline suite in fresh package/build partitions: 7,213
  passed / 77 skipped / 0 failed. This follows the repository's per-shard
  memory rationale (`.github/workflows/test.yml:24-34`).

## Next

1. Select and code-cite the deterministic household-CD assignment authority,
   including its spec/schema declaration and seed/checkpoint identity.
2. Add atomic nullable-H5 root-attribute publication and format-aware release
   preflight reading without weakening the guard.
3. Add the real tiny stacked-pool-to-release-preflight integration test,
   regenerate the anti-rot chain and legitimate identity pins, and keep every
   coherent commit suite-green.
4. If integration evidence needs a build, first prove no other `build_us_*`
   process is running and run at most the sanctioned off-chain 1% command.

## Historical prior lane

The PolicyEngine-US 1.819.0 lock-bump journal previously in this file is
historical: that lane merged into `origin/main` at `7b90bb18` on 2026-08-24.
Its final state remains available at commit `05d254aa` and its detailed
receipts remain in the historical section of `_LANE-NOTES.md`.
