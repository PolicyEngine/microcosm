The requested 19-node attribution cannot run on main `16c8e78d2`: its native US runner and authentication modules are absent. This draft records that prerequisite, a source-cited executor design, and the unchanged graph baseline so the measurement-base decision is reviewable.

This is WIP evidence only. It does not implement `max_workers`, memoization, amendment 26, or native measurements. The design identifies why independent kernels need canonical coordinator admission: ordinary nodes patch shared population versions, and patch order affects later frame bytes. The attribution audit separates #938's graph optimizations from native owner work that runs before executor entry.

Validation:
- Unchanged graph suite: 632 passed, 1 skipped in 72.27s.
- CI test grouping and graph acceptance burndown: verification=ok.
- Spec coverage: 42156/42156 fields; 41/41 inventory checks.
- No Python source, producer pin, interface lock, or dependency lock changes.

Pending Max's measurement-base choice: documented native overlay while retaining a main-only implementation, native-branch rebase, or strict-main synthetic scope. See `experiments/graph-parallel-executor/out.md` for the complete findings and explicit unmeasured items.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
