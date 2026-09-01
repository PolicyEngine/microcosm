# Lane B — the red acceptance suite for `microcosm-graph`

**State:** started. Branch `node-graph-acceptance` off `origin/node-graph`
(`517891f4`). The charter branch name `node-graph/acceptance` is impossible:
git cannot hold both `refs/heads/node-graph` and a `refs/heads/node-graph/`
directory, so the branch is `node-graph-acceptance`.

## Done

- Read `docs/graph-acceptance.md`, `decl.py`, `kernel.py`, `test_graph_decl.py`,
  `microcosm-frame` (`bundle.py`, `weights.py`, `schema.py`), `CLAUDE.md`,
  `tools/ci_test_groups.py`, `.github/workflows/test.yml`, and the
  architecture-review synthesis.
- Confirmed `--import-mode=importlib`: a test module cannot `import _toy`
  directly; the repo's established pattern (`test_uk_release_assembler.py`)
  loads a sibling helper through `importlib.util.spec_from_file_location`.
- Confirmed `microcosm-graph` test files classify as `rest` (fast lane) and
  `us-am`/`other-shards` (engine lane); no `[defaulted]` risk.

## Next

1. Toy country fixtures + `_toy.py`.
2. Acceptance files A–H + replays.
3. Burndown tool + its unit test + the CI step.
4. `out.md`.
