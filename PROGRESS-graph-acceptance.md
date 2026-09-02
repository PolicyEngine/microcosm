# Lane B — the red acceptance suite for `microcosm-graph`

Session journal for the lane that turns `docs/graph-acceptance.md` into an
executable, red test suite. Accurate when written; check git/GitHub for
current truth afterwards.

Two naming notes, both forced by the repository rather than chosen:

- The branch is `node-graph-acceptance`, not `node-graph/acceptance`. Git
  cannot hold both `refs/heads/node-graph` and a `refs/heads/node-graph/`
  directory, so the slash form is impossible while the base branch exists.
- This journal is not the root `PROGRESS.md`. That file is the
  `acs-predictor-release-join` lane's journal and predates this work.

## State

> Historical (2026-09-02): this journal describes the suite the day it was
> written. The `node-graph` branch merged into `main` in #836 with every
> charter property green (burndown 37/37, including H1–H3 parity); the
> current state is `docs/graph-acceptance.md` and git.

The suite is complete and red. `uv run pytest packages/microcosm-graph`
reports **41 xfailed, 37 passed**: one strict `xfail` per charter property in
groups A–H plus the four incident replays, and green guards that protect the
fixtures those properties are written against.

## Done

- Read `docs/graph-acceptance.md`, the frozen interfaces (`decl.py`,
  `kernel.py`), `test_graph_decl.py`, `microcosm-frame` (`bundle.py`,
  `weights.py`, `schema.py`), `CLAUDE.md`, `tools/ci_test_groups.py`,
  `.github/workflows/test.yml`, and the architecture review's synthesis.
- Toy country: 200 households, 502 persons, two strata, typed household
  design weights, a nullable `boolean` with 41 nulls and a float column with
  31 negative zeros (`packages/microcosm-graph/tests/fixtures/toy_country/`).
- `_toy.py`: 19 kernels — the honest ones, five that misbehave on purpose,
  and one that exists only to be counted — plus the graph builders and the
  byte-comparison helpers.
- Nine acceptance files, one per charter group plus the replays.
- `tools/graph_acceptance_burndown.py` and its unit test; one added step in
  the `lint` job of `.github/workflows/test.yml`.

## Next

Nothing in this lane. The runtime lane implements against the API assumptions
listed in the lane report (`out.md`), and deletes a marker per property.
