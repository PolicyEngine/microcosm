# Populace — agent guide

Agent-operational notes only. The [README](README.md) covers usage and release
operations; [DESIGN.md](DESIGN.md) is the architectural authority. On any
conflict, executable configuration (pyproject.toml, `.github/workflows/`,
`tools/`) and DESIGN.md win over this file.

## Layout

uv workspace monorepo. Shards live in `packages/populace-<x>/` and import as
the PEP 420 namespace `populace.<x>`: `frame`, `fit`, `calibrate`, `build`,
`data`. There is no top-level `populace` package directory — never add an
`__init__.py` to the namespace root.

## Commands

```bash
uv sync --all-packages   # set up the whole workspace
uv run pytest            # behavioral contract suite + unit tests (all shards)
uv run ruff check .      # lint
```

PR CI (`.github/workflows/test.yml`) runs exactly those plus a wheel-packaging
gate (build every shard's wheel, install into a clean venv, import and test).
Editable installs hide packaging breaks — if you touch packaging, build wheels
locally before pushing.

## The PR-CI / certification boundary

PR CI is secrets-free and never touches restricted microdata. Green PR checks
mean the code contracts hold — they do **not** certify data artifacts.
Builds, calibrations, and releases run outside PR CI, need gated Hugging Face
data and credentials, and cannot run from forks. Release publication is a
deliberate human step (`tools/publish_release.sh` →
`populace-publish-release`), gated by `tools/preflight_us_release_gates.py`;
see README "Releasing & alerts". Publication also refuses a release whose
build recorded staging telemetry that never reached its repo
(`--allow-missing-staging` overrides); a build that declared `--no-staging`
publishes without the flag. Never publish or promote artifacts as a side
effect of another task.

## Root journals are history, not state

The root `PROGRESS*.md`, `FINAL_REPORT.md`, `*_COVERAGE_PROGRESS.md`, and
similar files are session-handoff journals: accurate when written, historical
afterward. Do not treat their "State"/"Next" sections as current truth —
check git/GitHub instead. When a branch carrying such a journal merges,
historicize any currency claims in it ("nothing was pushed", "do not merge",
"in progress") in place with a dated note, so the file cannot mislead later
readers. Adjudicated verdicts belong in `experiments/` or the tracking issue,
with the journal pointing to them.

## Review this file

Update this guide in the same PR whenever the workspace layout, test
commands, or release flow change. If you find it contradicting the repo,
trust the repo and fix this file.
