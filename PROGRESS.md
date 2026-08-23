# Progress: PolicyEngine-US 1.819.0 lock bump

## State

The `bump-policyengine-us` lane is active from `origin/main` at `31640b91`.
The required pre-bump US-extra environment sync is complete against the
unchanged lock (`policyengine-us==1.764.6`). No lock or source change has been
made yet.

The managed sandbox cannot write the user-wide uv cache and has no DNS. The
existing exact-lock `microcosm-scorecard` environment was therefore cloned
copy-on-write, then all five editable Microcosm packages were rebuilt and
relinked to this worktree with the prior lane's writable Hatch build cache.
This is an environment bootstrap only; `.venv` is untracked.

No pool or release build, push, gate change, threshold change, or band change
has occurred.

## Done

- Read `CLAUDE.md` and recorded the PR-CI/certification boundary and required
  one-process-per-shard test shape.
- Attempted `uv sync --all-packages --extra us` first as ordered. The default
  cache failed before resolution because the sandbox cannot initialize
  `/Users/maxghenis/.cache/uv`; an empty writable cache then reached PyPI but
  could not resolve DNS.
- Completed the equivalent unchanged-lock sync offline with
  `UV_CACHE_DIR=/private/tmp/microcosm-scorecard-uv.0rntvY/cache uv sync
  --offline --all-packages --extra us` after cloning the exact-lock Python
  3.14 environment. The sync rebuilt and relinked every workspace package to
  this worktree.
- Selected the GitNexus debugging workflow for the expected compatibility
  failures. Its query/context tools are not exposed in this session, so the
  recorded fallback is repository-wide search, direct installed-package
  inspection, and focused tests.

## Next

1. Validate and commit this lane-start journal checkpoint.
2. Run `uv lock --upgrade-package policyengine-us`, record every resolved
   version movement in `_LANE-NOTES.md`, and sync the upgraded US-extra
   environment.
3. Run each package test shard separately, repair verified upstream ownership
   and consumer changes, and re-pin legitimate spec/seed identities with the
   repository's own tools.
4. Scan the PolicyEngine-US changelog from 1.764.6 through 1.819.0, record the
   compatibility note, rerun every shard plus Ruff, and write `FINAL_REPORT.md`.
