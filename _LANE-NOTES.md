# QBI ownership lane notes

## 2026-08-21 intake

- Branch/worktree: `battery-qbi-ownership` at `2c7a7218`, rooted at
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-qbi-ownership`.
- Binding build policy: at most a 1% sample; off-chain only; never pass
  `--logbook-prev-row-digest`; never touch `logbook-pending-chain.txt`; remain
  below 15 GiB RSS; check for a live `build_us_multispine_pool` before building.
- Binding model policy: do not tune gates, bands, ceilings, folds, or seeds;
  exclusions are owner-only and no register entry may be added.
- Binding evidence policy: every mechanism claim in the final record must cite
  current `module:line` locations.

## Environment receipt

- `uv sync --all-packages --extra us` failed before resolution because the
  managed sandbox denies `/Users/maxghenis/.cache/uv`.
- Retry with `UV_CACHE_DIR=/private/tmp/microcosm-qbi-uv-cache` resolved the
  workspace but could not download `pydantic==2.13.4` because network/DNS is
  disabled.
- `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-f1/uv.lock` has SHA-256
  `ea7af7806a0beefe7394adefd5516649f3eba4740ae95ccdaa9aaa252249bc3a`,
  exactly matching this worktree's lock, and its `.venv` contains the locked US
  and development dependencies. Test commands use that environment read-only
  with `UV_NO_SYNC=1`, a writable uv cache, and `PYTHONPATH` entries pointing
  only to this worktree's five package `src` directories.

## Source-of-truth evidence map

- Remediation order: adjudication line 42 requires recomputing each realized
  QRF regime from frozen donor support for every availability pattern and
  persisting it before any regime-specific remedy.
- Workstream: adjudication line 112 assigns eight BLOCKER checks/four physical
  QBI amount legs to ownership attribution and coupled-surface refit.
- Separate fix: adjudication lines 69-71 and rows 186-187 assign the two SSTB
  booleans to clone 1; they are not exclusions.
- Reproducibility: adjudication line 228 binds the failed simulated checkpoint
  SHA-256 `5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a`
  to release/run IDs and states that all nine QBI invariants were recomputed at
  tolerance `1e-8`.

## Open evidence questions

- For W2 and UBIA, distinguish the clone-1 PUF producer margin from the clone-0
  late-transfer margin before reconciliation.
- For BDC and REIT/PTP, also record the post-transfer QBI reconciliation delta,
  because exposure caps can change both roles.
- A margin must not be assigned an origin unless the receipt includes the exact
  check ID, stage metrics, and a nonempty closed-set origin channel.

## Initial commit gate

Commands used the exact-lock read-only dependency environment described above
and this worktree's package sources:

- `uv run pytest packages/microcosm-frame/tests`: 294 passed, 36 skipped.
- `uv run pytest packages/microcosm-fit/tests`: 93 passed.
- `uv run pytest packages/microcosm-calibrate/tests`: 201 passed.
- `uv run pytest packages/microcosm-data/tests`: 275 passed, 1 skipped.
- `uv run pytest packages/microcosm-build/tests`: 5,959 passed, 39 skipped.
- `uv run ruff check .`: clean.
- `git diff --check`: clean.
