# Rare signed-tail battery lane progress

## State

Lane initialization is in progress on `battery-rare-signed-tails` at
`2c7a7218`, matching `origin/main`. No product code, gate, band, ceiling, fold,
seed, exclusion register, logbook chain, or build artifact has been changed.
The authoritative frozen inputs are the arm-split lane's
`experiments/battery_burndown/ADJUDICATION.md` and `adjudication.json`.

The mandated `uv sync --all-packages --extra us` was attempted first. The
default cache is inaccessible in the managed sandbox; a retry with a writable
cache reached resolution but could not download `pandas==3.0.3` because network
DNS is unavailable. The primary-checkout environment proved sufficient for the
fit, calibrate, data, and frame shards but does not carry the build shard's
`jsonschema` dependency. Full verification therefore uses the compatible,
already-synced `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-f1/.venv`
read-only, with `UV_NO_SYNC=1` and this worktree's package source roots on
`PYTHONPATH`.

## Done

- Read `CLAUDE.md` and the GitNexus debugging workflow.
- Confirmed the branch starts at `origin/main` and recorded the pre-existing,
  untracked `.gitnexus/` directory without deleting or staging it.
- Attempted the required GitNexus symptom query. Its local index is current,
  but the global registry does not contain this worktree and is outside the
  writable sandbox; source tracing remains the authority.
- Located the frozen adjudication and established `FINAL_REPORT.md` as the
  committed output report because no separate output-file environment variable
  is present.
- Confirmed the shared environment imports this lane's `microcosm.frame` from
  the lane-local source tree and provides pytest, Ruff, pandas, and NumPy.
- Read the adjudication's comparator-mechanisms section in full and traced its
  cited incidence-first, five-carrier QED-skip, early-gap-fill, late-producer,
  donor-projection, regime-detection, and declared-absence seams.
- Recomputed all 48 red QED donor regimes from the frozen checkpoint: every
  availability pattern is gated (35 zero-inflated-positive targets and 13
  three-sign targets); none is degenerate or single-sign.
- Recomputed the five rare-tail physical legs on their actual route donors:
  collectibles +18, alimony +61, casualty +27, farm operations -89, and
  prior-year self-employment -48 sign carriers.
- Established that Keogh is not structurally absent: native ASEC has two
  positive values (`2,040` and `30,000`), but the forced late ASEC clone-1 donor
  is degenerate zero and its 1,736,840 finite banked recipient draws are all
  exactly zero.

## Next

- Finish the per-check mechanism/route matrix and record exact code citations.
- Add future-checkpoint regime provenance and the smallest non-threshold fix for
  the Keogh support loss, with mechanism-specific regression tests.
- Preserve the adjudication's BLOCKER posture for sparse tails whose frozen
  donors remain sign-capable but do not contain dense enough evidence for an
  honest target-specific refit.
- Guard any off-chain 1% build with the host queue/RSS checks and compare exact
  before/after failure lines against the frozen arm-split baseline.
