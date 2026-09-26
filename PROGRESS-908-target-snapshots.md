# microcosm#908 — per-target calibration estimate snapshots (bounded slice 1)

Lane journal for branch `calibration-target-snapshots-908-20260912`.
Journals are history, not state (see CLAUDE.md): check git/GitHub for current truth.

Historical note, 12 September 2026: the initial implementation below was
subsequently merged onto main `116d46ee9` and repaired through `b8380c475`.
Its original review and source-pin claims describe earlier revisions.
[The maintained guide](docs/calibration-target-snapshots.md) records current
component scope; country integration and issue 908 remain incomplete.

## State

Bounded first slice implemented, tested and committed on this branch. Not
pushed, no PR opened. #908 is NOT complete — see "Remaining #908 acceptance
scope" below.

## Scope of this slice

1. A shared, leaf-level typed snapshot codec in `microcosm-calibrate`:
   schema name + version, aggregate-only per-target rows, ordered target
   identity digest, finite-value and ordering validation, signed relative
   error with the repo's existing zero-target convention.
2. Real emission from the actual Adam solver loop in
   `microcosm.calibrate.solve`, at a configurable bounded cadence with an
   explicit every-epoch option, opt-in (off by default).
3. Honest iterate labelling: current / best_retained / selected, with the
   selected snapshot taken from the weights the solver actually returns.
4. Atomic local `latest.json` plus immutable, bounded history chunks.
5. A synthetic-only benchmark at US/UK-sized dimensions (invented matrices).

## Out of scope for this slice (remaining #908 acceptance scope)

- Version-2 staging upload / remote publication (PR896's country host).
- Calibration Diagnostics UI (different repository, by the issue's own text).
- Real UK/US native calibration measurements — this host is forbidden native
  microdata, engine runs and installs, so no real-run cadence default is
  announced here.

## Done

- `packages/microcosm-calibrate/src/microcosm/calibrate/target_snapshots.py`:
  shared leaf codec. Schema name + version, aggregate-only validation
  (unknown top-level keys and record-level key names refused at any depth),
  compiled-order row check, finite-value check, signed relative error with
  the solver's existing zero-target convention, ordered `(name, value)`
  sha256 identity digest, `TargetSnapshotCadence` (bounded + `EVERY_EPOCH`),
  and `TargetSnapshotWriter` (atomic `latest.json`, write-once history
  chunks, bounded retention with recorded drops).
- Solver integration in `solve.py`: emission from the Adam loop and the
  proximal loop off the estimate tensor the epoch's loss was already
  computed from (no second forward pass, so the hard-concrete gates' RNG
  stream is untouched); budget-search probes carry their own search
  identity; `l0_selection` / `post_l0_refit` carry their phase; one monotone
  sequence counter spans every phase; the closing `selected` snapshot comes
  off the float64 estimates the final diagnostics use.
- Public opt-in parameter `target_snapshots=` on `calibrate`,
  `refit_l0_selection` and `calibrate_l0_refit`; exports in the shard
  `__init__`. Off by default.
- 24 new tests in `packages/microcosm-calibrate/tests/test_target_snapshots.py`
  (flat path; `tools/ci_test_groups.py --verify` = ok, lands in fast `rest`
  and engine `us-am`, never `[defaulted]`).
- Synthetic benchmark + receipts under `experiments/`, explicitly labelled
  synthetic.
- changelog.d fragment `908-calibration-target-snapshots.added.md`.

## Remaining #908 acceptance scope (NOT done here)

- Version-2 staging persistence/upload of the snapshot and its history.
  PR #896 is open, draft, CONFLICTING and absent from `main`; its
  `calibration_progress` filters on `kind == "calibration_epoch"` and its
  per-event schema is `additionalProperties: false`, so it needs a new
  schema name, not a widened one. Nothing here claims remote publication.
- US host wiring (`tools/build_us_fiscal_refresh_release.py` still builds v1
  `StagingTelemetry`) and UK host cadence flags.
- Exact-k ladder phases in `microcosm.build.us_runtime.exact_k_ladder` and
  the UK size-search phases in `uk_runtime.dataset_size` are not wired.
- A test asserting intermediate snapshots never substitute for the canonical
  `calibration_diagnostics.json`.
- Real UK/US target/epoch counts, wall-clock, and upload time, and therefore
  the production default cadence. Forbidden on this host; the default stays
  off.
- Calibration Diagnostics UI (different repository, by the issue's own text).

## Adversarial review round (same lane, before hand-off)

Three independent review lenses (determinism, label honesty, codec/store)
raised 21 findings; each was handed to an adversarial verifier told to refute
it, and 10 survived. All 10 are fixed on this branch, with a regression test
each:

- Enabling the observer aborted runs that succeed without it, in two ways —
  duplicate compiled row labels (`row_name` is the lossy `f"{name}@{period}"`,
  so `("income", 2024)` and `("income", "2024")` collide) and a non-finite
  float32 in-loop estimate (the capped loss absorbs it and the run returns
  weights). The digest now carries the row index instead of refusing
  duplicates, and a non-finite estimate or target serializes as null with a
  `non_finite_rows` count, following `diagnostics._finite`.
- The `selected` snapshot stamped the closing epoch on a retained-best
  iterate, and claimed `best_retained.available: false` on a run that
  retained and returned a best. Both now read the selection receipt.
- The store silently collided across runs sharing a directory (sequences
  restart at 1 per observer) and could prune another run's chunks, or the
  chunk it had just written. Construction now refuses a populated directory
  unless explicitly adopted, a store refuses a second run's snapshots, and
  pruning is scoped to the chunks that writer wrote.
- The aggregate-only scan covered a hand-listed subset of keys; it now walks
  the whole payload. Numeric fields are type-checked, so a stringified number
  no longer validates.

## Source identity re-pins (disclosed)

`microcosm.calibrate.solve` is an attested module: it is in
`_DIRECT_KERNEL_MODULES` (`spec_engine/seeds.py`) and it defines the graph
shard's `calibrate.adam@1` kernel. Editing it legitimately moves several
pinned identities. Each was re-pinned to a value computed from this branch,
after first confirming with the *same* interpreter that pristine `origin/main`
reproduces the committed pins exactly — so the drift is attributable to the
source edit, not to an environment leak:

- `packages/microcosm-graph/tests/fixtures/parity/kernels/calibrate/pins.json`
  (`implementation_hash`, `node_key`). `fit.qrf` and `simulate` untouched; the
  regeneration asserted `graph.json`, `inputs.csv` and `direct.csv` come back
  byte-identical, so the kernel's numbers did not change.
- `EXPECTED_HASHES["seed_protocol"]` and `["seed_map"]` in
  `spec_engine/inventory_coverage.py`, plus the regenerated
  `docs/evidence/spec-engine/us-f0-coverage.json`.
- The resolved-spec golden vectors in
  `test_spec_engine_loader.py` and `test_us_multispine_pool_tool.py`.

No other pin was touched.

**These pins are correct for this branch's base (0c3f4f651) only.**
`origin/main` has since advanced to 116d46ee9 (#912,
`amend-keyed-seed-and-uniform-draws`), which itself edited the attested
`microcosm.fit.qrf` and re-pinned the same loader golden vector and pool-tool
spec digest. Every identity pin above must therefore be recomputed on the
merge ref before this branch merges, per CLAUDE.md's "CI tests the merge ref,
so merge main and re-pin". Do not treat the values here as final.

## How this lane ran the tests (no installs, no uv sync)

An isolated interpreter with this worktree's shard sources ahead of a shared
venv's site-packages, plus an import-path assertion so `microcosm.*` can never
resolve outside this worktree:

```python
# run.py — invoke as: <venv>/bin/python -I -B -S run.py <pytest args>
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_SITE = Path("<a venv with numpy/pandas/scipy/torch/pytest>/site-packages")
sys.path[:0] = [
    *sorted(str(p) for p in ROOT.glob("packages/*/src")),
    str(VENV_SITE),
    str(ROOT),
]
import microcosm.calibrate as _cal
assert str(ROOT) in str(Path(_cal.__file__).resolve()), _cal.__file__
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
```

Note pytest `addopts` already carries `-q`, so gate on the exit code rather
than adding another `-q` (which hides the summary line).

## Next

- Independent review of this slice, then the staging/host wiring as a
  separate slice once #896 lands.
