# Rare signed-tail battery lane progress

## State

Complete on `battery-rare-signed-tails`, based on `2c7a7218`. The mechanism
fix, frozen-evidence classification, receipt closure, regression coverage,
full per-package test suite, Ruff check, journal, and final reports are in the
solution commit containing this file. No pool build was started by this lane.

No gate, incidence band, QED ceiling, support floor, training-cap value, fold,
seed value, exclusion register, logbook chain, or pending-chain file changed.
The host queue still owns the off-chain 1% after-build. The frozen 1% before
projection and deterministic diff tooling are committed, but an actual after
artifact does not exist for this branch.

Authoritative inputs are the arm-split lane's
`experiments/battery_burndown/ADJUDICATION.md` and `adjudication.json`.
Outputs are:

- `FINAL_REPORT.md` — requested completion output;
- `experiments/battery_rare_signed_tails/REPORT.md` — full code-cited report;
- `experiments/battery_rare_signed_tails/realized_regimes.json` — all 48
  check classifications and smallest honest remedies;
- `experiments/battery_rare_signed_tails/baseline_1pct_projection.json` —
  frozen 1% before side.

## Done

- Initialized and committed the lane journal before product work
  (`a5c7310c`). Inspected all four salvage snapshots (`6597ca87`, `8131e7af`,
  `930a3de1`, and newest `bfe794ad`); the newest snapshot supplied the
  overlapping working state, while older broad/unrelated material was not
  recovered. Exact disposition is in `_LANE-NOTES.md`.
- Completed `uv sync --all-packages --extra us` using the writable lane cache.
- Read the comparator-mechanisms authority in full and recomputed every frozen
  fit-boundary regime: 48/48 checks are gated (35
  `zero_inflated_positive`, 13 `three_sign`; 42 targets: 35/7), with no
  degenerate or single-sign QED target. Verified all 42 banked-draw sign
  surfaces against those regimes.
- Classified the 48 checks into nine mechanism classes across fitter regime,
  early/late transfer ownership, and donor support. Five checks (16/28/33/46/75)
  remain additional-evidence starvation blockers; 40 remain owner-calibration
  blockers; three retirement checks (78/80/82) share the fixed support-deletion
  mechanism.
- Proved Keogh is not structurally absent: native ASEC has two positive
  carriers (`2,040`, `30,000`), the old uniform 5,000-row cap retained 0/2,
  and all 1,736,840 finite frozen bank draws are zero. Implemented a
  support-preserving carrier-union cap with common-zero weight calibration;
  cap and seed values remain unchanged.
- Persisted exact realized QRF regimes through ordinary/banked transfer,
  early/late/aggregate receipts, H5 identity, and fail-closed checkpoint
  validation. Advanced only the semantic receipt/materializer identities
  needed to invalidate regime-free or support-deleting artifacts.
- Fixed canonical-JSON resume validation so sorted regime-map keys do not make
  an authentic checkpoint appear corrupt; missing target keys still fail
  closed. Added the serialization regression.
- Regenerated and checked all source-attested bundle/coverage identities.
  `generate_us_bundle_from_constants.py --check` reports US bundle
  `5b0014c3eb6cb121f0a9f2138ab860be30ef2251dd6ff4ec38cbf5f778899554`;
  coverage is 41,381/41,381 fields and 40/40 inventory checks.
- Reproduced the frozen baseline projection byte-for-byte: 127 failure lines,
  93 legs, and 47/48 red QED checks visible at 1%. A baseline-vs-baseline diff
  is exactly empty. Keogh and ordinal 16 are too sparse to be visible at 1%.

## Suite receipts

Final source tree, one pytest process per package shard:

- `microcosm-calibrate`: 201 passed; 2 warnings; 36.25s.
- `microcosm-data`: 275 passed, 1 skipped; 37.50s.
- `microcosm-fit`: 93 passed; 1 warning; 249.72s.
- `microcosm-frame`: 294 passed, 36 skipped; 1 warning; 888.52s.
- `microcosm-build`: 5,973 passed, 39 skipped; 1,941 warnings; 7,858.53s.
- `uv run ruff check .`: all checks passed.
- `tools/generate_us_bundle_from_constants.py --check`: passed.
- `experiments/battery_rare_signed_tails/build_realized_regime_evidence.py
  --check`: passed.

The first complete build-shard run exposed one stale derived-payload test pin;
the authoritative generator was byte-identical, the structural schedule hash
was unchanged, the test pin was corrected to the live derived hash, its focused
node passed, and the complete build shard above was rerun green.

## Next

- Host-queue owner: run the committed 1% off-chain command, then compare its
  `pool.gates.json` with the frozen baseline using `diff_1pct_failures.py`.
- Owner-scheduled full-scale build: verify the Keogh gate-level flip, which a
  1% sample cannot observe.
- Owners of the seven blocked shape families: supply held-out/dense evidence
  before the target-scoped remedies in REPORT §2 are implemented.
- Owners of the five intact-but-sparse donors: supply a dense rung or approve
  a target-specific sparse-tail model. No exclusion is proposed.
