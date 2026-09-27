# Grouped solver x calibration target snapshot integration — 2026-09-12

Historical scope note, 12 September 2026: this records grouped core
`2f62bb8448010bcc9b9e1419ea24aafcb85917d1` and its earlier test run. The later
[fiscal-host integration](fiscal-target-snapshot-host-20260912.md) adds the
host observer and PR #914's completed-update convention. The 443-test result
and host-wiring gaps below remain evidence of this earlier revision.

Scope, evidence and residual risks for branch
`grouped-target-snapshot-integration-20260912`. Source-only; no native build,
no PUF fixture, no country build, no release action, nothing pushed.

## What this is

The bounded integration of the reviewed calibration target snapshot observer
(microcosm#908, head `b43369dc49e175803f62020cc1e72fa53926aed8`) with the
reviewed US grouped/fixed-zero Adam solver (head
`536f1ceefcdafda3cc619c14b4da18e012a7be57`), on a fresh `origin/main`
(`116d46ee9dc2aafdc68259b7c06e4c3462522e8b`). Main had not moved from the
shared base, so no main drift had to be preserved.

Executed against the read-only design checklist
`grouped-snapshot-integration-review.md` and its pins. All four pinned source
hashes on both heads matched the pin file byte for byte before the merge.

## What it is not

- **Not** host wiring. `us_runtime/graph_fiscal_dense_calibration.py` still
  calls `calibrate` without an observer, so this branch produces no snapshot
  file for the actual US fiscal path. Wiring it is a separate, explicitly
  host-owned step. No global observer registry, no callback or filesystem path
  in graph `Node.params`, and no claim that any replay reruns the optimizer.
- **Not** a dashboard, a staging upload, or a cadence/performance acceptance.
- **Not** a release, a publication or a promotion of any artifact.

## The one real code change

A clean textual merge leaves grouped runs uninstrumented: `_optimize` returns
early into `_optimize_grouped` before reaching any snapshot hook S added to the
ordinary Adam loop, so a caller passing `target_snapshots` to a grouped
`calibrate` received exactly one closing snapshot and no trajectory.

1. `_optimize` forwards its bound observer into `_optimize_grouped`
   (`snapshots=snapshots`), and `_optimize_grouped` takes a default-`None`
   `snapshots` argument.
2. `_optimize_grouped` emits inside the epoch loop, after the existing progress
   callback and before `backward()`, from the exact float32 `estimate` tensor
   the epoch's loss was computed from: `iterate=current`, `precision=float32`,
   `loss=trajectory[epoch]`, `best_retained={available: False, epoch: None,
   loss: None}`.
3. `calibrate` reads retain-best off the solver actually used rather than
   inferring it from an empty `iterate_selection_receipt`, and stamps the
   grouped closing snapshot at `epoch == epochs`. The receipt itself stays
   empty, exactly as before.
4. Grouped snapshots carry three bounded aggregate scalars in `selection`:
   `rule: "closing_state"`, `constraint_mode: "grouped_upper_bounds"`,
   `grouped_preserve_zeros: <bool>`.

Nothing else in either solver moved. No optimizer or matrix dtype, active mask,
zero template, L2 denominator, log/exp operation, optimizer construction, seed,
projection call, accepted-buffer assignment or `log_w` overwrite changed; there
is still exactly one `log_w` overwrite per accepted projection, the final no-op
projection and accepted-byte equality are unchanged, and the stored-result
admission (weight bytes, fixed-zero mask, bounds, ordered household IDs) still
runs after the last sink.

### Why the `selection` labels

Grouped Adam is a closing-state algorithm; it never runs the retain-best rule.
Without a label, its `best_retained.available: False` is indistinguishable from
an ordinary run whose retain-best rule happened to be off, and a consumer would
read the two the same way. Checklist section 2 sanctions exactly this —
"Optional grouped labels must be bounded aggregate scalars, e.g. constraint
mode and whether fixed-zero support is enabled". The labels mirror the run's
own `options["iterate_selection"]` and reuse the receipt's `rule` vocabulary;
they are three JSON scalars carrying no record-level content. This is the one
place the integration goes beyond the checklist's literal field list for
in-loop snapshots, and it is a single expression to remove if root disagrees.

## Evidence

- Grouped runs are bit-identical with the observer on and off across
  dense x CSR, positive x fixed-zero support, and plain / warm-started / L2:
  returned weights, loss trajectory, options, closing loss, stored frame
  weights, ordered IDs, fixed-zero bytes, and the full sequence of private
  post-projection proof payloads.
- Exactly 20 `_apply_constraint` and 3 `problem.estimates` calls with the
  observer on and off on the 20-epoch fixture: cadence buys observation, not
  evaluation. Torch and numpy RNG state are identical after each pair of runs.
- Each in-loop row carries the exact float32 tensor the epoch's loss was
  computed from, verified against a wrapped `_apply_constraint`, and differs on
  all 20 epochs from a recomputation on the accepted vector the private
  observer reports next.
- The closing row equals the reused float64 final diagnostics estimates.
- A sink that ruins every delivered payload changes neither the weights nor the
  next payload. A sink that mutates live frame IDs at the closing emission
  still trips the ordered-ID admission guard.
- Zero epochs through the internal grouped seam emits nothing and evaluates
  nothing.
- The deliberately oscillating example still selects grouped closing output
  while ordinary Adam selects an earlier best, and each solver's closing
  snapshot now says which rule produced it.

Identity recomputation. Editing `solve.py` — an attested kernel module — moves
the seed protocol digest, every resolved-spec digest that folds it in, and the
calibrate kernel's implementation hash. Each value below was recomputed against
this checkout with the isolated Python 3.14 interpreter; none equals either
branch's value, so none is an ours/theirs choice. The simulate parity pin and
the fit.qrf pin were deliberately left alone: neither hashes
`microcosm.calibrate.solve`, and both are byte-unchanged from the grouped head.

## Residual risks

1. **Cadence cost on a real US run is unmeasured here.** The only overhead
   evidence in the tree is #908's synthetic benchmark; this branch adds no
   native measurement. A country-scale cadence choice is still unowned.
2. **The `selection` labels are an integration judgement**, not a checklist
   requirement, on the in-loop rows specifically. See above.
3. **Host wiring remains the gap between this and any dashboard.** Green tests
   here say the solver seam is correct; they say nothing about whether any real
   US calibration emits anything, because none does yet.
4. **The identity recomputation is only as good as its interpreter.** Digests
   fold locked dependency versions; a lane on different locked versions will
   compute different parity keys. The values here were taken under
   numpy 2.4.6 / pandas 3.0.3 / scipy 1.17.1 / torch 2.12.0 on
   arm64/darwin/py3.14, which is what the calibrate parity pin records.
5. **The engine-gated identity tests were exercised with the engines present**
   in the borrowed environment. A lane without the US extra will skip
   `test_spec_engine_loader.py`'s golden and the country-bundle proofs rather
   than check them.
