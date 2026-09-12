# Grouped solver x calibration target snapshot integration — 2026-09-12

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
4. Grouped snapshots carry four bounded aggregate scalars in `selection`:
   `rule: "closing_state"`, `constraint_mode: "grouped_upper_bounds"`,
   `grouped_preserve_zeros: <bool>` and `epoch_convention`.

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
mode and whether fixed-zero support is enabled". `rule` is the run's own
`options["iterate_selection"]` and reuses the receipt's `rule` vocabulary; all
four are JSON scalars carrying no record-level content.

`epoch_convention` was added in the review round for a concrete reason: at full
cadence a grouped run stamps `epoch == epochs` **twice**, on the last in-loop
row and on the closing row, and those are two states one accepted projection
apart. More generally, because the loop's single `log_w` overwrite means the
parameters at the top of iteration `e` already are accepted state `e`, the row
stamped `epoch = e + 1` is the fit of the accepted state the private proof seam
stamps `epoch = e` — a systematic one-epoch lag. A host correlating a published
snapshot with the proof ledger by `epoch` would otherwise attribute each row to
the projection after the one it describes.

This is the one place the integration goes beyond the checklist's literal field
list for in-loop snapshots. It is four labels and the documentation of their
contract, not one expression, so removing it is a deliberate choice rather than
a trivial revert — root's call.

### The one change to a file inside the reviewed snapshot head

`target_snapshots.py` is no longer byte-identical to S. Its module docstring
stated that `solve.py` "emits `selection` as the iterate selection receipt",
which this branch made false. The correction is docstring-only and proven so:
parsing both versions and dropping the module docstring yields identical ASTs,
so not one executable line moved, and the module is not in the attested source
inventory, so no identity depends on it. Reverting that one hunk is a clean
option if root prefers exact byte-identity with the reviewed head to a true
statement in the file.

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

## The review round

After the four implementation commits, five independent reviewers (checklist
compliance, numerics/RNG safety, snapshot semantics and privacy, test quality,
merge and identity completeness) audited the branch, each followed by an
adversarial verifier instructed to refute its findings. All seven checklist
sections were confirmed; no blocker survived; the numerical, privacy and
merge-completeness claims were independently re-derived. What the round changed
is recorded above and in the fifth commit. Two process notes worth keeping: the
reviewers observed the worktree while it was still being edited, so only the
numbers taken on the final commit below count; and every identity had to be
recomputed a second time, because the review round edited `solve.py` again.

## Corrections to earlier commit messages on this branch

- Commit `7ac2b864a` says "the us value there is the same 15e61194 …" of
  `test_spec_engine_country_bundles.py`. There is no `us` row in that file: its
  parametrization is `am`/`be`/`uk` only, and the `if country == "us"` branch is
  pre-existing dead code. The US resolved-spec digest is pinned in
  `docs/evidence/spec-engine/us-f0-coverage.json` and in
  `test_us_multispine_pool_tool.py`, not there.
- The digests listed in `7ac2b864a` are **superseded**. The review round edited
  `solve.py` again, so every source-derived identity moved a second time. The
  values in the companion JSON are the final ones; `7ac2b864a`'s are the
  intermediate ones.

## Inherited red: this branch does not have a green PR gate

`packages/microcosm-build/tests/test_us_spine_blindness.py` fails three cases
here — `test_us_runtime_module_classification_is_complete`,
`test_runtime_classification_rejects_a_new_unreviewed_module` and
`test_runtime_population_operators_are_source_spine_blind` (3 failed, 495
passed). They are **inherited, not caused by this work**: that test file and the
four modules it names (`graph_fiscal_dense_calibration.py`,
`graph_fiscal_measurement.py`, `graph_puf55_route_attachment.py`,
`graph_survey_puf55.py`) are byte-identical to the grouped head, which added
those modules without classifying them in the blindness registries. The file
runs in the always-on `fast`/`rest` lane with no engine extras, so it would
block a PR from this branch. Fixing it means classifying four US runtime modules
and resolving dynamic-selector findings in three of them — that is the grouped
branch's own work and is deliberately not attempted here.

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
6. **`selection` is absent, not merely differently shaped, on a third kind of
   run.** An ungrouped closing-state run — gated, mass-conserving, or
   `l2_lambda > 0` — reports `options["iterate_selection"] == "closing_state"`
   yet leaves `selection` null on every row. A consumer cannot read the
   selection rule off `selection` in that case. Documented, not changed:
   touching the ordinary path's `selection` is outside this scope.
7. **A `selected` snapshot is not proof a run finished.** The closing emission
   is checklist-mandated to precede the grouped admission guard, so a sink can
   have persisted a `selected` payload for a run that then raised and returned
   no result. Documented; the ordering itself must not change.
8. **Grouped RNG parity is degenerate evidence.** Grouped Adam draws nothing
   after `calibrate`'s seeding and the grouped validators refuse every gated
   mode, so the RNG comparison can only catch a draw introduced by the emission
   itself, not a reordering of a stream that does not exist. The test says so in
   a comment rather than letting a green assertion imply more.
9. **`microcosm.calibrate.target_snapshots` is not in the attested source
   inventory** (`_DIRECT_KERNEL_MODULES`) and not hashed by
   `CalibrateAdamKernel.implementation_hash`, so a future edit confined to the
   codec — cadence rule, row schema, serialization — would move no identity
   while changing observer semantics. Harmless while no kernel enables
   snapshots, and checklist section 6 explicitly defers it to the owned seam;
   recorded so the deferral is not forgotten at host wiring.
10. **Two pre-existing defects in the reviewed codec, neither touched here.**
    `signed_relative_error` returns null on an overflowing relative error while
    `validate_target_snapshot` refuses exactly that null when both estimate and
    target are finite, so a `|estimate/target| > 1.8e308` row would abort a run
    the module promises never to abort (unreachable while weights are group-
    capped). And the ternary at the signed-relative-error zero-target branch has
    an unreachable arm, the preceding line having already returned.
