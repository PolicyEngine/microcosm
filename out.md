# Lane C — legacy kernels behind the graph protocol

Branch: `node-graph/kernels` (base `node-graph` at `517891f41d091e139b1e34f79c772e0f1265b8a3`)

Draft PR: **not created; no PR URL is available.** `git push -u origin node-graph/kernels` failed with `Could not resolve host: github.com`; `gh auth status` also reports that the stored token is invalid. The installed GitHub connector's write fallback returned `user cancelled MCP tool call`, so it could not create the branch. The complete verified local branch is ready for the two commands shown below when an authenticated network transport is available.

```bash
git push -u origin node-graph/kernels
gh pr create --draft --base node-graph --head node-graph/kernels \
  --title "Wrap legacy computations as graph kernels"
```

## `fit.qrf@1`

Direct computation wrapped:

- Public constructor/fit: `packages/microcosm-fit/src/microcosm/fit/__init__.py:101` (`fit`), called by the adapter at `packages/microcosm-fit/src/microcosm/fit/kernels.py:166`.
- Public first-draw path: `packages/microcosm-fit/src/microcosm/fit/qrf.py:1499` (`FittedRegimeGatedQRF.predict`), called at `packages/microcosm-fit/src/microcosm/fit/kernels.py:178`.

The kernel subsets the declared entity into donor and recipient rows, passes the donor subset of `context.weights[entity].values` to the public DataFrame fit API, then returns the public prediction indexed by recipient entity IDs. It serializes the fitted object immediately before the first draw as `artifacts["model"]`; the docstring explicitly warns that pickle is executable and must only be loaded from a trusted, content-verified store. The receipt carries row counts, seed and provenance, input/resolved weight kinds, donor weight mass, detected regime, fit options, and donor target summary statistics.

Parity fixture and expected-output production:

- `packages/microcosm-fit/tests/fixtures/graph_parity/qrf_donors.csv`: 40 donor rows with two predictors, observed target, IDs, and nonuniform weights.
- `packages/microcosm-fit/tests/fixtures/graph_parity/qrf_recipients.csv`: 12 recipient rows with separate IDs and deliberately large irrelevant recipient weights, proving only donor weights enter the fit.
- `packages/microcosm-fit/tests/fixtures/graph_parity/qrf_expected_hex.csv`: each expected float64 draw encoded with `float.hex()`.
- The expected values were produced from `microcosm.fit.fit(..., seed=947)` followed by the fitted object's first `predict(...)`, with `POPULACE_FIT_N_JOBS=1` and `POPULACE_FIT_PREDICT_WORKERS=1`. The test independently reruns that public path. Kernel output bytes, direct-call bytes, decoded pinned hex bytes, the pre-draw pickle bytes, and the trusted reloaded model's first draw all match exactly.

Capability record (`packages/microcosm-fit/src/microcosm/fit/kernels.py:112`):

- `determinism=SEEDED`: bootstrap, forest, and row draws all depend on the resolved seed.
- `numeric=TOLERANCE_BOUND`: the forest stack includes compiled NumPy/scikit-learn/quantile-forest behavior, so the kernel does not claim cross-platform bitwise stability. H1 parity itself uses no nonzero tolerance (`rtol=0`, `atol=0` in effect through `tobytes()` equality) and is byte-exact in the locked environment.
- `seed_source=PARAM` for `QRF_PARAM_KERNEL` and `EXECUTOR` for `QRF_EXECUTOR_KERNEL`: a literal seed exactly reproduces the legacy call; the executor variant forbids a literal and consumes exactly one integer from `context.rng`.
- `structural=NONE`, `consumes_se=False`: the kernel only owns recipient cells and has no calibration uncertainty input.
- `dependencies=("numpy", "pandas", "scikit-learn", "quantile-forest")`: these distributions govern fitting, bootstrap, and draw behavior and enter the implementation hash.

Frozen-interface accommodations:

- `Node` forbids a node from reading and owning the same column even under complementary row masks. Legal graph declarations therefore use a donor input alias (`donor_target`, e.g. `observed_y`) that the wrapper privately renames to the owned target before the public call. A same-name fallback remains for direct protocol contexts.
- The public QRF constructor does not expose `min_samples_leaf`; the wrapper accepts only its existing effective value `1` and rejects any other value rather than reimplementing the model.
- `Capabilities` is instance-level and context-free. PARAM and EXECUTOR modes are therefore separate configured objects sharing `fit.qrf@1`; a single frozen `KernelRegistry` cannot register both objects simultaneously. A registry must choose the legacy-parity or executor-seeded binding for a run.

## `calibrate.adam@1`

Direct computation wrapped:

- Public solver: `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:1331` (`calibrate`), called by the adapter at `packages/microcosm-calibrate/src/microcosm/calibrate/kernels.py:219` with `method="adam"` and `seed=0`.

The kernel rebuilds the minimal valid `Frame` around the declared calibrated entity without changing its row or weight order, compiles each five-item declaration into the public `Target`/`TargetSet`, and returns `result.frame.weights_for(entity)` from the public result. The node must declare `WeightTransition(entity, "calibrated")`, and the solver `mass` option must agree with the transition. The receipt carries the five-item target tuples unchanged and the public diagnostics payload.

Parity fixture and expected-output production:

- `packages/microcosm-calibrate/tests/test_kernels.py:38` builds six synthetic household records with float64 income/filter measures and nonuniform importance weights, plus the minimal person linkage needed for a direct `Frame` call.
- `TARGET_PARAMS` at line 25 carries float, integer, and separately tested `None` standard errors. Solver options are pinned at line 29 (`max_weight_ratio=2.0`, `epochs=24`, `learning_rate=0.03`, `mass="conserve"`).
- Expected weights are produced at test time by the public `calibrate(...)` call at lines 86–111 on the same values, targets, options, order, and seed. `Weights.values.tobytes()` is identical, and the wrapper receipt's diagnostics equal `diagnostics_payload(direct)`.

Capability record (`packages/microcosm-calibrate/src/microcosm/calibrate/kernels.py:154`):

- `determinism=DETERMINISTIC`, `numeric=BITWISE`, `seed_source=NONE`: the wrapper fixes the legacy Adam seed to zero and returns the direct call's array without recomputation or casting.
- `structural=NONE`: it changes weights, not rows.
- `consumes_se=False`: today's public `calibrate()` has no `se` argument and drops target uncertainty. Declared positive finite `se` values (or `None`) are nevertheless validated and preserved byte/type-for-type in `receipt["declared_targets"]`, satisfying D5 honestly.
- `dependencies=("numpy", "pandas", "scipy", "torch")`: these libraries implement frame/matrix construction and optimization. The implementation hash additionally covers local target, matrix, solver, diagnostics, Frame schema, bundle, and weight sources.

No tolerance was needed: returned weights are byte-identical to the direct call.

## `simulate.rules@1`

Direct computation wrapped:

- Protocol call: `packages/microcosm-frame/src/microcosm/frame/rules.py:80` (`RulesEngine.materialize`), invoked once by the adapter at `packages/microcosm-frame/src/microcosm/frame/kernels.py:173`.
- Real US implementation: `packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py:728` (`PolicyEngineUSEngine.materialize`).

The kernel binds a `RulesEngine` instance to a serializable `engine_ref`, reconstructs a complete Frame from structural IDs/memberships plus declared true-input slices, and calls `materialize` once for the requested period and variables. ID-only group tables absent from the context are recovered exactly from person memberships; no engine input or formula is fabricated. Each returned array keeps its dtype and byte order and is indexed by the owning entity's IDs. The receipt records engine reference, period, variables, and per-output row counts.

Parity fixtures and expected-output production:

- Pure-Python fixture (`packages/microcosm-frame/tests/test_kernels.py:103`): four persons and two households with shifted pandas indexes, nonuniform household weights, two true inputs, and outputs on different entities. The expected arrays come from a separate stub engine's direct `materialize(frame, variables, 2025)` call. Bytes, dtype, shape, IDs, and one-call behavior match.
- Real-US fixture (`packages/microcosm-frame/tests/test_kernels.py:251`): 20 one-person households spanning two states and a range of employment income. The registered `@pytest.mark.requires_us` case at line 296 computes `employment_income` and `household_net_income` for 2024 directly and through the kernel. With cached `policyengine-us==1.819.0` / `policyengine-core==3.31.0`, every returned byte, dtype, shape, and entity ID matches; the test passes. Without the engine it cleanly reports one registered skip.

Capability record (`packages/microcosm-frame/src/microcosm/frame/kernels.py:76`):

- `determinism=DETERMINISTIC`, `numeric=BITWISE`, `seed_source=NONE`: the wrapper returns the bound engine's arrays without numerical transformation and uses no RNG.
- `structural=NONE`, `consumes_se=False`: it owns columns on existing rows and has no calibration targets.
- `dependencies` is supplied by the binding. The pure-Python stub uses `()`. The production US binding uses `("policyengine-us",)`, so the installed engine version enters the implementation hash; the adapter module and local Frame/schema/rules sources are hashed as well.

No tolerance was needed: both stub and real-engine parity checks compare exact bytes.

## Verification

Green checks:

- `uv lock --check --offline`: `Resolved 125 packages`; the regenerated lock differs only by the three requested graph dependency edges.
- Full source suite: `uv run pytest packages/microcosm-fit packages/microcosm-calibrate packages/microcosm-frame -q` completed successfully. A single-quiet rerun reported `566 passed, 55 skipped, 3 warnings`; the registered engine-free kernel case is `1 passed, 1 skipped`.
- Real engine: an isolated environment resolved the cached `policyengine-us==1.819.0`, and `python -m pytest packages/microcosm-frame/tests/test_kernels.py -m requires_us` reported `1 passed, 1 deselected`.
- Frozen graph baseline: `16 passed`.
- `uv run ruff check .`: clean. The six owned Python files also pass `ruff format --check` (`6 files already formatted`).
- `git diff --check`: clean.
- `uv run python tools/ci_test_groups.py --verify`: `tracked_test_files=326`, `verification=ok`.
- `git diff --stat origin/node-graph -- packages/microcosm-graph`: empty.

Wheel boundary command (run once per `microcosm-fit`, `microcosm-calibrate`, and `microcosm-frame`, plus `microcosm-graph` to supply the clean install):

```bash
UV_CACHE_DIR=/private/tmp/microcosm-kernels-uv-cache-clone \
  uv build --wheel --offline --no-progress --no-build-isolation \
  --python /private/tmp/microcosm-kernels-uv-cache-clone/builds-v0/.tmpoJ8xk2/bin/python \
  --out-dir /private/tmp/microcosm-wheel-check.cvle94/wheels \
  packages/<shard>
```

The three touched wheels each contain `Requires-Dist: microcosm-graph<0.2,>=0.1`. Installing all four local wheels together into a fresh Python 3.14 environment resolved and installed 25 packages; `uv pip check` returned `All installed packages are compatible`. Isolated `-I` imports resolved all three kernel modules from that environment's `site-packages`, not the source checkout. Build isolation could not fetch the uncached `packaging==26.3` artifact because network/DNS is disabled, so the successful offline builds used uv's existing complete Hatchling build environment with `--no-build-isolation`.

Base/out-of-scope verification limitations:

- Repository-wide `uv run ruff format --check .` reports 122 pre-existing files on `origin/node-graph` as needing format; none is an owned kernel/test file. Reformatting them would violate this lane's ownership rule.
- The classifier places fit and calibrate in `rest` and `us-am:other-shards`, and frame in `rest` and `shared-spec`. It also reports the mandated `packages/microcosm-frame/tests/test_kernels.py` under `[defaulted]`; removing that requires an out-of-scope edit to `tools/ci_test_groups.py`. The verifier itself is green.
- The requested initial online sync and ordinary shell push cannot resolve external hosts in this sandbox. All runtime dependencies needed for the requested tests were recovered from existing verified caches. The authenticated GitHub connector was attempted as the branch/PR fallback, but its write was cancelled, leaving the push and draft PR as the only incomplete deliverables.
