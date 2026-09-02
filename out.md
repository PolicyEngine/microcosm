# Interface amendments 11–13 implementation report

Branch: `impl-11-13`

Implementation SHA: `fa9c290bcc1d1fde0dadb59079375ec67b61839c`
(the code/test/progress HEAD immediately before this report-only commit)

Worktree: `/private/tmp/microcosm-impl-11-13`

No network access or push occurred. The frozen files
`docs/graph-interface.lock`, `microcosm/graph/decl.py`, and
`microcosm/graph/kernel.py` are byte-unchanged from
`graph-amendments-11-13`.

## B6 — declared entrants

- `packages/microcosm-graph/src/microcosm/graph/population.py:380` emits
  `(new_id, source_id)` lineage with JSON null for entrants;
  `population.py:408` restores nullable integer and boolean sources from a
  cached receipt without turning null into a value.
- `population.py:458` validates target/source dtypes, collision-free new ids,
  known non-null sources, and `entrants=True` for null lineage.
- `population.py:913` builds entrant rows without borrowing an incumbent row,
  requires every carried column, validates declared dtypes, applies structural
  membership overlays, and lets `Frame` linkage validation require memberships
  to name incumbent or entrant groups. Entrant design weights become their
  explicit anchors for later cap checks.
- `packages/microcosm-graph/src/microcosm/graph/executor.py:830` requires every
  entrant data column to have a downstream non-rewrite ownership claim through
  `materialized_expand_outputs`, on both cold execution and cached replay.
- `packages/microcosm-graph/src/microcosm/graph/manifest.py:38` and
  `manifest.py:456` provide entity-name table convenience on attached frames
  through a same-layout, per-instance `Frame` subclass. Object identity is
  preserved and the global `Frame` class is not mutated.

The receipt shape is preserved, including for a warm cache hit:

```text
{'household': ((201, None),), 'person': ((503, 1),), 'release': ()}
False True
```

## C5 — declared tolerance

- `packages/microcosm-graph/src/microcosm/graph/executor.py:85` writes
  `capabilities.tolerance` into every run/cache receipt as either null or the
  exact `{rtol, atol, ulps}` JSON object.
- `executor.py:540` resolves a tolerance for every declared input coordinate.
  Local owners and carried structural owners follow compilation/key reader
  resolution; a rewrite reads its incumbent structural holder rather than
  itself. The immutable mapping is installed in `KernelContext.tolerances`.
- `packages/microcosm-graph/src/microcosm/graph/keys.py:98` handles rewrite
  input identity and, at `keys.py:190`, binds a non-null kernel tolerance into
  the producer key. A changed tolerance therefore invalidates the producer and
  its readers instead of replaying stale gate evidence. Numerically identical
  integer/float declarations canonicalize to the same key.
- `packages/microcosm-graph/src/microcosm/graph/manifest.py:84` and
  `manifest.py:719` save/load the exact tolerance without changing manifest
  schema version; integer-valued inputs normalize to float-valued `rtol` and
  `atol`, so bytes and objects round-trip.
- `packages/microcosm-graph/src/microcosm/graph/view.py:94` and
  `packages/microcosm-graph/src/microcosm/graph/explain.py:463` expose the
  declaration in human-readable views.

Independent cache probes confirmed that changing an owner from `rtol=1e-6` to
`rtol=.25` leaves the source hit but misses the owner and reader; both new
receipts report `.25`. A rewrite probe resolves the bitwise structural
incumbent and executes without a self-key dependency. Manifest byte/object
round trips and `Tolerance(1,2,3)` versus `Tolerance(1.0,2.0,3)` key equality
all returned `True`.

## D6 — partitioned mass

- `packages/microcosm-graph/src/microcosm/graph/population.py:112` extends the
  transient `MassRecord`; `population.py:202` creates the authoritative flat
  plus nested public receipt with collision-checked JSON-string keys.
- `population.py:1692` computes partition accounting before enforcing
  conservation, so a failure names the changed partition value before any
  aggregate failure.
- `population.py:1773` validates the runtime coordinate and broadcasts a group
  partition to persons through membership ids. `population.py:1810` groups
  effective person weights by `(partition, stratum)`; null partition values
  contribute to no partition, as required by the charter, while aggregate
  conservation still covers their mass.
- `population.py:1852` and `population.py:1890` validate optional
  kernel-declared flat/partition accounting, including the exact entity,
  column, and nested before/after shape. Numeric keys round-trip through cache
  via their JSON spellings.
- `packages/microcosm-graph/src/microcosm/graph/executor.py:932` threads the
  graph partition through cold and cached structural application. At
  `executor.py:1558` it authors the receipt alongside kernel evidence and
  converts malformed/colliding evidence into a node-naming rejection.
  Ordinary nodes are rejected if they try to own the structural partition.
- `packages/microcosm-graph/src/microcosm/graph/explain.py:1003` carries the
  nested partition block into mass explanation payloads.

Person- and group-partition cold/warm probes matched, and `conserve` rejection
named partition `2024`. A partition column declared by CREATE but absent from
its runtime frame is rejected by the existing exact CREATE-output validation.

## fit.qrf tolerance measurement

The H1 `fit.qrf@1` fixture was generated independently on native arm64 and on
x86_64 CPython under Rosetta. Both used Python 3.14.4, NumPy 2.4.6, pandas
3.0.3, scikit-learn 1.8.0, and quantile-forest 1.4.2, with fit and prediction
workers pinned to one. Relative difference is
`abs(arm64 - x86_64) / abs(arm64)`; the fixture has no zero reference outputs.

| Comparison | Differing cells | Max absolute | Max relative | Max ULP |
|---|---:|---:|---:|---:|
| arm64 vs x86_64/Rosetta | 0 / 12 | 0.0 | 0.0 | 0 |

Both CSVs equal committed `direct.csv` byte-for-byte:

- direct CSV SHA-256:
  `7b8dbd56c91ee71552ff6d892a42c56494b1813fb5d4b11553a8a8ccc9b90dca`
- float64 payload SHA-256:
  `377763cc9918fc21e1c58e9c7e87c01717475888118b11a752accfc76f3eb525`

`packages/microcosm-fit/src/microcosm/fit/kernels.py:57` now declares
`Tolerance(ulps=1)`: the smallest non-bitwise bound, giving one ULP of margin
above the observed zero-ULP drift. The measurement and margin are recorded in
the constant's comment. Only the fit.qrf implementation hash in
`packages/microcosm-graph/tests/fixtures/parity/kernels/fit.qrf/pins.json`
changed; `inputs.csv` and `direct.csv` did not.

## Acceptance flip

Immediately before the sanctioned flip, the three properties together said:

```text
3 passed in 0.62s
```

The first `uv run python tools/graph_acceptance_flip.py` invocation completed
inside a detached terminal session whose wrapper did not retain stdout. Its
resulting diff removed exactly these three decorators and no other acceptance
test content:

```text
packages/microcosm-graph/tests/test_acceptance_b_ownership.py::test_b6_entrants_are_declared
packages/microcosm-graph/tests/test_acceptance_c_seeds.py::test_c5_tolerance_is_declared
packages/microcosm-graph/tests/test_acceptance_d_weights.py::test_d6_mass_is_partitioned
```

A concurrent second invocation reached the already-flipped files and reported
`flipped 0 properties` with one “marker was not found” warning for each of
those same three tests. The committed flip is therefore evidenced by the exact
three-line deletion diff rather than an invented console line. After restoring
the suite pins, the three properties plus both pins reported:

```text
5 passed in 1.63s
```

The burndown is now `total 0`; the explorer is pinned to `44 green` / `0 red`.

## Verification

The requested graph command ran exactly as supplied (with the offline cache
environment) and exited 0:

```text
uv run pytest packages/microcosm-graph -q -p no:warnings
...............................................................          [100%]
exit 0
```

The repository itself already supplies `-q`, so the requested second `-q`
suppresses pytest's count line. A single-quiet equivalent, preserving importlib
mode and pinning QRF workers to avoid an unrelated process-pool stall, emitted:

```text
207 passed in 573.09s (0:09:33)
```

This is genuinely 207 rather than the requested expected 206. A separate
collection check emitted:

```text
207 tests collected in 2.17s
```

No graph test was added on this branch; the only graph-test edits are the three
sanctioned marker deletions and the two requested suite-pin updates. The count
mismatch is recorded rather than claiming the requested 206. There are zero
xfail markers according to the authoritative burndown below.

The requested cross-package command also ran exactly as supplied and exited 0:

```text
uv run pytest packages/microcosm-fit/tests/test_kernels.py packages/microcosm-calibrate/tests/test_kernels.py packages/microcosm-frame/tests/test_kernels.py packages/microcosm-build/tests/test_uk_graph.py packages/microcosm-build/tests/test_us_graph.py -q -p no:warnings
.............................                                            [100%]
exit 0
```

Its single-quiet equivalent emitted:

```text
29 passed in 52.86s
```

Dedicated byte-exact H1–H3 parity:

```text
3 passed in 114.66s (0:01:54)
```

Focused implementation summaries, verbatim:

```text
39 passed in 3.35s
1 passed in 0.73s
56 passed in 2.08s
37 passed in 3.91s
86 passed in 113.19s (0:01:53)
1 passed in 2.41s
2 passed in 0.91s
38 passed in 1.79s
34 passed in 0.87s
8 passed in 4.96s
3 passed in 0.62s
5 passed in 1.63s
```

Repository hygiene:

```text
uv run ruff check .
All checks passed!

uv run ruff format --check <13 touched Python files>
13 files already formatted

uv run python tools/ci_test_groups.py --verify
tracked_test_files=353
verification=ok

uv run python tools/graph_acceptance_burndown.py --verify
total         0
red        0  none
missing    0  none
verification=ok
```

`git diff --check` and the frozen-file diff check both exited 0 with no output.
No demo structural pin moved or required regeneration.

Two non-evidence retries are not presented as green runs: one summary-only
graph retry was interrupted after `42 passed in 421.17s (0:07:01)` when its
unbounded QRF process pool stalled; a worker-pinned rerun produced the 207-pass
line above. One cross-package summary retry omitted the repository's required
importlib mode and ended `2 errors in 24.88s`; restoring importlib mode produced
the 29-pass line above.

## Interface limitation

`packages/microcosm-graph/src/microcosm/graph/population.py:978` deliberately
rejects a person entrant: frozen `KernelResult` has no field with which a
kernel can supply that new person's mandatory non-null `Frame.strata` value.
The frozen B6 acceptance fixture explicitly uses a household entrant for this
reason. Group entrants, and copied persons assigned to incumbent or entrant
groups, are implemented and green. Supporting a genuinely new person requires
a future interface amendment; neither frozen interface file was edited here.
