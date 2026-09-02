# PR #851 review fix report

- Date: 2026-09-02
- Branch: `fix-851`
- Worktree: `/private/tmp/microcosm-fix-851`
- Base: `graph-amendments-impl` at `6348ed7b5f8ee8a76f103193f9a631b52df4faea`
- Verified implementation SHA: `bfa8342758596cd37fda713daab261789a0ee108`
- Push status: not pushed

## Outcome

Every owner-ruled must-fix item and requested follow-up from the local PR #851 review is implemented and covered by a runtime unit regression. The requested graph, kernel, country, byte-parity, lint, format, CI-inventory, and acceptance-burndown commands all exited 0 with the exact summaries recorded below.

The frozen `decl.py`, `kernel.py`, and `docs/graph-interface.lock`, all acceptance tests, and `packages/microcosm-graph/tests/_toy.py` are unchanged from the requested base. No merge, push, or network operation was performed.

## Environment

The worktree and branch were created exactly as requested. The global uv cache was read-only, so I used writable `UV_CACHE_DIR=/private/tmp/microcosm-fix-851-uv-cache`. Because the lane has no network, I seeded that task-local cache from the existing local uv cache, cloned the complete environment into this worktree, and then completed the exact locked sync inside `/private/tmp/microcosm-fix-851` with `UV_OFFLINE=1` and the locally cached build requirements. All six editable package bindings resolve to this worktree.

## Findings

### 1. EXPAND entity-ID integrity

Commit: `cb6f9a00` (`Reject EXPAND overlays of entity IDs`).

- `packages/microcosm-graph/src/microcosm/graph/population.py:1173` rejects an `expand_cells` coordinate equal to its entity ID column while leaving membership overlays legal.
- `packages/microcosm-graph/src/microcosm/graph/population.py:1307` compares every cold final ID index with `target_ids[entity]` after all overlays.
- `packages/microcosm-graph/src/microcosm/graph/population.py:766` requires the warm cached frame’s IDs to equal the exact incumbent-plus-lineage sequence.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:910` proves a malicious ID overlay is rejected on repeated shared-store runs without committing a reusable cache entry.
- Regression: `packages/microcosm-graph/tests/test_graph_population.py:618` rejects a warm frame whose IDs contain the right set but not the exact normative lineage sequence.

### 2. Structural entrant partition values

Commit: `ff9596d4` (`Treat entrant partition overlays as structural`).

- `packages/microcosm-graph/src/microcosm/graph/executor.py:854` adds the graph’s partition coordinate to the structural columns for entrants of the partition entity, so the EXPAND overlay supplies it without a downstream claimant.
- The ordinary-node partition ownership rejection remains at `packages/microcosm-graph/src/microcosm/graph/executor.py:949`.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:974` composes a household entrant with a partitioned graph, checks the cold and warm population/receipt, and separately proves an ordinary partition claimant is still rejected.

### 3. Full Capabilities cache identity

Commit: `df78ae9f` (`Bind cache identity to full capabilities`). The implementation-hash proof was tightened in `5f912b17` and import ordering was normalized in `43d22c6b`.

- `packages/microcosm-graph/src/microcosm/graph/keys.py:103` defines one canonical projection covering determinism, numeric, seed source, structural delta, role, `consumes_se`, dependencies, and tolerance.
- `packages/microcosm-graph/src/microcosm/graph/keys.py:98` normalizes both `-0.0` and `0.0` tolerance values to canonical `0.0`; `packages/microcosm-graph/src/microcosm/graph/keys.py:219` replaces the tolerance-only key factor with the full projection.
- `packages/microcosm-graph/src/microcosm/graph/executor.py:1121` stores that same projection, and `packages/microcosm-graph/src/microcosm/graph/executor.py:1188` makes a cache load miss when the stored contract disagrees with the registered kernel.
- Tolerance was not added to `implementation_hash`; the frozen `kernel.py` is unchanged.
- Regression: `packages/microcosm-graph/tests/test_graph_keys.py:289` varies every Capabilities field and proves positive and negative zero key identically.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:1542` mutates a stored capability block and requires `StoreMiss`.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:1574` uses one store for PARAM-seeded then EXECUTOR-seeded `fit.qrf@1`; their implementation hashes are equal, but the EXECUTOR path does not hit and reaches its parameter-specific rejection.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:1621` changes only QRF tolerance and proves the source-derived implementation hash remains unchanged.

### 4. Rewrite-incumbent tolerances

Commit: `48a5e06a` (`Project tolerances for rewrite incumbents`).

- `packages/microcosm-graph/src/microcosm/graph/executor.py:520` resolves tolerances over the union of explicit input coordinates and rewrite outputs; rewrite incumbents resolve against `input_version`, matching compilation and node-key treatment.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:674` covers both the implicit rewrite incumbent and the same incumbent declared explicitly, asserting the same boundary-owner tolerance in both cases.

### 5. Capabilities construction validation

No lane change was required. The requested base already contains #847’s owner-approved Capabilities validation, and the ruling for this lane prohibited editing frozen `kernel.py`. Its existing coverage ran as part of the graph shard.

### 6. Full-row entrant bridge claims

Commit: `9817a98c` (`Require full-row entrant bridge claims`).

- `packages/microcosm-graph/src/microcosm/graph/executor.py:896` requires entrant materialization bridge ownership to use `Owned.rows == ROWS_ALL` and rejects masked claims before claimant execution.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:1637` uses a claimant mask that excludes the entrant and proves the bridge and downstream claimant kernels remain uncalled.

### 7. Stable manifest population type

Commit: `38af953c` (`Return a stable manifest population view`).

- `packages/microcosm-graph/src/microcosm/graph/manifest.py:38` replaces `_AttachedFrame` class mutation with the documented, zero-copy `PopulationView(Frame)` accessor type.
- `packages/microcosm-graph/src/microcosm/graph/manifest.py:345` normalizes every attached Frame once; `packages/microcosm-graph/src/microcosm/graph/manifest.py:469` always returns `PopulationView`.
- B5 uses `.person`; B6 and B7 use `.person` and `.household`; D6 reads the manifest receipt. Inherited `Frame.person` remains the person path, while `PopulationView.__getattr__` supplies named group-table access such as `.household`. The source Frame’s class is not changed.
- Regression: `packages/microcosm-graph/tests/test_graph_manifest.py:119` proves one return type across versions, repeated-access identity, source-Frame type preservation, zero-copy tables/weights/strata, `.person`, `.household`, and missing-entity behavior.

## Follow-ups

### QRF tolerance wording and pin

Commit: `f63e498b` (`Clarify and repin QRF tolerance budget`), with the implementation-hash regression strengthened by `5f912b17`.

- `packages/microcosm-fit/src/microcosm/fit/kernels.py:52` now uses the owner-recommended provisional one-ULP wording.
- `packages/microcosm-graph/tests/fixtures/parity/kernels/fit.qrf/pins.json:1` is generator-repinned to `9b412a44b3e9d2cfc44b6ed2635e46edefc09c942af10235c56b0839cac906fd`.
- Regression: `packages/microcosm-graph/tests/test_graph_executor.py:1621` binds the declared one-ULP tolerance to the pin and proves tolerance remains outside `implementation_hash`.

### Undeclared partition receipt

Commit: `4bf22e7d` (`Reject undeclared partition receipts`).

- `packages/microcosm-graph/src/microcosm/graph/population.py:2157` rejects `receipt["mass"]["partition"]` when `Graph.mass_partition` is absent.
- Regression: `packages/microcosm-graph/tests/test_graph_population.py:397` supplies an otherwise-valid unpartitioned mass receipt with the reserved block and requires rejection.

### Signed-zero tolerance keying

Included in `df78ae9f` with the full Capabilities projection.

- `packages/microcosm-graph/src/microcosm/graph/keys.py:98` canonicalizes zero-valued `rtol` and `atol`.
- Regression: `packages/microcosm-graph/tests/test_graph_keys.py:324` proves `Tolerance(rtol=-0.0, ...)` and `Tolerance(rtol=0.0, ...)` produce the same node key.

## Verification

All commands below ran in `/private/tmp/microcosm-fix-851` with `UV_CACHE_DIR=/private/tmp/microcosm-fix-851-uv-cache`, `UV_OFFLINE=1`, `POPULACE_FIT_N_JOBS=1`, and `POPULACE_FIT_PREDICT_WORKERS=1` where applicable.

### Graph shard

Command:

```text
uv run pytest packages/microcosm-graph -p no:warnings
```

Summary (exit 0):

```text
233 passed in 91.19s (0:01:31)
```

Count command:

```text
uv run pytest packages/microcosm-graph -p no:warnings -o addopts='-q --import-mode=importlib'
```

Summary (exit 0):

```text
233 passed in 90.87s (0:01:30)
```

### Kernel and country graph tests

Command:

```text
uv run pytest packages/microcosm-fit/tests/test_kernels.py packages/microcosm-calibrate/tests/test_kernels.py packages/microcosm-frame/tests/test_kernels.py packages/microcosm-build/tests/test_uk_graph.py packages/microcosm-build/tests/test_us_graph.py -p no:warnings -o addopts='--import-mode=importlib' -q
```

Summary (exit 0):

```text
29 passed in 47.73s
```

### H1-H3 byte-exact parity

Command:

```text
uv run pytest packages/microcosm-graph/tests/test_acceptance_h_parity.py::test_h1_kernel_parity packages/microcosm-graph/tests/test_acceptance_h_parity.py::test_h2_uk_spine_parity packages/microcosm-graph/tests/test_acceptance_h_parity.py::test_h3_us_post_transfer_parity -p no:warnings -o addopts='--import-mode=importlib' -q
```

Summary (exit 0):

```text
3 passed in 88.27s (0:01:28)
```

### Ruff lint

Command: `uv run ruff check .`

Summary (exit 0):

```text
All checks passed!
```

### Ruff format check

Command: `uv run ruff format --check` on the 10 touched Python files.

Summary (exit 0):

```text
10 files already formatted
```

### CI test-group inventory

Command: `uv run python tools/ci_test_groups.py --verify`

Summary lines (exit 0):

```text
tracked_test_files=353
verification=ok
```

### Acceptance burndown

Command: `uv run python tools/graph_acceptance_burndown.py --verify --baseline origin/main`

Complete summary (exit 0):

```text
group   markers  properties
total         0

 markers  file
       0  packages/microcosm-graph/tests/test_acceptance_a_identity.py
       0  packages/microcosm-graph/tests/test_acceptance_b_ownership.py
       0  packages/microcosm-graph/tests/test_acceptance_c_seeds.py
       0  packages/microcosm-graph/tests/test_acceptance_d_weights.py
       0  packages/microcosm-graph/tests/test_acceptance_e_store.py
       0  packages/microcosm-graph/tests/test_acceptance_f_gates.py
       0  packages/microcosm-graph/tests/test_acceptance_g_legibility.py
       0  packages/microcosm-graph/tests/test_acceptance_h_parity.py
       0  packages/microcosm-graph/tests/test_acceptance_replays.py

green     41  A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 B6 B7 C1 C2 C3 C4 C5 D1 D2 D3 D4 D5 D6 E1 E2 E3 E4 E5 F1 F2 F3 F4 F5 G1 G2 G3 H1 H2 H3
red        0  none
missing    0  none

baseline=origin/main
  ok     packages/microcosm-graph/tests/test_acceptance_a_identity.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_b_ownership.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_c_seeds.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_d_weights.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_e_store.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_f_gates.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_g_legibility.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_h_parity.py: 0 -> 0
  ok     packages/microcosm-graph/tests/test_acceptance_replays.py: 0 -> 0
verification=ok
```

## Uncompleted work

Nothing. Every requested implementation, regression, and verification step completed. The initial global-cache restriction was resolved with the permitted writable task-local cache. No push was performed.
