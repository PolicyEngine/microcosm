# Lane D: release semantics, manifest persistence, and H1 fixtures

Date: 2026-09-01

Branch: `node-graph-release`

Checkout: `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/node-graph/release/repo`

## Outcome

The runtime, persistence, serialization, and deterministic real-kernel fixture
work possible without contradicting the frozen interfaces or unowned fixtures
is implemented and committed. F2, both F3 cases, and F5 are green; the
sanctioned flip tool removed exactly those four passing markers. The graph
package finishes with **175 passed and 4 xfailed**.

The requested endpoint of only H2/H3 remaining xfailed is not honest on the
checked-in acceptance fixtures. F4 and H1 contain independent contradictions
with the frozen interfaces and the assignment's explicit rulings. Their exact
corrections are under “Interface change requests” below. I did not weaken role,
tier, numeric-capability, graph-deserialization, or population-version semantics
to manufacture two false greens.

`decl.py` and `kernel.py` are unchanged from `origin/node-graph`. No network,
push, PR, release, or publication operation occurred.

## Implementation by file

- `packages/microcosm-graph/src/microcosm/graph/executor.py`
  - Validates role-gate outcomes against `GATE_OUTCOMES`.
  - Converts only role-gate exceptions into failed string verdict artifacts and
    exception evidence; non-gate exceptions retain abort behavior.
  - Derives a release tier from transitive role-gate ancestry, validates the
    release kernel's owned tier, and records the derived tier and ancestry.
  - Overlays the decision-dependent release outcome without putting decisions
    into node keys, artifacts, cached results, or the derived tier.
- `packages/microcosm-graph/src/microcosm/graph/manifest.py`
  - Adds canonical `save`, `load`, and `load_certified` persistence with schema
    version 1, manifest key, content-addressed body, tier, known failures,
    decisions, complete portable node receipts, and artifact identities.
  - Rederives the release tier from the recorded gate receipts, recomputes known
    failures, validates every duplicated top-level projection and the manifest
    key, and confirms every column/frame/weight/opaque artifact exists.
  - Converts malformed persisted data, including noncanonical body values, to a
    `StoreCorruptError` whose message identifies the claimed manifest key.
- `packages/microcosm-graph/src/microcosm/graph/serialize.py`
  - Adds canonical, lossless `graph_to_json(graph) -> str` and
    `graph_from_json(text) -> Graph` for the frozen declarations. Enums use their
    values, parameters are JSON values, and declaration tuples use JSON arrays
    and are restored as tuples.
- `packages/microcosm-graph/src/microcosm/graph/__init__.py`
  - Exports only `graph_to_json` and `graph_from_json`; the public `Graph` remains
    the frozen declaration class.
- `packages/microcosm-graph/tests/test_graph_executor.py`
  - Covers all five gate outcomes, gate exception evidence, non-gate aborts,
    transitive ancestry, tier disagreement, and decision/cache invariance.
- `packages/microcosm-graph/tests/test_graph_manifest.py`
  - Covers persistence, every duplicated-field mismatch, artifact existence,
    certified/unreached loading, failed-gate tier rederivation, noncanonical
    body errors, and explicit rejected-node accounting. All 24 tests pass.
- `packages/microcosm-graph/tests/test_graph_serialize.py`
  - Covers canonical round-trip and malformed declarations, plus an honest H1
    proxy that validates real kernel pins and direct bytes. Fit and simulate run
    through the executor; calibration is checked at the real wrapper's weight
    result boundary pending the population-contract ruling below.
- `packages/microcosm-graph/tests/fixtures/parity/kernels/**`
  - Commits `graph.json`, `inputs.csv`, `direct.csv`, and `pins.json` for
    `fit.qrf`, `calibrate`, and `simulate`, with producer provenance and ignored
    local `_store/` directories.
- `tools/graph_parity_fixtures.py`
  - Generates all three fixture cases from real `QRF_PARAM_KERNEL`,
    `CALIBRATE_ADAM`, and `SimulateRulesKernel` bindings.
  - Defines the importable fixture CREATE kernel and pure-Python rules stub.
    The test-accessible stub is
    `tools.graph_parity_fixtures.ParityRulesEngine`; the executable registry is
    `tools.graph_parity_fixtures.parity_registry()`.
- `PROGRESS.md`
  - Preserves the shared historical ledger and adds the committed Lane D state,
    completed work, evidence, and next owner actions.
- Acceptance marker files
  - `tools/graph_acceptance_flip.py` was the only writer. It removed F2, F3,
    F5, and replay-F3, then a final audit reported `no strict xpass`.

## Exact gate and release rules

1. A node is a gate only when its registered kernel has
   `Capabilities.role == KernelRole.GATE`.
2. A role-gate receipt outcome must be exactly one of `pass`, `fail`,
   `evidence_absent`, `not_applicable`, or `unreached`; any other value rejects
   the node.
3. An exception from a role-gate kernel produces `outcome="fail"`, writes
   `"fail"` to every declared string verdict cell, records exception type and
   message under `evidence`, and continues. An exception from a compute or
   release kernel rejects the node and aborts the run.
4. A release's gate set is every transitive predecessor whose registered role
   is `GATE`. The tier is `certified` iff every such outcome is `pass` or
   `not_applicable`; otherwise it is `evidence`. Thus `fail`,
   `evidence_absent`, and `unreached` block certification. An empty gate set is
   vacuously certified.
5. A release must own exactly one string `tier` column. Every owned value must
   equal the derived tier; disagreement raises `NodeRejectedError`. The
   release receipt's `tier` is always the executor-derived value.
6. `params["requires_decisions"]` must be a tuple of unique nonempty decision
   names. If any name is absent, the release outcome is `unreached`; otherwise
   it is `pass` for a certified tier and `fail` for an evidence tier.
7. Decisions affect the run manifest and release receipt outcome only. They do
   not affect any node key, artifact, cached release result, or derived tier.

## Exact manifest rules

- `save(path)` writes canonical JSON with integer `schema_version=1`, the
  content-addressed manifest `key`, derived `tier` (or null without a release),
  sorted `known_failures`, decisions, the content-addressed body, portable node
  receipts, and run metadata.
- The body contains sorted node keys and canonically sorted decisions. The
  manifest key is recomputed from that body.
- Tier is rederived from the release receipt's unique role-gate ancestry and
  the corresponding gate receipts; the release receipt's stored tier must
  agree. Gate outcomes are revalidated against the closed set.
- `known_failures` is the sorted union of role-gate ids whose outcome is not
  `pass`/`not_applicable` and nodes explicitly marked rejected.
- `load(path, store)` requires every persisted top-level field, checks the body,
  key, tier, known failures, decisions, and portable provenance, then checks
  each referenced artifact in the store with its declared kind. Missing
  artifacts raise `StoreMissError`; malformed or inconsistent persistence
  raises key-bearing `StoreCorruptError`.
- `load_certified` first refuses a release outcome of `unreached`, then refuses
  any tier other than `certified` with an evidence-tier error. A successful
  load preserves the saved manifest key.

## H1 fixture generation

Standard command:

```sh
uv run python tools/graph_parity_fixtures.py
```

The sandbox's prebuilt offline environment was invoked as:

```sh
UV_CACHE_DIR=/private/tmp/node-graph-release-uv-cache \
  uv run --no-sync python tools/graph_parity_fixtures.py
```

`fit.qrf` pins seed 947 and the real parameter-seeded QRF adapter;
`calibrate` pins seed 0 and the real Adam adapter's weight result; `simulate`
pins the real rules adapter bound to `ParityRulesEngine`. Running the generator
a second time changed no tracked byte (`git status --porcelain` was empty).

## Verification

- Owned-file format check: `8 files already formatted`.
- Repository Ruff: `All checks passed!`.
- Graph package: **175 passed, 4 xfailed, 179 collected**.
  Remaining xfails are F4, H1, H2, and H3.
- Acceptance burndown: **green 33; red 4 (F4 H1 H2 H3); missing 0;
  `verification=ok`**.
- CI inventory: `tracked_test_files=345`; `verification=ok`.
- Real kernel regressions: **15 passed, 1 skipped**.
- Frozen-interface diff for `decl.py` and `kernel.py`: empty.
- `git diff --check origin/node-graph..HEAD`: clean.
- The checkout was clean before this report was written.

## Interface change requests

1. **F4 toy fixture correction (no frozen-interface change).** The checked-in
   `ReleaseTier` certifies only an all-`pass` input, but the ruling says
   `not_applicable` also certifies and requires executor rejection on a kernel
   disagreement. Change that predicate to accept both certifying outcomes.
   Separately, F4 replaces the gate kernel with `bad.raise@1`, which the toy
   registry declares as role `COMPUTE`; the ruling requires compute exceptions
   to abort. Add a distinct throwing kernel registered as role `GATE` and use
   it in F4, preserving `bad.raise@1` for the non-gate abort contract.
2. **H1 acceptance consumer correction (no frozen-interface change).** H1 calls
   `Graph(**json.loads(graph.json))`, which cannot construct nested frozen
   declarations from JSON dictionaries. It must call `graph_from_json`.
   It also passes `toy_registry()`, which contains neither the fixture source
   nor any real parity kernel; it must use `parity_registry()`. Assert the real
   per-kernel numeric capability (`fit.qrf@1` is deliberately
   `tolerance_bound`, while calibrate/simulate are bitwise), validate the pin's
   ref/hash/dependencies, and compare calibration through `NodeReceipt.weight_key`
   rather than iterating only column artifacts.
3. **Calibration population-version ruling.** `CALIBRATE_ADAM` declares
   `StructuralDelta.NONE` while requiring and returning a `WeightTransition`.
   The integrated population runtime currently refuses that nonstructural
   weight mutation; allowing it as-is would create implicit ordering with no
   new population version, while declaring the fixture node `REWEIGHT`
   conflicts with the real kernel capability. Choose and authorize one coherent
   contract: make the calibration adapter a structural `REWEIGHT` kernel and
   declare its base, or
   amend compiler/population semantics to give nonstructural weight transitions
   explicit version/dependency behavior. Until then, executor-level calibration
   parity cannot be asserted honestly.

No change to the frozen `decl.py` or `kernel.py` is requested for the completed
F2/F3/F5 behavior itself.

## Local handoff

- Branch: `node-graph-release`
- Path: `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/node-graph/release/repo`
- Push status: not pushed, as required.
