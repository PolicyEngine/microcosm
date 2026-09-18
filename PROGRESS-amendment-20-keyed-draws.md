# Amendment 20 — keyed draw streams (lane journal)

Branch: `amend-keyed-seed-and-uniform-draws`, cut from `origin/main` at `3094bfe84`.
Worktree: `~/PolicyEngine/_worktrees/microcosm-amend-keyed`. No push, no new branches.
Platform of record for H1 pins: **arm64/darwin/py3.14** (Python 3.14.4) — the
fixture's authoring platform.

*Journal, not state: accurate as written on 2026-09-11. Check git and the
tracking issue for what is true later.*

Full receipts — every identity, command and exit code, and the three open
decisions with their evidence — are in
[`experiments/amendment-20-keyed-draws-receipts.md`](experiments/amendment-20-keyed-draws-receipts.md).

## State

Implementation, tests, pins, charter and changelog are landed and committed.
`packages/microcosm-graph/tests` and `packages/microcosm-fit/tests` are green.

> Historicized 2026-09-12: the paragraph below described the branch as the lane
> left it. The seed digests were re-pinned when the branch was stacked on
> amendment 19 and pushed as PR #912, so nothing here is red any more; the
> `-0.0` question in item 7.3 of the receipts was settled by normalising
> signed zero in `_coordinate`.

**The branch cannot go green in CI as it stands**, and that is deliberate: the
`qrf.py` edit moves the spec-engine seed digests, leaving seven tests red in a
lane that runs on every PR. The lane brief said report the drift, not re-pin
it, so the branch carries it and the decision goes to the merge owner — see
**Open for decision** item 1, which carries a tested recipe.

## Done

1. `graph/kernel.py`: `SeedSource.KEYED` plus the `KernelContext.rng` docstring.
   `ArtifactValue` / `KernelContext.artifacts` deliberately NOT brought over.
2. `graph/randomness.py` (new, 69 lines): `keyed_uniform`, taken verbatim from
   `origin/microcosm-us-launch-integration-20260909`.
3. `graph/__init__.py`: `keyed_uniform` exported, inserted in sorted position
   among the lowercase callables (`graph_to_json`, `keyed_uniform`, `load_source`).
4. `fit/qrf.py`: `_draw_target_from_uniforms` + `predict_from_uniforms`, taken
   verbatim from the same branch (`6e3907f86`); the diff applied with no conflict.
5. `docs/graph-interface.lock`: `kernel.py` re-recorded
   `eaf07da2… → 3483d091b03b19ae35c0268c01cb9e0f76c4cd63742083130567321d70da6048`.
   The lock is plain `shasum -a 256` of the file bytes (confirmed against the
   unchanged `decl.py` line).
6. Tests: `test_graph_randomness.py` (new), `SeedSource.KEYED` contracts in
   `test_graph_kernel_contract.py`, `test_qrf_stateless.py` (new, verbatim from
   the integration branch), a `QRFKernel` non-widening guard in microcosm-fit's
   `test_kernels.py`, and `test_graph_parity_pins.py` (new).
7. `tools/graph_parity_repin.py` (new) + the re-pinned `fit.qrf` fixture.
8. `docs/graph-acceptance.md` amendment 20 + a `changelog.d` fragment.

**No `test_acceptance_*.py` file was edited at all**, so the "acceptance-suite
edit is its own commit" rule never had to be exercised.

### Second pass, after an adversarial review of the branch (same day)

Six independent reviewers read the working tree; each finding was then put to
two adversarial verifiers. Four findings survived and were fixed:

9. **The re-pin now checks the environment it derives foreign keys from**
   (`test_graph_parity_repin.py`, new, 8 tests). The guard added in the first
   pass reproduced every pinned key from `pins["implementation_hash"]` — which
   substitutes away the only input carrying dependency versions, so *any*
   environment reproduced the pins and the check could not see the drift it
   existed to catch. `pins.json` already recorded the versions the keys were
   taken under; nothing read them. `repin` now refuses unless they equal this
   machine's installed versions, before deriving anything.
10. **The re-pin docstring no longer claims what the code does not do.** It had
    said the reproduction loop "proves the assumption rather than asserting it"
    and that "only one locked environment can do that". Neither was true. It now
    states what each of the two checks establishes.
11. **The final-bin closure is pinned.** Amendment 20 claims two deliberate
    differences from the generator path; only the strict comparison was tested.
    Deleting `cumulative[:, -1] = 1.0` left all fourteen stateless tests green.
12. **The keyed-kernel contract test asserts the half it only asserted in
    prose.** Its toy body never referenced `context.rng`, so "a keyed kernel
    simply does not spend it" held by construction. It now deep-copies the bit
    generator state across the call; adding `context.rng.random(1)` to the body
    makes it fail.

Each fix was mutation-tested: the mutation that breaks the behaviour makes the
new assertion, and only it, go red.

## Key findings (verified this session)

- The integration branch carries **no** executor change for `KEYED` and **no**
  `fit/kernels.py` change. `grep` over `origin/main`'s `executor.py` finds no
  `seed_source` branch. Honouring `KEYED` therefore required no executor edit.
- `QRFKernel.implementation_hash()` hashes `microcosm.fit.qrf`'s module bytes,
  so editing `qrf.py` moved `fit.qrf@1`'s implementation hash
  (`02db8f5c… → d1f8b192…`) and all three pinned platform node keys.
- **Foreign-platform node keys are derivable here because the environments
  agree — not because the fingerprint is the only channel.** A platform reaches
  a key two ways: the fingerprint string, and the implementation hash, into
  which `source_hash` folds `f"{distribution}=={version}"` for every declared
  dependency. Re-deriving the three OLD pinned keys with the OLD implementation
  hash substituted and only the fingerprint varied reproduces all three — which
  establishes that each pin is the key its platform computed *under that hash*,
  and establishes nothing about the dependency channel, because substituting
  the hash is exactly what removes it. That channel is now checked separately
  and explicitly. (The first pass wrote "so the platform reaches a key as that
  string and nothing else"; the experiment could not support it.)
- **`tools/graph_parity_fixtures.py` must not be edited.** `ParityCsvSource`
  and `ParityRulesEngine` are defined there, so its bytes are inside
  `ParityCsvSource.implementation_hash()` and
  `SimulateRulesKernel.implementation_hash()`. Measured: adding the re-pin code
  there moved the calibrate node key `184ccd0a → f8c9ed4b` with its
  implementation hash unchanged. The re-pin logic therefore lives in a sibling
  module. Reverted.
- `generate()` resets `pins["platforms"]` to the local platform alone, so a bare
  regeneration would have dropped both `x86_64/linux` pins and silently put H1
  on its off-platform branch there (which asserts no bytes).
- `direct.csv` is byte-identical before and after on every platform
  (`7b8dbd56c91ee71552ff6d892a42c56494b1813fb5d4b11553a8a8ccc9b90dca`).
- **`origin/main` moved during the lane** (to `e6d362b7e`, PR #909, object-dtype
  storage hashing). It touches `graph/store.py` and `graph/population.py`;
  neither enters `QRFKernel.implementation_hash()`. Merging is clean
  (`git merge-tree`, no file touched by both), and running H1 parity, the pin
  tests and the tolerance pin with main's post-#909 versions of those two files
  in place is green, so main's advance does not disturb this lane's pins.

## Open for decision (Max / the merge owner)

1. **The spec-engine seed digests are stale on this branch, and that is
   merge-blocking.** `tools/spec_engine_coverage.py --check` exits 1 (0 on
   `origin/main`). The cause is `packages/microcosm-fit/src/microcosm/fit/qrf.py`
   alone, bisected: `microcosm.fit.qrf` is in `_QRF_KERNEL_MODULES`
   (`spec_engine/seeds.py:353-362`), whose `source_inventory_sha256` hashes each
   module's exact installed source bytes. `SeedSource.KEYED`, `randomness.py`
   and the export move none of it. **Seven tests are red**: one failure in
   `test_spec_engine_inventory_coverage.py` and six fixture errors in
   `test_spec_engine_coverage_tool.py`. Both files run in the **`engine-shared`**
   CI lane, which has no `if:` condition — it runs on every PR and the aggregate
   gate requires it — so this cannot be merged as it stands. (The integration
   branch re-pinned these same two digests itself, in `1734b9e90`; its values
   cannot be copied here because they also fold in its own `acs_transfer` and
   housing changes.) The lane brief said
   report, do not re-pin, so the branch carries the drift. The re-pin is two
   pin values plus a regenerated evidence file; it was applied, verified green,
   and reverted this session, so the recipe in the lane report is tested rather
   than proposed.
2. **Amendment numbering.** `main`'s list ends at 18, so 19 is the next free
   number by its own arithmetic. The brief assigns 20 (the artifacts lane owns
   19). Two unmerged commits on the
   `candidate-quality-producer-integration-20260905` family already claim both:
   `3ff92b0ae` = 19 ("Typed artifacts and stable draw coordinates", which
   bundles this lane's subject), `d2043d85e` = 20 ("Failed typed evidence
   remains a failed gate"). Neither is on `main` or on the integration branch.
   Concrete consequence of leaving the gap: CommonMark renumbers an ordered
   list from its first item, so the entry *renders* as 19 however it is
   written — colliding with the number the artifacts lane expects.
3. **`keyed_uniform` gives `-0.0` and `0.0` different draws** (verified:
   `0.3464688…` vs `0.3251263…`), though they are `==` and hash-equal in Python.
   `canonical_json` emits them distinctly and `_coordinate` does not normalise
   the sign, so a float coordinate arriving as `-0.0` on one run re-randomises
   that row. Not fixed here on purpose: `randomness.py` is byte-identical to the
   integration branch that owns it, and `"sha256-u53-v1"` is a *versioned*
   normative algorithm — changing which uniform a coordinate draws is a version
   bump, not a drive-by fix. For that lane to decide.

## Next

- Nothing blocking in this lane's own work. A reviewer should decide (1), (2)
  and (3) above; (1) must be resolved before the branch can go green in CI.
