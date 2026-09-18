# Build lane report — amendment 20, keyed draw streams

Repo: `PolicyEngine/microcosm`. Branch: `amend-keyed-seed-and-uniform-draws`,
cut from `origin/main` at `3094bfe84`. Worktree:
`~/PolicyEngine/_worktrees/microcosm-amend-keyed`. Nothing pushed; no new
branches; the stash was never touched; `uv.lock` unchanged.

**Platform string of record for every H1 measurement below:
`arm64/darwin/py3.14`** (Python 3.14.4) — the `fit.qrf` fixture's own authoring
platform, so the local pin is *produced* rather than derived.

**Three things need your decision, in §7.** One of them is merge-blocking:
the branch cannot go green in CI as it stands, by design of the brief.

> Historicized 2026-09-12: the decisions in §7 were taken when the branch
> was stacked on amendment 19 and pushed as PR #912 — see the notes under
> §5, §7.1, §7.2, §7.3 and §8. The report below is the lane's record, not
> the branch's state.

This report is also committed at
`experiments/amendment-20-keyed-draws-receipts.md`, because the root `out.md`
is a tracked file holding a different lane's report — see §7.4.

---

## 1. What landed

Sixteen commits. Red-before-green wherever behaviour changed.

**First pass — the amendment itself:**

```
b46955507  Start the amendment 20 lane journal (keyed draw streams)
8d55f8172  Red: keyed draw streams and SeedSource.KEYED have no implementation yet
e75bd6b91  Green: SeedSource.KEYED and keyed_uniform; relock kernel.py
2c4dc34fd  Red: FittedRegimeGatedQRF has no predict_from_uniforms yet
82e11113e  Green: QRF draws from caller-supplied per-row uniforms
80c41e081  Add a re-pin path that keeps every pinned H1 platform, and check them all
c96a3f412  Re-pin the fit.qrf H1 fixture; direct.csv is byte-identical
77b7daabe  Amendment 20: keyed draw streams; changelog fragment
2d0681c8f  Bring the lane journal up to the landed state
```

**Second pass — after an adversarial review of the branch (§6):**

```
457b36944  Red: the re-pin derives foreign keys without checking the environment
8b284c584  Green: a re-pin checks the environment it derives foreign keys from
12637772a  Pin the final-bin closure the amendment claims
3beca70a5  Make the keyed-kernel contract test assert the half it only asserted in prose
6ea291779  Assert the authoring pin against the fixture, not against the helper
8fb973941  Say precisely which node keys amendment 20 moves, and why
74abdbbaf  Bring the lane journal up to the reviewed state
```

`git diff --stat $(git merge-base HEAD origin/main)...HEAD` — 16 files,
**+1495 / −6**:

| File | ± | What |
|---|---|---|
| `packages/microcosm-graph/src/microcosm/graph/kernel.py` | +4 −2 | `SeedSource.KEYED = "keyed"` and the `KernelContext.rng` docstring. **Nothing else** — no `ArtifactValue`, no `ArtifactType` import, no `KernelContext.artifacts`, no `__all__` entry for them. |
| `packages/microcosm-graph/src/microcosm/graph/randomness.py` | +69 | New. `keyed_uniform` + `_coordinate`, **byte-identical** to the integration branch's copy (sha256 `797fa7e4c7cc2fc81312f7886521e9b0f3e02be577e601a9cabc24362485847a`; `diff` empty). |
| `packages/microcosm-graph/src/microcosm/graph/__init__.py` | +2 | `keyed_uniform` imported and placed in `__all__`'s lowercase tail between `graph_to_json` and `load_source`. |
| `packages/microcosm-fit/src/microcosm/fit/qrf.py` | **+84 −0** | `_draw_target_from_uniforms` and `predict_from_uniforms`. Blob-identical to `6e3907f86` (`e8bb2c95611c1d650dfc78785ac90f7325cc3f75`). The `−0` is the mechanical form of the additivity claim: no existing line is touched. |
| `docs/graph-interface.lock` | +1 −1 | `kernel.py` re-recorded. |
| `docs/graph-acceptance.md` | +35 | Amendment 20. |
| `changelog.d/amend-keyed-seed-and-uniform-draws.added.md` | +1 | towncrier fragment. |
| `packages/microcosm-graph/tests/test_graph_randomness.py` | +205 | New, 12 tests. |
| `packages/microcosm-graph/tests/test_graph_kernel_contract.py` | +98 −8 | 3 new `SeedSource.KEYED` contract tests, one of them strengthened in the second pass. |
| `packages/microcosm-fit/tests/test_qrf_stateless.py` | +182 | New. 14 tests verbatim from the integration branch, plus the final-bin-closure test the second pass added. |
| `packages/microcosm-fit/tests/test_kernels.py` | +16 | One guard: `SeedSource.KEYED` must not widen `QRFKernel`. |
| `packages/microcosm-graph/tests/test_graph_parity_pins.py` | +116 | New, 11 tests — every pinned H1 platform key is checkable from every platform. |
| `packages/microcosm-graph/tests/test_graph_parity_repin.py` | +204 | New, 8 tests — the re-pin tool's refusals (second pass). |
| `tools/graph_parity_repin.py` | +328 | New. The re-pin path. |
| `packages/…/fixtures/parity/kernels/fit.qrf/pins.json` | +1 −1 | Re-pinned. |
| `PROGRESS-amendment-20-keyed-draws.md` | +151 | Lane journal, committed from the first commit. |

**`tools/graph_parity_fixtures.py` is NOT touched** — §4 explains why that
matters. **No `test_acceptance_*.py` file is touched at all**, so the
"acceptance-suite edit is its own commit" rule never had to be exercised.

---

## 2. What was deliberately NOT brought over

- **`ArtifactValue` / `KernelContext.artifacts`** — a separate lane owns them.
  Verified absent: neither `git diff origin/main...HEAD` nor the working-tree
  diff contains any occurrence of `ArtifactValue`, `ArtifactType`, or
  `artifacts=` in code. The integration branch's `__post_init__`, its
  `__init__.py` exports of `ArtifactInput`/`ArtifactOutput`/`ArtifactType`/
  `ArtifactValue`/`SourceBytesCodec`/`load_source_bytes`, and its
  artifacts-dependent new module `fit/qrf_target.py` are all absent.
- **Any `executor.py` change.** The brief allowed "the minimal coherent
  subset". The measured answer is **nothing**:
  - The integration branch's executor diff is +419 lines of typed artifacts,
    lazy population retention and receipt metadata, and contains **no
    occurrence of `SeedSource`, `KEYED`, or `keyed`**.
  - `grep -n "seed_source"` over `origin/main`'s `executor.py` returns nothing.
    The executor never branches on the seed source; it always builds
    `KernelContext.rng` from the node key and lets the kernel decide whether to
    spend it. A `KEYED` kernel simply does not.
  - Every other `SeedSource` use site in the repo was checked for a member
    enumeration that would reject `"keyed"`; none rejects it. The only site
    where a new member changes accept/reject behaviour is the capability
    projection, which the contract tests cover.
- **Any `microcosm-fit/kernels.py` change.** The integration branch's diff
  there is **empty**: it wires no `KEYED` draws into `fit.qrf@1`.
  `QRFKernel.__init__` still refuses anything but `PARAM`/`EXECUTOR`, and this
  branch adds a test asserting it keeps refusing.

---

## 3. H1 parity — the pins, and the `direct.csv` byte-identity proof

### 3.1 Why the pins had to move

`QRFKernel.implementation_hash()` is
`source_hash(type(self), fit_qrf, fit_model_module, qrf_module, dependencies=…)`,
and `source_hash` digests each object's **defining module's bytes** plus
`f"{distribution}=={version}"` for each declared dependency (read this session
at `graph/kernel.py:387-427`). `qrf_module` is `microcosm.fit.qrf`. So an
additive edit to `qrf.py` moves the implementation hash, and `node_key` folds
`kernel_impl_hash` in — for a `PLATFORM_BITWISE` kernel alongside the platform
fingerprint. All four pinned identities move.

```
implementation_hash  02db8f5c849d876be20a95152b5302a5cacc0a7c77c58d8b436a3a00f57b4c92
                  →  d1f8b1929e6452aa507b0d9c64ba42a59851dc31c71021303c25f060e34c075b

node keys            OLD             NEW
arm64/darwin/py3.14  8878352d…  →  35af6a452bd32ca39d313d78255236df01877d684c2d847b1bd5a6016a68e237  (produced)
x86_64/linux/py3.13  f6984280…  →  6c43edcb3edf2bdef20d915b1be16503d37583c678aced78f1442aca1139e1ae  (derived)
x86_64/linux/py3.14  9e80ee3a…  →  ecb6b20c02b0bc442c4baf3fd7def6d6aec06e6b9567910294f945d66c43bbf8  (derived)
```

### 3.2 The `direct.csv` byte-identity proof

`direct.csv` did **not** change, on any of the three platforms. Three
independent ways:

1. **Content digests, before and after.** All three files, unchanged:

   ```
   7b8dbd56c91ee71552ff6d892a42c56494b1813fb5d4b11553a8a8ccc9b90dca  fit.qrf/direct.csv
   7b8dbd56c91ee71552ff6d892a42c56494b1813fb5d4b11553a8a8ccc9b90dca  fit.qrf/platforms/x86_64-linux-py3_13/direct.csv
   7b8dbd56c91ee71552ff6d892a42c56494b1813fb5d4b11553a8a8ccc9b90dca  fit.qrf/platforms/x86_64-linux-py3_14/direct.csv
   ```
   (These three were already byte-equal to each other on `main`; this fixture's
   twelve recipient draws happen to agree across the pinned platforms.)

2. **Git.** `git diff --numstat` over the fixture tree shows `1 1
   …/fit.qrf/pins.json` and **no other fixture file**. Re-verified this
   session: after running the re-pin on all three cases, `git status
   --porcelain packages/microcosm-graph/tests/fixtures` is empty.

3. **The tool refuses otherwise.** `repin()` recomputes the direct call from the
   builder and compares it to the stored bytes *before* writing anything; a
   byte difference raises `SystemExit`. The re-pin succeeded, so that comparison
   passed — and `test_a_moved_direct_call_refuses_rather_than_re_pinning` now
   pins that the refusal actually fires.

   `predict()` also still draws the same values under test:
   `test_stateless_replays_legacy_uniforms_across_all_regimes` replays
   `predict()` against an independently-advanced generator and asserts exact
   frame equality across all seven regimes.

### 3.3 What licenses the two derived foreign keys — corrected

A node key is a pure function of the declaration, the resolved input
identities, the implementation hash, the capability projection, and — for a
platform-bitwise kernel — the platform fingerprint **string**.

The first pass claimed the reproduction of the three old pins established that
"the platform reaches a key as that string **and nothing else**". **That
inference was wrong, and the review caught it.** A platform also reaches its
key through the implementation hash, which folds in that machine's installed
dependency versions; the reproduction experiment substituted the recorded
implementation hash, which is precisely what removes that channel. It could not
be evidence about it.

What the reproduction does establish is narrower and still useful: each pinned
key is the one its platform computed *under the recorded hash*. The dependency
channel is now checked separately and explicitly — `repin` refuses unless
`pins["dependencies"]` (the versions the existing keys were taken under, already
recorded in the fixture and previously never read) equals this machine's
installed versions. Together the two checks say the pinned platforms and this
one shared one locked environment at pin time, which is the condition that makes
deriving the new foreign keys sound.

`repin()` also re-derives the local key alongside executing the graph and
refuses if the two disagree, so the derivation cannot drift from what the
executor computes.

**What is derived and what is not.** Keys are derived. **Bytes are never
derived**: each platform's `direct.csv` is left exactly as that platform
recorded it. I did not, and cannot, re-measure x86_64 floats from this machine.

### 3.4 The re-pin is idempotent

Re-verified this session, after the second pass changed the tool:

```
tools/graph_parity_repin.py fit.qrf    exit 0, pins.json unchanged
tools/graph_parity_repin.py calibrate  exit 0, pins.json unchanged
tools/graph_parity_repin.py simulate   exit 0, pins.json unchanged
git status --porcelain <fixtures>      empty
```

### 3.5 Standing guards

- `test_graph_parity_pins.py` derives **every** pinned platform's key on
  whatever platform is running and compares it to the pin. Before this, a stale
  foreign pin survived until that Linux lane happened to run, and H1's
  off-platform branch asserts no bytes at all — so a stale x86 pin was invisible
  on a Mac.
- `test_graph_parity_repin.py` (second pass) pins each of the tool's refusals,
  including the two the review found untested. Mutation-tested: deleting the
  dependency check makes exactly the two behavioural tests red.

### 3.6 `origin/main` moved during the lane — checked, no effect

`origin/main` advanced from `3094bfe84` to `e6d362b7e` (PR #909, object-dtype
storage hashing) while this lane ran; worktrees share refs, so another session
fetched it. It touches `graph/store.py` and `graph/population.py`. Neither is in
`QRFKernel.implementation_hash()`'s source set, and neither is an attested
seed-kernel module. Measured rather than assumed:

- `git merge-tree --write-tree HEAD origin/main` → exit 0, no conflict; zero
  files touched by both sides.
- With main's post-#909 `store.py` and `population.py` swapped into the tree,
  `test_acceptance_h_parity.py` + `test_graph_parity_pins.py` +
  `test_fit_qrf_tolerance_source_hash_pin_is_current` → **exit 0**. Tree
  restored and verified byte-identical afterwards.

---

## 4. Why `tools/graph_parity_fixtures.py` was not edited (a trap worth recording)

The first pass's initial attempt added `repin` to that module. It works — and it
silently moved **every parity node key in all three cases**:

```
calibrate node key  184ccd0a25e1d00cc4393b2880ff13a601ac04207bc37b0160b7ae60596cb3d7
                 →  f8c9ed4b33c75f3886471e48beb7c83685ead036e8335735ee9b2334cbbd2b1d
with calibrate.adam@1's implementation hash unchanged.
```

Because `ParityCsvSource` and `ParityRulesEngine` are **defined in** that
module, its bytes are inside `ParityCsvSource.implementation_hash()` and
`SimulateRulesKernel.implementation_hash()`. Every parity graph's source node
re-keys when the file changes, and every downstream node with it.

So the re-pin lives in a sibling module, `tools/graph_parity_repin.py`, which
defines no kernel and imports the generator's `_pins` / `_write_pins` / case
builders so a re-pinned fixture stays indistinguishable from a generated one.

The second reason not to use `generate()`: it rewrites `pins["platforms"]` as
the local platform alone. Running it here would have dropped both
`x86_64/linux` entries, orphaned their `direct.csv` files, and put CI's Linux
lanes onto H1's off-platform branch (no byte assertion) without any test going
red.

---

## 5. Every command, with its direct exit code

> Historicized 2026-09-12: the two red rows below (`spec_engine_coverage
> --check`, the two coverage test files) were cleared by the re-pin recorded
> under §7.1; the lock line predates the merge with amendment 19, after which
> `docs/graph-interface.lock` was re-recorded over the merged files.


Environment prepared with `uv sync --all-packages --locked --extra us --extra uk`;
everything after used `uv run --no-sync`.

The review agents mutation-tested by editing source files and reverting them,
so a mid-review run could in principle have measured a mutant. The headline
suite figure was therefore re-measured after every agent had finished, against
a tree verified to carry no `MUTANT` marker and no `.py` differing from `HEAD`
(`packages` tree `112501f5c275182b6bd4ea10769e44bce514d4b9`); it agrees with
the earlier separate runs.

| Command | Exit | Result |
|---|---|---|
| `pytest packages/microcosm-graph/tests packages/microcosm-fit/tests` | **0** | **487 passed**, 1 warning, 117s — the authoritative run, on the committed tree with nothing else touching it (the warning is a pre-existing pydantic deprecation from `policyengine_uk`) |
| `pytest packages/microcosm-graph/tests` | **0** | **371 passed**, 143s |
| `pytest packages/microcosm-fit/tests` | **0** | **116 passed**, 389s |
| `pytest test_acceptance_h_parity.py test_graph_serialize.py::test_generated_parity_graphs_bind_real_kernels_and_direct_bytes test_graph_executor.py::test_fit_qrf_tolerance_source_hash_pin_is_current` | **0** | 6 passed — the three named H1 checks, on `arm64/darwin/py3.14` |
| `pytest test_graph_parity_repin.py` (new) | **0** | 8 passed |
| `pytest test_graph_parity_repin.py test_graph_parity_pins.py` | **0** | 19 passed |
| `python tools/ci_test_groups.py --verify` | **0** | `verification=ok`. The three new test files land in fast `rest` and engine `us-am`; **none** under `[defaulted]` (that section is 53 pre-existing `microcosm-build` files, and `ci_test_groups.py` is byte-identical to main's) |
| `ruff check .` | **0** | All checks passed |
| `ruff format --check` on every file this branch touches | **0** | all formatted |
| `shasum -a 256 -c docs/graph-interface.lock` | **0** | `decl.py: OK`, `kernel.py: OK` |
| `python tools/graph_parity_repin.py fit.qrf` / `calibrate` / `simulate` | **0** | idempotent; fixtures unchanged |
| `git merge-tree --write-tree HEAD origin/main` | **0** | clean merge with current main |
| **`python tools/spec_engine_coverage.py --check`** | **1** | **drift — §7.1** |
| **`pytest test_spec_engine_inventory_coverage.py test_spec_engine_coverage_tool.py`** | **1** | **1 failed + 6 errors = 7 red — §7.1** |

Note on `ruff format --check .` repo-wide: it reports 81 files would be
reformatted, but that is **pre-existing on `main`**, and CI runs only
`ruff check .`. Every file this branch touches is format-clean.

---

## 6. The second pass: what an adversarial review of the branch found

Six reviewers read the working tree along separate axes (charter claims, test
rigor, the re-pin tool, brief compliance, implementation correctness, repo
integration); every finding was then put to two adversarial verifiers whose
default was to refute. 42 agents, no errors. Of eighteen findings, **3 survived
both verifiers, 7 split, 8 were refuted by both**. Adjudicating the splits
myself: four defects were fixed in code (each mutation-tested), two survivors
plus one split are the decisions in §7, and the rest were prose or duplicates
of the same root cause.

1. **The re-pin tool's central claim was false.** Its docstring said the
   reproduction loop "proves the assumption rather than asserting it" and that
   "only one locked environment can do that". Neither was true: the loop
   substitutes `pins["implementation_hash"]`, which is the only input carrying
   dependency versions, so *any* environment reproduces the pins. Probed
   directly — with the local kernel hash made uncallable, and with a faked
   `scikit-learn` version, the loop still passed. Fixed by making the code do
   what the prose claimed (read the recorded dependency versions, which
   `pins.json` already carried and nothing read) and by rewriting the prose to
   state what each check establishes.
2. **The final-bin closure had no test.** Amendment 20 claims two deliberate
   differences from the generator path; only the strict comparison was pinned.
   Deleting `cumulative[:, -1] = 1.0` left all fourteen stateless tests green.
   The guard is load-bearing — on a `predict_proba` row summing to just under
   1.0, a uniform above the sum makes every comparison false and `argmax`
   returns 0, silently selecting the *first*, most negative, class. Now pinned
   by a test that is the only one to fail without the closure.
3. **A contract test could not fail for the defect it named.** Its toy body
   never referenced `context.rng`, so "a keyed kernel does not spend it" held by
   construction. It now asserts the bit generator state across the call; adding
   `context.rng.random(1)` makes it fail. Its docstring also claimed the
   behaviour was "exercised end to end at the kernel protocol level" — it is
   not, because the executor has no `seed_source` branch — and now says so.
4. **A pins test asserted about its own helper.** It read through `_platforms`,
   which back-fills the top-level pin, so the assertion that the authoring
   platform appears in the mapping was satisfied by the helper rather than by
   `pins.json`.

Findings raised and **refuted** on inspection, recorded so they are not
re-litigated: the `_RELEASED` guard in `_draw_target_from_uniforms` is untested
but mirrors an existing tested one; the amendment's line-wrapping (fixed
anyway); and the "no test covers the re-pin tool" framing (now moot).

What the review checked and found **clean**: no forbidden artifacts code;
`kernel.py` exactly two hunks with the `KEYED` line character-identical to the
integration branch's; `randomness.py` and `qrf.py` byte/blob-identical to their
sources; the executor conclusion sound on both its grounds; `graph-interface.lock`
digests reproduce and `shasum -c` passes; `__all__` placement correct (that list
is not globally sorted on main, and this does not make it worse); test placement
and CI grouping correct; changelog fragment name and type correct; `uv.lock`
untouched.

---

## 7. Needs your decision

### 7.1 The spec-engine seed digests are stale — and this is merge-blocking

> Historicized 2026-09-12: applied in PR #912 — `seed_protocol` and
> `seed_map` re-pinned in `inventory_coverage.py`, the evidence file
> regenerated, and the pool-tool test's `spec_sha256` pin moved with it;
> `spec_engine_coverage.py --check` exits 0 on the pushed head.


| | `origin/main` | this branch |
|---|---|---|
| `tools/spec_engine_coverage.py --check` | exit **0** (42156/42156 fields, 41/41 checks) | exit **1** |
| `test_spec_engine_inventory_coverage.py` | pass | **1 failed** |
| `test_spec_engine_coverage_tool.py` | pass | **6 errors** (fixture setup) |

**Cause, bisected** (first pass, re-confirmed this session by computing the
digests directly): `packages/microcosm-fit/src/microcosm/fit/qrf.py` **alone**.
`microcosm.fit.qrf` is listed in `_QRF_KERNEL_MODULES`
(`spec_engine/seeds.py:353-362`); `source_inventory_sha256` hashes each module's
logical name and **exact installed source bytes**; that is the
`regime_gated_qrf` attestation's `source_sha256`, which is inside
`SeedProtocol.implementation_sha256`, which the inventory compares against
`EXPECTED_HASHES`. `SeedSource.KEYED`, `randomness.py` and the export move none
of it. This is exactly the situation `CLAUDE.md` describes: *"Spec identities …
attest kernel source and locked RNG-library versions, so they legitimately move
when main changes an attested module."*

```
seed_protocol   pinned fd3e4b06f11be4e8c13ea19fef9469ab95cbbe3e2dcfce351e860dd3e00709e4
                   now d8f595a05dd71d7eccce5976d640baf97541e3ca159fd6179e7b1ab1a1286d8e
seed_map        pinned 87ba50531d9fa6683096ecb39a31655b331ab8acff3a5876c2f62b33562a0885
                   now 373786d5ba5e115c3317c42662b3998cdef8061441cf0ba48c36828bd7d55e69
```

**Why it blocks the merge.** Both test files run in the `engine-shared` CI job,
which — read in `.github/workflows/test.yml:202` — carries **no `if:`
condition**, so it runs on every PR, and line 367 (`require_success
engine-shared`) makes the aggregate gate depend on it. The branch's changes also
classify as `shared` (`packages/*/src/**`, `tools/**`), so `engine-us` runs too.
`gh pr checks` cannot go green as the branch stands.

**The brief said report, do not re-pin, so the branch carries the drift.** The
fix is two pin values plus a regenerated evidence file (8 lines). I applied it,
measured it, and reverted it this session, so this is tested rather than
proposed:

```diff
--- a/packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py
@@ -359,8 +359,8 @@ EXPECTED_HASHES = {
-    "seed_map": "87ba50531d9fa6683096ecb39a31655b331ab8acff3a5876c2f62b33562a0885",
-    "seed_protocol": "fd3e4b06f11be4e8c13ea19fef9469ab95cbbe3e2dcfce351e860dd3e00709e4",
+    "seed_map": "373786d5ba5e115c3317c42662b3998cdef8061441cf0ba48c36828bd7d55e69",
+    "seed_protocol": "d8f595a05dd71d7eccce5976d640baf97541e3ca159fd6179e7b1ab1a1286d8e",
```

then `uv run python tools/spec_engine_coverage.py` (no `--check`) to regenerate
`docs/evidence/spec-engine/us-f0-coverage.json`. Verified afterwards — **all
seven tests, both files**, not just the one the first pass measured:

```
tools/spec_engine_coverage.py --check                                  exit 0
pytest test_spec_engine_inventory_coverage.py test_spec_engine_coverage_tool.py
                                                                       exit 0  (22 passed)
```

Tree restored and digest-verified against its pre-state both times.

**One fact that should weigh on the decision:** the integration branch that
this lane is landing code from re-pinned *these exact two digests itself*, in
`1734b9e90` ("Re-pin the seed protocol and compiled seed map digests for the
launch integration") — verified as an ancestor of that branch this session. Its
replacement values differ from the ones this branch needs (`c15bff65…` /
`a775cccd…`), because they also fold in that branch's own `acs_transfer` and
housing changes, so they cannot be copied. But it establishes that the upstream
lane treated this re-pin as the required companion to the `qrf.py` change rather
than as an open question. CLAUDE.md's own guidance ("merge main and re-pin")
points the same way. My recommendation is to apply it — on the merge ref, so the
digests attest the tree that actually merges.

### 7.2 Amendment numbering — and it renders as 19 regardless

> Historicized 2026-09-12: amendment 19 (typed artifacts) merged first as
> PR #911, so this entry is 20 by main's own arithmetic and renders as 20.


`main`'s list ends at **18**, so by its own arithmetic the next free number is
**19**. The entry is numbered **20** because the brief assigns 20 and gives 19 to
the artifacts lane. Two unmerged commits on the
`candidate-quality-producer-integration-20260905` family already claim both:
`3ff92b0ae` = 19 ("Typed artifacts and stable draw coordinates", which *bundles*
this lane's subject), `d2043d85e` = 20. Neither is on `main` or on the
integration branch, and the integration branch's `docs/graph-acceptance.md`
diff against main is empty — it carries the code with no amendment at all, which
is what this lane was sent to fix.

**New this pass:** leaving the gap has a concrete cost. The amendments are one
contiguous CommonMark ordered list, and renderers renumber from the first item —
so the entry **displays as "19."** on GitHub however it is written, colliding
with the number the artifacts lane expects and contradicting the item's own
text, the changelog fragment, the test docstrings and the journal, all of which
say 20. Nothing executable parses the list (the burndown and `explain.py` read
only `| A1 |`-style table rows), so nothing breaks — but the document already
*reads* as 19. Renumbering to 19, or landing a 19 placeholder, are both
one-token edits; which is right depends on what the artifacts lane lands as.

### 7.3 `keyed_uniform` draws differently for `-0.0` and `0.0`

> Historicized 2026-09-12: fixed in PR #912 — `_coordinate` normalises a
> float zero to `0.0` (as `keys._canonical_tolerance_float` already did for
> tolerances), with a contract test and a sentence in the amendment entry.
> No version bump: the algorithm string had not shipped on `main`.


Verified directly:

```
canonical_json([["float", 0.0]])  -> b'[["float",0.0]]'
canonical_json([["float", -0.0]]) -> b'[["float",-0.0]]'
draw((0.0,))  = 0.3464688222542801
draw((-0.0,)) = 0.3251263802628821
0.0 == -0.0: True | hash equal: True | draws equal: False
```

`_coordinate` normalises numpy scalars to Python scalars precisely so "a
coordinate read out of a column must not draw differently", but it does not
normalise float signed zero, and `canonical_json` passes floats straight to
`json.dumps`. A float coordinate arriving as `-0.0` on one run and `0.0` on
another — trivially produced by negation or a CSV literal — silently
re-randomises that row. (The int-vs-float distinction, by contrast, is
deliberate and documented: the tags are meant to differ.) I checked for a
documented signed-zero convention in the graph shard and found none.

**Not fixed here, on purpose.** `randomness.py` is byte-identical to the
integration branch that owns it, and `"sha256-u53-v1"` is a *versioned*
normative algorithm: changing which uniform a coordinate draws under the same
version string is a version bump, not a drive-by fix, and would collide with
that branch on exactly the normative draw algorithm. The one-line change is
`value = value + 0.0` in `_coordinate`'s float branch. For the owning lane to
decide.

### 7.4 The tracked root `out.md`

The brief's `-o` path is `out.md`, which is also a **tracked** file on `main`
carrying a different lane's stale report ("F1 portable worker identity — Sol
gate round 1", last written by `71dbe2979`, 2026-09-04). Writing this report
there and committing it would silently delete that record from `main`; leaving
it uncommitted means it dies with the worktree.

So: this report is written to `out.md` as the brief requires and left
**uncommitted**, and the identical content is **committed** at
`experiments/amendment-20-keyed-draws-receipts.md` — the home `CLAUDE.md`
sanctions for lane receipts. Someone should still decide whether a tracked
`out.md` belongs in the repo at all; it is a collision waiting to happen for
every lane that gets handed the same `-o` path.

---

## 8. Things checked that turned out clean

> Historicized 2026-09-12: the interface lock is now enforced in the suite
> by `packages/microcosm-graph/tests/test_graph_interface_lock.py` (PR #910).


- **No other stale pin.** Repo-wide grep for the old identities `02db8f5c`,
  `8878352d`, `f6984280`, `9e80ee3a` across `*.py`, `*.json`, `*.md`, `*.yaml`
  and `*.yml` finds no hit in any source file, test or fixture — only in this
  report and the lane journal, which quote them deliberately.
- **No existing node key moves.** Adding an enum *member* is not adding a
  normative *field*: `_capabilities_projection` serialises
  `capabilities.seed_source.value`, so every kernel that declares `EXECUTOR`,
  `PARAM` or `NONE` projects exactly what it projected before. The `fit.qrf`
  keys move because its implementation hash moved, not because `KEYED` exists —
  the charter entry now says this explicitly rather than leaving the reader to
  reconcile "no existing node key moves" with a re-pinned fixture in the same
  diff.
- **C4's static check still holds.** It ASTs every file under `microcosm/graph/`;
  `randomness.py` contains no `RandomState`, no `*.random.seed`, and no
  `default_rng` call at all. `test_acceptance_c_seeds.py` is green.
- **`docs/graph-interface.lock` is plain `shasum -a 256` of the file bytes** —
  `shasum -a 256 -c` reports both OK. Worth knowing: **nothing in the repo
  verifies this lock.** No test, tool, or workflow step reads it; the only
  references are prose. That is pre-existing, not a branch defect, but a lock
  nothing checks is a lock that will drift.
- **CI grouping needs no bookkeeping.** All three new test files land in fast
  `rest` and engine `us-am` automatically; `--verify` is green and none appears
  under `[defaulted]`.
- **The `KernelContext` field order contract is untouched** —
  `test_context_numerics_default_empty_and_carry_scopes` asserts the last two
  dataclass fields are `["tolerances", "numerics"]`, which is why bringing
  `artifacts` over would have broken it. It is green.
