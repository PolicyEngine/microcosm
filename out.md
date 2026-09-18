# Amendment 19 — typed opaque artifacts on the graph interface

Lane report. Branch `amend-typed-artifacts`, off `origin/main` at `3094bfe84`,
in the worktree `~/PolicyEngine/_worktrees/microcosm-amend-artifacts`.
Date 2026-09-11. **Nothing pushed, no PR opened, no branch created, `uv.lock`
untouched, the stash never used.**

> Historicized 2026-09-12: written before the push; the branch is PR #911 (pushed 2026-09-11, peer-gated). The lock-enforcement test named in §8 merged as #910 on 2026-09-11.


The interface change the integration branch
`origin/microcosm-us-launch-integration-20260909` had made to the two frozen
files without an amendment is now extracted and landed the charter's way:
declaration types and kernel context field, the executor support that makes
them mean something, amendment entry 19, a re-recorded lock, contract tests,
the frozen acceptance suite's B2 field set in its own commit, a changelog
fragment, and sorted exports.

---

## 1. Commits

Twenty-six commits, `3094bfe84..HEAD`, oldest first. The red commits are
marked; each is followed by the commit that makes it green.

| SHA | Subject |
| --- | --- |
| `d88e86d5b` | Open the Amendment 19 lane journal (typed opaque artifacts) |
| `68a6ecc4b` | **(red)** Amendment 19 (red): declaration contracts for typed opaque artifacts |
| `e591c52d7` | Amendment 19: typed opaque artifacts on the frozen declaration interface |
| `a2b6dfb0b` | **(frozen acceptance suite, alone)** B2: the kernel context carries declared artifacts |
| `1cce8eceb` | **(red)** Amendment 19 (red): executor contracts for typed artifact edges |
| `15f9d2c67` | Amendment 19: the executor honours declared artifact edges |
| `38b9a9e4d` | Amendment 19 in the charter; relock decl.py and kernel.py |
| `8c2e7faab` | The graph explorer shows a node's typed artifact provenance |
| `41fb8bc94` | Record the Amendment 19 lane's landed commits and gate results |
| `51362a506` | A gate kernel may not declare a typed artifact output |
| `749640255` | A gate reached only through a byte edge still derives the release tier |
| `433996d6f` | Widen the amendment 19 node-key measurement to both country graphs |
| `b4ac66c1f` | The manifest authenticates typed artifact edges on load |
| `5a13c1abf` | Artifact edges cross population versions under every resume policy |
| `3a0726f93` | A corrupt typed manifest loads as StoreCorruptError, not NodeRejectedError |
| `9ebb60e4b` | Say what ArtifactValue.key actually is; relock kernel.py |
| `402d9a631` | Keep gate_ancestry validation where it was; name the graphs actually measured |
| `5c4a8efde` | Keep the artifact miss decision inside the recompute fallback |
| `64ae64867` | Assert every field of a declared artifact edge is normative |
| `6037ce458` | Descendant-exact invalidation reaches through a byte edge (A3) |
| `9018c4420` | Typed bytes keep the derivation, not the identity; read payloads only to run |
| `79ef58e9b` | Record the adversarial review and what it changed |
| `df04a9375` | Re-verify the Amendment 19 lane end to end, independently of the landing |
| `315119e65` | Write the Amendment 19 lane report |
| `e3f69a4c4` | Exercise every refusal a foreign typed descriptor can trip |
| `0c22b9d30` | Correct the red-commit count: 1cce8eceb was 8 of 8, not 7 |

---

## 2. Diff summary, per file

`git diff --stat origin/main...HEAD` — 22 files, +2784 / −121. (The 121
deletions are almost entirely this file: `out.md` still held an unrelated
earlier lane's report and was replaced.)

### The two frozen files

**`packages/microcosm-graph/src/microcosm/graph/decl.py`** (+137 / −1)

- `ArtifactType` — a nominal `name` and a positive `schema_version`. The graph
  never parses a payload; it carries the pair and refuses a mismatch.
- `ArtifactOutput` — `name` (the key the kernel already returns bytes under)
  and `type`. Declaring it is what makes the output addressable and *required*.
- `ArtifactInput` — `name` (consumer-local alias), `producer`, `artifact`
  (the producer's output name), `type` (exact).
- `Node.artifact_inputs` / `Node.artifact_outputs`, both defaulting to `()`,
  validated in `__post_init__` for tuple-ness, element type, and unique names.
- `Node.normative()` elides both fields when empty — the mechanism behind the
  "no node key moves" answer in §4.
- `compile_graph` gains the artifact-edge arm: refuses self-dependence, an
  unknown producer, an undeclared output, and a type the producer does not
  declare; on success adds the producer to `predecessors[node.id]`, so an
  artifact cycle is refused by the same depth computation as a cell cycle.

**`packages/microcosm-graph/src/microcosm/graph/kernel.py`** (+72 / −1)

- `ArtifactValue` — frozen `payload: bytes`, `type: ArtifactType`, `key: str`,
  `producer_key: str`, `numerics: NumericScope`, with `__post_init__`
  validating the payload is `bytes`, the type and scope are the right classes,
  and both identities are 64-char lowercase hex.
- `KernelContext.artifacts: Mapping[str, ArtifactValue]`, defaulting to `{}`,
  placed **before** `tolerances`, with a new `__post_init__` that rejects an
  empty or non-string alias and a non-`ArtifactValue` value, then re-binds the
  field to a `MappingProxyType` over a defensive copy.

### Executor-side support (the minimal coherent subset)

**`artifact_edges.py`** (new, 210 lines) — the glue between declaration and
kernel. `numeric_scope`, `scope_payload`, `scope_from_payload`,
`require_compatible_scope` (the anti-laundering rule),
`descriptor` (the portable per-edge provenance a receipt records),
`typed_contracts` (every typed edge of one node, resolved against the compiled
graph; empty for a node declaring none), and `value_from_descriptor` (rebuilds
a verified `ArtifactValue`, re-deriving and checking the identity).

**`keys.py`** (+41) — `opaque_artifact_key(node_key, name)` promoted from the
executor's private `_opaque_artifact_key` into the public key module under the
identical domain and formula (`sha256_domain("node-artifact", …)`), so typed
and undeclared bytes share one derivation; and a `typed_artifacts` term added
to `node_key` **only** when the node declares an input.

**`executor.py`** (+121 / −3) — `_opaque_artifact_key` now delegates to
`graph_keys.opaque_artifact_key`; `_context_digest` folds each artifact's
alias, identities, type, scope and payload in (so B4's mutation check covers
artifacts); `_project_context` takes and passes `artifacts`; `_validate_result`
rejects a kernel that omits a declared output (`StoreCorrupt` on a restored
result, `NodeRejected` on a fresh one); `_write_node` records
`typed_artifacts` and moves the cache record to schema 2 **only** when the node
declares artifacts; `_require_record_shape` requires the contract to match, and
treats a missing declared output as a *miss* and a wrong identity as
*corruption*; `run_graph` refuses a gate kernel that declares an artifact
output, authenticates each edge against the producer receipt before deciding
hit-or-miss, and loads payloads only on the recompute path.

**`manifest.py`** (+140 / −2) — `NodeReceipt.typed_artifacts` (kw-only,
validated, frozen, elided from the payload when empty);
`_TYPED_SCHEMA_VERSION = 3`, emitted only when some receipt carries a typed
edge and refused on load if a v3 manifest carries none; `typed_artifacts` in a
receipt payload requires schema 3; and `_validate_typed_ancestry`, which on
every construction checks each output descriptor names its own node and key,
each input resolves to a producer whose output descriptor is byte-identical,
the scope matches the producer's declared capabilities and is one the consumer
may read, the edges form a DAG, and a release's `gate_ancestry` covers gates
reached through bytes.

**`serialize.py`** (+74) — both declarations serialize and round-trip, elided
when empty so a pre-amendment graph's JSON is byte-identical.

**`view.py`** (+17), **`explain.py`** (+5) — `describe` shows declared edges,
and the explorer's receipt payload carries `typed_artifacts` when present.

**`__init__.py`** (+8) — `ArtifactInput`, `ArtifactOutput`, `ArtifactType`
(from `decl`) and `ArtifactValue` (from `kernel`) added to the imports and to
`__all__`, in sorted position, per `ed36f6cb3`.

### Charter, lock, changelog, tests

- `docs/graph-acceptance.md` (+85) — amendment entry 19.
- `docs/graph-interface.lock` (±2) — both hashes re-recorded.
- `changelog.d/amend-typed-artifacts.added.md` (new, 3 lines).
- `packages/microcosm-graph/tests/` (+1346 / −3) — 36 new tests across seven
  files, of which exactly one line is in the frozen acceptance suite.

### What was deliberately left on the integration branch

The branch's `packages/microcosm-graph/src` diff against main is +2165 across
17 files; this extraction is +818 across 10. Excluded entirely, each its own
lane: `SeedSource.KEYED` and `randomness.py`; `availability.py` and the
execution-state / `unreached` / `blocked_by` / `gate_exception` machinery with
its manifest schema 4; `attachments.py`, `_PopulationRetention` and lazy
populations; `store.py`'s Frame-metadata `microcosm-graph-frame-v2` and
non-finite JSON decode hooks; `keys.py`'s `_stream_file` chunked source
hashing; `codecs.py`'s `SourceBytesCodec`; `schema.py`; and the `_write_node`
per-coordinate memory refactor.

`store.py` needed no change at all: `load_bytes`, `load_json` and the
`put_bytes` write of opaque artifacts under `opaque_artifact_key` all already
exist on `origin/main` (`store.py:952`, `store.py:983`,
`executor.py:1456` on main). The typed layer rides on that machinery, which is
why the amendment introduces no second identity scheme.

---

## 3. Field placement

`artifacts` was placed **before** `tolerances`, as the brief directed, not last
as the integration branch had it. There was no documented reason on the branch
for the trailing position, and the leading position keeps amendment 17's
contract assertion literally true. Both assertions now hold and are enforced:

- `test_graph_kernel_contract.py:204` (amendment 17, untouched):
  `assert fields[-2:] == ["tolerances", "numerics"]`
- `test_graph_kernel_contract.py:338` (amendment 19, new): the same assertion
  again, plus `assert fields[fields.index("artifacts") + 1] == "tolerances"`.

The amendment entry states the choice: *"It rides before `tolerances`, so
amendment 17's statement that `numerics` rides at the end of the context stays
literally true."*

---

## 4. The node-key answer, measured

**No node key moves for any graph that declares no artifact edge — and there
are none on `main`. Where an edge is declared, keys move for its two ends and,
by A3, for every descendant of either end (the consumer's cell readers and any
structural node whose base version contains the producer or consumer).**

`Node.artifact_inputs` and `Node.artifact_outputs` are normative fields with
defaults, which for amendments 11 and 13's sibling field `entrants` meant every
node's projection and key moved. This amendment avoids that by eliding both
fields from `Node.normative()` when empty, and by adding the `typed_artifacts`
term to `node_key` only when the node declares an input.

### Evidence — measured twice, independently

The landing session measured it; this session re-measured it from scratch with
a script that resolves `microcosm.graph` **twice from one process image** —
once from this branch's sources and once from `origin/main`'s, shadowed through
`PYTHONPATH` — while `microcosm.build` stays identical in both runs, so the
only variable is the graph-kernel code.

```
$ git archive origin/main packages/microcosm-graph/src | tar -x -C $SCRATCH/mainsrc
$ uv run --no-sync python measure_keys.py                       # this branch
$ PYTHONPATH=$SCRATCH/mainsrc/packages/microcosm-graph/src \
    uv run --no-sync python measure_keys.py                     # origin/main
```

Both exited 0. The runs loaded different `decl.py` bytes, confirming the shadow
took effect: `ed0a859a…` (this branch's locked hash) against
`635fef92…` (`origin/main`'s). The measurement was repeated at the final tree
after the audit fixes, with the same result.

| Graph | Nodes | Keys identical | Projections identical |
| --- | ---: | --- | --- |
| `_toy.small_graph()` | 5 | yes | yes |
| `_toy.chained_graph()` | 5 | yes | yes |
| `_toy.chained_graph(leaves=("leaf_a",))` | 6 | yes | yes |
| `_toy.full_graph()` | 9 | yes | yes |
| `uk_spine_graph(load_country_spec("uk"))` | 41 | yes | yes |
| `us_post_transfer_graph()` | 8 | yes | yes |
| **total** | **74** | **74 / 74** | **74 / 74** |

```
TOTAL node keys compared: 74
node keys that moved:     0
projections that moved:   0
```

The amendment entry's per-graph counts (5, 5, 6, 9, 41, 8; "74 node keys in
all") are each correct.

### The converse, also measured

Declaring an edge *does* move both ends, and the amendment says so rather than
claiming a free upgrade. Enforced by
`test_graph_keys.py::test_a_declared_artifact_edge_enters_both_ends_of_the_key`,
`::test_every_part_of_an_artifact_declaration_is_normative` (a type name, a
type version and the producer's output name move both ends; the consumer-local
alias moves only the consumer, because the producer never sees it), and
`::test_a_byte_edge_carries_descendant_exact_invalidation` (A3 holds over a
byte edge: re-hashing only the producer's implementation moves the consumer,
and without the declaration it does not).

A given output therefore does **not** keep its store identity when a type is
declared for it — `opaque_artifact_key` is derived from the producing node's
key, which moved. The amendment entry and the `opaque_artifact_key` docstring
both say this explicitly.

---

## 5. Commands run, with direct exit codes

Every command was run from the worktree with `uv run --no-sync`, after
`uv sync --all-packages --locked --extra us --extra uk` (exit 0 in the landing
session; the environment was reused unchanged). **Every row below was re-run at
the final tree, after the audit fixes** — none is carried over from the landing
session, and none predates the last source change.

| Command | Exit | Result |
| --- | ---: | --- |
| `pytest packages/microcosm-graph/tests` | **0** | **373 passed** |
| `pytest packages/microcosm-graph/tests -k acceptance` | **0** | 113 passed, 257 deselected |
| `pytest .../test_graph_kernel_contract.py` | **0** | 15 passed |
| `pytest` (every graph-dependent test outside the graph package) | **0** | **81 passed** |
| `pytest` (KernelContext consumers, 5 files) | **0** | 36 passed |
| `python tools/ci_test_groups.py --verify` | **0** | `verification=ok`, 388 tracked test files |
| `python tools/spec_engine_coverage.py --check` | **0** | 42156/42156 configuration fields; 41/41 inventory checks |
| `python tools/graph_acceptance_burndown.py --verify` | **0** | `verification=ok` |
| `ruff check .` | **0** | All checks passed |
| `ruff format --check` (the 17 changed `.py` files) | **0** | 17 files already formatted |
| `shasum -a 256` on the two frozen files | **0** | both match `docs/graph-interface.lock` |
| `git diff --stat origin/main...HEAD -- uv.lock` | **0** | empty |

**The graph-dependent closure.** Rather than trusting that a change confined to
`packages/microcosm-graph/src` cannot reach elsewhere, the set was computed:
seven source modules outside the graph package import `microcosm.graph`
(`microcosm.build.{uk,us}_runtime.{graph,graph_kernels}`,
`microcosm.{calibrate,fit,frame}.kernels`), and 29 test files reach the graph
package directly or through one of those seven. Running all 29 — the whole
graph package plus the 8 outside it — is green: **373 + 81 passed, both exit
0.** That is every test in the workspace this change could affect.

**The `KernelContext` consumers.** The narrower run the brief asks for is
`packages/microcosm-{calibrate,fit,frame}/tests/test_kernels.py` plus
`packages/microcosm-build/tests/test_{us,uk}_graph.py` — 36 passed, exit 0. A
grep of `KernelContext(` across `packages/`, `tools/` and `examples/` finds 17
sites; the five outside `packages/microcosm-graph/tests` are the three
`test_kernels.py` fixtures and the executor's own two constructions, and
**every one of them passes keyword arguments** — no consumer constructs
`KernelContext` positionally, so the new field's placement could not break one.

### Two commands that did not exit 0

**`ruff format --check .` → exit 1.** 81 of 825 files would be reformatted.
**None of them is touched by this lane** — the intersection of the 81 with
`git diff --name-only origin/main...HEAD` is empty, and all 17 changed Python
files pass `ruff format --check` individually (exit 0, above). This is
pre-existing repo drift; CI's `lint` lane runs only `ruff check .`
(`.github/workflows/test.yml:181`), which is green.

**`pytest packages/microcosm-build/tests/test_release_target_parity.py` →
exit 1.** 2 failed, 35 passed:
`TestRegeneration::test_committed_artifacts_match_regeneration` and
`TestRegeneration::test_gate_passes_on_real_compiled_registry`, both raising
`LedgerHierarchyMetadataError: … dimension 'bea_nipa.series_code' requires
exactly one non-empty label, got []`. **Not this lane, and not a CI failure.**
Four pieces of evidence:

1. Both tests are guarded by `TestRegeneration._feed_or_skip`
   (`test_release_target_parity.py:484`), which skips when the pinned feed is
   absent. That feed is `~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`
   — 131 MB, **outside the repository**, untracked, mtime **2026-07-23**. CI
   does not have it, so CI skips both tests.
2. The July artifact predates #855's hierarchy-label requirement, which is
   exactly the diagnosis the #791 lane already recorded for the UK twin
   (`test_uk_target_references.py`) in
   `experiments/791-household-composition-receipts.md:111`.
3. The same two tests fail identically with `origin/main`'s graph sources
   shadowed in (exit 1, same two names), and
   `microcosm/build/ledger_targets.py` imports no `microcosm.graph` at all.
4. `origin/main`'s 16 commits since this branch point (`3094bfe84..e6d362b7e`)
   are all the #907 object-dtype storage-hashing lane; none touches
   `ledger_targets.py`, `release_target_parity` or any Chronicle fact.

---

## 6. The adversarial audit of the landed work

The landing session ran a five-dimension adversarial review of the extraction
itself; seven of its nine findings were real and are among the commits in §1.
This session ran a *second*, independent audit of the finished result: five
auditors (extraction fidelity and minimality; charter mechanics; contract-test
coverage and red-first discipline; executor correctness; are the prose claims
true), each finding then put to two skeptics instructed to refute by default
and to reproduce rather than reason.

**Nine findings raised. One survived both skeptics.**

### The one that survived

**The red commit `1cce8eceb` records "7 of 8 fail"; all 8 failed.** Two
skeptics independently reconstructed the tree at that commit and ran its own
test file against its own sources; I then reproduced it a third time:
`git archive 1cce8eceb | tar -x`, `PYTHONPATH` over that tree's six shard
`src` directories (confirmed by `microcosm.graph.executor` resolving into the
extract and `microcosm.graph.artifact_edges` having no spec there, so the
executor support genuinely had not landed yet), then the commit's own
`test_graph_executor.py`: **8 failed, 63 passed**, the 8 being exactly the
amendment-19 tests it added.

Red-first discipline holds — the commit was *redder* than claimed — but the
tally in its message is wrong. `0c22b9d30` corrects the journal and says why
the commit message stays as it is: rewriting landed history to fix a tally
would be worse than recording the correction.

### What the audit changed anyway

One refuted finding still pointed at something worth fixing. A line trace of
the whole graph suite (`sys.settrace` pinned to the module) showed **eleven
non-docstring statements of `artifact_edges.py` never executed** — all of them
on the surface that parses provenance written elsewhere:
`value_from_descriptor`'s four refusals, `scope_from_payload`'s malformed-scope
and malformed-tolerance refusals plus its `Tolerance` reconstruction, and
`require_compatible_scope`'s platform-plus-tolerance refusal.

The skeptics were right that the brief did not ask for this and that no latent
bug hid behind those lines — but the amendment says a manifest "authenticates
every edge on load", and those arms are what make that true against a foreign
manifest. `e3f69a4c4` adds three tests driving all of them through the public
`NodeReceipt` / `RunManifest` surface. The same trace now reports **zero**
unexercised statements in the module.

The same finding's sharpest claim — that `run_graph`'s consumer-side
producer-receipt comparison (`executor.py:2059`) is never exercised — I
measured directly and confirmed in part: line 2059 runs on every
artifact-consuming node, but its `raise` arm never does. Both skeptics refuted
the finding correctly: the charter claims the *ordering of a check*, which does
execute. The arm is unreachable by construction — a producer that hit its
record had that identity checked by `_require_record_shape`, and one that ran
got it from `_write_node` under the same derivation — so `e3f69a4c4` marks it
with the comment convention `executor.py` already uses for such guards
(`# generated records cannot reach this branch`) rather than chasing it with a
test that cannot be written honestly. The misleading inline comment in
`test_a_cache_hit_authenticates_its_artifact_edges_without_reading_them`, which
implied the receipt comparison was the guard firing, now names the guard that
actually fires.

### The seven refuted, and why

| Finding | Why it was refuted |
| --- | --- |
| The node-key note says "six graphs" but `chained_graph()` with default args *is* `small_graph()` | True but not a defect: the text claims no distinctness, and the two are separate constructions whose keys were separately computed. A wording preference. |
| `PROGRESS.md` said "`ruff format --check` clean" | A skeptic extracted `origin/main` and got an identical set of 81 unformatted files, so the condition is pre-existing; and the journal already discloses the full precision forty lines below. |
| "Schema moves to 2 only for a node that declares artifacts" has no test | Both skeptics measured it rather than reading it: a plain node's record is written and accepted at schema 1, an artifact-declaring node's at 2. Code-true and checkable. |
| `explain.py`'s typed-artifact payload has no test | A skeptic executed the branch by hand (it emits `typed_artifacts` correctly) and found the identical untested-conditional pattern already on `origin/main` from `8549530c2`. |
| Two guards inside the required contract areas survive deletion | Reproduced by mutation — and the amendment-17 precedent the brief told this lane to mirror has exactly the same property on `origin/main`. |
| The producer-receipt refusal is never exercised | See above: the claim conflated "the raise arm is not taken" with "the check is not exercised". |
| A recomputed producer could hand a consumer an orphaned earlier payload | Real property of the store's write-once semantics (`_put` with `verify_existing=True` returns the existing object without comparing), but **byte-identical to `origin/main`** — this lane changed no `verify_existing` line — and reachable only if a kernel violates the determinism contract that makes same-node-key-same-bytes hold, in which case columns, frames and weights diverge the same way. |

The fidelity dimension returned **zero** findings: it classified all 3,193
diff lines of the integration branch's graph-src change, confirmed every
artifact-needed hunk landed and every excluded hunk is genuinely absent
(`grep` for `KEYED`, `keyed_uniform`, `population_retention`,
`_LazyPopulations`, `_stream_file`, `SourceBytesCodec` across the landed source
and tests returns nothing), and found no dangling dependency and no scope leak.

---

## 7. Charter mechanics, item by item

1. **Amendment entry 19** — `docs/graph-acceptance.md`, +85 lines, in the house
   style of entries 1–18: what the gap was (byte dependencies had no channel
   but undeclared `KernelResult.artifacts`, invisible to the compiler and
   outside every key), what the new declarations are, which acceptance items
   are touched (C3 "declared predecessors only" now covers bytes; B4's mutation
   check covers artifacts; E1's payload validation; F2's gate ancestry; and
   amendments 16 and 17's numeric classes), the one refused shape, the node-key
   answer stated as measured with its six graphs and 74 keys, and the closing
   *"Raised by the US launch integration branch
   (`microcosm-us-launch-integration-20260909`), which extended both frozen
   files without an amendment; extracted and adopted 2026-09-11."* The
   pre-existing paragraph about amendments 11 and 13 moving keys is extended
   with one sentence saying this one does not.
2. **Lock re-recorded** — `docs/graph-interface.lock` now holds
   `ed0a859a…  decl.py` and `97c3ec9f…  kernel.py`, both confirmed against
   `shasum -a 256` of the current bytes (§5).
3. **Contract tests** — `test_graph_kernel_contract.py` +208, mirroring
   amendment 17's shape in `cdbf71888`:
   `test_artifact_value_validates_its_payload_type_and_identities`,
   `test_context_artifacts_default_empty_and_are_immutable`, and
   `test_artifact_declarations_round_trip_through_canonical_json`.
4. **Frozen acceptance suite, alone** — `a2b6dfb0b` is a one-file, one-line
   commit adding `"artifacts"` to `test_acceptance_b_ownership.py`'s exact
   `KernelContext` field set, with a trailing comment naming the amendment,
   exactly mirroring `80b63ba14`'s shape for amendment 17. No other
   `test_acceptance_*` file is touched anywhere on the branch.
5. **Changelog and exports** — one fragment,
   `changelog.d/amend-typed-artifacts.added.md`, `added` being the correct
   towncrier type for new interface; four names added to `graph/__init__.py`'s
   imports and `__all__` in sorted position.
6. **Runs** — §5.

---

## 8. Interim rulings and anything needing Max

> Historicized 2026-09-12: this report was written before the branch was pushed
> as PR #911 (2026-09-11). The lock-enforcement test it names below merged as
> #910 on 2026-09-11, so "no automated enforcement" and "unmerged sibling
> branch" describe the state at writing time, not now.

**One interim ruling this lane made.** A **gate kernel may not declare a typed
artifact output**, refused outright in `run_graph` before any node executes. A
gate whose kernel raises becomes a `fail` verdict and the run continues
(amendment 7), so its synthesized result carries no artifacts; a declared
output would turn that verdict into an aborted run. Amendment 19 carries no
regime for an output a node was unable to produce, nor for the consumers
thereby unreachable. Refusing the declaration keeps amendment 7 literally true
for every legal node shape, and leaves the design space open. If Max would
rather model it — an unproduced-output regime, with `blocked_by` semantics for
the consumers — that is a follow-up amendment, and it interacts with the
`availability.py` / execution-state lane this extraction deliberately left on
the integration branch.

**The root cause of this lane, and the sibling branch that fixes it.** The
charter's "Interface freeze" section (`docs/graph-acceptance.md:153-159`) says
changing `decl.py` or `kernel.py` "requires the owner's sign-off on the pull
request and re-recording the lock" — a human gate. Nothing in the repository
compares the lock with the files: `grep` for `graph-interface` across the tree
finds only the charter sentence and the lock file itself, and CI's `lint` lane
runs only `ruff check .`. That is exactly how the integration branch extended
both frozen files with no amendment and no one noticing until this extraction.

There is already an unmerged sibling branch for it — **`graph-interface-lock-test`,
one commit `8bd6e05ec` ("Enforce the graph interface lock with a test"),
branched from the same `3094bfe84`** — adding
`packages/microcosm-graph/tests/test_graph_interface_lock.py`, which asserts the
lock names exactly the two frozen files and that each one's SHA-256 matches.
**This branch already satisfies it:** its test file run against this tree exits
0, 2 passed (run from a temporary copy, removed afterwards; the tree is clean).
Merging that sibling would make the freeze enforceable instead of advisory, and
would have caught the integration branch at PR time. It is a governance change
— every future frozen-file edit becomes a hard CI failure until re-locked,
which is the point — so it is Max's call, not this lane's, and the two branches
do not conflict.

**Two more, neither this lane's:**

- `ruff format --check .` is red on 81 pre-existing files — a skeptic confirmed
  the identical 81-file set on `origin/main`. CI does not gate on it, so it has
  drifted silently. Worth one formatting sweep, on its own PR.
- The local `_buildh-runtime` feed artifact is from 2026-07-23 and now fails
  the #855 hierarchy-label requirement, so two `test_release_target_parity.py`
  tests are red on any machine that has it and green (skipped) everywhere else.
  The UK twin was hit by the #791 lane in August. Re-pinning or refreshing that
  local artifact would stop it ambushing future lanes' full-workspace runs.

**No spec identity was re-pinned.** `tools/spec_engine_coverage.py --check`
passes unchanged at 42156/42156 and 41/41, so no drift arose to report.

---

## 9. Deliberately not done

No push, no PR, no branch created, no stash used, no `uv.lock` edit, no spec
re-pin, no artifact build, no release, no publication, and no acceptance-test
change beyond the single B2 field-set line the amendment requires.


> Historicized 2026-09-12: "no push, no PR" was true when written; the branch was pushed as PR #911 on 2026-09-11.
