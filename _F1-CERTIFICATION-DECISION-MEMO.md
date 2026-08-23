# F1 certification owner decision memo: node reuse and calibration scope

Date: 2026-08-22
Branch: `spec-engine-f1-cert`
Status: **OWNER RULING REQUIRED; STOP on both items**

## Purpose and fixed constraints

This memo asks for two independent owner decisions. It does not propose changing
the comparator to turn missing evidence into a pass. The current comparator is
correctly fail-closed, and the lane must preserve all of these constraints:

- normative artifacts remain raw-byte equal;
- the 72-site seed-ledger binding remains exact;
- no gate, band, ceiling, fold, seed, or owner-only exclusion is tuned toward a
  pass;
- no pool or release build runs in this lane; and
- the concurrently owned brokered-QRF draw path is not edited here.

The D4 ruling is controlling. Every H5 weight vector and calibration weight is
normative and admits neither canonicalization nor exclusion
(`_F1-CHARTER.md:156-164`). Only provenance, publication, and operational
receipt fields named
by the sealed comparison vector may use
`equal_after_normalizing_prefix`, `expected_to_differ_by_generation`, or
`operational_excluded` (`_F1-CHARTER.md:166-179`). The R6 continuation was
runner-and-host-handoff only and deliberately did not recreate executor wiring
(`_F1-LANE-NOTES.md:2473-2479`, `_F1-LANE-NOTES.md:2506-2514`).

## Decision 1: semantic node-reuse evidence

### Exact failing checks at this head

The production evidence emitter supplies an empty node inventory, declares it
incomplete, and emits an empty reuse-key map:

- `_emit_f1_evidence` receives pool output paths and the checkpoint store, but
  no dispatcher, completed-node journal, or semantic-reuse identity input
  (`tools/build_us_multispine_pool.py:5966-5975`).
- It calls `complete_coverage_evidence` with `node_reuse_ids=()` and
  `node_reuse_inventory_complete=False`
  (`tools/build_us_multispine_pool.py:6017-6025`).
- It publishes `node_reuse_keys={}`
  (`tools/build_us_multispine_pool.py:6042-6050`).

The comparator then fails in three independent, intentional ways:

1. Coverage requires `node_reuse_inventory_complete`, and requires the emitted
   ids to equal the compiler's complete `producer_order`
   (`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:373-397`).
2. Each within-mode pair fails unless both inventories are complete and their
   reuse-key maps are exactly equal
   (`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:1574-1613`).
3. Each constants/bundle pair immediately fails with
   `node reuse inventory is incomplete` if either inventory is incomplete
   (`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:1623-1631`).

The four-receipt verdict ANDs vector coverage, cold-build evidence, within-mode
equality, and cross-mode equality
(`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:1382-1444`),
and the CLI returns status 1 for that false verdict
(`tools/f1_certification_run.py:227-249`). This failure is therefore the exact
declared behavior, not incidental formatting or publication drift.

### Why this is not a plan-lock canonicalization bug

The D4 receipt comparison vector is compiled as rows containing an artifact
role, JSON pointer, rule, and category
(`packages/microcosm-build/src/microcosm/build/spec_engine/compiler_ir.py:2681-2708`).
Node-reuse maps are a separate typed input. The comparator binds each map to the
entire compiler producer inventory and explicitly says there is no
caller-authored escape hatch
(`packages/microcosm-build/src/microcosm/build/spec_engine/artifact_comparison.py:548-569`),
validates exact inventory closure
(`packages/microcosm-build/src/microcosm/build/spec_engine/artifact_comparison.py:1839-1862`),
and records a differing key under `rule="raw_exact"`
(`packages/microcosm-build/src/microcosm/build/spec_engine/artifact_comparison.py:634-647`).
No D4 normalization or exclusion applies.

Nor can the compiler's static `CompiledNode.node_key` honestly fill this map.
That key covers the resolved compiler slice, kernel implementation, and seed
protocol (`packages/microcosm-build/src/microcosm/build/spec_engine/compiler_ir.py:2063-2087`).
The semantic runtime API additionally requires behavior-relevant run inputs,
transitive input content hashes, an implementation-dependency digest,
broker-issued RNG and source behavior identities, materializer ABIs, and any
output-sensitive backend ABI
(`packages/microcosm-build/src/microcosm/build/spec_engine/executor.py:1523-1539`,
`packages/microcosm-build/src/microcosm/build/spec_engine/executor.py:1542-1681`).
It also enforces the compiled node's exact RNG site inventory
(`packages/microcosm-build/src/microcosm/build/spec_engine/executor.py:1577-1608`).

The missing production seam is concrete:

- the physical authority retains the compiled late-producer nodes, but its
  materializer kwargs omit them
  (`packages/microcosm-build/src/microcosm/build/us_runtime/pool_physical_authority.py:267-293`);
- the kernel authority therefore constructs `StackedLateProducerAuthority`
  only from those structural kwargs
  (`packages/microcosm-build/src/microcosm/build/us_runtime/pool_kernel_authority.py:75-114`),
  and that adapter has registry/schedule/transfer fields but no compiled nodes
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8853-8861`);
- the live late DAG accepts no physical executor
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:9990-10003`)
  and invokes legacy callbacks directly through readiness scheduling
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:10166-10229`);
  and
- although the standalone physical dispatcher calls `execute_node`, its journal
  records the static `node.node_key`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/pool_physical_executor.py:679-717`).
  That module itself classifies the journal as operational and separate from
  semantic node-reuse identity
  (`packages/microcosm-build/src/microcosm/build/us_runtime/pool_physical_executor.py:1-15`).

Thus this is not a small evidence-emitter omission. Production does not yet
produce the inputs required by the existing semantic identity API, and the
closed execution-ABI schema has no declared node-reuse input/backend authority
(`packages/microcosm-build/src/microcosm/build/spec_engine/schema/locks.schema.json:565-582`,
`packages/microcosm-build/src/microcosm/build/spec_engine/schema/locks.schema.json:701-738`).

### Options and costs

#### Option NR-A — authorize the complete D3 semantic contract and production dispatch

The owner authorizes one compiler-sealed contract for every producer's:

- behavior-relevant run-input projection;
- transitive content-hash projection;
- implementation-dependency digest;
- artifact-materializer and output-sensitive backend ABIs; and
- broker-issued primary and supplemental RNG-owner behavior, without changing
  or reconstructing seed material outside the ledger.

Implementation then routes the real bundle graph through the compiled-node
dispatcher, carries the semantic inputs and completed execution evidence to the
publisher, derives `node_reuse_identity(...)` for the exact compiler order, and
makes constants mode produce the same semantic keys without claiming bundle
provenance. The sibling QRF lane's brokered draw fix is an integration
prerequisite; this lane must consume that result rather than modify the draw
path.

Cost: **high implementation scope**. At minimum this changes the execution-ABI
compiler/schema, runtime authority projection, US physical authority/adapter,
dispatcher integration, stacked driver, evidence emitter, and their generated
identity pins and regressions. Required tests include rejection of omitted or
extra nodes, changed run/content/dependency/materializer/backend inputs,
supplemental RNG-owner drift, duplicate or skipped dispatch, and constants vs
bundle equality. Repository package suites and Ruff must be green per coherent
commit. On the owner host this adds no role beyond the existing four cold pool
roles; the semantic evidence is collected during those builds. If calibration
Option CAL-A is also selected, the combined handoff has eight heavy child
invocations, as costed below.

Recommendation: **NR-A** if the current F1 charter is to become green. It is the
only option that satisfies the existing exact node-reuse gate.

#### Option NR-B — explicitly defer node reuse and retain the red certificate

The owner may defer the compiler/runtime contract and production dispatcher.
No production code changes are needed. The present empty map, incomplete flag,
and status-1 verdict remain intact. The existing four pool roles may still run
as diagnostic evidence, but they cannot certify F1.

Cost: **documentation/decision only**, no additional heavy child. The cost is
that F1 remains incomplete; this option is not an exclusion, normalization, or
green ruling.

### STOP boundary and forbidden shortcuts

**STOP NR:** do not change node-reuse evidence, coverage, or comparison code
until the owner selects NR-A or NR-B. NR-A additionally requires the owner to
ratify the sealed input/backend/materializer and supplemental-RNG ownership
contract, and requires the sibling QRF prerequisite to be available.

Forbidden shortcuts include:

- copying `CompiledNode.node_key` into the semantic reuse map;
- treating the operational executor journal or late-producer receipt as a
  semantic reuse identity;
- hashing the whole frame as a substitute for a compiler-declared transitive
  input projection (the current helper hashes the full frame and available
  inputs at
  `packages/microcosm-build/src/microcosm/build/us_runtime/pool_physical_executor.py:202-215`);
- omitting supplemental RNG owners or weakening the 72-site ledger;
- adding a shadow collector that never dispatches the real graph; or
- adding a D4 receipt normalization/exclusion for node-reuse keys.

## Decision 2: calibration scope

### Exact failing checks at this head

The certification runner is a pool-only runner. Its child command invokes
`build_us_multispine_pool.py` and stops after that child's evidence is loaded
(`tools/f1_certification_run.py:139-207`). The pool tool explicitly describes
its output as pre-calibration and says calibration is downstream
(`tools/build_us_multispine_pool.py:2-16`). Its evidence emitter therefore
hard-codes `calibration_scope_complete=False` and the reason
`normative_artifact_vector_omits_calibration_weights`
(`tools/build_us_multispine_pool.py:6017-6032`).

That receipt cannot honestly be flipped in isolation. The production receipt
normalizer rejects `complete=True` with
`production calibration completion has no sealed inventory contract`
(`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:2593-2607`),
and builds only the plan-linked false receipt
(`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:2593-2638`).
The validator accepts that production domain only when the verdict is false and
the same omission reason is present
(`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:1962-2016`).
Nevertheless, complete F1 coverage requires `calibration_scope_complete=True`
(`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:373-397`),
and failed vector coverage is included in the final status
(`packages/microcosm-build/src/microcosm/build/spec_engine/f1_certification.py:1382-1444`).

The missing artifact exists only in the downstream release path today. That
tool writes `household_weight`, `initial_household_weight`, target values and
diagnostics into a calibration NPZ
(`tools/build_us_fiscal_refresh_release.py:5727-5746`) and emits it after
calibration (`tools/build_us_fiscal_refresh_release.py:11221-11224`). The spec
declares three distinct release modes
(`packages/microcosm-build/src/microcosm/build/us/spec/calibration.yaml:226-241`),
while the pool spine lists `calibrate` and `select_exact_k` only as auxiliary
operations (`packages/microcosm-build/src/microcosm/build/us/spec/spine.yaml:71-77`).
The current pool runtime plan has no calibration authority field
(`packages/microcosm-build/src/microcosm/build/us_runtime/pool_runtime_plan.py:954-1045`),
and its closed authority projection enum has no calibration projection
(`packages/microcosm-build/src/microcosm/build/us_runtime/spec_authority.py:33-43`).

This is therefore a scope and authority mismatch between the normative F1
claim and the pool-only legacy payload—not a boolean bug. The downstream route
also requires a Ledger facts artifact and supports explicit hash pins
(`tools/build_us_fiscal_refresh_release.py:829-896`); its strict exact-k route
requires the mode inputs, authenticated pool manifest/release id, explicit seed,
and frozen Ledger/target pins as one closed set
(`tools/build_us_fiscal_refresh_release.py:1534-1605`).

### Why D4 normalization or exclusion cannot resolve it

Calibration weights are expressly inside the normative byte-identity vector
and must be raw-byte equal (`_F1-CHARTER.md:156-164`). A receipt rule can only
classify provenance, publication, or operational differences
(`_F1-CHARTER.md:166-179`); it cannot make absent normative weights present and
cannot exempt their bytes.
The current compiler's artifact helper marks every emitted artifact normative,
required, and `raw_byte_exact`
(`packages/microcosm-build/src/microcosm/build/spec_engine/compiler_ir.py:2513-2536`),
but the closed pool artifact construction ends with pool, checkpoint, target
bank, terminal-gate, seed-map, and publication artifacts and never adds the
downstream calibration NPZ
(`packages/microcosm-build/src/microcosm/build/spec_engine/compiler_ir.py:2538-2679`).

### Options and costs

#### Option CAL-A — authorize strict composite pool-plus-release certification

The owner authorizes the certification unit to consist of a cold pool build
followed by a cold release/calibration build for each of constants A, constants
B, bundle A, and bundle B. The ruling must ratify one release mode from
`calibration.yaml`, the exact Ledger facts and manifest pins, pool manifest and
release-id binding, target-surface/incumbent inputs where the mode requires
them, the release seed, and the compiler/runtime authority that owns all of
those inputs. It must also ratify the normative calibration-weight surface. The
existing release writes both a fresh PolicyEngine-US H5 and a compressed NPZ
(`tools/build_us_fiscal_refresh_release.py:1-7`,
`tools/build_us_fiscal_refresh_release.py:11142-11148`,
`tools/build_us_fiscal_refresh_release.py:11221-11224`), so the ruling must
choose between a compiler-declared weight selector in the release H5 and named
raw `.npy` members in the NPZ. For the NPZ choice, the owner must say whether
`initial_household_weight.npy` joins `household_weight.npy` in the normative
set, and classify the target, estimate, error, and registry arrays as normative
or as sealed provenance/publication fields. The writer currently puts all of
those arrays in one archive (`tools/build_us_fiscal_refresh_release.py:5727-5746`);
that does not by itself make the whole archive normative. Constants and bundle
must each have an explicit release route, and both routes must emit the same
owner-ratified closed calibration-member inventory for raw-byte comparison.

Implementation extends the compiler execution ABI and runtime authority with
that sealed calibration contract, teaches the runner to launch and bind the
release child after each pool child, collects the selected H5 member or named
raw NPZ member inventory plus release receipts, and permits
`calibration_scope_complete=True` only after exact closure. An NPZ selector
must reject duplicate ZIP names, unsafe paths, object-dtype arrays, unexpected
shape/dtype/size, oversized headers or members, archive replacement and
mutation, and must compare the selected raw `.npy` member bytes exactly rather
than summaries or decompressed numeric equality. Regressions must also cover
missing/extra declared members, altered weights, mode or Ledger/manifest drift,
constants/bundle route drift, warm or resumed input rejection, and false
completion claims.

Cost: **high implementation and host cost**. The four certification roles
become eight heavy child invocations (four pool plus four release), run strictly
sequentially. The owner host must measure a new peak-RSS, elapsed-time, scratch
disk, and artifact-retention envelope for both child types; the pool-only
envelope cannot be assumed to cover a composite role. No such profiling is
authorized in this under-15-GiB lane. CI cost also includes affected compiler,
runtime, release, comparator/runner tests, every package test shard, and Ruff.

Recommendation: **CAL-A** if the current charter's calibration claim is to
become green. The owner must name both the release mode and normative weight
surface; this lane must infer neither from whichever legacy route happens to
pass.

#### Option CAL-B — amend the F1 flip scope to defer calibration

The owner explicitly narrows this flip to the pre-calibration pool and defers
composite release certification. Calibration weights remain normative for the
release that eventually claims them; they are not reclassified as provenance,
operational, or excluded. The four existing pool roles remain the complete host
workload.

Cost: **charter/versioning and documentation only**, with four pool children and
no release children. The current evidence must continue to say
`calibration_scope_complete=False`, and the current comparator must continue to
return status 1. Consequently CAL-B does not make the present F1 certificate
green; it creates a narrower owner claim and leaves full calibration
certification for a separately authorized contract.

### STOP boundary and forbidden shortcuts

**STOP CAL:** do not edit the runner, compiler artifact vector, calibration
receipt, or coverage gate until the owner selects CAL-A or CAL-B. CAL-A also
requires the owner to ratify the release mode, normative calibration-weight
surface and classifications, and the complete frozen input and authority set
before implementation or host-command changes begin.

Forbidden shortcuts include:

- setting `calibration_scope_complete=True` without a compiler-sealed,
  collected member inventory;
- reclassifying calibration weights as provenance, operational, or excluded;
- comparing summaries or diagnostics instead of the raw normative weight
  members;
- treating the entire compressed NPZ or all of its diagnostics as normative
  without the owner-ratified selector and classifications;
- running release only for bundle mode, only for one A/B role, or with different
  mode-independent inputs;
- reusing a warm calibration result or unpinned Ledger/pool/target inputs;
- changing modes, gates, bands, ceilings, folds, seeds, or target inputs toward
  a pass; or
- making the pre-calibration pool publisher pretend it performed downstream
  calibration.

## Combined owner choice and host-build consequence

| Owner choice | Heavy child invocations | Honest current outcome |
| --- | ---: | --- |
| NR-A + CAL-A | 8: four pool, four release | Can become green only after both contracts are implemented and all exact checks pass |
| NR-A + CAL-B | 4 pool | Red under the current comparator because calibration remains incomplete |
| NR-B + CAL-A | 8: four pool, four release | Red because node-reuse evidence remains incomplete |
| NR-B + CAL-B | 4 pool | Red on both deferred items |

The host handoff must not be expanded from four to eight children unless CAL-A
is selected and implemented. No combination authorizes overlapping heavy
children, reuse of an existing role root, weakening raw-byte equality, or any
change to the seed-ledger binding.

## Ruling requested

The owner should answer both lines independently:

```text
NODE REUSE: NR-A / NR-B
CALIBRATION: CAL-A / CAL-B
```

If `NR-A` is selected, also ratify the semantic input/backend/materializer and
supplemental-RNG ownership contract. If `CAL-A` is selected, also ratify the
release mode, frozen Ledger/manifest/target inputs, seed, and compiler/runtime
authority boundary. Until those selections are explicit, both items remain
stopped and the comparator remains honestly fail-closed.
