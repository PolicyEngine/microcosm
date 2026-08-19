# F1: drive — the generic executor, brokers, and bundle-mode authorities (RFC v3 APPROVED 2026-08-16)

**Why this lane exists, in one line:** F0 proved the bundle compiles to
today's payload byte-for-byte. F1 makes the bundle *drive* the build.
Everything Microcosm publishes after this lane runs off the spec bundle;
the constants-era executor becomes the thing F2 deletes. This is THE
migration. Nothing in this charter is about any single program or leg.

Work in `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-f1` (branch
`spec-engine-f1` @ 71485133 = spec-engine-schema + main). Run
`uv sync --all-packages --extra us` first. NO network beyond that sync;
commit locally on this branch only (do NOT push; owner pushes); journal
in `_F1-LANE-NOTES.md`; NEVER touch PROGRESS.md; NEVER stash; NEVER run
a build above 1% sample on this host (the owner runs the 25% rungs —
another 25% build is running concurrently and the box has rebooted 4×
today under memory pressure; keep every process under 20 GiB RSS).

Read in this order, fully, before writing code: `docs/spec-engine.md`
(v3 approved — BINDING; §"Kernels: executor + brokers", §"The producer
graph", §"Rollout" F1 paragraph, §"Decisions" D1–D5), the 15 schemas in
`packages/microcosm-build/src/microcosm/build/spec_engine/schema/`,
every module in `packages/microcosm-build/src/microcosm/build/spec_engine/`
(F0's compiler: loader → resolver → compiler_ir → legacy_adapter,
seeds.py = the draw-site ledger, engine_abi.py, inventory_coverage.py),
`_F0-LANE-NOTES.md`, `docs/evidence/spec-engine/us-f0-coverage.json`,
`_698-SOL-REVIEW-R2.md` + `_698-PRO-REVIEW-R2.md`, and
`tools/build_us_multispine_pool.py` end to end (this is what you are
replacing the *authority* of, stage by stage — read how it constructs
authorities from constants today: gap_fill_plan, transfer surfaces,
predictor tuples, late registry/schedule/ownership, take-up contract,
tail contract, seeds, rung grammar, release regex).

## What F1 is (verbatim intent from the approved RFC)

"F1 — drive. Generic executor + brokers; producer-graph compile-back
byte-identical; bundle mode constructs the authorities; per-PR cold
dual-mode fixtures; the four-build restricted f004 certification,
flipping stage by stage; geography = exact legacy behavior. Derived
closure/segments/dashboard retarget to compiler outputs here (not the
held authored-class tests)."

Round-2 design that binds the executor (RFC §Kernels): immutable
projection in, patch out, full structural diff, rejection outside
declared entity/column/row scopes; ambient access BROKERED (pure and
seeded kernels run with ambient access prohibited or instrumented;
file/env/clock/RNG access only through explicit brokers); capabilities
are ORTHOGONAL fields (determinism / numeric_reproducibility / effects /
structural_delta / retry_safety) — `structural_effect` is dead; row
scopes are a closed predicate algebra. Seed protocol `legacy-v1` is
enforced by the RNG broker + draw-site ledger (D5): every stochastic
callsite draws through a broker stream token whose site id is in
`seeds.py`'s ledger — no private RNG, no ambient numpy global state.
Provenance: `run_provenance_identity` records the triad; `node_reuse_key`
never includes provenance (D3).

## What F1 is NOT (violating any = FAIL)

- NOT a behavior change. Bundle mode must reproduce constants mode
  byte-for-byte on every artifact the four-build certification covers.
  If bundle mode "improves" anything, that is a bug in bundle mode.
- NOT deletion. Constants-mode stays runnable and is the oracle. F2
  deletes, machine-decidably, after ≥1 certified full release.
- NOT a program-specific effort. If a stage's flip needs a program
  named in code, you have found a leak — file it in the notes and
  route it through the generic contract (take-up = ownership × typed
  steps over the contract programs; the engine ABI lock is the column
  authority).
- NOT F3: no block-first geo, no 37-target re-unification, no
  derived-v2 seeds. Geography = exact legacy behavior.
- NOT manual dual-editing, ever (F0 rule stands).

## Deliverables (each a commit; suite green after each; notes updated)

1. **Sync + baseline + authority inventory.** Full suite green (record
   count). Then in `_F1-LANE-NOTES.md` §1: every authority the pool
   tool constructs from constants today, by module:line, mapped to the
   compiled-IR object that will replace it, with the stage it feeds
   and the artifact(s) it affects. This inventory is the flip plan and
   the certification target. Include the exact list of stochastic
   callsites (grep `np.random`, `default_rng`, `RandomState`, hashlib
   draws, `random.`) with their draw-site ledger id — every one must
   have one or the ledger is incomplete (that would be an F0 defect;
   fix it and re-run inventory coverage, do not paper over it).
2. **Executor core** (`spec_engine/executor.py` + tests): the generic
   executor per §Kernels — takes a compiled node + immutable frame
   projection, returns a patch; applies structural diff; rejects writes
   outside declared scopes; validates orthogonal capability fields
   before dispatch; deterministic total order for incomparable nodes.
   Fixture-scale tests: synthetic frames, at least one kernel per
   structural_delta kind, adversarial out-of-scope write refused.
3. **Brokers** (`spec_engine/brokers.py` + tests): RNG broker (stream
   token → seeded generator per the ledger's rng_family/kernel/
   seed_material/consumption_order/reset_boundary; legacy-v1 semantics
   verified against a captured constants-mode draw for ≥3 real sites),
   file broker (declared_source_read only; refuses undeclared paths),
   env/clock brokers (instrumented; kernels declared pure/seeded get
   refusal on ambient access — prove it with a test that monkeypatches
   `os.environ` reads and `time` and asserts refusal). Provenance:
   broker access log is a receipt surface (operational), NOT in
   `spec_sha256` and NOT in `node_reuse_key`.
4. **Bundle-mode authorities.** A `config_authority=bundle` path in
   the pool tool that constructs every inventoried authority from the
   compiled IR (not from constants, not from the generated legacy
   payload — from the IR objects). Constants mode untouched. Per-PR
   cold dual-mode fixture: fixture-scale build in both modes → every
   artifact byte-identical (extend the existing D4 byte-identity gate
   to run BOTH modes; the gate output records both identities and the
   diff = empty). This is the first real proof the bundle drives.
5. **Stage-by-stage flip receipts.** For each stage in the inventory,
   a receipt row: constants-mode digest, bundle-mode digest, equal
   yes/no, at fixture scale, then at 1% (`--sample-fraction 0.01
   --sample-seed 578`, off-chain, no logbook — omit
   `--logbook-prev-row-digest`; NEVER touch logbook-pending-chain.txt).
   Any stage that is not byte-equal is a defect in bundle mode: fix,
   do not tolerate. Record wall-clock and peak RSS per mode.
6. **Four-build restricted f004 certification** (D4): four cold builds
   (constants×2, bundle×2) at 1%; within-mode determinism + cross-mode
   equality on the plan-derived artifact vector incl. publication +
   banks; concrete resume predicate exercised (kill one build mid-stage,
   resume, artifacts identical to the uninterrupted one). Report as
   `docs/evidence/spec-engine/us-f1-certification.json` + a human
   summary in `docs/evidence/spec-engine/us-f1-certification.md`.
7. **Retarget derived closure/segments/dashboard** to compiler outputs
   (RFC F1 paragraph, last sentence). Existing tests that read the
   held authored-class fixtures move to reading compiled IR.
8. **Handoff**: `_F1-LANE-NOTES.md` final section = exact state, what
   the 25% owner-run flip needs (command line, expected identities,
   what to compare), and every leak/defect found with its disposition.
   Update `docs/spec-engine.md`'s rollout section status line for F1.

## Acceptance (the owner adjudicates; sol self-audits first)

- Full suite green in constants mode AND bundle mode.
- Dual-mode fixture gate: zero byte differences.
- 1% four-build certification: within-mode determinism PASS, cross-mode
  equality PASS, resume predicate PASS.
- Zero program names in executor/broker/authority-construction code.
- Every stochastic callsite draws through a broker token in the ledger.
- Notes complete; nothing pushed; nothing above 1% built.

If a deliverable cannot be met honestly, write WHY in the notes and stop
at that deliverable with the suite green — a partial-but-honest F1 beats
a green-washed one. Do not tune any gate/threshold/band/seed toward
passing.

---

## OWNER RULING 2026-08-19 07:35Z — deliverable-4 identity scope (resolves the preflight finding)

Read `_F1-LANE-NOTES.md` §"Preflight finding for deliverable 4". The three
contradictions are real at the raw-byte level and the resolution is the
RFC's own five-surface doctrine (§identity split; only NORMATIVE content
hashes into spec identity; run-provenance and operational surfaces are
receipt material). Apply it to the D4 gate exactly as follows:

1. **The byte-identity gate covers the NORMATIVE artifact vector**: pool
   H5 datasets (every entity table, every column, every weight vector),
   checkpoint payload bytes, gap-fill/transfer/QRF banks, calibration
   weights, gates.json verdicts and per-target rows, target banks, the
   compiled seed-stream map, and the plan-derived artifact list the D4
   text names ("plan-derived vector incl. publication + banks"). These
   MUST be raw-byte equal between constants mode and bundle mode. No
   canonicalization, no exclusions. If a NORMATIVE artifact differs,
   bundle mode is wrong.

2. **Provenance / operational surfaces are compared STRUCTURALLY under a
   DECLARED, SEALED canonicalization vector** that you author as a
   compiler-emitted execution-ABI document (`plan_lock` is the right
   home; extend it — do not invent a second lock). The vector names each
   receipt field that is (a) identity-generation-dependent — the D3
   triad, `identity_generation`, `config_authority` — (b) publication-line
   text — the D6 prefix — or (c) operational — absolute paths, durations,
   UUID nonces, clock values, host/executable strings — and states the
   comparison rule per field: `equal_after_normalizing_prefix`,
   `expected_to_differ_by_generation`, or `operational_excluded`. The
   gate FAILS on any receipt difference NOT covered by a sealed vector
   row. This is not weakening byte equality; it is stating precisely
   which bytes are normative — the same distinction spec_sha256 already
   makes.

3. **D6 at F1**: bundle mode publishes `microcosm-us-2024-*`; constants
   mode keeps `populace-us-2024-*`; the vector row is
   `equal_after_normalizing_prefix` and readers accept both (already the
   D6 ruling). Do NOT make constants mode emit the new prefix and do NOT
   make the bundle select a legacy field.

4. **D3 at F1**: bundle receipts carry generation 1 + triad; constants
   receipts carry generation 0. Vector row `expected_to_differ_by_
   generation`, with the gate asserting the bundle-side values equal
   the loader's `run_provenance_identity` (so the receipt is checked, not
   ignored). `node_reuse_key` never includes any of these fields (D3
   stands) — assert equality of `node_reuse_key` across modes.

5. The seed-ledger correction (53→72 sites, vehicle-QRF literal 42→0,
   Torch reset sites, PCG64(0) state restoration) is an F0 defect fix
   and is IN SCOPE — commit it as deliverable 1's second commit with the
   regenerated protocol digest, coverage evidence, and pins. It moves
   `spec_sha256`; that is legitimate. Record the old→new digests in the
   notes so the owner can re-pin the spec-engine-schema branch too.

Proceed through deliverable 4 under this ruling. The four-build 1%
certification (deliverable 6) uses the same two-tier vector: NORMATIVE
raw-byte, provenance/operational under the sealed vector. Everything
else in the charter stands.
