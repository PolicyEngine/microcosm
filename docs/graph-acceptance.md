# Node graph: acceptance charter

The `microcosm-graph` shard replaces stages, families, batches, banks, and
whole-run authority receipts with one object: a content-addressed DAG of
cell-ownership nodes. This document is the definition of done. Every
property below is an executable test in
`packages/microcosm-graph/tests/test_acceptance_*.py`, committed **red**
(`pytest.mark.xfail(strict=True)`) before the implementation exists, and
flipped to green by the pull request that implements it. The shard is done
when the acceptance suite carries zero `xfail` markers. Nothing else counts.

Three process rules keep the suite honest:

1. **Strict xfail.** A property that starts passing before its
   implementation PR fails CI, so a test can never be green by accident.
2. **Monotone burndown.** `tools/graph_acceptance_burndown.py --verify`
   fails if the number of `xfail` markers in the suite is higher than on
   `origin/main` (`origin/node-graph` until the shard merged). Nobody re-reds a property.
3. **Implementer ≠ author.** The lane that writes a property's test is a
   different lane from the one that makes it pass, and implementation
   pull requests never edit the suite. Ownership is in the last section.

Properties are named by group letter and number. Each carries the review
finding it closes (numbers refer to the 2026-09-01 architecture review) and
its owner.

## A. Identity and reuse

| Id | Property | Closes | Owner |
|---|---|---|---|
| A1 | **Determinism.** Two runs of the same graph over the same source bytes with the same kernels produce identical node keys and byte-identical artifacts, across process restarts and a reloaded store. Keys never depend on paths, hostnames, or clocks. | F2 | María writes, Max's session implements |
| A2 | **Memoization.** The second run of an unchanged graph executes zero kernels; the run manifest records a store hit for every node. | F2 | same |
| A3 | **Descendant-exact invalidation.** Changing one node's normative parameter re-executes exactly that node and its transitive descendants. Every ancestor and every sibling is a store hit. The test asserts the exact set of misses. | F2 (over-invalidation: 21 stores, 117 files on a `reason` edit) | same |
| A4 | **Inert-field invariance.** Changing a non-normative field (`description`, `citation`, a label) changes no node key. The schema marks every field normative or descriptive; the canonicalizer hashes only the normative projection. | F2, leg 2 §3.1 | same |
| A5 | **Code identity.** Changing a kernel's implementation hash (its source, or a pinned behavior-bearing dependency version) invalidates exactly the nodes bound to that kernel and their descendants. | F2 (leg 6: stacked checkpoint omits code) | same |
| A6 | **Input content identity.** Changing the bytes of a source input under the same name invalidates exactly its consumers. Renaming a source file without changing bytes invalidates nothing. | F2 | same |
| A7 | **Provenance is separate from reuse.** Adding a human decision, a release label, or run-request metadata to the run manifest changes no node key. | leg 3 identity triad; `docs/spec-engine.md:157-223` | same |

## B. Ownership and mutation

| Id | Property | Closes | Owner |
|---|---|---|---|
| B1 | **Ownership is total and exclusive.** Every non-source column has exactly one owning node. The compiler rejects a graph with two owners, and a graph that consumes a column nobody owns. | F5, leg 2 §3.3 | María / Max's session |
| B2 | **Executor enforces ownership.** A kernel that returns cells outside its declared owned positions is rejected; nothing it wrote reaches the population. Enforcement lives in the executor, and a kernel has no handle to the population. | F5, WIC incident | same |
| B3 | **Storage-preserving patch.** Patching owned positions preserves the incumbent column's dtype (nullable `boolean` stays nullable `boolean`; float bits including negative zero survive) and leaves every non-owned position byte-identical. This is the WIC guard, made structural. | leg 2 §3.3 | same |
| B4 | **Inputs are immutable.** A kernel receives read-only views; an in-place write raises inside the kernel and the node fails. | leg 1 finding 5 | same |
| B5 | **Null means absence.** A node declares each owned cell as *produced* or *absent*. A kernel writing a non-null value into an absent-declared cell is rejected. | `DESIGN.md:128-134` | same |
| B6 | **Entrants are declared.** An `EXPAND` node with `entrants=True` may return rows with null lineage; the executor requires the kernel to materialize every carried column for such a row (dtype-checked), records them as entrants rather than copies in the lineage receipt, and refuses null lineage on a node without the declaration. `entrants=True` with `mass='conserve'` is a compile error. An entrant needs a design anchor: it is admitted only while the weighted entity still carries `design` weights, and its admitted weight becomes that anchor; after any reweight the executor refuses entrant rows on that entity, because a derived weight has no design origin to record (ruled 2026-09-03, #847 gate round 3). | Dynamics: immigrant cohorts (microcosm-dynamics#412, #218) | Max's session; amendment 11 |
| B7 | **Entrant persons carry their stratum.** An entrant row on the person entity takes its stratum from `KernelResult.strata` (indexed by its new id); an entrant person absent from it, a label for a copied or incumbent person, or a label for an unknown id rejects the node. Entrant persons join incumbent or entrant groups through the materialized membership columns, and the mass ledger counts them from the node that admits them. | Dynamics: immigrant cohorts are persons (microcosm-dynamics#412, #218) | Max's session; amendment 14 |

## C. Seeds and factorization

| Id | Property | Closes | Owner |
|---|---|---|---|
| C1 | **Order invariance.** Permuting the declaration order of nodes, or the scheduler's packing of independent nodes into batches, changes no node key and no output byte. | F5 (`0347a009`) | María / Max's session |
| C2 | **Removal invariance.** Removing a node that nothing depends on, or adding a new leaf node, changes no other node's key or output. This is the `0347a009` replay: five targets removed, zero survivors re-modeled. | F5 | same |
| C3 | **Declared predecessors only.** A chained target's predictors are exactly its declared predecessors. The executor hands a kernel only its declared slices, so an undeclared read is impossible rather than merely detected. | F5, leg 3 §legibility | same |
| C4 | **Seed from identity.** A node's RNG seed is a pure function of its node key. Two nodes with identical declarations, inputs, and kernels in different graphs draw identical values. No positional RNG consumption exists anywhere in the shard (static check). | F4, `docs/spec-engine.md:254-282` | same |
| C5 | **Tolerance is declared.** A kernel claiming `tolerance_bound` numerics without a `Tolerance` is refused at registration, and a bitwise kernel may not carry one. The tolerance is recorded in every receipt, and a kernel reading a cell sees the declared tolerance of the node that produced it in `KernelContext.tolerances`: a structural version (`FILTER`, `EXPAND`, `REWEIGHT`) carries a column's tolerance through unchanged, so a bitwise carrier neither tightens nor erases a producer's bound, and a rewrite sees the incumbent producer's. Where more than one node wrote rows of a column in a version — a producer plus an `EXPAND` kernel that materialized entrant rows, or a claimant that took them over — the reader sees the loosest declared tolerance among those writers (componentwise maximum of `rtol`, `atol`, `ulps`; a bitwise writer contributes none), and a claimant's `KernelContext.tolerances` includes the coordinates it claims; a gate comparing against anything else says so in its evidence. | H2 (arm64/x86 one-ulp weights); microcosm-dynamics#412 | Max's session; amendment 13 |

## D. Weights and mass

| Id | Property | Closes | Owner |
|---|---|---|---|
| D1 | **Weight transitions are typed nodes.** `design → importance → calibrated` are the only legal transitions; the executor rejects a regression and rejects a transition declared on inherited (non-explicit) weights. | F9 (leg 1 finding 1) | María / Max's session |
| D2 | **Mass ledger.** Every population-changing node (select, concat, clone, reweight) emits a mass record with before/after totals and per-stratum mass. Mass is weighted person mass per stratum, within each declared partition (amendment 12). Under `conserve`, a stratum losing mass fails the node. An expansion that conserves its weight entity's mass while changing group composition changes person mass and must say so: `declared`, with a receipt stating the invariant it does hold (ruled 2026-09-02 on #844). `select` cannot drop mass silently. | F9 | same |
| D3 | **Cap anchored to design.** A calibration node's `max_weight_ratio` is asserted against the declared anchor across composed stages; a selection-then-refit chain that ships a record above `R × design` fails. | F9 (#493) | same |
| D4 | **Filters are binary.** A target filter containing NaN or a non-binary value is rejected at compile. | F9 | same |
| D5 | **Uncertainty travels.** A target's declared standard error reaches the calibration kernel's inputs; a kernel that ignores a declared `se` must say so in its capability record. | scoreboard row 5 (leg 1 finding 7) | same |
| D6 | **Mass is partitioned.** With `Graph.mass_partition` set, the ledger reports per stratum within each partition value, `conserve` holds within each partition, and a node that moves mass between partitions under `conserve` fails. Every `CREATE` node declares the partition column with a partition dtype, and no later node may own it (write or rewrite), or compilation fails: a partition value is fixed when the row is created, because a reassignment with the total unchanged is invisible to every mass policy. A row contributes mass only to the partitions it exists in. | Dynamics: person-period residency (microcosm-dynamics#412) | Max's session; amendment 12 |

## E. Store and resume

| Id | Property | Closes | Owner |
|---|---|---|---|
| E1 | **Content validation on load.** A stored artifact whose bytes were altered is rejected on load with `StoreCorrupt`; it is never used. | F3 | María / Max's session |
| E2 | **Verifier unavailability is fatal.** If the codec or dependency needed to load an artifact is unavailable, the executor raises `StoreUnavailable` and does not recompute. The 8/31 `ImportError` replay: the run stops before any kernel runs. | F3 | same |
| E3 | **Resume policy is real.** `require` refuses to execute any node without a store hit; `forbid` never reads the store; `auto` memoizes. All three are tested against the same graph. | F3 (`docs/spec-engine.md:454-460`) | same |
| E4 | **Atomic writes.** An interruption during an artifact write leaves no partial artifact visible; the next run treats the node as a miss. | leg 5 finding 8 | same |
| E5 | **Manifest completeness.** The run manifest lists every node key, hit or miss, kernel receipt, and seed; two runs of the same graph produce manifests that differ only in run-level fields. | F7 | same |

## F. Gates and release

| Id | Property | Closes | Owner |
|---|---|---|---|
| F1 | **A gate is a node.** Its verdict is an artifact keyed like any other; identical inputs hit the store; changed inputs re-evaluate. | F7, #611 | María |
| F2 | **Tier is derived.** A release node whose ancestry contains a failed gate cannot be certified; it is evidence-tier by construction. The certified loader rejects any manifest whose ancestry carries a failed gate or an evidence-tier node. | F1 (critical), #506 | María |
| F3 | **The one-field flip is impossible.** Mutating `tier` or `schema_version` in a serialized release manifest is detected, because both are derived from content-addressed ancestry and the manifest is keyed by its content. The review's reproduction is the regression test. | F1 (critical) | María |
| F4 | **Five outcomes, no accidental pass.** Gate outcomes are exactly `pass`, `fail`, `evidence_absent`, `not_applicable`, `unreached`. A kernel exception inside a gate becomes `fail` with the exception as evidence. | F7 | María |
| F5 | **Human decisions gate publication without entering keys.** A publication decision is a signed record with an owner, carried in the run manifest. The release node's owned `tier` derives from gate ancestry alone (so A7 holds: a decision changes no key); the executor reports the release's outcome as `unreached` when a required decision is absent, and the certified loader refuses such a manifest. A release is never certified by default. | F7 | María |

## G. Legibility and country neutrality

| Id | Property | Closes | Owner |
|---|---|---|---|
| G1 | **One-screen view.** `describe(node)` renders predecessors, parameters, seed derivation, owned cells, and kernel identity from the graph alone, under 40 lines, with no other file consulted. | F14 | Max's session |
| G2 | **The executor knows no country.** `microcosm-graph` imports no country package and contains no country string (static AST check). A UK graph and a US graph run through the same executor. | leg 5 finding 7 | Max's session |
| G3 | **Toy country in CI.** A synthetic country graph runs source → two chained QRF targets → calibrate → simulate (stub engine) → gate → release end-to-end in under 60 seconds on the fast lane, with zero restricted data. This is #378 step 2. | #378 | Max's session |

## V. Visuals (human review surface)

Every run of the executor can render itself. These are the artifacts María
reviews; they are generated from the run manifest and the graph, never
hand-drawn, so they stay true as the code moves.

| Id | Property | Closes | Owner |
|---|---|---|---|
| V1 | **Graph explorer.** `microcosm-graph explain <manifest>` renders one HTML page: the DAG with node keys, owned cells per node, store hit/miss per node, seed derivation, kernel identity, and the one-screen `describe()` for any node on click. Works on the toy country in CI and on a real run. | F14, G1 | Max's session builds; María reviews |
| V2 | **Acceptance burndown.** The same page (or its sibling) shows every property in this charter with its state (red/green), the PR that flipped it, and the four incident replays with their current verdicts. Published as an artifact on every PR into `node-graph`. | process | same |
| V3 | **Calibration view.** For a calibrate node: the target table with declared `se`, the weight-ratio distribution against the design anchor, and the mass ledger before/after, rendered from the node's artifacts. This is the existing calibration dashboard rehomed onto graph artifacts. | D3, D5 | same |
| V4 | **Incident replays, animated.** The four replays rendered as before/after diffs of node keys and cells, so a reader can see a removal changing zero survivor keys, or a dtype breach stopping at the node boundary. | narrative | same |

## H. Parity (migration acceptance)

| Id | Property | Closes | Owner |
|---|---|---|---|
| H1 | **Kernel parity.** Each wrapped legacy kernel (QRF fit and draw via `microcosm-fit`, calibrate via `microcosm-calibrate`, simulate via a `RulesEngine`) produces byte-identical output to the direct call on a pinned fixture and seed. A platform-bitwise kernel (amendment 16) is byte-identical within a platform: its fixture carries one pin per platform CI runs on plus the authoring platform, bytes are asserted on each of those, and on any other platform the node key must differ from every pinned key (identity partitioning). | #378 step 3 | Max's session |
| H2 | **UK spine parity.** The UK 26-stage spine expressed as a graph reproduces the current spine's `uk_frame_content_identity` on the fixture. Both sides run all 26 transforms from the fixture's raw tables in the test's own process, the graph through a CREATE kernel bound to the root transform, and nothing is pinned: the root transform's household weights differ by one ulp between machines (2026-09-02: two of 135 households on x86 versus this Mac), which the LCFS raking and every weight split then inherit. Stage order is derived from declared `consumes`; the hand-maintained `_STAGE_NAMES` tuple is deleted. | F12, F8 | María |
| H3 | **US post-transfer parity.** The stacked spine's derive → seed → simulate subgraph reproduces a pinned fixture output. | #378 step 3 | Max's session, later |

## The four incident replays

The candidate-26 incident is the acceptance fixture for the properties that
matter most. Each replay is a named test that reconstructs the incident's
shape on synthetic data:

1. **WIC dtype breach → B3.** A kernel returns a dense `bool` over a
   nullable `boolean` incumbent. The executor rejects the node.
2. **`0347a009` repack → C1 + C2.** Five leaf nodes are removed; every
   surviving node's key and output are unchanged.
3. **Engine-less environment → E2.** The codec for the engine manifest is
   unavailable; the run stops before recomputing anything.
4. **Evidence flip → F3.** The serialized release manifest's tier field is
   edited; the certified loader refuses it.

## Measured payoffs, once green

The suite proves properties. These are the numbers to publish alongside it,
so the payoff is measured rather than asserted:

- **Refit share on an unrelated edit.** On the US graph, an edit to a
  gap-fill `reason` string: store misses = 0. (Today: 21 stores, up to 117
  files.)
- **Verifier line count.** Lines of after-the-fact verification code
  deleted from `stacked_spine.py` as each invariant becomes structural.
  (Baseline: 9,546 of 13,931.)
- **Legibility.** Lines a reader must open to answer "what predicts
  `charitable_non_cash_donations`": today 424 across three files; target
  one `describe()` screen.
- **Cold-cache cutover cost.** Wall clock of the first full US run through
  the executor, receipted, so the number stops being an estimate.

## Interface freeze

`packages/microcosm-graph/src/microcosm/graph/decl.py` and `kernel.py`
define the contract both sides build against. Their canonical hash is
recorded in `docs/graph-interface.lock` at the start of parallel work.
Changing either file requires the owner's sign-off on the pull request and
re-recording the lock. Everything else moves freely.

Amendments so far (each re-locked):

1. **Structural kernels return data; the executor does the structural
   work.** Only `CREATE` returns `KernelResult.frame`. `FILTER` returns the
   surviving-row mask as `KernelResult.keep`; `REWEIGHT` and declared weight
   transitions return `KernelResult.weights`. No other kernel holds a
   population (B2). Raised by the acceptance lane; adopted 2026-09-01.
2. **Kernel roles.** `Capabilities.role` is `compute`, `gate`, or
   `release`. A gate's receipt carries `outcome` from `GATE_OUTCOMES` (now
   an interface constant) and `evidence`; a release owns a `tier` derived
   from the gate verdicts in its ancestry.
3. **Shared exception types** in `errors.py`, with the repository's
   `Error` suffix: `NodeRejectedError`, `StoreCorruptError`,
   `StoreUnavailableError`, `StoreMissError`.
4. **Decisions and keys** (F5, above): decisions live in the manifest, not
   in any key.
5. **Row masks are typed at compile time.** Every column's dtype is declared
   by its owner, so `compile_graph` refuses a row mask whose declared dtype
   is not `bool` or `boolean` (`MASK_DTYPES`); nulls inside a nullable mask
   are a run-time rejection. Raised by the runtime lane for D4; adopted
   2026-09-01.
6. **A weight transition is a REWEIGHT node.** Changing weights changes the
   population every later node reads, so `Node.weights` is legal only on a
   `REWEIGHT` node with a `base`, a `REWEIGHT` node must declare its
   transition, and the node's mass policy equals the transition's.
   `calibrate.adam@1` is therefore a structural kernel. Raised by the
   release lane; adopted 2026-09-01.
7. **Gate exceptions are verdicts.** A kernel whose role is `gate` and which
   raises produces outcome `fail` with the exception as evidence, and the
   run continues; a compute or release kernel that raises still aborts the
   run. `certified` requires every ancestral gate to be `pass` or
   `not_applicable`.
8. **Rewrites are declared cells.** `Owned.rewrite=True` says a node
   replaces a column its version carries from its base: the kernel receives
   the incumbent under the same name, the base's declared dtype must match,
   and the node owns the column in its own version. A rewrite therefore
   needs a version with a base (an identity `FILTER` opens one). Raised by
   the UK lane, which had ten rewriting stages; adopted 2026-09-01.
9. **Expansion lineage is a result field.** `KernelResult.expand` maps, per
   entity, the ids a new version adds to the base ids they copy; the
   executor carries columns from source rows, records lineage in the
   receipt, and records mass. Raised by the UK lane for the three clone
   stages; adopted 2026-09-01.
10. **Strings are `string`.** The graph stores text as pandas `string`
    (python storage); a population entering the graph with `object`
    strings is normalized at `CREATE`. Parity fixtures compare identities
    after the same normalization on the legacy side, and say so.
11. **Entrants are declared.** `Node.entrants=True` (EXPAND only) lets a
    kernel add rows that copy no base row: their lineage is null, the
    kernel materializes every carried column for them, the executor
    records them as entrants, and the node's mass policy cannot be
    `conserve`. An entrant is admitted only while the weighted entity
    carries `design` weights (its admitted weight is its design anchor);
    after a reweight the executor refuses entrant rows on that entity,
    since a derived weight has no design origin to record. Raised by the
    dynamics program (immigrant cohorts through the scheduled-entries seam,
    microcosm-dynamics#412 / #218); Max ruled go 2026-09-02; adopted
    2026-09-02; the design-anchor rule was stated 2026-09-03.
12. **Mass is partitioned.** `Graph.mass_partition = (entity, column)`
    partitions mass accounting (per stratum within each partition value;
    `conserve` per partition). Every `CREATE` node declares the column
    with a dtype in `PARTITION_DTYPES`, and `compile_graph` refuses any
    later owner of it, rewrite or not: a partition value is fixed when
    the row is created (review finding, 2026-09-02). The field is
    normative: the executor folds it into every structural node's key, so
    structural keys move once when a graph adopts it. Raised by the
    dynamics program for person-period residency; adopted 2026-09-02.
13. **Tolerance is declared.** `Capabilities.tolerance: Tolerance | None`
    (`rtol`, `atol`, `ulps`) is required for `tolerance_bound` kernels and
    forbidden for bitwise ones; `KernelContext.tolerances` hands each
    reader the declared tolerance of every input cell's producer, resolved
    through structural carriers to the node that wrote the values (a
    rewrite sees the incumbent producer's; where several nodes wrote rows
    of one column — entrant materialization, claims — the loosest declared
    tolerance among them). The whole `Capabilities`
    projection, tolerance included, is part of a node's identity and is
    compared on every cache hit. Raised by the H2 parity finding (root
    weights differ by one ulp between arm64 and x86) and the dynamics
    review; adopted 2026-09-02.

14. **Entrant persons carry their stratum.** `KernelResult.strata` (EXPAND
    kernels on an `entrants=True` node only) names the stratum of every
    entrant person by its new id; the executor requires exactly the entrant
    persons there. Raised by the implementation of amendment 11, which
    found the frozen result had no channel for a new person's mandatory
    stratum and left person entrants fail-closed; adopted 2026-09-02.

15. **Names are dot-free.** `compile`-time declarations refuse an entity,
    column, or row-mask name containing `.`: receipts, keys, and gate
    evidence spell a coordinate `entity.column`, and a dot inside either
    part would let two coordinates collide. `EntitySchema` itself still
    permits dots; the graph does not. Raised by the #847 gate review;
    adopted 2026-09-03.

16. **Platform-bitwise numerics.** `Numeric.PLATFORM_BITWISE`: identical
    bytes on one platform (architecture and locked dependencies), with no
    bound on cross-platform movement; a tolerance is forbidden on it as on
    `bitwise`. Adopted when measuring `fit.qrf@1` showed that a one-ulp
    difference in the forest flips which donor a quantile draw lands on
    (45 of 6,000 cells moved by up to 7% between arm64 and x86_64 while the
    rest agreed to one ulp; `docs/graph-qrf-cross-platform.md`), so no
    per-cell `Tolerance` is true of it. The node key of a platform-bitwise
    kernel carries a platform fingerprint (architecture, OS, Python minor),
    so a shared store never serves another platform's output, and parity
    pins record the platform: H1 asserts bytes on the pinned platform and
    identity partitioning elsewhere; a cross-platform gate on such a kernel
    says so in its evidence. Raised by the #847 gate review; adopted
    2026-09-03.

17. **Numeric scope per input coordinate.** `KernelContext.numerics` maps
    each declared input coordinate to a `NumericScope`: the loosest
    `Numeric` class among its writers (`bitwise` < `platform_bitwise` <
    `tolerance_bound`), the loosest declared `Tolerance` among the
    bounded writers, and the platform fingerprint the contract holds on.
    A platform-bitwise writer never disappears into a bound: it sets
    `platform`, so a sole platform-bitwise writer reaches a gate as
    `platform_bitwise` rather than as `None`, and a bounded writer mixed
    with one yields a bound that holds on that platform only.
    `KernelContext.tolerances` stays as the projection of `numerics`. A
    gate that compares across platforms must consult the scope and refuse
    or evidence a platform-scoped input (amendment 16). Raised by the
    #847 gate review (round 3); adopted 2026-09-03.

18. **Receipts are deterministic.** `KernelResult.receipt` is never hashed
    into a node key, but the run manifest's key hashes every node receipt
    less the executor's run-level fields (`hit`, `wall_time`, and a release
    node's decision-derived `outcome`), so a receipt must be a deterministic
    function of the computation: gate outcomes and evidence belong there;
    timings, host names, and iteration diagnostics that vary between runs of
    one computation do not. Manifest schema 2 records this identity; a
    schema-1 manifest loads as legacy in full (its receipts unauthenticated,
    `hit` forced to false) and `load_certified` refuses it. Raised by the
    #847 gate review; adopted 2026-09-03.

Adding a normative field with a default changes the canonical projection
of every node that carries it, so node keys moved with amendments 11 and
13's sibling field `entrants`; no released artifact pins a graph key yet.

## Ownership

Max's ruling (2026-09-01): the agents build all of it. The "implementer ≠
author" rule is kept by lane, not by person: the acceptance suite is
written by one sol lane from this charter against the frozen interfaces,
and implementation lanes never edit the suite; a suite change is its own
pull request reviewed by the Fable main.

- **Max's session (Fable main; sol lanes for every bounded leg):** the
  shard core, the legacy-kernel wrappers, the toy country, the acceptance
  suite (as its own lane), the visuals, and both country migrations.
- **María:** human review of the visual and interactive artifacts (group
  V), starting with the calibration view (V3), plus UK domain review of H2
  (open: H2 landed on `main` in #836 and awaits that review).
- **Max:** the rulings in the review's "Decisions that are yours" as they
  come due, and the `node-graph → main` merge (done: #836, 2026-09-02).

Branch history: `node-graph` was cut from `origin/main` at `4c6cc58c`; work
landed as pull requests into `node-graph`, and `node-graph` merged into
`main` in #836 on 2026-09-02 once the suite had zero `xfail` markers and
H1–H3 were green. Amendments to this charter now land as pull requests
into `main`.
