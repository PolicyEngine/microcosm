# GPT-5.6 Pro round-2 review of RFC v2 — VERDICT: REQUEST CHANGES
Extracted 2026-08-16 from https://chatgpt.com/c/6a81be80-1ba0-83ea-89fe-6d9a5f5cdde4
(sent 11:21 ET, "Worked for 17m 41s"; full text verified against page)

## Round-diff of its 15 findings
3 ADOPTED-CORRECTLY (dependency graph/plan lock; kernel capability typing;
shared core + country extensions), 11 ADOPTED-BUT-MUTATED, 1 LOST.

The LOST one: the semantic-versioning triad — v2's four identity classes
"solve a different problem"; semantic authority_version has no field, bump
rules, public-contract diff, precedence, or cache/logbook effects.

Key mutations flagged:
- Node keys adopted BUT global spec_binding also flows into QRF/transfer-bank
  bindings and "resume requires equality of all four classes" — global
  identities must describe the run, node identities must determine reuse;
  both gating reuse defeats node granularity.
- Canonicalization strong but no schema-migration translator, compiler
  implementation identity, or immutable schema-content digest.
- Five surfaces sound (fifth = chain state, good) but the YAML examples
  violate the separation (k, device, Supabase store, HF destination inside
  bundle files; sim batch size double-classified).
- Catalogs re-declare owner/lineage class = new drift pair; entity keys,
  membership cardinalities, artifact keys, executable row-scope language
  missing.
- legacy-v1 as named-stream map = documentation unless enforced (see (d)).
- Equivalence: fixed the tautology but needs FOUR builds (constants A/B,
  bundle C/D; require A=B, C=D, A=C) to separate authority difference from
  nondeterminism; comparison set must derive from plan.lock, not "the three
  H5 files"; must include model banks, final release artifact, manifests,
  mass history, receipts.
- ELIGIBILITY GUARD TOO WEAK: family-level required_concepts:[eligibility]
  with a generic eligibility tag does NOT make the CHAMPVA defect
  structurally impossible (drop veteran_va, keep own_coverage → still
  passes). Must be target-specific concept sets (CHAMPVA → veteran_status +
  military_coverage_context).
- Vintages: no typed compatibility constraints; engine pin duplicated
  between vintages.yaml and sources.yaml.
- Calibration/exact-k: names, not an executable mathematical contract (no
  scaling, weighting, zero/negative targets, optimizer, dtype, stopping,
  infeasibility, target priority; post-selection weight semantics absent).
- Sealed attempts: "sealed" conflicting with a running status; no atomic
  seal, idempotency, state transitions, orphan reconciliation; release DAG
  never actually defined as distinct from the strict-linear audit chain
  (needs derived_from / supersedes / revokes).
- Machine-decidable gates: metrics still lack populations, formulas,
  baselines, support minima, uncertainty, thresholds, missing-slice rules.

## Fusion adjudication
(a) Four classes = sound vocabulary, wrong composition: grammar already
folded into spec_sha256's envelope; global kernel_set can invalidate
unrelated nodes; runtime lock/output contract duplicate other classes; node
key lacks root-seed VALUE / behavior-relevant run inputs. Prescribes:
run_provenance_identity {source_grammar_receipt, global_resolved_spec_digest,
global_code_inventory_digest, global_artifact_protocol_inventory,
run_request, execution_receipt} for receipts/logbook, and node_reuse_key =
H(compiler_ir_abi_and_digest, resolved_transitive_node_slice,
behavior_relevant_run_inputs, transitive_input_content_hashes,
per_node_implementation_and_dependency_digest,
rng_protocol_and_actual_seed_material, input_and_output_artifact_contracts,
per_artifact_materializer_abi, output_sensitive_backend_abi) for reuse.
Schema migrations: schema_version selects an immutable migration chain with
recorded IDs + implementation digests; semantics-preserving migration
changes the audit receipt, not node reuse.

(b) Execution profile class in the node key contradicts the surface model.
Proven byte-invariant → attempt receipt only. Can change bits → resolved
backend/numeric ABI in the AFFECTED node keys. Suspected → temporary
cache-compatibility fence, never called output-invariant. device:auto can
never be output-invariant by declaration; receipt the resolved backend,
dtype, library stack, deterministic-algorithm mode; CPU/GPU share a key only
after per-kernel byte-equality conformance. Release labels never invalidate
computational nodes.

(c) F0 fast path = dual-authority trap as written, salvageable: manual
constants+bundle dual-editing is not honest authority. F0 needs nearly the
complete compiler FRONT END (parse/canonicalize/defaults/migrate, cross-ref,
typed entities/artifacts/scopes/columns, full stage DAG + producer graph
compile, seed-protocol resolution, complete normalization, compile-to-legacy
adapter covering every normative field, usage/coverage report proving no
field ignored, round-trip + mutation tests) — though not the executor. Safe
fast path: author the bundle ONCE; the compiler GENERATES the legacy payload
the constants-era executor consumes; record config_authority=constants_adapter
until F1; call spec_sha256 a mirror-attested configuration identity, not
proof the bundle executor built the artifact.

(d) legacy-v1 normative only if the runtime is FORCED to obey it: versioned
RNG broker; direct np.random/SeedSequence/random/framework RNG construction
outside the broker fails static + runtime checks; stochastic kernels receipt
consumed stream IDs. Normative: draw-site IDs, site→stream map, literal base
seeds, RNG family/version, spawn count/index, consumed order, reset/reuse
boundaries, entity/clone ordering, digest width/endianness/encoding, and the
immutable protocol implementation ID + digest. Rationale/code-paths =
descriptive.

## Fresh attack (new MAJORs)
1. Five-surface model contradicted by example schemas + migration map (k,
   device, logbook store, HF destination, batch size) — compile five
   physically separate typed objects; hash only the normative projection;
   declared precedence for default-vs-run-request values.
2. fingerprint → spec_sha256 boundary named, not specified: gen-0 keeps raw
   fingerprint; gen-1 raw = transport/package-integrity receipt only;
   spec_sha256 = semantic authority receipt; node reuse via compiled slices;
   {legacy_fingerprint, canonical_spec_sha256} compatibility map; NO second
   resource manifest (country_package.json only); immutable digests for
   schemas/migrators/composition. A Belgian SMOKE BUILD must precede the
   identity cutover (compile-only insufficient).
3. Producer graph: field coverage ≠ graph-semantic losslessness — needs
   explicit read-after-write edges, deterministic order for incomparable
   nodes (commute-or-disjoint rule), ordered fallback precedence with
   exhaustive/disjoint predicates, temp/validation-only outputs,
   entity-key/cardinality effects, typed link/membership/order/weight/mass
   mutations, unique ownership per CELL SEGMENT, retry/idempotence; plus a
   closed executable row-scope predicate algebra (labels like acs_rows can't
   police writes otherwise).
4. Take-up enum mixes ownership, mechanism, invocation optimization, policy
   interaction, and an untyped escape hatch (dedicated_stage undermines
   "kernels are the only escape hatch"; batched_seeded isn't a treatment;
   near_universal is a rate/edge policy). Fix: orthogonal ownership
   (measured|transferred|modeled|engine) × ordered typed pipeline-step kinds
   (probability_seed, count_calibration, …) + dependence group +
   final_owner_stage. Column/entity derivation needs a total-and-injective
   assertion.
5. Catalogs/vintages = authored drift pairs: split catalog into normative
   column contract / derived lineage report / non-normative docs; vintages =
   typed reference records (tax_period_ref, survey_period_ref,
   target_period_ref, geography_vintage_ref, policy_engine_surface_ref,
   release_series_ref) with validated compatibility relationships.
6. Equivalence gate: four-build structure; plan-derived comparison set; and
   adversarial rare/boundary fixtures (CHAMPVA-scale donor scarcity,
   empty/saturated take-up domains, county complements, crosswalk
   boundaries, overlapping producer fallbacks, zero/negative calibration
   targets, infeasible exact-k, clone tails, mixed-ownership columns) — a
   generic small fixture + one f004 may never exercise the motivating
   failures.
7. Deletion checklist additions: no is-guards/constant imports/alternate
   dispatch; node-invalidation matrix (unrelated edit reuses unaffected
   nodes, invalidates descendants); schema-migration + old-bundle reader
   fixtures; four-build determinism; fault injection around
   checkpoint/manifest/seal/promotion/logbook append; cross-profile
   byte-equivalence for every claimed-invariant profile; RNG-broker +
   ambient-read enforcement; real Belgian build + UK walking-skeleton
   EXECUTION (not compile).
8. Publication state machine: append-only attempt events → one immutable
   terminal seal (running ≠ sealed); temp artifact namespace, atomic
   manifest seal, output verification, idempotency key, promotion
   transaction, recovery for seal-ok/db-fail and db-ok/alias-fail, orphan +
   expiry reconciliation; strict_linear stays the tamper-evident AUDIT
   sequence, a separate release relationship graph (derived_from,
   supersedes, revokes) is the release DAG.
9. Machine-decidable gates: per-gate formula, input artifact/stage,
   population/denominator, slices, reference release/digest, minimum
   support, absolute+relative thresholds, uncertainty/multi-seed rule,
   missing-slice treatment, fail/warn/report status, typed reason mapping;
   calibration math contract; exact-k post-selection weight semantics.
10. Generic executor cannot enforce ambient reads (module globals, env,
    files, network, time, independent RNG) — brokers for file/env/clock/RNG;
    orthogonal capability fields: determinism | numeric_reproducibility |
    effects | structural_delta (none|filter|expand|join|relink|reorder|
    reweight) | retry_safety; structural_effect replaced by the specific
    delta + pre/postconditions.
11. MINOR: D6 can't stay an untyped open decision inside normative YAML —
    resolve before golden bundles freeze, or mark the release line
    explicitly provisional/non-normative and excluded from spec_sha256.

## Final verdict
REQUEST CHANGES — "v2 is not ready for owner sign-off as an implementation
contract … directionally much stronger than v1 … the remaining blockers are
internal contradictions in identity, authority, and rollout — not polish.
The most serious is that v2 promises per-node reuse while still globally
binding spec_sha256 and all four identity classes into checkpoint and bank
eligibility."

Ranked minimal v3 list: (1) separate run provenance from node reuse; (2)
restore the versioning triad with precedence/bump rules + immutable
schema/migrator/canonicalizer/compiler digests; (3) make F0 single-authored
(bundle → generated legacy payload; full front-end coverage, compile-back,
usage accounting, mutation tests before CHAMPVA-class edits use the path);
(4) reconcile the five surfaces mechanically; (5) legacy-v1 executable via
RNG broker + draw-site inventory + receipts; (6) correct domain schemas
(target-specific eligibility concepts; orthogonal take-up; no
dedicated_stage); (7) complete graph/contract semantics (row-scope predicate
language, deterministic ordering, fallback disjointness, segmented
ownership, derived catalogs, typed vintage references); (8) strengthen
certification/deletion (four builds, plan-derived comparison, adversarial
fixtures, node-invalidation tests, cross-profile determinism, migration
fixtures, crash/recovery injection, real BE + UK smoke executions); (9)
finish publication semantics (attempt events, atomic seal, idempotent
promotion, audit chain ≠ release DAG); (10) executable quality/calibration
gates; (11) resolve D6 or exclude the provisional release line from
normative identity.
