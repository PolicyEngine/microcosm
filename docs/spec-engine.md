# The spec engine: one declared bundle drives the build

Status: **v3 — APPROVED by Max 2026-08-16** ("sure i approve") after two
rounds of cross-family review. Sign-off covers the schema shape, the
P→F0→F3 phasing, and the schema set in `specs/schema/`. D6 RULED same day:
**`microcosm-us-2024-*`** is the spec-engine-era release line (Max,
2026-08-16); `populace-us-*` rows stay valid-historical in the chain; the
HF dataset destination is unchanged and outside this ruling.
Nothing here is wired; excerpts below are **non-normative pseudocode until
the committed schemas and the complete US bundle exist** (both reviews
correctly rejected "schema-conforming" as a label for drafts).

Review provenance:

- Round 1: sol (code-grounded, 16 MAJOR) + GPT-5.6 Pro (design, 15
  findings) on v1 → v2 (d865ba40).
- Round 2 on v2: **both returned request-changes.**
  - sol (`_698-SOL-REVIEW-R2.md`): 11 of 20 r1 findings RESOLVED, 6
    partial, 3 renamed-not-resolved; 9 new MAJORs (N1–N9). Every
    spot-checked claim held again (5 batches/37 targets; 13 contract
    programs with no wic/social_security; YAML banned by the package
    test).
  - Pro (`_698-PRO-REVIEW-R2.md`): 3 of 15 adopted-correctly, 11
    mutated, 1 LOST (the versioning triad); fusion adjudication +
    11 fresh MAJORs.
  - Convergent blockers (both, independently): F0 as written contains
    F1's compiler and manual dual-editing is a dual-authority trap; the
    node key over-keys execution profile and under-keys run values;
    catalogs/vintages recreate authored drift pairs; the take-up enum
    conflates dimensions; the producer graph is not yet lossless; the
    equivalence vector stops too early; the deletion checklist is not
    machine-decidable.
- v2 factual errors corrected here: `puf_tax_itemization` is **five**
  batches / **37** targets (19 bounded groups / 70 targets across the
  registry), not 4/32; `wic` and `social_security` are not contract
  programs; `takes_up_eitc`/`takes_up_dc_ptc` break the suffix naming
  rule (column mapping must come from an engine ABI projection).

## The ruling this implements

Max, 2026-08-15: *"It should be immediately obvious which predictors we're
using for each variable … this part should just be a yaml file. Also the
attributes of the ML model. … Let's do it right before making atomic
fixes. Scope out the right schema for everything first."*

Today the spec (#695/#697 line) **mirrors** code and CI enforces the
mirror. The spec engine **flips the authority**: the build compiles the
spec bundle into an executable plan, constructs its plan objects from it,
and the Python constants are deleted. Custom logic survives as **named
kernels** — code keeps the *how*, the spec owns the *what, from what,
with what*.

## What exists today (five authority surfaces, one disposition each)

1. **`CountrySpec`** (`country_spec.py`): packaged spec-only country
   system; US/UK/BE packages live; Belgium is its first full consumer.
   Its manifest is a bare filename list, every resource parses as a JSON
   mapping, typed behavior keys off hard-coded filenames, and the
   package test bans YAML (`SPEC_SUFFIXES = {".json", ".jsonld"}`).
   **Disposition: replaced-in-place through an explicit seam** — see
   "One spec system." Calling this an "extension" without the seam was
   v2's mistake; the migration is real work and is named as such.
2. **`specs/us_imputation_lineage.yaml`** (#695): mirror-mode lineage
   spec. **Disposition: retired at the flip**, with its named consumers
   migrated, not orphaned: `test_imputation_lineage_spec.py`,
   `tools/emit_lineage_dashboard.py`, the stale packaging comment, and
   the external dashboard handoff.
3. **The take-up contract** (`us/take_up_contract.json` +
   `source_stages.json` rows): a *reviewed snapshot* of engine facts
   whose currentness assertion is a deliberate tripwire.
   **Disposition: absorbed — but the tripwire survives** as a generated,
   committed **engine ABI lock** (below); derive-and-assert against the
   live engine alone would be circular on an engine bump.
4. **Python constants** across `us_runtime/` and the stacked tool.
   **Disposition: bundle-normative per the migration map; deletion gated
   by the machine-decidable checklist.**
5. **`builder_code_identity`** (`code_identity.py`): used by UK/PUF
   builders, absent from f004 stacked identity. **Disposition: joins the
   code identity class.**

## Principles

1. **Total, or failing.** Closure over the selected country/profile
   registry namespace; unknown referenced ids and duplicates always
   fail; library-only implementations don't.
2. **Declarations carry mechanisms, not just names.** Every stochastic
   write declares draw universe, conditioning, parameter source, and
   seed stream — and the runtime is **forced to obey** (RNG broker,
   below), or the declaration is documentation.
3. **Provenance is global; reuse is per-node.** Global identities
   describe every run in receipts and logbook rows; node keys decide
   cache/checkpoint reuse. Neither substitutes for the other.
4. **Kernels are the only escape hatch — behind a generic executor and
   brokers.** No second escape hatch anywhere else in the schema (the
   v2 take-up `dedicated_stage` value died for this reason).
5. **Country files instantiate a shared core plus declared extensions.**
   UK and Belgian semantics must be *expressible*; Belgium must
   **build** (smoke run, not compile-only) before the identity cutover.
6. **The dashboard reads the same bundle the build reads.**
7. **Derive, don't declare — in one direction only.** A field derivable
   from another authority is compiled and asserted. Where a *reviewed
   snapshot* of an external authority is the point (engine facts), the
   snapshot is a **generated lock**, regenerated deliberately, never
   silently re-derived at build time.
8. **Single-authored always.** At no phase do humans hand-edit two
   representations of the same fact. The fast path is generation, not
   parallel editing.

## One spec system: the CountrySpec seam (explicit)

- `country_package.json` becomes the **single typed resource manifest**:
  `{path, kind, schema_id}` rows replace the bare filename list. The
  bundle introduces **no second file inventory** — v2's
  `bundle.yaml: files` is dead; the root `bundle.yaml` shrinks to
  bundle-level settings (country, seed protocol, generation).
- The loader returns one `ResolvedCountrySpec` carrying (a)
  migration-era **compatibility projections** (today's `sources`,
  `gates`, take-up contract views — so `load_take_up_contract()` and
  UK `load_country_spec("uk").gates` keep working until their consumers
  migrate) and (b) the compiled spec-engine IR.
- The spec-only package test is updated **intentionally** to admit YAML
  + kernel ids as declared kinds.
- **Generation semantics** (Pro r2): generation-0 artifacts keep the
  historical raw `fingerprint`. In generation 1 the raw file-set
  fingerprint is a **transport/package-integrity receipt only**;
  `spec_sha256` is the semantic authority receipt; node reuse uses
  compiled node slices, never either global hash. A compatibility
  record may map `{legacy_fingerprint, canonical_spec_sha256}`; old
  artifacts are never retro-labeled.
- Named consumer migrations: the gate battery's `spec_fingerprint`
  (today composed only from `gates.json` while its contract doc claims
  the whole package) gets one owner and one definition; every
  fingerprint/spec-hash consumer is assigned a generation.
- Schema files, migration translators, and the composition manifest all
  carry **immutable content digests**.

## Canonicalization, schema migration, and locks

The v2 canonicalization contract stands (YAML 1.2 restrictions,
closed-world schemas, default injection into a typed `ResolvedSpec`,
domain-separated envelope, ordered-arrays-stay-ordered, golden vectors,
invalid fixtures). Round 2 adds the missing pieces:

- **`schema_version` selects an immutable migration chain**: migration
  ids + implementation digests are recorded in the grammar receipt. A
  semantics-preserving migration changes the audit receipt, never node
  reuse, because the compiled node slice is unchanged.
- The **compiler itself has an identity**: `compiler_ir_abi` + digest.
  Two compiler implementations resolving the same declared schema
  version differently is an identity difference, not a silent fork.
- Emitted, never authored: `bundle.lock.json` (file hashes + grammar
  receipt), `plan.lock.json` (typed IR: stage DAG, resolved node
  params, kernel pins, node keys), `engine_abi.lock.json` (below).
  Generated locks are reproducible from their authorities and rejected
  if hand-edited.

## Identity: the triad restored, provenance split from reuse

Round 2's sharpest convergent finding: v2 bound everything into
everything. v3 separates three questions.

**1. What is this run? — `run_provenance_identity`** (in every receipt,
manifest, terminal attempt, and logbook row; never a reuse gate):

```text
run_provenance_identity:
  identity_generation: 1          # absent/0 = historic, readable, never
                                  # promotable; 1 = binding required;
                                  # unknown = refuse
  source_grammar_receipt          # schema_version + canonicalizer +
                                  # migration chain ids/digests
  spec_binding:                   # {country, schema_id, schema_version,
                                  #  canonicalizer_version, spec_sha256}
  authority_versions:             # semantic contract version per named
                                  # authority (stacked authority v10 is
                                  # the live example) — field, bump
                                  # rules, and precedence below
  code_inventory_digest           # builder_code_identity + kernel set
  artifact_protocol_inventory     # materializer/serialization versions
  run_request                     # rung, seeds, k, release label
  execution_receipt               # resolved backend/profile, workers
```

**The triad and its precedence** (Pro r1's finding, dropped by v2,
restored): `schema_version` (grammar era; unsupported → refuse before
execution) → `spec_sha256` (exact resolved configuration; any semantic
config edit changes it) → `authority_version` (the *semantic contract
era* of a named authority; bumped when the contract's meaning changes
even under unchanged YAML shape; recorded per authority; a bump
invalidates dependent caches by entering the affected node slices).
Code and serialization digests pin implementations underneath. A
semantic behavior change may never ride a materializer bump unless
representation also changed.

**2. What may be reused? — `node_reuse_key`** (per executable node, once
the compiled plan drives execution):

```text
node_reuse_key = H(
  compiler_ir_abi_and_digest,
  resolved_transitive_node_slice,        # this node's plan + upstream
  behavior_relevant_run_inputs,          # rung, k where consumed, and
                                         # ACTUAL seed material for this
                                         # node's streams
  transitive_input_content_hashes,
  per_node_implementation_and_dependency_digest,
  rng_protocol_and_seed_material,
  input_and_output_artifact_contracts,
  per_artifact_materializer_abi,
  output_sensitive_backend_abi           # ONLY backends proven to
)                                        # affect bytes; nothing else
```

Execution profile is **not** in the key. Proven byte-invariant settings
live only in the attempt receipt; settings that can change bits are a
resolved backend/numeric ABI in the *affected* nodes; suspected-but-
unproven settings may serve as a temporary cache-compatibility fence and
are never called output-invariant. `device: auto` is never
output-invariant by declaration — the launcher receipts the resolved
backend/dtype/library/deterministic-mode, and CPU/GPU share a key only
after a per-kernel byte-equality conformance test. Release labels never
invalidate computational nodes.

**3. Interim reality.** Until `plan.lock` drives execution, the existing
whole-run identity machinery keeps gating resume exactly as today —
conservative over-invalidation, which is safe. `identity_generation` +
`spec_binding` enter `_configured_stacked_identity` and
`_stacked_checkpoint_base_identity` (stage and bank identities inherit),
`run_config`, manifests, terminal attempts, and logbook rows — the
concrete field homes sol specified. Discovery passes both through when
reconstructing expected identity.

## Configuration surfaces — five compiled objects, zero leaks

The five surfaces stand (normative / run request / execution profile /
operational bindings / external chain state). Round 2 caught the
examples violating them, so the rule is now mechanical: **the compiler
emits five physically separate typed objects; every schema field
declares its surface; the canonicalizer hashes only the normative
projection.** A value may be a normative default *and* a run-request
override only with declared precedence and a resolved receipt.

Example-level corrections baked into the excerpts below: `k` is a
run-request knob (the bundle may declare a default with precedence);
`device` is profile/backend, out of calibration.yaml; the logbook store
and HF destination are operational bindings, out of publication.yaml's
normative block; simulation batch size is classified by an invariance
proof, not by assertion.

Failed spec loads carry `spec_binding_status` + attempted grammar +
(when readable) a raw file-set digest never called `spec_sha256`.

## Seeds: an enforced protocol, not a map

`seed_protocol: legacy-v1` ships with the flip and changes nothing;
`derived-v2` (stateless counter streams, root seed in the run request)
is an F3 behavior edit with statistical gates. Round 2's demand — from
both reviewers independently — is enforcement:

- **The three-way boundary.** *Spec-normative:* stable draw-site ids,
  site→stream mapping, literal base seeds, RNG family/version, spawn
  count and index assignment, consumed target/program order,
  reset/reuse boundaries, entity/clone ordering where draws depend on
  it, digest width/endianness/encoding, and the immutable protocol
  implementation id + digest. *Kernel-contract-normative:* internal
  consumption patterns (QRF's `SeedSequence(seed).spawn(2)` and shared
  fit-RNG advance in target order), pinned by the kernel/code digest.
  *Receipt-descriptive:* realized seeds, saved states, rationale.
- **The RNG broker.** Production code obtains RNGs only through a
  versioned broker; direct `np.random`/`SeedSequence`/`random`/framework
  RNG construction outside it fails static and runtime checks; every
  stochastic kernel receipts the stream ids it consumed. Without the
  broker the map is documentation.
- **The exhaustive draw-site ledger.** v2's 578/0/42+ACS/QRF summary was
  not an inventory. The ledger additionally covers (sol N9): SSI
  training/model seeds, SIPP vehicle/asset stable-string hash
  algorithms, the ACS-rent archived hash, the tips training seed, and
  the SCF composite `SeedSequence` — plus anything a repo-wide audit
  finds. Golden ACS label/seed vectors and a multi-regime QRF chain
  fixture pin the algorithms.

## Kernels: executor + brokers + orthogonal capabilities

The generic executor stands (immutable projection in, patch out, full
structural diff, rejection outside declared entity/column/row scopes).
Round 2 additions:

- **Ambient access is brokered.** The executor cannot stop a Python
  callable from reading globals/env/files/network/clock/private RNG —
  so pure and seeded kernels run with ambient access prohibited or
  instrumented, and file/env/clock/RNG access exists only through
  explicit brokers.
- **Capabilities become orthogonal fields**:

```text
determinism: deterministic | seeded | nondeterministic
numeric_reproducibility: bitwise | tolerance_bound | unspecified
effects: none | declared_source_read | declared_sink_write
structural_delta: none | filter | expand | join | relink | reorder | reweight
retry_safety: idempotent | attempt_scoped | nonretryable
```

  `structural_effect` is replaced by the specific delta with pre/post
  conditions — otherwise it is an unbounded second escape hatch.
- **Row scopes are a closed predicate algebra.** Labels like `acs_rows`
  compile to canonical predicates whose overlap, exhaustiveness, and
  equality the compiler can decide; segmented lineage is unenforceable
  otherwise.

## The producer graph: lossless means graph-semantic

v2's sketch was not lossless (sol verified: it dropped
`producing_stage`, receipt ids, OR-of-AND alternatives, kind-specific
virtual-resource payloads, and the **18-row target × origin × clone-role
ownership matrix**). The v3 contract:

- Field-lossless: every `ProducerInput`/`ProducerOutput` field —
  entity/column/`required_scope`/`producing_stage`, ordered
  tolerated-absence receipt ids, alternatives as ordered OR-of-AND lists
  of `{entity, column, value_kind}` with declared precedence and
  disjoint/exhaustive predicates; outputs with `coverage_scope`; typed
  virtual resources (manifests, resolved weights, execution configs,
  transition/producer receipts, target banks) with each kind's semantic
  payload and digest rules; transfer groups; the full conditional
  ownership matrix with finalization and every non-owner action; the
  execution-receipt and transition-authority contracts.
- Graph-semantic: explicit read-after-write dependency edges;
  deterministic total order for incomparable nodes plus the rule that
  incomparable nodes commute or occupy disjoint write scopes;
  temporary/validation-only outputs; entity-key and cardinality
  effects; typed link/membership/order/weight/mass-history mutations;
  ownership unique per **cell segment**; retry/idempotence behavior.
- Acceptance: golden byte-identical compile-back of today's
  schedule/ownership payloads before the flip.

**The frozen split ledger, corrected**: at the flip, the entire current
ordered split structure — **19 bounded groups / 70 targets overall;
five batches / 37 targets for `puf_tax_itemization`** — is declared
explicitly with a reason per split, compiled back against both greedy
splitter implementations and every literal batch-name consumer
(ownership constants included), after which the two splitters and
`DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT = 8` die at F2. The F3
re-unification is scoped honestly: it changes the donor complete-case
population (intersection across 37 targets), family-label-derived
seeds, shared-RNG consumption order, cross-boundary chained predictors,
overlap-owner references, and bank identities — fresh node/bank
identities, cold caches, OOS/statistical gates.

## Lineage, closure, catalogs, vintages

- Whole-column closure derives from declared outputs; authored remnants
  are the inventory fixture and time-limited waivers. **The catalog
  splits three ways** (both reviewers, independently): a *normative
  column contract* (stable id, entity, dtype, unit, period, nullability,
  domain, public-stability status); a *derived lineage report*
  (producer, owner, origin class, scopes, stages — compiler-emitted,
  never authored); and a *non-normative documentation overlay*
  (descriptions, citations) hashed into a separate documentation
  digest, outside `spec_sha256`.
- Cell-scope segments over `(entity, column, row_scope, stage,
  write_policy)` stand, now backed by the row-scope predicate algebra.
- **Vintages become typed references, not repeated literals**:
  `tax_period_ref`, `survey_period_ref`, `target_period_ref`,
  `geography_vintage_ref`, `policy_engine_surface_ref`,
  `release_series_ref` — each living on the pinned source/lock record
  it describes, referenced by id everywhere else, with compiler-checked
  compatibility relationships. The engine pin appears in exactly one
  runtime/source-lock record; a CI check rejects duplicate literal
  authorities per normalized key.

## Take-up: orthogonal ownership × typed pipeline steps, 13 real programs

The v2 eight-value treatment enum conflated ownership, mechanism,
invocation grouping, policy interaction, and an escape hatch. v3:

```yaml
# take_up schema (pseudocode until schemas land)
ownership: measured | transferred | modeled | engine
pipeline:                # ordered, typed step kinds — closed enum of
  - kind: probability_seed | count_calibration | delivery_gate |
          assignment | measured_map        # STEP kinds, not program kinds
    kernel: kernel:…
    …step-typed params…
dependence: {group: …}   # batching/correlation grouping — an execution
                         # property, never a treatment
final_owner_stage: …
```

- **All 13 contract programs get committed rows** (no ellipsis): SNAP
  (national prior → state anchored count calibration, eligible-only
  domain, final owner); TANF (seeded rate); EITC (seeded rate,
  rate-by-approximated-qualifying-children); Medicaid (anchored count
  calibration); CHIP / Basic Health Program / DC PTC / Early Head Start
  (engine default with named debt); Medicare (measured ASEC map +
  support-clone propagation); SSI (target-derived age-band probability
  seed + delivered-recipient gate — it never count-matches flags);
  Head Start (imputed/transferred: SIPP-trained QRF); housing
  assistance (mixed row-scope: measured on ASEC rows, imputed on PUF
  support — segments, not one whole-column treatment); ACA (typed
  dedicated pipeline steps for Marketplace assignment + calibration).
- **The engine ABI lock**: from the single engine pin, the compiler
  generates `engine_abi.lock.json` — `{program → variable, entity,
  value_type, default, engine_class, consumers}` — committed and
  reviewed. CI compares a fresh derivation to the lock **before** bundle
  compilation; engine bumps fail closed until the lock is regenerated
  and every bundle-owned treatment re-reviewed. The engine owns its
  facts; the bundle owns treatments and pipelines; neither derives from
  the other at build time.
- Column/entity mapping comes from the ABI projection with a
  **total-and-injective assertion** (the suffix naming rule died on
  `takes_up_eitc`/`takes_up_dc_ptc`).

## Eligibility guards: target-specific concepts

v2's family-level `required_concepts: [eligibility]` with a generic tag
does **not** make CHAMPVA-class defects impossible (drop `veteran_va`,
keep `own_coverage`, still passes — Pro r2). v3: a concept registry maps
named concepts to predictor columns, and requirements are per-target and
concept-specific:

```yaml
concepts:
  veteran_status: [is_veteran, receives_va_payments]
  military_coverage_context: [acs_hins_va]
  disability_status: [has_hearing_difficulty, has_vision_difficulty]
families:
  - id: gap_fill/asec_survey_to_acs/person/benefit_participation
    targets:
      - name: has_champva_health_coverage_at_interview
        requires_concepts: [veteran_status, military_coverage_context]
      - name: has_tricare_health_coverage_at_interview
        requires_concepts: [veteran_status]
```

A target whose resolved predictor set fails to cover its concept set
fails the **load**.

## The equivalence gate: four builds, plan-derived vector

- **Four cold isolated builds**: constants A, constants B, bundle C,
  bundle D — require A=B and C=D (within-mode determinism) and A=C
  (cross-authority equivalence). Two runs cannot distinguish authority
  difference from nondeterminism.
- **The comparison set derives from `plan.lock.json`**, never a
  hand-named file list: every sealed stage artifact, trained-model bank
  used for resume, the final published logical frame/schema/period,
  normalized final manifest + diagnostics + gates, mass history, and
  behavior-bearing receipts. A missing output fails certification.
  Downstream calibration/selection/release nodes join the vector when
  they are in flip scope.
- **Resume-forbidden is a concrete predicate** over existing receipts:
  null `deepest_resumed_stage`, primary QRF `resume_status ==
  initialized`, and no ACS target bank entry with
  `load_status: resumed` / `source: checkpoint` — plus
  `--resume-policy=forbid` (refuse pre-existing manifests before
  loading) and one typed `resume_audit` with per-stage/per-target
  counts, all required zero.
- **Adversarial fixtures**, because a generic fixture never exercises
  the motivating failures: CHAMPVA-scale donor scarcity, empty and
  saturated take-up domains, county complements, crosswalk boundaries,
  overlapping producer fallbacks, zero/negative calibration targets,
  infeasible exact-k, clone tails, mixed-ownership columns.
- Joining the binding is an explicit `identity_generation` bump —
  operationally a cold-cache cutover; pre-spec artifacts stay
  generation 0, readable, never promotable, never retro-labeled.

## Gates and calibration: executable, not named

Every gate specifies: exact metric formula, input artifact + stage,
population + denominator, slices, reference release/data digest,
minimum support / effective sample size, absolute + relative
thresholds, uncertainty or multi-seed rule, missing-slice treatment,
fail/warn/report-only status, and typed failure-reason mapping.
`report_all_never_widen` names a *policy over that schema*, not a
metric. The calibration spec exposes the full mathematical contract:
target scaling, row-weight construction, cap behavior, zero/negative
target treatment, objective aggregation, mass constraints,
initialization/warm-start, optimizer + schedule + dtype, stopping
conditions, infeasibility policy, target-priority policy, attainment
verdict. Exact-k states post-selection weight semantics explicitly:
selected records carry their calibrated weights unchanged; non-selected
records are absent; no re-normalization unless declared. Gates evaluate
the **final selected artifact**, not only the pool.

## Publication: attempt events, atomic promotion, two graphs

- **Attempts are append-only events ending in one immutable terminal
  seal** (`landed | failed | expired`); a running attempt is an event
  stream, never a "sealed" status.
- Two-phase publication defines: temporary artifact namespace → atomic
  manifest seal → output-content verification → idempotency key →
  promotion transaction → recovery for seal-ok/append-fail and
  append-ok/alias-fail → orphan and expiry reconciliation.
- **The strict-linear logbook chain is the tamper-evident audit
  sequence; it is not the release topology.** A separate release
  relationship graph carries `derived_from`, `supersedes`, `revokes`;
  serial appends to the audit chain imply nothing about release
  parentage.
- `latest` promotion stays a human gate; eval artifacts never promote
  on a red battery.
- **D6 (release line rename)**: still open — the release line is marked
  **provisional/non-normative and excluded from `spec_sha256`** until
  Max rules, so golden bundles freeze without baking an unratified
  name. HF destination unchanged without an explicit ruling.

## File excerpts (all pseudocode until schemas + the full US bundle land)

### bundle.yaml — bundle-level settings only (no file inventory)

```yaml
country: us
identity_generation: 1
seed_protocol: legacy-v1
# The resource inventory lives in country_package.json {path,kind,schema_id}.
# Emitted: bundle.lock.json, plan.lock.json, engine_abi.lock.json.
```

### sources.yaml / spine.yaml

As in v2 (asec mass anchor; support roles with `puf_tax_detail`,
`clone_index: 1`), with paths/URIs strictly operational bindings.

### geography.yaml — block-first + ASEC complement (**phase F3**)

As in v2 (complement as sole rule, pinned identified-county source, no
sampled fallback) — with the round-2 correction that the held #696
branch does **not** implement this: it records ASEC county as absent
from the v3 checkpoint and falls back to state-wide draws, and its
loader rejects county fields. F3 here is a **coordinated kernel + ASEC
checkpoint-schema/source + block-artifact + bundle migration** with
complement/leakage tests and refusal on empty complements — the held
branch gets reworked, not merged as-is.

### imputation.yaml — chains without the 8; concept guards

As in v2 (declared order, `splits: declared_only`,
`release_after_draw`, keep-together pairs) with the corrected frozen
ledger (five/37; 19/70) declared explicitly at the flip and the
per-target `requires_concepts` blocks from the concept registry.

### take_up.yaml — see the orthogonal schema above; 13 committed rows.

### battery.yaml / calibration.yaml / selection.yaml

Per "Gates and calibration": each gate a full executable record; the
solver surface fully resolved; `k` a run-request knob with a declared
bundle default and precedence; no `device` in normative content.

### publication.yaml

Attempt-event model + promotion protocol + release relationship graph;
audit chain settings; release line marked provisional (D6).

## Rollout — honest phases

**P (now, before any freeze): the CHAMPVA lane lands first.** The held
predictor-sets branch is a behavior change (widened loaders, 15 carried
columns, new participation target order, alternatives-precedence and
registry-schema bumps), not a mirror edit. It lands under the existing
#695 mirror plus its **own OOS/statistical acceptance gate**, before any
baseline freezes. The #697 inventory fixture is then **regenerated and
re-certified against the post-predictor artifact profile** (the held
fixture is missing 56 of its columns — the v2 lane order was backwards).
The value fix does not wait for the spec engine at all.

**F0 — compiler front end, single-authored.** The full front end:
parsing/canonicalization/defaults/migration, cross-reference resolution,
typed entities/artifacts/scopes/columns, stage DAG + producer graph
compilation, seed-protocol resolution, complete normalization of every
domain file, the compile-to-legacy-payload adapter covering every
normative field, a usage/coverage report proving no field is ignored,
and round-trip + mutation tests. **The bundle is authored once; the
compiler generates the legacy payload the constants-era executor
consumes** (`config_authority=constants_adapter`). No manual
dual-editing, ever. `spec_sha256` is labeled a *mirror-attested
configuration identity* until F1. The CountrySpec seam lands here;
UK + BE bundles compile; **a minimal Belgian smoke build runs before
the identity-generation cutover**. No executor yet.

**F1 — drive.** Generic executor + brokers; producer-graph compile-back
byte-identical; bundle mode constructs the authorities; per-PR cold
dual-mode fixtures; the four-build restricted f004 certification,
flipping stage by stage; geography = exact legacy behavior. Derived
closure/segments/dashboard retarget to compiler outputs here (not the
held authored-class tests).

**F1 status (2026-08-20): PARTIAL; stopped at incomplete D4.** The generic
executor/brokers, bundle authority projections, sealed comparison contracts,
and compiler-derived closure/segments/dashboard exist, but the production pool
driver still dispatches physical stages through constants and does not invoke
the sealed artifact collector/comparator. Exact artifact-member and calibration
ownership also remain open. The cold D4 dual-mode gate, D5 stage receipts, and
D6 four-build/resume certification have therefore not run and are not claimed.

**F2 — delete, machine-decidably.** After ≥1 certified full release on
bundle mode, deletion requires generated inventories + zero-reference
tombstone gates: no `is`-guards, constant imports, or alternate
dispatch paths; no production reference to `CANONICAL_US_LATE_*`,
schedule/ownership receipt constructors, either greedy splitter, or the
max-width constant; no nonhistorical reference to the retired lineage
file/emitter/dashboard schema (external dashboard verified against the
compiled catalog); no direct take-up-contract/source-stage authority
outside the generation-0 reader; every reachable stochastic/hash-draw
callsite consuming a broker stream token; no duplicate engine / period /
geography-vintage / catalog-owner literals; one typed resource manifest
and no root drafting copies; a **node-invalidation matrix** proving an
unrelated bundle edit reuses unaffected nodes and invalidates
descendants of the changed node; schema-migration and old-bundle reader
fixtures; four-build within-mode + cross-mode determinism; fault
injection around checkpoint write / manifest write / seal / promotion /
logbook append; cross-profile byte-equivalence for every
claimed-invariant profile; RNG-broker and ambient-read enforcement
tests; a real Belgian build and a UK walking-skeleton **execution**;
and a **dated** authority-selector retention deadline. Generated locks
stay reproducible-or-rejected.

**F3 — intentional-change train.** Each lands as a bundle diff with
authority-version bumps, fresh node/bank identities, cold caches, and
statistical/OOS gates: block-first + ASEC complement (as the
coordinated migration above), the 37-target chain re-unification (full
scope), `derived-v2` seed streams, then the remaining calibration
stages.

## Migration map

The v2 table stands with these round-2 amendments: the late-DAG row
routes through the **lossless producer graph** as specified above; the
take-up row absorbs the contract **via the engine ABI lock**; catalogs
and vintages enter as split contract/derived/documentation and typed
references respectively; the seed row expands to the exhaustive
draw-site ledger; `us_imputation_lineage.yaml`'s row names its three
consumers; and the CountrySpec row is labeled replacement-through-seam.

## Decisions

| # | Decision | Round-2 status | v3 resolution |
|---|---|---|---|
| D1 | Bundle is build authority | both: direction stands | Via the explicit CountrySpec seam; single-authored from F0; constants become a generated adapter, then delete at F2 |
| D2 | Kernels are the escape hatch | both: amend | Executor + brokers + orthogonal capabilities + predicate-algebra scopes; no schema-level escape hatches anywhere |
| D3 | spec_sha256 joins identity | both: amend | As `run_provenance_identity` (with the restored triad + `identity_generation` on concrete fields); provenance never gates node reuse |
| D4 | Frozen-behavior gate | both: amend | Four cold builds; plan-derived vector incl. publication + banks; concrete resume predicate; adversarial fixtures |
| D5 | Seed streams | both: split stands | `legacy-v1` enforced by the RNG broker + draw-site ledger; `derived-v2` at F3 |
| D6 | Release line rename | Pro: don't freeze it unresolved | **RULED (Max, 2026-08-16): `microcosm-us-2024-*`** at the flip; the line becomes normative in publication.yaml at F0 (release regex + rung grammar; readers accept both prefixes; populace-* rows valid-historical); HF destination unchanged, separate explicit ruling |

## Review provenance

- Round 1: `_698-SOL-REVIEW.md`, Pro conversation (receipt in
  `_buildo-runtime/out/stacked-full-LAUNCH.md`).
- Round 2: `_698-SOL-REVIEW-R2.md` (verdict: request changes; 12-item
  v3 list — all folded), `_698-PRO-REVIEW-R2.md` (verdict: request
  changes; 11-item ranked v3 list — all folded; its "triad LOST"
  finding drove the identity restoration).
- Where the reviews pulled differently (four classes vs triad;
  whole-run vs node granularity), v3 adopts the composition stated in
  "Identity": triad + classes as the provenance vocabulary, node keys
  as the reuse mechanism, whole-run binding as the safe interim.
