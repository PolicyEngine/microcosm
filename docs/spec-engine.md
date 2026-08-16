# The spec engine: one declared bundle drives the build

Status: **v2 — revised after two independent reviews** (2026-08-16).
Nothing here is wired; `specs/us/*.yaml` in this branch are skeleton
drafts of the target shape.

Reviews folded in:

- **Sol** (code-grounded adversarial review, `_698-SOL-REVIEW.md`): 16
  MAJOR, 3 MINOR, 1 NIT. Every claim spot-checked against source held
  (7/7), including the existence of a live packaged `CountrySpec` system
  with US/UK/**Belgian** country packages that v1 missed entirely.
- **GPT-5.6 Pro** (design review, 15 findings): compiled plan lock,
  per-node execution keys, four-layer configuration separation, kernel
  capability typing, stateless RNG, catalogs, vintages, sealed attempts,
  machine-decidable verdicts, UK walking skeleton, and a phasing warning
  that value fixes must not queue behind infrastructure.
- **Max's rulings** (2026-08-16): chains never truncate opaquely (the
  `max_targets_per_fit: 8` split dies); take-up keys by program with
  column/entity derived; `column_lineage.yaml` dissolves into a compiler
  derivation; block-first geography with the ASEC identified-county
  complement.

v1 factual errors corrected in this revision: the production mass anchor
is **asec**, not acs (`stacked_spine.py` `BASE_ASEC_SUPPORT_CHANNEL`);
the PUF tax-detail support attachment is **clone 1** referenced by role,
not clone 0 (`support_provenance.py` `PUF_TAX_DETAIL_CLONE_INDEX = 1`);
and the repo already has a packaged country-spec system (`country_spec.py`,
`us|uk|be/country_package.json`) that this design must extend, not
duplicate.

## The ruling this implements

Max, 2026-08-15: *"It should be immediately obvious which predictors we're
using for each variable … this part should just be a yaml file. Also the
attributes of the ML model. … Let's do it right before making atomic
fixes. Scope out the right schema for everything first."*

Today the spec (#695/#697 line) **mirrors** code and CI enforces the
mirror. The spec engine **flips the authority**: the build loads the spec
bundle, constructs its plan objects from it, and the Python constants are
deleted. Custom logic survives as **named kernels** that spec entries
reference by id — code keeps the *how*, the spec owns the *what, from
what, with what*.

## What exists today (the inventory v1 skipped)

The flip does not start from zero. Five authority surfaces already exist,
and the design below gives each exactly one disposition:

1. **`CountrySpec`** (`country_spec.py`, ~40KB): a packaged spec-only
   country system. It hashes every declared resource, rejects undeclared
   or missing files, type-validates its resource kinds, and compiles the
   source/geography plan with a no-fallback posture.
   `CountrySpec.fingerprint` hashes the composition of raw resource
   bytes. `us/`, `uk/`, and `be/country_package.json` are live; Belgium
   is the first full consumer (microcosm#261). Its manifests carry a
   `schema_version` the loader does not yet read. **Disposition: the
   bundle becomes new resource kinds inside this system** — see "One
   spec system" below.
2. **`specs/us_imputation_lineage.yaml`** (#695): the mirror-mode lineage
   spec, consumed by a conformance test and the dashboard. **Disposition:
   absorbed by `imputation.yaml`; the file and its emitter retire at the
   flip** — no third surface survives.
3. **The take-up contract** (`us/take_up_contract.json` +
   `source_stages.json` rows): a curated authority over engine facts,
   treatments, rates, calibration targets, and scope owners.
   **Disposition: absorbed into `take_up.yaml` as first-class bundle
   content** — not referenced as an external "parameter source" (that
   would leave two authorities).
4. **Python constants** across `us_runtime/` and the stacked tool: gap
   plans, predictor lists, schedules, registries, tail contracts, seeds,
   solver defaults. **Disposition: bundle-normative, per the migration
   map**, with the deletion checklist gating removal.
5. **`builder_code_identity`** (`code_identity.py`): hashes packaged
   sources + numeric dependency versions. US PUF support and the UK
   builder use it; **the f004 stacked identity does not reference it at
   all** (verified: zero references in `build_us_multispine_pool.py`).
   **Disposition: joins stacked identity as the code-identity class** —
   see "Identity" below.

## Principles

1. **Total, or failing.** The bundle must account for every column of the
   artifact (the #697 closure survives the flip as a *derived* check —
   see "Lineage and closure"). An unclaimed column, an unreferenced
   kernel in the selected country/profile namespace, or a dangling
   cross-reference fails the load, not a review. Closure scope is the
   selected country + profile registry namespace: unknown referenced ids
   and duplicate ids always fail; unused library-only implementations do
   not.
2. **Declarations carry mechanisms, not just names.** Every stochastic
   write declares its draw universe, conditioning, parameter source, and
   seed stream. Stage topology alone is what let the geography flaw hide.
3. **Identity is four orthogonal classes, jointly required.** One YAML
   hash is not enough (a bundle hash cannot see a kernel-code change).
   See "Identity".
4. **Kernels are the only escape hatch — behind a generic executor.** A
   kernel is a registered callable with a declared io signature, a params
   schema, a capability type, and an implementation version. The registry
   refuses unknown ids. The *executor* enforces the discipline the
   declaration claims: projected inputs, patch-style outputs, full-frame
   diff. A name lookup plus today's specialized guards is not enough
   (today's `ProducerContract` is metadata-only and the dispatcher passes
   the whole frame).
5. **Country files instantiate a shared core plus declared extensions.**
   A small country-neutral core (source binding, typed stage DAG, kernel
   contracts, lineage/identity, artifact profile) plus discriminated
   country extensions (support kinds, geography kinds, entity/channel
   ids, take-up/publication extensions). "Same schema" never means the
   US file shapes verbatim — the UK's two-stage constituency→output-area
   geography, person/benunit/household entities, and synthetic-prior FRS
   support must be *expressible*, and the Belgian package's existing
   semantics must compile as a compatibility bundle, before the schema
   is signed off.
6. **The dashboard reads the same bundle the build reads.** Drift between
   page and build becomes impossible by construction, ending the
   emitter-regeneration step.
7. **Derive, don't declare.** A field that is a deterministic function of
   another authority — engine variable metadata, a naming rule, the block
   geoid, a contract row — is derived by the loader/compiler and
   asserted, never independently declared. Every independent second copy
   is a future drift. (Take-up column names and entities derive from the
   program + engine metadata; county/tract/PUMA derive from the block;
   whole-column closure derives from declared outputs. The assertion
   failing is a load refusal.)

## One spec system (sol MAJOR 10)

The bundle is **an extension of `CountrySpec`, not a parallel loader**:

- Bundle files live as **package data** under the country package
  (`microcosm/build/us/…`), declared in `country_package.json` as new
  resource kinds, found via the same `importlib.resources` lookup. Root
  `specs/us/*.yaml` in this branch is a drafting location only; it moves
  into the package at implementation so a built wheel carries it (root
  `specs/` is not under the wheel path today).
- One loader, one composition binding. The transition from raw-byte
  `CountrySpec.fingerprint` to canonical `spec_sha256` is an explicit
  identity-generation boundary — they are different functions and are
  never aliased.
- The inert package `schema_version` becomes read, validated, and
  versioned.
- `us_imputation_lineage.yaml` + emitter retire at the flip (surface 2
  above).
- The Belgian package's existing source/geography/target/gate/release
  resources compile as a **compatibility bundle** in rollout step F0 —
  Belgium is a conformance obligation, not future work.
- A clean-wheel test (build wheels, install into a fresh venv, load the
  bundle) gates the loader PR.

## Canonicalization contract (sol MAJOR 5)

`spec_sha256` is computed over a **typed resolved bundle**, never raw
concatenation:

1. Parse one YAML 1.2 document per file; reject duplicate keys, merge
   keys, custom tags, non-string keys, implicit timestamps, non-finite
   numbers.
2. Validate against closed-world JSON Schemas
   (`additionalProperties: false`); inject every schema default into one
   typed `ResolvedSpec`; reject unknown fields before hashing.
3. Hash a domain-separated envelope: canonicalizer id/version, schema
   id/version, country, and a map from normalized POSIX-relative file
   names to typed values.
4. Sort object keys; normalize tuple/list to arrays; normalize numbers to
   their schema-declared integer/float type; validate lowercase sha256
   and canonical ids; require NFC identifiers.
5. **Ordered arrays stay ordered** for stages, channels, directions,
   families, predictors, targets, fallbacks, absence rules — order is
   behavior-load-bearing (QRF consumes a shared RNG in target order;
   clone validation requires channel-major order). Only fields declared
   as mathematical sets may sort/deduplicate.
6. Status/notes/documentation fields are declared normative or
   non-normative per schema; authority payloads stay structured objects.

Golden canonical bytes/hashes, invalid fixtures, and a full
legacy-equivalent US bundle are committed **before schema sign-off**.
During the equivalence window the existing receipt constructors and
serializers are reused exactly (the stacked authority serializer and the
tool's identity serializer differ on `ensure_ascii` — unifying them is a
later, explicit identity-format change).

## Identity: four classes, node keys, and the compiled plan

Merging sol MAJOR 12 with Pro's versioning triad and plan lock:

1. **Grammar**: `schema_id`/`schema_version` + `canonicalizer_version` —
   what syntax is accepted and how it resolves. Unsupported versions fail
   before execution.
2. **Configuration**: `spec_sha256` — the exact fully resolved normative
   configuration. Any semantic config/default/order/pin edit changes it.
3. **Code**: each kernel/authority exposes an implementation/contract
   version, aggregated as a `kernel_set_sha256` (id, implementation
   version, params schema, io schema) **backed by a code/dependency
   digest** — `builder_code_identity` finally joins stacked identity
   here, closing the "unchanged YAML resumes changed code" hole.
4. **Serialization**: artifact/checkpoint schema and materializer
   versions stay code-owned (checkpoint schema v1/materializer v7,
   stacked materializer v11, frame checkpoint v3, outer context v2, …)
   and describe representation compatibility only.

**Resume requires equality of all four; none masks another.** A schema
migration bumps class 1 and normally class 2 even when behavior is
provably unchanged; a config-only edit changes class 2 without touching
class 3; a kernel edit bumps class 3 with unchanged YAML.

The loader **compiles** the resolved bundle into a canonical execution
plan and emits `plan.lock.json` (the typed IR: stage DAG, per-node
resolved params, kernel pins) plus `bundle.lock.json` (file hashes,
schema/canonicalizer versions). Each executable node gets a **node key**
`H(canonicalizer, resolved node plan, input content hashes, kernel
digest, runtime lock, execution profile class, seed stream id, output
contract)` — the checkpoint/bank resume key at node granularity. The
compiled artifacts are emitted and receipted, never hand-authored.

Identity binding is **namespaced**: a
`spec_binding: {country, schema_id, schema_version, canonicalizer_version,
spec_sha256}` object joins `_configured_stacked_identity` and
`_stacked_checkpoint_base_identity` (and flows into QRF/transfer bank
bindings); it is recorded in outer-stage `run_config`, not in structural
`FrameIdentity`. The existing nested authority receipts are preserved
through the equivalence window. A top-level `schema_version` field never
collides with the checkpoint-envelope field of the same name because the
binding is namespaced.

**Singleton canonicality dies.** Today production recognizes the
authority by Python object identity (`is` against module constants), so
any loader-built authority is rejected as non-canonical. That guard is
replaced by loader provenance + live component digests + semantic
authority version + the spec binding.

## Configuration surfaces (sol MAJOR 14 × Pro's four layers)

Every input to a build belongs to exactly one surface:

- **Normative (hashed into `spec_sha256`)**: logical source ids, roles,
  content pins, loader kernel ids, semantic options, stage topology,
  model params, seed *protocol* and stream map.
- **Run request (joins identity, not the bundle)**: rung/sample fraction,
  root seed value (under `derived-v2`), k, release label. These are the
  per-run knobs; each is identity-bound exactly as today.
- **Execution profile (receipted, asserted output-invariant)**: worker
  counts, device, batch sizing that provably cannot change bytes; any
  knob that *can* change bytes is reclassified normative or code.
- **Operational bindings (receipted, never hashed)**: source id → local
  path/URI mapping (the launcher supplies a bijection; the loader
  verifies pinned hash/size and never hashes the host location),
  checkpoint/output/spool roots, credentials. Host paths and secrets
  never enter identity.
- **External chain state**: the logbook predecessor digest, bound by the
  chain protocol (strict-linear, off-chain probes), not by the spec hash.
  A failed spec load cannot carry `spec_sha256` — failure rows carry
  `spec_binding_status` + attempted schema/canonicalizer + (when bytes
  were readable) a separate raw file-set digest that is **never** called
  `spec_sha256` (sol MINOR 3).

## Seeds: `legacy-v1` now, `derived-v2` after (sol MAJOR 3 × Pro's RNG)

Today's seed reality: shared literal 578 for ASEC/ACS sampling and clone
attachment, literal 0 for pool/QRF/source/take-up (bound into identity),
literal 42 for PUF aggregate allocation, plus an existing ACS-transfer
derivation protocol (SHA-256 over NUL-separated labels, first four bytes
little-endian) and QRF's `SeedSequence(seed).spawn(2)` consumed in target
order. A generic `hash(root_seed, stage, family)` cannot reproduce any of
that — so it can never pass an equivalence gate.

- **`seed_protocol: legacy-v1`** ships with the flip: a versioned,
  explicit named-stream map that pins every literal and every existing
  substream algorithm exactly as they are. The resolved map is hashed and
  exposed on /lineage per family. Behavior change: zero.
- **`seed_protocol: derived-v2`** is a later, intentionally
  behavior-changing bundle edit: stateless counter-based streams keyed by
  `(node, column, entity, draw index)` — order-independent and
  parallel-safe, with domain separation, length-prefixed UTF-8 labels,
  declared digest/width/endianness, and golden vectors. The root seed
  value moves to the run request. It lands with fresh identity
  namespaces, cold caches, and statistical/OOS gates — never a byte
  equality claim.

## Kernels: capability typing + the generic executor (sol MAJOR 6 × Pro)

Registry records bind: callable id, implementation + contract version,
params schema, io schema, supported spec range, and a **capability
type** — determinism class (pure / seeded-stochastic / io), grain
(cell / column / entity / frame), write mode (`fill_missing`,
`overwrite_scope`, `assert_equal_noop`, `structural_effect`), and
side-effect declaration.

Every build kernel runs through **one generic executor**:

- receives an immutable, schema-aware projection containing only declared
  physical and virtual resources (never the whole frame);
- returns a patch/output object, never an arbitrary replacement frame;
- the executor snapshots and diffs tables, links, ids/order, weights,
  strata, metadata, and mass history, and rejects any change outside the
  declared entity/column/**row** scopes.

Today's specialized guards (education/retirement/transfer overlap, tail
preservation, the QBI diff guard) are retained as extra invariants;
adversarial tests cover undeclared reads, undeclared writes, and
structural mutation. This executor is what makes "kernels are the only
escape hatch" true rather than nominal.

**Producer graph (sol MAJOR 7).** The late-producer registry migrates
losslessly: a `producer_graph` schema representing every current
`ProducerInput`/`ProducerOutput` field — entity/column, value-kind,
required scope, producing stage, alternative physical column sets,
tolerated-absence receipt ids, coverage scope, final owner, non-owner
actions. Order/waves derive from the graph. Before the flip, the compiled
bundle must reproduce the current schedule/ownership payload
byte-identically.

**No library-default dual authority (sol MAJOR 13).** Registered
production adapters accept a fully materialized, schema-complete config
and pass every build-facing parameter explicitly — today
`QRF(n_estimators=…, seed=…)` at both production call sites silently
inherits `zero_atol`/`max_samples_leaf` from library defaults, so a
library release could change artifacts without changing `spec_sha256`.
Standalone `microcosm.fit` users keep convenience defaults. Two tests
enforce the boundary: monkeypatch library defaults and prove the resolved
plan/output unchanged; statically assert every registered invocation
supplies its declared parameter set.

## Lineage and closure (sol MAJOR 8 + Max's dissolution ruling)

`column_lineage.yaml` **is not an authored file** (Max, 2026-08-16):
whole-column closure derives from each bundle file's declared outputs —
imputed ← imputation targets, computed ← producer outputs,
assigned_by_ladder ← geography derive list, take_up_draw ← programs +
naming rule, measured_native ← loader output contracts, receipt ← kernel
receipt outputs, structural ← entity schema. What stays authored: the
observed inventory fixture (cross-check), time-limited waivers, and the
human-facing column catalog. A closure failure names the producer that
failed to declare its output.

Column-level classes alone cannot police mixed ownership — ACS transfer
fills only null cells and preserves non-null measured values; generic
take-up does the same — so one physical column legitimately holds
measured-native and imputed cells under different owners, and presence
can be conditional (`tract_geoid` only when `assign_tract=True`).
Therefore:

- The bundle identity-binds a **closed artifact profile**; the expected
  column set resolves at load. Every output is `required`, `forbidden`,
  or guarded by a closed spec predicate.
- Runtime closure compares expected↔actual **in both directions** at
  each checkpoint and the final artifact. Conditional skips write a
  canonical skip receipt; required outputs materialize with canonical
  dtype even for zero-row scopes.
- Below the whole-column class (kept for documentation), lineage is
  exhaustive, non-overlapping **segments over (entity, column, row_scope,
  stage, write_policy)** — cell-scope ownership, matching how the code
  actually writes.

**Catalogs (Pro).** The bundle carries entity/artifact/column catalogs —
one human-facing row per column (description, units, owner, lineage
class) — and closure reporting derives from them. This is the dashboard's
direct data source.

**Vintages (Pro).** `vintages.yaml` pins period semantics: survey years,
tax year, geography vintages (2020 blocks, 119th CDs), engine version —
one place where "2024" is defined.

## File excerpts (corrected; representative, schema-conforming)

Placeholders from v1 are gone or explicitly marked as pseudocode outside
the YAML (sol NIT 1).

### sources.yaml — normative pins only

```yaml
sources:
  - id: asec_raw_stage_v3
    role: spine_channel_source          # asec channel
    sha256: 51e9fafcd6f16140018fa90c7afbeb6d79008bfc8c122e437d23a399b30553fe
    loader: kernel:load_asec_raw_stage_checkpoint
  - id: acs_2024_person_zip
    role: spine_channel_source
    sha256: afdc6d90c6e2f0bab365ed32d95ba4c4d8ac651162f46ac7861295b2dc469894
    loader: kernel:acs_pums_parse       # canonicalizing boundary (#672)
engine:
  policyengine_us: "1.764.6"            # pinned simulation surface
# Local paths/URIs are OPERATIONAL BINDINGS supplied at launch (id → path
# bijection, hash-verified) — they never appear in this file.
```

### spine.yaml — channels and assembly

```yaml
channels:                     # ORDERED — channel-major order is structural
  - id: asec
    source: asec_raw_stage_v3
    observed_geography: state           # county where identified (F3)
  - id: acs
    source: [acs_2024_household_zip, acs_2024_person_zip]
    observed_geography: puma
assembly:
  mass_anchor_channel: asec             # production default (v1 wrongly said acs)
  shared_dtype_policy: canonical_string_storage   # #672
support_roles:
  - id: puf_tax_detail                  # attachment by ROLE, not raw integer
    kind: puf_attachment                # UK: synthetic_prior_replacement; or none
    clone_index: 1                      # today's value, stated not implied
    tail_support: {strata: filing_status, policy: declared_min_support_with_receipt}
```

### geography.yaml — block-first (**phase F3**, not the flip)

```yaml
# PHASE F3 (intentional behavior change). The equivalence flip encodes
# LEGACY geography exactly — current PUMA-ladder semantics, observed-PUMA
# preservation, CD-then-block draw — because no existing kernel performs
# what this file declares (block ladder lacks PUMA; PUMA ladder never
# writes block). This file is the F3 target state, landing with a new
# versioned block artifact (exact 2020-PUMA relationship), fresh identity,
# cold caches, and statistical gates.
assignment:
  anchor: census_block_2020
  order: before_gap_fill                # so PUMA/county serve as predictors
  kernels:
    assign: kernel:block_first_assignment
    validate: kernel:block_geography_validation
  draw:
    acs:
      universe: observed_puma
      weight: block_population_2020
    asec:
      identified:
        universe: identified_county
        weight: block_population_2020
      unidentified:
        universe: state_minus_identified_counties   # the complement ruling —
        weight: block_population_2020               # never state-overall
  identified_county_source: source:census_cps_identified_counties  # pinned id
  derive: [tract_geoid, county_fips, puma_2020, congressional_district_geoid,
           place_fips, sldu, sldl, cbsa_code]       # all from block
  assertions:
    - observed_acs_state_and_puma_preserved
    - tract_to_puma_exact
  ladder_source: us_block_ladder_2020
```

### imputation.yaml — families × predictors × models × chains

```yaml
predictor_blocks:
  core_demographics: {columns: [age, is_female, state_fips], tags: [required]}
  income_components: {columns: [employment_income, self_employment_income,
    social_security_income, retirement_income, interest_dividend_rental_income],
    availability: observed}
  household_structure: {columns: [household_size, marital_status,
    relationship_to_head, is_household_head, tenure_code]}
  veteran_va: {columns: [is_veteran, receives_va_payments], tags: [eligibility]}
  disability: {columns: [has_hearing_difficulty, has_vision_difficulty],
    tags: [eligibility]}
  own_coverage: {columns: [acs_hins_va, acs_hins_medicaid], tags: [eligibility]}
  geography: {columns: [puma, county_fips]}

models:
  regime_gated_qrf:
    kernel: kernel:regime_gated_qrf
    params: {n_estimators: 100, max_samples_leaf: null, zero_atol: 1.0e-06}
    # FULLY materialized — production adapters pass every param explicitly;
    # library defaults never reach a build (sol MAJOR 13).

chaining:
  order: declared              # the family's target list IS the chain order
  splits: declared_only        # a chain break exists ONLY as an explicit
                               # `split_after:` entry with a reason — the
                               # greedy max_targets_per_fit=8 truncation is
                               # DEAD (Max: "we should never drop columns as
                               # predictors in such an opaque way").
  memory_policy: release_after_draw   # banked target-at-a-time; models
                                      # released post-draw, so width costs
                                      # bank size, not peak fit memory
  keep_together:
    - [ssn_card_type, immigration_status_str]
  # NOTE: re-unifying today's puf_tax_itemization__batch_1..4 into one
  # declared 32-target chain CHANGES RNG consumption (QRF consumes a shared
  # stream in target order) → it is an F3 behavior edit with statistical
  # gates, not part of the equivalence flip, which pins the frozen batch
  # structure as explicit declared splits.

families:
  - id: gap_fill/asec_survey_to_acs/person/benefit_participation
    stage: early_gap_fill
    donor: {channel: asec}
    recipient: {channel: acs}
    model: regime_gated_qrf
    predictors: [core_demographics, income_components, household_structure,
                 veteran_va, own_coverage, geography]
    required_concepts: [eligibility]    # per-target predictor-concept gate
                                        # (D4 as amended): a participation
                                        # target whose predictor set carries
                                        # no eligibility-tagged block fails
                                        # the LOAD — the CHAMPVA defect
                                        # becomes structurally impossible
    targets: [has_champva_health_coverage_at_interview,
              has_tricare_health_coverage_at_interview]
  - id: late_transfer/person/adult_care
    stage: late_transfer
    donor: {support_role: puf_tax_detail}   # role ref, not a raw clone int
    model: regime_gated_qrf
    predictors: [puf_tax_detail]
    targets: [is_incapable_of_self_care, pre_subsidy_care_expenses]

producer_graph:               # lossless late-DAG representation (sol MAJOR 7)
  - id: acs_earnings_universe
    kernel: kernel:acs_earnings_universe
    inputs:
      - {entity: person, column: employment_income, value_kind: amount,
         required_scope: acs_rows, alternatives: [], absence: fatal}
    outputs:
      - {entity: person, column: earnings_universe_flag, coverage: acs_rows,
         final_owner: true}
```

### take_up.yaml — discriminated programs, honest mechanisms

The v1 shape (one per-flag draw with a contract "parameter source") was
factually false for SNAP and structurally false in general (sol MAJOR 9):
the real system is a curated contract + per-program source stages, where
SNAP is `out_of_scope` for generic seeding because a national prior stage
is followed by a dedicated **state anchored count-calibration** stage
that overwrites the flag. v2 absorbs the contract and declares
**discriminated treatments with ordered pipelines**:

```yaml
naming_rule: takes_up_{program}{treatment_suffix}
# column + entity are DERIVED (naming rule + engine metadata) and asserted.
programs:
  - id: snap
    treatment: anchored_count_calibrated
    pipeline:                            # ORDERED; final_owner overwrites
      - stage: national_prior
        kernel: kernel:take_up_seed_national
        rate_source: source:take_up_reported_anchor
      - stage: state_count_calibration
        kernel: kernel:snap_state_take_up
        anchor: source:fns_state_participation_counts
        domain: eligible_only            # calibration domain ≠ assignment universe
        assignment: unmasked_stable_source_id
        prior: runtime_target_over_weighted_modeled_eligibles
        saturation: cap_at_universe
        final_owner: true
  - id: wic
    treatment: batched_seeded            # generic seeder, one kernel, many flags
    rate_source: contract:take_up
  - id: social_security
    treatment: measured                  # ASEC-native; no draw, no kernel
# Treatment enum (closed): measured | imputed_transferred | seeded_rate |
# batched_seeded | anchored_count_calibrated | engine_default_with_debt |
# near_universal | dedicated_stage. Output ownership is separate from kernel
# invocation: one kernel may own many flags; one flag may have multiple
# stages with exactly one final owner. Coverage is asserted against
# installed engine metadata exactly as today.
```

### battery.yaml / calibration.yaml / selection.yaml

```yaml
# battery.yaml
completeness: {targets: 131, source: contract:us_pool_inputs}
by_origin:
  legs: [incidence_ratio, quantile_envelope, boolean_liveness]
  verdict_policy: report_all_never_widen
verdicts: machine_decidable              # every gate: pass|fail + typed reason;
                                         # prose is commentary, never the verdict
# calibration.yaml — fully resolved solver surface (no library defaults)
solver:
  kernel: kernel:calibrate_prox
  loss: capped_relative_error
  max_weight_ratio: 7.0
  l0: {mode: exact_k_projection}
  warm_start: declared
  device: auto                           # execution profile, asserted
                                         # output-invariant per backend contract
targets:
  source: chronicle_facts                # sum-only doctrine
  geography_layers: [national, state, congressional_district, county]
  cd_policy: always_present_report_attainment   # Max 8/15 ruling
# selection.yaml — knobs only; validity gates are KERNEL CONTRACT, not config
exact_k:
  kernel: kernel:sampford_exact_k        # contract: cardinality gate (len==k
                                         # never clamped, unique, integer,
                                         # in-pool, sorted; real exceptions),
                                         # pi ∈ [0,1] finite, PCG64 isolated,
                                         # ulp-corrected inclusion probs,
                                         # 6-scalar receipt
  k: 50000
  pi_hi: 1.0
  group_ids: none
  on_infeasible: refuse
# Gates evaluate the FINAL selected artifact, not only the pool (Pro):
# battery + calibration attainment re-verify post-selection.
```

### publication.yaml — sealed attempts, release DAG

```yaml
release:
  # D6 (open, Max): the flip is the natural rename boundary — spec-engine-era
  # artifacts adopt the microcosm-* line; populace-* rows stay valid-historical.
  # HF destination is a separate named decision (never changed implicitly).
  line: microcosm-us-2024
  pattern: "{line}-stacked-f{rung}-s{seed}"
  rungs: [f001, f004, f010, f025, f100]
attempts: sealed                # every build attempt gets a sealed record
                                # (running|landed|failed|expired); publication
                                # is two-phase: seal, then promote
promotion:
  latest_flip: human_gate       # unchanged doctrine; eval artifacts never
                                # promote on a red battery
logbook:
  chain: strict_linear
  store: supabase:logbook
  release_dag: chain_protocol   # predecessor binding lives in the chain
                                # protocol, not the spec hash
distribution: hf:policyengine/populace-us   # UNCHANGED until explicit ruling
```

## The equivalence gate, as amended (sol MAJOR 2)

v1's "identical checkpoint identities and gate outputs" is not a behavior
proof — identity hashes configuration and structure, not transferred
cells, and a shared checkpoint root lets the second run *resume* the
first and "prove" equality tautologically. The real gate:

- **Both modes load, validate, and hash the same committed bundle before
  the authority implementation is selected** — `--config-authority=
  {bundle,constants}` (renamed from `--legacy-constants`; interaction
  with `--legacy-two-spine` explicitly rejected). The constants adapter
  asserts its resolved payload equals the legacy payload byte-for-byte
  and differs only in plan-object construction.
- **Cold, isolated, resume-forbidden**: both runs start in separate empty
  checkpoint roots at the same commit/dependency/input pins; the proof
  fails if either reports a checkpoint or model-bank resume.
- **Content equality, not identity equality**: compare SHA-256 of the
  three deterministic stage checkpoint H5 files (the serializer already
  makes equivalent frames byte-identical). Where operational metadata
  must differ, compare an exhaustive canonical frame digest (every table
  value, column order/dtype/index, links, weights, strata, metadata, mass
  records, stage/input receipts) instead.
- **Normalized gate payloads**: canonical terminal gate output with an
  enumerated exclusion list (timestamps, paths, run ids, authority-mode
  receipts). Release envelopes contain nonces and are compared only in
  normalized form.
- Per-table/per-column diagnostics on mismatch. Fixture-scale versions
  run per-PR on every change to bundle, loader, registry, identity, or
  kernels — parameterized over **both** authorities; the full cold f004
  proof runs in the restricted certification lane.
- Adding the spec binding to identity is operationally a **cold-cache
  cutover** even though semantically unchanged: it lands as an explicit
  `identity_generation` bump. Pre-spec artifacts keep generation 0,
  readable forever, **never retro-labeled** with a hash they didn't bind;
  release promotion after the cutoff requires the new binding.

## Migration map (v1 rows + sol MAJOR 11's missing classes)

| Today (constants / lanes) | Disposition |
|---|---|
| `INPUT_SHAS`, engine pin | `sources.yaml` (normative); paths → operational bindings |
| `_STACKED_SAMPLE_RUNG_TOKENS`, release regex, logbook rungs | `publication.yaml` |
| `stacked_gap_fill_plan()` literals | `imputation.yaml` families |
| `ACS_*_TRANSFER_PREDICTORS`, `PUF_TAX_DETAIL_DEFAULT_PREDICTORS` | `predictor_blocks` |
| predictor-sets lane (per-family blocks + widened loader) | `imputation.yaml` + `sources.yaml` loader columns |
| `CANONICAL_US_LATE_*` registry/groups/schedule + tolerated absences + alternatives + ownership | `producer_graph` (lossless; compile-back byte-identical) |
| #696 lane (block-first) | `geography.yaml` — **F3**, after legacy geography is encoded + equivalence-gated |
| #697 lane (classes, inventory, closure) | derived closure + inventory fixture + waivers + catalogs (no authored lineage file) |
| take-up modules + `take_up_contract.json` + `source_stages.json` rows | `take_up.yaml` programs (contract absorbed, not referenced) |
| battery/gate constants | `battery.yaml` |
| calibrate solver args + pruning/backend cutoffs, exact-k args | `calibration.yaml`, `selection.yaml` (fully resolved) |
| model defaults reaching builds via omitted kwargs | forbidden: adapters pass full config; `microcosm.fit` defaults are library-only |
| pipeline/runtime: mass shares, operator + stage order, seed, period, model sizes, sim batch size | bundle-normative; serialization/materializer versions stay code-owned |
| PUF support/clone/tail: clone role + index, tail thresholds/quantiles/topcode/filing statuses/AGI proxies, aggregate seed/RECIDs/bounds, SOI bands, primary QRF target order + absence doctrine | bundle resources/params for normative choices; role names not raw ints; explicit asset pins; envelope versions code-owned |
| resume/operations: checkpoint-root + bank layout, exact-identity discovery, logbook env fallback, spool/receipt roots, fit worker env/interpreter/CPU | operational surface, receipted; never normative |
| `us_imputation_lineage.yaml` + emitter + mirror test | retired at the flip |
| `CountrySpec` resource kinds + fingerprints (US/UK/BE) | extended in place; fingerprint→spec_sha256 = explicit generation boundary |

An audited inventory generated from the current full base identity,
authority receipts, packaged resource loads, runtime signatures, env
reads, and stage manifests assigns **every** item exactly one owner:
bundle-normative, kernel ABI/implementation, artifact
schema/materializer, operational receipt, or external chain state. A
reviewed static allowlist/denylist test stands until deletion.

## Decisions (as amended by both reviews)

| # | Decision | Sol | Pro | v2 resolution |
|---|---|---|---|---|
| D1 | Bundle is build authority | amend | accept+lock | Stands, as a `CountrySpec` extension with compiled `plan.lock.json`; kernels/artifact contracts stay code-owned and identity-bound |
| D2 | Registered kernels are the escape hatch | amend | capability typing | Stands with the generic executor (projection/patch/diff), capability types, and the lossless producer graph |
| D3 | `spec_sha256` joins identity/logbook | amend | four-layer split | Stands as a namespaced `spec_binding`, one of four identity classes; both modes load the same bundle; failure rows carry `spec_binding_status`; explicit `identity_generation` cutover |
| D4 | Frozen-behavior equivalence gate | does not stand as written | 4 cold builds | Rebuilt: cold isolated resume-forbidden runs, content digests (H5 SHA / exhaustive frame digest), normalized gates, per-PR fixtures over both authorities |
| D5 | One root seed, derived streams | does not stand as written | stateless RNG | Split: `legacy-v1` named streams for the flip (zero change); `derived-v2` stateless counters as an F3 bundle edit with statistical gates |
| D6 | `microcosm-us-2024-*` line at the flip | — | — | **Open — Max.** populace-* stays valid-historical; HF destination unchanged without explicit ruling |

## Rollout: F0 → F3, with the value-fix fast path

Pro's sharpest process point, and Max's standing concern: **CHAMPVA-class
fixes must not queue behind infrastructure.** The fast path is the mirror
discipline — after F0, behavior edits land as constants+bundle edited
together, with CI enforcing byte-equality of the resolved payloads, so
value work proceeds while F1 is still building.

- **F0 — Bind (small, fast).** Schemas (closed-world) + canonicalizer +
  golden vectors + typed `ResolvedSpec`; the full legacy-equivalent US
  bundle generated from constants; minimal UK bundle + Belgian
  compatibility bundle **compile** (not run); `CountrySpec` extension +
  clean-wheel test; both authority paths load/hash the bundle; namespaced
  `spec_binding` joins identity as one intentional cold-cache
  `identity_generation` bump; `seed_protocol: legacy-v1` pinned; mirror
  test: resolved bundle ≡ constants payloads byte-for-byte.
  *After F0*: `spec_sha256` answers "what built this," /lineage and the
  dashboard read the bundle, and **eligibility-predictor fixes (CHAMPVA),
  per-family predictor sets, and calibration part-1 land as mirrored
  edits immediately** — reviewed as bundle diffs, executed by constants.
- **F1 — Drive.** Generic kernel executor + producer graph behind the
  legacy path (compile-back byte-identical before any data comparison);
  bundle mode constructs the authorities; per-PR cold fixture equivalence
  over both modes; restricted cold f004 content certification, flipping
  stage by stage. Geography here is **exact legacy behavior**.
- **F2 — Delete.** After at least one certified full release on bundle
  mode and the deletion checklist below: remove constants,
  `--config-authority=constants`, and interim mirror tests. Keep schema/
  canonicalizer golden tests, kernel-discipline tests, closure tests, and
  content-equivalence fixtures.
- **F3 — Intentional-change train.** Each lands as a bundle diff with
  authority/materializer bumps, fresh identity, cold caches, and
  f025/OOS or statistical gates — never an equivalence claim:
  block-first geography + ASEC complement (#696), chain re-unification
  (`puf_tax_itemization` 32-target chain; kill-the-8 semantics fully
  realized), `derived-v2` seed streams, then the calibration epic's
  remaining stages. UK/BE run **real** conformance bundles before country
  extensions expand.

The three wave-1 lane branches stay held: block-first lands at F3 as
bundle content; predictor-sets lands **right after F0** as a mirrored
edit (it is the CHAMPVA fix); closure's inventory fixture + tests land at
F0 with the derivation replacing the authored classes.

## Deletion checklist (constants may be removed only when all hold)

1. One packaged loader + one composition binding cover all current
   `CountrySpec`, root lineage, stage-manifest, contract, and resource
   authority.
2. Schemas, canonicalizer golden vectors, invalid fixtures, the full US
   resolved bundle, a minimal UK bundle, and the Belgian compatibility
   bundle pass from a clean wheel.
3. Constants and bundle compile to byte-identical plan, schedule,
   ownership, authority, and identity payloads; both modes load the same
   bundle.
4. Cold fixture dual-mode tests run on every relevant PR; restricted cold
   f004 certification has compared all stage content + canonical gates;
   at least one full release certified on bundle mode.
5. All production kernels run through the generic executor; no production
   path consumes library defaults or undeclared package assets, env
   values, stage strings, source pins, or params.
6. Checkpoint/logbook/manifest/dashboard surfaces carry the spec binding
   + kernel/authority + artifact/materializer identities.
7. Identity-generation retention, reader compatibility, and cleanup dates
   documented; no old artifact relabeled.
8. Only then delete — retaining schema golden tests, kernel-discipline
   tests, closure tests, and content-equivalence fixtures.

## Review provenance

- Sol's full review: `_698-SOL-REVIEW.md` (this branch);
  16 MAJOR / 3 MINOR / 1 NIT; verdict "request changes" — every
  amendment above that cites a MAJOR is sol's, folded after code
  verification (7/7 spot-checks held, including `country_spec.py`
  Belgium, `BASE_ASEC_SUPPORT_CHANNEL`, `PUF_TAX_DETAIL_CLONE_INDEX = 1`,
  `QRF(n_estimators=…, seed=…)` default leakage, and zero
  `code_identity` references in the stacked tool).
- GPT-5.6 Pro's review: ChatGPT conversation (receipt in
  `_buildo-runtime/out/stacked-full-LAUNCH.md`); its plan-lock, node-key,
  four-layer, capability-typing, stateless-RNG, catalogs, vintages,
  sealed-attempts, machine-verdict, and UK-walking-skeleton findings are
  adopted above; its phasing warning shaped F0's fast path.
- Where the two disagreed on granularity (whole-run four-class equality
  vs per-node keys), v2 adopts both: the classes define the vocabulary,
  node keys are the resume mechanism.
