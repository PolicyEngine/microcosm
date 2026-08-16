# Adversarial review of the spec-engine RFC (PR #698)

Reviewed locally at `952d5add` against `main` at `f7173120`. I treated
`packages/microcosm-build/src/microcosm/build/us_runtime/` as `us_runtime/`
below. This was a source-and-design review only; I did not run the restricted
build or use the network.

**Bottom line: request changes.** The authority flip is a sound direction, but
the RFC cannot preserve current behavior or current identity as written. The
two known ordering holes are real, the proposed equivalence assertion is not a
behavioral equivalence assertion, the generic kernel discipline does not yet
exist, and the draft would create a second spec system alongside the packaged
`CountrySpec` system already in this repository.

## MAJOR findings

### MAJOR 1 — Rollout steps 2 and 3 cannot both be true, and a loaded bundle cannot currently pass the production authority guard

- **Claim or gap:** Step 2 joins `spec_sha256` into checkpoint identity while
  both authorities are live; step 3 then requires the constants-built and
  bundle-built runs to have identical identities (`docs/spec-engine.md:244-258,
  283-290`). The RFC does not say that constants mode loads the same bundle
  before the execution paths branch. It also assumes a bundle can reconstruct
  today's authority objects.
- **Code evidence:** The stacked base identity is assembled in
  `tools/build_us_multispine_pool.py:1043-1147`; a stage identity is exactly
  that mapping plus `stage` and `stage_index` at `:1298-1308`, and canonical JSON
  is hashed at `:1311-1322`. Discovery requires exact identity equality at
  `:1181-1244`, while the configured identity determines the checkpoint
  namespace at `:1150-1178`. Separately, production authority is recognized by
  Python object identity: `_production_stacked_authority` compares module
  constants with `is` at `us_runtime/stacked_spine.py:3089-3147`,
  `_authority_receipt` sets `canonical_identity = authority is
  _canonical_authority` at `:3349-3376`, and production rejects false at
  `:3672-3695`. A content-equivalent object constructed by a loader is therefore
  non-canonical today.
- **Rollout failure:** If only bundle mode carries the new hash, the two logical
  identities differ and the gate fails before comparing behavior. If neither
  path carries it, step 2 has not happened. If the loader creates new plan
  objects, production fails even before that contradiction. Adding a top-level
  `schema_version` also collides with the existing checkpoint-envelope field of
  that name (`tools/build_us_multispine_pool.py:1068-1070`).
- **Concrete fix:** Load, strictly validate, resolve, canonicalize, and hash the
  same committed bundle **before** selecting an authority implementation. Put a
  namespaced object such as
  `spec_binding:{country,schema_id,schema_version,canonicalizer_version,spec_sha256}`
  into both `_configured_stacked_identity` and
  `_stacked_checkpoint_base_identity`; allow that base binding to flow into the
  QRF/transfer bank bindings. Record the binding in outer-stage `run_config`
  (`outer_stage_runtime.py:227-282,587-608`), not in structural `FrameIdentity`.
  `--config-authority=constants` should use an adapter from the same
  `ResolvedSpec`, assert that its resolved payload equals the legacy payload,
  and differ only in plan construction. Replace singleton canonicality with
  loader provenance plus live component digests, semantic authority version,
  and the spec binding. Preserve the existing nested authority receipts during
  the equivalence window; do not replace them with the one bundle hash.

### MAJOR 2 — “Identical checkpoint identities and gate outputs” is not a frozen-behavior proof and can become a resume tautology

- **Claim or gap:** The RFC calls identity equality plus gate-output equality a
  frozen-behavior proof (`docs/spec-engine.md:252-260`). Neither side of that
  conjunction is an exhaustive digest of stage output.
- **Code evidence:** For an f004 build, the current logical identity contains:

  - envelope fields: artifact kind, pool-checkpoint schema v1, stacked
    materializer v11, pipeline, period 2024, model seed 0, and the installed
    `policyengine-us` version (`tools/build_us_multispine_pool.py:1067-1078`);
  - the six verified roles `asec_raw_stage`, `acs_household`, `acs_person`,
    `acs_rent_donor`, `processed_puf`, and `puf_source_year`, each with actual
    SHA-256 and byte size (`:689-746,1079`, `:1011-1019`);
  - sampling fraction `0.04`, rung token, literal sample seed, the full stack
    manifest and its digest, clone fraction/seed, and the complete stacked
    authority receipt (`:1080-1093`); that authority receipt in turn carries
    authority id/version plus live/declared component digests and payloads for
    the gap plan, post-PUF transfer/producer surfaces, declared surface,
    metric/joint registries, support/tail contract, and late schedule
    (`us_runtime/stacked_spine.py:3317-3471`);
  - `pool_code`: operator/pre-clone/post-clone/derive order, gap-fill and late
    schedules, late resource semantics, remaining-stage input manifest, primary
    QRF target order/schema, ACS earnings and QBI contracts, the entire take-up
    contract identity, capital-gains tail schema/support contract, both estimator
    counts, max targets per fit, and simulation batch size (`:1094-1146`);
  - the cut point (`assembled`, `transferred`, or `simulated`) and its index
    (`:1298-1308`; stage order is `us_runtime/multispine_pool.py:200-201`).

  That is a strong configuration/authority identity, but it does not hash all
  transferred or simulated cells. `FrameIdentity` is also structural: it hashes
  entity/link schema, ordered IDs, memberships, clone/source provenance, and
  their dtypes (`outer_stage_runtime.py:611-651,821-852`); it omits ordinary
  columns, weights, strata values, frame metadata, and the mass log. The
  checkpoint writer separately stores frame metadata/receipts and computes the
  **file** SHA-256 (`tools/build_us_multispine_pool.py:1499-1553`). Its serializer
  deliberately makes equivalent frames byte-identical
  (`frame_checkpoint.py:1-7,91-150`). Finally, if the harness reuses the same
  caller-selected/default checkpoint root, equal configured identity routes
  both runs to the same store (`tools/build_us_multispine_pool.py:1169-1178,
  4200-4238`), and `load_deepest` can make the second run load the first run's
  result. Release IDs and terminal receipts contain timestamps/UUIDs/build IDs
  (`:1274-1295,3569-3570,4067-4086`), so whole publication envelopes are not
  naturally byte-equal either.
- **Rollout failure:** A changed imputed value can leave identity unchanged and
  can evade aggregate gates. Worse, the bundle run can resume the constants
  run and “prove” equality without executing its own path. Conversely,
  comparing unnormalized publication manifests yields false failures from
  nonce fields.
- **Concrete fix:** Run both modes cold at the same commit/dependency/input pins
  in separate, initially empty checkpoint roots, and fail the proof if either
  run reports a checkpoint or model-bank resume. Compare the SHA-256 of all
  three deterministic checkpoint H5 files. If any operational metadata must
  differ, instead compare an exhaustive canonical frame digest covering every
  table value, column order/dtype/index, link, weight, stratum, metadata, mass
  record, and canonical stage/input receipt. Also compare canonical terminal
  gate payloads, with only an enumerated set of timestamps, paths, run IDs, and
  authority-mode receipts excluded. Emit per-table/per-column diagnostics on a
  mismatch. Keep the two logical identities equal by recording authority mode
  only in a separate operational receipt.

### MAJOR 3 — D5 seed derivation is necessarily a behavior change and cannot participate in the equivalence flip

- **Claim or gap:** D5 replaces current seeds with
  `hash(root_seed, stage_id, family_id)` (`docs/spec-engine.md:244-250`), while
  the frozen gate requires unchanged output.
- **Code evidence:** CLI sampling and clone-attachment seeds default to 578
  (`tools/build_us_multispine_pool.py:496-517`). The same sample seed restarts
  the ASEC and ACS samplers (`us_runtime/stacked_spine.py:619-631`), which use
  `np.random.default_rng(seed)` (`frame_sampling.py:252-268`). Partial PUF clone
  attachment has another literal generator (`us_runtime/puf_support.py:710-742`).
  Pool imputation/source/take-up uses the separate constant seed 0
  (`us_runtime/multispine_pool.py:238-242,2970-2990`), and that 0 is bound into
  checkpoint identity (`tools/build_us_multispine_pool.py:1073,1110-1113`). ACS
  transfer already derives family/pattern seeds, but with a precise, different
  protocol: SHA-256 over NUL-separated labels, first four bytes little-endian
  (`us_runtime/acs_transfer.py:2902-2916`). QRF then uses
  `SeedSequence(seed).spawn(2)` and consumes one fit RNG in target order
  (`microcosm-fit/qrf.py:1077-1107,1128-1148,1333-1355,1428-1429`). PUF aggregate
  disaggregation uses literal seed 42 (`us_runtime/puf_source_agi.py:21-52,
  379-403`). The RFC's word “hash” specifies no digest, encoding, width,
  endianness, label grammar, or collision/domain-separation rule.
- **Rollout failure:** A root seed of 578 cannot reproduce the current shared
  `578`, `0`, and `42` regimes through the proposed generic rule. It also breaks
  today's deliberate ASEC/ACS stream sharing and changes QRF results because
  target order controls shared-RNG consumption. The equivalence gate must fail
  if D5 is actually wired.
- **Concrete fix:** The equivalence release needs a versioned
  `seed_protocol: legacy-v1` with explicit named streams: shared ASEC/ACS survey
  sampling `578`, clone attachment `578`, pool/QRF/source/take-up `0`, PUF
  aggregate allocation `42`, legacy geography seeds, and the existing ACS/QRF
  substream algorithms. Hash and expose that resolved map in lineage. Only
  after the authority flip and constant deletion should a separate,
  intentionally behavior-changing bundle edit select `derived-v2`. Define
  domain-separated, length-prefixed UTF-8 inputs, SHA/HMAC choice, output width,
  byte order/range, label normalization, and golden vectors; bump the applicable
  authority/materializer identities, invalidate checkpoints, and use
  statistical/OOS gates rather than legacy byte equality.

### MAJOR 4 — `geography.yaml` describes behavior no existing geography kernel or artifact can perform

- **Claim or gap:** The skeleton requires ACS block draws inside observed PUMA,
  ASEC county/complement draws, a tract-to-PUMA assertion, and block-derived
  layers before gap fill (`specs/us/geography.yaml:3-32`). Rollout step 3 puts
  that #696 content inside the supposedly equivalence-gated flip
  (`docs/spec-engine.md:288-290`).
- **Code evidence:** The block ladder contains no PUMA or tract-to-PUMA field
  (`us_runtime/geography_ladder.py:77-119`). Its current assignment samples
  blocks inside an already assigned congressional district
  (`:236-305`) and writes block/tract/county/place/SLD/CBSA, but not PUMA,
  state, or CD (`:326-334`). The current PUMA ladder instead preserves observed
  ACS PUMA, draws PUMA for state-only rows, and draws CD/county within PUMA;
  tract is optional and it never writes block/place/SLD/CBSA
  (`us_runtime/puma_ladder.py:293-383`). The ACS path runs transfer before pooled
  PUMA geography (`us_runtime/acs_multispine.py:127-165`) and explicitly records
  block and tract as unresolved for ACS (`:188-218`). No current artifact or
  kernel implements `identified_county_set` or
  `state_minus_identified_counties`.
- **Rollout failure:** Reusing the block kernel loses the observed-PUMA
  invariant; reusing the PUMA kernel cannot produce the declared block-first
  surface. Either choice changes assignment, schema, predictor availability,
  RNG consumption, and downstream imputation. This is not a representation-only
  flip.
- **Concrete fix:** First encode and equivalence-gate the exact legacy geography
  behavior (including any current no-op in the stacked tool). Move block-first
  geography and the ASEC complement ruling to the post-flip intentional-change
  phase. Before enabling it, produce a versioned block artifact that includes an
  exact 2020 PUMA relationship, pin the official CPS identified-county source,
  implement explicit ACS and ASEC row scopes, derive every final layer from the
  selected block, and assert observed ACS state/PUMA. Name both assignment and
  validation kernels in the spec.

### MAJOR 5 — Canonical construction is underspecified, and the examples already encode identity- and behavior-changing values

- **Claim or gap:** The loader computes a hash over “canonical concatenation”
  (`docs/spec-engine.md:65-68`) but gives no canonicalization contract. It also
  assumes a semantically equivalent bundle will construct identity-stable
  payloads.
- **Code evidence:** Current build identity converts tuple/list to JSON arrays,
  sorts sets by canonical JSON, normalizes NumPy/enum/path scalars, rejects
  non-finite floats, and uses compact UTF-8 JSON with `ensure_ascii=False` and
  sorted keys (`tools/build_us_multispine_pool.py:1315-1322,3769-3802`). Stacked
  authority digests use a different serializer whose default is
  `ensure_ascii=True` (`us_runtime/stacked_spine.py:2349-2357`). Some registries
  sort explicitly, but plan/family/target sequences retain declared iteration
  order (`:2137-2163,2273-2299,3264-3314`). That order is behavior-load-bearing
  for QRF (`microcosm-fit/qrf.py:1087-1098,1523-1533`). Channel order is also
  structural: clone validation requires exactly two channel-major arms in the
  supplied order (`outer_stage_runtime.py:675-733`). Yet `spine.yaml` is shown as
  a mapping and says `mass_anchor_channel: acs` (`docs/spec-engine.md:99-108`),
  while production defaults to ASEC (`us_runtime/stacked_spine.py:542-552,
  677-703`). The late-transfer example declares donor clone 0
  (`docs/spec-engine.md:172-177`), while production declares PUF tax-detail clone
  1 (`us_runtime/support_provenance.py:31-35`; `stacked_spine.py:3390-3405`).
  Finally, eight of the ten files named by `specs/us/bundle.yaml:7-17` do not
  exist and there is no `specs/schema/`, so there is no executable unknown-key,
  default, or canonical-type policy to review.
- **Rollout failure:** Two reasonable loaders can hash or execute different
  bundles because of map order, file boundaries, `1` versus `1.0`, absent versus
  null/defaulted fields, Unicode/paths, or set/list treatment. An alphabetical
  channel compiler can reverse clone order; the literal RFC mass anchor and
  donor clone already change behavior. Merely adding a spec-origin string to a
  receipt can make deterministic checkpoint bytes differ.
- **Concrete fix:** Before schema sign-off, commit complete closed-world JSON
  Schemas (`additionalProperties:false`), a complete legacy-equivalent US
  bundle, compiling UK and Belgian compatibility bundles, invalid fixtures, and
  golden canonical bytes/hashes. The
  normative algorithm should:

  1. parse one YAML 1.2 document per file; reject duplicate keys, merge keys,
     custom tags, non-string keys, implicit timestamps, and non-finite numbers;
  2. validate types and inject every schema default into one typed
     `ResolvedSpec`; reject unknown fields before hashing;
  3. hash a domain-separated envelope containing canonicalizer id/version,
     schema id/version, country, and a map from normalized POSIX-relative file
     names to typed values—not ambiguous raw concatenation;
  4. sort object keys; normalize tuple/list to arrays; normalize each number to
     its schema-declared integer/float type; validate lowercase SHA-256 and
     canonical IDs; require NFC identifiers rather than silently trimming or
     case-folding arbitrary strings;
  5. preserve ordered arrays exactly for stages, channels, directions,
     families, predictors, targets, fallback alternatives, and absence rules;
     only fields explicitly declared as mathematical sets may deduplicate/sort;
  6. state whether status/notes/documentation are normative hash input, and
     keep authority payloads as structured objects rather than JSON strings.

  During equivalence, reuse today's receipt constructors/serialization exactly.
  Unifying their serializers is a later identity-format change. Replace the
  examples with ASEC as the legacy mass anchor and the named
  `puf_tax_detail` support role (not a raw clone integer).

### MAJOR 6 — The RFC's generic kernel write discipline does not exist

- **Claim or gap:** The RFC says kernels receive only declared inputs, write
  only declared outputs, and that existing ownership/tail guards already enforce
  the write side (`docs/spec-engine.md:231-242`).
- **Code evidence:** `ProducerContract` is metadata only; it has no callable or
  parameter schema (`us_runtime/late_producer_dag.py:139-163`).
  `run_producer_when_ready` validates readiness counts and invokes an opaque
  zero-argument callback without projecting reads or diffing writes
  (`:426-518`). The stacked dispatcher closes over and passes the full `Frame`
  and hard-codes a `contract.kind` chain (`us_runtime/stacked_spine.py:9331-9383`).
  Post-execution it checks declared outputs for type/presence/content evidence,
  not mutations to undeclared columns, weights, links, strata, metadata, or mass
  history (`:9445-9471`; declared-only evidence at `:5972-6043`). CPS source
  operators happen to merge only declared outputs
  (`us_runtime/multispine_pool.py:2681-2789`), but their row projection still
  contains all columns (`:2110-2130,2362-2392`) and whole-pool operators replace
  the frame. Existing overlap guards are specialized to particular education,
  retirement, transfer, and tail surfaces (`:2441-2454,2490-2623`;
  `stacked_spine.py:10250-10681`). The tail guard is invoked after the late DAG
  and again at later stage boundaries, not around every kernel
  (`tools/build_us_multispine_pool.py:3117-3120,3198-3201,3214-3217,
  3233-3236,3262`). QBI does have a strong undeclared-surface diff guard, but it
  is hard-coded to one fixed QBI output set rather than supplied by the registry
  (`us_runtime/qbi_inputs.py:1075-1148`).
- **Rollout failure:** A registered kernel can read an undeclared predictor or
  mutate an unrelated existing cell and still satisfy every current check. The
  bundle's IO declaration would be false, so “kernels are the only escape
  hatch” would merely hide behavior behind a named id.
- **Concrete fix:** Route every build kernel through one generic executor. Pass
  an immutable, schema-aware projection containing only declared physical and
  virtual resources; require a returned patch/output object rather than an
  arbitrary replacement frame; snapshot/diff all tables, links, IDs/order,
  weights, strata, metadata, and mass history; and reject every change outside
  declared entity/column/**row** scopes. Output contracts need policies such as
  `fill_missing`, `overwrite_scope`, `assert_equal_noop`, and
  `structural_effect`. Registry records must bind callable id, explicit
  implementation/contract version, parameter schema, IO schema, and supported
  spec range. Retain tail/overlap checks as extra invariants and add adversarial
  tests for undeclared reads/writes and structural mutation.

### MAJOR 7 — The late-producer migration row is lossy

- **Claim or gap:** The migration map says the canonical late registry/groups/
  schedule become only imputation families plus `computed_producers`
  (`docs/spec-engine.md:268-272`); the example gives bare input/output lists
  (`:178-183`).
- **Code evidence:** Current inputs include entity/column, value-kind,
  `required_scope`, `producing_stage`, alternative physical column sets, and
  tolerated-absence receipt ids (`us_runtime/late_producer_dag.py:55-119`).
  Outputs include coverage scope (`:122-137`). Optional inventory rows become
  producer-bound absence receipts (`us_runtime/us_late_producer_registry.py:
  1559-1594`). The identity payload carries overlap ownership, execution-receipt
  rules, schedule order/waves/edges, transfer groups, and every source/primary/
  ACS/transfer inventory (`:2047-2103`).
- **Rollout failure:** Two compilers cannot reconstruct the same readiness
  semantics or current schedule receipt. One may choose first-present, another
  any-present, and a third zero-fill. Optional inputs can become fatal or
  silently optional; row-scope ownership and non-owner actions disappear. That
  changes behavior and checkpoint identity before any intended edit.
- **Concrete fix:** Add a lossless `producer_graph` schema that can represent
  every current `ProducerInput`, `ProducerOutput`, virtual resource, alternative,
  value-kind, absence receipt, coverage scope, final owner, and non-owner action.
  Derive order/waves from it. Before flipping, compile the bundle back into the
  current schedule/ownership payload and require byte-identical canonical
  payloads and receipts.

### MAJOR 8 — Whole-column closure is insufficient for mixed ownership and conditional outputs

- **Claim or gap:** `column_lineage.yaml` gives each artifact column exactly one
  class and closure is checked against one committed inventory and the final
  artifact (`docs/spec-engine.md:223-229`). The RFC does not define expected
  presence by profile or cell-scope ownership.
- **Code evidence:** ACS transfer explicitly preserves every existing non-null
  target and fills only null cells (`us_runtime/acs_transfer.py:894-900,
  992-1043,3006-3035`). Generic take-up likewise preserves measured/source-owned
  non-null cells and fills only missing cells (`us_runtime/take_up.py:307-386`).
  Thus one physical column can contain measured-native and imputed cells under
  different owners. Presence can also be conditional: `tract_geoid` is written
  only when `assign_tract=True` (`us_runtime/puma_ladder.py:81-82,293-383`), and
  its gate changes the expected set under the same flag (`:496-498`). ACS
  transfer skips fitting and returns the canonicalized recipient when no target
  is active (`acs_transfer.py:943-975`),
  and an absent ACS source returns the base frame (`us_runtime/acs_multispine.py:
  98-103`). I found no current `f001`/`f004`/`f010`/`f025`/`f100` branch that
  changes column presence: those tokens select sample fractions/release grammar
  (`tools/build_us_multispine_pool.py:277-293,496-505`). The concrete current
  conditional is configuration/profile-dependent (`assign_tract`), not
  rung-dependent.
- **Rollout failure:** A column-level class cannot enforce who may write which
  cells. A one-sided “no unclaimed extras” check can miss required absent
  outputs, while equality to one fixture rejects legitimate profiles. Two
  implementers can disagree about whether a skipped kernel must materialize an
  all-null/empty column.
- **Concrete fix:** Identity-bind a closed artifact profile and resolve the
  expected set at load. Every output/lineage declaration must be `required`,
  `forbidden`, or guarded by a closed spec predicate; runtime closure compares
  expected and actual in both directions at each checkpoint and final output.
  Conditional skips require a canonical skip receipt and required outputs must
  materialize with canonical dtype even for zero-row scopes. Keep a primary
  whole-column classification for documentation, but add exhaustive,
  non-overlapping lineage/ownership segments over
  `(entity,column,row_scope,stage,write_policy)`.

### MAJOR 9 — `take_up.yaml` is factually false and assumes the wrong decomposition

- **Claim or gap:** The only concrete row says SNAP is one per-flag
  `snap_state_take_up` draw, with an eligibility interaction and
  `calibration_status: none` (`specs/us/take_up.yaml:4-12`). The RFC says every
  `takes_up_*` flag has that mechanism/source shape and leaves the existing
  contract as a parameter source (`docs/spec-engine.md:185-198,274`).
- **Code evidence:** The hashed take-up contract is itself a curated authority
  over engine facts, treatments, rates, calibration targets, scope owners, and
  debt states (`us/take_up_contract.json:1-27`;
  `us_runtime/take_up_contract.py:123-230`). SNAP is marked `out_of_scope` for
  generic seeding because a national reported-anchor/rate prior is followed by
  a dedicated state count-calibration stage (`take_up_contract.json:19-27`).
  The separate source manifest declares the national prior stage and then the
  state anchored count-calibration stage, including unmasked assignment and an
  eligible-only calibration domain (`us/source_stages.json:1950-2036`).
  That stage derives a runtime state prior as target divided by weighted modeled
  eligibles (`us_runtime/snap_state_take_up.py:186-210`), runs anchored assignment
  plus count calibration, and overwrites the final flag (`:225-305`). Eligibility
  is the calibration/engine domain, not the entire assignment universe. The
  generic seeder is batched across all `seed` programs and special-cases EITC
  (`us_runtime/take_up.py:242-294,307-386`). Across the contract, source
  manifests, and runtime modules—not all as literal treatment-enum labels—other
  flags are measured, transferred, count-calibrated, unsourced/defaulted,
  near-universal, or owned by dedicated stages (`take_up_contract.py:60-68` and
  the program inventory).
- **Rollout failure:** An implementation faithful to the YAML either omits SNAP
  state calibration, changes the assignment universe, or secretly continues to
  read the JSON contract/source manifest. Editing YAML may do nothing, leaving
  two authorities. The shown per-flag schema has no invocation/group id,
  deduplication rule, ordered pipeline, or final-owner semantics, so two
  compilers can invoke a batched kernel once, once per flag, or not at all, and
  cannot agree on a prior plus later finalizer.
- **Concrete fix:** Replace `draws` with a discriminated program inventory and
  ordered mechanism pipelines. Required treatments include at least measured,
  imputed/transferred, seeded-rate, batched-seeded, anchored-count-calibrated,
  engine-default-with-debt, near-universal, and out-of-scope/dedicated-stage.
  For SNAP declare reported anchor, national prior, stable-source-ID draw
  universe, runtime state-prior derivation, eligibility calibration domain,
  FNS target source, saturation rule, final-owner stage, and diagnostics/gate.
  Separate output ownership from kernel invocation so one kernel may own many
  flags and one flag may have multiple stages. Either absorb the curated
  contract and relevant `source_stages.json` rows into the bundle or make those
  exact resources bundle components; do not duplicate a subset while calling
  the old authority a mere parameter source. Retain exact coverage checks
  against installed engine metadata.

### MAJOR 10 — The RFC creates a second country-spec system and a second composition hash

- **Claim or gap:** The RFC presents root `specs/<country>/` plus a new loader as
  the shared country-spec mechanism (`docs/spec-engine.md:39-68`) without
  migrating the existing one.
- **Code evidence:** `country_spec.py` already defines a spec-only country
  package, hashes every declared resource, rejects undeclared/missing files,
  type-validates its recognized resource kinds, and separately compiles the
  source/geography plan with a no-fallback posture (`country_spec.py:1-10,
  797-920,923-996`). `CountrySpec.fingerprint` hashes the composition of every
  resource (`:756-794`) using a defined sorted-hash algorithm
  (`trace.py:96-123`). US and
  UK already have packaged manifests (`us/country_package.json:1-26`;
  `uk/country_package.json:1-20`), including US source, support, PUF, take-up, and
  fiscal resources. Belgium—not US—is explicitly described as the first full
  consumer and already declares source, geography, target, gate, and release
  resources (`country_spec.py:12-16`; `be/country_package.json:1-11`). The
  `schema_version` present in those package manifests is not read or validated
  by the current loader (`country_spec.py:820-859`), another version seam the RFC
  must absorb. The loader finds installed package resources with
  `importlib.resources` (`country_spec.py:813-817`). Root `specs/us/*.yaml` is not
  under the wheel package path declared by
  `packages/microcosm-build/pyproject.toml:55-63`. There is also an existing root
  `specs/us_imputation_lineage.yaml` described as source of truth and consumed by
  a conformance test/dashboard (`specs/us_imputation_lineage.yaml:1-4`;
  `packages/microcosm-build/tests/test_imputation_lineage_spec.py:1-31`).
- **Rollout failure:** Production can have two resource trees, two loaders, and
  two fingerprints answering “what spec built this?” The existing fingerprint
  composes hashes of raw resource bytes (`country_spec.py:786-794`;
  `trace.py:109-123`), whereas the proposed hash canonicalizes parsed YAML, so
  they even disagree on whitespace/key-order-only edits. Checkout tests may pass
  while an installed wheel cannot find the new YAML. Old US or Belgian resources
  can continue steering execution outside `spec_sha256`.
- **Concrete fix:** Make the RFC an explicit extension/replacement migration for
  `CountrySpec`, not a parallel loader. Choose one packaged manifest, one
  installed-resource lookup, and one public composition binding. Prefer placing
  the resolved bundle/schemas under package data and adding its new file kinds
  to `CountrySpec`; alternatively explicitly package root `specs/`, but delete
  the duplicate package resources in the same staged migration. Version the
  transition from raw-byte `CountrySpec.fingerprint` to canonical
  `spec_sha256` with an explicit identity-generation boundary; they cannot be
  aliases. Begin enforcing/versioning the previously inert package
  `schema_version`, test a clean built wheel, and migrate a real Belgian bundle
  or explicitly scope/deprecate that system. Explicitly retire/move
  `us_imputation_lineage.yaml` and its emitter rather than leaving a third
  surface.

### MAJOR 11 — The migration map is not total enough to support constant deletion

- **Claim or gap:** The RFC's migration table claims the relevant authority has
  a bundle home and then deletes constants/conformance tests
  (`docs/spec-engine.md:262-281`). It omits multiple behavior- and
  identity-bearing classes.
- **Code evidence:** At minimum, the audit found:

  | Missing class | Current evidence | Required disposition |
  |---|---|---|
  | Pipeline/runtime | mass shares, operator and checkpoint-stage order, seed, period, source-reuse doctrine, model sizes, and simulation batch size in `us_runtime/multispine_pool.py:183-251`; checkpoint schemas/materializer ledgers, estimator counts, filenames, pipeline id, rung grammar, and release regex in `tools/build_us_multispine_pool.py:216-293` | Normative order/params in bundle; serialization/materializer versions remain code-owned and identity-bound |
  | Late DAG semantics | registry/receipt/transition versions, hard-coded stage/scope/entity/virtual-resource names at `us_runtime/us_late_producer_registry.py:126-169`; tolerated absences and alternatives at `late_producer_dag.py:69-119` | Lossless producer graph or explicitly versioned kernel ABI |
  | PUF support/clone/tail | support channel and clone index at `us_runtime/support_provenance.py:31-35`; tail thresholds, quantiles, topcode, filing statuses, AGI proxies, support/no-widen doctrine at `puf_capital_gains_tail.py:76-240`; PUF aggregate seed/RECIDs/bounds at `puf_source_agi.py:21-52,280-421`; silently loaded and hashed SOI bands at `puf_interest_components.py:100-205` and aggregate-record spec at `puf_aggregate_records.py:226-232`; primary QRF target order, absence doctrine, and checkpoint layout at `puf_qrf_chain.py:80-129` | Bundle resources/params for normative choices; role names instead of raw clone integers; explicit asset pins; code-owned envelope versions |
  | Resume/operations | checkpoint-root and bank layout at `tools/build_us_multispine_pool.py:487-554,572-587`; exact-identity discovery at `:1169-1244`; logbook predecessor env fallback at `:3965-3977`; hard-coded spool/receipt roots at `:3985-4064`; fit worker env/interpreter/CPU bindings at `stacked_spine.py:4620-4699` | Formal normative/operational/external-state classification and receipts |
  | Calibration/selection | public solver defaults and mass/loss/warm-start/L0/L1/L2 surface at `microcosm-calibrate/solve.py:1306-1331`; pruning and backend cutoffs at `:111-128,410-414`; exact-k requires `pi_hi`, seed, and optional grouping at `exact_k.py:424-480`; registry artifact version at `registry.py:42-44` | Fully resolved build params; internal algorithm choices bound by kernel contract/materializer version |

- **Rollout failure:** Deleting a listed subset either breaks construction or
  leaves hidden Python/package/environment authorities. An unchanged bundle can
  then change output or resume stale checkpoints, defeating D1 and D3.
- **Concrete fix:** Generate an audited inventory from the current full base
  identity, authority receipts, packaged resource loads, runtime signatures,
  environment reads, and stage manifests. Give every item exactly one owner:
  bundle-normative, versioned kernel ABI/implementation, artifact schema/
  materializer, operational receipt, or external mutable chain state. Require a
  reviewed static allowlist/denylist test until deletion, and require the
  resolved legacy bundle to reproduce the current identity/authority payloads
  byte-for-byte. Delete a constant only after its replacement or deliberate
  code-owned classification is tested.

### MAJOR 12 — `schema_version`, `spec_sha256`, authority versions, and materializer versions have no interaction policy

- **Claim or gap:** The RFC introduces three apparent configuration/version
  concepts but says only that the bundle hash becomes the primary answer and
  semantic versions remain (`docs/spec-engine.md:30-33,244-247`). It does not say
  which changes invalidate compatibility/resume or what a schema bump means.
- **Code evidence:** Current code needs several independent axes: checkpoint
  schema/materializer v1/v7 with an explicit semantic invalidation ledger
  (`tools/build_us_multispine_pool.py:216-258`), stacked materializer v11
  (`:284-289`), frame checkpoint schema v3 (`frame_checkpoint.py:44-48`), outer
  context schema v2 (`outer_stage_runtime.py:42-48`), stacked authority v10 and
  component digests (`us_runtime/stacked_spine.py:1690-1702,2338-2365`), late
  registry/receipt/transition versions
  (`us_runtime/us_late_producer_registry.py:126-140`),
  and target-registry artifact version (`microcosm-calibrate/registry.py:42-44`).
  The materializer comment explicitly requires a bump when implementation
  changes under the same registry name (`tools/build_us_multispine_pool.py:
  245-250`). The repository already has `builder_code_identity`, specifically
  because pins and seeds alone can blend old-code and new-code checkpoints; it
  hashes packaged sources and numeric dependency versions
  (`code_identity.py:1-10,27-75`). US PUF support and UK builders use it
  (`tools/build_us_puf_support_base.py:699-736`;
  `tools/build_uk_national_dataset.py:1032-1069`), but the f004 stacked base
  identity does not; its git pin is written only to Logbook
  (`tools/build_us_multispine_pool.py:3919-3934,3991-4004`). A YAML hash cannot
  notice such Python/dependency changes.
- **Rollout failure:** If “one hash” replaces those invalidators, unchanged YAML
  can resume output from changed kernel code. If every schema bump is treated as
  a semantic authority bump, harmless grammar evolution becomes impossible. If
  semantic versions override the hash, config edits can share identity.
- **Concrete fix:** Define four orthogonal, jointly required identity classes:

  1. `schema_id`/`schema_version` plus `canonicalizer_version` define accepted
     syntax and typed resolution; unsupported versions fail before execution.
  2. `spec_sha256` identifies the exact fully resolved normative configuration;
     any semantic config/default/order/pin edit changes it.
  3. Each kernel/authority exposes an implementation/contract version (or a
     `kernel_set_sha256` over id, implementation version, params schema, and IO
     schema), backed by a code/dependency artifact digest rather than a manual
     version alone; the bundle pins a supported version/range and the loader
     checks it.
  4. Artifact/checkpoint schema and materializer versions remain code-owned and
     describe serialization/materialization compatibility.

  Resume requires equality of all four; none masks another. A schema migration
  produces a new schema version and normally a new spec identity even when an
  equivalence fixture proves behavior unchanged. A config-only edit changes the
  spec hash without requiring an authority bump. A semantic kernel edit bumps
  its contract/materializer even with unchanged YAML. An explicit tested
  translator/semantic-IR hash may permit migration, but it must be a separate
  mechanism, never implicit precedence.

### MAJOR 13 — Keeping model defaults as library fallback reintroduces dual authority

- **Claim or gap:** The migration map says build model settings move into the
  bundle while `microcosm.fit` keeps defaults for library users
  (`docs/spec-engine.md:277`). It does not forbid build kernels from omitting
  those arguments.
- **Code evidence:** QRF defaults remain live at
  `microcosm-fit/qrf.py:83-89,1033-1044`. Production APIs also have their own
  defaults, e.g. ACS transfer seed/estimators/max-targets
  (`us_runtime/acs_transfer.py:880-892`) and primary QRF predictors/outputs/seed/
  estimators/absence doctrine (`us_runtime/puf_qrf_chain.py:118-130`). Calibration
  has an even larger default surface (`microcosm-calibrate/solve.py:1306-1331`).
  This is live production behavior, not hypothetical: primary QRF and ACS
  transfer instantiate QRF with only `n_estimators` and `seed`, so `zero_atol`
  and `max_samples_leaf` come from the library defaults
  (`us_runtime/puf_qrf_chain.py:219-225`; `acs_transfer.py:1341-1345`).
  The current mirror test pins some code defaults precisely because omitted
  kwargs otherwise matter (`test_imputation_lineage_spec.py:87-97`).
- **Rollout failure:** A library release can change a default and alter a
  bundle-built artifact without changing `spec_sha256`. That is the same
  code/spec drift the authority flip is intended to remove.
- **Concrete fix:** Registered production adapters must accept a fully
  materialized, schema-complete config and pass every build-facing parameter
  explicitly. Production code may not use `dict.get(...library_default)`, omit a
  behavior-bearing kwarg, or infer a build default from the library. Standalone
  library calls may retain convenience defaults. Add a test that monkeypatches
  library defaults and proves the resolved build plan/output is unchanged, plus
  a static/contract test that every registered build invocation supplies its
  declared parameter set. Internal algorithm changes remain covered by the
  kernel/materializer identity from MAJOR 12.

### MAJOR 14 — Source identity and deployment/operational bindings are conflated

- **Claim or gap:** `sources.yaml` purportedly replaces launcher paths and pins
  every external input (`docs/spec-engine.md:72-95,266`), but the RFC calls the
  locator “documentation” and gives no runtime binding protocol.
- **Code evidence:** The tool currently requires six local paths and six pins
  (`tools/build_us_multispine_pool.py:406-478`). Verified manifest entries include
  resolved absolute paths (`:341-355`), while logical checkpoint identity wisely
  includes only role, actual hash, and size (`:1011-1019`). Checkpoint roots are
  operational fallbacks (`:487-554`); logbook chain head can come from an env var
  (`:3965-3977`); spool/receipt directories are derived operationally
  (`:3985-4064`). Worker environment and interpreter/CPU choices are separately
  captured in fit bindings (`us_runtime/stacked_spine.py:4620-4699`).
- **Rollout failure:** Embedding host paths, credentials, checkpoint roots, or
  logbook chain state in the bundle makes identical semantic builds hash
  differently and can leak restricted locations. Leaving all env/CLI state out
  without classification lets behavior-bearing inputs escape identity. Calling
  a locator documentation does not make the bundle drive the build.
- **Concrete fix:** Define three surfaces: (1) normative and hashed logical
  source ids, roles, content pins, loader ids, and semantic options; (2)
  operational and receipted bindings from source id to local path/URI,
  checkpoint/output/spool roots, credentials, and demonstrably output-invariant
  worker counts; (3) external mutable chain state such as logbook predecessor,
  bound by the chain protocol rather than spec hash. The launcher supplies an
  exact id-to-location mapping; the loader requires a bijection, verifies the
  bundled hash/size, and never hashes the host location. Explicitly classify
  any worker/backend option that can change bytes as normative or implementation
  identity, not merely operational.

### MAJOR 15 — The proposed “shared schema” is US-shaped and already fails the UK implementation

- **Claim or gap:** Principle 5 and rollout step 6 say UK instantiates the same
  schema after the US flip (`docs/spec-engine.md:39-41,295`).
- **Code evidence:** The examples assume ASEC/ACS channel names, PUF attachment,
  numeric clone roles, US census blocks/PUMA/FIPS/CD119, `takes_up_*`, and a US
  publication grammar (`docs/spec-engine.md:97-134,162-198,218-220`). Production
  codifies PUF role `asec`/`puf_tax_detail` and clone 1
  (`us_runtime/support_provenance.py:31-35`). The UK geography mechanism is a
  two-stage constituency-then-output-area draw with UK-specific OA/LSOA/MSOA/
  LAD/ward/ITL layers (`uk_runtime/geography_ladder.py:1-25,96-121`), and that
  module explicitly says the existing shared country schema only models
  `clone_assign_uniform` and needs a new `anchor_sample_oa_ladder` method plus
  resource wiring (`:66-74`). UK uses person/benunit/household entities
  (`uk_runtime/national_frame.py:61`), FRS/SPI support with 10,000 synthetic
  households and 50% prior mass (`uk_runtime/spi_support.py:29-42`), and
  different exported versus in-memory clone-column shapes
  (`uk_runtime/rowwise_dataset.py:56-78`). Those conventions are not PUF clone
  attachment. Belgium's already-live country package is a third compatibility
  obligation, not mentioned by the RFC (`country_spec.py:12-16`).
- **Rollout failure:** The first UK bundle will require discriminators or a
  schema break after US code/constants have already been deleted. US names can
  leak into supposedly generic loader/compiler APIs.
- **Concrete fix:** Split a small country-neutral core (source/resource binding,
  typed stage DAG, kernel contracts, lineage/identity, artifact profile) from
  discriminated country extensions. At minimum use support kinds such as
  `puf_attachment`, `synthetic_prior_replacement`, and `none`; geography kinds
  such as `single_anchor` and `two_stage_anchor`; arbitrary entity/channel/role
  ids; a geography layer graph instead of fixed FIPS keys; and optional country
  take-up/publication extensions. Compiling a minimal UK bundle and a
  compatibility bundle for existing Belgian semantics must validate in rollout
  step 1. “Same schema” should mean the
  same versioned core plus declared extensions, not the US file shapes verbatim.

### MAJOR 16 — Legacy-mode lifecycle and pre-spec artifact handling are unspecified

- **Claim or gap:** The RFC introduces a temporary `--legacy-constants` mode,
  immediately joins the hash, later deletes constants, and says nothing about
  in-flight artifacts or how long dual authority stays tested
  (`docs/spec-engine.md:252-258,283-295`).
- **Code evidence:** Adding any identity field routes QRF/ACS banks under a new
  digest (`tools/build_us_multispine_pool.py:572-587`). Stacked discovery requires
  exact identity equality (`:1181-1244`): adding the binding only to the base
  identity rejects old mappings inside the existing configured namespace, while
  the required fix of also adding it to `_configured_stacked_identity` routes to
  a new namespace (`:1150-1178`). Either way, pre-spec checkpoints cannot resume
  under the new identity. The tool already has a separate whole-pipeline
  `--legacy-two-spine` flag (`:528-531`), creating four undefined flag
  combinations. PR CI also cannot perform the restricted-input certification
  build (`CLAUDE.md:31-40`).
- **Rollout failure:** Step 2 is operationally a cold-cache cutover even if it is
  semantically unchanged. Relabeling an old artifact with a hash it never bound
  is false provenance. A constants path not exercised on every relevant change
  will rot, while accidental interaction with `--legacy-two-spine` can compare
  different pipelines.
- **Concrete fix:** Call the selector
  `--config-authority={bundle,constants}`, apply it only to the current stacked
  pipeline, and reject or precisely define its interaction with
  `--legacy-two-spine`. Parameterize fixture integration tests over both
  authorities on every change to bundle, loader, registry, identity, or kernels;
  require both to load/hash the same bundle. Run the cold f004 proof in the
  restricted certification lane and retain the selector through at least one
  certified full release and a stated retention deadline. For old artifacts,
  make a hard cut: drain old runs on old code, keep old identity namespaces
  read-only, add an `identity_generation`, and cold-build the new namespace.
  Never retrofit `spec_sha256`; a legacy reader may inspect old artifacts for
  comparison but must not promote them. Readers/logbook schemas should accept
  historic generation 0 and require the new binding for release promotion after
  the cutoff.

## MINOR findings

### MINOR 1 — The RFC and committed geography skeleton disagree about the ASEC fallback

- **Claim or gap:** The RFC says unidentified ASEC rows draw from
  `identified_county_else_state` (`docs/spec-engine.md:121-123`); the committed
  skeleton says state **minus** identified counties
  (`specs/us/geography.yaml:8-15`).
- **Code evidence:** `identified_county_set.preferred` and `.fallback` are prose
  tokens, not source ids (`specs/us/geography.yaml:16-19`), despite the source
  pin rule at `docs/spec-engine.md:72-95`; neither existing geography kernel
  consumes them.
- **Rollout failure:** Implementers can choose different universes, and a
  fallback derived from a sampled pooled file can vary by rung.
- **Concrete fix:** Make the complement ruling the sole text, reference a pinned
  official source id, and disallow the fallback in production. If a fallback is
  unavoidable, define an exact pre-sampling derivation and bind the resolved
  county list plus digest into identity/receipt.

### MINOR 2 — “Unreferenced kernel fails load” has no registry scope

- **Claim or gap:** Total closure rejects an unreferenced kernel
  (`docs/spec-engine.md:22-25`).
- **Code evidence:** The same RFC expects a shared multi-country schema and
  country-specific bundles (`:39-41`), while the current late registry already
  contains multiple kinds and stages (`us_runtime/us_late_producer_registry.py:
  126-169`).
- **Rollout failure:** If “unreferenced” means every callable in a process-global
  registry, the US bundle fails merely because UK, library-only, diagnostic, or
  test kernels are installed.
- **Concrete fix:** Require closure over the selected country/profile registry
  namespace and instantiated aliases. Unknown referenced ids and duplicate ids
  always fail; unused library implementations do not, or are explicitly marked
  `library_only`.

### MINOR 3 — “Every logbook row has spec_sha256” is impossible for spec-load failures

- **Claim or gap:** D3 says every logbook row carries the validated bundle hash
  (`docs/spec-engine.md:30-33,244-247`) while malformed bundles must refuse at
  load.
- **Code evidence:** The tool creates a terminal-attempt state before input or
  configuration validation and records failure rows using its current identity
  digest (`tools/build_us_multispine_pool.py:3985-4015,4138-4165`). A missing,
  duplicate-key, or unparsable bundle has no valid canonical `spec_sha256`.
- **Rollout failure:** Implementers must either omit the row, lie with a partial
  hash, or violate the “every row” schema.
- **Concrete fix:** Make `spec_sha256` required only after successful spec
  resolution. Failure rows carry `spec_binding_status`, schema/canonicalizer
  attempt, and—when bytes were readable—a separate raw manifest/file-set digest
  plus the validation error. Never call that raw digest `spec_sha256`.

## NIT findings

### NIT 1 — The advertised “real excerpts” and sign-off artifacts still contain placeholders

- **Claim or gap:** The RFC calls each example a real excerpt
  (`docs/spec-engine.md:70`) and asks for schema sign-off in rollout step 1
  (`:283-286`).
- **Code evidence:** The excerpts contain literal `"…"`, `<schema-v2 artifact
  sha>`, `<hard bound>`, and an abbreviated release grammar
  (`docs/spec-engine.md:88-92,150-151,171,181-182,209-210,218-219`); the committed
  take-up skeleton promises future rows rather than declaring them
  (`specs/us/take_up.yaml:1-12`).
- **Rollout failure:** Reviewers cannot distinguish schema syntax from prose or
  validate whether placeholders are legal values.
- **Concrete fix:** Label snippets pseudocode, or replace every placeholder with
  valid schema-conforming data and put explanatory omissions outside the YAML.

## Deletion checklist

Do not delete legacy constants or the authority selector until all of the
following are true:

1. One packaged loader and one canonical composition binding cover all current
   `CountrySpec`, root lineage, stage-manifest, contract, and resource authority.
2. Complete schemas, canonicalizer golden vectors, invalid fixtures, a full US
   resolved bundle, and compiling a minimal UK bundle plus a compatibility
   bundle for existing Belgian semantics pass from a clean wheel.
3. Constants and bundle compile to byte-identical current plan, schedule,
   ownership, authority, and identity payloads; both modes load the same bundle.
4. Cold fixture dual-mode tests run on every relevant PR, and a restricted cold
   f004 certification compares all stage content plus canonical gates. At least
   one full release has been certified before expiry.
5. Production registered kernels use the generic projection/diff executor and
   no production path consumes library defaults or undeclared package assets,
   env values, stage strings, source pins, or params.
6. Checkpoint/logbook/manifest/dashboard surfaces carry the namespaced spec
   binding plus kernel/authority and artifact/materializer identities.
7. Historic identity-generation retention, reader compatibility, hard-cut
   rebuild, and cleanup dates are documented; no old artifact is relabeled.
8. Only then remove constants, `--config-authority=constants`, and interim
   code-equals-bundle tests. Retain schema/canonicalizer golden tests, generic
   kernel-discipline tests, closure tests, and content-equivalence fixtures.

## Verdict on D1–D5 and rollout order

Using the RFC's five operative decisions as D1 authority flip, D2 registered
kernels, D3 spec identity binding, D4 frozen-behavior gate, and D5 derived seed
streams:

| Decision | Verdict | Required amendment |
|---|---|---|
| D1 — bundle is build authority | **Needs amendment; direction stands, wording does not.** | Extend/replace the existing packaged `CountrySpec`; make the fully resolved bundle the only normative build config; use cell-scope lineage; separate runtime source bindings; keep code-owned kernel and artifact contracts explicit. |
| D2 — registered kernels are the escape hatch | **Needs amendment.** | Add the generic projected-input/patch-output/diff executor and a lossless producer graph. A name lookup plus current specialized guards is insufficient. |
| D3 — `spec_sha256` joins identity/logbook | **Needs amendment.** | Both modes load the same bundle before branching; use a namespaced binding; retain kernel/authority/materializer identity; define invalid-spec log rows and the pre-spec artifact cutover. |
| D4 — identical identities and gates prove the flip | **Does not stand as written.** | Use two cold isolated executions, forbid resume, and compare deterministic stage content/exhaustive frame digests plus normalized gates. |
| D5 — derive every stream from one root seed in the flip | **Does not stand as written.** | Equivalence uses explicit `legacy-v1` streams. A fully specified `derived-v2` is a later intentional behavior change with fresh identity/caches and statistical gates. |

Therefore **none of D1–D5 stands completely as written**, although D1's
high-level direction remains worth pursuing.

The current rollout order is not sound. A coherent order is:

1. Reconcile the RFC with `CountrySpec`; land the complete schemas, typed IR,
   canonicalizer and golden vectors, full legacy-equivalent US bundle, minimal
   real UK bundle, Belgian compatibility fixture, version-interaction policy,
   operational binding model, and old-artifact policy.
2. Make both current-stacked authority paths load/hash that same bundle. Add the
   namespaced spec binding everywhere in one intentional cold-cache identity
   generation; retain explicit legacy seeds and current geography/mass/clone
   semantics.
3. Build the generic kernel executor and lossless producer graph behind the
   legacy execution path. Require byte-identical compiled plan/authority/
   schedule payloads before comparing data.
4. Run per-PR cold fixture equivalence and restricted cold f004 content
   certification, flipping one stage at a time. Geography at this point must be
   the exact legacy behavior, not #696's block-first change.
5. After at least one certified full release and the deletion checklist, remove
   Python configuration constants and the temporary authority selector.
6. Land intentional behavior edits separately: block-first/ASEC complement,
   eligibility and calibration changes, then `derived-v2` seed streams. Each is
   a bundle diff with the appropriate authority/materializer bump, cold caches,
   and f025/OOS or statistical gates—not a legacy equivalence claim.
7. Expand country extensions only after the real UK and Belgian conformance
   bundles have already proved the shared core.
