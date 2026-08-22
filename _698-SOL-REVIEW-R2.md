# Round-2 adversarial review of the spec-engine RFC v2 (PR #698)

Reviewed locally at `d865ba40` against repository source and the three held
wave-1 branch heads: `per-family-predictor-sets` at `016cb662`,
`lineage-column-closure-697` at `323f6c69`, and
`block-first-geography-696` at `7fff4489`. I did not use the network, a virtual
environment, a pipeline build, or certification inputs. This is a
design/source review, not an implementation review.

**Bottom line: request changes.** V2 fixes several important R1 defects, but it
is not sign-off-ready. Three R1 findings are asserted away without a schema
capable of representing the current system, and six more are only partial. The
new machinery adds independent blockers: F0 contains the compiler work assigned
to F1, the node key can collide across different stochastic run requests, the
frozen split ledger is factually wrong (five/37, not four/32), take-up engine
validation is circular, catalogs/vintages recreate dual authority, and the
equivalence vector stops before final publication.

## 1. Round-diff audit

Verdict count: **11 RESOLVED, 6 PARTIALLY RESOLVED, 3
RENAMED-NOT-RESOLVED, 0 REGRESSED**.

| R1 | Verdict | V2 text | Code-grounded adjudication |
|---|---|---|---|
| M1 | **RESOLVED** | `docs/spec-engine.md:217-231,630-635` puts one namespaced `spec_binding` in configured/base identity, makes both authorities load the same bundle before branching, and replaces singleton canonicality. | This fixes the R1 ordering contradiction. Current namespace routing is `_configured_stacked_identity` (`tools/build_us_multispine_pool.py:1150-1178`); stage discovery reconstructs and exactly compares `_stacked_checkpoint_base_identity` (`:1043-1147,1181-1244`); stage and bank identities inherit that base (`:1298-1322,2602-2633`). Loading and hashing before the current configured-identity call at `:4192` is implementable. The real `is`-based guards are at `stacked_spine.py:3089-3147,3349-3376,3672-3695`, and v2 explicitly replaces them. The missing concrete `identity_generation` field is charged to M12 rather than double-counted here. |
| M2 | **RESOLVED** | `docs/spec-engine.md:623-652` now requires two cold isolated roots, forbids checkpoint/model-bank resume, compares all three deterministic stage H5 files (or an exhaustive frame digest), normalizes terminal gates, and emits mismatch diagnostics. | This addresses both original failures for the original three pool cutpoints plus terminal gates: a changed ordinary cell is covered by checkpoint content, and run two cannot resume run one. The serializer is intentionally deterministic (`frame_checkpoint.py:1-7,91-155,318-396`) and current shared-root discovery/resume is real (`tools/build_us_multispine_pool.py:4192-4239`). V2's broader bundle now claims publication and downstream surfaces outside that original vector; that new scope insufficiency is N6 below. |
| M3 | **PARTIALLY RESOLVED** | `docs/spec-engine.md:258-279` correctly preserves `legacy-v1` for the flip and defers stateless `derived-v2` to F3; `specs/us/bundle.yaml:13-14` selects it. | The phasing defect is fixed, but the promised named-stream declaration does not exist, and a static stream map cannot declaratively reproduce code-internal, data-dependent consumption. QRF creates two `SeedSequence` children and advances one shared fit RNG in target order (`qrf.py:1077-1098,1128-1148,1333-1380,1428-1429`); ACS uses exact NUL-separated labels, SHA-256 truncation, and little-endian decoding (`acs_transfer.py:2902-2916`). V2 does not say what is spec-normative, kernel-contract-normative, or merely receipt-descriptive. |
| M4 | **RESOLVED** | `docs/spec-engine.md:408-442,669,719-735` and `specs/us/geography.yaml:3-50` put block-first geography and the ASEC complement in F3 and require exact legacy geography through the flip. | This is coherent with the current gap: the legacy block ladder lacks PUMA and samples inside a prior CD, while the PUMA ladder preserves/draws PUMA and never assigns a block. An implementer no longer has to fake block-first behavior during byte equivalence. The held branch's failure to include the later complement ruling is a rollout issue (N8), not a recurrence of M4's phase error. |
| M5 | **RESOLVED** | `docs/spec-engine.md:152-182` defines closed-world typed resolution, canonical envelope/file boundaries, numeric normalization, explicit ordered-vs-set fields, golden bytes/hashes, and preservation of the current serializers during equivalence; corrected ASEC/clone-role examples are at `:388-406`. | Following this contract avoids v1's ambiguous concatenation and ordering/default/numeric failures. The current serializers really differ (`tools/build_us_multispine_pool.py:1315-1322`; `stacked_spine.py:2349-2357`), and v2 correctly does not silently unify them at the flip. |
| M6 | **RESOLVED** | `docs/spec-engine.md:281-303,719-723,757-759` requires one projected-input/patch-output executor, full structural diffing, capability types, and adversarial read/write tests before bundle mode drives production. | This is an implementable replacement for the current metadata-only `ProducerContract` and opaque callback (`late_producer_dag.py:139-163,426-518`) and the full-frame dispatcher. V2 no longer claims today's specialized guards already provide the generic discipline. |
| M7 | **RENAMED-NOT-RESOLVED** | `docs/spec-engine.md:305-312,508-516,668` calls `producer_graph` lossless and promises byte-identical compile-back. | The only sketched record is not lossless. It omits `producing_stage` and tolerated receipt IDs, flattens the OR-of-AND `alternatives`, puts an invalid `value_kind: amount` on the logical input, spells `coverage_scope` as `coverage`, and invents a per-output `final_owner` Boolean. Actual fields are in `late_producer_dag.py:47-137,176-209`; kind-specific virtual resources and complete receipt semantics are in `us_late_producer_registry.py:1597-1736,2047-2103`; final ownership is an 18-row target × origin × clone-role matrix (`us_late_overlap_ownership.py:60-199,219-261`). An implementer following the shown shape cannot reproduce the current payload. |
| M8 | **PARTIALLY RESOLVED** | `docs/spec-engine.md:325-354` adds a closed artifact profile, bidirectional checkpoint/final closure, predicates, skip receipts, canonical empty outputs, and exhaustive `(entity,column,row_scope,stage,write_policy)` segments. | Those provisions directly address mixed-cell ownership and conditional presence. But `:356-359` immediately reauthors one catalog row per column with `owner` and `lineage class` and says closure derives from it. That is again a whole-column second authority and conflicts with `:327-335`, where owner/class derive from producer outputs. The runtime segment model can be correct only if catalog owner/class become compiler-generated, not authored. |
| M9 | **RENAMED-NOT-RESOLVED** | `docs/spec-engine.md:519-558` and `specs/us/take_up.yaml:17-51` replace one draw per flag with a closed treatment enum and ordered pipelines. | SNAP is materially better, but the supposedly schema-conforming inventory is still false. `wic` and `social_security` are not among the 13 engine take-up programs; EITC is conditioned by approximated child count, not `filer_conditioning`; and `dedicated_stage` is an untyped escape hatch for unlike SSI, ACA, and SNAP mechanisms. SSI and housing cannot be represented honestly by one enum value without forced fits. The complete inventory follows below. |
| M10 | **RENAMED-NOT-RESOLVED** | `docs/spec-engine.md:128-150,678-679` and `specs/us/bundle.yaml:4-8` repeatedly call the bundle a `CountrySpec` extension and require package data/one composition boundary. | The intended consolidation is right, and v2 now explicitly retires the old lineage file, test, and emitter. But “new resource kinds” do not exist in the current manifest or loader: resources are bare filenames (`country_spec.py:835-839`), every resource is parsed as a JSON mapping (`:849-860`), typed behavior is selected by hard-coded filenames (`:862-907`), `CountrySpec` has fixed fields (`:756-784`), and compilation only builds source/geography `StagePlan`s (`:923-996`). YAML is currently forbidden by the package contract (`test_spec_only_country_packages.py:11,44-57,80-87`). `bundle.yaml.files` also adds a second file inventory beside `country_package.json.resources`. This is a loader/manifest/dataclass/compiler replacement unless v3 defines the adapter and consumer migration. |
| M11 | **PARTIALLY RESOLVED** | `docs/spec-engine.md:659-686` adds the missing pipeline/runtime, PUF support/tail, resume/operations, calibration/selection, and default-leakage classes, plus a generated authority inventory and static allow/deny test; `:743-765` gates deletion. | The classification is much more complete. It is not yet total enough to authorize deletion because new surfaces and their old consumers have no explicit tombstones: catalog/vintage duplicates, country-specific producer constants/receipts, the two greedy splitters and max-width constant, engine ABI snapshots, and the full stochastic-callsite surface. N9 gives the required zero-reference gates. A general sentence saying an inventory assigns “every item” is not a deletion protocol. |
| M12 | **PARTIALLY RESOLVED** | `docs/spec-engine.md:184-225` defines four jointly required identity classes and exact resume interaction; `:653-657,707-714,760-763` declares an `identity_generation` cutover. | The four-class policy is coherent with the repository's independent semantic/materializer/checkpoint versions and correctly adds `builder_code_identity`. But `identity_generation` has no field location, legacy default, reader branch, discovery behavior, or promotion/logbook representation; `rg` finds no source implementation. Current routing hashes configured identity and discovery reconstructs/exact-compares base identity (`tools/build_us_multispine_pool.py:1150-1244`). “Bump” alone does not make generation 0/1 machine-decidable. |
| M13 | **RESOLVED** | `docs/spec-engine.md:314-323,460-465,674` requires schema-complete adapters, every build-facing kwarg passed explicitly, a monkeypatched-default test, and a static invocation contract. | This directly closes the current leakage from QRF defaults at production call sites (`puf_qrf_chain.py:219-225`; `acs_transfer.py:1341-1345`). Standalone library convenience defaults remain appropriately code-owned. |
| M14 | **RESOLVED** | `docs/spec-engine.md:233-256,370-386,663,677` separates normative config, identity-bound run request, output-invariant execution profile, operational bindings, and external chain state; paths are bijective/hash-verified but not identity-hashed. | This matches current logical input pins versus absolute-path receipts and cleanly classifies checkpoint roots, credentials, paths, workers, and logbook predecessor. The contradictory inclusion of execution profile in the new node key is N2, not the original source/path conflation. |
| M15 | **RESOLVED** | `docs/spec-engine.md:106-115,146-150,707-710,735-736` requires a country-neutral core plus discriminated extensions, UK two-stage/OA geography and synthetic prior to be expressible before sign-off, a minimal UK compile, and Belgian compatibility compile. | This addresses the actual UK mismatch (`uk_runtime/geography_ladder.py:1-74,96-121`; `national_frame.py:61`; `spi_support.py:29-43`) rather than promising literal reuse of US shapes. F0's size is separately wrong, but the schema direction no longer forces US PUF/FIPS/take-up conventions onto the UK. |
| M16 | **PARTIALLY RESOLVED** | `docs/spec-engine.md:630-638,649-657,724-728,743-765` names `--config-authority`, rejects interaction with `--legacy-two-spine`, requires both-mode PR fixtures, restricted cold certification, a full bundle-mode release, a cold generation cutover, and no retro-labeling. | Most lifecycle policy is now present. It is not fully implementable until M12's generation field/read protocol exists, and no retention/cleanup date is stated—only a checklist requiring one later. Without explicit absent/0/1/unknown reader rules, “generation 0 readable forever but never promotable” is prose rather than artifact policy. |
| MIN1 | **RESOLVED** | `docs/spec-engine.md:408-442` and `specs/us/geography.yaml:20-34` make `state_minus_identified_counties` the sole rule, name a pinned source id, and prohibit a sampled fallback. | The conflicting `else_state` rule is gone and the behavior is correctly postponed to F3. |
| MIN2 | **RESOLVED** | `docs/spec-engine.md:84-91` scopes closure to the selected country/profile registry namespace, always rejects unknown/duplicate referenced IDs, and permits unused library-only implementations. | This avoids failing a US load merely because UK, diagnostic, test, or general library kernels are installed. |
| MIN3 | **RESOLVED** | `docs/spec-engine.md:251-256` says failed resolution has no `spec_sha256`; failure rows carry status, attempted grammar/canonicalizer, a raw file-set digest when available, and the validation error. | This is coherent with terminal attempts being opened before validation (`tools/build_us_multispine_pool.py:3985-4015,4138-4167`). |
| NIT1 | **PARTIALLY RESOLVED** | `docs/spec-engine.md:365-368` relabels snippets “representative, schema-conforming” and says placeholders are gone or outside YAML. | The claim remains unreviewable because no schemas exist, 8 of 11 bundle files are absent, and `specs/us/take_up.yaml:48-51` still contains an ellipsis promising a future inventory. More importantly, three rows already present are factually invalid under the claimed exact engine-coverage rule. “Draft” is honest; “schema-conforming” is not yet supportable. |

### M1/M12 pressure point: where `identity_generation` must live

The same-bundle-before-branch rule now works, provided the loader runs before
`_configured_stacked_identity()` at `tools/build_us_multispine_pool.py:4192`.
The generation cutover needs one equally concrete representation:

```text
configured identity:
  identity_generation: 1
  spec_binding: {...}

stacked checkpoint base identity:
  identity_generation: 1
  spec_binding: {...}
```

`_discover_stacked_checkpoint_identity()` must pass both through when it
reconstructs the expected base identity. Stage identities and the primary-QRF /
ACS-transfer bank identities already derive from the base, so they inherit the
cutover. The same fields must be copied into outer `run_config`, checkpoint and
publication manifests, terminal attempts, and Logbook rows. Readers need a
closed rule: absent or `0` = historic/readable/non-promotable; `1` = binding
required and fully validated; unknown = refuse. Without this, discovery cannot
distinguish a genuine generation-0 artifact from malformed new output.

### M3 pressure point: a behavior-preserving `legacy-v1` boundary

A legacy declaration can preserve behavior, but it cannot truthfully expose
QRF's internal advances as independently addressable streams.

- **Spec-normative:** literal/run-request seed source; which stages/families
  share it; exact ACS label grammar, digest, truncation, and byte order; target
  order; and a versioned kernel algorithm-contract ID.
- **Kernel-contract-normative:** `SeedSequence(seed).spawn(2)`, child roles,
  bit-generator type, ordered traversal, and regime/data-dependent RNG
  consumption. These stay protected by the kernel/code digest.
- **Receipt-descriptive:** realized derived seeds, target order, saved RNG
  states, and checkpoint evidence.

The flip should pass the declared base seed unchanged to the pinned legacy
kernel. It must not replace the shared generator with per-target streams. Add
golden ACS label/seed vectors and a full multi-regime QRF chain fixture.
`derived-v2`, not `legacy-v1`, is where streams become stateless declarations.

### M7 pressure point: minimum lossless producer graph

A compilable graph needs, at minimum:

- graph-level external stages and the scope-coverage relation currently hashed
  in `late_producer_dag.py:373-400`;
- each input's `entity`, `column`, `required_scope`, `producing_stage`, ordered
  tolerated-absence receipt IDs, and alternatives as OR-of-AND lists of
  `{entity,column,value_kind}`;
- outputs `{entity,column,coverage_scope}`;
- typed virtual resource declarations for manifests, resolved weights,
  execution configs, transition/producer receipts, and target banks, including
  each kind's semantic payload and digest rules;
- transfer groups and the full overlap matrix keyed by target, origin, and clone
  role, with finalization plus every non-owner action; and
- the execution-receipt and transition-authority contracts in
  `us_late_producer_schedule_payload()` (`us_late_producer_registry.py:2047-2103`).

The RFC's example at `docs/spec-engine.md:508-516` must either be replaced with
that shape or be labeled non-normative pseudocode. In its present form the
promised compile-back equality is impossible.

### M9 pressure point: actual take-up inventory

The checked-in contract has exactly 13 programs
(`us/take_up_contract.json:17-207`); tests require exact equality with installed
engine metadata (`test_us_take_up_contract.py:33-38,152-168`). The actual
mechanisms across that contract, `source_stages.json`, and runtime modules are:

| Actual program(s) | Current contract label | Actual production mechanism | Honest v2 representation |
|---|---|---|---|
| SNAP | `out_of_scope` | National reported-anchor/FNS-rate prior, then reporter-anchored state count calibration that overwrites the flag (`source_stages.json:1950-2036`). | Ordered two-stage pipeline; final semantic treatment can be `anchored_count_calibrated`. FNS counts are targets, not the “anchor” claimed at `take_up.yaml:32`. |
| TANF | `seed` | Generic stable-ID Bernoulli kernel at a scalar administrative rate (`take_up.py:242-294,307-386`). | `seeded_rate`; batching is a separate invocation-group property. |
| EITC | `seed` | Same kernel, with rate selected by approximated qualifying-child count (`take_up.py:200-239,262-269`). | `seeded_rate` + `rate_by_num_children`; not `filer_conditioning`. |
| Medicaid | `count_calibrated` | Reporter anchor, runtime state prior, greedy state count calibration (`source_stages.json:2788-2836`). | `anchored_count_calibrated`. |
| CHIP, Basic Health Program, DC PTC, Early Head Start | `rate_unsourced` | The engine's default-true leaf survives, with explicit source/debt follow-up. | `engine_default_with_debt`. |
| Medicare | `out_of_scope` | Measured ASEC `MCARE` mapping and support-clone propagation (`source_stages.json:2068-2109`). | `measured` pipeline. |
| SSI | `count_calibrated` in the contract | Reporter-anchored, target-derived age-band Bernoulli prior plus delivered-recipient gate; it explicitly **never count-matches flags** (`take_up_contract.json:121-145`; test at `test_us_take_up_contract.py:86-116`). | Needs a typed `target_derived_seeded_with_delivery_gate` operation/pipeline. `anchored_count_calibrated` is false. |
| Head Start | `out_of_scope` | Weighted QRF trained on direct SIPP response and transferred to frame identities (`source_stages.json:1357-1486`). | `imputed_transferred`, with donor/training/recipient scopes. |
| Housing assistance | `out_of_scope` | Measured ASEC receipt on one row surface and ACS/QRF-imputed receipt on PUF support; the flag equals the receipt (`source_stages.json:2390-2530`). | Mixed row-scope `measured` + `imputed_transferred`; one whole-column treatment is insufficient. |
| ACA | `out_of_scope` | Dedicated Marketplace assignments and several calibration operations (`source_stages.json:2680-2785`). | A typed dedicated pipeline, not a bare escape-hatch label. |

No current contract program is `near_universal` or `model_simulated`. WIC and
Social Security are not contract programs. `batched_seeded` mixes execution
grouping with semantic treatment, and `dedicated_stage` makes the supposedly
closed enum open-ended. V3 should separate engine class/debt status, semantic
pipeline operations, row-scope ownership, and kernel invocation group, then
commit all 13 real rows. Column mapping must come from an engine ABI projection,
not the insufficient optional-suffix naming rule (`takes_up_eitc` and
`takes_up_dc_ptc` already show why).

### M10 pressure point: this is a replacement unless the seam is explicit

Current `CountrySpec` combines four assumptions:

1. one flat `country_package.json.resources: [filename]` manifest;
2. JSON-object parsing for every declared resource;
3. filename-specific typed fields on the `CountrySpec` dataclass; and
4. source/geography-only `StagePlan` compilation.

V2 adds another file manifest, YAML parsing, closed schemas/default injection,
arbitrary kernel registries, catalogs/vintages, a typed execution IR, node keys,
and canonical typed hashing. Calling that “new resource kinds” does not define
an extension: the existing manifest has no kind field. Minimal coherent wording
is:

- replace the bare filename list with one authoritative typed resource table
  `{path,kind,schema_id}`, or make the versioned bundle manifest itself replace
  `country_package.json`; do not keep both file lists;
- return one `ResolvedCountrySpec` carrying both migration-era compatibility
  projections (`sources`, `gates`, etc.) and the compiled spec-engine IR;
- intentionally update the spec-only package test to allow YAML and kernel IDs;
- define which generation exposes raw `fingerprint`, which exposes
  `spec_sha256`, and migrate every validator/receipt/attestation consumer. In
  particular, `gate_battery.py:787-792,958-983` currently emits a
  `spec_fingerprint` composed only from `gates.json`, while
  `docs/gate-battery-contract.md:89-90` calls it the whole country package; and
- list direct runtime consumers such as `load_take_up_contract()` and UK
  `load_country_spec("uk").gates` (`uk_runtime/national_build.py:467-472`) in the
  compatibility/deletion plan.

V2 does get one part right: it explicitly retires the old lineage surface and
its consumers. The active direct consumers are
`packages/microcosm-build/tests/test_imputation_lineage_spec.py:31-124` and
`tools/emit_lineage_dashboard.py:21-83` (plus the stale package comment at
`packages/microcosm-build/pyproject.toml:20-22`). They must be replaced, not left
pointing at the retired file.

## 2. New v2 findings

All new findings below are **MAJOR**. There are no standalone new MINOR or NIT
findings: each defect can invalidate identity, equivalence, the authority flip,
or the promised fast path.

### MAJOR N1 — F0's “fast mirror” contains F1's compiler/compile-back work

**Evidence.** F0 promises closed schemas, the canonicalizer, one typed
`ResolvedSpec`, a full constants-generated US bundle, UK and Belgian compile
compatibility, the `CountrySpec` loader/wheel rewrite, both authority modes,
the identity cutover, a complete legacy seed map, and byte equality of all
resolved constants/bundle payloads (`docs/spec-engine.md:707-718`). F1 is then
said to introduce the producer graph, generic executor, compile-back, and
bundle-built authorities (`:719-723`). Those boundaries are circular:
high-level references, defaults, graph topology, ownership, and kernel config
can be compared to today's low-level plan/schedule payloads only after the
common typed IR and adapters/extractors exist. A textual YAML comparison would
not prove resolved equality.

The current mirror is deliberately narrower. It checks ordered family/target
parity, predictor sets, a few model/default values, and computed producer
outputs (`test_imputation_lineage_spec.py:43-112`); it is not a complete plan
compiler. The existing `CountrySpec` incompatibilities in M10 make F0 larger
again. Likewise, `spec_sha256` cannot truthfully answer “what built this” at
`docs/spec-engine.md:715` while constants outside the compared subset still
drive behavior.

**Rollout failure.** Either F0 balloons into F1 and blocks the CHAMPVA-class
work it exists to unblock, or F0 certifies a partial mirror while attaching a
whole-build hash and identity generation to unbound behavior. The latter is
worse: it gives a false provenance claim.

**Concrete fix.** Define an honest F0a with one of two orders:

1. Preferred: land the predictor/CHAMPVA value change first under today's #695
   mirror plus its required OOS/statistical gate, then generate the legacy
   baseline from that commit.
2. If schema work must land first, F0a only packages/parses a narrow
   `PredictorMirrorPayload` covering exact ordered families/targets,
   predictor columns/blocks, and every model argument used by the held edit.
   CI compares that payload to constants. Constants remain explicitly
   authoritative and the digest is descriptive, not `spec_sha256`.

Move the full `ResolvedCountrySpec`, all-file canonical bundle hash,
`plan.lock.json`, CountrySpec migration, global identity cutover, full seed
inventory, derived closure, dual authority, and plan/schedule/ownership
compile-back to F1. Alternatively rename the current F0 “compiler binding,”
estimate it as F1-sized, and stop advertising it as the value-fix fast path.

### MAJOR N2 — The node key overkeys output-invariant profiles and underkeys run values

**Evidence.** The node formula includes “execution profile class”
(`docs/spec-engine.md:211-214`), while `:243-245` defines execution profiles as
receipted and proven output-invariant. Conversely, root seed, rung/sample
fraction, `k`, and release label are identity-bound run-request values
(`:240-242`), but the node formula names only a `seed stream id`, not the root
seed or a value-bearing run-request digest. A stream identifier names a
derivation function; it does not distinguish seed 1 from seed 2. `plan.lock`
is described as compiled bundle IR (`:208-210`), so it cannot be assumed to
contain per-run values.

Current checkpoint identities bind exact values: period/model seed, sample
fraction and seed, clone fraction and seed, engine, and inputs are explicit in
`tools/build_us_multispine_pool.py:1067-1093`; the configured namespace binds
the same run controls at `:1150-1165`.

**Rollout failure.** Different stochastic runs can share a node key and resume
one another, while harmless worker/device/batch profiles unnecessarily split
cache namespaces. “Runtime lock” is too vague to repair the omission; it also
does not distinguish semantic dependency/code identity from operational
scheduling compatibility.

**Concrete fix.** Define two records:

```text
semantic node key = H(
  identity_generation,
  grammar + canonicalizer,
  resolved semantic node plan,
  exact consumed run-request values (including root seed),
  direct input content hashes,
  kernel ABI + implementation/output-affecting dependency digest,
  seed stream id,
  artifact/materializer + output contract
)

attempt/scheduling receipt = {
  semantic_node_key,
  output-invariant execution profile,
  operational bindings
}
```

If a device, batch size, worker count, or backend can change bytes, reclassify
that exact field as semantic/code identity and include it; do not include a
vague profile “class.” Golden tests should prove both that changing root seed
changes the key and that changing a certified output-invariant worker count
does not.

### MAJOR N3 — The frozen split ledger is wrong, and F3 reunification changes more than RNG order

**Evidence.** V2 says today's
`puf_tax_itemization__batch_1..4` are 32 targets
(`docs/spec-engine.md:479-483,732-734`). The committed mirror has **five**
batches: four groups of eight and `batch_5` with five—**37 targets**
(`specs/us_imputation_lineage.yaml:280-356`; the operator documentation confirms
all five at `docs/us-multispine-operator-ordering.md:724-728`). The overall
registry requires 19 bounded groups/70 targets
(`us_late_producer_registry.py:1393-1404`).

The split is reproducible given today's ordered input and width: runtime
normalization preserves target order and sorts family names, then greedily
packs atoms while keeping the immigration pair together
(`acs_transfer.py:2268-2380`). But it is not yet an independent declaration:
the registry duplicates the splitter (`us_late_producer_registry.py:1338-1390`)
and ownership literals name particular batches
(`us_late_overlap_ownership.py:29-33`). Those copies can co-drift.

Reunification has at least four behavioral effects:

- the family label changes `_family_seed`/`_pattern_seed` for every target
  (`acs_transfer.py:2902-2916`);
- QRF's shared RNG is consumed across a different target sequence
  (`qrf.py:1077-1098,1333-1380`);
- the donor complete-case mask is now the intersection across all 37 targets,
  not per old batch (`acs_transfer.py:1260-1270,1427-1439`); and
- later targets can condition on drawn targets across former batch boundaries
  (`qrf.py:1523-1533`).

**Rollout failure.** Literal implementation drops five live outputs before any
intended behavior edit and breaks registry/ownership equality. Treating F3 as
only “RNG consumption order” under-scopes donor selection, chained features,
bank identities/layout, and overlap-owner references.

**Concrete fix.** At the flip, declare and golden the entire current ordered
split ledger—19 groups/70 targets overall and five/37 for tax itemization—with
an explicit reason for every frozen split. Compile it back against both current
splitter outputs and every literal owner/receipt consumer, then delete the two
greedy production splitters and width constant at F2. At F3 declare the full
37-target chain, assign fresh node/bank/materializer identities, and require
OOS/statistical tests that cover the new donor complete-case population,
family-label seed change, and cross-boundary chained predictors.

### MAJOR N4 — Take-up derive-and-assert is circular on an engine bump

**Evidence.** V2 removes authored `column`/`entity`, derives them from a naming
rule plus installed engine metadata, and asserts coverage against that same
installed metadata (`docs/spec-engine.md:530-557`; `specs/us/take_up.yaml:12-15`).
Today the checked-in contract is intentionally a reviewed snapshot of engine
facts: entity/value type/default/class keys are explicit
(`take_up_contract.py:49-58`), `assert_take_up_contract_current()` compares
every field and set against the installed engine (`:327-391`), and tests prove
the equality and its ability to fail (`test_us_take_up_contract.py:33-38,
152-168`). Deriving facts and then asserting them against their own source
removes that tripwire: existing variables whose entity/default/class changes
would be silently accepted unless another snapshot remains.

The version authority is already split. Project metadata permits
`policyengine-us>=1.745.0,<2` (`packages/microcosm-build/pyproject.toml:30`), the
lock resolves 1.764.6 (`uv.lock:1364-1365`), the current contract records a
1.752.2 review vintage (`take_up_contract.json:5-9`), and v2 places 1.764.6 in
both sources and vintages (`docs/spec-engine.md:361-363,382-383`).

**Rollout failure.** On an engine bump, either the compiler silently changes
the resolved take-up ABI, or unrelated duplicate pins fail in an undefined
order. A newly discovered default-true flag can ship without treatment if the
derived inventory is mistaken for reviewed bundle content. The optional-suffix
naming rule is also not a bijection for actual names such as `takes_up_eitc`
and `takes_up_dc_ptc`.

**Concrete fix.** Choose one exact engine artifact/version pin. From precisely
that pin, compile a complete engine ABI projection
`{program_id -> variable, entity, type, default, engine_class, consumers}` into
a generated, committed, reviewable lock. CI compares a fresh derivation to the
lock before bundle compilation. Engine bumps require an explicit pin/ABI-lock
refresh and human review of all bundle-owned treatments/pipelines; added,
removed, or changed facts fail closed first. The engine owns its facts, the
bundle owns treatments and scope pipelines, and neither is derived from the
other. Other files refer to the one engine pin by ID.

### MAJOR N5 — Catalogs and vintages recreate the drift pairs v2 claims to dissolve

**Evidence: catalogs.** `docs/spec-engine.md:327-335` says closure owner/class
derive from bundle outputs and catalogs are human-facing. Lines `:356-359` then
author `owner` and `lineage class` in catalogs and make closure reporting derive
from them. That is a second `column_lineage.yaml` under another name. It also
conflicts with the repository rule that entity/dtype/period metadata come
through the RulesEngine adapter, never per-tool guesses (`DESIGN.md:71-72,
82-89`).

**Evidence: vintages.** `docs/spec-engine.md:361-363` calls `vintages.yaml` the
one place for engine, geography, and “2024,” but the same RFC pins the engine in
`sources.yaml` (`:382-383`) and assigns the engine pin to sources in the
migration table (`:663`). The geography draft repeats `cd119`
(`specs/us/geography.yaml:35-40`). Current code has `POOL_TIME_PERIOD = 2024`
(`multispine_pool.py:238-242`), 2024 release parsing/IDs
(`tools/build_us_multispine_pool.py:290-293,1288-1292`), and
`CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE = "119th_congress"`
(`congressional_district_vintage.py:16-25`). Source artifacts also carry their
own factual survey years, which should not be overwritten by one global year.

**Rollout failure.** Catalog ownership can disagree with producer-graph
ownership, and an engine/year/geography change can update only one of
sources/vintages/code. If prose is part of `spec_sha256`, a spelling edit can
invalidate builds; if it is not, the schema still needs to distinguish the
non-normative overlay.

**Concrete fix.** Authored catalog rows are documentation overlays keyed by a
compiler-derived stable column key: description, citations, and only display
units that are not engine metadata. The compiler injects entity, dtype/period,
presence profile, owner, row-scope segments, and lineage class. Hash prose in a
separate documentation digest, not normative `spec_sha256`.

Make vintages a normalized reference/index over content-pinned source records:
survey/tax/geography vintages live on the source artifact they describe, and
other specs refer to a vintage/source ID. Put engine compatibility in exactly
one runtime/source-lock record and assert the installed engine against it.
Derive publication period from the dataset/run contract. Add a CI check that
rejects duplicate literal authorities for each normalized key.

### MAJOR N6 — Three stage H5 hashes do not cover the bundle's claimed behavior, and resume-forbidden is underspecified and non-fail-fast

**Evidence: output coverage.** The three durable stages are only `assembled`,
`transferred`, and `simulated` (`multispine_pool.py:200-201`). The transferred
checkpoint precedes derive/seed/simulate (`tools/build_us_multispine_pool.py:
2947-3175`); the simulated checkpoint is written at `:3182-3250`; terminal
gates run afterward at `:3264-3275`. V2 separately compares normalized gates,
which is good, but final publication is still outside the SHA vector: it uses a
different H5 materializer plus publication run ID and writes the final manifest
and diagnostics later (`tools/build_us_multispine_pool.py:3664-3748`). The
manifest explicitly says calibration is downstream (`:3507-3512`), even though
calibration/selection/publication are bundle surfaces in v2.

Operational/model-bank evidence is deliberately excluded from canonical stage
H5 metadata: `_split_checkpoint_stage_receipts()` removes primary-QRF resume
status/routing and ACS target-bank receipts to a sidecar
(`tools/build_us_multispine_pool.py:2103-2152`). Thus H5 equality cannot prove
that both paths executed cold.

**Evidence: enforceability.** Existing receipts are sufficient for a post-run
no-resume assertion, but v2 neither enumerates the required assertion nor
provides a fail-fast policy. Stage provenance exposes
`deepest_resumed_stage` and per-stage source
(`tools/build_us_multispine_pool.py:1614-1714`); primary QRF exposes an
aggregate `initialized|resumed` status (`:2326-2368,2462-2470`); and ACS target
banks expose per-target `load_status: resumed` / `source: checkpoint`
(`acs_transfer_bank.py:107-228,337-377`). The aggregate primary status is
adequate: the chain resumes a contiguous prefix when a manifest exists, and it
rejects a nonempty manifestless bank (`puf_qrf_chain.py:254-312`). A harness
can therefore require null `deepest_resumed_stage`, primary
`resume_status == initialized`, and no resumed/checkpoint-sourced ACS target.
What is missing is one specified policy/audit spanning those receipts and a
pre-load refusal; empty roots are only convention.

**Rollout failure.** Bundle and constants modes can agree at the three pool
cutpoints while differing in final serialization/manifest behavior or a later
bundle-owned calibration/selection/release node. An implementer can also check
only stage provenance and miss model-bank reuse because “fail if either run
reports a resume” does not define the three-receipt assertion above. The gate
is enforceable today, but not specified as an executable predicate.

**Concrete fix.** Define an equivalence artifact vector, scoped explicitly to
every node under bundle authority: canonical logical digest of every stage
frame; final published logical frame/schema/period/materializer; normalized
final manifest, diagnostics, and gates; compiler lock files; and downstream
calibration/selection/release artifacts when those are in the flip. Compare raw
H5 SHA only where the same deterministic materializer and normalized embedded
metadata are guaranteed.

For F0, spell out the exact post-run predicate over the existing receipts:
null `deepest_resumed_stage`, primary `resume_status == initialized`, and no ACS
target with resumed/checkpoint source. Then add `--resume-policy=forbid` and
propagate it through stage discovery, primary QRF, and every ACS bank. In forbid
mode, reject any pre-existing manifest/stage/target before loading and emit one
typed `resume_audit` with per-stage and per-target attempted/resumed counts;
equivalence requires every count to be zero. The flag is fail-fast hardening,
not a prerequisite for enforcing the current post-run gate.

### MAJOR N7 — The wave-1 predictor and closure lane order is internally inconsistent

**Evidence: predictor lane.** The held `per-family-predictor-sets` branch is not
only a YAML mirror edit. It widens ACS PUMS required inputs with dozens of raw
fields
(`per-family-predictor-sets:packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72-131`)
and passes the resulting person table into the Frame
(`per-family-predictor-sets:packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:297-341`).
It materializes 15 canonical carried predictor columns
(`per-family-predictor-sets:packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:111-135,617-666`),
and its pool test asserts they remain on the person table
(`per-family-predictor-sets:packages/microcosm-build/tests/test_us_multispine_pool.py:3670-3688`). Final H5
serialization writes every Frame table/column (`us_runtime/h5_io.py:917-927`).
The lane also installs a new participation target order
(`per-family-predictor-sets:packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:640-658,805-825`), changes
alternative precedence from sorted to declaration order
(`per-family-predictor-sets:packages/microcosm-build/src/microcosm/build/late_producer_dag.py:111-117`), and bumps the late
registry schema
(`per-family-predictor-sets:packages/microcosm-build/src/microcosm/build/us_late_producer_registry.py:102-130`).
Its byte-equal mirror therefore proves synchronization with code,
not that the behavioral edit is acceptable. Its own lane notes say no f025 OOS
sweep or artifact build was run
(`per-family-predictor-sets:_predictor-sets-LANE-NOTES.md:6-10,29-33`).

**Evidence: closure lane.** The held `lineage-column-closure-697` branch pins a
392-column f025 inventory
(`lineage-column-closure-697:packages/microcosm-build/tests/test_lineage_column_closure.py:54-79`). A direct set
comparison finds 56 raw/canonical predictor names introduced by the predictor
branch absent from that fixture. Live-H5 comparison is opt-in and normally
skips without `MICROCOSM_LINEAGE_POOL_H5`
(`lineage-column-closure-697:packages/microcosm-build/tests/test_lineage_column_closure.py:265-272`). More fundamentally, the
closure test reads the authored lineage file, dynamically imports the retiring
dashboard emitter, and reasserts the old take-up contract
(`lineage-column-closure-697:packages/microcosm-build/tests/test_lineage_column_closure.py:20-46,81-185,187-237`).
That is the opposite of v2's compiler-derived closure.

**Rollout failure.** Landing closure at F0 and predictors after F0 either makes
the first value edit immediately fail the frozen profile or silently leaves the
fixture stale. Transplanting the closure branch as-is also preserves the three
authorities v2 says F0 dissolves.

**Concrete fix.** Land the predictor/CHAMPVA work before freezing the F0
baseline, with its real OOS/statistical acceptance gate. Then either:

- declare/catalog its added columns, regenerate a certified inventory, and
  treat the change as artifact-profile/materializer work; or
- make predictor features typed transient/virtual resources and prove the
  executor strips them before checkpoints/final H5.

Only `emit_artifact_column_inventory.py` and the versioned inventory fixture
from #697 can land unchanged. Retarget exact closure to compiler-emitted
profiles/segments and compiled take-up after the full compiler exists in F1.

### MAJOR N8 — The held block-first lane does not contain the owner-ruled ASEC complement

**Evidence.** The held branch explicitly records that the v3 ASEC checkpoint
has only `household_id`, `state_fips`, and `H_TENURE`, then chooses state-only
fallback (`block-first-geography-696:_696-LANE-NOTES.md:33-40`). Its kernel says
ASEC rows draw within the state because identified county is unavailable
(`block-first-geography-696:packages/microcosm-build/src/microcosm/build/us_runtime/geography_ladder.py:1-7,326-345`)
and receipts `asec: state` / `asec_county_status: absent_from_v3_checkpoint`
(`block-first-geography-696:packages/microcosm-build/src/microcosm/build/us_runtime/geography_ladder.py:494-516,596-602`).
The checkpoint loader actively rejects any county field until a dedicated
schema change
(`block-first-geography-696:packages/microcosm-build/src/microcosm/build/us_runtime/asec_checkpoint.py:179-182,207-229`).
The branch therefore cannot implement v2's
`state_minus_identified_counties` rule (`specs/us/geography.yaml:20-34`).

It is also not “bundle content,” as `docs/spec-engine.md:738` says: the lane
adds a new block artifact schema/source pin, assignment/validation kernel,
checkpoint identity fields, and source/checkpoint plumbing.

**Rollout failure.** Merging the held branch at F3 would violate the revised
owner ruling by sampling unidentified ASEC households over the whole state,
including counties they are known not to inhabit. Treating it as config-only
would omit required artifact and checkpoint-schema changes.

**Concrete fix.** Keep F3, but describe this as a coordinated kernel + ASEC
checkpoint-schema/source + block-artifact + bundle migration. Add a bound
official identified-county source, county-identified and state-complement
universes, refusal for empty complements, leakage/complement tests, new
identity/materializer versions, and cold certification. Rework/rebase the held
branch; do not merge it as the implementation of the final YAML.

### MAJOR N9 — The F2 deletion checklist lacks zero-reference gates for the new surfaces

**Evidence.** `docs/spec-engine.md:743-765` gives good global conditions but no
specific tombstones for several authorities introduced or absorbed by v2:

- **Country composition:** raw `CountrySpec.fingerprint`, the gate-battery
  `spec_fingerprint`, bare manifest filenames, and old JSON compatibility
  projections need an explicit generation/consumer migration (M10).
- **Lineage/dashboard:** current consumers are
  `test_imputation_lineage_spec.py:31-124` and
  `tools/emit_lineage_dashboard.py:21-83`; held predictor/closure branches add
  more tests against that file. The package comment at
  `packages/microcosm-build/pyproject.toml:20-22` remains stale, and the external
  dashboard handoff is not verified in this repo.
- **Producer graph:** country-specific `CANONICAL_US_LATE_*`, schedule/ownership
  receipt constructors, and their tool/runtime imports survive. For example,
  the stacked tool consumes the schedule at
  `tools/build_us_multispine_pool.py:1105,2621,3015`; the generic
  `ProducerInput`/`ProducerOutput`/DAG validators should remain, but the US
  declaration constants must not.
- **Frozen splits:** both greedy splitter implementations and
  `DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT = 8` remain live
  (`acs_transfer.py:88-91,939-942,2338-2380`;
  `us_late_producer_registry.py:1338-1396`).
- **Take-up:** `take_up_contract.json`, its loader/currentness assertions, and
  the absorbed `source_stages.json` rows remain direct authorities until every
  runtime consumer uses compiled bundle projections.
- **Seed protocol:** v2's summary of 578/0/42 plus ACS/QRF is not an exhaustive
  production callsite inventory. Reachable source stages also own SSI training
  and model seeds (`ssi_disability_criteria.py:241-243`, duplicated in
  `source_stages.json:1331-1337`), vehicle/asset stable-string hash algorithms
  (`sipp_vehicles.py:299-314`; `sipp_financial_assets.py:307-322`), the ACS-rent
  archived hash (`housing_inputs.py:740-756`), the tips training seed
  (`sipp_tips.py:114-120`), and the SCF composite `SeedSequence`
  (`scf_wealth.py:830-855`). A central map would become a third copy if those
  literals remain.
- **Catalogs/vintages:** existing code/docs pins must be removed or converted to
  references after the one-authority model in N5 exists; root `specs/us`
  drafting copies must disappear after package-data migration.

**Rollout failure.** F2 can satisfy the eight broad sentences while leaving old
runtime loaders/constants active. The build then still has two authorities even
though the selector and mirror tests—the mechanisms most likely to expose the
drift—have been deleted.

**Concrete fix.** Make F2 machine-decidable with generated inventories and
zero-reference/tombstone tests:

1. every package resource/file belongs to exactly one typed manifest and only
   the generation-appropriate composition identity is emitted;
2. repository search finds no nonhistorical reference to the root lineage file,
   old emitter, or old dashboard schema; the external dashboard is verified
   against the compiled catalog/export;
3. no production import/reference to country declaration constants, old
   ownership/schedule receipts, greedy splitters, or the width constant remains;
4. every reachable stochastic/hash-draw callsite consumes a resolved stream
   token, with code retaining only versioned algorithm implementations and
   golden vectors; no build seed/default literal remains outside the audited
   legacy-kernel contract;
5. no direct take-up-contract/source-stage authority survives outside the
   compatibility reader for generation 0; and
6. no duplicate engine, period, geography-vintage, catalog-owner, or lineage
   literal survives. Generated lock files are regenerated outputs, never a new
   authored authority.

## 3. Rollout-order adjudication

The current phase labels are not credible as written.

| Work item | Actual size/home | Required order |
|---|---|---|
| Narrow predictor/model mirror sufficient for CHAMPVA-class edits | Honest F0a | Land the held predictor behavior change first (with OOS/statistical acceptance), or land only the narrow mirror schema/parser and keep constants authoritative. |
| Full schemas + canonical `ResolvedCountrySpec` + US/UK/BE compatibility + one CountrySpec manifest/loader + all-payload equality | F1-sized compiler binding | Must precede global `spec_sha256`, bundle authority mode, and the identity-generation cutover. It cannot be split from compile-back by calling the comparison a mirror. |
| `plan.lock.json`, producer graph, generic executor, plan/schedule/ownership compile-back | F1 | Compile-back fixtures must exist before bundle mode can construct authorities. |
| `legacy-v1` seed contract | F1 unless F0 is renamed compiler-sized | Build the reachable stochastic-callsite inventory and golden legacy-kernel vectors before claiming full equivalence. |
| #697 inventory tool/fixture | F0a, **after** predictor materialization is settled | Rebuild/certify the fixture against the post-predictor artifact profile. |
| #697 derived closure/segments/dashboard | F1 | Retarget to compiler outputs; do not land the authored-class test/emitter as held. |
| Held predictor branch | First behavior lane, but not a byte-equality acceptance proof | Treat loader columns, retained artifact columns, target order, kernel changes, and profile/materializer effects explicitly; run its promised OOS/statistical gate. |
| Held block-first branch | F3 coordinated behavior/artifact/schema change | Rework it to include identified-county/complement semantics; it is not merely a bundle diff. |
| Five-to-one tax-itemization chain | F3 | Correct the frozen baseline to five/37 first, then gate the full donor/seed/chain/bank behavior change. |

Accordingly, the advertised order “closure at F0; predictor right after F0” is
reversed at the artifact boundary. The inventory baseline cannot precede a lane
that adds persistent columns unless those columns are declared transient and
removed before every checkpoint/publication surface.

The deletion checklist is **not complete**. N9's zero-reference gates should be
added as explicit F2 conditions, alongside these final cleanup requirements:

1. one typed country resource manifest; no second `bundle.files` inventory;
2. an explicit generation-0 compatibility reader and generation-1
   `ResolvedCountrySpec`, with every fingerprint/spec-hash consumer assigned;
3. no root drafting bundle after package-data installation;
4. old lineage file/emitter/tests and external dashboard schema migrated;
5. old take-up contract/source rows and direct loaders migrated or retained only
   in the generation-0 reader;
6. US producer/schedule/ownership constants, splitters, and max-width constant
   absent from production;
7. catalog/vintage/pin literals reduced to one authority plus references;
8. exhaustive seed-callsite inventory at zero unbound callsites;
9. authority selector retained through a certified full release and deleted only
   after a dated retention deadline; and
10. generated bundle/plan/ABI locks reproducible from their authorities and
    rejected if hand-edited.

## 4. Final verdict and minimal v3 change list

**V2 is not sign-off-ready for the owner.** Its architectural direction is now
substantially better than v1, but following the current document can still
produce a false equivalence proof, stale stochastic cache reuse, dropped live
targets, an engine-sensitive take-up plan with no review tripwire, and a
CountrySpec implementation that is a rewrite hidden behind the word
“extension.”

The minimal v3 change list is:

1. **Make F0 honest.** Either reduce it to a narrow, non-authoritative predictor
   mirror or rename/estimate it as the full compiler-binding phase. Do not attach
   whole-build `spec_sha256`, identity generation, or dual authority before the
   full compile-back surface exists.
2. **Specify the CountrySpec replacement seam.** Use one typed resource manifest
   and one file inventory; define YAML/package tests, `ResolvedCountrySpec`
   compatibility projections, and the complete raw-fingerprint → canonical-spec
   consumer migration (including gate battery and UK/BE readers).
3. **Put identity on concrete fields.** Add `identity_generation` and
   `spec_binding` to configured and base identities and all readers/receipts;
   define absent/0/1/unknown behavior. Rewrite the node key to include exact
   consumed run-request values/root seed and exclude output-invariant execution
   profiles.
4. **Draw the legacy RNG boundary.** The spec owns seed sources/sharing, label
   grammars, target order, and algorithm IDs; pinned kernels own internal
   consumption. Commit an exhaustive stochastic-callsite ledger and golden
   ACS/QRF/source-stage vectors.
5. **Replace the producer-graph sketch with a genuinely lossless schema.** Cover
   nested alternatives/value kinds, producing stages, receipt IDs, virtual
   resource payloads, scope coverage, transfer groups, the full conditional
   ownership matrix, execution receipts, and transition authority. Require a
   golden byte-identical compile-back fixture.
6. **Replace the take-up examples with all 13 real programs.** Separate engine
   ABI facts, semantic pipeline operations, row-scope ownership, and invocation
   grouping; add an exact pinned-engine ABI lock and fail engine bumps before
   plan compilation. Do not use `dedicated_stage` as an untyped escape hatch.
7. **Correct split facts to five/37 (19/70 overall).** Freeze the exact current
   ordered groups as declarations. Describe F3 reunification as a donor-mask,
   family-seed, shared-RNG, chained-feature, owner, and bank-identity change—not
   only a consumption-order change.
8. **Remove new dual authorities.** Catalogs are non-normative documentation
   overlays; owner/class/entity/dtype/period are compiled. Vintages live on
   pinned source records and are referenced by ID; engine compatibility appears
   exactly once.
9. **Expand the equivalence contract.** Compare the declared vector through
   final published logical H5/manifest/diagnostics and every downstream node in
   flip scope. Add an enforced end-to-end `resume-policy=forbid` and one complete
   per-stage/per-target resume audit.
10. **Correct the lane map.** Settle/validate predictor behavior before freezing
    the inventory; move derived closure to the compiler phase; rework #696 for
    the ASEC complement and identify it as coordinated code/artifact/schema
    work.
11. **Make F2 deletions machine-decidable.** Add the explicit tombstones and
    zero-reference/unbound-callsite gates in N9, plus a dated authority-selector
    retention deadline.
12. **Stop calling drafts schema-conforming.** Commit schemas and a complete US
    bundle (including all take-up rows and correct split ledger), or label all
    incomplete excerpts pseudocode until those artifacts exist.

With those amendments, the owner could sign off the architecture before the
implementation exists. At `d865ba40`, the document still asks the owner to sign
off several invariants that its shown schemas and rollout cannot satisfy.
