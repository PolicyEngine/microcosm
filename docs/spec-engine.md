# The spec engine: one declared bundle drives the build

Status: **draft for Max's review** (2026-08-15). Nothing here is wired;
`specs/us/*.yaml` in this branch are skeleton drafts of the target shape.

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

## Principles

1. **Total, or failing.** The bundle must account for every column of the
   artifact (the #697 closure test survives the flip unchanged). An
   unclaimed column, an unreferenced kernel, or a dangling cross-reference
   fails the load, not a review.
2. **Declarations carry mechanisms, not just names.** Every stochastic
   write declares its draw universe, conditioning, parameter source, and
   seed derivation. Stage topology alone is what let the geography flaw
   hide.
3. **One hash.** The canonical bundle content hash (`spec_sha256`) binds
   into checkpoint identity and every logbook row, replacing scattered
   version constants as the primary "what configuration built this"
   answer. Authority *semantic* versions remain for contract escalations.
4. **Kernels are the only escape hatch.** A kernel is a registered
   callable with a declared io signature (`inputs`, `outputs`, `params`).
   Spec entries reference kernels by id; the registry refuses unknown ids;
   closure refuses kernel outputs the spec doesn't claim. Per-variable
   custom logic lives here — pointed at, never hidden.
5. **Country files instantiate a shared schema.** `specs/schema/*.json`
   (JSON Schema) validates any country bundle; `specs/us/` is the first
   instance, `specs/uk/` the second (#578 one-suite-per-country).
6. **The dashboard reads the same bundle the build reads.** Drift between
   page and build becomes impossible by construction, ending the
   emitter-regeneration step.
7. **Derive, don't declare.** A field that is a deterministic function of
   another authority — engine variable metadata, a naming rule, the block
   geoid, a contract row — is derived by the loader and asserted, never
   independently declared. Every independent second copy is a future
   drift. (Examples: take-up column names and entities derive from the
   program + contract; county/tract/PUMA derive from the block; the
   assertion failing is a load refusal.)

## Bundle layout

```
specs/
  schema/            # JSON Schema per file kind + bundle cross-ref rules
  us/
    bundle.yaml      # manifest: file list, schema_version, country, root_seed
    sources.yaml     # every external input, sha-pinned
    spine.yaml       # channels, assembly, clones, mass shares
    geography.yaml   # block-first assignment (#696 ruling)
    imputation.yaml  # families × targets × predictor blocks × models
    take_up.yaml     # every takes_up_* draw: mechanism + parameter source
    battery.yaml     # completeness surface + by-origin legs + verdicts
    calibration.yaml # loss, bounds, target refs, CD attainment reporting
    selection.yaml   # sparse / exact-k
    publication.yaml # release grammar, rungs, logbook, distribution
    column_lineage.yaml  # closure classes over the artifact inventory
```

`bundle.yaml` is the root: the loader reads it, loads every listed file,
validates each against its schema, resolves cross-references
(predictor-block ids, kernel ids, source ids, target refs), and computes
`spec_sha256` over the canonical concatenation. Any failure is a refusal.

## File schemas (shape + real excerpt each)

### sources.yaml — inputs with pins

Replaces the launcher's `INPUT_SHAS` and scattered path plumbing. Every
external artifact: id, role, locator (documentation), sha256, and the
loader kernel that parses it.

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
  - id: us_block_ladder_2020
    role: geography_ladder
    sha256: <schema-v2 artifact sha from the #696 lane>
    loader: kernel:load_us_block_ladder
  # … puf_2024, puf_2015_csv, acs_2022_rent, csv_hus …
engine:
  policyengine_us: "1.764.6"            # pinned simulation surface
```

### spine.yaml — channels and assembly

```yaml
channels:
  asec: {source: asec_raw_stage_v3, observed_geography: state}   # → county after #696 gap 1
  acs:  {source: [acs_2024_household_zip, acs_2024_person_zip], observed_geography: puma}
assembly:
  mass_anchor_channel: acs
  shared_dtype_policy: canonical_string_storage   # #672
clones:
  attachment_fraction: 1.0
  attachment_seed: derive               # from root_seed + stage label
  tail_support:                         # #652 per-stratum receipts
    strata: filing_status
    policy: declared_min_support_with_receipt
```

### geography.yaml — the #696 ruling, verbatim as config

```yaml
assignment:
  anchor: census_block_2020
  order: before_gap_fill
  draw:
    acs:  {universe: observed_puma,        weight: block_population_2020}
    asec: {universe: identified_county_else_state, weight: block_population_2020}
  derive:
    tract_geoid:  structural_prefix
    county_fips:  structural_prefix
    puma:         tract_relationship      # must equal observed for acs — asserted
    congressional_district_geoid: {equivalency: cd119_block_bef, vintage: 119}
    place_fips: block_crosswalk
    sldu: block_crosswalk
    sldl: block_crosswalk
    cbsa_code: block_crosswalk
  ladder_source: us_block_ladder_2020
  validation: puma_cd_overlap_consistency   # vs (PUMA, CD) overlap shares
```

### imputation.yaml — the heart: families × predictors × models

Predictor **blocks** are the reusable unit; families compose them.
Participation families are schema-required to include an
`eligibility`-tagged block (the CHAMPVA lesson, enforced at load).

```yaml
predictor_blocks:
  core_demographics: {columns: [age, is_female, state_fips], tags: [required]}
  income_components: {columns: [employment_income, self_employment_income,
    social_security_income, retirement_income, interest_dividend_rental_income],
    availability: observed}              # availability-pattern partition
  household_structure: {columns: [household_size, marital_status, relationship_to_head, is_household_head, tenure_code]}
  veteran_va: {columns: [is_veteran, receives_va_payments], tags: [eligibility]}
  disability: {columns: [has_hearing_difficulty, has_vision_difficulty, "…"], tags: [eligibility]}
  own_coverage: {columns: [acs_hins_va, acs_hins_medicaid, "…"], tags: [eligibility]}
  geography: {columns: [puma, county_fips], channel_caveat:
    "exact for acs recipients; ladder-drawn for asec donors (attenuating, not biasing)"}

models:
  regime_gated_qrf:
    kernel: kernel:regime_gated_qrf
    params: {n_estimators: 100, max_samples_leaf: null, zero_atol: 1.0e-06}
    # gate/magnitude/weighting/chaining live in the kernel's contract doc;
    # every knob a family may override is a declared param here.

chaining:                       # chain semantics are DECLARED, not accidental
  order: declared               # the family's target list IS the chain order
  max_targets_per_fit: 8        # chains split greedily at this width
  cross_batch: independent_given_predictors   # split chains share only base
  keep_together:                # joint-critical targets must share one chain
    - [ssn_card_type, immigration_status_str]   # existing joint codec
    # coverage booleans, etc. — declared per family as needed

families:
  - id: gap_fill/asec_survey_to_acs/person/benefit_participation
    stage: early_gap_fill
    donor: {channel: asec}
    recipient: {channel: acs}
    model: regime_gated_qrf
    predictors: [core_demographics, income_components, household_structure,
                 veteran_va, own_coverage, geography]
    calibration: {participation: donor_weighted_rate, amounts: none}  # two-part, part 1
    targets: [has_champva_health_coverage_at_interview, has_tricare_health_coverage_at_interview, "…"]
  - id: late_transfer/person/adult_care
    stage: late_transfer
    donor: {channel: asec, clone: 0}     # #675 alignment, now just config
    model: regime_gated_qrf
    predictors: [puf_tax_detail]
    targets: [is_incapable_of_self_care, pre_subsidy_care_expenses]
computed_producers:
  - id: acs_earnings_universe
    kernel: kernel:acs_earnings_universe
    inputs: ["…declared, with fallbacks…"]
    outputs: ["…"]
```

### take_up.yaml — every stochastic flag, mechanism + source

Populated by the #697 lane's code trace; schema:

Entries key by **program**, not by flag column (Max, 2026-08-16). The
column name derives from a declared naming rule and the entity derives
from the take-up contract's engine metadata (`TakeUpProgram.entity`, which
the engine's own variable definition sourced) — both derived-and-asserted,
never independently declared, so neither can drift from the program.

```yaml
naming_rule: takes_up_{program}{treatment_suffix}   # suffix from contract
                                                    # treatment kind; legacy
                                                    # irregulars regularized
                                                    # at the flip
draws:
  - program: snap                       # the key — a contract row
    kernel: kernel:snap_state_take_up
    probability_source: {contract: take_up_contract, grain: state}
    conditioning: eligibility_interaction
    seed: derive
    calibration_status: none            # visible debt
    # derived + asserted (loader refuses on mismatch):
    #   column: takes_up_snap_if_eligible   (naming_rule)
    #   entity: spm_unit                    (contract -> engine metadata)
```

### battery.yaml / calibration.yaml / selection.yaml / publication.yaml

```yaml
# battery.yaml
completeness: {targets: 131, source: contract:us_pool_inputs}
by_origin:
  legs: [incidence_ratio, quantile_envelope, boolean_liveness]
  verdict_policy: report_all_never_widen
# calibration.yaml
solver: {kernel: kernel:calibrate_prox, loss: capped_relative_error,
         max_weight_ratio: <hard bound>, device: auto}   # torch; local MPS / Modal GPU
targets:
  source: chronicle_facts               # sum-only doctrine
  geography_layers: [national, state, congressional_district, county]
  cd_policy: always_present_report_attainment   # Max 8/15 ruling
# selection.yaml — knobs only; validity gates are KERNEL CONTRACT, not config
exact_k:
  kernel: kernel:sampford_exact_k     # contract: exact-k cardinality gate
                                      # (len==k never clamped, unique, in-pool,
                                      # sorted; real exceptions, never assert),
                                      # pi in [0,1] finite, PCG64(seed) isolated,
                                      # ulp-corrected inclusion probs + 6-scalar
                                      # receipt. "support" here = index validity;
                                      # renamed cardinality_gate to avoid clash
                                      # with donor support.
  k: 50000
  pi_hi: <boundary parameter>
  group_ids: <declared semantics or none>
  on_infeasible: refuse
# publication.yaml
release:
  # D6 (open, Max): the flip is the natural rename boundary — spec-engine-era
  # artifacts adopt the microcosm-* line; populace-* rows stay valid-historical
  # in the chain. HF destination is a separate named decision (never changed
  # implicitly).
  line: microcosm-us-2024            # proposed; was populace-us-2024 (frozen v1)
  pattern: "{line}-stacked-f{rung}-s{seed}-…"
  rungs: [f001, f004, f010, f025, f100]
distribution: hf:policyengine/populace-us   # unchanged until explicit ruling
logbook: {chain: strict_linear, store: supabase:logbook}
```

### column_lineage.yaml — closure over the artifact

The #697 classes, unchanged in role: every artifact column claimed by
exactly one of `measured_native | imputed | computed | assigned_by_ladder
| take_up_draw | receipt | structural | unclassified(reason)`. The loader
runs closure at spec-load time against the committed inventory fixture;
the build re-runs it against the artifact it actually wrote.

## The kernel registry

```python
@register_kernel("regime_gated_qrf", params_schema=…, io=…)
def fit_and_draw(frame, *, family_config, rng): ...
```

Rules: registry refuses unknown ids and duplicate registrations; a kernel's
`params_schema` validates the spec's params at load; kernels receive ONLY
declared inputs and may write ONLY declared outputs (the existing
ownership/tail-preservation guards already enforce the write side —
r7's machinery becomes the kernel discipline).

## Identity, seeds, determinism

- `spec_sha256` + `schema_version` join checkpoint identity and every
  logbook row. Two builds with different bundles can never share identity.
- One `root_seed` in `bundle.yaml`; every stage/family derives its stream
  as `hash(root_seed, stage_id, family_id)` — declared, reproducible, and
  visible on /lineage per family.

## The equivalence gate (how we know the flip changed nothing)

The flip PR ships with a **frozen-behavior proof**: build the f004 rung
twice at the same commit — once through the legacy constants (kept behind
a temporary `--legacy-constants` flag), once through the bundle — and
assert **identical checkpoint identities and gate outputs**. CI carries a
fixture-scale version of the same assertion. Only after that proof lands
do the atomic fixes (eligibility predictors, calibration part 1, ASEC
county) go in — each as a bundle edit whose diff IS the review.

## Migration map

| Today (constants / lanes) | Bundle home |
|---|---|
| `INPUT_SHAS`, launcher paths, engine pin | `sources.yaml` |
| `_STACKED_SAMPLE_RUNG_TOKENS`, release regex, logbook rungs | `publication.yaml` |
| `stacked_gap_fill_plan()` literals | `imputation.yaml` families |
| `ACS_*_TRANSFER_PREDICTORS`, `PUF_TAX_DETAIL_DEFAULT_PREDICTORS` | `predictor_blocks` |
| predictor-sets lane (per-family blocks + widened loader) | `imputation.yaml` + `sources.yaml` loader columns |
| `CANONICAL_US_LATE_*` registry/groups/schedule | `imputation.yaml` (families + computed_producers) |
| #696 lane (block-first) | `geography.yaml` + ladder pin in `sources.yaml` |
| #697 lane (classes, inventory, closure) | `column_lineage.yaml` + loader closure |
| take-up modules' constants | `take_up.yaml` (+ contract stays the parameter source) |
| battery/gate constants | `battery.yaml` |
| calibrate solver args, exact-k args | `calibration.yaml`, `selection.yaml` |
| model defaults in `microcosm.fit` | `models` (fit keeps kernels + defaults as fallback for library users) |

Conformance tests that assert code==spec are **deleted** with their
constants; what remains: schema validation, cross-ref resolution, closure,
kernel-registry existence, and the equivalence fixtures.

## Rollout (each step lands green before the next)

1. **This RFC + skeletons** — Max sign-off on the schema.
2. **Loader + schema validation + `spec_sha256` identity join** (no
   behavior change; bundle generated from constants, both live).
3. **The flip, stage by stage, equivalence-gated**: imputation →
   geography (#696 lane content lands *as bundle config*) → take-up →
   sources/publication → battery/calibration/selection.
4. **Delete constants + interim conformance tests**; /lineage reads the
   bundle directly; coverage meter reads closure.
5. **Atomic fixes as bundle edits**, swept at f025 (OOS quantile loss):
   eligibility predictors, ASEC identified-county, calibration part 1/2.
6. **UK bundle** instantiates the same schema (#578 parity).

The three wave-1 lane branches are held un-pushed; their content merges
into steps 2–4 as bundle files rather than as parallel code+mirror PRs
(the loader widening and block-first implementation code land intact;
their YAML sections land re-shaped into the bundle).
