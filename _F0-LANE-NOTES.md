# F0 lane notes

These notes are the implementation journal for the approved F0 compiler-front-end
lane.  Line citations in section 1 refer to the starting tree at
`e45620992d14fdc52117fc3b1b2f7e96f86257ad` unless a later commit is named.
`PROGRESS.md` is deliberately untouched.

## 1. Sync, baseline, and legacy-authority coverage inventory

### Environment and baseline

- Branch/start: `spec-engine-schema` at `e4562099`; the worktree was clean.
- The first `uv sync --all-packages --extra us` could not write the sandboxed
  global uv cache, and a task-local empty cache then hit the sandbox's DNS
  denial while fetching SciPy.  No further network access was attempted.
- Sync was completed offline from the host's pre-populated, read-only uv cache
  through a writable cache proxy:
  `UV_CACHE_DIR=/private/tmp/microcosm-f0-uv-cache-proxy uv sync --offline
  --all-packages --extra us`.  It installed 95 packages and rebuilt all five
  workspace shards into this worktree's `.venv`.
- Baseline suite: `.venv/bin/python -m pytest -q` completed green with
  **6,267 passed, 37 skipped (6,304 collected), 0 failed**.  The run emitted
  only the pre-existing numerical, pandas fragmentation/chained-assignment,
  and sparse-tensor warnings summarized by pytest.
- GitNexus was installed locally, but its analyzer requires a write to
  `~/.gitnexus/registry.json`, which this sandbox forbids.  The partial local
  index was moved out of the worktree and all blast-radius work below used
  direct source/import analysis.

The starting branch predates the held per-family predictor lane: current ACS
transfer still has one shared predictor contract.  That is recorded rather
than silently importing a behavior change.  F0 compiles the current constants
at this assigned starting commit, and eligibility-blind participation targets
receive explicit F-P waiver records rather than invented predictors, as the
approved F0 brief requires.

### Coverage inventory and bundle homes

This is the coverage target for deliverable 5.  “Adapter target” means the
canonical JSON-shaped object that `compile_to_legacy_payload()` must reproduce;
F0 does not execute any of it.

#### A. Early gap-fill plan, target surfaces, and predictor tuples

| Inventory item | Current authority / transfer surface | Planned single-authored home | Adapter target |
| --- | --- | --- | --- |
| Two-direction early gap-fill plan | `GapFillDirection` and absence-rule grammar at `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1736`; plan construction at `stacked_spine.py:1853`; canonical binding at `stacked_spine.py:2983` | `imputation.yaml.families` plus resolved direction metadata | `_plan_payload()` at `stacked_spine.py:2145` and `stacked_gap_fill_plan()` at `stacked_spine.py:3258` |
| Early transfer surface | Raw declared ACS families at `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:313`; nonnative source-operator additions at `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:692`; early/late partition at `multispine_pool.py:740`; producer sub-surfaces at `multispine_pool.py:814` | Ordered `imputation.yaml.families` and producer nodes | `_surface_payload()` at `stacked_spine.py:2137`; current early ledger is 13 families / 48 targets, pinned by `packages/microcosm-build/tests/test_imputation_lineage_spec.py:43` |
| Shared ACS person predictors | Required ordered tuple at `acs_transfer.py:97`; optional ordered tuple at `acs_transfer.py:148`; combined precedence at `acs_transfer.py:186`; tenure codec at `acs_transfer.py:214` | `imputation.yaml.predictor_blocks.acs_person_transfer` | Effective required + observed-optional tuple consumed by family fits at `acs_transfer.py:1260` and `acs_transfer.py:1307` |
| Shared ACS grouped predictors | Required aggregates at `acs_transfer.py:166`; person-to-group optional mapping at `acs_transfer.py:176`; combined precedence at `acs_transfer.py:186` | `imputation.yaml.predictor_blocks.acs_group_transfer` | Same family-fit adapter, with group projection |
| Primary PUF predictors and chained tuples | Ordered base eight at `packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:197`; ordered person/tax-unit target surface at `puf_support.py:170` and `puf_support.py:208`; fixed target order/hash at `packages/microcosm-build/src/microcosm/build/us_runtime/puf_qrf_chain.py:80`; append-after-draw semantics at `packages/microcosm-fit/src/microcosm/fit/qrf.py:1307` and `qrf.py:1523` | `imputation.yaml.predictor_blocks.puf_tax_detail` and ordered primary family | 65 effective tuples (base eight plus prior targets), asserted at `packages/microcosm-build/tests/test_us_multispine_pool.py:2067`; primary manifest at `puf_qrf_chain.py:227` |
| QRF model defaults currently leaking through call sites | Library defaults `n_estimators=100`, `max_samples_leaf=None`, `zero_atol=1e-6` at `packages/microcosm-fit/src/microcosm/fit/qrf.py:83` and `qrf.py:1033`; build calls at `puf_qrf_chain.py:219` and `acs_transfer.py:1341` | `imputation.yaml.models.regime_gated_qrf.params`, all explicit | Primary config binding at `stacked_spine.py:4702`; per-transfer execution contract at `acs_transfer.py:229` and binding at `stacked_spine.py:5411` |

#### B. Frozen late transfer ledger and producer graph

| Inventory item | Current authority / payload | Planned single-authored home | Adapter target |
| --- | --- | --- | --- |
| Frozen 19-group / 70-target split ledger | Ordered source rows and greedy grouping at `packages/microcosm-build/src/microcosm/build/us_runtime/us_late_producer_registry.py:1313`; width-eight/keep-together splitter at `us_late_producer_registry.py:1338`; import-time 19/70 and 51 numeric / 17 Boolean / 2 string guards at `us_late_producer_registry.py:1393` | `imputation.yaml.chaining.split_after` plus 19 explicitly ordered late families | `transfer_groups` inside `us_late_producer_schedule_payload()` at `us_late_producer_registry.py:2047` |
| Five-batch / 37-target tax-itemization split | Exact ordered five groups are mirrored at `specs/us_imputation_lineage.yaml:280`; batch-name ownership consumers at `packages/microcosm-build/src/microcosm/build/us_runtime/us_late_overlap_ownership.py:29` | Five declared late families, with a reason at every split | Exact group ids and target arrays in the schedule payload; never recomputed by a compiler splitter |
| Producer input/output grammar | `ProducerInputColumn`, `ProducerInput`, `ProducerOutput`, and `ProducerContract` at `packages/microcosm-build/src/microcosm/build/us_runtime/late_producer_dag.py:47`; exact contract projection at `late_producer_dag.py:176` | `imputation.yaml.producer_graph.nodes` | `_contract_payload()` plus graph scope coverage |
| Deterministic DAG | Scope coverage at `late_producer_dag.py:27`; deterministic validation/toposort/hash at `late_producer_dag.py:256`; tolerated-absence receipt rules at `late_producer_dag.py:405` | `producer_graph.external_stages`, explicit `depends_on`, `ordering: deterministic_total` | Schedule order, waves, edges, hashes in `us_late_producer_schedule_payload()` |
| Full 38-producer registry | Registry/receipt/transition ABI constants at `us_late_producer_registry.py:102`; common/effective inventories at `us_late_producer_registry.py:375`; 16 source inventories at `us_late_producer_registry.py:640`; primary and ACS inventories at `us_late_producer_registry.py:1017` and `us_late_producer_registry.py:1123`; node construction at `us_late_producer_registry.py:1597` | Complete producer graph, including virtual resources and all effective input rows | `us_late_producer_schedule_payload()` / `us_late_producer_schedule_receipt()` at `us_late_producer_registry.py:2047` / `us_late_producer_registry.py:2106`; baseline schedule SHA `b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5` |
| Resource/config semantics | Virtual kinds/versions at `stacked_spine.py:4050`; source-stage binding at `stacked_spine.py:5106`; source, finalizer, ACS, transfer configs at `stacked_spine.py:5172`, `stacked_spine.py:5303`, `stacked_spine.py:5379`, and `stacked_spine.py:5411` | Node virtual resources and resolved kernel/config params | `stacked_late_producer_resource_semantics_receipt()` at `stacked_spine.py:5529` |
| Transition/producer receipt contract | Exact execution-row keys at `stacked_spine.py:6112`; genesis at `stacked_spine.py:6726`; order validation at `stacked_spine.py:6799`; transition authority at `stacked_spine.py:6046` | Producer-graph receipt declarations only | Schedule payload's `execution_receipt_contract` and `transition_authority`; no F0 execution/binding |

#### C. Conditional ownership matrix

- The three overlap targets and their source/primary/transfer producers are
  declared at `packages/microcosm-build/src/microcosm/build/us_runtime/us_late_overlap_ownership.py:26`.
- The exact matrix is 3 targets × 2 origins × 3 clone roles = **18 rows** at
  `us_late_overlap_ownership.py:98`, with the import-time row-count guard at
  `us_late_overlap_ownership.py:243`.  Tuition is transfer-owned on clone 0,
  primary-owned on clone 1, and clone 2 byte-inherits clone 1; retirement is
  source-owned for ASEC rows and transfer/primary/inherited for ACS clone roles.
  The preservation/masking doctrine is at `us_late_overlap_ownership.py:185`.
- Bundle home: `imputation.yaml.producer_graph.ownership_matrix`, all 18 rows
  authored explicitly.  Adapter target:
  `us_late_overlap_ownership_receipt()` at
  `us_late_overlap_ownership.py:178`; baseline payload SHA
  `5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356`.

#### D. Source-stage order and take-up contract

| Inventory item | Current authority / mechanism | Planned single-authored home | Adapter target |
| --- | --- | --- | --- |
| Source-stage grammar | Operation allow-list at `packages/microcosm-build/src/microcosm/build/source_manifest.py:41`; exact operation/stage records at `source_manifest.py:145` and `source_manifest.py:169`; runtime seed/period config at `packages/microcosm-build/src/microcosm/build/source_runtime.py:29` | `sources.yaml` logical pins plus producer nodes/configs; compatibility projection remains `SourceManifest` | Exact resolved `SourceStageSpec`, resource hash, resolver and execution config bound at `stacked_spine.py:5106` and `stacked_spine.py:5172` |
| 16 post-clone source operators | Total order at `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:203`; phase/mechanism/scope registry at `multispine_pool.py:475`; callback params at `multispine_pool.py:1808`; finalizer/deferred assets at `multispine_pool.py:1948` and `multispine_pool.py:661` | Producer nodes in that exact order, with source evidence/config virtual resources | Source nodes, finalizer and resource-semantics receipt |
| Full take-up ABI snapshot | Complete 13-row JSON authority at `packages/microcosm-build/src/microcosm/build/us/take_up_contract.json:1`; dataclass/loader/full-resource hashing at `packages/microcosm-build/src/microcosm/build/us_runtime/take_up_contract.py:72` and `take_up_contract.py:123`; installed-engine drift tripwire at `take_up_contract.py:327` | `take_up.yaml` owns treatments/pipelines; generated `engine_abi.lock.json` owns reviewed engine facts | `take_up_contract_identity()` at `take_up_contract.py:212`, including every raw row and resource SHA; baseline SHA `495dc6ed195eae372a6ba098c6fb894323638a4a7dce1b4fe7efaaf6beb69446` |

The 13 program mechanisms that must survive normalization are:

1. SNAP: reported/FNS national prior then eligible-only state count calibration,
   `take_up_contract.json:18` and `source_stages.json:1950` / `source_stages.json:1988`.
2. TANF: stable-id scalar-rate seed, `take_up_contract.json:29`.
3. EITC: stable-id rate-by-approximated-child-count seed,
   `take_up_contract.json:47`.
4. Medicaid: reporter anchor plus state count calibration,
   `take_up_contract.json:65` and `source_stages.json:2788`.
5. CHIP: engine default with source debt, `take_up_contract.json:84`.
6. Basic Health Program: engine default with source debt,
   `take_up_contract.json:96`.
7. Medicare: measured ASEC mapping plus clone propagation,
   `take_up_contract.json:108` and `source_stages.json:2068`.
8. SSI: target-derived age-band probability seed plus delivery gate, explicitly
   never a flag count-match, `take_up_contract.json:120` and
   `source_stages.json:1489`.
9. DC PTC: engine default with source debt, `take_up_contract.json:147`.
10. Head Start: SIPP-trained weighted QRF transfer,
    `take_up_contract.json:159` and `source_stages.json:1357`.
11. Early Head Start: engine default with source debt,
    `take_up_contract.json:171`.
12. Housing assistance: measured ASEC segment plus QRF-imputed PUF support
    segment, `take_up_contract.json:183` and `source_stages.json:2390`.
13. ACA: dedicated typed assignment/calibration sequence,
    `take_up_contract.json:195` and `source_stages.json:2680`.

The generic TANF/EITC draw grammar is also normative: plausibility bands at
`packages/microcosm-build/src/microcosm/build/us_runtime/take_up.py:73`,
BLAKE2b stable draw key at `take_up.py:112`, child-bin selection at
`take_up.py:200`, and assignment/idempotence at `take_up.py:242` and
`take_up.py:307`.

#### E. Capital-gains tail contract

- All literal contract values live at
  `packages/microcosm-build/src/microcosm/build/us_runtime/puf_capital_gains_tail.py:76`:
  stage/channel/schema versions; positive-mass target `1.2709e12`;
  top-100 share cap `0.75`; minimum 500 nonzero rows; `.995`/`.999`
  quantiles; ASEC topcode `1,999,998`; the four-person-column joint vector;
  tax-unit unrecaptured-section-1250 target; and overlap ownership.
  Provenance/output columns are at `puf_capital_gains_tail.py:123`, filing-status
  and AGI-proxy precedence at `puf_capital_gains_tail.py:136`, and the
  worsening tolerance at `puf_capital_gains_tail.py:557`.
- Bundle home: the PUF support role's typed tail support plus the primary node's
  fully resolved kernel parameters.  Adapter targets are
  `puf_capital_gains_tail_support_contract_identity()`, aggregate identity,
  concentration controls and execution inputs at
  `puf_capital_gains_tail.py:155`, `puf_capital_gains_tail.py:186`,
  `puf_capital_gains_tail.py:216`, and `puf_capital_gains_tail.py:243`.

#### F. Stacked authority and identity aggregation

- `_authority_component_payloads()` at
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2311`
  defines the eight exact components: gap-fill plan, post-PUF transfer surface,
  declared surface, metric registry, joint registry, support profile, tail
  contract, and late schedule.  Canonical JSON/hash rules are at
  `stacked_spine.py:2349`; `_authority_receipt()` is at `stacked_spine.py:3349`.
- `stacked_gap_fill_producer_schedule_receipt()` at `stacked_spine.py:3264`
  is a separate executable precedence proof that the adapter must reproduce.
- Baseline stacked authority SHA is
  `f0b676f6508dbf6bb2b787c42e6b85331bacc57c6649ac7ad15fdaa5884a1b2d`;
  its late-component SHA is
  `bf95c78ea4168c81fa319872276002835f19ac27461eb3b69349c9637bc14f86`.
- Final aggregation/consumer is `_stacked_checkpoint_base_identity()` at
  `tools/build_us_multispine_pool.py:1043`: it binds stack authority; source,
  pre-clone, post-clone and derive orders; gap-fill schedule; late schedule and
  resource semantics; primary order/schema; earnings/QBI/take-up/tail
  contracts; estimator counts; transfer width; simulation batch size; engine
  version; period; run seeds; and input pins (`:1067-1145`).  The adapter's
  identity component projection is compared field-for-field to this object,
  but F0 never constructs an executor or retro-labels a generation-0 identity.

#### G. Seeds, rung grammar, and release regex

`legacy-v1` preserves the following distinct protocols; equal literals do not
mean shared advancing generators.

- **Run-request 578 streams.** Stacked CLI defaults independently to sampling
  seed 578 and clone-attachment seed 578
  (`tools/build_us_multispine_pool.py:496`).  The old pilot constant is ACS-only
  10%/578 (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:286`).
  ASEC and ACS sampling are separate calls (`stacked_spine.py:542` and
  `stacked_spine.py:613`), each sorting strata and ids and starting a fresh
  `default_rng(seed)`; full population consumes no draw
  (`packages/microcosm-build/src/microcosm/build/frame_sampling.py:208`).  PUF
  clone attachment likewise draws only below full fraction, after sorting
  source household ids (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:682`).
- **Build/model seed 0.** `POOL_RANDOM_SEED = 0` at
  `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:238`
  flows unchanged to pre-clone sources (`multispine_pool.py:1701`), all 16
  post-clone producers (`multispine_pool.py:1808`), TANF/EITC
  (`multispine_pool.py:2970`), and stacked gap-fill/primary/tail/transfers
  (`tools/build_us_multispine_pool.py:2984`).  Bundle home is the selected
  immutable `legacy-v1` protocol registry; node/family records carry the
  resolved stream ids and consumed order.
- **Two different PUF 42 contracts.** Archived processed-PUF disaggregation
  uses one shared `default_rng(42)` advancing across three bounded buckets
  (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_source_agi.py:21`
  and `puf_source_agi.py:355`).  Live raw-source disaggregation merely has a
  library default 42 but receives build seed 0 through runtime
  (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_aggregate_records.py:392`;
  `packages/microcosm-build/src/microcosm/build/source_runtime.py:1148`).  The
  existing source manifest records both archived 42 and live build-seed
  semantics (`packages/microcosm-build/src/microcosm/build/us/source_stages.json:27`,
  `source_stages.json:103`, and `source_stages.json:123`).
- **SSI fixed streams.** Weighted replacement seed
  `8_386_123_572_872_638_692`, model seed 42, 20k cap, replace=true, and ignored
  build seed are fixed at
  `packages/microcosm-build/src/microcosm/build/us_runtime/ssi_disability_criteria.py:236`
  and consumed at `ssi_disability_criteria.py:271` and
  `ssi_disability_criteria.py:904`.  ASEC and PUF deep-copy the pristine model
  so their draw streams reset identically (`ssi_disability_criteria.py:978`).
- **Stable-string training-cap hashes.** Vehicle and financial-asset caps use
  UTF-8 polynomial-x31 modulo 2^64 followed by the fixed xor/multiply mixer and
  modulo 2^63 (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:299`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_financial_assets.py:307`).
  Vehicles use model seed 42 and per-target/fill salts (`sipp_vehicles.py:137`
  and `sipp_vehicles.py:317`).  Assets preserve bank→stock→bond order and
  derive three QRF seeds from `SeedSequence([base_seed, 374]).spawn(3)`
  (`sipp_financial_assets.py:78` and `sipp_financial_assets.py:663`).
  Archived ACS-rent caps use the same stable hash, with rent then real-estate
  tax ordering (`packages/microcosm-build/src/microcosm/build/us_runtime/housing_inputs.py:740`
  and `housing_inputs.py:759`).
- **Tips fixed cap.** Seed `5_559_651_045_748_063_828` for
  `calibration_sipp_tip_training_sample:tip_income`, 10k cap, no replacement,
  and sorted selected positions live at
  `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_tips.py:109`
  and `sipp_tips.py:399`.  This was absent from the old source manifest and is
  therefore an explicit bundle-home gap the generator must close.
- **SCF composite stream.** The household source selector sorts ids and uses
  `default_rng(SeedSequence([seed, time_period, 374]))`, one uniform per
  household at p=.5 (`packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py:830`);
  SCF QRF and SIPP submodels retain their separate protocols
  (`scf_wealth.py:744` and `scf_wealth.py:866`).
- **ACS transfer derivation.** Family seed is the little-endian integer from
  the first four SHA-256 bytes of UTF-8
  `base\0entity\0family`; pattern/QRF seed appends NUL-joined ordered optional
  predictors.  Pattern id is the first eight lowercase hash hex characters,
  prefixed by its two-digit position (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2902`).
  Declared availability-pattern and target order drive both fresh and banked
  execution (`acs_transfer.py:1247` and `acs_transfer.py:1477`).
- **QRF kernel consumption.** `SeedSequence(seed).spawn(2)` assigns child 0 to
  one shared fit RNG and child 1 to one shared draw RNG; both advance in target
  order, and prior targets append to predictors
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:1073`).  Canonical PCG64
  states/order/config are checkpointed (`qrf.py:622` and `qrf.py:1126`).
  Degenerate, one-sided, and gated regimes consume different pinned sequences
  (`qrf.py:1333`); predict draws consume ordered quantiles and then gated sign
  uniforms (`qrf.py:950`).  These advances are kernel-contract-normative and
  must not be split into per-target streams in F0.
- **BLAKE2b stable draws.** Generic source operations hash UTF-8
  `{seed}:{salt}:{stable-key}` using 8-byte BLAKE2b, big-endian unsigned divided
  by 2^64 (`packages/microcosm-build/src/microcosm/build/source_runtime.py:1297`),
  with frozen ACA/calibration/joint salts at `source_runtime.py:451`,
  `source_runtime.py:520`, and `source_runtime.py:632`.  TANF/EITC use the same
  shape (`packages/microcosm-build/src/microcosm/build/us_runtime/take_up.py:112`).
  Named call sites include SNAP (`us_runtime/snap_take_up.py:130`), pregnancy
  (`us_runtime/pregnancy.py:128`), WIC (`us_runtime/wic_claim.py:370`), ABAWD
  (`us_runtime/snap_discretionary_exemption.py:133`), immigration
  (`us_runtime/immigration.py:431`), SSI (`us_runtime/ssi_take_up.py:502`),
  Medicaid (`us_runtime/medicaid_take_up.py:119`), and state SNAP
  (`us_runtime/snap_state_take_up.py:64`).
- **Other reachable stochastic sites.** Adult-care weighted-prefix assignment
  (`us_runtime/adult_care.py:461`), capital-gains-tail random ranking
  (`us_runtime/puf_capital_gains_tail.py:1400`), Torch calibration reseeding
  (`packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:1030`),
  exact-k PCG64/Sampford selection
  (`packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py:168` and
  `exact_k.py:428`), the build-seeded 5k pandas caps in
  `us_runtime/prior_year_income.py:450`, `childcare.py:298`,
  `retirement_contributions.py:339`, `disability_benefits.py:343`,
  `housing_inputs.py:1078`, `workers_compensation.py:303`,
  `retirement_distributions.py:432`, `child_support.py:324`,
  `energy_subsidy.py:326`, `other_health_insurance.py:388`, and
  `weeks_unemployed.py:904`, plus legacy geography RNG construction in
  `us_runtime/geography_ladder.py:286`, `puma_ladder.py:334`, and
  `congressional_district_geography.py:128`.

The rung map is exactly `{.01:f001, .04:f004, .10:f010, .25:f025,
1.00:f100}` and the historical writer regex is
`^populace-us-2024-stacked-f(?:001|004|010|025|100)-s[0-9]+-asec[0-9]+-acs[0-9]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$`
at `tools/build_us_multispine_pool.py:277`.  `_stacked_rung()` consumes the map
at `tools/build_us_multispine_pool.py:1002`; the writer composes and validates
the full realized-count/timestamp/nonce id at
`tools/build_us_multispine_pool.py:1274`.  Logbook independently admits exactly
the same five tokens and signed nonnegative 64-bit seeds at
`packages/microcosm-build/src/microcosm/build/logbook.py:91` and
`logbook.py:964`.

Bundle home is `publication.yaml.release`: F0 emits new
`microcosm-us-2024-*` ids and retains `populace-us-2024-*` as reader-only
history.  The compiler must tighten the current schema's generic `fNNN` rule to
the exact five rungs and correct the drafting pattern, which currently both
double-prefixes `f` and omits realized counts/timestamp/nonce
(`specs/us/publication.yaml:10`; schema boundary at
`specs/schema/publication.schema.json:66`).

#### H. CountrySpec seam and expressibility findings

- The starting seam is a JSON-only bare-filename manifest with a fixed
  `CountrySpec` projection and hard-coded filenames
  (`packages/microcosm-build/src/microcosm/build/country_spec.py:1`,
  `country_spec.py:756`, and `country_spec.py:797`).  `country_stage_plan()`
  requires all declared sources plus geography and rejects missing/unknown
  implementations (`country_spec.py:923`).  The package test permits only JSON
  suffixes and assumes string resources
  (`packages/microcosm-build/tests/test_spec_only_country_packages.py:9` and
  `test_spec_only_country_packages.py:44`).  Deliverable 2 must replace this in
  place with typed rows and a single `ResolvedCountrySpec`, while preserving
  the old projections.
- Belgium declares `silc_load` only
  (`packages/microcosm-build/src/microcosm/build/be/source_stages.json:1`) and
  `clone_assign_communes` with 20 clones / 2025 NIS
  (`packages/microcosm-build/src/microcosm/build/be/geography_spine.json:1`).
  There is no runtime implementation of either declared stage; today's only
  “compile” test supplies no-op lambdas
  (`packages/microcosm-build/tests/test_country_spec.py:193`).  The Axiom pilot
  directly constructs a synthetic frame and additionally needs the external
  Axiom engine and `POPULACE_RULESPEC_BE`
  (`packages/microcosm-frame/tests/test_axiom_adapter.py:447`), while the build
  shard exposes only US/UK extras
  (`packages/microcosm-build/pyproject.toml:25`).  Deliverable 6 therefore needs
  a new fixture-backed shared smoke implementation if it is to do more than the
  presently available compile proof; it must not pretend gated SILC exists.
- The approved 15 schemas can select `seed_protocol: legacy-v1` and reference
  stream ids, but contain no authored site→stream/draw-ledger shape:
  `specs/schema/bundle.schema.json:7` is only the protocol enum,
  `specs/schema/defs.schema.json:22` only defines a stream-ref string, and
  `specs/schema/locks.schema.json:105` only records emitted node refs.  The
  binding RFC nevertheless makes sites, literal seeds, spawn/order/reset/key
  grammar normative.  F0 will treat `legacy-v1` as a content-digested immutable
  compiler protocol selected by the authored scalar, not as an untyped bundle
  extension; its complete resolved ledger enters `bundle.lock`, compiled IR,
  coverage, and the adapter.  This preserves the approved 15-schema shape.  If
  review instead requires each seed literal to be country-YAML-authored, the
  approved schema set itself must be amended; silently inventing a 16th resource
  would violate the assigned contract.

### Machine-counted coverage target

Deliverable 5 must at least account for: 8 stacked-authority components; 13
early families / 48 targets; 65 primary chained targets/tuples; 19 late groups
/ 70 targets (including five / 37 itemization); 38 producer nodes; all producer
inputs, outputs and virtual-resource rows; 18 ownership rows; 16 source
operators; 13 take-up programs; every tail-control field; every legacy-v1 draw
site; and all stacked identity/rung/release fields inventoried above and in
subsection G.  A normative field without a compiler usage path or any one of
these inventory surfaces without a bundle home is a failing report.

## 2. Loader, canonicalizer, and CountrySpec seam

- The authored parser is a deliberately restricted YAML 1.2 implementation:
  it composes before construction so duplicate keys, merge keys, tags,
  timestamps, recursive aliases, non-string keys, multi-document streams, and
  non-finite values refuse with stable source positions
  (`packages/microcosm-build/src/microcosm/build/spec_engine/yaml12.py:295`).
- `SchemaRegistry` loads exactly the 15 approved draft-2020-12 documents,
  checks their ids and schemas, pre-resolves every local reference through a
  retrieval-refusing `referencing.Registry`, injects only declared defaults,
  and reports all validation failures in deterministic path order
  (`packages/microcosm-build/src/microcosm/build/spec_engine/schemas.py:162`).
  The only current authored default is selection precedence, now explicit at
  `specs/schema/selection.schema.json:44`; omitted and explicit forms resolve
  identically.
- Every authored schema inherits a normative surface declaration (for example
  `specs/schema/bundle.schema.json:5`).  The two ruled exceptions are encoded
  rather than guessed: catalog docs are documentation
  (`specs/schema/catalogs.schema.json:77`) and the publication audit store is
  operational (`specs/schema/publication.schema.json:129`).  The
  schema-directed walker produces six physically separate immutable objects,
  normalizes finite numbers and NFC strings, preserves arrays unless the
  schema explicitly declares a set, and sorts the manifest's declared set
  (`packages/microcosm-build/src/microcosm/build/spec_engine/canonical.py:170`;
  `specs/schema/resource_manifest.schema.json:23`).
- `load_bundle()` validates manifest path containment and exact file closure,
  resolves typed domains/defaults/cross-references, and returns the immutable
  dataclass root at
  `packages/microcosm-build/src/microcosm/build/spec_engine/loader.py:235` and
  `packages/microcosm-build/src/microcosm/build/spec_engine/model.py:323`.
  Its semantic envelope includes the typed composition and normative domain
  projections; documentation/operational edits do not change `spec_sha256`.
  `bundle.lock.json` is emitted only from that result
  (`spec_engine/loader.py:411`) and is never admitted as an authored row.
- The seam is one type: `CountrySpec` is an exact alias of
  `ResolvedCountrySpec` (`packages/microcosm-build/src/microcosm/build/country_spec.py:873`
  and `country_spec.py:928`).  Generation-0 filename rows
  become visible `legacy_json` descriptors without changing their historical
  fingerprint; a typed v1 manifest loads the compiler result through this same
  object (`country_spec.py:1116`).
- Packaging gate: all five shard wheels built; the build wheel contains exactly
  the 15 schemas via the force-include rule
  (`packages/microcosm-build/pyproject.toml:66`).  A clean Python 3.14 venv
  installed the wheels offline, imported `microcosm.build.spec_engine` from the
  wheel, resolved the schema root inside site-packages, and loaded the shipped
  US, UK, and BE compatibility packages.
- Validation gate: the exact pre-commit tree collected 6,356 tests and completed
  at 100% with 6,319 passed, 37 skipped, and zero failures.  Ruff and
  `git diff --check` are clean.  The build-shard wheel was then rebuilt from
  that same tree and installed offline in a second clean Python 3.14 venv; its
  imported package contained all 15 schema resources.

## 3. Complete generated US bundle

### One-way migration and package ownership

- The retained one-shot migration generator has one tombstone,
  `2026-11-16`, and one ordered 12-domain registry
  (`tools/generate_us_bundle_from_constants.py:37`, `:50`, and `:66`).  One
  scoped pass builds all documents, renders deterministic alias-free YAML,
  attests the frozen compatibility resources, emits the engine lock, supports
  byte-for-byte `--check`, and compiles the result through `load_bundle()`
  (`tools/generate_us_bundle_from_constants.py:192`, `:242`, `:304`, `:328`,
  `:353`, and `:404`).
- `country_package.json` is the one typed `{path, kind, schema_id}` inventory:
  12 packaged YAML domains plus 20 explicit `legacy_json` evidence resources
  (`tools/generate_us_bundle_from_constants.py:224` and
  `packages/microcosm-build/src/microcosm/build/us/country_package.json:1`).
  The admitted kinds are closed, and `CountrySpec` remains the exact alias of
  `ResolvedCountrySpec`
  (`packages/microcosm-build/src/microcosm/build/country_spec.py:76`, `:849`,
  and `:939`).
- The seam calls `load_bundle()` once and derives the legacy source,
  support-spine, and take-up views from that same resolved object.  Packaged
  generation-0 JSON is equality-attested evidence, not a selectable second
  authority (`country_spec.py:984`, `:1116`, `:1279`, `:1335`, and `:1405`).
- The retired root drafting location is a symlink-free pointer only
  (`specs/us/README.md:1`), enforced together with generator/package byte
  equality at
  `packages/microcosm-build/tests/test_us_spec_bundle.py:1015` and `:1034`.

### Complete normalized US surface

- Machine-counted contents are 7 pinned sources / 37 source stages; early
  13 families / 48 targets; primary 1 / 65; late 19 / 70; and itemization
  batches `[8, 8, 8, 8, 5]`
  (`packages/microcosm-build/tests/test_us_spec_bundle.py:281`).
- The producer graph has 38 nodes, 18 ownership rows, 71 derived RAW edges,
  and 6 deterministic waves.  Only 92 structural/virtual outputs are authored;
  the family-owned join compiles the complete 227-output surface
  (`test_us_spec_bundle.py:338` and
  `packages/microcosm-build/tests/test_spec_engine_typed_closure.py:50`).
  `compile_producer_outputs()` makes family targets the sole modeled-output
  authority and refuses mirrors/collisions; the same module then closes typed
  columns, artifacts, and scopes
  (`packages/microcosm-build/src/microcosm/build/spec_engine/typed_closure.py:162`
  and `:921`).
- Take-up contains 13 programs and 24 typed steps: 17 thin source-operation
  references and 7 local operations over 8 source stages.  Ownership is engine
  4, measured 1, mixed 1, modeled 6, transferred 1
  (`test_us_spec_bundle.py:381`).  Source-backed steps contain neither copied
  compatibility payloads nor copied stage arrays
  (`packages/microcosm-build/tests/test_spec_engine_take_up_semantics.py:107`).
- The battery has 131 scalar plus 1 joint metric, catalogs close 173 columns,
  and typed resolution yields 84 artifacts and 7 finite scopes
  (`test_us_spec_bundle.py:440` and
  `test_spec_engine_typed_closure.py:50`).
- The selected `legacy-v1` protocol resolves 53 draw sites over 14 streams;
  geography remains today's PUMA-anchored legacy phase; the vintage index has
  15 referenced authorities; and publication owns the normative
  `microcosm-us-2024` line plus the exact five-rung grammar
  (`test_us_spec_bundle.py:884` and
  `packages/microcosm-build/tests/test_us_bundle_core_contracts.py:227`).
- Required participation concepts and today's exact predictor gaps are recorded
  in one F-P waiver rather than filled with invented predictors
  (`tools/us_bundle_generation/imputation.py:3324` and
  `test_us_spec_bundle.py:484`).

### Single-authored derivations and compatibility identities

- Graph edges, waves, order, write scopes, input inventories, and transfer
  groups are derived from node inputs, family targets, scopes, and the authored
  ownership rows (`tools/us_bundle_generation/imputation.py:2438`, `:2657`,
  `:2721`, `:2803`, and `:2831`).  The typed projection reconstructs the seven
  constants-era imputation identity components at `imputation.py:2947`.
- Exact graph identities remain schedule SHA
  `b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5`,
  schedule-payload SHA
  `7766f2e94476cceb93d9730a74afb2ca6fed836068053f96fa4141bcc2f6154e`,
  and ownership SHA
  `5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356`
  (`test_us_spec_bundle.py:364` and `:678`).
- Primary effective predictor tuples derive from the one base block plus the
  preceding target order (`imputation.py:2905`); battery summary views derive
  from the metric registries (`tools/us_bundle_generation/contracts.py:1033`).
- Calibration's retired flat aliases are not authored.  A pure projector
  derives them from the normalized solver contract
  (`packages/microcosm-build/src/microcosm/build/spec_engine/calibration_semantics.py:31`
  and `:101`); the constants-era canonical SHA remains
  `8cc14b405d775640123eafeda89405ab0cb28455c554df8e9bf0eb950d78a806`
  (`packages/microcosm-build/tests/test_spec_engine_calibration_semantics.py:27`).
- Take-up source references join to source operations, then project the typed
  ownership/step contract with reviewed generated ABI facts
  (`packages/microcosm-build/src/microcosm/build/spec_engine/take_up_semantics.py:331`,
  `:348`, and `:646`).  The result reconstructs frozen
  `take_up_contract.json` exactly
  (`packages/microcosm-build/tests/test_spec_engine_take_up_semantics.py:260`).
- `engine_abi.lock.json` derives from normalized take-up plus the fresh installed
  engine, is canonicalized/schema-validated, and is refused when missing or
  stale (`packages/microcosm-build/src/microcosm/build/spec_engine/engine_abi.py:292`,
  `:375`, and `:410`).  It pins PolicyEngine US `1.764.6` once, covers all 13
  programs, and contains the 993-row remaining-stage input manifest
  (`packages/microcosm-build/tests/test_spec_engine_engine_abi.py:331`).
- The seed implementation is source/dependency-attested and remains explicitly
  mirror-only until F1.  Every selected site binds exhaustively to a producer
  node, source stage, or pipeline operation
  (`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:35`,
  `:149`, and
  `packages/microcosm-build/src/microcosm/build/spec_engine/identity_contracts.py:68`).
  The generation-0 checkpoint envelope was extracted without relabeling
  (`tools/build_us_multispine_pool.py:302`; exact gate at
  `packages/microcosm-build/tests/test_spec_engine_identity_contracts.py:58`).
- Generation-0 resource bytes are unchanged: source stages
  `a3e9ca87f43d74b3d83320ca77559f28452036cf60dfc16bee10a22d4784f672`,
  support spine
  `68f37dc6ae6e0cde7ebccb53f88dd4a800e63456f838fa214ff98d1db8d815be`,
  and take-up
  `5852e96582793313782d1c3edfc4cfdd0358a1a9cfd54bfea5844cbb09e89bd4`.
  The generator attests both raw evidence and typed projections
  (`tools/generate_us_bundle_from_constants.py:117` and `:304`).
  `PROGRESS.md` remains untouched at SHA
  `7de938a5d5c9128303392edcfbb6d5c1f5f437b05521346a858ff37b6b572d78`.

### Packaging and validation gate

- All package resources are swept by the typed package-closure test, while the
  build wheel force-includes the 15 schemas
  (`packages/microcosm-build/tests/test_spec_only_country_packages.py:47` and
  `packages/microcosm-build/pyproject.toml:63`).  All five shard wheels were
  rebuilt offline following the CI shard loop at `.github/workflows/test.yml:45`.
- Final source and isolated installed-wheel identities match exactly: 15
  schemas, 32 resources, spec SHA
  `c78c1ce8c951e5f745d7a03c3e887e151f8fcfa0bd3adeafb12cf73f2983acac`,
  seed implementation SHA
  `76a79351cb919e1a3119b6a0c1bfbc4ef392381f8a7add61b34c1527c444eacd`,
  and package fingerprint
  `30bb9dcd0d902c9bd8fd19471f5e01e82539f3845a4c58f3bf1aa3b134078ecc`.
- The synced numerical stack installs a process-wide warning ignore filter
  during the deprecated ACS shim's imports.  The shim now scopes an `always`
  filter to its one public deprecation notice, leaving the caller's filter
  unchanged (`tools/build_us_acs_multispine_base.py:85`; contract test at
  `packages/microcosm-build/tests/test_us_acs_multispine_base_builder.py:202`).
- Final D3 suite gate: 6,521 tests collected; `.venv/bin/python -m pytest -q`
  reached 100% with exit 0 and no failures.  The generator `--check`, Ruff,
  `git diff --check`, raw resource hashes, and source/wheel identity parity are
  rerun immediately before the commit.

## 4. Compiled IR and constants-era adapter

### Compiler boundary and emitted plan

- `compile_spec()` is a pure `ResolvedSpec -> CompiledSpecIR` boundary; it
  compiles and revalidates graph closure without importing or invoking an
  executor (`packages/microcosm-build/src/microcosm/build/spec_engine/compiler_ir.py:1414`).
  The IR retains the six physically separated surfaces, typed
  entities/artifacts/scopes/columns/references, generated and vintage
  authorities, the normalized domain inventory, and the compiler ABI
  (`compiler_ir.py:401`).
- The US plan contains **38 nodes, 71 edges, 6 deterministic waves, 227
  compiled outputs, and all 18 conditional-ownership cells**.  Its selected
  `legacy-v1` map contains **53 sites, 112 site-to-owner bindings, and 54
  distinct typed owners** (`compiler_ir.py:198`, `:312`, and `:1414`).  A
  shared-core bundle without an imputation graph compiles to the same closed
  types with an empty executable graph rather than acquiring a US-specific
  default.
- `plan.lock.json` is emitted canonically and reproducibly, validated against
  the closed draft-2020-12 plan-lock definition, and refused with pointer-level
  differences after hand editing
  (`packages/microcosm-build/src/microcosm/build/spec_engine/plan_lock.py:54`
  and `:164`; `specs/schema/locks.schema.json:277`).  The lock schema adds only
  emitted compiler objects; the existing bundle-lock and engine-lock
  definitions remain closed and unchanged.

### One-way compatibility compilation

- `compile_to_legacy_payload()` first compiles the IR and then returns the
  complete named constants-era projections: source manifest, support spine,
  all seven imputation objects, take-up raw contract and identity, calibration
  and resolved tail contracts, battery contract, publication/rung grammar,
  spine sampling, stacked authority, and the static checkpoint identity
  components (`packages/microcosm-build/src/microcosm/build/spec_engine/legacy_adapter.py:261`).
  It accepts only a normalized US `ResolvedSpec`, imports no executor, and
  leaves `spec_binding.attestation=mirror-attested`.
- The seven imputation payloads are projected from the normalized bundle in
  production code, including the declared 5/37 split, 19/70 late ledger,
  primary 65-target chain, exact resource semantics, and 18-row ownership
  matrix (`spec_engine/imputation_semantics.py:1810`).  The retained migration
  generator delegates to these projectors; it no longer owns a duplicate
  semantic projection.
- Battery registry views and legacy gates derive from the 131 scalar plus one
  joint registry (`spec_engine/battery_semantics.py:136`).  The publication
  projector derives the exact five rung tokens and both current/historical
  reader regexes (`spec_engine/publication_semantics.py:152` and `:188`).
  Calibration tail references resolve to the PUF support role and primary
  virtual resource rather than copying constants
  (`spec_engine/calibration_semantics.py:151`), and the compiled take-up
  identity hashes the generated 13-program contract
  (`spec_engine/take_up_semantics.py:797`).
- The eight-component stacked receipt and the complete
  `_stacked_checkpoint_base_identity` projection are rebuilt only from the
  resolved bundle and generated ABI lock
  (`spec_engine/stacked_authority_semantics.py:494`, `:940`, and `:1128`).
  Full nondefault input-pin/run-request vectors compare field-for-field and
  byte-for-byte with generation 0; F0 does not construct bundle-mode
  authority.

### Byte-identity gate

- The integrated adapter gate compares canonical JSON bytes against the live
  source/support/take-up payloads, `_authority_receipt`, gap-fill schedule,
  late producer schedule, and overlap ownership
  (`packages/microcosm-build/tests/test_spec_engine_legacy_adapter.py:126`).
  Result: **PASS**, aggregate gate SHA-256
  `fefccb6a0c10cdfd7c4b21ef2957cf77891f300d357ce61223ef2f34b5bcdeb3`.
- Generation-0 identities are unchanged: stacked authority
  `f0b676f6508dbf6bb2b787c42e6b85331bacc57c6649ac7ad15fdaa5884a1b2d`,
  late schedule
  `b1d00afea69b2009d862ca73fff1b63ce56628a8a0790be49918e4bbbecc9fc5`,
  ownership
  `5f64f0aac49e2313177564f71876bffc8c81b3ded4df701e70930e60e9c98356`,
  resource semantics
  `b23c74dcfaaa5780d09caadec2cd110e6b9e268e31057b8584ae814f866bfe17`,
  and take-up resource
  `495dc6ed195eae372a6ba098c6fb894323638a4a7dce1b4fe7efaaf6beb69446`.
  The direct live oracles also compare canonical bytes for every stacked
  authority component and the full checkpoint base identity
  (`test_spec_engine_stacked_authority_semantics.py:62` and `:104`).
- Final D4 repository gate: **6,567 tests collected**;
  `.venv/bin/python -m pytest -q` reached 100% with exit 0.  The only warnings
  were the pre-existing numerical, pandas fragmentation/chained-assignment,
  and sparse-tensor warnings.  The generator `--check`, Ruff, production
  forbidden-import sweep, JSON-schema parse, and `git diff --check` were rerun
  immediately before the commit.
