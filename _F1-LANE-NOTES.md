# F1 lane notes

These notes are the implementation journal for the approved F1 drive lane.
Line citations in section 1 refer to the starting tree at
`7148513302070d9ff39a2d2729726487e709c047` unless a later commit is named.
`PROGRESS.md` is deliberately untouched, as required by the lane charter and
the repository journal convention.

## State

- Branch: `spec-engine-f1`; no push has been performed.
- Deliverable 1 is committed as `cd2b1d60`; deliverable 2 is committed as
  `21297028`. Deliverable 3's implementation and authoritative verification
  are complete and are being closed by the current broker commit.
- The required source-reading sequence is complete.
- The untouched serial baseline was interrupted after 1,489.16 seconds at 10%
  once two independent failures and six cascading fixture errors had appeared.
  No build has been run at any sample fraction.
- The stochastic audit found an F0 defect: the declared 53-site protocol is
  internally closed but is not exhaustive over reachable stochastic kernel
  invocations. The itemized defect is corrected in this lane by an
  independent source-callsite scanner and a 72-site protocol. Generator,
  coverage, focused regression, whole-repository lint, and the complete serial
  suite all pass.
- The generic executor, closed row-scope algebra, compiler direct-node lifts,
  and adversarial fixture suite are complete. The 430-test spec-engine gate is
  green; the authoritative 6,887-test whole-repository suite is also green.
  Deliverable 2 is complete and committed.
- The legacy-v1 RNG, declared-source file, environment, and clock brokers are
  complete. Broker behavior inputs are typed node-reuse inputs; operational
  access receipts carry run provenance but are excluded from both the spec
  identity and node-reuse identity. The authoritative post-broker suite is
  green across 6,976 collected tests.

## Done

- `uv sync --all-packages --extra us` was attempted first. The sandboxed
  global cache refused its metadata write; a writable empty cache then reached
  the expected no-network boundary.
- Sync was recovered without network access by using a writable cache overlay
  whose registry artifacts resolve to the host's lock-identical read-only uv
  cache. The exact successful command was
  `uv sync --all-packages --extra us --offline --no-python-downloads --cache-dir /private/tmp/microcosm-f1-uv-cache-overlay`.
  It installed 100 packages and rebuilt all five workspace packages.
- The authority and stochastic callsite inventories below were checked against
  the current source rather than copied from the F0 pass label.
- The seed correction now gives every one of the 19 missing logical sites a
  typed owner and exact legacy kernel contract. It also corrects the existing
  vehicle-QRF seed from the false archived literal 42 to the live build seed 0,
  covers both Torch reset sites and its `Tensor.uniform_` draw, describes
  pandas' sample-return order exactly, and removes ambient entropy from QRF
  state restoration by constructing `Generator(PCG64(0))` before applying the
  serialized state.
- A later full-suite attempt reached 53% before exposing one additional stale
  US spec-identity assertion in the pool-tool tests. Both occurrences in that
  test module now name the regenerated US digest, and the two affected tests
  pass. The elapsed-time footer for this interrupted attempt was lost when the
  terminal's buffered failure trace was truncated; the reached percentage and
  failing node were recovered from pytest's cache before any edit.
- The authoritative post-correction full suite completed green at 100% in
  3,020.35 seconds (`real`; 2,977.39 user, 451.68 system). A separate
  importlib-mode collection gate counted exactly 6,749 tests. The repository's
  configured double-quiet invocation suppresses pytest's numeric pass/skip
  footer; the run itself showed only passes and expected skips, with no failure
  or error report.
- `seed_callsite_coverage.py` is deliberately independent of `seeds.py`. Its
  filesystem-derived AST scan covers 200 production modules and exactly 274
  physical stochastic/hash/ambient-entropy calls: 119 are ledger-bound and 155
  are typed exemptions, with zero unbound or stale rows. All 154 hash calls are
  classified independently as 91 content identities, 51 source-integrity
  checks, 11 stochastic draws, and one operational subset. The other nine new
  rows are `uuid.uuid4` operational nonces, exclusively typed as operational
  exemptions rather than ledger draws. Lexical fixed-point facts, assignment
  kills, alias chains, class attributes, a runtime API-drift pin, and explicit
  unresolved escape rows make ambiguous new stochastic flows fail the exact
  manifest instead of receiving a guessed family. The existing total
  source-home audit remains in place for all 72 logical sites.
- Six helper-return `.choice` calls (the paired vehicle, financial-asset, and
  archived-rent cap draws) are intentionally labeled
  `stochastic.unresolved.choice` in the physical manifest because their helper
  return types are not locally provable. Each remains bound to its exact
  legacy-v1 training-cap site; this is conservative physical typing, not an
  unowned draw.

## Next

1. Begin deliverable 4: construct every inventoried pool-tool authority from
   compiled IR in `config_authority=bundle` mode while leaving constants mode
   untouched.
2. Extend the cold fixture identity gate to run both modes and require an
   empty byte diff over the complete plan-derived artifact vector.

## 1. Sync, baseline, and authority inventory

### Environment and baseline

- Start: clean `spec-engine-f1` at `71485133`.
- Sync: complete offline as recorded above.
- Baseline attempt: `/usr/bin/time -l .venv/bin/python -m pytest -q`, serial
  and without xdist, reached 10% before being interrupted after the first
  independent failures were fully captured. It ran for 1,489.16 seconds.
- The two independent failures are stale BE and UK `spec_sha256` pins. Six US
  coverage tests then error from their shared fixture because the seed
  protocol/map digest gate is stale. The starting checkout resolves the
  protocol as `6dade07562ec29c56d96ab8e299a4416c679f1c44b18b228e0ef10f21bd6f6ec`
  and the compiled seed map as
  `96140220b6b248c1b3a3567dc0c97df6c08176e6745d8dd55786053f26c43a32`,
  rather than the committed pre-merge pins. The actual starting spec hashes
  are BE `262091db8c7b01b2a3b596aa2468d95855a63703ba9f8ebba2940cf5834c2c83`,
  UK `ed0a0c365dc77f6a0798caeb4ad5de6bcd8e81dd57e3293835c26ffa2b0296da`,
  and US `6e9dce8f0fd3e3f0101103a14d6a08ac8527b90b82d48fa8bad2c4cc70dbdfde`.
  The merge changed `microcosm.calibrate.solve`, which is part of the direct
  stochastic-kernel source attestation, without regenerating these pins.
  This pre-existing F0 defect will be corrected through the generators in the
  same coherent step as the exhaustive-ledger repair.
- A first post-correction full-suite attempt was intentionally stopped after
  356.03 seconds at 7% when independent review found the focused scanner and
  QRF consumption metadata were still incomplete. No failure had appeared in
  that run. The corrected scanner, kernel attestation, and exact QRF/hash
  semantics were then completed before restarting the authoritative baseline.
- The next full-suite attempt was stopped at 11% after 1,235.75 seconds once
  two loader failures were captured. The stateless organization lottery had
  introduced a bespoke `pandas_default_hash_key` value-source label outside
  the closed lock schema, and the minimal-bundle semantic-hash vector still
  described the 53-site protocol. The site now uses the existing exact
  `literal` type (with the implicit pandas key pinned in `seed_material`), the
  lock validates, and only the resulting reviewed semantic/spec/protocol/map
  identities were refreshed. The loader, country, inventory, seed, generator,
  and coverage regression groups all pass after that correction.
- The following full-suite attempt reached 53% and found a pool-tool test that
  still expected the old US `spec_sha256` in two constants-adapter receipt
  fixtures. Runtime resolution produced the already-reviewed regenerated
  digest. Both assertions now use
  `9699f76c5c3146b36c9300be3726471d3272576748d816d9684b50f6cde795d8`,
  and their focused regression passes 2/2. The run was stopped after the first
  independent failure so it could not be mistaken for a passing baseline.
- Host-safety check during the run: `vm_stat` reported 4,137,397 free 16-KiB
  pages (about 63.1 GiB). No process in this lane has approached the 20-GiB
  RSS limit.
- Authoritative corrected baseline: `/usr/bin/time -p .venv/bin/python -m
  pytest -q` completed at 100% with zero failures/errors in 3,020.35 seconds.
  `.venv/bin/python -m pytest -o addopts='--import-mode=importlib'
  --collect-only -q` counted 6,749 tests. The generated bundle and evidence
  `--check` gates, `ruff check .`, and `git diff --check` were rerun after the
  suite and all pass.

### Constants-era authority flip inventory

Artifact shorthand used below:

- `A`, `T`, `S`: assembled, transferred, and simulated checkpoint H5,
  manifest, and operational-receipt sidecar.
- `PB`: primary-QRF bank and manifest.
- `EB`, `LB`: early-gap-fill and late-transfer target banks.
- `P`: final logical pool, normalized manifest, and terminal-gates diagnostic.
- `R`: immutable terminal-gate receipt and Logbook spool row.

| Constants-era authority and current construction site | Compiled-IR authority that must replace it | Stage | Affected artifacts |
| --- | --- | --- | --- |
| Source pins/loaders: ACS package manifest and local-pin validation (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_sources.py:39`; pool tool `:734`) plus rent digest (`us_runtime/housing_inputs.py:149`) | `ir.resource("sources")["sources"]`, source stages/manifest, and `ir.vintage_authorities` | preflight/source load | all descendants; source receipts in `P/R` |
| Compatibility support pool and source-stage JSON selected at pool tool `:3937` and `:3959` | normalized `spine.support_source_pool_metadata`, `spine.support_source_pool`, `sources.stage_manifest`, and `sources.stages` | upstream source preparation | source receipt, `A/T/S`, all descendants |
| Assembly channels and mass shares (`us_runtime/support_provenance.py:31`, `us_runtime/stacked_spine.py:263`, pool tool `:4569`) | `ir.resource("spine")["channels"]` and `["assembly"]`; run-request sampling overlays remain separate | assembly | `A/T/S`, banks through identity, `P/R` |
| Sampling default/rungs/exact-count rule (pool tool `:297`, `:531`; `frame_sampling.py:43`) | `spine.sampling`, referenced `publication.release.rung_fractions`, and `ir.seed_stream_map` | assembly | sampled frame/stack receipt and all descendants |
| Checkpoint envelope, target period, pipeline, operator orders, and batch size (pool tool `:284`, `:1088`; `us_runtime/multispine_pool.py:183`) | `spine.pipeline_contract.artifact_protocol`, its operator orders and batch-size contract; `bundle.dataset_run.target_period`; generated engine ABI lock | all stages | bank routing, `A/T/S`, `P/R` |
| Early gap-fill plan/schedule (`us_runtime/stacked_spine.py:2983`, `:3258`; direction banks at pool tool `:3017`) | early `imputation.families` plus `imputation.gap_fill_schedule`; direct IR projectors are `spec_engine/imputation_semantics.py:711` and `:775` | gap fill | `EB`, `T/S/P/R` |
| Early and post-PUF transfer surfaces (`us_runtime/stacked_spine.py:2983`; embedded at pool tool `:3184`) | ordered imputation families plus `ir.producer_graph.nodes[*].outputs` | gap fill/late DAG | `EB/LB`, `T/S/P/R` |
| ACS predictor tuples, execution profiles, model arguments, and width (`us_runtime/acs_transfer.py:88`, `:97`, `:229`; pool tool `:3026`, `:3095`) | `imputation.predictor_blocks`, `transfer_execution`, `models`, family references, and per-node resolved parameters | gap fill/late DAG | every ACS fit and bank; `T/S/P/R` |
| Primary PUF predictors/outputs/order/checkpoint ABI/model (`us_runtime/puf_support.py:197`; `puf_qrf_chain.py:80`; pool tool `:3060`) | `CompiledNode("primary_puf_qrf")`, primary family, `puf_tax_detail` predictor block, model, and primary-checkpoint contract | primary PUF | `PB`, `T/S/P/R` |
| Frozen 19-group/70-target late ledger (`us_runtime/us_late_producer_registry.py:1393`; bank construction at pool tool `:3043`) | authored-order late `imputation.families`; never the generation-0 width-eight splitter | late DAG | `LB`, `T/S/P/R` |
| Full 38-node producer registry, 71 edges, six waves, schedule/order (`us_runtime/us_late_producer_registry.py:2013`; pool tool `:3095`, bank identity `:2655`) | `ir.producer_graph` and `ir.nodes` | late DAG | `PB/LB`, `T/S/P/R` |
| Conditional 18-cell ownership matrix (`us_runtime/us_late_overlap_ownership.py:98`, receipt `:212`; adapter surface pool tool `:3970`) | `ir.producer_graph.ownership_matrix` and each `ProducerNodeIR.write_scopes` | primary/late DAG | transferred cells and transition receipts; `T/S/P/R` |
| Late input inventories, virtual resources/configs, resource semantics, and transition contract (`us_runtime/stacked_spine.py:4826`, `:4978`, `:5529`; pool tool `:3082`) | producer-node source/input/output/resource/capability/mutation records, normalized graph resource semantics/receipt contract, and compiled node parameters | primary/late DAG | `PB/LB`, transition receipts, `T/S/P/R` |
| Capital-gains tail (`us_runtime/puf_capital_gains_tail.py:76`, identity `:155`; pool tool `:3073`, `:3143`) | `spine.support_roles[puf_tax_detail].tail_support` and the primary node's `@primary_puf_execution_config.capital_gains_tail` binding | primary PUF/preservation/gates | tail receipt, `T/S/P/R` |
| Take-up contract and reviewed engine facts (`us_runtime/take_up_contract.py:234`; identity pool tool `:1175`; seed execution `:3247`) | normalized `take_up.programs`, generated engine ABI lock, referenced source steps, and `ir.seed_stream_map` | source/seed/simulate | seeded state and `S/P/R` |
| Remaining-stage engine manifest, ACS earnings universe, and QBI contract (pool tool `:1162`; derive `:3226`) | generated ABI remaining-stage manifest, producer virtual resource for ACS earnings, and `spine.pipeline_contract.qbi_reconciliation` | late DAG/derive/simulate | `T/S/P/R` and transition authorities |
| Battery thresholds, scalar/joint registries, support profile (`us_runtime/stacked_battery_contract.py:66`; terminal calls pool tool `:3306`) | `ir.resource("battery")` and compiler battery projections | terminal gates | gates, normalized manifest, `R` |
| Eight-component stacked authority and static checkpoint identity (`us_runtime/stacked_spine.py:3069`; pool tool `:1135`, `:1192`) | direct derivation from IR bundle/battery/imputation/publication/sources/spine/take-up plus generated engine lock (`spec_engine/stacked_authority_semantics.py:494`, `:940`) | configuration/all | configured namespace, every bank, `A/T/S/P/R` |
| Seed literals/derivations/private RNG construction (`us_runtime/multispine_pool.py:238`, CLI defaults at pool tool `:531`) | `ir.seed_stream_map` plus the F1 RNG broker; the current 53-site map is incomplete as detailed below | every stochastic stage | stochastic frames, banks, checkpoints, final outputs |
| Release line, ID pattern, rungs, and reader regex (pool tool `:297`, `:304`, `:1316`) | `ir.resource("publication")["release"]` and compiler publication projection | publication | release/H5 identifiers, manifests/gates, `R` |

#### Authorities not yet closed in the IR

The following constants have no complete normalized owner at the starting
commit: `POOL_CHECKPOINT_STAGE_ORDER` (`us_runtime/multispine_pool.py:200`),
the pool stage checkpoint artifact kinds/materializer ledger (pool tool
`:236`–`:290`), final H5/manifest/diagnostic artifact protocols
(`us_runtime/h5_io.py:57`–`:78`), the clone-safe source-id ceiling
(`us_runtime/puf_support.py:2369`), and the sealed plan-derived comparison
vector. F1 must classify each as compiler-emitted artifact protocol or an
explicit code/materializer ABI; silently retaining it as configuration is not
an acceptable flip.

The physical checkpoints also combine logical stages: `A` is assembly;
`T` combines source preparation, gap fill, primary PUF/tail, and the late DAG;
`S` combines derive, seed, and simulation. Stage-by-stage receipts therefore
need intermediate logical digests in addition to the three durable files.

### Stochastic callsite audit

Audit command classes were `np.random`, `default_rng`, `RandomState`, standard
`random.`, stochastic uses of `hashlib`/pandas hashing, seeded framework/model
constructors, and their reachable QRF invocations. No production import or use
of the stdlib `random` module was found. Ordinary SHA-256 artifact/identity
hashes are deterministic content digests, not draws, and are excluded.

The existing 53-site result is circular: `inventory_coverage.py:1643` compares
the protocol to `EXPECTED_SEED_GROUPS`, while
`test_spec_engine_seeds.py:255` only checks that each *declared* site has some
source anchor. Neither scan discovers undeclared calls.

#### Declared sites and exact reachable callsites

| Ledger id | Current stochastic callsite(s) |
| --- | --- |
| `survey_sample_asec`, `survey_sample_acs` | `frame_sampling.py:258` generator and `:267` choice; channel calls `stacked_spine.py:621`, `:627` |
| `puf_clone_attachment` | `puf_support.py:740`–`:741` |
| `puf_archived_aggregate_disaggregation` | `puf_source_agi.py:380` generator; draws `:106`, `:110`, `:180` |
| `puf_live_aggregate_disaggregation` | `puf_aggregate_records.py:418` generator; draws `:920`, `:1434` |
| `ssi_weighted_replacement_training` | `ssi_disability_criteria.py:660`–`:665` audit replay and `:913`–`:919` runtime sample |
| `ssi_archived_qrf_model` | `ssi_disability_criteria.py:972`–`:998` and the shared QRF kernel below |
| `sipp_vehicle_training_cap` | x31 mixer/generator/choices `sipp_vehicles.py:299`–`:357` |
| `sipp_vehicle_qrf_model` | QRF `sipp_vehicles.py:846`–`:853` (does **not** cover the vehicle-count sklearn RNG gap below) |
| `sipp_financial_asset_training_cap` | x31 mixer/generator/choices `sipp_financial_assets.py:307`–`:378` |
| `sipp_financial_asset_qrf_models` | `sipp_financial_assets.py:667`–`:695` |
| `acs_rent_archived_training_cap` | x31 mixer/choices `housing_inputs.py:740`–`:810` |
| `sipp_tip_training_cap` | `sipp_tips.py:420`–`:423` |
| `scf_household_source_selector` | `scf_wealth.py:853`–`:857` |
| `scf_financial_asset_qrf_model`, `scf_net_worth_qrf_model` | shared QRF helper `scf_wealth.py:773`–`:781`, called at `:812`–`:818`, `:976`–`:982` |
| `scf_auto_loan_qrf_model` | `scf_auto_loans.py:441`–`:447` |
| `acs_transfer_family_seed` | SHA-256 derivation `acs_transfer.py:2902`–`:2904` |
| `acs_transfer_pattern_seed` | SHA-256 derivation `acs_transfer.py:2907`–`:2922` |
| `primary_qrf_fit_draw` | `puf_qrf_chain.py:219`–`:225`, `:363`–`:375` |
| `acs_qrf_fit_draw` | `acs_transfer.py:1341`–`:1358`, `:1513`–`:1524`, `:1598`–`:1607` |
| `source_aca_assignment` | call `source_runtime.py:491`–`:495`; BLAKE2b helper `:1297`–`:1324` |
| `source_count_calibration` | call `source_runtime.py:571`–`:575`; same helper |
| `source_joint_count_calibration` | call `source_runtime.py:728`–`:732`; same helper |
| `snap_take_up_assignment` | `snap_take_up.py:130`–`:160`, `:235` |
| `pregnancy_assignment` | `pregnancy.py:128`–`:160`, `:205` |
| `wic_claim_assignment` | `wic_claim.py:387`–`:405`, `:424` |
| `snap_discretionary_exemption_assignment` | `snap_discretionary_exemption.py:133`–`:165`, `:209` |
| `immigration_ead_workers_assignment`, `immigration_ead_students_assignment` | helper `immigration.py:431`–`:465`; calls `:583`–`:595` |
| `ssi_take_up_assignment` | hash `ssi_take_up.py:502`–`:511`; call `:695` |
| `medicaid_take_up_assignment` | `medicaid_take_up.py:119`–`:133` delegates to hash `take_up.py:148` |
| `snap_state_take_up_assignment` | `snap_state_take_up.py:164` delegates to `snap_take_up.py:152` |
| `tanf_take_up_assignment`, `eitc_take_up_assignment` | call `take_up.py:255`–`:260`; hash `:148` |
| `adult_care_weighted_prefix_assignment` | `adult_care.py:475`–`:476` |
| `capital_gains_tail_random_rank` | `puf_capital_gains_tail.py:1416`–`:1417` |
| `torch_calibration_reseed` | `packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:1126`, `:1600`; the declared reset boundary currently omits the first reset |
| `exact_k_pcg64_selection` | `packages/microcosm-calibrate/src/microcosm/calibrate/exact_k.py:170`, `:241`, `:248`, `:333`, `:501` |
| eleven `*_training_cap` sites | generator/sample pairs in `prior_year_income.py:450`, `childcare.py:298`, `retirement_contributions.py:339`, `disability_benefits.py:343`, `housing_inputs.py:1078`, `workers_compensation.py:303`, `retirement_distributions.py:432`, `child_support.py:324`, `energy_subsidy.py:326`, `other_health_insurance.py:388`, `weeks_unemployed.py:904` |
| `legacy_geography_ladder` | `geography_ladder.py:286`, `:300` |
| `legacy_puma_ladder` | `puma_ladder.py:334`, `:660`, `:693` |
| `legacy_congressional_district_assignment` | `congressional_district_geography.py:128`, `:145` |

All declared QRF sites ultimately consume shared kernel draws at
`packages/microcosm-fit/src/microcosm/fit/qrf.py:216`, `:228`, `:507`, `:640`,
`:960`, `:985`, `:1084`–`:1085`, `:1129`–`:1131`, `:1352`, `:1428`, and
`:1487`. These are kernel-contract consumption points, not independent
per-target streams.

#### F0 defect: reachable sites with no ledger id at the starting commit

The following are reachable from the US build but have no valid site/owner
binding. The superficially related training-cap entries cover only pandas
sampling, not the subsequent model stream.

| Missing logical site | Unbound callsite | Required owner |
| --- | --- | --- |
| child-support QRF | `child_support.py:352` | source stage `child_support_inputs` |
| childcare QRF | `childcare.py:326` | `childcare_inputs` |
| disability-benefits QRF | `disability_benefits.py:371` | `disability_benefits_input` |
| energy-subsidy QRF | `energy_subsidy.py:354` | `energy_subsidy` |
| archived ACS-rent QRF | `housing_inputs.py:886` | `acs_rent` |
| housing-assistance QRF | `housing_inputs.py:1102` | `acs_rent` |
| organization-wages QRF | `org_wages.py:710` | `org_wages` |
| other-health-insurance QRF | `other_health_insurance.py:416` | `other_health_insurance_premiums` |
| prior-year-income QRF | `prior_year_income.py:490` | `prior_year_income` |
| primary-PUF non-checkpoint fallback | `puf_support.py:1683` | existing primary producer plus an attested fallback path |
| retirement-contributions QRF | `retirement_contributions.py:370` | `retirement_contributions` |
| retirement-distributions QRF | `retirement_distributions.py:462` | `retirement_distributions` |
| Head Start QRF | `sipp_head_start.py:681` | `sipp_head_start` |
| tips QRF | `sipp_tips.py:425` | `sipp_tips` |
| voluntary-filing QRF | `voluntary_filing.py:893` | `voluntary_filing_input` |
| weeks-unemployed QRF | `weeks_unemployed.py:943` | `weeks_unemployed_input` |
| workers-compensation QRF | `workers_compensation.py:331` | `workers_compensation_input` |
| vehicle-count sklearn forest | `sipp_vehicles.py:809`–`:825` | `vehicle_assets`; not described by the QRF site |
| organization-wages weighted hash lottery | `org_wages.py:667`–`:689` | `org_wages`; pandas hash is converted to a uniform draw |

#### F0 defect disposition

The correction assigns, in table order, the ledger ids
`child_support_puf_qrf_model`, `childcare_puf_qrf_model`,
`disability_benefits_puf_qrf_model`, `energy_subsidy_puf_qrf_model`,
`acs_rent_qrf_model`, `housing_assistance_puf_qrf_model`,
`org_wages_qrf_model`, `other_health_insurance_puf_qrf_model`,
`prior_year_income_puf_qrf_model`, `primary_puf_monolithic_qrf_model`,
`retirement_contributions_puf_qrf_model`,
`retirement_distributions_puf_qrf_model`, `sipp_head_start_qrf_model`,
`sipp_tip_qrf_model`, `voluntary_filing_qrf_model`,
`weeks_unemployed_puf_qrf_model`, `workers_compensation_puf_qrf_model`,
`sipp_vehicle_count_random_forest_model`, and `org_union_hash_lottery`.
They are bound to the required source-stage or producer-node owners in the
generated spine. The corrected closed inventories are 72 sites, 14 streams,
57 owner rows, and 131 site-owner bindings.

Regeneration yields:

- US `spec_sha256`:
  `9699f76c5c3146b36c9300be3726471d3272576748d816d9684b50f6cde795d8`.
- BE `spec_sha256`:
  `0938096be78feaa48a73a94a642595b679504814eadf78d332259b5e55f0dda3`.
- UK `spec_sha256`:
  `284bd4de3984c0b5a7650cb1bad994c52c3edf8db73781a76933999543bf0790`.
- Legacy-v1 protocol digest:
  `5c6ca5ccfbbd3897b23d29dc46196ef1ae110d6d1f2974ce360fac722075a73d`.
- Compiled seed-map digest:
  `40628dc8242840fb09106389d8887e778f121dbe320a2c2d47f2aeb0b0baad48`.
- Field-usage closure: 41,867/41,867 configuration fields consumed (32,218
  authored normative fields plus 9,649 resolved binding fields); inventory
  coverage remains 40/40.

The owner-requested starting-tree to corrected identity transition is:

| Identity | Starting checkout | Corrected deliverable 1 |
| --- | --- | --- |
| US `spec_sha256` | `6e9dce8f0fd3e3f0101103a14d6a08ac8527b90b82d48fa8bad2c4cc70dbdfde` | `9699f76c5c3146b36c9300be3726471d3272576748d816d9684b50f6cde795d8` |
| BE `spec_sha256` | `262091db8c7b01b2a3b596aa2468d95855a63703ba9f8ebba2940cf5834c2c83` | `0938096be78feaa48a73a94a642595b679504814eadf78d332259b5e55f0dda3` |
| UK `spec_sha256` | `ed0a0c365dc77f6a0798caeb4ad5de6bcd8e81dd57e3293835c26ffa2b0296da` | `284bd4de3984c0b5a7650cb1bad994c52c3edf8db73781a76933999543bf0790` |
| `legacy-v1` protocol | `6dade07562ec29c56d96ab8e299a4416c679f1c44b18b228e0ef10f21bd6f6ec` | `5c6ca5ccfbbd3897b23d29dc46196ef1ae110d6d1f2974ce360fac722075a73d` |
| compiled seed map | `96140220b6b248c1b3a3567dc0c97df6c08176e6745d8dd55786053f26c43a32` | `40628dc8242840fb09106389d8887e778f121dbe320a2c2d47f2aeb0b0baad48` |

Focused verification is green: the seed/source-census tests pass (23 tests),
the complete QRF regression file passes (43 tests), the coverage/field/
inventory batch passes, and the identity/compiler/plan-lock/US-bundle batch
passes. Both generated-file `--check` commands, `git diff --check`, and
whole-repository `ruff check .` pass. The source audit explicitly proves that
the QRF implementation digest covers its public dispatch (`microcosm.fit`),
weight resolution (`microcosm.fit.model`), and implementation
(`microcosm.fit.qrf`). An independent post-fix audit reran the namespace,
alias-chain, reassignment, branch-merge, class-attribute, runtime-API, and UUID
probes and returned PASS.

`acs_pums.py:428`–`:431` is a callable smoke-subset hash branch, but the pool
constructs `AcsPumsSource` without `max_households`; it is classified as
an exact typed exemption for F1: deterministic operational smoke subsetting,
outside the published producer graph, with no valid source-stage or pipeline
owner. If raw ACS construction enters the graph, this must become an owned
`acs_smoke_household_hash_rank` site. Exported `fit/holdout.py:52` has no
production pool caller. UK-only stochastic modules are outside the selected
US `legacy-v1` namespace.

The starting shared QRF restore helper at
`microcosm-fit/src/microcosm/fit/qrf.py:640` called `default_rng()` before
replacing its state. Although that ambient entropy did not affect output
bytes, it violated the F1 ambient-access contract. Deliverable 1 now constructs
`Generator(PCG64(0))`, restores the validated state, and keeps the callsite
bound to `primary_qrf_fit_draw`.

Operational UUID entropy occurs at pool tool `:1327`, `:3625`, `:4084`, and
`:4388`; `logbook.py:1085`, `:1101`; `logbook_adoption.py:254`; and
`us_runtime/h5_io.py:274`, `:884`. It does not affect numerical node reuse,
but it is an ambient operational effect that the F1 broker/access receipt must
classify and that certification must freeze or normalize. The scanner proves
all nine are exclusively `operational_nonce` exemptions.

### Known host-safety boundary for later certification

Existing cold 1% (`f001`) telemetry proves that the primary-QRF path cannot be
launched under this lane's per-process 20-GiB ceiling without a
behavior-preserving memory redesign or another host. The current-RSS maxima in
four prior `stage_profile.json` receipts are 84,729,479,168 bytes (78.91 GiB),
90,351,255,552 bytes (84.15 GiB), 103,374,684,160 bytes (96.28 GiB), and
104,102,936,576 bytes (96.95 GiB); every row has `sampling_error: null`.
These are not estimates of cumulative allocation. Full PUF donors remain
unsampled at a 1% output rung, and D4 requires cold execution rather than bank
reuse.

The evidence files are respectively:

- `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r10/populace_us_2024_stacked_pool_verify.checkpoints/stacked/2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c/primary-qrf/stage_profile.json`
- `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r9/populace_us_2024_stacked_pool_verify.checkpoints/stacked/99376eea69594de6c88e2f68f76e35e6590a3f1cdc2849953257f0de3a7d2f46/primary-qrf/stage_profile.json`
- `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r7/populace_us_2024_stacked_pool_verify.checkpoints/stacked/c16d0206589b466d7614e9ed3536b80cb7c9da48fa4b6b4a5f6f3fca42f2cb91/primary-qrf/stage_profile.json`
- `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/smoke-f001-r6/populace_us_2024_stacked_pool_verify.checkpoints/stacked/c16d0206589b466d7614e9ed3536b80cb7c9da48fa4b6b4a5f6f3fca42f2cb91/primary-qrf/stage_profile.json`

Inputs are present, so this is a resource-safety blocker rather than a
missing-data blocker. Per the charter, this lane will not start a real 1%
primary-QRF process that is known to breach the ceiling; if the blocker remains
when deliverable 5 is reached, the lane stops there honestly with fixture
receipts and a green suite.

### Preflight finding and owner resolution for deliverable 4

The approved requirements contain a raw-byte contradiction that no current IR
projection can resolve:

1. D6 makes `microcosm-us-2024-*` the normative bundle release line at the F1
   flip (`publication.yaml`), while untouched constants mode emits and embeds
   `populace-us-2024-*` (`tools/build_us_multispine_pool.py:304`–`:318`,
   `:1316`). Choosing the bundle's legacy prefix would reproduce constants but
   would not let the normative bundle authority drive publication.
2. D3 requires bundle mode to carry identity generation 1 and the spec-binding
   triad, while constants-mode generation-0 sidecars do not. Those receipts
   cannot be raw-byte equal if constants mode is untouched.
3. Existing cold-run envelopes also contain absolute paths, durations, UUIDs,
   and clock values. A canonical comparison could exclude or normalize these,
   but the present plan lock has no sealed comparison/canonicalization vector,
   and the charter separately says every covered artifact is byte-identical.

The current IR also lacks the physical whole-pipeline DAG, logical-stage to
checkpoint/bank mapping, typed operation-to-kernel bindings, final artifact
protocols, and a closed certification plan. These can be added as
compiler-emitted execution ABI rather than country-specific configuration.

The owner resolved the apparent contradiction on 2026-08-19 in
`_F1-CHARTER-R2.md`. Deliverables 4 and 6 use two sealed tiers:

- The normative artifact vector remains raw-byte exact with no exclusions or
  canonicalization.
- Provenance and operational receipt fields compare structurally under a
  compiler-emitted, sealed `plan_lock` vector. Every differing field must have
  exactly one declared rule: `equal_after_normalizing_prefix`,
  `expected_to_differ_by_generation`, or `operational_excluded`; any unlisted
  receipt difference fails.
- Constants mode keeps `populace-us-2024-*`, bundle mode uses
  `microcosm-us-2024-*`, and readers accept both. Bundle receipts must match
  the loader's generation-1 provenance triad while constants receipts remain
  generation 0. `node_reuse_key` must be byte-equal across modes and must not
  include provenance.

This ruling permits the lane to proceed through deliverable 4 without changing
constants mode or weakening normative byte identity.

## 2. Generic executor core

### State

- D2 source, fixture coverage, and the authoritative full suite are complete;
  no pool build or sample rung has been invoked.
- New generic code is confined to `spec_engine/executor.py` and
  `spec_engine/scope_algebra.py`, with direct compiler projections in
  `compiler_ir.py`. The executor and scope modules contain no country or
  program literals and do not import the pool runtime.
- The F1 compiler ABI is version 2 with digest
  `845d3963bc03efb68cdc27e013450bd7048598bc55e73903fc16e908e12724ea`.
  The D1 ABI was version 1 at
  `3f6996bc1d4d187d669c55b525ff3910d6f19dc8ccadff464c4561aeca127f27`.

### Done

- `CompiledNode` now carries compiler-bound execution rank, dependencies,
  inputs, outputs, orthogonal capabilities, mutation triples, exact write
  scopes, the closed scope registry, row-classifier pin, kernel pin, effective
  seed sites/streams, and transitive node slices. Each direct lift is checked
  against its hashed resolved parameter before dispatch. The compiler emits a
  separate kernel projection so changing a kernel ref or pin cannot be hidden
  by recomputing `node_key` while leaving the authored node unchanged.
- Producer ordering uses the compiled ranks with a deterministic Kahn order.
  It rejects missing/duplicate/noncontiguous ranks, backward or unknown
  dependencies, cycles, an incorrect transitive dependency closure, and
  incomparable nodes with overlapping exact or structural write authority.
  The empty graph has the total order `()`.
- The scope registry implements the finite Boolean algebra over compiler-owned
  atoms, including canonical union/intersection/complement/difference,
  equality, overlap, exhaustiveness, serialization identity, and refusal of
  unknown predicates. Runtime classifiers are orchestration-owned and must
  match the compiler-bound classifier ref, digest, and predicate space before
  an expand kernel runs.
- `execute_node` gives a kernel a detached immutable projection and accepts
  only a declarative `KernelPatch`. It deep-copies object cells, detects direct
  input mutation, applies to a private projection, computes an exact structural
  diff, checks delta semantics and all executable mutation pre/postconditions,
  validates entity/column/row scope, and returns a sealed transaction. Base,
  result, patch, diff, node id, and node key are bound in the application
  envelope, and a patch cannot be replayed on another base.
- The full diff covers table/link cells, stable keys, row order and index
  association, exact dtype representations, column and row axis contracts,
  DataFrame flags/attrs/subclass metadata, typed weights and storage order,
  strata values/name/attrs/dtype/index/subclass metadata, links, memberships,
  virtual receipts, mass history, frame metadata, row atoms, and all inventory
  order/target contracts. Exact comparison distinguishes `1` from `True`,
  `+0.0` from `-0.0`, and `None` from `NaN`.
- All seven `structural_delta` kinds have executable fixture kernels. The
  validator refuses stable-key rewrites, unrelated cell/strata/index effects,
  non-exact membership/link mirrors, column additions outside JOIN, output
  order/dtype/nullability violations, weight-kind changes outside REWEIGHT,
  invalid filter alignment, malformed clone lineage/block order/remapping,
  and failure to conserve mass per source lineage.
- Input projection validation is fail-closed: undeclared columns are hidden,
  every physical input must exist, ordered OR-of-AND alternatives enforce
  `column_present`/`finite_numeric`/`non_null` over the declared closed row
  scope, and an explicit absence receipt can excuse only actual absence, never
  a present-but-invalid value. Physical outputs require a compiled column
  contract before dispatch and are validated after execution even when the
  column already existed.
- The generator no longer labels seedless post-clone nodes `seeded` through a
  program-name exception. It derives effective seed ownership generically from
  typed node/source joins. This corrected two F0 declaration defects
  (`with_us_education_inputs` and `with_us_medicare_take_up_input`) to
  deterministic/bitwise without changing the generation-0 payload or runtime
  behavior. The compiled US plan has 34 seeded and four deterministic nodes.
- The resulting semantic identities are US `spec_sha256`
  `f6ae6cde8c08e8695128ab9f29509168c1ee042faa97a28a3c94977262987309`
  (D1: `9699f76c5c3146b36c9300be3726471d3272576748d816d9684b50f6cde795d8`)
  and authored producer-node digest
  `754cccfa2e9fdcb0137669e76f0f7b839a263ff047550e5e560231ae80d7f95a`
  (D1: `a83363de26cad0144b5a98b36b4bca49542e37a7b9fee3d7e541f692deeff864`).
  The generated legacy payload remains byte-identical; schedule/order and the
  72-site legacy-v1 protocol are unchanged. Node keys intentionally change
  because the ABI and their direct kernel/execution projections changed.
- Adding the executor introduced one deterministic SHA-256 content-identity
  call and two production modules to the independent seed scan. The closed D2
  census is 202 modules and 275 physical calls: 119 ledger bindings plus 156
  typed exemptions; all 155 hash calls are independently classified. There
  are zero unbound, stale, or conflicting callsite rows.

### Adversarial review and verification

Two independent read-only reviews reproduced twelve fail-closed defects in
the initial implementation. None was waived. Regression tests now prove the
disposition of every finding:

1. categorical dtype order, DataFrame flags, row-axis contracts, strata
   metadata, and Series subclass metadata cannot piggyback on a valid write;
2. stable key bytes and membership/link mirrors use representation-exact
   equality, and REORDER cannot mutate an unrelated entity index;
3. kernel direct lifts, required inputs and alternatives, existing output
   contracts, node-envelope identity, and dependency transitive slices are
   checked independently; and
4. absence receipts cannot mask malformed present inputs and EXPAND cannot
   promote a typed-weight kind.

Final D2 fixture evidence:

- executor: 89 tests pass;
- executor + scope algebra + compiler IR: 138 tests pass in 21.38 seconds;
- compiler IR + plan lock + executor + scope algebra + country bundles: 159
  tests pass after the three final regressions;
- complete spec-engine plus US-bundle gate: 430 tests pass in 542.57 seconds;
- generated US bundle `--check`: pass at the spec identity above;
- coverage generation and `--check`: 41,867/41,867 fields and 40/40 inventory
  checks;
- focused Ruff, Python compilation, and `git diff --check`: pass.
- authoritative serial whole-repository suite: 100% green across 6,887
  collected tests in 2,633.70 seconds (`real`; 2,786.77 user, 557.83 system).

The first authoritative D2 full-suite attempt was stopped at 54% after
1,490.90 seconds when the constants-adapter pool-tool test found two assertions
still pinned to the D1 US `spec_sha256`. Runtime resolution produced the
reviewed D2 identity above and the live-constants payload equality assertion
had already passed. Both stale expectations now name the generated D2 digest;
their focused tests pass 2/2. This was an identity-fixture update, not a gate,
seed, threshold, or runtime behavior change.

The kernel and row-classifier registries are explicit trusted orchestration
boundaries: a caller cannot substitute a different ref/digest than the
compiler pin, but the F0 kernel namespace digest inventories implementation
ids rather than callable bytecode. Deliverable 4 must bind the actual adapter
implementation/code inventory before the final `node_reuse_key` is certified;
the executor deliberately does not introspect or guess Python callable hashes.

### Next

1. Implement D3 brokers. Pure/seeded ambient-access instrumentation and the
   legacy-v1 RNG broker must use these compiled capability and seed grants.

## 3. Brokers

### State and identity boundary

- `spec_engine/brokers.py` now supplies the generic RNG, file, environment,
  and clock broker boundary used by the executor. A `BrokerSession` is bound
  to one compiled node, one node key, and one `run_provenance_identity`; the
  kernel receives only the restricted `KernelBrokerSession` projection.
- The US `spec_sha256` remains
  `f6ae6cde8c08e8695128ab9f29509168c1ee042faa97a28a3c94977262987309`
  and the seed-protocol identity remains
  `5c6ca5ccfbbd3897b23d29dc46196ef1ae110d6d1f2974ce360fac722075a73d`.
  The generated legacy payload remains byte-identical. The compiler ABI is
  generation 3 with digest
  `79a931e68a69bc83577322c0f7f5ffcecd39a7cd830200e73115d64872695804`;
  it binds executor contract `compiled-node-brokered-contracts-v2` and broker
  semantics `legacy-v1-ledger-broker-semantics-v2`.
- Broker event logs are an operational receipt surface. They contain the full
  provenance triad and a receipt digest, but neither the receipt nor the
  provenance triad enters `spec_sha256` or `node_reuse_key`. The reuse key
  admits only the required identity-generation scalar plus typed RNG/source
  behavior identities, transitive content identities, implementation digest,
  and materializer/backend ABI inputs. Tests refuse provenance or receipt
  fields even when disguised under aliases.
- Source behavior identity is path-free: it binds logical source id, content
  digest, and byte size. Equal declared bytes at different host paths yield
  the same reuse identity; content changes rekey it. RNG and source behavior
  objects are sealed, owner-bound, and same-session-bound, so forged or
  cross-session values cannot be injected into a reuse key.

### Generic broker contract

- Every RNG stream token is validated against the compiled node's seed grants
  and the 72-site ledger. Dispatch enforces `rng_family`, exact legacy kernel,
  ordered seed material, consumption order, and reset boundary. There is no
  arbitrary integer-seed escape hatch and no caller-owned generator state.
- The broker-owned adapters cover serialized-state NumPy PCG64 generators,
  QRF child/target pairs, SHA-derived chained seeds, BLAKE2b and pandas-hash
  stateless draws, exact pandas sampling, sklearn random-forest fit/predict,
  and serialized-state Torch generators. Derived seeds are opaque handles and
  a downstream draw must reference a handle produced earlier in the same
  session at the ledger-declared boundary.
- Captured constants-mode vectors prove legacy-v1 semantics for real survey
  sampling, SCF composite seeding, QRF paired streams, ACS derived QRF seeds,
  BLAKE2b uniforms, pandas sample ordering, sklearn random-forest predictions,
  and the Torch reset/uniform sequence. The QRF state-restore path constructs
  a fixed PCG64 shell before applying state and consumes no ambient entropy.
- The file broker accepts logical `declared_source_read` grants only, records
  content digest and byte count, and refuses undeclared or changed sources.
  Its read lease exposes bytes/text/iteration without exposing the raw stream;
  an unclosed or privately probed lease taints the receipt and prevents a
  successful seal.
- Environment and clock reads are explicit, logged broker operations.
  Deterministic/pure and seeded kernels execute with ambient file, environment,
  clock, NumPy/Python RNG, entropy, UUID, process, socket, and loaded-Torch
  access instrumented and refused. Direct pandas sampling is also guarded.
  A caught refusal is sticky: the kernel cannot suppress it and then obtain a
  complete receipt.
- Kernel callables are required to be plain functions and their prebound
  object graph is inspected before dispatch. Same-module aliases, captured
  original ambient primitives, imported-module re-exports, private RNGs,
  mutable/uninspectable containers, and arbitrary mutable authority objects
  are refused. Row classifiers run inside the same guard and cannot use the
  kernel's broker grants.
- The executor verifies that a complete receipt belongs to the exact node,
  node key, session, and provenance context before applying its structural
  transaction. Receipt identity is carried in the operational application
  envelope while patch, result, and reuse identities remain behavior-only.

This is an instrumented Python execution boundary, not an operating-system
sandbox. Ordinary kernel code cannot reach private sessions, streams, raw
generators, or ambient aliases, and the static/dynamic guards are fail-closed
for supported callable shapes. Python's deliberate reflection primitives such
as `object.__getattribute__` are outside that threat model; bundle-mode
adapters must therefore remain reviewed orchestration code rather than
untrusted plugins.

### Adversarial review and verification

Two independent read-only reviews exercised the initial broker implementation.
Every reproduced defect was fixed and pinned by a regression test. The final
dispositions include:

1. raw integer seeds, arbitrary SHA material/chaining, and unproduced derived
   seeds are refused;
2. source behavior and RNG behavior are separately typed, path-free where
   required, and bound to the issuing session and node owner;
3. kernels receive no raw session and cannot capture raw/imported broker or
   ambient authority through helpers, module aliases, or container graphs;
4. generator, Torch, and file leases do not expose their live mutable state;
5. receipts carry schema-valid run provenance and recursively immutable event
   details, and caught refusals cannot be sealed as complete; and
6. operational broker implementation/logging details do not silently rekey
   nodes: semantic changes require an explicit broker-semantics ABI bump.

Final D3 evidence:

- focused broker + executor + compiler-IR + schema tests: pass;
- complete `test_spec_engine_*.py` gate: pass;
- generated US bundle `--check`: pass;
- coverage generation and `--check`: 41,867/41,867 configuration fields and
  40/40 inventory checks, with 72 seed sites and 38 compiled nodes;
- whole-repository Ruff and `git diff --check`: pass;
- authoritative serial whole-repository suite: 100% green across exactly
  6,976 collected tests in 2,949.50 seconds (`real`; 3,009.34 user, 793.00
  system), with only expected skips and the repository's existing warnings.

### Production-routing boundary for deliverable 4

The broker API and executor enforcement boundary are complete, but the pool
tool has not yet been flipped to invoke production kernels through it. The
independent source scanner therefore still reports 119 physical stochastic
bindings whose current constants-mode callsites draw directly. `seeds.py`
correctly remains `mirror-only-until-f1`; this section does **not** claim the
acceptance condition that every reachable production stochastic callsite is
broker-routed. Deliverable 4 must route those callsites through compiled seed
grants while preserving the exact constants-mode byte behavior, then change
the enforcement status only after the independent inventory proves it.

### Next

1. Compile the bundle once at pool-tool entry and build a generic immutable
   runtime-authority object directly from `CompiledSpecIR` (never from the
   generated legacy payload).
2. Thread that authority object stage by stage through identity, resume,
   assembly, gap fill, primary and late imputation, seeding, gates, manifest,
   publication, and banks, keeping constants mode as the untouched oracle.
3. Add hostile-constants tests and the cold dual-mode fixture gate before any
   sample build.

## 4. Bundle-mode authorities

### Compiler-issued runtime capability and execution ABI

- The compiler ABI is now generation 4 with digest
  `10e17024929eb7304e00dd655c29e300d275d524c0ebf72b4453361523f0f56e`.
  Its new `execution_abi` contract is authored by an exact ordered stage
  partition in `spine.pipeline_contract.execution_stages`; it contains four
  logical stages, three durable checkpoints, the code/materializer ABI, 30
  concrete normative artifact rows, scoped receipt-comparison rules, and the
  resume predicate. The execution-ABI digest is
  `694bb34e1d41a23143b9c02194810161b54e2228df8810f0c0deafdfd81fff80`.
- The stage partition is generated from the existing physical operator order,
  is schema-closed, must be a contiguous exact partition, and binds the
  producer graph exactly once. It replaces hard-coded numeric stage splits in
  compiler output. The US normative identity consequently moved from
  `f6ae6cde8c08e8695128ab9f29509168c1ee042faa97a28a3c94977262987309`
  to `a57b484c8993ec81e1c2c0edb9ef29dbae33c17051bdec58ed710774d73906b2`;
  this is an authored F1 execution contract, not a runtime behavior change.
- `RuntimeAuthorities` is the immutable compiler-to-runtime capability. It
  retains only normative, execution-profile, run-request, generated/vintage,
  declared-source, execution-ABI, seed-map, compiled-node, and narrow
  compatibility projections. It does not retain `CompiledSpecIR`, normalized
  documents, the legacy aggregate payload, or any country constants object.
  Its current capability digest is
  `853ce81c5ddb44273c6c8e19094d0eb9d7b06c2199c7fd0f6e506d47e578005e`.
- Executable producer graph/node compilation now uses only the merged
  normative and execution-profile projection. The full authored graph remains
  a compiler-only plan-lock/coverage surface. Documentation and operational
  producer fields therefore cannot rekey a node or escape into the runtime
  capability; the coverage inventory separately proves exact authored-graph
  preservation and exact behavior-projection compilation.
- `USSpecAuthority` is a narrow US runtime adapter over that capability. It
  exposes recursively frozen behavior resources, compatibility projections,
  declared sources, and typed node queries; it cannot reopen the bundle or
  invoke the legacy aggregate adapter. Program ids remain bundle data rather
  than named accessors.

### Source-manifest closure

- The two ACS archive records now carry generated `acquisition` objects. The
  filename, URL, and source directory are operational; `verified_on` is
  documentation. These eight scalar bindings are generated directly from the
  existing packaged manifest and therefore are not a second hand-authored
  authority. They do not change `spec_sha256`, the normative source digest, or
  the engine ABI lock.
- The runtime compiler selects only source id/role/content pins, loader and
  vintage references, and acquisition receipts into a sealed declared-source
  registry. Its digest is
  `491dd8df598f18c0a94ec03a3448733bc6b596bc59484dc626fb02f557fe2f22`.
  Operational or documentation acquisition edits change this runtime receipt
  authority but leave `spec_sha256`, node keys, and node-reuse inputs alone.

### Provenance/reuse correction and pool selection seam

- D3's journal wording that the reuse key admitted an identity-generation
  scalar was incorrect under approved decision D3. `node_reuse_key` now
  rejects `identity_generation` alongside every other provenance field. Two
  otherwise identical generation-0 and generation-1 run-provenance receipts
  produce the same reuse key. Provenance remains recorded only in
  `run_provenance_identity` and operational broker/build receipts.
- `--config-authority bundle` now performs exactly one
  `load_bundle("us") -> compile_spec -> compile_runtime_authorities` selection
  before configured identity construction. It never invokes
  `compile_to_legacy_payload`. The hidden capability is excluded from the
  mapping/repr receipt view so its digest and provenance cannot accidentally
  enter checkpoint, bank, semantic-artifact, or reuse identities. Constants
  and constants-adapter modes retain their existing paths.
- This is the authority-selection seam, not yet a claim that every physical
  stage is flipped. The pool driver still constructs stage authorities from
  constants after selection. The required next work is to retain the typed
  execution ABI and seed map in the US adapter, build sealed typed stage
  materializers, and thread them through the existing stage entry points with
  `None` preserving constants mode.

### Verification at this checkpoint

- Generated US bundle `--check`: pass.
- Coverage generation: 41,886/41,886 normative/resolved fields and 40/40
  inventory items; refreshed evidence is committed with this step.
- Spec-engine run through the first production seed-inventory assertion:
  417 tests passed. The sole failure was the fail-closed production module
  count increasing from 202 to 204 for the two new runtime-authority modules;
  after reviewing the new modules and updating that exact count, the remaining
  142 spec/US-bundle/source/adapter tests passed.
- Focused bundle-selection/constants-adapter pool-tool gate: 14/14 pass,
  including constants-adapter byte-identical fixture checkpoints.
- No build, sample or otherwise, was run in this step.

### Next

1. Complete the typed source, support, publication, checkpoint, seed, and
   generated-engine materializers without constants or legacy-payload fallbacks.
2. Thread a compiler-sealed US pool runtime plan stage by stage; dynamic
   checkpoint identities must use the actual clone/run fields rather than
   overlaying the static default projection.
3. Add the plan-derived raw-byte/receipt comparator and the cold dual-mode
   fixture gate before attempting any 1% receipt run.

### Sealed pool plan, materializers, and comparison authority

The authority-selection seam is now narrowed once more before physical use.
`USPoolRuntimePlan` is the single immutable pool capability derived from
`USSpecAuthority`. Its own recursive seal covers every domain authority,
compiled node, seed map, and typed execution/checkpoint wrapper, so a frozen
dataclass replacement cannot forge a plan after compiler validation. The plan
contains 10 physical operations, four logical stages, three durable
checkpoints, 30 required normative artifacts, 16 sealed receipt rules, seven
declared sources, 72 seed sites owned by 57 typed owners, and 38 compiled
producer nodes.

Source materializers now accept only the plan's `SourceAuthority`, not the
broader US capability. That authority contains the compiler-issued declared
source registry, authored source and vintage contracts, and resolved vintage
records. The reconstructed ACS manifest is exactly equal to the constants
oracle without invoking its packaged JSON loader. Dynamic stacked-checkpoint
identity construction similarly accepts only the sealed pool plan and combines
its one dedicated static projection with the actual input pins, stack receipt,
sampling request, and clone-attachment request. It updates the nested late
resource binding as well as the top-level request and preserves the legacy
ASCII canonical hash codec.

The plan-derived comparator is fail closed:

- every normative artifact is required and compared as raw bytes;
- receipt differences are accepted only when exactly one compiler-emitted
  rule names the concrete field;
- `populace` to `microcosm` is the only release-prefix normalization;
- generation transitions are sealed as constants to bundle, absent to
  resolved, identity generation 0 to 1, and a typed generation-one
  `run_provenance_identity` whose execution-ABI binding is revalidated;
- the full receipt surface is digested even though declared operational fields
  are excluded from equality; and
- checkpoint receipt sidecars are split only by a trusted raw-sidecar
  constructor. Their outer binding stays exact and only the nested operational
  receipt is excluded.

An adversarial review found that optional checkpoint sidecars could otherwise
be absent in both modes and pass. The generic stage contract now authors
`operational_receipts_sidecar` as `forbidden`, `required`, or
`not_applicable`, with durable/policy coherence enforced by schema, identity
contracts, compiler IR, the sealed runtime plan, and the comparator. The
current compiled order declares assembled forbidden, transferred required,
simulated required, and terminal not applicable. Resume remains deliberately
independent of these operational receipts, preserving generation-0 behavior;
certification nevertheless fails if required evidence is missing or forbidden
evidence appears. This metadata is single-authored through the inert
generation-0 execution-stage registry and the bundle generator, not manually
duplicated in YAML or inferred from stage names.

This reviewed change moves the committed checkpoint identities as follows:

- US `spec_sha256`:
  `a57b484c8993ec81e1c2c0edb9ef29dbae33c17051bdec58ed710774d73906b2`
  -> `f8508f6d00de1ccd79f951d7aeaa6d0f13b46db2e9f7b4ab7155cb010bea9f18`;
- compiler-IR ABI:
  `10e17024929eb7304e00dd655c29e300d275d524c0ebf72b4453361523f0f56e`
  -> `8067bd5ee91af2e5e3b096d41e61fd61c9a712a8832fc1fac4f3b3df9ca3a265`;
- execution ABI:
  `694bb34e1d41a23143b9c02194810161b54e2228df8810f0cdeafdfd81fff80`
  -> `f3be6359c4497d794b0101198da8a71ae2a1a0b7f182d877332d0ec653a5231a`;
- runtime authority:
  `853ce81c5ddb44273c6c8e19094d0eb9d7b06c2199c7fd0f6e506d47e578005e`
  -> `b35ed037c1ae8a0cb0708285c50345b1d7ccd9bd4b86dcedb9f5ff3c26d5f9b1`;
- narrow US capability seal:
  `76af83aee649a45ee7b50f21a11986a2bc0a1c904098cd14718e1a50028ce5b2`;
  and
- pool-plan seal:
  `aa444c078da472523963459cbba3fb34e46aaddc137973b8614112e79a7ba079`.

Verification at this checkpoint:

- generated US bundle `--check`: pass at the new spec SHA;
- coverage generation and `--check`: 41,890/41,890 configuration fields and
  40/40 inventory items, with the pointer inventory re-pinned to
  `ffeeb52f5cdff5e4d516f270a4e45011b2cca255cab4c17c59d50a673a88e522`;
- focused comparator, identity-contract, runtime-authority, source,
  checkpoint, and sealed-plan gates: pass;
- focused plan/source/checkpoint seam: 47 tests pass; and
- Ruff and diff whitespace checks on the touched authority layer: pass.

No sample build was run. Production artifact collection and physical stage
threading are still pending; this subsection does not claim the dual-mode
fixture gate or that physical stochastic calls have all moved behind brokers.

### Next after the sealed-plan checkpoint

1. Retain only `USPoolRuntimePlan` beside runtime provenance in bundle mode and
   thread its identity, rung, release, checkpoint, publication, source,
   imputation, take-up, remaining-stage, battery, and seed authorities through
   the physical tool.
2. Collect the actual plan-derived normative vector and raw checkpoint
   sidecars through trusted locator/selector implementations.
3. Run the cold constants-versus-bundle fixture comparison and require an
   empty difference vector before considering any sample certification.

### Plan-derived artifact collector

The execution-ABI vector now has a generic fail-closed materialization
boundary. `ArtifactLocatorRegistry` binds only opaque compiler locator refs to
construction-time paths or compiler JSON values; it rejects duplicate or
extra bindings, paths outside explicit roots, symlinks, special files, and
incomplete inventories. `collect_artifact_surfaces` validates both execution
ABI seals, applies all seven declared selectors, preserves raw structural
receipts, enforces checkpoint sidecar presence policy, and cross-authenticates
each checkpoint payload, manifest, and raw sidecar before comparison. The live
30-artifact fixture produces equal selected bytes from copied constants and
bundle trees while retaining their raw receipt differences. Ten focused tests,
Ruff, formatting, and whitespace checks pass.

Two production integration gaps remain explicit rather than inferred from
names. A deep transferred/simulated resume does not instantiate the 22 bank
stores, while the current IR carries opaque bank locators but no sealed
relative-path recipes. The collector therefore requires exact caller bindings
and refuses to decode producer ids or import constants. The execution code ABI
also currently seals selector names and locator grammar, not a detailed
selector framing contract; that contract must be compiler-emitted before the
collector can claim a cryptographic selector implementation binding. No sample
build was run in this step.

### R3 provenance/cache correction and sealed selector semantics

The first driver seam was rejected before commit because it would have let a
bundle run reuse generation-zero checkpoints and would have described a
constants-dispatched physical build as though the plan had driven it. The
corrected boundary now issues exactly one typed `run_provenance_identity` for
each run. Its four-field receipt view is `config_authority`,
`spec_binding_status`, `identity_generation`, and
`run_provenance_identity`; the broader runtime capability remains hidden.
Generation and the provenance binding select distinct configured checkpoint
and bank namespaces, while provenance remains absent from node-reuse keys.

Checkpoint H5 payloads deliberately retain the generation-zero-compatible
identity bytes required by D4. Bundle checkpoint JSON manifests, instead,
carry and validate the exact generation-one run configuration, and the
compiler now emits the four corresponding structural comparison rules for
every durable checkpoint manifest. Normal production routing therefore cannot
promote or resume a generation-zero checkpoint into bundle mode, while the
normative payload can still be raw-byte equal. The plan-authored sidecar policy
is also enforced on both write and load: assembled forbids an operational
sidecar; transferred and simulated require a nonempty, valid sidecar.

Bundle source preflight authenticates each of the six compiler-declared source
grants through the file broker and records the sealed broker receipt as an
operational publication-manifest surface. The entire broker receipt is
excluded by one compiler rule; it enters neither `spec_sha256` nor a reuse key.
This is not yet a claim that later source consumers are brokered: after
preflight, the current loaders reopen the paths through their legacy APIs.
That remaining ambient-read seam must be removed before the physical bundle
flip is complete.

The selector gap recorded in the preceding section is closed. A
single-authored selector contract now fixes directory framing, file semantics,
logical H5 table/index/column/scalar framing, receipt normalization, and the
H5 header. Both H5 selectors include the exact `_time_period` value and
`_populace_staging_metadata` object; only `publication_run_id` is removed as
operational. The contract digest and locator grammar are sealed in the code
ABI, and both the collector and comparison kernel refuse a different digest.
Tests prove that period and normative metadata changes alter both logical H5
surfaces, while a publication-run nonce does not.

This checkpoint still does **not** claim that the bundle drives physical
execution. Assembly, gap-fill, primary-QRF, late-producer, take-up, remaining,
and terminal calls must consume the compiler materializers rather than their
constants defaults before the dual-mode fixture gate is meaningful. No sample
build was run.

### Adversarial review correction: production-shaped receipts and D3

The first reduced artifact fixture was not sufficient evidence for the D4
gate. A production-shaped review found that the then-sealed rules normalized
only the `populace`/`microcosm` prefix of a release id, leaving its independently
generated timestamp and nonce to differ; nested publication UUIDs, physical
container hashes, checkpoint paths and write timings were also uncovered. The
same review found that a rule covering the whole
`run_provenance_identity` could hide a behavior-relevant run-request mismatch,
and that the selector implementation files were absent from the compiler code
inventory. No dual-mode or certification PASS is claimed from the earlier
reduced fixture.

The correction makes the provenance comparison leaf-specific. Constants and
bundle runs now construct the same behavior request, code-pin digest, artifact
protocol inventory, pipeline, and compatibility authority versions. Only the
source-grammar/spec-binding triad, identity generation, generation-one runtime
and execution-ABI seals, and authority-mode receipt may differ. The comparison
kernel binds both complete typed provenance identities rather than trusting a
caller-authored exception for an entire object. Release-id normalization is
being bound to an explicit parser: it validates each authority's brand,
retains the semantic country/year/pipeline/rung/seed/realized-count middle,
and removes only the terminal UTC timestamp and eight-hex nonce.

Commit `2c6f84d3` adds the independent Logbook part of this correction. New
attempt rows may carry the full closed D3 identity and include it in the row
digest. Historical absent-key rows preserve their exact JSON and digest. The
repo-owned nullable JSONB migration and live exporter distinguish historical
SQL `NULL` from a present identity and validate the full object. The pool tool
caller is wired in the following driver commit; resolution failures that have
not issued a valid identity remain explicitly absent rather than receiving a
fabricated provenance object.

The review also recorded two still-open closure requirements. Logical-H5
selectors must receive a compiler-sealed required entity/column/weight
inventory, and target-bank directory selectors must receive exact member
descriptors; observationally comparing the files that happen to exist in both
modes is not enough. Actual bundle source consumers must parse broker-issued
leases rather than reopen paths after preflight. Both are treated as blockers
to the physical flip, not deferred evidence. No sample build was run.

## SPLIT-OUT COORDINATION — F1 deliverable 7 (closure/segments/dashboard)

This section is append-only coordination from the deliverable-7 split-out.
Its commits use the `F1-d7:` prefix. The split-out owns only
`tools/emit_lineage_dashboard.py` and the directly affected lineage, typed
closure, take-up semantics, and US bundle tests; it does not stage or alter the
main lane's deliverables 5/6/8 work.

Final handoff (`ef036572`, `F1-d7: retarget lineage surfaces to compiler IR`):

- Consumer audit is complete. The active dashboard read 92 authored producer
  outputs and no compiled segments. Production-shaped test fixtures also read
  the constants generator or pre-IR `ResolvedSpec` surfaces.
- The held #697 392-column f025 artifact inventory is not being resurrected as
  current closure: the approved RFC review records that it predates predictor
  work and misses 56 later columns. Deliverable 7 instead reports the exact
  compiler-declared closure while artifact-presence closure remains owned by
  the plan-derived selector/certification surface.
- The implementation now compiles the packaged bundle and projects 173 typed
  column contracts, 227 compiler-expanded producer-output occurrences, 241
  exact producer cell segments, and two mixed take-up semantic segments. The
  held authored lineage classes are not imported or reproduced.
- The dashboard regression is green: seven tests in 479.80 seconds. A broader
  focused run passed every D7-targeted module; its only failures were the
  existing BE/UK exact `spec_sha256` pins after the main lane's concurrent
  uncommitted compiler/code-inventory changes. D7 did not re-pin those
  main-lane identities and removed its optional edit to that module.
- No pool build or sample rung has run, and nothing has been pushed.
- Main-lane verification blocker: the broader
  `test_spec_engine_*.py` gate otherwise passed, but
  `test_spec_engine_seeds.py::test_production_callsites_are_independent_exact_classified_and_attested`
  still pins 208 production modules while the current main-lane HEAD exposes
  212. The failure reproduced alone (63.90 seconds). Deliverable 7 adds no
  production modules and leaves this main-lane-owned attestation count
  untouched.

The parallel lane is now complete. Its committed files were:

- `tools/emit_lineage_dashboard.py`
- `packages/microcosm-build/tests/test_imputation_lineage_spec.py`
- `packages/microcosm-build/tests/test_spec_engine_typed_closure.py`
- `packages/microcosm-build/tests/test_spec_engine_take_up_semantics.py`
- `packages/microcosm-build/tests/test_us_spec_bundle.py`

### D7 final verification addendum (2026-08-19)

This append-only addendum supersedes two stale count phrases in the handoff
above; it does not alter the main lane's deliverables 5/6/8 state.

- The exact compiler-derived take-up authority surface is **14 ownership
  leaves across 13 typed columns**, not "two mixed take-up semantic segments."
  Housing has two explicit leaves and the other 12 programs each totalize to
  one whole-universe leaf.
- The complete exclusive typed authority projection contains 192 variants:
  150 graph-final segments across 132 columns, 28 family-only segments, and 14
  take-up leaves. The graph final-owner proof independently covers 763 exact
  cells with 170 segments before Medicare and Housing defer to take-up.
- After the cached compiler-schema merge, all four D7 modules passed again on
  merged HEAD `da45dfcd` in 426.33 seconds. Ruff check, Ruff format check, JSON
  round-trip coverage, and diff whitespace checks are clean for the D7 files.
- The broader pre-merge `test_spec_engine_*.py` run passed every other case but
  exposed the main-lane seed-attestation pin (208 expected production modules,
  212 observed). After the merge, the isolated owner test still fails at 208
  expected versus 213 observed (56.46 seconds). D7 adds no production modules
  and does not change that owner-maintained pin.
- The scoped final output is `FINAL_REPORT_F1_D7.md`. No build or sample rung
  was run, nothing was pushed, and the split-out stayed under its 15 GiB RSS
  ceiling.

### SPLIT-OUT COORDINATION — F1 D7 exact-cell correction (2026-08-20)

This append-only section supersedes the prior D7 counts and the statement that
Medicare/Housing "defer" to take-up. It does not alter the main lane's
deliverables 5/6/8 state.

- A continuation audit committed at `823a0100` found that the first retarget's
  whole-column subtraction dropped 20 early-family segments whenever their
  typed column also appeared in the producer graph. The compiler cells do not
  overlap: graph and family use disjoint exact atoms on every shared column.
- The correction retains authority per exact compiler predicate cell and does
  not invent supersession between predicate spaces. The current closure is
  214 segments over all 173 typed contracts: 152 graph segments/134 columns,
  48 early-family segments/columns, and 14 take-up leaves/13 columns. It has
  809 unique `(predicate_space, column_key, atom)` cells.
- Before typed filtering, graph final ownership remains 170 segments/763
  exact cells and exactly equals the 241-segment raw compiler write-cell
  union. The graph/family shared set has 20 columns and zero exact-cell
  collisions.
- Column surface membership is 113 graph-only, 28 family-only, 11 take-up-only,
  19 graph+family, one graph+take-up (Medicare), and one graph+family+take-up
  (Housing). The value summary now accounts for every variable: 45 flags, 132
  amounts, five categories, and one count, totaling 183.
- The source/test correction is present in concurrent commit `5875be22`
  (`Correct F0 kernel and seed contracts`). A main-lane process committed the
  shared index while the D7 patch was verified, sweeping the two D7 files into
  its unrelated F0 commit. The split-out did not rewrite shared history; this
  prefixed handoff documents exact file ownership and provenance.
- Post-correction verification is green: all 70 tests in the four D7 modules
  passed in 407.29 seconds; the exact projection test passed alone in 85.94
  seconds; direct emission completed in 43.10 seconds; and Ruff, formatting,
  bytecode, and whitespace checks passed.
- Two read-only independent audits approve the result. One reconstructed the
  809-cell closure directly from `CompiledSpecIR` without dashboard imports;
  the other confirmed byte-identical output under two distinct hash seeds.
- The seed-attestation blocker recorded above was resolved by the concurrent
  owner commit, which pins 213 production modules and 285 classified calls.
  No build or sample rung ran, nothing was pushed, and D7 makes no broader
  certification claim. The scoped output is `FINAL_REPORT_F1_D7.md`.

### SPLIT-OUT COORDINATION — F1 D7 current-head closure audit (2026-08-20)

This append-only section is the final D7 coordination receipt. It supersedes
only prior D7 verification wording and does not alter main-lane D4--D6/D8
ownership.

- A tracked-tree consumer sweep and two independent read-only audits found no
  missed production-shaped consumer and no held authored-class dependency.
  The exact compiler projection remains 173 contracts, 214 segments, and 809
  unique cells, including all 20 disjoint graph/family shared columns.
- All four D7 modules passed together: 70 passed in 420.42 seconds. Two hash-
  seeded processes emitted byte-identical dashboard JSON with SHA-256
  `03c08eb0e59b4ce2fa0d2ffe2bcf62e503005569b1171520a46068dec99ef7df`.
  Scoped Ruff check/format and whitespace checks pass.
- Serial all-shard testing passed calibrate, data, fit, and frame (865 passed,
  37 skipped). The build run reached 100% with 6,322 passed, 37 skipped, and
  three spine-blindness failures collected before concurrent main-lane commit
  `12df8c45` classified six new runtime-authority modules and changed the
  import-graph pin from 65 to 70. The exact three tests pass at current HEAD
  (3 passed in 4.42 seconds), and the main lane records the full 495-test
  module passing. This is decomposed green evidence, not a clean one-shot
  current-HEAD rerun.
- Repository-wide Ruff has one main-lane `I001` at
  `tools/build_us_multispine_pool.py:1157`. The D7 files are clean; the
  split-out did not edit that D5/D6/D8 driver.
- The scoped final output is `FINAL_REPORT_F1_D7.md`. No push, build, sample
  rung, restricted-data access, or publication occurred. Validation was
  serialized; sandbox restrictions prevented an exact RSS observation.

## F1 continuation r4 final main-lane receipt — honest stop before D5 (2026-08-20)

This section is the final main-lane receipt for continuation r4. It supersedes
the D7 addendum's decomposed-suite and Ruff wording, but it does not alter any
D7 implementation or exact-cell result above.

### Sync, merge, and concurrent-lane custody

- The required first command was attempted exactly as
  `git fetch origin && git merge origin/main --no-edit`. Fetch failed because
  the sandbox could not resolve `github.com`, so the chained merge did not
  execute. No network fallback was available.
- Cached `origin/main` at `164027e2` was then merged in `35fb3ed0`. Conflict
  resolution retained both main's integer-typed schemas, narrowed
  `exc_type` guards, scoped archive/resource work, and this lane's F1
  provenance/checkpoint seams.
- Cached final #698 at `c4e1eb7f` was merged in `da45dfcd`. Conflict
  resolution explicitly retained the 72-site ledger over #698's interim
  53-site ledger, then regenerated rather than hand-editing derived pins.
- `UV_CACHE_DIR=/private/tmp/microcosm-f1-uv-cache-overlay uv sync
  --all-packages --extra us` completed successfully after the merges
  (`Checked 100 packages`).
- Deliverable 7 remained in its parallel lane. Its final documentation commits
  are `5228cf5c` and `c0a75253`; the exact-cell source/test correction was swept
  into shared-index commit `5875be22` while both lanes were active. History was
  not rewritten, and the D7 sections above document that custody boundary.

### Final F0 correction and identity re-pins

Commit `5875be22` corrected confirmed F0 contract defects before any
certification attempt:

- `source_finalizer` now declares `structural_delta: join` for its three
  appended deferred columns;
- all 19 late-transfer nodes declare source-read and sink-write effects;
- the SIPP financial-asset and archived-rent cap sites carry their exact seed
  materials and fill boundaries;
- the vehicle, financial-asset, and rent target-balanced cap sites carry their
  exact conditional draw/fill predicates; and
- five deterministic SHA-256 callsites are classified in the exhaustive
  callsite inventory rather than being mistaken for stochastic draws.

The final field-usage projection is 41,911/41,911 configuration fields:
32,260 authored plus 9,651 resolved, including 19 effect leaves and two fill
leaves. The pointer-inventory digest is
`de879c7bca5f2fa914b43de5130e6a138dd14d0a7059d02952f747c6e578406c`.
The physical census covers 213 production modules and 285 classified
callsites: 119 ledger bindings and 166 exemptions, of which 165 are
deterministic-hash classifications. The compiled protocol contains 72 logical
sites, 57 owners, and 131 owner/site bindings.

The pre-merge 72-site pins moved as follows:

| Surface | Before r4 merge | Final current pin |
|---|---|---|
| compiler ABI v5 | `72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a` | unchanged |
| seed protocol | `f7d51987846236198d9e07dd562b8851f62c48f64e71ebcb0513c122c07c1d7b` | `fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e` |
| compiled seed map | `746a8abde5b6ead279680a76fe15c4522ec52382301d73cd00f39bf7cf079044` | `f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30` |
| normalized graph nodes | `754cccfa2e9fdcb0137669e76f0f7b839a263ff047550e5e560231ae80d7f95a` | `f0d1341b2077da85698ce9993497a366bcbb3fdda0b3721509fafcce7f82b64a` |
| BE `spec_sha256` | `775902c9eedeeb376bdfa3ed5bfda983b6b5a0517b520fa5789867c4e7812172` | `1c5a9449438e1bd30e17105d954f3c5375d20564456be1d005dcab6a798d4aaf` |
| UK `spec_sha256` | `cff22ddb86ddbf814479a0a996f6474182a2b36e2474ad519c3881859a0abec7` | `1de6c5f462370cafd5acf0a861dc29603b0e2cb18d6a5c54318114fa08df5ad5` |
| US `spec_sha256` | `1a25c5c55609a7a565da7a1bf7f80a5c71fabdd92d20231af9c6804a3efd0c68` | `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa` |
| minimal-loader golden | `227ed01608e9a1103651d35711ac0bf39e17da681c75170635068d3514f81201` | `aee96b0afefbbc4776f5dcb0c3e268e2be4351e535314fa6d49025c416e1ab16` |

The cached #698 53-site interim pins were deliberately not adopted. For
auditability, that superseded set was seed protocol
`6dade07562ec29c56d96ab8e299a4416c679f1c44b18b228e0ef10f21bd6f6ec`,
seed map `96140220b6b248c1b3a3567dc0c97df6c08176e6745d8dd55786053f26c43a32`,
graph nodes `a83363de26cad0144b5a98b36b4bca49542e37a7b9fee3d7e541f692deeff864`,
BE `262091db8c7b01b2a3b596aa2468d95855a63703ba9f8ebba2940cf5834c2c83`,
UK `beed723a64944a2dc45c85a6853a51666885aba0aeca0e1ada6680a6352fb44c`,
US `6e9dce8f0fd3e3f0101103a14d6a08ac8527b90b82d48fa8bad2c4cc70dbdfde`,
and loader `9afebedeebc4194243e8307703ebc0feac975a69a186d28e266f1c4ef356bbe3`.
They are recorded only to prevent an owner re-pin from accidentally restoring
the smaller ledger.

### Validation receipts

- Generated US bundle `--check`: pass at US spec
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`.
- Inventory coverage generation and `--check`: 41,911/41,911 fields and 40/40
  inventory checks; evidence refreshed at
  `docs/evidence/spec-engine/us-f0-coverage.json`.
- F0 correction-focused suite: 101 passed.
- Merge/D7 integration suite: 229 passed, zero failures/skips.
- Final D7 exact-cell suite: 70 passed; the D7 receipt above carries its
  independent hash-seed and projection evidence.
- The first full-suite run reached 100% and exposed only three fail-closed
  test-audit pins: six reviewed runtime-authority modules were unclassified and
  the reachable runtime graph was still pinned at 65 rather than 70. Commit
  `12df8c45` corrected that reviewed inventory; all 495 tests in the complete
  spine-blindness module then passed.
- The clean committed-tree full-suite rerun collected 7,262 tests, reached
  100%, and exited 0 with expected skips only.
- Repository-wide `ruff check .` passes. Commit `030c0613` contains the sole
  mechanical import-order correction; the complete pool-tool test module also
  exits 0 after it. `git diff --check` is clean.

The test suite is green. This does **not** convert the missing production
dual-mode build into a D4/D5/D6 certification claim.

### Deliverable 4 closure audit — incomplete

Deliverable 4 remains incomplete, so this lane stops before deliverables 5 and
6. The current pool driver truthfully states that "Physical stage execution
remains on the constants implementation in this seam." Concrete blockers are:

1. There is no production call to `spec_engine.executor.execute_node`; only
   executor tests invoke it. `USPoolKernelAuthorities` is materialized but is
   not carried into the physical build configuration.
2. The honest physical integration seam is per compiled node inside
   `stacked_spine.run_stacked_late_producer_dag`: 38 nodes resolve to 20
   kernel references with mixed `none`, `join`, and `expand` deltas. A single
   stage-level compatibility kernel would bypass the approved executor
   contract.
3. Production stochastic code still constructs private RNGs from integer
   seeds. The 119 classified physical bindings are not yet routed through
   broker stream tokens, even though the 72-site protocol and broker tests are
   complete.
4. Bundle source access is authenticated at preflight, but legacy physical
   consumers are not all operating on broker-issued snapshots/leases. There is
   also no complete declared sink/process/target-bank broker for the physical
   QRF and transfer paths.
5. No closed callable registry, narrow immutable Frame projection, validated
   patch-to-full-Frame merger, row-classifier integration, virtual-receipt
   codec, per-node checkpoint journal, or production node-reuse map is wired.
6. The artifact collector/comparator is not invoked by the production pool
   tool. Its H5 and directory selectors still lack compiler-sealed exact
   required table/column/weight/member inventories, so equal omissions could
   pass an observational comparison.
7. Calibration is absent from the pool execution graph and its normative
   weight/bank ownership is unresolved. The existing reduced fixture is not a
   production-shaped proof of the owner ruling's full artifact vector.

Accordingly, the cold constants-versus-bundle D4 fixture gate has not passed.
Bundle mode constructs typed plan/provenance/checkpoint surfaces, but it does
not yet drive every physical stage or collect the sealed two-tier result.

### Deliverables 5 and 6 — not run, not certified

No stage flip receipt row was emitted at fixture or 1% scale. Therefore there
are no honest constants/bundle stage digests, equality verdicts, wall-clock
rows, or peak-RSS rows to report for deliverable 5.

The four cold 1% builds and kill/resume leg required by deliverable 6 were not
started. Within-mode determinism, cross-mode normative raw-byte equality, the
sealed receipt comparison, and the concrete resume predicate are all
**NOT RUN**, not PASS. The files
`docs/evidence/spec-engine/us-f1-certification.json` and
`docs/evidence/spec-engine/us-f1-certification.md` were intentionally not
created because no certification occurred.

The existing implementation is independently ineligible for the requested
rung under the unchanged lane rule: the four recorded cold f001 primary-QRF
peaks are 78.91, 84.15, 96.28, and 96.95 GiB RSS, versus the per-process
20 GiB ceiling. Additional host RAM alone does not authorize those processes.
No sample build ran in this continuation, no build above 1% ran, and no
threshold, band, gate, seed, or comparator was tuned.

### Owner handoff

There is no valid 25% owner-run command to issue at this state. Running the
current `--config-authority bundle` path would still dispatch physical stages
through constants and would not invoke the production artifact comparator; a
command would therefore misstate what is being certified. The same objection
applies at 1%.

Before an exact command can be published, a future authorized continuation
must:

1. route every inventoried physical stage, including all 38 producer nodes,
   through the compiled executor/broker authorities while preserving constants
   mode as the oracle;
2. seal exact H5, checkpoint, bank, target, publication, and calibration member
   inventories and invoke the two-tier comparator in production;
3. pass the cold dual-mode D4 fixture gate and freeze the resulting complete
   expected-identity vector;
4. resolve the six source path/SHA pairs on the owner host and allocate
   separate fresh output/checkpoint namespaces for constants, bundle, and the
   kill/resume leg;
5. make the 1% path keep every process below 20 GiB RSS before D5/D6, or obtain
   explicit owner authority to change that constraint; only then consider
   authorizing f025; and
6. for off-chain work, omit `--logbook-prev-row-digest` **and** unset
   `POPULACE_LOGBOOK_PREV_ROW_DIGEST`, because the environment is a fallback.

The current core pins to preserve while closing those seams are the final
identity values in the table above. No owner certification comparison target
exists beyond them until D4 produces a complete vector.

Nothing was pushed. `logbook-pending-chain.txt` was never touched. The three
untracked `_F1-CHARTER-R*.md` files were left untouched and uncommitted.

## F1 continuation r4 resumed audit — D5 remains blocked (2026-08-20)

This append-only receipt records the owner-requested attempt to resume
deliverables 5 and 6 after #698 merged. It supersedes no implementation result
above and does not alter the parallel deliverable-7 surfaces.

### Main reconciliation and identity recomputation

- The required first command was attempted exactly as `git fetch origin && git
  merge origin/main --no-edit`. Sandbox DNS again prevented the fetch, so the
  chained merge did not execute. Cached `origin/main` remains stale at
  `164027e2`; the GitHub merge commit itself is not present in any local ref,
  reflog, or object scan and therefore cannot be named honestly here.
- Final #698 head `c4e1eb7f` is already an ancestor through this lane's merge
  `da45dfcd`. Its scoped-archive, integer-typed schema, and narrowed-guard
  content is present. Current `seeds.py` remains the 72-site source of truth;
  its blob `6c8b4ae1` differs from #698's superseded 53-site blob `36e6c27e`.
- `UV_CACHE_DIR=/private/tmp/microcosm-f1-uv-cache-overlay uv sync
  --all-packages --extra us` completed successfully (`Checked 100 packages`).
- The identities were recomputed from the resolved source rather than copied
  from pins. No re-pin edit was necessary: generated US bundle `--check`
  passes at spec
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`;
  coverage `--check` reports 41,911/41,911 fields and 40/40 inventory checks;
  the protocol has 72 sites and digest
  `fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e`;
  the compiled map has 57 owners, 131 bindings, and digest
  `f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30`;
  and compiler ABI v5 remains
  `72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a`.
  BE and UK recompute unchanged at `1c5a9449...` and `1de6c5f4...`.

### Renewed deliverable-5/6 readiness finding

Two independent read-only audits and direct source tracing reproduce the prior
D4 stop. `tools/build_us_multispine_pool.py` explicitly states that physical
stage execution remains on constants. Bundle resolution retains a typed runtime
plan, but the production path calls none of `execute_node`,
`USPoolKernelAuthorities`, `collect_artifact_surfaces`, or
`compare_artifact_sets`. It continues to select the private model seeds, QRF
parameters, gap-fill plan, late-transfer groups, and derive/seed/simulate
functions in the constants path.

The reusable compiler pieces do not close that gap. The current plan has 30
normative rows across configured, assembled, transferred, simulated, and
terminal stages, but its selectors do not seal exact required H5
entity/column/weight inventories or exact bank member descriptors; calibration
weights/banks are absent; no production node-reuse map is persisted; and the
collector/comparator are exercised only by synthetic tests. The plan's coarse
stage rows also omit the inventory's intermediate source, gap-fill, primary
QRF, late-DAG, derive, seed, and simulation digests required by deliverable 5.

There is no production dual-mode fixture command, per-stage wall/RSS receipt,
four-build certification runner, `--resume-policy=forbid`, typed zero-count
resume audit, subprocess kill/resume gate, or F1 certification evidence model.
The existing constants-versus-constants-adapter fixture does not run bundle
physical authorities and compares only three checkpoint payloads.

Independently, a real 1% run remains prohibited: four authenticated cold f001
primary-QRF profiles peak at 78.91--96.95 GiB RSS, versus the unchanged 20 GiB
per-process ceiling. The production driver itself documents a 64-GiB 1% host
envelope. No process in this resumed audit ran a pool build.

Therefore deliverable 5 cannot emit an honest fixture or 1% receipt row and
deliverable 6 cannot start. Their verdict remains **NOT RUN**, not PASS. No
`us-f1-certification.json` or `.md` file is created. Per the charter's explicit
honest-stop rule, this continuation stops at deliverable 5 after the committed-
tree validation gates, with no threshold, band, seed, artifact selector, or
memory limit changed.

### SPLIT-OUT COORDINATION — F1 D7 green-suite/resource addendum (2026-08-20)

This append-only D7 receipt supersedes the earlier decomposed-suite and
unobservable-RSS wording. It does not change main-lane D4--D6/D8 ownership.

- Post-report D7 verification at `c0a75253` passed all four scoped modules: 70
  passed in 350.29 seconds. Darwin `ru_maxrss` was 4,231,233,536 bytes (3.94
  GiB), below the split-out's 15 GiB ceiling.
- A fresh build-shard run completed with 6,325 passed and 37 expected skips in
  6,820.57 seconds. Combined with the other four serial shard receipts, the
  observed total is 7,190 passed and 74 expected skips. The main lane also
  records its independent clean committed-tree full-suite result.
- The shared branch advanced during that long run through `030c0613`,
  `e14355c6`, `2d919ce9`, and `4deb8fb8`. Those commits changed one mechanical
  import ordering plus journals/docs, and no D7 implementation/test file.
- The build-shard process peaked at 30,950,326,272 bytes (28.82 GiB) RSS. That
  exceeded the user's 15 GiB ceiling despite being test-only. Heavy testing
  stopped as soon as the terminal resource receipt exposed the overrun. This
  is an explicit operating-order violation, not a PASS or a pool-build claim.
- Current-HEAD repository-wide Ruff, scoped D7 formatting, held-fixture
  searches, and whitespace checks pass. Nothing was pushed; no pool build,
  sample rung, restricted-data access, or publication occurred.

### Resumed-main validation and final disposition

The renewed committed-tree validation was deliberately kept separate from a
pool build. A serial `uv run pytest` invocation completed its collection and
test execution in 8,044.09 seconds with 7,189 passed, 74 expected skips, and
one failure. The failure was a `subprocess.TimeoutExpired` in
`test_exchange_interrupted_before_cleanup_reclaims_displaced_set`: its child
process exceeded the test's fixed 60-second limit while importing the bulk
trade tool and exercising the durable directory-exchange crash path.

Direct source tracing found no blocking protocol in that child: its publisher
lock is nonblocking, paths are unique under `tmp_path`, there are no wait loops
or grandchildren, and the patched cleanup exits immediately after one atomic
exchange. Heavy NumPy/pandas import and filesystem `fsync` latency are the only
unbounded phases. The host was materially contended during the run, and the
same test has passed in later recorded full-suite receipts without a relevant
production or timeout change. No timeout, test, or publication implementation
was changed.

The exact failed node then passed unchanged. The complete 78-test
`test_us_trade_imdb_bulk.py` module also passed unchanged (exit 0; 331.51
seconds), including the crash-recovery node. Repository-wide `ruff check .`
and `git diff --check` pass. These receipts establish decomposed current-head
green coverage for the transient failure, but they do not rewrite the one-shot
full-suite exit as zero.

This validation does not alter the readiness finding: deliverable 4 still does
not make bundle authorities drive physical execution or invoke the sealed
production comparator, and the known 1% implementation remains above the
20 GiB per-process ceiling. Deliverables 5 and 6 are therefore **NOT RUN**.
No fixture/1% stage receipt, four-build result, resume verdict, certification
JSON, or certification Markdown exists. No pool build or sample rung ran, no
threshold or timeout was tuned, the logbook pending chain remained untouched,
and nothing was pushed. This is the charter-required honest stop at
deliverable 5.

## F1 continuation r4 verification reprise (2026-08-20)

This append-only main-lane receipt rechecks the committed honest stop at
`cddb6f18`. It changes no deliverable-7 closure, segment, dashboard, source, or
test surface.

- The required `git fetch origin && git merge origin/main --no-edit` was again
  the first command attempted. Sandbox DNS prevented the fetch. A separate
  local `git merge origin/main --no-edit` then reported already up to date.
  Cached `origin/main` remains `164027e2`; final #698 `c4e1eb7f` is independently
  an ancestor through `da45dfcd`.
- `uv sync --all-packages --extra us` completed from the task-local cache with
  100 packages checked. The generated US bundle `--check` passes at
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`,
  and coverage `--check` passes at 41,911/41,911 fields and 40/40 checks.
- Independent source/identity inspection confirms 72 sites, 57 owners, 131
  bindings, seed protocol
  `fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e`,
  seed map
  `f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30`,
  and compiler ABI v5
  `72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a`.
  No identity re-pin is required.
- The decisive production seam remains explicit at
  `tools/build_us_multispine_pool.py:5402`: physical stage execution stays on
  constants. Production has no caller of `execute_node`,
  `USPoolKernelAuthorities.from_runtime_plan`, `collect_artifact_surfaces`, or
  `compare_artifact_sets`. Source reads are now snapshot-brokered in bundle
  mode, correcting one older blocker description, but RNG/effect dispatch,
  exact artifact-member and calibration closure, node reuse, and the dual-mode
  comparison harness remain open.
- No dual/four-build certification runner, `--resume-policy=forbid`, typed
  zero-count resume audit, or real subprocess kill/resume gate exists. The
  constants-versus-constants-adapter fixture remains a stubbed three-checkpoint
  comparison rather than a bundle-driven D4 gate.
- Four authenticated cold f001 primary-QRF profiles still peak at
  78.91--96.95 GiB RSS, 3.9--4.8 times the hard 20 GiB/process ceiling. No pool
  build, fixture build, or sample rung was started in this reprise.

Deliverables 5 and 6 therefore remain **NOT RUN**, not PASS. The certification
JSON and Markdown remain absent by design. No threshold, timeout, seed, band,
selector, or memory limit changed; the pending logbook chain was not touched;
nothing was pushed. The charter-required disposition remains an honest stop at
deliverable 5.

## F1 continuation r5 kickoff (2026-08-20)

### State

- The owner accepts the r4 honest stop and authorizes exactly two
  implementation deliverables before a host handoff: bundle-only physical
  executor/broker wiring and a host-invokable four-receipt certification
  runner. The host, not this lane, will execute every sample build.
- `_F1-CHARTER.md` and this journal were reread before code work. Their
  explicit `NEVER touch PROGRESS.md` / `no PROGRESS.md` rule controls the
  contradictory generic standing-order sentence, so this file remains the
  committed state/done/next journal and `PROGRESS.md` remains untouched.
- Current HEAD is `58269355` on `spec-engine-f1`. The only pre-existing
  worktree entries are the four untracked owner charter copies
  `_F1-CHARTER-R2.md` through `_F1-CHARTER-R5.md`; they remain untouched and
  uncommitted. Nothing has been pushed or stashed.
- The r4 production finding is the starting seam: bundle selection compiles a
  `USPoolRuntimePlan` and snapshot-brokers source reads, but
  `build_stacked_pool` still dispatches constants-era physical calls and no
  production caller invokes `execute_node`,
  `USPoolKernelAuthorities.from_runtime_plan`,
  `collect_artifact_surfaces`, or `compare_artifact_sets`.
- The lane will run only focused/unit and fixture-scale tests. It will not run
  a sample build, including 1%, because authenticated primary-QRF profiles
  peak at 78.91--96.95 GiB RSS and exceed the unchanged 20-GiB process
  ceiling.

### Done

- Read the complete r4 journal, approved charter including the D4 two-tier
  ruling, repository instructions, and the r5 continuation order.
- Confirmed the tracked worktree is clean and reviewed the current authority
  selection, source snapshot, physical stage, artifact collector/comparator,
  and kernel-authority entry points before editing production code.
- Began three read-only audits in parallel: physical dispatch seams,
  certification receipt/CLI reuse, and artifact-vector/fixture closure.
  GitNexus graph resources are not exposed in this environment, so the
  required exploration workflow falls back to direct source, tests, and Git
  history.

### Next

1. Wire bundle-only physical execution through compiler-materialized kernel
   authorities, per-node `execute_node` dispatch, source snapshots, and ledger
   RNG sessions while preserving the constants branch as the oracle.
2. Add and pass a cold fixture-scale constants-versus-bundle full-stage graph
   gate using the sealed D4 two-tier comparison.
3. Build and unit-test the one-build receipt runner, four-receipt comparator,
   and document-only kill/resume predicate command.
4. Append the exact ordered host commands and expected RSS envelope, update
   the final output report, and stop without running certification.

## F1 continuation r5 artifact-member coverage step (2026-08-20)

### State

- The sealed artifact vector contains 22 directory-tree selectors whose
  observed members can be checked against exact physical target-bank
  authority, plus two final-pool H5 selectors that currently enumerate only
  observed entity/column/weight members.
- Treating those observational selectors as proof of vector closure would be
  circular. Coverage therefore fails closed until the compiler issues an
  independent exact final-H5 inventory.

### Done

- Added a typed plan/authority-bound artifact coverage contract for all 22
  early, primary-QRF, and late-transfer target banks. It derives 208 exact
  file/directory descriptors with the production filename algorithms and
  validates missing, extra, symlink, special-file, and locator-binding cases.
- Added a typed coverage receipt that can report complete bank-member
  coverage while refusing to promote the overall container vector to PASS
  when final-H5 closure is unsupported.
- Focused coverage verification passes: 6 tests; Ruff format/check passes.
  No build or sample rung ran.

### Next

1. Add a compiler-issued exact final-H5 entity/column/weight inventory, or
   retain the typed unsupported verdict in certification; never infer closure
   from the members returned by the selector itself.
2. Bind this coverage receipt into the production evidence sidecar and the
   four-receipt comparator.

## F1 continuation r5 physical-authority plumbing step (2026-08-20)

### State

- Bundle configuration now compiles one `USPoolKernelAuthorities` value from
  the sealed runtime plan and retains it as a typed, non-receipt capability.
  Constants mode still constructs no compiler authority and continues down
  its explicit no-authority calls.
- This step closes the constants-at-physical-leaves defect, but per-node
  `execute_node`/broker dispatch is a separate next step and is not claimed by
  this commit.

### Done

- Threaded compiler-issued assembly, model, gap-fill, primary-QRF,
  late-producer, remaining-stage, take-up, simulation, terminal, receipt,
  checkpoint, and publication-validation authorities through the bundle-only
  production path. Bundle pre-clone preparation requires its remaining-stage
  and simulation authorities as an all-or-nothing pair.
- Derived early/late target-bank schedules and target surfaces from the typed
  authorities. Checkpoint resume/write and terminal publication validation
  now retain the same authority rather than reopening constants.
- Added the cold-run `--resume-policy forbid` preflight before any publication
  or checkpoint discovery; default `allow` preserves the oracle path.
- Root verification: 10 focused authority/config/publication tests pass after
  correcting the newly fail-closed publication fixture; Ruff and diff checks
  pass. Earlier decomposed checks covered the five constants fixture terminal,
  resume, and checkpoint cases. No build ran.

### Next

1. Construct the bundle-only per-node executor dispatcher from
   `kernel_authorities.physical.late_producers.nodes` and prove every compiled
   node executes exactly once in sealed order.
2. Route stochastic work through node ledger sessions and refuse undeclared
   file/process/sink access; then run the full dual-mode fixture graph.

## F1 continuation r5 executor audit and certification-runner step (2026-08-20)

### State — honest stop on deliverable A

- Bundle selection now compiles one
  `USPoolKernelAuthorities.from_runtime_plan(...)` capability, threads its
  authorities through the bundle-only physical-stage calls, and keeps explicit
  constants branches as the behavior oracle. That is authority plumbing; it
  is not the charter's required per-node execution proof. Production still has
  no caller of `execute_node`, so no full physical stage graph is dispatched as
  sealed `CompiledNode` transactions.
- A legacy callback wrapper cannot honestly close the gap. `execute_node`
  requires a broker-bound
  `(ImmutableFrameProjection, ExecutionContext) -> KernelPatch` transaction.
  Primary QRF and transfer kernels also require checkpoint writes and rereads,
  subprocess execution, and target-bank sinks, but the executor has no sealed
  sink/checkpoint/process broker for those effects. Calling the legacy
  `Frame -> Frame` functions inside a registered kernel would bypass the
  executor's effect and patch boundaries.
- The current compiled execution projection is also insufficient for a
  byte-faithful adapter: the physical family/marital/SPM/tax callbacks need
  source identifiers absent from their node projections; late transfer needs
  dynamic RNG sequencing; and the required clone-attachment and capital-gain
  random-rank draws do not have usable grants on the physical nodes that own
  them. Legacy QRF state embeds the derived integer seed in
  `QRFChainState.model_config.seed` and serialized H5 state, while the ledger
  broker deliberately exposes an opaque handle. Exposing or independently
  recomputing the seed would violate the broker contract and can change bytes.
- A sketched adapter fixture was deliberately removed: with physical work
  stubbed, it could only prove that a vacuous callback loop produced equal
  empty/stub artifacts. It could not prove the real clone/tail/cap/checkpoint/
  target-bank graph or exact broker consumption. No such false gate is
  committed.
- Therefore the required dual-mode full-stage fixture, exact once-per-node
  journal, broker-consumption receipt, node-reuse map, and D4 two-tier
  byte-equality verdict do not exist. Deliverable A is **NOT COMPLETE**, not
  PASS. This is the charter-required honest stop rather than an assertion that
  authority arguments alone constitute executor dispatch.

### Done — deliverable B's fail-closed runner surface

- Added `tools/f1_certification_run.py` with three host-facing subcommands:
  `run` (alias `build`), `compare`, and documentation-only `resume-gate`.
  `run` exclusively claims an absent output root, launches exactly one pool
  child in `constants` or `bundle` mode with `--resume-policy forbid`, removes
  ambient logbook/ledger variables, and writes a typed
  `us-f1-build-receipt.json` only after a zero child exit.
- The pool's opt-in `--f1-evidence-out` seam is allowed only for stacked
  constants/bundle cold builds. It rejects any pre-existing publication,
  checkpoint, or evidence state before load; freezes the compiler plan before
  source reads; binds bundle execution authority to that snapshot; requires a
  tracked-clean, unchanged HEAD; and revalidates the identical plan before
  artifact collection. The receipt parser validates the complete plan-lock
  schema rather than trusting an execution-ABI fragment.
- Normative vector collection now supports streaming SHA-256/size collection:
  ordinary file and directory-member bodies stream in 1-MiB chunks and H5
  logical encodings write into a digest sink instead of retaining the complete
  encoded vector. The receipt still records the plan row plus normative
  selected-byte digest/size; provenance and operational surfaces remain under
  the sealed `plan_lock.execution_abi.receipt_comparison_vector`.
- The production locator registry binds all 29 unique artifact locators (30
  artifact rows) plus six checkpoint manifest/optional-receipt roles. The
  independent target-bank contract covers 22 banks and 208 exact members.
  Selector receipts, their coverage contracts, calibration receipts, and the
  full plan are sealed and validated before comparison.
- `compare` requires four distinct typed receipts in constants-A,
  constants-B, bundle-A, bundle-B roles, one current plan lock, and one request.
  It reports within-mode normative/receipt/node-key determinism, paired
  cross-mode D4 equality, cold zero-resume audits, and vector coverage, then
  writes `us-f1-certification.json` and `us-f1-certification.md`. Exit status is
  0 for PASS, 1 for a well-formed FAIL, and 2 for malformed input/error.
- `resume-gate` compiles the sealed resume predicate into a Markdown host
  procedure and explicitly starts no build, kill, or resume. The four cold
  receipts separately require zero resumed durable/QRF/target-bank counts.
- Coverage is intentionally incomplete rather than fabricated: production has
  no runtime node-reuse inventory, the compiler has no independent exact final
  H5 entity/column/weight inventory, and calibration weights are absent from
  the normative artifact vector. Production receipts seal those three facts as
  false. The current comparator is therefore expected to write a structurally
  valid **FAIL**, even if every observed artifact digest matches; it is not a
  certification-capable PASS path until deliverable A and the two inventory
  defects close.

### Validation and next

- The combined runner/schema, streaming collector, D4 comparator, and exact
  bank-coverage run passes 78/78 tests. A separate pool-tool authority,
  bundle, evidence-freeze, parser, and resume-policy selection passes 19/19
  tests. Ruff format/check and `git diff --check` pass on every touched source
  and test file.
- The real `resume-gate` subcommand compiled the current US plan and wrote its
  documentation-only predicate with exit 0 under `/private/tmp`; it launched
  no pool child. The final review's two sealing findings (post-build plan
  recompilation and unvalidated/self-asserted receipt inputs) were fixed and
  regression-tested before this commit.
- A final independent read-only audit found no commit-blocking correctness
  issue. It identified two low-risk spy-test gaps (direct CLI coverage-contract
  compilation and freeze-before-input-verification call order), but no concrete
  false-PASS path; both remain explicit follow-up hardening rather than claimed
  certification evidence.
- All checks are unit or fixture scale. The repository-wide build shard was
  not rerun because its authenticated prior peak was 28.82 GiB, above this
  lane's 20-GiB process limit; no repository-wide green claim is made. No pool
  build, sample rung, four-build comparison, or kill/resume exercise ran.
- Next is deliverable C only: append the exact sequential host commands and
  observed RSS envelope, update the final output report, and stop. The host
  owns every build and the owner adjudicates any emitted verdict.

## F1 continuation r5 host handoff (2026-08-20)

### State

- Deliverable A remains **NOT COMPLETE**: production does not execute the
  physical graph through `execute_node`, and the required non-vacuous
  constants-versus-bundle full-stage fixture does not exist.
- Deliverable B is mechanically host-invokable and fail-closed. With the
  present node-reuse, final-H5, and calibration coverage gaps, its comparator
  is expected to emit a structurally valid **FAIL**, not certification PASS,
  even if all observed raw-byte artifact digests match.
- No command below was run in this lane. The high-memory host owns the four
  builds and comparator, and the owner adjudicates their evidence.

### Exact host sequence

Run from the repository root on the high-memory host. Replace each placeholder
with an authenticated absolute path or its lowercase 64-hex SHA-256 pin. Every
build root and the resume-gate output must be absent before its command starts.

```zsh
export F1_CERT_ROOT='/absolute/host/path/us-f1-certification'

export F1_ASEC_RAW_STAGE_H5='/absolute/path/asec-raw-stage.h5'
export F1_ASEC_RAW_STAGE_H5_SHA256='<64-lowercase-hex>'
export F1_ACS_HOUSEHOLD_ZIP='/absolute/path/acs-household.zip'
export F1_ACS_HOUSEHOLD_ZIP_SHA256='<64-lowercase-hex>'
export F1_ACS_PERSON_ZIP='/absolute/path/acs-person.zip'
export F1_ACS_PERSON_ZIP_SHA256='<64-lowercase-hex>'
export F1_ACS_RENT_H5='/absolute/path/acs-rent.h5'
export F1_ACS_RENT_H5_SHA256='<64-lowercase-hex>'
export F1_PUF_H5='/absolute/path/puf.h5'
export F1_PUF_H5_SHA256='<64-lowercase-hex>'
export F1_PUF_SOURCE_YEAR_CSV='/absolute/path/puf-source-year.csv'
export F1_PUF_SOURCE_YEAR_CSV_SHA256='<64-lowercase-hex>'

mkdir -p "$F1_CERT_ROOT/logs"
set -o pipefail
```

First write the documentation-only kill/resume predicate. This starts no build
and performs no kill or resume:

```zsh
.venv/bin/python tools/f1_certification_run.py resume-gate \
  --mode bundle \
  --sample-fraction 0.01 \
  --seed 578 \
  --output "$F1_CERT_ROOT/us-f1-resume-gate.md"
```

Then run the four cold builds strictly sequentially. Require status zero from
each timed pipeline before proceeding. The `tee` logs are intentionally
outside the absent run roots.

1. Constants A:

```zsh
env \
  -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  -u POPULACE_LEDGER_URL \
  -u POPULACE_LEDGER_KEY \
  -u POPULACE_LEDGER_API_KEY \
  /usr/bin/time -l \
  .venv/bin/python tools/f1_certification_run.py run \
  --mode constants \
  --sample-fraction 0.01 \
  --seed 578 \
  --output-root "$F1_CERT_ROOT/constants-a" \
  --asec-raw-stage-h5 "$F1_ASEC_RAW_STAGE_H5" \
  --asec-raw-stage-h5-sha256 "$F1_ASEC_RAW_STAGE_H5_SHA256" \
  --acs-household-zip "$F1_ACS_HOUSEHOLD_ZIP" \
  --acs-household-zip-sha256 "$F1_ACS_HOUSEHOLD_ZIP_SHA256" \
  --acs-person-zip "$F1_ACS_PERSON_ZIP" \
  --acs-person-zip-sha256 "$F1_ACS_PERSON_ZIP_SHA256" \
  --acs-rent-h5 "$F1_ACS_RENT_H5" \
  --acs-rent-h5-sha256 "$F1_ACS_RENT_H5_SHA256" \
  --puf-h5 "$F1_PUF_H5" \
  --puf-h5-sha256 "$F1_PUF_H5_SHA256" \
  --puf-source-year-csv "$F1_PUF_SOURCE_YEAR_CSV" \
  --puf-source-year-csv-sha256 "$F1_PUF_SOURCE_YEAR_CSV_SHA256" \
  2>&1 | tee "$F1_CERT_ROOT/logs/constants-a.time.log"
```

2. Constants B:

```zsh
env \
  -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  -u POPULACE_LEDGER_URL \
  -u POPULACE_LEDGER_KEY \
  -u POPULACE_LEDGER_API_KEY \
  /usr/bin/time -l \
  .venv/bin/python tools/f1_certification_run.py run \
  --mode constants \
  --sample-fraction 0.01 \
  --seed 578 \
  --output-root "$F1_CERT_ROOT/constants-b" \
  --asec-raw-stage-h5 "$F1_ASEC_RAW_STAGE_H5" \
  --asec-raw-stage-h5-sha256 "$F1_ASEC_RAW_STAGE_H5_SHA256" \
  --acs-household-zip "$F1_ACS_HOUSEHOLD_ZIP" \
  --acs-household-zip-sha256 "$F1_ACS_HOUSEHOLD_ZIP_SHA256" \
  --acs-person-zip "$F1_ACS_PERSON_ZIP" \
  --acs-person-zip-sha256 "$F1_ACS_PERSON_ZIP_SHA256" \
  --acs-rent-h5 "$F1_ACS_RENT_H5" \
  --acs-rent-h5-sha256 "$F1_ACS_RENT_H5_SHA256" \
  --puf-h5 "$F1_PUF_H5" \
  --puf-h5-sha256 "$F1_PUF_H5_SHA256" \
  --puf-source-year-csv "$F1_PUF_SOURCE_YEAR_CSV" \
  --puf-source-year-csv-sha256 "$F1_PUF_SOURCE_YEAR_CSV_SHA256" \
  2>&1 | tee "$F1_CERT_ROOT/logs/constants-b.time.log"
```

3. Bundle A:

```zsh
env \
  -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  -u POPULACE_LEDGER_URL \
  -u POPULACE_LEDGER_KEY \
  -u POPULACE_LEDGER_API_KEY \
  /usr/bin/time -l \
  .venv/bin/python tools/f1_certification_run.py run \
  --mode bundle \
  --sample-fraction 0.01 \
  --seed 578 \
  --output-root "$F1_CERT_ROOT/bundle-a" \
  --asec-raw-stage-h5 "$F1_ASEC_RAW_STAGE_H5" \
  --asec-raw-stage-h5-sha256 "$F1_ASEC_RAW_STAGE_H5_SHA256" \
  --acs-household-zip "$F1_ACS_HOUSEHOLD_ZIP" \
  --acs-household-zip-sha256 "$F1_ACS_HOUSEHOLD_ZIP_SHA256" \
  --acs-person-zip "$F1_ACS_PERSON_ZIP" \
  --acs-person-zip-sha256 "$F1_ACS_PERSON_ZIP_SHA256" \
  --acs-rent-h5 "$F1_ACS_RENT_H5" \
  --acs-rent-h5-sha256 "$F1_ACS_RENT_H5_SHA256" \
  --puf-h5 "$F1_PUF_H5" \
  --puf-h5-sha256 "$F1_PUF_H5_SHA256" \
  --puf-source-year-csv "$F1_PUF_SOURCE_YEAR_CSV" \
  --puf-source-year-csv-sha256 "$F1_PUF_SOURCE_YEAR_CSV_SHA256" \
  2>&1 | tee "$F1_CERT_ROOT/logs/bundle-a.time.log"
```

4. Bundle B:

```zsh
env \
  -u POPULACE_LOGBOOK_PREV_ROW_DIGEST \
  -u POPULACE_LEDGER_URL \
  -u POPULACE_LEDGER_KEY \
  -u POPULACE_LEDGER_API_KEY \
  /usr/bin/time -l \
  .venv/bin/python tools/f1_certification_run.py run \
  --mode bundle \
  --sample-fraction 0.01 \
  --seed 578 \
  --output-root "$F1_CERT_ROOT/bundle-b" \
  --asec-raw-stage-h5 "$F1_ASEC_RAW_STAGE_H5" \
  --asec-raw-stage-h5-sha256 "$F1_ASEC_RAW_STAGE_H5_SHA256" \
  --acs-household-zip "$F1_ACS_HOUSEHOLD_ZIP" \
  --acs-household-zip-sha256 "$F1_ACS_HOUSEHOLD_ZIP_SHA256" \
  --acs-person-zip "$F1_ACS_PERSON_ZIP" \
  --acs-person-zip-sha256 "$F1_ACS_PERSON_ZIP_SHA256" \
  --acs-rent-h5 "$F1_ACS_RENT_H5" \
  --acs-rent-h5-sha256 "$F1_ACS_RENT_H5_SHA256" \
  --puf-h5 "$F1_PUF_H5" \
  --puf-h5-sha256 "$F1_PUF_H5_SHA256" \
  --puf-source-year-csv "$F1_PUF_SOURCE_YEAR_CSV" \
  --puf-source-year-csv-sha256 "$F1_PUF_SOURCE_YEAR_CSV_SHA256" \
  2>&1 | tee "$F1_CERT_ROOT/logs/bundle-b.time.log"
```

Finally compare the four receipts:

```zsh
.venv/bin/python tools/f1_certification_run.py compare \
  --constants-a "$F1_CERT_ROOT/constants-a/us-f1-build-receipt.json" \
  --constants-b "$F1_CERT_ROOT/constants-b/us-f1-build-receipt.json" \
  --bundle-a "$F1_CERT_ROOT/bundle-a/us-f1-build-receipt.json" \
  --bundle-b "$F1_CERT_ROOT/bundle-b/us-f1-build-receipt.json" \
  --output-root "$F1_CERT_ROOT/verdict"
```

The comparator returns 0 for PASS, 1 for a well-formed D4 FAIL, and 2 for
malformed input or an execution error. Status 1 is expected at the present
head; it still writes `us-f1-certification.json` and
`us-f1-certification.md`, so automation must retain those files.

### Memory, order, and recovery constraints

- The four authenticated cold f001 primary-QRF profiles observed
  84,729,479,168 bytes (78.91 GiB), 90,351,255,552 bytes (84.15 GiB),
  103,374,684,160 bytes (96.28 GiB), and 104,102,936,576 bytes (96.95 GiB)
  peak RSS. They are not mapped to a particular mode/A-B position. Provision
  every build for more than 96.95 GiB plus host margin and never overlap them.
- On Darwin, `/usr/bin/time -l` reports maximum resident set size in bytes in
  the external log. Streaming post-build collection has not been separately
  profiled and must not be assumed to reduce the build peak.
- A failed `run` leaves its exclusively claimed root in place. Diagnose it and
  choose a new fresh root; do not blindly delete or reuse the claimed root.
- Source pins must be exactly 64 lowercase hexadecimal characters.
  `resume-gate` refuses to overwrite its output, and `compare` refuses to
  overwrite either verdict file.
- `set -o pipefail` is required because each timed build is piped through
  `tee`. Do not start the next command unless the current pipeline returns 0.

### Done / next

- Deliverable C is complete as a handoff only. This lane ran none of the host
  commands, performed no push or stash, and did not create or modify
  `PROGRESS.md`.
- The host may run the commands in the exact order above. The owner, not this
  lane, adjudicates the emitted verdict. A current valid FAIL is evidence of
  the deliberately open executor/node-reuse/final-H5/calibration work and is
  not certification.

## F1 continuation r5 completion pass (2026-08-20)

### State

- The prior r5 handoff is retained as an honest historical receipt, but it did
  not satisfy Deliverable A: bundle physical execution still has no production
  `execute_node` caller and no non-vacuous full-stage dual-mode fixture.
- This pass is reopened solely to close that missing executor/broker wiring,
  then revalidate Deliverables B and C. Constants mode remains the byte oracle.
- The four untracked owner charter copies remain untouched. No push, stash,
  sample build, or process above the 20-GiB lane ceiling is authorized.
- The lane-specific `NEVER touch PROGRESS.md` rule controls the contradictory
  generic standing order; this committed journal remains the state/done/next
  record.

### Done

- Reread the complete lane journal, approved charter and D4 owner ruling, and
  repository instructions before editing.
- Confirmed current HEAD `88833a30` contains the fail-closed certification
  runner and bundle authority plumbing but records Deliverable A as incomplete.
- Started independent read-only audits of the physical call graph, executor /
  broker adapter contract, and fixture/certification harness. GitNexus graph
  resources are unavailable in this session, so source tracing uses `rg`,
  direct reads, tests, and history.
- Added the lossless US `Frame` / executor-projection codec. It narrows every
  node view to compiler-declared physical columns plus structural keys,
  memberships, links, weights, strata, mass history, virtual receipts, and
  compiler-derived row atoms. Legacy results are rejected if they alter an
  unprojected surface; EXPAND lineage is classified from explicit support ids
  or an unambiguous canonical remap inverse.
- The codec's seven focused fixture-scale tests pass, including projection
  narrowing, virtual inputs, patch conversion, external-surface refusal,
  clone-copy proof, row classification, and validated merge. Targeted Ruff
  and `git diff --check` also pass.
- Added a one-shot `PhysicalOperation` capability to compiled-node broker
  sessions. The registered kernel can invoke only the prebound zero-argument
  operation with the compiled implementation pin and exact input-binding
  digest. Source reads stay denied inside this compatibility scope; sink
  access is confined to absolute compiler-owner roots; seeded NumPy
  constructors require explicit seed material; raw RNG authority cannot
  escape in the result.
- The bridge records every compiler-granted RNG site as an operational event.
  This transitional `legacy-v1` trust boundary cannot prove which private
  implementation subcall consumed which site, so the limitation remains
  explicit rather than being mislabeled as typed draw-by-draw consumption.
- All five focused physical-operation broker tests pass. Targeted Ruff and
  `git diff --check` pass.
- Added the bundle physical dispatcher. It consumes one exact compiled-node
  order, merges uniquely owned tolerated-absence receipts into the node
  projection, binds each callback input, invokes real `execute_node` and
  `apply_patch`, registers the compiler-pinned classifier only for EXPAND,
  restores validated frames without losing legacy sidecars, and refuses
  completion until every node ran exactly once.
- The dispatcher resolves only the three authored virtual-output forms
  (`frame.@...`, source receipts, and resolved weights), failing closed on an
  unknown meaning. Its operational journal records node ids/keys and validated
  patch, result-projection, and broker-receipt digests without entering
  normative artifact bytes.
- All three focused dispatcher tests, Ruff, `git diff --check`, and a real
  `execute_node` fixture-scale JOIN smoke pass.

### Next

1. Finish validating the production wiring behind `config_authority=bundle`
   and prove exact once-per-node
   full-stage dispatch plus D4 two-tier artifact equality at fixture scale.
2. Reconcile runner coverage with the completed node journal/reuse surface,
   run only fixture/unit verification, update the host commands and final
   output report, and stop before any host build.

## F1 continuation r5 executor input-state repair (2026-08-20)

### State

- The first real full-graph bundle dispatch reached `execute_node` and failed
  before its first callback. The late readiness contract treats scoped nulls
  in optional availability-pattern inputs as missing and emits the declared
  absence receipt, while the generic executor previously collapsed those
  nulls together with present non-null invalid values.
- Changing the fixture would hide the same production incompatibility. The
  executor input state therefore needs the same missing-versus-invalid
  distinction as the compiled late contract.

### Done

- Added a closed satisfied/missing/invalid input-cell state to executor
  validation. A declared absence receipt can now cover an absent column or
  scoped null cells; it still cannot cover infinity, a boolean numeric input,
  or nonnumeric content.
- Aligned `finite_numeric` validation with the late readiness contract by
  checking value coercibility rather than requiring a numeric storage dtype.
  Numeric strings such as the physical `TYPEHUGQ` surface are accepted;
  coercion failures and nonfinite values remain invalid.
- Focused validation passes for the null-with-receipt case, all three invalid
  refusal cases, numeric-string acceptance, and the broader required-input
  alternative selection cases. Ruff and `git diff --check` pass. No build ran.

### Next

1. Rerun the full 38-node dual-mode fixture and repair only further genuine
   projection/patch incompatibilities.
2. Resolve the separate audit finding that the compatibility scope currently
   records compiled RNG grants without consuming broker tokens; do not call
   that ledger-routed execution until the broker owns the draws.

## F1 continuation r5 physical-contract correction (2026-08-20)

### State

- The full-graph executor fixture next stopped at the primary-PUF node because
  eight optional boolean allocation-basis columns were compiled as
  `finite_numeric`. This is a real registry defect: the late readiness
  contract preserves their logical `non_null` value kind, and changing the
  fixture or weakening numeric validation would conceal it.
- The generated US spec digest consequently changes from
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`
  to `44893ea9631bca8bea53692abfc8f51eabb8dd04f3bcf5874a65459253a285b8`.

### Done

- Bumped the late-producer schedule payload from v16 to v17 and now derive
  each optional primary-PUF allocation input's value kind from the target
  column. Regenerated `imputation.yaml`; only the version and the eight
  boolean input kinds changed. Updated the two tests that pin the current US
  spec binding.
- `tools/generate_us_bundle_from_constants.py --check` passes with full
  generated-bundle validation. The canonical schedule test, live constants
  adapter binding test, adapter checkpoint byte-equality test, typed
  imputation reconstruction test, and every-checked-in-byte generator test
  pass. Targeted Ruff and `git diff --check` pass.
- One initial combined pytest command was interrupted after it remained in
  import collection with no output for several minutes. Re-running the tests
  separately with numerical-library thread counts pinned to one produced the
  passing results above; no sample build ran.
- Completed read-only audits of all seeded source, primary-PUF, and late
  transfer callbacks. They establish that the current compatibility scope is
  not ledger-routed: it logs grants and re-enables explicitly seeded private
  RNG, but never consumes broker tokens. The pending 38-node fixture also
  stubs every stochastic callback and therefore cannot prove this property.

### Next

1. Replace the compatibility exception with a broker-owned operation context,
   dynamic per-dispatch invocation plans, and broker-controlled QRF streams.
2. Preserve the three existing RNG owners inside the primary callback via a
   compiler-bound composite/delegated authority and subprocess continuation,
   or change the compiled stage graph explicitly; a local adapter cannot
   honestly bridge those owner boundaries.
3. Keep the bundle production activation uncommitted until a non-vacuous
   fixture consumes real broker tokens and remains byte-equal to constants.

## F1 continuation r5 broker-stream migration (2026-08-20)

### State

- The authority-preserving route keeps constants mode on its existing seeded
  calls and gives bundle mode narrow generator leases issued by the compiled
  ledger. Physical algorithms must accept those leases without learning
  broker internals or changing normative receipts.

### Done

- Added optional narrow generator seams at the two non-QRF primary-stage draw
  boundaries: partial clone household selection and capital-gains-tail random
  ranking. The legacy `None` paths still construct the same NumPy generators;
  an injected PCG64 stream produces identical entity tables, weights,
  metadata, and tail manifest.
- The combined fixture-scale equality test passes. Targeted Ruff and
  `git diff --check` pass; no sample build ran.

### Next

1. Bind these seams to the clone and tail pipeline-owner grants in the
   compiler's seed stream map, while retaining the primary producer-node
   owner for QRF.
2. Complete typed QRF, source, and late-transfer broker seams, then activate
   the physical executor only when every planned invocation is consumed.

## F1 continuation r5 QRF stream seam (2026-08-20)

### State

- Primary, source, and transfer QRFs all implement the same legacy-v1 split:
  `SeedSequence(seed).spawn(2)` yields persistent fit and draw PCG64 streams.
  The broker already issues that exact pair, but QRF previously had no way to
  consume it.

### Done

- Added an optional structural generator-pair input to monolithic fit,
  `start_chain`, and `fit_draw_next`. The default path is unchanged. The
  injected path uses only the pair's narrow draw methods and opaque state
  snapshot; fitted prediction retains the broker draw lease.
- Targetwise execution validates both supplied stream states against the
  checkpoint before either can advance, then writes post-step states into the
  existing JSON fields. No new checkpoint member or receipt field is added.
- Four focused tests pass for monolithic byte equality, targetwise state/draw
  equality, pre-draw drift refusal, and the existing JSON-resume equivalence.
  Targeted Ruff and `git diff --check` pass; no sample build ran.

### Next

1. Thread broker-issued QRF pairs into source, primary, and transfer physical
   callbacks, with exact dynamic invocation plans prepared before dispatch.
2. Prove existing checkpoint member bytes remain equal when the primary chain
   runs in-process under the bundle session.

## F1 continuation r5 late-transfer ledger route (2026-08-20)

### State

- Every late-transfer producer derives one family receipt seed, then one
  pattern seed and one persistent QRF fit/draw pair per live optional-predictor
  availability pattern. Pattern counts and boundaries depend on the current
  recipient frame and therefore cannot be frozen at executor construction.

### Done

- Added a side-effect-free per-family invocation planner that reproduces the
  live readiness/availability partition and emits ordered
  `acs_transfer_family_seed`, `acs_transfer_pattern_seed`, and
  `acs_qrf_fit_draw` invocations with the ledger's exact semantic material.
- Added an optional narrow transfer RNG broker. It consumes and verifies both
  legacy SHA-derived receipt seeds, obtains QRF pairs by pattern boundary, and
  passes those pairs into monolithic and cold target-banked QRF chains. The
  constants path remains unchanged.
- A one-family monolithic fixture and a three-target cold-bank fixture are
  byte-exact between private-seed and injected-stream paths. The cold-bank
  test compares every H5 target member as raw bytes. Both focused tests,
  targeted Ruff, and `git diff --check` pass; no sample build ran.
- Brokered target-bank prefix resume currently fails closed with a named
  restoration-grant error instead of reconstructing private RNG. Cold
  certification roots are unaffected; the documented host resume predicate
  still needs an exact compiled state-restoration route before it can pass.

### Next

1. Supply each plan per dispatch through the physical executor and require
   exact token consumption at broker seal.
2. Add compiler-bound QRF state restoration for target-bank resume; never
   infer or accept caller-authored state outside a predeclared grant.

## F1 continuation r5 narrow physical input (2026-08-20)

### State

- Validating a callback's output patch is insufficient if the callback can
  still read undeclared columns from a full `Frame` captured outside the
  executor. Bundle physical callbacks must receive the compiler projection as
  their only frame input.

### Done

- Added a lossless conversion from `ImmutableFrameProjection` to a narrow
  legacy `Frame`. It preserves only projected entity/link tables, typed
  weights, strata, mass history, and metadata; virtual receipts remain
  separate. No unprojected physical column can enter the callback through
  this boundary.
- The existing projection-boundary fixture now reconstructs the narrow frame
  and verifies its columns, links, weight kind/bits, metadata, and mass log.
  The focused test, targeted Ruff, and `git diff --check` pass; no sample build
  ran.

### Next

1. Change physical dispatch callbacks to accept this narrow frame plus the
   broker-owned operation context.
2. Treat any resulting missing read as a compiled input-contract defect; do
   not reopen the full frame inside bundle execution.
