# F1 lane notes

These notes are the implementation journal for the approved F1 drive lane.
Line citations in section 1 refer to the starting tree at
`7148513302070d9ff39a2d2729726487e709c047` unless a later commit is named.
`PROGRESS.md` is deliberately untouched, as required by the lane charter and
the repository journal convention.

## State

- Branch: `spec-engine-f1`; no push has been performed.
- Deliverable 1 is committed as `cd2b1d60`; deliverable 2 is committed in the
  current executor step; current deliverable: **3 — brokers**.
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

1. Commit deliverable 2 with the suite green.
2. Begin deliverable 3: RNG/file/env/clock brokers and operational receipts.

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
