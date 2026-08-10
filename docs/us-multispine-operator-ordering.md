# US multispine operator ordering

This note records both executable pool orderings in
`tools/build_us_multispine_pool.py`: the production stacked pipeline selected
by default and the byte-compatible retiring lineage selected explicitly with
`--legacy-two-spine`. The assembly and agreement contracts originated in
microcosm#581; the ratified microcosm#578 adoption replaces agreement with a
complete, origin-aware terminal battery on one stack. This documentation and
its fixture tests do not certify a full-data output artifact.

## Retired serial ordering

The earlier lineage used two serial builds. The first build produced an
operated ASEC-by-PUF-detail donor. The second build created ACS records,
transferred inputs from that donor, and only then appended ACS.

### `build_us_puf_support_base.py`

`PIPELINE_STEPS` and `STAGE_BOUNDARIES` are executable ordering
configuration. The monolithic and checkpointed paths implement the same
sequence.

| Boundary | Operators | State consumed |
|---|---|---|
| `source_construction` | `_load_base_frame_from_args` | Either an existing US base H5 or pooled ASEC unit frames built from the declared source years. |
| `pre_clone_enrichment` | `derive_us_cps_carried_inputs`; prior-year income, relationships, Medicare take-up, housing, eligibility, pregnancy, WIC, child support, disability benefits, workers compensation, weeks unemployed, childcare, energy subsidy, retirement contribution/distribution, and immigration operators | The ASEC-only frame, build year and seed; some operators also consume pinned external source tables. The adjacent-year income join and rent draw happen here, before row expansion. Housing can consume the pinned ACS 2022 rent donor when its input gate is not already satisfied. |
| `clone_feature_extraction` | `clone_us_frame_for_puf_support`; PUF donor extraction; primary-QRF initialization | The already enriched ASEC frame and processed PUF donor arrays. Cloning creates an ASEC copy and a PUF-tax-detail copy and splits weights across them. |
| `primary_qrf_chain` and `qrf_finalization` | Chained weighted QRF fits and finalization | Predictors on the cloned frame, PUF donor targets, design weights, fit seed, and estimator count. Predictions are assigned to the PUF-detail copy. |
| PUF tail and derived-detail stages | Capital-gains tail transfer, capital-gain distributions, QBI reconciliation | The cloned, QRF-imputed frame plus PUF donor detail and declared deterministic reconciliation rules. |
| post-clone input stages | WIC, housing assistance, prior-year income, child support, disability benefits, workers compensation, weeks unemployed, childcare, adult care, energy subsidy, retirement contributions/distributions, and education inputs | The operated ASEC/PUF-detail frame, source columns, build year, seed, and operator-specific external tables. Prior-year income's second invocation performs the PUF-support QRF from the values derived before cloning; it does not repeat the adjacent-year join. Signal gates run between these mutations. |
| geography and export | Congressional-district assignment, block-ladder assignment, H5 export | The operated frame, geography artifacts and seeds, followed by the final frame writer. |

Thus many derivation, imputation, and seeded-assignment operators run before
any ACS peer spine exists. The exported H5 is not a raw donor: it contains the
results of the pre-clone, PUF-transfer, post-clone, and optional geography
stages.

### `build_us_acs_multispine_base.py`

The former ACS builder took that exported H5 as `--base-h5`. Its runtime call
graph was:

1. Load and validate the dense ASEC-by-PUF-detail base and declared transfer
   coverage.
2. Fetch the hash-pinned ACS archives and build an ACS unit frame.
3. Map ACS-native inputs.
4. Preflight pooled geography when a PUMA ladder is supplied.
5. Run `transfer_acs_inputs(mapped_acs, operated_base, ...)`. The operated
   base is the QRF donor; declared input leaves and deterministic
   post-transfer structure are added to ACS.
6. Run `with_optional_acs_spine(operated_base, transferred_acs, ...)`.
   This is the first point at which the two household spines share one
   frame. It rescales their household mass, remaps colliding IDs, aligns
   nullable columns, and can assign the pooled PUMA geography ladder.
7. Audit fits and nullable inputs and write a pre-calibration staging H5.

Calibration is downstream of this tool. Simulation is not run by either
builder in this call graph.

The important ordering flaw is structural: ACS receives model-input
transfers from a donor after the donor has crossed the ASEC-only operator
sequence. Appending the transferred ACS records later does not cause those
operators to run over the combined population.

`build_us_acs_multispine_base.py` remains a deprecated but executable
compatibility path until microcosm#578 increment 4 retires the ACS local-release
overlay. The public command warns and delegates to the preserved implementation
under `tools/_legacy`; its summary and reviewed-null receipts remain the inputs
expected by `build_us_acs_local_release.py`. New multispine work uses the pool
builder below, but the supported legacy release recipe is not left half-working.

## Production stacked pool build

`build_us_multispine_pool.py` consumes only explicit local files and their
declared SHA-256 values:

- the dedicated `microcosm_us_asec_raw_stage` artifact emitted alongside the
  producer's `source_construction` checkpoint. Its stage tag is
  `raw_source_mapping` and its operator status is `operator_untouched`;
- the ACS household and person PUMS archives, whose caller-supplied hashes
  must also match the checked-in ACS source manifest;
- the canonical ACS 2022 rent donor used by the post-assembly housing
  operator; and
- the processed PUF H5 and source-year PUF CSV used by the existing donor
  loader.

The raw artifact is a second producer output, not a relabeling of
`pre_clone_enrichment`. It contains pooled ASEC unit structure and measured raw
columns. The only enrichment allowed there is faithful source mapping:
`LKWEEKS` and `ED_VAL` are restored by exact, pinned Census identity joins.
No `weeks_unemployed`, `educational_assistance`, carried-income split,
eligibility, pregnancy, take-up, childcare, retirement, or immigration output
is present. The producer still emits its historical enriched checkpoint and
final H5 for the sparse/dense single-spine release lineage; their operator
sequence and bytes are unchanged.

The dedicated raw artifact is produced by the checkpointed producer recipe
(`--stage all --checkpoint-dir ...`) at
`<checkpoint-dir>/asec_raw_stage.checkpoint.h5`. The legacy monolithic recipe
continues to produce only its historical release outputs.

The tool does not download any source. It verifies all file pins and validates
the raw artifact kind, stage, frame identity, operator status, and complete
operator-output absence before loading the peer frames. Measured ACS mappings
are allowed only when named by the ACS native-input receipt.

### Default sequence

1. `assemble_stacked_spine(...)` selects whole households independently from
   both survey arms with the single `sample_fraction` and `sample_seed`,
   restores each sample to its full-source design-weight mass, and assembles
   one origin-labeled frame. Standard rungs are `f001`, `f010`, and `f100`;
   the manifest binds the fraction, seed, exact realized ASEC/ACS counts,
   selected-lineage digests, the complete ordered native ACS household and
   person support/raw/household-parent/classification mappings, and the sampled
   native ACS TYPEHUGQ 2/3 household and person lineage digests. The full PUF
   remains a donor and is never sampled.
2. The spine-blind source-preparation chain derives the native predictors and
   pre-clone operator outputs needed by the early declared cross-origin fills.
   Historical kernels run on the raw-`PERIDNUM` CPS/ASEC availability
   projection and merge only their declared outputs back into the stack. In
   particular, the pinned ACS rent artifact trains `with_us_housing_inputs`,
   which materializes `pre_subsidy_rent` on ASEC; native ACS `RNTP`/`GRNTP`
   remain predictors and are not relabeled as that model input.
   Targets produced only by the later PUF pass or late source producers are
   excluded from this early authority surface. No population operator selects
   behavior from the source-channel labels.
3. `gap_fill_stacked_spine(...)` runs the two immutable directions over the
   same frame: ASEC survey fields fill ACS nulls, then ASEC-produced housing
   rent fills ACS housing-unit nulls in a separately banked direction.
   Activation authority is source-and-role exact, observed zero is not
   absence, and native donor cells must remain byte-identical. Every declared
   target resolves through the operator-output registry to a producer channel
   and stage that must strictly precede its direction's check. Only the
   explicit `cps_source` and `whole_pool` execution scopes carry authority;
   unknown scopes fail rather than inheriting whole-pool authority. ACS TYPEHUGQ
   2/3 people remain null only when their live mask matches the assembly-bound
   native group-quarters lineage in every clone role and the declared
   structural-absence rule; relabeling a housing unit later cannot create
   absence authority. Filling those rows with zero or a donor housing value
   would synthesize an unobserved housing unit. Every other early target must
   finish with zero `unmodeled_rows` and zero residual nulls: transfer
   accounting alone is not terminal absence authority. The #608 per-target
   banks sit beneath the stack-bound checkpoint identity.
4. `run_stacked_puf_pass(...)` attaches the separately controlled PUF clone
   arm (`clone_attachment_fraction`, default `1.0`) and runs one primary QRF
   pass across both survey origins. The strict recipient surface applies the
   [2024 ACS PUMS Data Dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf)
   universes for `WAGP` and `SEMP`. The ASEC evidence is the established
   producer, not a new convention: [`derive_us_cps_carried_inputs`](../packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py#L150-L155)
   maps `WSAL_VAL` and `SEMP_VAL`, while its [`_source`](../packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py#L355-L362)
   coercion materializes source blanks as numeric zero. A direct audit of the
   certified Build J artifact (`populace-us-2024-buildj-sparse-rmloss100-75d5add-20260710T094201Z`,
   built with PolicyEngine US 1.764.6) found all 15,509 under-15 ASEC people
   numeric zero on raw and mapped earnings and retained all seven all-child
   ASEC tax units with zero earnings sums. The PUF-detail ASEC clone likewise
   has 21,737 under-15 zeros and 22 retained all-child units. Cross-arm
   comparison therefore requires the ACS produced frame to use the same
   semantics: a
   named pre-QRF operator materializes `0.0` only on declared under-15 mapped
   leaves, records the exact person/unit counts and the
   `acs_2024_pums_wagp_age_15_plus` and
   `acs_2024_pums_semp_age_15_plus` rule IDs in its receipt, and leaves raw
   `WAGP`/`SEMP` blanks unchanged. Eligible mapped or raw nulls still fail in
   [`_require_complete_recipient_predictor_sources`](../packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py#L3036-L3167)
   with the original greppable `missing values before coercion` diagnostic and
   the responsible rule ID. All-child units remain
   recipients with receipted zero earnings predictors; they are not generic
   `fillna` results. PUF earnings allocation is limited to age-15-plus people,
   so an under-15 first person and every member of an all-child unit remain
   zero even when the unit prediction is positive. The mapped leaves' exact
   universe zeros must agree with the raw `WAGP`/`SEMP` authority columns,
   which are mandatory on scoped ACS rows. A
   same-named tax-unit/person source
   collision is ambiguous and fails instead of letting receipt and feature
   construction choose different grains.
   The full ACS 2024 archive audit found exactly 510,098 under-15 people, equal
   to the complete WAGP/SEMP blank set. At the 1% rung the original failure set
   was exactly 2,998 recipient tax units touching 5,294 under-15 people: 2,980
   mixed-age units and 18 units with no eligible member. These populations are
   counted by the universe receipt; the latter 18 remain zero-basis recipients
   to match the certified ASEC arm.
   Exact structural-person, affected-unit, mixed-unit, and empty-unit counts
   and lineage/feature digests bind the root QRF manifest, both immutable
   banks, every target checkpoint, live finalization, and the outer stacked
   receipt. PUF donors stay full. The clone-2 capital-gains-tail operator runs
   inside this pass; exact tail-owned and QRF-owned cells are checked after
   source completion and every later phase.
   The tail declares support separately for each filing status. Its required
   minimum is the number of selected q99.5 PUF tail donors in that status. Its
   observed support is the number of unique, single-tax-unit PUF-detail
   recipient households in that status whose half-weight can carry the global
   maximum assigned tail-donor weight. When observed support falls below the
   minimum, the operator skips the whole status and emits a named, counted
   `insufficient_support` receipt with the status, observed count, and required
   minimum. It never borrows recipients from another status or partially
   attaches the status. All 22 AGI bands remain nearest-first fallback choices
   within a filing status; they are not separate hard partitions. A status
   with zero selected donors, such as `SURVIVING_SPOUSE` in the pinned tail,
   receipts `not_applicable` rather than a skip.

   At the standard 1% rung, `SINGLE` and `HEAD_OF_HOUSEHOLD` attach,
   `JOINT` and `SEPARATE` receipt `insufficient_support`, and
   `SURVIVING_SPOUSE` receipts `not_applicable`. Every positive-requirement
   status meets support at the 10% and full rungs. Filtering occurs only after
   the operator constructs the original global-capacity candidate pool, so
   attached statuses retain the pre-change assignment bytes. Full-scale output
   therefore remains unchanged.

   The authority versions distinguish the two contracts. The primary-QRF root
   and target checkpoint schema remains version 6. The capital-gains tail
   manifest uses schema version 2 and binds its support contract and receipt.
   The canonical stacked authority and outer stacked checkpoint materializer
   use version 8, while the pool stage checkpoint materializer uses version 4.
   The outer base identity binds primary-QRF version 6, the ACS universe and
   QBI reconciliation contracts, the tail schema and support contract, and
   late-producer registry schema version 2. The companion pool manifest uses
   schema version 5.
   Older outer authority or materializer payloads are stale; primary-QRF
   version 6 remains current.
5. One declared late-producer DAG schedules the primary PUF/tail pass, all 16
   post-clone source operators, and all 19 bounded transfer groups. Each node
   declares every effective input and output. A callback cannot run until each
   input is filled on its required scope or has the exact counted
   declared-absence receipt which that input contract tolerates. Import
   validation rejects unknown producers, ambiguous ownership, uncovered
   targets, and cycles, naming a deterministic cycle path. Lexical Kahn waves
   make the order independent of registry iteration. The resulting first wave
   is the primary PUF pass alone; later waves interleave source and transfer
   work. In particular, PUF batch 5 transfers
   `sstb_self_employment_income_before_lsr` before adult care consumes it, PUF
   batch 2 transfers `qualified_tuition_expenses` before education consumes it,
   pregnancy precedes WIC, and childcare precedes adult care. This order is
   derived from producer/input edges, never imposed as a second hand-written
   list.

   The complete model donor is the ASEC-origin PUF-detail role. Authority is
   target-specific: every live positive-index clone must already observe a
   PUF-produced target, every ASEC-origin clone must already observe a
   source-produced target, and dual-produced targets require the union. A null
   on any producer row is terminal; only complementary recipient rows may be
   filled from QRF predictions. No blanket null-to-zero synthesis occurs,
   every producer cell stays byte-identical, and zero residual nulls are
   required.
6. The transferred checkpoint records the early gap-fill banks, 19 distinct
   late-transfer banks, the primary-QRF bank, the complete 36-node DAG receipt,
   tail manifest and its per-status support receipt, weights audit,
   stack-manifest digest, fraction/seed, clone controls, and the channel-aware
   producer-precedence schedule. The DAG receipt binds all 54 edges, all input
   inventories, five derived waves, exact execution rows, the once-only source
   finalizer, and the 19-group/70-target aggregate. The same identity regime
   governs cold and resumed builds. Checkpoint emission, resume, and final
   publication each reject the receipt unless it carries the exact canonical
   stacked authority; NON-CANONICAL test receipts cannot ship.
7. Schedule-D preparation, deterministic derivation, seeded inputs, and
   batched simulation run on the transferred stack. QBI reconciliation uses
   the same source declaration: it fails on any in-universe self-employment
   null and preserves the receipted ACS under-15 base self-employment zero in
   every clone role. That narrow source exception does not suppress independent QBI
   identities: every QBI detail row is still reconciled and checked. The
   operator declares the base self-employment rewrite alongside its QBI
   outputs and receipts a whole-person input-table digest, a declared-output
   digest, changed-row counts, exact preservation of every undeclared person
   column, entity, link, weight, stratum, mass-log, and metadata surface, an
   exact structural-source exclusion count, and zero forbidden
   structural-source mutations. Receipt generation recomputes the complete
   deterministic before/after transition and validates exact keys, types,
   nonnegative counts, nested universe receipts, preservation claims, and
   every SHA-256. Only after that succeeds does the derive boundary bind the
   canonical receipt SHA and its preimage/output digests into deeply immutable
   frame metadata and a separately carried transition-authority field.
   Checkpoint H5 metadata and its sidecar content-bind that independent field.
   Simulated checkpoint emission, durable checkpoint write and load, simulated
   resume, legacy and stacked manifest construction, and both publication
   entry points require the live receipt SHA to match the carried authority.
   They also recompute the declared-output and QBI-driver digests, exact live
   universe receipt, kernel fixed point, preservation-count equations, and
   exact person-column inventory. The only later person additions permitted by
   that inventory are outputs named by the checked-in take-up contract and the
   seed-stage receipt; an arbitrary receipt cannot whitelist a column. Legacy
   receipts must use
   `derive.qbi_input_reconciliation`; stacked receipts must use
   `derive.pool_derivation.qbi_input_reconciliation`; missing, wrong-route, or
   ambiguous receipts fail. A forged envelope, a laundered input inventory,
   and even a self-consistent alternate fixed point paired with a freshly
   generated receipt therefore fail rather than becoming publication
   authority. The tool
   retains the existing `assembled`, `transferred`, and `simulated` #599
   boundaries.
8. A fresh `us_stacked_completeness` gate proves every declared input is
   observed or has exact source-by-role absence authority. The terminal
   `us_by_origin_battery` then evaluates all 131 declared targets (114 person,
   9 tax-unit, 8 SPM-unit), plus joint immigration structure, using an
   immutable live-digested per-column metric registry. Metric choice never
   dispatches from physical dtype. A digest-bound structural-absence rule may
   remove only its exact proven cells from a comparison's applicability scope;
   any additional null or filled structural cell is terminal. Manifest
   emission revalidates the exact structural-rule schema, row arithmetic,
   per-role proofs, and battery exclusion count from the immutable gate
   snapshot, so authority metadata cannot be grafted onto invented absence.
   Both terminal gates reauthenticate the tail manifest and project its exact
   per-status support receipt into gate details. A missing, altered, or rebound
   status, observed count, required minimum, attachment decision, or manifest
   digest fails closed.
   At small rungs, comparisons outside the validity domain receipt
   `insufficient_support`; tolerances do not widen.
9. Only after both gates run does publication write the nullable H5,
   diagnostics, and readiness manifest. Success, failed gate, and exception
   paths each append a durable Logbook spool row beside the output, with the
   fraction token, seed, code/input/identity pins, phases, gate-receipt
   pointers, wall time, artifact location, and disposition.

### Late producer/input DAG

The late stage is a declared producer/input graph, not a fixed source loop
followed by a fixed transfer loop. Its registry contains 36 producers: the
primary PUF/tail producer, 16 post-clone source producers, and 19 bounded
late-transfer producers. Import derives and validates the schedule. Unknown
producers, duplicate ownership, uncovered transfer targets, and cycles fail at
import; a cycle error prints its deterministic cycle path. Readiness is checked
again immediately before each callback. Every required input must be nonnull
on its declared scope, finite when marked numeric, or carry one of that input's
explicitly tolerated counted-absence receipts. A receipt tolerated by one
input does not authorize another input, and no missing value is converted to
zero.

The notation below is executable-contract shorthand. `p`, `tu`, `s`, and `h`
mean person, tax unit, SPM unit, and household. `F(x)` requires numeric finite
values; `+` is an all-of alternative; `|` separates alternatives; and `?R`
means that only the named, counted absence receipt may replace that optional
input. `@weight` is the Frame-resolved entity weight and `@sidecar` or `@bank`
is an authenticated resource receipt, not a physical column.

The primary PUF producer's complete 15-input inventory is:

```text
filing status = tu.filing_status_input | tu.filing_status
tax-unit membership = p.person_tax_unit_id + tu.tax_unit_id
F(p.employment_income_before_lsr)
F(p.self_employment_income_before_lsr)
F(p.taxable_interest_income)
dividends = F(p.dividend_income)
          | F(p.qualified_dividend_income) + F(p.non_qualified_dividend_income)
          | F(tu.dividend_income)
short-term gains = F(p.short_term_capital_gains)
                 | F(tu.short_term_capital_gains)
long-term gains = F(p.long_term_capital_gains_before_response)
                | F(p.long_term_capital_gains)
                | F(tu.long_term_capital_gains)
p.person_id
tu.tax_unit_id
p.person_support_channel
p.person_support_clone_index
tu.@weight
tu.@puf_donor_tax_units
tu.@primary_qrf_checkpoint
```

The common role-aware source bundle `C` is the following complete set:

```text
p.person_id; p.@weight; p.person_support_channel;
p.person_support_clone_index ?R;
F(p.age) | F(p.A_AGE);
p.is_male | p.is_female | p.A_SEX;
p.has_esi; p.person_tax_unit_id; p.tax_unit_role_input;
F(p.employment_income_before_lsr) | F(p.WSAL_VAL);
F(p.self_employment_income_before_lsr) | F(p.SEMP_VAL);
[F(p.social_security_retirement) + F(p.social_security_disability)
 + F(p.social_security_survivors) + F(p.social_security_dependents)]
 | F(p.SS_VAL);
tu.tax_unit_id; tu.filing_status_input | tu.filing_status
```

Every source node also has a required whole-pool
`p.person_support_clone_index` scheduling input produced by primary PUF; this
turns clone attachment into an edge even where the kernel does not inspect the
column. The table gives every kernel input in addition to that structural
input. `C + ...` expands exactly to the bundle above.

| Post-clone source producer | Complete effective kernel input set |
|---|---|
| `impute_us_housing_assistance_to_puf_support` | `C + p.person_spm_unit_id + s.spm_unit_id + s.receives_housing_assistance + s.takes_up_housing_assistance_if_eligible + s.spm_unit_support_channel + s.spm_unit_support_clone_index ?R` |
| `with_us_adult_care_inputs` | `F(p.age) + F(p.employment_income_before_lsr) + F(p.self_employment_income_before_lsr) + F(p.sstb_self_employment_income_before_lsr) + p.PEDISDRS + p.is_full_time_college_student + p.tax_unit_role_input + p.person_tax_unit_id + p.person_spm_unit_id + p.person_id + (p.person_support_clone_index | p.person_support_channel) + F(s.spm_unit_pre_subsidy_childcare_expenses) + s.spm_unit_id + tu.tax_unit_id + p.@weight + s.@weight + tu.@weight` |
| `with_us_child_support_inputs` | `C + p.CSP_VAL + p.CHSP_VAL` |
| `with_us_childcare_inputs` | `C + p.person_spm_unit_id + p.SPM_CHILDCAREXPNS + s.spm_unit_id` |
| `with_us_disability_benefits` | `C + p.DIS_VAL1 + p.DIS_SC1 + p.DIS_VAL2 + p.DIS_SC2` |
| `with_us_education_inputs` | `(p.ED_VAL | p.@education_assistance_sidecar) + F(p.qualified_tuition_expenses) + p.person_id + p.@weight` |
| `with_us_energy_subsidy_input` | `C + p.person_spm_unit_id + p.SPM_ENGVAL + s.spm_unit_id` |
| `with_us_immigration_inputs` | `p.PRCITSHP + p.PEINUSYR + p.PENATVTY + p.A_AGE + p.A_MARITL + p.A_SPOUSE + p.A_HSCOL + p.WSAL_VAL + p.SEMP_VAL + p.MCARE + p.CAID + p.IHSFLG + p.CHAMPVA + p.MIL + p.PEN_SC1 + p.PEN_SC2 + p.RESNSS1 + p.RESNSS2 + p.SS_YN + p.SSI_YN + p.PEIO1COW + p.A_MJOCC + p.PEAFEVER + p.SPM_CAPHOUSESUB + p.person_id + p.@weight + ([p.source_year + p.source_person_id] | p.person_id)` |
| `with_us_medicare_take_up_input` | `p.MCARE + p.person_id + p.@weight` |
| `with_us_pregnancy_inputs` | `p.A_SEX + p.A_AGE + p.person_id + p.@weight + ([p.source_year + p.source_household_id + p.source_person_id] | p.person_id)` |
| `with_us_prior_year_income_inputs` | `C + p.source_year + p.PERIDNUM + p.WSAL_VAL + p.SEMP_VAL + p.I_ERNVAL + p.I_SEVAL` |
| `with_us_retirement_contribution_inputs` | `C + p.RETCB_VAL + p.WSAL_VAL + p.SEMP_VAL` |
| `with_us_retirement_distribution_inputs` | `C + p.DST_SC1 + p.DST_VAL1 + p.DST_SC2 + p.DST_VAL2 + p.DST_SC1_YNG + p.DST_VAL1_YNG + p.DST_SC2_YNG + p.DST_VAL2_YNG + p.taxable_ira_distributions` |
| `with_us_weeks_unemployed` | `p.source_year + p.PERIDNUM + (p.LKWEEKS | p.@weeks_unemployed_sidecar) + (p.age | p.A_AGE) + (p.is_male | p.is_female | p.A_SEX) + (p.tax_unit_is_joint | [p.person_tax_unit_id + tu.tax_unit_id + tu.filing_status_input] | [p.person_tax_unit_id + tu.tax_unit_id + tu.filing_status]) + (p.tax_unit_role_input | [p.is_tax_unit_head + p.is_tax_unit_spouse + p.is_tax_unit_dependent]) + (p.unemployment_compensation | p.UC_VAL) ?R + p.person_support_channel + p.@weight` |
| `with_us_wic_claim_input` | `p.age + p.is_female + p.is_pregnant + p.own_children_in_household + p.person_family_id + p.@weight + ([p.source_year + p.source_household_id + p.source_person_id] | p.person_support_source_id | p.person_id)` |
| `with_us_workers_compensation` | `C + p.WC_VAL` |

For a transfer whose target entity is `E`, the complete common transfer input
bundle `T(E)` is:

```text
p.person_id + p.person_support_channel + p.person_support_clone_index + p.@weight
+ E.E_id + E.@weight
+ p.person_E_id                         # only when E is not person
+ F(p.age) + p.is_female
+ [p.state_fips | (p.person_household_id + h.household_id + h.state_fips)]
+ F(p.employment_income_before_lsr) ?R
+ F(p.self_employment_income_before_lsr) ?R
+ [(p.social_security_retirement + p.social_security_disability
     + p.social_security_dependents + p.social_security_survivors)
    | p.acs_social_security_income] ?R
+ [(p.taxable_private_pension_income + p.tax_exempt_private_pension_income
     + p.taxable_ira_distributions) | p.acs_retirement_income] ?R
+ [(p.taxable_interest_income + p.tax_exempt_interest_income
     + p.qualified_dividend_income + p.non_qualified_dividend_income
     + p.rental_income + p.estate_income)
    | p.acs_interest_dividend_rental_income] ?R
+ (p.is_household_head | p.RELSHIPP | p.A_EXPRRP | p.A_LINENO) ?R
+ (p.tenure_type | s.spm_unit_tenure_type | h.TEN | h.H_TENURE) ?R
```

Every one of the 19 transfer nodes also requires PUF-clone producer evidence
for `p.tax_exempt_interest_income` and `p.estate_income`, plus producer evidence
for every target listed below. A target shown in both producer columns requires
both scopes; that is the two-target PUF/source overlap. Thus the table is the
complete per-node input delta over `T(E)`, as well as the exact 70-target
partition. Transfer rows abbreviate the registry's leading `transfer:`; source
names in these tables abbreviate the leading `source:`.

| Transfer producer | Targets | PUF target inputs | Source target inputs |
|---|---|---|---|
| `person/adult_care` | `is_incapable_of_self_care`, `pre_subsidy_care_expenses` | — | both from `with_us_adult_care_inputs` |
| `person/model_required_boolean` | `is_pregnant` | — | from `with_us_pregnancy_inputs` |
| `person/puf_tax_itemization__batch_1` | `tax_exempt_interest_income`, `long_term_capital_gains_on_collectibles`, `non_sch_d_capital_gains`, `alimony_expense`, `salt_refund_income`, `charitable_cash_donations`, `charitable_non_cash_donations`, `home_mortgage_interest` | all targets | — |
| `person/puf_tax_itemization__batch_2` | `investment_interest_expense`, `investment_income_elected_form_4952`, `student_loan_interest`, `educator_expense`, `qualified_tuition_expenses`, `casualty_loss`, `unreimbursed_business_employee_expenses`, `traditional_ira_contributions_desired` | all targets | `traditional_ira_contributions_desired` from `with_us_retirement_contribution_inputs` |
| `person/puf_tax_itemization__batch_3` | `self_employed_pension_contributions_desired`, `estate_income`, `farm_income`, `farm_rent_income`, `partnership_income`, `partnership_self_employment_net_earnings`, `estate_income_would_be_qualified`, `farm_operations_income_would_be_qualified` | all targets | `self_employed_pension_contributions_desired` from `with_us_retirement_contribution_inputs` |
| `person/puf_tax_itemization__batch_4` | `farm_rent_income_would_be_qualified`, `partnership_s_corp_income_would_be_qualified`, `rental_income_would_be_qualified`, `self_employment_income_would_be_qualified`, `sstb_self_employment_income_would_be_qualified`, `business_is_sstb`, `qualified_bdc_income`, `qualified_reit_and_ptp_income` | all targets | — |
| `person/puf_tax_itemization__batch_5` | `sstb_self_employment_income_before_lsr`, `sstb_unadjusted_basis_qualified_property`, `sstb_w2_wages_from_qualified_business`, `unadjusted_basis_qualified_property`, `w2_wages_from_qualified_business` | all targets | — |
| `person/source_operator_child_support` | `child_support_expense`, `child_support_received` | — | both from `with_us_child_support_inputs` |
| `person/source_operator_disability_benefits` | `disability_benefits` | — | from `with_us_disability_benefits` |
| `person/source_operator_education_inputs` | `attends_eligible_educational_institution_for_american_opportunity_credit`, `educational_assistance`, `has_american_opportunity_credit_1098_t_or_exception`, `has_american_opportunity_credit_institution_ein`, `is_enrolled_at_least_half_time_for_american_opportunity_credit`, `is_pursuing_credential_for_american_opportunity_credit` | — | all from `with_us_education_inputs` |
| `person/source_operator_immigration` | `ssn_card_type`, `immigration_status_str` | — | both from `with_us_immigration_inputs` |
| `person/source_operator_medicare_take_up` | `takes_up_medicare_if_eligible` | — | from `with_us_medicare_take_up_input` |
| `person/source_operator_retirement_contributions` | `roth_401k_contributions_desired`, `roth_ira_contributions_desired`, `traditional_401k_contributions_desired` | — | all from `with_us_retirement_contribution_inputs` |
| `person/source_operator_retirement_distributions` | `keogh_distributions`, `tax_exempt_ira_distributions`, `taxable_401k_distributions`, `taxable_403b_distributions`, `taxable_sep_distributions` | — | all from `with_us_retirement_distribution_inputs` |
| `person/source_operator_weeks_unemployed` | `weeks_unemployed` | — | from `with_us_weeks_unemployed` |
| `person/source_operator_wic_claim` | `would_claim_wic` | — | from `with_us_wic_claim_input` |
| `person/source_operator_workers_compensation` | `workers_compensation` | — | from `with_us_workers_compensation` |
| `tax_unit/puf_tax_itemization` | `domestic_production_ald`, `unrecaptured_section_1250_gain`, `first_home_mortgage_balance`, `first_home_mortgage_interest`, `first_home_mortgage_origination_year`, `health_savings_account_ald` | all targets | — |
| `spm_unit/source_operator_energy_subsidy` | `spm_unit_energy_subsidy` | — | from `with_us_energy_subsidy_input` |

#### Complete dependency edges

The following three tables enumerate all 54 unique producer-to-consumer edges.
Multiple values on one row are the input reasons carried by that edge. Bare
source names carry the registry prefix `source:` and transfer paths carry
`transfer:`.

The 16 primary-PUF-to-source edges are:

| Consumer source | Late/structural inputs supplied by primary PUF |
|---|---|
| `impute_us_housing_assistance_to_puf_support` | clone index; employment; self-employment; four Social Security components |
| `with_us_adult_care_inputs` | clone index; employment; self-employment |
| `with_us_child_support_inputs` | clone index; employment; self-employment; four Social Security components |
| `with_us_childcare_inputs` | clone index; employment; self-employment; four Social Security components |
| `with_us_disability_benefits` | clone index; employment; self-employment; four Social Security components |
| `with_us_education_inputs` | clone index |
| `with_us_energy_subsidy_input` | clone index; employment; self-employment; four Social Security components |
| `with_us_immigration_inputs` | clone index |
| `with_us_medicare_take_up_input` | clone index |
| `with_us_pregnancy_inputs` | clone index |
| `with_us_prior_year_income_inputs` | clone index; employment; self-employment; four Social Security components |
| `with_us_retirement_contribution_inputs` | clone index; employment; self-employment; four Social Security components |
| `with_us_retirement_distribution_inputs` | clone index; employment; self-employment; four Social Security components; `taxable_ira_distributions` |
| `with_us_weeks_unemployed` | clone index |
| `with_us_wic_claim_input` | clone index |
| `with_us_workers_compensation` | clone index; employment; self-employment; four Social Security components |

There are also 19 primary-PUF-to-transfer edges: one to every row of the
transfer table above. Each carries the shared PUF-clone investment predictors
`tax_exempt_interest_income` and `estate_income`; a PUF-owned target in that
row is an additional reason on the same edge.

The remaining 19 edges are:

| Producer | Consumer | Input reason |
|---|---|---|
| `with_us_adult_care_inputs` | `transfer:person/adult_care` | `is_incapable_of_self_care`, `pre_subsidy_care_expenses` |
| `with_us_child_support_inputs` | `transfer:person/source_operator_child_support` | `child_support_expense`, `child_support_received` |
| `with_us_childcare_inputs` | `with_us_adult_care_inputs` | `spm_unit_pre_subsidy_childcare_expenses` |
| `with_us_disability_benefits` | `transfer:person/source_operator_disability_benefits` | `disability_benefits` |
| `with_us_education_inputs` | `transfer:person/source_operator_education_inputs` | six education outputs listed above |
| `with_us_energy_subsidy_input` | `transfer:spm_unit/source_operator_energy_subsidy` | `spm_unit_energy_subsidy` |
| `with_us_immigration_inputs` | `transfer:person/source_operator_immigration` | `ssn_card_type`, `immigration_status_str` |
| `with_us_medicare_take_up_input` | `transfer:person/source_operator_medicare_take_up` | `takes_up_medicare_if_eligible` |
| `with_us_pregnancy_inputs` | `with_us_wic_claim_input` | `is_pregnant` |
| `with_us_pregnancy_inputs` | `transfer:person/model_required_boolean` | `is_pregnant` |
| `with_us_retirement_contribution_inputs` | `transfer:person/puf_tax_itemization__batch_2` | `traditional_ira_contributions_desired` source scope |
| `with_us_retirement_contribution_inputs` | `transfer:person/puf_tax_itemization__batch_3` | `self_employed_pension_contributions_desired` source scope |
| `with_us_retirement_contribution_inputs` | `transfer:person/source_operator_retirement_contributions` | three contribution outputs listed above |
| `with_us_retirement_distribution_inputs` | `transfer:person/source_operator_retirement_distributions` | five distribution outputs listed above |
| `with_us_weeks_unemployed` | `transfer:person/source_operator_weeks_unemployed` | `weeks_unemployed` |
| `with_us_wic_claim_input` | `transfer:person/source_operator_wic_claim` | `would_claim_wic` |
| `with_us_workers_compensation` | `transfer:person/source_operator_workers_compensation` | `workers_compensation` |
| `transfer:person/puf_tax_itemization__batch_2` | `with_us_education_inputs` | `qualified_tuition_expenses` |
| `transfer:person/puf_tax_itemization__batch_5` | `with_us_adult_care_inputs` | `sstb_self_employment_income_before_lsr` |

The lexically canonical waves have sizes `(1, 17, 14, 3, 1)`:

1. `primary_puf_qrf`.
2. Housing assistance; child support; childcare; disability; energy;
   immigration; Medicare; pregnancy; prior-year income; retirement
   contributions; retirement distributions; weeks unemployed; workers'
   compensation; person PUF batches 1, 4, and 5; tax-unit PUF transfer.
3. Adult care; WIC; pregnancy transfer; person PUF batches 2 and 3; child
   support, disability, immigration, Medicare, retirement-contribution,
   retirement-distribution, weeks-unemployed, workers'-compensation, and
   SPM-energy transfers.
4. Education; adult-care transfer; WIC transfer.
5. Education transfer.

Registry schema version 2 binds the canonical input declarations, outputs,
edges, and waves. The schedule SHA-256 is
`67cf85077a0fb4611208129977f783c316a26802728b8d4b723a34d6eb0e7b8e`;
the full payload SHA-256 is
`a16b15e65703d7a563c9efb6aea004119336855611d8371aa11d42bd7b7b541a`.
Reversing registry iteration produces those same bytes.

### Downstream hard-completeness audit

This table makes the stacked 1% supplier and starvation behavior explicit at
every remaining boundary. An early `unmodeled_rows` receipt is merely an
accounting result. The tail stage may issue its declared per-status
`insufficient_support` receipt before terminal evaluation; the by-origin
battery uses the same status name only after a comparison surface is complete
and valid. Neither receipt authorizes an upstream null.

| Boundary | Hard requirement | Stacked 1% supplier | Can an upstream insufficient-support/unmodeled state starve it? |
|---|---|---|---|
| Early gap-fill handoff | Donors observe every declared target; every recipient null is filled except the exact ACS group-quarters rent rule. Nonstructural `unmodeled_rows` and residual nulls are forbidden. | The pre-clone ASEC source operators supply 48 early targets to the two ASEC-to-ACS directions. | No accepted starvation remains. A nonstructural residual fails before cloning; literal `insufficient_support` is not an early-transfer outcome. |
| Clone attachment | Input rows are all clone 0; the seeded whole-household selection, lineage, pair weights, fraction, and seed agree exactly. | The completed gap-filled stack and the attachment sampler. | A permitted structural rent null is copied with its authority. Any other early residual has already failed. |
| PUF raw predictor sources | Every filing-status, count, and income component is observed in its declared source universe. Raw WAGP/SEMP authority is present and agrees with mapped leaves; a cross-grain source collision is rejected. A null on any eligible member fails before coercion. | Structure supplies status/count; ACS-native or ASEC-carried earnings supply earnings; early transfer supplies interest, dividends, and gains. | No. ACS under-15 WAGP/SEMP blanks are an exact source-universe state, not transfer starvation; all other source nulls fail. |
| PUF tax-unit features | Every clone-1 recipient has a finite feature vector. Post-aggregation NaN, `+inf`, and `-inf` are counted by named predictor and rejected before fitting; none is coerced or snapped to zero. | Universe-aware person sums plus tax-unit structural inputs. | No. Eligible member values must be complete; the only special case is an all-child unit whose numeric-zero predictor is explicitly owned and counted by the named universe-zero rule. |
| Primary QRF banks and chain | Donor/recipient banks are immutable; target order and RNG prefix are contiguous; all targets complete; live recipient identity, source-universe receipt, and feature digest match before finalization. | The processed full PUF donor and strict recipient checkpoint initialized above. | No. Mutation or missing receipt invalidates the bank; it cannot resume under legacy semantics. |
| Outer pool checkpoint identity and resume | Primary-QRF schema v6, tail-manifest schema v2, late-registry schema v2, stacked checkpoint/authority v8, pool checkpoint materializer v4, pool manifest schema v5, and the ACS-universe, QBI-mutation, tail-support, and complete late-DAG identities must match exactly before any cached stage is discovered. | Fresh input pins, live stack receipt, scale controls, code identity, and all semantic contract identities. | No. An older outer materializer or authority payload is stale; a self-consistent old receipt cannot reopen a checkpoint. Primary-QRF v6 remains current. |
| Clone-2 capital-gains tail | Each filing status requires as many eligible recipient households as selected q99.5 donors. Eligibility requires unique single-tax-unit PUF-detail lineage and half-weight capacity for the global maximum assigned donor weight. An adequate status assigns every selected donor once; a thin status skips as a whole with a named, counted `insufficient_support` receipt. | Completed clone-1 QRF output and full PUF tail donors. At 1%, `SINGLE` and `HEAD_OF_HOUSEHOLD` attach, `JOINT` and `SEPARATE` skip, and zero-requirement `SURVIVING_SPOUSE` is `not_applicable`. | No widening or partial attachment is permitted. All 22 AGI bands provide nearest-first fallback only inside a status. Universe-aware PUF recipients remain eligible, including explicitly receipted empty-universe tax units. |
| Late producer DAG | Before any callback, all declared inputs are filled on their required scopes or carry an input-specific counted absence receipt; numeric inputs are finite. The exact derived order, readiness rows, once-only source finalizer, and bounded transfer receipts must validate. | Primary PUF/tail, 16 source producers, and 19 bounded transfer groups execute in five derived waves. | No. The refusing producer names the unfilled input and its declared producing stage. A cycle fails at import with its path. |
| Late transfer completion | Every declared PUF-clone or ASEC source-producer cell is nonnull; all complementary recipients are filled; the allowed count for both unmodeled and residual rows is zero. | Forty-three PUF and 29 source targets, with two overlaps, supply the 70-target late surface. | No. A missing producer or recipient value is terminal at this boundary. |
| Fit-weight audit | Every primary and post-PUF QRF fit receipts its resolved entity weight kind, and the collected fit records pass the weights audit before a transferred checkpoint can exist. | Calibrated household weights mapped by the frame to each modeled entity. | No. A missing, inconsistent, or manually substituted weight declaration fails before checkpoint emission. |
| Tail preservation | Tail manifest, support decisions, attached descendants, IDs, weights, provenance, joint vector, and non-tail QRF cells remain exact after completion, transfer, derive, seed, and simulation. | The schema-v2 tail manifest and support receipt bound during the PUF pass and projected into both terminal gates. | A support receipt cannot authorize mutation. Any byte or identity change in an attached status, any descendant for a skipped status, or any receipt change fails. |
| Schedule-D derive | Both transferred parent columns are finite for every person and align to every tax unit. | Completed late transfer plus tail replacements. | No. A residual would fail late transfer first and derive again by name. |
| QBI derive | All QBI detail outputs are finite; self-employment is finite wherever its source applies; every independent archived QBI identity holds. The declared surface includes the base self-employment rewrite and binds pre/post digests. Its exact receipt is recomputed and authenticated at every persisted and publication boundary. | PUF/source detail plus ACS/ASEC native self-employment. Raw under-15 ACS `SEMP` remains structurally blank; mapped `self_employment_income_before_lsr` is a named, receipted universe zero. | No silent starvation. Every mapped ACS under-15 base value is held at its receipted universe zero across clone roles; all derived QBI cells remain in scope, and an in-universe null, forged receipt, or non-kernel output fails. |
| Take-up seed | Every administratively seeded variable completes; transfer-owned take-up cannot use a default; only explicitly non-transfer-owned inputs may use receipted engine defaults. | Seed kernels, the complete transfer surface, and declared defaults. | Transfer-owned residuals fail. A declared default is a separate modeled state, not an insufficient-support receipt. |
| SSI simulation projection | Every nullable engine input has a declared default on the disposable projection; the engine returns exactly one SSI value per person. | The persistent derived/seeded pool plus separately receipted ephemeral defaults. | A projection default can enable simulation but cannot cure the persistent pool; terminal evaluation returns to the original inputs plus SSI. |
| Simulated checkpoint pair and resume | The persistent input-only frame and temporary evaluation frame must share exact assembly provenance; SSI exists only on the evaluation half. The live QBI receipt must authenticate the persistent frame at emission, durable write/load, and resume. | Derived/seeded persistent inputs plus the separately materialized SSI evaluation output. | No. A forged QBI receipt, altered persistent value, invalid SSI binding, or mismatched pair invalidates the simulated checkpoint and falls back only to an independently valid earlier stage. |
| Terminal completeness | All 131 registered targets exist; every positive-weight value is metric-valid; a null needs exact source/role authority, and post-PUF targets forbid absence authority. | The 48 early targets, 70 late targets, derived leaves, take-up inputs, and SSI output. | No. Only the canonical group-quarters rent rule reaches this gate as null; base WAGP/SEMP leaves are outside the 131-target terminal surface. |
| By-origin battery | All 131 clone-0 comparison surfaces are complete and valid before support is measured. | The terminal simulation frame, comparing ASEC and ACS native origins. | No. `insufficient_support` is assigned only after null and validity checks, so it cannot hide an upstream missing value. |
| Manifest construction and canonical publication closure | Legacy and stacked builders reauthenticate QBI live output, canonical stacked authority, terminal-gate receipts, H5/diagnostics run IDs, and artifact digests before readiness can be asserted. | The validated persistent pool, immutable stage receipts, terminal gate snapshot, and atomically staged publication files. | No. Construction rejects forged or wrong-route receipts; publication begins with a non-ready tombstone, and only one fully authenticated run can replace it with a ready manifest. |

The audit leaves no generic “receipted but null” path into a hard consumer.
Structural absence is target- and universe-exact. Sample-size support affects
only whether a complete filing-status tail can attach and whether an otherwise
complete terminal comparison is testable.

### Retiring `--legacy-two-spine` sequence

The explicit compatibility flag preserves the previous assemble-first pool
path, including its publication bytes. It remains reproducible for lineage
comparison but is not the production default. That path runs this fixed
sequence:

1. `assemble_spines({"asec": ..., "acs": ...})` creates the first shared
   population state and binds the immutable assembly receipt.
2. The coarse `clone` stage first prepares the clone-sensitive source inputs
   on the assembled pool, then calls `clone_us_frame_for_puf_support(...)`.
   Preparation selects rows only by raw `PERIDNUM` availability and removes
   every `*_support_channel` and `*_support_clone_index` column from the
   historical kernel's ephemeral projection. This matters because assembly
   assigns clone index zero: exposing that metadata would make the legacy
   prior-year wrapper treat an unexpanded frame as support-cloned. CPS-carried
   inputs, direct measured hours-worked mappings, the adjacent-year
   prior-income join, relationships, the household rent draw, and parent-pointer
   eligibility inputs run in that order. Only their declared outputs are merged
   into the still-receipted pool. Physical cloning then copies those values and
   remaps structural IDs. Unavailable peer rows remain nullable; no operator
   reads source-channel identity.
3. The primary PUF QRF chain and capital-gains tail transfer run over the
   combined, physically cloned frame. Clone-index provenance, not source-spine
   identity, controls PUF-detail routing.
4. The post-clone source chain runs the prior-year PUF-support QRF, Medicare,
   pregnancy, WIC, housing-assistance support transfer, child support,
   disability benefits, workers compensation, weeks unemployed, childcare,
   adult care, energy subsidy, retirement contributions/distributions,
   immigration, and education inputs. The prior-year wage target is carried
   through cloning only long enough to train that QRF and is then deleted from
   the whole pool because it is formula-owned.
5. The pool-specific ACS transfer plan fills only still-null peer cells from
   those post-assembly results. Existing measured/native cells remain
   byte-for-byte unchanged, and transfer receipts record fitted and imputed
   rows. The three SCF/SIPP financial-asset leaves are excluded only from this
   pool-local transfer plan because none of the six pinned pool inputs is their
   donor. They remain hard release requirements and legacy ACS-transfer
   targets. The pool persists each as a typed all-null column with an explicit
   `with_us_scf_wealth_inputs` deferral receipt; the disposable SSI agreement
   view separately receipts the engine-default fills it needs for evaluation.
6. Schedule-D and QBI deterministic reconciliation run over that same pool.
7. The seed stage preserves existing take-up values, applies the sourced
   TANF and EITC mechanisms, and explicitly receipts live engine defaults
   used for unresolved, non-transfer-owned take-up inputs. Those defaults
   are not described as fitted or administrative mechanisms.
8. SSI is materialized only on an ephemeral agreement view in fixed
   household batches. Any engine defaults required solely for that
   calculation are separately receipted; formula output is not written into
   the input pool.
9. The spine-agreement gate is terminal. Its immutable pool registry is built
   from the complete pool-specific transfer plan plus derived, take-up, and SSI
   surfaces. Numeric columns retain the fixed incidence and conditional-
   quantile tolerances. Categorical columns use a fixed weighted total-
   variation-distance ceiling of `0.25`; the immigration fields are also
   checked jointly so matching marginals cannot conceal incompatible pairs.
   The gate batches all failures and controls the manifest's simulation-ready
   status.

On the retiring path, the 23-operator pool contract registry makes clone
placement total and executable. Adding an operator without a phase
declaration, or calling one in an undeclared phase, fails before the kernel
runs:

| Operator | Clone phase | Mechanism receipt |
|---|---|---|
| CPS-carried inputs | pre | Deterministic measured mappings supply rent and primary-QRF predictors. |
| Hours worked | pre | Direct ASEC mappings are computed once per source person and copied to support clones. |
| Prior-year income | pre and post | The `(source_year, PERIDNUM)` join must be unique before cloning; the PUF-only QRF requires clone roles afterward. |
| Relationships | pre | Household-head status is a rent-recipient predictor. |
| Medicare take-up | post | Rowwise carry/completion is clone-safe. |
| Housing inputs | pre | Rent is drawn once per source household and then cloned unchanged. |
| Eligibility inputs | pre | Raw `PH_SEQ`/`A_LINENO` parent pointers would count every cloned child twice. |
| Pregnancy | post | Stable source-identity hashes share draws across clones. |
| WIC | post | Remapped family grouping and source-identity draws are clone-safe. |
| Housing assistance | post | The QRF intentionally replaces only the PUF support role. |
| Child support | post | Role-aware QRF uses remapped structural IDs. |
| Disability benefits | post | Role-aware QRF uses remapped structural IDs. |
| Workers compensation | post | Role-aware QRF uses remapped structural IDs. |
| Weeks unemployed | post | Role-aware QRF uses remapped structural IDs. |
| Childcare | post | Role-aware QRF uses remapped SPM-unit IDs. |
| Adult care | post | Clone-local unit imputation requires support roles. |
| Energy subsidy | post | Role-aware QRF uses remapped SPM-unit IDs. |
| Retirement contributions | post | Role-aware QRF uses remapped structural IDs. |
| Retirement distributions | post | Forced PUF imputation requires support roles. |
| Immigration | post | Source-keyed draws preserve equality across clones. |
| Education | post | Deterministic rowwise derivation follows PUF tuition imputation. |
| Schedule-D completion | post | Transferred tax-unit parents exist only on the physically cloned pool. |
| QBI reconciliation | post | All-or-nothing identities reconcile the post-transfer PUF detail surface. |

The ASEC checkpoint remains the operator-untouched `raw_source_mapping`
artifact. Clone-stage preparation deliberately happens after loading,
operator-free validation, and assembly, so this ordering correction neither
changes nor requires reproduction of that checkpoint.

The output H5 is a nullable, input-only, pre-calibration pool. Its companion
manifest carries input pins, the stack/assembly receipts, per-source and
per-clone counts, operator receipts, and the complete terminal-gate result.
Publication first atomically replaces any prior manifest with a non-ready
tombstone, then
stages the H5 and diagnostics under one publication run ID and renames them,
and finally writes the readiness manifest. The manifest records the H5 and
diagnostics run IDs and SHA-256 digests. The readiness loader requires a green
manifest whose run ID and digests match the H5 metadata and diagnostics
payload. An interrupted, substituted, or failed publication therefore
self-reports not ready even beside stale files. A failed stacked gate writes
diagnostics and a non-ready final manifest and exits nonzero. Calibration is
deliberately absent; the downstream k-ladder may consume only a pool whose
terminal stacked battery passed.

## Provenance axes

The retired lineage used two related metadata schemes:

- PUF support cloning adds, on every entity,
  `*_source_id`, `*_support_channel`, and
  `*_support_clone_index`. The default support-channel values are `asec` and
  `puf_tax_detail`; clone index zero keeps the original IDs and later clones
  receive remapped IDs.
- Late ACS pooling adds `*_spine`, with `asec_puf` on the dense donor and
  `acs_2024_1yr` on ACS. When the donor already has complete support metadata,
  pooling synthesizes ACS support metadata with its native ID, channel
  `acs_2024_1yr`, and clone index zero.
- `transfer_acs_inputs` keeps fit and transfer provenance outside the frame,
  including the target family, donor spine/channel, predictors, seeds,
  weight kind, recipient patterns, and unmodeled-row count.

Those fields mixed two concepts: the population source that carried a record
and the PUF-detail copy created by an operator. The assembly seam keeps those
concepts separate.

## Canonical executable ordering

The production US multispine build order is:

```text
source ingestion and faithful schema harmonization
    -> uniformly sample both survey arms and assemble one stack
    -> prepare native predictors
    -> banked cross-origin gap-fill
    -> derived 36-node late producer DAG:
       PUF QRF plus clone-2 capital-gains tail
       -> interleaved source completion and 19 bounded transfer groups
       -> exact source finalization and transfer aggregation
    -> derive
    -> seed take-up and other stochastic inputs
    -> simulate
    -> completeness gate plus 131-target by-origin battery
    -> emit input-only pool, receipts, and terminal Logbook row
```

Calibration is a downstream consumer boundary, not a stage in this tool.

`assemble_spines(...)` is the boundary between source preparation and
population operators. It receives nullable, schema-compatible peer frames
and produces one combined frame before clone-stage preparation, physical
cloning, fitted transfer, seeded assignment, simulation, or calibration.
Downstream pool-stage entrypoints receive that combined frame and operate on
measured characteristics without selecting behavior by source spine. A
historical source kernel that requires CPS-only raw fields receives an
ephemeral availability projection after the combined-frame boundary is
validated. Such a projection is not published or described as a full-pool
lineage state; its declared outputs are merged back into the combined frame,
whose immutable assembly receipt remains the authority.

ASEC and ACS are peer household spines. A future household source can join
the same assembly contract. PUF tax detail is not a peer spine: it remains a
clone operator applied once after assembly and banked cross-origin gap-fill,
so every assembled household source reaches the PUF pass with the same
declared predictor surface.

The new provenance contract is:

- `*_support_channel` records source-spine provenance. Its vocabulary is the
  set of source names declared at assembly, such as `asec`, `acs`, and future
  peer channels.
- `*_spine_source_id` is the entity ID in the source frame before assembly
  remaps colliding ID spaces.
- `*_source_id` is the assembly-unique structural ID before cloning. Operator
  clones retain it so a source record and its copies remain one lineage even
  when two peer spines reused the same local ID.
- `*_support_clone_index` records operator-created copies. Index zero is the
  assembled source record; the PUF-detail clone is identified by its clone
  index rather than by changing the source channel.

`Frame.table()` exposes mutable pandas tables, so the provenance columns are
not made read-only by the dataframe API. Enforcement instead uses a private,
deeply frozen frame-metadata receipt. At assembly it records the declared
channel set and the native row count for every entity/channel pair. Assembly
output, PUF clone entry/output, and both stacked terminal gates validate live
channels and clone-index-zero row counts against that receipt. They also
require every person's channel to agree with each linked group row, including
its household. An unknown/forged channel, a drifted native count, a missing
receipt on an assembled frame, or a cross-grain mismatch raises `ValueError`
that names the assembly manifest. US runtime frame rebuilds carry the receipt
whenever they carry the source frame's mass log, and a structural test rejects
a mass-log-preserving rebuild that drops it.

Assembly, provenance reporting, gap-fill routing, completeness, and the
by-origin battery may read source channels. Population operators must not. In
particular, an operator may route PUF-detail behavior using clone provenance,
but it may not make a fit,
draw, transformation, or overwrite conditional on `asec`, `acs`, or another
source channel. The current unassembled lineage remains compatible: when the
raw-spine ID field is absent, its historical `asec`/`puf_tax_detail` channel
labels are validated and translated to clone roles centrally.

The structural guard covers person, household, tax-unit, SPM-unit, family,
marital-unit, and benefit-unit naming variants. It recognizes direct,
aliased-helper, dynamic-subscript, `getattr`, and `*_spine_source_id` reads.
Every US runtime module is classified as a reviewed population operator or an
explicit non-operator/provenance owner, so a new unclassified module fails the
guard. A second fail-closed scan starts at `build_us_multispine_pool.py` and
covers the tool plus its transitive US-runtime import graph under the same
owner registry. `transfer_acs_inputs` selects fit donors by the centrally
derived clone role; assembled source-channel names never determine donor
eligibility.

Assembly accepts only integer-typed, nonnegative structural source IDs and
names the source spine and offending IDs on failure. PUF cloning revalidates
integrality without truncating fractional values and accepts only the
canonical native/PUF-detail role pair.

## Source harmonization and geography boundary

“Assemble before operators” starts after the minimum work needed to represent
each source as a valid US `Frame`. File decoding, unit construction, ID
normalization, column naming, categorical decoding, and faithful mappings of
observed source fields are source ingestion or schema harmonization. They may
occur before assembly and may be source-specific.

That exception is narrow:

- Measured values remain untouched. A native mapping can carry an observed
  value into its canonical column, including its source provenance; it is
  not permission to fill, smooth, reconcile, or overwrite the measurement.
- A donor transfer, fitted imputation, deterministic synthetic allocation,
  seeded take-up assignment, or model simulation is a population operator
  and belongs after assembly.
- Observed geography such as an ACS PUMA can be normalized and preserved
  before assembly. Drawing missing PUMAs, congressional districts, counties,
  tracts, or other geography is an operator. It belongs after assembly and
  must condition on the available measured geography, not on the source
  channel.

This distinction leaves the exact source-to-schema adapters to their own
contracts while making the first shared mutable population state explicit.

## Gate and calibration boundary

The completeness gate and by-origin battery run after simulation and before
publication or calibration. Their authority bundle freezes the complete
declared surface, gap-fill plan, per-column metric registry, joint metrics,
and support profile; live content digests are recomputed at evaluation and
manifest emission. Each canonical result is also sealed to the exact evidence
snapshot minted by its evaluator, so even another internally valid canonical
receipt cannot be grafted onto it. Production entrypoints accept no
caller-supplied surface, metric, or tolerance authority.

Each declared comparison uses positive record weights and exactly one named
metric: boolean/rare incidence, monetary sign-separated incidence plus
conditional quantiles, or categorical total variation. Incidence ratios use
`[0.8, 1.25]`; conditional q10/q25/q50/q75/q90 envelopes and categorical TVD
use `0.25`. A one-sided supported hole fails. Small-rung comparisons below
the immutable effective-support floor are explicitly untestable and receipt
`insufficient_support`; they do not silently pass as tested and do not alter
the production tolerances.

The one canonical applicability exception is ACS group-quarters rent. Its
rule is part of the gap-fill-plan digest. Assembly separately binds the native
sampled TYPEHUGQ 2/3 household and person lineage counts and SHA-256 digests,
plus complete ordered native mappings of household support/raw/classification
and person support/raw/household-parent/classification identity. Each later
boundary proves the live native set and full native mapping, then expands those
mappings to every clone and proves exact person coverage of each clone's
receipted household selection. Clone roles are lifecycle-exact in every entity:
unattached `{0}`, full or partial PUF detail `{0, 1}`, or tail-descendant
`{0, 1, 2}` backed by its fully validated attachment receipt. The
`pre_subsidy_rent` null mask must equal those linked people across every clone
role and receipts those cells as recipient-exact structural absence.
Completeness proves the exact mask; the battery excludes only the same clone-0
cells before applying the unchanged support floor and tolerances.

Both gates return batched `GateResult` failures. Calibration cannot consume a
frame whose completeness or by-origin gate failed, and no source-specific
target, loss term, seed, inferred dtype metric, or caller threshold may shape
a passing result.

## Validation boundary

The production adoption is covered by synthetic module fixtures and a tiny
tool-entrypoint build, but this documentation does not claim a restricted-data
smoke or full production artifact. The tool never downloads a source dataset,
and calibration, exact-k selection, certification, and release promotion
remain downstream operations.
