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
  operator;
- the processed PUF H5 and source-year PUF CSV used by the existing donor
  loader;
- the canonical 2020-PUMA population-overlap ladder used by the post-assembly
  household-geography operator; and
- the packaged 117th-to-119th-Congress crosswalk that authenticates the
  ladder's current congressional-district target universe.

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

The exact outer `US_STACKED_POOL_OPERATOR_ORDER` is byte-stable and contains
these eleven entries. The numbered paragraphs below explain those phases; nested
callbacks are not additional outer entries.

```text
assemble_stacked_spine
assign_us_puma_ladder
prepare_multispine_source_inputs_for_clone
gap_fill_stacked_spine
run_stacked_late_producer_dag
prepare_stacked_tail_derivation
derive_multispine_pool_inputs
seed_multispine_pool_inputs
materialize_multispine_agreement_outputs
stacked_completeness_gate
by_origin_battery
```

1. `assemble_stacked_spine(...)` selects whole households independently from
   both survey arms with the single `sample_fraction` and `sample_seed`,
   restores each sample to its full-source design-weight mass, and assembles
   one origin-labeled frame. Standard rungs are `f001`, `f004`, `f010`, `f025`,
   and `f100`;
   the manifest binds the fraction, seed, exact realized ASEC/ACS counts,
   selected-lineage digests, the complete ordered native ACS household and
   person support/raw/household-parent/classification mappings, and the sampled
   native ACS TYPEHUGQ 2/3 household and person lineage digests. The full PUF
   remains a donor and is never sampled.
2. `assign_us_puma_ladder(...)` runs on that first shared frame, after both
   source-boundary checks and before source preparation. It authenticates the
   pinned 2020-PUMA population-overlap ladder and the packaged
   117th-to-119th-Congress crosswalk, preserves observed ACS PUMAs, and uses the
   ledgered `geography_legacy` seed stream to assign missing PUMAs plus household
   congressional districts and counties. Its ordered household/geography
   digest, positive-support counts, target universe, authority hashes,
   algorithm, and seed are bound into every stacked checkpoint and the terminal
   manifest; publication writes the crosswalk hash and `119th_congress` target
   vintage into the nullable H5 root attributes.
3. The spine-blind source-preparation chain derives the native predictors and
   pre-clone operator outputs needed by the early declared cross-origin fills.
   Historical kernels run on the raw-`PERIDNUM` CPS/ASEC availability
   projection and merge only their declared outputs back into the stack. In
   particular, the pinned ACS rent artifact trains `with_us_housing_inputs`,
   which materializes `pre_subsidy_rent` on ASEC; native ACS `RNTP`/`GRNTP`
   remain predictors and are not relabeled as that model input.
   Targets produced only by the later PUF pass or late source producers are
   excluded from this early authority surface. No population operator selects
   behavior from the source-channel labels.
4. `gap_fill_stacked_spine(...)` runs the two immutable directions over the
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
5. The derived second node of `run_stacked_late_producer_dag(...)` invokes
   `run_stacked_puf_pass(...)`; it is not a second outer operator-order entry.
   The callback attaches the separately controlled PUF clone arm
   (`clone_attachment_fraction`, default `1.0`) and runs one primary QRF pass
   across both survey origins. The strict recipient surface applies the
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
   The canonical stacked authority is version 12, the outer stacked checkpoint
   materializer uses version 13, and the stacked pool stage checkpoint
   materializer uses version 7.
   The outer base identity binds primary-QRF version 6, the ACS universe and
   QBI reconciliation contracts, the tail schema and support contract, and
   late-producer registry schema version 17, including execution-receipt
   contract 4, transition authority 2, and resource-semantics receipt 2. The
   primary execution config is version 5, its portable worker identity is
   version 1, and its checkpoint input sidecar is version 2. The companion pool
   manifest uses schema version 10.
   Older outer authority or materializer payloads are stale; primary-QRF
   version 6 remains current.

   The saved 10% failure checkpoint makes the ordering mechanism concrete.
   Before PUF it contains 385,992 clone-0 people: 342,732 ACS-origin and 43,260
   ASEC-origin. After clone attachment, the PUF-detail clone-1 cells for
   `sstb_self_employment_income_before_lsr` are finite while the corresponding
   clone-0 recipient cells remain null. The adult-care source projection is
   ASEC-scoped, so its strict earned-income accumulator encounters exactly the
   43,260 ASEC-origin clone-0 nulls and raises. It does not encounter 43,260
   ACS-origin rows; that origin description is contradicted by the checkpoint.
   The former driver called all post-clone source completion before the late
   transfer. The declared batch-5-to-adult-care edge below therefore derives
   the repair from data dependency rather than installing another manual
   ordering exception.
6. One declared late-producer DAG schedules the ACS earnings-universe
   materializer, primary PUF/tail pass, all 16 post-clone source operators,
   their once-only finalizer, and all 19 bounded transfer groups. Each node
   declares every effective input and output. A callback cannot run until each
   input is filled on its required scope or has the exact counted
   declared-absence receipt which that input contract tolerates. Import
   validation rejects unknown producers, ambiguous ownership, uncovered
   targets, and cycles, naming a deterministic cycle path. Lexical Kahn waves
   make the order independent of registry iteration. The universe node is the
   unique first wave, the primary PUF pass is the unique second wave, and later
   waves interleave source and transfer work. In particular, PUF batch 5 transfers
   `sstb_self_employment_income_before_lsr` before adult care consumes it, PUF
   batch 2 transfers `qualified_tuition_expenses` before education consumes it,
   pregnancy precedes WIC, and childcare precedes adult care. This order is
   derived from producer/input edges, never imposed as a second hand-written
   list. Thus the ACS structural-zero rule is also scheduled from declared
   inputs rather than hidden inside the primary callback.

   The complete model donor is the ASEC-origin PUF-detail role. Authority is
   target-specific: every live positive-index clone must already observe a
   PUF-produced target, every ASEC-origin clone must already observe a
   source-produced target, and dual-produced targets require the union. A null
   on any producer row is terminal; only complementary recipient rows may be
   filled from QRF predictions. No blanket null-to-zero synthesis occurs,
   every producer cell stays byte-identical, and zero residual nulls are
   required.
7. The transferred checkpoint records the early gap-fill banks, 19 distinct
   late-transfer banks, the primary-QRF bank, the complete 38-node DAG receipt,
   tail manifest and its per-status support receipt, weights audit,
   stack-manifest digest, fraction/seed, clone controls, and the channel-aware
   producer-precedence schedule. The DAG receipt binds all 71 edges, all input
   inventories, six derived waves, exact execution rows, the once-only source
   finalizer, and the 19-group/70-target aggregate. Every row hashes the live
   content of every declared alternative and output, the callback receipt, and
   the preceding row. The top receipt hashes the entry/output frames and chain
   terminus. Its digest is anchored in immutable Frame metadata and carried
   independently through checkpoints and publication. The same identity
   regime governs cold and resumed builds. Checkpoint emission, resume, and
   final publication reject a missing, stale, or reissued authority;
   NON-CANONICAL test receipts cannot ship.

   Outer materializer v10 also embeds one signed resource-semantics row for
   every DAG producer. Static configs are exact; donor tables are bound by the
   declared canonical scalar-content codec; primary and transfer banks name
   their outer/stage identity derivations; and source receipts name their
   callback-receipt digest derivation. The resource keys must exactly equal the
   producer's virtual input surface. Registry iteration cannot change the
   bytes, and checkpoint discovery positively accepts the current identity but
   rejects a stale source asset/config even when every engine and sampling pin
   is otherwise identical.
8. Schedule-D preparation, deterministic derivation, seeded inputs, and
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
9. A fresh `us_stacked_completeness` gate proves every declared input is
   observed or has exact source-by-role absence authority. The terminal
   `us_by_origin_battery` then evaluates all 134 declared targets (114 person,
   12 tax-unit, 8 SPM-unit), plus joint immigration structure, using an
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
10. Only after both gates run does publication write the nullable H5,
   diagnostics, and readiness manifest. Success, failed gate, and exception
   paths each append a durable Logbook spool row beside the output, with the
   fraction token, seed, code/input/identity pins, phases, gate-receipt
   pointers, wall time, artifact location, and disposition.

### Late producer/input DAG

The late stage is a declared producer/input graph, not a fixed source loop
followed by a fixed transfer loop. Its registry contains 38 producers: the ACS
earnings-universe producer, the primary PUF/tail producer, 16 post-clone source
producers, their explicit once-only finalizer, and 19 bounded late-transfer
producers. Import derives
and validates the schedule. Unknown
producers, duplicate ownership, uncovered transfer targets, and cycles fail at
import; a cycle error prints its deterministic cycle path. Readiness is checked
again immediately before each callback. Every required input must be nonnull
on its declared scope, finite when marked numeric, or carry one of that input's
explicitly tolerated counted-absence receipts. A receipt tolerated by one
input does not authorize another input. The readiness fence never converts a
missing value to zero; the separately declared ACS universe producer is the
only named structural-zero materializer. Execution receipts do not trust their
summarized readiness counts:
validation recomputes them from the exact physical-alternative evidence,
requires the exact kind-specific input/output schema, and rejects a completed
producer whose declared output is absent.

The notation below is executable-contract shorthand. `p`, `tu`, `s`, and `h`
mean person, tax unit, SPM unit, and household. `F(x)` requires numeric finite
values; `+` is an all-of alternative; `|` separates alternatives; and `?R`
means that only the named, counted absence receipt may replace that optional
input. `@weight` is the Frame-resolved entity weight and `@sidecar` or `@bank`
is an authenticated resource receipt, not a physical column.

The first producer, `acs_pums_earnings_universe`, has this complete ten-row
ACS-scoped inventory:

```text
F(p.age)
p.person_support_channel
F(p.person_support_clone_index)
p.person_tax_unit_id
p.person_source_id | p.person_id
p.WAGP present
p.SEMP present
F(p.employment_income_before_lsr) ?R
F(p.self_employment_income_before_lsr) ?R
p.@acs_pums_earnings_universe_execution_config
```

The two optional mapped numeric rows tolerate only their producer- and
requirement-specific `optional_input:acs_pums_earnings_universe:*` receipts.
The two raw columns must exist, but their legitimate structural nulls remain
part of the source authority. The execution config binds its runtime owner,
the ordered raw-to-mapped column pairs, ACS-only scope, and the complete
universe-rule identity. The producer leaves raw
`WAGP`/`SEMP` untouched, materializes mapped zero only for the declared
under-15 structural universe, and emits both mapped earnings columns plus
`frame.@acs_pums_earnings_universe_application`. Primary PUF consumes all
three outputs directly, making the universe-to-primary edge unavoidable.

The primary PUF producer has 114 external logical requirements: the following
83-input QRF/tail bundle `Q83`, plus the 31-item validation bundle `V0` below.
`Q83` consists of 17 required core rows, the optional finite
`is_full_time_college_student` tuition fallback, all 56 optional person-output
allocation bases, and all nine optional tax-unit passthroughs. Each optional
row can be absent only under its own counted receipt; a present value must be
finite. `V0` is the common 32-item late-transfer validation bundle `V` with
only the post-PUF clone-attachment manifest removed, because primary PUF
creates that manifest.
The raw ACS `WAGP`/`SEMP` authority, the two ACS-scoped mapped-earnings outputs,
and the application receipt add five direct dependencies, giving the executable
primary contract exactly 119 inputs.

```text
F(p.age)
filing status = tu.filing_status_input
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
F(p.is_full_time_college_student) ?R
tu.@puf_donor_tax_units
tu.@primary_qrf_checkpoint
tu.@primary_puf_execution_config
```

The 56 optional finite person allocation bases are, in canonical order:

```text
employment_income_before_lsr, self_employment_income_before_lsr,
taxable_interest_income, qualified_dividend_income,
non_qualified_dividend_income, tax_exempt_interest_income,
short_term_capital_gains, long_term_capital_gains_before_response,
long_term_capital_gains_on_collectibles, non_sch_d_capital_gains,
taxable_private_pension_income, taxable_ira_distributions,
social_security_retirement, social_security_disability,
social_security_dependents, social_security_survivors, alimony_income,
alimony_expense, salt_refund_income, charitable_cash_donations,
charitable_non_cash_donations, real_estate_taxes, home_mortgage_interest,
investment_interest_expense, investment_income_elected_form_4952,
student_loan_interest, educator_expense, qualified_tuition_expenses,
casualty_loss, unreimbursed_business_employee_expenses,
traditional_ira_contributions_desired,
self_employed_pension_contributions_desired, rental_income, estate_income,
farm_income, farm_operations_income, farm_rent_income, miscellaneous_income,
partnership_income, s_corp_income,
partnership_self_employment_net_earnings,
estate_income_would_be_qualified,
farm_operations_income_would_be_qualified,
farm_rent_income_would_be_qualified,
partnership_s_corp_income_would_be_qualified,
rental_income_would_be_qualified,
self_employment_income_would_be_qualified,
sstb_self_employment_income_would_be_qualified, business_is_sstb,
qualified_bdc_income, qualified_reit_and_ptp_income,
sstb_self_employment_income_before_lsr,
sstb_unadjusted_basis_qualified_property,
sstb_w2_wages_from_qualified_business,
unadjusted_basis_qualified_property, w2_wages_from_qualified_business
```

The nine optional finite tax-unit passthroughs are:

```text
domestic_production_ald, unrecaptured_section_1250_gain,
first_home_mortgage_balance, second_home_mortgage_balance,
first_home_mortgage_interest, second_home_mortgage_interest,
first_home_mortgage_origination_year,
second_home_mortgage_origination_year, health_savings_account_ald
```

```text
V0 = support channel + F(clone index) on p, h, tu, s, family, marital_unit
   + F(p.person_id)
   + F(p.person_household_id) + F(p.person_tax_unit_id)
   + F(p.person_spm_unit_id) + F(p.person_family_id)
   + F(p.person_marital_unit_id)
   + F(h.household_id) + F(tu.tax_unit_id) + F(s.spm_unit_id)
   + F(family.family_id) + F(marital_unit.marital_unit_id)
   + F(p.person_spine_source_id) + F(p.person_source_id)
   + F(h.household_spine_source_id) + F(h.household_source_id)
   + F(h.TYPEHUGQ) + h.@weight
   + frame.@us_spine_assembly_manifest
   + frame.@us_stacked_spine_manifest
```

Those are 28 physical provenance/structure columns, one resolved household
weight, and two metadata receipts. Primary PUF declares the same structural
surface, all six resolved-weight resources, 65 PUF/tail columns, and the clone
attachment manifest as outputs, so downstream dependencies are ownership
edges rather than incidental observations.

The three primary virtual resources are semantic, not row-count assertions.
The donor receipt hashes canonical typed scalar content, ordered columns, and
dtypes. The checkpoint receipt binds the outer routed identity, cache mode,
primary-QRF schema, manifest name, and exact target order; the physical
checkpoint directory basename must independently equal that bound identity.
The execution-config receipt resolves and hashes the actual predictor/output
sequences, all 65 optional allocation/passthrough reads, clone fraction/seed,
QRF seed/estimator count, worker module/interpreter/argv and reviewed fit
environment, strict-recipient and null-preserving doctrines, the once-resolved
aggregate-disaggregation spec and SOI AGI-band bytes/semantics, every tail
selection/topcode/five-times/concentration control, and enabled audit sinks.
The same three receipts form an exact SHA-bound sidecar beside the primary-QRF
manifest; resume refuses a missing or different sidecar, including a
same-row-count donor with changed bytes. This closes stale-bank reuse under a
newly claimed outer route.

#### Portable primary-QRF `worker_execution` identity

Worker identity schema v1 separates semantic authentication from launcher
aliases. `semantic_identity` binds the interpreter-launcher bytes, the exact
loaded Python runtime library (or the executable for a static build), and a
digest of the source and extension bytes for stdlib modules observed during or
present after a clean worker import. It also binds implementation, version and
ABI, cache tag,
canonical `pyvenv.cfg` fields, worker-module source, the statically resolved
Python closure plus every Microcosm namespace file opened by that clean import,
the exact approved `uv.lock`, installed-distribution/RECORD and combined
transitive environment/code digests, arguments after `argv[0]` with `argv[0]`
replaced by `{python_interpreter}`, and configured/resolved fit-job and
prediction-worker controls. `TORCH_DEVICE_BACKEND_AUTOLOAD=0` is a forced,
bound worker-bootstrap override applied before the QRF runtime can import
Torch. Before the clean import, identity construction enumerates
all installed `torch.backends` entry-point declarations, refuses duplicate
distribution identities or providers outside the selected RECORD closure, and
binds declarations belonging to selected distributions. Authentication
compares that payload and its
`semantic_identity_sha256`, so byte-identical interpreters reached through
different worktrees remain the same worker. `audit_aliases` records absolute
`sys_executable`, `sys_prefix`, and raw `argv_template_0`, but those aliases are
never compared for authentication.

A schema-9 gate-failed pool receives no implicit alias exception. Its only
relocation path is the scoring-only loader with an operator-supplied
compatibility-attestation JSON binding the sealed manifest and H5 SHA-256s, the
exact plan-published campaign-tree token `b8819b3f`, campaign `uv.lock` SHA-256
`27f47e385cfa35e2644a37410d1804b361ad9aee123577551c8421547bda65ee`,
installed transitive environment/code digest, recorded worker binding,
semantic identity, exact permitted mismatch set
`["argv_template[0]", "interpreter.executable"]`, and
`purpose: scoring_only`. Every semantic field must still equal the live
worker. The attestation's `plan_signature` is an exact plan-defined
authorization tuple (`gate`, `plan_sha256`, `prompt_sha256`,
`checklist_sha256`, and `evidence_sha256`) checked as data; this boundary does
not claim public-key or cryptographic signature verification.

Current manifests, diagnostics, `release_manifest.json`, and scoring receipts
surface `worker_execution_authentication`: manifest, execution-config, and
worker schema versions, `semantic_identity_sha256`, and audit aliases.
Attested legacy scoring additionally surfaces
`compatibility_attestation_sha256` and `purpose: scoring_only`. Schema-9 or
scoring-only compatibility evidence cannot enter a simulation-ready or release
receipt. The scoring loader may authenticate a deny-listed pool for
diagnostics, but candidate-26 remains denied for release independently of this
relocation check.

This F1 identity still assumes the interpreter's broader inherited startup path
is trusted. It now closes Torch backend autoloading, loaded-runtime and imported
stdlib bytes, and observed Microcosm import-time files, but does not additionally
authenticate `PYTHONPATH`, executable `.pth` startup hooks, or
`sitecustomize`/`usercustomize`. Hardening those remaining interpreter-startup
mechanisms would change the worker launch contract and is outside this
portability repair.

Every one of the 16 source producers consumes the following 16-requirement
wrapper bundle `W`. It is added to the operator-specific kernel inventory in
the table below, even where a kernel requirement names the same physical
column again. The schema-v3 execution config names the operator and binds the
post-clone phase, complete operator registry and contract, declared output
family and formula-owned removals, seed `0`; period `2024`, except housing's
`None`; `force_puf_imputation=True` only for retirement distributions; strict
existing-surface policy; housing QRF controls; and explicit `not_supplied` mode
for the education and weeks-unemployed sidecar arguments. For the 15
manifest-backed kernels it also hashes the exact packaged `source_stages.json`
bytes, resolved `SourceStageSpec`, and live resolver module/callable, refusing
runtime-helper drift. Thus no callback control or unreachable sidecar
alternative sits outside the registry:

```text
W = p.@post_clone_source_execution_config
  + frame.@us_spine_assembly_manifest + p.PERIDNUM
  + F(p.person_support_clone_index) + h.@weight
  + F(p.person_id) + F(p.person_household_id) + F(p.person_tax_unit_id)
  + F(p.person_spm_unit_id) + F(p.person_family_id)
  + F(p.person_marital_unit_id)
  + F(h.household_id) + F(tu.tax_unit_id) + F(s.spm_unit_id)
  + F(family.family_id) + F(marital_unit.marital_unit_id)
```

The common role-aware kernel bundle `C`, used by the source rows marked with
`C`, is:

```text
F(p.person_id); p.@weight; p.person_support_channel;
F(p.person_support_clone_index) ?R;
F(p.age) | F(p.A_AGE);
F(p.is_male) | F(p.is_female) | F(p.A_SEX);
F(p.has_esi); F(p.person_tax_unit_id); p.tax_unit_role_input;
F(p.employment_income_before_lsr) | F(p.WSAL_VAL);
F(p.self_employment_income_before_lsr) | F(p.SEMP_VAL);
[F(p.social_security_retirement) + F(p.social_security_disability)
 + F(p.social_security_survivors) + F(p.social_security_dependents)]
 | F(p.SS_VAL);
F(tu.tax_unit_id); tu.filing_status_input | tu.filing_status
```

The table gives every kernel input in addition to `W`; `C + ...` expands
exactly to the kernel bundle above. All raw CPS codes and amounts shown in the
table carry `F(...)` finite-numeric semantics unless they are explicitly
domain-checked booleans or strings. An optional receipt is not permission to
excuse a present invalid raw value.

| Post-clone source producer | Complete effective kernel input set |
|---|---|
| `impute_us_housing_assistance_to_puf_support` | `C + p.person_spm_unit_id + s.spm_unit_id + F(s.receives_housing_assistance) + F(s.takes_up_housing_assistance_if_eligible) + s.spm_unit_support_channel + s.spm_unit_support_clone_index ?R` |
| `with_us_adult_care_inputs` | `F(p.age) + F(p.employment_income_before_lsr) + F(p.self_employment_income_before_lsr) + F(p.sstb_self_employment_income_before_lsr) + F(p.PEDISDRS) + F(p.is_full_time_college_student) + p.tax_unit_role_input + F(p.person_tax_unit_id) + F(p.person_spm_unit_id) + F(p.person_id) + [p.person_support_channel + F(p.person_support_clone_index)] + F(s.spm_unit_pre_subsidy_childcare_expenses) + F(s.spm_unit_id) + F(tu.tax_unit_id) + p.@weight + s.@weight + tu.@weight` |
| `with_us_child_support_inputs` | `C + p.CSP_VAL + p.CHSP_VAL` |
| `with_us_childcare_inputs` | `C + F(p.person_spm_unit_id) + F(p.SPM_CHILDCAREXPNS) + s.spm_unit_id` |
| `with_us_disability_benefits` | `C + p.DIS_VAL1 + p.DIS_SC1 + p.DIS_VAL2 + p.DIS_SC2` |
| `with_us_education_inputs` | `F(p.ED_VAL) + F(p.qualified_tuition_expenses) + p.person_id + p.@weight` |
| `with_us_energy_subsidy_input` | `C + F(p.person_spm_unit_id) + F(p.SPM_ENGVAL) + s.spm_unit_id` |
| `with_us_immigration_inputs` | `p.PRCITSHP + p.PEINUSYR + p.PENATVTY + p.A_AGE + p.A_MARITL + p.A_SPOUSE + p.A_HSCOL + p.WSAL_VAL + p.SEMP_VAL + p.MCARE + p.CAID + p.IHSFLG + p.CHAMPVA + p.MIL + p.PEN_SC1 + p.PEN_SC2 + p.RESNSS1 + p.RESNSS2 + p.SS_YN + p.SSI_YN + p.PEIO1COW + p.A_MJOCC + p.PEAFEVER + p.SPM_CAPHOUSESUB + p.person_id + p.@weight + ([p.source_year + p.source_person_id] | p.person_id)` |
| `with_us_medicare_take_up_input` | `p.MCARE + p.person_id + p.@weight` |
| `with_us_pregnancy_inputs` | `p.A_SEX + p.A_AGE + p.person_id + p.@weight + ([p.source_year + p.source_household_id + p.source_person_id] | p.person_id)` |
| `with_us_prior_year_income_inputs` | `C + F(p.source_year) + F(p.PERIDNUM) + F(p.WSAL_VAL) + F(p.SEMP_VAL) + F(p.I_ERNVAL) + F(p.I_SEVAL) + F(p.employment_income_last_year) + F(p.self_employment_income_last_year)` |
| `with_us_retirement_contribution_inputs` | `C + p.RETCB_VAL + p.WSAL_VAL + p.SEMP_VAL` |
| `with_us_retirement_distribution_inputs` | `C + p.DST_SC1 + p.DST_VAL1 + p.DST_SC2 + p.DST_VAL2 + p.DST_SC1_YNG + p.DST_VAL1_YNG + p.DST_SC2_YNG + p.DST_VAL2_YNG + p.taxable_ira_distributions` |
| `with_us_weeks_unemployed` | `F(p.source_year) + F(p.PERIDNUM) + F(p.LKWEEKS) + (F(p.age) | F(p.A_AGE)) + (F(p.is_male) | F(p.is_female) | F(p.A_SEX)) + (F(p.tax_unit_is_joint) | [p.person_tax_unit_id + tu.tax_unit_id + tu.filing_status_input] | [p.person_tax_unit_id + tu.tax_unit_id + tu.filing_status]) + (p.tax_unit_role_input | [F(p.is_tax_unit_head) + F(p.is_tax_unit_spouse) + F(p.is_tax_unit_dependent)]) + (F(p.unemployment_compensation) | F(p.UC_VAL)) ?R + p.person_support_channel + p.@weight` |
| `with_us_wic_claim_input` | `F(p.age) + F(p.is_female) + F(p.is_pregnant) + F(p.own_children_in_household) + F(p.person_family_id) + p.@weight + ([p.source_year + p.source_household_id + p.source_person_id] | p.person_support_source_id | p.person_id)` |
| `with_us_workers_compensation` | `C + p.WC_VAL` |

The table plus `W` is the complete external logical inventory. The executable
contract additionally carries direct evidence from every declared producer of
an inventory column, plus the cross-source dependencies named in the edge
table. These counts make that expansion auditable:

| Source producer | External inventory rows | Executable contract inputs |
|---|---:|---:|
| `impute_us_housing_assistance_to_puf_support` | 36 | 59 |
| `with_us_adult_care_inputs` | 33 | 54 |
| `with_us_child_support_inputs` | 32 | 53 |
| `with_us_childcare_inputs` | 33 | 54 |
| `with_us_disability_benefits` | 34 | 55 |
| `with_us_education_inputs` | 20 | 35 |
| `with_us_energy_subsidy_input` | 33 | 54 |
| `with_us_immigration_inputs` | 43 | 57 |
| `with_us_medicare_take_up_input` | 19 | 33 |
| `with_us_pregnancy_inputs` | 21 | 35 |
| `with_us_prior_year_income_inputs` | 38 | 59 |
| `with_us_retirement_contribution_inputs` | 33 | 54 |
| `with_us_retirement_distribution_inputs` | 39 | 61 |
| `with_us_weeks_unemployed` | 26 | 41 |
| `with_us_wic_claim_input` | 23 | 38 |
| `with_us_workers_compensation` | 31 | 52 |

The 17th source-side node is the explicit `source_finalizer`. Its complete
17-input set is the 16 virtual resources
`p.@source_receipt:<operator>`, one for every table row above, plus
`p.@source_finalizer_execution_config`. Each source resource hashes the exact
corresponding callback receipt. The schema-v2 finalizer config binds the phase,
source registry, formula-owned exclusions, complete deferred-input declarations,
and deferred status. Only after all 17 exist may the finalizer materialize the
three deliberately deferred SCF columns
`bank_account_assets`, `bond_assets`, and `stock_assets` with their declared
absence receipts. This makes finalization a DAG node rather than a hidden
mutation after the schedule.

Every transfer consumes the 46-row external logical inventory `V + T(E)` plus
direct producer evidence for every primary/source-owned physical input and
target in the next table. The adult-care transfer alone adds the required
47th logical row `p.tax_unit_role_input`, consumed by its deterministic
post-fit reconciliation. `V` is the exact common validation surface: 28
physical columns, the resolved household weight, and three immutable metadata
receipts.

```text
V = support channel + F(clone index) on p, h, tu, s, family, marital_unit
  + F(p.person_id)
  + F(p.person_household_id) + F(p.person_tax_unit_id)
  + F(p.person_spm_unit_id) + F(p.person_family_id)
  + F(p.person_marital_unit_id)
  + F(h.household_id) + F(tu.tax_unit_id) + F(s.spm_unit_id)
  + F(family.family_id) + F(marital_unit.marital_unit_id)
  + F(p.person_spine_source_id) + F(p.person_source_id)
  + F(h.household_spine_source_id) + F(h.household_source_id)
  + F(h.TYPEHUGQ) + h.@weight
  + frame.@us_spine_assembly_manifest
  + frame.@us_stacked_spine_manifest
  + frame.@us_puf_clone_attachment_manifest
```

For a transfer whose target entity is `E`, the complete 14-requirement model
and weight bundle `T(E)` is:

```text
T(E) = F(p.age) + F(p.is_female) + p.@weight + E.@weight
     + E.@late_transfer_model_config + E.@late_transfer_target_bank
     + [F(p.state_fips)
        | (F(p.person_household_id) + F(h.household_id) + F(h.state_fips))]
     + F(p.employment_income_before_lsr) ?R
     + F(p.self_employment_income_before_lsr) ?R
     + [(F(p.social_security_retirement)
         + F(p.social_security_disability)
         + F(p.social_security_dependents)
         + F(p.social_security_survivors))
        | F(p.acs_social_security_income)] ?R
     + [(F(p.taxable_private_pension_income)
         + F(p.tax_exempt_private_pension_income)
         + F(p.taxable_ira_distributions))
        | F(p.acs_retirement_income)] ?R
     + [(F(p.taxable_interest_income) + F(p.tax_exempt_interest_income)
         + F(p.qualified_dividend_income)
         + F(p.non_qualified_dividend_income) + F(p.rental_income)
         + F(p.estate_income))
        | F(p.acs_interest_dividend_rental_income)] ?R
     + (F(p.is_household_head) | F(p.RELSHIPP) | F(p.A_EXPRRP)
        | F(p.A_LINENO)) ?R
     + (p.tenure_type | s.spm_unit_tenure_type | F(h.TEN)
        | F(h.H_TENURE)) ?R
```

Every one of the 19 transfer nodes also requires direct primary-PUF evidence
for every primary-owned physical alternative in `V + T(E)`, including
`p.tax_exempt_interest_income` and `p.estate_income`, plus direct producer
evidence for every target listed below. A target shown in both producer columns
requires both scopes; that is the two-target PUF/source overlap. Thus the table
is the complete per-node producer delta over `V + T(E)`, as well as the exact
70-target partition. Transfer rows abbreviate the registry's leading
`transfer:`; source names in these tables abbreviate the leading `source:`.

For every transfer, schema-v3 `@late_transfer_model_config` binds that node's
name, entity, family, ordered targets, seed, estimator count, canonical maximum
targets per fit, required/optional predictor order, combined-feature mappings,
target codecs, housing-head and tenure precedence/codes, immigration codec,
and deterministic post-fit structure. Bounded DAG groups explicitly disable
the opportunistic Schedule-D derivation; the later whole-pool tax-unit derive
operator is its sole owner. Adult care keeps its declared reconciliation and
therefore gates on `tax_unit_role_input`. `@late_transfer_target_bank` binds
either the durable bank's
validated identity SHA-256 or the explicit `ephemeral_no_checkpoint` mode; a
non-null bank without an identity is rejected before dispatch. Each virtual
receipt has an exact kind-specific inner schema and its own SHA-256, and its
execution-row input evidence must hash the identical receipt. The source
finalizer applies the same rule to each of its sixteen source-receipt inputs.

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
| `person/source_operator_wic_claim` | `takes_up_wic_if_eligible` | — | from `with_us_wic_claim_input` |
| `person/source_operator_workers_compensation` | `workers_compensation` | — | from `with_us_workers_compensation` |
| `tax_unit/puf_tax_itemization` | `domestic_production_ald`, `unrecaptured_section_1250_gain`, `first_home_mortgage_balance`, `first_home_mortgage_interest`, `first_home_mortgage_origination_year`, `health_savings_account_ald` | all targets | — |
| `spm_unit/source_operator_energy_subsidy` | `spm_unit_energy_subsidy` | — | from `with_us_energy_subsidy_input` |

The resulting executable transfer contracts contain 92–100 inputs; 46 is the
common logical inventory, not the complete contract count:

| Transfer producer | Executable contract inputs |
|---|---:|
| `person/adult_care` | 94 |
| `person/model_required_boolean` | 92 |
| `person/puf_tax_itemization__batch_1` | 98 |
| `person/puf_tax_itemization__batch_2` | 100 |
| `person/puf_tax_itemization__batch_3` | 99 |
| `person/puf_tax_itemization__batch_4` | 99 |
| `person/puf_tax_itemization__batch_5` | 96 |
| `person/source_operator_child_support` | 93 |
| `person/source_operator_disability_benefits` | 92 |
| `person/source_operator_education_inputs` | 97 |
| `person/source_operator_immigration` | 93 |
| `person/source_operator_medicare_take_up` | 92 |
| `person/source_operator_retirement_contributions` | 94 |
| `person/source_operator_retirement_distributions` | 96 |
| `person/source_operator_weeks_unemployed` | 92 |
| `person/source_operator_wic_claim` | 92 |
| `person/source_operator_workers_compensation` | 92 |
| `spm_unit/source_operator_energy_subsidy` | 93 |
| `tax_unit/puf_tax_itemization` | 98 |

#### Complete dependency edges

The following grouped tables enumerate all 71 unique producer-to-consumer edges.
Multiple values on one row are the input reasons carried by that edge. Bare
source names carry the registry prefix `source:` and transfer paths carry
`transfer:`.

The first edge is `acs_pums_earnings_universe -> primary_puf_qrf`, carried by
the two ACS-scoped mapped earnings columns and the whole-frame universe
application receipt.

Every primary-PUF-to-source edge carries the same 14-item ownership base `S14`:

```text
S14 = p.person_support_clone_index + p.person_id
    + p.person_household_id + p.person_tax_unit_id + p.person_spm_unit_id
    + p.person_family_id + p.person_marital_unit_id
    + h.household_id + tu.tax_unit_id + s.spm_unit_id
    + family.family_id + marital_unit.marital_unit_id
    + p.@weight + h.@weight
```

The following table gives the complete additions to `S14` on each of the 16
primary-PUF-to-source edges:

| Consumer source | Additional inputs supplied by primary PUF |
|---|---|
| `impute_us_housing_assistance_to_puf_support` | person support channel; employment; self-employment; four Social Security components; SPM-unit support channel and clone index |
| `with_us_adult_care_inputs` | person support channel; employment; self-employment; SPM-unit and tax-unit weights |
| `with_us_child_support_inputs` | person support channel; employment; self-employment; four Social Security components |
| `with_us_childcare_inputs` | person support channel; employment; self-employment; four Social Security components |
| `with_us_disability_benefits` | person support channel; employment; self-employment; four Social Security components |
| `with_us_education_inputs` | — |
| `with_us_energy_subsidy_input` | person support channel; employment; self-employment; four Social Security components |
| `with_us_immigration_inputs` | — |
| `with_us_medicare_take_up_input` | — |
| `with_us_pregnancy_inputs` | — |
| `with_us_prior_year_income_inputs` | person support channel; employment; self-employment; four Social Security components |
| `with_us_retirement_contribution_inputs` | person support channel; employment; self-employment; four Social Security components |
| `with_us_retirement_distribution_inputs` | person support channel; employment; self-employment; four Social Security components; `taxable_ira_distributions` |
| `with_us_weeks_unemployed` | person support channel |
| `with_us_wic_claim_input` | — |
| `with_us_workers_compensation` | person support channel; employment; self-employment; four Social Security components |

There are also 19 primary-PUF-to-transfer edges: one to every row of the
transfer table above. `B(E)` is the exact set of primary-owned physical
alternatives in the published `V + T(E)` inventory plus the shared PUF-clone
predictors `tax_exempt_interest_income` and `estate_income`. `B(person)` has 45
inputs; `B(tax_unit)` and `B(spm_unit)` each have 46 because their target
weight is distinct from the person weight. Every primary-to-transfer edge
carries `B(E)` plus each PUF-owned target in that row which is not already in
`B(E)`. The transfer target table and its contract counts therefore enumerate
the complete input reasons for all 19 edges.

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
| `with_us_wic_claim_input` | `transfer:person/source_operator_wic_claim` | `takes_up_wic_if_eligible` |
| `with_us_workers_compensation` | `transfer:person/source_operator_workers_compensation` | `workers_compensation` |
| `transfer:person/puf_tax_itemization__batch_2` | `with_us_education_inputs` | `qualified_tuition_expenses` |
| `transfer:person/puf_tax_itemization__batch_5` | `with_us_adult_care_inputs` | `sstb_self_employment_income_before_lsr` |

Finally, there are 16 source-to-finalizer edges: each of the 16 source
producers in the source-input table has one edge to `source_finalizer`, carried
by its exact `p.@source_receipt:<operator>` resource. Thus the exhaustive count
is 1 universe-to-primary + 16 primary-to-source + 19 primary-to-transfer + 19
cross/source-to-transfer + 16 source-to-finalizer = 71.

The lexically canonical waves have sizes `(1, 1, 17, 14, 3, 2)`:

1. `acs_pums_earnings_universe`.
2. `primary_puf_qrf`.
3. Housing assistance; child support; childcare; disability; energy;
   immigration; Medicare; pregnancy; prior-year income; retirement
   contributions; retirement distributions; weeks unemployed; workers'
   compensation; person PUF batches 1, 4, and 5; tax-unit PUF transfer.
4. Adult care; WIC; pregnancy transfer; person PUF batches 2 and 3; child
   support, disability, immigration, Medicare, retirement-contribution,
   retirement-distribution, weeks-unemployed, workers'-compensation, and
   SPM-energy transfers.
5. Education; adult-care transfer; WIC transfer.
6. `source_finalizer` and education transfer.

Registry schema version 17 and execution-receipt contract version 4 bind the
canonical input declarations, outputs, edges, waves, exact kind-specific
virtual-resource bindings, content-hashed execution-row schema, and immutable
transition authority version 2. The schedule SHA-256 is
`e59c019d3d454eac99ac0ac209b6c5b6faaf9bdfcaeee18c36a25be19bf7da2f`;
the full payload SHA-256 is
`7be038d34f228d66c12b53558fc5f30c93f1b376f1058c5e4fd7e7563a88d67f`.
Reversing registry iteration produces those same bytes.

The virtual-resource payload ledger is independently versioned: ACS-universe
config v2, primary execution config v5, source execution config v3, source
finalizer config v2, and transfer model config v3. Donor content,
primary-checkpoint routing, source callback receipts, and transfer-bank routing
remain v1. The outer all-producer resource-semantics receipt is v2 and binds
both these static schemas and every dynamic derivation mode.

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
| Outer pool checkpoint identity and resume | Primary-QRF schema v6, primary execution config v5, portable worker identity v1, tail-manifest schema v2, late-registry schema v17/receipt contract v4, outer stacked materializer v13/authority v12, stacked pool-stage materializer v7, pool manifest schema v10, and the ACS-universe, QBI-mutation, tail-support, late-DAG, and signed virtual-resource-semantics identities must match exactly before any cached stage is discovered. The retiring legacy envelope remains manifest schema v4/materializer v3. | Fresh input pins, live stack receipt, scale controls, code identity, and all semantic contract identities. | No. An older stacked materializer or authority payload is stale; a self-consistent old receipt cannot reopen a checkpoint. Primary-QRF v6 remains current. |
| Clone-2 capital-gains tail | Each filing status requires as many eligible recipient households as selected q99.5 donors. Eligibility requires unique single-tax-unit PUF-detail lineage and half-weight capacity for the global maximum assigned donor weight. An adequate status assigns every selected donor once; a thin status skips as a whole with a named, counted `insufficient_support` receipt. | Completed clone-1 QRF output and full PUF tail donors. At 1%, `SINGLE` and `HEAD_OF_HOUSEHOLD` attach, `JOINT` and `SEPARATE` skip, and zero-requirement `SURVIVING_SPOUSE` is `not_applicable`. | No widening or partial attachment is permitted. All 22 AGI bands provide nearest-first fallback only inside a status. Universe-aware PUF recipients remain eligible, including explicitly receipted empty-universe tax units. |
| Late producer DAG | Before any callback, all declared inputs are filled on their required scopes or carry an input-specific counted absence receipt; numeric inputs are finite. The exact derived order, readiness rows, once-only source finalizer, and bounded transfer receipts must validate. | ACS earnings-universe materialization, primary PUF/tail, 16 source producers, and 19 bounded transfer groups execute in six derived waves. | No. The refusing producer names the unfilled input and its declared producing stage. A cycle fails at import with its path. |
| Late transfer completion | Every declared PUF-clone or ASEC source-producer cell is nonnull; all complementary recipients are filled; the allowed count for both unmodeled and residual rows is zero. | Forty-three PUF and 29 source targets, with two overlaps, supply the 70-target late surface. | No. A missing producer or recipient value is terminal at this boundary. |
| Fit-weight audit | Every primary and post-PUF QRF fit receipts its resolved entity weight kind, and the collected fit records pass the weights audit before a transferred checkpoint can exist. | Calibrated household weights mapped by the frame to each modeled entity. | No. A missing, inconsistent, or manually substituted weight declaration fails before checkpoint emission. |
| Tail preservation | Tail manifest, support decisions, attached descendants, IDs, weights, provenance, joint vector, and non-tail QRF cells remain exact after completion, transfer, derive, seed, and simulation. | The schema-v2 tail manifest and support receipt bound during the PUF pass and projected into both terminal gates. | A support receipt cannot authorize mutation. Any byte or identity change in an attached status, any descendant for a skipped status, or any receipt change fails. |
| Schedule-D derive | Both transferred parent columns are finite for every person and align to every tax unit. Bounded late-transfer groups do not write this leaf; the whole-pool tax-unit derive is its sole canonical owner. | Completed late transfer plus tail replacements. | No. A residual would fail late transfer first and derive again by name. |
| QBI derive | All QBI detail outputs are finite; self-employment is finite wherever its source applies; every independent archived QBI identity holds. The declared surface includes the base self-employment rewrite and binds pre/post digests. Its exact receipt is recomputed and authenticated at every persisted and publication boundary. | PUF/source detail plus ACS/ASEC native self-employment. Raw under-15 ACS `SEMP` remains structurally blank; mapped `self_employment_income_before_lsr` is a named, receipted universe zero. | No silent starvation. Every mapped ACS under-15 base value is held at its receipted universe zero across clone roles; all derived QBI cells remain in scope, and an in-universe null, forged receipt, or non-kernel output fails. |
| Take-up seed | Every administratively seeded variable completes; transfer-owned take-up cannot use a default; only explicitly non-transfer-owned inputs may use receipted engine defaults. | Seed kernels, the complete transfer surface, and declared defaults. | Transfer-owned residuals fail. A declared default is a separate modeled state, not an insufficient-support receipt. |
| SSI simulation projection | Every nullable engine input has a declared default on the disposable projection; the engine returns exactly one SSI value per person. | The persistent derived/seeded pool plus separately receipted ephemeral defaults. | A projection default can enable simulation but cannot cure the persistent pool; terminal evaluation returns to the original inputs plus SSI. |
| Simulated checkpoint pair and resume | The persistent input-only frame and temporary evaluation frame must share exact assembly provenance; SSI exists only on the evaluation half. The live QBI receipt must authenticate the persistent frame at emission, durable write/load, and resume. | Derived/seeded persistent inputs plus the separately materialized SSI evaluation output. | No. A forged QBI receipt, altered persistent value, invalid SSI binding, or mismatched pair invalidates the simulated checkpoint and falls back only to an independently valid earlier stage. |
| Terminal completeness | All 134 registered targets exist; every positive-weight value is metric-valid; a null needs exact source/role authority, and post-PUF targets forbid absence authority. | The 48 early targets, 70 late targets, derived leaves, take-up inputs, and SSI output. | No. Only the canonical group-quarters rent rule reaches this gate as null; base WAGP/SEMP leaves are outside the 134-target terminal surface. |
| By-origin battery | All 134 clone-0 comparison surfaces are complete and valid before support is measured. | The terminal simulation frame, comparing ASEC and ACS native origins. | No. `insufficient_support` is assigned only after null and validity checks, so it cannot hide an upstream missing value. |
| Manifest construction and canonical publication closure | Legacy and stacked builders reauthenticate QBI live output, canonical stacked authority, terminal-gate receipts, H5/diagnostics run IDs, and artifact digests before readiness can be asserted. | The validated persistent pool, immutable stage receipts, terminal gate snapshot, and atomically staged publication files. | No. Construction rejects forged or wrong-route receipts; publication begins with a non-ready tombstone, and only one fully authenticated run can replace it with a ready manifest. |

The audit leaves no generic “receipted but null” path into a hard consumer.
Structural absence is target- and universe-exact. Sample-size support affects
only whether a complete filing-status tail can attach and whether an otherwise
complete terminal comparison is testable.

### Retiring `--legacy-two-spine` sequence

The explicit compatibility flag preserves the previous assemble-first pool
path, including its publication bytes. It remains reproducible for lineage
comparison but is not the production default. That path runs this fixed
sequence under its preserved manifest schema 4/checkpoint materializer 3
identity. The loader uses the complete schema/envelope surface and rejects a
stacked artifact whose pipeline markers were stripped, so schema lowering
cannot bypass late-DAG validation:

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
   pool-local transfer plan because none of the six pinned model-data inputs is
   their donor; the two additional pinned inputs are geography authorities, not
   financial-asset donors. They remain hard release requirements and legacy
   ACS-transfer targets. The pool persists each as a typed all-null column with an explicit
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
    -> derived 38-node late producer DAG:
       ACS earnings-universe materialization
       -> PUF QRF plus clone-2 capital-gains tail
       -> interleaved source completion and 19 bounded transfer groups
       -> exact source finalization and transfer aggregation
    -> derive
    -> seed take-up and other stochastic inputs
    -> simulate
    -> completeness gate plus 134-target by-origin battery
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
