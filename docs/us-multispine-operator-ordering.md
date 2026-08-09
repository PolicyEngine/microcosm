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
   Targets produced only by the later PUF pass or source-completion chain are
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
   This semantic change is authority-gated: the primary-QRF root and target
   checkpoint schema, outer stacked checkpoint materializer, and canonical
   stacked authority are all version 6. The outer base identity binds the
   primary-QRF schema plus the ACS universe and QBI reconciliation contract
   identities. Every v1--v5 payload is stale and refused, including the former
   strict v5 two-control payload.
5. The post-clone source-completion chain runs, then the declared post-PUF
   transfer fills the targets first materialized by that chain or the PUF pass.
   Its complete model donor is the ASEC-origin PUF-detail role. Authority is
   target-specific: every live positive-index clone must already observe a
   PUF-produced target, every ASEC-origin clone must already observe a
   source-produced target, and dual-produced targets require the union. A null
   on any such producer row is terminal; only the complementary recipient
   rows may be filled from QRF predictions. No blanket null-to-zero synthesis
   occurs, every producer cell stays byte-identical, and zero residual nulls
   are required.
6. The transferred checkpoint records the early gap-fill banks, post-PUF
   transfer bank, primary-QRF bank, tail manifest, weights audit,
   stack-manifest digest, fraction/seed, clone controls, and the channel-aware
   producer-precedence schedule. The same identity regime governs cold and
   resumed builds. Checkpoint emission, resume, and final publication each
   reject the post-PUF receipt unless it carries the exact canonical stacked
   authority; NON-CANONICAL test receipts cannot ship.
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
   At small rungs, comparisons outside the validity domain receipt
   `insufficient_support`; tolerances do not widen.
9. Only after both gates run does publication write the nullable H5,
   diagnostics, and readiness manifest. Success, failed gate, and exception
   paths each append a durable Logbook spool row beside the output, with the
   fraction token, seed, code/input/identity pins, phases, gate-receipt
   pointers, wall time, artifact location, and disposition.

### Downstream hard-completeness audit

This table makes the stacked 1% supplier and starvation behavior explicit at
every remaining boundary. An early `unmodeled_rows` receipt is merely an
accounting result; `insufficient_support` is a later battery status reached
only after a comparison surface is complete and valid.

| Boundary | Hard requirement | Stacked 1% supplier | Can an upstream insufficient-support/unmodeled state starve it? |
|---|---|---|---|
| Early gap-fill handoff | Donors observe every declared target; every recipient null is filled except the exact ACS group-quarters rent rule. Nonstructural `unmodeled_rows` and residual nulls are forbidden. | The pre-clone ASEC source operators supply 48 early targets to the two ASEC-to-ACS directions. | No accepted starvation remains. A nonstructural residual fails before cloning; literal `insufficient_support` is not an early-transfer outcome. |
| Clone attachment | Input rows are all clone 0; the seeded whole-household selection, lineage, pair weights, fraction, and seed agree exactly. | The completed gap-filled stack and the attachment sampler. | A permitted structural rent null is copied with its authority. Any other early residual has already failed. |
| PUF raw predictor sources | Every filing-status, count, and income component is observed in its declared source universe. Raw WAGP/SEMP authority is present and agrees with mapped leaves; a cross-grain source collision is rejected. A null on any eligible member fails before coercion. | Structure supplies status/count; ACS-native or ASEC-carried earnings supply earnings; early transfer supplies interest, dividends, and gains. | No. ACS under-15 WAGP/SEMP blanks are an exact source-universe state, not transfer starvation; all other source nulls fail. |
| PUF tax-unit features | Every clone-1 recipient has a finite feature vector. Post-aggregation NaN, `+inf`, and `-inf` are counted by named predictor and rejected before fitting; none is coerced or snapped to zero. | Universe-aware person sums plus tax-unit structural inputs. | No. Eligible member values must be complete; the only special case is an all-child unit whose numeric-zero predictor is explicitly owned and counted by the named universe-zero rule. |
| Primary QRF banks and chain | Donor/recipient banks are immutable; target order and RNG prefix are contiguous; all targets complete; live recipient identity, source-universe receipt, and feature digest match before finalization. | The processed full PUF donor and strict recipient checkpoint initialized above. | No. Mutation or missing receipt invalidates the bank; it cannot resume under legacy semantics. |
| Outer pool checkpoint identity and resume | Schema/materializer/authority v6 plus the ACS-universe and QBI-mutation contract identities must match exactly before any cached stage is discovered. | Fresh input pins, live stack receipt, scale controls, code identity, and both semantic contract identities. | No. Every v1--v5 root, target, materializer, or authority payload is stale; a self-consistent old receipt cannot reopen a checkpoint. |
| Clone-2 capital-gains tail | Candidate recipients have the required filing-status/AGI support, positive donor mass, unique household lineage, and sufficient weight capacity; every selected donor is assigned once. | Completed clone-1 QRF output and full PUF tail donors. | No early residual is accepted. Universe-aware PUF recipients remain eligible, including explicitly receipted empty-universe tax units. |
| Post-clone source completion | Each source operator preserves structure and emits its declared ASEC-evidenced outputs; unavailable peer cells remain null only until late transfer. | ASEC evidence rows plus completed PUF clone outputs. | Temporarily: peer nulls are intentional here, but the next zero-residual transfer must consume them. |
| Post-PUF transfer | Every declared PUF-clone or ASEC source-producer cell is nonnull; all complementary recipients are filled; the allowed count for both unmodeled and residual rows is zero. | Forty-three PUF and 30 source targets, with three overlaps, supply the 70-target late surface. | No. A missing producer or recipient value is terminal at this boundary. |
| Fit-weight audit | Every primary and post-PUF QRF fit receipts its resolved entity weight kind, and the collected fit records pass the weights audit before a transferred checkpoint can exist. | Calibrated household weights mapped by the frame to each modeled entity. | No. A missing, inconsistent, or manually substituted weight declaration fails before checkpoint emission. |
| Tail preservation | Tail manifest, descendants, IDs, weights, provenance, joint vector, and non-tail QRF cells remain exact after completion, transfer, derive, seed, and simulation. | The tail manifest bound during the PUF pass. | Completeness receipts cannot authorize a mutation; any byte or identity change fails. |
| Schedule-D derive | Both transferred parent columns are finite for every person and align to every tax unit. | Completed post-PUF transfer plus tail replacements. | No. A residual would fail late transfer first and derive again by name. |
| QBI derive | All QBI detail outputs are finite; self-employment is finite wherever its source applies; every independent archived QBI identity holds. The declared surface includes the base self-employment rewrite and binds pre/post digests. Its exact receipt is recomputed and authenticated at every persisted and publication boundary. | PUF/source detail plus ACS/ASEC native self-employment. Raw under-15 ACS `SEMP` remains structurally blank; mapped `self_employment_income_before_lsr` is a named, receipted universe zero. | No silent starvation. Every mapped ACS under-15 base value is held at its receipted universe zero across clone roles; all derived QBI cells remain in scope, and an in-universe null, forged receipt, or non-kernel output fails. |
| Take-up seed | Every administratively seeded variable completes; transfer-owned take-up cannot use a default; only explicitly non-transfer-owned inputs may use receipted engine defaults. | Seed kernels, the complete transfer surface, and declared defaults. | Transfer-owned residuals fail. A declared default is a separate modeled state, not an insufficient-support receipt. |
| SSI simulation projection | Every nullable engine input has a declared default on the disposable projection; the engine returns exactly one SSI value per person. | The persistent derived/seeded pool plus separately receipted ephemeral defaults. | A projection default can enable simulation but cannot cure the persistent pool; terminal evaluation returns to the original inputs plus SSI. |
| Simulated checkpoint pair and resume | The persistent input-only frame and temporary evaluation frame must share exact assembly provenance; SSI exists only on the evaluation half. The live QBI receipt must authenticate the persistent frame at emission, durable write/load, and resume. | Derived/seeded persistent inputs plus the separately materialized SSI evaluation output. | No. A forged QBI receipt, altered persistent value, invalid SSI binding, or mismatched pair invalidates the simulated checkpoint and falls back only to an independently valid earlier stage. |
| Terminal completeness | All 131 registered targets exist; every positive-weight value is metric-valid; a null needs exact source/role authority, and post-PUF targets forbid absence authority. | The 48 early targets, 70 late targets, derived leaves, take-up inputs, and SSI output. | No. Only the canonical group-quarters rent rule reaches this gate as null; base WAGP/SEMP leaves are outside the 131-target terminal surface. |
| By-origin battery | All 131 clone-0 comparison surfaces are complete and valid before support is measured. | The terminal simulation frame, comparing ASEC and ACS native origins. | No. `insufficient_support` is assigned only after null and validity checks, so it cannot hide an upstream missing value. |
| Manifest construction and canonical publication closure | Legacy and stacked builders reauthenticate QBI live output, canonical stacked authority, terminal-gate receipts, H5/diagnostics run IDs, and artifact digests before readiness can be asserted. | The validated persistent pool, immutable stage receipts, terminal gate snapshot, and atomically staged publication files. | No. Construction rejects forged or wrong-route receipts; publication begins with a non-ready tombstone, and only one fully authenticated run can replace it with a ready manifest. |

The audit leaves no generic “receipted but null” path into a hard consumer.
Structural absence is target- and universe-exact; sample-size support affects
only whether an otherwise complete terminal comparison is testable.

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
    -> one PUF QRF pass plus clone-2 capital-gains tail
    -> source completion
    -> banked post-PUF transfer of newly materialized targets
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
