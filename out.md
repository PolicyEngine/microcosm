# Weeksgate: stacked release gates and fractional-weeks final report

Date: 2026-08-27

Branch: `stacked-release-gate-alignment`

Lane base: `4f453746` (`origin/main` at kickoff; includes #786)

## Outcome

The 369 fractional `weeks_unemployed` values are genuine pool-content
defects, but the proposed post-transfer-calibration mechanism is refuted. They
were emitted earlier by the ACS-transfer QRF because PolicyEngine-US physically
types this integer-supported variable as `float`. The calibration maps ACS
clone 0 onto actual observed ASEC support and removes all 342 fractions in that
scope; it never touches the 360 ACS clone-1 and 9 ACS clone-2 fractions that
survive into the pool.

The source fix registers `weeks_unemployed` as a discrete numeric ACS-transfer
target. Predictions now snap deterministically to actual donor support, with
lower-support tie breaking, and are returned on integer support. The transfer
execution contract and generated imputation authority receipt the policy.
The strict post-transfer calibration receipt and validator remain unchanged
because that kernel did not cause the defect; their complete contract suite
still passes.

The weeks release gate now distinguishes physical source channel from
clone-operator role. It derives an assembled roster (`asec+acs`) or retains a
legacy roster (`asec+puf_tax_detail`), validates raw `LKWEEKS` only where an
ASEC source exists, reconciles direct carries only on native ASEC rows, and
checks the UC rule only on rows owned by that constraint. Every plausibility
band and numeric threshold is unchanged.

The full release-side sweep fixed all unambiguous physical-source,
clone-layout, and stable-identity archaeology. Six archived-model predictor
contracts remain owner decisions and are reported precisely below. In
particular, ORG wages/FLSA is guaranteed to fail on the stacked pool, while
SIPP tips' global tipped-occupation component passes despite a dead ACS
channel. Those are not safe gate-only edits.

No network access, pool build, release build, artifact publication, push,
plausibility-band change, or by-origin battery change was performed. Issue
#782's weeks-incidence-band adjudication was not touched.

## Task 1: provenance of all 369 noninteger weeks

### Evidence read

The analysis read the supplied fixed-format pool directly:

`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/pool.h5`

It contains 1,970,973 person rows: 234,133 physical ASEC rows and 1,736,840
physical ACS rows. It also read the exact pre-calibration late-transfer target
bank:

`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/candidate-25/pool/checkpoints/stacked/3847d30e1488ae18891780a7a5de6a1b75d1329f2491796e35639fc9e802c26e/acs-transfer/0914d1b858dff5171f7743f0e535349be4459fb99b3386bbc2e3be72d40dd5fc/late_producer_dag/person/source_operator_weeks_unemployed/targets/000__weeks_unemployed.h5`

The raw-draw value SHA-256 is
`6ea20782bcb53fa730bb7e7045c7d79ab096dfe69bc5ceb540158d93f6672db4`.
The final fractional-row-index SHA-256 is
`3431befe89f007c353296b0c59792bc31e45a1031e62a6f9c99e003df286305f`;
the sorted 369-value SHA-256 is
`314ba4459bb57169eaf492dd28c8ea04af460104b962bb29285104e70eee976c`.

### Counts by physical channel, clone, and UC

| Physical channel | Clone index | All rows | Fractional, UC=0 | Fractional, UC>0 |
|---|---:|---:|---:|---:|
| ASEC | 0 | 108,073 | 0 | 0 |
| ASEC | 1 | 108,073 | 0 | 0 |
| ASEC | 2 | 17,987 | 0 | 0 |
| ACS | 0 | 856,626 | 0 | 0 |
| ACS | 1 | 856,626 | 355 | 5 |
| ACS | 2 | 23,588 | 9 | 0 |
| **Total** |  | **1,970,973** | **364** | **5** |

All 369 values are positive, have null ASEC `LKWEEKS`, and are distinct at
their exact IEEE-754 representation. Every fractional row maps one-to-one by
`person_source_id` to a unique ACS clone-0 sibling, and
`unemployment_compensation` is identical across that sibling group. Of the
364 UC-zero rows, the clone-0 sibling has zero weeks. The five UC-positive
rows have clone-0 integer weeks in `{2, 4, 40, 48, 50}`.

The five fractional value / annual-UC pairs are:

| Fractional weeks | Unemployment compensation |
|---:|---:|
| 3.3297787140375217 | 10,800 |
| 3.948788299342264 | 20,000 |
| 17.23699569220113 | 7,200 |
| 18.64447048256536 | 10,800 |
| 21.97077954067273 | 4,400 |

### Value distribution

| Statistic | Value |
|---|---:|
| Distinct values | 369 |
| Minimum | 1.0003521955067698 |
| p10 | 2.306610095577396 |
| p25 | 6.226837695984924 |
| Median | 12.319331262236448 |
| Mean | 14.643462963265243 |
| p75 | 23.618846787450813 |
| p90 | 27.706623369972394 |
| Maximum | 37.796501228614694 |

| Weeks interval | All | ACS clone 1, UC=0 | ACS clone 1, UC>0 | ACS clone 2, UC=0 |
|---|---:|---:|---:|---:|
| [1, 5) | 74 | 70 | 2 | 2 |
| [5, 10) | 85 | 84 | 0 | 1 |
| [10, 15) | 45 | 44 | 0 | 1 |
| [15, 20) | 40 | 36 | 2 | 2 |
| [20, 25) | 60 | 58 | 1 | 1 |
| [25, 30) | 39 | 39 | 0 | 0 |
| [30, 35) | 16 | 15 | 0 | 1 |
| [35, 40) | 10 | 9 | 0 | 1 |

The nine clone-2 values are
`1.3112775468533615`, `3.155651627690991`,
`8.163243626548894`, `13.03279351472897`,
`15.40528737169489`, `15.477036717133474`,
`24.60208367015022`, `31.658750512333974`, and
`37.10384792802646`.

### Mechanism verdict: calibration hypothesis refuted

The late-transfer target bank contains 711 fractional predictions:

| ACS clone | Fractional before calibration | Fractional in final pool | Rows changed by calibration |
|---:|---:|---:|---:|
| 0 | 342 | 0 | 13,417 |
| 1 | 360 | 360 | 0 |
| 2 | 9 | 9 | 0 |

Every surviving fractional bit is identical between the target bank and final
pool. The stacked calibration explicitly defines ASEC clone 0 as reference and
ACS clone 0 as recipient in
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8975-8985`.
Its amount mapping selects values from the sorted observed donor array rather
than interpolating in
`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:577-680`.

The pool receipt confirms the calibration ran:

- 8,419 allowed-carrier/addition-candidate rows within 856,626 mutable ACS
  clone-0 rows;
- 13,417 changed clone-0 rows, including 4,998 cleared and 8,290 added;
- `capacity_limited=true`;
- 8,419 mapped positive amounts;
- reference quantiles `[2, 6, 12, 26, 36]`;
- recipient quantiles `[2, 8, 22, 24, 32]` before and
  `[2, 6, 12, 26, 36]` after;
- amount QED `0.5882352941176471 -> 0.0`;
- zero donor-support violations.

Weeks' declared post-transfer calibration spec is at
`packages/microcosm-build/src/microcosm/build/us_runtime/post_transfer_calibration.py:237-263`.
The actual cause precedes calibration. Before this branch,
`_target_encoding` treated numeric `weeks_unemployed` as continuous because
the engine metadata says `float`; a quantile-regression-forest prediction may
therefore interpolate between integer observations. The relevant encoding and
decoding seam is
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:3037-3120`
and `:3240-3270`. Calibration then repaired only clone 0 by design.

### Why the gate said “5,218 PUF rows”

There is no physical PUF channel in this pool. The compatibility helper
`support_role_series` deliberately labels clone 0 as the ASEC-compatible
operator role and every clone above zero as the PUF-tax-detail operator role,
independent of physical source. See
`packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py:390-489`.
The old summary confused that operator role with physical channel.

The 5,218 rows are all ACS-origin, UC-zero, positive-week non-native clones:

| Actual rows | Integer weeks | Fractional weeks | Total |
|---|---:|---:|---:|
| ACS clone 1 | 4,733 | 355 | 5,088 |
| ACS clone 2 | 121 | 9 | 130 |
| **Total** | **4,854** | **364** | **5,218** |

The UC-zero constraint is not defined for those ACS non-native QRF clones.
The updated gate preserves the compatibility role for operator plausibility
but uses physical source and clone provenance to own source/UC checks.

## Task 2: source fix and receipt contract

`weeks_unemployed` is now in `_DISCRETE_NUMERIC_TARGETS` at
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:133-141`.
The existing discrete codec:

1. rejects nonintegral donor support;
2. snaps finite predictions to the nearest value in actual observed support;
3. resolves exact-distance ties to the lower support value;
4. emits integer/nullable-integer output.

The snap is at
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:3327-3339`.
The sorted discrete-target roster is part of the transfer execution identity at
`:271`; the generated authority records it at
`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:408-411`.
The final regenerated bundle spec SHA-256 is
`5f44d96d45e9aabcea2d565ef063d68bfc0652df1b38b08aa31ce6896d15f371`.
This invalidates affected target-bank identities instead of silently changing
content under an old receipt.

The release-gate and WIC edits also change exact source bytes attested by the
shared legacy seed protocol. Its reviewed WIC grammar now explicitly records
assembled `person_source_id` before the unchanged legacy fallbacks at
`packages/microcosm-build/src/microcosm/build/spec_engine/seeds.py:720-737`.
The fail-closed spec proof was coherently re-pinned to 42,122/42,122 claimed
fields (32,352 authored and 9,770 resolved) and 41/41 inventory checks. The
committed coverage report, producer-semantics digest, seed protocol/map,
checkpoint identity, pointer inventory, and affected country-envelope goldens
all move together. A detached `origin/main` comparison confirmed that the
BE/UK/minimal-envelope changes come from this branch's attested source and seed
grammar, not environment drift.

The regression at
`packages/microcosm-build/tests/test_us_acs_transfer.py:1391-1430` sends a
fraction-producing mean QRF through the real target codec, verifies that output
lies on observed integer donor support, and verifies that the execution
contract declares `weeks_unemployed`.

Because the hypothesis was refuted, no change was made to
`post_transfer_calibration.py` or its strict validator. Its observed-support
amount mapping, before/after margin fidelity, hashes, anchors, capacity status,
and donor-support validation remain mandatory; the complete 47-test receipt
contract passes.

**Artifact consequence:** pool content changes. The host session must rerun
`late_transfer -> simulated -> terminal-gates` from checkpoints for the
candidate pool. This lane did not run any pool or release build.

## Task 3: stacked-aware weeks release gate

The gate now uses the narrow, read-only physical-source accessor
`support_gate_source_channel_series` at
`packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py:496-524`.
That accessor is restricted to reviewed gates/reporters by the static
source-blindness contract.

| Frame layout | Plausibility roster | Raw LKWEEKS validity | Exact LKWEEKS carry | UC-zero consistency |
|---|---|---|---|---|
| No provenance | ASEC | all rows | all rows | none, matching legacy behavior |
| Legacy ASEC+PUF roles | ASEC + PUF | ASEC role | ASEC role | PUF role |
| Assembled ASEC+ACS | actual physical channels | every physical ASEC clone | physical ASEC clone 0 | physical ASEC non-native clones + non-ASEC clone 0 |

The assembled scope logic is at
`packages/microcosm-build/src/microcosm/build/us_runtime/weeks_unemployed.py:1320-1380`;
summary and failure construction are at `:1204-1317` and `:1383-1435`.
The ASEC plausibility bands and the legacy recipient/PUF bands remain
byte-identical, and every non-ASEC physical channel uses the unchanged
recipient band.

A read-only replay on the supplied pool reports:

- roster `asec, acs`;
- 234,133 raw ASEC source rows, zero invalid;
- 108,073 direct native reconciliation rows, zero mismatches;
- 982,686 UC-constrained rows, zero mismatches;
- ASEC positive share `0.01956717285126748`, weighted mean
  `0.32416674108944377`, weighted weeks `52,743,981.18128455`;
- ACS positive share `0.00800370929739283`, weighted mean
  `0.1376988711272782`, weighted weeks `21,987,101.082573153`;
- the unchanged channel bands pass;
- the gate fails only on the genuine 369 noninteger values.

Stacked ASEC+ACS and legacy ASEC+PUF fixtures cover roster derivation, raw
source scoping, native reconciliation, and UC ownership. The complete focused
weeks file passed: 26 passed, 1 skipped.

## Task 4: complete release-side archaeology audit

The audit started from every release call in
`tools/build_us_fiscal_refresh_release.py:8690-11726`, traced each gate and
its immediately preceding producer/wrapper, and classified physical-source,
operator-role, clone-layout, entity-layout, and raw-column assumptions. The
table groups gates only where they share the same architectural conclusion;
every release gate in the roster is named.

| Gate/stage | Stacked assumption found | Disposition |
|---|---|---|
| Validation input coverage; register consistency; release target parity; target-profile coverage; base-population scale | Manifest, registry, target, or aggregate-weight contracts; no legacy per-row source roster | Already compatible; no change |
| PUF capital-gains-tail presence and post-selection preservation | Authenticated tail-column presence/preservation checks, not a demand for a physical PUF channel (`build_us_fiscal_refresh_release.py:8837-8841,8937-8941`) | Already compatible; no change |
| Exact-k PUF tail support; exact-k frozen-register fit; fiscal-target materialization/skip/zero-support; critical/SOI fit | Calibrated diagnostics and authenticated exact-k inputs, not physical source labels | Already compatible; no change |
| Weeks input and post-selection weeks input | Used clone operator roles as physical ASEC/PUF channels and applied raw/UC checks to the wrong rows | **Fixed** with physical roster and separate validity/reconciliation/UC scopes |
| QBI input | Already has an explicit stacked ACS path | Already compatible; no change |
| Workers' compensation; alimony; Medicare; retirement contributions; retirement distributions | Raw ASEC-only columns were validated/reconciled through legacy operator roles | **Fixed**: all physical ASEC clones own raw validity; native physical ASEC rows own exact direct-carry reconciliation where applicable |
| SSI reporter capture/take-up | Strict `SSI_VAL` validation treated null ACS raw-source cells as invalid | **Fixed**: capture reporters from physical ASEC; assignment remains source-blind and consumes the captured ID set |
| SIPP Head Start; voluntary filing | Assumed exactly an occurrence pair / two clones | **Fixed**: explicit assembly clone index, arbitrary clone count, duplicate source+clone refusal, clone-0-or-lowest canonical decision, fanout to every clone |
| Prior-year income clone diagnostic | Compared only the first two occurrences | **Fixed**: group all assembled clones; existing availability band unchanged |
| SSI disability clone-divergence diagnostic | Compared only a pair and missed clone-2-only divergence | **Fixed** for all assembled clones; whether divergence becomes fatal still requires owner ruling |
| WIC deterministic draw | Raw source-local identity could collide across physical ASEC and ACS | **Fixed**: assembled frames prefer unique `person_source_id`; clones remain draw-stable and legacy precedence is unchanged |
| Farm business; domestic production; child support; disability benefits; educator expenses; Form 4952; SALT refund; capital-gain details; energy subsidy; housing; other health insurance | These use clone-operator roles for reviewed producer/plausibility semantics, not as claims about physical source | Intentionally unchanged; stacked-compatible role diagnostics |
| Childcare; casualty loss; miscellaneous itemized; immigration; generic take-up; hours; SNAP take-up; relationship; eligibility; education; pregnancy; reported-coverage vintage; SNAP discretionary exemption | Output/input signal checks do not assume a physical PUF channel or require raw ASEC cells on all rows | Already compatible; no change |
| Local health input; Medicaid take-up; SNAP state take-up; SSI final/delivery | Output or diagnostics-based checks | Already compatible after SSI reporter fix |
| Input-mass reference; degenerate input; eCPS parity; release input coverage; export input mass; QRF tail concentration; reform-coverage smoke; source coverage | Export-wide schema/mass/tail/simulation contracts, not legacy source rosters | Already compatible; no change |
| Export count-calibrated take-up staleness | Export consistency against recorded count-calibration receipts (`build_us_fiscal_refresh_release.py:11702-11726`), independent of physical source roster | Already compatible; no change |
| SCF wealth | Archived predictors silently map ACS-null CPS race/Hispanic source cells to `Other` | **Owner ruling required**, detailed below |
| SSI disability criteria | Archived receiver strictly requires six ASEC disability predictors and `SSI_VAL` on every row | **Owner ruling required**, detailed below |
| SCF auto loans | Imports the SCF wealth CPS-race mapping, silently mapping ACS recipients to `Other` | **Owner ruling required**, detailed below |
| SIPP vehicles | Missing ACS `SPM_TENMORTSTATUS` defaults every ACS recipient to tenant code 3 | **Owner ruling required**, detailed below |
| SIPP tips | Missing ACS `PEIOOCC` defaults to non-tipped; global band can conceal a dead ACS channel | **Owner ruling required**, detailed below |
| ORG wages/FLSA | Missing ACS race/ethnicity/occupation defaults to zero; unchanged global gates are guaranteed to fail | **Owner ruling required**, detailed below |

### Unambiguous repairs

- `support_provenance.py:331-354,496-524` centralizes detection of assembled
  metadata and physical channels. General derive/imputation code cannot call
  the physical accessor: `test_us_spine_blindness.py` pins the exact reviewed
  call graph.
- `alimony.py:282-327` and `workers_compensation.py:560-604` split physical
  source validity from direct-carry reconciliation while preserving all role
  bands.
- `medicare_take_up.py:134-143,248-270`,
  `retirement_contributions.py:172-199,602-628`, and
  `retirement_distributions.py:274-301,709-739` scope release diagnostics
  without routing their population operators by physical origin.
- `ssi_take_up.py:515-550` captures reporters from physical ASEC and accepts
  ACS nulls; the assignment consumes captured lineage rather than reading
  origin.
- `sipp_head_start.py:579-621` and
  `voluntary_filing.py:749-783` use explicit clone indices on assembled
  frames. `prior_year_income.py:744-765` and
  `ssi_disability_criteria.py:1119-1152` inspect all clones.
- `wic_claim.py:337-376` prefers the assembly-unique identity only when
  assembled metadata exists.

### Owner rulings required

1. **SSI disability criteria.** Release call
   `tools/build_us_fiscal_refresh_release.py:9840-9847`. The receiver
   requires and strictly evaluates all six `PEDIS*` predictors on every row
   at
   `packages/microcosm-build/src/microcosm/build/us_runtime/ssi_disability_criteria.py:755-859`,
   then strictly evaluates `SSI_VAL` at the same file's `:1023-1034`; the
   support-role prediction loop is `:965-1005`. Each input is finite on all
   234,133 physical ASEC rows and null on all 1,736,840 ACS rows, so the stage
   fails before its gate.
   Required decision: transfer/map canonical ACS disability inputs and
   reporter-anchor semantics, or revise/retrain the archived recipient model.

2. **SCF wealth.** Release call
   `tools/build_us_fiscal_refresh_release.py:9798-9808`.
   `packages/microcosm-build/src/microcosm/build/us_runtime/scf_wealth.py:654-667`
   requires `PRDTRACE`, `PRDTHSP`, `A_MARITL`, `PEPAR1/2`, `PH_SEQ`, and
   `A_LINENO`; `:670-691` selects the archived reference person, and
   `:694-740` builds race/marriage/children predictors. Only
   `PRDTRACE/PRDTHSP` are ACS-null; `:611-620` maps those to zero/`Other`.
   The other structural fields are finite and mapped on every ACS row. All
   three asset leaves are all-null and `net_worth` is absent, so the wrapper's
   recompute branch at `:1109-1159` is certain and every ACS recipient is
   silently forced to `Other`. Required decision: authoritative ACS
   race/Hispanic mapping or transfer, or a reviewed fallback/retrained model.

3. **SCF auto loans.** Release call
   `tools/build_us_fiscal_refresh_release.py:9990-9996`.
   `packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:125-137`
   requires the archived layout; `:320-362` selects the reference person and
   `:365-417` supplies race/marriage/children inputs. `A_LINENO` is required
   by the selector's presence check but is not subsequently used; the mapped
   structural fields are finite. The actual defect is the imported SCF-wealth
   CPS-race mapping of ACS-null `PRDTRACE/PRDTHSP` to `Other`. All three
   auto-loan outputs are absent, so `:475-513` certainly takes the QRF path.
   Required decision: the same authoritative mapping or reviewed model change
   as SCF wealth.

4. **SIPP vehicles.** Release call
   `tools/build_us_fiscal_refresh_release.py:10026-10033`.
   `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:150-157`
   declares the archived layout; `:577-622` reads person
   `SPM_TENMORTSTATUS` and fills missing values to tenant code 3, while
   `:649-763` uses marriage, `A_LINENO` reference selection, and tenure.
   The pool's mapped `A_LINENO` is finite, but person
   `SPM_TENMORTSTATUS` is ACS-null and the household field is absent. Both
   vehicle outputs are absent, so `:872-917` certainly recomputes and
   classifies ACS tenure as tenant. Required decision: canonical ACS tenure
   mapping/transfer and whether assembled `A_LINENO` preserves the archived
   reference-person semantics, or model revision/retraining.

5. **SIPP tips.** Release call
   `tools/build_us_fiscal_refresh_release.py:10103-10109`.
   `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_tips.py:249-263`
   maps null `PEIOOCC` to tipped code 0; recipient use is `:363-396`, wrapper
   branching is `:450-473`, and the gate is `:484-549`. The pool lacks both
   tip output columns, so the wrapper takes its imputation branch.
   `PEIOOCC` is ASEC-finite and ACS-null. The ASEC-conditional tipped-code
   share is `0.0700204513`; ACS is exactly zero; the global share is
   nevertheless `0.0353393976`, inside the unchanged `[0.02, 0.15]` band.
   This proves that gate component conceals the dead ACS channel, not that the
   separate tip-income component necessarily passes. Required decision:
   map/transfer ACS occupation or select a reviewed alternate model, then add
   channel-aware diagnostics once those semantics are owned.

6. **ORG wages/FLSA.** Release call
   `tools/build_us_fiscal_refresh_release.py:10144-10150`.
   `packages/microcosm-build/src/microcosm/build/us_runtime/org_wages.py:535-559`
   fills null `PRDTRACE/PRDTHSP/POCCU2` to zero; these enter features at
   `:581-625`; wrapper `:901-945` always recomputes; gate `:959-1034`. All
   three raw fields are ASEC-finite and ACS-null. On real weights, CPS-race
   nonzero share is `0.5047010838` versus unchanged minimum `0.95`, and
   detailed-occupation nonzero share is `0.4155259611` versus minimum `0.65`.
   The next release is guaranteed to fail here. Required decision:
   authoritative ACS race/ethnicity/occupation mapping or a revised model
   contract, not threshold weakening.

### Read-only real-pool replay after the fixes

The repaired alimony, Medicare, retirement-contribution,
retirement-distribution, workers'-compensation, and SSI-reporter checks pass.
Their physical ASEC source count is 234,133; direct-carry gates use 108,073
native ASEC rows where applicable. The weeks gate fails only on 369 fractional
values.

Two other observed failures are genuine data/spec outcomes, not row-label
archaeology:

- prior-year-income weighted availability is `0.042839`, outside the
  unchanged `[0.05, 0.50]` band;
- WIC finds `is_pregnant=true` on nonfemale rows, with example row positions
  `129405, 167076, 171133, 192443, 195546`.

No threshold was changed to hide either outcome.

## Diff summary by file

| File | Rationale |
|---|---|
| `PROGRESS.md` | Maintains the required state/done/next journal from kickoff through verified handoff while preserving historical lanes |
| `changelog.d/stacked-release-gate-alignment.fixed.md` | Records the user-visible integer-support and stacked-gate fix under repository convention |
| `out.md` | This provenance, audit, verification, and handoff report |
| `us/spec/imputation.yaml` | Generated authority now receipts `weeks_unemployed` in the discrete numeric transfer roster |
| `us_runtime/acs_transfer.py` | Declares weeks integer-supported through the existing deterministic observed-support codec |
| `us_runtime/support_provenance.py` | Adds assembled-metadata detection and a narrow read-only physical-source accessor |
| `us_runtime/__init__.py`, `us_runtime/puf_support.py` | Re-export assembled-layout detection while keeping the physical accessor confined to its owner |
| `us_runtime/weeks_unemployed.py` | Derives stacked/legacy rosters and separates source validity, native reconciliation, and UC scopes |
| `us_runtime/alimony.py`, `us_runtime/workers_compensation.py` | Scope raw-source and exact-carry checks to their physical/native owners |
| `us_runtime/medicare_take_up.py`, `us_runtime/retirement_contributions.py`, `us_runtime/retirement_distributions.py` | Make release summaries source-aware while retaining source-blind producer kernels |
| `us_runtime/ssi_take_up.py` | Captures SSI reporters from physical ASEC and permits null ACS source cells without origin-routing assignment |
| `us_runtime/sipp_head_start.py`, `us_runtime/voluntary_filing.py` | Replace pair/occurrence layouts with explicit arbitrary-clone assembled layouts |
| `us_runtime/prior_year_income.py`, `us_runtime/ssi_disability_criteria.py` | Detect divergence across all assembled clones |
| `us_runtime/wic_claim.py` | Uses assembly-unique person identity for cross-origin deterministic draws |
| `spec_engine/seeds.py` | Receipts assembled `person_source_id` as WIC's first seed-key source |
| `spec_engine/field_usage.py`, `spec_engine/inventory_coverage.py`, `tools/spec_engine_coverage.py` | Re-pin the exact two-field expansion and downstream producer/seed/checkpoint/pointer identities |
| `docs/evidence/spec-engine/us-f0-coverage.json` | Regenerates the closed 42,122-field, 41-item coverage attestation |
| `tests/test_us_acs_transfer.py` | Proves fractional QRF output is snapped to observed integer weeks support and receipted |
| `tests/test_us_weeks_unemployed.py` | Covers stacked ASEC+ACS and legacy ASEC+PUF roster/scope contracts |
| `tests/test_us_alimony.py`, `test_us_workers_compensation.py`, `test_us_medicare_take_up.py`, `test_us_retirement_contributions.py`, `test_us_retirement_distributions.py`, `test_us_ssi_take_up.py` | Cover physical ASEC source validity, native reconciliation, legacy behavior, and SSI reporter lineage |
| `tests/test_us_sipp_head_start.py`, `test_us_voluntary_filing.py`, `test_us_prior_year_income.py`, `test_us_ssi_disability_criteria.py` | Cover clone 2+, duplicate clone refusal, deterministic canonical rows/fanout, and all-clone divergence |
| `tests/test_us_wic_claim.py` | Covers cross-origin identity collision avoidance and clone-stable draws |
| `tests/test_us_multispine_puf_clone.py`, `test_us_spine_blindness.py` | Cover assembled provenance and pin physical-source access to the reviewed gate/reporter call graph |
| `tests/test_spec_engine_seeds.py` | Pins WIC's assembled-first seed grammar |
| `tests/test_spec_engine_field_usage.py`, `test_spec_engine_coverage_tool.py`, `test_spec_engine_country_bundles.py`, `test_spec_engine_loader.py` | Pin the reviewed field totals, report identities, and source-attested envelope goldens |
| `tests/test_us_multispine_pool_tool.py` | Re-pins the constants adapter's live US spec identity after regeneration while retaining the separate arbitrary checkpoint identity fixture |

All source paths in the table are under
`packages/microcosm-build/src/microcosm/build/`; all abbreviated test paths
are under `packages/microcosm-build/tests/`.

## Judgment calls

- The post-transfer calibration was not modified merely because its receipt is
  visible near the symptom. Checkpoint bits prove the surviving values predate
  it, and its mapping already uses actual donor support.
- The integer contract is explicit by target rather than inferred from the
  current PolicyEngine physical dtype. That dtype is `float` and caused the
  bug; reviewed domain semantics are the authority.
- Snapping to observed donor support was preferred over generic rounding. It
  cannot invent an unsupported week count, is deterministic, and reuses an
  existing receipted codec.
- Physical source identity is exposed only to read-only gates/reporters.
  Medicare and retirement producer changes considered during review were
  reverted: routing population treatment by origin would violate the
  source-blind operator boundary. The authenticated release path consumes
  already-produced pool surfaces.
- Raw validity covers all physical ASEC clones because they carry the raw
  source. Exact reconciliation covers only native ASEC where transferred
  non-native clones are intentionally allowed to differ.
- Operator roles remain appropriate for producer/plausibility bands. They are
  not aliases for physical ASEC/ACS source, which was the old weeks bug.
- Clone divergence in SSI disability remains diagnostic rather than newly
  fatal; changing that policy needs an owner ruling.
- The six archived-model cases were not “fixed” with zero fills, gate
  rescoping, or weaker thresholds. Each requires substantive decisions about
  ACS predictors and model semantics.

## Verification evidence

All commands ran offline against the prebuilt environment, with
`UV_CACHE_DIR=/private/tmp/microcosm-weeksgate-uv-cache` so the required
`uv run --no-sync` command could operate inside the sandbox. Each pytest
shard ran in one independent process.

- `uv run --no-sync ruff check .`: PASS, `All checks passed!`
- `uv run --no-sync python tools/ci_test_groups.py --verify`: PASS,
  `tracked_test_files=309`, `verification=ok`
- Eight directly affected spec/receipt files: PASS, 102 tests
- `tools/generate_us_bundle_from_constants.py --check`: PASS, final US spec
  SHA-256
  `5f44d96d45e9aabcea2d565ef063d68bfc0652df1b38b08aa31ce6896d15f371`
- `tools/spec_engine_coverage.py --check`: PASS, 42,122/42,122 fields and
  41/41 inventory checks
- `uv run --no-sync pytest packages/microcosm-calibrate/tests -q`: PASS,
  203 passed
- `uv run --no-sync pytest packages/microcosm-data/tests -q`: PASS,
  318 passed, 2 skipped
- `uv run --no-sync pytest packages/microcosm-fit/tests -q`: PASS,
  93 passed
- `uv run --no-sync pytest packages/microcosm-frame/tests -q`: PASS,
  295 passed, 36 skipped
- `uv run --no-sync pytest packages/microcosm-build/tests -q`: PASS,
  6,608 tests collected, 100% reached, exit code 0, expected skips only
- Complete ACS-transfer file: PASS, 65 tests
- Complete post-transfer calibration receipt-contract file: PASS, 47 tests
- Complete weeks file: PASS, 26 passed, 1 skipped
- Complete alimony/workers'/SSI files: PASS, 28 / 21 / 71 tests
- Medicare + retirement contribution/distribution focused files: PASS, 65 tests
- Repository `git diff --check`: PASS

The shard warnings observed are existing numerical, sparse-tensor,
joblib core-detection, and PolicyEngine divide warnings; none is a failure and
none originates in the new transfer/source-scope paths.

## Commit inventory

Implementation and evidence commits preceding the final report carrier:

1. `a3331db6` Start stacked release gate alignment journal
2. `9979d101` Record fractional weeks provenance
3. `d7ad753e` Preserve integer support for transferred weeks
4. `0c5b05a8` Receipt integer weeks transfer support
5. `c22a6799` Align weeks gate with stacked source roles
6. `9bbd6dfe` Centralize stacked source channel scopes
7. `49052e6e` Scope alimony source checks to physical ASEC
8. `17ed9a0b` Key WIC draws by stacked person identity
9. `c7dd53ad` Support stacked Head Start clone layouts
10. `3f934688` Scope Medicare source gate to stacked ASEC
11. `5852f4ba` Scope retirement contribution source diagnostics
12. `e40d5db7` Support stacked voluntary filing clones
13. `148881c6` Check all stacked prior-year income clones
14. `52b27f78` Separate alimony source validation and carry scopes
15. `058f27b5` Record stacked clone-layout repairs
16. `97de22e7` Scope workers compensation sources in stacked pools
17. `74e96187` Scope retirement distribution source gates
18. `41007992` Anchor SSI reporters to physical ASEC sources
19. `eb5cd4b6` Keep SSI assignment source blind
20. `3dc5df51` Keep stacked source repairs gate scoped
21. `9b36720b` Record stacked raw-source gate repairs
22. `9520c03f` Report divergence across all SSI disability clones
23. `b8e04e9c` Confine physical source access to release gates
24. `b2de92b9` Document stacked release gate alignment
25. `cd8c55a7` Repin spec engine for integer weeks
26. `12a918ed` Repin multispine live spec fixture

The final report/journal carrier follows this inventory and is necessarily
self-excluded; the clean-worktree handoff lists it in the final response.
