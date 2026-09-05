# #834 / #789 receipts — childcare and Tax-Free Childcare port, Chronicle re-pin, bus targets

Plan: `repos/uk-834-childcare-tfc-port-plan.md` (canonical, María's rulings A0–A20). Licensed
evidence lives under `data/ukds/acceptance/834-childcare-tfc/` (SDC-safe aggregates, minimum
cell count 3); this file carries the committed, digest-pinned receipts the PR's signed entries
anchor to.

## Part A — Chronicle feed re-pin (increment I0)

Feed rebuilt from Chronicle `PolicyEngine/chronicle` main `6fb700e` (2026-09-04; carries #220
childcare, #231 the recovered #202, #239 the UC crosses / Child Benefit / census composition)
with `chronicle build-bundle --suite uk` → `build-consumer-artifact`:

| | value |
|---|---|
| `consumer_facts.jsonl` sha256 | `6ae49d7d7ab297df25a0b9bfe2d6776827c672d284fbb360957fe8337089549f` |
| `manifest.json` sha256 | `dcda51d6496aea67f768a284e7955c7520e7c8b91e2bed3569f247567b7153f0` |
| rows / schema | 128 717 / `policyengine_ledger.consumer_artifact.v2` |
| previous pin | `33ca98a` / `4395a4e7…`, 108 112 rows (local census); the national surface had been generated on `226358e7…`, 107 550 rows |

### Why the re-pin had to repair the contract (plan §5c)

On the previously pinned feed the committed national surface did not compile: 19 references
unsupported (`compile_uk_target_registry`), which `tools/calibrate_uk_national_dataset.py`
refuses. On the rebuilt feed, 12 were unsupported before the I0 corrections. After I0:

| surface | before (committed) | after I0 on `6ae49d7d…` |
|---|---|---|
| national membership | active 408, multi_fact 1, no_fact 7, signed_excluded 1 | **active 408**, no_fact 7, signed_excluded 2 |
| national runtime compile | 389 compiled / 19 unsupported on the pinned feed | **408 compiled / 0 unsupported** |
| local membership | active 19 618, no_fact_at_or_before_period 314, no_fact_for_area 1 586, compile_error 1 | **active 19 932**, no_fact_for_area 1 586, compile_error 1 (the signed `E14001416` SPI mean) |

Corrections (each a declaration, verified by the regeneration test and the runtime compile):

- UC composition rows (`dwp.uc.households_{single,couple}_{no,with}_children`,
  `_children_{1..5_or_more}`): exact dimension-key pins (`dimensions: [family_type]` /
  `[number_of_children]`), so the #239 child-entitlement and payment-indicator crosses no
  longer match; the generator's sum inference no longer treats a `dimensions` list as a sum.
- Caseload `dwp.uc.households`: `source_measure_id: total_units` +
  `groupby_dimension: dwp.uc_deductions_month` → binds DWP UC deductions statistics Table 1
  only; calendar-year average over April–December 2025 = **6 758 889** (the #735 value),
  1.45% below the Stat-Xplore family-type sum 6 858 287 (chronicle#247 asks for the
  cube's Total row).
- CGT totals: `groupby_dimension: hmrc.cgt_table1_line` (Table 5's UK rows carried the same
  concepts and values, £127.316bn gains at TY2024).
- One temporal basis for `dwp_universal_credit`: every row is a calendar-year average; the 100
  payment-band rows move from the May-2025 snapshot to the 2025 mean; the Scotland
  youngest-child row is re-bound at benefit-unit grain (`youngest_child_age < 1`).
- `ons.population.scotland_households_3plus_children` signed-excluded (its selector reaches
  person-level mid-year population rows; #736).
- Local `ons.rent.private_rent`: the stale deferral `private_rent_pipr_after_target_period`
  retired; 314 rows bind the calendar-year average of the 2025 PIPR months (12 members each).
- Feed guard: the calibration runner refuses a feed whose facts digest differs from the
  committed pin (`--allow-unpinned-feed` is a recorded diagnostic override); a hermetic test
  regenerates the surfaces from the pinned identity and skips only when the feed is absent.

## Part B — Licensed before-receipts (L0, L1)

All at design weights on full-scale spines (52 846 households, 113 626 persons, 61 234 benefit
units), engine policyengine-uk 2.94.0 for the receipts.

### L0 — spine-n (main `47c74225` + #842 + #850, built at 2.92.1), read at 2.94.0

`l0_spine_n_childcare_receipt.json`. The routed share reads its engine default (1.0): the column
is absent from the spine.

| row (model / fact) | 2024 basis (FY2024-25 / Jan-2024; extended Jan-2025 ages 2–4) | 2025 basis (FY2025-26 / Jan-2025) |
|---|---|---|
| TFC spending (£0.6322bn / £0.5998bn) | 1.073× | 1.174× |
| TFC children (1 085 020 / 1 151 515) | 1.020× | 0.951× |
| Working-parent children ages 2–4, England (621 482) | 1.283× | 1.261× |
| Early learning 2-year-olds, England (115 852 / 95 031) | 0.596× | 0.738× |
| Universal-only children, England (416 537 / 396 965) | 1.104× | 1.177× |

### L1 — engine-only twin (branch `eb8abcdd`: floor ≥2.93, lock 2.94.0; no new column)

`l1-engine-twin/l1-engine.h5`, 28 stages, spine battery 15/15. Payload against spine-n
(`l1_vs_spine_n_payload.json`): benunit and time_period tables byte-equal; person differs only
in `student_loan_balance` (26 rows); household differs in 37 columns — the WAS, LCFS, ETB,
property and consumption outputs of the QRF stages whose predictors include
`household_net_income` (which carries `tax_free_childcare`, moved by 2.92.2's gross-rate
arithmetic). No income, benefit or childcare input column moves. The childcare receipt on L1
(`l1_engine_childcare_receipt.json`) is identical to L0.

## Part C — column twin (L2), release round trip (L3), re-fit (L5), composition (L4)

Pending the Lane A/B increments.

## Part D — bus (#789)

E9's Part D (`experiments/685-net-new-stages-receipts.md`) is the sizing evidence: at design
weights England fares 0.56× BUS05ai and net support 0.71× BUS05bi; London 0.28×; England outside
London 0.74× / 0.97×; the LCFS fare gradient runs against NTS0705a (#790's disposition).
