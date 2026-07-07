# Build H — identify the 4 floating PUF-input dimensions + re-run to certification

Task: PolicyEngine/populace **#299 Build H**. Add real SOI calibration targets so the 4 export-mass
residual columns become IDENTIFIED, then re-run the two Build G candidates (DENSE + FROZEN-SPARSE) to
certification. **STAGING/LOCAL ONLY** — never touch prod HF / latest.json. Publication is Max's call.
Detached compute (nohup+pidfile+logs). Three-strike stop with precise diagnosis. Every number from source.

Runtime home (OUTSIDE repos, durable): `/Users/maxghenis/PolicyEngine/_buildh-runtime/`.
Worktrees: populace `build-h-run` (off build-g-run lineage), ledger `build-h-soi-income-targets`.

## Frozen invariants from Build G (only the FEED changes: v7 -> v8)
base **18833fb6**, pe-us **1.764.6** / core 3.26.11, period 2024, λ=0, seed 0, feed v7 `735f326a`,
CD-crosswalk `383a6666`, reform = default `US_JCT_TAX_EXPENDITURE_REFORMS`.
Export-mass reference (corrected, #327) = **live-default 57k `c2065b642ab0…`** via
`--export-input-mass-reference-h5`. Registry: dense `e035007b` (5521) / sparse `a9897709` (5502, 19-cell
zero-support exclusions). Selection manifest `identities_sha256=77363a47`.

## Build G candidate results (from #299 comment 4902533514) — Build H must MEET/BEAT
| metric | DENSE parent | FROZEN-57k (headline) | certified band |
|---|---:|---:|---:|
| records | 337,704 | 57,240 | 57,240 / 75,112 |
| final loss | 0.04139 | **0.02964** | 0.044 / 0.0423 |
| within-10% | 86.16% | 89.15% | 94.7% / 86.4% |
| fed income tax (SOI) | +1.23% | **+0.50%** | ~target |
| SS (SSA) | +0.05% | −0.13% | ~target |
| mortgage tax-exp (JCT) | +59.7% | **+44.5%** (real miss) | must SHRINK |
| all structural gates | PASS | PASS | — |
| zero-support | 0 | 0 | — |
| export-mass gate | FAIL (4) | FAIL (4) | must PASS |

## Export-mass gate: the 4 residuals + PASS criteria (from build-g release logs)
Reference = populace_us_2024.h5 (live-default 57k, c2065b64); ±50% band; min_reference_total $1B.

| column | ref ($B) | ±50% band ($B) | DENSE now | SPARSE now | Build H direction |
|---|---:|---|---:|---:|---|
| estate_income | 98.434 | [49.22, 147.65] | 45.66 (−53.6%) | 32.71 (−66.8%) | pull UP (SOI estate/trust income) |
| home_mortgage_interest | 311.126 | [155.56, 466.69] | 526.45 (+69.2%) | 474.25 (+52.4%) | pull DOWN (SOI Sched A mtg-int ded) |
| first_home_mortgage_interest | 310.839 | [155.42, 466.26] | 502.84 (+61.8%) | in-band | pull DOWN (follows home_mtg) |
| miscellaneous_income | 47.401 | [23.70, 71.10] | 15.05 (−68.3%) | 21.66 (−54.3%) | pull UP (SOI other income) OR reviewed exclusion (this col only) |
| non_sch_d_capital_gains | 75.747 | [37.87, 113.62] | in-band | 160.10 (+111.4%) | pull DOWN (SOI split line) OR reviewed exclusion+justif |

## 4 columns — identification class (from #299 comment) + Build H plan
- **home_mortgage_interest** — partially identified: JCT `deductible_mortgage_interest.revenue_loss`
  binds it (+44.5% frozen real miss). Add DIRECT SOI itemized-deduction target (amount + returns).
  **PIVOTAL OPEN Q**: is PE-US `home_mortgage_interest` the Schedule-A itemizer-deducted amount (SOI $171B
  correct → pull down fixes JCT + export gate) or ALL-household mortgage interest (SOI itemizer too low →
  targeting distorts)? Raw base $345B, ref $311B, export $474–526B, SOI Sched A $171B. MUST verify PE-US
  variable definition + imputation population + how a populace target maps to the export INPUT column.
- **non_sch_d_capital_gains** — component split of net_capital_gains (aggregate constrained ±0.2%, split
  not). Add SOI split line if it exists; else reviewed exclusion w/ component-split justification.
- **estate_income** — purely unidentified. Add SOI Table 1.4 estate/trust net income line.
- **miscellaneous_income** — purely unidentified + sign-unstable (raw base −$7.9B). Add SOI Table 1.4
  "other income (less loss)" line; reviewed exclusion acceptable for THIS column only if untargetable.

## SOI values (prior-agent sourced; VERIFY each from raw workbook before use)
- Table 2.1 (Pub 1304) https://www.irs.gov/pub/irs-soi/23in21id.xls sha df6cf04ed3b7:
  mortgage_interest_paid (financial insts) $167,675,863,000 / 11,490,340 returns;
  personal_seller $3,688,924,000 / 269,588 returns; total Sched A home mtg int **$171,364,787,000**.
  Already in v7 as irs_soi.ty2023.table_2_1 (per prior agent) — VERIFY.
- Table 1.4 (Pub 1304) https://www.irs.gov/pub/irs-soi/23in14ar.xls sha b6c1f87fbb55:
  net_capital_gains (all) $966,168,014,000 = TAXABLE NET agg (not the split). estate/trust, other income,
  capital-gain-distributions lines NOT yet in ledger table_1_4 → must author. VERIFY all from 23in14ar.xls.

## Feed mechanics (v7 -> v8)
Feed = arch.consumer_fact.v1 JSONL. v7 = `_buildf-runtime/inputs/consumer_facts_buildf_v7.jsonl`
sha 735f326a, 37,058 facts. Dedup key = lineage.source_record_id. SOI facts carry native tax_year period;
populace maps to @2024 at compile time. v8 = v7 + new SOI target rows; sha-pin + ledgered supersession
(v6/v7 discipline). Ledger export = policyengine_ledger/consumer.py build_consumer_artifact. SOCA package
may need parking for local ledger runs (Build E notes).

## Run recipe (mirror buildg chain; ONLY --ledger-facts changes to v8)
tools/build_us_fiscal_refresh_release.py, base-h5 18833fb6, --export-input-mass-reference-h5 live-default
57k (c2065b64), --seed 0 --no-staging. DENSE = full base + --dense-default-dataset (no per-run exclusions).
SPARSE = --selection-source-manifest 57k + --dense-default-dataset + --zero-support-exclusions 19-cell.
DETACHED (nohup+pidfile+logs in _buildh-runtime/logs/buildh-run). Real exit codes, no pipe-masking.

## HARD RULES
Three-strike stop + precise diagnosis. Detached compute only. Real exit codes. Staging only; prod
HF/latest.json untouched. Every number cited from its source document.

## TIMELINE
### 2026-07-07
- **Step 0 (setup).** Read #299 Build G comment 4902533514, #327, PROGRESS_BUILDG_RUN, build-g release
  logs (dense/sparse/chain). Verified venv resync clean (94 pkgs). Recorded exact export-mass residuals +
  ref bands (above). Prior agent left no repo commits (both worktrees clean); groundwork = SOI sourcing +
  23in14ar.xls in scratch. Dispatched code-exploration on (1) populace fact→target→export-column mechanics
  + export-mass gate + #256 coverage pattern, (2) PE-US mortgage/income variable semantics.

- **Step 1 (SOI values VERIFIED from committed workbooks).** Ledger has the raw .xls committed at
  `db/data/irs_soi/table_{1_4,2_1}/23in{14ar,21id}.xls`; sha256 of 23in14ar.xls = b6c1f87fbb55… matches
  manifest + prior-agent scratch. Parsed with xlrd (value_scale 1000, amounts in $1000s):

  **Table 1.4 (TY2023) "All returns, total" (row idx 8):**
  | concept | returns col/val | amount col/val ($) |
  |---|---|---|
  | Capital gain distributions reported on Form 1040 | AJ 3,209,131 | AK **$9,340,820,000** |
  | Estate and trust — Net income | BX 648,583 | BY **$47,892,046,000** |
  | Estate and trust — Net loss | BZ 36,592 | CA $4,898,828,000 → NET **$42,993,218,000** |
  | Other income — Net income | CN 7,468,434 | CO **$60,816,508,000** |
  | Other income — Net loss | CP 318,839 | CQ $12,213,628,000 → NET **$48,602,880,000** |
  | Sales of cap assets Sch D taxable net gain (xcheck) | — | AM $966,168,014,000 ✓ |

  **Table 2.1 (TY2023) itemizers "All returns, total" (row idx 9), Home mortgage interest:**
  | concept | returns | amount ($) |
  |---|---|---|
  | **Total** | CW 11,644,348 | CX **$171,364,787,000** |
  | Paid to financial institutions | CY 11,490,340 | CZ $167,675,863,000 |
  | Paid to individuals | DA 269,588 | DB $3,688,924,000 |
  (CZ+DB = CX ✓. table_2_1 pkg already has CZ+DB measures; missing the CW/CX "Total".)

- **Step 1 RECONCILIATION vs export-mass ref bands (PIVOTAL — contradicts the optimistic "all pass"):**
  | column | SOI target ($B) | ref ($B) | SOI vs ref | in ±50% band? |
  |---|---:|---:|---:|---|
  | home_mortgage_interest | 171.365 (Sched A total) | 311.126 | −44.9% | **YES** (just inside) → gate passes IF input==itemizer amt |
  | miscellaneous_income | 48.603 net / 60.817 gross | 47.401 | +2.5% / +28.3% | **YES** → gate passes, real target |
  | estate_income | 42.993 net / 47.892 gross | 98.434 | −56.3% / −51.3% | **NO** (just below) → ref is ~2× SOI, incidental |
  | non_sch_d_capital_gains | 9.341 (cap-gain distrib) | 75.747 | −87.7% | **NO** (far below) → ref ~8× SOI, likely concept mismatch |

  So the live-default reference is itself off-SOI on estate (~2×) and non_sch_d (~8×). Targeting to true SOI
  makes those two FAIL the parity-vs-incidental-reference check. Two coherent completions per column: (a)
  SOI ∈ band → clean gate pass (mortgage, misc); (b) SOI ∉ band because ref is incidental → SOI target
  identifies the column + reviewed exclusion in the parity gate justified BY that target (estate); (c)
  concept mismatch / magnitude fights constrained aggregate → reviewed exclusion w/ component-split
  justification (non_sch_d). Final per-column decision GATED on agent findings (PE-US variable semantics +
  exact gate comparison: does a targeted column re-reference to its target, or stay vs live-default?).
  NEXT: consume both agents; resolve mortgage definitional Q; author ledger table_1_4 lines + wire targets.
