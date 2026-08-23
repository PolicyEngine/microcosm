# QBI amount-margin ownership evidence (workstream 5)

Adjudication basis: `experiments/battery_burndown/ADJUDICATION.md` workstream 5
("Establish QBI ownership and refit the coupled amount surface", 8 BLOCKER
checks / 4 physical legs, adjudication line 112) and its remediation rule
(line 42): recompute each realized QRF regime from frozen donor support for
every availability pattern before any regime-specific fix, and persist the
regime in future receipts.

Extraction: `extract_qbi_ownership_evidence.py` (this directory), read-only
over the SHA-verified failed-attempt checkpoints. Machine-readable results:
`evidence.json`. **The extraction reproduces all eight amount-check values and
their supporting amount statistics with zero mismatches**: the four legs'
positive-leg incidence ratios, per-origin incidences, carrier counts, and QEDs
agree by exact binary64 equality, while the two SSTB clone-1 statistics agree
at the six-significant-digit precision published by adjudication rows 65–66
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:441-448,1586-1612,1795-1810`).

## Artifact identity

- Simulated checkpoint SHA-256
  `5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a`
  (matches the adjudication pin, ADJUDICATION.md:228); transferred
  `bdc9355d92659bb28d58b1ddcd647ec303f2ad217661e17d5b4b0984e04532e8`;
  assembled
  `3b50bbd6abca781ea5dc23c63e6128e4ee934042068fb32b94afb98eeb4d2540`
  (all recorded under
  `evidence.json.artifact_bindings.stages.<stage>.computed_sha256`, recomputed
  at extraction time). The external QRF target-bank H5 files used below are
  separately byte-hashed and matched to their SHA-bound publication receipts
  (`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:809-914`).
- The adjudication binds this checkpoint to release ID
  `populace-us-2024-stacked-f025-s578-asec42213-acs382903-20260816T145820Z-80e26cb5`
  and run ID `378f7af26eb24667be35de7cfe595d27`, publication state
  `gate_failed`/`simulation_ready=false` (ADJUDICATION.md:69). This evidence
  inherits that binding; nothing here implies release readiness.
- Stage semantics: the `transferred` checkpoint captures phases
  `{impute}` only, and `simulated` captures `{impute, derive, seed,
  simulate}` (`tools/build_us_multispine_pool.py:2139-2141`); the shared QBI
  reconciliation runs inside the `derive` stage after the late transfer
  (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2851-2908`).
  `transferred` is therefore the pre-reconciliation measurement point and
  `simulated` the terminal one.
- Row surface: 1,970,973 persons = 964,699 clone-0 + 964,699 clone-1 +
  41,575 clone-2; ASEC clone-1 donors: 108,073.

## Comparator replication

Battery statistics were recomputed with the terminal battery's exact math:
clone-scoped positive-weight rows split by support channel
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11943-12001`),
positive-leg incidence over the full origin scope with carriers as raw row
counts (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:12177-12234`), weighted inverse-ECDF
p10/p25/p50/p75/p90 magnitude quantiles and the QED envelope
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:12307-12338`), frozen tolerances
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11623-11626`), and person weights inherited from
household importance weights through membership (microcosm-frame
`packages/microcosm-frame/src/microcosm/frame/bundle.py:516-546`).

## Ownership decomposition (the four red amount legs)

Per-stage positive-leg ACS/ASEC incidence ratio and QED:

| leg | clone-1 producer (transferred) | clone-0 post-transfer, pre-reconciliation | clone-0 terminal (battery) |
|---|---|---|---|
| `qualified_bdc_income` | ratio **1.06846** (in band), QED 0.584598 | ratio **0.329192**, QED 0.827384 | ratio **0.328928**, QED 0.751361 |
| `qualified_reit_and_ptp_income` | ratio **1.04731** (in band), QED 0.597071 | ratio **0.385159**, QED 1.16264 | ratio **0.446585**, QED 1.15762 |
| `unadjusted_basis_qualified_property` | ratio **1.10702** (in band), QED 0.187807 (green) | ratio **0.774548**, QED 0.590155 | ratio **0.774548**, QED 0.590155 |
| `w2_wages_from_qualified_business` | ratio **1.1723** (in band), QED 0.451897 | ratio **0.760975**, QED 1.33713 | ratio **0.760975**, QED 1.33713 |

Supporting per-channel levels (weighted incidence, `evidence.json`):

| leg | donor (ASEC clone-1) | ACS clone-1 (producer) | ASEC clone-0 (drawn) | ACS clone-0 (drawn) |
|---|---|---|---|---|
| BDC | 0.0085506 | 0.0091359 | 0.0092141 | 0.0030332 |
| REIT/PTP | 0.0575926 | 0.0603171 | 0.0576607 | 0.0222085 |
| UBIA | 0.0584381 | 0.0646919 | 0.0587794 | 0.0455274 |
| W2 | 0.0032949 | 0.0038625 | 0.0049113 | 0.0037374 |

### Attribution per margin

**Terminal value origin and first-failing stage are different questions.**
The failed attempt predates the explicit `origin.channel` field, so
`qrf_transfer` is a code-plus-receipt inference, not a claim that the old
origin field existed. The extractor byte-verifies each exporting QRF bank
target and validates the SHA-pinned pool's `stacked_post_puf_transfer` receipt:
for each amount target the only producer role is `puf_clone`, all 964,699
authorized complement rows were imputed, and residual/unmodeled rows are zero
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:1087-1195,1668-1715`).
The live code defines PUF-produced rows as clone>0, sends their complement
through the transfer, preserves producer bytes, and records the completion
equations (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8728-8772,8832-8852,8945-9040`). Together those bindings attribute every
terminal clone-0 value in these eight checks to `qrf_transfer`; the first stage
outside the frozen comparator still differs by criterion:

| leg | incidence first fails | QED first fails | terminal origin channel |
|---|---|---|---|
| `qualified_bdc_income` | late transfer | clone-1 producer | `qrf_transfer` |
| `qualified_reit_and_ptp_income` | late transfer | clone-1 producer | `qrf_transfer` |
| `unadjusted_basis_qualified_property` | late transfer | late transfer | `qrf_transfer` |
| `w2_wages_from_qualified_business` | late transfer | clone-1 producer | `qrf_transfer` |

Thus all four incidence failures and the UBIA QED failure are transfer-first.
The BDC, REIT/PTP, and W2 QEDs are already red on the clone-1 producer
(0.584598, 0.597071, and 0.451897 against the frozen 0.25 ceiling) and are
then amplified in the clone-0 transfer result (0.827384, 1.16264, and
1.33713 before reconciliation). The producer's cross-channel incidence is
in band on every leg (1.047–1.172), while the transfer's drawn ASEC clone-0
incidence tracks the donor level (REIT 0.05766 vs donor 0.05759; UBIA
0.05878 vs 0.05844; BDC 0.00921 vs 0.00855) and the drawn ACS clone-0
incidence is lower than both the donor and the ACS clone-1 producer level
(REIT 0.02221 vs 0.06032; UBIA 0.04553 vs 0.06469; BDC 0.00303 vs
0.00914). The chained qualification booleans drawn by the same models for
the same recipients stay in band on clone-0 (`business_is_sstb` 1.05464,
SSTB qualification 1.05341 pre-reconciliation); their membership in the
same byte-verified chained target bank is explicit
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:297-317,809-995`).
These coarse comparisons
localize the failing stages but do not, by themselves, isolate the causal
feature inside either model.

Stage-specific findings:

1. **All four incidence failures and the UBIA QED are transfer-first.** QBI
   reconciliation rewrites only the SSTB-side splits and never
   the non-SSTB W2/UBIA columns
   (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1324-1341`);
   empirically the clone-0 ratio, QED, carriers, and all five quantiles are
   identical between `transferred` and `simulated` to every reported digit,
   and the row-level reconciliation delta is zero on both channels
   (`evidence.json.amount_legs.*.reconciliation_deltas`). The terminal red
   W2/UBIA values are exactly the transfer's output. UBIA's producer QED is
   green (0.187807), whereas W2's producer QED is already red (0.451897), so
   only UBIA has transfer-first ownership for both criteria.
   - W2 additionally shows the transfer overdrawing ASEC clone-0 carriers
     (0.00491 vs donor 0.00329, +49%) while ACS clone-0 lands near the
     producer level (0.00374): the two channels err in opposite directions.
   - Drawn magnitude shapes collapse against donor support: W2 donor median
     95,201 vs drawn ASEC clone-0 median 17,942 and ACS clone-0 median
     3,564 (QED 1.33713).
2. **`qualified_bdc_income`: transfer-first incidence, producer-first QED;
   reconciliation is level-changing but ratio-neutral.** The exposure cap
   `min(bdc, max(non_qualified_dividend_income, 0))`
   (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1342-1349,1354-1356`) kills 704/932 (75.5%) ASEC and
   2,181/3,044 (71.6%) ACS clone-0 carriers — near-proportionally — so the
   ratio barely moves (0.329192 → 0.328928) while both levels drop ~3.6×.
   The red ratio pre-exists the cap.
3. **`qualified_reit_and_ptp_income`: transfer-first incidence,
   producer-first QED; reconciliation is secondary and ratio-improving.**
   The cap
   `min(reit, max(nqdi,0) + max(partnership_s_corp_income, 0))`
   (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1350-1353,1357-1359`) kills 71.2% of ASEC and 65.3% of
   ACS clone-0 carriers, moving the ratio 0.385159 → 0.446585 — toward the
   band but far outside it.
4. **Producer conditional-shape divergence is part of the current coupled
   defect.** Clone-1 QEDs are 0.584598 (BDC), 0.597071 (REIT), and 0.451897
   (W2) — above the 0.25 ceiling — with UBIA green (0.187807). Although the
   terminal battery evaluates clone 0 for these amount targets (production
   declares the terminal role at
   `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2145-2166`), these three
   QEDs establish that the shape failure is present in the upstream donor
   surface before the transfer. The transfer then magnifies it, so the
   producer and transfer magnitude surfaces must be evaluated as a coupled
   refit rather than assigning those QED failures solely downstream.
5. **The transfer violates coupled QBI support pre-reconciliation.** At
   `transferred`, 2,996 rows carry `qualified_bdc_income` above their
   non-qualified-dividend base and 22,350 rows carry REIT/PTP above the
   dividend+partnership base; all nine coupled invariant counts are zero at
   `simulated` (`evidence.json.invariants`), confirming reconciliation
   enforces the identities the transfer ignored — by destroying ~3.6× of
   BDC/REIT carrier mass on both channels. The transfer does not expose the
   exact coupled bases as separate predictors: non-qualified dividends are
   folded into a broader donor investment aggregate, its recipient analog is
   an ACS aggregate, and partnership/S-corporation income is absent from the
   transfer feature list
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:203-229,2206-2225,2251-2286`).

## Recipient-context constraint on the collapse mechanism

The extraction records a limited set of coarse predictor-source marginals
across the two recipient channels (`evidence.json.recipient_context`,
transferred stage):
employment-income positive share 0.4920 (ASEC clone-0) vs 0.5097 (ACS
clone-0) with medians 50,000 vs 48,529, and self-employment positive share
0.0376 vs 0.0472. These summaries do not cover the joint predictor
distribution, per-row availability-pattern assignment, gate scores, or gate
outcomes, so they cannot establish which model feature causes the
2.6–3.0× drawn-incidence gap. The four availability
patterns (478 / 45,032 / 235,192 / 683,997 of the 964,699 clone-0
recipients; distinguished by observation of the social-security, retirement,
interest/dividend/rental, and tenure-code optional predictors) are fitted
separately because availability codes partition eligible recipients and each
partition gets its own donor mask, QRF fit, and draws
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1542-1703,2152-2171`).
Current origin receipts persist each target's realized regime and canonical
pattern catalog, but their closed fields contain no recipient-level pattern
assignments, gate scores/outcomes, or channel×pattern gate-rate cross-tabs
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`).
A host-owned 1% diagnostic build must add or derive those measurements before
choosing a mechanism-specific refit (`REFIT_PLAN.md`, action 2).

## Realized-regime recomputation (remediation-rule precondition)

Recomputed from frozen donor support with the banked fitter tolerance
(`zero_atol = 1e-06`) via `microcosm.fit.qrf.detect_regime`
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:118-150`), mirroring the
transfer's own recomputation contract
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1512-1532`):

- Both QBI late-producer batches (`puf_tax_itemization__batch_4`, `__batch_5`)
  fit every availability pattern on the **byte-identical frozen donor set**:
  banked donor index SHA-256
  `8ecfa8bcb4a52bdb34654a00a910a698463337d6d10cc2da8750bb29bd47a8d2`,
  108,073 rows, identical across all four patterns of every target step in
  both batches (`evidence.json.bank_patterns`). The reconstructed donor set
  (ASEC clone-1 rows complete for all family targets) matches the banked
  count exactly.
- Realized regimes by target: `zero_inflated_positive` for all four red
  amounts, all six chained qualification booleans, and both SSTB splits;
  `three_sign` only for `sstb_self_employment_income_before_lsr`.
- Consequence: every terminal red amount value passed through a **gated
  zero-inflated-positive** chain step — a sample-weighted zero-vs-positive
  gate followed by a positive-magnitude forest
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:104-105,950-1003,1333-1429`).
  The observed incidence change occurs across that gate, but the historical
  artifacts do not record enough row-level diagnostics to distinguish gate
  calibration, pattern routing, or chained conditioning as its cause; the
  closed current origin-receipt schema shows exactly which aggregate fields
  exist and that those row diagnostics do not
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`)
  (pattern predictors: `age`, `is_female`, `__acs_transfer_state_fips`,
  `__acs_transfer_employment_income`,
  `__acs_transfer_self_employment_income`,
  `__acs_transfer_is_household_head`, plus per-pattern optional additions;
  `evidence.json.bank_patterns`).
- The receipt gap named by the adjudication (regimes returned by the fitter
  but not persisted in the failed attempt's ACS-transfer provenance, bank
  metadata, or public receipts) is closed going forward: target origin
  receipts now carry a closed pattern catalog plus realized regimes, bank
  receipts carry and validate per-pattern regimes, and the post-PUF receipt
  validator requires the origin record
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:350-420,612-780`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8945-9040`).
  This extraction recomputed the failed attempt's regimes exactly as the rule
  requires for historical artifacts.

## SSTB boolean role evidence (supports the separate two-check fix)

- Pre-reconciliation, the transfer's drawn clone-0 SSTB signals are live and
  in band (`business_is_sstb` 1.05464, carriers 3,786/33,512;
  qualification flag 1.05341, carriers 3,743/33,182).
- The derive-stage reconciliation then forces both flags false on every
  native-role row: `support_role_series` assigns the ASEC-compatible role to
  all clone-0 records on **both** channels
  (`packages/microcosm-build/src/microcosm/build/us_runtime/support_provenance.py:370-379`),
  and the reconciliation kernel sets both SSTB flags false on that role mask
  (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1246-1264`), making the terminal clone-0 comparison dead
  by construction (0/0 carriers) — reproduced here.
- The clone-1 signal stays live and in band at terminal:
  `sstb_self_employment_income_would_be_qualified` 0.0350747/0.0372869
  (ratio 1.06307, carriers 3,735/33,902), `business_is_sstb`
  0.0352413/0.0375576 (ratio 1.06573, carriers 3,749/34,166) — matching
  ADJUDICATION.md:69 digit-for-digit — and is already stable at the
  transferred stage (1.06444 / 1.06554).
- This is direct ownership evidence for the adjudicated two-check terminal
  role fix (assign these two targets to clone 1 in the canonical metric
  authority; ADJUDICATION.md:71): the native scope is intentionally dead and
  the PUF-detail clone scope carries the only live, in-band signal.

## Reproduction

```bash
uv run python experiments/qbi_ownership/extract_qbi_ownership_evidence.py
# Debug-only iteration without the ~13 GB digest pass must use a noncanonical output:
uv run python experiments/qbi_ownership/extract_qbi_ownership_evidence.py \
  --skip-sha --output /private/tmp/qbi-ownership-evidence-debug.json
```

The script asserts the stage, publication-manifest, gates, transition-authority,
and target-bank artifact pins, then fails before its atomic replacement on any
disagreement with the adjudicated battery numbers
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:809-914,999-1215,1990-2003`).
A drifted checkpoint, target bank, or comparator therefore cannot silently
reuse this evidence.
