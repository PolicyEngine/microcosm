# Coupled QBI amount-surface refit plan

Grounded in `EVIDENCE.md` / `evidence.json` (this directory): stage-attributed
ownership for the four red QBI amount legs of adjudication workstream 5, the
recomputed realized regimes, and the nine-invariant reruns. Nothing here
tunes a gate, band, ceiling, fold, or seed; every action is a model/data or
receipt change to be verified against the frozen comparator.

## Evidence-derived defects the refit must address

1. **Transfer-first incidence failures, plus the UBIA QED failure.** One QRF
   per availability pattern is fit on the 108,073
   ASEC clone-1 donors and fills every clone-0 row on both channels
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1542-1703,2152-2171`;
   `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8784-8920,8974-8994`). Drawn carrier incidence tracks
   the donor level on ASEC clone-0 recipients but collapses on ACS clone-0
   recipients (REIT 0.0222 vs 0.0603 ACS-producer; UBIA 0.0455 vs 0.0647;
   BDC 0.0030 vs 0.0091), while the BDC/REIT batch's chained models keep the
   qualification booleans in band on the same recipients. The realized
   regime for every red amount is `zero_inflated_positive`; that regime fits a
   sample-weighted zero-vs-positive gate and draws its predicted sign before a
   positive-magnitude forest
   (`packages/microcosm-fit/src/microcosm/fit/qrf.py:92-150,950-1003,1333-1442`),
   so the failing incidence transition crosses that gate,
   but the historical receipts do not identify whether gate calibration,
   availability-pattern routing, or chained conditioning causes it (three
   required predictors plus three empirically always-observed optional
   predictors, followed by the pattern-specific optional predictors;
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:173-181,2206-2286`;
   `evidence.json.bank_patterns`). Current closed origin receipts persist the
   pattern catalog and realized regimes but not row assignments, gate scores,
   or gate outcomes
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`).
2. **The transfer does not condition on the exact coupled exposure bases.**
   At the transferred stage 2,996 rows carry `qualified_bdc_income` above
   `max(non_qualified_dividend_income, 0)` and 22,350 rows carry REIT/PTP
   above the dividend+partnership base; the derive-stage caps
   (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1342-1359`) then kill 65–76% of drawn carriers across the two
   targets and channels. The transfer feature surface includes non-qualified dividends
   only inside a broader donor investment aggregate, maps recipients through
   an ACS investment aggregate, and does not include partnership/S-corporation
   income, so it never observes either exact cap base as its own predictor
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:203-229,2206-2225,2251-2286`).
3. **W2 ASEC-recipient overdraw.** The transfer draws 49% more W2 carriers
   on ASEC clone-0 than the donor support carries (0.00491 vs 0.00329),
   while the ACS side lands near the producer level — the two channels err
   in opposite directions on one fitted gate.
4. **Coupled producer/transfer conditional-shape divergence.** Clone-1 QEDs
   at the transferred stage are already red for BDC 0.5846, REIT 0.5971,
   and W2 0.4519 (UBIA is green at 0.1878). Their clone-0 transfer QEDs then
   worsen to 0.8274, 1.1626, and 1.3371 before reconciliation. These three
   criteria are producer-first and transfer-amplified, so producer shape is
   part of this workstream's current coupled refit rather than a deferred
   follow-up.

## Refit actions

Ordered; each action names its verification. The comparator bands, QED
ceiling, quantile set, and support floor stay frozen throughout.

1. **Couple the amount draws to their exposure bases (defect 2).** Add the
   observed exposure-base columns (`non_qualified_dividend_income`,
   `partnership_income + s_corp_income`) to the QBI batch predictor surface
   and use a bounded output construction (for example, draw a share of the
   available positive base), or otherwise restructure batch_4 so BDC/REIT
   amounts cannot exceed that base. Adding a predictor alone does not prove
   the coupled identity. Introduce the QBI-specific surface at the
   family-fit/feature-surface
   seam; today the feature surface is selected for each family fit but is
   shared across person families except for the explicit housing special case
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1542-1578,2174-2225`).
   This is a structural predictor change, not a band change. Verification:
   post-transfer exposure-invariant violation counts (currently 2,996 /
   22,350) reach zero and the derive cap becomes a near-no-op, on a 1% build
   first, then 25%.
2. **Instrument the host-owned 1% diagnostic before selecting a gate fix
   (defects 1 and 3).** The available coarse marginals include
   employment-income positive share 0.492 ASEC vs 0.510 ACS clone-0 and
   self-employment positive share 0.038 vs 0.047
   (`evidence.json.recipient_context`). They do not describe the joint
   predictor distribution and cannot establish a cause. The 964,699
   clone-0 recipients split into four patterns (478 / 45,032 / 235,192 /
   683,997 rows) fit as four separate models distinguished by which of
   `__acs_transfer_social_security_income`, `__acs_transfer_retirement_income`,
   `__acs_transfer_interest_dividend_rental_income`, and
   `__acs_transfer_tenure_code` are observed (`evidence.json.bank_patterns`)
   — pattern membership is availability-driven because the transfer encodes
   each recipient's finite optional predictors into a pattern code
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2152-2171`).
   Current closed origin receipts contain the realized regime and pattern
   catalog, but not per-row pattern assignments, gate scores/outcomes, or a
   channel×pattern×gate cross-tab
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`).
   Add those aggregate diagnostics to the
   host 1% build receipt (or derive them from its artifact), then compare
   donor gate rate with drawn rate by target, channel, and pattern. Candidate
   causes to discriminate are: (a) channel-correlated pattern routing;
   (b) sample-weighted gate calibration interacting with donor weights; and
   (c) chained-boolean conditioning. The fitter weights the gate directly by
   donor `sample_weight` and chains each predicted target into later target
   features (`packages/microcosm-fit/src/microcosm/fit/qrf.py:1333-1442,1499-1531`).
   Apply only the structural fix supported by that evidence, with no channel
   special case.
3. **Refit producer and transfer magnitude shape together (defect 4).** The
   BDC, REIT/PTP, and W2 QED failures first appear on clone 1 and worsen on
   clone 0. The 1% diagnostic must therefore compare producer and transfer
   quantiles for the same target and pattern before changing either surface.
   A supported refit may change predictors, conditioning, or model/data
   structure, but must not tune the frozen QED ceiling, bands, folds, or
   seeds toward passing. Acceptance requires both the upstream clone-1
   shape evidence and the terminal clone-0 criteria to improve without a new
   red criterion.
4. **Rerun whole-pool reconciliation and the coupled identities after any
   transfer change.** The derive-stage reconciliation and the nine
   invariants at `1e-8` are the acceptance harness: all nine must stay
   zero at terminal. Extend the change receipt with target-level
   positive-to-zero/carrier deltas, or derive those deltas in the host
   diagnostic, to show that the caps no longer destroy drawn carrier mass
   (defect 2's verification); the current aggregate receipt alone cannot.
   The production summary computes the same nine identities after applying
   the whole-pool SSTB splits and BDC/REIT caps
   (`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1324-1359,1377-1487`),
   and the extraction script in this directory reruns both checkpoint stages
   without a build.

## What ran here vs. what needs the host queue

Safe work completed in this lane (read-only analysis and receipt/test
infrastructure, within the 15 GiB lane budget):

- Whole-pool reconciliation delta measurement, exposure-cap attribution,
  producer/transfer stage decomposition, and the nine-invariant rerun on
  both stages of the SHA-verified failed-attempt checkpoint
  (`extract_qbi_ownership_evidence.py`).
- Realized-regime recomputation from frozen donor support for every
  availability pattern of both QBI batches, cross-checked against banked
  donor-index digests.
- Persistence and validation of realized regimes and origin channels in
  future transfer receipts
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:509-555,582-671`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:350-420,612-780`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4025-4153,4178-4208`).
  No amount-model refit, gate change, or pool build was performed in this
  lane.

Needs the host build queue (NOT run in this lane):

- **The 1% diagnostic and demonstration build** of actions 1–3. Start from
  the existing gated baseline at
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/baseline1pct/`
  for before/after comparison. The build must add or derive target×channel×
  availability-pattern gate counts/rates and producer/transfer quantiles;
  the present receipts alone cannot supply those cross-tabs. The build
  tool's own
  contract expects the 1% rung to take "roughly 2–4 hours" within "a 64 GiB
  memory envelope" (`tools/build_us_multispine_pool.py:8-9,18-19`) because PUF
  donors stay full at every rung; that envelope exceeds this lane's binding
  15 GiB RSS cap. The host queue owns the build, which must remain off-chain
  and must not use `--logbook-prev-row-digest`.
- **The 25% host certification run** only after the instrumented 1% rung is
  clean: full battery
  with the frozen tolerances, whole-pool reconciliation receipts, the nine
  invariants, and the role-aware SSTB clone-1 checks. Acceptance for
  workstream 5 is the eight amount checks (four legs × incidence+quantile)
  leaving the red set with no new red introduced, per the adjudication's
  minimal-set rule.
