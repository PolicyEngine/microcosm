# PROGRESS — populace #300 (build-level weights audit) — EXTENSION per lead

Branch: `weights-audit` | Worktree: populace-weights-audit
Issue: PolicyEngine/populace#300 | PR: #308 (OPEN, extending in place)

## STATUS (2026-07-04, sole agent — resumed after connection death)
- [x] Rebased onto post-#306 origin/main — VERIFIED this session:
      merge-base(HEAD, origin/main) == origin/main HEAD 7e9a32d; #306 formula
      guard (`resolve_formula_owned_outputs` etc.) present in puf_support.py.
- [x] STEP 1 DONE + re-verified reading code this session:
      - model.py: `resolved_weight_kind(frame_or_df, entity, weight_values)` +
        `EXPLICIT_WEIGHTS="explicit"`; exported from populace.fit.
      - qrf.py: fit() computes `weight_kind` (line ~464); FittedRegimeGatedQRF
        has read-only `weight_kind` property (line ~657).
      - gates.py: RESOLVED_WEIGHT_KINDS includes "explicit" (lockstep with fit).
- [~] STEP 2 IN PROGRESS: seam is the OPT-IN `fit_records` out-param on
      `impute_us_puf_tax_detail_support`. Design is PINNED by predecessor tests
      (test_us_puf_support.py::TestPufSupportWeightsAuditWiring, lines 1163-1247,
      ALREADY WRITTEN and committed): when a `fit_records` list is passed, the
      impute appends `FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, fitted.weight_kind)`;
      when omitted it is byte-for-byte unchanged. This resolves PROGRESS's earlier
      wrapper-vs-out-param deliberation in favor of the out-param (existing
      callers untouched; no second public fn).
      SUB-STEPS:
        (a) puf_support.py: add `fit_records` kwarg; append record after fit. TODO
        (b) us_runtime/__init__.py: re-export US_PUF_SUPPORT_FIT_NAME. TODO
      NOTE: impute has NO non-test production caller in-repo (only the __init__
      re-export + tests). The out-param IS the production seam a future build
      stage uses to feed weights_audit_gate; the "real build manifest carries
      resolved_weight_kinds and a none fit fails" surface is proven by the
      synthetic wiring tests (engine-less) plus a PE-US-gated real-path test.
- [x] STEP 3 DONE: synthetic wiring tests were ALREADY WRITTEN (incl. prove-it-
      can-fail `test_wired_gate_would_fail_a_none_fit`). ADDED
      `test_production_fit_records_design_kind_under_the_real_engine`
      (importorskip policyengine_us): runs the audited impute with the LIVE
      formula-owned metadata guard active over real leaf outputs; asserts a
      "design" record + passing gate. VERIFIED it actually PASSES (not just
      skips) via `uv run --with policyengine-us pytest ...` — green with engine.
      Skips locally/CI (no [us] extra) but is proven runnable.
- [ ] Full suite + ruff (pipefail); PR body boundary update; delete PROGRESS.
      Commits: 86edf4a (seam) + this step (gated test).

## Extension design (locked — as implemented)
### 1. populace-fit resolved-kind surface — DONE (see STEP 1 above).
### 2. Seam = opt-in `fit_records` out-param on `impute_us_puf_tax_detail_support`
- After `fitted = QRF(...).fit(donor_frame, ..., weights="design")` the impute
  has `fitted.weight_kind` ("design" for this DESIGN-weighted Frame fit).
- `if fit_records is not None: fit_records.append(FitWeightRecord(
  US_PUF_SUPPORT_FIT_NAME, fitted.weight_kind))`.
- Opt-in so every existing caller (all pass nothing) is unaffected — same Frame
  return, no new required arg. A future build stage passes a shared list, then
  runs `weights_audit_gate(fit_records)` and writes its details into the release
  summary / aborts on failure (the geography-gate pattern).
### 3. Tests
- Synthetic (no policyengine_us): TestPufSupportWeightsAuditWiring — records a
  "design" kind, gate passes, forced "none" fails (prove-it-can-find-something),
  and no-sink call is unaffected. ALREADY WRITTEN.
- PE-US-gated (importorskip policyengine_us): TO ADD — real engine frame path.

## Scope discipline
- No other fits. No declarative-config work (issue item 2 stays deferred).
- PR body boundary paragraph: recording now END-TO-END for the PUF-support fit;
  remaining follow-up is only declarative-config + future fits.

## Verify at the end
- `set -o pipefail; uv run pytest` exit 0; `uv run ruff check .` exit 0.
- Push; re-check CI green. Delete this PROGRESS.md in final cleanup.

## Notes carried
- Repo has NO changelog system → no fragment (org CLAUDE.md towncrier does NOT
  apply here; predecessor + merged-PR evidence confirm).
- CI `uv sync --all-packages` omits `[us]`; policyengine_us absent (verified
  locally too) → PE-US paths importorskip.
- Do NOT merge (Max merges after lead review). PR via gh, no @-mentions.
- Known flake: wheels(3.13) may redden on the #307 ESS test — rerun once; if it
  re-fails on that specific test, note it (being deflaked separately), don't chase.
