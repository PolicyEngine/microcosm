# PROGRESS — populace #300 (build-level weights audit) — EXTENSION per lead

Branch: `weights-audit` | Worktree: populace-weights-audit
Issue: PolicyEngine/populace#300 | PR: #308 (OPEN, extending in place)

## STATUS (2026-07-04, sole agent)
- [x] Rebased onto post-#306 origin/main (clean, no conflicts; #306 formula-owned
      code present in puf_support.py). Force-pushed.
- [x] STEP 1 DONE: fit-side resolved-weight-kind surface.
      - model.py: `resolved_weight_kind(frame_or_df, entity, weight_values)` +
        `EXPLICIT_WEIGHTS="explicit"`; exported from populace.fit.
      - qrf.py: fit() computes it; FittedRegimeGatedQRF gains read-only
        `weight_kind` property. 7 fit tests green, full populace-fit green.
      - gates.py: RESOLVED_WEIGHT_KINDS now includes "explicit" (lockstep);
        +2 build tests. build gates green.
- [ ] STEP 2: wire PUF-support seam to emit FitWeightRecord + run gate.
- [ ] STEP 3: non-gated synthetic prove-it-can-fail test + PE-US-gated real path.
- [ ] Full suite + ruff (pipefail); PR body boundary update; delete PROGRESS.

## Where we are
- DONE + PUSHED: `FitWeightRecord` + `weights_audit_gate` in build/gates.py,
  exported, 13 tests, full suite + ruff green (commits f238afc, c7b981b).
- Lead ruling: gate is well built but an UNWIRED gate is dead code. EXTEND this
  PR: (1) fit-side resolved-kind property, (2) wire the PUF-support production
  seam to emit records + run the gate, (3) minimal beyond that, update PR body.

## Extension design (locked)
### 1. populace-fit — expose resolved weight kind on the fitted estimator (TDD)
- `resolve_fit_weights` (Frame) already knows the resolved kind: it is
  `frame.resolve_weights(entity).kind` (a WeightKind, reflecting INHERITED
  weights — the issue's word "resolved") when weighted, or `"none"` when
  `weights="none"`.
- Add a sibling that returns BOTH the vector and the resolved-kind STRING so
  `fit()` can stash it, without changing the existing single-return callers.
  Plan: new `resolved_weight_kind(frame, entity, weights) -> str` (Frame) and
  the DataFrame analogue, OR return a small tuple. Chosen: a thin
  `resolve_fit_weights_kind(...)` returning the kind string, called alongside
  the existing resolver in `fit()`. Keeps existing resolver signature intact
  (public API, used by puf_support indirectly via QRF).
- Vocabulary of the resolved kind:
  - Frame weighted: `"design" | "importance" | "calibrated"` (WeightKind.value).
  - Frame or DataFrame unweighted (`weights="none"`): `"none"`.
  - DataFrame explicit vector/column: `"explicit"` — weighted, but no kernel
    WeightKind provenance. Honest; NOT one of the typed kinds. The audit treats
    any non-`none` kind as weighted, so `"explicit"` passes the gate correctly.
- Store on `FittedRegimeGatedQRF` as a read-only `weight_kind: str` property,
  set in `fit()`. No behavior change (weights still resolved/materialized
  exactly as today) — additive.
- Extend build-side `RESOLVED_WEIGHT_KINDS` to include `"explicit"` so the two
  vocabularies stay in lockstep (build gate accepts what the fit can emit).

### 2. Wire the PUF-support production seam
- Real seam: `impute_us_puf_tax_detail_support` (puf_support.py) does
  `QRF(...).fit(donor_frame, ..., weights="design")` on a Frame with
  WeightKind.DESIGN → resolved kind `"design"`.
- Make `impute_us_puf_tax_detail_support` capture `fitted.weight_kind` and
  expose a `FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, fitted.weight_kind)` so the
  build tool can record it. Minimal seam: return the record, OR attach to a
  small result. Chosen: add a module-level fit name constant + have the impute
  fn return the record via a new optional out-param is ugly; instead expose a
  helper `us_puf_support_fit_weight_record(fitted)` and thread the fitted
  model's kind out. Simplest that keeps signature stable: the impute fn already
  builds `fitted`; add the record to the returned Frame? No — Frame is typed.
  DECISION: change `impute_us_puf_tax_detail_support` to ALSO record the fit by
  appending to a caller-provided list is stateful. Cleanest: expose the fit name
  constant `US_PUF_SUPPORT_FIT_NAME` and, in `build_us_puf_support_base.py`,
  after the impute, we don't have the fitted object. So: have the impute fn
  return `(Frame, FitWeightRecord)` — but that breaks its callers/tests.
  FINAL: keep impute returning Frame; expose the resolved kind by having the
  impute attach it to the frame's mass_log? No. Instead: the build TOOL is the
  production seam that must emit + gate. Give puf_support a thin public fn
  `impute_us_puf_tax_detail_support_with_audit(...) -> tuple[Frame,
  FitWeightRecord]` that wraps the fit, returns the record. Existing
  `impute_us_puf_tax_detail_support` stays byte-for-byte. Build tool calls the
  audited variant, records the FitWeightRecord, runs `weights_audit_gate`,
  writes details into the summary, and aborts on failure (like the geography
  gate does). ← verify against actual code before coding; adjust if a cleaner
  seam exists (e.g. the fit already isolated in a helper).

### 3. Tests
- Non-gated (no policyengine_us): the wiring SHAPE on a synthetic Frame —
  audited-impute returns a `FitWeightRecord` with kind `"design"`, and
  `weights_audit_gate([record])` passes; a forced `"none"` record fails. This
  is the "prove it can find something" test that does not need the engine.
- PE-US-gated (importorskip policyengine_us) where the real build path needs
  the engine.
- Keep every new import that pulls policyengine_us behind importorskip.

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
