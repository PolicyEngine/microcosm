# PROGRESS — populace #300 (build-level weights audit) — FINAL INCH per lead

Branch: `weights-audit` @ f3542d7 (pushed) | Worktree: populace-weights-audit
Issue: PolicyEngine/populace#300 | PR: #308 (OPEN, extending in place)

## DONE + PUSHED (prior sessions, all verified green)
- STEP 1: `resolved_weight_kind` in populace-fit.model + read-only
  `FittedRegimeGatedQRF.weight_kind` property. gates RESOLVED_WEIGHT_KINDS has
  "explicit" (lockstep).
- STEP 2: seam = opt-in `fit_records` out-param on
  `impute_us_puf_tax_detail_support` (append FitWeightRecord when a list passed;
  byte-for-byte unchanged when omitted). US_PUF_SUPPORT_FIT_NAME re-exported.
- STEP 3: synthetic wiring tests + PE-US-gated real-engine test (verified green
  under real engine via `uv run --with policyengine-us`).
- Full suite + ruff pipefail exit 0; CI green after one ESS-flake rerun.
- PR body updated once.
- Commits: 86edf4a (seam), d48eb4d (gated test), f3542d7 (PROGRESS removal).

## LEAD RE-REVIEW RULING (the reason this PROGRESS is back)
`grep fit_records tools/` was EMPTY: `tools/build_us_puf_support_base.py:200`
calls `impute_us_puf_tax_detail_support` WITHOUT the sink — so no production
build records anything and `weights_audit_gate` never runs in a release. Seam
without caller = dead code one layer down. THE LAST INCH: wire the base build.

## STATUS THIS SESSION
- [x] tool wired: extracted `impute_and_audit_us_puf_support(expanded, donor, *,
      seed, n_estimators, predictors=..., person_outputs=..., tax_unit_outputs=...)`
      in build_us_puf_support_base.py. Creates fit_records, calls impute with the
      sink, runs weights_audit_gate, `raise SystemExit` on failure (mirrors the
      geography-ladder gate), returns (imputed, {"passed","failures","details"}).
      main() line ~200 now calls it; summary carries `"weights_audit"`. The
      *_outputs/predictors passthrough (defaults = production set) exists ONLY so
      the seam is testable engine-free; production calls with defaults → unchanged.
- [x] tests added to test_us_puf_support_base_builder.py (TestBaseBuildWeightsAudit):
      records "design" in summary details, JSON round-trips, and (prove-it-fails)
      a monkeypatched "none" fit makes the helper raise SystemExit naming the fit.
      Engine-free (formula-owned guard degrades to static seed w/o policyengine_us).
      All 12 builder tests green; ruff clean on tool+test.
- [ ] Full suite + ruff pipefail exit 0; PR body boundary; push; CI green; rm PROGRESS.

## PLAN (this session)
1. tools/build_us_puf_support_base.py:
   - Extract a small TESTABLE helper `impute_and_audit_us_puf_support(expanded,
     donor, *, seed, n_estimators) -> tuple[Frame, dict]` that:
       * creates `fit_records: list[FitWeightRecord] = []`
       * calls `impute_us_puf_tax_detail_support(..., fit_records=fit_records)`
       * runs `report = weights_audit_gate(fit_records)`
       * `if not report.passed: raise SystemExit("Weights audit failed:\n  " +
         "\n  ".join(report.failures))`  (mirrors the geography-ladder gate at
         lines 262-265 — a failing gate fails the tool with nonzero exit)
       * returns `(imputed, {"passed", "failures", "details"})` where details is
         the gate's `to_manifest`-style dict (dict(report.details), carrying
         resolved_weight_kinds).
   - main() line ~200 calls the helper instead of the bare impute; threads the
     returned imputed frame onward; adds `"weights_audit": weights_audit` to the
     `summary` dict (alongside geography_ladder_assignment etc.).
   - Import weights_audit_gate + FitWeightRecord from populace.build.
   The audit is UNCONDITIONAL (the PUF fit always runs at line 200), unlike the
   optional geography ladder. No allowed_unweighted (the fit is design-weighted).
2. TEST (packages/populace-build/tests/ or tools test dir — find where tool tests
   live; if none, add tests/test_build_us_puf_support_base.py importing the
   helper). Non-engine synthetic frame (reuse the _minimal_us_frame pattern):
   assert the returned dict["details"]["resolved_weight_kinds"] ==
   {US_PUF_SUPPORT_FIT_NAME: "design"} and dict["passed"] is True. Engine-gate
   ONLY if the helper path needs the engine (the impute itself does NOT — the
   formula-owned guard degrades to static seed without policyengine_us — so this
   test can be engine-FREE, same as TestPufSupportWeightsAuditWiring). Confirm.
3. PR body: boundary line now says the PUF-support BASE BUILD records + enforces
   end-to-end (not just the seam).
4. Full suite + ruff pipefail exit 0; push; CI green (1 flake rerun allowed);
   delete PROGRESS in final cleanup.

## Notes carried
- Repo has NO changelog system → no fragment (verified; org towncrier N/A here).
- CI omits [us] extra; engine-gated paths importorskip. Tool build path itself is
  engine-free for the impute+audit (write_dataset needs engine, but that is NOT
  in the extracted helper).
- Do NOT merge (Max merges after lead review). PR via gh, no @-mentions.
- Known ESS flake: wheels/test may redden on #307 test_refit_l0_selection...;
  rerun once, note if recurs, don't chase.
