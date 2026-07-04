# PROGRESS — populace #300 (build-level weights audit)

Branch: `weights-audit` | Worktree: populace-weights-audit
Issue: PolicyEngine/populace#300

## State
- Exploration DONE. Design locked. Writing failing test (TDD).

## Design (locked)
Add a release gate `weights_audit_gate(fit_records, *, allowed_unweighted)` to
`packages/populace-build/src/populace/build/gates.py`, plus a small frozen
dataclass `FitWeightRecord(fit_name, weight_kind)`. This is the auditable,
build-level enforcement point the issue's item 1 names:
- **Records** the resolved weight kind per named production fit into
  `GateResult.details` (the manifest surface — DESIGN.md "stage manifests are
  load-bearing"; `GateReport.to_manifest()` already serializes details).
- **Fails the release** when any fit's resolved kind is `"none"` (unweighted)
  and it is not in the `allowed_unweighted` allowlist.
- `allowed_unweighted` reuses the `_reviewed_exclusion_reasons` idiom already in
  gates.py: every entry needs a non-empty reason; unused entries are reported so
  the register cannot rot (matches every other gate here).
- `weight_kind` vocabulary = the resolved-kind strings `resolve_fit_weights`
  already keys on: `"design" | "importance" | "calibrated"` (WeightKind values)
  + `"none"`. Any other string is refused (a fit with an uninterpretable weight
  kind is unauditable — same principle as support_gate's missing-range failure).

Export from `populace.build.__init__` alongside the other gates.

## Scope boundary (deliberate; in final report)
- This PR is the GATE + its manifest surface. Threading each production fit to
  *emit* a `FitWeightRecord` (e.g. out of `impute_us_puf_tax_detail_support`
  into the puf-support summary) is the larger follow-up the issue's item 1
  implies ("every production fit records ..."). The gate is the guardrail the
  microplex-263 regression class needs; per-fit emission is additive.
- NOT added to `country_spec.ALLOWED_GATE_FUNCTIONS`: that registry is the
  declarative allowlist for per-frame-evidence gates a country spec selects; the
  weights audit consumes recorded fit records, not a frame, so listing it there
  would imply a capability the executor does not have. Reversible.

## Semantic questions deferred (guardian-owned → final report)
- Which production fits, if any, may legitimately be unweighted (i.e. what the
  seed allowlist should contain). Today every production path fits weighted
  (`weights="design"` in puf_support); the allowlist ships empty.
- Whether `FittedRegimeGatedQRF` should expose its resolved `weight_kind`
  directly (fit-side change in populace-fit) so the build can read it rather
  than the caller asserting it. That is a populace-fit API decision; flagged.

## Plan (TDD — failing test first)
1. [x] Read spec + repo. Spec matches plan.
2. [x] Locate fit weight-kind resolution (fit/model.py resolve_fit_weights;
       FittedRegimeGatedQRF does NOT record kind today) + gate/manifest home
       (build/gates.py; GateReport.to_manifest).
3. [ ] Failing test: tests/test_gates.py::TestWeightsAuditGate.
4. [ ] Implement weights_audit_gate + FitWeightRecord; export.
5. [ ] Full `uv run pytest` green (set -o pipefail) + `uv run ruff check .`.
6. [ ] Delete this file, push, open PR (--body-file, no @-mentions, no merge).

## Notes
- Repo has NO changelog system (no changelog.d/, no towncrier). No fragment.
- CI: `uv sync --all-packages` omits the `[us]` extra → policyengine_us absent;
  PE-US-gated tests must importorskip. (The gate + its test need NO
  policyengine_us — pure evidence in, GateResult out.)
- ruff: line-length 88 but E501 ignored; select E,F,I,N,W,UP,B. py313 target.
