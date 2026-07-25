# Final report: floor-aware SSI take-up prior

## Outcome

Implemented the populace#507/#508 assignment-law correction on
`ssi-floor-aware-prior`, branched directly from `origin/main` at
`0ac342270633740dbfc4dd8bde6eeb4b1c0430e0`.

The one-shot source-identity Bernoulli assignment, stable draw, SSA targets,
calibration losses, and absence of an in-process reconcile loop are unchanged.
No target, loss-shaping term, or calibration knob was added. Nothing was
pushed and no PR was opened.

## Prior shipped

For target `T`, candidate capacity `C`, forced reporter candidate floor `F`,
and drawable candidate capacity `D = C - F`, the feasible unsaturated prior is:

```text
p = (T - F) / (C - F)
E[selected candidate mass] = F + p(C - F) = T
```

Attempt 5's exact inputs now produce:

```text
T = 3,905,779
C = 4,840,938.516496049
F = 2,169,544.5101818806
p = 0.6499357585269396
E[selected] = 3,905,779.000000
```

The old `T / C` prior was `0.8068226825626091` and, under the actual anchored
law, expected `4,324,885.7885` recipients (`+10.73043%`). In the deterministic
1,000-draw regression, the fixed prior delivers `3,908,622.0083`
(`+0.07279%`, inside the 5% gate), while the old prior delivers
`4,317,345.2913` (`+10.53737%`, outside it).

## Explicit edge behavior

- `0 < F < T < C`: `p = (T - F) / (C - F)`, status `floor_aware`.
- `F > T`: `p = 0`, status `reporter_floor_exceeds_target`, and
  `anchor_excess = F - T`. An enforced adult band hard-fails as a
  reporter-identification/support defect even when the excess is inside the
  ordinary delivery tolerance. A draw cannot shrink forced reporters.
- `F == T`: `p = 0`, status `reporter_floor_meets_target`. This is exact, not
  over-delivery, and passes the delivery gate when selected mass equals the
  floor, including the boundary `T == F == C`.
- `F == C < T`: `p = 0`, status `no_drawable_candidate_capacity`; only
  reporters remain selected. An enforced band hard-fails because no Bernoulli
  draw can close the shortfall.
- `0 <= F < C <= T`: the existing nonconstant saturation fallback remains
  `p = F / C`, status `saturated_reporter_rate`. `saturated` now consistently
  includes equality (`C <= T`). Enforced adult bands hard-fail as insufficient
  modeled candidate support; under-18 remains explicitly fenced.
- `C == 0`: `p = 0`, status `zero_candidate_capacity`. The artifact separately
  records `empty_band`, so a truly empty band is distinguishable from a
  populated band with no candidates. The assignment-integrity gate names zero
  capacity and fails.
- `F > C`, negative capacity/floor, nonfinite inputs, or any negative drawable
  denominator raise an explicit `ValueError`; nothing is clamped.

`target_shortfall` still means realized `max(T - selected, 0)`.
`anchor_excess` still means current-weight `max(F - T, 0)`. The new status and
drawable-capacity fields expose the arithmetic regime without changing those
meanings.

Run-level retry guidance is also fail-closed: a prior-weight-basis retry is
advertised only for ordinary delivery misses with no structural-support or
invalid-diagnostics blocker. Mixed failures retain their evidence but do not
recommend a retry that cannot clear the run.

## Determinism and artifact contract

The source-keyed BLAKE2 draw and assignment law remain:

```text
selected(source) = reporter_anchor(source) OR stable_draw(source, seed) < p
```

Thus a fixed seed reproduces the same flags, and increasing `p` produces a
superset for fixed support and seed. There is still exactly one draw per
source identity and no count matching or reconciliation pass.

The SSI diagnostics artifact is schema 4. New band fields are:

- `empty_band`
- `drawable_candidate_capacity`
- `assignment_prior_status`
- `prior_basis_drawable_candidate_capacity`
- `prior_status_recomputed_from_current_weights`

Current artifacts must satisfy schema 4. The prior-basis loader deliberately
continues to accept schema 2, 3, and 4 artifacts because all three carry the
release-weight `candidate_capacity` and `reporter_candidate_floor` needed by
the corrected arithmetic. The local literal pins, builder fixtures, source
contract, CLI help, and tests were updated. There is no analogous
cross-package SSI schema lockstep pin; the cross-package calibration
diagnostics lockstep does not cover this artifact.

## `--ssi-take-up-prior-weight-basis` assessment

The flag remains present and correct. Its artifact loader passes both recorded
`C` and `F` into the same floor-aware formula, and
`test_release_artifact_loader_basis_drives_floor_aware_band_priors` exercises
the loader-to-assignment chain.

The floor-aware law removes the cross-attempt ladder's use as compensation for
forced-floor double counting. It does not, by itself, eliminate the separate
pre-calibration versus post-calibration weight-basis drift documented by
populace#507/#508: the release weights can still change both `C` and `F` after
the one-shot flags freeze. Therefore the flag should not be retired solely in
this PR. Retirement should be a separate, evidence-based change after release
builds show that current-frame bases reliably satisfy the adult delivery gate.

## Effect on the 65+ band

The 65+ assignment is affected whenever its basis has `0 < F < T < C`.
Algebraically:

```text
p_old - p_new = F(C - T) / [C(C - F)] > 0
```

So, for fixed support, seed, and basis, the new 65+ threshold is lower and its
selected source-identity set is a subset of the old one. The final
post-calibration delivered mass is not guaranteed to move monotonically,
because changed flags feed the simulations and the calibration weights can
respond. Consequently the current `+1.0%` delivered result cannot be projected
numerically from the prior alone; a release rebuild must determine its new
landed value and whether it remains inside the ±5% gate. The assignment-basis
expectation itself is now exactly the target instead of including the old
forced-floor overshoot.

## Test coverage

Key new or updated regressions include:

- `test_attempt_5_floor_aware_prior_hits_expected_target`
- `test_attempt_5_seeded_delivery_removes_floor_blind_overshoot`
- `test_reporter_floor_above_target_never_drops_an_anchor`
- `test_reporter_floor_equal_target_is_explicit_and_exact`
- `test_reporter_floor_equal_target_and_capacity_passes_delivery_gate`
- `test_zero_candidate_capacity_is_named_and_fails_the_gate`
- `test_empty_age_band_is_named_and_fails_the_gate`
- `test_no_drawable_candidate_capacity_is_named`
- `test_target_equal_capacity_uses_explicit_saturated_fallback`
- `test_band_prior_rejects_invalid_or_negative_drawable_capacity`
- `test_assignment_is_deterministic_source_keyed_and_seed_sensitive`
- `test_fixed_seed_selections_are_monotone_in_floor_aware_prior`
- `test_release_artifact_loader_basis_drives_floor_aware_band_priors`
- `test_prior_basis_loader_accepts_schema_4_3_and_2_artifacts`
- `test_delivery_gate_names_saturated_enforced_support_inside_tolerance`
- `test_mixed_structural_and_ordinary_misses_are_not_retryable`

Existing assertions and source contracts that encoded the old `T / C`
expectation were corrected explicitly; no test or tolerance was weakened.

## Verification

The required pytest command was unpiped, and its return code was captured
immediately:

```text
uv run pytest packages/populace-build/tests/ -q
PYTEST_RC=0
```

The first full run returned `PYTEST_RC=1` only because three parameterizations
of an existing terminal-gate test double omitted the newly explicit
ordinary-miss retry classification. The fixture was corrected without changing
production behavior (`c6b5b55`), its focused cases returned `FOCUSED_RC=0`,
and the complete suite then returned `PYTEST_RC=0`. After a final independent
audit found the exact-boundary and mixed-failure interactions, their focused
regressions returned `FOCUSED_EDGE_RC=0` and the complete suite was rerun from
the committed tree, again with final `PYTEST_RC=0`.

Final lint and whitespace gates:

```text
uv run ruff check <all six changed Python files>
RUFF_CHECK_RC=0

uv run ruff format --check <all six changed Python files>
RUFF_FORMAT_RC=0

git diff --check
DIFF_CHECK_RC=0
```

Pytest emitted only existing non-failing warnings from target-support overflow,
joblib core detection, pandas categorical deprecation, and the legacy
snap-to-observed comparison.

## Brief corrections and commits

The supplied attempt-5 diagnosis and numerical correction were accurate. Three
qualifications emerged:

1. `T == F` is exact rather than reporter over-delivery; only `F > T` is an
   unavoidable excess.
2. Floor awareness removes the floor-double-counting reason for a ladder, but
   does not mathematically remove post-calibration weight-basis drift, so flag
   retirement needs separate evidence.
3. The repository has no cross-package lockstep pin for the SSI artifact
   schema; only the SSI-local pins and consumers required updates.

Committed implementation history:

- `97bafd5` — track the clean baseline and work plan.
- `8c14bed` — ship the floor-aware law, diagnostics, contracts, and tests.
- `c6b5b55` — keep the ordinary-miss terminal fixture retryable.
- `7197aa0` — make exact-boundary and mixed-failure gate remedies truthful.

No commit was pushed and no PR was opened.
