# Final report: populace #462 critical-row loss alignment

## Outcome

Built `loss-contract-alignment` from `origin/main` at `c3e378a`. The US
release builder now fails on every row class enforced by the populace-data
publish contract, and the calibration objective gives those same critical rows
a configurable final loss-weight overlay. Target values, tolerances, and
unrelated gate semantics are unchanged.

The change addresses the observed stable equilibrium: Build N attempt 5 passed
the builder gates with national medical/dental expense at +21.2% against the
publish contract's 15% limit, while attempt 6 reduced overall loss from
0.03009 to 0.02824 but moved that row only to +20.78%.

## Register alignment

Before this branch, the builder had 13 exact critical requirement classes plus
the existing Table 1.4 blanket. The publish contract had 16 exact/semantic
classes plus that blanket. The builder was missing:

| Requirement class | Newly builder-gated exact rows | Maximum absolute relative error |
| --- | ---: | ---: |
| Itemized deduction amount | 2 | 0.15 |
| State and local tax deduction amount | 2 | 0.10 |
| Medical expense deduction amount | 1 | 0.25 (adjudicated #490; restores to 0.15 once a boosted run holds it) |

The complete register now lives in
`populace.data.us_critical_targets` as dependency-light frozen data and is
imported by both the builder and publication contract. This preserves the
existing 0.15 credit class and the existing 0.25 Table 1.4 blanket as well as
the other established tolerances.

Matching is aligned across both gates:

- exact compiled row identity, including `@2024`;
- `family + target_role` aliases for live Table 1.2, 2.1, and 4.3 variants;
- the existing Table 1.4 name-pattern blanket;
- congressional-district layouts excluded;
- no incumbent-improvement escape for itemized, SALT, medical, or Table 1.4
  rows.

The builder's US extra now declares its cycle-free `populace-data` workspace
dependency. The anti-drift test compares the actual builder and publish
registers by requirement ID, tolerance, selectors, and incumbent strictness; a
new publisher requirement without at least equally strict builder coverage
fails the test.

## Loss pipeline

`_fiscal_target_loss_weights` applies weighting in this order:

1. derive the existing square-root target-value basis weights;
2. normalize expanded rows to their concept budgets;
3. balance amount/count basis budgets and normalize the vector to mean 1;
4. apply optional family multipliers and renormalize to mean 1;
5. apply `US_CRITICAL_TARGET_LOSS_MULTIPLIER` to rows matching the shared
   critical register.

The final critical overlay defaults to `5.0`, is applied once even if
selectors overlap, excludes congressional-district rows, and is deliberately
not renormalized. Non-critical weights therefore remain bit-for-bit unchanged
and each critical weight is exactly the configured multiple of its normalized
base weight. The solver already divides weighted loss by the final weight sum,
so this changes relative priority without introducing a second loss
definition.

Adjudication runs can set
`--critical-target-loss-multiplier MULTIPLIER`; non-positive and non-finite
values fail argument validation. The effective value is recorded in
calibration-stage telemetry and the failure-safe calibration diagnostics build
record. Current fiscal/state score-only diagnostics also record it in solver
options, build provenance, and summaries. Historical Build H/J replay scripts
explicitly use `1.0` so their old objectives remain reproducible.

## Verification

- `packages/populace-data/tests/test_contract.py`: 65 passed.
- Complete `packages/populace-data/tests`: 132 passed, 1 optional-engine skip.
- Focused builder register/release-gate group: 18 passed.
- Focused loss/CLI/diagnostics/orchestration group: 15 passed.
- Complete fiscal refresh builder plus state-file scorer tests: 135 passed.
- `ruff check --fix`, `ruff format`, final `ruff check`, lockfile
  consistency, and `git diff --check`: passed.
- Independent reviews covered register semantics, loss placement/determinism,
  CLI propagation, diagnostics provenance, package dependency direction, and
  downstream call sites.

Pytest emitted non-failing macOS temporary-directory cleanup warnings in the
builder run; no test failed.

## Implementation commits

- `3b51e22` — start and commit the #462 progress record.
- `6a2ffb9` — share and align builder/publish critical requirements.
- `f3e5dfc` — add the post-normalization critical-row loss priority.
- `c1b794d` — keep manifest test registries faithful to production.
- `92951f6` — declare the shared-register package dependency.
- `323b58a` — make scorer provenance and historical replay behavior explicit.

Nothing was pushed.
