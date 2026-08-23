# Rare signed-tail regime evidence

[`realized_regimes.json`](realized_regimes.json) is the deterministic projection
of the 48 red quantile-envelope checks in the frozen f025 arm-split
adjudication. It records all three requested mechanism dimensions for every
check:

- reconstructed donor sign support and realized fitter regime;
- early gap-fill versus late producer-complement ownership; and
- donor-support starvation, including the separately identified upstream
  retirement-cap deletion.

The result is 48 checks over 42 unique target checkpoints. Every target has four
recipient availability patterns, for 168 unique target-pattern records and 192
check-pattern links. All patterns for a target bind the same complete donor
identity. At check level, 35 regimes are `zero_inflated_positive` and 13 are
`three_sign`; none is degenerate or single-sign.

The schema-1 checkpoint metadata authenticates the target, pattern, predictors,
donor index, weight vector, and zero tolerance, but not the realized regime.
The output therefore labels its sign counts and regimes
`fit_boundary_reconstruction`. Do not infer an early target's support from the
final transferred pool because later producers can mutate that field. The
schema-2 target-bank change on this branch persists the fit result directly for
future builds.

Rebuild or check the evidence with the frozen artifacts:

```bash
python experiments/battery_rare_signed_tails/build_realized_regime_evidence.py \
  --adjudication-dir /Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/experiments/battery_burndown \
  --bank-root /Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/populace_us_2024_stacked_pool.checkpoints/stacked/9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6/acs-transfer/d20ba8f3d33d235485b5da720e7bbe8fc79a72fab7b14df4b9fe2fc72ad756c5 \
  --f025-gates /Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/populace_us_2024_stacked_pool.gates.json \
  --transferred-receipts /Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/populace_us_2024_stacked_pool.checkpoints/stacked/9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6/transferred.checkpoint.receipts.json \
  --check
```

The extractor refuses changed authority hashes, missing or extra QED checks,
changed pattern inventories, donor/support non-closure, or a changed 48/42/168
counting identity.
