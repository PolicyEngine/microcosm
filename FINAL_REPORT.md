# Final report: populace #462 fix 3b

## Outcome

Implemented the settled weighted-donor-quantile tail bound on
`qrf-tail-bound-462`. The branch is based on current `origin/main`, all requested
local tests pass, and nothing was pushed. The #481 weight-aware leaf-draw fix and
#482 dead manifest `support_clip` cleanup remain deferred as directed.

## Delivered

- Added the module configuration for `non_sch_d_capital_gains` at q=0.999.
- Reused `populace.frame.wquantile` for an inverse-CDF quantile over strictly
  positive donor values using the donor's original design weights.
- Clipped only raw tax-unit draws strictly above the bound, before person
  allocation. Rows are never dropped or redrawn, so participation counts are
  preserved and sub-bound values remain bit-identical.
- Added fail-loud validation for missing outputs, invalid quantiles, absent
  positive donor support, and overlap with snapping, sparse-pruning, or signed
  calibration sets.
- Published per-target diagnostics through monolithic and checkpointed
  finalization into build summaries. `clipped_mass_before` and
  `clipped_mass_after` are recipient-design-weighted masses.
- Added focused arithmetic, failure, passthrough, telemetry, and real-donor
  coverage while retaining the existing finalizer behavior tests.

No fitting code, manifest, or unrelated pipeline stage was changed.

## Verification

- Pinned donor SHA-256:
  `7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df`.
- Real-donor weighted p99.9: `211500.84797884867`, finite, positive, and below
  the required `594483.0` ceiling. `BLOCKED.md` was therefore not created.
- Focused PUF support, QRF checkpoint, builder telemetry, source-policy, and
  legacy finalizer suites: passed.
- Full `packages/populace-build/tests` suite: 100%, exit code 0. The exact
  lockfile-pinned `policyengine-uk==2.89.0` cached wheel was exposed read-only so
  cached licensed-artifact regeneration checks executed rather than skipped.
- `ruff check --fix`, `ruff format`, and `git diff --check`: passed.

## Implementation commits

- `d6e8fee` — Implement #462 fix 3b tail bound; defer draws to #481 and
  `support_clip` to #482.
- `0de6d38` — Harden and test #462 fix 3b tail bound; retain #481/#482
  follow-ups.
- `2f966c3` — Keep the #462 real-donor pin within repository source policy.
- `0849253` — Harden #462 fix 3b activation and telemetry; preserve #481/#482
  follow-ups.

Earlier committed progress/design records are `54509aa` and `f5bebb8`.
