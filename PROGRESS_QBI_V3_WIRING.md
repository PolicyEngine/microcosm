# QBI v3 wiring progress

## State

The offline `qbi-v3-wiring` branch is complete in
`.claude/worktrees/populace-wt-530`. The evidence-consuming assumptions
builder, persisted full-artifact calibration, version-3 simulation path, host
transform, replay diagnostics, and final report are implemented. Full-workspace
pytest, Ruff, repository-contract, reproducibility, and offline wheel checks
are green.

## Done

- Created `qbi-v3-wiring` from local `qbi-v2-content`.
- Merged local `qbi-v3-evidence` with no network access.
- Resolved the sole merge conflict in the shared `PROGRESS.md` journal by
  retaining both dedicated sibling ledgers.
- Confirmed the country-package manifest merged without conflict.
- Read the GitNexus exploration skill. No GitNexus query/context tools are
  exposed in this environment, so source flows will be traced directly.
- Traced the v1/v2 donor simulation, QRF target selection, post-QRF host SSTB
  transform, monolithic/checkpoint builder dispatch, public exports, package
  manifest, and restricted replay tests.
- Confirmed v1 must remain the unchanged production default and that adding
  version 3 to the supported-version tuple automatically exposes it only as an
  explicit CLI choice.
- Recorded the byte-identity boundary: v1's combined W-2/UBIA stream and v2's
  qualification, SSTB, W-2, UBIA, and investment families must not gain or
  lose draws.
- Confirmed the evidence forms, finest-industry partition, usable
  receipts/wage/UBIA rows, SCF income bands, and full-artifact person-weight
  mapping.
- Resolved the record-level legal-form seam for implementation: a positive
  qualified combined partnership/S-corporation source selects the required
  17/53 latent split; remaining positive-QBI records use the sole-proprietor
  proxy. The assumptions resource will state this rule explicitly.
- Chosen the evidence-preserving UBIA branch allowed by the binding design:
  a mean-one lognormal residual whose per-form sigma is the
  receipts-weighted standard deviation of log SOI industry intensities divided
  by the square root of the receipts-weight effective industry count. The
  latent industry draw carries the observed cross-industry heterogeneity; this
  standard-error scale adds only modest residual dispersion and preserves the
  selected component's intensity in expectation.
- Confirmed the output-file convention is a committed
  `FINAL_REPORT_QBI_V3_WIRING.md`; no output-path environment variable is
  present.
- Added a deterministic assumptions-build command that maps tax-unit weights
  to person rows, consumes the packaged SCF/SOI resources, solves the
  full-artifact employer shifts, pins all input digests, and emits the strict
  spec-only `qbi_assumptions_v3.json`.
- Persisted log-odds shifts of `-2.598762342032285` for sole proprietorships,
  `-2.3076956692827935` for partnerships, and `-0.9906098787763655` for
  S corporations.
- Added independent v3 entity-split, latent-industry, employer-gate,
  margin-quantile, and UBIA-dispersion streams without changing the v1/v2
  stream paths or production default.
- Wired v3 through the public runtime, post-QRF host SSTB transform, source
  manifest exclusions, and country-package resource declaration.
- Added assumptions-builder unit tests, new-family independence coverage,
  v2/v3 host-SSTB byte-parity coverage, synthetic diagnostics, exact restricted
  replay pins, and evidence/replay digest checks.
- Restricted replay passes: realized zero-employee shares are
  `0.9510065290364592`, `0.791442063158404`, and `0.3774158749014317`
  by form and `0.8430139401618544` overall; W-2 aggregate is
  `$1,472,347,345,828.11`, and the v3 nonzero share is
  `0.020430544809711643`.
- Rebuilt the committed assumptions resource from the pinned H5 and confirmed
  byte-for-byte equality.
- Full-workspace pytest passed with 3,293 tests, 132 skips, and zero failures
  in 121.40 seconds using both required restricted-data environment variables.
- Full-workspace Ruff, changed-file format, diff, country-spec, manifest,
  incumbent-reference, and entrypoint-heuristic checks pass.
- Built the `populace-build` wheel offline from the cached build backend and
  confirmed that it contains the v3 assumptions and both evidence resources.
- Wrote `FINAL_REPORT_QBI_V3_WIRING.md` with all required receipts, the one
  merge conflict, and an explicit no-deviation statement.

## Next

- Supervisor pushes the completed local branch and opens the pull request.
