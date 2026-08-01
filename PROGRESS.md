# Progress: PR #589 round-1 remediation

## State

Remediation is in progress on `acs-transfer-dtypes-578` from clean starting
commit `fcdd857`. All four findings are fixed with focused regressions; broad
sandbox-safe verification remains.

## Done

- Read `CLAUDE.md` and the supplied verdict
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/reviews/sol_589_r1.log`.
- Confirmed the requested branch, clean starting state, and no-push boundary.
- Attempted the GitNexus debugging workflow; its integration is unavailable in
  this session, so source/call-site tracing and executable regressions are the
  fallback.
- Established this committed progress ledger before implementation.
- Made the known-boolean encoder reject object-backed values unless their
  observed semantics are strictly Python/NumPy boolean, with target-named and
  offending-type diagnostics; generic donor validation now deliberately routes
  observed known-boolean objects to that boundary.
- Added Sol's exact `is_blind=object([True, 0.0])` regression, bool/int,
  bool/string, NumPy-boolean near misses, uniform object 0/1 rejection, and
  valid Python/NumPy boolean-object acceptance. All eight rejection/acceptance
  cases pass, with no QRF invocation on failures.
- Adjudicated and retained physical numeric 0/1 compatibility: the real
  primary-PUF finalizer emits all eight QBI engine-boolean leaves as finite
  `float64` `{0, 1}` before ACS transfer, confirmed in the persisted
  `base-p1/004_qrf_finalization.frame.h5` checkpoint. The HDF round-trip
  compatibility regression now uses the actual `business_is_sstb` producer
  target. The focused Major-1 batch passes: 9 passed, 29 deselected.
- Classified columns with zero finite/non-null observed values as degenerate in
  the shared default-valued gate, including a named failure, report detail, and
  reason-preserving release-coverage finding.
- Added Sol's exact isolated coverage regression: required
  `bank_account_assets`, `bond_assets`, and `stock_assets`, each
  `float64 [NaN, NaN]`, now produce three named failures with no columns
  misclassified as missing.
- Confirmed no pool exclusion is needed or added. Pool readiness runs the
  terminal agreement gate and publishes explicit typed-null asset-deferral
  receipts; release input coverage binds later to the calibrated fiscal export
  after the unconditional SCF/SIPP restoration and its signal gate.
- Major-2 verification passes: direct repro 2 passed; complete generic/US gate
  suites 235 passed; UK shared-gate regression suite 39 passed, 3 skipped.
- Wired `us_hours_worked_signal_gate` immediately after the pool's pre-clone
  hours producer. Failures now abort the pool source chain; passing gate name,
  status, failures, and details are visible in the operator receipt.
- Added a pool-path regression with all three hours outputs present,
  nonconstant, and implausible. It fails on worked share 1.000 and worker mean
  67.5 rather than being trusted by the idempotent producer. The existing real
  pre-clone test now pins the passing receipt; both focused tests pass.
- Replaced the hard-coded all-family dtype table with a small end-to-end
  producer fixture. It executes all 21 real pool source operators (including
  the twice-bound prior-year operator), physical PUF-support cloning, the real
  primary-PUF imputer/finalizer, donor-channel resolution, donor validation,
  complete-case masking, and the actual target encoder for all 117 active
  targets; only the underlying QRF predictions are deterministic test doubles.
- Added a Sol-style mutation that wraps the real hours producer and changes its
  numeric output to object-backed strings. The real signal gate still observes
  the numeric values, but the encoder rejects the physical dtype and names
  `hours_worked_last_week`; assertions prove every source producer and the
  primary-PUF QRF fit/predict path ran. Both producer-guard tests pass (2 passed,
  27 deselected).
- Expanded focused verification initially found one stale structural-test
  assumption: it required each operator-map value to have the same symbol name
  as its registry key. The audit now explicitly pins the hours key to its sole
  allowed gated wrapper while retaining exact-name checks for every other
  operator; the structural audit and implausible-hours regression both pass.

## Next

- Run focused and broad sandbox-safe verification, record exact counts, update
  this ledger, and write the final report to the requested output channel.
