# Progress: PR #589 round-1 remediation

## State

Remediation is in progress on `acs-transfer-dtypes-578` from clean starting
commit `fcdd857`. Major 1 is fixed and verified; the all-nonfinite release
coverage, producer-observing dtype guard, and pool-path hours signal gate remain.

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

## Next

- Make required columns with no observed finite/non-null values fail release
  input coverage without weakening receipted pool deferrals.
- Replace the all-family synthetic dtype guard with actual producer execution.
- Port the hours plausibility/nonconstant signal gate to the pool path.
- Run focused and broad sandbox-safe verification, record exact counts, update
  this ledger, and write the final report to the requested output channel.
