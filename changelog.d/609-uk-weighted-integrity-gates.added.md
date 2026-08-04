Ported the two incident-purchased US weighted-integrity gates into the UK
terminal battery (#609, increment 4 of the #578 parity plan). The shared
`input_mass_totals` helper was promoted from `populace.build.us_runtime` to
`populace.build.input_mass` (the US name remains a re-export), and a new
`uk_runtime.weighted_integrity` module supplies the UK evidence plumbing:
`uk_dataset_input_mass_totals` broadcasts household weights through the
national person/benunit/household tables, `uk_input_mass_parity_gate` keeps
the #278 semantics verbatim (zero candidate mass fails at any tolerance,
candidate-only columns are reported and never fail, near-zero reference
columns are skipped) and records the frozen reference's filename, revision,
sha256, and vintage, and `uk_qrf_tail_concentration_gate` derives its column
surface from the `fit_weighted_qrf_stage*` outputs declared in the HMRC
source manifest with no sparsity filter — every declared output is checked,
with `min_nonzero_records` as the sole thinness guard.

Both gates join `uk_terminal_gate_report` under the optional-evidence rule: a
path with no frozen reference or reviewed thresholds omits them instead of
inventing passes, and an armed gate missing either fails closed by name.
Thresholds carry no committed defaults — arming requires explicit
`UKInputMassParityPolicy` / `UKQRFTailConcentrationPolicy` values, which are
sealed into `policy_sha256`, alongside new `input_mass_parity` and
`qrf_tail_concentration` evidence digests; the terminal-gate report schema
moved to 3 and the attestation to 5, with the populace-data publication
contract updated in lockstep. Empty reviewed-exclusion registers are
committed under the universal discipline (mandatory reason, dormant entries
reported, stale entries fail — added on top of the shared input-mass gate,
which lacked staleness detection). The staging launcher grew the matching
`--input-mass-*` and `--qrf-tail-*` flags, and the #609 measurement pass is
now runnable: `tools/measure_uk_weighted_integrity_baselines.py` records
weighted totals and top-k concentration for any national artifact, and
`tools/build_uk_efrs_parity_reference.py --emit-weighted-totals` extracts the
pinned eFRS incumbent's totals to an uncommitted file pending the UKDS EUL
disclosure confirmation.
