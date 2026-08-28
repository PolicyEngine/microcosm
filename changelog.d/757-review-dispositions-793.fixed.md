The #793 review round (four findings, all the reports-success-without-
testing class): a release shipping national-line gate artifacts without
`release_certification.json` now refuses instead of validating clean by
omission (the id-keyed required-files rule waits on the canonical national
release-id); the rule-1 cross-pin reads `artifacts.candidate.sha256` - the
field that names what the scorer measured - instead of a substring scan; a
malformed sidecar fit-weight block raises as corruption instead of
degrading to the empty tuple (a genuinely empty fitting stage still fails
the audit, never vacuously passes); and the parity evidence's shared
name@period grain is now a loud refusal rather than an implicit
convention. Alongside, the duplication the round pointed at is
consolidated: one scope-filtering helper (`uk_scoped_gate_manifest`, with
a source parameter serving the spine driver's stub point) replaces three
copies and a dead zero-caller variant, one `finalize_uk_scoped_gate_report`
grafts and re-signs every scoped report, the certification's private
cross-module imports become public names, and `GateBatteryRun` exposes its
attested digests as properties so derivation stops routing through a
signed payload.
