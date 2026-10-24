# ACS local county calibration rung

This implements microcosm#922 item 1: an optional county rung on the US
ACS local release line. County logic lives in
`tools/us_acs_local_county_surface.py`; the release driver and district helper
only call its hooks. The initial surface uses ladder population and public IRS
county shares. Future ACS county families can use the same county row,
holdout, and reliability interfaces.

## Plan and invariants

- Population measures are household size times county membership. County
  ladder populations are normalized within each state to its state target.
- County SOI targets use within-state county fact shares of an already
  materialized state concept. County measures split that state's stored sparse
  row, so they require no additional rules-engine columns.
- Every household belongs to exactly one county within its state for each
  concept; county rows reconstruct the state row. County population targets
  and each SOI concept's targets sum to their state target.
- Missing counties, facts, or household support are reported and refused.
  Only an explicitly capped smoke may continue with missing support.
- Holdout roles are deterministic given their salt. Scoring compares held-out
  county targets with state target times county population share.
- Reliability reports show supporting rows, distinct source households, Kish
  ESS, training and holdout error. At fixed error, increasing ESS cannot worsen
  the grade. Missing holdout evidence is explicit.

Engine-free fixture and Hypothesis tests cover these invariants. No full
materialization, release build, publication, or annual projection is performed
as part of this implementation. The county rung applies to the base-year
ACS local calibration; 2026/2027 outputs still need the separate annual
projection and acceptance path.

Feed inspection, exact run commands, resource estimates, grade thresholds,
and validation results will be recorded here with the implementation.
