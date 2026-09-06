# US household reported money income — source-measurement contract

Session-handoff journal for branch
`microcosm-us-household-reported-income-20260906`, branched from
`ab10f77854b7b9e8057a07dbec40d15c10b96297` (branch
`us-launch-integration-20260906`). That base is a descendant of
`origin/main` at `c1b8324183819cf91511c002bf724a2676524e87`: the merge base is
exactly `origin/main`, and the base is 124 commits ahead with nothing behind.

Scope: a comparable *reported* household total money income measurement on the
prepared ASEC arm (`asec_HTOTVAL`) and the native ACS arm (`HINCP`). This lane
owns primary-source authority, the ACS household source-column contract
version, and one pure measurement module. It does not own CD references,
housing authority, dynamics, the dashboard, or graph assembly.

## State

- Primary Census authority established and byte-verified (see
  `docs/us-household-reported-income.md`).
- ACS household source-column contract v2 adds `HINCP` and the housing-record
  `ADJINC`; v1 stays byte-identical and remains the default.
- `microcosm.build.us_runtime.household_reported_income` is a pure measurement
  module: no engine inputs, no Frame, no graph registration.

## Next

- Two-arm consumer (other lane) binds the documented API.
- One genuine small extension is required before any measured comparison; the
  exact extension is stated in the doc and the lane report.

## Not done, deliberately

- No push, merge, publication, target activation, or candidate scoring.
- No genuine raw-source read or run; only public publisher documentation.
- No eight-component reconstruction and no railroad / private-public split.
