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

- Primary Census authority established and byte-verified. Quoted locators,
  digests and limits are in `docs/us-household-reported-income.md`; the full
  evidence package, including the retrieved PDFs and their extracted text, is
  preserved outside this repository at
  `architecture-reviews/microcosm-20260904/us-household-reported-income-authority/`.
- The publisher household total is admissible on both arms, so no total was
  manufactured. `HINCP` is the ACS aggregate; `asec_HTOTVAL` is the ASEC one.
  Neither arm sums components: the executed ACS person roster carries six of
  the eight published income types.
- `acs_household_source_columns` declares the versioned household roster.
  Version 1 restates the executed roster and is asserted against the loader's
  own tuples. Version 2 adds exactly `HINCP` and the housing-record `ADJINC`
  through a separate additive read, so `acs_pums.py` is unedited and its
  implementation identity and source cache are unchanged.
- `household_reported_income` is the pure measurement module: no Frame, no
  archive, no country model, no engine output. A test pins its whole import
  closure.
- B19001 is expressed through the existing `quality_aggregates` distribution
  helper and stays a reserved same-source holdout: no target is activated and
  no candidate is scored.

## Next

- The separate assembly child binds the API documented in
  `docs/us-household-reported-income.md` ("The two-arm surface"). This lane
  registers no source codec and no graph node, so the two additions do not
  collide.
- One genuine small extension is needed before any measured comparison: run
  `read_acs_household_income_columns` over the already-pinned `csv_hus.zip`
  from `acs_2024_1yr_sources.json`. The ASEC arm needs no new input.

## Not done, deliberately

- No push, merge, publication, target activation, or candidate scoring.
- No genuine raw-source read or run; only public publisher documentation.
- No eight-component reconstruction, and no railroad or private/public split.
- No CD reference, housing authority, dynamics or dashboard change.
