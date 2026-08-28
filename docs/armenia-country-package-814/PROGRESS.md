# Armenia country package progress

## State

The engine-free Armenia package, Chronicle/rules handoff, and explicit regression
coverage are complete. The schema, golden, and focused country-package tests
pass; repository-wide verification remains in progress.

## Done

- Read the repository `AGENTS.md` and `CLAUDE.md` operating rules.
- Confirmed that the branch starts from `origin/main` with no existing `am/`
  package changes.
- Read the complete Belgium spec-only package, the country-spec schema and
  golden tests, both New Zealand plan documents, and the calibration/holdout
  doctrine.
- Moved this journal under `docs/`, following the New Zealand planning-doc
  convention. Country directories are closed spec inventories: in-package
  Markdown would make `am/` fail both its loader and spec-only tests.
- Confirmed that New Zealand has planning documents but no committed package.
  Armenia will derive the populace-US donor posture from that plan and say so.
- Decided to omit `support_spine.json`: its only supported method constructs
  raw current/prior ASEC pools and cannot truthfully describe consuming an
  existing populace-US support artifact.
- Identified the shared compiler additions needed for honest contract-only
  Armenia source and geography kernel IDs. These IDs will compile but remain
  explicitly non-executable.
- Confirmed that survey-measured poverty and tax-benefit quantities remain
  permanent holdouts; the ILCS poverty snapshot and external CEQ/World Bank
  results cannot become calibration targets.
- Drafted the complete closed `am/` resource inventory: six typed schema
  resources plus five generation-zero JSON projections, mirroring Belgium.
- Split donor loading from Armenia marz assignment so the typed spine honestly
  declares country-level observed geography and target-derived marz codes.
- Added contract-only compiler identities for the unimplemented Armenia donor,
  marz, and community stages; no runtime fallback or rules engine is present.
- Confirmed the draft resolves through `load_country_spec("am")` with both
  source stages and the community geography contract.
- Added a one-row-per-reference Chronicle harvest contract, the architecture
  and solver notes, the verified-facts-only future rules boundary, and the
  changelog fragment.
- Added Armenia to the deterministic country golden, shared-core compile proof,
  contract-only registry closure, spec-only discovery checks, and runtime
  refusal tests.
- Ran the focused country/spec test files successfully and passed Ruff plus
  `git diff --check`.

## Next

1. Run the full `packages/microcosm-build` suite and the repository's spec-only
   CI shard commands; fix any regressions.
2. Verify the CI group layout and final diff/commit hygiene.
3. Record exact test counts and maintainer questions here and in the requested
   final output report.
