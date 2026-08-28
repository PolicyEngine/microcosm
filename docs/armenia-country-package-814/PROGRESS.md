# Armenia country package progress

## State

Required reconnaissance is complete. Issue #814 is ready for the first package
draft; no Armenia spec resource has landed yet.

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

## Next

1. Draft the engine-free Armenia pool-reweight specifications and harvest notes.
2. Add compiler contract IDs, explicit package tests, and the golden fixture.
3. Run the spec-only and shard test suites and fix regressions.
4. Record final test evidence and maintainer questions here and in the requested
   final output report.
