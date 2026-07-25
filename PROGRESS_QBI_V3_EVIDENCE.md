# QBI v3 evidence progress

## State

- Active branch: `qbi-v3-evidence`.
- Dedicated worktree:
  `.claude/worktrees/populace-wt-530-v3`.
- Base: local `repeal-validation-298` at `e45f797`.
- Offline, local commits only. This lane produces provisional evidence
  resources and reproducible estimation code; simulation wiring remains out
  of scope.

## Done

- Created the requested isolated worktree and branch.
- Read the GitNexus exploration skill; its MCP tools are unavailable in this
  session, so codebase exploration will use direct repository searches.
- Read/inventoried the SCF, SOI, and Section 199A factsheets plus the restricted
  SCF archive and six SOI workbooks in place.
- Mapped the package-resource, spec-only, restricted-builder, and schema-test
  conventions.
- Declared the offline-cached `xlrd` reader needed for the two genuine BIFF
  SOI workbooks and regenerated its lock entry without network access.
- Implemented the pure SCF business-record stack, X42001/5 implicate pooling,
  owned-net bands, legal-form groups, employer proxy, independent presence/size
  collapse, weighted margin quantiles, JCT comparison, and strict resource
  validator.
- Added synthetic tests for implicate arithmetic, `-1` economic-zero handling,
  ownership scaling, thin-cell fallback, inverse-CDF margins, and schema
  refusal. Four focused tests and Ruff pass.

## Next

- Implement and validate the SOI workbook parsers and ratio builder on
  synthetic fixtures.
- Run the builder on restricted real inputs, commit only derived resources,
  and record commands and input digests.
- Declare resources, add changelog/tests, run Ruff and the full workspace
  suite, and write the final report.
