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

## Next

- Extract the verified factsheet findings without transcript noise.
- Map country-package, resource, builder, and testing conventions.
- Implement and validate SCF and SOI estimation logic on synthetic fixtures.
- Run the builder on restricted real inputs, commit only derived resources,
  and record commands and input digests.
- Declare resources, add changelog/tests, run Ruff and the full workspace
  suite, and write the final report.
