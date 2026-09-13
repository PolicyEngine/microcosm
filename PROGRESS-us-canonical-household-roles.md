# US canonical household-role fragment — lane journal

Lane: `us-canonical-household-roles-20260913`, based on integration
`a96fc850942d3fd96d24251c07d9ebc38b79f8de`. Independent parallel US
source-only build. No push, no merge, no runtime staging, no test execution.

## Scope

Bounded canonical household-role fragment for the current Microcosm US graph:

- A thin **qualifier** over the genuine retained ACS/ASEC preparation/issuers.
- A **common graph fragment** mapping exact original ACS/ASEC source
  identities onto both initial clone families, preserving int64 ids,
  household relationships, assigned geography, weights, provenance and every
  other Frame cell.
- `is_household_head` bound **only** where the actual documented source
  definition matches. Ambiguous source roles stay unbound and explicitly
  unknown.
- Independent-minor roles: **source/consumer scope audit only** unless an
  existing explicit canonical contract plus genuine source authority
  establish the mapping.

Out of scope (owned elsewhere or deferred): canonical sex, raw
disability/student observations, housing participation, native restoration,
parent-link preservation, wiring the shared country host.

## State

- [x] Worktree/branch/base verified; AGENTS.md + CLAUDE.md read.
- [ ] Trace maintained source owners + canonical `is_household_head`.
- [ ] Mapping/unknownness specification.
- [ ] Qualifier module.
- [ ] Graph fragment module.
- [ ] Pure/graph/fixture tests (written, not run).
- [ ] CI source classification.
- [ ] Docs + changelog fragment.
- [ ] BUILD-RESULT.md.

## Next

Trace owners with a parallel read sweep over the ACS/ASEC source runtime.
