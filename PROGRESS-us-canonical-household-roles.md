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

## Original lane checklist (historical)

- [x] Worktree/branch/base verified; AGENTS.md + CLAUDE.md read.
- [ ] Trace maintained source owners + canonical `is_household_head`.
- [ ] Mapping/unknownness specification.
- [ ] Qualifier module.
- [ ] Graph fragment module.
- [ ] Pure/graph/fixture tests (written, not run).
- [ ] CI source classification.
- [ ] Docs + changelog fragment.
- [ ] BUILD-RESULT.md.

## Original next step (historical)

Trace owners with a parallel read sweep over the ACS/ASEC source runtime.

## Source-only continuation, 2026-09-13

The external builder stopped after commit `593aa157a46d4736238b4535ed3f9d9eb6dd7a52`, containing a qualifier module without graph/tests/docs. Its commits are preserved. The continuation rechecked the clean assigned branch and fetched main: 239 commits ahead, zero behind before further edits.

Review found the declared live fence was never called by the qualifier. The continuation checks it around every owner callback, binds individual column aliases, and independently reconstructs the final projection after owner I/O. Root also explicitly confirmed that unsupported known incumbents must refuse at canonical binding, including artificial group-quarters heads; reconciliation retains the original evidence for adjudication.

The source continuation adds a common two-node graph fragment, pure/graph/actual-owner tests, the mapping and independent-minor audit in `docs/us-current-survey-household-roles.md`, and a changelog fragment. No country-host wiring, housing routing, independent-minor calculation, source staging, project import, pytest, native or installed-model read, push or release is performed. The packet's source pins and bounded runtime proposal record the final scope; written tests remain unrun until separately authorized.

Source AST parsing/compilation, Ruff formatting/lint, diff whitespace and CI inventory verification pass. The inventory contains 122 planned cases (58 pure, 50 graph, 14 actual owners), not executed results. All three new files receive the existing fast/rest and engine/us-am/build classification; no CI classifier change is required.

## Independent-review corrections, 2026-09-13

Fable reviewed exact `8a3226c37a2b0a3e9aee0090f901bfdca4efd549`. The original commit, review and source inventory remain preserved. The concrete graph-test receipt key is corrected from `acs_allocation_provenance` to the actual `relationship_allocation_provenance`. Five additional pure cases cover non-nullable Boolean preservation/refusal, invalid incumbent dtypes and the published current ASEC maximum roster line. Five additional real-owner cases exercise detached ASEC native/line/household mismatches and ACS household/retained-line coverage failures without modifying the issuer, original source or owner buffer.

The source review's proposed legitimate ASEC line above 20 is not supported by the exact 2025 dictionary: A_LINENO is printed 01–16. No production bound is widened. The docs explicitly distinguish the operative 64 MiB serialized-artifact cap, the qualifier row ceiling and the ASEC owner's 600,000-row cap; no national capacity is asserted. The revised planned roster is 132 cases: 63 pure, 50 graph and 19 actual-owner cases (A113/B19). All remain unrun. The two production modules are unchanged.
