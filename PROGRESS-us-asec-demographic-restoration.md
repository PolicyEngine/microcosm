# US ASEC demographic source restoration and explicit binding

Lane journal. History, not state: check git and the report for current truth.

## Scope

Prepare the next bounded source increment named by
`us-asec-demographic-source-root-adjudication.md` (2026-09-06): an explicit
source contract for `A_SEX`/`AXSEX` and `A_EXPRRP` restored from each pinned
original ASEC CSV cohort, with `P_SEQ` preserved only as a separately
diagnosed crosscheck.

Fences honoured by this lane: no source-data parsing of genuine Census
members, no genuine build, no environment installs, no remote writes, no
nested agents, no broad test sweeps. Tiny invented fixtures and direct/graph
tests only. The accepted composition at `84e7a011f` is the base and is not
modified in place.

## State

2026-09-06: local source preparation and focused fixture verification completed.
Genuine extraction and live demographic binding were not run. The external
`us-asec-demographic-restoration-REPORT.md/.json` receipts record the final clean
head and exact tested-code attestation for root review.

## Done

- Read the root adjudication and the source audit it cites, and recomputed the
  audit's own SHA256 (`1e42991b...60c70f`) — matches the memo.
- Re-derived the primary-source locators independently with PyMuPDF over the
  three pinned dictionaries rather than trusting the audit's page numbers:
  `A_SEX` pages 22/23/24, `A_EXPRRP` 21/22/23, `P_SEQ` 20/21/22 and `AXSEX`
  27/28/29 in the 2023/2024/2025 dictionaries. `AXSEX`'s page was not in the
  audit. All four entries are textually identical across the three years.
- Created the child worktree from `origin/main` (`c1b832418`) and integrated
  the accepted composition dependency by fast-forward to `84e7a011f`, which
  is the tree this lane builds on.
- Preserved the original source-contract and runtime-classification commits and
  finished the partial composed-binding metadata edits after the external
  author's session stopped.
- Declared exact source tokens, domains, allocation provenance, reference-person
  cardinality, cohort periods, additive columns and per-cohort diagnostics.
- Fixed invented counterexamples for incumbent relationship recodes, raw-code
  sequence diagnostics, mixed/invalid ACS relationship households, and binding
  payloads that could rewrite the declared demographic contract.
- Ran 115 focused cases: all passed, no skips. All 278 loaded Microcosm modules
  came from this worktree and stayed unchanged during testing. The final report
  also compares those tested bytes and the test files with the committed blobs.
- Verified that the original source/schema/restoration identities and eight
  graph-stage identities are unchanged; only the composed ASEC binding stage
  changes for its corrected metadata contract. The new reader remains unwired.

## Next

- Root/peer review of the exact local source revision before genuine use.
- A separately authorized new extraction from the existing pinned members,
  with aggregate per-cohort diagnostics and preserved frozen parents.
- A later explicit prepared-source/graph integration decision; no sex/head
  booleans or microunit relationship construction are changed by this lane.
