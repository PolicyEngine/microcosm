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

Lane opened. Base integrated; implementation not started.

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

## Next

- Declare the explicit demographic source contract and its reader.
- Correct the composed binding's unbound-demographic registry so it names the
  explicit household reference field rather than `P_SEQ`.
- Focused direct and graph tests on invented fixtures; report.
