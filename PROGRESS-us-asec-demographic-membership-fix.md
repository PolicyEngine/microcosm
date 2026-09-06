# US ASEC demographic household-membership verification

Lane journal. History, not state: check git and the report for current truth.

## Scope

Close the one material finding the Fable 5.1 audit raised against the
finalized demographic source head `2f3423a2b`
(`asec-demographic-source-fable51-capture.md`): `_reconstruct` classifies
household reference status grouped by `person_household_id`, but nothing
proves that grouping is the same partition as the publisher's own household,
`(income_year, source_household_id)`, which is the unit `A_EXPRRP` is defined
on. The per-row `source_household_id == PH_SEQ` crosscheck does not establish
group equivalence, and the classifier's own cohort-uniformity check does not
either.

Fences honoured by this lane: no genuine Census member or store is stat'd,
read or run; no environment install; no remote write; no nested agents; no
broad test sweep. Source pins, observation semantics, graph stages and the
prior money/restoration metadata stay unchanged, except the expected new
demographic source implementation digest. `is_female` and
`is_household_head` stay unbound.

## State

2026-09-06: lane opened on a fresh worktree created by root from
`origin/main` `c1b832418`, fast-forwarded to `2f3423a2b`. Nothing implemented
yet.

## Done

- Read `AGENTS.md`/`CLAUDE.md`, the restoration report (`.md`/`.json`) and the
  Fable 5.1 capture; located the finding in
  `asec_demographic_source.py::_reconstruct` and
  `classify_asec_demographic_observations`.

## Next

- Add explicit integer coordinate validation and a bidirectional partition
  equivalence proof before classification.
- Record the aggregate verification in the artifact receipt, with no person
  identifier in any refusal.
- Add split, merged, cross-year native-id reuse and cohort-collision tests.
- Run the focused demographic and graph binding/spine fence tests in the
  existing explicit source environment; prove tested bytes against the
  committed blobs.
