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

2026-09-06: implemented, verified and committed locally. 159/159 focused cases
pass at a clean head; genuine extraction and live sex/head binding remain
unrun and unauthorized. External receipts:
`us-asec-demographic-membership-fix-REPORT.md/.json`.

## Correction to the reported finding

The equivalence was already enforced upstream, which the audit did not find.
`asec_household_observations._household_keys` refuses both directions —
a grouping household carrying more than one `(source_year,
source_household_id)` key, and one key claimed by more than one grouping
household — and it is called from `asec_current_money_source._scope`, so from
every `AuthenticatedCurrentMoneySource.validate()`, which `_reconstruct`
performs through `_parent(source)` before and after the join. The audit's
"holds by construction in `asec_pool.py`, but rests on an unverified parent
property" is therefore inaccurate: the property was verified, just not by the
module that forms the headship verdict and not recorded in its receipt. The
fix as directed still stands on its own merits — it is a module-local
restatement that survives a change to the parent loader, and it is what puts
the aggregate verification in this artifact's own receipt.

## Done

- Added `verify_household_membership`: explicit int64/positivity/cohort
  coordinate admission plus a bidirectional partition-equivalence proof, run
  in `_reconstruct` after the per-row `source_household_id == PH_SEQ`
  crosscheck and before any household verdict. Merged and split published
  households and cross-cohort grouping collisions each refuse with their own
  value-free reason naming only a column; native-id reuse across income years
  is admitted and counted.
- Typed the grouping column explicitly at the reconstruct boundary instead of
  letting `to_numpy(dtype=...)` coerce it.
- Recorded the aggregate verification in the artifact receipt under
  `household_membership`, with per-cohort household and native-id counts, and
  declared it in `contract_document()`.
- Derived the join receipt's `unreferenced_source_rows`, `incumbent_conflicts`
  and `native_key_or_age_conflicts` instead of asserting them zero, and made
  every structurally-zero counter name itself in the receipt.
- Documented, in the contract doc, what those zeros and the whole-run refusals
  do not measure, plus two deliberately unchanged refusal-surface limits.
- Added 44 cases: split, merged, cross-cohort reuse, cohort collision,
  order-independence, coordinate dtype/positivity/cohort admission, the
  classifier's own cohort self-check, and blank/non-canonical coordinate and
  person-key tokens.
- Verified against the base tree at `2f3423a2b`: all nine graph-stage hashes
  (including `composed_asec_binding_v1` `3ec79ca7...`), source pins, member
  pins, money schema, person-income restoration identity, prepared-source kind
  and binding schema are unchanged. Only the demographic source module digest,
  its implementation identity, and one added contract key change. The reader
  remains in no live stage inventory.

## Next

- Root and Fable re-review of this exact clean head before genuine use.
- Then the separately authorized bounded extraction from the pinned members,
  which has not been run here.

