# Validating the ACS SPM independence role against the ASEC

Run 2026-10-13 at commit `fe8a19cbe` (branch `us-acs-spm-independent-minor-role`) with
`experiments/acs_spm_role_asec_validation.py`. The receipt is
[`acs-spm-role-asec-validation-receipt.json`](acs-spm-role-asec-validation-receipt.json).
Every input was verified against its pin before it was read, and the receipt
holds aggregate counts and shares only. The run took 482 s of wall time and
9.6 GB peak RSS.

## The question

The ACS has no Census SPM fields and no SPM partition: each ACS household is
one SPM unit. `with_acs_spm_independence_role` reads the ASEC rule
(`SPM_HEAD == 1 OR (A_FAMTYP in {1,4} AND A_FAMREL in {1,2})`) off
`RELSHIPP` as "reference person, spouse of the reference person, or a
group-quarters record's sole person". The ASEC can test that reading. Its CPS
encoding is `A_EXPRRP` 1 to 4: reference person with or without relatives, or
spouse. In households that are one SPM unit, the ASEC also carries Census's own
answer.

## ASEC: four pinned vintages, households that are one SPM unit

| Income year | One-unit households | SPM head is the reference person | Units matching Census adult and child counts (ACS reading) | 15-to-17-year-olds agreeing |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 55,076 | 55,076 | 55,076 of 55,076 | 6,518 of 6,518 |
| 2023 | 54,455 | 54,455 | 54,455 of 54,455 | 6,351 of 6,351 |
| 2024 | 54,015 | 54,015 | 54,014 of 54,015 | 6,283 of 6,284 |
| 2025 | 51,797 | 51,797 | 51,797 of 51,797 | 5,798 of 5,798 |

- **The premise holds.** In every one-unit household of every vintage, the SPM
  head is the household's reference person, so "the unit's head" and "the
  reference person" are the same person on this partition.
- **The counts match.** The ACS reading reproduces Census's `SPM_NUMADULTS` and
  `SPM_NUMKIDS` in 215,342 of 215,343 units. The ASEC rule itself reproduces
  all 215,343.
- **It never over-assigns.** In each vintage, across all households, the ACS
  reading is never true where the ASEC rule is false.
- **What it misses.** Every person-level disagreement in one-unit households is
  a reference person or spouse of an unrelated subfamily (`A_FAMTYP` 4). That
  is a branch the ACS cannot see. There are 118, 124, 101 and 83 such persons
  in the four vintages. Among them, one 15-to-17-year-old in income year 2024
  changes a unit's adult count.
- **Weighted shares.** In one-unit households, the role's weighted share of
  persons is 0.597 to 0.606 under both readings in every vintage. Among
  15-to-17-year-olds in one-unit households, the two readings give 0.0103,
  0.0114 and 0.0089 alike in income years 2022, 2023 and 2025, and 0.0131
  against the ASEC rule's 0.0134 in 2024.
- **Teens in multi-unit households.** In all households, 24 to 40 teens per
  vintage hold the ASEC role without being a reference person or spouse. All
  but one of the 139 across the four vintages live in households of several
  SPM units. Most have
  `A_FAMTYP` 5, outside the family types the rule's family branch reads
  ({1, 4}). So they hold the ASEC rule only through `SPM_HEAD`: they head their
  own Census SPM unit inside someone else's household. The ACS household
  partition cannot represent that (see "What the ACS spine cannot see" in
  [`docs/us-acs-spm-role.md`](../docs/us-acs-spm-role.md)).

## ACS: the derivation on the full pinned 2024 1-year PUMS

The pinned PUMS loads as 3,422,888 persons in 1,531,614 households, after the
loader drops 100,355 vacant housing-unit records. That makes 1,531,614
one-household SPM units, 183,206 of them group-quarters records.

- **Who holds the role.** 2,216,180 persons hold it: 1,348,408 reference
  persons, 684,566 spouses of the reference person and 183,206 group-quarters
  sole persons. Its weighted share of persons is 0.608. 3,570 of the 122,658
  persons aged 15 to 17 hold it, a weighted share of 0.0100.
- **Housing units.** The role is the only classified adult in 117 housing units,
  and no housing unit is left without one. The derivation would have refused
  the spine otherwise.
- **The residue.** 1,946 SPM units have no classified adult. All are
  group-quarters children under 15, and the independent composition check
  (`check_spm_composition`) counts the same 1,946. No role can classify a
  person under 15. The source places ACS group quarters outside the SPM
  universe, and how these units leave SPM measurement is recorded on
  microcosm#1263. The release gate refuses any release on which its engine
  would still measure them.
