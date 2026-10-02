# FRS EMPSTATI 11 → OTHER_INACTIVE: receipts

The UK `frs_employment` stage now maps EMPSTATI codes 1-11 one data-dictionary
label each (UKDS SN 9563, FRS 2024-25 adult table), so code 11 ("Other
Inactive") is `OTHER_INACTIVE` rather than `LONG_TERM_DISABLED`. This is the
same mapping as the incumbent's uk-data#526 (head `d3984002`). Everything below
was run on 2026-10-02 against the pinned `adult.tab`
(sha256 `4eaea080…658d`). Only aggregates are reported; no cell here is under
10 survey people.

## The licensed input

Every one of the 27,714 adults in `adult.tab` carries a code from 1 to 11. No
code is blank, non-integer or outside that range, and no `person_id` repeats.
So the new refusal of unknown adult codes does not stop the real build.
`child.tab` has no EMPSTATI column.

| EMPSTATI | Data-dictionary label | Status | Adults | Grossed (GROSS4) |
|---:|---|---|---:|---:|
| 1 | Full-time Employee | FT_EMPLOYED | 10,297 | 22.51m |
| 2 | Part-time Employee | PT_EMPLOYED | 2,701 | 5.57m |
| 3 | Full-time Self-Employed | FT_SELF_EMPLOYED | 1,446 | 2.95m |
| 4 | Part-time Self-Employed | PT_SELF_EMPLOYED | 664 | 1.25m |
| 5 | Unemployed | UNEMPLOYED | 484 | 1.31m |
| 6 | Retired | RETIRED | 8,585 | 12.04m |
| 7 | Student | STUDENT | 383 | 1.34m |
| 8 | Looking after family/home | CARER | 429 | 1.00m |
| 9 | Permanently sick/disabled | LONG_TERM_DISABLED | 1,678 | 3.24m |
| 10 | Temporarily sick/injured | SHORT_TERM_DISABLED | 131 | 0.29m |
| 11 | Other Inactive | OTHER_INACTIVE (was LONG_TERM_DISABLED) | 916 | 2.05m |

Before this change, `LONG_TERM_DISABLED` held 2,594 adults (codes 9 and 11);
it now holds the 1,678 with code 9. These are the same counts uk-data#526
reports for its FRS 2024-25 build.

## Differential against uk-data#526

`differential_vs_uk_data_526.py` loads uk-data's code table and
`derive_employment_status_from_frs` from the PR head by AST and compares them
with microcosm's:

```
code tables identical: 11 codes
fixed cases agree: 34
hypothesis cases agree: 2000 examples
licensed FRS 2024-25 people: 34966; mismatches: 0
```

The fixed cases are codes 0-12, -1, 11.5, NaN and 99, each as an adult and as
a child row. Agreement means both return the same statuses or both refuse.
The licensed comparison runs microcosm's stage derivation
(`derive_frs_employment`, reading `adult.tab` through the sha-pinned reader)
against uk-data's function fed the way `create_frs` feeds it (child rows filled
with 0, adult records by `adult.tab` membership), over all 34,966 people.

## What reads `employment_status`

- `frs_legacy_proxies` reads the frame's `employment_status`.
  `ESA_HEALTH_EMPLOYMENT_STATUSES` is (`LONG_TERM_DISABLED`,
  `SHORT_TERM_DISABLED`), and the support group needs `LONG_TERM_DISABLED`.
  Code-11 adults therefore leave both ESA proxies. `legacy_jobseeker_proxy`
  reads `UNEMPLOYED` only and does not change.
- In policyengine-uk 2.100.0 (microcosm's lock), no variable formula reads
  `employment_status`. The labour-supply dynamics read the self-employed and
  student statuses only, and the three proxies are not engine variables.
- The eFRS parity reference (uk-data 1.56.16), the release input-coverage gate
  and the input-mass parity compare `employment_status` only as a non-empty
  or non-default (engine default `UNEMPLOYED`) share, which this change leaves
  as it was. No CI job reads a released uk-data artifact. Until uk-data ships
  #526, microcosm's code-11 adults and ESA proxies differ from the pinned
  incumbent, and no committed instrument measures that difference.
